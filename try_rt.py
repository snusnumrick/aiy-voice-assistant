#!/usr/bin/env python3
"""
Sample app demonstrating OpenAI Realtime API with AIY audio interface.
"""
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import queue
import time
import wave
import base64
import websockets
import numpy as np
from dotenv import load_dotenv
import soundfile as sf
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from aiy.board import Board
from aiy.voice.audio import AudioFormat, BytesPlayer, Recorder
from aiy.leds import Leds, Color

# Audio configuration
# Original recording format
RECORD_FORMAT = AudioFormat(sample_rate_hz=24000, num_channels=1, bytes_per_sample=2)
# Required OpenAI format
OPENAI_SAMPLE_RATE = 16000
CHUNK_DURATION_SECS = 0.1  # 100ms chunks
MIN_AUDIO_BUFFER = 0.2  # 200ms minimum buffer size


def resample_audio(audio_data, src_rate=24000, target_rate=16000):
    """Resample audio data to target sample rate"""
    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Calculate resampling ratio
    ratio = target_rate / src_rate

    # Calculate new length
    new_length = int(len(audio_array) * ratio)

    # Use numpy to resample
    resampled = np.array(list(audio_array[int(i / ratio)] for i in range(new_length)))

    # Convert back to bytes
    return resampled.astype(np.int16).tobytes()


class RealtimeAssistant:
    def __init__(self):
        self.board = Board()
        self.player = BytesPlayer()
        self.recorder = Recorder()
        self.recording = False
        self.recording_thread = None
        self.led = Leds()
        self.websocket = None

        # Use a thread-safe Queue for audio chunks
        self.audio_queue = queue.Queue()
        self.buffer_time = 0

        # Create WAV files for debug recording
        timestamp = int(time.time())
        self.original_wav_filename = f"original_{timestamp}.wav"
        self.resampled_wav_filename = f"resampled_{timestamp}.wav"
        self.response_wav_filename = f"response_{timestamp}.wav"
        self.original_wav_file = None
        self.resampled_wav_file = None

        # Counter for response chunks
        self.response_chunk_count = 0

        # Set up button handlers
        self.board.button.when_pressed = self._handle_button_press
        self.board.button.when_released = self._handle_button_release

    def _open_wav_files(self):
        """Open WAV files for writing"""
        # Original audio WAV file
        self.original_wav_file = wave.open(self.original_wav_filename, 'wb')
        self.original_wav_file.setnchannels(RECORD_FORMAT.num_channels)
        self.original_wav_file.setsampwidth(RECORD_FORMAT.bytes_per_sample)
        self.original_wav_file.setframerate(RECORD_FORMAT.sample_rate_hz)

        # Resampled audio WAV file
        self.resampled_wav_file = wave.open(self.resampled_wav_filename, 'wb')
        self.resampled_wav_file.setnchannels(1)
        self.resampled_wav_file.setsampwidth(2)
        self.resampled_wav_file.setframerate(OPENAI_SAMPLE_RATE)

        logger.info(
            f"Opened WAV files: {self.original_wav_filename}, {self.resampled_wav_filename}, and response will be saved to {self.response_wav_filename}")

    def _close_wav_files(self):
        """Close WAV files if open"""
        if self.original_wav_file:
            self.original_wav_file.close()
            self.original_wav_file = None
        if self.resampled_wav_file:
            self.resampled_wav_file.close()
            self.resampled_wav_file = None
        logger.info("Closed WAV files")

    async def handle_server_events(self):
        """Handle events from the OpenAI Realtime API"""
        # Start audio processing task
        audio_task = asyncio.create_task(self.process_audio_chunks())

        try:
            logger.info("Waiting for events...")
            async for message in self.websocket:
                event = json.loads(message)
                logger.info(f"Received event: {json.dumps(event, indent=2)}")

                if event["type"] == "error":
                    logger.error(f"Error event: {event.get('error')}")
                    continue

                elif event["type"] == "response.audio.delta":
                    try:
                        # Convert hex string to bytes
                        audio_data = bytes.fromhex(event["delta"])

                        # Save to response WAV file
                        append_to_wav(self.response_wav_filename, audio_data)

                        # Play audio
                        self.player.play(AUDIO_FORMAT)(audio_data)

                        # Log audio chunk info
                        self.response_chunk_count += 1
                        chunk_size = len(audio_data)
                        chunk_duration = chunk_size / (AUDIO_FORMAT.sample_rate_hz * AUDIO_FORMAT.bytes_per_sample)
                        logger.debug(
                            f"Response audio chunk {self.response_chunk_count}: {chunk_size} bytes ({chunk_duration:.3f}s)")

                    except Exception as e:
                        logger.error(f"Error handling audio response: {e}")

                elif event["type"] == "response.text.delta":
                    # Print text as it comes in
                    logger.info(f"Response text: {event.get('delta')}")

        except Exception as e:
            logger.error(f"Event handling error: {e}")
        finally:
            await audio_task
            logger.info(f"Event handling completed. Total response chunks: {self.response_chunk_count}")

    async def send_audio_message(self, audio_chunks):
        """Send audio as a conversation item message"""
        if not audio_chunks:
            return

        try:
            # Combine chunks
            combined_audio = b''.join(audio_chunks)

            # Resample to 16kHz
            resampled_audio = resample_audio(combined_audio,
                                             src_rate=RECORD_FORMAT.sample_rate_hz,
                                             target_rate=OPENAI_SAMPLE_RATE)

            # Write to debug WAV file
            if self.resampled_wav_file:
                self.resampled_wav_file.writeframes(resampled_audio)

            # Calculate duration for logging
            audio_bytes = len(resampled_audio)
            audio_duration = audio_bytes / (OPENAI_SAMPLE_RATE * 2)  # 2 bytes per sample
            logger.info(f"Sending audio message: {audio_bytes} bytes ({audio_duration:.3f}s)")

            # Encode to base64
            encoded_audio = base64.b64encode(resampled_audio).decode()

            # Create message event
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "audio": encoded_audio
                    }]
                }
            }

            # Send the message
            await self.websocket.send(json.dumps(event))

        except Exception as e:
            logger.error(f"Error sending audio message: {e}")

    # ... [rest of the class implementation remains the same] ...

    async def cleanup(self):
        """Cleanup resources"""
        self.stop_recording()
        self._close_wav_files()
        if self.websocket:
            await self.websocket.close()
        self.board.close()
        self.player.join()
        self.recorder.join()
        self.led.reset()
        logger.info(f"Cleanup complete. Check debug files:\n"
                    f"  Input original: {self.original_wav_filename}\n"
                    f"  Input resampled: {self.resampled_wav_filename}\n"
                    f"  Response audio: {self.response_wav_filename}")

async def main():
    assistant = RealtimeAssistant()

    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        asyncio.create_task(assistant.cleanup())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await assistant.run()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        await assistant.cleanup()


if __name__ == "__main__":
    asyncio.run(main())