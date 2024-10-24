me
Assistant
with Fixed WebSocket

```python
# !/usr/bin/env python3
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
import io
import websockets
from pydub import AudioSegment
from dotenv import load_dotenv

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
RECORD_FORMAT = AudioFormat(sample_rate_hz=24000, num_channels=1, bytes_per_sample=2)
OPENAI_SAMPLE_RATE = 16000
CHUNK_DURATION_SECS = 0.1  # 100ms chunks
MIN_AUDIO_BUFFER = 0.2  # 200ms minimum buffer size


def resample_audio(audio_data, src_rate=24000, target_rate=16000):
    """Resample audio data using pydub"""
    try:
        # Create WAV in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(src_rate)
            wav.writeframes(audio_data)
        wav_io.seek(0)

        # Load with pydub
        audio = AudioSegment.from_wav(wav_io)

        # Resample
        resampled_audio = audio.set_frame_rate(target_rate)

        # Export to bytes
        output_io = io.BytesIO()
        resampled_audio.export(output_io, format='wav')

        # Extract raw PCM from WAV
        with wave.open(output_io, 'rb') as wav:
            return wav.readframes(wav.getnframes())

    except Exception as e:
        logger.error(f"Error resampling audio: {e}")
        return None


def append_to_wav(wav_file, audio_data, sample_rate=16000, channels=1, sample_width=2):
    """Helper function to append audio data to an existing WAV file"""
    try:
        # Convert hex to bytes if needed
        if isinstance(audio_data, str):
            audio_data = bytes.fromhex(audio_data)

        # Create WAV file if it doesn't exist
        if not os.path.exists(wav_file):
            with wave.open(wav_file, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
        else:
            # Append to existing file
            with wave.open(wav_file, 'ab') as wf:
                wf.writeframes(audio_data)
    except Exception as e:
        logger.error(f"Error appending to WAV file: {e}")


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

    async def send_audio_message(self, audio_chunks):
        """Send audio as a conversation item message"""
        if not audio_chunks:
            return

        try:
            # Combine chunks
            combined_audio = b''.join(audio_chunks)

            # Write original audio to file
            if self.original_wav_file:
                self.original_wav_file.writeframes(combined_audio)

            # Resample to 16kHz using pydub
            resampled_audio = resample_audio(combined_audio,
                                             src_rate=RECORD_FORMAT.sample_rate_hz,
                                             target_rate=OPENAI_SAMPLE_RATE)

            if resampled_audio is None:
                logger.error("Failed to resample audio")
                return

            # Write resampled audio to file
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