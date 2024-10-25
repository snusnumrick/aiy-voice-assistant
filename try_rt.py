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
from aiy.leds import Leds, Color, Pattern

# Audio configuration
RECORD_FORMAT = AudioFormat(sample_rate_hz=24000, num_channels=1, bytes_per_sample=2)
AUDIO_FORMAT = AudioFormat(sample_rate_hz=24000, num_channels=1, bytes_per_sample=2)
OPENAI_SAMPLE_RATE = 24000
CHUNK_DURATION_SECS = 0.1  # 100ms chunks
MIN_AUDIO_BUFFER = 0.2  # 200ms minimum buffer size
LISTENING_COLOR = Color.WHITE
LISTENING_PATTERN = Pattern.breathe(5000)
TRANSCRIBING_COLOR = Color.GREEN
TRANSCRIBING_PATTERN = None


def resample_audio(audio_data, src_rate=24000, target_rate=16000):
    if src_rate == target_rate:
        return audio_data

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

        # response audio WAV file
        self.response_wav_file = wave.open(self.response_wav_filename, 'wb')
        self.response_wav_file.setnchannels(1)
        self.response_wav_file.setsampwidth(2)
        self.response_wav_file.setframerate(OPENAI_SAMPLE_RATE)

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
        if self.response_wav_file:
            self.response_wav_file.close()
            self.response_wav_file = None
        logger.info("Closed WAV files")

    def _handle_button_press(self):
        """Callback for button press event"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def _handle_button_release(self):
        """Callback for button release event"""
        pass

    async def connect_websocket(self):
        """Connect to OpenAI Realtime API websocket"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "realtime=v1"
        }

        try:
            self.websocket = await websockets.connect(url, extra_headers=headers)
            logger.info("Connected to OpenAI Realtime API")

            # Send initial configuration
            await self.websocket.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": "alloy",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    }
                }
            }))
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            sys.exit(1)

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
                        # if self.response_wav_file:
                        #     self.response_wav_file.writeframes(audio_data)

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

            # Write original audio to file
            # if self.original_wav_file:
            #     for chunk in audio_chunks:
            #         self.original_wav_file.writeframes(chunk)
                # self.original_wav_file.writeframes(combined_audio)

            # Resample to 16kHz using pydub
            resampled_audio = resample_audio(combined_audio,
                                             src_rate=RECORD_FORMAT.sample_rate_hz,
                                             target_rate=OPENAI_SAMPLE_RATE)

            if resampled_audio is None:
                logger.error("Failed to resample audio")
                return

            # Write resampled audio to file
            # if self.resampled_wav_file:
            #     self.resampled_wav_file.writeframes(resampled_audio)

            # Calculate duration for logging
            audio_bytes = len(resampled_audio)
            audio_duration = audio_bytes / (OPENAI_SAMPLE_RATE * 2)  # 2 bytes per sample
            logger.debug(f"Sending audio message: {audio_bytes} bytes ({audio_duration:.3f}s)")

            # Encode to base64
            encoded_audio = base64.b64encode(resampled_audio).decode()
            # test = base64.b64decode(encoded_audio)
            # assert resampled_audio == test

            # Create message event
            event = {
                "type": "input_audio_buffer.append",
                "audio": encoded_audio
            }

            # Send the message
            await self.websocket.send(json.dumps(event))

        except Exception as e:
            logger.error(f"Error sending audio message: {e}")

    async def process_audio_chunks(self):
        """Process audio chunks from the queue and send to websocket"""
        chunks_buffer = []
        buffer_duration = 0
        last_send_time = time.time()
        logger.info("Audio processing started...")

        while True:
            try:
                # Non-blocking get from queue
                chunk = self.audio_queue.get_nowait()

                if chunk is None:  # Sentinel value for stopping
                    logger.info("Audio processing stopped")
                    # Send any remaining buffered audio
                    if chunks_buffer:
                        await self.send_audio_message(chunks_buffer)
                    break

                # Write original chunk to WAV file
                # if self.original_wav_file:
                #     self.original_wav_file.writeframes(chunk)

                # Calculate chunk duration in seconds
                chunk_duration = len(chunk) / (RECORD_FORMAT.bytes_per_sample * RECORD_FORMAT.sample_rate_hz)
                buffer_duration += chunk_duration
                chunks_buffer.append(chunk)
                logger.debug(f"buffer: {buffer_duration:.3f}s ({len(chunk)} bytes)")

                # If we've accumulated enough audio and have a websocket connection
                current_time = time.time()
                if buffer_duration >= MIN_AUDIO_BUFFER and (current_time - last_send_time) >= 0.1:
                    if self.websocket:
                        # Send accumulated audio as a message
                        await self.send_audio_message(chunks_buffer)
                        last_send_time = current_time
                        logger.debug(f"Sent {buffer_duration:.3f}s of audio")

                    # Clear the buffer after sending
                    chunks_buffer = []
                    buffer_duration = 0

                self.audio_queue.task_done()

            except queue.Empty:
                # If queue is empty, wait a bit before checking again
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                await asyncio.sleep(0.01)

    def record_audio(self):
        """Record audio in a separate thread"""
        logger.info("Recording started...")
        self.led.pattern = LISTENING_PATTERN
        self.led.update(Leds.rgb_pattern(LISTENING_COLOR))

        # Open new WAV files for recording
        self._open_wav_files()

        try:
            # Initialize timing variables
            chunk_count = 0
            start_time = time.time()

            for chunk in self.recorder.record(
                    RECORD_FORMAT,
                    chunk_duration_sec=CHUNK_DURATION_SECS,
                    on_start=lambda: None,
                    on_stop=lambda: None
            ):
                if not self.recording:
                    break

                # Put the chunk in the thread-safe queue
                self.audio_queue.put(chunk)
                chunk_count += 1

                # Log throughput every second for debugging
                if chunk_count % 10 == 0:  # Every second (10 * 0.1s chunks)
                    elapsed = time.time() - start_time
                    logger.debug(f"Audio throughput: {chunk_count / elapsed:.2f} chunks/sec")

        except Exception as e:
            logger.error(f"Recording error: {e}")
        finally:
            # Signal the audio processing to stop
            self.audio_queue.put(None)
            self._close_wav_files()
            self.led.update(Leds.rgb_off())
            logger.info("Recording stopped")

    def start_recording(self):
        """Start recording audio"""
        if not self.recording:
            self.recording = True
            # Start the recording thread
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()

    def stop_recording(self):
        """Stop recording audio"""
        if self.recording:
            self.recording = False
            self.led.update(Leds.rgb_off())
            if self.recording_thread:
                self.recording_thread.join()
                self.recording_thread = None

    async def handle_server_events(self):
        """Handle events from the OpenAI Realtime API"""
        # Start audio processing task
        audio_task = asyncio.create_task(self.process_audio_chunks())

        try:
            logger.info("Waiting for events...")
            async for message in self.websocket:
                event = json.loads(message)
                # logger.info(f"Received event: {json.dumps(event, indent=2)}")

                if event["type"] == "error":
                    logger.error(f"Error event: {event.get('error')}")
                    continue

                elif event["type"] == "session.created":
                    logger.info(f"{json.dumps(event, indent=2)}")

                elif event["type"] == "session.updated":
                    logger.info(f"{json.dumps(event, indent=2)}")

                elif event["type"] == "input_audio_buffer.speech_started":
                    logger.info(f"speech_started")
                    self.led.update(Leds.rgb_on(TRANSCRIBING_COLOR))

                elif event["type"] == "input_audio_buffer.speech_stopped":
                    logger.info(f"speech_stopped")
                    self.led.pattern = LISTENING_PATTERN
                    self.led.update(Leds.rgb_pattern(LISTENING_COLOR))

                elif event["type"] == "response.created":
                    logger.info(f"response.created")

                elif event["type"] == "input_audio_buffer.committed":
                    logger.info(f"input_audio_buffer.committed")

                elif event["type"] == "conversation.item.created":
                    item = event["item"]
                    logger.info(f"{item['role']}: {item['content']}")

                elif event["type"] == "response.audio.delta":
                    logger.info("received response.audio.delta")
                    # Play audio chunk
                    audio_data = base64.b64decode(event["delta"])
                    self.player.play(AUDIO_FORMAT)(audio_data)

                elif event["type"] == "response.audio_transcript.done":
                    logger.info(f"Response text: {event.get('transcript')}")

                else:
                    logger.info(f"Event not handled: {event['type']}")

        except Exception as e:
            logger.error(f"Event handling error: {e}")
        finally:
            await audio_task
            logger.info("Event handling completed")

    async def run(self):
        """Main run loop"""
        await self.connect_websocket()
        await self.handle_server_events()

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