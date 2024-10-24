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
import websockets
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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
AUDIO_FORMAT = AudioFormat(sample_rate_hz=16000, num_channels=1, bytes_per_sample=2)
CHUNK_DURATION_SECS = 0.1


class RealtimeAssistant:
    def __init__(self):
        self.board = Board()
        self.player = BytesPlayer()
        self.recorder = Recorder()
        self.recording = False
        self.ws = None
        self.recording_thread = None
        self.led = Leds()
        self.websocket = None

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
            logger.info("Sending initial configuration...")
            await self.websocket.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": "Please assist the user.",
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
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

    async def send_audio(self, chunk):
        """Send audio chunk to websocket"""
        if self.websocket:
            try:
                logger.info("sending audio chunk")
                await self.websocket.send(json.dumps({
                    "type": "audio.chunk",
                    "chunk": chunk.hex()  # Convert bytes to hex string
                }))
            except Exception as e:
                logger.error(f"Failed to send audio: {e}")

    def record_audio(self):
        """Record audio in a separate thread"""
        logger.info("Recording started...")
        self.led.update(Leds.rgb_on(Color.RED))

        try:
            for chunk in self.recorder.record(
                    AUDIO_FORMAT,
                    chunk_duration_sec=CHUNK_DURATION_SECS,
                    on_start=lambda: None,
                    on_stop=lambda: None
            ):
                if not self.recording:
                    break
                if self.websocket:
                    asyncio.run(self.send_audio(chunk))
        except Exception as e:
            logger.error(f"Recording error: {e}")
        finally:
            self.led.update(Leds.rgb_off())
            logger.info("Recording stopped")

    def start_recording(self):
        """Start recording audio"""
        if not self.recording:
            self.recording = True
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()

    def stop_recording(self):
        """Stop recording audio"""
        if self.recording:
            self.recording = False
            if self.recording_thread:
                self.recording_thread.join()
                self.recording_thread = None

    async def handle_server_events(self):
        """Handle events from the OpenAI Realtime API"""
        try:
            while True:
                logger.info("Waiting for events...")
                message = await self.websocket.recv()
                event = json.loads(message)

                if event.get("type") == "error":
                    logger.error(f"Error event: {event.get('error')}")
                    continue

                if event.get("type") == "audio.delta":
                    # Play audio chunk
                    audio_data = bytes.fromhex(event["delta"])  # Convert hex string back to bytes
                    self.player.play(AUDIO_FORMAT)(audio_data)

                elif event.get("type") == "text.delta":
                    # Print text as it comes in
                    logger.info(f"Response text: {event.get('delta')}")

        except Exception as e:
            logger.error(f"Event handling error: {e}")

    async def run(self):
        """Main run loop"""
        await self.connect_websocket()

        def handle_button(button):
            if button.pressed:
                self.start_recording()
            else:
                self.stop_recording()

        self.board.button.when_pressed = lambda: handle_button(self.board.button)
        self.board.button.when_released = lambda: handle_button(self.board.button)

        # Start event handling
        await self.handle_server_events()

    async def cleanup(self):
        """Cleanup resources"""
        self.stop_recording()
        if self.websocket:
            await self.websocket.close()
        self.board.close()
        self.player.join()
        self.recorder.join()
        self.led.reset()


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