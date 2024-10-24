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

from aiy.board import Board, MultiColorLed
from aiy.voice.audio import AudioFormat, BytesPlayer, Recorder
from aiy.leds import Leds, Color
import openai

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

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

    async def connect_websocket(self):
        """Connect to OpenAI Realtime API websocket"""
        try:
            self.ws = await openai.beta.realtime.connect(
                model="gpt-4o-realtime-preview-2024-10-01",
                voice="alloy",
                input_audio_format="pcm16",
                output_audio_format="pcm16",
                turn_detection={
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            )
            logger.info("Connected to OpenAI Realtime API")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            sys.exit(1)

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
                if self.ws:
                    asyncio.run(self.ws.send_audio(chunk))
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
            async for event in self.ws:
                if event.type == "error":
                    logger.error(f"Error: {event.error}")
                    continue

                if event.type == "response.audio.delta":
                    # Play audio chunk
                    audio_data = event.delta
                    self.player.play(AUDIO_FORMAT)(audio_data)

                elif event.type == "response.text.delta":
                    # Print text as it comes in
                    logger.info(f"Response text: {event.delta}")

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

                # Setup button handler

        self.board.button.when_pressed = lambda: handle_button(self.board.button)
        self.board.button.when_released = lambda: handle_button(self.board.button)

        # Start event handling
        await self.handle_server_events()

    def cleanup(self):
        """Cleanup resources"""
        self.stop_recording()
        if self.ws:
            asyncio.run(self.ws.close())
        self.board.close()
        self.player.join()
        self.recorder.join()
        self.led.reset()


def main():
    assistant = RealtimeAssistant()

    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        assistant.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(assistant.run())
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        assistant.cleanup()


if __name__ == "__main__":
    main()
