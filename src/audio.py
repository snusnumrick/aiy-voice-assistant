"""
Audio processing module.

This module provides functionality for speech transcription and synthesis,
interfacing with various STT and TTS engines.
"""

from enum import Enum
from typing import Optional, Iterator
from subprocess import Popen
from collections import deque
import time
import logging
import contextlib

from aiy.board import Button
from aiy.leds import Leds, Color, Pattern
from aiy.voice.audio import AudioFormat, Recorder

from src.config import Config
from src.stt import create_stt_service, BaseSTTService
from src.tts import create_tts_engine, synthesize_speech

logger = logging.getLogger(__name__)


class RecognitionStatus(Enum):
    """Enum representing the current status of speech recognition."""
    IDLE = 0
    LISTENING = 1
    PROCESSING = 2


@contextlib.contextmanager
def led_pattern(leds: Leds, pattern: Pattern, color: tuple):
    """
    Context manager for setting and cleaning up LED patterns.

    Args:
        leds (Leds): The LED controller.
        pattern (Pattern): The LED pattern to set.
        color (tuple): RGB color tuple for the LED.
    """
    leds.pattern = pattern
    leds.update(Leds.rgb_pattern(color))
    try:
        yield
    finally:
        leds.update(Leds.rgb_off())


class SpeechTranscriber:
    """
    A class to handle speech transcription using various speech recognition services.

    This class manages the button interface, LED indicators, and the transcription process.
    """

    def __init__(self, button: Button, leds: Leds, config: Config):
        """
        Initialize the SpeechTranscriber.

        Args:
            button (Button): The AIY Kit button object.
            leds (Leds): The AIY Kit LED object.
            config (Config): The application configuration object.
        """
        self.button = button
        self.leds = leds
        self.config = config
        self.button_is_pressed = False
        self.stt_service = create_stt_service(config)
        self.tts_engine = create_tts_engine(config)

        # Load configuration settings
        self.breathing_period_ms = self.config.get('ready_breathing_period_ms', 10000)
        self.led_breathing_color = self.config.get('ready_breathing_color', (0, 1, 0))
        self.led_recording_color = self.config.get('recording_color', (0, 255, 0))
        self.led_breathing_duration = self.config.get('ready_breathing_duration', 60)
        self.led_processing_color = self.config.get('processing_color', (0, 1, 0))
        self.led_processing_blink_period_ms = self.config.get('processing_blink_period_ms', 300)
        self.audio_sample_rate = self.config.get('audio_sample_rate', 16000)
        self.audio_recording_chunk_duration_sec = self.config.get('audio_recording_chunk_duration_sec', 0.3)
        self.post_button_release_record_chunks = self.config.get('post_button_release_record_chunks', 3)

    def transcribe_speech(self, player_process: Optional[Popen] = None) -> str:
        """
        Transcribe speech from the microphone input.

        Args:
            player_process (Optional[Popen]): A subprocess.Popen object representing a running audio player.

        Returns:
            str: The transcribed text.
        """
        self.setup_button_callbacks()
        logger.info('Press the button and speak')

        with Recorder() as recorder:
            audio_generator = self.generate_audio_chunks(recorder)

            try:
                text = self.stt_service.transcribe_stream(audio_generator)
            except Exception as e:
                logger.error(f"Error transcribing speech: {str(e)}")
                text = ""

        return text

    def generate_audio_chunks(self, recorder: Recorder) -> Iterator[bytes]:
        """
        Generate audio chunks for transcription.

        Args:
            recorder (Recorder): The audio recorder object.

        Yields:
            bytes: Audio chunk data.
        """
        chunks_deque: deque = deque()
        status = RecognitionStatus.IDLE
        record_more = 0

        with led_pattern(self.leds, Pattern.breathe(self.breathing_period_ms), self.led_breathing_color):
            for chunk in recorder.record(self.get_audio_format(),
                                         chunk_duration_sec=self.audio_recording_chunk_duration_sec):
                if self.handle_button_press(status, chunks_deque):
                    status = RecognitionStatus.LISTENING

                if self.handle_button_release(status):
                    status = RecognitionStatus.PROCESSING
                    record_more = self.post_button_release_record_chunks

                if self.should_record(status, record_more):
                    chunks_deque.append(chunk)
                    if status == RecognitionStatus.IDLE and len(chunks_deque) > 3:
                        chunks_deque.popleft()

                if status != RecognitionStatus.IDLE:
                    yield chunks_deque.popleft()

                if status == RecognitionStatus.PROCESSING:
                    record_more -= 1

    def get_audio_format(self) -> AudioFormat:
        """
        Get the audio format for recording.

        Returns:
            AudioFormat: The audio format configuration.
        """
        return AudioFormat(sample_rate_hz=self.audio_sample_rate, num_channels=1, bytes_per_sample=2)

    def handle_button_press(self, status: RecognitionStatus, chunks_deque: deque) -> bool:
        """
        Handle button press event.

        Args:
            status (RecognitionStatus): Current recognition status.
            chunks_deque (deque): Deque of audio chunks.

        Returns:
            bool: True if the status should change to LISTENING, False otherwise.
        """
        if status == RecognitionStatus.IDLE and self.button_is_pressed:
            chunks_deque.clear()
            return True
        return False

    def handle_button_release(self, status: RecognitionStatus) -> bool:
        """
        Handle button release event.

        Args:
            status (RecognitionStatus): Current recognition status.

        Returns:
            bool: True if the status should change to PROCESSING, False otherwise.
        """
        return status == RecognitionStatus.LISTENING and not self.button_is_pressed

    def should_record(self, status: RecognitionStatus, record_more: int) -> bool:
        """
        Determine if audio should be recorded.

        Args:
            status (RecognitionStatus): Current recognition status.
            record_more (int): Number of additional chunks to record after button release.

        Returns:
            bool: True if audio should be recorded, False otherwise.
        """
        return status != RecognitionStatus.IDLE or (status == RecognitionStatus.PROCESSING and record_more > 0)

    def setup_button_callbacks(self) -> None:
        """Set up callbacks for button press and release events."""
        self.button.when_pressed = self.button_pressed
        self.button.when_released = self.button_released

    def button_pressed(self) -> None:
        """Callback function for button press event."""
        self.button_is_pressed = True
        logger.debug('Button pressed')

    def button_released(self) -> None:
        """Callback function for button release event."""
        self.button_is_pressed = False
        logger.debug('Button released')

    def generate_speech(self, text: str, output_filename: str) -> None:
        """
        Generate speech from text and save it to a file.

        This method uses the TTS engine to synthesize speech.

        Args:
            text (str): The text to convert to speech.
            output_filename (str): The filename to save the generated audio.
        """
        synthesize_speech(self.tts_engine, text, output_filename, self.config)
