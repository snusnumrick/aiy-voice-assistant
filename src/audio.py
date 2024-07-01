"""
Audio processing module.

This module provides functionality for speech transcription and synthesis,
interfacing with the Google Cloud Speech-to-Text API and various TTS engines.
"""

from typing import Optional
from subprocess import Popen
from aiy.board import Button
from aiy.leds import Leds, Color, Pattern
from aiy.voice.audio import AudioFormat, Recorder
from google.cloud import speech
import logging
import os

logger = logging.getLogger(__name__)


class SpeechTranscriber:
    """
    A class to handle speech transcription using Google Cloud Speech-to-Text API.

    This class manages the button interface, LED indicators, and the transcription process.

    Attributes:
        button (Button): The AIY Kit button object.
        leds (Leds): The AIY Kit LED object for visual feedback.
        config (Config): The application configuration object.
        button_is_pressed (bool): Flag to track button press state.
        speech_client (speech.SpeechClient): Google Cloud Speech client.
        streaming_config (speech.StreamingRecognitionConfig): Configuration for streaming recognition.
    """

    def __init__(self, button: Button, leds: Leds, config):
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
        self.setup_speech_client()

    def setup_speech_client(self):
        """
        Set up the Google Cloud Speech client and streaming configuration.
        """
        from google.oauth2 import service_account
        service_account_file = self.config.get('service_account_file', '~/gcloud.json')
        service_account_file = os.path.expanduser(service_account_file)
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        self.speech_client = speech.SpeechClient(credentials=credentials)
        self.setup_streaming_config()

    def setup_streaming_config(self):
        """
        Configure the streaming recognition settings.
        """
        speech_language_code = self.config.get("language_code", "ru-RU")
        speech_sample_rate_hertz = self.config.get("sample_rate_hertz", 16000)
        config = speech.types.RecognitionConfig(
            encoding=speech.types.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=speech_sample_rate_hertz,
            language_code=self.config.get("language_code", speech_language_code),
            enable_automatic_punctuation=True
        )
        self.streaming_config = speech.types.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False
        )

    def transcribe_speech(self, player_process: Optional[Popen] = None) -> str:
        """
        Transcribe speech from the microphone input, including pre and post buffering.

        Args:
            player_process (Optional[Popen]): A subprocess.Popen object representing a running audio player.

        Returns:
            str: The transcribed text.
        """
        self.setup_button_callbacks()
        logger.info('Press the button and speak')
        self.wait_for_button_press()

        if player_process:
            player_process.terminate()

        text = ""
        with Recorder() as recorder:
            chunks = deque(maxlen=int(self.config.get('pre_buffer_seconds', 0.5) / 0.1))

            def generate_audio_chunks():
                nonlocal chunks
                audio_format = AudioFormat(sample_rate_hz=16000, num_channels=1, bytes_per_sample=2)

                # Pre-buffering
                for chunk in recorder.record(audio_format, chunk_duration_sec=0.1):
                    chunks.append(chunk)
                    if self.button_is_pressed:
                        break

                # Main recording
                self.leds.update(Leds.rgb_on(Color.GREEN))
                logger.debug('Listening...')
                while self.button_is_pressed:
                    chunk = next(recorder.record(audio_format, chunk_duration_sec=0.1))
                    chunks.append(chunk)
                    yield chunk

                # Post-buffering
                post_buffer_chunks = int(self.config.get('post_buffer_seconds', 0.5) / 0.1)
                for _ in range(post_buffer_chunks):
                    chunk = next(recorder.record(audio_format, chunk_duration_sec=0.1))
                    chunks.append(chunk)
                    yield chunk

                self.leds.update(Leds.rgb_off())

            audio_generator = generate_audio_chunks()
            requests = (speech.types.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
            responses = self.speech_client.streaming_recognize(self.streaming_config, requests)

            for response in responses:
                for result in response.results:
                    if result.is_final:
                        text += result.alternatives[0].transcript

        return text

    def setup_button_callbacks(self):
        """
        Set up callbacks for button press and release events.
        """
        self.button.when_pressed = self.button_pressed
        self.button.when_released = self.button_released

    def button_pressed(self):
        """
        Callback function for button press event.
        """
        self.button_is_pressed = True
        logger.debug('Button pressed')

    def button_released(self):
        """
        Callback function for button release event.
        """
        self.button_is_pressed = False
        logger.debug('Button released')

    def wait_for_button_press(self):
        """
        Wait for the button to be pressed, with visual LED feedback.
        """
        self.leds.pattern = Pattern.breathe(10000)
        self.leds.update(Leds.rgb_pattern((0, 1, 0)))
        self.button.wait_for_press()
        self.leds.update(Leds.rgb_off())


def synthesize_speech(engine, text: str, filename: str, config):
    """
    Synthesize speech from text and save it to a file.

    Args:
        engine: The TTS engine to use for synthesis.
        text (str): The text to synthesize.
        filename (str): The filename to save the synthesized audio.
        config (Config): The application configuration object.
    """
    logger.debug('Synthesizing speech for: %s', text)
    engine.synthesize(text, filename)
    logger.debug(f"Saved at {filename}")
