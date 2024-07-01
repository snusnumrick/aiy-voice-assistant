"""
Text-to-Speech (TTS) Engine module.

This module provides abstract and concrete implementations of TTS engines,
including OpenAI's TTS model and Google's Text-to-Speech.
"""

from abc import ABC, abstractmethod
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """
    Abstract base class for Text-to-Speech engines.
    """

    @abstractmethod
    def synthesize(self, text: str, filename: str) -> None:
        """
        Synthesize speech from text and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
        """
        pass


class OpenAITTSEngine(TTSEngine):
    """
    Implementation of TTSEngine using OpenAI's TTS model.
    """

    def __init__(self, config):
        """
        Initialize the OpenAI TTS engine.

        Args:
            config (Config): The application configuration object.
        """
        from openai import OpenAI
        self.client = OpenAI()
        self.model = config.get('openai_tts_model', 'tts-1')
        self.voice = config.get('openai_tts_voice', 'alloy')

    def synthesize(self, text: str, filename: str) -> None:
        """
        Synthesize speech using OpenAI's TTS model and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
        """
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            response_format="wav",
            input=text
        )
        response.stream_to_file(filename)


class GoogleTTSEngine(TTSEngine):
    """
    Implementation of TTSEngine using Google's Text-to-Speech.
    """

    def __init__(self, config):
        """
        Initialize the Google TTS engine.

        Args:
            config (Config): The application configuration object.
        """
        from google.oauth2 import service_account
        from google.cloud import texttospeech

        service_account_file = config.get('google_service_account_file', '~/gcloud.json')
        service_account_file = os.path.expanduser(service_account_file)

        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        self.client = texttospeech.TextToSpeechClient(credentials=credentials)
        self.language_code = config.get('google_tts_language', 'ru-RU')
        self.voice = config.get('google_tts_voice', 'ru-RU-Wavenet-A')
        self.audio_encoding = texttospeech.AudioEncoding.LINEAR16

    def synthesize(self, text: str, filename: str) -> None:
        """
        Synthesize speech using Google's Text-to-Speech and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
        """
        from google.cloud import texttospeech

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=self.audio_encoding
        )

        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with open(filename, "wb") as out:
            out.write(response.audio_content)

        logger.debug(f"Audio content written to file {filename}")
