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


class YandexTTSEngine(TTSEngine):
    """
    Implementation of TTSEngine using Yandex SpeechKit via the speechkit module.
    """

    def __init__(self, config):
        """
        Initialize the Yandex TTS engine.

        Args:
            config (Config): The application configuration object.
        """
        from speechkit import model_repository, configure_credentials, creds

        # Try to get the API key from environment variable first, then fall back to config
        self.api_key = os.environ.get('YANDEX_API_KEY') or config.get('yandex_api_key')
        if not self.api_key:
            raise ValueError("Yandex API key is not provided in environment variables or configuration")

        # Configure credentials
        configure_credentials(
            yandex_credentials=creds.YandexCredentials(
                api_key=self.api_key
            )
        )

        self.voice = config.get('yandex_tts_voice', 'ermil')
        self.role = config.get('yandex_tts_role', 'good')
        self.language_code = config.get('yandex_tts_language', 'ru-RU')
        self.speed = config.get('yandex_tts_speed', 1.0)

        # Initialize the synthesis model
        self.model = model_repository.synthesis_model()
        self.model.voice = self.voice
        self.model.role = self.role
        self.model.language = self.language_code
        self.model.speed = self.speed

        logger.info(f"Initialized Yandex TTS Engine with language {self.language_code}, voice {self.voice}, and role {self.role}")

    def synthesize(self, text: str, filename: str) -> None:
        """
        Synthesize speech using Yandex SpeechKit and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
        """
        try:
            logger.debug(f"Synthesizing text: {text[:50]}...")
            result = self.model.synthesize(text, raw_format=False)
            result.export(filename, 'wav')
            logger.info(f"Audio content written to file {filename}")
        except Exception as e:
            logger.error(f"Error during speech synthesis: {str(e)}")
            raise
