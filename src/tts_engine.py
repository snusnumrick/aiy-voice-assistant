"""
Text-to-Speech (TTS) Engine module.

This module provides abstract and concrete implementations of TTS engines,
including OpenAI's TTS model and Google's Text-to-Speech.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum

import aiofiles
import aiohttp

# Set up logging
logger = logging.getLogger(__name__)


class Tone(Enum):
    PLAIN = 0
    HAPPY = 1


class TTSEngine(ABC):
    """
    Abstract base class for Text-to-Speech engines.
    """

    @abstractmethod
    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN) -> None:
        """
        Synthesize speech from text and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use.
        """
        pass

    @abstractmethod
    def max_text_length(self) -> int:
        return -1

    @abstractmethod
    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str,
                               tone: Tone = Tone.PLAIN) -> bool:
        """
        Asynchronously synthesize speech from text and save it to a file.

        Args:
            session (aiohttp.ClientSession): An aiohttp client session for making HTTP requests.
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use.

        Returns:
            bool: True if synthesis was successful, False otherwise.
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

    def max_text_length(self) -> int:
        return 4096

    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN) -> None:
        """
        Synthesize speech using OpenAI's TTS model and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use.
        """
        response = self.client.with_streaming_response.audio.speech.create(model=self.model, voice=self.voice,
                                                                           response_format="wav", input=text)
        response.stream_to_file(filename)

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str,
                               tone: Tone = Tone.PLAIN) -> bool:
        url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {self.client.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "input": text, "voice": self.voice, "response_format": "wav"}

        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    async with aiofiles.open(filename, mode='wb') as f:
                        await f.write(await response.read())
                else:
                    raise Exception(f"OpenAI API request failed with status {response.status}")
        except Exception as e:
            logger.error(f"Error in speech synthesis: {str(e)}")
            return False

        return True


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

        credentials = service_account.Credentials.from_service_account_file(service_account_file, scopes=[
            "https://www.googleapis.com/auth/cloud-platform"])

        self.client = texttospeech.TextToSpeechClient(credentials=credentials)
        self.language_code = config.get('google_tts_language', 'ru-RU')
        self.voice = config.get('google_tts_voice', 'ru-RU-Wavenet-A')
        self.audio_encoding = texttospeech.AudioEncoding.LINEAR16

    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN) -> None:
        """
        Synthesize speech using Google's Text-to-Speech and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use.
        """
        from google.cloud import texttospeech

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(language_code=self.language_code, name=self.voice)

        audio_config = texttospeech.AudioConfig(audio_encoding=self.audio_encoding)

        response = self.client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        with open(filename, "wb") as out:
            out.write(response.audio_content)

        logger.debug(f"Audio content written to file {filename}")

    def max_text_length(self) -> int:
        return -1

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str,
                               tone: Tone = Tone.PLAIN) -> bool:
        from google.cloud import texttospeech

        # Google Cloud TTS doesn't have an async API, so we'll run it in an executor
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=self.language_code, name=self.voice)
        audio_config = texttospeech.AudioConfig(audio_encoding=self.audio_encoding)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.client.synthesize_speech, synthesis_input, voice, audio_config)

        async with aiofiles.open(filename, "wb") as out:
            await out.write(response.audio_content)

        logger.debug(f"Audio content written to file {filename}")
        return True


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
        from speechkit.tts import SynthesisConfig, AudioEncoding

        # Try to get the API key from environment variable first, then fall back to config
        self.api_key = os.environ.get('YANDEX_API_KEY') or config.get('yandex_api_key')
        if not self.api_key:
            raise ValueError("Yandex API key is not provided in environment variables or configuration")

        # Configure credentials
        configure_credentials(yandex_credentials=creds.YandexCredentials(api_key=self.api_key))

        self.voice = config.get('yandex_tts_voice', 'ermil')
        self.role_plain = config.get('yandex_tts_role_plain', 'neutral')
        self.role_happy = config.get('yandex_tts_role_happy', 'good')
        self.language_code = config.get('yandex_tts_language', 'ru-RU')
        self.speed = config.get('yandex_tts_speed', 1.0)

        # Initialize the synthesis model
        self.model_plain = model_repository.synthesis_model()
        self.model_plain.voice = self.voice
        self.model_plain.role = self.role_plain
        self.model_plain.language = self.language_code
        self.model_plain.speed = self.speed

        self.model_happy = model_repository.synthesis_model()
        self.model_happy.voice = self.voice
        self.model_happy.role = self.role_happy
        self.model_happy.language = self.language_code
        self.model_happy.speed = self.speed

        self.synthesis_config = SynthesisConfig(
            audio_encoding=AudioEncoding.WAV,
            voice=self.voice
        )

        logger.info(
            f"Initialized Yandex TTS Engine with language {self.language_code}, voice {self.voice}")

    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN) -> None:
        """
        Synthesize speech using Yandex SpeechKit and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (tone): The tone to use.
        """
        model = self.model_happy if tone == Tone.PLAIN else self.model_plain
        try:
            logger.debug(f"Synthesizing text: {text[:50]}...")
            result = model.synthesize(text, raw_format=False)
            result.export(filename, 'wav')
            logger.debug(f"Audio content written to file {filename}")
        except Exception as e:
            logger.error(f"Error during speech synthesis: {str(e)}")
            raise

    def max_text_length(self) -> int:
        return -1

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str,
                               tone: Tone = Tone.PLAIN) -> bool:
        # Yandex SpeechKit doesn't have an async API, so we'll run it in an executor

        def synthesize_wrapper(par: dict) -> bytes:
            """Wrapper method to call synthesize with the correct parameters."""
            return par["model"].synthesize(
                par["text"],
                raw_format=True
            )

        loop = asyncio.get_event_loop()
        model = self.model_plain if tone == Tone.HAPPY else self.model_plain
        result = await loop.run_in_executor(None, synthesize_wrapper, {"model": model, "text":text})

        async with aiofiles.open(filename, "wb") as out:
            await out.write(result)

        return True
