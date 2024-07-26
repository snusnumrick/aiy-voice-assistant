"""
Text-to-Speech (TTS) Engine module.

This module provides abstract and concrete implementations of TTS engines,
including OpenAI's TTS model, Google's Text-to-Speech, Yandex SpeechKit, and ElevenLabs.
It defines a common interface for TTS engines and implements various concrete classes
that interact with different TTS services. The module supports multiple languages
and provides both synchronous and asynchronous synthesis methods.

Classes:
    TTSEngine (ABC): Abstract base class for all TTS engines.
    OpenAITTSEngine: Implementation using OpenAI's TTS model.
    GoogleTTSEngine: Implementation using Google's Text-to-Speech.
    YandexTTSEngine: Implementation using Yandex SpeechKit.
    ElevenLabsTTSEngine: Implementation using ElevenLabs API.

Enums:
    Tone: Enumeration of speech tones.
    Language: Enumeration of supported languages.
    AudioFormat: Enumeration of supported audio formats.
    HTTPStatus: Enumeration of relevant HTTP status codes.
"""

import asyncio
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from typing import Optional, List, Dict, Any

import aiofiles
import aiohttp
import requests
from pydub import AudioSegment
from speechkit import model_repository

from src.config import Config
from src.tools import retry_async

# Set up logging
logger = logging.getLogger(__name__)


class Tone(Enum):
    """Enumeration of speech tones."""
    PLAIN = 0
    HAPPY = 1


class Language(Enum):
    """Enumeration of supported languages."""
    RUSSIAN = 0
    ENGLISH = 1
    GERMAN = 2


class AudioFormat(Enum):
    """Enumeration of supported audio formats."""
    WAV = "audio/wav"
    MP3 = "audio/mpeg"


class HTTPStatus(IntEnum):
    """Enumeration of relevant HTTP status codes."""
    OK = 200
    TOO_MANY_REQUESTS = 429


class TTSEngine(ABC):
    """
    Abstract base class for Text-to-Speech engines.
    Defines the interface that all concrete TTS engine implementations must follow.
    """

    @abstractmethod
    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> None:
        """
        Synthesize speech from text and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use for speech synthesis.
            lang (Language): The language to use for speech synthesis.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        pass

    @abstractmethod
    def max_text_length(self) -> int:
        """
        Get the maximum allowed text length for synthesis.

        Returns:
            int: The maximum number of characters allowed, or -1 if there's no limit.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        return -1

    @abstractmethod
    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str, tone: Tone = Tone.PLAIN,
                               lang=Language.RUSSIAN) -> bool:
        """
        Asynchronously synthesize speech from text and save it to a file.

        Args:
            session (aiohttp.ClientSession): An aiohttp client session for making HTTP requests.
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use for speech synthesis.
            lang (Language): The language to use for speech synthesis.

        Returns:
            bool: True if synthesis was successful, False otherwise.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        pass


class OpenAITTSEngine(TTSEngine):
    """
    Implementation of TTSEngine using OpenAI's TTS model.
    This class provides methods for text-to-speech synthesis using the OpenAI API.
    """

    def __init__(self, config):
        """
        Initialize the OpenAI TTS engine.

        Args:
            config (Config): The application configuration object containing OpenAI-specific settings.
        """
        from openai import OpenAI
        self.client = OpenAI()
        self.model = config.get('openai_tts_model', 'tts-1')
        self.voice = config.get('openai_tts_voice', 'alloy')

    def max_text_length(self) -> int:
        """
        Get the maximum allowed text length for OpenAI's TTS API.

        Returns:
            int: The maximum number of characters allowed (4096 for OpenAI).
        """
        return 4096

    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> None:
        """
        Synthesize speech using OpenAI's TTS model and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use (not used in OpenAI API).
            lang (Language): The language to use (not used in OpenAI API).

        Raises:
            Exception: If the OpenAI API request fails.
        """
        response = self.client.with_streaming_response.audio.speech.create(model=self.model, voice=self.voice,
                                                                           response_format="wav", input=text)
        response.stream_to_file(filename)

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str, tone: Tone = Tone.PLAIN,
                               lang=Language.RUSSIAN) -> bool:
        """
        Asynchronously synthesize speech using OpenAI's TTS model and save it to a file.

        Args:
            session (aiohttp.ClientSession): An aiohttp client session for making HTTP requests.
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use (not used in OpenAI API).
            lang (Language): The language to use (not used in OpenAI API).

        Returns:
            bool: True if synthesis was successful, False otherwise.

        Raises:
            Exception: If the OpenAI API request fails.
        """
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

    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> None:
        """
        Synthesize speech using Google's Text-to-Speech and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use.
            lang (Language): The language to use.
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

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str, tone: Tone = Tone.PLAIN,
                               lang=Language.RUSSIAN) -> bool:
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
        from speechkit import configure_credentials, creds

        # Try to get the API key from environment variable first, then fall back to config
        self.api_key = os.environ.get('YANDEX_API_KEY') or config.get('yandex_api_key')
        if not self.api_key:
            raise ValueError("Yandex API key is not provided in environment variables or configuration")

        # Configure credentials
        configure_credentials(yandex_credentials=creds.YandexCredentials(api_key=self.api_key))

        self.langs = {Language.RUSSIAN: "ru-RU", Language.ENGLISH: "en-GB", Language.GERMAN: "de-DE"}
        self.lang_voices = {Language.RUSSIAN: config.get('yandex_tts_voice_russian', 'ermil'),
                            Language.ENGLISH: config.get('yandex_tts_voice_english', 'john'),
                            Language.GERMAN: config.get('yandex_tts_voice_german', 'lea')}
        self.roles = {Tone.PLAIN: config.get('yandex_tts_role_plain', 'neutral'),
                      Tone.HAPPY: config.get('yandex_tts_role_happy', 'good')}
        self.role_plain = config.get('yandex_tts_role_plain', 'neutral')
        self.role_happy = config.get('yandex_tts_role_happy', 'good')
        self.speed = config.get('yandex_tts_speed', 1.0)

    def voice_model(self, tone=Tone.PLAIN, lang=Language.RUSSIAN):
        model = model_repository.synthesis_model()
        model.voice = self.lang_voices[lang]
        if lang == Language.RUSSIAN:
            model.role = self.roles[tone]
        model.language = self.langs[lang]
        model.speed = self.speed
        return model

    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> None:
        """
        Synthesize speech using Yandex SpeechKit and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (tone): The tone to use.
            lang (Language): The language to use.
        """
        model = self.voice_model(tone=tone, lang=lang)
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

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str, tone: Tone = Tone.PLAIN,
                               lang=Language.RUSSIAN) -> bool:
        # Yandex SpeechKit doesn't have an async API, so we'll run it in an executor

        def synthesize_wrapper(par: dict) -> bytes:
            """Wrapper method to call synthesize with the correct parameters."""
            return par["model"].synthesize(par["text"], raw_format=True)

        loop = asyncio.get_event_loop()
        args = {"model": self.voice_model(tone=tone, lang=lang), "text": text}
        logger.info(f"Synthesizing text: {text[:50]}...")
        result = await loop.run_in_executor(None, synthesize_wrapper, args)

        logger.info(f"Audio content being written to file {filename}")
        async with aiofiles.open(filename, "wb") as out:
            await out.write(result)

        return True


class ElevenLabsAPIError(Exception):
    """Custom exception for ElevenLabs API errors."""
    pass


class ElevenLabsTTSEngine(TTSEngine):
    """
    Implementation of TTSEngine using ElevenLabs API.
    This class provides methods for text-to-speech synthesis using ElevenLabs,
    including support for multiple languages and voice settings.
    """

    def __init__(self, config: Config):
        """
        Initialize the ElevenLabs TTS engine.

        Args:
            config (Config): Configuration object containing ElevenLabs-specific settings.

        Raises:
            ValueError: If the API key is missing or if required configuration is invalid.
        """
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key is not provided in environment variables or configuration")

        # Initialize voice IDs for different languages
        self.voice_ids = {Language.ENGLISH: config.get('elevenlabs_voice_id_en', 'N2lVS1w4EtoT3dr4eOWO'),
                          Language.GERMAN: config.get('elevenlabs_voice_id_de', 'Ay1WwRHxUsu3hEeAp8JZ'),
                          Language.RUSSIAN: config.get('elevenlabs_voice_id_ru', 'cjVigY5qzO86Huf0OWal'), }
        if not all(self.voice_ids.values()):
            raise ValueError("Voice IDs for all languages must be provided in the configuration")

        # Set up voice parameters
        self.stability = config.get('elevenlabs_stability', 0.5)
        self.similarity_boost = config.get('elevenlabs_similarity_boost', 0.75)
        self.style = config.get('elevenlabs_style', 0.0)
        self.use_speaker_boost = config.get('elevenlabs_use_speaker_boost', False)

        # Configure retry mechanism
        self.max_retries = config.get('elevenlabs_max_retries', 5)
        self.initial_retry_delay = config.get('elevenlabs_initial_retry_delay', 1)
        self.jitter_factor = config.get('elevenlabs_retry_jitter_factor', 0.1)

        # Set up API-specific parameters
        self.query = '{"output_format":config.get("elevenlabs_output_format", "mp3_22050_32")}'
        self.model_id = config.get('elevenlabs_model', "eleven_multilingual_v2")
        self.base_url = "https://api.elevenlabs.io/v1"

        # Initialize voice IDs for different languages
        self._validate_config()

        # Validate API key asynchronously
        asyncio.get_event_loop().run_until_complete(self._validate_api_key())

    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.

        Raises:
            ValueError: If any of the configuration parameters are invalid.
        """
        if not isinstance(self.stability, float) or not 0 <= self.stability <= 1:
            raise ValueError("stability must be a float between 0 and 1")
        if not isinstance(self.similarity_boost, float) or not 0 <= self.similarity_boost <= 1:
            raise ValueError("similarity_boost must be a float between 0 and 1")
        if not isinstance(self.style, float) or not 0 <= self.style <= 1:
            raise ValueError("style must be a float between 0 and 1")

    async def _validate_api_key(self) -> None:
        """
        Validate the API key by making a test request to the ElevenLabs API.

        Raises:
            ElevenLabsAPIError: If the API key is invalid or the connection to the API fails.
        """
        url = f"{self.base_url}/user"
        headers = {"xi-api-key": self.api_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise ElevenLabsAPIError("Invalid API key or unable to connect to ElevenLabs API")

    @retry_async()
    async def _get_history_items_async(self, session: aiohttp.ClientSession, voice_id: str) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieve history items for a specific voice from ElevenLabs API.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session to use for the request.
            voice_id (str): The ID of the voice to retrieve history for.

        Returns:
            List[Dict]: A list of history items, each as a dictionary.

        Raises:
            ElevenLabsAPIError: If there's an error in the API request or response.
        """

        url = f"{self.base_url}/history"
        headers = {"xi-api-key": self.api_key}
        params = {"voice_id": voice_id}

        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("history", [])
            else:
                raise ElevenLabsAPIError(f"Failed to get history items: {response.status} - {await response.text()}")

    def _get_history_items(self, voice_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve history items for a specific voice from ElevenLabs API.

        Args:
            voice_id (str): The ID of the voice to retrieve history for.

        Returns:
            List[Dict[str, Any]]: A list of history items, each as a dictionary.

        Raises:
            ElevenLabsAPIError: If there's an error in the API request or response.
        """
        url = f"{self.base_url}/history"
        headers = {"xi-api-key": self.api_key}
        params = {"voice_id": voice_id}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get("history", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get history items: {response.status_code} - {response.text}")
            raise ElevenLabsAPIError(f"Failed to get history items: {str(e)}")

    async def _find_matching_history_item_async(self, session: aiohttp.ClientSession, text: str, voice_id: str) -> \
            Optional[str]:
        """
        Asynchronously find a matching history item for the given text and voice ID.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session to use for the request.
            text (str): The text to search for in the history.
            voice_id (str): The voice ID to match in the history.

        Returns:
            Optional[str]: The history item ID if a match is found, None otherwise.
        """
        history_items = await self._get_history_items_async(session, voice_id)
        for item in history_items:
            if item.get("text") == text and item.get("voice_id") == voice_id:
                return item.get("history_item_id")
        return None

    def _find_matching_history_item(self, text: str, voice_id: str) -> Optional[str]:
        """
        Find a matching history item for the given text and voice ID.

        This method searches through the history items to find an exact match
        for the text and voice ID combination.

        Args:
            text (str): The text to search for in the history.
            voice_id (str): The voice ID to match in the history.

        Returns:
            Optional[str]: The history item ID if a match is found, None otherwise.
        """
        history_items = self._get_history_items(voice_id)
        for item in history_items:
            if item.get("text") == text and item.get("voice_id") == voice_id:
                return item.get("history_item_id")
        return None

    @retry_async()
    async def _download_audio_async(self, session: aiohttp.ClientSession, history_item_id: str, filename: str) -> None:
        """
        Asynchronously download audio file for a specific history item.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session to use for the request.
            history_item_id (str): The ID of the history item to download.
            filename (str): The path to save the downloaded audio file.

        Raises:
            Exception: If the download fails.
        """
        url = f"{self.base_url}/history/{history_item_id}/audio"
        headers = {"xi-api-key": self.api_key}

        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                content = await response.read()
                async with aiofiles.open(filename, mode='wb') as f:
                    await f.write(content)
            else:
                raise ElevenLabsAPIError(f"Failed to download audio: {response.status} - {await response.text()}")

    def _download_audio(self, history_item_id: str, filename: str) -> None:
        """
        Download audio file for a specific history item.

        Args:
            history_item_id (str): The ID of the history item to download.
            filename (str): The path to save the downloaded audio file.

        Raises:
            ElevenLabsAPIError: If there's an error in the API request or response.
        """
        url = f"{self.base_url}/history/{history_item_id}/audio"
        headers = {"xi-api-key": self.api_key}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            raise ElevenLabsAPIError(f"Failed to download audio: {str(e)}")

    @retry_async()
    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str, tone: Tone = Tone.PLAIN,
                               lang: Language = Language.RUSSIAN) -> bool:
        """
        Asynchronously synthesize speech using ElevenLabs API and save it to a file.

        This method first attempts to check if the audio already exists in the history.
        If history check fails, it proceeds with synthesis. If existing audio is found, it's downloaded.
        Otherwise, it generates new audio.

        Args:
            session (aiohttp.ClientSession): An aiohttp client session for making HTTP requests.
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use for speech synthesis (currently not used in API call).
            lang (Language): The language to use for speech synthesis.

        Returns:
            bool: True if synthesis was successful, False otherwise.

        Raises:
            ValueError: If the input text is empty or exceeds the maximum length.
            ElevenLabsAPIError: If there's an error in the API request or response during synthesis.
        """
        if not text:
            raise ValueError("Text cannot be empty")
        if len(text) > self.max_text_length():
            raise ValueError(f"Text exceeds maximum length of {self.max_text_length()} characters")

        voice_id = self.voice_ids.get(lang)
        if not voice_id:
            raise ValueError(f"No voice ID configured for language: {lang}")

        # Attempt to check history, but continue if it fails
        try:
            history_item_id = await self._find_matching_history_item_async(session, text, voice_id)
            if history_item_id:
                logger.info(f"Found existing audio for text: {text[:30]}...")
                mp3_filename = f"{filename}.mp3"
                await self._download_audio_async(session, history_item_id, mp3_filename)
                await self._convert_mp3_to_wav_async(mp3_filename, filename)
                return True
        except Exception as e:
            logger.warning(
                f"Failed to check history or download existing audio: {str(e)}. Proceeding with new synthesis.")

        # If no existing audio found or history check failed, proceed with synthesis
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        headers = {"Content-Type": "application/json", "xi-api-key": self.api_key}
        data = {"text": text, "model_id": self.model_id,
            "voice_settings": {"stability": self.stability, "similarity_boost": self.similarity_boost}}

        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=data, headers=headers, params=self.query) as response:
                    if response.status == 200:
                        mp3_filename = f"{filename}.mp3"
                        audio_content = await response.read()
                        async with aiofiles.open(mp3_filename, mode='wb') as f:
                            await f.write(audio_content)
                        await self._convert_mp3_to_wav_async(mp3_filename, filename)
                        return True
                    elif response.status == HTTPStatus.TOO_MANY_REQUESTS.value:
                        retry_after = int(
                            response.headers.get('Retry-After', self.initial_retry_delay * (2 ** attempt)))
                        retry_time = self._get_retry_time(attempt, retry_after)
                        logger.warning(f"Too many requests, retrying after {retry_time:.2f} seconds...")
                        await asyncio.sleep(retry_time)
                    else:
                        raise ElevenLabsAPIError(
                            f"Error from ElevenLabs API: {response.status} - {await response.text()}")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return False
                logger.warning(f"An error occurred: {str(e)}. Retrying...")
                await asyncio.sleep(self.initial_retry_delay * (2 ** attempt))

        return False

    @staticmethod
    async def _convert_mp3_to_wav_async(mp3_file: str, wav_file: str) -> None:
        """Asynchronously convert an MP3 file to WAV format."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, AudioSegment.from_mp3(mp3_file).export, wav_file, "wav")
        finally:
            if os.path.exists(mp3_file):
                os.remove(mp3_file)

    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN, lang: Language = Language.RUSSIAN) -> None:
        """
        Synthesize speech using ElevenLabs API and save it to a file.

        This method first attempts to check if the audio already exists in the history.
        If history check fails, it proceeds with synthesis. If existing audio is found, it's downloaded.
        Otherwise, it generates new audio.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use for speech synthesis (currently not used in API call).
            lang (Language): The language to use for speech synthesis.

        Raises:
            ValueError: If the input text is empty or exceeds the maximum length.
            ElevenLabsAPIError: If there's an error in the API request or response during synthesis.
        """
        if not text:
            raise ValueError("Text cannot be empty")
        if len(text) > self.max_text_length():
            raise ValueError(f"Text exceeds maximum length of {self.max_text_length()} characters")

        voice_id = self.voice_ids.get(lang)
        if not voice_id:
            raise ValueError(f"No voice ID configured for language: {lang}")

        # Attempt to check history, but continue if it fails
        try:
            history_item_id = self._find_matching_history_item(text, voice_id)
            if history_item_id:
                logger.info(f"Found existing audio for text: {text[:30]}...")
                self._download_audio(history_item_id, filename + ".mp3")
                self._convert_mp3_to_wav(filename + ".mp3", filename)
                os.remove(filename + ".mp3")
                return
        except Exception as e:
            logger.warning(
                f"Failed to check history or download existing audio: {str(e)}. Proceeding with new synthesis.")

        # If no existing audio found or history check failed, proceed with synthesis
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        headers = {"Content-Type": "application/json", "xi-api-key": self.api_key}
        data = {"text": text, "model_id": self.model_id,
            "voice_settings": {"stability": self.stability, "similarity_boost": self.similarity_boost}}

        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=data, headers=headers, params=self.query)
                response.raise_for_status()

                mp3_filename = filename + ".mp3"
                with open(mp3_filename, "wb") as f:
                    f.write(response.content)

                self._convert_mp3_to_wav(mp3_filename, filename)
                os.remove(mp3_filename)
                return

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                    retry_after = int(e.response.headers.get('Retry-After', self.initial_retry_delay * (2 ** attempt)))
                    retry_time = self._get_retry_time(attempt, retry_after)
                    logger.warning(f"Too many requests, retrying after {retry_time:.2f} seconds...")
                    time.sleep(retry_time)
                elif attempt == self.max_retries - 1:
                    raise ElevenLabsAPIError(f"HTTP error occurred: {e}")
                else:
                    logger.warning(f"HTTP error occurred: {e}. Retrying...")
                    time.sleep(self.initial_retry_delay * (2 ** attempt))

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise ElevenLabsAPIError(f"Request failed: {e}")
                else:
                    logger.warning(f"Request failed: {e}. Retrying...")
                    time.sleep(self.initial_retry_delay * (2 ** attempt))

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise ElevenLabsAPIError(f"An unexpected error occurred: {e}")
                else:
                    logger.warning(f"An unexpected error occurred: {e}. Retrying...")
                    time.sleep(self.initial_retry_delay * (2 ** attempt))

        raise ElevenLabsAPIError(f"Failed to synthesize speech after {self.max_retries} attempts")

    def _get_retry_time(self, attempt: int, retry_after: Optional[int] = None) -> float:
        """
        Calculate the retry time with exponential backoff and jitter.

        Args:
            attempt (int): The current attempt number.
            retry_after (Optional[int]): The 'Retry-After' time suggested by the server, if any.

        Returns:
            float: The calculated retry time in seconds.
        """
        if retry_after is not None:
            base_delay = retry_after
        else:
            base_delay = self.initial_retry_delay * (2 ** attempt)

        jitter = random.uniform(0, self.jitter_factor * base_delay)
        return base_delay + jitter

    def max_text_length(self) -> int:
        """
        Get the maximum allowed text length for ElevenLabs API.

        Returns:
            int: The maximum number of characters allowed (2500 for ElevenLabs).
        """
        return 2500  # ElevenLabs has a limit of 2500 characters per request

    @staticmethod
    def _convert_mp3_to_wav(mp3_file: str, wav_file: str) -> None:
        """
        Convert an MP3 file to WAV format.

        Args:
            mp3_file (str): Path to the input MP3 file.
            wav_file (str): Path to save the output WAV file.
        """
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")


async def main():
    """
    Main function for testing the ElevenLabsTTSEngine.
    """
    config = Config()
    engine = ElevenLabsTTSEngine(config)
    async with aiohttp.ClientSession() as session:
        await engine.synthesize_async(session, "to be or not to be", "test.wav")


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()

    asyncio.run(main())
