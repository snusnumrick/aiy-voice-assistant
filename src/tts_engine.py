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
from speechkit import model_repository
import requests
from pydub import AudioSegment

# Set up logging
logger = logging.getLogger(__name__)

from src.config import Config


class Tone(Enum):
    PLAIN = 0
    HAPPY = 1


class Language(Enum):
    RUSSIAN = 0
    ENGLISH = 1
    GERMAN = 2


class TTSEngine(ABC):
    """
    Abstract base class for Text-to-Speech engines.
    """

    @abstractmethod
    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> None:
        """
        Synthesize speech from text and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use.
            lang (Language): The language to use.
        """
        pass

    @abstractmethod
    def max_text_length(self) -> int:
        return -1

    @abstractmethod
    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str,
                               tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> bool:
        """
        Asynchronously synthesize speech from text and save it to a file.

        Args:
            session (aiohttp.ClientSession): An aiohttp client session for making HTTP requests.
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use.
            lang (str): The language to use.

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

    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> None:
        """
        Synthesize speech using OpenAI's TTS model and save it to a file.

        Args:
            text (str): The text to synthesize into speech.
            filename (str): The path to save the synthesized audio file.
            tone (Tone): The tone to use.
            lang (Language): The language to use.
        """
        response = self.client.with_streaming_response.audio.speech.create(model=self.model, voice=self.voice,
                                                                           response_format="wav", input=text)
        response.stream_to_file(filename)

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str,
                               tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> bool:
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

    def synthesize(self, text: str, filename: str,
                   tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> None:
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

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str,
                               tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> bool:
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

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str,
                               tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> bool:
        # Yandex SpeechKit doesn't have an async API, so we'll run it in an executor

        def synthesize_wrapper(par: dict) -> bytes:
            """Wrapper method to call synthesize with the correct parameters."""
            return par["model"].synthesize(par["text"], raw_format=True)

        loop = asyncio.get_event_loop()
        args = {"model": self.voice_model(tone=tone, lang=lang), "text": text}
        result = await loop.run_in_executor(None, synthesize_wrapper, args)

        async with aiofiles.open(filename, "wb") as out:
            await out.write(result)

        return True


class ElevenLabsTTSEngine(TTSEngine):
    def __init__(self, config):
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("Eleven Labs API key is not provided in environment variables or configuration")

        self.voice_id = config.get('elevenlabs_voice_id', 'N2lVS1w4EtoT3dr4eOWO')
        self.stability = config.get('elevenlabs_stability', 0.5)
        self.similarity_boost = config.get('elevenlabs_similarity_boost', 0.75)
        self.style = config.get('elevenlabs_style', 0.0)
        self.use_speaker_boost = config.get('elevenlabs_use_speaker_boost', False)

        self.model_id = config.get('elevenlabs_model', "eleven_multilingual_v2")

    def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> None:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

        headers = {
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost
            }
        }

        response = requests.post(url, json=data, headers=headers)

        mp3_filename = filename + ".mp3"
        if response.status_code == 200:
            with open(mp3_filename, "wb") as f:
                f.write(response.content)
            audio = AudioSegment.from_mp3(mp3_filename)
            audio.export(filename, format="wav")
        else:
            raise Exception(f"Error from ElevenLabs API: {response.status_code} - {response.text}")

    def max_text_length(self) -> int:
        return 2500  # Eleven Labs has a limit of 2500 characters per request

    async def synthesize_async(self, session: aiohttp.ClientSession, text: str, filename: str,
                               tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> bool:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

        headers = {
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost
            }
        }

        try:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    audio_content = await response.read()
                    mp3_filename = filename + ".mp3"
                    async with aiofiles.open(mp3_filename, mode='wb') as f:
                        await f.write(audio_content)
                    audio = AudioSegment.from_mp3(mp3_filename)
                    audio.export(filename, format="wav")
                    logger.info(f"Audio content written to file {filename}")
                    return True
                else:
                    logging.error(f"Error from ElevenLabs API: {response.status} - {await response.text()}")
                    return False
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return False


async def main():
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

