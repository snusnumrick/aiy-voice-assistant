"""
Text-to-Speech (TTS) Engine module.

This module provides abstract and concrete implementations of TTS engines,
including OpenAI's TTS model and Google's Text-to-Speech.
"""

from abc import ABC, abstractmethod


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
        from google.cloud import texttospeech
        self.client = texttospeech.TextToSpeechClient()
        self.language_code = config.get('google_tts_language', 'en-US')

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
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig
