"""
Speech-to-Text (STT) Engine module.

This module provides abstract and concrete implementations of STT engines,
including OpenAI's Whisper model and Google's Speech Recognition.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class STTEngine(ABC):
    """
    Abstract base class for Speech-to-Text engines.
    """

    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe speech from an audio file to text.

        Args:
            audio_file (str): Path to the audio file to transcribe.

        Returns:
            str: The transcribed text.
        """
        pass


class OpenAISTTEngine(STTEngine):
    """
    Implementation of STTEngine using OpenAI's Whisper model.
    """

    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe speech using OpenAI's Whisper model.

        Args:
            audio_file (str): Path to the audio file to transcribe.

        Returns:
            str: The transcribed text.
        """
        import openai

        with open(audio_file, "rb") as f:
            return openai.audio.transcriptions.create(
                model="gpt-4o-transcribe", file=f, response_format="text"
            )


class GoogleSTTEngine(STTEngine):
    """
    Implementation of STTEngine using Google's Speech Recognition.
    """

    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe speech using Google's Speech Recognition.

        Args:
            audio_file (str): Path to the audio file to transcribe.

        Returns:
            str: The transcribed text.
        """
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language="en-US")
        except sr.UnknownValueError:
            logger.error("Unknown error occurred")
            return ""
        except sr.RequestError:
            logger.error(
                "Could not request results from Google Speech Recognition service"
            )
            return ""
