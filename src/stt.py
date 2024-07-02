"""
Speech-to-Text (STT) module.

This module provides classes for different STT services.
"""

from abc import ABC, abstractmethod
from typing import Iterator
import os
import grpc
import logging
from google.cloud import speech
from google.oauth2 import service_account
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc
import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
from src.config import Config

logger = logging.getLogger(__name__)


class BaseSTTService(ABC):
    """Abstract base class for STT services."""

    def __init__(self, config: Config):
        """
        Initialize the STT service.

        Args:
            config (Config): Configuration object.
        """
        self.config = config

    @abstractmethod
    def transcribe_stream(self, audio_generator: Iterator[bytes]) -> str:
        """
        Transcribe an audio stream to text.

        Args:
            audio_generator (Iterator[bytes]): Generator yielding audio chunks.

        Returns:
            str: Transcribed text.
        """
        pass


class GoogleSTTService(BaseSTTService):
    """Google Cloud Speech-to-Text service."""

    def __init__(self, config: Config):
        """
        Initialize the Google STT service.

        Args:
            config (Config): Configuration object.
        """
        super().__init__(config)
        self.setup_client()

    def setup_client(self) -> None:
        """Set up the Google Cloud Speech client."""
        service_account_file = self.config.get('google_service_account_file', '~/gcloud.json')
        service_account_file = os.path.expanduser(service_account_file)
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        self.client = speech.SpeechClient(credentials=credentials)

    def transcribe_stream(self, audio_generator: Iterator[bytes]) -> str:
        """
        Transcribe audio stream using Google Cloud Speech-to-Text API.

        Args:
            audio_generator (Iterator[bytes]): Generator yielding audio chunks.

        Returns:
            str: Transcribed text.
        """
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.config.get("sample_rate_hertz", 16000),
                language_code=self.config.get("language_code", "en-US"),
                enable_automatic_punctuation=True
            ),
            interim_results=True
        )

        requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
        responses = self.client.streaming_recognize(streaming_config, requests)

        return self._process_responses(responses)

    def _process_responses(self, responses) -> str:
        """
        Process responses from Google Cloud Speech-to-Text API.

        Args:
            responses: Iterator of responses from the API.

        Returns:
            str: Concatenated transcribed text.
        """
        text = ""
        for response in responses:
            for result in response.results:
                if result.is_final:
                    text += result.alternatives[0].transcript + " "
        return text.strip()


class YandexSTTService(BaseSTTService):
    """Yandex SpeechKit service."""

    def __init__(self, config: Config):
        """
        Initialize the Yandex STT service.

        Args:
            config (Config): Configuration object.
        """
        super().__init__(config)
        self.setup_client()

    def setup_client(self) -> None:
        """Set up the Yandex SpeechKit client."""
        self.api_key = os.environ.get('YANDEX_API_KEY') or self.config.get('yandex_api_key')
        if not self.api_key:
            raise ValueError("Yandex API key is not provided in environment variables or configuration")

        cred = grpc.ssl_channel_credentials()
        self.channel = grpc.secure_channel('stt.api.cloud.yandex.net:443', cred)
        self.stub = stt_service_pb2_grpc.RecognizerStub(self.channel)

    def transcribe_stream(self, audio_generator: Iterator[bytes]) -> str:
        """
        Transcribe audio stream using Yandex SpeechKit API.

        Args:
            audio_generator (Iterator[bytes]): Generator yielding audio chunks.

        Returns:
            str: Transcribed text.
        """

        def request_generator():
            yield stt_pb2.StreamingRequest(session_options=self._get_streaming_options())
            for chunk in audio_generator:
                yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=chunk))

        metadata = [('authorization', f'Api-Key {self.api_key}')]
        responses = self.stub.RecognizeStreaming(request_generator(), metadata=metadata)

        return self._process_responses(responses)

    def _get_streaming_options(self) -> stt_pb2.StreamingOptions:
        """
        Get streaming options for Yandex SpeechKit API.

        Returns:
            stt_pb2.StreamingOptions: Configured streaming options.
        """
        return stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=self.config.get("sample_rate_hertz", 16000),
                        audio_channel_count=1
                    )
                ),
                text_normalization=stt_pb2.TextNormalizationOptions(
                    text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                    profanity_filter=self.config.get("profanity_filter", False),
                    literature_text=False
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=[self.config.get("language_code", "ru-RU")]
                ),
                audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME
            )
        )

    def _process_responses(self, responses) -> str:
        """
        Process responses from Yandex SpeechKit API.

        Args:
            responses: Iterator of responses from the API.

        Returns:
            str: Concatenated transcribed text.
        """
        full_text = ""
        for response in responses:
            event_type = response.WhichOneof('Event')
            if event_type == 'final_refinement':
                full_text += response.final_refinement.normalized_text.alternatives[0].text + " "
        return full_text.strip()

    def __del__(self):
        """Close the gRPC channel when the object is destroyed."""
        if hasattr(self, 'channel'):
            self.channel.close()


def create_stt_service(config: Config) -> BaseSTTService:
    """
    Factory function to create STT service based on configuration.

    Args:
        config (Config): Configuration object.

    Returns:
        BaseSTTService: Initialized STT service.

    Raises:
        ValueError: If an unsupported STT service is specified.
    """
    service_name = config.get('speech_recognition_service', 'yandex').lower()
    if service_name == 'google':
        return GoogleSTTService(config)
    elif service_name == 'yandex':
        return YandexSTTService(config)
    else:
        raise ValueError(f"Unsupported STT service: {service_name}")
