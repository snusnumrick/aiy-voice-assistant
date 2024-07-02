"""
Audio processing module.

This module provides functionality for speech transcription and synthesis,
interfacing with the Google Cloud Speech-to-Text API and various TTS engines.
"""

from typing import Optional, List, Iterator
from subprocess import Popen
from aiy.board import Button
from aiy.leds import Leds, Color, Pattern
from aiy.voice.audio import AudioFormat, Recorder
from google.cloud import speech
import logging
from abc import ABC, abstractmethod
from collections import deque
import time
import os
from pydub import AudioSegment
import tempfile
import shutil
import re
import grpc
from src.tts_engine import TTSEngine
from src.config import Config

logger = logging.getLogger(__name__)


class SpeechRecognitionService(ABC):
    @abstractmethod
    def setup_client(self, config):
        pass

    @abstractmethod
    def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        pass


class GoogleSpeechRecognition(SpeechRecognitionService):
    def setup_client(self, config):
        from google.oauth2 import service_account
        from google.cloud import speech

        logger.info("Setting up Google Speech client")
        service_account_file = config.get('google_service_account_file', '~/gcloud.json')
        service_account_file = os.path.expanduser(service_account_file)
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        self.client = speech.SpeechClient(credentials=credentials)

    def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        from google.cloud import speech

        logger.info("Transcribing audio stream (google)")
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=config.get("sample_rate_hertz", 16000),
                language_code=config.get("language_code", "ru-RU"),
                enable_automatic_punctuation=True
            ),
            interim_results=True
        )
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
        responses = self.client.streaming_recognize(streaming_config, requests)

        text = ""
        for response in responses:
            logger.info("Received response: %s", response)
            for result in response.results:
                logger.info("Received result: %s", result)
                if result.is_final:
                    text += result.alternatives[0].transcript + " "
        return text.strip()


class YandexSpeechRecognition(SpeechRecognitionService):
    def setup_client(self, config):
        from yandex.cloud.ai.stt.v3 import stt_service_pb2_grpc

        self.api_key = os.environ.get('YANDEX_API_KEY') or config.get('yandex_api_key')
        if not self.api_key:
            raise ValueError("Yandex API key is not provided in environment variables or configuration")

        # Create a secure channel
        creds = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel('stt.api.cloud.yandex.net:443', creds)
        self.stub = stt_service_pb2_grpc.RecognizerStub(channel)

    def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        from yandex.cloud.ai.stt.v3 import stt_service_pb2

        # Prepare the recognition config
        recognition_config = stt_service_pb2.RecognitionConfig(
            specification=stt_service_pb2.RecognitionSpec(
                language_code=config.get("language_code", "ru-RU"),
                profanity_filter=config.get("profanity_filter", False),
                model=config.get("model", "general"),
                partial_results=config.get("partial_results", True),
                audio_encoding=stt_service_pb2.LINEAR16_PCM,
                sample_rate_hertz=config.get("sample_rate_hertz", 16000),
            )
        )

        # Prepare the streaming recognition request
        def request_stream():
            yield stt_service_pb2.StreamingRequest(
                session_options=stt_service_pb2.StreamingOptions(recognition_model=recognition_config))
            for chunk in audio_generator:
                yield stt_service_pb2.StreamingRequest(chunk=stt_service_pb2.AudioChunk(data=chunk))

        # Make the gRPC call
        metadata = [('authorization', f'Api-Key {self.api_key}')]
        responses = self.stub.RecognizeStreaming(request_stream(), metadata=metadata)

        text = ""
        for response in responses:
            if response.HasField('final'):
                for alternative in response.final.alternatives:
                    text += alternative.text + " "
            elif response.HasField('partial'):
                # Handle partial results if needed
                pass

        return text.strip()


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
        self.setup_speech_service()
        self.breathing_period_ms = self.config.get('ready_breathing_period_ms', 10000)
        self.led_breathing_color = self.config.get('ready_breathing_color', (0, 1, 0))  # dark green
        self.led_breathing_duration = self.config.get('ready_breathing_duration', 60)
        self.led_processing_color = self.config.get('processing_color', (0, 1, 0))  # dark green
        self.led_processing_blink_period_ms = self.config.get('processing_blink_period_ms', 300)
        self.audio_sample_rate = self.config.get('audio_sample_rate', 16000)
        self.audio_recording_chunk_duration_sec = self.config.get('audio_recording_chunk_duration_sec', 0.3)

    def setup_speech_service(self):
        service_name = self.config.get('speech_recognition_service', 'google').lower()
        if service_name == 'google':
            self.speech_service = GoogleSpeechRecognition()
        elif service_name == 'yandex':
            self.speech_service = YandexSpeechRecognition()
        else:
            raise ValueError(f"Unsupported speech recognition service: {service_name}")
        self.speech_service.setup_client(self.config)

    def transcribe_speech(self, player_process: Optional[Popen] = None) -> str:
        """
        Transcribe speech from the microphone input, including pre and post buffering.

        Args:
            player_process (Optional[Popen]): A subprocess.Popen object representing a running audio player.

        Returns:
            str: The transcribed text.
        """

        chunks_deque = deque()
        status = 0  # 0 - not started, 1 - started, 2 - finished

        def generate_audio_chunks():
            nonlocal status, chunks_deque

            audio_format = AudioFormat(sample_rate_hz=self.audio_sample_rate,
                                       num_channels=1,
                                       bytes_per_sample=2)
            record_more = 0
            self.leds.pattern = Pattern.breathe(self.breathing_period_ms)
            self.leds.update(Leds.rgb_pattern(self.led_breathing_color))
            time_breathing_started = time.time()
            breathing_on = True
            for chunk in recorder.record(audio_format, chunk_duration_sec=self.audio_recording_chunk_duration_sec):
                # if breathing is on for more than maX_breathing_duration seconds, switch off LED
                if time.time() - time_breathing_started > self.led_breathing_duration and breathing_on:
                    logger.info('Breathing off')
                    self.leds.update(Leds.rgb_off())
                    breathing_on = False
                if status < 2 or (status == 2 and record_more > 0):
                    logger.info(f"Recording audio chunk, status={status}, record_more={record_more}")
                    if status == 2:
                        record_more -= 1
                    chunks_deque.append(chunk)
                    if status == 0 and len(chunks_deque) > 3:
                        chunks_deque.popleft()

                if status == 0 and self.button_is_pressed:
                    if player_process:
                        try:
                            player_process.terminate()
                        except Exception as e:
                            logger.error(f"Error terminating player process: {str(e)}")
                    self.leds.update(Leds.rgb_on(Color.GREEN))
                    logger.info('Listening...')
                    status = 1

                if not chunks_deque:
                    logger.debug("No audio chunk available")
                    break

                if status > 0:
                    chunk = chunks_deque.popleft()
                    logger.debug("Yielding audio chunk")
                    yield chunk

                if status == 1 and not self.button_is_pressed:
                    logger.info('Stopped listening')
                    self.leds.pattern = Pattern.blink(self.led_processing_blink_period_ms)
                    self.leds.update(Leds.rgb_pattern(self.led_processing_color))
                    status = 2
                    record_more = 2

        self.setup_button_callbacks()
        logger.info('Press the button and speak')

        with Recorder() as recorder:
            audio_generator = generate_audio_chunks()

            for _ in audio_generator:
                if status:
                    break

            logger.info('Processing audio...')
            text = self.speech_service.transcribe_stream(audio_generator, self.config)

        self.leds.update(Leds.rgb_off())  # Ensure LED is turned off after transcription
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


def split_text(text: str, max_length: int) -> List[str]:
    """
    Split text into chunks of maximum length.

    Args:
        text (str): The text to split.
        max_length (int): The maximum length of each chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    sentences = re.split('(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def combine_audio_files(file_list: List[str], output_filename: str) -> None:
    """
    Combine multiple audio files into a single file.

    Args:
        file_list (List[str]): List of audio file paths to combine.
        output_filename (str): Path to save the combined audio file.
    """
    logger.debug(f"Combining {len(file_list)} audio files into {output_filename}")
    combined = AudioSegment.empty()
    for file in file_list:
        audio = AudioSegment.from_wav(file)
        combined += audio
    logger.debug(f"Exporting combined audio to {output_filename}")
    combined.export(output_filename, format="wav")
    logger.debug(f"Exported combined audio to {output_filename}")


def synthesize_speech(engine: TTSEngine, text: str, filename: str, config: Config) -> None:
    """
    Synthesize speech from text, handling long texts by splitting and combining audio chunks.

    Args:
        engine (TTSEngine): The text-to-speech engine to use.
        text (str): The text to synthesize into speech.
        filename (str): The path to save the synthesized audio file.
        config (Config): The application configuration object.
    """
    logger.debug('Synthesizing speech for: %s', text)
    max_size_tts = config.get('max_size_tts', 4096)  # Default to 4096 if not in config
    chunks = split_text(text, max_length=max_size_tts)

    temp_dir = tempfile.mkdtemp()
    try:
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(temp_dir, f"chunk_{i}.wav")
            logger.debug(f"Synthesizing chunk {i}: {chunk}")
            engine.synthesize(chunk, chunk_file)
            logger.debug(f"Saved chunk {i} to {chunk_file}")
            chunk_files.append(chunk_file)

        if len(chunk_files) > 1:
            combine_audio_files(chunk_files, filename)
        else:
            shutil.move(chunk_files[0], filename)
    finally:
        shutil.rmtree(temp_dir)

    logger.debug(f"Final synthesized speech saved at {filename}")
