"""
Audio processing module.

This module provides functionality for speech transcription and synthesis,
interfacing with the Google Cloud Speech-to-Text API and various TTS engines.
"""

import asyncio
import datetime
import logging
import os
import queue
import re
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Optional, List, Iterator, Callable

import aiohttp
import grpc
from aiy.board import Button
from aiy.leds import Leds, Pattern
from aiy.voice.audio import AudioFormat, Recorder
from google.cloud import speech

from src.config import Config
from src.responce_player import ResponsePlayer
from src.tools import time_string_ms, get_timezone, combine_audio_files
from src.tts_engine import TTSEngine
from src.tailscale_manager import TailscaleManager

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

        logger.debug("Setting up Google Speech client")
        service_account_file = config.get(
            "google_service_account_file", "~/gcloud.json"
        )
        service_account_file = os.path.expanduser(service_account_file)
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file
        )
        self.client = speech.SpeechClient(credentials=credentials)

    def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        logger.debug("Transcribing audio stream (google)")
        streaming_config = speech.types.StreamingRecognitionConfig(
            config=speech.types.RecognitionConfig(
                encoding=speech.types.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=config.get("sample_rate_hertz", 16000),
                language_code=config.get("language_code", "ru-RU"),
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )
        requests = (
            speech.types.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in audio_generator
        )
        responses = self.client.streaming_recognize(streaming_config, requests)

        text = ""
        for response in responses:
            logger.debug("Received response: %s", response)
            for result in response.results:
                logger.debug("Received result: %s", result)
                if result.is_final:
                    text += result.alternatives[0].transcript + " "
        return text.strip()


class YandexSpeechRecognition(SpeechRecognitionService):
    def setup_client(self, config):
        import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc
        import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2

        self.api_key = os.environ.get("YANDEX_API_KEY") or config.get("yandex_api_key")
        if not self.api_key:
            raise ValueError(
                "Yandex API key is not provided in environment variables or configuration"
            )

        cred = grpc.ssl_channel_credentials()
        self.channel = grpc.secure_channel("stt.api.cloud.yandex.net:443", cred)
        self.stub = stt_service_pb2_grpc.RecognizerStub(self.channel)

        self.recognize_options = stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=config.get("sample_rate_hertz", 16000),
                        audio_channel_count=1,
                    )
                ),
                text_normalization=stt_pb2.TextNormalizationOptions(
                    text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                    profanity_filter=config.get("profanity_filter", False),
                    literature_text=config.get("literature_text", True),
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=[config.get("language_code", "ru-RU")],
                ),
                audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
            )
        )

    def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        def request_generator():
            import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2

            yield stt_pb2.StreamingRequest(session_options=self.recognize_options)

            for chunk in audio_generator:
                yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=chunk))

        metadata = [("authorization", f"Api-Key {self.api_key}")]
        responses = self.stub.RecognizeStreaming(request_generator(), metadata=metadata)

        full_text = ""
        current_segment = ""
        try:
            for response in responses:
                event_type = response.WhichOneof("Event")
                if event_type == "partial" and response.partial.alternatives:
                    logger.debug(f"Partial: {response.partial.alternatives[0].text}")
                elif event_type == "final":
                    current_segment = response.final.alternatives[0].text
                    logger.debug(f"Final: {current_segment}")
                elif event_type == "final_refinement":
                    refined_text = (
                        response.final_refinement.normalized_text.alternatives[0].text
                    )
                    logger.debug(f"Refined: {refined_text}")
                    full_text += refined_text + " "
                    current_segment = ""  # Reset current segment
                elif event_type == "eou_update":
                    # If we have a current segment that wasn't refined, add it to full_text
                    if current_segment:
                        full_text += current_segment + " "
                        logger.info(f"Added unrefined segment: {current_segment}")
                    current_segment = ""  # Reset current segment

        except Exception as err:
            logger.error(err)
            return ""

        return full_text.strip()

    def __del__(self):
        if hasattr(self, "channel"):
            self.channel.close()


class RecordingStatus(Enum):
    NOT_STARTED = 0
    STARTED = 1
    FINISHED = 2


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

    def __init__(
        self,
        button: Button,
        leds: Leds,
        config,
        cleaning: Optional[Callable] = None,
        timezone: Optional[str] = None,
    ) -> None:
        """
        Initialize the SpeechTranscriber.

        Args:
            button (Button): The AIY Kit button object.
            leds (Leds): The AIY Kit LED object.
            config (Config): The application configuration object.
            cleaning (Optional[Callable]): Optional callback function to clean the audio stream.
            timezone (Optional[str]): The timezone of the current location.
        """
        self.button = button
        self.leds = leds
        self.config = config
        self.button_is_pressed = False
        self.setup_speech_service()
        self.breathing_period_ms = self.config.get("ready_breathing_period_ms", 10000)
        self.led_breathing_color = self.config.get(
            "ready_breathing_color", (0, 1, 0)
        )  # dark green
        self.led_recording_color = self.config.get(
            "recording_color", (0, 255, 0)
        )  # bright green
        self.led_breathing_duration = self.config.get("ready_breathing_duration", 60)
        self.led_processing_color = self.config.get(
            "processing_color", (0, 1, 0)
        )  # dark green
        self.led_processing_blink_period_ms = self.config.get(
            "processing_blink_period_ms", 300
        )
        self.audio_sample_rate = self.config.get("audio_sample_rate", 16000)
        self.audio_recording_chunk_duration_sec = self.config.get(
            "audio_recording_chunk_duration_sec", 0.1
        )
        self.max_number_of_chunks = self.config.get("max_number_of_chunks", 5)
        self.number_of_chuncks_to_record_after_button_depressed = self.config.get(
            "number_of_chuncks_to_record_after_button_depressed", 3
        )
        self.cleaning_routine: Callable = cleaning
        self.cleaning_task = None
        self.last_clean_date: Optional[datetime.date] = None
        self.timezone: str = get_timezone() if timezone is None else timezone
        self.tailscale_manager = TailscaleManager(config)

    async def check_and_schedule_cleaning(self) -> None:
        now: datetime.datetime = datetime.datetime.now()
        cleaning_time_start = datetime.time(
            hour=self.config.get("cleaning_tine_start_hour", 3)
        )  # 3 AM
        cleaning_time_stop = datetime.time(
            hour=self.config.get("cleaning_time_stop_hour", 4)
        )  # 4 AM

        if cleaning_time_start <= now.time() < cleaning_time_stop:
            if self.last_clean_date != now.date():
                logger.debug(f"Cleaning date: {now.date()}")
                await self.cleaning_routine()
                self.last_clean_date = now.date()

        # First check Tailscale state
        await self.tailscale_manager.check_and_update_state()

    def setup_speech_service(self):
        service_name = self.config.get("speech_recognition_service", "yandex").lower()
        if service_name == "google":
            self.speech_service = GoogleSpeechRecognition()
        elif service_name == "yandex":
            self.speech_service = YandexSpeechRecognition()
        else:
            raise ValueError(f"Unsupported speech recognition service: {service_name}")
        self.speech_service.setup_client(self.config)

    async def transcribe_speech(
        self, player_process: Optional[ResponsePlayer] = None
    ) -> str:
        """
        Transcribe speech from the microphone input, including pre and post buffering.

        Args:
            player_process (Optional[ResponsePlayer]): Object representing a running audio player.

        Returns:
            str: The transcribed text.
        """

        chunks_deque = deque()
        status = RecordingStatus.NOT_STARTED

        async def generate_audio_chunks():
            nonlocal status, chunks_deque, player_process

            audio_format = AudioFormat(
                sample_rate_hz=self.audio_sample_rate,
                num_channels=1,
                bytes_per_sample=2,
            )
            record_more = 0
            breathing_on = False

            def start_idle() -> bool:
                nonlocal status, time_breathing_started, breathing_on, player_process
                if player_process is None or not player_process.is_playing():
                    logger.info(
                        f"({time_string_ms(self.timezone)}) Ready to listen...  LED: breathing"
                    )
                    self.leds.pattern = Pattern.breathe(self.breathing_period_ms)
                    self.leds.update(Leds.rgb_pattern(self.led_breathing_color))
                    time_breathing_started = time.time()
                    breathing_on = True
                    return True
                else:
                    return False

            def start_listening():
                nonlocal status, breathing_on, recoding_started_at
                logger.info(
                    f"({time_string_ms(self.timezone)}) Recording audio... LED solid"
                )
                self.leds.update(Leds.rgb_on(self.led_recording_color))
                breathing_on = False
                recoding_started_at = time.time()

            def start_processing():
                nonlocal status, record_more
                logger.info(
                    f"({time_string_ms(self.timezone)}) Processing audio... LED blinking"
                )
                self.leds.pattern = Pattern.blink(self.led_processing_blink_period_ms)
                self.leds.update(Leds.rgb_pattern(self.led_processing_color))
                record_more = self.number_of_chuncks_to_record_after_button_depressed

            def stop_breathing():
                nonlocal breathing_on
                logger.info("Breathing off LED OFF")
                self.leds.update(Leds.rgb_off())
                breathing_on = False

            def stop_playing():
                nonlocal player_process
                if player_process:
                    try:
                        logger.info("Stopping player process")
                        chunks_deque.clear()
                        player_process.stop()
                        logger.info("Player process stopped")
                    except Exception as e:
                        logger.error(f"Error stopping player process: {str(e)}")

            chunks = []
            idle = start_idle()
            status = RecordingStatus.NOT_STARTED

            recoding_started_at = time.time()
            time_breathing_started = time.time()
            for chunk in recorder.record(
                audio_format, chunk_duration_sec=self.audio_recording_chunk_duration_sec
            ):
                if status == RecordingStatus.NOT_STARTED and not idle:
                    idle = start_idle()

                if self.cleaning_routine:
                    await self.check_and_schedule_cleaning()

                if (
                    time.time() - time_breathing_started > self.led_breathing_duration
                ) and breathing_on:
                    stop_breathing()

                if (status != RecordingStatus.FINISHED) or (
                    status == RecordingStatus.FINISHED and record_more > 0
                ):
                    if status == RecordingStatus.FINISHED:
                        record_more -= 1
                    chunks_deque.append(chunk)
                    if (status == RecordingStatus.NOT_STARTED) and (
                        len(chunks_deque) > self.max_number_of_chunks
                    ):
                        chunks_deque.popleft()

                if (status == RecordingStatus.NOT_STARTED) and self.button_is_pressed:
                    stop_playing()
                    start_listening()
                    logger.debug(f"{len(chunks_deque)} audio chunks buffered")
                    status = RecordingStatus.STARTED
                    continue

                if not chunks_deque:
                    logger.debug("No audio chunk available")

                    # import wave
                    #
                    # with wave.open("recording.wav", 'wb') as wav_file:
                    #     wav_file.setnchannels(audio_format.num_channels)
                    #     wav_file.setsampwidth(audio_format.bytes_per_sample)
                    #     wav_file.setframerate(audio_format.sample_rate_hz)
                    #     for chunk in chunks:
                    #         wav_file.writeframes(chunk)

                    break

                if status != RecordingStatus.NOT_STARTED:
                    chunks.append(chunk)
                    yield chunks_deque.popleft()

                if status == RecordingStatus.STARTED and not self.button_is_pressed:
                    start_processing()
                    status = RecordingStatus.FINISHED

        self.setup_button_callbacks()
        logger.info("Press the button and speak")

        with Recorder() as recorder:
            audio_generator = generate_audio_chunks()

            async for _ in audio_generator:
                if status != RecordingStatus.NOT_STARTED:
                    break

            logger.debug("Processing audio...")

            try:
                audio_queue = queue.Queue()

                async def fill_queue():
                    async for chunk in audio_generator:
                        audio_queue.put(chunk)
                    audio_queue.put(None)  # Сигнал окончания

                def sync_audio_generator():
                    while True:
                        chunk = audio_queue.get()
                        if chunk is None:
                            break
                        yield chunk

                # fill queue in background
                fill_queue_task = asyncio.create_task(fill_queue())

                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    self.speech_service.transcribe_stream,
                    sync_audio_generator(),
                    self.config,
                )

                # wait queue
                await fill_queue_task

            except Exception as e:
                logger.error(f"Error transcribing speech: {str(e)}")
                text = ""
        return text

    def setup_button_callbacks(self):
        """
        Set up callbacks for button press and release events.
        """
        self.button.when_pressed = self.button_pressed
        self.button.when_released = self.button_released
        logger.debug("set button callback")

    def button_pressed(self):
        """
        Callback function for button press event.
        """
        self.button_is_pressed = True
        logger.debug("Button pressed")

    def button_released(self):
        """
        Callback function for button release event.
        """
        self.button_is_pressed = False
        logger.debug("Button released")

    def wait_for_button_press(self):
        """
        Wait for the button to be pressed, with visual LED feedback.
        """
        logger.info("Waiting for button press... LED solid")
        self.leds.pattern = Pattern.breathe(10000)
        self.leds.update(Leds.rgb_pattern((0, 1, 0)))
        self.button.wait_for_press()
        self.leds.update(Leds.rgb_off())
        logger.info("Button pressed LED OFF")


def split_text(text: str, max_length: int) -> List[str]:
    """
    Split text into chunks of maximum length.

    Args:
        text (str): The text to split.
        max_length (int): The maximum length of each chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    sentences = re.split("(?<=[.!?]) +", text)
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


def synthesize_speech(
    engine: TTSEngine, text: str, filename: str, config: Config
) -> bool:
    """
    Synthesize speech from text, handling long texts by splitting and combining audio chunks.

    Args:
        engine (TTSEngine): The text-to-speech engine to use.
        text (str): The text to synthesize into speech.
        filename (str): The path to save the synthesized audio file.
        config (Config): The application configuration object.

    Returns:
        bool: True if the speech was successfully synthesized, False otherwise.
    """
    logger.debug("Synthesizing speech for: %s", text)
    max_size_tts = engine.max_text_length()
    result = True
    if (max_size_tts > 0) and (len(text) > max_size_tts):
        # split long text
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
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            result = False
        finally:
            shutil.rmtree(temp_dir)
    else:
        engine.synthesize(text, filename)

    logger.debug(f"Final synthesized speech saved at {filename}")
    return result


async def synthesize_speech_async(
    engine: TTSEngine, text: str, filename: str, config: Config
) -> bool:
    """
    Asynchronous version of synthesize_speech function.
    """
    logger.debug("Synthesizing speech for: %s", text)
    max_size_tts = engine.max_text_length()
    result = True
    if (max_size_tts > 0) and (len(text) > max_size_tts):
        chunks = split_text(text, max_length=max_size_tts)
        temp_dir = tempfile.mkdtemp()
        try:
            chunk_files = []
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i, chunk in enumerate(chunks):
                    chunk_file = os.path.join(temp_dir, f"chunk_{i}.wav")
                    logger.debug(f"Synthesizing chunk {i}: {chunk}")
                    task = asyncio.create_task(
                        engine.synthesize_async(session, chunk, chunk_file)
                    )
                    tasks.append(task)

                await asyncio.gather(*tasks)

                for i, task in enumerate(tasks):
                    chunk_file = os.path.join(temp_dir, f"chunk_{i}.wav")
                    chunk_files.append(chunk_file)

            if len(chunk_files) > 1:
                combine_audio_files(chunk_files, filename)
            else:
                shutil.move(chunk_files[0], filename)
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            result = False
        finally:
            shutil.rmtree(temp_dir)
    else:
        await engine.synthesize_async(None, text, filename)

    logger.debug(f"Final synthesized speech saved at {filename}")
    return result
