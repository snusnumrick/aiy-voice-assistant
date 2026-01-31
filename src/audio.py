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
from aiy.board import Button, ButtonState
from aiy.leds import Leds, Pattern
from aiy.voice.audio import AudioFormat, Recorder
from google.cloud import speech

from src.config import Config
from src.responce_player import ResponsePlayer
from src.tools import time_string_ms, get_timezone, combine_audio_files
from src.tts_engine import TTSEngine
from src.background_tasks import BackgroundTaskManager
from src.emotion_engine import EmotionEngine, format_annotation

logger = logging.getLogger(__name__)


class SpeechRecognitionService(ABC):
    @abstractmethod
    def setup_client(self, config):
        pass

    @abstractmethod
    async def transcribe_stream(
        self, audio_generator: Iterator[bytes], config
    ) -> str:
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

    async def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._transcribe_stream_sync, audio_generator, config
        )

    def _transcribe_stream_sync(self, audio_generator: Iterator[bytes], config) -> str:
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

    async def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._transcribe_stream_sync, audio_generator, config
        )

    def _transcribe_stream_sync(self, audio_generator: Iterator[bytes], config) -> str:
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


class OpenAISpeechRecognition(SpeechRecognitionService):
    """
    OpenAI Realtime API speech recognition implementation.

    Uses WebSocket connection to stream audio in real-time to OpenAI's
    gpt-4o-transcribe model with VAD (Voice Activity Detection).
    """

    def setup_client(self, config):
        """
        Initialize OpenAI client for speech recognition.

        Args:
            config: Configuration object containing API settings

        Raises:
            ValueError: If OpenAI API key is not provided
        """
        import base64
        import json
        import asyncio
        import websockets
        from websockets.exceptions import ConnectionClosed

        logger.info("Setting up OpenAI Realtime Speech client")

        # Get API key from environment variable or config
        api_key = os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")

        if not api_key:
            raise ValueError(
                "OpenAI API key is not provided. Set OPENAI_API_KEY environment variable "
                "or 'openai_api_key' in configuration."
            )

        # Store configuration
        self.api_key = api_key
        self.model = config.get("openai_transcription_model", "gpt-4o-transcribe")
        self.language = config.get("language_code", "")
        self.sample_rate = config.get("sample_rate_hertz", 24000)  # OpenAI Realtime API requires at least 24kHz
        self.base64 = base64
        self.json = json
        self.asyncio = asyncio
        self.websockets = websockets
        self.ConnectionClosed = ConnectionClosed

    async def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        """
        Transcribe audio stream using OpenAI Realtime API.

        Streams PCM16 audio chunks in real-time with transcription results.

        Args:
            audio_generator: Iterator yielding audio chunks as bytes
            config: Configuration object

        Returns:
            str: Transcribed text
        """
        logger.debug("Transcribing audio stream (openai realtime)")

        try:
            return await self._transcribe_stream_async(audio_generator)
        except Exception as e:
            logger.error(f"Error transcribing audio with OpenAI: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    async def _transcribe_stream_async(self, audio_generator: Iterator[bytes]) -> str:
        """
        Async implementation of stream transcription.

        Args:
            audio_generator: Iterator yielding audio chunks as bytes

        Returns:
            str: Transcribed text
        """
        import websockets

        logger.debug("Starting OpenAI Realtime transcription")

        # Get ephemeral token
        try:
            logger.debug("Getting ephemeral token...")
            token_response = await self._get_ephemeral_token()
            if not token_response:
                logger.error("Failed to get ephemeral token")
                return ""
            logger.debug("Ephemeral token received successfully")
        except Exception as e:
            logger.error(f"Error getting ephemeral token: {str(e)}")
            return ""

        # Establish WebSocket connection
        client_secret = token_response.get("client_secret", {}).get("value")
        if not client_secret:
            logger.error("No client_secret in token response")
            return ""

        uri = f"wss://api.openai.com/v1/realtime?intent=transcription&client_secret={client_secret}"
        logger.debug(f"Connecting to WebSocket: {uri[:80]}...")

        full_transcript = ""
        interim_results = []
        audio_chunk_count = 0

        try:
            async with websockets.connect(
                uri,
                extra_headers={
                    "Authorization": f"Bearer {self.api_key}"
                }
            ) as websocket:
                # Send session configuration
                session_config = {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.sample_rate,
                                },
                                "transcription": {
                                    "model": self.model,
                                    "language": self.language,
                                },
                            }
                        },
                    },
                }
                logger.debug(
                    "Sending session config: sample_rate=%s, model=%s",
                    self.sample_rate,
                    self.model,
                )
                await websocket.send(self.json.dumps(session_config))
                logger.debug("Session config sent")

                # Wait for config acknowledgment
                try:
                    config_ack = await asyncio.wait_for(websocket.recv(), timeout=5)
                    logger.debug(f"Config acknowledgment received: {config_ack[:100]}")
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for config acknowledgment")

                # Stream audio chunks with pacing for real-time transcription
                logger.debug("Starting to stream audio chunks with pacing...")
                import time
                t_next = time.monotonic()
                chunk_duration = 0.025  # 25ms between chunks for real-time pacing

                async for chunk in self._async_generator(audio_generator):
                    if chunk:
                        audio_chunk_count += 1
                        if audio_chunk_count % 10 == 0:
                            logger.debug(f"Processed {audio_chunk_count} audio chunks...")

                        # Encode audio to base64 (PCM16)
                        audio_b64 = self.base64.b64encode(chunk).decode('utf-8')

                        # Send audio buffer
                        audio_message = {
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64
                        }
                        await websocket.send(self.json.dumps(audio_message))

                        # Add pacing delay for real-time transcription
                        t_next += chunk_duration
                        await asyncio.sleep(max(0, t_next - time.monotonic()))

                # Signal end of audio
                logger.debug(
                    f"Finished streaming {audio_chunk_count} audio chunks. Sending commit..."
                )
                await websocket.send(self.json.dumps({"type": "input_audio_buffer.commit"}))
                logger.debug("Audio commit sent. Requesting response...")

                # Process responses with timeout
                message_count = 0

                async def process_messages():
                    nonlocal message_count, full_transcript
                    async for message in websocket:
                        message_count += 1
                        logger.debug(f"Received WebSocket message #{message_count}: {message}")

                        try:
                            response = self.json.loads(message)
                            event_type = response.get("type")

                            # Log all event types for debugging
                            if event_type:
                                logger.debug(f"Event type received: {event_type}")
                                logger.debug(f"Full response: {response}")

                            # Handle transcription events
                            if event_type == "input_audio_buffer.committed":
                                # VAD detected speech commit
                                logger.debug("VAD commit received")

                            elif event_type == "input_audio_transcription.completed":
                                # Final transcription result
                                transcript = response.get("transcript", "")
                                logger.debug(f"Received transcription: {transcript}")
                                if transcript:
                                    interim_results.append(transcript)
                                    full_transcript = " ".join(interim_results)

                            elif event_type == "input_audio_transcription.final_logprobs":
                                # Final result with logprobs
                                transcript = response.get("text", "")
                                logger.debug(f"Received final transcription: {transcript}")
                                if transcript:
                                    interim_results.append(transcript)
                                    full_transcript = " ".join(interim_results)

                            elif event_type == "input_audio_transcription.failed":
                                # Transcription failed
                                error_msg = response.get("error", {}).get("message", "Unknown error")
                                logger.error(f"Transcription failed: {error_msg}")

                            elif event_type == "conversation.item.input_audio_transcription.delta":
                                # Streaming delta - accumulate partial transcription
                                delta = response.get("delta", "")
                                if delta:
                                    logger.debug(f"Delta received: {delta}")
                                    # Keep accumulating in current transcript
                                    full_transcript += delta

                            elif event_type == "conversation.item.input_audio_transcription.completed":
                                # Final transcription result - move current to completed
                                logger.debug("Transcription completed")
                                if full_transcript:
                                    interim_results.append(full_transcript)
                                    logger.debug(f"Final transcript: {full_transcript}")
                                    # Return immediately when transcription is complete
                                    return full_transcript.strip()
                                full_transcript = ""  # Reset for next item

                            elif event_type == "conversation.item.done":
                                # Check if this item now has a transcript
                                item = response.get("item", {})
                                content = item.get("content", [])
                                for content_item in content:
                                    if content_item.get("type") == "input_audio":
                                        transcript = content_item.get("transcript", "")
                                        if transcript and transcript != "None":
                                            logger.debug(f"Received transcript from conversation item: {transcript}")
                                            return transcript.strip()


                        except Exception as e:
                            logger.error(f"Error processing WebSocket message: {str(e)}")
                            continue
                    return None

                try:
                    # Wait for responses with timeout after flush
                    await asyncio.wait_for(process_messages(), timeout=15)

                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for transcription response after flush")

                logger.debug(f"Total WebSocket messages received: {message_count}")
                logger.debug(f"Final transcript: '{full_transcript}'")

                return full_transcript.strip()

        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Error in async transcription: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    async def _get_ephemeral_token(self):
        """
        Get ephemeral token for WebSocket authentication.

        Returns:
            dict: Response containing client_secret
        """
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.openai.com/v1/realtime/sessions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                }

                logger.debug(f"Requesting ephemeral token from {url}")
                async with session.post(url, headers=headers, json=payload) as response:
                    logger.debug(f"Ephemeral token response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        logger.debug("Ephemeral token received successfully")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to get ephemeral token: {response.status} - {error_text}")
                        return {}
        except Exception as e:
            logger.error(f"Error getting ephemeral token: {str(e)}")
            return {}

    async def _async_generator(self, sync_generator: Iterator[bytes]):
        """
        Convert synchronous generator to async generator.

        Args:
            sync_generator: Synchronous iterator of bytes

        Yields:
            bytes: Audio chunks
        """
        try:
            for chunk in sync_generator:
                # Small delay to prevent overwhelming the API
                await asyncio.sleep(0.01)
                yield chunk
        except asyncio.CancelledError:
            logger.debug("Async generator cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in async generator: {str(e)}")
            raise


class ElevenLabsSpeechRecognition(SpeechRecognitionService):
    """
    ElevenLabs Realtime Speech-to-Text API implementation.

    Uses WebSocket connection to stream audio and receive transcripts.
    """

    def setup_client(self, config):
        import base64
        import json
        import asyncio
        import websockets
        from websockets.exceptions import ConnectionClosed

        logger.info("Setting up ElevenLabs Realtime Speech client")

        api_key = os.environ.get("ELEVENLABS_API_KEY") or config.get(
            "elevenlabs_api_key"
        )
        token = config.get("elevenlabs_token", "")

        if not api_key and not token:
            raise ValueError(
                "ElevenLabs API key or token is not provided. Set ELEVENLABS_API_KEY "
                "environment variable or 'elevenlabs_api_key'/'elevenlabs_token' in configuration."
            )

        self.api_key = api_key
        self.token = token
        self.model_id = config.get("elevenlabs_model_id", "scribe_v2_realtime")
        self.language_code = config.get("language_code") or config.get(
            "elevenlabs_language_code"
        )
        self.sample_rate = config.get("sample_rate_hertz", 16000)
        self.audio_format = config.get("elevenlabs_audio_format") or self._infer_audio_format(
            self.sample_rate
        )
        self.commit_strategy = config.get(
            "elevenlabs_commit_strategy", "manual"
        ).lower()
        self.include_timestamps = bool(
            config.get("elevenlabs_include_timestamps", False)
        )
        self.include_language_detection = bool(
            config.get("elevenlabs_include_language_detection", False)
        )
        self.enable_logging = config.get("elevenlabs_enable_logging")
        self.vad_silence_threshold_secs = config.get(
            "elevenlabs_vad_silence_threshold_secs"
        )
        self.vad_threshold = config.get("elevenlabs_vad_threshold")
        self.min_speech_duration_ms = config.get(
            "elevenlabs_min_speech_duration_ms"
        )
        self.min_silence_duration_ms = config.get(
            "elevenlabs_min_silence_duration_ms"
        )
        self.response_timeout_sec = config.get(
            "elevenlabs_response_timeout_sec", 15
        )
        self.previous_text = config.get("elevenlabs_previous_text")
        self.base64 = base64
        self.json = json
        self.asyncio = asyncio
        self.websockets = websockets
        self.ConnectionClosed = ConnectionClosed

        if self.commit_strategy not in {"manual", "vad"}:
            logger.warning(
                "Unsupported ElevenLabs commit strategy '%s', defaulting to manual",
                self.commit_strategy,
            )
            self.commit_strategy = "manual"
        logger.info("Setting up ElevenLabs Realtime Speech client completed")

    async def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        """
        Transcribe audio stream using ElevenLabs Realtime Speech-to-Text API.

        """
        logger.debug("Transcribing audio stream (elevenlabs realtime)")

        try:
            return await self._transcribe_stream_async(audio_generator)
        except Exception as e:
            logger.error(f"Error transcribing audio with ElevenLabs: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    async def _transcribe_stream_async(self, audio_generator: Iterator[bytes]) -> str:
        import urllib.parse

        uri = self._build_ws_uri(urllib.parse.urlencode)
        logger.debug("Connecting to ElevenLabs WebSocket: %s", uri)

        extra_headers = {}
        if self.api_key:
            extra_headers["xi-api-key"] = self.api_key

        send_done = self.asyncio.Event()
        commit_sent = self.asyncio.Event()

        full_transcript = []

        try:
            async with self.websockets.connect(
                uri,
                extra_headers=extra_headers if extra_headers else None,
            ) as websocket:
                send_task = self.asyncio.create_task(
                    self._send_audio(websocket, audio_generator, send_done, commit_sent)
                )
                receive_task = self.asyncio.create_task(
                    self._receive_transcripts(
                        websocket, send_done, commit_sent, full_transcript
                    )
                )

                await send_task
                result = await receive_task
                return result.strip() if result else ""

        except self.ConnectionClosed as e:
            logger.error(f"ElevenLabs WebSocket closed: {str(e)}")
            return " ".join(full_transcript).strip()
        except Exception as e:
            logger.error(f"Error in ElevenLabs async transcription: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    async def _send_audio(
        self,
        websocket,
        audio_generator: Iterator[bytes],
        send_done,
        commit_sent,
    ):
        first_sent = False
        pending_chunk = None
        previous_text = (self.previous_text or "").strip()
        if previous_text:
            previous_text = previous_text[:50]

        async for chunk in self._async_generator(audio_generator):
            if not chunk:
                continue
            if pending_chunk is None:
                pending_chunk = chunk
                continue

            await self._send_chunk(
                websocket,
                pending_chunk,
                commit=False,
                include_previous=not first_sent,
                previous_text=previous_text,
            )
            first_sent = True
            pending_chunk = chunk

        if pending_chunk is not None:
            commit = self.commit_strategy == "manual"
            await self._send_chunk(
                websocket,
                pending_chunk,
                commit=commit,
                include_previous=not first_sent,
                previous_text=previous_text,
            )
            if commit:
                commit_sent.set()

        send_done.set()

    async def _send_chunk(
        self,
        websocket,
        chunk: bytes,
        commit: bool,
        include_previous: bool,
        previous_text: str,
    ):
        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": self.base64.b64encode(chunk).decode("utf-8"),
            "sample_rate": self.sample_rate,
        }
        if include_previous and previous_text:
            payload["previous_text"] = previous_text
        if self.commit_strategy == "manual":
            payload["commit"] = bool(commit)

        await websocket.send(self.json.dumps(payload))

    async def _receive_transcripts(
        self,
        websocket,
        send_done,
        commit_sent,
        full_transcript: List[str],
    ) -> str:
        error_types = {
            "auth_error",
            "quota_exceeded",
            "transcriber_error",
            "input_error",
            "error",
            "commit_throttled",
            "unaccepted_terms",
            "scribe_auth_error",
            "scribe_quota_exceeded_error",
            "scribe_transcriber_error",
            "scribe_input_error",
            "scribe_error",
            "scribe_throttled_error",
            "scribe_rate_limited_error",
            "scribe_unaccepted_terms_error",
            "scribe_queue_overflow_error",
            "scribe_resource_exhausted_error",
            "scribe_session_time_limit_exceeded_error",
            "scribe_chunk_size_exceeded_error",
            "scribe_insufficient_audio_activity_error",
        }

        while True:
            try:
                message = await self.asyncio.wait_for(
                    websocket.recv(), timeout=self.response_timeout_sec
                )
            except self.asyncio.TimeoutError:
                if send_done.is_set():
                    break
                continue
            except self.ConnectionClosed:
                break

            logger.info("Received message from ElevenLabs: %s", message)

            try:
                response = self.json.loads(message)
            except Exception:
                logger.warning("Non-JSON message from ElevenLabs: %s", message)
                continue

            message_type = response.get("message_type")
            if not message_type:
                continue

            if message_type == "session_started":
                logger.debug("ElevenLabs session started")
                continue

            if message_type == "partial_transcript":
                partial_text = response.get("text", "")
                if partial_text:
                    logger.debug("ElevenLabs partial: %s", partial_text)
                continue

            if message_type in {
                "committed_transcript",
                "committed_transcript_with_timestamps",
            }:
                text = response.get("text", "")
                if text:
                    full_transcript.append(text)
                    if self.commit_strategy == "manual" and commit_sent.is_set():
                        return " ".join(full_transcript)
                continue

            if message_type in error_types:
                logger.error(
                    "ElevenLabs error: %s",
                    response.get("message") or response.get("error") or response,
                )
                return ""

        return " ".join(full_transcript)

    async def _async_generator(self, sync_generator: Iterator[bytes]):
        try:
            for chunk in sync_generator:
                await self.asyncio.sleep(0)
                yield chunk
        except self.asyncio.CancelledError:
            logger.debug("ElevenLabs async generator cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in ElevenLabs async generator: {str(e)}")
            raise

    def _infer_audio_format(self, sample_rate: int) -> str:
        mapping = {
            8000: "pcm_8000",
            16000: "pcm_16000",
            22050: "pcm_22050",
            24000: "pcm_24000",
            44100: "pcm_44100",
            48000: "pcm_48000",
        }
        audio_format = mapping.get(sample_rate)
        if not audio_format:
            logger.warning(
                "Unsupported ElevenLabs sample_rate_hertz %s; defaulting to pcm_16000",
                sample_rate,
            )
            audio_format = "pcm_16000"
        return audio_format

    def _build_ws_uri(self, urlencode):
        query_params = {
            "model_id": self.model_id,
            "audio_format": self.audio_format,
            "commit_strategy": self.commit_strategy,
        }

        if self.language_code:
            query_params["language_code"] = self.language_code

        if self.token:
            query_params["token"] = self.token

        if self.include_timestamps:
            query_params["include_timestamps"] = "true"

        if self.include_language_detection:
            query_params["include_language_detection"] = "true"

        if self.enable_logging is not None:
            query_params["enable_logging"] = (
                "true" if bool(self.enable_logging) else "false"
            )

        if self.commit_strategy == "vad":
            if self.vad_silence_threshold_secs is not None:
                query_params["vad_silence_threshold_secs"] = str(
                    self.vad_silence_threshold_secs
                )
            if self.vad_threshold is not None:
                query_params["vad_threshold"] = str(self.vad_threshold)
            if self.min_speech_duration_ms is not None:
                query_params["min_speech_duration_ms"] = str(
                    self.min_speech_duration_ms
                )
            if self.min_silence_duration_ms is not None:
                query_params["min_silence_duration_ms"] = str(
                    self.min_silence_duration_ms
                )

        return (
            "wss://api.elevenlabs.io/v1/speech-to-text/realtime?"
            + urlencode(query_params)
        )


class SonioxSpeechRecognition(SpeechRecognitionService):
    """
    Soniox Real-time WebSocket Speech-to-Text implementation.
    """

    def setup_client(self, config):
        import json
        import asyncio
        import websockets
        from websockets.exceptions import ConnectionClosed

        logger.info("Setting up Soniox Realtime Speech client")

        api_key = os.environ.get("SONIOX_API_KEY") or config.get("soniox_api_key")
        if not api_key:
            raise ValueError(
                "Soniox API key is not provided. Set SONIOX_API_KEY environment variable "
                "or 'soniox_api_key' in configuration."
            )

        self.api_key = api_key
        self.model = config.get("soniox_model", "stt-rt-v3")
        self.audio_format = config.get("soniox_audio_format", "pcm_s16le")
        self.sample_rate = config.get("sample_rate_hertz", 16000)
        self.num_channels = config.get("soniox_num_channels", 1)
        self.language_hints = config.get("soniox_language_hints")
        if not self.language_hints:
            language_code = config.get("language_code")
            if language_code:
                self.language_hints = [language_code]
        self.language_hints_strict = config.get("soniox_language_hints_strict")
        self.enable_endpoint_detection = config.get(
            "soniox_enable_endpoint_detection"
        )
        self.enable_language_identification = config.get(
            "soniox_enable_language_identification"
        )
        self.enable_speaker_diarization = config.get(
            "soniox_enable_speaker_diarization"
        )
        self.client_reference_id = config.get("soniox_client_reference_id")
        self.response_timeout_sec = config.get("soniox_response_timeout_sec", 15)
        self.send_finalize = bool(config.get("soniox_send_finalize", False))

        self.json = json
        self.asyncio = asyncio
        self.websockets = websockets
        self.ConnectionClosed = ConnectionClosed

    async def transcribe_stream(self, audio_generator: Iterator[bytes], config) -> str:
        logger.debug("Transcribing audio stream (soniox realtime)")

        try:
            return await self._transcribe_stream_async(audio_generator)
        except Exception as e:
            logger.error(f"Error transcribing audio with Soniox: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    async def _transcribe_stream_async(self, audio_generator: Iterator[bytes]) -> str:
        uri = "wss://stt-rt.soniox.com/transcribe-websocket"
        logger.debug("Connecting to Soniox WebSocket: %s", uri)

        send_done = self.asyncio.Event()
        transcript_parts = []
        last_partial = ""

        # Transcribes audio stream; handles connection and exceptions
        try:
            async with self.websockets.connect(uri) as websocket:
                config_message = self._build_config_message()
                logger.debug("Sending Soniox config message: %s", config_message)
                await websocket.send(self.json.dumps(config_message))

                send_task = self.asyncio.create_task(
                    self._send_audio(websocket, audio_generator, send_done)
                )
                receive_task = self.asyncio.create_task(
                    self._receive_transcripts(
                        websocket, send_done, transcript_parts, last_partial
                    )
                )

                await send_task
                result = await receive_task
                return result.strip() if result else ""

        except self.ConnectionClosed as e:
            logger.error(f"Soniox WebSocket closed: {str(e)}")
            return "".join(transcript_parts).strip()
        except Exception as e:
            logger.error(f"Error in Soniox async transcription: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def _build_config_message(self) -> dict:
        message = {
            "api_key": self.api_key,
            "model": self.model,
            "audio_format": self.audio_format,
        }

        if self.audio_format != "auto":
            message["sample_rate"] = self.sample_rate
            message["num_channels"] = self.num_channels

        if self.language_hints:
            message["language_hints"] = self.language_hints

        if self.language_hints_strict is not None:
            message["language_hints_strict"] = bool(self.language_hints_strict)

        if self.enable_endpoint_detection is not None:
            message["enable_endpoint_detection"] = bool(
                self.enable_endpoint_detection
            )

        if self.enable_language_identification is not None:
            message["enable_language_identification"] = bool(
                self.enable_language_identification
            )

        if self.enable_speaker_diarization is not None:
            message["enable_speaker_diarization"] = bool(
                self.enable_speaker_diarization
            )

        if self.client_reference_id:
            message["client_reference_id"] = self.client_reference_id

        return message

    async def _send_audio(self, websocket, audio_generator: Iterator[bytes], send_done):
        async for chunk in self._async_generator(audio_generator):
            if not chunk:
                continue
            await websocket.send(chunk)

        if self.send_finalize:
            finalize = self.json.dumps({"type": "finalize"})
            logger.debug("Sending finalize message: %s", finalize)
            await websocket.send(finalize)

        await websocket.send(b"")
        send_done.set()

    async def _receive_transcripts(
        self,
        websocket,
        send_done,
        transcript_parts: List[str],
        last_partial: str,
    ) -> str:
        # Receives and processes transcript messages from websocket
        while True:
            try:
                message = await self.asyncio.wait_for(
                    websocket.recv(), timeout=self.response_timeout_sec
                )
            except self.asyncio.TimeoutError:
                if send_done.is_set():
                    logger.info("Soniox WebSocket timed out after send_done was set")
                    break
                continue
            except self.ConnectionClosed:
                logger.info("Soniox WebSocket closed")
                break

            logger.debug("Received message from Soniox: %s", message)

            if not message:
                continue
            if isinstance(message, (bytes, bytearray)):
                try:
                    message = message.decode("utf-8")
                except Exception:
                    logger.warning("Non-UTF8 message from Soniox")
                    continue

            try:
                response = self.json.loads(message)
            except Exception:
                logger.warning("Non-JSON message from Soniox: %s", message)
                continue

            if response.get("error_code"):
                logger.error(
                    "Soniox error %s: %s",
                    response.get("error_code"),
                    response.get("error_message"),
                )
                return ""

            if response.get("finished"):
                logger.info("Soniox transcription finished")
                break

            tokens = response.get("tokens", [])
            if tokens:
                partial_parts = []
                for token in tokens:
                    text = token.get("text", "")
                    if not text:
                        continue
                    if token.get("is_final"):
                        if text == "<fin>":
                            logger.debug("Soniox final transcript received")
                            break
                        transcript_parts.append(text)
                    else:
                        partial_parts.append(text)
                if partial_parts:
                    last_partial = "".join(partial_parts)
                elif send_done.is_set():
                    logger.debug("Soniox: all parts are final")
                    break

        if transcript_parts:
            return "".join(transcript_parts)
        logger.warning("No final transcript received from Soniox, returning last partial: %s", last_partial)
        return (last_partial or "").strip()

    async def _async_generator(self, sync_generator: Iterator[bytes]):
        try:
            for chunk in sync_generator:
                await self.asyncio.sleep(0)
                yield chunk
        except self.asyncio.CancelledError:
            logger.debug("Soniox async generator cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Soniox async generator: {str(e)}")
            raise


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
        emotion_engine: Optional[EmotionEngine] = None,
    ) -> None:
        """
        Initialize the SpeechTranscriber.

        Args:
            button (Button): The AIY Kit button object.
            leds (Leds): The AIY Kit LED object.
            config (Config): The application configuration object.
            cleaning (Optional[Callable]): Optional callback function to clean the audio stream.
            timezone (Optional[str]): The timezone of the current location.
            emotion_engine (Optional[EmotionEngine]): Optional emotion detection engine.
        """
        self.button = button
        self.leds = leds
        self.config = config
        self.emotion_engine = emotion_engine
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
        self.cleaning_task = None
        self.last_clean_date: Optional[datetime.date] = None
        self.timezone: str = get_timezone() if timezone is None else timezone
        self.task_manager = BackgroundTaskManager(config, timezone)
        if cleaning:
            self.task_manager.set_cleaning_routine(cleaning)

    async def check_and_schedule_tasks(self) -> None:
        """Check and run scheduled background tasks."""
        await self.task_manager.check_and_run_tasks()

    def setup_speech_service(self):
        service_name = self.config.get("speech_recognition_service", "yandex").lower()
        if service_name == "google":
            self.speech_service = GoogleSpeechRecognition()
        elif service_name == "yandex":
            self.speech_service = YandexSpeechRecognition()
        elif service_name == "openai":
            logger.info("using openai realtime speech recognition")
            self.speech_service = OpenAISpeechRecognition()
        elif service_name == "elevenlabs":
            logger.info("using elevenlabs realtime speech recognition")
            self.speech_service = ElevenLabsSpeechRecognition()
        elif service_name == "soniox":
            logger.info("using soniox realtime speech recognition")
            self.speech_service = SonioxSpeechRecognition()
        else:
            raise ValueError(f"Unsupported speech recognition service: {service_name}")
        self.speech_service.setup_client(self.config)

    async def transcribe_speech(
        self, player_process: Optional[ResponsePlayer] = None
    ) -> tuple:
        """
        Transcribe speech from the microphone input, including pre and post buffering.
        Optionally runs emotion detection in parallel with STT.

        Args:
            player_process (Optional[ResponsePlayer]): Object representing a running audio player.

        Returns:
            tuple: (transcribed_text, emotion_annotation) where emotion_annotation is
                   a string like "[User emotion: excited (0.82)]" or empty string.
        """

        chunks_deque = deque()
        status = RecordingStatus.NOT_STARTED
        prebuffer_chunks_to_skip = 0

        async def generate_audio_chunks():
            nonlocal status, chunks_deque, player_process, prebuffer_chunks_to_skip

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
                        if player_process.is_playing():
                            logger.info(
                                "Stopping playback; clearing %s buffered audio chunks",
                                len(chunks_deque),
                            )
                            chunks_deque.clear()
                            player_process.stop()
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

                await self.check_and_schedule_tasks()

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

                if (status == RecordingStatus.NOT_STARTED) and self.button.state == ButtonState.PRESSED:
                    stop_playing()
                    start_listening()
                    prebuffer_chunks_to_skip = len(chunks_deque)
                    logger.info(f"{len(chunks_deque)} audio chunks buffered")
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

                if status == RecordingStatus.STARTED and self.button.state != ButtonState.PRESSED:
                    start_processing()
                    status = RecordingStatus.FINISHED

        logger.info("Press the button and speak")

        with Recorder() as recorder:
            audio_generator = generate_audio_chunks()

            async for _ in audio_generator:
                if status != RecordingStatus.NOT_STARTED:
                    break

            logger.info("Processing audio...")

            try:
                if prebuffer_chunks_to_skip > 0:
                    prebuffer_chunks_to_skip -= 1

                emotion_limit_sec = self.config.get("emotion_audio_limit_sec", 5.0)
                emotion_max_chunks = None
                try:
                    if emotion_limit_sec is not None:
                        limit_sec = float(emotion_limit_sec)
                        if limit_sec <= 0:
                            emotion_max_chunks = 0
                        else:
                            chunk_duration = float(
                                self.audio_recording_chunk_duration_sec
                            )
                            if chunk_duration > 0:
                                emotion_max_chunks = int(limit_sec / chunk_duration)
                            else:
                                emotion_max_chunks = None
                except Exception as e:
                    logger.error(
                        "Invalid emotion_audio_limit_sec (%s): %s",
                        emotion_limit_sec,
                        str(e),
                    )

                debug_wav = None
                debug_wav_path = None
                if self.config.get("stt_debug_recording_enabled", False):
                    import wave

                    debug_wav_path = self.config.get("stt_debug_recording_path")
                    if not debug_wav_path:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_wav_path = os.path.join(
                            "logs", f"stt_debug_{timestamp}.wav"
                        )
                    os.makedirs(os.path.dirname(debug_wav_path), exist_ok=True)
                    debug_wav = wave.open(debug_wav_path, "wb")
                    debug_wav.setnchannels(1)
                    debug_wav.setsampwidth(2)
                    debug_wav.setframerate(self.audio_sample_rate)
                    logger.info("Recording STT debug audio to %s", debug_wav_path)

                # Two queues distribute audio chunks to STT and emotion detection.
                # Different queue types match each consumer's execution model:
                #
                # - stt_queue (queue.Queue): STT consumes chunks from a sync generator.
                #   Sync queue's .get() blocks the thread until chunk arrives.
                #
                # - emotion_queue (asyncio.Queue): Emotion detection is async-native,
                #   using await with websockets. Async queue's .get() yields control
                #   to event loop while waiting, enabling true concurrency.
                stt_queue = queue.Queue()
                emotion_queue = asyncio.Queue()

                async def fill_queues():
                    """Distribute each audio chunk to both consumers."""
                    remaining_prebuffer = prebuffer_chunks_to_skip
                    emotion_chunks_sent = 0
                    emotion_done = False
                    async for chunk in audio_generator:
                        stt_queue.put(chunk)        # Non-blocking for sync queue
                        if not emotion_done:
                            if remaining_prebuffer > 0:
                                remaining_prebuffer -= 1
                            elif (emotion_max_chunks is None) or (
                                emotion_chunks_sent < emotion_max_chunks
                            ):
                                await emotion_queue.put(chunk)  # Async put
                                emotion_chunks_sent += 1
                                if (
                                    emotion_max_chunks is not None
                                    and emotion_chunks_sent >= emotion_max_chunks
                                ):
                                    await emotion_queue.put(None)
                                    emotion_done = True
                            else:
                                await emotion_queue.put(None)
                                emotion_done = True
                        if debug_wav is not None:
                            debug_wav.writeframes(chunk)
                    # Signal end-of-stream to both consumers
                    stt_queue.put(None)
                    if not emotion_done:
                        await emotion_queue.put(None)

                def stt_generator():
                    """Sync generator for STT - blocks on queue.get() in thread."""
                    while True:
                        chunk = stt_queue.get()  # Blocks thread until chunk available
                        if chunk is None:
                            break
                        yield chunk

                async def emotion_generator():
                    """Async generator for emotion - awaits on queue.get() in event loop."""
                    while True:
                        chunk = await emotion_queue.get()  # Yields to event loop
                        if chunk is None:
                            break
                        yield chunk

                async def run_stt():
                    """Run STT in the event loop (async service interface)."""
                    try:
                        return await self.speech_service.transcribe_stream(
                            stt_generator(), self.config
                        )
                    except Exception as e:
                        logger.error(f"Error in STT: {str(e)}")
                        return ""

                async def run_emotion_detection():
                    """Run emotion detection in event loop (async-native)."""
                    if not self.emotion_engine:
                        # Drain queue to avoid blocking fill_queues
                        async for _ in emotion_generator():
                            pass
                        return ""

                    try:
                        result = await self.emotion_engine.detect_stream(
                            emotion_generator(),
                            sample_rate=self.audio_sample_rate
                        )
                        annotation = format_annotation(result)
                        if annotation:
                            logger.debug(f"Detected emotion: {annotation}")
                        return annotation
                    except Exception as e:
                        logger.error(f"Error in emotion detection: {str(e)}")
                        return ""

                # Start distributing chunks to both queues
                fill_queues_task = asyncio.create_task(fill_queues())

                # Run STT and emotion detection in parallel, both streaming
                logger.debug("Starting parallel streaming STT and emotion detection")
                text, emotion_annotation = await asyncio.gather(
                    run_stt(),
                    run_emotion_detection()
                )

                # Ensure fill_queues completes
                await fill_queues_task

                logger.debug(
                    f"Parallel processing complete: text={len(text) if text else 0} chars, "
                    f"emotion={emotion_annotation}"
                )

            except Exception as e:
                logger.error(f"Error transcribing speech: {str(e)}")
                text = ""
                emotion_annotation = ""
            finally:
                if debug_wav is not None:
                    try:
                        debug_wav.close()
                        logger.info("STT debug audio saved to %s", debug_wav_path)
                    except Exception as e:
                        logger.error(f"Error closing STT debug audio file: {str(e)}")

        return text, emotion_annotation

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
