"""
Emotion Detection Engine module.

This module provides abstract and concrete implementations of emotion detection engines,
including Hume AI's Expression Measurement API for voice emotion detection.
"""

import asyncio
import base64
import hashlib
import inspect
import io
import json
import logging
import os
import random
import threading
import time
import uuid
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Optional

import aiohttp

try:
    from src.tools import retry_async
except Exception:
    def retry_async(
        max_retries: int = 5,
        initial_retry_delay: float = 1,
        backoff_factor: float = 2,
        jitter_factor: float = 0.1,
    ):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except Exception:
                        if attempt == (max_retries - 1):
                            raise
                        retry_time = initial_retry_delay * (backoff_factor**attempt)
                        jitter = random.uniform(0, jitter_factor * retry_time)
                        await asyncio.sleep(retry_time + jitter)

            return wrapper

        return decorator

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """
    Result of emotion detection from audio.

    Attributes:
        top_emotions: List of (emotion_name, score) tuples for top emotions
        raw_scores: Dictionary of all emotion scores (48 prosody dimensions for Hume)
        confidence: Overall confidence score (typically the top emotion's score)
    """
    top_emotions: list[tuple[str, float]] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


class EmotionEngine(ABC):
    """
    Abstract base class for emotion detection engines (mirrors STTEngine pattern).
    """

    @abstractmethod
    async def detect(self, audio_file: str) -> Optional[EmotionResult]:
        """
        Detect emotions from audio file.

        Args:
            audio_file: Path to the audio file to analyze.

        Returns:
            EmotionResult with detected emotions, or None on failure.
        """
        pass

    async def detect_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000
    ) -> Optional[EmotionResult]:
        """
        Detect emotions from streaming audio chunks.

        Args:
            audio_chunks: Async iterator yielding audio data chunks.
            sample_rate: Audio sample rate in Hz.

        Returns:
            EmotionResult with detected emotions, or None on failure.
        """
        # Default implementation: not supported, subclasses can override
        return None

def format_annotation(result: Optional[EmotionResult]) -> str:
    """
    Format emotion result as text annotation for LLM.

    Args:
        result: EmotionResult to format.

    Returns:
        Formatted string like "[User emotion: excited (0.82), curious (0.65)]"
        or empty string if no emotions detected.
    """
    if not result or not result.top_emotions:
        return ""
    emotions_str = ", ".join(
        f"{name} ({score:.2f})"
        for name, score in result.top_emotions
    )
    return f"[User emotion: {emotions_str}]"


class EmotionComparisonLogger:
    """Append-only JSONL logger for primary vs shadow emotion comparisons."""

    def __init__(self, path: str, include_raw_scores: bool = True) -> None:
        self.path = path
        self.include_raw_scores = include_raw_scores
        self._lock = threading.Lock()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def record(self, row: dict) -> None:
        """Persist one comparison row."""
        if not self.path:
            return

        payload = json.loads(json.dumps(row))
        payload.setdefault("recorded_at", self._now_iso())
        if not self.include_raw_scores:
            for side in ("primary", "shadow"):
                if isinstance(payload.get(side), dict):
                    payload[side].pop("raw_scores", None)

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")


def get_emotion_comparison_logger(config) -> Optional[EmotionComparisonLogger]:
    """Return a process-shared emotion comparison logger stored on config."""
    if not config.get("emotion_comparison_enabled", False):
        return None
    existing = getattr(config, "_emotion_comparison_logger_instance", None)
    if existing:
        return existing
    path = config.get("emotion_comparison_log_path", "logs/emotion_comparison.jsonl")
    include_raw_scores = config.get("emotion_comparison_include_raw_scores", True)
    tracker = EmotionComparisonLogger(path, include_raw_scores=include_raw_scores)
    setattr(config, "_emotion_comparison_logger_instance", tracker)
    return tracker


class ComparisonEmotionEngine(EmotionEngine):
    """
    Runs primary and shadow emotion engines in parallel.

    The primary result is returned to the caller. The shadow result is collected
    for comparison logging during a transition period.
    """

    def __init__(
        self,
        primary_provider: str,
        primary_engine: EmotionEngine,
        shadow_provider: str,
        shadow_engine: EmotionEngine,
        comparison_logger: Optional[EmotionComparisonLogger] = None,
    ) -> None:
        self.primary_provider = primary_provider
        self.primary_engine = primary_engine
        self.shadow_provider = shadow_provider
        self.shadow_engine = shadow_engine
        self.comparison_logger = comparison_logger
        self._pending_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _normalize_emotion_name(name: str) -> str:
        return str(name or "").strip().lower()

    @staticmethod
    def _serialize_result(result: Optional[EmotionResult]) -> Optional[dict]:
        if result is None:
            return None
        return {
            "top_emotions": [
                {"name": name, "score": round(score, 6)}
                for name, score in result.top_emotions
            ],
            "raw_scores": {
                name: round(score, 6)
                for name, score in result.raw_scores.items()
            },
            "confidence": round(result.confidence, 6),
        }

    async def _run_provider(self, provider: str, coro) -> dict:
        started = time.perf_counter()
        error = None
        result = None
        try:
            result = await coro
        except Exception as e:
            error = str(e)
            logger.error("Emotion provider %s failed inside comparison engine: %s", provider, e)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "provider": provider,
            "result": result,
            "error": error,
            "latency_ms": latency_ms,
        }

    def _build_record(self, mode: str, audio_meta: dict, primary_outcome: dict, shadow_outcome: dict) -> dict:
        primary_result = primary_outcome.get("result")
        shadow_result = shadow_outcome.get("result")
        primary_latency = primary_outcome.get("latency_ms")
        shadow_latency = shadow_outcome.get("latency_ms")

        primary_names = [
            self._normalize_emotion_name(name)
            for name, _ in (primary_result.top_emotions if primary_result else [])
        ]
        shadow_names = [
            self._normalize_emotion_name(name)
            for name, _ in (shadow_result.top_emotions if shadow_result else [])
        ]
        primary_top_1 = primary_names[0] if primary_names else None
        shadow_top_1 = shadow_names[0] if shadow_names else None

        return {
            "schema_version": 1,
            "comparison_id": uuid.uuid4().hex,
            "mode": mode,
            "primary_provider": self.primary_provider,
            "shadow_provider": self.shadow_provider,
            "audio": audio_meta,
            "timing": {
                "primary_return_latency_ms": primary_latency,
                "shadow_latency_ms": shadow_latency,
                "shadow_extra_latency_ms": (
                    round(max(0.0, shadow_latency - primary_latency), 2)
                    if isinstance(primary_latency, (int, float))
                    and isinstance(shadow_latency, (int, float))
                    else None
                ),
            },
            "primary": {
                "latency_ms": primary_latency,
                "error": primary_outcome.get("error"),
                **(
                    self._serialize_result(primary_result)
                    or {"top_emotions": [], "raw_scores": {}, "confidence": 0.0}
                ),
            },
            "shadow": {
                "latency_ms": shadow_latency,
                "error": shadow_outcome.get("error"),
                **(
                    self._serialize_result(shadow_result)
                    or {"top_emotions": [], "raw_scores": {}, "confidence": 0.0}
                ),
            },
            "agreement": {
                "top_1_match": bool(primary_top_1 and primary_top_1 == shadow_top_1),
                "primary_top_1": primary_top_1,
                "shadow_top_1": shadow_top_1,
                "top_overlap_count": len(set(primary_names) & set(shadow_names)),
            },
        }

    def _log_comparison(self, mode: str, audio_meta: dict, primary_outcome: dict, shadow_outcome: dict) -> None:
        if not self.comparison_logger:
            return
        try:
            self.comparison_logger.record(
                self._build_record(mode, audio_meta, primary_outcome, shadow_outcome)
            )
        except Exception as e:
            logger.warning(f"Failed to persist emotion comparison data: {e}")

    async def _finish_shadow_logging(
        self,
        mode: str,
        audio_meta: dict,
        primary_outcome: dict,
        shadow_task: asyncio.Task,
    ) -> None:
        shadow_outcome = await shadow_task
        self._log_comparison(mode, audio_meta, primary_outcome, shadow_outcome)

    def _track_background_task(self, task: asyncio.Task) -> None:
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    @staticmethod
    async def _iter_pcm_bytes(pcm_bytes: bytes, chunk_size: int = 3200) -> AsyncIterator[bytes]:
        for i in range(0, len(pcm_bytes), chunk_size):
            yield pcm_bytes[i:i + chunk_size]

    async def detect(self, audio_file: str) -> Optional[EmotionResult]:
        """
        Run both providers on an audio file and return the primary result.
        """
        audio_meta = {
            "file_path": audio_file,
        }
        try:
            if os.path.exists(audio_file):
                audio_bytes = os.path.getsize(audio_file)
                audio_meta["file_size_bytes"] = audio_bytes
        except Exception:
            pass

        primary_task = asyncio.create_task(
            self._run_provider(self.primary_provider, self.primary_engine.detect(audio_file))
        )
        shadow_task = asyncio.create_task(
            self._run_provider(self.shadow_provider, self.shadow_engine.detect(audio_file))
        )

        primary_outcome = await primary_task
        if shadow_task.done():
            shadow_outcome = shadow_task.result()
            self._log_comparison("file", audio_meta, primary_outcome, shadow_outcome)
        else:
            background = asyncio.create_task(
                self._finish_shadow_logging("file", audio_meta, primary_outcome, shadow_task)
            )
            self._track_background_task(background)
        return primary_outcome.get("result")

    async def detect_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000
    ) -> Optional[EmotionResult]:
        """
        Run both providers on the same buffered PCM stream and return the primary result.
        """
        pcm_data = bytearray()
        async for chunk in audio_chunks:
            pcm_data.extend(chunk)

        if not pcm_data:
            return None

        audio_bytes = bytes(pcm_data)
        duration_ms = round((len(audio_bytes) / (sample_rate * 2)) * 1000, 2) if sample_rate else None
        audio_meta = {
            "sample_rate": sample_rate,
            "pcm_bytes": len(audio_bytes),
            "duration_ms": duration_ms,
            "pcm_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        }

        primary_task = asyncio.create_task(
            self._run_provider(
                self.primary_provider,
                self.primary_engine.detect_stream(self._iter_pcm_bytes(audio_bytes), sample_rate),
            )
        )
        shadow_task = asyncio.create_task(
            self._run_provider(
                self.shadow_provider,
                self.shadow_engine.detect_stream(self._iter_pcm_bytes(audio_bytes), sample_rate),
            )
        )

        primary_outcome = await primary_task
        if shadow_task.done():
            shadow_outcome = shadow_task.result()
            self._log_comparison("stream", audio_meta, primary_outcome, shadow_outcome)
        else:
            background = asyncio.create_task(
                self._finish_shadow_logging("stream", audio_meta, primary_outcome, shadow_task)
            )
            self._track_background_task(background)
        return primary_outcome.get("result")


class HumeEmotionEngine(EmotionEngine):
    """
    Hume AI WebSocket-based emotion detection using prosody model.

    Uses Hume's Expression Measurement WebSocket API for real-time
    emotion detection from voice prosody (tone, rhythm, etc.).

    Supports both file-based and streaming detection.
    """

    def __init__(self, config):
        """
        Initialize HumeEmotionEngine.

        Args:
            config: Configuration object containing Hume API settings.
        """
        self.api_key = os.environ.get("HUME_API_KEY") or config.get("hume_api_key")
        if not self.api_key:
            raise ValueError(
                "Hume API key is not provided. Set HUME_API_KEY environment variable "
                "or 'hume_api_key' in configuration."
            )

        self.websocket_url = "wss://api.hume.ai/v0/stream/models"
        self.top_n = config.get("emotion_top_n", 3)
        self.min_score = config.get("emotion_min_score", 0.3)
        self.timeout = config.get("emotion_detection_timeout", 10.0)

        logger.info(
            f"HumeEmotionEngine initialized: top_n={self.top_n}, "
            f"min_score={self.min_score}, timeout={self.timeout}"
        )

    @retry_async()
    async def _connect_websocket_with_retry(self, websockets):
        connection = websockets.connect(
            self.websocket_url,
            extra_headers={"X-Hume-Api-Key": self.api_key},
        )
        if inspect.isawaitable(connection):
            return await connection
        if hasattr(connection, "__aenter__"):
            return await connection.__aenter__()
        return connection

    async def _close_websocket(self, ws) -> None:
        close_method = getattr(ws, "close", None)
        if close_method is None:
            return
        close_result = close_method()
        if inspect.isawaitable(close_result):
            await close_result

    async def detect(self, audio_file: str) -> Optional[EmotionResult]:
        """
        Send audio file to Hume WebSocket API, get emotion predictions.

        Args:
            audio_file: Path to the audio file to analyze.

        Returns:
            EmotionResult with detected emotions, or None on failure.
        """
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. Run: pip install websockets")
            return None

        try:
            ws = await self._connect_websocket_with_retry(websockets)
            try:
                # Read and encode audio file
                with open(audio_file, "rb") as f:
                    audio_data = f.read()
                audio_b64 = base64.b64encode(audio_data).decode()

                # Send request with prosody model config
                request = {
                    "data": audio_b64,
                    "models": {"prosody": {}},
                }
                await ws.send(json.dumps(request))
                logger.debug(f"Sent {len(audio_data)} bytes of audio to Hume API")

                # Receive and parse response with timeout
                response_text = await asyncio.wait_for(ws.recv(), self.timeout)
                response = json.loads(response_text)
                logger.debug(f"Received Hume API response: {response.keys()}")

                return self._parse_response(response)
            finally:
                await self._close_websocket(ws)

        except asyncio.TimeoutError:
            logger.warning(f"Emotion detection timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return None

    async def detect_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000
    ) -> Optional[EmotionResult]:
        """
        Stream audio chunks to Hume WebSocket API, get emotion predictions.

        Opens WebSocket connection and streams audio chunks as they arrive.
        Returns final emotion prediction after all chunks are processed.

        Args:
            audio_chunks: Async iterator yielding raw PCM audio data chunks.
            sample_rate: Audio sample rate in Hz (default 16000).

        Returns:
            EmotionResult with detected emotions, or None on failure.
        """
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. Run: pip install websockets")
            return None

        try:
            ws = await self._connect_websocket_with_retry(websockets)
            try:
                # Accumulate chunks and send periodically for better predictions
                accumulated_data = bytearray()
                chunk_count = 0
                last_result = None

                async for chunk in audio_chunks:
                    accumulated_data.extend(chunk)
                    chunk_count += 1

                # Send all accumulated audio at end of stream
                if accumulated_data:
                    # Wrap raw PCM data in WAV format - Hume API requires valid audio format
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(bytes(accumulated_data))
                    wav_data = wav_buffer.getvalue()

                    audio_b64 = base64.b64encode(wav_data).decode()
                    request = {
                        "data": audio_b64,
                        "models": {"prosody": {}},
                    }
                    await ws.send(json.dumps(request))
                    logger.debug(
                        f"Sent {len(wav_data)} bytes WAV ({len(accumulated_data)} PCM, "
                        f"{chunk_count} chunks, {sample_rate}Hz) to Hume API"
                    )

                    # Receive final response
                    response_text = await asyncio.wait_for(ws.recv(), self.timeout)
                    logger.debug(f"Received Hume API raw response: {response_text[:500]}")
                    response = json.loads(response_text)
                    last_result = self._parse_response(response)

                return last_result
            finally:
                await self._close_websocket(ws)

        except asyncio.TimeoutError:
            logger.warning(f"Emotion detection stream timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"Emotion detection stream failed: {e}")
            return None

    def _parse_response(self, response: dict) -> Optional[EmotionResult]:
        """
        Parse Hume API response into EmotionResult.

        Args:
            response: Raw JSON response from Hume API.

        Returns:
            EmotionResult with parsed emotions, or None if no predictions.
        """
        # Check for API error response
        if "error" in response:
            logger.error(f"Hume API returned error: {response['error']}")
            return None

        try:
            # Navigate to prosody predictions
            prosody = response.get("prosody", {})
            predictions = prosody.get("predictions", [])

            if not predictions:
                logger.warning(
                    f"No prosody predictions in Hume response. "
                    f"Response keys: {list(response.keys())}"
                )
                return None

            # Get emotions from first prediction
            emotions = predictions[0].get("emotions", [])
            if not emotions:
                logger.warning("No emotions in prosody predictions")
                return None

            # Sort by score, filter by threshold, take top N
            sorted_emotions = sorted(
                emotions,
                key=lambda e: e.get("score", 0),
                reverse=True
            )

            top = [
                (e["name"], e["score"])
                for e in sorted_emotions
                if e.get("score", 0) >= self.min_score
            ][:self.top_n]

            if not top:
                logger.debug(
                    f"No emotions above threshold {self.min_score}. "
                    f"Highest score: {sorted_emotions[0].get('score', 0):.2f} "
                    f"({sorted_emotions[0].get('name', 'unknown')})"
                )

            raw = {e["name"]: e["score"] for e in emotions}
            confidence = top[0][1] if top else 0.0

            logger.debug(f"Detected emotions: {top}")
            return EmotionResult(
                top_emotions=top,
                raw_scores=raw,
                confidence=confidence
            )

        except (KeyError, IndexError, TypeError) as e:
            logger.error(
                f"Failed to parse Hume response: {e}. "
                f"Response structure: {json.dumps(response, default=str)[:500]}"
            )
            return None


class GeminiEmotionEngine(EmotionEngine):
    """
    Gemini-based emotion detection using multimodal audio understanding.

    This is a best-effort fallback for voice emotion detection. It asks Gemini
    to score a fixed emotion taxonomy from the acoustic delivery of the voice
    and maps that into EmotionResult.
    """

    EMOTION_LABELS = (
        "neutral",
        "calm",
        "joy",
        "excitement",
        "interest",
        "curiosity",
        "surprise",
        "sadness",
        "frustration",
        "anger",
        "anxiety",
        "fear",
    )

    MIME_TYPES = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".aiff": "audio/aiff",
        ".aif": "audio/aiff",
    }

    def __init__(self, config):
        """
        Initialize GeminiEmotionEngine.

        Args:
            config: Configuration object containing Gemini API settings.
        """
        self.api_key = os.environ.get("GEMINI_API_KEY") or config.get("gemini_api_key")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not provided. Set GEMINI_API_KEY environment variable "
                "or 'gemini_api_key' in configuration."
            )

        self.base_url = config.get(
            "gemini_base_url", "https://generativelanguage.googleapis.com"
        ).rstrip("/")
        self.model_id = config.get(
            "gemini_emotion_model", "gemini-3.1-flash-lite-preview"
        )
        self.url = f"{self.base_url}/v1beta/models/{self.model_id}:generateContent"
        self.top_n = config.get("emotion_top_n", 3)
        self.min_score = config.get("emotion_min_score", 0.3)
        self.timeout = config.get("emotion_detection_timeout", 10.0)
        self.thinking_level = config.get("gemini_emotion_thinking_level", "minimal")

        logger.info(
            "GeminiEmotionEngine initialized: model=%s top_n=%s min_score=%s timeout=%s",
            self.model_id,
            self.top_n,
            self.min_score,
            self.timeout,
        )

    def _guess_mime_type(self, audio_file: str) -> str:
        _, extension = os.path.splitext(audio_file.lower())
        return self.MIME_TYPES.get(extension, "audio/wav")

    @classmethod
    def _classification_prompt(cls) -> str:
        labels = ", ".join(cls.EMOTION_LABELS)
        return (
            "You classify vocal emotion from audio. "
            "Prioritize paralinguistic cues in the voice: pitch, energy, rhythm, pacing, "
            "hesitation, tension, loudness, stability, breathiness, and intensity. "
            "Ignore the literal meaning of the words as much as possible. "
            "If lexical meaning conflicts with vocal delivery, prioritize vocal delivery. "
            "Score every label independently from 0.0 to 1.0. "
            "Use higher scores only when there is audible evidence in the voice. "
            "If uncertain, keep scores low and favor neutral or calm. "
            "Return JSON only with numeric scores for these labels: "
            f"{labels}."
        )

    @classmethod
    def _response_schema(cls) -> dict:
        return {
            "type": "OBJECT",
            "properties": {
                label: {"type": "NUMBER"}
                for label in cls.EMOTION_LABELS
            },
            "required": list(cls.EMOTION_LABELS),
        }

    def _generation_config(self) -> dict:
        generation_config = {
            "temperature": 0,
            "max_output_tokens": 256,
            "response_mime_type": "application/json",
            "response_schema": self._response_schema(),
        }
        if self.model_id.startswith("gemini-3"):
            generation_config["thinking_config"] = {
                "thinking_level": self.thinking_level,
            }
        return generation_config

    @retry_async()
    async def _request_scores(self, audio_bytes: bytes, mime_type: str) -> Optional[dict]:
        audio_b64 = base64.b64encode(audio_bytes).decode()
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": self._classification_prompt()},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": audio_b64,
                            }
                        },
                    ],
                }
            ],
            "generation_config": self._generation_config(),
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.url,
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    response_text = await response.text()
                    response.raise_for_status()
        except asyncio.TimeoutError:
            logger.warning("Gemini emotion detection timed out after %ss", self.timeout)
            return None
        except Exception as e:
            logger.error(f"Gemini emotion detection failed: {e}")
            return None

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Gemini emotion detection returned non-JSON: {e}")
            return None

        candidates = data.get("candidates", [])
        if not candidates:
            logger.warning("Gemini emotion detection returned no candidates")
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if part.get("text")).strip()
        if not text:
            logger.warning("Gemini emotion detection returned no text payload")
            return None

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(
                "Gemini emotion detection returned invalid JSON text: %s payload=%s",
                e,
                text[:500],
            )
            return None
        return parsed

    def _parse_scores(self, scores_payload: dict) -> Optional[EmotionResult]:
        try:
            raw = {}
            for label in self.EMOTION_LABELS:
                score = float(scores_payload.get(label, 0.0))
                raw[label] = max(0.0, min(1.0, score))

            sorted_emotions = sorted(raw.items(), key=lambda item: item[1], reverse=True)
            top = [
                (name, score)
                for name, score in sorted_emotions
                if score >= self.min_score
            ][:self.top_n]
            confidence = sorted_emotions[0][1] if sorted_emotions else 0.0

            logger.debug(f"Gemini detected emotions: {top}")
            return EmotionResult(
                top_emotions=top,
                raw_scores=raw,
                confidence=confidence,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to parse Gemini emotion scores: {e}")
            return None

    async def detect(self, audio_file: str) -> Optional[EmotionResult]:
        """
        Detect emotions from an audio file using Gemini.

        Args:
            audio_file: Path to the audio file to analyze.

        Returns:
            EmotionResult with detected emotions, or None on failure.
        """
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None

        try:
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            logger.error(f"Failed to read audio file {audio_file}: {e}")
            return None

        scores_payload = await self._request_scores(
            audio_bytes,
            self._guess_mime_type(audio_file),
        )
        if not scores_payload:
            return None
        return self._parse_scores(scores_payload)

    async def detect_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000
    ) -> Optional[EmotionResult]:
        """
        Detect emotions from streaming PCM audio chunks using Gemini.

        The current pipeline accumulates a short utterance and sends it as WAV
        audio at end-of-turn.
        """
        accumulated_data = bytearray()
        async for chunk in audio_chunks:
            accumulated_data.extend(chunk)

        if not accumulated_data:
            return None

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(accumulated_data))

        scores_payload = await self._request_scores(wav_buffer.getvalue(), "audio/wav")
        if not scores_payload:
            return None
        return self._parse_scores(scores_payload)


def normalize_emotion_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases used in config."""
    value = str(provider or "hume").strip().lower()
    aliases = {
        "off": "none",
        "noop": "none",
        "disabled": "none",
    }
    return aliases.get(value, value)


def default_shadow_emotion_provider(primary_provider: str) -> Optional[str]:
    """Return a sensible default shadow provider for comparison mode."""
    if primary_provider == "hume":
        return "gemini"
    if primary_provider == "gemini":
        return "hume"
    return None


def create_provider_emotion_engine(provider: str, config) -> EmotionEngine:
    """Instantiate a concrete provider-specific emotion engine."""
    normalized = normalize_emotion_provider(provider)
    if normalized == "none":
        return NoOpEmotionEngine()
    if normalized == "gemini":
        return GeminiEmotionEngine(config)
    if normalized == "hume":
        return HumeEmotionEngine(config)
    raise ValueError(f"Unsupported emotion engine provider: {provider}")


def create_emotion_engine(config) -> EmotionEngine:
    """Create the configured emotion engine, optionally with shadow comparison."""
    if not config.get("emotion_detection_enabled", False):
        return NoOpEmotionEngine()

    primary_provider = normalize_emotion_provider(config.get("emotion_engine_provider", "hume"))
    primary_engine = create_provider_emotion_engine(primary_provider, config)

    if primary_provider == "none":
        return primary_engine

    if not config.get("emotion_comparison_enabled", False):
        return primary_engine

    shadow_provider = normalize_emotion_provider(
        config.get(
            "emotion_comparison_shadow_provider",
            default_shadow_emotion_provider(primary_provider),
        )
    )
    if shadow_provider in {"", "none"}:
        logger.info("Emotion comparison requested but no shadow provider is configured")
        return primary_engine
    if shadow_provider == primary_provider:
        logger.warning(
            "Emotion comparison shadow provider matches primary provider (%s); disabling comparison",
            primary_provider,
        )
        return primary_engine

    try:
        shadow_engine = create_provider_emotion_engine(shadow_provider, config)
    except Exception as e:
        logger.warning(
            "Failed to initialize shadow emotion engine %s: %s. Using primary only.",
            shadow_provider,
            e,
        )
        return primary_engine

    logger.info(
        "Emotion comparison enabled: primary=%s shadow=%s",
        primary_provider,
        shadow_provider,
    )
    return ComparisonEmotionEngine(
        primary_provider=primary_provider,
        primary_engine=primary_engine,
        shadow_provider=shadow_provider,
        shadow_engine=shadow_engine,
        comparison_logger=get_emotion_comparison_logger(config),
    )


class NoOpEmotionEngine(EmotionEngine):
    """
    No-operation emotion engine for when emotion detection is disabled.
    """

    async def detect(self, audio_file: str) -> Optional[EmotionResult]:
        """
        Return None (no emotion detection).

        Args:
            audio_file: Ignored.

        Returns:
            None
        """
        return None

    async def detect_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000
    ) -> Optional[EmotionResult]:
        """
        Return None (no emotion detection).

        Args:
            audio_chunks: Ignored.
            sample_rate: Ignored.

        Returns:
            None
        """
        # Consume the iterator to avoid blocking
        async for _ in audio_chunks:
            pass
        return None


async def main() -> int:
    """CLI helper to test emotion detection on a local WAV file."""
    import argparse

    from dotenv import load_dotenv

    from src.config import Config

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run the configured emotion engine against a local audio file."
    )
    parser.add_argument(
        "audio_file",
        help="Path to a local WAV/audio file to analyze",
    )
    parser.add_argument(
        "--provider",
        choices=["hume", "gemini", "none"],
        help="Override emotion_engine_provider from config",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Enable comparison mode using config defaults for shadow provider",
    )
    args = parser.parse_args()

    config = Config()
    config["emotion_detection_enabled"] = True
    if args.provider:
        config["emotion_engine_provider"] = args.provider
    if args.comparison:
        config["emotion_comparison_enabled"] = True

    engine = create_emotion_engine(config)
    result = await engine.detect(args.audio_file)

    print(f"engine={engine.__class__.__name__}")
    print(f"audio_file={args.audio_file}")
    print(f"annotation={format_annotation(result)}")
    if result is None:
        print("result=None")
    else:
        print(
            json.dumps(
                {
                    "top_emotions": result.top_emotions,
                    "raw_scores": result.raw_scores,
                    "confidence": result.confidence,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(asyncio.run(main()))
