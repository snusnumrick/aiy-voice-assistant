"""
Emotion Detection Engine module.

This module provides abstract and concrete implementations of emotion detection engines,
including Hume AI's Expression Measurement API for voice emotion detection.
"""

import asyncio
import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple

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
    top_emotions: List[Tuple[str, float]] = field(default_factory=list)
    raw_scores: Dict[str, float] = field(default_factory=dict)
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

    async def detect(self, audio_file: str) -> Optional[EmotionResult]:
        """
        Send audio file to Hume WebSocket API, get emotion predictions.

        Args:
            audio_file: Path to the audio file to analyze.

        Returns:
            EmotionResult with detected emotions, or None on failure.
        """
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. Run: pip install websockets")
            return None

        try:
            async with websockets.connect(
                self.websocket_url,
                additional_headers={"X-Hume-Api-Key": self.api_key}
            ) as ws:
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

        except asyncio.TimeoutError:
            logger.warning(f"Emotion detection timed out after {self.timeout}s")
            return None
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_file}")
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
            async with websockets.connect(
                self.websocket_url,
                additional_headers={"X-Hume-Api-Key": self.api_key}
            ) as ws:
                # Accumulate chunks and send periodically for better predictions
                accumulated_data = bytearray()
                chunk_count = 0
                last_result = None

                async for chunk in audio_chunks:
                    accumulated_data.extend(chunk)
                    chunk_count += 1

                # Send all accumulated audio at end of stream
                if accumulated_data:
                    audio_b64 = base64.b64encode(bytes(accumulated_data)).decode()
                    request = {
                        "data": audio_b64,
                        "models": {"prosody": {}},
                    }
                    await ws.send(json.dumps(request))
                    logger.debug(
                        f"Sent {len(accumulated_data)} bytes ({chunk_count} chunks) to Hume API"
                    )

                    # Receive final response
                    response_text = await asyncio.wait_for(ws.recv(), self.timeout)
                    response = json.loads(response_text)
                    last_result = self._parse_response(response)

                return last_result

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
        try:
            # Navigate to prosody predictions
            prosody = response.get("prosody", {})
            predictions = prosody.get("predictions", [])

            if not predictions:
                logger.debug("No prosody predictions in Hume response")
                return None

            # Get emotions from first prediction
            emotions = predictions[0].get("emotions", [])
            if not emotions:
                logger.debug("No emotions in prosody predictions")
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

            raw = {e["name"]: e["score"] for e in emotions}
            confidence = top[0][1] if top else 0.0

            logger.debug(f"Detected emotions: {top}")
            return EmotionResult(
                top_emotions=top,
                raw_scores=raw,
                confidence=confidence
            )

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse Hume response: {e}")
            return None


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
