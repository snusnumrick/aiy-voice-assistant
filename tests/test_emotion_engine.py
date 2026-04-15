"""
Tests for the emotion detection engine module.
"""

import asyncio
import json
import os
import tempfile
import unittest
import wave
from unittest.mock import AsyncMock, MagicMock, patch

from src.emotion_engine import (
    ComparisonEmotionEngine,
    EmotionEngine,
    EmotionResult,
    GeminiEmotionEngine,
    HumeEmotionEngine,
    NoOpEmotionEngine,
    format_annotation,
)


class FakeEmotionEngine(EmotionEngine):
    """Minimal test double for comparison-engine tests."""

    def __init__(self, result=None, wait_event=None, error=None):
        self.result = result
        self.wait_event = wait_event
        self.error = error

    async def detect(self, audio_file: str):
        if self.wait_event is not None:
            await self.wait_event.wait()
        if self.error is not None:
            raise self.error
        return self.result

    async def detect_stream(self, audio_chunks, sample_rate: int = 16000):
        async for _ in audio_chunks:
            pass
        if self.wait_event is not None:
            await self.wait_event.wait()
        if self.error is not None:
            raise self.error
        return self.result


class TestEmotionResult(unittest.TestCase):
    """Tests for EmotionResult dataclass."""

    def test_empty_result(self):
        """Test creating an empty EmotionResult."""
        result = EmotionResult()
        self.assertEqual(result.top_emotions, [])
        self.assertEqual(result.raw_scores, {})
        self.assertEqual(result.confidence, 0.0)

    def test_with_emotions(self):
        """Test creating EmotionResult with data."""
        result = EmotionResult(
            top_emotions=[("joy", 0.85), ("excitement", 0.72)],
            raw_scores={"joy": 0.85, "excitement": 0.72, "sadness": 0.1},
            confidence=0.85,
        )
        self.assertEqual(len(result.top_emotions), 2)
        self.assertEqual(result.top_emotions[0], ("joy", 0.85))
        self.assertEqual(result.confidence, 0.85)


class TestNoOpEmotionEngine(unittest.TestCase):
    """Tests for NoOpEmotionEngine."""

    def test_detect_returns_none(self):
        """Test that NoOpEmotionEngine always returns None."""
        engine = NoOpEmotionEngine()
        result = asyncio.run(engine.detect("/some/audio.wav"))
        self.assertIsNone(result)

    def test_format_annotation_empty(self):
        """Test that format_annotation returns empty string for None."""
        annotation = format_annotation(None)
        self.assertEqual(annotation, "")


class TestEmotionEngineFormatAnnotation(unittest.TestCase):
    """Tests for the format_annotation method."""

    def test_format_with_emotions(self):
        """Test formatting with multiple emotions."""
        result = EmotionResult(
            top_emotions=[("excited", 0.82), ("curious", 0.65)],
            raw_scores={},
            confidence=0.82,
        )
        annotation = format_annotation(result)
        self.assertEqual(annotation, "[User emotion: excited (0.82), curious (0.65)]")

    def test_format_with_single_emotion(self):
        """Test formatting with single emotion."""
        result = EmotionResult(
            top_emotions=[("joy", 0.91)],
            raw_scores={},
            confidence=0.91,
        )
        annotation = format_annotation(result)
        self.assertEqual(annotation, "[User emotion: joy (0.91)]")

    def test_format_with_no_emotions(self):
        """Test formatting with empty emotions list."""
        result = EmotionResult(
            top_emotions=[],
            raw_scores={},
            confidence=0.0,
        )
        annotation = format_annotation(result)
        self.assertEqual(annotation, "")

    def test_format_with_none_result(self):
        """Test formatting with None result."""
        annotation = format_annotation(None)
        self.assertEqual(annotation, "")


class TestHumeEmotionEngine(unittest.TestCase):
    """Tests for HumeEmotionEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = MagicMock()
        self.mock_config.get = MagicMock(side_effect=lambda key, default=None: {
            "emotion_top_n": 3,
            "emotion_min_score": 0.3,
            "emotion_detection_timeout": 10.0,
        }.get(key, default))

    def test_init_requires_api_key(self):
        """Test that initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            self.mock_config.get = MagicMock(return_value=None)
            with self.assertRaises(ValueError) as context:
                HumeEmotionEngine(self.mock_config)
            self.assertIn("Hume API key", str(context.exception))

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict(os.environ, {"HUME_API_KEY": "test-key"}):
            engine = HumeEmotionEngine(self.mock_config)
            self.assertEqual(engine.api_key, "test-key")

    def test_init_with_config_api_key(self):
        """Test initialization with API key from config."""
        with patch.dict(os.environ, {}, clear=True):
            self.mock_config.get = MagicMock(side_effect=lambda key, default=None: {
                "hume_api_key": "config-key",
                "emotion_top_n": 3,
                "emotion_min_score": 0.3,
                "emotion_detection_timeout": 10.0,
            }.get(key, default))
            engine = HumeEmotionEngine(self.mock_config)
            self.assertEqual(engine.api_key, "config-key")

    def test_parse_response_valid(self):
        """Test parsing a valid Hume API response."""
        with patch.dict(os.environ, {"HUME_API_KEY": "test-key"}):
            engine = HumeEmotionEngine(self.mock_config)

            response = {
                "prosody": {
                    "predictions": [{
                        "emotions": [
                            {"name": "Joy", "score": 0.85},
                            {"name": "Excitement", "score": 0.72},
                            {"name": "Interest", "score": 0.45},
                            {"name": "Sadness", "score": 0.1},  # Below threshold
                        ]
                    }]
                }
            }

            result = engine._parse_response(response)

            self.assertIsNotNone(result)
            self.assertEqual(len(result.top_emotions), 3)
            self.assertEqual(result.top_emotions[0], ("Joy", 0.85))
            self.assertEqual(result.top_emotions[1], ("Excitement", 0.72))
            self.assertEqual(result.top_emotions[2], ("Interest", 0.45))
            self.assertEqual(result.confidence, 0.85)

    def test_parse_response_empty_predictions(self):
        """Test parsing response with no predictions."""
        with patch.dict(os.environ, {"HUME_API_KEY": "test-key"}):
            engine = HumeEmotionEngine(self.mock_config)

            response = {
                "prosody": {
                    "predictions": []
                }
            }

            result = engine._parse_response(response)
            self.assertIsNone(result)

    def test_parse_response_no_prosody(self):
        """Test parsing response without prosody key."""
        with patch.dict(os.environ, {"HUME_API_KEY": "test-key"}):
            engine = HumeEmotionEngine(self.mock_config)

            response = {}

            result = engine._parse_response(response)
            self.assertIsNone(result)

    def test_detect_file_not_found(self):
        """Test detect with non-existent file."""
        with patch.dict(os.environ, {"HUME_API_KEY": "test-key"}):
            engine = HumeEmotionEngine(self.mock_config)
            result = asyncio.run(engine.detect("/nonexistent/audio.wav"))
            self.assertIsNone(result)


class TestGeminiEmotionEngine(unittest.TestCase):
    """Tests for GeminiEmotionEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = MagicMock()
        self.mock_config.get = MagicMock(side_effect=lambda key, default=None: {
            "gemini_emotion_model": "gemini-3.1-flash-lite-preview",
            "emotion_top_n": 3,
            "emotion_min_score": 0.3,
            "emotion_detection_timeout": 10.0,
            "gemini_emotion_thinking_level": "minimal",
        }.get(key, default))

    def test_init_requires_api_key(self):
        """Test that initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            self.mock_config.get = MagicMock(return_value=None)
            with self.assertRaises(ValueError) as context:
                GeminiEmotionEngine(self.mock_config)
            self.assertIn("Gemini API key", str(context.exception))

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            engine = GeminiEmotionEngine(self.mock_config)
            self.assertEqual(engine.api_key, "test-key")
            self.assertEqual(engine.model_id, "gemini-3.1-flash-lite-preview")

    def test_parse_scores_valid(self):
        """Test parsing valid Gemini emotion scores."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            engine = GeminiEmotionEngine(self.mock_config)
            result = engine._parse_scores(
                {
                    "neutral": 0.12,
                    "calm": 0.18,
                    "joy": 0.81,
                    "excitement": 0.74,
                    "interest": 0.66,
                    "curiosity": 0.29,
                    "surprise": 0.15,
                    "sadness": 0.03,
                    "frustration": 0.04,
                    "anger": 0.02,
                    "anxiety": 0.09,
                    "fear": 0.01,
                }
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.top_emotions[0], ("joy", 0.81))
            self.assertEqual(result.top_emotions[1], ("excitement", 0.74))
            self.assertEqual(result.top_emotions[2], ("interest", 0.66))
            self.assertEqual(result.confidence, 0.81)
            self.assertEqual(result.raw_scores["curiosity"], 0.29)

    def test_parse_scores_clamps_values(self):
        """Test out-of-range values are clamped safely."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            engine = GeminiEmotionEngine(self.mock_config)
            result = engine._parse_scores(
                {
                    "neutral": 1.2,
                    "calm": -0.5,
                    "joy": 0.4,
                    "excitement": 0.3,
                    "interest": 0.2,
                    "curiosity": 0.1,
                    "surprise": 0.0,
                    "sadness": 0.0,
                    "frustration": 0.0,
                    "anger": 0.0,
                    "anxiety": 0.0,
                    "fear": 0.0,
                }
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.raw_scores["neutral"], 1.0)
            self.assertEqual(result.raw_scores["calm"], 0.0)

    def test_detect_stream_uses_request_scores(self):
        """Test streaming detection wraps PCM and parses returned scores."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            engine = GeminiEmotionEngine(self.mock_config)
            engine._request_scores = AsyncMock(
                return_value={
                    "neutral": 0.1,
                    "calm": 0.2,
                    "joy": 0.12,
                    "excitement": 0.88,
                    "interest": 0.64,
                    "curiosity": 0.15,
                    "surprise": 0.08,
                    "sadness": 0.04,
                    "frustration": 0.05,
                    "anger": 0.03,
                    "anxiety": 0.07,
                    "fear": 0.01,
                }
            )

            async def async_chunk_generator():
                for _ in range(4):
                    yield b"\x00" * 3200

            result = asyncio.run(engine.detect_stream(async_chunk_generator()))

            self.assertIsNotNone(result)
            self.assertEqual(result.top_emotions[0], ("excitement", 0.88))
            engine._request_scores.assert_awaited_once()


class TestHumeEmotionEngineIntegration(unittest.TestCase):
    """Integration-style tests for HumeEmotionEngine with mocked WebSocket."""

    def create_test_wav_file(self):
        """Create a temporary WAV file for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # Write 1 second of silence
            wav_file.writeframes(b'\x00' * 32000)
        return temp_file.name

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = MagicMock()
        self.mock_config.get = MagicMock(side_effect=lambda key, default=None: {
            "emotion_top_n": 3,
            "emotion_min_score": 0.3,
            "emotion_detection_timeout": 10.0,
        }.get(key, default))
        self.test_wav = self.create_test_wav_file()

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_wav):
            os.unlink(self.test_wav)

    def test_detect_success(self):
        """Test successful emotion detection flow."""
        # Create a mock websockets module
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps({
            "prosody": {
                "predictions": [{
                    "emotions": [
                        {"name": "Joy", "score": 0.85},
                        {"name": "Excitement", "score": 0.72},
                    ]
                }]
            }
        })

        mock_websockets = MagicMock()
        mock_websockets.connect.return_value.__aenter__.return_value = mock_ws

        with patch.dict(os.environ, {"HUME_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"websockets": mock_websockets}):
                engine = HumeEmotionEngine(self.mock_config)
                result = asyncio.run(engine.detect(self.test_wav))

                self.assertIsNotNone(result)
                self.assertEqual(len(result.top_emotions), 2)
                self.assertEqual(result.top_emotions[0], ("Joy", 0.85))

                # Verify WebSocket was called correctly
                mock_ws.send.assert_called_once()
                sent_data = json.loads(mock_ws.send.call_args[0][0])
                self.assertIn("data", sent_data)
                self.assertIn("models", sent_data)
                self.assertIn("prosody", sent_data["models"])

    def test_detect_stream_success(self):
        """Test successful streaming emotion detection."""
        # Create a mock websockets module
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps({
            "prosody": {
                "predictions": [{
                    "emotions": [
                        {"name": "Excitement", "score": 0.90},
                        {"name": "Interest", "score": 0.75},
                    ]
                }]
            }
        })

        mock_websockets = MagicMock()
        mock_websockets.connect.return_value.__aenter__.return_value = mock_ws

        async def async_chunk_generator():
            """Simulate streaming audio chunks."""
            for _ in range(5):
                yield b'\x00' * 3200  # 100ms of silence at 16kHz

        with patch.dict(os.environ, {"HUME_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"websockets": mock_websockets}):
                engine = HumeEmotionEngine(self.mock_config)
                result = asyncio.run(engine.detect_stream(async_chunk_generator()))

                self.assertIsNotNone(result)
                self.assertEqual(len(result.top_emotions), 2)
                self.assertEqual(result.top_emotions[0], ("Excitement", 0.90))

                # Verify WebSocket was called correctly
                mock_ws.send.assert_called_once()
                sent_data = json.loads(mock_ws.send.call_args[0][0])
                self.assertIn("data", sent_data)
                self.assertIn("models", sent_data)


class TestComparisonEmotionEngine(unittest.TestCase):
    """Tests for parallel comparison mode."""

    def test_shadow_engine_does_not_block_primary_result(self):
        """Primary result should return before a slow shadow engine finishes."""
        primary_result = EmotionResult(
            top_emotions=[("Joy", 0.9)],
            raw_scores={"Joy": 0.9},
            confidence=0.9,
        )
        shadow_result = EmotionResult(
            top_emotions=[("excitement", 0.7)],
            raw_scores={"excitement": 0.7},
            confidence=0.7,
        )

        async def scenario(log_path: str):
            shadow_gate = asyncio.Event()
            engine = ComparisonEmotionEngine(
                primary_provider="hume",
                primary_engine=FakeEmotionEngine(result=primary_result),
                shadow_provider="gemini",
                shadow_engine=FakeEmotionEngine(result=shadow_result, wait_event=shadow_gate),
                comparison_logger=None,
            )

            async def audio_chunks():
                yield b"\x00" * 3200

            result = await asyncio.wait_for(engine.detect_stream(audio_chunks()), timeout=0.05)
            self.assertEqual(result.top_emotions[0], ("Joy", 0.9))
            self.assertFalse(os.path.exists(log_path))

            shadow_gate.set()
            await asyncio.wait_for(asyncio.gather(*engine._pending_tasks), timeout=0.2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            asyncio.run(scenario(os.path.join(tmp_dir, "emotion_compare.jsonl")))

    def test_comparison_logging_writes_jsonl_row(self):
        """Comparison mode should persist a row with both provider outcomes."""
        primary_result = EmotionResult(
            top_emotions=[("Joy", 0.9)],
            raw_scores={"Joy": 0.9},
            confidence=0.9,
        )
        shadow_result = EmotionResult(
            top_emotions=[("Joy", 0.8), ("Interest", 0.5)],
            raw_scores={"Joy": 0.8, "Interest": 0.5},
            confidence=0.8,
        )

        async def scenario(log_path: str):
            logger = MagicMock()
            logger.record = MagicMock()
            engine = ComparisonEmotionEngine(
                primary_provider="hume",
                primary_engine=FakeEmotionEngine(result=primary_result),
                shadow_provider="gemini",
                shadow_engine=FakeEmotionEngine(result=shadow_result),
                comparison_logger=logger,
            )

            async def audio_chunks():
                yield b"\x00" * 6400

            result = await engine.detect_stream(audio_chunks(), sample_rate=16000)
            self.assertEqual(result.top_emotions[0], ("Joy", 0.9))
            logger.record.assert_called_once()
            payload = logger.record.call_args[0][0]
            self.assertEqual(payload["mode"], "stream")
            self.assertEqual(payload["primary_provider"], "hume")
            self.assertEqual(payload["shadow_provider"], "gemini")
            self.assertTrue(payload["agreement"]["top_1_match"])
            self.assertEqual(payload["audio"]["pcm_bytes"], 6400)
            self.assertIn("primary_return_latency_ms", payload["timing"])
            self.assertIn("shadow_latency_ms", payload["timing"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            asyncio.run(scenario(os.path.join(tmp_dir, "emotion_compare.jsonl")))


class TestNoOpEmotionEngineStream(unittest.TestCase):
    """Tests for NoOpEmotionEngine streaming."""

    def test_detect_stream_returns_none(self):
        """Test that NoOpEmotionEngine.detect_stream returns None."""
        engine = NoOpEmotionEngine()

        async def async_chunk_generator():
            for _ in range(3):
                yield b'\x00' * 1600

        result = asyncio.run(engine.detect_stream(async_chunk_generator()))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
