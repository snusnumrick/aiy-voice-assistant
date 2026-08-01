"""Tests for passive speaker embeddings and progressive local profiles."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.speaker_engine import (
    GeminiSpeakerEmbeddingProvider,
    NoOpSpeakerEngine,
    ProfiledSpeakerEngine,
    SpeakerEmbedding,
    SpeakerEmbeddingProvider,
    SpeakerProfileStore,
    SpeakerResult,
    WeSpeakerRemoteEmbeddingProvider,
    cosine_similarity,
    create_speaker_engine,
    format_speaker_annotation,
    normalize_embedding,
)


class DictConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


class FakeEmbeddingProvider(SpeakerEmbeddingProvider):
    def __init__(self, embedding):
        self.embedding = embedding
        self.calls = []

    async def embed(self, audio_bytes: bytes, mime_type: str):
        self.calls.append((audio_bytes, mime_type))
        return self.embedding


class TestSpeakerMath(unittest.TestCase):
    def test_normalize_and_cosine_similarity(self):
        normalized = normalize_embedding([3.0, 4.0])
        self.assertAlmostEqual(normalized[0], 0.6)
        self.assertAlmostEqual(normalized[1], 0.8)
        self.assertAlmostEqual(cosine_similarity(normalized, normalized), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

class TestSpeakerProfiles(unittest.TestCase):
    def test_progressive_profile_update_and_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SpeakerProfileStore(os.path.join(temp_dir, "profiles.json"))
            first = SpeakerEmbedding([1.0, 0.0], "test:model:2")
            second = SpeakerEmbedding(normalize_embedding([0.9, 0.1]), "test:model:2")

            self.assertTrue(store.update("Anton", first, minimum_similarity=0.7))
            self.assertTrue(store.update("Anton", second, minimum_similarity=0.7))
            with self.assertLogs("src.speaker_engine", level="INFO") as captured_logs:
                result = store.match(second, minimum_similarity=0.7)

            self.assertIsNotNone(result)
            self.assertEqual(result.name, "Anton")
            self.assertGreater(result.confidence, 0.9)
            self.assertTrue(
                any("Speaker recognized from current audio" in line for line in captured_logs.output)
            )
            with open(os.path.join(temp_dir, "profiles.json"), encoding="utf-8") as profile_file:
                saved = json.load(profile_file)
            profile = saved["spaces"]["test:model:2"]["speakers"]["Anton"]
            self.assertEqual(profile["sample_count"], 2)

    def test_inconsistent_declared_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SpeakerProfileStore(os.path.join(temp_dir, "profiles.json"))
            self.assertTrue(
                store.update(
                    "Anton",
                    SpeakerEmbedding([1.0, 0.0], "test:model:2"),
                    minimum_similarity=0.7,
                )
            )
            self.assertFalse(
                store.update(
                    "Anton",
                    SpeakerEmbedding([0.0, 1.0], "test:model:2"),
                    minimum_similarity=0.7,
                )
            )

    def test_ambiguous_match_is_rejected_by_margin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SpeakerProfileStore(os.path.join(temp_dir, "profiles.json"))
            query = SpeakerEmbedding([1.0, 0.0], "test:model:2")
            near_query = SpeakerEmbedding(normalize_embedding([0.99, 0.1]), "test:model:2")
            store.update("Anton", query, minimum_similarity=0.7)
            store.update("Tanya", near_query, minimum_similarity=0.7)

            with self.assertLogs("src.speaker_engine", level="INFO") as captured_logs:
                result = store.match(
                    query,
                    minimum_similarity=0.7,
                    minimum_margin=0.05,
                )

            self.assertIsNone(result)
            self.assertTrue(
                any("Speaker match is ambiguous" in line for line in captured_logs.output)
            )


class TestProfiledSpeakerEngine(unittest.TestCase):
    def test_llm_declared_turn_enrolls_and_later_turn_recognizes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding = SpeakerEmbedding([1.0, 0.0], "fake:model:2")
            provider = FakeEmbeddingProvider(embedding)
            engine = ProfiledSpeakerEngine(
                provider,
                DictConfig(
                    {
                        "speaker_profiles_path": os.path.join(temp_dir, "profiles.json"),
                        "speaker_match_threshold": 0.8,
                        "speaker_profile_update_threshold": 0.7,
                    }
                ),
            )

            async def chunks():
                yield b"\x00\x00" * 160

            first_turn = engine.begin_turn()
            generated = asyncio.run(engine.embed_stream(chunks()))
            original, recognized = engine.resolve_transcript(
                "Any natural self-introduction can be here",
                generated,
                turn_id=first_turn,
            )
            self.assertIsNone(recognized)
            with self.assertLogs("src.speaker_engine", level="INFO") as declaration_logs:
                self.assertTrue(engine.declare_speaker("Anton"))
            self.assertFalse(engine.declare_speaker("anton"))
            self.assertFalse(engine.declare_speaker("Maria"))

            second_turn = engine.begin_turn()
            _, recognized = engine.resolve_transcript(
                "Continue",
                embedding,
                turn_id=second_turn,
            )

            self.assertEqual(original, "Any natural self-introduction can be here")
            self.assertEqual(recognized.name, "Anton")
            self.assertTrue(
                any(
                    "Applied LLM speaker declaration to current-turn embedding" in line
                    for line in declaration_logs.output
                )
            )
            profile = engine.profile_store.data["spaces"]["fake:model:2"]["speakers"]["Anton"]
            self.assertEqual(profile["sample_count"], 1)
            self.assertEqual(provider.calls[0][1], "audio/wav")
            self.assertTrue(provider.calls[0][0].startswith(b"RIFF"))

    def test_llm_declaration_waits_for_late_embedding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ProfiledSpeakerEngine(
                FakeEmbeddingProvider(None),
                DictConfig(
                    {
                        "speaker_profiles_path": os.path.join(temp_dir, "profiles.json"),
                        "speaker_context_max_age_sec": 300,
                    }
                ),
            )
            turn_id = engine.begin_turn()
            engine.resolve_transcript("Unrestricted introduction wording", None, turn_id=turn_id)
            with self.assertLogs("src.speaker_engine", level="INFO") as declaration_logs:
                self.assertTrue(engine.declare_speaker("Anton"))
                engine.resolve_transcript(
                    "Unrestricted introduction wording",
                    SpeakerEmbedding([1.0, 0.0], "fake:model:2"),
                    turn_id=turn_id,
                )

            engine.begin_turn()
            text, result = engine.resolve_transcript("Continue the story", None)

            self.assertEqual(text, "Continue the story")
            self.assertEqual(result.name, "Anton")
            self.assertEqual(result.source, "context")
            self.assertTrue(
                any("Stored LLM speaker declaration pending" in line for line in declaration_logs.output)
            )
            self.assertTrue(
                any(
                    "Applied pending LLM speaker declaration to late embedding" in line
                    for line in declaration_logs.output
                )
            )

            matched = engine.profile_store.match(
                SpeakerEmbedding([1.0, 0.0], "fake:model:2"),
                minimum_similarity=0.8,
            )
            self.assertEqual(matched.name, "Anton")

    def test_recent_context_survives_until_mismatch_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ProfiledSpeakerEngine(
                FakeEmbeddingProvider(None),
                DictConfig(
                    {
                        "speaker_profiles_path": os.path.join(temp_dir, "profiles.json"),
                        "speaker_match_threshold": 0.8,
                        "speaker_context_misses_to_clear": 2,
                    }
                ),
            )
            enrolled = SpeakerEmbedding([1.0, 0.0], "fake:model:2")
            different = SpeakerEmbedding([0.0, 1.0], "fake:model:2")

            first_turn = engine.begin_turn()
            engine.resolve_transcript("Introduction", enrolled, turn_id=first_turn)
            engine.declare_speaker("Anton")

            second_turn = engine.begin_turn()
            _, first_mismatch = engine.resolve_transcript(
                "Continuation",
                different,
                turn_id=second_turn,
            )
            third_turn = engine.begin_turn()
            _, second_mismatch = engine.resolve_transcript(
                "Another speaker",
                different,
                turn_id=third_turn,
            )

            self.assertEqual(first_mismatch.name, "Anton")
            self.assertEqual(first_mismatch.source, "context")
            self.assertIsNone(second_mismatch)

    def test_annotation_format(self):
        self.assertEqual(
            format_speaker_annotation(SpeakerResult("Anton", 1.0, "declared")),
            "[User speaker: Anton]",
        )
        self.assertEqual(
            format_speaker_annotation(SpeakerResult("Anton", 0.824)),
            "[User speaker: Anton (0.82)]",
        )
        self.assertEqual(
            format_speaker_annotation(SpeakerResult("Anton", 1.0, "context")),
            "",
        )


class TestGeminiSpeakerEmbeddingProvider(unittest.TestCase):
    def setUp(self):
        self.config = DictConfig(
            {
                "gemini_speaker_embedding_model": "gemini-embedding-2",
                "speaker_embedding_dimension": 2,
                "speaker_detection_timeout": 5,
            }
        )

    def test_init_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                GeminiSpeakerEmbeddingProvider(self.config)

    def test_embed_parses_and_normalizes_response(self):
        response = MagicMock()
        response.text = AsyncMock(return_value=json.dumps({"embedding": {"values": [3, 4]}}))
        response.raise_for_status = MagicMock()
        post_context = MagicMock()
        post_context.__aenter__ = AsyncMock(return_value=response)
        post_context.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.post.return_value = post_context
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            provider = GeminiSpeakerEmbeddingProvider(self.config)
            with patch("src.speaker_engine.aiohttp.ClientSession", return_value=session_context):
                result = asyncio.run(provider.embed(b"audio", "audio/wav"))

        self.assertEqual(result.space_id, "gemini:gemini-embedding-2:2")
        self.assertAlmostEqual(result.values[0], 0.6)
        self.assertAlmostEqual(result.values[1], 0.8)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["output_dimensionality"], 2)
        self.assertEqual(payload["content"]["parts"][0]["inline_data"]["mime_type"], "audio/wav")


class TestWeSpeakerRemoteEmbeddingProvider(unittest.TestCase):
    def setUp(self):
        self.config = DictConfig(
            {
                "wespeaker_embedding_url": "https://speaker.example/v1/embeddings",
                "wespeaker_model_id": "resnet34-lm-voxceleb",
                "wespeaker_embedding_dimension": 2,
                "wespeaker_detection_timeout": 5,
            }
        )

    def test_init_requires_service_url(self):
        with self.assertRaises(ValueError):
            WeSpeakerRemoteEmbeddingProvider(DictConfig())

    def test_embed_posts_wav_and_parses_normalized_embedding(self):
        response = MagicMock()
        response.text = AsyncMock(
            return_value=json.dumps(
                {
                    "embedding": [3, 4],
                    "space_id": "wespeaker:resnet34-lm-voxceleb:2",
                }
            )
        )
        response.raise_for_status = MagicMock()
        post_context = MagicMock()
        post_context.__aenter__ = AsyncMock(return_value=response)
        post_context.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.post.return_value = post_context
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"WESPEAKER_API_KEY": "test-key"}):
            provider = WeSpeakerRemoteEmbeddingProvider(self.config)
            with patch("src.speaker_engine.aiohttp.ClientSession", return_value=session_context):
                result = asyncio.run(provider.embed(b"RIFFaudio", "audio/wav"))

        self.assertEqual(result.space_id, "wespeaker:resnet34-lm-voxceleb:2")
        self.assertAlmostEqual(result.values[0], 0.6)
        self.assertAlmostEqual(result.values[1], 0.8)
        request = session.post.call_args
        self.assertEqual(request.kwargs["data"], b"RIFFaudio")
        self.assertEqual(request.kwargs["headers"]["Content-Type"], "audio/wav")
        self.assertEqual(request.kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_repeated_request_failures_pause_requests_until_cooldown(self):
        provider = WeSpeakerRemoteEmbeddingProvider(self.config)

        with patch(
            "src.speaker_engine.aiohttp.ClientSession",
            side_effect=OSError("server unavailable"),
        ) as client_session:
            for _ in range(3):
                self.assertIsNone(asyncio.run(provider.embed(b"RIFFaudio", "audio/wav")))
            self.assertIsNone(asyncio.run(provider.embed(b"RIFFaudio", "audio/wav")))

        self.assertEqual(client_session.call_count, 3)
        self.assertGreater(provider.disabled_until, 0)

    def test_successful_probe_resumes_requests_after_cooldown(self):
        response = MagicMock()
        response.text = AsyncMock(
            return_value=json.dumps(
                {
                    "embedding": [3, 4],
                    "space_id": "wespeaker:resnet34-lm-voxceleb:2",
                }
            )
        )
        response.raise_for_status = MagicMock()
        post_context = MagicMock()
        post_context.__aenter__ = AsyncMock(return_value=response)
        post_context.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.post.return_value = post_context
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=None)
        provider = WeSpeakerRemoteEmbeddingProvider(self.config)
        provider.consecutive_failures = provider.failure_threshold
        provider.disabled_until = 1.0

        with (
            patch("src.speaker_engine.time.monotonic", return_value=2.0),
            patch("src.speaker_engine.aiohttp.ClientSession", return_value=session_context),
            self.assertLogs("src.speaker_engine", level="INFO") as captured_logs,
        ):
            result = asyncio.run(provider.embed(b"RIFFaudio", "audio/wav"))

        self.assertIsNotNone(result)
        self.assertEqual(provider.consecutive_failures, 0)
        self.assertEqual(provider.disabled_until, 0.0)
        self.assertTrue(any("recognition resumed" in line for line in captured_logs.output))


class TestSpeakerFactory(unittest.TestCase):
    def test_disabled_factory_does_not_require_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            engine = create_speaker_engine(DictConfig({"speaker_recognition_enabled": False}))
        self.assertIsInstance(engine, NoOpSpeakerEngine)

    def test_wespeaker_factory_uses_remote_provider(self):
        engine = create_speaker_engine(
            DictConfig(
                {
                    "speaker_recognition_enabled": True,
                    "speaker_embedding_provider": "wespeaker",
                    "wespeaker_embedding_url": "https://speaker.example/v1/embeddings",
                }
            )
        )

        self.assertIsInstance(engine, ProfiledSpeakerEngine)
        self.assertIsInstance(engine.provider, WeSpeakerRemoteEmbeddingProvider)


if __name__ == "__main__":
    unittest.main()
