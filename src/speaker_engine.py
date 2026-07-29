"""Passive speaker recognition built around remotely generated audio embeddings."""

import asyncio
import base64
import io
import json
import logging
import math
import os
import time
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SpeakerEmbedding:
    """A normalized vector produced by a specific embedding model."""

    values: list[float]
    space_id: str


@dataclass
class SpeakerResult:
    """Best-effort speaker identity for one user turn."""

    name: str
    confidence: float
    source: str = "recognized"


class SpeakerEmbeddingProvider(ABC):
    """Provider interface for converting utterance audio into a speaker vector."""

    @abstractmethod
    async def embed(self, audio_bytes: bytes, mime_type: str) -> Optional[SpeakerEmbedding]:
        """Return an embedding for an encoded audio clip, or None on failure."""
        pass


class GeminiSpeakerEmbeddingProvider(SpeakerEmbeddingProvider):
    """Generate audio embeddings with the Gemini multimodal embedding API."""

    def __init__(self, config):
        self.api_key = os.environ.get("GEMINI_API_KEY") or config.get("gemini_api_key")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not provided. Set GEMINI_API_KEY environment variable "
                "or 'gemini_api_key' in configuration."
            )

        self.base_url = config.get(
            "gemini_base_url", "https://generativelanguage.googleapis.com"
        ).rstrip("/")
        self.model_id = config.get("gemini_speaker_embedding_model", "gemini-embedding-2")
        self.dimension = int(config.get("speaker_embedding_dimension", 768))
        self.timeout = float(config.get("speaker_detection_timeout", 10.0))
        self.url = f"{self.base_url}/v1beta/models/{self.model_id}:embedContent"
        self.space_id = f"gemini:{self.model_id}:{self.dimension}"

        logger.info(
            "Gemini speaker embedding provider initialized: model=%s dimension=%s timeout=%s",
            self.model_id,
            self.dimension,
            self.timeout,
        )

    async def embed(self, audio_bytes: bytes, mime_type: str) -> Optional[SpeakerEmbedding]:
        if not audio_bytes:
            return None

        payload = {
            "content": {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                        }
                    }
                ]
            },
            "output_dimensionality": self.dimension,
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
            logger.warning("Gemini speaker embedding timed out after %ss", self.timeout)
            return None
        except Exception as e:
            logger.error("Gemini speaker embedding failed: %s", e)
            return None

        try:
            data = json.loads(response_text)
            embedding = data.get("embedding")
            if embedding is None:
                embeddings = data.get("embeddings") or []
                embedding = embeddings[0] if embeddings else None
            values = embedding.get("values") if isinstance(embedding, dict) else None
            normalized = normalize_embedding(values or [])
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            logger.error("Invalid Gemini speaker embedding response: %s", e)
            return None

        if len(normalized) != self.dimension:
            logger.error(
                "Gemini returned speaker embedding dimension %s; expected %s",
                len(normalized),
                self.dimension,
            )
            return None
        return SpeakerEmbedding(values=normalized, space_id=self.space_id)


def normalize_embedding(values: list[float]) -> list[float]:
    """Return a unit-length float vector."""
    vector = [float(value) for value in values]
    magnitude = math.sqrt(sum(value * value for value in vector))
    if not vector or magnitude <= 0:
        return []
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity, tolerating unnormalized stored vectors."""
    if not left or len(left) != len(right):
        return -1.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class SpeakerProfileStore:
    """Persist compact speaker centroids, namespaced by embedding model."""

    def __init__(self, path: str, max_centroid_weight: int = 20):
        self.path = Path(path)
        self.max_centroid_weight = max(1, int(max_centroid_weight))
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "spaces": {}}
        try:
            with self.path.open(encoding="utf-8") as profile_file:
                data = json.load(profile_file)
            if not isinstance(data, dict) or not isinstance(data.get("spaces"), dict):
                raise ValueError("profile file has an invalid structure")
            return data
        except Exception as e:
            logger.error("Failed to load speaker profiles from %s: %s", self.path, e)
            return {"version": 1, "spaces": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as profile_file:
            json.dump(self.data, profile_file, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_path, self.path)

    def _speakers(self, space_id: str) -> dict:
        spaces = self.data.setdefault("spaces", {})
        space = spaces.setdefault(space_id, {"speakers": {}})
        return space.setdefault("speakers", {})

    def match(
        self,
        embedding: SpeakerEmbedding,
        minimum_similarity: float,
    ) -> Optional[SpeakerResult]:
        best_name = None
        best_score = -1.0
        for name, profile in self._speakers(embedding.space_id).items():
            centroid = profile.get("centroid", [])
            score = cosine_similarity(embedding.values, centroid)
            if score > best_score:
                best_name = name
                best_score = score
        if best_name is None:
            return None
        if best_score < minimum_similarity:
            logger.info(
                "Speaker match below threshold: best=%s similarity=%.3f threshold=%.3f",
                best_name,
                best_score,
                minimum_similarity,
            )
            return None
        logger.info(
            "Speaker recognized from current audio: name=%s similarity=%.3f threshold=%.3f",
            best_name,
            best_score,
            minimum_similarity,
        )
        return SpeakerResult(name=best_name, confidence=best_score)

    def update(
        self,
        name: str,
        embedding: SpeakerEmbedding,
        minimum_similarity: float,
    ) -> bool:
        speakers = self._speakers(embedding.space_id)
        stored_name = next(
            (candidate for candidate in speakers if candidate.casefold() == name.casefold()),
            name,
        )
        profile = speakers.get(stored_name)
        if profile is None:
            speakers[stored_name] = {
                "centroid": embedding.values,
                "sample_count": 1,
                "updated_at": int(time.time()),
            }
            self._save()
            logger.info("Created provisional speaker profile for %s", stored_name)
            return True

        centroid = profile.get("centroid", [])
        if len(centroid) != len(embedding.values):
            logger.error(
                "Rejected speaker sample for %s because profile dimensions differ",
                stored_name,
            )
            return False
        similarity = cosine_similarity(embedding.values, centroid)
        if similarity < minimum_similarity:
            logger.warning(
                "Rejected inconsistent speaker sample for %s: similarity=%.3f threshold=%.3f",
                stored_name,
                similarity,
                minimum_similarity,
            )
            return False

        sample_count = max(1, int(profile.get("sample_count", 1)))
        existing_weight = min(sample_count, self.max_centroid_weight)
        combined = [
            ((old_value * existing_weight) + new_value) / (existing_weight + 1)
            for old_value, new_value in zip(centroid, embedding.values)
        ]
        normalized = normalize_embedding(combined)
        if not normalized:
            return False
        profile.update(
            centroid=normalized,
            sample_count=sample_count + 1,
            updated_at=int(time.time()),
        )
        self._save()
        logger.info(
            "Updated speaker profile for %s: samples=%s similarity=%.3f",
            stored_name,
            sample_count + 1,
            similarity,
        )
        return True


class SpeakerEngine(ABC):
    """Conversation-facing interface for passive speaker recognition."""

    @abstractmethod
    async def embed_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000,
    ) -> Optional[SpeakerEmbedding]:
        pass

    @abstractmethod
    def begin_turn(self) -> int:
        """Begin an audio turn and return its local identifier."""
        pass

    @abstractmethod
    def declare_speaker(self, speaker_id: str) -> bool:
        """Associate an LLM-extracted identity with the current audio turn."""
        pass

    @abstractmethod
    def resolve_transcript(
        self,
        text: str,
        embedding: Optional[SpeakerEmbedding],
        turn_id: Optional[int] = None,
    ) -> tuple[str, Optional[SpeakerResult]]:
        pass


class NoOpSpeakerEngine(SpeakerEngine):
    """Disabled speaker engine which preserves the input unchanged."""

    async def embed_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000,
    ) -> Optional[SpeakerEmbedding]:
        async for _ in audio_chunks:
            pass
        return None

    def begin_turn(self) -> int:
        return 0

    def declare_speaker(self, speaker_id: str) -> bool:
        return False

    def resolve_transcript(
        self,
        text: str,
        embedding: Optional[SpeakerEmbedding],
        turn_id: Optional[int] = None,
    ) -> tuple[str, Optional[SpeakerResult]]:
        return text, None


class ProfiledSpeakerEngine(SpeakerEngine):
    """Match provider embeddings against locally persisted progressive profiles."""

    def __init__(self, provider: SpeakerEmbeddingProvider, config):
        self.provider = provider
        self.match_threshold = float(config.get("speaker_match_threshold", 0.8))
        self.update_threshold = float(config.get("speaker_profile_update_threshold", 0.7))
        self.context_max_age_sec = float(config.get("speaker_context_max_age_sec", 300))
        self.context_misses_to_clear = max(
            1,
            int(config.get("speaker_context_misses_to_clear", 2)),
        )
        self.current_speaker: Optional[SpeakerResult] = None
        self.current_speaker_at = 0.0
        self.current_speaker_misses = 0
        self._active_turn_id = 0
        self._turn_embeddings: dict[int, SpeakerEmbedding] = {}
        self._turn_declarations: dict[int, str] = {}
        self._trained_declaration_turns: set[int] = set()
        self.profile_store = SpeakerProfileStore(
            config.get("speaker_profiles_path", "speaker_profiles.json"),
            max_centroid_weight=int(config.get("speaker_profile_max_centroid_weight", 20)),
        )

    async def embed_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000,
    ) -> Optional[SpeakerEmbedding]:
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
        return await self.provider.embed(wav_buffer.getvalue(), "audio/wav")

    def begin_turn(self) -> int:
        """Create a boundary so late embeddings cannot affect a newer turn."""
        self._active_turn_id += 1
        oldest_turn_to_keep = self._active_turn_id - 4
        self._turn_embeddings = {
            turn_id: embedding
            for turn_id, embedding in self._turn_embeddings.items()
            if turn_id >= oldest_turn_to_keep
        }
        self._turn_declarations = {
            turn_id: speaker_id
            for turn_id, speaker_id in self._turn_declarations.items()
            if turn_id >= oldest_turn_to_keep
        }
        self._trained_declaration_turns = {
            turn_id
            for turn_id in self._trained_declaration_turns
            if turn_id >= oldest_turn_to_keep
        }
        return self._active_turn_id

    def declare_speaker(self, speaker_id: str) -> bool:
        """Apply a speaker ID extracted from the assistant's hidden metadata."""
        normalized_id = " ".join(str(speaker_id or "").split()).strip()
        if not normalized_id or len(normalized_id) > 80 or self._active_turn_id <= 0:
            logger.warning("Ignored invalid or out-of-turn speaker declaration")
            return False

        turn_id = self._active_turn_id
        existing_id = self._turn_declarations.get(turn_id)
        if existing_id:
            if existing_id.casefold() == normalized_id.casefold():
                logger.debug(
                    "Ignored repeated LLM speaker declaration for turn %s: %s",
                    turn_id,
                    normalized_id,
                )
                return False
            logger.warning(
                "Ignored conflicting LLM speaker declaration for turn %s: %s then %s",
                turn_id,
                existing_id,
                normalized_id,
            )
            return False

        self._turn_declarations[turn_id] = normalized_id
        embedding = self._turn_embeddings.get(turn_id)
        if embedding is not None:
            profile_updated = self.profile_store.update(
                normalized_id,
                embedding,
                minimum_similarity=self.update_threshold,
            )
            logger.info(
                "Applied LLM speaker declaration to current-turn embedding: "
                "name=%s turn=%s profile_updated=%s",
                normalized_id,
                turn_id,
                profile_updated,
            )
            self._trained_declaration_turns.add(turn_id)
        else:
            logger.info(
                "Stored LLM speaker declaration pending current-turn embedding: name=%s turn=%s",
                normalized_id,
                turn_id,
            )

        self._remember_current_speaker(
            SpeakerResult(name=normalized_id, confidence=1.0, source="declared")
        )
        logger.info("LLM declared current speaker as %s", normalized_id)
        return True

    def _remember_current_speaker(self, result: SpeakerResult) -> None:
        self.current_speaker = result
        self.current_speaker_at = time.time()
        self.current_speaker_misses = 0

    def _context_result(self) -> Optional[SpeakerResult]:
        if self.current_speaker is None:
            return None
        if time.time() - self.current_speaker_at > self.context_max_age_sec:
            self.current_speaker = None
            self.current_speaker_at = 0.0
            self.current_speaker_misses = 0
            return None
        return SpeakerResult(
            name=self.current_speaker.name,
            confidence=self.current_speaker.confidence,
            source="context",
        )

    def resolve_transcript(
        self,
        text: str,
        embedding: Optional[SpeakerEmbedding],
        turn_id: Optional[int] = None,
    ) -> tuple[str, Optional[SpeakerResult]]:
        resolved_turn_id = turn_id if turn_id is not None else self._active_turn_id
        if embedding is not None:
            self._turn_embeddings[resolved_turn_id] = embedding

        declared_name = self._turn_declarations.get(resolved_turn_id)
        if (
            declared_name
            and embedding is not None
            and resolved_turn_id not in self._trained_declaration_turns
        ):
            profile_updated = self.profile_store.update(
                declared_name,
                embedding,
                minimum_similarity=self.update_threshold,
            )
            logger.info(
                "Applied pending LLM speaker declaration to late embedding: "
                "name=%s turn=%s profile_updated=%s",
                declared_name,
                resolved_turn_id,
                profile_updated,
            )
            self._trained_declaration_turns.add(resolved_turn_id)

        if declared_name and embedding is not None:
            result = SpeakerResult(
                name=declared_name,
                confidence=1.0,
                source="declared",
            )
            if resolved_turn_id == self._active_turn_id:
                self._remember_current_speaker(result)
            return text, result

        if embedding is None:
            return text, self._context_result()
        result = self.profile_store.match(
            embedding,
            minimum_similarity=self.match_threshold,
        )
        if result is not None:
            if resolved_turn_id == self._active_turn_id:
                self._remember_current_speaker(result)
            return text, result

        if resolved_turn_id == self._active_turn_id:
            self.current_speaker_misses += 1
            if self.current_speaker_misses >= self.context_misses_to_clear:
                self.current_speaker = None
                self.current_speaker_at = 0.0
                self.current_speaker_misses = 0
            else:
                context_result = self._context_result()
                if context_result is not None:
                    logger.info(
                        "Retaining contextual speaker %s after embedding mismatch (%s/%s)",
                        context_result.name,
                        self.current_speaker_misses,
                        self.context_misses_to_clear,
                    )
                return text, context_result
        return text, None


def format_speaker_annotation(result: Optional[SpeakerResult], score_decimals: int = 2) -> str:
    """Format a speaker result as compact context for the conversation model."""
    if result is None or not result.name:
        return ""
    if result.source == "context":
        return ""
    if result.source == "declared":
        return f"[User speaker: {result.name}]"
    return f"[User speaker: {result.name} ({result.confidence:.{score_decimals}f})]"


def normalize_speaker_provider(provider: Optional[str]) -> str:
    value = str(provider or "gemini").strip().lower()
    return {"off": "none", "noop": "none", "disabled": "none"}.get(value, value)


def create_speaker_embedding_provider(provider: str, config) -> SpeakerEmbeddingProvider:
    """Create the configured embedding provider."""
    normalized = normalize_speaker_provider(provider)
    if normalized == "gemini":
        return GeminiSpeakerEmbeddingProvider(config)
    raise ValueError(f"Unsupported speaker embedding provider: {provider}")


def create_speaker_engine(config) -> SpeakerEngine:
    """Create a disabled or provider-backed speaker engine."""
    if not config.get("speaker_recognition_enabled", False):
        return NoOpSpeakerEngine()
    provider_name = normalize_speaker_provider(
        config.get("speaker_embedding_provider", "gemini")
    )
    if provider_name == "none":
        return NoOpSpeakerEngine()
    provider = create_speaker_embedding_provider(provider_name, config)
    return ProfiledSpeakerEngine(provider, config)
