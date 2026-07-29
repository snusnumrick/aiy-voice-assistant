#!/usr/bin/env python3
"""Measure whether configured audio embeddings separate real speakers."""

# ruff: noqa: E402

import argparse
import asyncio
import mimetypes
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.speaker_engine import (
    SpeakerEmbedding,
    cosine_similarity,
    create_speaker_embedding_provider,
)


@dataclass
class Sample:
    speaker: str
    path: Path
    embedding: SpeakerEmbedding


def parse_sample(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/audio.wav")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser()
    if not name.strip() or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid speaker name or audio file: {value}")
    return name.strip(), path


async def embed_samples(raw_samples: list[tuple[str, Path]], config) -> list[Sample]:
    provider_name = config.get("speaker_embedding_provider", "gemini")
    provider = create_speaker_embedding_provider(provider_name, config)
    samples = []
    for speaker, path in raw_samples:
        mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
        embedding = await provider.embed(path.read_bytes(), mime_type)
        if embedding is None:
            raise RuntimeError(f"Embedding failed for {path}")
        samples.append(Sample(speaker=speaker, path=path, embedding=embedding))
    return samples


def print_comparison(samples: list[Sample]) -> None:
    same_scores = []
    different_scores = []
    print("speaker_a\tspeaker_b\tsimilarity\tfiles")
    for left, right in combinations(samples, 2):
        if left.embedding.space_id != right.embedding.space_id:
            raise RuntimeError("Embedding provider returned incompatible vector spaces")
        score = cosine_similarity(left.embedding.values, right.embedding.values)
        if left.speaker.casefold() == right.speaker.casefold():
            same_scores.append(score)
        else:
            different_scores.append(score)
        print(
            f"{left.speaker}\t{right.speaker}\t{score:.4f}\t"
            f"{left.path.name} <> {right.path.name}"
        )

    print()
    if same_scores:
        print(f"same-speaker range:      {min(same_scores):.4f} .. {max(same_scores):.4f}")
    if different_scores:
        print(
            f"different-speaker range: {min(different_scores):.4f} .. "
            f"{max(different_scores):.4f}"
        )
    if same_scores and different_scores:
        gap = min(same_scores) - max(different_scores)
        print(f"separation gap:          {gap:.4f}")
        if gap > 0:
            threshold = (min(same_scores) + max(different_scores)) / 2
            print(f"candidate match threshold: {threshold:.4f}")
        else:
            print("No clean separation; this embedding model is not reliable on these samples.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Gemini audio embeddings across labeled speaker recordings."
    )
    parser.add_argument(
        "samples",
        nargs="+",
        type=parse_sample,
        metavar="NAME=AUDIO",
        help="Labeled WAV or MP3 file; provide several files per speaker.",
    )
    args = parser.parse_args()
    if len(args.samples) < 2:
        parser.error("provide at least two labeled recordings")

    load_dotenv()
    config = Config()
    samples = asyncio.run(embed_samples(args.samples, config))
    print_comparison(samples)


if __name__ == "__main__":
    main()
