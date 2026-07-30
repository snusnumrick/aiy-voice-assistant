"""HTTP service for multilingual WeSpeaker ONNX embeddings."""

import asyncio
import hashlib
import hmac
import io
import logging
import math
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi
from fastapi import FastAPI, Header, HTTPException, Request

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get("WESPEAKER_MODEL_PATH", "/models/wespeaker.onnx"))
MODEL_ID = os.environ.get(
    "WESPEAKER_MODEL_ID",
    "resnet34-lm-voxceleb",
).strip()
API_KEY = os.environ.get("WESPEAKER_API_KEY", "")
MAX_AUDIO_BYTES = int(os.environ.get("WESPEAKER_MAX_AUDIO_BYTES", 2_000_000))
EXPECTED_SAMPLE_RATE = int(os.environ.get("WESPEAKER_SAMPLE_RATE", 16000))
THREADS = max(1, int(os.environ.get("WESPEAKER_THREADS", "1")))
EMBEDDING_SPACE_VERSION = "fbank80-cmn-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WeSpeakerOnnxModel:
    """Official WeSpeaker fbank preprocessing plus ONNX inference."""

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"WeSpeaker model not found: {model_path}")

        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = THREADS
        session_options.intra_op_num_threads = THREADS
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model_sha256 = _sha256(model_path)

    @staticmethod
    def _read_wav(audio_bytes: bytes) -> torch.Tensor:
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                frames = wav_file.readframes(frame_count)
        except (EOFError, wave.Error) as e:
            raise ValueError(f"Invalid WAV audio: {e}") from e

        if sample_width != 2:
            raise ValueError("WAV must contain 16-bit PCM samples")
        if channels < 1:
            raise ValueError("WAV must contain at least one channel")
        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"WAV sample rate must be {EXPECTED_SAMPLE_RATE} Hz; received {sample_rate}"
            )

        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if samples.size < sample_rate // 4:
            raise ValueError("WAV must contain at least 250 ms of audio")
        return torch.from_numpy(samples).unsqueeze(0)

    def embed_wav(self, audio_bytes: bytes) -> list[float]:
        waveform = self._read_wav(audio_bytes)
        features = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=EXPECTED_SAMPLE_RATE,
            window_type="hamming",
            use_energy=False,
        )
        features = features - torch.mean(features, dim=0)
        batch = features.unsqueeze(0).numpy()
        output = self.session.run(
            output_names=[self.output_name],
            input_feed={self.input_name: batch},
        )[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        magnitude = math.sqrt(float(np.dot(vector, vector)))
        if not np.isfinite(vector).all() or magnitude <= 0:
            raise ValueError("Model produced an invalid embedding")
        return (vector / magnitude).tolist()


model = WeSpeakerOnnxModel(MODEL_PATH)
app = FastAPI(title="Cubie WeSpeaker Embeddings", version="1.0")


def _check_authorization(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    supplied = authorization or ""
    expected = f"Bearer {API_KEY}"
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


async def _read_limited_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio body is too large")
    if not body:
        raise HTTPException(status_code=400, detail="Audio body is empty")
    return bytes(body)


@app.get("/healthz")
async def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_sha256": model.model_sha256,
    }


@app.post("/v1/embeddings")
async def create_embedding(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    _check_authorization(authorization)
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in {"audio/wav", "audio/x-wav"}:
        raise HTTPException(status_code=415, detail="Content-Type must be audio/wav")
    audio_bytes = await _read_limited_body(request)
    try:
        embedding = await asyncio.to_thread(model.embed_wav, audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("WeSpeaker inference failed")
        raise HTTPException(status_code=500, detail="Speaker embedding failed") from e

    return {
        "embedding": embedding,
        "dimension": len(embedding),
        "model": MODEL_ID,
        "space_id": (
            f"wespeaker:{MODEL_ID}:{EMBEDDING_SPACE_VERSION}:"
            f"{model.model_sha256[:12]}:{len(embedding)}"
        ),
    }
