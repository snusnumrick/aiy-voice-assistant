"""Build reusable LED timelines from the volume envelope of recorded music."""

import colorsys
import json
import logging
import math
import os
import tempfile
import wave
from array import array
from pathlib import Path
from sys import byteorder

logger = logging.getLogger(__name__)

TRACK_VERSION = 1
FRAME_SECONDS = 0.05


def _samples(data: bytes, width: int):
    if width == 1:
        return [sample - 128 for sample in data]
    if width in (2, 4):
        samples = array("h" if width == 2 else "i")
        samples.frombytes(data)
        if byteorder != "little":
            samples.byteswap()
        return samples
    if width == 3:
        return [
            int.from_bytes(data[i : i + 3], "little", signed=True) for i in range(0, len(data), 3)
        ]
    raise ValueError(f"Unsupported PCM sample width: {width}")


def generate_light_track(wav_path: Path) -> list[dict]:
    """Follow loudness and rising accents with smooth brightness and color."""
    levels = []
    with wave.open(str(wav_path), "rb") as audio:
        rate, width = audio.getframerate(), audio.getsampwidth()
        duration = audio.getnframes() / rate
        block_frames = max(1, round(rate * FRAME_SECONDS))
        full_scale = float(1 << (width * 8 - 1))
        while True:
            data = audio.readframes(block_frames)
            if not data:
                break
            samples = _samples(data, width)
            # Measure every channel separately so opposite-phase stereo stays audible.
            rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
            levels.append(rms / full_scale)

    if not levels:
        return []
    ordered = sorted(levels)
    reference = max(ordered[int((len(ordered) - 1) * 0.95)], 0.01)
    smoothed = previous = slow_energy = 0.0
    frames = []
    for index, level in enumerate(levels):
        at = index * block_frames / rate
        energy = min(1.0, level / reference)
        accent = max(0.0, energy - previous)
        previous = energy
        # Gentle attack/release, darkness in silence, and extra emphasis on accents.
        target = min(1.0, energy**0.7 + 0.2 * accent)
        alpha = 0.35 if target > smoothed else 0.15
        smoothed += alpha * (target - smoothed)
        slow_energy += 0.02 * (energy - slow_energy)
        hue = (0.58 + at / 24.0 + 0.12 * slow_energy) % 1.0
        color = [round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.8, 1.0)]
        # Fade the final half-second, without modifying the audio itself.
        fade = min(1.0, max(0.0, (duration - at) / 0.5))
        frames.append(
            {"time": round(at, 6), "color": color, "brightness": round(smoothed * fade, 4)}
        )
    frames.append({"time": round(duration, 6), "color": [0, 0, 0], "brightness": 0.0})
    return frames


def valid_light_track(frames) -> bool:
    """Reject malformed cached timelines before they reach the LED scheduler."""
    if not isinstance(frames, list) or not frames:
        return False
    previous = -1.0
    for frame in frames:
        if not isinstance(frame, dict):
            return False
        at, color, brightness = frame.get("time"), frame.get("color"), frame.get("brightness")
        if (
            not isinstance(at, (int, float))
            or not math.isfinite(at)
            or at < 0
            or at < previous
            or not isinstance(brightness, (int, float))
            or not math.isfinite(brightness)
            or not 0 <= brightness <= 1
            or not isinstance(color, list)
            or len(color) != 3
            or any(type(channel) is not int or not 0 <= channel <= 255 for channel in color)
        ):
            return False
        previous = at
    return frames[0]["time"] == 0


def load_or_generate_light_track(wav_path: Path, source_path: Path) -> list[dict]:
    """Cache beside the original song; regenerate when the song or algorithm changes."""
    source = source_path.stat()
    signature = {"version": TRACK_VERSION, "size": source.st_size, "mtime_ns": source.st_mtime_ns}
    cache_path = source_path.with_name(source_path.name + ".light.json")
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and cached.get("source") == signature
            and valid_light_track(cached.get("frames"))
        ):
            return cached["frames"]
    except (OSError, ValueError):
        pass

    frames = generate_light_track(wav_path)
    if not frames:
        return frames
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_path.parent, suffix=".tmp", delete=False
        ) as output:
            temp_path = Path(output.name)
            json.dump({"source": signature, "frames": frames}, output, separators=(",", ":"))
        os.replace(temp_path, cache_path)
        logger.info("Generated music light track: %s (%d frames)", cache_path, len(frames))
    except OSError as exc:
        logger.warning("Cannot cache music light track %s: %s", cache_path, exc)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return frames
