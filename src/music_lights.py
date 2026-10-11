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

TRACK_VERSION = 3
FRAME_SECONDS = 0.05


def detect_rhythm(levels: list[float], step: float) -> tuple[float, float]:
    """Estimate beat period/phase from repeated rises in the volume envelope."""
    onsets = [0.0] + [max(0.0, b - a) for a, b in zip(levels, levels[1:])]
    best_score, best_lag = 0.0, 0
    for lag in range(max(1, round(60 / 180 / step)), round(60 / 60 / step) + 1):
        if len(onsets) < lag * 4:
            continue
        left, right = onsets[:-lag], onsets[lag:]
        norm = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
        if norm <= 1e-12:
            continue
        score = sum(a * b for a, b in zip(left, right)) / norm
        # Break half/double-time ties near a moderate tempo.
        score -= 0.02 * abs(math.log((60 / (lag * step)) / 120))
        if score > best_score:
            best_score, best_lag = score, lag
    if best_score < 0.3:
        return 0.0, 0.0
    phase = max(range(best_lag), key=lambda offset: sum(onsets[offset::best_lag]))
    return best_lag * step, phase * step


def valid_light_plan(plan, duration: float) -> bool:
    """Validate the small artistic plan independently of fine-grained audio timing."""
    if not isinstance(plan, dict) or not isinstance(plan.get("segments"), list):
        return False
    segments = plan["segments"]
    if not 1 <= len(segments) <= 24:
        return False
    previous = -1.0
    for segment in segments:
        if not isinstance(segment, dict):
            return False
        at, color = segment.get("start"), segment.get("color")
        intensity, strength = segment.get("intensity", 1), segment.get("pulse_strength", 1)
        if (
            type(at) not in (int, float)
            or not math.isfinite(at)
            or not previous < at < duration
            or at < 0
            or not isinstance(color, list)
            or len(color) != 3
            or any(type(c) is not int or not 0 <= c <= 255 for c in color)
            or segment.get("pulse") not in ("beat", "breathe", "steady")
            or type(intensity) not in (int, float)
            or not 0 <= intensity <= 1
            or type(strength) not in (int, float)
            or not 0 <= strength <= 1
        ):
            return False
        previous = at
    return segments[0]["start"] == 0


def load_light_plan(source_path: Path):
    """Accept a cached artistic plan only for the exact original song."""
    source = source_path.stat()
    try:
        cached = json.loads(
            source_path.with_name(source_path.name + ".light-plan.json").read_text(encoding="utf-8")
        )
        if isinstance(cached, dict) and cached.get("source") == {
            "size": source.st_size,
            "mtime_ns": source.st_mtime_ns,
        }:
            return cached.get("plan")
    except (OSError, ValueError):
        pass
    return None


def _signature(source_path: Path) -> dict:
    source = source_path.stat()
    plan = load_light_plan(source_path)
    return {
        "version": TRACK_VERSION,
        "size": source.st_size,
        "mtime_ns": source.st_mtime_ns,
        "plan": plan,
    }


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


def generate_light_track(wav_path: Path, plan=None) -> list[dict]:
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
    if not valid_light_plan(plan, duration):
        plan = None
    beat_period, beat_phase = detect_rhythm(levels, block_frames / rate)
    last_beat = beat_phase
    ordered = sorted(levels)
    reference = max(ordered[int((len(ordered) - 1) * 0.95)], 0.01)
    floor = ordered[int((len(ordered) - 1) * 0.10)]
    # Expand even compressed music's small volume changes into a visible LED range.
    if reference - floor < max(0.000001, reference * 0.001):
        floor = 0.0
    smoothed = previous = slow_energy = 0.0
    frames = []
    segment_index = 0
    for index, level in enumerate(levels):
        at = index * block_frames / rate
        energy = min(1.0, max(0.0, (level - floor) / (reference - floor)))
        accent = max(0.0, energy - previous)
        previous = energy
        if beat_period and accent >= 0.15 and at - last_beat >= beat_period * 0.6:
            # Re-anchor on real accents instead of letting a rounded tempo drift.
            last_beat = at
        # Deep valleys and quick attacks make accents visible through the cube's diffuser.
        target = min(1.0, 0.04 + 0.96 * energy**1.8 + 0.4 * accent) if level > 0.0001 else 0.0
        alpha = 0.9 if target > smoothed else 0.5
        smoothed += alpha * (target - smoothed)
        slow_energy += 0.02 * (energy - slow_energy)
        hue = (0.58 + at / 24.0 + 0.12 * slow_energy) % 1.0
        color = [round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.8, 1.0)]
        mode, strength, intensity = "beat", 1.0, 1.0
        if plan:
            segments = plan["segments"]
            while segment_index + 1 < len(segments) and at >= segments[segment_index + 1]["start"]:
                segment_index += 1
            segment = segments[segment_index]
            color = segment["color"]
            if segment_index:
                previous_color = segments[segment_index - 1]["color"]
                blend = min(1.0, (at - segment["start"]) / 1.0)
                color = [round(a + (b - a) * blend) for a, b in zip(previous_color, color)]
            mode = segment["pulse"]
            strength = segment.get("pulse_strength", 1)
            intensity = segment.get("intensity", 1)
        pulse = 1.0
        if mode == "beat" and beat_period:
            pulse = max(0.0, math.cos(2 * math.pi * (at - last_beat) / beat_period)) ** 2
        elif mode == "breathe":
            period = beat_period * 4 if beat_period else 4.0
            pulse = (1 + math.cos(2 * math.pi * (at - beat_phase) / period)) / 2
        modulation = 1 - strength * 0.85 * (1 - pulse)
        brightness = smoothed * modulation
        if mode == "beat" and beat_period:
            # Give the beat a bright, visible crest even in a compressed passage.
            brightness = max(brightness, strength * pulse * min(1.0, level / reference) ** 0.5)
        # Fade the final half-second, without modifying the audio itself.
        fade = min(1.0, max(0.0, (duration - at) / 0.5))
        frames.append(
            {
                "time": round(at, 6),
                "color": color,
                "brightness": round(brightness * intensity * fade, 4),
            }
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


def load_cached_light_track(source_path: Path):
    """Read a current timeline without decoding the song again."""
    signature = _signature(source_path)
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
    return None


def load_or_generate_light_track(wav_path: Path, source_path: Path) -> list[dict]:
    """Cache beside the original song; regenerate when the song or algorithm changes."""
    cached = load_cached_light_track(source_path)
    if cached is not None:
        return cached
    signature = _signature(source_path)
    cache_path = source_path.with_name(source_path.name + ".light.json")

    frames = generate_light_track(wav_path, signature["plan"])
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
