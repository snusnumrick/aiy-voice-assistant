import json
import math
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from src.music_lights import (
    detect_rhythm,
    generate_light_track,
    load_or_generate_light_track,
    valid_light_plan,
    valid_light_track,
)


def write_song(path, amplitudes, channels=1, width=2):
    rate = 8000
    pcm = bytearray()
    for amplitude in amplitudes:
        for index in range(rate):
            sample = round(amplitude * math.sin(2 * math.pi * 200 * index / rate))
            for channel in range(channels):
                value = -sample if channel % 2 else sample
                if width == 1:
                    pcm.extend(bytes([value + 128]))
                else:
                    pcm.extend(value.to_bytes(width, "little", signed=True))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(pcm)


def test_track_follows_volume_and_silence_without_changing_audio(tmp_path):
    song = tmp_path / "song.wav"
    write_song(song, [1000, 20000, 0, 20000])
    original = song.read_bytes()
    frames = generate_light_track(song)
    assert valid_light_track(frames)
    assert len(frames) == 81
    assert frames[0]["time"] == 0
    assert frames[-1]["time"] == 4
    assert frames[-1]["brightness"] == 0
    quiet = next(frame for frame in frames if frame["time"] == 0.75)
    loud = next(frame for frame in frames if frame["time"] == 1.75)
    silence = next(frame for frame in frames if frame["time"] == 2.95)
    assert loud["brightness"] > quiet["brightness"] * 3
    assert silence["brightness"] < 0.05
    assert len({tuple(frame["color"]) for frame in frames}) > 10
    assert song.read_bytes() == original


@pytest.mark.parametrize("width,amplitude", [(1, 100), (2, 25000), (3, 6000000), (4, 1500000000)])
def test_pcm_formats_and_opposite_phase_stereo(tmp_path, width, amplitude):
    song = tmp_path / "stereo.wav"
    write_song(song, [amplitude, amplitude], channels=2, width=width)
    frames = generate_light_track(song)
    assert valid_light_track(frames)
    assert frames[20]["brightness"] > 0.9


def test_silent_and_empty_audio(tmp_path):
    song = tmp_path / "silence.wav"
    write_song(song, [0])
    assert all(frame["brightness"] == 0 for frame in generate_light_track(song))
    write_song(song, [])
    assert generate_light_track(song) == []


def test_cache_reuse_invalidation_and_corruption(tmp_path):
    song = tmp_path / "song.wav"
    write_song(song, [5000])
    frames = load_or_generate_light_track(song, song)
    cache = Path(str(song) + ".light.json")
    assert json.loads(cache.read_text())["frames"] == frames
    with patch("src.music_lights.generate_light_track", side_effect=AssertionError("cache miss")):
        assert load_or_generate_light_track(song, song) == frames

    write_song(song, [5000, 0])
    changed = load_or_generate_light_track(song, song)
    assert changed[-1]["time"] == 2
    cache.write_text("broken json")
    assert load_or_generate_light_track(song, song) == changed
    cached = json.loads(cache.read_text())
    cached["frames"][0]["brightness"] = float("nan")
    cache.write_text(json.dumps(cached))
    assert load_or_generate_light_track(song, song) == changed


def test_read_only_cache_does_not_prevent_track_generation(tmp_path):
    song = tmp_path / "song.wav"
    write_song(song, [5000])
    with patch("src.music_lights.tempfile.NamedTemporaryFile", side_effect=PermissionError):
        assert valid_light_track(load_or_generate_light_track(song, song))


@pytest.mark.parametrize(
    "frames",
    [
        None,
        [],
        [None],
        [{"time": 0, "color": [0, 0, 256], "brightness": 0.5}],
        [{"time": 0, "color": [0, 0, 0], "brightness": -1}],
        [{"time": float("inf"), "color": [0, 0, 0], "brightness": 0.5}],
        [{"time": 1, "color": [0, 0, 0], "brightness": 0.5}],
    ],
)
def test_invalid_tracks(frames):
    assert not valid_light_track(frames)


def test_detects_repeated_beats_and_ignores_constant_audio():
    levels = [1.0 if i % 10 < 2 else 0.9 for i in range(160)]
    period, phase = detect_rhythm(levels, 0.05)
    assert period == 0.5
    assert phase == 0
    assert detect_rhythm([1.0] * 160, 0.05) == (0, 0)


def test_compressed_music_gets_deep_beat_pulses(tmp_path):
    song = tmp_path / "beats.wav"
    with wave.open(str(song), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        for frame in range(160):
            amplitude = 10000 if frame % 10 < 2 else 9000
            audio.writeframes(
                b"".join(
                    round(amplitude * math.sin(2 * math.pi * 200 * i / 8000)).to_bytes(
                        2, "little", signed=True
                    )
                    for i in range(400)
                )
            )
    frames = generate_light_track(song)
    peak = next(frame for frame in frames if frame["time"] == 2)
    valley = next(frame for frame in frames if frame["time"] == 2.25)
    assert peak["brightness"] > 0.85
    assert valley["brightness"] < 0.08


def test_artistic_plan_colors_modes_and_cache_invalidation(tmp_path):
    song = tmp_path / "song.wav"
    write_song(song, [20000] * 4)
    automatic = load_or_generate_light_track(song, song)
    plan = {
        "segments": [
            {"start": 0, "color": [255, 0, 0], "pulse": "steady", "intensity": 0.4},
            {"start": 2, "color": [0, 0, 255], "pulse": "breathe", "pulse_strength": 0.7},
        ]
    }
    assert valid_light_plan(plan, 4)
    assert not valid_light_plan(plan, 2)
    info = song.stat()
    Path(str(song) + ".light-plan.json").write_text(
        json.dumps(
            {
                "source": {"size": info.st_size, "mtime_ns": info.st_mtime_ns},
                "plan": plan,
            }
        )
    )
    designed = load_or_generate_light_track(song, song)
    assert designed != automatic
    assert designed[0]["color"] == [255, 0, 0]
    assert max(frame["brightness"] for frame in designed if frame["time"] < 2) <= 0.4
    assert next(frame for frame in designed if frame["time"] == 3)["color"] == [0, 0, 255]


def test_beat_pulses_reanchor_when_tempo_does_not_fit_the_frame_grid(tmp_path):
    song = tmp_path / "123-bpm.wav"
    peaks = []
    previous = False
    with wave.open(str(song), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        for index in range(800):
            is_peak = (index * 0.05) % (60 / 123) < 0.1
            if is_peak and not previous:
                peaks.append(index)
            previous = is_peak
            amplitude = 10000 if is_peak else 9000
            audio.writeframes(
                (
                    amplitude.to_bytes(2, "little", signed=True)
                    + (-amplitude).to_bytes(2, "little", signed=True)
                )
                * 200
            )
    frames = generate_light_track(song)
    assert all(
        frames[index]["brightness"] > 0.85 for index in peaks if 1 < frames[index]["time"] < 39
    )
