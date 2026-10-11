import json
import math
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from src.music_lights import (
    generate_light_track,
    load_or_generate_light_track,
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
