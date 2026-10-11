import asyncio
import json
import wave
from unittest.mock import patch

from src.config import Config
from src.music_light_design import prepare_light_design


def song_file(tmp_path):
    song = tmp_path / "moon.wav"
    with wave.open(str(song), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x10" * 16000)
    return song


def settings(**kwargs):
    return Config(
        config_file="__missing__.json",
        user_config_file="__missing_user__.json",
        lyrics_model="configured-model",
        **kwargs,
    )


def test_model_design_is_validated_saved_and_reused(tmp_path):
    song = song_file(tmp_path)
    plan = {
        "segments": [
            {
                "start": 0,
                "color": [20, 60, 255],
                "pulse": "breathe",
                "intensity": 0.5,
                "pulse_strength": 0.6,
            }
        ]
    }

    class Model:
        async def get_response_async(self, messages):
            context = json.loads(messages[1]["content"])
            assert context["duration"] == 2
            assert context["title"] == "moon"
            yield "```json\n" + json.dumps(plan) + "\n```"

    with patch("src.lyrics_tool.LyricsTool._get_model", return_value=Model()) as get_model:
        asyncio.run(prepare_light_design(song, settings()))
        asyncio.run(prepare_light_design(song, settings()))
    assert get_model.call_count == 1
    assert json.loads(song.with_name(song.name + ".light-plan.json").read_text())["plan"] == plan


def test_invalid_model_output_and_timeout_leave_local_analysis_available(tmp_path):
    song = song_file(tmp_path)

    class InvalidModel:
        async def get_response_async(self, messages):
            yield '{"segments":[{"start":0,"color":[999,0,0],"pulse":"beat"}]}'

    class SlowModel:
        async def get_response_async(self, messages):
            await asyncio.sleep(1)
            yield "{}"

    for model in (InvalidModel(), SlowModel()):
        with patch("src.lyrics_tool.LyricsTool._get_model", return_value=model):
            asyncio.run(prepare_light_design(song, settings(music_lights_llm_timeout=0.01)))
        assert not song.with_name(song.name + ".light-plan.json").exists()


def test_disabled_design_does_not_request_a_model(tmp_path):
    with patch("src.lyrics_tool.LyricsTool._get_model") as get_model:
        asyncio.run(
            prepare_light_design(song_file(tmp_path), settings(music_lights_llm_enabled=False))
        )
    get_model.assert_not_called()
