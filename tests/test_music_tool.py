import asyncio
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Config
from src.music_tool import (
    GeminiMusicTool,
    MiniMaxMusicTool,
    MusicTool,
    create_music_tool,
    normalize_music_provider,
)


class TestMusicToolFactory(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    @patch("src.music_tool.MusicTool._discover_music_folder", return_value=Path("/tmp"))
    def test_factory_defaults_to_minimax(self, _mock_discover):
        tool = create_music_tool(self._config(), response_player=MagicMock())
        self.assertIsInstance(tool, MiniMaxMusicTool)
        self.assertIsInstance(tool, MusicTool)

    @patch("src.music_tool.MusicTool._discover_music_folder", return_value=Path("/tmp"))
    def test_factory_uses_gemini_provider(self, _mock_discover):
        tool = create_music_tool(
            self._config(music_tool_provider="gemini"),
            response_player=MagicMock(),
        )
        self.assertIsInstance(tool, GeminiMusicTool)
        self.assertIsInstance(tool, MusicTool)

    def test_normalize_music_provider_accepts_lyria_alias(self):
        self.assertEqual(normalize_music_provider("lyria"), "gemini")

    @patch("src.music_tool.MusicTool._discover_music_folder", return_value=Path("/tmp"))
    def test_exposes_generation_and_saved_music_tools(self, _mock_discover):
        tool = create_music_tool(self._config(), response_player=MagicMock())
        self.assertEqual(
            [definition.name for definition in tool.tool_definitions()],
            ["generate_music", "play_music"],
        )


class TestSavedMusicLibrary(unittest.TestCase):
    def _tool(self, music_dir, response_player=None):
        config = Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
        )
        with patch("src.music_tool.MusicTool._discover_music_folder", return_value=music_dir):
            return MiniMaxMusicTool(config, response_player or MagicMock())

    def test_empty_query_plays_newest_saved_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            music_dir = Path(temp_dir)
            old_song = music_dir / "old_song.wav"
            new_song = music_dir / "homework_pop.wav"
            old_song.write_bytes(b"old")
            new_song.write_bytes(b"new")
            os.utime(old_song, (1, 1))
            os.utime(new_song, (2, 2))
            player = MagicMock()
            tool = self._tool(music_dir, player)

            result = asyncio.run(tool.play_music_async({}))

            self.assertEqual(result, "Playing saved music: homework_pop")
            player.add.assert_called_once_with(
                (None, str(new_song), "saved music: homework_pop")
            )

    def test_query_filters_saved_music_by_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            music_dir = Path(temp_dir)
            (music_dir / "bedtime_lullaby.wav").write_bytes(b"lullaby")
            expected = music_dir / "homework_pop.wav"
            expected.write_bytes(b"pop")
            player = MagicMock()
            tool = self._tool(music_dir, player)

            result = asyncio.run(tool.play_music_async({"query": "homework"}))

            self.assertEqual(result, "Playing saved music: homework_pop")
            self.assertEqual(player.add.call_args.args[0][1], str(expected))

    def test_list_does_not_queue_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            music_dir = Path(temp_dir)
            (music_dir / "song.wav").write_bytes(b"song")
            player = MagicMock()
            tool = self._tool(music_dir, player)

            result = asyncio.run(tool.play_music_async({"action": "list"}))

            self.assertEqual(result, "Saved music (1 shown): song")
            player.add.assert_not_called()


class TestMiniMaxMusicTool(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    @patch("src.music_tool.MusicTool._discover_music_folder", return_value=Path("/tmp"))
    @patch("src.music_tool.MiniMaxMusicTool._process_generated_audio", return_value=Path("/tmp/out.mp3"))
    @patch("src.music_tool.aiohttp.ClientSession.get")
    @patch("src.music_tool.aiohttp.ClientSession.post")
    def test_generate_music_uses_minimax_endpoint(
        self,
        mock_post,
        mock_get,
        _mock_process_audio,
        _mock_discover,
    ):
        mock_post_response = AsyncMock()
        mock_post_response.raise_for_status = MagicMock()
        mock_post_response.text.return_value = (
            '{"data": {"audio": "https://example.com/generated.mp3"}, "base_resp": {"status_code": 0}}'
        )
        mock_post.return_value.__aenter__.return_value = mock_post_response

        mock_get_response = AsyncMock()
        mock_get_response.raise_for_status = MagicMock()
        mock_get_response.read.return_value = b"fake-mp3"
        mock_get.return_value.__aenter__.return_value = mock_get_response

        with patch.dict("os.environ", {"MINIMAX_API_KEY": "test-minimax-key"}, clear=False):
            tool = MiniMaxMusicTool(self._config(minimax_music_model="music-2.6-free"), MagicMock())
            result = asyncio.run(
                tool.generate_music_async(
                    {
                        "prompt": "Upbeat electronic music with warm synths",
                        "model": "music-2.6-pro",
                    }
                )
            )

        self.assertEqual(
            result,
            "Music generated and playing. Download URL (valid 24h): "
            "https://example.com/generated.mp3\nLocal MP3: /tmp/out.mp3",
        )
        called_url = mock_post.call_args.args[0]
        self.assertEqual(called_url, "https://api.minimax.io/v1/music_generation")
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-minimax-key",
        )
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "music-2.6-pro")


class TestGeminiMusicTool(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    @patch("src.music_tool.MusicTool._discover_music_folder", return_value=Path("/tmp"))
    @patch("src.music_tool.GeminiMusicTool._process_generated_audio", return_value=Path("/tmp/gemini.mp3"))
    @patch("src.music_tool.aiohttp.ClientSession.post")
    def test_generate_music_uses_gemini_model_override(
        self,
        mock_post,
        _mock_process_audio,
        _mock_discover,
    ):
        audio_b64 = base64.b64encode(b"fake-audio").decode("ascii")
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text.return_value = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "data": audio_b64,
                                        "mimeType": "audio/mpeg",
                                    }
                                },
                                {"text": "Generated chorus"},
                            ]
                        }
                    }
                ]
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}, clear=False):
            tool = GeminiMusicTool(
                self._config(gemini_music_model="lyria-3-pro-preview"),
                MagicMock(),
            )
            result = asyncio.run(
                tool.generate_music_async(
                    {
                        "prompt": "Dreamy ambient music with airy vocals",
                        "model": "lyria-3-clip-preview",
                    }
                )
            )

        self.assertEqual(
            result,
            "Music generated and playing. Local MP3: /tmp/gemini.mp3\n"
            "Generated lyrics/description:\nGenerated chorus",
        )
        called_url = mock_post.call_args.args[0]
        self.assertIn("/v1beta/models/lyria-3-clip-preview:generateContent", called_url)
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["x-goog-api-key"],
            "test-gemini-key",
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"],
            "Dreamy ambient music with airy vocals",
        )


if __name__ == "__main__":
    unittest.main()
