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
    ElevenLabsMusicTool,
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

    @patch("src.music_tool.MusicTool._discover_music_folder", return_value=Path("/tmp"))
    def test_claude_schema_includes_style_constraints_for_both_providers(self, _mock_discover):
        from src.ai_models_with_tools import ClaudeAIModelWithTools

        for provider in ("minimax", "gemini", "elevenlabs"):
            with self.subTest(provider=provider):
                tool = create_music_tool(
                    self._config(music_tool_provider=provider), response_player=MagicMock()
                )
                definition = tool.tool_definition()
                schema = ClaudeAIModelWithTools._create_tools_description([definition])[0]
                self.assertIn("no artist or band names", schema["description"])
                prompt = schema["input_schema"]["properties"]["prompt"]["description"]
                self.assertIn("generate_lyrics intent", prompt)
                self.assertIn("before the first call", prompt)
                self.assertIn("separate lyrics parameter", prompt)
                self.assertIn("Перед самым первым вызовом", definition.rule_instructions["russian"])
                self.assertIn("Before the very first call", definition.rule_instructions["english"])


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

            self.assertEqual(result, "Saved music queued after your spoken response: homework_pop")
            player.add_music.assert_called_once_with(
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

            self.assertEqual(result, "Saved music queued after your spoken response: homework_pop")
            self.assertEqual(player.add_music.call_args.args[0][1], str(expected))

    def test_list_does_not_queue_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            music_dir = Path(temp_dir)
            (music_dir / "song.wav").write_bytes(b"song")
            player = MagicMock()
            tool = self._tool(music_dir, player)

            result = asyncio.run(tool.play_music_async({"action": "list"}))

            self.assertEqual(result, "Saved music (1 shown): song")
            player.add_music.assert_not_called()


class TestMusicMetadata(unittest.TestCase):
    def test_readable_lyrics_removes_tags_and_preserves_stanzas(self):
        source = '[Verse]\nСЕрый подъЕЗД.\n\nДЕвочка: "СТОЙ!"\nпоДАрок'
        self.assertEqual(
            MusicTool._readable_lyrics(source),
            'Серый подъезд.\n\nДевочка: "Стой!"\nПодарок',
        )

    def test_stress_capitals_inside_sentences_are_removed(self):
        self.assertEqual(
            MusicTool._readable_lyrics('МАма ПРЯчет РУки. ПУдель беЖИТ.\nМАма: «Я не моГУ…»\nВ Этом ДОМе Окурок.'),
            'Мама прячет руки. Пудель бежит.\nМама: «Я не могу…»\nВ этом доме окурок.',
        )

    def test_section_tags_become_blank_lines(self):
        self.assertEqual(
            MusicTool._readable_lyrics("[Verse]\nFirst line\n[Chorus]\nSecond line\n\n[Outro]\nLast line"),
            "First line\n\nSecond line\n\nLast line",
        )

    @patch("src.music_tool.HAS_PYDUB", False)
    @patch("src.music_tool.MusicTool._discover_music_folder", return_value=Path("/tmp"))
    def test_title_language_and_lyrics(self, _discover):
        from mutagen.id3 import ID3

        config = Config(config_file="__missing_config__.json", user_config_file="__missing_user__.json")
        tool = MiniMaxMusicTool(config, MagicMock())
        for lyrics, title, expected_title, language in (
            ("[Verse]\nТвой поДАрок\nВторая строка", "Подарок", "Подарок", "rus"),
            ("[Verse]\nТвой поДАрок", "", "Твой подарок", "rus"),
            ("[Verse]\nA quiet morning", "Morning", "Morning", "eng"),
            ("", "", "Generated instrumental", None),
        ):
            with self.subTest(title=title, lyrics=lyrics), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "test.mp3"
                path.write_bytes(b"")
                tool._add_mp3_metadata(path, "Long style description", lyrics, title=title)
                tags = ID3(path)
                self.assertEqual(str(tags["TIT2"]), expected_title)
                self.assertIn("Long style description", str(tags.getall("COMM")[0]))
                frames = tags.getall("USLT")
                if language:
                    self.assertEqual(frames[0].lang, language)
                    self.assertEqual(frames[0].text, tool._readable_lyrics(lyrics))
                else:
                    self.assertEqual(frames, [])


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


class TestElevenLabsMusicTool(unittest.TestCase):
    def setUp(self):
        sleep = patch("src.tools.asyncio.sleep", new_callable=AsyncMock)
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        self.discover = patch("src.music_tool.MusicTool._discover_music_folder", return_value=Path("/tmp"))
        self.discover.start()
        self.addCleanup(self.discover.stop)
        self.env = patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _tool(self, **kwargs):
        return create_music_tool(
            Config(config_file="__missing_config__.json", user_config_file="__missing_user__.json",
                   music_tool_provider="elevenlabs", **kwargs),
            MagicMock(),
        )

    def test_factory_and_aliases(self):
        self.assertIsInstance(self._tool(), ElevenLabsMusicTool)
        for alias in ("ElevenLabs", "eleven labs", "eleven_labs"):
            self.assertEqual(normalize_music_provider(alias), "elevenlabs")

    @patch("src.music_tool.ElevenLabsMusicTool._process_generated_audio", return_value=Path("/tmp/song.mp3"))
    @patch("src.music_tool.aiohttp.ClientSession.post")
    def test_generation_preserves_lyrics_and_processes_binary_audio(self, post, process):
        response = AsyncMock()
        response.raise_for_status = MagicMock()
        response.status = 200
        response.read.return_value = b"mp3-audio"
        post.return_value.__aenter__.return_value = response
        lyrics = "[Verse]\nВ бумаге лежал твой поДАрок"
        emotion = {"voice": {"tone": "happy"}}
        tool = self._tool(elevenlabs_music_model="music_v1", elevenlabs_music_length_ms=60000,
                          elevenlabs_music_base_url="https://example.com/", elevenlabs_music_timeout=123)
        result = asyncio.run(tool.generate_music_async(
            {"prompt": "Warm acoustic folk", "lyrics": lyrics, "model": "music_v2", "emotion": emotion, "title": "Подарок"}
        ))
        self.assertEqual(result, "Music generated and playing. Local MP3: /tmp/song.mp3")
        self.assertEqual(post.call_args.args[0], "https://example.com/v1/music")
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["xi-api-key"], "test-key")
        self.assertEqual(kwargs["params"], {"output_format": "mp3_44100_128"})
        self.assertEqual(kwargs["timeout"].total, 123)
        self.assertEqual(kwargs["json"], {
            "prompt": f"Warm acoustic folk\n\nSing these lyrics:\n{lyrics}",
            "model_id": "music_v2", "force_instrumental": False, "music_length_ms": 60000,
        })
        process.assert_called_once_with(
            title="Подарок", audio_data=b"mp3-audio", prompt="Warm acoustic folk", lyrics=lyrics, emotion=emotion,
            comment_lines=["Generated by ElevenLabs Music", "Model: music_v2"],
        )

    @patch("src.music_tool.ElevenLabsMusicTool._process_generated_audio")
    @patch("src.music_tool.aiohttp.ClientSession.post")
    def test_instrumental_and_empty_audio(self, post, process):
        response = AsyncMock()
        response.raise_for_status = MagicMock()
        response.status = 200
        response.read.return_value = b""
        post.return_value.__aenter__.return_value = response
        result = asyncio.run(self._tool().generate_music_async({"prompt": "Quiet ambient piano"}))
        self.assertEqual(result, "Error: No audio data received from API")
        self.assertEqual(post.call_args.kwargs["json"], {
            "prompt": "Quiet ambient piano", "model_id": "music_v1", "force_instrumental": True,
        })
        process.assert_not_called()

    @patch("src.music_tool.aiohttp.ClientSession.post")
    def test_validation_prevents_requests(self, post):
        for parameters in ({}, {"prompt": "short"}, {"prompt": "x" * 4101},
                           {"prompt": "x" * 4090, "lyrics": "поДАрок"}):
            with self.subTest(parameters=parameters):
                self.assertTrue(asyncio.run(self._tool().generate_music_async(parameters)).startswith("Error:"))
        for duration in (True, "60000", 2999, 600001):
            with self.subTest(duration=duration):
                result = asyncio.run(self._tool(elevenlabs_music_length_ms=duration).generate_music_async(
                    {"prompt": "Quiet ambient piano"}
                ))
                self.assertIn("elevenlabs_music_length_ms", result)
        tool = self._tool()
        tool.api_key = None
        self.assertIn("ELEVENLABS_API_KEY", asyncio.run(tool.generate_music_async({"prompt": "Quiet ambient piano"})))
        post.assert_not_called()

    @patch("src.music_tool.ElevenLabsMusicTool._process_generated_audio")
    @patch("src.music_tool.aiohttp.ClientSession.post")
    def test_error_body_and_retry_after_are_reported(self, post, process):
        response = AsyncMock()
        response.status = 429
        response.headers = {"Retry-After": "30"}
        post.return_value.__aenter__.return_value = response
        for body, expected in (
            ('{"detail":{"status":"system_busy","message":"Try later"}}', "system_busy: Try later"),
            ('{"error":{"code":"rate_limit_exceeded"}}', "rate_limit_exceeded"),
            ("Service busy", "Service busy"),
        ):
            with self.subTest(body=body):
                response.text.return_value = body
                result = asyncio.run(self._tool().generate_music_async({"prompt": "Quiet ambient piano"}))
                self.assertIn("HTTP 429", result)
                self.assertIn(expected, result)
                self.assertIn("Retry-After: 30", result)
                self.assertEqual(self.sleep.await_count, 2)
                self.sleep.reset_mock()
        process.assert_not_called()

    @patch("src.music_tool.ElevenLabsMusicTool._process_generated_audio", return_value=Path("/tmp/song.mp3"))
    @patch("src.music_tool.aiohttp.ClientSession.post")
    def test_transient_error_retries_then_processes_once(self, post, process):
        for status in (429, 503):
            with self.subTest(status=status):
                post.reset_mock()
                process.reset_mock()
                self.sleep.reset_mock()
                failure = AsyncMock()
                failure.status = status
                failure.headers = {}
                failure.text.return_value = '{"detail":{"status":"system_busy","message":"Busy"}}'
                success = AsyncMock()
                success.status = 200
                success.raise_for_status = MagicMock()
                success.read.return_value = b"audio"
                contexts = []
                for response in (failure, success):
                    context = MagicMock()
                    context.__aenter__.return_value = response
                    contexts.append(context)
                post.side_effect = contexts
                result = asyncio.run(self._tool().generate_music_async({"prompt": "Quiet ambient piano"}))
                self.assertIn("Music generated and playing", result)
                self.assertEqual(post.call_count, 2)
                self.sleep.assert_awaited_once()
                self.assertGreaterEqual(self.sleep.call_args.args[0], 2)
                process.assert_called_once()

    @patch("src.music_tool.ElevenLabsMusicTool._process_generated_audio")
    @patch("src.music_tool.aiohttp.ClientSession.post")
    def test_permanent_errors_do_not_retry(self, post, process):
        for status in (400, 401, 403, 422):
            with self.subTest(status=status):
                post.reset_mock()
                response = AsyncMock()
                response.status = status
                response.headers = {}
                response.text.return_value = '{"detail":{"status":"invalid_request","message":"Rejected"}}'
                post.return_value.__aenter__.return_value = response
                result = asyncio.run(self._tool().generate_music_async({"prompt": "Quiet ambient piano"}))
                self.assertIn(f"HTTP {status}", result)
                post.assert_called_once()
        self.sleep.assert_not_awaited()
        process.assert_not_called()

    @patch("src.music_tool.ElevenLabsMusicTool._process_generated_audio")
    @patch("src.music_tool.aiohttp.ClientSession.post")
    def test_http_and_connection_errors(self, post, process):
        import aiohttp

        response = AsyncMock()
        response.status = 200
        response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            MagicMock(), (), status=429, message="Too many requests"
        ))
        post.return_value.__aenter__.return_value = response
        result = asyncio.run(self._tool().generate_music_async({"prompt": "Quiet ambient piano"}))
        self.assertIn("HTTP 429", result)
        post.side_effect = aiohttp.ClientConnectionError("Connection failed")
        result = asyncio.run(self._tool().generate_music_async({"prompt": "Quiet ambient piano"}))
        self.assertIn("Failed to connect to ElevenLabs", result)
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
