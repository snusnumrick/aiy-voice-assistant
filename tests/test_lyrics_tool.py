import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Config
from src.lyrics_tool import LyricsTool, main, normalize_lyrics_provider


class TestLyricsTool(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    def test_tool_definition_exposes_compact_flexible_inputs(self):
        definition = LyricsTool(self._config()).tool_definition()

        self.assertEqual(definition.name, "generate_lyrics")
        self.assertEqual(definition.required, ["intent"])
        self.assertEqual(
            [parameter.name for parameter in definition.parameters],
            ["intent", "existing_lyrics"],
        )
        intent_description = definition.parameters[0].description
        self.assertIn("minimal normalization", intent_description)
        self.assertIn("genuine semantic ambiguity", intent_description)
        self.assertIn("obvious repetitions", intent_description)
        self.assertIn("speech-recognition artifacts", intent_description)
        self.assertIn("Do not mention such cleanup", intent_description)
        self.assertIn("Do not invent", intent_description)

    def test_provider_aliases_are_normalized(self):
        self.assertEqual(normalize_lyrics_provider("google"), "gemini")
        self.assertEqual(normalize_lyrics_provider("open-router"), "openrouter")

    @patch("src.lyrics_tool.GeminiAIModel")
    def test_generates_lyrics_with_gemini_model_from_config(self, model_class):
        model = MagicMock()
        model.get_response.return_value = (
            "[Verse]\nMorning finds the window\n[Chorus]\nCarry the light home"
        )
        model_class.return_value = model
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}, clear=False):
            tool = LyricsTool(
                self._config(
                    lyrics_model="gemini-3.1-pro-preview",
                )
            )
            result = asyncio.run(
                tool.generate_lyrics_async(
                    {
                        "intent": "Write a hopeful folk song in English",
                        "existing_lyrics": "Old opening line",
                    }
                )
            )

        self.assertEqual(
            result,
            "[Verse]\nMorning finds the window\n[Chorus]\nCarry the light home",
        )
        model_class.assert_called_once_with(
            tool.config,
            model_id="gemini-3.1-pro-preview",
            max_tokens=8192,
            thinking_level="medium",
            request_timeout_sec=120,
        )
        messages = model.get_response.call_args.args[0]
        self.assertIn("Existing lyrics:\nOld opening line", messages[1]["content"])

    @patch("src.lyrics_tool.OpenRouterModel")
    def test_generates_lyrics_with_openrouter_model_from_config(self, model_class):
        model = MagicMock()
        model.get_response.return_value = "[Verse]\nA red cup by the sink"
        model_class.return_value = model
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-router-key"}, clear=False):
            tool = LyricsTool(
                self._config(
                    lyrics_provider="openrouter",
                    lyrics_model="google/gemini-3.1-pro-preview",
                )
            )
            result = asyncio.run(
                tool.generate_lyrics_async({"intent": "Write a song about a forgotten red cup"})
            )

        self.assertEqual(result, "[Verse]\nA red cup by the sink")
        model_class.assert_called_once_with(
            tool.config,
            model_id="google/gemini-3.1-pro-preview",
            max_tokens=8192,
            reasoning_effort=None,
        )

    def test_requires_configured_model(self):
        tool = LyricsTool(self._config())
        result = asyncio.run(
            tool.generate_lyrics_async({"intent": "Write a gentle bedtime lullaby"})
        )

        self.assertEqual(result, "Error: lyrics_model is not configured")

    @patch("src.lyrics_tool.load_dotenv")
    @patch("src.lyrics_tool.LyricsTool.generate_lyrics_async", new_callable=AsyncMock)
    @patch("builtins.print")
    def test_script_prints_generated_lyrics(self, mock_print, generate, _load_dotenv):
        generate.return_value = "[Verse]\nA small remembered world"

        exit_code = main(["Write a song about a girl and her poodle"])

        self.assertEqual(exit_code, 0)
        mock_print.assert_called_once_with("[Verse]\nA small remembered world")


if __name__ == "__main__":
    unittest.main()
