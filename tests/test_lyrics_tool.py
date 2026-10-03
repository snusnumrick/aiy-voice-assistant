import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_models import TruncatedResponseError
from src.config import Config
from src.lyrics_tool import LyricsTool, main, normalize_lyrics_provider


class TestLyricsTool(unittest.TestCase):
    def _config(self, **kwargs):
        kwargs.setdefault("lyrics_review_max_passes", 0)
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
            ["intent", "audience", "existing_lyrics"],
        )
        intent_description = definition.parameters[0].description
        self.assertIn("minimal normalization", intent_description)
        self.assertIn("genuine semantic ambiguity", intent_description)
        self.assertIn("obvious repetitions", intent_description)
        self.assertIn("speech-recognition artifacts", intent_description)
        self.assertIn("Do not mention such cleanup", intent_description)
        self.assertIn("Do not invent", intent_description)
        self.assertIn("Include known audience context", intent_description)
        self.assertIn("Do not invent an age", intent_description)
        self.assertIn("известный контекст аудитории", definition.rule_instructions["russian"])
        self.assertIn("known audience and age in intent", definition.rule_instructions["english"])

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
            max_tokens=32768,
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
            max_tokens=32768,
            reasoning_effort=None,
        )

    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_requests_syllable_stress_and_preserves_capitalization(self, get_model):
        lyrics = "[Verse]\nпоДАрок и доРОга\n[Chorus]\nмоЛОко"
        get_model.return_value.get_response.return_value = lyrics
        tool = LyricsTool(self._config(lyrics_model="test-model"))

        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про подарок"}))

        system = get_model.return_value.get_response.call_args.args[0][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("заглавными буквами ударного", system["content"])
        self.assertIn("поДАрок", system["content"])
        self.assertIn("Не используй знак +", system["content"])
        self.assertEqual(result, lyrics)

    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_requests_yo_spelling_and_preserves_it(self, get_model):
        lyrics = "[Verse]\nПЁС иДЁТ по голоЛЁДу\n[Chorus]\nВСЁ гоТОво"
        model = get_model.return_value
        model.get_response.return_value = lyrics
        tool = LyricsTool(self._config(lyrics_model="test-model"))

        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про пса зимой"}))

        system = model.get_response.call_args.args[0][0]["content"]
        self.assertIn('не заменяй её на «е»', system)
        self.assertIn('сохраняй букву «ё»', system)
        self.assertIn('Не заменяй все «е» на «ё» механически', system)
        self.assertIn('все люди / всё готово', system)
        self.assertEqual(result, lyrics)

    @patch("src.tools.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_truncated_lyrics_retry_with_larger_budget(self, get_model, sleep):
        model = get_model.return_value
        model.max_tokens = 8192
        budgets = []
        def response(*args, **kwargs):
            budgets.append(model.max_tokens)
            if len(budgets) < 3:
                raise TruncatedResponseError("MAX_TOKENS")
            return "Он ждёт у маминых ног."
        model.get_response.side_effect = response
        tool = LyricsTool(self._config(lyrics_model="test-model", lyrics_max_output_tokens=8192))
        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про пуделя"}))
        self.assertEqual(result, "Он ждёт у маминых ног.")
        self.assertEqual(budgets, [8192, 16384, 32768])
        self.assertEqual(sleep.await_count, 2)
        self.assertEqual(model.max_tokens, 8192)

    @patch("src.tools.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_truncation_retries_are_bounded(self, get_model, sleep):
        model = get_model.return_value
        model.max_tokens = 8192
        model.get_response.side_effect = TruncatedResponseError("MAX_TOKENS")
        tool = LyricsTool(self._config(lyrics_model="test-model"))
        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про пуделя"}))
        self.assertIn("Error", result)
        self.assertEqual(model.get_response.call_count, 3)
        self.assertEqual(sleep.await_count, 2)
        self.assertEqual(model.max_tokens, 8192)

    @patch("src.tools.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_permanent_lyrics_error_is_not_retried(self, get_model, sleep):
        model = get_model.return_value
        model.get_response.side_effect = RuntimeError("Invalid API key")
        tool = LyricsTool(self._config(lyrics_model="test-model"))
        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про пуделя"}))
        self.assertIn("Invalid API key", result)
        model.get_response.assert_called_once()
        sleep.assert_not_awaited()

    @patch("src.tools.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_child_and_unknown_audiences_rewrite_smoking(self, get_model, sleep):
        for audience in ("child", "unknown"):
            with self.subTest(audience=audience):
                model = get_model.return_value
                model.get_response.side_effect = [
                    "МАма броСАет в УРну быЧОК.", "МАма клаДЁТ в карМАН клюЧИ.",
                ]
                tool = LyricsTool(self._config(lyrics_model="test-model"))
                result = asyncio.run(tool.generate_lyrics_async({
                    "intent": "Песня про девочку с пуделем", "audience": audience,
                }))
                self.assertEqual(result, "МАма клаДЁТ в карМАН клюЧИ.")
                self.assertIn("Не включай курение", model.get_response.call_args.args[0][0]["content"])

    @patch("src.tools.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_child_smoking_rejection_is_bounded(self, get_model, sleep):
        model = get_model.return_value
        model.get_response.return_value = "Мама курит сигарету."
        tool = LyricsTool(self._config(lyrics_model="test-model"))
        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про прогулку", "audience": "child"}))
        self.assertTrue(result.startswith("Error"))
        self.assertEqual(model.get_response.call_count, 3)

    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_adult_audience_keeps_original_lyrics(self, get_model):
        lyrics = "Мама бросает в урну бычок."
        get_model.return_value.get_response.return_value = lyrics
        tool = LyricsTool(self._config(lyrics_model="test-model"))
        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про прогулку", "audience": "adult"}))
        self.assertEqual(result, lyrics)

    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_review_revises_until_approved(self, get_model):
        model = get_model.return_value
        model.get_response.side_effect = ["Initial draft", "Better version", "Final version", "LYRICS_APPROVED"]
        tool = LyricsTool(self._config(lyrics_model="test-model", lyrics_review_max_passes=5))
        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про прогулку"}))
        self.assertEqual(result, "Final version")
        self.assertEqual(model.get_response.call_count, 4)
        messages = model.get_response.call_args.args[0]
        self.assertEqual([m["content"] for m in messages if m["role"] == "assistant"],
                         ["Initial draft", "Better version", "Final version"])
        self.assertIn("правильность ударений", messages[-1]["content"])
        self.assertIn("аудитории", messages[-1]["content"])

    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_review_stops_when_initial_draft_is_approved(self, get_model):
        model = get_model.return_value
        model.get_response.side_effect = ["Complete lyrics", "LYRICS_APPROVED"]
        tool = LyricsTool(self._config(lyrics_model="test-model", lyrics_review_max_passes=5))
        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про прогулку"}))
        self.assertEqual(result, "Complete lyrics")
        self.assertEqual(model.get_response.call_count, 2)

    @patch("src.lyrics_tool.LyricsTool._get_model")
    def test_review_limit_does_not_return_unapproved_lyrics(self, get_model):
        model = get_model.return_value
        model.get_response.side_effect = ["Draft", "Revision", "Another revision"]
        tool = LyricsTool(self._config(lyrics_model="test-model", lyrics_review_max_passes=2))
        result = asyncio.run(tool.generate_lyrics_async({"intent": "Песня про прогулку"}))
        self.assertEqual(result, "Error: Lyrics were not approved after 2 review passes")
        self.assertEqual(model.get_response.call_count, 3)

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
