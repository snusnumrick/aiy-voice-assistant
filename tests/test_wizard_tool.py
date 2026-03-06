# -*- coding: utf-8 -*-
import unittest
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.config import Config
from src.wizard_tool import WizardTool, fix_markdown_formatting


class TestWizardToolConfig(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    @patch("src.wizard_tool.OpenAIModel")
    def test_defaults_to_openai_model(self, mock_openai_model):
        wizard = WizardTool(self._config())

        self.assertIsNotNone(wizard)
        mock_openai_model.assert_called_once()
        _, kwargs = mock_openai_model.call_args
        self.assertEqual(kwargs["model_id"], "gpt-5")
        self.assertEqual(kwargs["max_tokens"], 8192)

    @patch("src.wizard_tool.GeminiAIModel")
    def test_explicit_gemini_provider_normalizes_shorthand_model(self, mock_gemini_model):
        wizard = WizardTool(
            self._config(
                wizard_model_api="gemini",
                wizard_model_id="3.1 pro",
                wizard_max_tokens=4096,
            )
        )

        self.assertIsNotNone(wizard)
        mock_gemini_model.assert_called_once()
        _, kwargs = mock_gemini_model.call_args
        self.assertEqual(kwargs["model_id"], "gemini-3.1-pro")
        self.assertEqual(kwargs["max_tokens"], 4096)

    @patch("src.wizard_tool.GeminiAIModel")
    def test_wizard_gemini_model_switches_provider_implicitly(self, mock_gemini_model):
        wizard = WizardTool(self._config(wizard_gemini_model="3.1 pro"))

        self.assertIsNotNone(wizard)
        mock_gemini_model.assert_called_once()
        _, kwargs = mock_gemini_model.call_args
        self.assertEqual(kwargs["model_id"], "gemini-3.1-pro")


class TestFixMarkdownFormatting(unittest.TestCase):
    def test_fixes_inline_headers_and_inline_numbered_lists(self):
        raw = """--- # Природа Времени и Сознания:Нейрокогнитивный и Философский Анализ
---
##1. Деконструкция основных компонентов
Анализ связи времени и сознания требует разделения проблемы на несколько фундаментальных элементов:- Физическое время:Объективная шкала.##2. Ключевые концепции
Последние исследования предлагают конкретные механизмы:1)Первый механизм:Описание.2)Второй механизм:Описание.
Сенсорные сигналы ( зрение, слух, осязание )обрабатываются мозгом, а ретенции )и протенции согласуются."""

        formatted = fix_markdown_formatting(raw)

        self.assertIn("---\n\n# Природа Времени и Сознания: Нейрокогнитивный и Философский Анализ", formatted)
        self.assertIn("## 1. Деконструкция основных компонентов", formatted)
        self.assertIn("\n\n## 2. Ключевые концепции", formatted)
        self.assertIn(":\n\n1) Первый механизм", formatted)
        self.assertIn(".\n\n2) Второй механизм", formatted)
        self.assertIn(") обрабатываются", formatted)
        self.assertIn(") и протенции", formatted)
        self.assertNotIn("##1", formatted)
        self.assertNotIn(".##", formatted)
        self.assertNotIn(":1)", formatted)

    def test_does_not_split_single_bullet_into_separate_paragraph(self):
        raw = (
            '- Специфическое настоящее ( specious present ):Психологическое "сейчас" '
            "не является математической точкой нулевой длительности."
            "Оно имеет психологическую протяженность ( от нескольких миллисекунд до секунд ), "
            "внутри которой события воспринимаются как происходящие одновременно или в "
            "непосредственной связке."
        )

        formatted = fix_markdown_formatting(raw)

        self.assertIn(
            '- Специфическое настоящее ( specious present ): Психологическое "сейчас" '
            "не является математической точкой нулевой длительности. "
            "Оно имеет психологическую протяженность",
            formatted,
        )
        self.assertNotIn("длительности.\n\nОно", formatted)

    def test_does_not_split_plain_paragraph_between_sentences(self):
        raw = (
            "С точки зрения современной науки, время и сознание неразрывно переплетены:"
            "наше восприятие течения времени является продуктом самого сознания, а не "
            "прямым отражением физической реальности.Мозг выступает в роли архитектора реальности."
        )

        formatted = fix_markdown_formatting(raw)

        self.assertIn(
            "физической реальности. Мозг выступает в роли архитектора реальности.",
            formatted,
        )
        self.assertNotIn("физической реальности.\n\nМозг", formatted)

    @patch("src.wizard_tool.OpenAIModel")
    def test_save_report_async_writes_raw_report_content(self, mock_openai_model):
        with TemporaryDirectory() as reports_dir, TemporaryDirectory() as docs_dir:
            wizard = WizardTool(
                Config(
                    config_file="__missing_config__.json",
                    user_config_file="__missing_user__.json",
                    wizard_reports_dir=reports_dir,
                )
            )
            raw = "##1. Заголовок\nТекст:1)Первый пункт.##2. Второй раздел"

            with patch("src.wizard_tool._discover_doc_folder", return_value=Path(docs_dir)):
                filepath = asyncio.run(wizard.save_report_async("Вопрос", raw))

            saved = Path(filepath).read_text(encoding="utf-8")
            doc_files = list(Path(docs_dir).glob("*.md"))

            self.assertIn(raw, saved)
            self.assertIn("##1. Заголовок", saved)
            self.assertEqual(len(doc_files), 1)
            formatted_doc = doc_files[0].read_text(encoding="utf-8")
            self.assertIn("## 1. Заголовок", formatted_doc)
            self.assertIn(":\n\n1) Первый пункт", formatted_doc)
            self.assertIn("\n\n## 2. Второй раздел", formatted_doc)
            self.assertNotIn("##1.", formatted_doc)
            self.assertNotIn(":1)", formatted_doc)



if __name__ == "__main__":
    unittest.main()
