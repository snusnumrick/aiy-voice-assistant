import unittest

from src.responce_player import extract_language


class TestExtractLanguage(unittest.TestCase):

    def test_no_tags(self):
        self.assertEqual(extract_language("hello", "ru"), [("ru", "hello")])

    def test_single_tag(self):
        self.assertEqual(
            extract_language("$lang: ru$ привет", "en"),
            [("ru", "привет")],
        )

    def test_two_tags(self):
        self.assertEqual(
            extract_language("$lang: ru$ привет $lang: en$ hello", "ru"),
            [("ru", "привет"), ("en", "hello")],
        )

    def test_leading_text_uses_default(self):
        result = extract_language("intro $lang: en$ hello", "ru")
        self.assertEqual(result, [("ru", "intro"), ("en", "hello")])

    def test_trailing_tag_no_text_updates_language(self):
        # "$lang: ru$\n" at end with no text — must still return ("ru", "") so
        # callers can update their language-state tracker.
        result = extract_language("$lang: pt$ Era uma vez $lang: ru$\n", "ru")
        # Portuguese text present, then Russian tag with empty segment
        self.assertIn(("pt", "Era uma vez"), result)
        # Language-state tuple for Russian must be present even though text is empty
        self.assertIn(("ru", ""), result)

    def test_mixed_language_story(self):
        text = (
            "$lang: ru$ Хорошо, вот история "
            "$lang: pt$ Era uma vez um pescador. "
            "$lang: ru$ Вот пересказ."
        )
        result = extract_language(text, "ru")
        self.assertEqual(result, [
            ("ru", "Хорошо, вот история"),
            ("pt", "Era uma vez um pescador."),
            ("ru", "Вот пересказ."),
        ])


if __name__ == "__main__":
    unittest.main()
