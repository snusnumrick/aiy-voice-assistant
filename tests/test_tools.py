import logging
import unittest
from typing import List

from src.tools import extract_sentences


class TestExtractSentences(unittest.TestCase):

    def setUp(self):
        # Disable logging for tests
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        # Re-enable logging after tests
        logging.disable(logging.NOTSET)

    def assert_sentences(self, input_text: str, expected_output: List[str]):
        self.assertEqual(extract_sentences(input_text), expected_output)

    def test_basic_sentences(self):
        self.assert_sentences("This is a sentence. This is another one!",
            ["This is a sentence.", "This is another one!"])

    def test_special_pattern_middle(self):
        self.assert_sentences("Hello $world$! This is a test.", ["Hello $world$!", "This is a test."])

    def test_special_pattern_end(self):
        self.assert_sentences("This is a $test.", ["This is a $test."])

    def test_multiple_special_patterns(self):
        self.assert_sentences("Hello $world$! This is a $test. And $another one$.",
            ["Hello $world$!", "This is a $test. And $another one$."])

    def test_special_pattern_no_punctuation(self):
        self.assert_sentences("Start$mid$end", ["Start$mid$end"])

    def test_special_pattern_immediate_punctuation(self):
        self.assert_sentences("$x.f$abc.123", ["$x.f$abc.", "123"])

    def test_multiple_special_patterns_immediate_punctuation(self):
        self.assert_sentences("$pattern1$.$pattern2$.", ["$pattern1$.", "$pattern2$."])

    def test_incomplete_special_pattern(self):
        self.assert_sentences("This is an $incomplete pattern", ["This is an $incomplete pattern"])

    def test_special_pattern_at_start(self):
        self.assert_sentences("$start$ of the sentence.", ["$start$ of the sentence."])

    def test_multiple_sentences_with_special_patterns(self):
        self.assert_sentences("First $special$. Second $special$! Third $special$?",
            ["First $special$.", "Second $special$!", "Third $special$?"])

    def test_empty_input(self):
        self.assert_sentences("", [])

    def test_only_special_pattern(self):
        self.assert_sentences("$pattern$", ["$pattern$"])

    def test_russian_sentences(self):
        self.assert_sentences("Это предложение на русском. И еще одно!", ["Это предложение на русском.", "И еще одно!"])

    def test_mixed_languages(self):
        self.assert_sentences("This is English. Это русский. $pattern$ Again English.",
            ["This is English.", "Это русский.", "$pattern$ Again English."])

    def test_ellipsis(self):
        self.assert_sentences("First sentence... Second sentence.", ["First sentence...", "Second sentence."])

    def test_multiple_punctuation(self):
        self.assert_sentences("Is this a question?! Yes, it is!", ["Is this a question?!", "Yes, it is!"])


if __name__ == '__main__':
    unittest.main()