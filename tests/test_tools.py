import logging
import unittest
from typing import List

from src.tools import clean_response, extract_sentences, split_long_sentence


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
        self.assert_sentences("$x.f$abc. 123", ["$x.f$abc.", "123"])

    def test_multiple_special_patterns_immediate_punctuation(self):
        self.assert_sentences("$pattern1$.$pattern2$.", ["$pattern1$.$pattern2$."])

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

    def test_whitespace(self):
        self.assert_sentences('$remember: начало факта. Второе предложение. $ Остальной текст.',
                              ['$remember: начало факта. Второе предложение. $ Остальной текст.'])

    def test_numeration(self):
        self.assert_sentences("1. First. And. 2. Second",
                              ["1. First.", "And.", "2. Second"])

    def test_complex_numeration(self):
        self.assert_sentences(
            "1. First point. 2. Second point with multiple sentences. It continues here. 3. Third point.",
            ["1. First point.", "2. Second point with multiple sentences.", "It continues here.",
             "3. Third point."])

    def test_mixed_numeration_and_regular_sentences(self):
        self.assert_sentences("This is a regular sentence. 1. Then a numbered point. Another regular sentence.",
                              ["This is a regular sentence.", "1. Then a numbered point.", "Another regular sentence."])

    def test_numeration_after_colon(self):
        self.assert_sentences("Intro: 1. one. 2. two.",
                              ['Intro:', '1. one.', '2. two.'])

    def test_numeration_after_curly_brace(self):
        self.assert_sentences("text) 1. one. 2. two.",
                              ['text)', '1. one.', '2. two.'])

    def test_numeration_after_time(self):
        self.assert_sentences("Швеция - 3:56.92 8. Нидерланды - 3:59.52",
                              ["Швеция - 3:56.92", "8. Нидерланды - 3:59.52"])

    def test_plain_text_no_punctuation_returns_empty(self):
        text = "встреча Путина и Трампа запланирована на 15 августа на Аляске"
        self.assert_sentences(text, [text])


class TestCleanResponse(unittest.TestCase):

    def test_italic_single_word(self):
        self.assertEqual(clean_response("Она может описать *как*, но не *зачем*."),
                         "Она может описать **как**, но не **зачем**.")

    def test_bold_word(self):
        # ** kept for Yandex SpeechKit emphasis
        self.assertEqual(clean_response("**bold** text"), "**bold** text")

    def test_bold_and_italic(self):
        # ** kept; single * converted to ** emphasis
        self.assertEqual(clean_response("**bold** and *italic* text"), "**bold** and **italic** text")

    def test_italic_phrase(self):
        # lowercase + spaces = emphasis, converted to Yandex ** emphasis
        self.assertEqual(clean_response("*some phrase here*"), "**some phrase here**")

    def test_meta_tags_removed(self):
        self.assertEqual(clean_response("$lang: ru$ hello"), " hello")

    def test_no_markers(self):
        self.assertEqual(clean_response("normal text"), "normal text")

    def test_empty_string(self):
        self.assertEqual(clean_response(""), "")

    def test_mixed_meta_and_asterisks(self):
        self.assertEqual(clean_response("$lang: ru$ *важно* текст"), " **важно** текст")

    def test_emoji_kept(self):
        # Yandex SpeechKit handles emoji as punctuation — keep them
        self.assertEqual(clean_response("Привет! 😄 Как дела?"), "Привет! 😄 Как дела?")

    def test_asterisk_with_quotes_and_emoji(self):
        result = clean_response('Вот: *"Olá!"* Это значит «Привет!» 😄 Пока!')
        self.assertEqual(result, 'Вот: "Olá!" Это значит «Привет!» 😄 Пока!')

    def test_stage_direction_removed(self):
        self.assertEqual(
            clean_response("*Мигающий красный свет, имитирующий раздражение*"),
            "",
        )

    def test_stage_direction_mid_sentence_removed(self):
        self.assertEqual(
            clean_response("Текст. *Резко переключается обратно в нормальный режим* Продолжение."),
            "Текст.  Продолжение.",
        )

    def test_lowercase_asterisk_with_spaces_converted(self):
        # lowercase + spaces = emphasis phrase, converted to Yandex ** emphasis
        self.assertEqual(
            clean_response("*светится нежно-розовым* продолжаем."),
            "**светится нежно-розовым** продолжаем.",
        )

    def test_lowercase_phrase_asterisk_converted(self):
        # lowercase emphasis phrase → converted to Yandex ** emphasis
        self.assertEqual(
            clean_response("Дискриминация *по гендерному признаку* недопустима."),
            "Дискриминация **по гендерному признаку** недопустима.",
        )

    def test_uppercase_single_word_asterisk_converted(self):
        # Single word (no spaces) — converted to Yandex ** emphasis
        self.assertEqual(clean_response("*Важно*"), "**Важно**")

    def test_underscore_emphasis_stripped(self):
        self.assertEqual(clean_response("_Куплет_"), "Куплет")

    def test_underscore_word_stripped(self):
        self.assertEqual(clean_response("Это _важно_ для понимания."), "Это важно для понимания.")

    def test_tts_pause_tag_removed(self):
        self.assertEqual(clean_response("Привет. <[small]> Как дела?"), "Привет.  Как дела?")

    def test_tts_pause_tag_all_sizes_removed(self):
        for size in ("tiny", "small", "medium", "large", "huge"):
            self.assertEqual(clean_response(f"<[{size}]>"), "")

    def test_tts_malformed_pause_tag_removed(self):
        self.assertEqual(clean_response("<small>[small]</small>"), "")

    def test_tts_malformed_pause_tag_in_sentence(self):
        self.assertEqual(
            clean_response("Раз. <small>[small]</small> Два."),
            "Раз.  Два.",
        )


class TestSplitLongSentence(unittest.TestCase):

    def test_short_text_unchanged(self):
        text = "Короткий текст."
        self.assertEqual(split_long_sentence(text, max_length=245), [text])

    def test_exactly_at_limit_unchanged(self):
        text = "a" * 245
        self.assertEqual(split_long_sentence(text, max_length=245), [text])

    def test_splits_at_comma(self):
        part1 = "a" * 200 + ","
        part2 = " " + "b" * 200
        result = split_long_sentence(part1 + part2, max_length=245)
        self.assertEqual(len(result), 2)
        self.assertLessEqual(len(result[0]), 245)
        self.assertLessEqual(len(result[1]), 245)

    def test_splits_at_em_dash(self):
        part1 = "a" * 100
        part2 = "b" * 100
        text = part1 + " — " + part2
        result = split_long_sentence(text, max_length=150)
        self.assertEqual(len(result), 2)
        self.assertIn("a" * 100, result[0])
        self.assertIn("b" * 100, result[1])

    def test_splits_at_semicolon(self):
        part1 = "x" * 200 + ";"
        part2 = " " + "y" * 200
        result = split_long_sentence(part1 + part2, max_length=245)
        self.assertEqual(len(result), 2)
        self.assertLessEqual(len(result[0]), 245)

    def test_no_split_point_hard_splits_at_word(self):
        # Long text with no commas/dashes — falls back to word split
        words = ["слово"] * 60  # ~360 chars with spaces
        text = " ".join(words)
        result = split_long_sentence(text, max_length=245)
        self.assertGreater(len(result), 1)
        for chunk in result:
            self.assertLessEqual(len(chunk), 245)

    def test_real_russian_news_sentence(self):
        text = (
            "Ну и сегодня ночью — иранский Новый год, Навруз. "
            "Люди должны прыгать через костры на улицах, "
            "но власти запретили выходить, говорят — израильские агенты хотят устроить хаос. "
            "Такие дела. Как тебе такой расклад?"
        )
        result = split_long_sentence(text, max_length=245)
        for chunk in result:
            self.assertLessEqual(len(chunk), 245)
        # Reassembled text should contain all content
        combined = " ".join(result)
        self.assertIn("Навруз", combined)
        self.assertIn("израильские агенты", combined)

    def test_multiple_delimiters_greedy_packing(self):
        # Short pieces that can be packed together into one chunk
        text = "один, два, три, четыре, пять"
        result = split_long_sentence(text, max_length=245)
        self.assertEqual(result, [text])

    def test_empty_string(self):
        self.assertEqual(split_long_sentence("", max_length=245), [""])


if __name__ == '__main__':
    unittest.main()
