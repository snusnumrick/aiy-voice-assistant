import unittest

from src.ai_models import ReasoningEffort, to_reasoning_effort


class TestReasoningEffortParse(unittest.TestCase):
    def test_basic_strings(self):
        self.assertEqual(to_reasoning_effort("quick"), ReasoningEffort.QUICK)
        self.assertEqual(to_reasoning_effort("thorough"), ReasoningEffort.THOROUGH)
        self.assertEqual(to_reasoning_effort("comprehensive"), ReasoningEffort.COMPREHENSIVE)

    def test_case_insensitive(self):
        self.assertEqual(to_reasoning_effort("Quick"), ReasoningEffort.QUICK)
        self.assertEqual(to_reasoning_effort("THOROUGH"), ReasoningEffort.THOROUGH)
        self.assertEqual(to_reasoning_effort("Comprehensive"), ReasoningEffort.COMPREHENSIVE)

    def test_passthrough_enum(self):
        self.assertEqual(to_reasoning_effort(ReasoningEffort.QUICK), ReasoningEffort.QUICK)

    def test_invalid_and_none(self):
        self.assertIsNone(to_reasoning_effort(None))
        self.assertIsNone(to_reasoning_effort("unknown"))


if __name__ == '__main__':
    unittest.main()
