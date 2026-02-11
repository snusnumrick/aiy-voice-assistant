import unittest

from src.llm_optimization import (
    PROFILE_CHAT_ONLY,
    PROFILE_CREATIVE,
    PROFILE_HOME_CONTROL,
    PROFILE_RESEARCH,
    classify_tool_profile,
    parse_volume_intent,
    resolve_tools_for_profile,
    wants_detailed_response,
)


class TestLlmOptimization(unittest.TestCase):
    def test_classify_tool_profile(self):
        self.assertEqual(classify_tool_profile("сделай тише"), PROFILE_HOME_CONTROL)
        self.assertEqual(classify_tool_profile("нарисуй кота"), PROFILE_CREATIVE)
        self.assertEqual(classify_tool_profile("найди новости"), PROFILE_RESEARCH)
        self.assertEqual(classify_tool_profile("просто поболтаем"), PROFILE_CHAT_ONLY)

    def test_parse_volume_intent(self):
        i1 = parse_volume_intent("тише")
        self.assertIsNotNone(i1)
        self.assertEqual(i1.action, "decrease")
        self.assertIsNone(i1.value)

        i2 = parse_volume_intent("громче на 15")
        self.assertIsNotNone(i2)
        self.assertEqual(i2.action, "increase")
        self.assertEqual(i2.value, 15)

        i3 = parse_volume_intent("громкость на 33")
        self.assertIsNotNone(i3)
        self.assertEqual(i3.action, "set")
        self.assertEqual(i3.value, 33)

    def test_wants_detailed_response(self):
        self.assertTrue(wants_detailed_response("объясни подробно"))
        self.assertTrue(wants_detailed_response("in detail please"))
        self.assertFalse(wants_detailed_response("коротко"))

    def test_resolve_tools_for_profile(self):
        available = {"control_speaker_volume", "generate_image", "recall_memory"}
        selected = resolve_tools_for_profile(PROFILE_HOME_CONTROL, available)
        self.assertEqual(selected, {"control_speaker_volume"})


if __name__ == "__main__":
    unittest.main()
