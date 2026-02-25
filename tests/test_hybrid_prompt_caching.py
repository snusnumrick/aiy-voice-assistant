import unittest
from unittest.mock import patch

from src.ai_models_with_tools import ClaudeAIModelWithTools
from src.config import Config
from src.conversation_manager import ConversationManager


class TestHybridPromptCaching(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    def test_message_breakpoint_added_in_hybrid_mode(self):
        cfg = self._config(
            claude_enable_prompt_caching=True,
            claude_prompt_caching_hybrid_enabled=True,
            claude_hybrid_cache_min_messages=4,
            claude_hybrid_cache_recent_uncached_messages=2,
        )
        model = ClaudeAIModelWithTools(cfg, tools=[])
        original = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "latest"},
        ]

        updated = model._apply_hybrid_message_cache_breakpoint(original)

        self.assertEqual(original[4]["content"], "u3")
        self.assertIsInstance(updated[4]["content"], list)
        block = updated[4]["content"][0]
        self.assertEqual(block["type"], "text")
        self.assertEqual(block["text"], "u3")
        self.assertEqual(block["cache_control"], {"type": "ephemeral"})
        # Keep newest messages uncached
        self.assertEqual(updated[5]["content"], "a3")
        self.assertEqual(updated[6]["content"], "latest")

    def test_message_breakpoint_not_added_when_disabled(self):
        cfg = self._config(
            claude_enable_prompt_caching=True,
            claude_prompt_caching_hybrid_enabled=False,
        )
        model = ClaudeAIModelWithTools(cfg, tools=[])
        original = [{"role": "user", "content": "hello"}]
        updated = model._apply_hybrid_message_cache_breakpoint(original)
        self.assertEqual(updated, original)

    def test_dynamic_prefix_frozen_for_cache_window(self):
        cm = ConversationManager.__new__(ConversationManager)
        cm.optimize_prompt_split_dynamic = True
        cm.config = self._config(claude_enable_prompt_caching=True)
        cm.freeze_dynamic_context_for_cache = True
        cm.prompt_cache_window_seconds = 300
        cm._cached_dynamic_context_prefix = None
        cm._cached_dynamic_context_expires_at = 0.0
        cm.timezone = "UTC"
        cm.location = "In Test."

        with patch("src.conversation_manager.time.time", side_effect=[1000.0, 1001.0, 1301.0]):
            with patch(
                "src.conversation_manager.get_current_datetime_english",
                side_effect=["T1", "T2"],
            ):
                p1 = cm._system_prompt_context_prefix()
                p2 = cm._system_prompt_context_prefix()
                p3 = cm._system_prompt_context_prefix()

        self.assertEqual(p1, "T1 In Test. ")
        self.assertEqual(p2, "T1 In Test. ")
        self.assertEqual(p3, "T2 In Test. ")


if __name__ == "__main__":
    unittest.main()
