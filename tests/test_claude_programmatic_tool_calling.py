import unittest

from src.ai_models_with_tools import ClaudeAIModelWithTools
from src.config import Config


class TestClaudeProgrammaticToolCalling(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    def test_programmatic_tool_is_added_and_forced_in_runtime_filter(self):
        cfg = self._config(
            claude_enable_programmatic_tool_calling=True,
            claude_programmatic_code_execution_type="code_execution_20260120",
            claude_programmatic_code_execution_name="code_execution",
        )
        model = ClaudeAIModelWithTools(cfg, tools=[])
        self.assertTrue(
            any(
                tool.get("type") == "code_execution_20260120"
                and tool.get("name") == "code_execution"
                for tool in model.tools_description
            )
        )

        model.set_request_options(tool_names=set(), response_max_tokens=None, system_blocks=None)
        filtered = model._get_runtime_tools_description()
        self.assertTrue(
            any(tool.get("name") == "code_execution" for tool in filtered)
        )

    def test_programmatic_and_caching_beta_headers_are_merged(self):
        cfg = self._config(
            claude_enable_prompt_caching=True,
            claude_prompt_caching_beta_header="prompt-caching-2024-07-31",
            claude_enable_programmatic_tool_calling=True,
            claude_programmatic_tool_beta_header="advanced-tool-use-2025-11-20",
        )
        model = ClaudeAIModelWithTools(cfg, tools=[])
        header = model.headers.get("anthropic-beta", "")
        self.assertIn("prompt-caching-2024-07-31", header)
        self.assertIn("advanced-tool-use-2025-11-20", header)

    def test_followup_user_text_is_added_after_tool_result_when_enabled(self):
        cfg = self._config(
            claude_enable_programmatic_tool_calling=True,
            claude_programmatic_append_followup_text=True,
            claude_programmatic_tool_followup_text="Continue.",
        )
        model = ClaudeAIModelWithTools(cfg, tools=[])
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x"}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        ]
        model._append_programmatic_tool_followup_message(messages)
        self.assertEqual(messages[-1], {"role": "user", "content": "Continue."})

    def test_followup_user_text_is_not_added_when_disabled(self):
        cfg = self._config(
            claude_enable_programmatic_tool_calling=True,
            claude_programmatic_append_followup_text=False,
        )
        model = ClaudeAIModelWithTools(cfg, tools=[])
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x"}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        ]
        model._append_programmatic_tool_followup_message(messages)
        self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()
