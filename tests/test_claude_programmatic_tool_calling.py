import unittest

from src.ai_models_with_tools import ClaudeAIModelWithTools, Tool, ToolParameter
from src.config import Config


class TestClaudeProgrammaticToolCalling(unittest.TestCase):
    @staticmethod
    async def _noop_processor(parameters):
        return "ok"

    def _tool(self, name: str, candidate: bool) -> Tool:
        return Tool(
            name=name,
            description=f"Tool {name}",
            iterative=True,
            parameters=[
                ToolParameter(
                    name="x",
                    type="string",
                    description="placeholder",
                )
            ],
            required=[],
            processor=self._noop_processor,
            programmatic_code_execution_candidate=candidate,
        )

    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    def test_programmatic_tool_is_added_only_for_active_candidate_tools(self):
        cfg = self._config(
            claude_enable_programmatic_tool_calling=True,
            claude_programmatic_code_execution_type="code_execution_20260120",
            claude_programmatic_code_execution_name="code_execution",
        )
        model = ClaudeAIModelWithTools(
            cfg,
            tools=[
                self._tool("enhanced_weather_info", candidate=True),
                self._tool("stress_marker", candidate=False),
            ],
        )

        # Candidate tool selected -> code execution tool is included.
        model.set_request_options(
            tool_names={"enhanced_weather_info"},
            response_max_tokens=None,
            system_blocks=None,
        )
        selected = model._get_runtime_tools_description()
        self.assertTrue(
            any(
                tool.get("type") == "code_execution_20260120"
                and tool.get("name") == "code_execution"
                and tool.get("allowed_callers") == ["direct"]
                for tool in selected
            )
        )

        # Non-candidate tool selected -> no code execution tool.
        model.set_request_options(
            tool_names={"stress_marker"},
            response_max_tokens=None,
            system_blocks=None,
        )
        filtered = model._get_runtime_tools_description()
        self.assertFalse(any(tool.get("name") == "code_execution" for tool in filtered))

    def test_programmatic_tool_not_added_when_no_candidates(self):
        cfg = self._config(claude_enable_programmatic_tool_calling=True)
        model = ClaudeAIModelWithTools(cfg, tools=[self._tool("stress_marker", candidate=False)])
        filtered = model._get_runtime_tools_description()
        self.assertFalse(any(tool.get("name") == "code_execution" for tool in filtered))

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
