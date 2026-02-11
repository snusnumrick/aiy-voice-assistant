import unittest

from src.ai_models_with_tools import ClaudeAIModelWithTools
from src.config import Config


class TestClaudeCostMetrics(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    def test_compute_turn_cost_with_rates(self):
        config = self._config(
            claude_cost_input_per_million=3.0,
            claude_cost_output_per_million=15.0,
            claude_cost_cache_write_per_million=3.75,
            claude_cost_cache_read_per_million=0.30,
        )
        model = ClaudeAIModelWithTools(config, tools=[])
        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 300,
        }
        expected = (
            (1000 / 1_000_000) * 3.0
            + (500 / 1_000_000) * 15.0
            + (200 / 1_000_000) * 3.75
            + (300 / 1_000_000) * 0.30
        )
        self.assertAlmostEqual(model._compute_turn_cost_usd(usage), expected)

    def test_compute_turn_cost_without_rates(self):
        config = self._config()
        model = ClaudeAIModelWithTools(config, tools=[])
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30,
        }
        self.assertIsNone(model._compute_turn_cost_usd(usage))

    def test_record_usage_from_event_merges_shapes(self):
        config = self._config()
        model = ClaudeAIModelWithTools(config, tools=[])
        model._turn_usage_accumulator = model._new_usage_totals()

        model._record_usage_from_event({"usage": {"input_tokens": 10, "output_tokens": 2}})
        model._record_usage_from_event({"message": {"usage": {"cache_read_input_tokens": 5}}})
        model._record_usage_from_event({"delta": {"usage": {"output_tokens": 3}}})

        usage = model._turn_usage_accumulator
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 5)
        self.assertEqual(usage["cache_read_input_tokens"], 5)
        self.assertEqual(usage["cache_creation_input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
