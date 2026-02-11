import os
import tempfile
import unittest

from src.tool_usage_stats import ToolUsageStats


class TestToolUsageStats(unittest.TestCase):
    def test_record_and_rank(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "tool_usage_stats.json")
            stats = ToolUsageStats(path)

            stats.record_call("a")
            stats.record_call("b")
            stats.record_call("a")

            self.assertEqual(stats.total_calls(), 3)
            self.assertEqual(stats.get_count("a"), 2)
            self.assertEqual(stats.get_count("b"), 1)
            self.assertEqual(stats.top_tools(2), ["a", "b"])

            reloaded = ToolUsageStats(path)
            self.assertEqual(reloaded.get_count("a"), 2)
            self.assertEqual(reloaded.total_calls(), 3)


if __name__ == "__main__":
    unittest.main()
