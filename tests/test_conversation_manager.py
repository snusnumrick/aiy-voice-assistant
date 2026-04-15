# ruff: noqa: E402

import sys
import unittest
from unittest.mock import Mock, patch

mock_duckduckgo_search = Mock()
mock_duckduckgo_search.DDGS = Mock()
mock_lxml = Mock()
mock_lxml.html = Mock()
mock_pydub = Mock()
mock_pydub.AudioSegment = Mock()

sys.modules["aiofiles"] = Mock()
sys.modules["geocoder"] = Mock()
sys.modules["pydub"] = mock_pydub
sys.modules["duckduckgo_search"] = mock_duckduckgo_search
sys.modules["lxml"] = mock_lxml
sys.modules["lxml.html"] = mock_lxml.html

from src.config import Config
from src.conversation_manager import ConversationManager


class _StreamingModel:
    def __init__(self, chunks):
        self._chunks = chunks

    async def get_response_async(self, messages):
        for chunk in self._chunks:
            yield chunk


class TestConversationManagerBuffer(unittest.IsolatedAsyncioTestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            token_threshold=999999,
            sentence_buffer_enabled=True,
            sentence_buffer_max_length=500,
            **kwargs,
        )

    async def test_buffer_flushes_when_emotion_changes(self):
        response = (
            '$lang: ru$ Сначала без света. '
            '$emotion:{"light":{"color":[255,100,0],"behavior":"breathing","brightness":"bright","period":2},"voice":{"tone":"plain"}}$ '
            'Потом оранжевый. '
            '$emotion:{"light":{"color":[0,100,255],"behavior":"blinking","brightness":"bright","period":1},"voice":{"tone":"plain"}}$ '
            'Потом синий. '
            '$emotion:{}$ И в конце без света.'
        )

        with patch("src.conversation_manager.WebSearcher"):
            with patch("src.conversation_manager.ClaudeAIModel"):
                with patch("src.conversation_manager.get_location", return_value="In Test."):
                    with patch("src.conversation_manager.get_tool_usage_stats", return_value=None):
                        with patch.object(ConversationManager, "load_facts", return_value=[]):
                            with patch.object(ConversationManager, "load_rules", return_value=[]):
                                manager = ConversationManager(
                                    self._config(),
                                    _StreamingModel([response]),
                                    timezone="UTC",
                                    enabled_tools=[],
                                )

        batches = []
        async for batch in manager.get_response("Проверь свет"):
            batches.extend(batch)

        self.assertEqual(
            [item["text"] for item in batches],
            [
                "Сначала без света.",
                "Потом оранжевый.",
                "Потом синий.",
                "И в конце без света.",
            ],
        )
        self.assertIsNone(batches[0]["emotion"])
        self.assertEqual(batches[1]["emotion"]["light"]["color"], [255, 100, 0])
        self.assertEqual(batches[2]["emotion"]["light"]["color"], [0, 100, 255])
        self.assertEqual(batches[3]["emotion"], {})


if __name__ == "__main__":
    unittest.main()
