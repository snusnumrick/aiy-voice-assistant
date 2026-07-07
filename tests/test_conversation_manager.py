# ruff: noqa: E402

import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

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


class _ToolProvenanceStreamingModel(_StreamingModel):
    def __init__(self, chunks, provenance_messages):
        super().__init__(chunks)
        self._provenance_messages = provenance_messages

    def consume_tool_provenance_messages(self):
        messages = self._provenance_messages
        self._provenance_messages = []
        return messages


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
                with patch("src.conversation_manager.get_location", return_value="In Test.", create=True):
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

    async def test_buffer_preserves_emotion_after_plain_text_when_stream_splits_inside_tag(self):
        chunks = [
            '$lang: ru$ Ну что, справедливо.$emotion:{"light":{"color":[255,80,0],',
            '"behavior":"breathing","brightness":"dark","period":3}, "voice":{"tone":"plain"}}$ '
            'Сказал — не подумав, попался.Буду стараться.',
        ]

        with patch("src.conversation_manager.WebSearcher"):
            with patch("src.conversation_manager.ClaudeAIModel"):
                with patch("src.conversation_manager.get_location", return_value="In Test.", create=True):
                    with patch("src.conversation_manager.get_tool_usage_stats", return_value=None):
                        with patch.object(ConversationManager, "load_facts", return_value=[]):
                            with patch.object(ConversationManager, "load_rules", return_value=[]):
                                manager = ConversationManager(
                                    self._config(),
                                    _StreamingModel(chunks),
                                    timezone="UTC",
                                    enabled_tools=[],
                                )

        batches = []
        async for batch in manager.get_response("Проверь эмоцию"):
            batches.extend(batch)

        self.assertEqual(
            [item["text"] for item in batches],
            [
                "Ну что, справедливо.",
                "Сказал — не подумав, попался.Буду стараться.",
            ],
        )
        self.assertIsNone(batches[0]["emotion"])
        self.assertEqual(batches[1]["emotion"]["light"]["color"], [255, 80, 0])
        self.assertEqual(batches[1]["emotion"]["light"]["brightness"], "dark")

    async def test_nightly_cleanup_prunes_old_web_search_reports(self):
        with patch("src.conversation_manager.WebSearcher") as mock_searcher_class:
            with patch("src.conversation_manager.ClaudeAIModel"):
                with patch("src.conversation_manager.get_location", return_value="In Test.", create=True):
                    with patch("src.conversation_manager.get_tool_usage_stats", return_value=None):
                        with patch.object(ConversationManager, "load_facts", return_value=[]):
                            with patch.object(ConversationManager, "load_rules", return_value=[]):
                                manager = ConversationManager(
                                    self._config(
                                        form_new_memories_at_night=False,
                                        clean_message_history_at_night=False,
                                    ),
                                    _StreamingModel([]),
                                    timezone="UTC",
                                    enabled_tools=[],
                                )

        with patch.object(manager, "_process_facts", new=AsyncMock()):
            with patch.object(manager, "_process_rules", new=AsyncMock()):
                await manager.process_and_clean()

        mock_searcher_class.return_value.cleanup_old_search_reports.assert_called_once_with()

    async def test_appends_tool_provenance_to_history_after_response(self):
        with patch("src.conversation_manager.WebSearcher"):
            with patch("src.conversation_manager.ClaudeAIModel"):
                with patch("src.conversation_manager.get_location", return_value="In Test.", create=True):
                    with patch("src.conversation_manager.get_tool_usage_stats", return_value=None):
                        with patch.object(ConversationManager, "load_facts", return_value=[]):
                            with patch.object(ConversationManager, "load_rules", return_value=[]):
                                manager = ConversationManager(
                                    self._config(),
                                    _ToolProvenanceStreamingModel(
                                        ["Answer from search."],
                                        [
                                            {
                                                "role": "assistant",
                                                "content": (
                                                    "[Tool context retained: internet_search "
                                                    "query='Taylor Swift wedding']"
                                                ),
                                            }
                                        ],
                                    ),
                                    timezone="UTC",
                                    enabled_tools=[],
                                )

        batches = []
        async for batch in manager.get_response("Tell me"):
            batches.extend(batch)

        self.assertEqual([item["text"] for item in batches], ["Answer from search."])
        self.assertEqual(manager.message_history[-2]["content"], "Answer from search.")
        self.assertEqual(
            manager.message_history[-1]["content"],
            "[Tool context retained: internet_search query='Taylor Swift wedding']",
        )

    async def test_nightly_cleanup_prunes_expired_reminders(self):
        with patch("src.conversation_manager.WebSearcher"):
            with patch("src.conversation_manager.ClaudeAIModel"):
                with patch("src.conversation_manager.get_location", return_value="In Test.", create=True):
                    with patch("src.conversation_manager.get_tool_usage_stats", return_value=None):
                        with patch.object(ConversationManager, "load_facts", return_value=[]):
                            with patch.object(ConversationManager, "load_rules", return_value=[]):
                                manager = ConversationManager(
                                    self._config(
                                        form_new_memories_at_night=False,
                                        clean_message_history_at_night=False,
                                        reminders_file="test-reminders.json",
                                    ),
                                    _StreamingModel([]),
                                    timezone="UTC",
                                    enabled_tools=[],
                                )

        with patch("src.conversation_manager.ReminderManager") as mock_reminder_manager:
            with patch.object(manager, "_process_facts", new=AsyncMock()):
                with patch.object(manager, "_process_rules", new=AsyncMock()):
                    await manager.process_and_clean()

        mock_reminder_manager.assert_called_once_with("test-reminders.json", timezone="UTC")
        mock_reminder_manager.return_value.cleanup_expired_reminders.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
