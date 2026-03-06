import asyncio
import unittest
from unittest.mock import Mock, patch

from src.config import Config
from src.web_search import GeminiSearch


class TestGeminiSearchGrounding(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    def test_search_model_uses_config_override(self):
        search = GeminiSearch(self._config(gemini_search_model="gemini-3.1-pro"))

        self.assertEqual(search.model_id, "gemini-3.1-pro")
        self.assertIn("/models/gemini-3.1-pro:generateContent", search.url)

    @patch("src.web_search.requests.post")
    def test_search_requests_grounding_but_returns_plain_text(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "Spain won Euro 2024. "
                                    "This victory marked Spain's record fourth European title."
                                )
                            }
                        ]
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/uefa", "title": "UEFA"}},
                            {"web": {"uri": "https://example.com/bbc", "title": "BBC Sport"}},
                        ],
                        "groundingSupports": [
                            {
                                "segment": {"endIndex": 20},
                                "groundingChunkIndices": [0],
                            },
                            {
                                "segment": {"endIndex": 74},
                                "groundingChunkIndices": [0, 1],
                            },
                        ],
                    },
                }
            ]
        }
        mock_post.return_value = mock_response

        result = asyncio.run(GeminiSearch(self._config()).search("Who won Euro 2024?"))

        self.assertEqual(
            result,
            "Spain won Euro 2024. This victory marked Spain's record fourth European title.",
        )
        sent_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(sent_payload["tools"], [{"google_search": {}}])
        self.assertIn(
            "Use Google Search grounding only when it improves factual accuracy.",
            sent_payload["contents"][0]["parts"][0]["text"],
        )

    @patch("src.web_search.requests.post")
    def test_search_returns_plain_text_when_grounding_metadata_missing(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Plain grounded answer text"}],
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = asyncio.run(GeminiSearch(self._config()).search("Simple query"))

        self.assertEqual(result, "Plain grounded answer text")


if __name__ == "__main__":
    unittest.main()
