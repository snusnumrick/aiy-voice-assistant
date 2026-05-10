import asyncio
import time
import unittest
from datetime import date
from unittest.mock import AsyncMock, Mock, patch

import httpx

from src.config import Config
from src.web_search import BraveLLMContext, GeminiSearch, ParallelSearch, Tavily, WebSearcher
from src.web_search_tool import WebSearchTool


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

        result = asyncio.run(
            GeminiSearch(self._config()).search(
                "Who won Euro 2024?",
                after_date="2024-01-01",
                location="Spain",
            )
        )

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
        self.assertIn(
            "Prefer sources published on or after 2024-01-01.",
            sent_payload["contents"][0]["parts"][0]["text"],
        )
        self.assertIn(
            "Use search context relevant to Spain",
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


class TestParallelSearch(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    def test_init_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                ParallelSearch(self._config())

    @patch("src.web_search.httpx.AsyncClient.post", new_callable=AsyncMock)
    def test_search_uses_parallel_payload_and_formats_results(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {
                    "url": "https://example.com/news",
                    "title": "Today News",
                    "publish_date": "2026-04-24",
                    "excerpts": ["News excerpt one.", "News excerpt two."],
                }
            ],
            "warnings": None,
        }
        mock_post.return_value = mock_response

        config = self._config(parallel_search_max_results=10)
        with patch.dict("os.environ", {"PARALLEL_API_KEY": "test-parallel-key"}):
            result = asyncio.run(
                ParallelSearch(config).search(
                    "news of today",
                    after_date="2026-04-23",
                    location="US",
                )
            )

        self.assertIn("### Today News", result)
        self.assertIn("**URL:** https://example.com/news", result)
        self.assertIn("**Published:** 2026-04-24", result)
        self.assertIn("News excerpt one.", result)

        call_args = mock_post.call_args
        self.assertEqual(call_args.args[0], ParallelSearch.BASE_URL)
        self.assertEqual(call_args.kwargs["headers"]["x-api-key"], "test-parallel-key")
        self.assertEqual(
            call_args.kwargs["json"],
            {
                "search_queries": ["news of today"],
                "mode": "advanced",
                "advanced_settings": {
                    "max_results": 10,
                    "location": "us",
                    "source_policy": {
                        "after_date": "2026-04-23",
                    },
                },
            },
        )


class TestWebSearcherProviders(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    def test_threaded_provider_results_keep_configured_order(self):
        searcher = WebSearcher.__new__(WebSearcher)
        searcher.config = self._config(web_search_max_workers=2)

        def search_single_provider(provider_name, query, after_date=None, location=None):
            if provider_name == "slow":
                time.sleep(0.02)
            return f"{provider_name}:{query}:{after_date}:{location}"

        searcher._search_single_provider = search_single_provider

        results = searcher._search_providers_sync(
            "query",
            ["slow", "fast"],
            after_date="2026-04-23",
            location="us",
        )

        self.assertEqual(
            results,
            ["slow:query:2026-04-23:us", "fast:query:2026-04-23:us"],
        )

    def test_enabled_providers_skips_missing_api_key_providers(self):
        searcher = WebSearcher.__new__(WebSearcher)
        searcher.config = self._config(web_search_providers=["parallel", "brave", "tavily"])
        searcher.parallel = None
        searcher.brave = object()
        searcher.tavily = object()

        self.assertEqual(searcher._get_enabled_providers(), ["brave", "tavily"])

    def test_threaded_provider_errors_include_exception_repr(self):
        class BlankMessageError(Exception):
            def __str__(self):
                return ""

        searcher = WebSearcher.__new__(WebSearcher)
        searcher.config = self._config(web_search_max_workers=1)

        def search_single_provider(provider_name, query, after_date=None, location=None):
            raise BlankMessageError("hidden detail")

        searcher._search_single_provider = search_single_provider

        with self.assertLogs("src.web_search", level="ERROR") as logs:
            results = searcher._search_providers_sync("query", ["tavily"])

        self.assertIsInstance(results[0], BlankMessageError)
        self.assertIn("BlankMessageError('hidden detail')", "\n".join(logs.output))


class TestTavilySearch(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    @patch("src.web_search.httpx.AsyncClient.post", new_callable=AsyncMock)
    def test_search_uses_bearer_auth_and_maps_optional_filters(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"answer": "fresh answer"}
        mock_post.return_value = mock_response

        config = self._config(tavily_search_topic="general")
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-tavily-key"}):
            result = asyncio.run(
                Tavily(config).search(
                    "news of today",
                    after_date="2026-04-23",
                    location="us",
                )
            )

        self.assertEqual(result, "fresh answer")
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-tavily-key")
        sent_payload = mock_post.call_args.kwargs["json"]
        self.assertNotIn("api_key", sent_payload)
        self.assertEqual(sent_payload["start_date"], "2026-04-23")
        self.assertEqual(sent_payload["country"], "united states")
        self.assertEqual(sent_payload["topic"], "general")

    @patch("src.web_search.httpx.AsyncClient.post", new_callable=AsyncMock)
    def test_search_returns_formatted_results_when_answer_missing(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Bremen exhibition",
                    "url": "https://example.com/bremen",
                    "content": "Exhibition details",
                }
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-tavily-key"}):
            result = asyncio.run(Tavily(self._config()).search("Bremen exhibitions"))

        self.assertIn("### Bremen exhibition", result)
        self.assertIn("**URL:** https://example.com/bremen", result)
        self.assertIn("Exhibition details", result)

    @patch("src.web_search.httpx.AsyncClient.post", new_callable=AsyncMock)
    def test_search_timeout_returns_empty_result(self, mock_post):
        mock_post.side_effect = httpx.ReadTimeout("timed out")

        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-tavily-key"}):
            result = asyncio.run(
                Tavily(self._config(tavily_search_timeout_sec=1)).search("slow query")
            )

        self.assertEqual(result, "")


class TestBraveSearch(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    @patch("src.web_search.httpx.AsyncClient.get", new_callable=AsyncMock)
    def test_search_maps_after_date_and_location(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "grounding": {
                "generic": [
                    {
                        "title": "Local result",
                        "url": "https://example.com/local",
                        "snippets": ["Local result excerpt."],
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"BRAVE_API_KEY": "test-brave-key"}):
            result = asyncio.run(
                BraveLLMContext(self._config()).search(
                    "best coffee",
                    after_date="2026-04-23",
                    location="San Francisco, CA, US",
                )
            )

        self.assertIn("### Local result", result)
        headers = mock_get.call_args.kwargs["headers"]
        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(headers["X-Loc-City"], "San Francisco")
        self.assertEqual(headers["X-Loc-State"], "CA")
        self.assertEqual(headers["X-Loc-Country"], "US")
        self.assertEqual(params["country"], "us")
        self.assertEqual(params["freshness"], f"2026-04-23to{date.today().isoformat()}")


class TestWebSearchTool(unittest.TestCase):
    def test_tool_definition_exposes_optional_search_parameters(self):
        definition = WebSearchTool.__new__(WebSearchTool).tool_definition()

        parameter_names = [parameter.name for parameter in definition.parameters]

        self.assertIn("query", parameter_names)
        self.assertIn("after_date", parameter_names)
        self.assertIn("location", parameter_names)
        self.assertEqual(definition.required, ["query"])

    def test_async_processor_passes_optional_search_parameters(self):
        tool = WebSearchTool.__new__(WebSearchTool)
        tool.web_searcher = Mock()
        tool.web_searcher.search_async = AsyncMock(return_value="search result")
        tool._start_processing = Mock()
        tool._stop_processing = Mock()

        result = asyncio.run(
            tool.do_search_async(
                {
                    "query": "news of today",
                    "after_date": "2026-04-23",
                    "location": "us",
                }
            )
        )

        self.assertEqual(result, "search result")
        tool.web_searcher.search_async.assert_awaited_once_with(
            "news of today",
            after_date="2026-04-23",
            location="us",
        )


if __name__ == "__main__":
    unittest.main()
