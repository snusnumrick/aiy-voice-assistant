import asyncio
import json
import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from typing import Optional

import httpx
import requests
from duckduckgo_search import DDGS
from lxml import html

if __name__ == "__main__":
    # add current directory to python path
    sys.path.append(os.getcwd())

from src.ai_models import OpenRouterModel
from src.config import Config

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
]


class SearchProvider(ABC):
    """
    Abstract base class for internet search.
    """

    @abstractmethod
    async def search(self, query: str) -> str:
        """
        Search for the given query.

        Args:
            query (str): what to search for.

        Returns:
            str: The search result.
        """
        pass


class Google(SearchProvider):
    def __init__(self, config: Config):
        self.session = httpx.AsyncClient()

    async def _fetch_data(self, term, lang):
        url = f"https://www.google.com/search?q={requests.utils.quote(term)}&hl={lang}"
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = await self.session.get(url, headers=headers)
        return html.fromstring(response.content)

    @staticmethod
    def _get_text(tree, selector):
        elements = tree.cssselect(selector)
        return (
            " ".join(element.text_content() for element in elements) if elements else ""
        )

    async def search(self, term: str) -> str:
        start_time = time.time()
        tree = await self._fetch_data(term, "en")
        logger.debug(f"Google search fetch_data time: {time.time() - start_time}")

        selectors = [".sXLaOe", ".hgKElc", ".wx62f", ".HwtpBd", ".yxjZuf span", ".IZ6rdc"]
        results = [self._get_text(tree, selector) for selector in selectors]
        await asyncio.sleep(0)

        a1 = (
            self._get_text(tree, ".UDZeY span")
            .replace("Описание", "")
            .replace("ЕЩЁ", "")
        )
        await asyncio.sleep(0)
        a1 += self._get_text(tree, ".LGOjhe span")
        await asyncio.sleep(0)
        a2 = self._get_text(tree, ".yXK7lf span")
        await asyncio.sleep(0)

        brief_result = "; ".join(filter(None, results))
        result = brief_result or a2 or a1

        logger.debug(f"Google search took {(time.time() - start_time):.2f} seconds")
        return result


class GoogleCustomSearch(SearchProvider):
    def __init__(self, config: Config):
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"
        self.cs_key = os.environ.get("GOOGLE_CUSTOMSEARCH_KEY")
        self.api_key = os.environ.get("GOOGLE_API_KEY")

    async def search(self, term: str) -> str:
        start_time = time.time()
        params = {
            "q": term,
            "key": self.api_key,
            "cx": self.cs_key,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes

            results = response.json()

            # Process and return the results
            text_result = (
                "\n".join([item["snippet"] for item in results["items"]])
                if "items" in results
                else ""
            )
            logger.debug(
                f"Google Custom Search search took {time.time() - start_time} seconds"
            )
            return text_result

        except requests.RequestException as e:
            logger.error(f"An error occurred: {e}")
            return ""


class Perplexity(SearchProvider):
    def __init__(self, config: Config):
        from src.ai_models import PerplexityModel

        self.model = PerplexityModel(config)

    async def search(self, query: str) -> str:
        start_time = time.time()
        messages = [
            {
                "role": "user",
                "content": (query),
            },
        ]
        response = "".join([r async for r in self.model.get_response_async(messages)])
        duration = time.time() - start_time
        logger.debug(f"Perplexity search took {duration:.2f} seconds")
        return response


class GeminiSearch(SearchProvider):
    """
    Gemini-backed search provider using Google Search grounding.
    Returns plain answer text and lets Gemini use grounding only when it
    improves factual quality.
    """

    def __init__(self, config: Config):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model_id = config.get("gemini_search_model", "gemini-flash-lite-latest")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"

    @staticmethod
    def _extract_text(candidate: dict) -> str:
        parts = candidate.get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts if part.get("text")).strip()

    async def search(self, query: str) -> str:
        start_time = time.time()
        prompt = (
            "Use Google Search grounding only when it improves factual accuracy. "
            "Return only the plain answer text without citations or source annotations.\n\n"
            f"Query: {query}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingLevel": "LOW",
                },
            },
            "tools": [{"google_search": {}}],
        }
        headers = {"Content-Type": "application/json"}
        answer = ""
        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            candidate = data["candidates"][0]
            answer = self._extract_text(candidate)
            logger.debug(
                "Gemini grounding used: %s",
                bool(candidate.get("groundingMetadata")),
            )
        except Exception as e:
            logger.error(f"gemini search failed: {e}")
        duration = time.time() - start_time
        logger.debug(f"Gemini search (raw) took {duration:.2f} seconds")
        return answer


class Tavily(SearchProvider):
    def __init__(self, config: Config):
        self.api_key = os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key is not provided in environment variables")

    async def search(self, query: str):
        url = "https://api.tavily.com/search"

        start_time = time.time()

        # Prepare the request payload
        payload = {
            "api_key": self.api_key,
            "query": query,
            "include_answer": True,
            "search_depth": "advanced",
            "topic": "news",
        }

        try:
            # Make the POST request to the API
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
            if response.status_code == 400:
                try:
                    logger.error(
                        f"Tavily: {json.loads(response.content)['detail']['error']}"
                    )
                except Exception:
                    pass
            response.raise_for_status()  # Raise an exception for bad status codes

            # Parse and return the JSON response
            answer = response.json()["answer"]
            duration = time.time() - start_time
            logger.debug(f"Tavily search took {duration:.2f} seconds")
            return answer

        except requests.exceptions.RequestException as e:
            print(f"Error making request to Tavily API: {e}")
            raise

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            raise


class PersistentDDGS(DDGS):
    def _get_url(
        self, method: str, url: str, **kwargs
    ) -> Optional[httpx._models.Response]:
        resp = self._client.request(method, url, follow_redirects=True, **kwargs)
        if resp.status_code == 202:
            # try again in a few seconds if there is no answer yet
            if "Location" in resp.headers:
                status_update_url = resp.headers["Location"]
            # If the URL is in the response body, you'll need to parse the body to find it
            # This is just a generic example, adjust it according to your API documentation
            elif hasattr(resp, "status_url"):
                status_update_url = resp.status_url
            else:
                status_update_url = resp.url
            if status_update_url:
                time.sleep(3)
                resp = self._client.request(
                    "GET", status_update_url, follow_redirects=True
                )
        if self._is_500_in_url(str(resp.url)) or resp.status_code == 202:
            raise httpx._exceptions.HTTPError("")

        if resp.status_code == 200:
            return resp
        return None


class DuckDuckGoSearch(SearchProvider):
    def __init__(self, config):
        try:
            self.ddgs = PersistentDDGS()
        except Exception as e:
            logger.error(f"DDGS search could not be initialized: {e}")
            self.ddgs = None

    async def search(self, query: str) -> str:
        start_time = time.time()
        if not self.ddgs:
            return ""
        try:
            search_results = [
                {
                    "title": r["title"],
                    "url": r["href"],
                    **({"exerpt": r["body"]} if r.get("body") else {}),
                }
                for r in self.ddgs.text(query)
            ]
        except Exception as e:
            logger.error(f"DDGS dsearch failed: {e}")
            return ""

        await asyncio.sleep(0)

        results = "## Search results\n" + "\n\n".join(
            '### "{}"\n**URL:** {}  \n**Excerpt:** {}'.format(
                r["title"],
                r["url"],
                '"{}"'.format(r.get("exerpt")) if r.get("exerpt") else "N/A",
            )
            for r in search_results
        )

        # make it safe
        results = results.encode("utf-8", "ignore").decode("utf-8")
        duration = time.time() - start_time
        logger.debug(f"DDGS search took {duration:.2f} seconds")
        return results


class BraveLLMContext(SearchProvider):
    """
    Brave LLM Context API — returns pre-extracted page content optimised for
    grounding LLM responses.  Requires BRAVE_API_KEY in the environment.
    """

    BASE_URL = "https://api.search.brave.com/res/v1/llm/context"
    _cached_location_headers: Optional[dict] = None  # process-level cache

    def __init__(self, config: Config):
        self.api_key = os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY is not set in environment variables")
        self.count = config.get("brave_count", 10)
        self.max_tokens = config.get("brave_max_tokens", 4096)
        self.threshold_mode = config.get("brave_threshold_mode", "balanced")

    @classmethod
    def _get_location_headers(cls) -> dict:
        """Resolve location via IP once per process and cache the result."""
        if cls._cached_location_headers is not None:
            return cls._cached_location_headers
        try:
            import geocoder  # lazy import — same pattern as ai_models_with_tools.py
            g = geocoder.ip("me")
            headers = {}
            if g.lat:
                headers["X-Loc-Lat"] = str(g.lat)
            if g.lng:
                headers["X-Loc-Long"] = str(g.lng)
            if g.city:
                headers["X-Loc-City"] = g.city
            if g.state:
                headers["X-Loc-State"] = g.state
            if g.country_code:
                headers["X-Loc-Country"] = g.country_code
            logger.debug(f"Brave location headers resolved: {headers}")
            cls._cached_location_headers = headers
        except Exception as e:
            logger.debug(f"Brave location resolution failed, skipping: {e}")
            cls._cached_location_headers = {}
        return cls._cached_location_headers

    async def search(self, query: str) -> str:
        start_time = time.time()
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            **self._get_location_headers(),
        }
        params = {
            "q": query,
            "count": self.count,
            "maximum_number_of_tokens": self.max_tokens,
            "context_threshold_mode": self.threshold_mode,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(self.BASE_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            generic = data.get("grounding", {}).get("generic", [])
            if not generic:
                logger.debug("Brave LLM Context: no grounding results returned")
                return ""

            parts = []
            for item in generic:
                title = item.get("title", "")
                url = item.get("url", "")
                snippets = item.get("snippets", [])
                if snippets:
                    body = "\n".join(snippets)
                    parts.append(f"### {title}\n**URL:** {url}\n{body}")

            result = "\n\n".join(parts)
            duration = time.time() - start_time
            logger.debug(f"Brave LLM Context search took {duration:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Brave LLM Context search failed: {e}")
            return ""


class WebSearcher:
    def __init__(self, config):
        """Initializes multi-provider search engines with fallback for missing API key"""
        try:
            self.tavily = Tavily(config)
        except ValueError:
            self.tavily = None
            logger.debug("Tavily search disabled: TAVILY_API_KEY not set")
        self.google = Google(config)
        self.google_cs = GoogleCustomSearch(config)
        self.ai_model = OpenRouterModel(config, use_simple_model=True)
        self.perplexity = Perplexity(config)
        self.gemini = GeminiSearch(config)
        self.ddgs = DuckDuckGoSearch(config)
        self.config = config
        try:
            self.brave = BraveLLMContext(config)
        except ValueError:
            self.brave = None
            logger.debug("Brave LLM Context disabled: BRAVE_API_KEY not set")

    async def search_providers_async(self, query: str, enabled_providers):
        logger.debug(f"Searching for {query} with providers: {enabled_providers}")

        # Run search in a thread pool to avoid HTTP event loop saturation
        # This prevents search from competing with TTS HTTP requests
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._search_providers_sync, query, enabled_providers
        )

        combined_result = ""
        for provider, result in zip(enabled_providers, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error while searching with provider {provider}: {str(result)}"
                )
            else:
                logger.debug(f"\n---------\n{provider} result: {result}")
                combined_result += f"Result from {provider}: \n{result}\n"

        return combined_result

    def _search_providers_sync(self, query: str, enabled_providers):
        """
        Synchronous wrapper for searching providers in a separate thread.
        This avoids HTTP event loop saturation when combined with TTS requests.
        """
        import concurrent.futures

        results = []

        # Create a thread pool for concurrent search
        with concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="search") as executor:
            # Submit all search tasks to the thread pool
            future_to_provider = {
                executor.submit(self._search_single_provider, provider, query): provider
                for provider in enabled_providers
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_provider):
                provider = future_to_provider[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in thread for provider {provider}: {str(e)}")
                    results.append(e)

        return results

    def _search_single_provider(self, provider_name: str, query: str):
        """
        Execute search for a single provider synchronously.
        Creates a new event loop for this thread since provider.search() is async.
        """
        provider = getattr(self, provider_name)
        if provider is None:
            logger.warning(f"Provider '{provider_name}' is not available (disabled or missing API key)")
            return ""
        # Run the async search in a new event loop for this thread
        return asyncio.run(provider.search(query))

    async def search_async(self, query: str) -> str:
        logger.debug(f"Searching for {query}")
        start_time = time.time()

        # providers = ["gemini", "perplexity", "tavily", "brave"]
        providers = ["brave", "tavily"]

        try:
            combined_result = await self.search_providers_async(query, providers)

            # logger.info(f"\n---------\n{query} combined result: {combined_result}")
            #
            # prompt = (f"Answer short. Based on result from internet search below, what is the answer to the question: "
            #           f"{query}\n\n{combined_result}")
            # result = self.ai_model.get_response([{"role": "user", "content": prompt}])
            result = combined_result

            duration = time.time() - start_time
            logger.debug(
                f"Final search took {duration:.2f} seconds; result for query '{query}' is: {result}"
            )
            return result

        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            raise


async def loop():
    config = Config()
    web_searcher = WebSearcher(config)
    while True:
        query = input(">")
        result = await web_searcher.search_providers_async(query, ["brave"])
        print(result)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()

    asyncio.run(loop())
