import asyncio
import json
import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import httpx
import requests
from duckduckgo_search import DDGS
from lxml import html
from openai import AuthenticationError

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

COUNTRY_NAME_TO_CODE = {
    "australia": "au",
    "canada": "ca",
    "france": "fr",
    "germany": "de",
    "great britain": "gb",
    "japan": "jp",
    "united kingdom": "gb",
    "united states": "us",
    "united states of america": "us",
    "usa": "us",
}

COUNTRY_CODE_TO_TAVILY_COUNTRY = {
    "au": "australia",
    "ca": "canada",
    "de": "germany",
    "fr": "france",
    "gb": "united kingdom",
    "jp": "japan",
    "uk": "united kingdom",
    "us": "united states",
}


def _clean_optional_search_param(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _country_code_from_location(location: Optional[str]) -> Optional[str]:
    location = _clean_optional_search_param(location)
    if not location:
        return None
    token = location.split(",")[-1].strip().lower().replace(".", "")
    if token in COUNTRY_NAME_TO_CODE:
        return COUNTRY_NAME_TO_CODE[token]
    if token == "uk":
        return "gb"
    if len(token) == 2 and token.isalpha():
        return token
    return None


def _tavily_country_from_location(location: Optional[str]) -> Optional[str]:
    location = _clean_optional_search_param(location)
    if not location:
        return None
    country_code = _country_code_from_location(location)
    if country_code:
        return COUNTRY_CODE_TO_TAVILY_COUNTRY.get(country_code)
    normalized = location.split(",")[-1].strip().lower()
    if normalized in COUNTRY_CODE_TO_TAVILY_COUNTRY.values():
        return normalized
    return None


def _brave_freshness_from_after_date(after_date: Optional[str]) -> Optional[str]:
    after_date = _clean_optional_search_param(after_date)
    if not after_date:
        return None
    return f"{after_date}to{date.today().isoformat()}"


def _brave_location_headers(location: Optional[str]) -> dict[str, str]:
    location = _clean_optional_search_param(location)
    if not location:
        return {}

    parts = [part.strip() for part in location.split(",") if part.strip()]
    if not parts:
        return {}

    headers = {}
    if len(parts) == 1:
        country_code = _country_code_from_location(parts[0])
        if country_code:
            headers["X-Loc-Country"] = country_code.upper()
        else:
            headers["X-Loc-City"] = parts[0]
        return headers

    headers["X-Loc-City"] = parts[0]
    if len(parts) == 2:
        country_code = _country_code_from_location(parts[1])
        if country_code:
            headers["X-Loc-Country"] = country_code.upper()
        else:
            headers["X-Loc-State"] = parts[1]
        return headers

    headers["X-Loc-State"] = parts[1]
    country_code = _country_code_from_location(parts[-1])
    if country_code:
        headers["X-Loc-Country"] = country_code.upper()
    return headers


def _query_with_search_constraints(
    query: str,
    after_date: Optional[str],
    location: Optional[str],
) -> str:
    constraints = []
    after_date = _clean_optional_search_param(after_date)
    location = _clean_optional_search_param(location)
    if after_date:
        constraints.append(f"Prefer sources published on or after {after_date}.")
    if location:
        constraints.append(f"Use location context: {location}.")
    if not constraints:
        return query
    return f"{query}\n\nSearch constraints:\n" + "\n".join(f"- {item}" for item in constraints)


def _positive_int(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _json_object_from_text(text: str) -> Optional[dict]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _json_objects_from_text(text: str, limit: int = 50) -> list[dict]:
    decoder = json.JSONDecoder()
    objects = []
    source = str(text or "")
    index = 0
    while len(objects) < limit:
        start = source.find("{", index)
        if start == -1:
            break
        try:
            parsed, offset = decoder.raw_decode(source[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
        index = start + max(offset, 1)
    return objects


def _json_objects_with_sources_from_evidence(
    evidence: str,
    limit: int = 50,
) -> list[tuple[dict, str]]:
    objects = []
    current_source = ""
    for raw_line in str(evidence or "").splitlines():
        line = raw_line.strip()
        if line.startswith("**URL:**"):
            current_source = line.removeprefix("**URL:**").strip()
            continue
        for item in _json_objects_from_text(line, limit=limit - len(objects)):
            objects.append((item, current_source))
            if len(objects) >= limit:
                return objects
    return objects


class SearchProvider(ABC):
    """
    Abstract base class for internet search.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        """
        Search for the given query.

        Args:
            query (str): what to search for.
            after_date (Optional[str]): prefer or filter results on or after YYYY-MM-DD.
            location (Optional[str]): prefer or filter results for a country or place.

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

    async def search(
        self,
        term: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
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

    async def search(
        self,
        term: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        start_time = time.time()
        params = {
            "q": term,
            "key": self.api_key,
            "cx": self.cs_key,
        }
        country_code = _country_code_from_location(location)
        if country_code:
            params["gl"] = country_code

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

    async def search(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        start_time = time.time()
        messages = [
            {
                "role": "user",
                "content": _query_with_search_constraints(query, after_date, location),
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

    async def search(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        start_time = time.time()
        prompt_parts = [
            (
                "Use Google Search grounding only when it improves factual accuracy. "
                "Return only the plain answer text without citations or source annotations."
            )
        ]
        after_date = _clean_optional_search_param(after_date)
        location = _clean_optional_search_param(location)
        if after_date:
            prompt_parts.append(f"Prefer sources published on or after {after_date}.")
        if location:
            prompt_parts.append(f"Use search context relevant to {location} when geography affects the answer.")
        prompt_parts.append(f"Query: {query}")
        prompt = "\n\n".join(prompt_parts)
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
        self.search_depth = config.get("tavily_search_depth", "advanced")
        self.topic = config.get("tavily_search_topic", "general")
        self.country = config.get("tavily_search_country")
        self.timeout = float(config.get("tavily_search_timeout_sec", 30))

    @staticmethod
    def _format_results(data: dict) -> str:
        results = data.get("results", [])
        if not results:
            return ""

        parts = []
        for item in results:
            title = str(item.get("title") or "Search result").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            meta = [f"### {title}"]
            if url:
                meta.append(f"**URL:** {url}")
            if content:
                meta.append(content)
            parts.append("\n".join(meta))
        return "\n\n".join(parts)

    async def search(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ):
        url = "https://api.tavily.com/search"

        start_time = time.time()

        # Prepare the request payload
        payload = {
            "query": query,
            "include_answer": False,
            "search_depth": self.search_depth,
            "topic": self.topic,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        after_date = _clean_optional_search_param(after_date)
        if after_date:
            payload["start_date"] = after_date

        country = (
            _tavily_country_from_location(location)
            or _tavily_country_from_location(self.country)
        )
        if country:
            if self.topic == "general":
                payload["country"] = country
            else:
                logger.debug("Tavily location skipped because country is only supported for general topic")

        try:
            # Make the POST request to the API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 400:
                try:
                    logger.error(
                        f"Tavily: {json.loads(response.content)['detail']['error']}"
                    )
                except Exception:
                    pass
            if response.status_code >= 400:
                logger.error(
                    "Tavily API returned %s: %s",
                    response.status_code,
                    response.text[:1000],
                )
            response.raise_for_status()  # Raise an exception for bad status codes

            # Parse and return the JSON response
            data = response.json()
            answer = self._format_results(data)
            duration = time.time() - start_time
            logger.debug(f"Tavily search took {duration:.2f} seconds")
            return answer

        except httpx.TimeoutException as e:
            logger.warning("Tavily search timed out after %.1fs: %r", self.timeout, e)
            return ""
        except httpx.HTTPError as e:
            logger.error("Tavily search HTTP error: %r", e)
            return ""
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            logger.error("Tavily search returned an unexpected response: %r", e)
            return ""


class ParallelSearch(SearchProvider):
    """
    Parallel Search API provider.

    Requires PARALLEL_API_KEY in the environment or parallel_api_key in config.
    """

    BASE_URL = "https://api.parallel.ai/v1/search"

    def __init__(self, config: Config):
        self.api_key = os.environ.get("PARALLEL_API_KEY") or config.get("parallel_api_key")
        if not self.api_key:
            raise ValueError("PARALLEL_API_KEY is not set in environment variables")
        self.mode = str(config.get("parallel_search_mode", "advanced"))
        self.max_results = int(config.get("parallel_search_max_results", 10))
        self.location = config.get("parallel_search_location", "us")
        self.after_date = config.get("parallel_search_after_date")
        self.max_chars_per_result = config.get("parallel_search_max_chars_per_result")
        self.timeout = float(config.get("parallel_search_timeout_sec", 30))

    def _build_payload(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> dict:
        after_date = _clean_optional_search_param(after_date) or self.after_date
        location = _clean_optional_search_param(location) or self.location
        advanced_settings = {
            "max_results": self.max_results,
        }
        if location:
            advanced_settings["location"] = str(location).lower()
        if after_date:
            advanced_settings["source_policy"] = {
                "after_date": str(after_date),
            }
        if self.max_chars_per_result:
            advanced_settings["excerpt_settings"] = {
                "max_chars_per_result": int(self.max_chars_per_result),
            }

        return {
            "search_queries": [query],
            "mode": self.mode,
            "advanced_settings": advanced_settings,
        }

    @staticmethod
    def _format_excerpts(excerpts) -> str:
        if isinstance(excerpts, str):
            excerpts = [excerpts]
        if not isinstance(excerpts, list):
            return ""

        parts = []
        for excerpt in excerpts:
            if isinstance(excerpt, dict):
                text = excerpt.get("text") or excerpt.get("content") or excerpt.get("excerpt")
                if not text:
                    text = json.dumps(excerpt, ensure_ascii=False)
            else:
                text = str(excerpt)
            text = text.strip()
            if text:
                parts.append(text)
        return "\n".join(parts)

    @classmethod
    def _format_results(cls, data: dict) -> str:
        results = data.get("results", [])
        if not results:
            return ""

        parts = []
        for item in results:
            title = str(item.get("title") or "Search result").strip()
            url = str(item.get("url") or "").strip()
            publish_date = item.get("publish_date")
            excerpts = cls._format_excerpts(item.get("excerpts", []))

            meta = []
            if url:
                meta.append(f"**URL:** {url}")
            if publish_date:
                meta.append(f"**Published:** {publish_date}")
            body = "\n".join([f"### {title}", *meta, excerpts]).strip()
            if body:
                parts.append(body)
        return "\n\n".join(parts)

    async def search(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        start_time = time.time()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.BASE_URL,
                    headers=headers,
                    json=self._build_payload(query, after_date, location),
                )
            if response.status_code >= 400:
                logger.error(
                    "Parallel Search API returned %s: %s",
                    response.status_code,
                    response.text[:1000],
                )
            response.raise_for_status()
            result = self._format_results(response.json())
            duration = time.time() - start_time
            logger.debug(f"Parallel search took {duration:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Parallel search failed: {e}")
            return ""


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

    async def search(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
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
        self.country = config.get("brave_country")
        self.freshness = config.get("brave_freshness")

    @classmethod
    def _get_location_headers(cls, location: Optional[str] = None) -> dict:
        """Resolve location via IP once per process and cache the result."""
        if location:
            return _brave_location_headers(location)
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

    async def search(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        start_time = time.time()
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            **self._get_location_headers(location),
        }
        params = {
            "q": query,
            "count": self.count,
            "maximum_number_of_tokens": self.max_tokens,
            "context_threshold_mode": self.threshold_mode,
        }
        country_code = _country_code_from_location(location) or _country_code_from_location(
            self.country
        )
        if country_code:
            params["country"] = country_code
        freshness = _brave_freshness_from_after_date(after_date) or self.freshness
        if freshness:
            params["freshness"] = freshness
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
        self.config = config
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
        try:
            self.parallel = ParallelSearch(config)
        except ValueError:
            self.parallel = None
            logger.debug("Parallel search disabled: PARALLEL_API_KEY not set")
        try:
            self.brave = BraveLLMContext(config)
        except ValueError:
            self.brave = None
            logger.debug("Brave LLM Context disabled: BRAVE_API_KEY not set")

    async def search_providers_async(
        self,
        query: str,
        enabled_providers,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ):
        logger.debug(f"Searching for {query} with providers: {enabled_providers}")

        # Run search in a thread pool to avoid HTTP event loop saturation
        # This prevents search from competing with TTS HTTP requests
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._search_providers_sync,
            query,
            enabled_providers,
            after_date,
            location,
        )

        combined_result = ""
        for provider, result in zip(enabled_providers, results):
            if isinstance(result, Exception):
                logger.error(
                    "Error while searching with provider %s: %r",
                    provider,
                    result,
                )
            elif result:
                logger.debug(f"\n---------\n{provider} result: {result}")
                combined_result += f"Result from {provider}: \n{result}\n"

        return combined_result

    def _search_providers_sync(
        self,
        query: str,
        enabled_providers,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """
        Synchronous wrapper for searching providers in a separate thread.
        This avoids HTTP event loop saturation when combined with TTS requests.
        """
        import concurrent.futures

        if not enabled_providers:
            return []

        results_by_provider = {}
        max_workers = min(
            len(enabled_providers),
            max(1, int(self.config.get("web_search_max_workers", 3))),
        )

        # Create a thread pool for concurrent search
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="search",
        ) as executor:
            # Submit all search tasks to the thread pool
            future_to_provider = {
                executor.submit(
                    self._search_single_provider,
                    provider,
                    query,
                    after_date,
                    location,
                ): provider
                for provider in enabled_providers
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_provider):
                provider = future_to_provider[future]
                try:
                    result = future.result()
                    results_by_provider[provider] = result
                except Exception as e:
                    logger.error("Error in thread for provider %s: %r", provider, e)
                    results_by_provider[provider] = e

        return [results_by_provider.get(provider, "") for provider in enabled_providers]

    def _search_single_provider(
        self,
        provider_name: str,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """
        Execute search for a single provider synchronously.
        Creates a new event loop for this thread since provider.search() is async.
        """
        provider = getattr(self, provider_name)
        if provider is None:
            logger.warning(f"Provider '{provider_name}' is not available (disabled or missing API key)")
            return ""
        # Run the async search in a new event loop for this thread
        return asyncio.run(provider.search(query, after_date=after_date, location=location))

    def _get_enabled_providers(self) -> list[str]:
        configured = self.config.get("web_search_providers", ["parallel", "brave", "tavily"])
        if isinstance(configured, str):
            providers = [p.strip() for p in configured.split(",")]
        else:
            providers = list(configured)

        enabled = []
        for provider in providers:
            if not provider:
                continue
            provider_name = str(provider).strip()
            if getattr(self, provider_name, None) is None:
                logger.debug("Skipping unavailable search provider: %s", provider_name)
                continue
            enabled.append(provider_name)
        return enabled

    @staticmethod
    def _evidence_metadata(evidence: str) -> dict:
        urls = set()
        published_dates = set()
        title_count = 0
        for raw_line in str(evidence or "").splitlines():
            line = raw_line.strip()
            if line.startswith("### "):
                title_count += 1
            elif line.startswith("**URL:**"):
                url = line.removeprefix("**URL:**").strip()
                if url:
                    urls.add(url)
            elif line.startswith("**Published:**"):
                published = line.removeprefix("**Published:**").strip()
                if published:
                    published_dates.add(published)

        return {
            "source_count": len(urls) or title_count,
            "title_count": title_count,
            "published_date_count": len(published_dates),
            "char_count": len(str(evidence or "")),
        }

    @staticmethod
    def _is_event_json_object(value: dict) -> bool:
        event_type = value.get("@type")
        if isinstance(event_type, list):
            return any("event" in str(item).lower() for item in event_type)
        return "event" in str(event_type or "").lower()

    @staticmethod
    def _format_event_location(location) -> str:
        if isinstance(location, str):
            return location.strip()
        if not isinstance(location, dict):
            return ""

        parts = []
        name = str(location.get("name") or "").strip()
        if name:
            parts.append(name)
        address = location.get("address")
        if isinstance(address, dict):
            for key in ("streetAddress", "addressLocality", "addressRegion", "addressCountry"):
                value = str(address.get(key) or "").strip()
                if value and value not in parts:
                    parts.append(value)
        return ", ".join(parts)

    @staticmethod
    def _first_table_value(row: dict, keys: tuple[str, ...]) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None:
                value = str(value).strip()
                if value:
                    return value
        return ""

    @staticmethod
    def _table_event_candidates(item: dict, source: str) -> list[dict[str, str]]:
        table = item.get("table")
        if not isinstance(table, list):
            return []

        candidates = []
        fallback_source = source or str(item.get("url") or item.get("@id") or "").strip()
        for row in table:
            if not isinstance(row, dict):
                continue
            name = WebSearcher._first_table_value(
                row,
                ("Event", "Veranstaltung", "Title", "Titel", "Name", "name", "title"),
            )
            date_time = WebSearcher._first_table_value(
                row,
                ("Termin", "Datum", "Date", "date", "When", "Zeit", "startDate"),
            )
            location = WebSearcher._first_table_value(
                row,
                ("Ort", "Venue", "Location", "location", "Wo", "Adresse"),
            )
            if not name or not date_time:
                continue
            candidates.append(
                {
                    "name": name,
                    "date_time": date_time,
                    "location": location,
                    "source": fallback_source or str(item.get("title") or "").strip(),
                }
            )
        return candidates

    def _extract_event_candidates(self, evidence: str) -> list[dict[str, str]]:
        max_candidates = _positive_int(
            self.config.get("web_search_max_structured_candidates", 24),
            24,
        )
        candidates = []
        seen = set()
        for item, surrounding_source in _json_objects_with_sources_from_evidence(
            evidence,
            limit=max_candidates * 4,
        ):
            extracted = []
            if self._is_event_json_object(item):
                name = str(item.get("name") or "").strip()
                start_date = str(item.get("startDate") or "").strip()
                end_date = str(item.get("endDate") or "").strip()
                source = str(item.get("url") or item.get("@id") or surrounding_source).strip()
                location = self._format_event_location(item.get("location"))
                if name or start_date:
                    extracted.append(
                        {
                            "name": name,
                            "date_time": (
                                f"{start_date} to {end_date}"
                                if end_date and end_date != start_date
                                else start_date
                            ),
                            "location": location,
                            "source": source,
                        }
                    )
            extracted.extend(self._table_event_candidates(item, surrounding_source))

            for candidate in extracted:
                identity = (
                    candidate.get("name", ""),
                    candidate.get("date_time", ""),
                    candidate.get("source", ""),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                candidates.append(candidate)
                if len(candidates) >= max_candidates:
                    return candidates
        return candidates

    def _format_event_candidates_for_prompt(self, evidence: str) -> str:
        lines = []
        for candidate in self._extract_event_candidates(evidence):
            parts = [
                candidate.get("name", ""),
                candidate.get("date_time", ""),
                candidate.get("location", ""),
                candidate.get("source", ""),
            ]
            line = " | ".join(part for part in parts if part)
            if line:
                lines.append(f"- {line}")
        if not lines:
            return ""
        return (
            "Structured event candidates extracted from evidence:\n"
            + "\n".join(lines)
        )

    def _suggest_answer_mode(
        self,
        evidence: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        setting = str(
            self.config.get("web_search_structured_extraction", "auto")
        ).strip().lower()
        if setting in {"structured", "always", "true", "yes"}:
            return "structured"
        if setting in {"direct", "off", "false", "no", "none"}:
            return "direct"

        metadata = self._evidence_metadata(evidence)
        source_threshold = _positive_int(
            self.config.get("web_search_structured_source_threshold", 5),
            5,
        )
        if self._extract_event_candidates(evidence):
            return "structured"
        has_explicit_constraints = bool(
            _clean_optional_search_param(after_date)
            or _clean_optional_search_param(location)
        )
        if has_explicit_constraints and metadata["source_count"] > 1:
            return "structured"
        if metadata["published_date_count"] > 1:
            return "structured"
        if metadata["source_count"] >= source_threshold:
            return "structured"
        return "direct"

    def _compact_search_evidence(self, evidence: str) -> str:
        max_chars = _positive_int(
            self.config.get("web_search_max_evidence_chars", 12000),
            12000,
        )
        max_sources = _positive_int(
            self.config.get("web_search_max_sources", 8),
            8,
        )
        lines = str(evidence or "").splitlines()
        compacted_lines = []
        source_count = 0
        skip_current_source = False

        for line in lines:
            if line.strip().startswith("### "):
                source_count += 1
                skip_current_source = source_count > max_sources
            if not skip_current_source:
                compacted_lines.append(line)

        compacted = "\n".join(compacted_lines).strip()
        if len(compacted) > max_chars:
            compacted = compacted[:max_chars].rstrip() + "\n[Search evidence truncated.]"
        return compacted

    def _build_answer_prompt(
        self,
        query: str,
        evidence: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        compacted_evidence = self._compact_search_evidence(evidence)
        suggested_mode = self._suggest_answer_mode(
            evidence,
            after_date=after_date,
            location=location,
        )
        event_candidates = self._format_event_candidates_for_prompt(evidence)
        event_candidate_section = (
            f"\n{event_candidates}\n"
            if event_candidates
            else "\nStructured event candidates extracted from evidence:\n- None found.\n"
        )
        constraints = []
        after_date = _clean_optional_search_param(after_date)
        location = _clean_optional_search_param(location)
        if after_date:
            constraints.append(f"- Prefer or require sources on or after {after_date}.")
        if location:
            constraints.append(f"- Prefer or require sources relevant to {location}.")
        constraint_text = "\n".join(constraints) if constraints else "- No explicit constraints."

        return f"""
You are a strict search-result extraction and answer-composition tool.

User question:
{query}

Search constraints:
{constraint_text}

{event_candidate_section}
Search evidence:
{compacted_evidence}

Suggested response mode: {suggested_mode}

Return only valid JSON. The JSON must use this shape:
{{
  "mode": "direct",
  "answer": "short direct user-facing answer",
  "items": [
    {{
      "name": "event, fact, place, article, or result name",
      "date_time": "date/time if available",
      "location": "venue or location if available",
      "source": "source URL or source name",
      "confidence": "high, medium, or low"
    }}
  ],
  "notes": ["brief ignored/noisy-result notes when useful"]
}}

Rules:
1. The mode value must be either "direct" or "structured".
2. Use "direct" for simple questions with one clear answer.
3. Use "structured" when the evidence contains several dated, location-specific, or conflicting items.
4. Do not summarize the whole result set.
5. Exclude date-mismatched, location-mismatched, generic, or duplicate results.
6. Prefer official venue, city, museum, or event pages over aggregators.
7. Use aggregators only when no official source is present, and mark them lower confidence.
8. Keep "answer" concise because it may be spoken aloud.
9. Include only items that directly answer the user question.
10. If the user question names a date window, include only items inside that window.
11. Do not claim that no specific events are listed when the structured event candidates contain dated items matching the question.
12. If the evidence is insufficient, say that in "answer" and leave "items" empty.
"""

    @staticmethod
    def _render_answer_json(data: dict) -> str:
        answer = str(data.get("answer") or "").strip()
        mode = str(data.get("mode") or "direct").strip().lower()
        if mode != "structured":
            return answer

        lines = [answer] if answer else []
        items = data.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                parts = [
                    str(item.get("name") or "").strip(),
                    str(item.get("date_time") or "").strip(),
                    str(item.get("location") or "").strip(),
                    str(item.get("source") or "").strip(),
                ]
                line = " - ".join(part for part in parts if part)
                confidence = str(item.get("confidence") or "").strip()
                if confidence:
                    line = f"{line} ({confidence})" if line else f"({confidence})"
                if line:
                    lines.append(f"- {line}")

        notes = data.get("notes")
        if isinstance(notes, list):
            note_text = " ".join(str(note).strip() for note in notes if str(note).strip())
            if note_text:
                lines.append(f"Notes: {note_text}")
        return "\n".join(lines).strip()

    def _render_answer_response(self, response: str) -> str:
        data = _json_object_from_text(response)
        if not data:
            return str(response or "").strip()
        return self._render_answer_json(data) or str(response or "").strip()

    def search(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        return asyncio.run(
            self.search_async(query, after_date=after_date, location=location)
        )

    def search_many(
        self,
        queries: list[str],
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        return asyncio.run(
            self.search_many_async(queries, after_date=after_date, location=location)
        )

    async def _search_evidence_async(
        self,
        query: str,
        providers: list[str],
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        combined_result = await self.search_providers_async(
            query,
            providers,
            after_date=after_date,
            location=location,
        )
        logger.info(f"\n---------\n{query} combined result: {combined_result}")
        return combined_result

    def _combined_question(self, queries: list[str]) -> str:
        if len(queries) <= 1:
            return queries[0] if queries else ""
        additional = "\n".join(f"- {query}" for query in queries[1:])
        return f"{queries[0]}\n\nAdditional search variants used:\n{additional}"

    async def search_many_async(
        self,
        queries: list[str],
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        cleaned_queries = []
        for query in queries:
            query = str(query).strip()
            if query and query not in cleaned_queries:
                cleaned_queries.append(query)
        if not cleaned_queries:
            return ""
        max_query_variants = _positive_int(
            self.config.get("web_search_max_query_variants", 3),
            3,
        )
        cleaned_queries = cleaned_queries[:max_query_variants]

        logger.debug(f"Searching for {cleaned_queries}")
        start_time = time.time()
        providers = self._get_enabled_providers()

        try:
            evidence_results = await asyncio.gather(
                *[
                    self._search_evidence_async(
                        query,
                        providers,
                        after_date=after_date,
                        location=location,
                    )
                    for query in cleaned_queries
                ]
            )
            combined_result = "\n\n".join(
                f"Query: {query}\n{result}"
                for query, result in zip(cleaned_queries, evidence_results)
                if result
            )
            if not combined_result:
                return ""

            result = combined_result
            prompt = self._build_answer_prompt(
                self._combined_question(cleaned_queries),
                combined_result,
                after_date=after_date,
                location=location,
            )
            try:
                response = self.ai_model.get_response([{"role": "user", "content": prompt}])
                result = self._render_answer_response(response)
            except AuthenticationError as e:
                logger.warning(f"web search summary: Authentication error: {e}, passing raw result")

            duration = time.time() - start_time
            logger.debug(
                "Final search took %.2f seconds; result for queries %s is: %s",
                duration,
                cleaned_queries,
                result,
            )
            return result
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            raise

    async def search_async(
        self,
        query: str,
        after_date: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        logger.debug(f"Searching for {query}")
        start_time = time.time()

        providers = self._get_enabled_providers()

        try:
            combined_result = await self._search_evidence_async(
                query,
                providers,
                after_date=after_date,
                location=location,
            )

            result = combined_result
            prompt = self._build_answer_prompt(
                query,
                combined_result,
                after_date=after_date,
                location=location,
            )
            try:
                response = self.ai_model.get_response([{"role": "user", "content": prompt}])
                result = self._render_answer_response(response)
            except AuthenticationError as e:
                logger.warning(f"web search summary: Authentication error: {e}, passing raw result")

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
        # query = input(">")
        query = "today news"
        result = await web_searcher.search_async(query)
        print(result)
        break


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()

    asyncio.run(loop())
