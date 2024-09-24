import asyncio
import json
import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Coroutine

import httpx
import requests
from duckduckgo_search import DDGS
from lxml import html

if __name__ == "__main__":
    # add current directory to python path
    sys.path.append(os.getcwd())

from src.config import Config
from src.ai_models import OpenRouterModel

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36", ]


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
        return " ".join(element.text_content() for element in elements) if elements else ""

    async def search(self, term: str) -> str:
        start_time = time.time()
        tree = await self._fetch_data(term, "en")
        logger.debug(f"Google search fetch_data time: {time.time() - start_time}")

        selectors = [".sXLaOe", ".hgKElc", ".wx62f", ".HwtpBd", ".yxjZuf span"]
        results = [self._get_text(tree, selector) for selector in selectors]
        await asyncio.sleep(0)

        a1 = self._get_text(tree, ".UDZeY span").replace("Описание", "").replace("ЕЩЁ", "")
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
        self.cs_key = os.environ.get('GOOGLE_CUSTOMSEARCH_KEY')
        self.api_key = os.environ.get('GOOGLE_API_KEY')

    async def search(self, term: str) -> str:
        start_time = time.time()
        params = {'q': term, 'key': self.api_key, 'cx': self.cs_key, }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes

            results = response.json()

            # Process and return the results
            text_result = "\n".join([item['snippet'] for item in results['items']]) if 'items' in results else ""
            logger.debug(f"Google Custom Search search took {time.time() - start_time} seconds")
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
        messages = [{"role": "user", "content": (query), }, ]
        response = "".join([r async for r in self.model.get_response_async(messages)])
        duration = time.time() - start_time
        logger.debug(f"Perplexity search took {duration:.2f} seconds")
        return response


class Tavily(SearchProvider):
    def __init__(self, config: Config):
        self.api_key = os.environ.get('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("Tavily API key is not provided in environment variables")

    async def search(self, query: str):
        url = "https://api.tavily.com/search"

        start_time = time.time()

        # Prepare the request payload
        payload = {"api_key": self.api_key, "query": query, "include_answer": True, "search_depth": "advanced",
                   "topic": "news"}

        try:
            # Make the POST request to the API
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
            if response.status_code == 400:
                try:
                    logger.error(f"Tavily: {json.loads(response.content)['detail']['error']}")
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
    def _get_url(self, method: str, url: str, **kwargs) -> Optional[httpx._models.Response]:
        resp = self._client.request(method, url, follow_redirects=True, **kwargs)
        if resp.status_code == 202:
            # try again in a few seconds if there is no answer yet
            if 'Location' in resp.headers:
                status_update_url = resp.headers['Location']
            # If the URL is in the response body, you'll need to parse the body to find it
            # This is just a generic example, adjust it according to your API documentation
            elif hasattr(resp, 'status_url'):
                status_update_url = resp.status_url
            else:
                status_update_url = resp.url
            if status_update_url:
                time.sleep(3)
                resp = self._client.request('GET', status_update_url, follow_redirects=True)
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
                {"title": r["title"], "url": r["href"], **({"exerpt": r["body"]} if r.get("body") else {}), } for r in
                self.ddgs.text(query)]
        except Exception as e:
            logger.error(f"DDGS dsearch failed: {e}")
            return ""

        await asyncio.sleep(0)

        results = "## Search results\n" + "\n\n".join(
            "### \"{}\"\n**URL:** {}  \n**Excerpt:** {}".format(r['title'], r['url'],
                "\"{}\"".format(r.get("exerpt")) if r.get("exerpt") else "N/A") for r in search_results)

        # make it safe
        results = results.encode("utf-8", "ignore").decode("utf-8")
        duration = time.time() - start_time
        logger.debug(f"DDGS search took {duration:.2f} seconds")
        return results


class WebSearcher:
    def __init__(self, config):
        self.tavily = Tavily(config)
        self.google = Google(config)
        self.google_cs = GoogleCustomSearch(config)
        self.ai_model = OpenRouterModel(config, use_simple_model=True)
        self.perplexity = Perplexity(config)
        self.ddgs = DuckDuckGoSearch(config)
        self.config = config

    async def search_providers_async(self, query: str, enabled_providers):

        tasks = [getattr(self, provider).search(query) for provider in enabled_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined_result = ""
        for provider, result in zip(enabled_providers, results):
            if isinstance(result, Exception):
                logger.error(f"Error while searching with provider {provider}: {str(result)}")
            else:
                logger.info(f"\n---------\n{provider} result: {result}")
                combined_result += f"Result from {provider}: \n{result}\n"

        return combined_result

    async def search_async(self, query: str) -> str:
        start_time = time.time()

        providers = ["google", "google", "tavily", "perplexity"]

        try:
            combined_result = await self.search_providers_async(query, providers)

            # logger.debug(f"\n---------\n{query} result: {combined_result}")
            #
            # prompt = (f"Answer short. Based on result from internet search below, what is the answer to the question: "
            #           f"{query}\n\n{combined_result}")
            # result = self.ai_model.get_response([{"role": "user", "content": prompt}])
            result = combined_result

            duration = time.time() - start_time
            logger.info(f"Final search took {duration:.2f} seconds; result for query '{query}' is: {result}")
            return result

        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            raise


async def loop():
    config = Config()
    web_searcher = WebSearcher(config)
    ai_model = OpenRouterModel(config, use_simple_model=True)
    while True:
        query = input(">")
        result = await web_searcher.search_async(query)

        prompt = (f"Answer short. Based on result from internet search below, what is the answer to the question: "
                  f"{query}\n\n{result}")
        result = ai_model.get_response([{"role": "user", "content": prompt}])
        print(result)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()

    asyncio.run(loop())
