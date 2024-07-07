import json
import logging
import os
import requests
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

if __name__ == "__main__":
    from config import Config
    from ai_models import OpenRouterModel
else:
    from src.config import Config
    from src.ai_models import OpenRouterModel

logger = logging.getLogger(__name__)


class SearchProvider(ABC):
    """
    Abstract base class for internet search.
    """

    @abstractmethod
    def search(self, query: str) -> str:
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
        pass

    def search(self, term: str) -> str:
        def fetch_data(term, lang):
            from bs4 import BeautifulSoup
            import random

            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36", ]

            url = f"https://www.google.com/search?q={requests.utils.quote(term)}&hl={lang}"
            headers = {"User-Agent": random.choice(user_agents)}
            response = requests.get(url, headers=headers)
            return BeautifulSoup(response.text, 'html.parser')

        soup = fetch_data(term, "en")

        brief = " ".join([element.text for element in soup.select(".sXLaOe")])
        extract = " ".join([element.text for element in soup.select(".hgKElc")])
        denotion = " ".join([element.text for element in soup.select(".wx62f")])
        place = " ".join([element.text for element in soup.select(".HwtpBd")])
        wiki = " ".join([element.text for element in soup.select(".yxjZuf span")])

        a1 = (" ".join([element.text for element in soup.select(".UDZeY span")]).replace("Описание", "").replace("ЕЩЁ",
                                                                                                                 "") + soup.select_one(
            ".LGOjhe span").text) if soup.select_one(".LGOjhe span") else ""
        a2 = " ".join([element.text for element in soup.select(".yXK7lf span")])

        brief_result = "; ".join(filter(None, [brief, extract, denotion, place, wiki]))
        result = brief_result or a2 or a1

        return result


class Perplexity(SearchProvider):
    def __init__(self, config: Config):
        if __name__ == "__main__":
            from ai_models import PerplexityModel
        else:
            from src.ai_models import PerplexityModel
        self.model = PerplexityModel(config)

    def search(self, query: str) -> str:
        messages = [
            {
                "role": "user",
                "content": (query),
            },
        ]
        return self.model.get_response(messages)


class Tavily(SearchProvider):
    def __init__(self, config: Config):
        self.api_key = os.environ.get('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("Tavily API key is not provided in environment variables")

    def search(self, query: str):
        url = "https://api.tavily.com/search"

        # Prepare the request payload
        payload = {"api_key": self.api_key, "query": query, "include_answer": True, "search_depth": "advanced", "topic": "news"}

        try:
            # Make the POST request to the API
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Parse and return the JSON response
            return response.json()["answer"]

        except requests.exceptions.RequestException as e:
            print(f"Error making request to Tavily API: {e}")
            raise

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            raise


class DuckDuckGoSearch(SearchProvider):
    def __init__(self, config):
        try:
            self.duckduckgo_max_attempts = config.get("duckduckgo_max_attempts", 5)
        except Exception as e:
            logger.error(f"DDGS search could not be initialized: {e}")

    def search(self, query: str):
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException, TimeoutException
        ddgs = DDGS()

        try:
            search_results = []
            attempts = 0

            while attempts < self.duckduckgo_max_attempts:
                if not query:
                    return json.dumps(search_results)

                search_results = ddgs.text(query)

                if search_results:
                    break

                time.sleep(1)
                attempts += 1

            if attempts > 0:
                logger.warning(f"DuckDuckGoSearch failed {attempts} times")
            if not search_results:
                logger.error(f"DDGS No results found for query {query}")

            search_results = [
                {
                    "title": r["title"],
                    "url": r["href"],
                    **({"exerpt": r["body"]} if r.get("body") else {}),
                }
                for r in search_results
            ]

            results = "## Search results\n" + "\n\n".join(
                "### \"{}\"\n**URL:** {}  \n**Excerpt:** {}".format(
                    r['title'],
                    r['url'],
                    "\"{}\"".format(r.get("exerpt")) if r.get("exerpt") else "N/A"
                )
                for r in search_results
            )

            # make it safe
            results = results.encode("utf-8", "ignore").decode("utf-8")

            return results

        except RatelimitException as e:
            logger.error(f"rate limit exceeded during DDGS search: {e}")
            return ""
        except TimeoutException as e:
            logger.error(f"timeout during DDGS search: {e}")
            return ""
        except DuckDuckGoSearchException as e:
            logger.error(f"DDGS search failed {e}")
            return ""
        except Exception as e:
            logger.error(f"exception during DDGS search: {e}")
            return ""


class WebSearcher:
    def __init__(self, config):
        self.tavily = Tavily(config)
        self.google = Google(config)
        self.ai_model = OpenRouterModel(config, use_simple_model=True)
        self.perplexity = Perplexity(config)
        self.ddgs = DuckDuckGoSearch(config)
        self.config = config

    async def search_async(self, query: str):
        providers = ["perplexity", "tavily", "google", "ddgs"]
        provider_defaults = [False, False, False, True]
        enabled_providers = [prov for (prov, default) in zip(providers, provider_defaults) if
                             self.config.get(f"use_{prov}", default)]

        def search_provider(provider, query):
            return getattr(self, provider).search(query)

        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(executor, search_provider, prov, query) for prov in enabled_providers]
            results = await asyncio.gather(*tasks)

        combined_result = ""
        joined_results = zip(enabled_providers,
                             ["Result from {}: \n{}\n".format(prov, res) for prov, res in zip(enabled_providers, results)])

        for provider, result in joined_results:
            logger.info(f"\n---------\n{provider} result: {result}")
            combined_result += result

        return combined_result

    def search(self, query: str) -> str:
        # return self.perplexity.search(query)

        try:
            loop = asyncio.get_event_loop()

            # use the loop to run your function
            combined_result = loop.run_until_complete(self.search_async(query))
            logger.info(f"combined result: {combined_result}")

            prompt = (f"Answer short. Based on result from internet search below, what is the answer to the question: "
                      f"{query}\n\n{combined_result}")
            result = self.ai_model.get_response([{"role": "user", "content": prompt}])

            logger.info(f"Final search result for query '{query}' is: {result}")
            return result

        except Exception as e:
            print(f"Error performing web search: {e}")
            raise


def main():
    if __name__ == "__main__":
        config = Config()
        web_searcher = WebSearcher(config)
        query = "Euro 2024 semifinal dates"
        result = web_searcher.search(query)
        logger.info(f"Web search result for query '{query}' is: {result}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    main()
