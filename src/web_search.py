import json
import logging
import os
import requests
import time

from src.config import Config

from src.ai_models import OpenRouterModel

logger = logging.getLogger(__name__)


def google_web_search(term, lang) -> str:
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

    soup = fetch_data(term, lang)

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


class Perplexity:
    def __init__(self, config: Config):
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


class Tavily:
    def __init__(self):
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


class DuckDuckGoSearch:
    def __init__(self, config):
        from duckduckgo_search import DDGS
        self.ddgs = DDGS()
        self.duckduckgo_max_attempts = config.get("duckduckgo_max_attempts", 5)
        self.duckduckgo_num_results = config.get("duckduckgo_num_results", 8)

    def search(self, query: str):
        search_results = []
        attempts = 0

        while attempts < self.duckduckgo_max_attempts:
            if not query:
                return json.dumps(search_results)

            search_results = self.ddgs.text(query, max_results=self.duckduckgo_num_results)

            if search_results:
                break

            time.sleep(1)
            attempts += 1

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


class WebSearcher:
    def __init__(self, config):
        self.tavily = Tavily()
        self.google = google_web_search
        self.ai_model = OpenRouterModel(config)
        self.perplexity = Perplexity(config)
        self.ddgs = DuckDuckGoSearch(config)

    def search(self, query: str) -> str:
        # return self.perplexity.search(query)

        try:
            perplexity_result = self.perplexity.search(query)
            logger.info(f"Perplexity result: {perplexity_result}")
            tavily_result = self.tavily.search(query)
            logger.info(f"Tavily result: {tavily_result}")
            google_result = self.google(query, "en")
            logger.info(f"Google result: {google_result}")
            ddgs_result = self.ddgs.search(query)
            logger.info(f"DDGS result: {ddgs_result}")
            combined_result = perplexity_result +"\n\n" + tavily_result + "\n\n" + google_result + "\n\n" + ddgs_result
            logger.info(f"combined search result for query '{query}' is: {combined_result}")

            prompt = f"Answer short. Based on result from internet search below, what is the answer to the question: {query}\n\n{combined_result}"
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
        query = "top attractions in Rybinsk Russia"
        result = web_searcher.search(query)
        logger.info(f"Web search result for query '{query}' is: {result}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    main()
