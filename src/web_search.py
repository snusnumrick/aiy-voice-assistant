import json
import logging
import os

import requests

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


class Tavily:
    def __init__(self):
        from tavily import TavilyClient

        api_key = os.environ.get('TAVILY_API_KEY')
        if not api_key:
            raise ValueError("Tavily API key is not provided in environment variables")
        self.tavily = TavilyClient(api_key="YOUR_API_KEY")

    def search(self, query: str):
        general = self.tavily.qna_search(query=query, topic = "general")
        news = self.tavily.qna_search(query=query, topic = "news")
        return f"general: {general}; news: {news}"


class WebSearcher:
    def __init__(self, config):
        self.tavily = Tavily()
        self.google = google_web_search
        self.ai_model = OpenRouterModel(config)

    def search(self, query: str) -> str:
        try:
            # combined_result = self.tavily.search(query) + "\n\n" + self.google(query, "en")
            # logger.info(f"Web search result for query '{query}' is: {combined_result}")
            # prompt = f"based on result from internet search below, what is the answer to the question: {query}\n\n{combined_result}"
            # result = self.ai_model.get_response([{"role": "user", "content": prompt}])

            result = self.tavily.search(query)

            logger.info(f"Web search result for query '{query}' is: {result}")
            return result

        except Exception as e:
            print(f"Error performing web search: {e}")
            raise
