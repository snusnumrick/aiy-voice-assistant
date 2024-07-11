from src.config import Config
from src.web_search import WebSearcher
import logging
from typing import Dict

logger = logging.getLogger(__name__)
config = Config()

class WebSearchTool:
    def __init__(self):
        self.web_searcher = WebSearcher(config)

    def do_search(self, parameters: Dict[str, any]) -> str:
        if 'query' in parameters:
            return self.web_searcher.search(parameters['query'])
        logger.error(f"missing parameter query: {parameters}")
        return ""

    async def do_search_async(self, parameters: Dict[str, any]) -> str:
        if 'query' in parameters:
            return await self.web_searcher.search_async(parameters['query'])
        logger.error(f"missing parameter query: {parameters}")
        return ""
