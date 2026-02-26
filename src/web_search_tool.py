import logging
from typing import Dict

from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
from src.web_search import WebSearcher

logger = logging.getLogger(__name__)


class WebSearchTool:
    """

    This class represents a web search tool that can be used to search the internet. It provides both synchronous and asynchronous search methods.

    Class: WebSearchTool

    Methods:
    - tool_definition(self) -> Tool
        Returns the definition of the tool. The Tool object includes the name, description, whether it is iterative, and the parameters required for the search.

    - __init__(self, config: Config)
        Initializes the WebSearchTool object.

    - _start_processing(self)
        (Private method) Starts the processing indicator (e.g. LED blinking).

    - _stop_processing(self)
        (Private method) Stops the processing indicator (e.g. LED off).

    - do_search(self, parameters: Dict[str, any]) -> str
        Performs a synchronous web search using the given parameters. It starts the processing indicator, performs the search using the web_searcher object, stops the processing indicator, and returns the result.

    - do_search_async(self, parameters: Dict[str, any]) -> str
        Performs an asynchronous web search using the given parameters. It starts the processing indicator, performs the asynchronous search using the web_searcher object, stops the processing indicator, and returns the result.

    """

    def tool_definition(self) -> Tool:
        return Tool(
            name="internet_search",
            description="Search Internet for actual information",
            iterative=True,
            programmatic_code_execution_candidate=True,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="A query to search for, preferable in English",
                )
            ],
            required=["query"],
            processor=self.do_search_async,
            rule_instructions={
                "russian": (
                    "Перед поиском в интернете скажи что собираешься поискать. "
                ),
                "english": (
                    "Before searching the internet, say that you are going to search. "
                )
            },
        )

    def __init__(self, config: Config):
        self.web_searcher = WebSearcher(config)
        self.led_processing_color = config.get(
            "processing_color", (0, 1, 0)
        )  # dark green
        self.led_processing_blink_period_ms = config.get(
            "processing_blink_period_ms", 300
        )

    def _start_processing(self):
        pass

    def _stop_processing(self):
        pass

    def do_search(self, parameters: Dict[str, any]) -> str:
        query = str(parameters.get("query", "")).strip() if isinstance(parameters, dict) else ""
        if not query:
            logger.error(f"missing parameter query: {parameters}")
            return (
                "Tool input error for 'internet_search': missing required parameter 'query'. "
                "Ask the user what exactly to search for and retry."
            )
        logger.info(f"searching for {query}")
        self._start_processing()
        result = self.web_searcher.search(query)
        self._stop_processing()
        return result

    async def do_search_async(self, parameters: Dict[str, any]) -> str:
        query = str(parameters.get("query", "")).strip() if isinstance(parameters, dict) else ""
        if not query:
            logger.error(f"missing parameter query: {parameters}")
            return (
                "Tool input error for 'internet_search': missing required parameter 'query'. "
                "Ask the user what exactly to search for and retry."
            )
        logger.info(f"searching async for {query}")
        self._start_processing()
        logger.info(f"searching for {query}")
        result = await self.web_searcher.search_async(query)
        logger.info(f"search result: {result}")
        self._stop_processing()
        return result
