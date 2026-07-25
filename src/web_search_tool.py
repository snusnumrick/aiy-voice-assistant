import logging
from typing import Any

from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
from src.web_search import WebSearcher

logger = logging.getLogger(__name__)


def _queries_from_parameters(parameters: dict[str, Any]) -> list[str]:
    queries = []
    query = str(parameters.get("query", "")).strip()
    if query:
        queries.append(query)

    additional_queries = parameters.get("additional_queries")
    if additional_queries:
        if isinstance(additional_queries, list):
            candidates = additional_queries
        else:
            candidates = str(additional_queries).replace("\n", ",").split(",")
        for candidate in candidates:
            candidate = str(candidate).strip()
            if candidate and candidate not in queries:
                queries.append(candidate)
    return queries


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

    - do_search_async(self, parameters: Dict[str, any]) -> str
        Performs an asynchronous web search using the given parameters. It starts the processing indicator, performs the asynchronous search using the web_searcher object, stops the processing indicator, and returns the result.

    """

    def tool_definitions(self) -> list[Tool]:
        return [
            self.tool_definition(),
            Tool(
                name="list_web_search_reports",
                description="Lists saved web search result reports with filenames. Use this to find previous internet_search results before repeating a similar search.",
                iterative=True,
                parameters=[],
                required=[],
                processor=self.web_searcher.list_search_reports_async,
            ),
            Tool(
                name="get_web_search_report",
                description="Retrieves the full markdown content of a saved web search result report by filename.",
                iterative=True,
                programmatic_code_execution_candidate=False,
                parameters=[
                    ToolParameter(
                        name="filename",
                        type="string",
                        description="The filename of the saved web search report to retrieve",
                    )
                ],
                required=["filename"],
                processor=self.web_searcher.get_search_report_async,
            ),
        ]

    def tool_definition(self) -> Tool:
        return Tool(
            name="internet_search",
            description="Search Internet for actual information",
            iterative=True,
            programmatic_code_execution_candidate=False,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="A query to search for. Use the local-language wording most likely to return relevant results.",
                ),
                ToolParameter(
                    name="additional_queries",
                    type="string",
                    description="Optional comma-separated additional search queries to run and combine with the main query, for local-language or venue-specific variants.",
                ),
                # ToolParameter(
                #     name="after_date",
                #     type="string",
                #     description="Optional freshness filter as YYYY-MM-DD. Use when the user asks for recent results after a specific date.",
                # ),
                ToolParameter(
                    name="location",
                    type="string",
                    description="Optional country or place for localized search, for example 'us' or 'San Francisco, CA, US'.",
                ),
            ],
            required=["query"],
            processor=self.do_search_async,
            rule_instructions={
                "russian": (
                    "Перед поиском в интернете скажи что собираешься поискать и оформи эту фразу "
                    "как $tool_filler: Сейчас поищу$. "
                    "Если нужен прошлый результат, сначала проверь list_web_search_reports."
                ),
                "english": (
                    "Before searching the internet, say that you are going to search and format "
                    "that phrase as $tool_filler: Let me search$. "
                    "If a previous result may answer the request, check list_web_search_reports first."
                ),
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

    async def do_search_async(self, parameters: dict[str, Any]) -> str:
        logger.info(f"searching async for {parameters['query']}")
        if "query" in parameters:
            self._start_processing()
            logger.info(f"searching for {parameters['query']}")
            result = await self.web_searcher.search_many_async(
                _queries_from_parameters(parameters),
                after_date=parameters.get("after_date"),
                location=parameters.get("location"),
            )
            logger.info(f"search result: {result}")
            self._stop_processing()
            return result
        logger.error(f"missing  parameter  query:  {parameters}")
        return ""
