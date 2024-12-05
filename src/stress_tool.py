from typing import Dict
from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
import aiohttp
import logging

logger = logging.getLogger(__name__)


def _convert_stress_format(word: str) -> str:
    # Replace U+0301 with '+' before the vowel
    # vowels = 'аеёиоуыэюя'
    result = ""
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i + 1] == "\u0301":
            result += "+" + word[i]
            i += 2
        else:
            result += word[i]
            i += 1
    return result


class StressTool:
    """
    This class represents a stress tool that can be used to add stress marks to Russian words.
    It provides both synchronous and asynchronous methods.

    The stress is marked with a '+' symbol placed before the stressed vowel.

    The implementation uses https://morpher.ru/ws3/#addstressmarks

    Class: StressTool

    Methods:
    - tool_definition(self) -> Tool
        Returns the definition of the tool. The Tool object includes the name, description,
        whether it is iterative, and the parameters required for the stress marking.

    - __init__(self, config: Config)
        Initializes the StressTool object.

    - _start_processing(self)
        (Private method) Starts the processing indicator.

    - _stop_processing(self)
        (Private method) Stops the processing indicator.

    - do_stress(self, parameters: Dict[str, any]) -> List[str]
        Performs a synchronous stress marking using the given parameters.

    - do_stress_async(self, parameters: Dict[str, any]) -> List[str]
        Performs an asynchronous stress marking using the given parameters.
    """

    def tool_definition(self) -> Tool:
        return Tool(
            name="stress_marker",
            description="Add stress mark to Russian word. Return list of possible stress variations for given word with stress marking. ",
            iterative=True,
            parameters=[
                ToolParameter(
                    name="word",
                    type="string",
                    description="A Russian word to add stress marks to",
                )
            ],
            required=["word"],
            processor=self.do_stress_async,
        )

    def __init__(self, config: Config):
        self.morpher_url = "https://ws3.morpher.ru/russian/addstressmarks"
        self.headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "application/json",
        }

    def _start_processing(self):
        pass

    def _stop_processing(self):
        pass

    def do_stress(self, parameters: Dict[str, any]) -> str:
        if "word" in parameters:
            self._start_processing()
            import requests

            response = requests.post(
                self.morpher_url,
                headers=self.headers,
                data=parameters["word"].encode("utf-8"),
            )
            self._stop_processing()
            if response.status_code == 200:
                return _convert_stress_format(response.json())
            else:
                logger.error(f"Error: {response.status_code}, {response.text}")
                return []
        logger.error(f"missing parameter word: {parameters}")
        return []

    async def do_stress_async(self, parameters: Dict[str, any]) -> str:
        if "word" in parameters:
            self._start_processing()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.morpher_url,
                    headers=self.headers,
                    data=parameters["word"].encode("utf-8"),
                ) as response:
                    self._stop_processing()
                    if response.status == 200:
                        return _convert_stress_format(await response.json())
                    else:
                        text = await response.text()
                        logger.error(f"Error: {response.status}, {text}")
                        return []
        logger.error(f"missing parameter word: {parameters}")
        return []
