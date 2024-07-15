from typing import List, Dict, Callable, AsyncGenerator, Awaitable
from dataclasses import dataclass
import logging
import asyncio
import json
import aiohttp

from src.ai_models import ClaudeAIModel
from src.config import Config
from src.llm_tools import WebSearchTool
from src.tools import get_token_count

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str


@dataclass
class Tool:
    name: str
    description: str
    iterative: bool
    parameters: List[ToolParameter]
    processor: Callable[[Dict[str, any]], Awaitable[str]]


class ClaudeAIModelWithTools(ClaudeAIModel):
    def __init__(self, config: Config, tools: List[Tool]) -> None:
        super().__init__(config)
        self.tools_description = [{'name': t.name,
                                   'description': t.description,
                                   'input_schema': {'type': 'object',
                                                    'properties': {
                                                        p.name: {
                                                            'type': p.type,
                                                            'description': p.description}
                                                        for p in
                                                        t.parameters}}}
                                  for t in tools]
        self.tools_processors = {t.name: t.processor for t in tools}
        self.tools = {t.name: t for t in tools}

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        response_dict = self._get_response(messages)
        response_text = ""
        messages.append({"role": "assistant", "content": response_dict['content']})
        for content in response_dict['content']:
            if content['type'] == 'text':
                response_text += content['text']

            elif content['type'] == 'tool_use':
                tool_name = content['name']
                tool_use_id = content['id']
                tool_parameters = content['input']
                tool_processor = self.tools_processors[tool_name]
                tool_result = tool_processor(tool_parameters)
                if self.tools[tool_name].iterative:
                    messages.append({"role": "user", "content": [
                        {'type': 'tool_result',  'content': tool_result, "tool_use_id": tool_use_id}]})
                    response = self.get_response(messages)
                    response_text += response
                pass
        return response_text

    async def _get_response_async(self, messages: List[Dict[str, str]]) -> dict:
        system_message_combined = " ".join([m["content"] for m in messages if m["role"] == "system"])
        non_system_message = [m for m in messages if m["role"] != 'system']

        data = {"model": self.model, "max_tokens": self.max_tokens, "tools": self.tools_description,
                "messages": non_system_message}
        if system_message_combined:
            data["system"] = system_message_combined

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=self.headers, json=data) as response:
                res = await response.text()
                return json.loads(res)

    async def get_response_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        logger.debug(f"get_response_async: {messages}")
        message_list = [m for m in messages]

        response_dict = await self._get_response_async(message_list)
        logger.debug(f"get_response_async: {json.dumps(response_dict, indent=2)}")
        if 'usage' in response_dict:
            logger.info(f"tokens usage: {response_dict['usage']} vs estimate {get_token_count(message_list)}")
        if 'content' not in response_dict:
            logger.error(f"No content in response: {json.dumps(response_dict, indent=2)}")
            return
        message_list.append({"role": "assistant", "content": response_dict['content']})
        for content in response_dict['content']:
            if content['type'] == 'text':
                yield content['text']

            elif content['type'] == 'tool_use':
                tool_name = content['name']
                tool_use_id = content['id']
                tool_parameters = content['input']
                tool_processor = self.tools_processors[tool_name]
                tool_result = await tool_processor(tool_parameters)
                if self.tools[tool_name].iterative:
                    message_list.append({"role": "user", "content": [
                        {'type': 'tool_result',  'content': tool_result, "tool_use_id": tool_use_id}]})
                    async for response in self.get_response_async(message_list):
                        yield response


def main():
    from aiy.leds import Leds

    config = Config()
    with Leds() as leds:
        search_tool = WebSearchTool(config, leds)

        tools = [Tool(name="internet_search", description="Search Internet", iterative=True,
                      parameters=[ToolParameter(name='query', type='string', description='A query to search for')],
                      processor=search_tool.do_search_async), ]
        model = ClaudeAIModelWithTools(config, tools=tools)
        messages = [{"role": "user", "content": "who will play at euro 2024 final?"}]
        response = model.get_response(messages)
        print(response)


async def loop():
    from aiy.leds import Leds

    config = Config()

    with Leds() as leds:

        search_tool = WebSearchTool(config, leds)

        tools = [Tool(name="internet_search", description="Search Internet", iterative=True,
                      parameters=[ToolParameter(name='query', type='string', description='A query to search for')],
                      processor=search_tool.do_search_async), ]
        model = ClaudeAIModelWithTools(config, tools=tools)
        message = "Today is July 11, 2024. who will play at euro 2024 final?"
        print(message)
        messages = [{"role": "user", "content": message}]
        async for response_part in model.get_response_async(messages):
            messages.append({"role": "assistant", "content": response_part})
            print(response_part, flush=True)

        pass


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(loop())

    # main()
