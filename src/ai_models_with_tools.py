from typing import List, Dict, Callable, AsyncGenerator
from dataclasses import dataclass
import logging
import asyncio
import json
import aiohttp

from src.ai_models import ClaudeAIModel
from src.config import Config
from src.llm_tools import WebSearchTool

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
    processor: Callable[[Dict[str, any]], str]


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
        responce_text = ""
        messages.append({"role": "assistant", "content": response_dict['content']})
        for content in response_dict['content']:
            if content['type'] == 'text':
                responce_text += content['text']

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
                    responce_text += response
                pass
        return responce_text

    async def _get_response_async(self, messages: List[Dict[str, str]]) -> dict:
        data = {"model": self.model, "max_tokens": self.max_tokens, "tools": self.tools_description,
                "system": messages[0]["content"], "messages": messages[1:]}

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=self.headers, json=data) as response:
                res = await response.text()
                return json.loads(res)

    async def get_response_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        logger.info(f"get_response_async: {messages}")

        response_dict = await self._get_response_async(messages)
        logger.info(f"get_response_async: {response_dict}")
        messages.append({"role": "assistant", "content": response_dict['content']})
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
                    messages.append({"role": "user", "content": [
                        {'type': 'tool_result',  'content': tool_result, "tool_use_id": tool_use_id}]})
                    async for response in self.get_response_async(messages):
                        yield response


def main():
    config = Config()

    search_tool = WebSearchTool()

    tools = [Tool(name="internet_search", description="Search Internet", iterative=True,
                  parameters=[ToolParameter(name='query', type='string', description='A query to search for')],
                  processor=search_tool.do_search), ]
    model = ClaudeAIModelWithTools(config, tools=tools)
    messages = [{"role": "user", "content": "who will play at euro 2024 final?"}]
    responce = model.get_response(messages)
    print(responce)


async def loop():
    config = Config()

    search_tool = WebSearchTool()

    tools = [Tool(name="internet_search", description="Search Internet", iterative=True,
                  parameters=[ToolParameter(name='query', type='string', description='A query to search for')],
                  processor=search_tool.do_search_async), ]
    model = ClaudeAIModelWithTools(config, tools=tools)
    messages = [{"role": "user", "content": "Today is July 11, 2024. who will play at euro 2024 final?"}]
    async for response_part in model.get_response_async(messages):
        print(response_part, end='', flush=True)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(loop())

    # main()
