import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Callable, AsyncGenerator, Awaitable

import aiohttp

from src.ai_models import ClaudeAIModel
from src.config import Config
from src.tools import extract_sentences, retry_async_generator

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
    required: List[str] = field(default_factory=list)

    def __post_init__(self):
        parameter_names = {p.name for p in self.parameters}
        for required in self.required:
            if required not in parameter_names:
                raise ValueError(f'Required field "{required}" does not exist in parameters')


class ClaudeAIModelWithTools(ClaudeAIModel):
    """
    A class representing an AI model with tool-using capabilities.

    This class extends the base ClaudeAIModel to include functionality for using
    various tools during the conversation process.
    """
    def __init__(self, config: Config, tools: List[Tool]) -> None:
        """
        Initialize the ClaudeAIModelWithTools instance.

        Args:
            config (Config): Configuration object for the AI model.
            tools (List[Tool]): List of Tool objects available to the model.
        """
        super().__init__(config)
        self.tools_description = [{'name': t.name,
                                   'description': t.description,
                                   'input_schema': {'type': 'object',
                                                    'properties': {p.name: {'type': p.type,
                                                                            'description': p.description
                                                                            }
                                                                   for p in t.parameters
                                                                   },
                                                    'required': t.required
                                                    }
                                   }
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
                        {'type': 'tool_result', 'content': tool_result, "tool_use_id": tool_use_id}]})
                    response = self.get_response(messages)
                    response_text += response
                pass
        return response_text

    @retry_async_generator()
    async def _get_response_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[dict, None]:
        """
        Asynchronously get responses from the AI model.

        This method sends messages to the AI model and yields the response events.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries to send to the model.

        Yields:
            dict: Response events from the AI model.
        """
        system_message_combined = " ".join([m["content"] for m in messages if m["role"] == "system"])
        non_system_message = [m for m in messages if m["role"] != 'system']

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "tools": self.tools_description,
            "messages": non_system_message,
            "stream": True  # Enable streaming
        }
        if system_message_combined:
            data["system"] = system_message_combined

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=self.headers, json=data) as response:
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                event_data = json.loads(line[6:])
                                yield event_data
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode JSON: {line}")
                                continue

    async def get_response_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Asynchronously process responses from the AI model and yield sentences.

        This method handles the streaming response from the AI model, processes tool
        usage, and yields complete sentences as they become available.

        Args:
           messages (List[Dict[str, str]]): List of message dictionaries to send to the model.

        Yields:
           str: Complete sentences from the AI model's response.
        """
        logger.debug(f"get_response_async: {messages}")
        message_list = [m for m in messages]

        current_text = ""
        current_tool_use = None

        async for event in self._get_response_async(message_list):
            event_type = event.get('type')

            if event_type == 'content_block_delta':
                delta = event.get('delta', {})
                if delta.get('type') == 'text_delta':
                    text = delta.get('text', '')
                    current_text += text
                    sentences = extract_sentences(current_text)

                    # If we have any complete sentences, yield them
                    if len(sentences) > 1:
                        logger.info(f"{len(sentences)} sentences extracted")
                        for sentence in sentences[:-1]:
                            logger.info(f"yield {sentence}")
                            yield sentence

                        # Keep the last (potentially incomplete) sentence
                        current_text = sentences[-1]

                elif delta.get('type') == 'input_json_delta':
                    if current_tool_use is None:
                        current_tool_use = {}
                    current_tool_use['input'] = current_tool_use.get('input', '') + delta.get('partial_json', '')

            elif event_type == 'content_block_start':
                content_block = event.get('content_block', {})
                if content_block.get('type') == 'tool_use':
                    current_tool_use = {
                        'name': content_block.get('name'),
                        'id': content_block.get('id'),
                        'input': ''
                    }

            elif event_type == 'content_block_stop':
                if current_tool_use:
                    try:
                        tool_input = json.loads(current_tool_use['input'])
                        tool_name = current_tool_use['name']
                        tool_use_id = current_tool_use['id']
                        tool_processor = self.tools_processors[tool_name]
                        tool_result = await tool_processor(tool_input)
                        if self.tools[tool_name].iterative:
                            message_list.append({"role": "user", "content": [
                                {'type': 'tool_result', 'content': tool_result, "tool_use_id": tool_use_id}]})
                            async for response in self.get_response_async(message_list):
                                logger.info(f"yield response {response}")
                                yield response
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode tool input JSON: {current_tool_use['input']}")
                    finally:
                        current_tool_use = None

            elif event_type == 'message_stop':
                # Yield any remaining text
                if current_text:
                    logger.info(f"yield current_text: {current_text}")
                    yield current_text

        # Yield any remaining text if the message_stop event wasn't received
        if current_text:
            yield current_text

def main():
    # from aiy.leds import Leds
    #
    # config = Config()
    # with Leds() as leds:
    #     search_tool = WebSearchTool(config, leds)
    #
    #     tools = [Tool(name="internet_search", description="Search Internet", iterative=True,
    #                   parameters=[ToolParameter(name='query', type='string', description='A query to search for')],
    #                   processor=search_tool.do_search_async), ]
    #     model = ClaudeAIModelWithTools(config, tools=tools)
    #     messages = [{"role": "user", "content": "who will play at euro 2024 final?"}]
    #     response = model.get_response(messages)
    #     print(response)
    pass


async def loop():
    # from aiy.leds import Leds
    #
    # config = Config()
    #
    # with Leds() as leds:
    #
    #     search_tool = WebSearchTool(config, leds)
    #
    #     tools = [Tool(name="internet_search", description="Search Internet", iterative=True,
    #                   parameters=[ToolParameter(name='query', type='string', description='A query to search for')],
    #                   processor=search_tool.do_search_async), ]
    #     model = ClaudeAIModelWithTools(config, tools=tools)
    #     message = "Today is July 11, 2024. who will play at euro 2024 final?"
    #     print(message)
    #     messages = [{"role": "user", "content": message}]
    #     async for response_part in model.get_response_async(messages):
    #         messages.append({"role": "assistant", "content": response_part})
    #         print(response_part, flush=True)

    pass


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(loop())

    # main()
