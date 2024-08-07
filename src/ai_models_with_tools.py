import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Callable, AsyncGenerator, Awaitable, Optional
import asyncio
import aiohttp
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from src.ai_models import ClaudeAIModel, OpenAIModel
from src.config import Config
from src.tools import extract_sentences, retry_async_generator, get_token_count

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

    def __init__(self, config: Config, timezone: str = "", tools: List[Tool] = []) -> None:
        """
        Initialize the ClaudeAIModelWithTools instance.

        Args:
            config (Config): Configuration object for the AI model.
            tools (List[Tool]): List of Tool objects available to the model.
        """
        super().__init__(config, timezone=timezone)

        def schema(t: Tool):
            return {'type': 'object',
                    'properties': {p.name: {'type': p.type, 'description': p.description} for p in t.parameters},
                    'required': t.required}

        self.tools_description = [
            {'name': t.name, 'description': t.description, 'input_schema': schema(t), 'required': t.required} for t in
            tools]
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
async def _get_response_async(self, messages: List[Dict[str, str]], streaming=False) -> AsyncGenerator[dict, None]:
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

    data = {"model": self.model, "max_tokens": self.max_tokens, "tools": self.tools_description,
            "messages": non_system_message, "stream": streaming  # Enable streaming
            }
    if system_message_combined:
        data["system"] = system_message_combined

    async with aiohttp.ClientSession() as session:
        async with session.post(self.url, headers=self.headers, json=data) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        line = line[6:]
                    if line.startswith('{'):
                        try:
                            event_data = json.loads(line)
                            yield event_data
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON: {line}")
                            continue


@retry_async_generator()
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
    logger.debug(f"{self._time_str()}AI get_response_async: {messages[-2:]}")
    streaming = self.config.get("llm_streaming", False)
    if streaming:
        async for response in self._get_response_async_streaming(messages):
            logger.debug(f"{self._time_str()}AI response: {response}")
            yield response
    else:
        async for response in self._get_response_async_plain(messages):
            logger.debug(f"{self._time_str()}AI response: {response}")
            yield response


async def _get_response_async_plain(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    message_list = [m for m in messages]

    async for response_dict in self._get_response_async(message_list, streaming=False):
        logger.debug(f"get_response_async: {json.dumps(response_dict, indent=2)}")
        if 'usage' in response_dict:
            logger.debug(f"tokens usage: {response_dict['usage']} vs estimate {get_token_count(message_list)}")
        if 'content' not in response_dict:
            logger.error(f"No content in response: {json.dumps(response_dict, indent=2)}")
            if 'error' in response_dict:
                response_dict['content'] = [{"type": "text", "text": response_dict["error"]["message"]}]
            else:
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
                        {'type': 'tool_result', 'content': tool_result, "tool_use_id": tool_use_id}]})
                    async for response in self.get_response_async(message_list):
                        yield response


async def _get_response_async_streaming(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    message_list = [m for m in messages]

    current_text = ""
    current_tool_use = None
    message_ended = False
    assistant_message = ""

    async for event in self._get_response_async(message_list, streaming=True):
        logger.debug(f"{self._time_str()}Received event: {event}")
        event_type = event.get('type')

        if event_type == 'error':
            logger.error(f"{self._time_str()}Error: {event['error']}")
            raise Exception(f"Claude error: {event['error']['message']}")

        if event_type == 'content_block_delta':
            delta = event.get('delta', {})
            if delta.get('type') == 'text_delta':
                text = delta.get('text', '')
                logger.debug(f"{self._time_str()}Received text delta: {text}")
                current_text += text
                logger.debug(f"{self._time_str()}Current text: {current_text}")
                sentences = extract_sentences(current_text)

                # If we have any complete sentences, yield them
                if len(sentences) > 1:
                    logger.debug(f"{self._time_str()}{len(sentences) - 1} sentences extracted")
                    for sentence in sentences[:-1]:
                        logger.debug(f"{self._time_str()}Yielding sentence: {sentence}")
                        assistant_message += f"{sentence} "
                        yield sentence

                    # Keep the last (potentially incomplete) sentence
                    current_text = sentences[-1]
                    logger.debug(f"{self._time_str()}Remaining text: {current_text}")

            elif delta.get('type') == 'input_json_delta':
                logger.debug(f"{self._time_str()}Received input JSON delta: {delta.get('partial_json', '')}")
                if current_tool_use is None:
                    current_tool_use = {}
                current_tool_use['input'] = current_tool_use.get('input', '') + delta.get('partial_json', '')

        elif event_type == 'content_block_start':
            content_block = event.get('content_block', {})
            if content_block.get('type') == 'tool_use':
                logger.debug(f"{self._time_str()}Tool use started: {content_block}")
                current_tool_use = {'type': 'tool_use', 'name': content_block.get('name'),
                                    'id': content_block.get('id'), 'input': ''}

        elif event_type == 'content_block_stop':
            logger.debug("Content block stopped")
            for sentence in sentences:
                logger.debug(f"{self._time_str()}Yielding sentence: {sentence}")
                assistant_message += f"{sentence} "
                yield sentence
            sentences = []
            current_text = ""
            if current_tool_use:
                try:
                    tool_input = json.loads(current_tool_use['input'])
                    current_tool_use['input'] = tool_input
                    message_list.append({"role": "assistant", "content": [current_tool_use]})
                    tool_name = current_tool_use['name']
                    tool_use_id = current_tool_use['id']
                    logger.debug(f"{self._time_str()}Processing tool use: {tool_name}, {tool_use_id}")
                    tool_processor = self.tools_processors[tool_name]
                    tool_result = await tool_processor(tool_input)
                    logger.debug(f"{self._time_str()}tool result: {json.dumps(tool_result, indent=2)}")
                    if self.tools[tool_name].iterative:
                        message_list.append({"role": "user", "content": [
                            {'type': 'tool_result', 'content': tool_result, "tool_use_id": tool_use_id}]})
                        async for response in self.get_response_async(message_list):
                            logger.debug(f"Yielding after tool response: {response}")
                            yield response
                except json.JSONDecodeError:
                    logger.error(f"{self._time_str()}Failed to decode tool input JSON: {current_tool_use['input']}")
                finally:
                    current_tool_use = None

        elif event_type == 'message_stop':
            logger.debug(f"{self._time_str()}Message stopped")
            message_ended = True

        # If the message has ended and we have any remaining text, yield it
        if message_ended:
            if current_text:
                logger.debug(f"{self._time_str()}Yielding final text: {current_text}")
                assistant_message += f"{current_text} "
                yield current_text
                current_text = ""  # Clear current_text to prevent duplication
            message_list.append({"role": "assistant", "content": assistant_message})
            assistant_message = ""

    # If the message somehow ended without a message_stop event and we still have text, yield it
    if current_text:
        logger.debug(f"{self._time_str()}Yielding remaining text: {current_text}")
        yield current_text


@dataclass
class ToolCall:
    arguments: str
    id: str


class OpenAIModelWithTools(OpenAIModel):
    """
    A class representing an AI model with tool-using capabilities.

    This class extends the base OpenAIModel to include functionality for using
    various tools during the conversation process.
    """

    def __init__(self, config: Config, tools: Optional[List[Tool]] = None) -> None:
        """
        Initialize the ClaudeAIModelWithTools instance.

        Args:
            config (Config): Configuration object for the AI model.
            tools (List[Tool]): List of Tool objects available to the model.
        """
        super().__init__(config)

        def schema(t: Tool):
            return {'type': 'object',
                    'properties': {p.name: {'type': p.type, 'description': p.description} for p in t.parameters},
                    'required': t.required}

        self.tools_processors = {t.name: t.processor for t in tools}
        self.tools = {t.name: t for t in tools}

        self.tools_description = [{"type": "function",
                       "function": {'name': t.name, 'description': t.description, 'parameters': schema(t),
                                    'required': t.required}} for t in tools]

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=messages,
                                                       tools=self.tools_description,
                                                       max_tokens=self.max_tokens)
        choice = response.choices[0]
        if choice.finish_reason != 'tool_calls':
            return choice.message.content.strip()
        else:
            _messages = [m for m in messages]
            _messages.append(choice.message)
            iterate = False

            response_text = ""
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_use_id = tool_call.id
                tool_parameters = json.loads(tool_call.function.arguments)
                tool_processor = self.tools_processors[tool_name]
                tool_result = asyncio.run(tool_processor(tool_parameters))
                if self.tools[tool_name].iterative:
                    _messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_use_id})
                    iterate = True
            if iterate:
                response = self.get_response(_messages)
                response_text += response
            return response_text

    @retry_async_generator()
    async def get_response_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        stream = await self.client_async.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools_description,
            stream=True,
        )
        tool_name = ""
        tools: Dict[str, ToolCall] = {}
        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason is None and choice.delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tool_name = tool_call.function.name or tool_name
                    if tool_name not in tools:
                        tools[tool_name] = ToolCall(arguments="", id=tool_call.id)
                    tools[tool_name].arguments += tool_call.function.arguments
            elif choice.finish_reason == 'tool_calls':
                tool_name = tool_call.function.name or tool_name
                arguments = tools[tool_name].arguments
                tool_use_id = tools[tool_name].id
                tool_parameters = json.loads(arguments)
                tool_processor = self.tools_processors[tool_name]
                tool_result = await tool_processor(tool_parameters)
                _messages = [m for m in messages]
                ass_msg = {'role': 'assistant',
                           'tool_calls': [
                               {'function': {'arguments': arguments, 'name': tool_name},
                                'id': tool_use_id, 'type': 'function'}]}
                _messages.append(ass_msg)
                iterate = False
                if self.tools[tool_name].iterative:
                    _messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_use_id})
                    iterate = True
                if iterate:
                    async for response in self.get_response_async(_messages):
                        yield response
            else:
                content = chunk.choices[0].delta.content
                yield content or ""


def main():
    from src.web_search_tool import WebSearchTool
    config = Config()
    search_tool = WebSearchTool(config)
    model = OpenAIModelWithTools(config, tools=[search_tool.tool_definition()])
    message = "Today is August 6, 2024. who got olympics gold today?"
    print(message)
    messages = [{"role": "user", "content": message}]
    print(model.get_response(messages))


async def main_async():
    from src.web_search_tool import WebSearchTool
    config = Config()
    search_tool = WebSearchTool(config)
    model = ClaudeAIModelWithTools(config, tools=[search_tool.tool_definition()])
    system = """Today is Auguts 6 2024. Now 12:15 PM PDT. In San Jose, California, US. Тебя зовут Кубик. Ты мой друг и помощник. Ты умеешь шутить и быть саркастичным.
    Отвечай естественно, как в устной речи. Говори максимально просто и понятно. Не используй списки и нумерации. Например, не говори 1. что-то; 2.
    что-то. говори во-первых, во-вторых или просто перечисляй. При ответе на вопрос где важно время, помни какое сегодня число. Если чего-то не знаешь,
    так и скажи. Я буду разговаривать с тобой через голосовой интерфейс. Будь краток, избегай банальностей и непрошенных советов."""
    # message = "Посмотри, в прогнозе сегодня жарко будет."
    message = "Today is August 6, 2024. who got olympics gold today?"
    print(message)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
    m = ""
    async for response_part in model.get_response_async(messages):
        print(response_part, flush=True, end="")
        m += response_part
    print()
    # messages.append({"role": "assistant", "content": m})
    # print(messages)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(main_async())
    # main()
