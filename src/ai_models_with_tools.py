"""
AI Models with Tools module.

This module extends the base AI models to include functionality for using
various tools during the conversation process. It provides classes for
Claude AI and OpenAI models with tool-using capabilities.
"""

import asyncio
import json
import logging
import os
import sys
from typing import List, Dict, Callable, AsyncGenerator, Optional, Coroutine

import aiohttp
from pydantic import BaseModel, Field

from src.ai_models import ClaudeAIModel, OpenAIModel, MessageList, normalize_messages, GeminiAIModel
from src.config import Config
from src.tools import extract_sentences, yield_complete_sentences, retry_async_generator, get_token_count

logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """Represents a parameter for a tool."""
    name: str
    type: str
    description: str


class Tool(BaseModel):
    """Represents a tool that can be used by the AI model."""
    name: str
    description: str
    iterative: bool
    parameters: List[ToolParameter]
    processor: Callable[[Dict[str, any]], Coroutine[any, any, str]]
    required: List[str] = Field(default_factory=list)

    def __post_init__(self):
        """Validate that all required fields exist in parameters."""
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

    def __init__(self, config: Config, timezone: str = "", tools: Optional[List[Tool]] = None) -> None:
        """
        Initialize the ClaudeAIModelWithTools instance.

        Args:
            config (Config): Configuration object for the AI model.
            timezone (str): Timezone string for timestamp generation.
            tools (Optional[List[Tool]]): List of Tool objects available to the model.
        """
        super().__init__(config, timezone=timezone)
        self.tools = {t.name: t for t in tools} if tools else {}
        self.tools_description = self._create_tools_description(tools) if tools else []
        self.tools_processors = {t.name: t.processor for t in tools} if tools else {}

    @staticmethod
    def _create_tools_description(tools: List[Tool]) -> List[Dict]:
        """
        Create a list of tool descriptions for the AI model.

        Args:
            tools (List[Tool]): List of Tool objects.

        Returns:
            List[Dict]: List of tool descriptions.
        """
        return [{'name': t.name, 'description': t.description, 'input_schema': {'type': 'object', 'properties': {
            p.name: {'type': p.type, 'description': p.description} for p in t.parameters}, 'required': t.required}} for
                t in tools]

    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response using the AI model with tool capabilities.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        messages = normalize_messages(messages)

        response_dict = self._get_response(messages)
        response_text = ""
        messages.append({"role": "assistant", "content": response_dict['content']})
        for content in response_dict['content']:
            if content['type'] == 'text':
                response_text += content['text']
            elif content['type'] == 'tool_use':
                response_text += self._process_tool_use(content, messages)
        return response_text

    def _process_tool_use(self, content: Dict, messages: List[Dict[str, str]]) -> str:
        """Process a tool use request and generate a response."""
        tool_name = content['name']
        tool_use_id = content['id']
        tool_parameters = content['input']
        tool_processor = self.tools_processors[tool_name]
        tool_result = asyncio.run(tool_processor(tool_parameters))
        if self.tools[tool_name].iterative:
            messages.append({"role": "user",
                             "content": [{'type': 'tool_result', 'content': tool_result, "tool_use_id": tool_use_id}]})
            return self.get_response(messages)
        return ""

    @retry_async_generator()
    async def _get_response_async(self, messages: List[Dict[str, str]], streaming=False) -> AsyncGenerator[dict, None]:
        """
        Asynchronously get responses from the AI model.

        This method sends messages to the AI model and yields the response events.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries to send to the model.
            streaming (bool): Whether to use streaming mode for the response.

        Yields:
            dict: Response events from the AI model.
        """
        system_message_combined = " ".join([m["content"] for m in messages if m["role"] == "system"])
        non_system_message = [m for m in messages if m["role"] != 'system']

        data = {"model": self.model, "max_tokens": self.max_tokens, "tools": self.tools_description,
                "messages": non_system_message, "stream": streaming}
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
    async def get_response_async(self, messages: MessageList) -> AsyncGenerator[str, None]:
        """
        Asynchronously process responses from the AI model and yield sentences.

        This method handles the streaming response from the AI model, processes tool
        usage, and yields complete sentences as they become available.

        Args:
           messages (MessageList): List of message dictionaries to send to the model.

        Yields:
           str: Complete sentences from the AI model's response.

        Raises:
            Exception: If there's an error in processing the AI response.
        """
        messages = normalize_messages(messages)
        logger.debug(f"{self._time_str()}AI get_response_async: {messages[-2:]}")

        streaming = self.config.get("llm_streaming", False)
        try:
            if streaming:
                async for response in self._get_response_async_streaming(messages):
                    logger.debug(f"{self._time_str()}AI response: {response}")
                    yield response
            else:
                async for response in self._get_response_async_plain(messages):
                    logger.debug(f"{self._time_str()}AI response: {response}")
                    yield response
        except Exception as e:
            logger.error(f"Error in get_response_async: {str(e)}")
            raise

    async def _get_response_async_plain(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Generate a plain (non-streaming) response asynchronously."""
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
                    yield await self._process_tool_use_async(content, message_list)

    async def _process_tool_use_async(self, content: Dict, message_list: List[Dict[str, str]]) -> AsyncGenerator[
        str, None]:
        """Process a tool use request asynchronously and generate a response."""
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
        """
        Generate a streaming response asynchronously.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.

        Yields:
            str: Sentences from the AI model's response.

        Raises:
            Exception: If there's an error in processing the AI response.
        """
        message_list = messages.copy()
        current_text = ""
        current_tool_use = None
        assistant_message = ""

        async def process_content_block_delta(event: Dict, current_tool_use: Optional[Dict] = None) -> \
                AsyncGenerator[str, None]:
            """
            Process a content block delta event.

            Args:
                event (Dict): The event data.
                current_tool_use (Optional[Dict]): The current tool use data, if any.

            Yields:
                str: Complete sentences extracted from the current text.
            """
            nonlocal current_text

            delta = event.get('delta', {})
            if delta.get('type') == 'text_delta':
                text = delta.get('text', '')
                current_text += text
                sentences = extract_sentences(current_text)

                if len(sentences) > 1:
                    for sentence in sentences[:-1]:
                        yield sentence
                    current_text = sentences[-1]

            elif delta.get('type') == 'input_json_delta':
                if current_tool_use is not None:
                    current_tool_use['input'] = current_tool_use.get('input', '') + delta.get('partial_json', '')

        def process_content_block_start(event: Dict) -> Optional[Dict]:
            """
            Process a content block start event.

            Args:
                event (Dict): The event data.

            Returns:
                Optional[Dict]: The tool use data if it's a tool use event, None otherwise.
            """
            content_block = event.get('content_block', {})
            if content_block.get('type') == 'tool_use':
                return {'type': 'tool_use', 'name': content_block.get('name'), 'id': content_block.get('id'),
                        'input': ''}
            return None

        async def process_content_block_stop(current_tool_use: Optional[Dict], current_text: str,
                                             message_list: List[Dict[str, str]]) \
                -> AsyncGenerator[str, None]:
            """
            Process a content block stop event.

            Args:
                current_tool_use (Optional[Dict]): The current tool use data, if any.
                current_text (str): The current text.
                message_list (List[Dict[str, str]]): The list of messages in the conversation.

            Yields:
                str: Any remaining text and the result of processing tool use, if applicable.
            """
            sentences = extract_sentences(current_text)
            for sentence in sentences:
                yield sentence

            if current_tool_use:
                async for r in self._process_tool_use_streaming(current_tool_use, message_list):
                    yield r

        async def process_message_stop(message_list: List[Dict[str, str]], assistant_message: str) -> \
                AsyncGenerator[str, None]:
            """
            Process a message stop event.

            Args:
                message_list (List[Dict[str, str]]): The list of messages in the conversation.
                assistant_message (str): The complete assistant message.

            Yields:
                str: Any remaining text.
            """
            nonlocal current_text, current_tool_use

            if current_text:
                logger.debug(f"{self._time_str()}Yielding final text: {current_text}")
                assistant_message += f"{current_text} "
                yield current_text

            message_list.append({"role": "assistant", "content": assistant_message})

        try:
            async for event in self._get_response_async(message_list, streaming=True):
                logger.debug(f"{self._time_str()}Received event: {event}")
                event_type = event.get('type')

                if event_type == 'error':
                    raise Exception(f"Claude error: {event['error']['message']}")

                if event_type == 'content_block_delta':
                    async for sentence in process_content_block_delta(event, current_tool_use):
                        if sentence:
                            yield sentence

                elif event_type == 'content_block_start':
                    current_tool_use = process_content_block_start(event)

                elif event_type == 'content_block_stop':
                    async for sentence in process_content_block_stop(current_tool_use, current_text, message_list):
                        if sentence:
                            yield sentence
                    current_text = ""
                    current_tool_use = None

                elif event_type == 'message_stop':
                    async for sentence in process_message_stop(message_list, assistant_message):
                        if sentence:
                            yield sentence
                    current_text = ""
                    assistant_message = ""

            if current_text:
                logger.debug(f"{self._time_str()}Yielding remaining text: {current_text}")
                yield current_text

        except StopAsyncIteration:
            logger.debug("AsyncGenerator completed normally.")
        except Exception as e:
            logger.error(f"Error in _get_response_async_streaming: {str(e)}")
            raise

    async def _process_tool_use_streaming(self, tool_use: Dict, message_list: List[Dict[str, str]]) -> AsyncGenerator[
        str, None]:
        """Process a tool use request in streaming mode and generate a response."""

        try:
            tool_input = json.loads(tool_use['input'])
            tool_use['input'] = tool_input
            message_list.append({"role": "assistant", "content": [tool_use]})
            tool_name = tool_use['name']
            tool_use_id = tool_use['id']
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
            logger.error(f"{self._time_str()}Failed to decode tool input JSON: {tool_use['input']}")


class ToolCall(BaseModel):
    """Represents a tool call made by the AI model."""
    arguments: str
    id: str


class GeminiAIModeWithTools(GeminiAIModel):
    """
    Implementation of AIModel using Google Gemini model.
    """

    def __init__(self, config: Config, model_id: Optional[str] = None, tools: Optional[List[Tool]] = None):
        sys_path = sys.path
        sys.path = [p for p in sys.path if p != os.getcwd()]
        import google.generativeai as genai
        sys.path = sys_path

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model_id = model_id or config.get("gemini_model_id", "gemini-1.5-pro-latest")

        self.tools = {t.name: t for t in tools} if tools else {}
        self.tools_description = self._create_tools_description(tools) if tools else None
        self.tools_processors = {t.name: t.processor for t in tools} if tools else {}

        def add(a: int, b: int) -> int:
            """returns a * b."""
            return a+b

        import inspect
        sig = inspect.signature(add)

        self.model = genai.GenerativeModel(model_id, tools=[add])

        self.model = genai.GenerativeModel(model_id, tools=self.tools_description)
        max_tokens = config.get('max_tokens', 4096)
        self.generation_config = genai.GenerationConfig(max_output_tokens=max_tokens)

    @classmethod
    def _create_tools_description(self, tools: List[Tool]) -> dict:
        """Create a list of tool descriptions for the Gemini model."""

        def schema(t: Tool):
            from google.generativeai.responder import to_type
            return {'type': 'object',
                    'properties': {p.name: {'type': p.type, 'description': p.description}
                                   for p in t.parameters},
                    'required': t.required}

        return {'function_declarations':
                    [{'name': t.name, 'description': t.description, 'parameters': schema(t)} for t in tools]}

    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response using Google Gemini model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            str: The generated response.
        """
        from google.generativeai import types
        import google.generativeai as genai

        messages = normalize_messages(messages)
        system_message_combined = " ".join([m["content"] for m in messages if m["role"] == "system"])
        non_system_messages = [m for m in messages if m["role"] != 'system']
        adapted_messages = []
        for m in non_system_messages:
            a = {"role": "model" if m["role"] == "assistant" else "user", "parts": m["content"]}
            adapted_messages.append(a)
        history = adapted_messages[:-1] if adapted_messages else []

        try:
            if system_message_combined:
                self.model._system_instruction = types.content_types.to_content(system_message_combined)
            chat = self.model.start_chat(history=history, enable_automatic_function_calling=True)
            response = chat.send_message(adapted_messages[-1], generation_config=self.generation_config)
            while True:
                part = response.candidates[0].content.parts[0]
                if not part.function_call:
                    return part.text.strip()
                else:
                    fc = part.function_call
                    args = {p.name: fc.args[p.name] for p in self.tools[fc.name].parameters if p.name in fc.args}
                    result = asyncio.run(self.tools_processors[fc.name](args))
                    if self.tools[fc.name].iterative:
                        response = chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name='multiply',
                                        response={'result': result}))]),
                            generation_config=self.generation_config)
        except Exception as e:
            logging.error(f"Error in Gemini API call: {str(e)}")
            raise

    @retry_async_generator()
    @yield_complete_sentences
    async def get_response_async(self, messages: MessageList) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response using Google gemini model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Yields:
            str: Parts of the generated response.
        """
        from google.generativeai import types
        import google.generativeai as genai

        messages = normalize_messages(messages)
        system_message_combined = " ".join([m["content"] for m in messages if m["role"] == "system"])
        non_system_messages = [m for m in messages if m["role"] != 'system']
        adapted_messages = []
        for m in non_system_messages:
            a = {"role": "model" if m["role"] == "assistant" else "user", "parts": m["content"]}
            adapted_messages.append(a)
        history = adapted_messages[:-1] if adapted_messages else []

        try:
            if system_message_combined:
                self.model._system_instruction = types.content_types.to_content(system_message_combined)
            chat = self.model.start_chat(history=history, enable_automatic_function_calling=True)
            response = chat.send_message(adapted_messages[-1], generation_config=self.generation_config)
            part = response.candidates[0].content.parts[0]
            if not part.function_call:
                yield part.text
            else:
                fc = part.function_call
                args = {p.name : fc.args[p.name] for p in self.tools[fc.name].parameters if p.name in fc.args}
                result = await self.tools_processors[fc.name](args)
                if self.tools[fc.name].iterative:
                    response = await chat.send_message_async(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name='multiply',
                                    response={'result': result}))]),
                        generation_config=self.generation_config
                    )
                    yield response.text

        except Exception as e:
            logging.error(f"Error in Gemini API call: {str(e)}")
            raise


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
        self.tools_processors = {t.name: t.processor for t in tools} if tools else {}
        self.tools = {t.name: t for t in tools} if tools else {}
        self.tools_description = self._create_tools_description(tools) if tools else []

    @classmethod
    def _create_tools_description(self, tools: List[Tool]) -> List[Dict]:
        """Create a list of tool descriptions for the OpenAI model."""

        def schema(t: Tool):
            return {'type': 'object',
                    'properties': {p.name: {'type': p.type, 'description': p.description} for p in t.parameters},
                    'required': t.required}

        return [{"type": "function", "function": {'name': t.name, 'description': t.description, 'parameters': schema(t),
                                                  'required': t.required}} for t in tools]

    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response using the OpenAI model with tool capabilities.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        messages = normalize_messages(messages)

        response = self.client.chat.completions.create(model=self.model, messages=messages,
                                                       tools=self.tools_description, max_tokens=self.max_tokens)
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
    async def get_response_async(self, messages: MessageList) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response using the OpenAI model with tool capabilities.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Yields:
            str: Parts of the generated response.
        """
        messages = normalize_messages(messages)

        logger.debug(f"open ai: Getting response: {messages}")
        stream = await self.client_async.chat.completions.create(model=self.model, messages=messages,
                                                                 tools=self.tools_description, stream=True, )
        tool_name = ""
        tools: Dict[str, ToolCall] = {}
        current_text = ""
        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            logger.debug(f"reason: {choice.finish_reason}; tools: {choice.delta.tool_calls}")
            if choice.finish_reason is None and choice.delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tool_name = tool_call.function.name or tool_name
                    if tool_name not in tools:
                        tools[tool_name] = ToolCall(arguments="", id=tool_call.id)
                    tools[tool_name].arguments += tool_call.function.arguments
            elif choice.finish_reason == 'tool_calls':
                _messages = messages.copy()
                tool_calls = [
                    {"function": {"arguments": tools[t].arguments, "name": t}, "id": tools[t].id, "type": "function"}
                    for t in tools]
                _messages.append({"role": "assistant", "tool_calls": tool_calls})
                async for r in self._process_tool_calls(tools, _messages):
                    yield r
            else:
                text = chunk.choices[0].delta.content or ""
                current_text += text
                sentences = extract_sentences(current_text)
                logger.debug(f"text: {text}; current text: {current_text}; sentences: {sentences}")
                # If we have any complete sentences, yield them
                if len(sentences) > 1:
                    for sentence in sentences[:-1]:
                        yield sentence

                    # Keep the last (potentially incomplete) sentence
                    current_text = sentences[-1]

        if current_text:
            yield current_text

    async def _process_tool_calls(self, tools: Dict[str, ToolCall], messages: List[Dict[str, str]]) -> AsyncGenerator[
        str, None]:
        """Process tool calls and generate a response."""
        _messages = messages.copy()
        iterate = False
        for tool_name, tool_call in tools.items():
            tool_use_id = tool_call.id
            tool_parameters = json.loads(tool_call.arguments)
            tool_processor = self.tools_processors[tool_name]
            tool_result = await tool_processor(tool_parameters)
            _messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_use_id})
            if self.tools[tool_name].iterative:
                iterate = True
        if iterate:
            async for response in self.get_response_async(_messages):
                yield response


def main():
    """
    Main function for testing the ClaudeAIModelWithTools.
    """
    from src.web_search_tool import WebSearchTool
    config = Config()
    search_tool = WebSearchTool(config)
    for model in [GeminiAIModeWithTools(config, tools=[search_tool.tool_definition()]),
                  ClaudeAIModelWithTools(config, tools=[search_tool.tool_definition()]),
                  OpenAIModelWithTools(config, tools=[search_tool.tool_definition()])]:
        message = "Today is August 6, 2024. who got olympics gold today?"
        print(message)
        messages = [{"role": "user", "content": message}]
        print(model.get_response(messages))


async def main_async():
    """
    Asynchronous main function for testing the ClaudeAIModelWithTools.
    """
    from src.web_search_tool import WebSearchTool
    config = Config()
    search_tool = WebSearchTool(config)
    for model in [GeminiAIModeWithTools(config, tools=[search_tool.tool_definition()]),
                  OpenAIModelWithTools(config, tools=[search_tool.tool_definition()]),
                  ClaudeAIModelWithTools(config, tools=[search_tool.tool_definition()])]:
        system = """Today is August 6 2024. Now 12:15 PM PDT. In San Jose, California, US. Тебя зовут Кубик. Ты мой друг и помощник. Ты умеешь шутить и быть саркастичным.
                Отвечай естественно, как в устной речи. Говори максимально просто и понятно. Не используй списки и нумерации. Например, не говори 1. что-то; 2.
                что-то. говори во-первых, во-вторых или просто перечисляй. При ответе на вопрос где важно время, помни какое сегодня число. Если чего-то не знаешь,
                так и скажи. Я буду разговаривать с тобой через голосовой интерфейс. Будь краток, избегай банальностей и непрошенных советов."""
        message = "Today is August 6, 2024. who got olympics gold today?"
        print(message)
        # messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
        messages = [{"role": "user", "content": message}]
        m = ""
        async for response_part in model.get_response_async(messages):
            print(response_part, flush=True, end="")
            m += response_part
        print()


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(main_async())
    # main()
