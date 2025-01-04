# -*- coding: utf-8 -*-
"""
AI Models module.

This module provides abstract and concrete implementations of AI models
for generating responses in conversations, including OpenAI's GPT, Anthropic's Claude,
OpenRouter, and Perplexity models.
"""

import asyncio
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, AsyncGenerator, Optional, Union
from pydantic import BaseModel
import aiohttp
import requests

from src.config import Config
from src.tools import time_string_ms, retry_async_generator, yield_complete_sentences


class MessageModel(BaseModel):
    role: str
    content: str


MessageList = List[Union[Dict[str, str], MessageModel]]


def normalize_messages(messages: MessageList) -> List[Dict[str, str]]:
    """
    Normalize messages to ensure they are in the correct format for API calls.

    Args:
        messages (MessageList): A list of messages.

    Returns:
        List[Dict[str, str]]: A list of normalized message dictionaries.
    """
    return [
        message.model_dump() if isinstance(message, MessageModel) else message
        for message in messages
    ]


class AIModel(ABC):
    """
    Abstract base class for AI models used in conversation.
    """

    @abstractmethod
    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response based on the conversation history.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            str: The generated response.
        """
        pass

    async def get_response_async(
        self, messages: MessageList
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response based on the conversation history.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Yields:
            str: Parts of the generated response.
        """
        pass


class GeminiAIModel(AIModel):
    """
    Implementation of AIModel using Google Gemini model.
    """

    def __init__(self, config: Config, model_id: Optional[str] = None):
        sys_path = sys.path
        sys.path = [p for p in sys.path if p != os.getcwd()]
        import google.generativeai as genai

        sys.path = sys_path

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model_id = model_id or config.get("gemini_model_id", "gemini-1.5-pro-exp-0801")
        self.model = genai.GenerativeModel(model_id)
        max_tokens = config.get("max_tokens", 4096)
        self.generation_config = genai.GenerationConfig(max_output_tokens=max_tokens)

    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response using Google Gemini model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            str: The generated response.
        """
        from google.generativeai import types

        messages = normalize_messages(messages)
        system_message_combined = " ".join(
            [m["content"] for m in messages if m["role"] == "system"]
        )
        non_system_messages = [m for m in messages if m["role"] != "system"]
        adapted_messages = []
        for m in non_system_messages:
            a = {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": m["content"],
            }
            adapted_messages.append(a)
        history = adapted_messages[:-1] if adapted_messages else []

        try:
            if system_message_combined:
                self.model._system_instruction = types.content_types.to_content(
                    system_message_combined
                )
            chat = self.model.start_chat(history=history)
            response = chat.send_message(
                adapted_messages[-1], generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error in Gemini API call: {str(e)}")
            raise

    @retry_async_generator()
    @yield_complete_sentences
    async def get_response_async(
        self, messages: MessageList
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response using Google gemini model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Yields:
            str: Parts of the generated response.
        """
        from google.generativeai import types

        messages = normalize_messages(messages)
        system_message_combined = " ".join(
            [m["content"] for m in messages if m["role"] == "system"]
        )
        non_system_messages = [m for m in messages if m["role"] != "system"]
        adapted_messages = []
        for m in non_system_messages:
            a = {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": m["content"],
            }
            adapted_messages.append(a)
        history = adapted_messages[:-1] if adapted_messages else []

        try:
            if system_message_combined:
                self.model._system_instruction = types.content_types.to_content(
                    system_message_combined
                )
            chat = self.model.start_chat(history=history)
            async for response in await chat.send_message_async(
                adapted_messages[-1],
                generation_config=self.generation_config,
                stream=True,
            ):
                yield response.text

        except Exception as e:
            logging.error(f"Error in Gemini API call: {str(e)}")
            raise


class OpenAIModel(AIModel):
    """
    Implementation of AIModel using OpenAI's GPT model.
    """

    def __init__(
        self,
        config: Config,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        """
        Initialize the OpenAI model.

        Args:
            config (Config): The application configuration object.
            base_url (Optional[str]): The base URL for the API.
            api_key (Optional[str]): The API key for authentication.
            model_id (Optional[str]): The specific model ID to use.
        """
        from openai import OpenAI, AsyncOpenAI

        self.model = model_id or config.get("openai_model", "gpt-4o")
        self.max_tokens = config.get("max_tokens", 4096)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.client_async = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response using OpenAI's GPT model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            str: The generated response.
        """
        messages = normalize_messages(messages)
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error in OpenAI API call: {str(e)}")
            raise

    @retry_async_generator()
    @yield_complete_sentences
    async def get_response_async(
        self, messages: MessageList
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response using OpenAI's GPT model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Yields:
            str: Parts of the generated response.
        """
        messages = normalize_messages(messages)
        stream = await self.client_async.chat.completions.create(
            model=self.model, messages=messages, stream=True
        )
        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""


class ClaudeAIModel(AIModel):
    """
    Implementation of AIModel using Anthropic's Claude model.
    """

    def __init__(self, config: Config, timezone: str = ""):
        """
        Initialize the Claude AI model.

        Args:
            config (Config): The application configuration object.
            timezone (str): The timezone to use for timestamps.
        """
        self.model = config.get("claude_model", "claude-3-5-sonnet-20240620")
        self.max_tokens = config.get("max_tokens", 4096)
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "content-type": "application/json",
            "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
        }
        self.config = config
        self.timezone = timezone

    def _time_str(self) -> str:
        """
        Generate a formatted time string.

        Returns:
            str: Formatted time string with timezone if available.
        """
        return f"({time_string_ms(self.timezone)}) " if self.timezone else ""

    def _get_response(self, messages: List[Dict[str, str]]) -> dict:
        """
        Send a request to the Claude API and get the response.

        Args:
            messages (List[Dict[str, str]]): A list of message models representing the conversation history.

        Returns:
            dict: The API response as a dictionary.
        """
        system_message_combined = " ".join(
            [m["content"] for m in messages if m["role"] == "system"]
        )
        non_system_message = [m for m in messages if m["role"] != "system"]
        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": non_system_message,
        }
        if system_message_combined:
            data["system"] = system_message_combined

        response = requests.post(self.url, headers=self.headers, json=data)
        return json.loads(response.content.decode("utf-8"))

    async def _get_response_async(self, messages: List[Dict[str, str]]) -> dict:
        """
        Asynchronously send a request to the Claude API and get the response.

        Args:
            messages (List[Dict[str, str]]): A list of message models representing the conversation history.

        Returns:
            dict: The API response as a dictionary.
        """
        system_message_combined = " ".join(
            [m["content"] for m in messages if m["role"] == "system"]
        )
        non_system_message = [m for m in messages if m["role"] != "system"]

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": non_system_message,
        }
        if system_message_combined:
            data["system"] = system_message_combined

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, headers=self.headers, json=data
            ) as response:
                res = await response.text()
                return json.loads(res)

    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response using Anthropic's Claude model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            str: The generated response.
        """
        messages = normalize_messages(messages)
        response_dict = self._get_response(messages)
        response_text = ""
        for content in response_dict["content"]:
            if content["type"] == "text":
                response_text += content["text"]
        return response_text

    @retry_async_generator()
    @yield_complete_sentences
    async def get_response_async(
        self, messages: MessageList
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response using Anthropic's Claude model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Yields:
            str: Parts of the generated response.

        Raises:
            Exception: If there's an error in the API response.
        """
        messages = normalize_messages(messages)
        response_dict = await self._get_response_async(messages)
        if "error" in response_dict:
            raise Exception(response_dict["error"])
        for content in response_dict["content"]:
            if content["type"] == "text":
                yield content["text"]


class OpenRouterModel(OpenAIModel):
    """
    Implementation of AIModel using OpenRouter's API.
    """

    def __init__(self, config: Config, use_simple_model: bool = False):
        """
        Initialize the OpenRouter model.

        Args:
            config (Config): The application configuration object.
            use_simple_model (bool): Whether to use the simple model or not.
        """
        if use_simple_model:
            model = config.get("openrouter_model_simple", "anthropic/claude-3-haiku")
        else:
            model = config.get("openrouter_model", "anthropic/claude-3.5-sonnet")
        base_url = config.get(
            "openrouter_model_base_url", "https://openrouter.ai/api/v1"
        )
        super().__init__(
            config,
            base_url=base_url,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_id=model,
        )


class PerplexityModel(OpenAIModel):
    """
    Implementation of AIModel using Perplexity's API.
    """

    def __init__(self, config: Config):
        """
        Initialize the Perplexity model.

        Args:
            config (Config): The application configuration object.
        """
        model = config.get("perplexity_model", "llama-3.1-sonar-small-128k-online")
        base_url = "https://api.perplexity.ai"
        api_key = os.getenv("PERPLEXITY_API_KEY")
        super().__init__(config, base_url=base_url, api_key=api_key, model_id=model)


# Debug functions
async def main_async():
    """
    Asynchronous main function for debugging purposes.
    """
    prompt = """
    Обобщи основные моменты разговора, сосредоточившись на наиболее важных фактах и контексте. Будь лаконичен. Отвечай от лицв user. Начни ответ с "Ранее мы говорили о ".
    Разговор:
    -----
    user:  Привет кубик!
    assistant:  Здор+ово! Как жизнь? Рад тебя слышать! Что нового у тебя произошло с нашего последнего разговора? Может, уже успел сыграть в футбол? Ведь сегодня четверг, а ты как раз начал играть по четвергам с первого августа.
    user:  Я решил, о чем только недавно тебе об этом сказал.
    assistant:  А, точно! Прошу прощения, я немного поторопился с выводами. Ты действительно только недавно рассказал мне о своём решении начать играть в футбол по четвергам с 1 августа. А сегодня как раз первый четверг августа! Ну что, готов к первой игре? Наверное, уже в предвкушении, а? Футбол - это же здорово! Бегаешь, пинаешь мяч, общаешься с друзьями. Отличный способ провести вечер! А как ты вообще решился на это? Может, друзья позвали или просто захотелось добавить активности в жизнь? В любом случае, молодец, что решился! Главное - не забудь взять с собой воды и, может быть, лёгкий перекус. После игры точно захочется подкрепиться. Кстати, если вдруг решишь поделиться впечатлениями после игры, буду рад послушать. Удачи тебе сегодня!
    user:  Спасибо, Кубик.
    assistant:  Да не за что, дружище! Всегда рад поддержать. Ты сегодня какой-то немногословный. Всё в порядке? Может, волнуешься перед первой игрой? Или просто устал? Знаешь, если что, я всегда тут, готов выслушать и поболтать. А если просто хочешь помолчать - это тоже нормально. Главное, чтобы ты знал, что твой кубический друг всегда на связи!
    user:  Привет кубик!
    assistant:
    Здор+ово! Рад тебя снова слышать! Как дела? Ты сегодня в хорошем настроении? Я-то точно в приподнятом духе, аж свечусь от радости! Ну, в каком-то смысле действительно свечусь, хе-хе. Расскажи, как прошёл твой день? Может, случилось что-нибудь интересное? Или, может быть, у тебя есть какие-то планы на вечер? Я тут весь в предвкушении хорошей беседы!
    user:  Я игрок в бокс.
    assistant:  Ого, вот это новость! Бокс? Серьёзно? Ну ты меня удивил, дружище! А я-то думал, ты по футболу специализируешься. Видимо, ты у нас настоящий спортсмен-универсал!
    Слушай, а это же очень круто! Бокс - спорт не для слабаков. Как давно ты им занимаешься? И в какой весовой категории выступаешь? Знаешь, я тут подумал... Хорошо, что ты мне об этом сказал. А то представь: я бы тебе про футбол заладил, а ты бы недоумевал, с чего это вдруг твой кубический друг решил, что ты футболист.
    Кстати, раз уж мы заговорили о боксе, у меня есть один вопрос.
    user:  Я играл в футбол.
    """
    config = Config()
    ai_model = GeminiAIModel(config)
    ai_model = ClaudeAIModel(config)
    # ai_model = OpenAIModel(config, model_id="gpt-4o-mini")
    result = ""
    async for response_part in ai_model.get_response_async(
        [{"role": "user", "content": prompt}]
    ):
        print(response_part)
        result += response_part
    print(result)


def main():
    """
    Main function for debugging purposes.
    """
    config = Config()
    model = GeminiAIModel(config)
    messages = [
        {"role": "system", "content": "Your name is Cubie."},
        {"role": "user", "content": "Hi! What's your name?"},
    ]
    response = model.get_response(messages)
    print(response)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    # main()
    asyncio.run(main_async())  # main()
