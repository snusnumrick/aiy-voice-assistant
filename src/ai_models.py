# -*- coding: utf-8 -*-
"""
AI Models module.

This module provides abstract and concrete implementations of AI models
for generating responses in conversations, including OpenAI's GPT and Anthropic's Claude.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, AsyncGenerator

import aiohttp
import requests

from src.config import Config
from src.tools import time_string_ms


class AIModel(ABC):
    """
    Abstract base class for AI models used in conversation.
    """

    @abstractmethod
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response based on the conversation history.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        pass

    async def get_response_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        pass


class OpenAIModel(AIModel):
    """
    Implementation of AIModel using OpenAI's GPT model.
    """

    def __init__(self, config):
        """
        Initialize the OpenAI model.

        Args:
            config (Config): The application configuration object.
        """
        from openai import OpenAI
        self.client = OpenAI()
        self.model = config.get('openai_model', 'gpt-4o')
        self.max_tokens = config.get('max_tokens', 4096)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using OpenAI's GPT model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=self.max_tokens)
        return response.choices[0].message.content.strip()


class ClaudeAIModel(AIModel):
    """
    Implementation of AIModel using Anthropic's Claude model.
    """

    def __init__(self, config: Config, timezone: str = ""):
        """
        Initialize the Claude AI model.

        Args:
            config (Config): The application configuration object.
            timezone (str): The timezone to use.
        """
        self.model = config.get('claude_model', 'claude-3-5-sonnet-20240620')
        self.max_tokens = config.get('max_tokens', 4096)
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {"content-type": "application/json", "x-api-key": os.getenv('ANTHROPIC_API_KEY'),
                        "anthropic-version": "2023-06-01"}
        self.config = config
        self.timezone = timezone

    def _time_str(self) -> str:
        return f"({time_string_ms(self.timezone)}) " if self.timezone else ""

    def _get_response(self, messages: List[Dict[str, str]]) -> dict:
        system_message_combined = " ".join([m["content"] for m in messages if m["role"] == "system"])
        non_system_message = [m for m in messages if m["role"] != 'system']
        data = {"model": self.model, "max_tokens": self.max_tokens, "messages": non_system_message}
        if system_message_combined:
            data["system"] = system_message_combined

        response = requests.post(self.url, headers=self.headers, json=data)
        return json.loads(response.content.decode('utf-8'))

    async def _get_response_async(self, messages: List[Dict[str, str]]) -> dict:
        system_message_combined = " ".join([m["content"] for m in messages if m["role"] == "system"])
        non_system_message = [m for m in messages if m["role"] != 'system']

        data = {"model": self.model, "max_tokens": self.max_tokens, "messages": non_system_message}
        if system_message_combined:
            data["system"] = system_message_combined

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=self.headers, json=data) as response:
                res = await response.text()
                return json.loads(res)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using Anthropic's Claude model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        response_dict = self._get_response(messages)
        responce_text = ""
        for content in response_dict['content']:
            if content['type'] == 'text':
                responce_text += content['text']
        return responce_text

    async def get_response_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Generate a response using Anthropic's Claude model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        response_dict = await self._get_response_async(messages)
        if 'error' in response_dict:
            raise Exception(response_dict['error'])
        for content in response_dict['content']:
            if content['type'] == 'text':
                yield content['text']


class OpenRouterModel(AIModel):
    """
    Implementation of AIModel using OpenAI's GPT model.
    """

    def __init__(self, config, use_simple_model=False):
        """
        Initialize the OpenRouter model.

        Args:
            config (Config): The application configuration object.
        """
        from openai import OpenAI
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        self.model = config.get('openrouter_model_simple',
                                'anthropic/claude-3-haiku') if use_simple_model else config.get('openrouter_model',
                                                                                                'anthropic/claude-3.5-sonnet')
        self.max_tokens = config.get('max_tokens', 4096)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using OpenRouter model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=self.max_tokens)
        return response.choices[0].message.content.strip()


class PerplexityModel(AIModel):
    """
    Implementation of AIModel using Perplexity.
    """

    def __init__(self, config):
        """
        Initialize the Perplexity model.

        Args:
            config (Config): The application configuration object.
        """
        from openai import OpenAI, AsyncOpenAI
        self.client = OpenAI(base_url="https://api.perplexity.ai", api_key=os.getenv("PERPLEXITY_API_KEY"))
        self.client_async = AsyncOpenAI(base_url="https://api.perplexity.ai", api_key=os.getenv("PERPLEXITY_API_KEY"))
        self.model = config.get('perplexity_model', 'llama-3-sonar-large-32k-online')
        self.max_tokens = config.get('max_tokens', 4096)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using Perplexity model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=self.max_tokens)
        return response.choices[0].message.content.strip()

    async def get_response_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        stream = await self.client_async.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""


async def main_async():
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
    ai_model = ClaudeAIModel(config)
    result = ""
    async for response_part in ai_model.get_response_async([{"role": "user", "content": prompt}]):
        result += response_part
        print(result)


def main():
    config = Config()
    model = ClaudeAIModel(config)
    messages = [{"role": "user", "content": "who will play at euro 2024 final?"}]
    responce = model.get_response(messages)
    print(responce)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(main_async())  # main()
