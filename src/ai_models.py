"""
AI Models module.

This module provides abstract and concrete implementations of AI models
for generating responses in conversations, including OpenAI's GPT and Anthropic's Claude.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, AsyncGenerator
import aiohttp

import requests

from src.config import Config


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

    def __init__(self, config: Config):
        """
        Initialize the Claude AI model.

        Args:
            config (Config): The application configuration object.
        """
        self.model = config.get('claude_model', 'claude-3-5-sonnet-20240620')
        self.max_tokens = config.get('max_tokens', 1000)
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {"content-type": "application/json", "x-api-key": os.getenv('ANTHROPIC_API_KEY'),
                        "anthropic-version": "2023-06-01"}

    def _get_response(self, messages: List[Dict[str, str]]) -> dict:
        data = {"model": self.model, "max_tokens": self.max_tokens, "tools": self.tools_description,
                "messages": messages}

        response = requests.post(self.url, headers=self.headers, json=data)
        return json.loads(response.content.decode('utf-8'))

    async def _get_response_async(self, messages: List[Dict[str, str]]) -> dict:
        data = {"model": self.model, "max_tokens": self.max_tokens, "tools": self.tools_description,
                "messages": messages}

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
        from openai import OpenAI
        self.client = OpenAI(base_url="https://api.perplexity.ai", api_key=os.getenv("PERPLEXITY_API_KEY"))
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


async def loop():
    config = Config()


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
    # asyncio.run(loop())

    main()
