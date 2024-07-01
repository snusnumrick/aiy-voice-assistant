"""
AI Models module.

This module provides abstract and concrete implementations of AI models
for generating responses in conversations, including OpenAI's GPT and Anthropic's Claude.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import os


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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()


class ClaudeAIModel(AIModel):
    """
    Implementation of AIModel using Anthropic's Claude model.
    """

    def __init__(self, config):
        """
        Initialize the Claude AI model.

        Args:
            config (Config): The application configuration object.
        """
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.get('anthropic_api_key'))
        self.model = config.get('claude_model', 'claude-3-5-sonnet-20240620')
        self.max_tokens = config.get('max_tokens', 1000)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using Anthropic's Claude model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        prompt = self._convert_messages_to_prompt(messages)
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens_to_sample=self.max_tokens
        )
        return response.completion.strip()

    @staticmethod
    def _convert_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Convert a list of message dictionaries to a Claude-compatible prompt string.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries.

        Returns:
            str: A formatted prompt string for Claude.
        """
        prompt = ""
        for message in messages:
            if message['role'] == 'system':
                prompt += f"System: {message['content']}\n\n"
            elif message['role'] == 'user':
                prompt += f"Human: {message['content']}\n\n"
            elif message['role'] == 'assistant':
                prompt += f"Assistant: {message['content']}\n\n"
        prompt += "Assistant:"
        return prompt


class OpenRouterModel(AIModel):
    """
    Implementation of AIModel using OpenAI's GPT model.
    """

    def __init__(self, config):
        """
        Initialize the OpenRouter model.

        Args:
            config (Config): The application configuration object.
        """
        from openai import OpenAI
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.getenv("OPENROUTER_API_KEY"))
        self.model = config.get('openrouter_model', 'anthropic/claude-3.5-sonnet')
        self.max_tokens = config.get('max_tokens', 4096)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using OpenRouter model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()
