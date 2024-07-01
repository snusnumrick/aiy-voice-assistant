"""
Conversation Manager module.

This module provides the ConversationManager class for managing the flow of conversation,
including message history, token counting, and interaction with AI models.
"""

from collections import deque
import re
from typing import List, Dict
from .ai_models import AIModel
import logging

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages the conversation flow, including message history and interaction with AI models.

    Attributes:
        config (Config): The application configuration object.
        ai_model (AIModel): The AI model used for generating responses.
        message_history (deque): A queue of message dictionaries representing the conversation history.
    """

    def __init__(self, config, ai_model: AIModel):
        """
        Initialize the ConversationManager.

        Args:
            config (Config): The application configuration object.
            ai_model (AIModel): The AI model to use for generating responses.
        """
        self.config = config
        self.ai_model = ai_model
        self.message_history = deque()
        system_prompt = config.get('system_prompt',
                                   "Тебя зовут Роби. "
                                   "Ты мой друг и помощник. Отвечай естественно, как в устной речи."
                                   "Если чего-то не знаешь, так и скажи.")
        self.message_history.append({"role": "system", "content": system_prompt})

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a given text.

        Args:
            text (str): The text to estimate tokens for.

        Returns:
            int: Estimated number of tokens.
        """
        words = len(re.findall(r'\b\w+\b', text))
        punctuation = len(re.findall(r'[.,!?;:"]', text))
        return int(words * 1.5 + punctuation)

    def get_token_count(self, messages: List[Dict[str, str]]) -> int:
        """
        Get the total token count for a list of messages.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries.

        Returns:
            int: Total estimated token count.
        """
        return sum(self.estimate_tokens(msg["content"]) for msg in messages)

    def summarize_and_compress_history(self):
        """
        Summarize and compress the conversation history to reduce token count.
        """
        summary_prompt = self.config.get('summary_prompt',
                                         "Summarize the key points of this conversation, "
                                         "focusing on the most important facts and context. Be concise:")
        for msg in self.message_history:
            if msg["role"] != "system":
                summary_prompt += f"\n{msg['role']}: {msg['content']}"

        summary = self.ai_model.get_response([{"role": "user", "content": summary_prompt}])

        system_message = self.message_history[0]
        self.message_history.clear()
        self.message_history.append(system_message)
        self.message_history.append({"role": "system", "content": f"Conversation summary: {summary}"})

        logger.info(f"Summarized conversation: {summary}")

    def get_response(self, text: str) -> str:
        """
        Get an AI response based on the current conversation state and new input.

        Args:
            text (str): The new input text to respond to.

        Returns:
            str: The AI-generated response.
        """
        self.message_history.append({"role": "user", "content": text})

        while self.get_token_count(list(self.message_history)) > self.config.get('token_threshold', 2500):
            self.summarize_and_compress_history()

        logger.info(f"Message history: \n{self.formatted_message_history()}")

        response_text = self.ai_model.get_response(list(self.message_history))
        self.message_history.append({"role": "assistant", "content": response_text})

        logger.debug(f"AI response: {text} -> {response_text}")
        return response_text

    def formatted_message_history(self):
        """
        Format the message history for logging purposes.

        Returns:
            str: A formatted string representation of the message history.
        """
        return "\n".join([f'{msg["role"]}:{msg["content"].strip()}' for msg in self.message_history])
