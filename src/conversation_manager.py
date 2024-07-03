"""
Conversation Manager module.

This module provides the ConversationManager class for managing the flow of conversation,
including message history, token counting, and interaction with AI models.
"""

from collections import deque
import re
from typing import List, Dict, Tuple
from .ai_models import AIModel
import logging
from .web_search import web_search

logger = logging.getLogger(__name__)


def process_and_search(input_string: str, searcher: web_search) -> Tuple[str, List[str]]:
    """
    Process the input string, perform web searches for queries, and return modified string and results.

    Args:
        input_string (str): The input string that may contain web search queries.
        searcher (web_search): An instance of the web_search class.

    Returns:
        Tuple[str, List[str]]: A tuple containing the modified string and a list of web search results.
    """
    # Regular expression to match {internet query: xxx} pattern
    pattern = r'\{internet query:(.*?)\}'

    # Find all matches
    matches = re.findall(pattern, input_string)

    # List to store search results
    search_results = []

    # Process each match
    for match in matches:
        logger.info(f"Performing web search for: {match}")
        try:
            result = searcher.search(match)
            search_results.append(result)
        except Exception as e:
            search_results.append(f"Error performing search: {str(e)}")

    # Remove all {internet query: xxx} substrings from the input string
    modified_string = re.sub(pattern, '', input_string)

    # Remove any extra whitespace that might have been left
    modified_string = ' '.join(modified_string.split())

    return (modified_string, search_results)


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
        self.searcher = web_search()
        self.ai_model = ai_model
        self.message_history = deque()
        system_prompt = config.get('system_prompt',
                                   "Тебя зовут Роби. "
                                   "Ты мой друг и помощник. Отвечай естественно, как в устной речи."
                                   "Говори мксимально просто и понятно. Не используй списки и нумерации."
                                   "Если чего-то не знаешь, так и скажи."
                                   "Я буду разговаривать с тобой через голосовой интерфейс."
                                   "Если чтобы ответить на мой вопрос, тебе нужно поискать в интернете, "
                                   "не отвечпй сразу, а пошли ине сообщение в таком формате:"
                                   "{internet query:<что ты хочешь поискать на английском языке>}")
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

        response_text_, search_results = process_and_search(response_text, self.searcher)
        logger.info(f"Response Text: {response_text_}; Search results: {search_results}")

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
