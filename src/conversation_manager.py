"""
Conversation Manager module.

This module provides the ConversationManager class for managing the flow of conversation,
including message history, token counting, and interaction with AI models.
"""

import datetime
import logging
import re
from collections import deque
from typing import List, Dict, Tuple

import geocoder
import pytz

from .ai_models import AIModel
from .config import Config
from .web_search import web_search

logger = logging.getLogger(__name__)


def get_current_date_time_location():
    # Get current date and time in UTC
    now_utc = datetime.datetime.now(pytz.utc)

    # Define the timezone you want to convert to (for example, PST)
    timezone = pytz.timezone('America/Los_Angeles')
    now_local = now_utc.astimezone(timezone)

    # Format the date with the month as a word
    months = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня', 7: 'июля', 8: 'августа',
              9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
    date_str = now_local.strftime(f"%d {months[now_local.month]} %Y")

    # Format the time in 12-hour format with AM/PM and timezone
    time_str = now_local.strftime("%I:%M:%S %p, %Z")

    # Get current location
    g = geocoder.ip('me')
    location_parts = [g.city, g.state, g.country]
    location = ', '.join([part for part in location_parts if part]) if any(location_parts) else ''

    # Prepare the message in Russian
    message = f"Сегодня {date_str}. Сейчас {time_str}."
    if location:
        message += f" Я нахожусь в {location}"

    return message


def get_system_prompt(config: Config):
    prompt = f"Тебя зовут Роби. " \
             "{get_current_date_time_location()}" \
             "Ты мой друг и помощник. Отвечай естественно, как в устной речи. " \
             "Говори максимально просто и понятно. Не используй списки и нумерации. " \
             "Если чего-то не знаешь, так и скажи. " \
             "Я буду разговаривать с тобой через голосовой интерфейс. " \
             "Если чтобы ответить на мой вопрос, тебе нужно поискать в интернете, " \
             "не отвечай сразу, а пошли мне сообщение в таком формате: " \
             "{internet query:<что ты хочешь поискать на английском языке>}"
    logger.info(f"System prompt: {prompt}")
    return prompt


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
        logger.debug(f"Performing web search for: {match}")
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
        self.message_history = deque([{"role": "system", "content": get_system_prompt(config)}])

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
        summary_prompt = self.config.get('summary_prompt', "Summarize the key points of this conversation, "
                                                           "focusing on the most important facts and context. Be concise:")
        first_message = True
        for msg in self.message_history:
            if not first_message:
                summary_prompt += f"\n{msg['role']}: {msg['content']}"
            else:
                first_message = False

        summary = self.ai_model.get_response([{"role": "user", "content": summary_prompt}])

        self.message_history = deque([{"role": "system", "content": get_system_prompt(self.config)},
                                      {"role": "system", "content": f"Conversation summary: {summary}"}])

        logger.info(f"Summarized conversation: {summary}")

    def get_response(self, text: str) -> str:
        """
        Get an AI response based on the current conversation state and new input.

        Args:
            text (str): The new input text to respond to.

        Returns:
            str: The AI-generated response.
        """

        # update system message
        self.message_history[0] = {"role": "system", "content": get_system_prompt(self.config)}

        self.message_history.append({"role": "user", "content": text})

        while self.get_token_count(list(self.message_history)) > self.config.get('token_threshold', 2500):
            self.summarize_and_compress_history()

        logger.info(f"Message history: \n{self.formatted_message_history()}")

        response_text = self.ai_model.get_response(list(self.message_history))
        logger.info(f"AI response: {text} -> {response_text}")
        self.message_history.append({"role": "assistant", "content": response_text})

        _, search_results = process_and_search(response_text, self.searcher)
        logger.info(f"Response Text: {_}; Search results: {search_results}")

        if search_results:
            result_message = f"результаты поиска: {search_results[0]}"
            self.message_history.append({"role": "system", "content": result_message})
            self.message_history.append({"role": "user", "content": "?"})
            logger.info(self.formatted_message_history())
            response_text = self.ai_model.get_response(list(self.message_history))
            self.message_history.pop()
            self.message_history.append({"role": "assistant", "content": response_text})
        logger.info(f"AI response: {text} -> {response_text}")

        return response_text

    def formatted_message_history(self):
        """
        Format the message history for logging purposes.

        Returns:
            str: A formatted string representation of the message history.
        """
        return "\n".join([f'{msg["role"]}:{msg["content"].strip()}' for msg in self.message_history])
