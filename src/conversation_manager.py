"""
Conversation Manager module.

This module provides the ConversationManager class for managing the flow of conversation,
including message history, token counting, and interaction with AI models.
"""

import datetime
import json
import logging
import os
import re
import sys
from collections import deque
from typing import List, Dict, Tuple

import geocoder
import pytz

if __name__ == "__main__":
    # add current directory to python path
    sys.path.append(os.getcwd())

from src.ai_models import AIModel
from src.config import Config
from src.web_search import WebSearcher

logger = logging.getLogger(__name__)


def get_timezone():
    import googlemaps
    import time

    g = geocoder.ip('me')

    key = os.environ.get('GOOGLE_API_KEY')
    gmaps = googlemaps.Client(key=key)
    timezone = gmaps.timezone(g.latlng, time.time())

    return timezone["timeZoneId"]


def get_current_date_time_location(timezone_string: str) -> str:
    # Get current date and time in UTC
    now_utc = datetime.datetime.now(pytz.utc)

    # Define the timezone you want to convert to (for example, PST)
    timezone = pytz.timezone(timezone_string)
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


def get_location() -> str:
    g = geocoder.ip('me')
    location_parts = [g.city, g.state, g.country]
    location = ', '.join([part for part in location_parts if part]) if any(location_parts) else ''

    return f"In {location}."


def get_current_datetime_english(timezone_string: str) -> str:
    # Set the timezone
    tz = pytz.timezone(timezone_string)

    # Get the current time in the timezone
    current_time = datetime.datetime.now(tz)

    # Format the date
    date_str = current_time.strftime("%d %B %Y")

    # Format the time
    time_str = current_time.strftime("%I:%M %p")

    # Determine if it's PDT or PST
    timezone_abbr = current_time.strftime("%Z")

    # Create the formatted string
    formatted_str = f"Today is {date_str}. Now {time_str} {timezone_abbr}."

    return formatted_str


def get_current_date_time_for_facts():
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

    # Prepare the message in Russian
    message = f"({date_str}, {time_str})"

    return message


async def process_and_search(input_string: str, searcher: WebSearcher) -> Tuple[str, List[str]]:
    """
    Process the input string, perform web searches for queries, and return modified string and results.

    Args:
        input_string (str): The input string that may contain web search queries.
        searcher (Tavily): An instance of the web_search class.

    Returns:
        Tuple[str, List[str]]: A tuple containing the modified string and a list of web search results.
    """
    # Regular expression to match {internet query: xxx} pattern
    pattern = r'\$internet query:(.*?)\$'

    # Find all matches
    matches = re.findall(pattern, input_string)

    # List to store search results
    search_results = []

    # Process each match
    for match in matches:
        logger.info(f"Performing web search for: {match}")
        try:
            result = await searcher.search_2(match)
            search_results.append(result)
        except Exception as e:
            search_results.append(f"Error performing search: {str(e)}")

    # Remove all {internet query: xxx} substrings from the input string
    modified_string = re.sub(pattern, '', input_string)

    # Remove any extra whitespace that might have been left
    modified_string = ' '.join(modified_string.split())

    return (modified_string, search_results)

def extract_facts(text: str) -> Tuple[str, List[str]]:
    """
    Extract facts from the input text and return the modified text and a list of extracted facts.

    Args:
        text (str): The input text to extract facts from.

    Returns:
        Tuple[str, List[str]]: A tuple containing the modified text and a list of extracted facts.
    """
    # Regular expression to match {remember: xxx} pattern
    pattern = r'\$remember:(.*?)\$'

    # Find all matches
    matches = re.findall(pattern, text)

    # List to store extracted facts
    extracted_facts = []

    # Process each match
    for match in matches:
        logger.debug(f"Extracted fact: {match}")
        fact = get_current_date_time_for_facts() + " : " + match
        extracted_facts.append(fact)

    # Remove all {remember: xxx} substrings from the input string
    modified_text = re.sub(pattern, '', text)

    # Remove any extra whitespace that might have been left
    modified_text = ' '.join(modified_text.split())

    return modified_text, extracted_facts


def extract_rules(text: str) -> Tuple[str, List[str]]:
    """
    Extract rules from the input text and return the modified text and a list of extracted facts.

    Args:
        text (str): The input text to extract facts from.

    Returns:
        Tuple[str, List[str]]: A tuple containing the modified text and a list of extracted facts.
    """
    # Regular expression to match {remember: xxx} pattern
    pattern = r'\$rule:(.*?)\$'

    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    # List to store extracted facts
    extracted_rules = []

    # Process each match
    for match in matches:
        logger.debug(f"Extracted rule: {match}")
        rule = get_current_date_time_for_facts() + " : " + match
        extracted_rules.append(rule)

    # Remove all {remember: xxx} substrings from the input string
    modified_text = re.sub(pattern, '', text)

    # Remove any extra whitespace that might have been left
    modified_text = ' '.join(modified_text.split())

    return modified_text, extracted_rules


def extract_emotions(text: str) -> List[Tuple[dict, str]]:
    """
        This function parses the given text and extracts 'emotion' dictionaries (if any) and the associated text following them.
        The structured data is returned as a list of tuples, each containing the dictionary and the corresponding text.

        An emotion dictionary is expected to be enclosed inside '$emotion:' and '$' markers in the input text.
        Any text not preceded by an emotion marker is associated with an empty dictionary.

        :param text: str, Input text which includes 'emotion' dictionaries and text.
        :return: List[Tuple[Dict, str]]. Each tuple contains:
            - dict: The parsed 'emotion' dictionary or an empty dictionary if no dictionary was found.
            - str: The associated text following the dictionary or preceding the next dictionary.
        """

    pattern = re.compile(r'(.*?)\$emotion:\s*(\{.*?\})?\$(.*?)(?=\$emotion:|$)', re.DOTALL)

    results = []
    pos = 0
    while pos < len(text):
        match = pattern.search(text, pos)
        if not match:
            remaining_text = text[pos:].strip()
            if remaining_text:
                results.append(({}, remaining_text))
            break
        preceding_text = match.group(1).strip()
        if preceding_text:
            results.append(({}, preceding_text))

        emotion_dict_str = match.group(2) if match.group(2) else '{}'
        associated_text = match.group(3).strip()
        try:
            emotion_dict = json.loads(emotion_dict_str)
            results.append((emotion_dict, associated_text))
        except json.JSONDecodeError:
            results.append(({}, associated_text))
        pos = match.end()

    return results


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
        self.searcher = WebSearcher(config)
        self.ai_model = ai_model
        self.facts = self.load_facts()
        self.rules = self.load_rules()
        self.location = get_location()
        self.timezone = get_timezone()
        self.hard_rules = ("Если чтобы ответить на мой вопрос, тебе нужно поискать в интернете, не отвечай сразу, "
                           "а пошли мне сообщение в таком формате: "
                           "$internet query:<что ты хочешь поискать на английском языке>$. "
                           "Таких запросов в твоем сообщении может быть несколько. "
                           "Если в ответе на твой запрос указано время без указания часового пояса, "
                           "считай что это Восточное стандартное время."
                           # "Если по этому запросу не нашел нужной информации, попробуй переформулировать запрос. "
                           "Если тебе надо что-то запомнить, "
                           "пошли мне сообщение в таком формате: $remember: <текст, который тебе нужно запомнить>$. "
                           "Таких фактов в твоем сообщении тоже может быть несколько. "
                           "Например, $remember: <первый текст, который тебе нужно запомнить>$ "
                           "{remember: $второрй текст, который тебе нужно запомнить>$."
                           "Если я прошу тебя как-то поменятся (например, не используй обсценную лексику); "
                           "чтобы запомнить это новое правило, пошли мне сообщение в таком формате: "
                           "$rule: <текст нового правила>$. "
                           "Таких запросов в твоем сообщении тоже может быть несколько. ")
        # self.hard_rules = ("For web searches: $internet query:<query in English>$. "
        #                    "To remember: $remember:<text>$. For new rules: $rule:<text>$ ")
        self.default_system_prompt = ("Тебя зовут Кубик. Ты мой друг и помощник. Ты умеешь шутить и быть саркастичным. "
                                      " Отвечай естественно, как в устной речи. "
                                      "Говори максимально просто и понятно. Не используй списки и нумерации. "
                                      "Например, не говори 1. что-то; 2. что-то. говори во-первых, во-вторых "
                                      "или просто перечисляй. "
                                      "При ответе на вопрос где важно время, помни какое сегодня число. "
                                      "Если чего-то не знаешь, так и скажи. "
                                      "Я буду разговаривать с тобой через голосовой интерфейс. "
                                      "Будь краток, избегай банальностей и непрошенных советов. ")
        # self.default_system_prompt = ("You're Kubik, my friendly AI assistant. Be witty and sarcastic. "
        #                               "Speak naturally, simply. Avoid lists. Consider date in time-sensitive answers. "
        #                               "Admit unknowns. I use voice interface. Be brief, avoid platitudes. "
        #                               "Use internet searches when needed for up-to-date or specific information. "
        #                               "Assume EST if timezone unspecified. Treat responses as spoken.")
        self.message_history = deque([{"role": "system", "content": self.get_system_prompt()}])

    def get_system_prompt(self):
        from src.responce_player import emotions_prompt

        prompt = f"{get_current_datetime_english(self.timezone)} {self.location} "
        prompt += self.config.get('system_prompt', self.default_system_prompt)
        prompt += self.hard_rules
        prompt += emotions_prompt()
        if self.facts:
            prompt += " Ты уже знаешь факты:" + " ".join(self.facts)
        if self.rules:
            prompt += " Ты уже помнишь правила:" + " ".join(self.rules)
        return prompt

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
        min_number_of_messages = self.config.get('min_number_of_messages', 10)
        new_history = [{"role": "system", "content": self.get_system_prompt()}]
        self.message_history.popleft()
        while len(self.message_history) > min_number_of_messages:
            msg = self.message_history.popleft()
            summary_prompt += f"\n{msg['role']}: {msg['content']}"
        summary = self.ai_model.get_response([{"role": "user", "content": summary_prompt}])
        logger.info(f"Summarized conversation: {summary}")
        new_history.append({"role": "system", "content": f"Earlier conversation summary: {summary}"})
        while self.message_history:
            new_history.append(self.message_history.popleft())
        self.message_history = deque(new_history)

    async def get_response(self, text: str) -> List[Tuple[dict, str]]:
        """
        Get an AI response based on the current conversation state and new input.

        Args:
            text (str): The new input text to respond to.

        Returns:
            List[Tuple[dict, str]]: The AI-generated response, marked with emotion response
        """

        # update system message
        self.message_history[0] = {"role": "system", "content": self.get_system_prompt()}

        self.message_history.append({"role": "user", "content": text})

        while self.get_token_count(list(self.message_history)) > self.config.get('token_threshold', 2500):
            self.summarize_and_compress_history()

        logger.debug(f"Message history: \n{self.formatted_message_history()}")

        response_text = self.ai_model.get_response(list(self.message_history))
        logger.debug(f"AI response: {text} -> {response_text}")
        self.message_history.append({"role": "assistant", "content": response_text})

        _, search_results = await process_and_search(response_text, self.searcher)
        logger.info(f"Response Text: {_}; Search results: {search_results}")

        if search_results:
            result_message = f"результаты поиска: {search_results[0]}"
            self.message_history.append({"role": "system", "content": result_message})
            self.message_history.append({"role": "user", "content": "?"})
            logger.debug(self.formatted_message_history())
            response_text = self.ai_model.get_response(list(self.message_history))
            self.message_history.pop()
            self.message_history.append({"role": "assistant", "content": response_text})

        response_text, facts = extract_facts(response_text)
        self.facts += facts
        self.save_facts(self.facts)

        if facts:
            logger.info(f"Extracted facts: {facts}")

        response_text, rules = extract_rules(response_text)
        self.rules += rules
        self.save_rules(self.rules)

        if rules:
            logger.info(f"Extracted rules: {rules}")

        text_with_emotions = extract_emotions(response_text)
        # logger.info(f"Extracted emotions: {text_with_emotions}")
        # response_text = " ".join([t for e, t in text_with_emotions])
        #
        # logger.debug(f"AI response: {text} -> {response_text}")

        print("\n" + self.formatted_message_history() + "\n")

        return text_with_emotions

    def formatted_message_history(self):
        """
        Format the message history for logging purposes.

        Returns:
            str: A formatted string representation of the message history.
        """
        return "\n\n".join([f'{msg["role"]}:{msg["content"].strip()}' for msg in self.message_history])

    def load_facts(self):
        try:
            with open('facts.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_facts(self, facts):
        with open('facts.json', 'w', encoding='utf8') as f:
            json.dump(facts, f, ensure_ascii=False, indent=4)

    def load_rules(self):
        try:
            with open('rules.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_rules(self, rules):
        with open('rules.json', 'w', encoding='utf8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=4)


def test():
    print(get_current_datetime_english(get_timezone()) + " " + get_location())


if __name__ == '__main__':
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    test()
