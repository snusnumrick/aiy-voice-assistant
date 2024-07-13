"""
Conversation Manager module.

This module provides the ConversationManager class for managing the flow of conversation,
including message history, token counting, and interaction with AI models.
"""

import json
import logging
import os
import re
import sys
from collections import deque
from typing import List, Dict, Tuple

from src.responce_player import extract_emotions

if __name__ == "__main__":
    # add current directory to python path
    sys.path.append(os.getcwd())

from src.ai_models import AIModel
from src.config import Config
from src.web_search import WebSearcher, process_and_search
from src.llm_tools import summarize_and_compress_history
from src.ai_models import AIModel, ClaudeAIModel
from src.tools import get_token_count, get_location, get_timezone, get_current_datetime_english, \
    get_current_date_time_for_facts

logger = logging.getLogger(__name__)


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


def extract_rules(text: str, current_timezone: str) -> Tuple[str, List[str]]:
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
        rule = get_current_date_time_for_facts(current_timezone) + " : " + match
        extracted_rules.append(rule)

    # Remove all {remember: xxx} substrings from the input string
    modified_text = re.sub(pattern, '', text)

    # Remove any extra whitespace that might have been left
    modified_text = ' '.join(modified_text.split())

    return modified_text, extracted_rules


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
        self.summarize_model = ClaudeAIModel(config)
        self.facts = self.load_facts()
        self.rules = self.load_rules()
        self.location = get_location()
        self.timezone = get_timezone()
        # self.hard_rules = (""
        #                    # "Если чтобы ответить на мой вопрос, тебе нужно поискать в интернете, не отвечай сразу, "
        #                    # "а пошли мне сообщение в таком формате: "
        #                    # "$internet query:<что ты хочешь поискать на английском языке>$. "
        #                    # "Таких запросов в твоем сообщении может быть несколько. "
        #                    "Если в ответе на твой запрос указано время без указания часового пояса, "
        #                    "считай что это Восточное стандартное время."
        #                    # "Если по этому запросу не нашел нужной информации, попробуй переформулировать запрос. "
        #                    "Если тебе надо что-то запомнить, "
        #                    "пошли мне сообщение в таком формате: $remember: <текст, который тебе нужно запомнить>$. "
        #                    "Таких фактов в твоем сообщении тоже может быть несколько. "
        #                    "Например, $remember: <первый текст, который тебе нужно запомнить>$ "
        #                    "{remember: $второрй текст, который тебе нужно запомнить>$."
        #                    "Если я прошу тебя как-то поменятся (например, не используй обсценную лексику); "
        #                    "чтобы запомнить это новое правило, пошли мне сообщение в таком формате: "
        #                    "$rule: <текст нового правила>$. "
        #                    "Таких запросов в твоем сообщении тоже может быть несколько. ")
        self.hard_rules = ("For web searches: $internet query:<query in English>$. "
                           "To remember: $remember:<text>$. For new rules: $rule:<text>$ ")
        # self.default_system_prompt = ("Тебя зовут Кубик. Ты мой друг и помощник. Ты умеешь шутить и быть саркастичным. "
        #                               " Отвечай естественно, как в устной речи. "
        #                               "Говори максимально просто и понятно. Не используй списки и нумерации. "
        #                               "Например, не говори 1. что-то; 2. что-то. говори во-первых, во-вторых "
        #                               "или просто перечисляй. "
        #                               "При ответе на вопрос где важно время, помни какое сегодня число. "
        #                               "Если чего-то не знаешь, так и скажи. "
        #                               "Я буду разговаривать с тобой через голосовой интерфейс. "
        #                               "Будь краток, избегай банальностей и непрошенных советов. ")
        self.default_system_prompt = ("You're Kubik, my friendly AI assistant. Be witty and sarcastic. "
                                      "Speak naturally, simply. Avoid lists. Consider date in time-sensitive answers. "
                                      "Admit unknowns. I use voice interface. Be brief, avoid platitudes. "
                                      "Use internet searches when needed for up-to-date or specific information. "
                                      "Assume EST if timezone unspecified. Treat responses as spoken.")
        self.message_history = deque([{"role": "system", "content": self.get_system_prompt()}])

    def get_system_prompt(self):
        from src.responce_player import emotions_prompt

        return self.config.get('system_prompt', self.default_system_prompt)

        prompt = f"{get_current_datetime_english(self.timezone)} {self.location} "
        prompt += self.config.get('system_prompt', self.default_system_prompt)
        prompt += self.hard_rules
        prompt += emotions_prompt()
        if self.facts:
            prompt += " Ты уже знаешь факты:" + " ".join(self.facts)
        if self.rules:
            prompt += " Ты уже помнишь правила:" + " ".join(self.rules)
        return prompt

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

        if get_token_count(list(self.message_history)) > self.config.get('token_threshold', 2500):
            self.message_history = summarize_and_compress_history(self.message_history, self.summarize_model,
                                                                  self.config)

        logger.debug(f"Message history: \n{self.formatted_message_history()}")

        response_text = ""
        async for response_part in self.ai_model.get_response_async(list(self.message_history)):
            response_text += response_part

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

        response_text, rules = extract_rules(response_text, self.timezone)
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
