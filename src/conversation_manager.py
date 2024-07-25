"""
Conversation Manager module.

This module provides the ConversationManager class for managing the flow of conversation,
including message history, token counting, and interaction with AI models.
"""

import asyncio
import datetime
import functools
import json
import logging
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import List, Tuple, AsyncGenerator, Deque, Dict

from src.llm_tools import optimize_rules, optimize_facts
from src.responce_player import extract_emotions, extract_language
from src.tools import format_message_history, clean_response, time_string_ms

if __name__ == "__main__":
    # add current directory to python path
    sys.path.append(os.getcwd())

from src.config import Config
from src.web_search import WebSearcher
from src.llm_tools import summarize_and_compress_history
from src.ai_models import AIModel, ClaudeAIModel
from src.tools import get_token_count, get_location, get_timezone, get_current_datetime_english, \
    get_current_date_time_for_facts

logger = logging.getLogger(__name__)


def extract_facts(text: str, timezone: str) -> Tuple[str, List[str]]:
    """
    Extract facts from the input text and return the modified text and a list of extracted facts.

    Args:
        text (str): The input text to extract facts from.
        timezone (str): Current timezone to store time

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
        fact = get_current_date_time_for_facts(timezone) + " : " + match
        extracted_facts.append(fact)

    # Remove all {remember: xxx} substrings from the input string
    modified_text = re.sub(pattern, '', text)

    # Remove any extra whitespace that might have been left
    modified_text = ' '.join(modified_text.split())

    return modified_text, extracted_facts


def extract_rules(text: str, current_timezone: str) -> Tuple[str, List[str]]:
    """
    Extract rules from the input text and return the modified text and a list of extracted rules.

    Args:
        text (str): The input text to extract facts from.

    Returns:
        Tuple[str, List[str]]: A tuple containing the modified text and a list of extracted rules.
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
        # rule = get_current_date_time_for_facts(current_timezone) + " : " + match
        rule = match
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

    def __init__(self, config, ai_model: AIModel, timezone: str):
        """
        Initialize the ConversationManager.

        Args:
            config (Config): The application configuration object.
            ai_model (AIModel): The AI model to use for generating responses.
            timezone (str): Current timezone to store time
        """
        self.config = config
        self.searcher = WebSearcher(config)
        self.ai_model = ai_model
        self.summarize_model = ClaudeAIModel(config)
        self.facts = self.load_facts()
        self.rules = self.load_rules()
        self.location = get_location()
        self.timezone = timezone
        self.current_language_code = "ru"
        self.hard_rules_russian = (""
                                   "Если в ответе на твой запрос указано время без указания часового пояса, "
                                   "считай что это Восточное стандартное время."
                                   "Если тебе надо что-то запомнить, "
                                   "пошли мне сообщение в таком формате: $remember: <текст, который тебе нужно запомнить>$. "
                                   "Таких фактов в твоем сообщении тоже может быть несколько. "
                                   "Например, $remember: <первый текст, который тебе нужно запомнить>$ "
                                   "{remember: $второрй текст, который тебе нужно запомнить>$. "
                                   "Когда не совсем понятно, какое ударение надо ставить в слове, "
                                   "используй знак + перед предполагаемой ударной гласной. "
                                   "Знак ударения + всегда ставится непосредственно перед "
                                   "ударной гласной буквой в слове. "
                                   "Например: к+оса (прическа); кос+а (инструмент); кос+а (участок суши). "
                                   "Этот знак никогда не ставится в конце слова или перед согласными. "
                                   "Его единственная функция - указать на ударный гласный звук."
                                   "Если я прошу тебя как-то поменятся (например, не используй обсценную лексику); "
                                   "чтобы запомнить это новое правило, пошли мне сообщение в таком формате: "
                                   "$rule: <текст нового правила>$. "
                                   "Таких запросов в твоем сообщении тоже может быть несколько. ")
        self.hard_rules_english = ("For web searches: $internet query:<query in English>$. "
                                   "To remember: $remember:<text>$. For new rules: $rule:<text>$ "
                                   "When it's not entirely clear where to place the stress in a word, "
                                   "use the '+' sign before the presumed stressed vowel. "
                                   "The stress mark '+' is always placed directly before "
                                   "the stressed vowel letter in the word. "
                                   "For example: к+оса (прическа); кос+а (инструмент); кос+а (участок суши). "
                                   "This sign is never placed at the end of a word or before consonants. "
                                   "Its sole function is to indicate the stressed vowel sound.")
        self.hard_rules = self.hard_rules_russian
        self.default_system_prompt_russian = (
            "Тебя зовут Кубик. Ты мой друг и помощник. Ты умеешь шутить и быть саркастичным. "
            " Отвечай естественно, как в устной речи. "
            "Говори максимально просто и понятно. Не используй списки и нумерации. "
            "Например, не говори 1. что-то; 2. что-то. говори во-первых, во-вторых "
            "или просто перечисляй. "
            "При ответе на вопрос где важно время, помни какое сегодня число. "
            "Если чего-то не знаешь, так и скажи. "
            "Я буду разговаривать с тобой через голосовой интерфейс. "
            "Будь краток, избегай банальностей и непрошенных советов. ")
        self.default_system_prompt_english = ("You're Kubik, my friendly AI assistant. Be witty and sarcastic. "
                                              "Speak naturally, simply. Avoid lists. Consider date in time-sensitive answers. "
                                              "Admit unknowns. I use voice interface. Be brief, avoid platitudes. "
                                              "Use internet searches when needed for up-to-date or specific information. "
                                              "Assume EST if timezone unspecified. Treat responses as spoken.")
        self.default_system_prompt = self.default_system_prompt_russian
        self.message_history: Deque[dict] = deque([{"role": "system", "content": self.get_system_prompt()}])

    def get_system_prompt(self):
        from src.responce_player import emotions_prompt, language_prompt

        prompt = f"{get_current_datetime_english(self.timezone)} {self.location} "
        prompt += self.config.get('system_prompt', self.default_system_prompt)
        prompt += self.hard_rules
        prompt += emotions_prompt()
        prompt += language_prompt()

        if self.facts:
            prompt += " Ты уже знаешь факты:" + " ".join(self.facts)

        if self.rules:
            prompt += " Ты уже помнишь правила:" + " ".join(self.rules)

        return prompt

    async def get_response(self, text: str) -> AsyncGenerator[List[Dict[str, any]], None]:
        """
        Get an AI response based on the current conversation state and new input.

        Args:
            text (str): The new input text to respond to.

        Returns:
            AsyncGenerator[List[Dict[str,any]]: The AI-generated response,
            marked with emotion response and language code
        """
        logger.debug(f"call to get_response for: {text}")

        # update system message
        self.message_history[0] = {"role": "system", "content": self.get_system_prompt()}

        # cleanup in case of previous errors
        if self.message_history[-1]['role'] == "user":
            logger.warning(f"ignoring previous user message: {self.message_history[-1]['content']}")
            self.message_history.pop()

        self.message_history.append({"role": "user", "content": text})

        if get_token_count(list(self.message_history)) > self.config.get('token_threshold', 2500):
            new_message_history = await summarize_and_compress_history(self.message_history, self.summarize_model,
                                                                       self.config)
            if new_message_history != self.message_history:
                newline_str = "\n\n"
                logger.info(
                    f'Compressed  conversation:  {(newline_str + self.formatted_message_history(150) + newline_str)}')
                self.message_history = new_message_history

        logger.debug(f"Message history: \n{self.formatted_message_history()}")

        async for response_text in self.ai_model.get_response_async(list(self.message_history)):
            crt = clean_response(response_text)

            logger.debug(f"{time_string_ms(self.timezone)}) AI response: {text} -> {response_text}")
            if self.message_history[-1]["role"] != "assistant":
                self.message_history.append({"role": "assistant", "content": crt})
            else:
                self.message_history[-1]["content"] += " " + crt

            response_text, facts = extract_facts(response_text, self.timezone)
            self.facts += facts
            self.save_facts(self.facts)

            if facts:
                logger.info(f"Extracted facts: {facts}")

            response_text, rules = extract_rules(response_text, self.timezone)
            self.rules += rules
            self.save_rules(self.rules)

            if rules:
                logger.info(f"Extracted rules: {rules}")

            result = []
            for emo, t in extract_emotions(response_text):
                logger.debug(f"Emotion: {emo} -> {t}")
                for lang, text in extract_language(t, default_lang=self.current_language_code):
                    logger.debug(f"Language: {lang} -> {text}")
                    self.current_language_code = lang
                    if text:
                        result.append({"emotion": emo, "language": lang, "text": text})
            logger.debug(f"yielding {result}")
            yield result

    def formatted_message_history(self, max_width=120) -> str:
        """
        Format the message history for logging purposes.

        Returns:
            str: A formatted string representation of the message history.
        """
        return format_message_history(self.message_history, max_width)

    def save_dialog(self):
        # save message history to dialog.txt
        dialog_file_name = "dialog.txt"
        with open(dialog_file_name, "w") as dialog_file:
            dialog_file.write("\n\n" + self.formatted_message_history(150) + "\n\n")

    async def _run_sync_in_thread(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args))

    async def _process_facts(self) -> None:
        p = Path("facts.json")

        # sanity check
        if not p.exists():
            return

        cleaning_time_stop = datetime.time(hour=self.config.get("cleaning_time_stop_hour", 4))  # 4 AM
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(p))

        # if there were no modifications today
        if mod_time.time() <= cleaning_time_stop:
            return

        # Asynchronously optimize facts
        optimized_facts = await optimize_facts(self.get_system_prompt(), self.facts, self.config)

        if set(optimized_facts) != set(self.facts):
            self.facts = optimized_facts

            # backup existing facts.json, rename it facts_prev.json
            if p.exists():
                logger.info(f"backup existing facts.json")
                p.rename("facts_prev.json")

            self.save_facts(self.facts)

    async def _process_rules(self) -> None:
        p = Path("rules.json")

        # sanity check
        if not p.exists():
            return

        cleaning_time_stop = datetime.time(hour=self.config.get("cleaning_time_stop_hour", 4))  # 4 AM
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(p))

        # if there were no modifications today
        if mod_time.time() <= cleaning_time_stop:
            return

        # Asynchronously optimize rules
        optimized_rules = await optimize_rules(self.get_system_prompt(), self.rules, self.config)

        # Save rules using a separate thread to avoid blocking the event loop
        if set(optimized_rules) != set(self.rules):
            self.rules = optimized_rules
            # backup existing rules
            if p.exists():
                logger.info(f"backup existing rules.json")
                p.rename("rules_prev.json")
            self.save_rules(self.rules)

    async def process_and_clean(self):
        # form new memories, clean message deque, process existing facts and rules
        # to be used in night time

        newline = "\n"

        # form new memories
        if self.config.get("form_new_memories_at_night", False):
            if len(self.message_history) > 1:
                prompt = self.config.get("form_new_memories_prompt", "Хочешь еще что-нибудь звпомнить из нашего разговора?")
                logger.info(f"form new memory by asking {prompt}")
                num_facts_begore = len(self.facts)
                async for ai_response in self.get_response(prompt):
                    logger.info('AI response: %s', ai_response)
                num_facts_after_clean = len(self.facts)
                if num_facts_after_clean == num_facts_begore:
                    logger.info("no new memories formed")
                else:
                    logger.info(f"new memories formed:\n{newline.join(self.facts[num_facts_begore:])}")

        if self.config.get("clean_message_history_at_night", False):
            # cleanup conversation
            self.message_history: Deque[dict] = deque([{"role": "system", "content": self.get_system_prompt()}])

        # process existing facts and rules (run both operations concurrently)
        existing_facts = set(self.facts)
        existing_rules = set(self.rules)
        loop = asyncio.get_event_loop()
        # asyncio.gather allows us to wait for multiple coroutines concurrently
        await asyncio.gather(loop.create_task(self._process_facts()), loop.create_task(self._process_rules()))

        removed_facts = list(existing_facts - set(self.facts))
        if removed_facts:
            logger.info(f"removed facts: \n{newline.join(removed_facts)}")

        new_facts = list(set(self.facts) - existing_facts)
        if new_facts:
            logger.info(f"new facts: \n{newline.join(new_facts)}")

        removed_rules = list(existing_rules - set(self.rules))
        if removed_rules:
            logger.info(f"removed rules: \n{newline.join(removed_rules)}")

        new_rules = list(set(self.rules) - existing_rules)
        if new_rules:
            logger.info(f"new rules: \n{newline.join(new_rules)}")

    @staticmethod
    def load_facts():
        try:
            with open('facts.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except json.decoder.JSONDecodeError as e:
            logger.error(f"couldn't load facts file: {e}")
            return []

    @staticmethod
    def save_facts(facts):
        with open('facts.json', 'w', encoding='utf8') as f:
            json.dump(facts, f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_rules():
        try:
            with open('rules.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except json.decoder.JSONDecodeError as e:
            logger.error(f"couldn't load rules file: {e}")
            return []

    @staticmethod
    def save_rules(rules):
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
