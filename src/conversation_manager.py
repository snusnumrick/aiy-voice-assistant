import re
import os
import logging
from collections import deque
from abc import ABC, abstractmethod
from typing import List, Dict

if __name__ == '__main__':
    from config import Config
else:
    from .config import Config


logger = logging.getLogger(__name__)


class AIModel(ABC):
    @abstractmethod
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        pass


class OpenAIModel(AIModel):
    def __init__(self, config: Config):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = config.get('openai_model', 'gpt-4o')
        self.max_tokens = config.get('max_tokens', 4096)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()


class ClaudeAIModel(AIModel):
    def __init__(self, config: Config):
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.get('anthropic_api_key'))
        self.model = config.get('claude_model', 'claude-2.1')
        self.max_tokens = config.get('max_tokens', 1000)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        # Convert the message format from OpenAI to Anthropic
        prompt = ClaudeAIModel._convert_messages_to_prompt(messages)

        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens_to_sample=self.max_tokens
        )
        return response.completion.strip()

    def _convert_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
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


class ConversationManager:
    def __init__(self, config: Config, ai_model: AIModel):
        self.config = config
        self.ai_model = ai_model
        self.message_history = deque()
        self.message_history.append({"role": "system", "content": config.get('system_message', "Your name is Robi.")})

    def estimate_tokens_russian(self, text: str) -> int:
        words = len(re.findall(r'\b\w+\b', text))
        punctuation = len(re.findall(r'[.,!?;:"]', text))
        return int(words * 1.5 + punctuation)

    def get_token_count(self, messages: List[Dict[str, str]]) -> int:
        return sum(self.estimate_tokens_russian(msg["content"]) for msg in messages)

    def summarize_and_compress_history(self):
        summary_prompt = self.config.get('summary_prompt',
                                         "Сделайте краткое изложение ключевых моментов этой беседы, "
                                         "сосредоточившись на самых важных фактах и контексте. Будьте лаконичны:")
        for msg in self.message_history:
            if msg["role"] != "system":
                summary_prompt += f"\n{msg['role']}: {msg['content']}"

        summary = self.ai_model.get_response([{"role": "user", "content": summary_prompt}])

        system_message = self.message_history[0]
        self.message_history.clear()
        self.message_history.append(system_message)
        self.message_history.append({"role": "system", "content": f"Краткое изложение беседы: {summary}"})

        logger.info(f"Summarized conversation: {summary}")

    def formatted_message_history(self):
        return "\n".join([f'{y["role"]}:{y["content"].strip()}' for y in self.message_history])

    def get_response(self, text: str) -> str:
        self.message_history.append({"role": "user", "content": text})

        while self.get_token_count(list(self.message_history)) > self.config.get('token_threshold', 2500):
            self.summarize_and_compress_history()

        logger.info(f"Message history: \n{self.formatted_message_history()}")

        response_text = self.ai_model.get_response(list(self.message_history))
        self.message_history.append({"role": "assistant", "content": response_text})

        logger.debug(f"AI response: {text} -> {response_text}")
        return response_text


def test():
    config = Config()
    ai_model = OpenAIModel(config)
    conversation_manager = ConversationManager(config, ai_model)

    while True:
        text = input("You: ")
        if text.lower() in ['exit', 'quit', 'bye']:
            break
        response = conversation_manager.get_response(text)
        print(f"Robi: {response}")


if __name__ == '__main__':
    x =  deque([{'role': 'assistant', 'content': 'Почти! Столица Канады — Оттава. \n\nПродолжим? Следующая страна: Бразилия.\n\nКакая у неё столица?'}, {'role': 'user', 'content': 'Бразилия\n'}, {'role': 'assistant', 'content': 'Столица Бразилии — Бразилиа.\n\nПродолжим? Назови столицу Австралии.'}, {'role': 'user', 'content': 'Комбинация\n'}, {'role': 'assistant', 'content': 'На самом деле, столица Австралии — Канберра.\n\nПродолжим? Назови, пожалуйста, столицу Индии.'}, {'role': 'user', 'content': 'Не удали.\n'}, {'role': 'assistant', 'content': 'Конечно, продолжим! Столица Индии — Нью-Дели.\n\nТеперь твоя очередь: какая столица Турции?'}, {'role': 'user', 'content': 'Долица Турции это Анкара.\n'}, {'role': 'assistant', 'content': 'Верно! Столица Турции — это Анкара.\n\nТеперь давай попробуем следующую страну: какая столица Египта?'}, {'role': 'user', 'content': 'Солица Египта это Каир.\n'}], maxlen=10)
    print("\n".join([f'{y["role"]}:{y["content"].strip()}' for y in x]))


    from dotenv import load_dotenv

    load_dotenv()
    test()
