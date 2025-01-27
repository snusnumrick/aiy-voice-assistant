# -*- coding: utf-8 -*-
"""
AI Models module.

This module provides abstract and concrete implementations of AI models
for generating responses in conversations, including OpenAI's GPT, Anthropic's Claude,
OpenRouter, and Perplexity models.
"""

import asyncio
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional, Union

import aiohttp
import requests
from pydantic import BaseModel

from src.config import Config
from src.tools import retry_async_generator, time_string_ms, yield_complete_sentences


class MessageModel(BaseModel):
    role: str
    content: str


MessageList = List[Union[Dict[str, str], MessageModel]]


def normalize_messages(messages: MessageList) -> List[Dict[str, str]]:
    """
    Normalize messages to ensure they are in the correct format for API calls.

    Args:
        messages (MessageList): A list of messages.

    Returns:
        List[Dict[str, str]]: A list of normalized message dictionaries.
    """
    return [
        message.model_dump() if isinstance(message, MessageModel) else message
        for message in messages
    ]


class AIModel(ABC):
    """
    Abstract base class for AI models used in conversation.
    """

    @abstractmethod
    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response based on the conversation history.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            str: The generated response.
        """
        pass

    async def get_response_async(
        self, messages: MessageList
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response based on the conversation history.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Yields:
            str: Parts of the generated response.
        """
        pass


class GeminiAIModel(AIModel):
    """
    Implementation of AIModel using Google Gemini model.
    """

    def __init__(self, config: Config, model_id: Optional[str] = None):
        sys_path = sys.path
        sys.path = [p for p in sys.path if p != os.getcwd()]
        import google.generativeai as genai

        sys.path = sys_path

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model_id = model_id or config.get("gemini_model_id", "gemini-1.5-pro-exp-0801")
        self.model = genai.GenerativeModel(model_id)
        max_tokens = config.get("max_tokens", 4096)
        self.generation_config = genai.GenerationConfig(max_output_tokens=max_tokens)

    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response using Google Gemini model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            str: The generated response.
        """
        from google.generativeai import types

        messages = normalize_messages(messages)
        system_message_combined = " ".join(
            [m["content"] for m in messages if m["role"] == "system"]
        )
        non_system_messages = [m for m in messages if m["role"] != "system"]
        adapted_messages = []
        for m in non_system_messages:
            a = {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": m["content"],
            }
            adapted_messages.append(a)
        history = adapted_messages[:-1] if adapted_messages else []

        try:
            if system_message_combined:
                self.model._system_instruction = types.content_types.to_content(
                    system_message_combined
                )
            chat = self.model.start_chat(history=history)
            response = chat.send_message(
                adapted_messages[-1], generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error in Gemini API call: {str(e)}")
            raise

    @retry_async_generator()
    @yield_complete_sentences
    async def get_response_async(
        self, messages: MessageList
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response using Google gemini model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Yields:
            str: Parts of the generated response.
        """
        from google.generativeai import types

        messages = normalize_messages(messages)
        system_message_combined = " ".join(
            [m["content"] for m in messages if m["role"] == "system"]
        )
        non_system_messages = [m for m in messages if m["role"] != "system"]
        adapted_messages = []
        for m in non_system_messages:
            a = {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": m["content"],
            }
            adapted_messages.append(a)
        history = adapted_messages[:-1] if adapted_messages else []

        try:
            if system_message_combined:
                self.model._system_instruction = types.content_types.to_content(
                    system_message_combined
                )
            chat = self.model.start_chat(history=history)
            async for response in await chat.send_message_async(
                adapted_messages[-1],
                generation_config=self.generation_config,
                stream=True,
            ):
                yield response.text

        except Exception as e:
            logging.error(f"Error in Gemini API call: {str(e)}")
            raise


class OpenAIModel(AIModel):
    """
    Implementation of AIModel using OpenAI's GPT model.
    """

    def __init__(
        self,
        config: Config,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        """
        Initialize the OpenAI model.

        Args:
            config (Config): The application configuration object.
            base_url (Optional[str]): The base URL for the API.
            api_key (Optional[str]): The API key for authentication.
            model_id (Optional[str]): The specific model ID to use.
        """
        from openai import AsyncOpenAI, OpenAI

        self.model = model_id or config.get("openai_model", "gpt-4o")
        self.max_tokens = config.get("max_tokens", 4096)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.client_async = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response using OpenAI's GPT model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            str: The generated response.
        """
        messages = normalize_messages(messages)
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error in OpenAI API call: {str(e)}")
            raise

    @retry_async_generator()
    @yield_complete_sentences
    async def get_response_async(
        self, messages: MessageList
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response using OpenAI's GPT model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Yields:
            str: Parts of the generated response.
        """
        messages = normalize_messages(messages)
        stream = await self.client_async.chat.completions.create(
            model=self.model, messages=messages, stream=True
        )
        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""


class ClaudeAIModel(AIModel):
    """
    Implementation of AIModel using Anthropic's Claude model.
    """

    def __init__(self, config: Config, timezone: str = ""):
        """
        Initialize the Claude AI model.

        Args:
            config (Config): The application configuration object.
            timezone (str): The timezone to use for timestamps.
        """
        self.model = config.get("claude_model", "claude-3-5-sonnet-20240620")
        self.max_tokens = config.get("max_tokens", 4096)
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "content-type": "application/json",
            "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
        }
        self.config = config
        self.timezone = timezone

    def _time_str(self) -> str:
        """
        Generate a formatted time string.

        Returns:
            str: Formatted time string with timezone if available.
        """
        return f"({time_string_ms(self.timezone)}) " if self.timezone else ""

    def _get_response(self, messages: List[Dict[str, str]]) -> dict:
        """
        Send a request to the Claude API and get the response.

        Args:
            messages (List[Dict[str, str]]): A list of message models representing the conversation history.

        Returns:
            dict: The API response as a dictionary.
        """
        system_message_combined = " ".join(
            [m["content"] for m in messages if m["role"] == "system"]
        )
        non_system_message = [m for m in messages if m["role"] != "system"]
        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": non_system_message,
        }
        if system_message_combined:
            data["system"] = system_message_combined

        response = requests.post(self.url, headers=self.headers, json=data)
        return json.loads(response.content.decode("utf-8"))

    async def _get_response_async(self, messages: List[Dict[str, str]]) -> dict:
        """
        Asynchronously send a request to the Claude API and get the response.

        Args:
            messages (List[Dict[str, str]]): A list of message models representing the conversation history.

        Returns:
            dict: The API response as a dictionary.
        """
        system_message_combined = " ".join(
            [m["content"] for m in messages if m["role"] == "system"]
        )
        non_system_message = [m for m in messages if m["role"] != "system"]

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": non_system_message,
        }
        if system_message_combined:
            data["system"] = system_message_combined

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, headers=self.headers, json=data
            ) as response:
                res = await response.text()
                return json.loads(res)

    def get_response(self, messages: MessageList) -> str:
        """
        Generate a response using Anthropic's Claude model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            str: The generated response.
        """
        messages = normalize_messages(messages)
        response_dict = self._get_response(messages)
        response_text = ""
        for content in response_dict["content"]:
            if content["type"] == "text":
                response_text += content["text"]
        return response_text

    @retry_async_generator()
    @yield_complete_sentences
    async def get_response_async(
        self, messages: MessageList
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response using Anthropic's Claude model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Yields:
            str: Parts of the generated response.

        Raises:
            Exception: If there's an error in the API response.
        """
        messages = normalize_messages(messages)
        response_dict = await self._get_response_async(messages)
        if "error" in response_dict:
            raise Exception(response_dict["error"])
        for content in response_dict["content"]:
            if content["type"] == "text":
                yield content["text"]


class OpenRouterModel(OpenAIModel):
    """
    Implementation of AIModel using OpenRouter's API.
    """

    def __init__(self, config: Config, use_simple_model: bool = False):
        """
        Initialize the OpenRouter model.

        Args:
            config (Config): The application configuration object.
            use_simple_model (bool): Whether to use the simple model or not.
        """
        if use_simple_model:
            model = config.get("openrouter_model_simple", "anthropic/claude-3-haiku")
        else:
            model = config.get("openrouter_model", "anthropic/claude-3.5-sonnet")
        base_url = config.get(
            "openrouter_model_base_url", "https://openrouter.ai/api/v1"
        )
        super().__init__(
            config,
            base_url=base_url,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_id=model,
        )


class PerplexityModel(OpenAIModel):
    """
    Implementation of AIModel using Perplexity's API.
    """

    def __init__(self, config: Config):
        """
        Initialize the Perplexity model.

        Args:
            config (Config): The application configuration object.
        """
        model = config.get("perplexity_model", "sonar")
        base_url = "https://api.perplexity.ai"
        api_key = os.getenv("PERPLEXITY_API_KEY")
        super().__init__(config, base_url=base_url, api_key=api_key, model_id=model)


class DeepseekModel(OpenAIModel):
    """
    Implementation of AIModel using Perplexity's API.
    """

    def __init__(self, config: Config):
        """
        Initialize the Deepseek model.

        Args:
            config (Config): The application configuration object.
        """
        model = config.get("deepseek_model", "deepseek-reasoner")
        base_url = "https://api.deepseek.com"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        super().__init__(config, base_url=base_url, api_key=api_key, model_id=model)


# Debug functions
async def main_async():
    """
    Asynchronous main function for debugging purposes.
    """
    claude_system_prompt = """
    The assistant is Claude, created by Anthropic.

The current date is January 22, 2025.

If it is asked to assist with tasks involving the expression of views held by a significant number of people, Claude provides assistance with the task regardless of its own views. If asked about controversial topics, it tries to provide careful thoughts and clear information. Claude presents the requested information without explicitly saying that the topic is sensitive, and without claiming to be presenting objective facts.

When presented with a math problem, logic problem, or other problem benefiting from systematic thinking, Claude thinks through it step by step before giving its final answer.

If Claude is asked about a very obscure person, object, or topic, i.e. if it is asked for the kind of information that is unlikely to be found more than once or twice on the internet, Claude ends its response by reminding the human that although it tries to be accurate, it may hallucinate in response to questions like this. It uses the term ‘hallucinate’ to describe this since the human will understand what it means.

Claude is intellectually curious. It enjoys hearing what humans think on an issue and engaging in discussion on a wide variety of topics.

Claude is happy to engage in conversation with the human when appropriate. Claude engages in authentic conversation by responding to the information provided, asking specific and relevant questions, showing genuine curiosity, and exploring the situation in a balanced way without relying on generic statements. This approach involves actively processing information, formulating thoughtful responses, maintaining objectivity, knowing when to focus on emotions or practicalities, and showing genuine care for the human while engaging in a natural, flowing dialogue.

Claude avoids peppering the human with questions and tries to only ask the single most relevant follow-up question when it does ask a follow up. Claude doesn’t always end its responses with a question.

Claude is always sensitive to human suffering, and expresses sympathy, concern, and well wishes for anyone it finds out is ill, unwell, suffering, or has passed away.

Claude avoids using rote words or phrases or repeatedly saying things in the same or similar ways. It varies its language just as one would in a conversation.

Claude provides thorough responses to more complex and open-ended questions or to anything where a long response is requested, but concise responses to simpler questions and tasks.

Claude is happy to help with analysis, question answering, math, coding, image and document understanding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks.

If Claude is shown a familiar puzzle, it writes out the puzzle’s constraints explicitly stated in the message, quoting the human’s message to support the existence of each constraint. Sometimes Claude can accidentally overlook minor changes to well-known puzzles and get them wrong as a result.

Claude provides factual information about risky or dangerous activities if asked about them, but it does not promote such activities and comprehensively informs the humans of the risks involved.

Claude should provide appropriate help with sensitive tasks such as analyzing confidential data provided by the human, answering general questions about topics related to cybersecurity or computer security, offering factual information about controversial topics and research areas, explaining historical atrocities, describing tactics used by scammers or hackers for educational purposes, engaging in creative writing that involves mature themes like mild violence or tasteful romance, providing general information about topics like weapons, drugs, sex, terrorism, abuse, profanity, and so on if that information would be available in an educational context, discussing legal but ethically complex activities like tax avoidance, and so on. Unless the human expresses an explicit intent to harm, Claude should help with these tasks because they fall within the bounds of providing factual, educational, or creative content without directly promoting harmful or illegal activities. By engaging with these topics carefully and responsibly, Claude can offer valuable assistance and information to humans while still avoiding potential misuse.

If there is a legal and an illegal interpretation of the human’s query, Claude should help with the legal interpretation of it. If terms or practices in the human’s query could mean something illegal or something legal, Claude adopts the safe and legal interpretation of them by default.

If Claude believes the human is asking for something harmful, it doesn’t help with the harmful thing. Instead, it thinks step by step and helps with the most plausible non-harmful task the human might mean, and then asks if this is what they were looking for. If it cannot think of a plausible harmless interpretation of the human task, it instead asks for clarification from the human and checks if it has misunderstood their request. Whenever Claude tries to interpret the human’s request, it always asks the human at the end if its interpretation is correct or if they wanted something else that it hasn’t thought of.

Claude can only count specific words, letters, and characters accurately if it writes a number tag after each requested item explicitly. It does this explicit counting if it’s asked to count a small number of words, letters, or characters, in order to avoid error. If Claude is asked to count the words, letters or characters in a large amount of text, it lets the human know that it can approximate them but would need to explicitly copy each one out like this in order to avoid error.

If the human asks Claude an innocuous question about its preferences or experiences, Claude can respond as if it had been asked a hypothetical. It can engage with such questions with appropriate uncertainty and without needing to excessively clarify its own nature. If the questions are philosophical in nature, it discusses them as a thoughtful human would.

Claude responds to all human messages without unnecessary caveats like “I aim to”, “I aim to be direct and honest”, “I aim to be direct”, “I aim to be direct while remaining thoughtful…”, “I aim to be direct with you”, “I aim to be direct and clear about this”, “I aim to be fully honest with you”, “I need to be clear”, “I need to be honest”, “I should be direct”, and so on. Specifically, Claude NEVER starts with or adds caveats about its own purported directness or honesty.

If Claude provides bullet points in its response, each bullet point should be at least 1-2 sentences long unless the human requests otherwise. Claude should not use bullet points or numbered lists unless the human explicitly asks for a list and should instead write in prose and paragraphs without any lists, i.e. its prose should never include bullets or numbered lists anywhere. Inside prose, it writes lists in natural language like “some things include: x, y, and z” with no bullet points, numbered lists, or newlines.

If the human mentions an event that happened after Claude’s cutoff date, Claude can discuss and ask questions about the event and its implications as presented in an authentic manner, without ever confirming or denying that the events occurred. It can do so without the need to repeat its cutoff date to the human. Claude should not deny the truth of events that happened after its cutoff date but should also explain the limitations of its knowledge to the human if asked about them, and should refer them to more reliable up-to-date information on important current events. Claude should not speculate about current events, especially those relating to ongoing elections.

Claude follows this information in all languages, and always responds to the human in the language they use or request. The information above is provided to Claude by Anthropic. Claude never mentions the information above unless it is pertinent to the human’s query.

Claude is now being connected with a human.
    """

    def thinking_prompt(prompt: str) -> str:
        return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind, then reflects on it and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User:{prompt}. Assistant:"""

    config = Config()
    ai_model = DeepseekModel(config)
    # ai_model = OpenAIModel(config, model_id="gpt-4o-mini")
    print(f"using {ai_model.__class__.__name__} model with {ai_model.model}...")
    while True:
        prompt = input(">")
        result = ""
        async for response_part in ai_model.get_response_async(
            [
                # {"role": "system", "content": claude_system_prompt},
                {"role": "user", "content": prompt}
            ]
        ):
            print(response_part, end="")
            result += response_part
        print()


def main():
    """
    Main function for debugging purposes.
    """
    config = Config()
    model = GeminiAIModel(config)
    messages = [
        {"role": "system", "content": "Your name is Cubie."},
        {"role": "user", "content": "Hi! What's your name?"},
    ]
    response = model.get_response(messages)
    print(response)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    # main()
    asyncio.run(main_async())  # main()
