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
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Union

import aiohttp
import requests
from pydantic import BaseModel

from src.config import Config
from src.tools import retry_async_generator, time_string_ms, yield_complete_sentences

# Compatibility shim: ensure openai module has attributes used by tests even on older SDKs
try:
    import openai as _openai  # type: ignore
    if not hasattr(_openai, "OpenAI"):
        setattr(_openai, "OpenAI", object)
    if not hasattr(_openai, "AsyncOpenAI"):
        setattr(_openai, "AsyncOpenAI", object)
except Exception:
    pass


class MessageModel(BaseModel):
    role: str
    content: str


MessageList = List[Union[Dict[str, str], MessageModel]]


class ReasoningEffort(Enum):
    """Abstracted reasoning-effort levels for internal use.

    Values map to OpenAI Responses API efforts as follows:
    - quick -> minimal
    - thorough -> medium
    - comprehensive -> high
    """
    QUICK = "quick"
    THOROUGH = "thorough"
    COMPREHENSIVE = "comprehensive"


def to_reasoning_effort(value: Optional[Union[str, "ReasoningEffort"]]) -> Optional["ReasoningEffort"]:
    """Convert a string ("quick", "thorough", or "comprehensive") into a ReasoningEffort enum.

    - Returns the value unchanged if it is already a ReasoningEffort.
    - Performs case-insensitive matching for the three supported values.
    - Returns None for unsupported strings or None input.
    """
    if value is None:
        return None
    if isinstance(value, ReasoningEffort):
        return value
    try:
        key = str(value).strip().lower()
    except Exception:
        return None
    mapping = {
        "quick": ReasoningEffort.QUICK,
        "thorough": ReasoningEffort.THOROUGH,
        "comprehensive": ReasoningEffort.COMPREHENSIVE,
    }
    return mapping.get(key)


def _to_openai_reasoning_effort(value: Optional[Union[str, ReasoningEffort]]) -> Optional[str]:
    """Normalize internal reasoning effort to OpenAI Responses API effort string.

    Accepts:
    - ReasoningEffort enum (preferred)
    - Internal strings: "quick", "thorough", "comprehensive"
    - OpenAI strings: "minimal", "low", "medium", "high" (pass-through)

    Returns one of: "minimal", "low", "medium", "high" or None.
    """
    if value is None:
        return None
    # Enum -> string
    if isinstance(value, ReasoningEffort):
        v = value.value
    else:
        v = str(value)
    key = v.strip().lower()
    # Direct pass-through for OpenAI-supported values
    if key in {"minimal", "low", "medium", "high"}:
        return key
    # Map internal names to OpenAI
    mapping = {
        "quick": "minimal",
        # The spec requests only three internal levels; map thorough to medium.
        "thorough": "medium",
        "comprehensive": "high",
    }
    return mapping.get(key)


def normalize_messages(messages: MessageList) -> List[Dict[str, str]]:
    """
    Normalize messages to ensure they are in the correct format for API calls.

    Args:
        messages (MessageList): A list of messages.

    Returns:
        List[Dict[str, str]]: A list of normalized message dictionaries.
    """
    return [
        (getattr(message, "model_dump", None)() if getattr(message, "model_dump", None) else message.dict())
        if isinstance(message, MessageModel) else message
        for message in messages
    ]


class AIModel(ABC):
    """
    Abstract base class for AI models used in conversation.
    """

    @abstractmethod
    def get_response(self, messages: MessageList, reasoning_effort: Optional[Union[str, ReasoningEffort]] = None) -> str:
        """
        Generate a response based on the conversation history.

        Args:
            messages (MessageList): A list of message models representing the conversation history.
            reasoning_effort (Optional[Union[str, ReasoningEffort]]): Optional per-call override for reasoning depth/effort. Implementations may ignore it.

        Returns:
            str: The generated response.
        """
        pass

    async def get_response_async(
        self, messages: MessageList,
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response based on the conversation history.

        Args:
            messages (MessageList): A list of message models representing the conversation history.
            reasoning_effort (Optional[Union[str, ReasoningEffort]]): Optional per-call override for reasoning depth/effort. Implementations may ignore it.

        Yields:
            str: Parts of the generated response.
        """
        pass

    def get_tokens_number(self, messages: MessageList) -> int:
        """
        Count the number of tokens in the given messages.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            int: The number of tokens in the messages.

        Raises:
            NotImplementedError: If the model implementation doesn't support token counting.
        """
        raise NotImplementedError(f"Token counting is not supported by {self.__class__.__name__}")


class GeminiAIModel(AIModel):
    """
    Implementation of AIModel using Google Gemini model.
    """

    def __init__(self, config: Config, model_id: Optional[str] = None, max_tokens: Optional[int] = None):
        sys_path = sys.path
        sys.path = [p for p in sys.path if p != os.getcwd()]
        import google.generativeai as genai

        sys.path = sys_path

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model_id = model_id or config.get("gemini_model_id", "gemini-1.5-pro-exp-0801")
        self.model = genai.GenerativeModel(model_id)
        max_tokens = max_tokens or config.get("max_tokens", 4096)
        self.generation_config = genai.GenerationConfig(max_output_tokens=max_tokens)

    def get_response(self, messages: MessageList, reasoning_effort: Optional[Union[str, ReasoningEffort]] = None) -> str:
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
        self, messages: MessageList,
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
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
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
        api: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ): 
        """
        Initialize the OpenAI model.

        Args:
            config (Config): The application configuration object.
            base_url (Optional[str]): The base URL for the API.
            api_key (Optional[str]): The API key for authentication.
            model_id (Optional[str]): The specific model ID to use.
            reasoning_effort (Optional[Union[str, ReasoningEffort]]): Preferred reasoning effort. Accepts ReasoningEffort (quick, thorough, comprehensive) or a string; mapped to OpenAI's "minimal"|"low"|"medium"|"high" for the Responses API.
            api (Optional[str]): Which OpenAI API to use: "responses" (default) or "chat_completions"/"completions".
            max_tokens (Optional[int]): The maximum number of tokens to generate.
        """
        # Import module to avoid ImportError on old SDKs and allow graceful fallback
        import importlib
        openai = importlib.import_module("openai")

        self.model = model_id or config.get("openai_model", "gpt-4o")
        logging.info(f"Using model: {self.model}")
        self.max_tokens = max_tokens or config.get("max_tokens", 4096)
        # Preserve for REST fallback
        self.base_url = base_url or getattr(openai, "base_url", None) or "https://api.openai.com/v1"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # API mode selection: default to "responses" to preserve current behavior/tests
        # Optionally allow override from config key "openai_api" if provided
        api_from_cfg = None
        try:
            api_from_cfg = config.get("openai_api")
        except Exception:
            api_from_cfg = None
        api_choice = (api or api_from_cfg or "responses").lower()
        if api_choice in ("chat_completions", "completions", "chat"):
            self.api_mode = "chat_completions"
        else:
            self.api_mode = "responses"
        # Optional reasoning effort for Responses API (GPT‑5)
        # Accept enum or string (internal or OpenAI), normalize to OpenAI values
        _eff = (
            reasoning_effort
            if reasoning_effort is not None
            else (config.get("openai_reasoning_effort") or os.getenv("OPENAI_REASONING_EFFORT"))
        )
        self.reasoning_effort = _to_openai_reasoning_effort(_eff)

        # Initialize modern SDK clients (OpenAI / AsyncOpenAI). If instantiation fails, leave as None to allow REST fallback.
        OpenAI_cls = getattr(openai, "OpenAI", None)
        AsyncOpenAI_cls = getattr(openai, "AsyncOpenAI", None)

        self.client = None
        self.client_async = None
        if OpenAI_cls is not None:
            try:
                self.client = OpenAI_cls(base_url=base_url, api_key=api_key)
            except Exception:
                try:
                    self.client = OpenAI_cls()
                except Exception:
                    self.client = None
        if AsyncOpenAI_cls is not None:
            try:
                self.client_async = AsyncOpenAI_cls(base_url=base_url, api_key=api_key)
            except Exception:
                try:
                    self.client_async = AsyncOpenAI_cls()
                except Exception:
                    self.client_async = None

    def get_response(self, messages: MessageList, reasoning_effort: Optional[Union[str, ReasoningEffort]] = None) -> str:
        """
        Generate a response using OpenAI's GPT model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.
            reasoning_effort (Optional[Union[str, ReasoningEffort]]): Per-call override for reasoning effort.

        Returns:
            str: The generated response.
        """
        messages = normalize_messages(messages)

        # Per-call override for reasoning effort only; do not fall back to instance/config defaults
        eff = _to_openai_reasoning_effort(reasoning_effort) if reasoning_effort is not None else None
        # logging.info(f"Using reasoning effort: {eff}")

        def _messages_to_input(msgs: MessageList):
            # For Responses API, allow passing rich input; otherwise fall back to last user message
            try:
                return [{"role": m.get("role"), "content": m.get("content")} for m in msgs]
            except Exception:
                return msgs

        def _extract_responses_output(resp_obj) -> str:
            # Try to extract content from Responses API shape
            out_text = []
            try:
                output = getattr(resp_obj, "output", None) or resp_obj.get("output")
            except Exception:
                output = None
            if output:
                for item in output:
                    content = getattr(item, "content", None) or item.get("content")
                    if content:
                        for c in content:
                            text = getattr(c, "text", None) or c.get("text")
                            if text:
                                out_text.append(text)
            # Fallback to text field
            if not out_text:
                try:
                    text = getattr(resp_obj, "text", None) or resp_obj.get("text")
                    if text:
                        out_text.append(text)
                except Exception:
                    pass
            return "".join(out_text).strip()

        # Branch by API mode
        if self.api_mode == "chat_completions":
            # Prefer SDK Chat Completions API if available
            if self.client is not None and hasattr(self.client, "chat") and hasattr(self.client.chat, "completions") and hasattr(self.client.chat.completions, "create"):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                    )
                    # Extract first choice message content
                    content = None
                    try:
                        choices = getattr(response, "choices", None) or response.get("choices")
                    except Exception:
                        choices = None
                    if choices:
                        first = choices[0]
                        message = getattr(first, "message", None) or first.get("message")
                        if message:
                            content = getattr(message, "content", None) or message.get("content")
                    if content:
                        return str(content)
                except Exception as e:
                    logging.error(f"Error in OpenAI Chat Completions SDK call: {str(e)}")
                    raise
            # REST fallback to /chat/completions
            try:
                url = f"{self.base_url.rstrip('/')}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                }
                payload: Dict[str, Union[str, int, Dict, List]] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                }
                resp = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=(10, 180),
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices")
                if choices:
                    message = choices[0].get("message")
                    if message and message.get("content"):
                        return str(message["content"]) 
                return json.dumps(data)
            except Exception as e:
                logging.error(f"Error in OpenAI REST Chat Completions call: {str(e)}")
                raise
        else:
            # Responses API (default)
            # Prefer SDK Responses API if available
            if self.client is not None and hasattr(self.client, "responses") and hasattr(self.client.responses, "create"):
                try:
                    kwargs = {"model": self.model, "input": _messages_to_input(messages)}
                    if eff:
                        kwargs["reasoning"] = {"effort": eff}
                    response = self.client.responses.create(**kwargs)
                    text = _extract_responses_output(response)
                    if text:
                        return text
                except Exception as e:
                    logging.error(f"Error in OpenAI Responses SDK call: {str(e)}")
                    raise
            # REST fallback to /responses
            try:
                url = f"{self.base_url.rstrip('/')}/responses"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                }
                payload: Dict[str, Union[str, int, Dict, List]] = {
                    "model": self.model,
                    "input": _messages_to_input(messages),
                }
                if eff:
                    payload["reasoning"] = {"effort": eff}
                resp = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self._timeout_profile(eff),
                )
                resp.raise_for_status()
                data = resp.json()
                text = _extract_responses_output(data)
                if text:
                    return text
                return json.dumps(data)
            except Exception as e:
                logging.error(f"Error in OpenAI REST Responses call: {str(e)}")
                raise

    def _timeout_profile(self, eff: Optional[str]) -> tuple:
        """
        Return (connect_timeout, read/total timeout) based on reasoning effort.
        eff is already normalized to OpenAI values: "minimal" | "low" | "medium" | "high" | None
        """
        connect = 10  # seconds
        mapping = {
            "minimal": 120,  # quick
            "low": 180,
            "medium": 300,   # thorough
            "high": 600,     # comprehensive
        }
        read_total = mapping.get(eff, 180 if eff else 180)
        return connect, read_total

    @retry_async_generator()
    @yield_complete_sentences
    async def get_response_async(
        self, messages: MessageList,
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously generate a response using OpenAI's GPT model.

        Args:
            messages (MessageList): A list of message models representing the conversation history.
            reasoning_effort (Optional[Union[str, ReasoningEffort]]): Per-call override for reasoning depth/effort.
                Implementations may ignore it. For OpenAI Responses API, maps to effort levels:
                - quick -> minimal
                - thorough -> medium
                - comprehensive -> high

        Yields:
            str: Parts of the generated response.
        """
        # logging.info(f"get_response_async: {messages}")
        messages = normalize_messages(messages)

        # Per-call override for reasoning effort only; do not fall back to instance/config defaults
        eff = _to_openai_reasoning_effort(reasoning_effort) if reasoning_effort is not None else None
        # logging.info(f"Using reasoning effort: {eff}")

        # Branch by API mode (non-streaming emission)
        if self.api_mode == "chat_completions":
            # Try SDK Chat Completions first
            if self.client_async is not None and hasattr(self.client_async, "chat") and hasattr(self.client_async.chat, "completions") and hasattr(self.client_async.chat.completions, "create"):
                try:
                    resp_obj = await self.client_async.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                    )
                    content = None
                    try:
                        choices = getattr(resp_obj, "choices", None) or resp_obj.get("choices")
                    except Exception:
                        choices = None
                    if choices:
                        first = choices[0]
                        message = getattr(first, "message", None) or first.get("message")
                        if message:
                            content = getattr(message, "content", None) or message.get("content")
                    if content:
                        yield str(content)
                        return
                except Exception as e:
                    logging.error(f"Error in OpenAI Chat Completions SDK async call: {str(e)}")
                    raise
            # REST fallback
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            }
            payload: Dict[str, Union[str, int, Dict, List]] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
            }
            try:
                timeout_cfg = aiohttp.ClientTimeout(total=180)
                async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        choices = data.get("choices")
                        if choices:
                            message = choices[0].get("message")
                            if message and message.get("content"):
                                yield str(message["content"]) 
                                return
                        yield json.dumps(data)
                        return
            except Exception as e:
                logging.error(f"Error in OpenAI REST Chat Completions async call: {str(e)}")
                raise
        else:
            # Responses API (default)
            # Try SDK first
            if self.client_async is not None and hasattr(self.client_async, "responses") and hasattr(self.client_async.responses, "create"):
                kwargs = {"model": self.model, "input": messages}
                if eff:
                    kwargs["reasoning"] = {"effort": eff}
                try:
                    resp_obj = await self.client_async.responses.create(**kwargs)
                except Exception as e:
                    logging.error(f"Error in OpenAI Responses SDK async call: {str(e)}")
                    raise
                # Extract
                output_text = ""
                try:
                    output = getattr(resp_obj, "output", None) or resp_obj.get("output")
                except Exception:
                    output = None
                if output:
                    for item in output:
                        content = getattr(item, "content", None)
                        if content is None and hasattr(item, "get"):
                            content = item.get("content")
                        if content:
                            for c in content:
                                text = getattr(c, "text", None) or c.get("text")
                                if text:
                                    output_text += text
                if not output_text:
                    try:
                        t = getattr(resp_obj, "text", None) or resp_obj.get("text")
                        if t:
                            output_text = str(t)
                    except Exception:
                        pass
                if output_text:
                    yield output_text
                    return
            # REST fallback (non‑streaming)
            url = f"{self.base_url.rstrip('/')}/responses"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            }
            payload: Dict[str, Union[str, int, Dict, List]] = {
                "model": self.model,
                "input": messages,
            }
            if eff:
                payload["reasoning"] = {"effort": eff}
            try:
                timeout_cfg = aiohttp.ClientTimeout(total=600)  # allow extended server processing
                async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        # Extract output text
                        output_text = ""
                        output = data.get("output")
                        if output:
                            for item in output:
                                content = item.get("content")
                                if content:
                                    for c in content:
                                        t = c.get("text")
                                        if t:
                                            output_text += t
                        if not output_text:
                            t = data.get("text")
                            if t:
                                output_text = str(t)
                        if output_text:
                            yield output_text
                            return
            except Exception as e:
                logging.error(f"Error in OpenAI REST Responses async call: {str(e)}")
                raise


class ClaudeAIModel(AIModel):
    """
    Implementation of AIModel using Anthropic's Claude model.
    """

    def __init__(self, config: Config, timezone: str = "", model_id: Optional[str] = None, max_tokens: Optional[int] = None):
        """
        Initialize the Claude AI model.

        Args:
            config (Config): The application configuration object.
            timezone (str): The timezone to use for timestamps.
            model_id (Optional[str]): The specific model ID to use.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
        """
        self.model = model_id or config.get("claude_model", "claude-sonnet-4-5")

        logging.info(f"Using model: {self.model}")
        self.max_tokens = max_tokens or config.get("max_tokens", 4096)
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
        if response.status_code in (401, 403):
            try:
                body = response.json()
                msg = body.get("error", {}).get("message") or response.text
            except Exception:
                msg = response.text
            from src.tools import NonRetryableError

            raise NonRetryableError(f"Claude error: {msg}")
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
                if response.status in (401, 403):
                    body = await response.text()
                    try:
                        err = json.loads(body)
                        msg = err.get("error", {}).get("message") or body
                    except Exception:
                        msg = body
                    from src.tools import NonRetryableError

                    raise NonRetryableError(f"Claude error: {msg}")
                res = await response.text()
                return json.loads(res)

    def get_response(self, messages: MessageList, reasoning_effort: Optional[Union[str, ReasoningEffort]] = None) -> str:
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
        self, messages: MessageList,
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
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

    def get_tokens_number(self, messages: MessageList) -> int:
        """
        Count the number of tokens in the given messages using Anthropic's count_tokens API.

        Args:
            messages (MessageList): A list of message models representing the conversation history.

        Returns:
            int: The number of tokens in the messages.
        """
        messages = normalize_messages(messages)
        system_message_combined = " ".join(
            [m["content"] for m in messages if m["role"] == "system"]
        )
        non_system_message = [m for m in messages if m["role"] != "system"]

        data = {
            "model": self.model,
            "messages": non_system_message,
        }
        if system_message_combined:
            data["system"] = system_message_combined

        count_tokens_url = "https://api.anthropic.com/v1/messages/count_tokens"
        response = requests.post(count_tokens_url, headers=self.headers, json=data)

        if response.status_code in (401, 403):
            try:
                body = response.json()
                msg = body.get("error", {}).get("message") or response.text
            except Exception:
                msg = response.text
            from src.tools import NonRetryableError

            raise NonRetryableError(f"Claude count_tokens error: {msg}")

        response_dict = json.loads(response.content.decode("utf-8"))
        return response_dict.get("input_tokens", 0)


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
            api="chat_completions",
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
        super().__init__(config, base_url=base_url, api_key=api_key, model_id=model, api="chat_completions")


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
        super().__init__(config, base_url=base_url, api_key=api_key, model_id=model, api="chat_completions")


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
    ai_model = ClaudeAIModel(config)
    tnum = ai_model.get_tokens_number([{"role": "system", "content": claude_system_prompt},{"role": "user", "content": "a"}])
    print(f"Tokens number: {tnum}")
    # Use OpenAI's high-reasoning model (GPT-5)
    ai_model = OpenAIModel(config)
    # ai_model = OpenAIModel(config, model_id="gpt-5", reasoning_effort=ReasoningEffort.COMPREHENSIVE)
    # ai_model = DeepseekModel(config)
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
