"""
Conversation Manager module.

This module provides the ConversationManager class for managing the flow of conversation,
including message history, token counting, and interaction with AI models.
"""

import asyncio
import datetime
import functools
import glob
import json
import logging
import os
import re
import sys
import time
from collections import deque
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

if __name__ == "__main__":
    # add current directory to python path
    sys.path.append(os.getcwd())

from src.ai_models import AIModel, ClaudeAIModel
from src.llm_optimization import (
    PROFILE_CHAT_ONLY,
    classify_tool_profile,
    parse_volume_intent,
    resolve_tools_for_profile,
    wants_detailed_response,
)
from src.llm_tools import (
    optimize_facts,
    optimize_rules,
    summarize_and_compress_history,
)
from src.responce_player import extract_emotions, extract_language
from src.tool_usage_stats import get_tool_usage_stats
from src.tools import (
    clean_response,
    fix_stress_marks_russian,
    format_message_history,
    get_current_date_time_for_facts,
    get_current_datetime_english,
    get_location,
    get_timezone,
    get_token_count,
)
from src.web_search import WebSearcher

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
    pattern = r"\$remember:(.*?)\$"

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
    modified_text = re.sub(pattern, "", text)

    # Remove any extra whitespace that might have been left
    modified_text = " ".join(modified_text.split())

    return modified_text, extracted_facts


def extract_rules(text: str) -> Tuple[str, List[str]]:
    """
    Extract rules from the input text and return the modified text and a list of extracted rules.

    Args:
        text (str): The input text to extract facts from.

    Returns:
        Tuple[str, List[str]]: A tuple containing the modified text and a list of extracted rules.
    """
    # Regular expression to match {remember: xxx} pattern
    pattern = r"\$rule:(.*?)\$"

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
    modified_text = re.sub(pattern, "", text)

    # Remove any extra whitespace that might have been left
    modified_text = " ".join(modified_text.split())

    return modified_text, extracted_rules


def _get_emotion_awareness_rule_russian() -> str:
    """Rule for interpreting emotion annotations in user messages."""
    return (
        "Сообщения пользователя могут начинаться с [User emotion: эмоция1 (оценка), ...]. "
        # "Учитывай эти эмоции при ответе. Если пользователь взволнован - отвечай энергично, "
        # "если грустит - будь поддерживающим, если раздражён - оставайся спокойным. "
    )


def _get_emotion_awareness_rule_english() -> str:
    """English version of emotion awareness rule."""
    return (
        "User messages may start with [User emotion: emotion1 (score), ...]. "
        # "Consider these emotions when responding. Match excited energy, "
        # "be supportive when user is sad, stay calm when frustrated. "
    )


def _get_base_rules_russian() -> str:
    """Base rules that are tool-independent (preserved from current implementation)"""
    return (
        "Если в ответе на твой запрос указано время без указания часового пояса, "
        "считай что это Восточное стандартное время. "
        "Если тебе надо что-то запомнить, "
        "пошли мне сообщение в таком формате: $remember: <текст, который тебе нужно запомнить>$. "
        "Таких фактов в твоем сообщении тоже может быть несколько. "
        "Например, $remember: <первый текст, который тебе нужно запомнить>$ "
        "$remember: <второй текст, который тебе нужно запомнить>$. "
        "Когда не совсем понятно, какое ударение надо ставить в слове, "
        "используй знак + перед предполагаемой ударной гласной. "
        "Знак ударения + всегда ставится непосредственно перед "
        "ударной гласной буквой в слове. "
        "Например: к+оса (прическа); кос+а (инструмент); кос+а (участок суши). "
        "Этот знак никогда не ставится в конце слова или перед согласными. "
        "Его единственная функция - указать на ударный гласный звук. "
        "Используй знак + только в русском языке. "
        "Если я прошу тебя как-то поменяться (например, не используй обсценную лексику), "
        "чтобы запомнить это новое правило, пошли мне сообщение в таком формате: "
        "$rule: <текст нового правила>$. "
        "Таких запросов в твоем сообщении тоже может быть несколько."
    )


def _get_base_rules_english() -> str:
    """English version of base rules"""
    return (
        "For web searches: $internet query:<query in English>$. "
        "To remember: $remember:<text>$. For new rules: $rule:<text>$ "
        "When it's not entirely clear where to place the stress in a word, "
        "use the '+' sign before the presumed stressed vowel. "
        "The stress mark '+' is always placed directly before "
        "the stressed vowel letter in the word. "
        "For example: к+оса (прическа); кос+а (инструмент); кос+а (участок суши). "
        "This sign is never placed at the end of a word or before consonants. "
        "Its sole function is to indicate the stressed vowel sound. Use it only in Russian."
    )


def _get_base_rules_russian_compact() -> str:
    """Compact base rules to reduce prompt tokens."""
    return (
        "Если время указано без часового пояса, считай EST. "
        "Для новых фактов используй: $remember: <текст>$. "
        "Для новых постоянных правил используй: $rule: <текст>$. "
        "Знак ударения '+' ставь только перед ударной гласной и только в русском языке."
    )


def _get_base_rules_english_compact() -> str:
    """Compact English base rules for lower token usage."""
    return (
        "If time is provided without timezone, assume EST. "
        "To store memory use: $remember: <text>$. "
        "To store a new persistent rule use: $rule: <text>$. "
        "Use '+' before stressed vowel only for Russian words."
    )


def _combine_rules(base_rules: str, dynamic_rules: str) -> str:
    """Combine base rules with dynamic tool rules"""
    if dynamic_rules:
        return f"{base_rules}\n\n{dynamic_rules}"
    return base_rules


class ConversationManager:
    """
    Manages the conversation flow, including message history and interaction with AI models.

    Attributes:
        config (Config): The application configuration object.
        ai_model (AIModel): The AI model used for generating responses.
        message_history (deque): A queue of message dictionaries representing the conversation history.
    """

    def __init__(
        self, config, ai_model: AIModel, timezone: str, enabled_tools: Optional[List] = None
    ):
        """
        Initialize the ConversationManager.

        Args:
            config (Config): The application configuration object.
            ai_model (AIModel): The AI model to use for generating responses.
            timezone (str): Current timezone to store time
            enabled_tools (Optional[List]): List of enabled tools for dynamic rule generation
        """
        self.config = config
        self.searcher = WebSearcher(config)
        self.ai_model = ai_model
        self.summarize_model = ClaudeAIModel(config)
        self.facts = self.load_facts()
        self.rules: List[str] = self.load_rules()
        self.location = get_location()
        self.timezone = timezone
        self.current_language_code = "ru"
        self.enabled_tools = enabled_tools or []
        self.enabled_tools_by_name = {t.name: t for t in self.enabled_tools if hasattr(t, "name")}
        self.emotion_detection_enabled = config.get("emotion_detection_enabled", False)
        self.tool_usage_stats = get_tool_usage_stats(config)
        self.optimize_prompt_compact = bool(config.get("optimize_prompt_compact", False))
        self.optimize_prompt_split_dynamic = bool(config.get("optimize_prompt_split_dynamic", False))
        self.hybrid_prompt_caching_enabled = bool(
            config.get("claude_prompt_caching_hybrid_enabled", False)
        )
        self.prompt_cache_window_seconds = int(
            config.get("claude_prompt_cache_window_seconds", 300)
        )
        self.freeze_dynamic_context_for_cache = bool(
            config.get(
                "claude_prompt_cache_freeze_dynamic_context",
                self.hybrid_prompt_caching_enabled,
            )
        )
        self._cached_dynamic_context_prefix: Optional[str] = None
        self._cached_dynamic_context_expires_at: float = 0.0
        self.optimize_prompt_internal_language = str(
            config.get("optimize_prompt_internal_language", "ru")
        ).strip().lower()
        if self.optimize_prompt_internal_language not in {"ru", "en"}:
            self.optimize_prompt_internal_language = "ru"

        self.default_system_prompt_russian = (
            "Тебя зовут Кубик. Ты мой друг и помощник. Ты умеешь шутить и быть саркастичным. "
            "Отвечай естественно, как в устной речи. "
            "Говори максимально просто и понятно. Не используй списки и нумерации. "
            "Например, не говори 1. что-то; 2. что-то. говори во-первых, во-вторых "
            "или просто перечисляй. "
            "Не использую markdown formatting, его не передаст устная речь. "
            "При ответе на вопрос где важно время, помни какое сегодня число. "
            "Если чего-то не знаешь, так и скажи. "
            "Я буду разговаривать с тобой через голосовой интерфейс. "
            "Будь краток, избегай банальностей и непрошенных советов."
        )

        self.default_system_prompt_english = (
            "You're Kubik, my friendly AI assistant. Be witty and sarcastic. "
            "Speak naturally, simply. Avoid lists. Consider date in time-sensitive answers. "
            "Admit unknowns. I use voice interface. Be brief, avoid platitudes. "
            "Don't use markdown formatting."
            "Use internet searches when needed for up-to-date or specific information. "
            "Assume EST if timezone unspecified. Treat responses as spoken."
        )

        self.compact_system_prompt_russian = (
            "Тебя зовут Кубик. Ты друг и помощник. Говори естественно, коротко и просто, "
            "как в устной речи. Без markdown и без списков. Если не знаешь, так и скажи."
        )

        self.compact_system_prompt_english = (
            "You are Kubik, a friendly voice assistant. Speak naturally, briefly, and simply. "
            "No markdown, no list formatting. If unsure, say so."
        )

        use_english_meta = self.optimize_prompt_internal_language == "en"
        if use_english_meta:
            self.default_system_prompt = (
                self.compact_system_prompt_english
                if self.optimize_prompt_compact
                else self.default_system_prompt_english
            )
            self.tool_rules_language = "english"
        else:
            self.default_system_prompt = (
                self.compact_system_prompt_russian
                if self.optimize_prompt_compact
                else self.default_system_prompt_russian
            )
            self.tool_rules_language = "russian"

        self.hard_rules = self._build_hard_rules()
        self.ephemeral_facts: List[str] = []

        use_structured_system_payload = (
            self.optimize_prompt_split_dynamic and hasattr(self.ai_model, "set_request_options")
        )
        initial_system_for_history = (
            self._system_prompt_body() if use_structured_system_payload else self.get_system_prompt()
        )
        self.message_history: Deque[dict] = deque(
            [{"role": "system", "content": initial_system_for_history}]
        )
        if use_structured_system_payload:
            initial_payload = self._build_runtime_system_blocks()
            self.last_system_payload_for_dialog = (
                initial_payload if initial_payload is not None else initial_system_for_history
            )
        else:
            self.last_system_payload_for_dialog = initial_system_for_history

    def _generate_tool_rules(self, language: str) -> str:
        """
        Collect and combine rule_instructions from all enabled tools.

        Args:
            language: Language code ('russian' or 'english')

        Returns:
            Combined natural language rules for all tools that provide them
        """
        prune = bool(self.config.get("optimize_tool_rules_prune_by_usage", False))
        min_calls = int(self.config.get("optimize_tool_rules_usage_min_calls", 3))
        keep_top_n = int(self.config.get("optimize_tool_rules_usage_keep_top_n", 6))
        warmup_calls = int(self.config.get("optimize_tool_rules_usage_warmup_calls", 30))
        never_prune = set(self.config.get("optimize_tool_rules_never_prune", []))

        top_tools = set()
        total_calls = 0
        if prune and self.tool_usage_stats:
            total_calls = self.tool_usage_stats.total_calls()
            top_tools = set(self.tool_usage_stats.top_tools(keep_top_n))

        pruned_tools = []
        rules = []
        for tool in self.enabled_tools:
            tool_name = getattr(tool, "name", "")

            if prune and self.tool_usage_stats and total_calls >= warmup_calls:
                tool_count = self.tool_usage_stats.get_count(tool_name)
                include = (
                    tool_name in never_prune
                    or tool_count >= min_calls
                    or tool_name in top_tools
                )
                if not include:
                    pruned_tools.append(tool_name)
                    continue

            if hasattr(tool, "rule_instructions") and language in tool.rule_instructions:
                rule_text = tool.rule_instructions[language].strip()
                if rule_text:
                    rules.append(rule_text)

        if pruned_tools:
            logger.info(f"Tool rules pruned by usage: {pruned_tools}")
        logger.debug(f"Generated tool rules: {rules}")
        return "".join(rules)

    def _build_hard_rules(self) -> str:
        use_english_meta = self.optimize_prompt_internal_language == "en"
        if use_english_meta:
            base_rules = (
                _get_base_rules_english_compact()
                if self.optimize_prompt_compact
                else _get_base_rules_english()
            )
            if self.emotion_detection_enabled:
                base_rules += _get_emotion_awareness_rule_english()
            return _combine_rules(base_rules, self._generate_tool_rules("english"))

        base_rules = (
            _get_base_rules_russian_compact()
            if self.optimize_prompt_compact
            else _get_base_rules_russian()
        )
        if self.emotion_detection_enabled:
            base_rules += _get_emotion_awareness_rule_russian()
        return _combine_rules(base_rules, self._generate_tool_rules("russian"))

    def _system_prompt_context_prefix(self) -> str:
        should_freeze = (
            self.optimize_prompt_split_dynamic
            and bool(self.config.get("claude_enable_prompt_caching", False))
            and self.freeze_dynamic_context_for_cache
        )
        if not should_freeze:
            return f"{get_current_datetime_english(self.timezone)} {self.location} "

        now_ts = time.time()
        if (
            self._cached_dynamic_context_prefix is not None
            and now_ts < self._cached_dynamic_context_expires_at
        ):
            return self._cached_dynamic_context_prefix

        prefix = f"{get_current_datetime_english(self.timezone)} {self.location} "
        ttl = max(1, int(self.prompt_cache_window_seconds))
        self._cached_dynamic_context_prefix = prefix
        self._cached_dynamic_context_expires_at = now_ts + ttl
        return prefix

    def _system_prompt_body(self) -> str:
        from src.responce_player import emotions_prompt, language_prompt

        prompt = self.config.get("system_prompt", self.default_system_prompt)
        # Refresh hard rules because optional usage-based pruning is time-dependent.
        self.hard_rules = self._build_hard_rules()
        prompt += self.hard_rules
        prompt += emotions_prompt()
        prompt += language_prompt()

        facts_for_prompt = self.facts
        combined_facts = []
        if facts_for_prompt:
            combined_facts.extend(facts_for_prompt)
        if self.ephemeral_facts:
            combined_facts.extend(self.ephemeral_facts)
            logger.info(f"Ephemeral facts: {self.ephemeral_facts}")
        else:
            logger.debug("Ephemeral facts Empty")
        if combined_facts:
            prompt += " Ты уже знаешь факты:" + " ".join(combined_facts)

        if self.rules:
            prompt += " Ты уже помнишь правила:" + " ".join(self.rules)

        # Log the generated system prompt and its token count
        # try:
        #     token_count = self.ai_model.get_tokens_number(
        #         [
        #             {"role": "system", "content": prompt},
        #             {"role": "user", "content": "a"},
        #         ]
        #     )
        #
        #     logger.info(f"Generated system prompt ({token_count} tokens):\n{prompt}")
        # except Exception as e:
        #     logger.warning(f"Could not count tokens for system prompt: {e}")
        #     logger.info(f"Generated system prompt:\n{prompt}")

        return prompt

    def get_system_prompt(self):
        return self._system_prompt_context_prefix() + self._system_prompt_body()

    def get_system_prompt_parts(self) -> Tuple[str, str]:
        """
        Return (static_body, dynamic_context).
        """
        return self._system_prompt_body(), self._system_prompt_context_prefix()

    def _build_runtime_system_blocks(self) -> Optional[List[Dict[str, Any]]]:
        if not self.optimize_prompt_split_dynamic:
            return None
        static_body, dynamic_context = self.get_system_prompt_parts()
        if not static_body and not dynamic_context:
            return None
        cache_enabled = bool(self.config.get("claude_enable_prompt_caching", False))
        if cache_enabled:
            return [
                {
                    "type": "text",
                    "text": static_body,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": dynamic_context},
            ]
        return [{"type": "text", "text": dynamic_context + static_body}]

    def _select_tool_names_for_text(self, text: str) -> Optional[set]:
        if not self.config.get("optimize_dynamic_tool_profiles", False):
            return None
        default_profile = str(
            self.config.get("optimize_default_tool_profile", PROFILE_CHAT_ONLY)
        ).strip()
        profile = classify_tool_profile(text, default_profile=default_profile)
        available_tool_names = set(self.enabled_tools_by_name.keys())
        if self.config.get("claude_use_search", False):
            available_tool_names.add("web_search")
        if self.config.get("openai_use_search", False):
            available_tool_names.add("web_search")
        selected = resolve_tools_for_profile(profile, available_tool_names)
        logger.info(f"Tool profile selected: {profile}, tools={sorted(selected)}")
        return selected

    def _select_max_tokens_for_text(self, text: str) -> Optional[int]:
        if not self.config.get("optimize_response_length_control", False):
            return None
        detailed = wants_detailed_response(text)
        default_limit = int(self.config.get("optimize_response_max_tokens_default", 180))
        detailed_limit = int(self.config.get("optimize_response_max_tokens_detailed", 360))
        chosen = detailed_limit if detailed else default_limit
        logger.info(f"Response token cap selected: {chosen} (detailed={detailed})")
        return chosen

    async def _maybe_route_volume_command(self, text: str) -> Optional[str]:
        if not self.config.get("optimize_volume_router_enabled", False):
            return None
        intent = parse_volume_intent(text)
        if not intent:
            return None
        tool = self.enabled_tools_by_name.get("control_speaker_volume")
        if not tool:
            return None
        params: Dict[str, Any] = {"action": intent.action}
        if intent.value is not None:
            params["value"] = intent.value
        try:
            result = await tool.processor(params)
            if self.tool_usage_stats:
                self.tool_usage_stats.record_call("control_speaker_volume")
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.warning(f"Direct volume route failed: {e}")
            return None

    def _log_turn_cost_metrics(self) -> None:
        if not self.config.get("cost_per_turn_logging_enabled", False):
            return
        usage_getter = getattr(self.ai_model, "get_last_turn_usage", None)
        if not callable(usage_getter):
            return
        usage = usage_getter()
        if not usage:
            return
        cost_getter = getattr(self.ai_model, "get_last_turn_cost_usd", None)
        cost = cost_getter() if callable(cost_getter) else None
        if cost is None:
            logger.info(
                "LLM turn usage tokens: input=%s output=%s cache_write=%s cache_read=%s; "
                "cost unavailable (configure claude_cost_*_per_million).",
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_creation_input_tokens", 0),
                usage.get("cache_read_input_tokens", 0),
            )
            return
        logger.info(
            "LLM turn cost: $%.6f (input=%s, output=%s, cache_write=%s, cache_read=%s)",
            cost,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
        )

    async def get_response(self, text: str) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        Get an AI response based on the current conversation state and new input.

        Sentences are buffered and combined before yielding to reduce TTS costs.
        Buffer flushes when:
        - Total characters exceed sentence_buffer_max_length (default: 200)
        - No new sentence arrives within sentence_buffer_timeout (default: 1.5s)

        Args:
            text (str): The new input text to respond to.

        Returns:
            AsyncGenerator[List[Dict[str,any]]: The AI-generated response,
            marked with emotion response and language code
        """
        logger.debug(f"call to get_response for: {text}")

        # Buffer settings
        buffer_enabled = self.config.get("sentence_buffer_enabled", True)
        buffer_timeout = self.config.get("sentence_buffer_timeout", 1.5)
        buffer_max_length = self.config.get("sentence_buffer_max_length", 200)

        if buffer_enabled:
            logger.debug(
                f"Sentence buffer: enabled, timeout={buffer_timeout}s, max_length={buffer_max_length}"
            )

        # Keep history/debug prompt aligned with the actual split payload mode.
        # In split mode, dynamic datetime/location is sent separately via system blocks.
        if self.optimize_prompt_split_dynamic and hasattr(self.ai_model, "set_request_options"):
            system_prompt_for_history = self._system_prompt_body()
        else:
            system_prompt_for_history = self.get_system_prompt()
        self.message_history[0] = {"role": "system", "content": system_prompt_for_history}
        # Default payload snapshot for dialog/debug (overridden below when split blocks are used).
        self.last_system_payload_for_dialog = system_prompt_for_history

        # cleanup in case of previous errors
        if self.message_history[-1]["role"] == "user":
            logger.warning(f"ignoring previous user message: {self.message_history[-1]['content']}")
            self.message_history.pop()

        self.message_history.append({"role": "user", "content": text})

        # Optional direct route for obvious volume commands to avoid LLM round-trip.
        direct_volume_response = await self._maybe_route_volume_command(text)
        if direct_volume_response:
            self.message_history.append({"role": "assistant", "content": direct_volume_response})
            lang = self.current_language_code or "ru"
            yield [{"emotion": None, "language": lang, "text": direct_volume_response}]
            return

        tool_names = self._select_tool_names_for_text(text)
        max_tokens = self._select_max_tokens_for_text(text)
        system_blocks = self._build_runtime_system_blocks()
        if system_blocks is not None:
            self.last_system_payload_for_dialog = system_blocks
        request_options = {
            "tool_names": tool_names,
            "response_max_tokens": max_tokens,
            "system_blocks": system_blocks,
        }
        if hasattr(self.ai_model, "set_request_options"):
            self.ai_model.set_request_options(**request_options)

        if get_token_count(list(self.message_history)) > self.config.get("token_threshold", 2500):
            new_message_history = await summarize_and_compress_history(
                self.message_history, self.summarize_model, self.config
            )
            if new_message_history != self.message_history:
                self.message_history = new_message_history
                newline_str = "\n\n"
                logger.debug(
                    f"Compressed  conversation:  {(newline_str + self.formatted_message_history(150) + newline_str)}"
                )

        # Buffer for combining sentences
        sentence_buffer: List[Dict[str, Any]] = []
        buffer_chars = 0

        def combine_buffer() -> List[Dict[str, Any]]:
            """Combine buffered sentences into one."""
            if not sentence_buffer:
                return []

            # Combine texts, keep first emotion/language
            combined_text = " ".join(s["text"] for s in sentence_buffer)
            return [
                {
                    "emotion": sentence_buffer[0]["emotion"],
                    "language": sentence_buffer[0]["language"],
                    "text": combined_text,
                }
            ]

        try:
            async for response_text in self.ai_model.get_response_async(list(self.message_history)):
                crt = clean_response(response_text)

                if self.message_history[-1]["role"] != "assistant":
                    self.message_history.append({"role": "assistant", "content": crt})
                else:
                    self.message_history[-1]["content"] += " " + crt

                response_text, facts = extract_facts(response_text, self.timezone)
                if facts:
                    self.facts += facts
                    self.save_facts(self.facts)
                    logger.debug(f"Extracted facts: {facts}")

                response_text, rules = extract_rules(response_text)
                self.rules += rules
                self.save_rules(self.rules)

                if rules:
                    logger.debug(f"Extracted rules: {rules}")

                # Check if this is a tool use signal
                if response_text.strip() == "[[TOOL_USE]]":
                    logger.debug("Received tool use signal, flushing sentence buffer")
                    # Flush buffer immediately when tool is about to be used
                    if buffer_enabled and sentence_buffer:
                        logger.debug(
                            f"Sentence buffer: tool use detected, flushing {len(sentence_buffer)} sentences "
                            f"({buffer_chars} chars)"
                        )
                        yield combine_buffer()
                        sentence_buffer.clear()
                        buffer_chars = 0
                    # remove from history
                    self.message_history.pop()
                    # Don't process further
                    continue

                # Process emotions and language for this sentence
                for emo, t in extract_emotions(response_text):
                    logger.debug(f"Emotion: {emo} -> {t}")
                    for lang, clean_text in extract_language(
                        t, default_lang=self.current_language_code
                    ):
                        logger.debug(f"Language: {lang} -> {clean_text}")
                        self.current_language_code = lang
                        if text and clean_text:
                            clean_text = fix_stress_marks_russian(clean_text)
                            sentence = {"emotion": emo, "language": lang, "text": clean_text}

                            if not buffer_enabled:
                                # No buffering, yield immediately
                                yield [sentence]
                            else:
                                sentence_len = len(clean_text)

                                # Check if adding this sentence would cross a billing unit boundary
                                # (Yandex v3 charges per 250-char unit)
                                would_cross_unit = (
                                    buffer_chars > 0 and buffer_chars + sentence_len > 250
                                )

                                # Also flush if max_length exceeded (safety check)
                                would_exceed_max = buffer_chars + sentence_len > buffer_max_length

                                # Flush if we should cross a unit boundary or exceed max_length
                                if (would_cross_unit or would_exceed_max) and buffer_chars > 0:
                                    logger.debug(
                                        f"Sentence buffer: optimizing for billing units "
                                        f"({buffer_chars}/{sentence_len}={buffer_chars + sentence_len} chars), "
                                        f"yielding {len(sentence_buffer)} sentences"
                                    )
                                    yield combine_buffer()
                                    sentence_buffer.clear()
                                    buffer_chars = 0

                                # Add sentence to buffer
                                sentence_buffer.append(sentence)
                                buffer_chars += sentence_len
                                logger.debug(
                                    f"Sentence buffer: added ({sentence_len} chars, "
                                    f"total: {buffer_chars}, count: {len(sentence_buffer)})"
                                )

            # Flush remaining buffer at the end
            if buffer_enabled and sentence_buffer:
                logger.debug(
                    f"Sentence buffer: end of response, yielding {len(sentence_buffer)} sentences "
                    f"({buffer_chars} chars)"
                )
                yield combine_buffer()
            self._log_turn_cost_metrics()
        finally:
            if hasattr(self.ai_model, "clear_request_options"):
                self.ai_model.clear_request_options()

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
        message_history_for_dialog = list(self.message_history)
        if message_history_for_dialog and message_history_for_dialog[0].get("role") == "system":
            payload = self.last_system_payload_for_dialog
            if (
                self.optimize_prompt_split_dynamic
                and hasattr(self.ai_model, "set_request_options")
                and not isinstance(payload, (list, dict))
            ):
                fallback_payload = self._build_runtime_system_blocks()
                if fallback_payload is not None:
                    payload = fallback_payload
                    self.last_system_payload_for_dialog = fallback_payload
            if isinstance(payload, (list, dict)):
                payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
            else:
                payload_text = str(payload)
            message_history_for_dialog[0] = {"role": "system", "content": payload_text}
        with open(dialog_file_name, "w", encoding="utf-8") as dialog_file:
            dialog_file.write(
                "\n\n" + format_message_history(message_history_for_dialog, 150) + "\n\n"
            )

    async def _run_sync_in_thread(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args))

    async def _process_facts(self) -> None:
        p = Path("facts.json")

        # sanity check
        if not p.exists():
            return

        cleaning_time_stop_hour = self.config.get("cleaning_time_stop_hour", 4)
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(p))

        cutoff = datetime.datetime.now() - datetime.timedelta(days=1)
        cutoff = cutoff.replace(hour=cleaning_time_stop_hour, minute=0, second=0, microsecond=0)

        # if there were no modifications today
        if mod_time < cutoff:
            logger.debug(f"facts modification time {mod_time} is too old to optimize")
            return

        # remove facts with /tmp/
        pre_optimized_facts = []
        for line in self.facts:
            if "/tmp/" in line:
                continue
            pre_optimized_facts.append(line)

        # Asynchronously optimize facts
        optimized_facts = await optimize_facts(pre_optimized_facts, self.config, self.timezone)

        if set(optimized_facts) != set(self.facts):
            self.facts = optimized_facts

            # backup existing facts.json, rename it facts_prev.json
            if p.exists():
                logger.debug("backup existing facts.json")
                p.rename("facts_prev.json")

            self.save_facts(self.facts)

    async def _process_rules(self) -> None:
        p = Path("rules.json")

        # sanity check
        if not p.exists():
            return

        cleaning_time_stop_hour = self.config.get("cleaning_time_stop_hour", 4)
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(p))

        cutoff = datetime.datetime.now() - datetime.timedelta(days=1)
        cutoff = cutoff.replace(hour=cleaning_time_stop_hour, minute=0, second=0, microsecond=0)

        # if there were no modifications today
        if mod_time < cutoff:
            logger.debug(f"rules modification time {mod_time} is too old to optimize")
            return

        # Asynchronously optimize rules
        optimized_rules = await optimize_rules(self.hard_rules, self.rules, self.config)

        # Save rules using a separate thread to avoid blocking the event loop
        if set(optimized_rules) != set(self.rules):
            self.rules = optimized_rules
            # backup existing rules
            if p.exists():
                logger.debug("backup existing rules.json")
                p.rename("rules_prev.json")
            self.save_rules(self.rules)

    async def process_and_clean(self, force=False):
        # form new memories, clean message deque, process existing facts and rules
        # to be used at night time

        newline = "\n"

        # form new memories
        if self.config.get("form_new_memories_at_night", True) or force:
            if len(self.message_history) > 2 or force:
                prompt = self.config.get(
                    "form_new_memories_prompt",
                    "Это необязательно, но может хочешь еще что-нибудь запомнить "
                    "из нашего разговора перед тем как его удалю? "
                    "Отвечай только текстом, используя $remember: <текст>$ если нужно. "
                    "Не вызывай никакие инструменты.",
                )
                logger.debug(f"form new memory by asking {prompt}")
                num_facts_before = len(self.facts)
                async for ai_response in self.get_response(prompt):
                    logger.debug("CM: AI response: %s", ai_response)
                num_facts_after_clean = len(self.facts)
                if num_facts_after_clean == num_facts_before:
                    logger.debug("no new memories formed")
                else:
                    logger.debug(f"{num_facts_after_clean - num_facts_before} new facts remembered")

        if self.config.get("clean_message_history_at_night", True) or force:
            # cleanup conversation
            self.message_history: Deque[dict] = deque(
                [{"role": "system", "content": self.get_system_prompt()}]
            )
            # Clear ephemeral facts during nightly cleanup.
            self.ephemeral_facts = []

        # process existing facts and rules (run both operations concurrently)
        existing_facts = set(self.facts)
        existing_rules = set(self.rules)
        loop = asyncio.get_event_loop()
        # asyncio.gather allows us to wait for multiple coroutines concurrently
        await asyncio.gather(
            loop.create_task(self._process_facts()),
            loop.create_task(self._process_rules()),
        )

        removed_facts = list(existing_facts - set(self.facts))
        if removed_facts:
            logger.debug(f"removed facts: \n{newline.join(removed_facts)}")

        new_facts = list(set(self.facts) - existing_facts)
        if new_facts:
            logger.debug(f"new facts: \n{newline.join(new_facts)}")

        removed_rules = list(existing_rules - set(self.rules))
        if removed_rules:
            logger.debug(f"removed rules: \n{newline.join(removed_rules)}")

        new_rules = list(set(self.rules) - existing_rules)
        if new_rules:
            logger.debug(f"new rules: \n{newline.join(new_rules)}")

        # remove temp wav files
        num_removed = 0
        for dir in ["/tmp", "."]:
            for filepath in glob.glob(dir + "/*.wav"):
                try:
                    os.remove(filepath)
                    logger.debug(f"File {filepath} has been removed successfully")
                    num_removed += 1
                except Exception as e:
                    logger.warning(f"Error occurred while trying to remove {filepath}. Error: {e}")
        logger.debug(f"removed {num_removed} temp wav files")

    def add_pending_reminder_fact(self, reminder: dict) -> None:
        message = reminder.get("message")
        if not isinstance(message, str) or not message.strip():
            return
        fact_text = reminder.get("fact_text")
        if not isinstance(fact_text, str) or not fact_text.strip():
            fact_text = f"Напомнил: {message.strip()}."
        fact_with_marker = f"{self.reminder_fact_marker} {fact_text}"
        if any(fact_with_marker in existing for existing in self.facts):
            return
        fact = get_current_date_time_for_facts(self.timezone) + " : " + fact_with_marker
        self.facts += [fact]
        self.save_facts(self.facts)

    def add_ephemeral_fact(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        fact = text.strip()
        if fact in self.ephemeral_facts:
            return
        self.ephemeral_facts.append(fact)

    @staticmethod
    def load_facts():
        try:
            with open("facts.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except json.decoder.JSONDecodeError as e:
            logger.error(f"couldn't load facts file: {e}")
            return []

    @staticmethod
    def save_facts(facts):
        with open("facts.json", "w", encoding="utf8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_rules() -> List[str]:
        try:
            with open("rules.json", "r") as f:
                result = json.load(f)
                if isinstance(result, dict):
                    result = result["rules"]
                return result
        except FileNotFoundError:
            return []
        except json.decoder.JSONDecodeError as e:
            logger.error(f"couldn't load rules file: {e}")
            return []

    @staticmethod
    def save_rules(rules):
        with open("rules.json", "w", encoding="utf8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=4)


async def test():
    from src.config import Config
    from src.ai_models_with_tools import ClaudeAIModelWithTools

    config = Config()
    timezone = get_timezone()
    ai_model = ClaudeAIModelWithTools(config)
    conversation_manager = ConversationManager(config, ai_model, timezone)
    await conversation_manager.process_and_clean(force=True)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(test())
