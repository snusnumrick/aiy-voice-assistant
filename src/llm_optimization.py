"""
Utilities for optional LLM cost optimizations.

This module provides lightweight, deterministic routing helpers that can be
enabled via config flags without changing default behavior.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Tool profile names
PROFILE_ALL_TOOLS = "all_tools"
PROFILE_CHAT_ONLY = "chat_only"
PROFILE_HOME_CONTROL = "home_control"
PROFILE_CREATIVE = "creative"
PROFILE_RESEARCH = "research"
PROFILE_ORGANIZER = "organizer"
PROFILE_MEMORY = "memory"


PROFILE_TOOL_NAMES = {
    PROFILE_HOME_CONTROL: {
        "control_speaker_volume",
        "stress_marker",
    },
    PROFILE_CREATIVE: {
        "generate_image",
        "generate_music",
        "play_music",
    },
    PROFILE_RESEARCH: {
        "internet_search",
        "web_search",
        "list_web_search_reports",
        "get_web_search_report",
        "wise_wizard",
        "list_wizard_reports",
        "get_wizard_report",
        "weather_info",
        "enhanced_weather_info",
    },
    PROFILE_ORGANIZER: {
        "manage_reminders",
        "list_reminders",
        "send_email_to_user",
        "code_interpreter",
    },
    PROFILE_MEMORY: {
        "recall_memory",
    },
    PROFILE_CHAT_ONLY: set(),
}


DETAIL_PATTERNS = [
    r"\bподроб",
    r"\bдеталь",
    r"\bразвернут",
    r"\bглубже",
    r"\bin detail\b",
    r"\bmore detail\b",
    r"\bdeep dive\b",
    r"\bexplain more\b",
    r"\bstep by step\b",
]


@dataclass
class VolumeIntent:
    """Represents a deterministic speaker volume intent."""

    action: str
    value: int | None = None


def wants_detailed_response(text: str) -> bool:
    """Return True when user explicitly asks for a detailed answer."""
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in DETAIL_PATTERNS)


def classify_tool_profile(text: str, default_profile: str = PROFILE_CHAT_ONLY) -> str:
    """
    Classify user text into a coarse tool profile.

    The matcher is intentionally conservative. Uncertain inputs use default_profile.
    """
    if not text:
        return default_profile

    t = text.lower()

    if re.search(
        r"\b(тише|громче|убав(ь|ить)|прибав(ь|ить)|volume|louder|quieter|turn up|turn down)\b",
        t,
    ):
        return PROFILE_HOME_CONTROL

    if re.search(
        r"\b(ударени|stress|как произнос|как читается|pronunciation)\b",
        t,
    ):
        return PROFILE_HOME_CONTROL

    if re.search(
        r"\b(нарис\w*|картинк|изображени|фото|иллюстрац|draw|image|picture|illustration)\b",
        t,
    ):
        return PROFILE_CREATIVE

    if re.search(
        r"\b(музык|песн|мелоди|трек|sing|song|music|melody|track)\b"
        r"|\b(включи|сыграй|play)\b.*\b(старую|старое|сохраненн\w*|previous|saved|old)\b",
        t,
    ):
        return PROFILE_CREATIVE

    if re.search(
        r"\b(погода|weather|новост|news|поиск|search|найди|look up|wizard|мудрец)\b",
        t,
    ):
        return PROFILE_RESEARCH

    if re.search(
        r"\b(напомни|напоминан|reminder|email|почт|письм|код|python)\b",
        t,
    ):
        return PROFILE_ORGANIZER

    if re.search(
        r"\b(помнишь|вспомни|remember|recall|раньше говорил|earlier)\b",
        t,
    ):
        return PROFILE_MEMORY

    return default_profile


def resolve_tools_for_profile(profile: str, available_tool_names: Iterable[str]) -> set[str]:
    """
    Resolve profile name to concrete tool names that exist in this runtime.
    """
    available = set(available_tool_names)
    if profile == PROFILE_ALL_TOOLS:
        return available

    requested = PROFILE_TOOL_NAMES.get(profile, set())
    return available.intersection(requested)


def parse_volume_intent(text: str) -> VolumeIntent | None:
    """
    Parse deterministic volume commands from natural language.

    Supported shapes:
    - "тише", "громче"
    - "тише на X", "громче на X"
    - "volume 35", "set volume to 35", "громкость на 35"
    """
    if not text:
        return None

    t = text.lower()

    set_match = re.search(
        r"(?:громкост[ьи]|volume)\D{0,20}(?:на|to)?\s*(\d{1,3})\b",
        t,
    )
    if set_match:
        value = max(0, min(100, int(set_match.group(1))))
        return VolumeIntent(action="set", value=value)

    if re.search(r"\b(тише|потише|убав(ь|ить)|quieter|turn down|lower volume)\b", t):
        by_match = re.search(r"\b(?:на|by)\s*(\d{1,3})\b", t)
        return VolumeIntent(
            action="decrease",
            value=int(by_match.group(1)) if by_match else None,
        )

    if re.search(r"\b(громче|погромче|прибав(ь|ить)|louder|turn up|raise volume)\b", t):
        by_match = re.search(r"\b(?:на|by)\s*(\d{1,3})\b", t)
        return VolumeIntent(
            action="increase",
            value=int(by_match.group(1)) if by_match else None,
        )

    return None
