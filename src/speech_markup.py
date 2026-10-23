"""Canonical application pauses and provider-specific rendering.

Yandex keeps its context-dependent cues. Other durations below are application
approximations, not claimed equivalents of Yandex's context-dependent timing.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)
PAUSE_SIZES = ("tiny", "small", "medium", "large", "huge")
_SECONDS = dict(zip(PAUSE_SIZES, ("0.1", "0.3", "0.6", "1.0", "2.0")))
_CANONICAL = re.compile(r"<\[\s*([a-z]+)\s*(\])?\s*>", re.IGNORECASE)
_HTML_PAUSE = re.compile(
    r"<(tiny|small|medium|large|huge)>\[\1\]</\1>", re.IGNORECASE
)
_RENDERED = re.compile(
    r'<\[(?:tiny|small|medium|large|huge|accented)\]>'
    r'|<break\s+time=["\x27][\d.]+s["\x27]\s*/>'
    r'|\[(?:short |long )?pause\]',
    re.IGNORECASE,
)


def pause_instructions(language: str) -> str:
    """The prompt uses one application format for every spoken language."""
    tags = ", ".join(f"<[{size}]>" for size in PAUSE_SIZES)
    if language == "english":
        return (
            f"For optional speech pauses in any language use application cues {tags}. "
            "Insert them literally between spoken words, not HTML or SSML. "
            "The application translates each cue for the selected speech provider. "
            "Use pauses sparingly; never return only pause cues. "
        )
    return (
        f"Для необязательных пауз в речи на любом языке используй метки приложения {tags}. "
        "Вставляй их дословно между произносимыми словами, это не HTML и не SSML. "
        "Приложение преобразует метку для выбранного голосового движка. "
        "Используй паузы умеренно; не отвечай одними метками. "
    )


def incomplete_pause_start(text: str) -> Optional[int]:
    """Reserve an unfinished <[... tail, including whitespace inside a tag."""
    start = text.rfind("<[")
    if start >= 0 and ">" not in text[start:]:
        return start
    return None


def _error(reason: str, enabled: bool) -> None:
    if enabled:
        logger.error("Malformed service metadata removed: tag=pause reason=%s", reason)


def normalize_pause_markup(text: str, *, log_errors: bool = True) -> str:
    """Recover known malformed cues; hide unknown or unfinished pause metadata."""
    def html_pause(match: re.Match) -> str:
        _error("html_pause_wrapper", log_errors)
        return f"<[{match[1].lower()}]>"

    text = _HTML_PAUSE.sub(html_pause, text)

    def pause(match: re.Match) -> str:
        size = match[1].lower()
        # Do not mistake Yandex's separate emphasis cue for an invalid pause.
        if size not in PAUSE_SIZES and size != "accented":
            _error("unknown_pause_size", log_errors)
            return " "
        if not match[2]:
            _error("missing_bracket", log_errors)
        return f"<[{size}]>"

    text = _CANONICAL.sub(pause, text)
    start = incomplete_pause_start(text)
    if start is not None:
        _error("truncated_metadata", log_errors)
        text = text[:start]
    return text


def strip_pause_markup(text: str, *, log_errors: bool = True) -> str:
    """Remove pauses from stored history and recognition context."""
    return _RENDERED.sub("", normalize_pause_markup(text, log_errors=log_errors))


def render_pause_markup(text: str, style: str) -> str:
    """Render only at the chosen provider boundary; already-rendered text is stable."""
    text = normalize_pause_markup(text)
    if not strip_pause_markup(text, log_errors=False).strip():
        return ""

    def render(match: re.Match) -> str:
        size = match[1].lower()
        if style == "yandex":
            return match[0]
        if size == "accented":
            return " "
        if style == "ssml":
            return f'<break time="{_SECONDS[size]}s" />'
        if style == "audio_tags":
            return "[short pause]" if size in ("tiny", "small", "medium") else "[long pause]"
        # Engines without a documented pause format get plain speech, not cue names.
        return " "

    return _CANONICAL.sub(render, text)


def pause_safe_split(text: str, split_at: int, limit: int) -> int:
    """Keep rendered pause tags atomic when enforcing a provider's text limit."""
    for match in _RENDERED.finditer(text):
        if match.start() < split_at < match.end():
            split_at = match.start() or match.end()
            if split_at > limit:
                raise ValueError("Pause markup exceeds the provider text limit")
            break
    return split_at
