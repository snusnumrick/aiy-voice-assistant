"""Recognize service metadata independently of whether its payload decodes.

Only known tag openings reserve text. A JSON object's balanced boundary lets us
recover speech after a missing '$'; an unfinished payload stays hidden at EOF.
"""

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

_NAMES = ("emotion", "lang", "remember", "rule", "speaker", "tool_filler")
_OPENING = re.compile(r"\$(" + "|".join(_NAMES) + r"):", re.IGNORECASE)
_PREFIXES = tuple("$" + name + ":" for name in _NAMES)
_QUOTES = {'"': '"', "“": "”", "'": "'"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceTag:
    """A hidden envelope; complete describes its boundary, not JSON validity."""

    name: str
    start: int
    end: int
    payload: str
    complete: bool = True


def _possible_opening(text: str) -> bool:
    lower = text.lower()
    return any(prefix.startswith(lower) or lower.startswith(prefix) for prefix in _PREFIXES)


def _object_end(text: str, start: int) -> int:
    """Locate a container boundary, including invalid JSON with smart quotes."""
    stack = []
    quote = ""
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in _QUOTES:
            quote = _QUOTES[char]
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack.pop() != char:
                return -1
            if not stack:
                return pos + 1
    return -1


def _closing_dollar(
    text: str, start: int, *, final: bool, nested: bool = False, quoted: bool = False
) -> int:
    """Find an explicit terminator without consuming the next service opening."""
    pos = start
    quote = ""
    escaped = False
    while pos < len(text):
        char = text[pos]
        if quoted and quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif quoted and char in _QUOTES:
            quote = _QUOTES[char]
        elif char == "$" and not text[pos + 1:pos + 2].isdigit():
            opening = _OPENING.match(text, pos)
            if opening:
                if not nested:
                    return -1
                end = _closing_dollar(text, opening.end(), final=final)
                if end < 0:
                    return -1
                pos = end + 1
                continue
            if _possible_opening(text[pos:]) and not (final and pos == len(text) - 1):
                return -1
            return pos
        pos += 1
    return -1


def _orphan_field_start(text: str, end: int) -> tuple[int, bool]:
    """Recognize an orphaned light/voice field from the old prompt example."""
    tail = text[end:].lstrip()
    if not tail.startswith(","):
        return -1, False
    tail = tail[1:].lstrip()
    if not tail:
        return -1, True
    close_quote = _QUOTES.get(tail[0])
    if close_quote is None:
        return -1, False
    key_end = tail.find(close_quote, 1)
    if key_end < 0:
        return -1, any(key.startswith(tail[1:]) for key in ("light", "voice"))
    if tail[1:key_end] not in ("light", "voice"):
        return -1, False
    tail = tail[key_end + 1:].lstrip()
    if not tail:
        return -1, True
    if not tail.startswith(":"):
        return -1, False
    tail = tail[1:].lstrip()
    if not tail:
        return -1, True
    return (len(text) - len(tail), False) if tail.startswith("{") else (-1, True)


def _optional_dollar(text: str, end: int, final: bool) -> tuple[int, bool]:
    pos = end
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos == len(text):
        return end, final
    if text[pos] == "$":
        suffix = text[pos:]
        if suffix == "$":
            return (pos + 1, True) if final else (end, False)
        if not _possible_opening(suffix) and not suffix[1:2].isdigit():
            return pos + 1, True
    return end, True


def iter_service_tags(text: str, *, final: bool = True) -> Iterator[ServiceTag]:
    """Yield metadata spans, marking ambiguous/unclosed tails as incomplete."""
    pos = 0
    while pos < len(text):
        start = text.find("$", pos)
        if start < 0:
            return
        match = _OPENING.match(text, start)
        if not match:
            suffix = text[start:]
            if _possible_opening(suffix) and (not final or suffix != "$"):
                names = [name for name in _NAMES if ("$" + name + ":").startswith(suffix.lower())]
                yield ServiceTag("/".join(names), start, len(text), "", False)
                return
            pos = start + 1
            continue

        name = match[1].lower()
        payload_start = match.end()
        value_start = payload_start
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        end = -1
        payload_end = payload_start
        complete = True
        if name == "emotion" and text[value_start:value_start + 1] == "{":
            payload_end = _object_end(text, value_start)
            if payload_end >= 0:
                end = payload_end
                while True:
                    # Recover the orphaned voice field shown by the old prompt's
                    # extra brace. Its boundary is still structural, not JSON decoding.
                    while end < len(text) and text[end] == "}":
                        end += 1
                    field_start, pending_field = _orphan_field_start(text, end)
                    if pending_field:
                        end = -1
                        break
                    if field_start < 0:
                        break
                    end = _object_end(text, field_start)
                    if end < 0:
                        break
                payload_end = end
                if end >= 0:
                    end, complete = _optional_dollar(text, end, final)
        elif name == "lang":
            code = re.match(r"[A-Za-z]+", text[value_start:])
            if code:
                payload_end = value_start + code.end()
                end, complete = _optional_dollar(text, payload_end, final)

        if end < 0:
            closing = _closing_dollar(
                text, payload_start, final=final, nested=name == "rule", quoted=name == "emotion"
            )
            if closing >= 0:
                payload_end, end = closing, closing + 1
            else:
                next_tag = _OPENING.search(text, payload_start) if name != "emotion" else None
                end = next_tag.start() if next_tag else len(text)
                payload_end, complete = end, False

        yield ServiceTag(name, start, end, text[payload_start:payload_end].strip(), complete)
        pos = end


def report_malformed_metadata(tag: ServiceTag, text: str) -> None:
    """Report one consumed envelope using only fixed diagnostic codes.

    Recognition is deliberately silent: speculative stream scans must not log.
    Call this only when removing a tag or committing it to downstream parsing.
    Typographic quotes recover the structural boundary, not JSON semantics.
    """
    reasons = []
    if not tag.complete:
        reasons.append("truncated_metadata")
    else:
        if text[tag.end - 1:tag.end] != "$":
            reasons.append("missing_delimiter")
        if tag.name == "emotion":
            try:
                decoded = json.loads(tag.payload or "{}")
                if not isinstance(decoded, dict):
                    reasons.append("invalid_payload_type")
            except json.JSONDecodeError:
                reasons.append("invalid_json")
                if any(quote in tag.payload for quote in ("“", "”")):
                    reasons.append("typographic_quotes")
        elif tag.name == "lang" and not re.fullmatch(r"[A-Za-z]+", tag.payload):
            reasons.append("invalid_language")
    if reasons:
        logger.error(
            "Malformed service metadata removed: tag=%s reason=%s",
            tag.name, ",".join(reasons),
        )


def strip_service_metadata(text: str, *, log_errors: bool = True) -> str:
    """Remove service data; retain speech explicitly wrapped as a tool filler."""
    pieces = []
    pos = 0
    for tag in iter_service_tags(text):
        if log_errors:
            report_malformed_metadata(tag, text)
        pieces.append(text[pos:tag.start])
        if tag.name == "tool_filler" and tag.complete:
            pieces.append(strip_service_metadata(tag.payload, log_errors=log_errors))
        pos = tag.end
    pieces.append(text[pos:])
    return "".join(pieces)


def split_complete_meta_tag_prefix(
    text: str, *, final: bool = False, log_errors: bool = True
) -> tuple[str, str]:
    """Buffer only potential service tags and trailing partial spoken words.

    Currency has no special meaning. Completed sentence chunks can pass through;
    partial words wait so an arbitrary network split cannot introduce a space.
    """
    tags = list(iter_service_tags(text, final=final))
    if final:
        pieces = []
        pos = 0
        for tag in tags:
            if log_errors:
                report_malformed_metadata(tag, text)
            if not tag.complete:
                pieces.append(text[pos:tag.start])
                pos = tag.end
        pieces.append(text[pos:])
        return "".join(pieces), ""
    for tag in tags:
        if not tag.complete:
            if log_errors:
                for committed in tags:
                    if committed.end <= tag.start:
                        report_malformed_metadata(committed, text)
            return text[:tag.start], text[tag.start:]
    if not text:
        return text, ""

    # Do not cut inside a complete tag (which may contain spaces or punctuation).
    pos = tags[-1].end if tags else 0
    safe = pos
    for index in range(pos, len(text)):
        char = text[index]
        sentence_end = (
            char in ".!?"
            and (index == 0 or not text[index - 1].isdigit())
            and (index + 1 == len(text) or text[index + 1].isspace())
        )
        if char.isspace() or sentence_end:
            safe = index + 1
    if log_errors:
        for tag in tags:
            if tag.end <= safe:
                report_malformed_metadata(tag, text)
    return text[:safe], text[safe:]
