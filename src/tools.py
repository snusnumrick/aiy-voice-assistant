import asyncio
import json
import logging
import os
import random
import re
import subprocess
import time
from datetime import datetime
from functools import wraps
from typing import (
    List,
    Dict,
    Union,
    Any,
    Callable,
    AsyncGenerator,
    Iterable,
    Tuple,
    AsyncIterator,
    Optional,
)

import aiofiles
import geocoder
import pytz
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a given text.

    Args:
        text (str): The text to estimate tokens for.

    Returns:
        int: Estimated number of tokens.
    """
    words = len(re.findall(r"\b\w+\b", text))
    punctuation = len(re.findall(r'[.,!?;:"]', text))
    return int(words * 1.5 + punctuation)


def get_token_count(messages: List[Dict[str, Union[str, Dict]]]) -> int:
    """
    Get the total token count for a list of messages.

    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries.

    Returns:
        int: Total estimated token count.
    """
    count = 0
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            count += estimate_tokens(content)
        elif isinstance(content, dict):
            if "text" in content:
                count += estimate_tokens(content["text"])
            elif "content" in content:
                count += get_token_count(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    count += estimate_tokens(item)
                elif (
                    isinstance(item, dict)
                    and "type" in item
                    and item["type"] == "text"
                    and "text" in item
                ):
                    count += estimate_tokens(item["text"])
        else:
            logger.error(f"unknown message type {content}")

    return count


def get_timezone() -> str:
    """Get the timezone of the current location.

    It uses paid service - use sparingly.
    Example of output: 'America/Los_Angeles'

    :return: The string id of the timezone.
    """
    import googlemaps
    import time

    g = geocoder.ip("me")

    key = os.environ.get("GOOGLE_API_KEY")
    gmaps = googlemaps.Client(key=key)
    timezone = gmaps.timezone(g.latlng, time.time())

    return timezone["timeZoneId"]

def time_string_ms(timezone_string: str) -> str:
    # 07:00.989
    return (
        datetime.now(pytz.utc)
        .astimezone(pytz.timezone(timezone_string))
        .strftime("%M:%S.%f")[:-3]
    )


def get_location_string() -> str:
    """
    :return: A string representing the current location.

    Example usage:
    ```
    result = get_location_string()
    print(result)
    ```

    Expected output:
    ```
    San Francisco, CA, United States
    ```
    """
    g = geocoder.ip("me")
    location_parts = [g.city, g.state, g.country]
    location = (
        ", ".join([part for part in location_parts if part])
        if any(location_parts)
        else ""
    )
    return location


def get_current_date_time_tuple(timezone_string: str) -> Tuple[str, str]:
    """
    :param timezone_string: A string representing the timezone to convert the current date and time to (e.g. 'America/Los_Angeles').
    :return: A tuple with the current date and time in the specified timezone in Russian.

    This method takes a timezone string as input and returns the current date and time in the specified timezone.
    The date is formatted with the month as a word, and the time is formatted in 12-hour format with AM/PM and timezone information.

    Example usage:
    ```
    timezone = 'America/Los_Angeles'
    result = get_current_date_time_location(timezone)
    print(result)
    ```

    Expected output:
    ```
    ("05 мая 2021", "08:30:45 PM, PDT")
    ```
    """
    # Get current date and time in UTC
    now_utc = datetime.now(pytz.utc)

    # Define the timezone you want to convert to (for example, PST)
    timezone = pytz.timezone(timezone_string)
    now_local = now_utc.astimezone(timezone)

    # Format the date with the month as a word
    months = {
        1: "января",
        2: "февраля",
        3: "марта",
        4: "апреля",
        5: "мая",
        6: "июня",
        7: "июля",
        8: "августа",
        9: "сентября",
        10: "октября",
        11: "ноября",
        12: "декабря",
    }
    date_str = now_local.strftime(f"%d {months[now_local.month]} %Y")

    # Format the time in 12-hour format with AM/PM and timezone
    time_str = now_local.strftime("%I:%M:%S %p, %Z")

    return date_str, time_str


def get_current_date_time_location(timezone_string: str) -> str:
    """
    :param timezone_string: A string representing the timezone to convert the current date and time to (e.g. 'America/Los_Angeles').
    :return: A string representing the current date and time in the specified timezone, along with the current location. The message is in Russian.

    This method takes a timezone string as input and returns the current date and time in the specified timezone, along with the current location. The date is formatted with the month as a word, and the time is formatted in 12-hour format with AM/PM and timezone information.

    Example usage:
    ```
    timezone = 'America/Los_Angeles'
    result = get_current_date_time_location(timezone)
    print(result)
    ```

    Expected output:
    ```
    Сегодня 05 мая 2021. Сейчас 08:30:45 PM, PDT. Я нахожусь в San Francisco, CA, United States
    ```
    """
    date_str, time_str = get_current_date_time_tuple(timezone_string)
    location = get_location_string()

    # Prepare the message in Russian
    message = f"Сегодня {date_str}. Сейчас {time_str}."
    if location:
        message += f" Я нахожусь в {location}"

    return message


def get_location() -> str:
    """Human-readable location.

    Example of return: 'In San Jose, California, US.'

    :return: a string representing the location of the user based on their IP address.
    """
    g = geocoder.ip("me")
    location_parts = [g.city, g.state, g.country]
    location = (
        ", ".join([part for part in location_parts if part])
        if any(location_parts)
        else ""
    )

    return f"In {location}."


def get_current_datetime_english(timezone_string: str = "") -> str:
    """Human-readable current date and time in English.

    Example of output: 'Today is 13 July 2024. Now 12:20 PM PDT.'

    :param timezone_string: The string representing the timezone in which the current datetime will be retrieved.
    :return: The formatted string representing the current datetime in English.

    This method takes a timezone string as input and returns a formatted string representing the current datetime in English. The timezone string should be in the format accepted by the pytz.timezone() method.

    The method first sets the timezone based on the provided timezone string. It then retrieves the current time in the specified timezone using the datetime.datetime.now() method. The date is formatted using the "%d %B %Y" format string, which represents the day, month name, and year. The time is formatted using the "%I:%M %p" format string, which represents the 12-hour time with AM/PM indicator.

    The timezone abbreviation is determined using the "%Z" format string. This will provide either "PDT" or "PST" depending on the time of the year.

    Finally, a formatted string is created using the date, time, and timezone abbreviation, and it is returned as the result.
    """
    # Set the timezone
    tz = pytz.timezone(timezone_string) if timezone_string else None

    # Get the current time in the timezone
    current_time = datetime.now(tz)

    # Format the date
    date_str = current_time.strftime("%d %B %Y")

    # Format the time
    time_str = current_time.strftime("%I:%M %p")

    # Determine if it's PDT or PST
    timezone_abbr = current_time.strftime("%Z")

    # Create the formatted string
    formatted_str = f"Today is {date_str}. Now {time_str} {timezone_abbr}."

    return formatted_str


def get_current_date_time_for_facts(timezone_string: str) -> str:
    """
    Get the current date and time in the specified timezone.

    :param timezone_string: the timezone to convert to (e.g. "PST")
    :return: a formatted string representing the current date and time in the specified timezone
    """
    # Get current date and time in UTC
    now_utc = datetime.now(pytz.utc)

    # Define the timezone you want to convert to (for example, PST)
    timezone = pytz.timezone(timezone_string)
    now_local = now_utc.astimezone(timezone)

    # Format the date with the month as a word
    months = {
        1: "января",
        2: "февраля",
        3: "марта",
        4: "апреля",
        5: "мая",
        6: "июня",
        7: "июля",
        8: "августа",
        9: "сентября",
        10: "октября",
        11: "ноября",
        12: "декабря",
    }
    date_str = now_local.strftime(f"%d {months[now_local.month]} %Y")

    # Format the time in 12-hour format with AM/PM and timezone
    time_str = now_local.strftime("%I:%M:%S %p, %Z")

    # Prepare the message in Russian
    message = f"({date_str}, {time_str})"

    return message


def indent_content(content, max_width=120):
    """
    Indent and format content, splitting by sentence boundaries and line length.
    Put newlines before and after special $<word>:<more text>$ parts.

    Args:
    content (str): The text content to format.
    max_width (int): Target maximum line width (default 80).

    Returns:
    str: Formatted and indented content.
    """

    # Replace special parts with placeholders and store them
    special_parts = []

    def replace_special(match):
        special_parts.append(match.group(0))
        return f"\n{{{len(special_parts) - 1}}}\n"

    content = re.sub(r"\$\w+:.+?\$", replace_special, content)
    # Split content into sentences
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    formatted_lines = []

    current_line = []
    current_length = 0
    for sentence in sentences:
        words = sentence.split()

        for word in words:
            if word.startswith("{") and word.endswith("}"):
                # This is a placeholder for a special part
                if current_line:
                    formatted_lines.append(" ".join(current_line))
                formatted_lines.append(special_parts[int(word[1:-1])])
                current_line = []
                current_length = 0

            elif current_length + len(word) + (1 if current_line else 0) > max_width:
                formatted_lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + (1 if current_line else 0)

    if current_line:
        formatted_lines.append(" ".join(current_line))

        # Add a blank line after each sentence, except the last one
        if sentence != sentences[-1]:
            formatted_lines.append("")

    # Join all lines with newline and indentation
    return "    " + "\n    ".join(formatted_lines)


def format_message_history(
    message_history: Iterable[Dict[str, str]], max_width=120
) -> str:
    """
    Format the message history into a formatted string with a specified maximum width.

    :param message_history: A deque containing the message history.
    :param max_width: An optional parameter specifying the maximum width of the formatted string. Default is 120.
    :return: The formatted message history as a string.
    """
    return "\n\n".join(
        [
            f'{msg["role"]}:\n{indent_content(msg["content"], max_width)}'
            for msg in message_history
        ]
    )


async def save_to_conversation(role: str, message: str, timezone: str, max_width=120):
    """Saves the given message to the conversation file."""
    date_str, time_str = get_current_date_time_tuple(timezone)
    indented_msg = indent_content(message.replace("+", ""), max_width)
    formatted = f'{role if role == "assistant" else date_str + ", " + time_str}:\n{indented_msg}\n\n'
    async with aiofiles.open("conversation.txt", "a", encoding="utf-8") as f:
        await f.write(formatted)


def extract_json(text):
    # Try to parse the entire text as JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # If it fails, continue to other methods

    # Patterns to match JSON content within triple backticks
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # For ```json
        r"```\s*([\s\S]*?)\s*```",  # For ```
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue  # Try next match if this one fails

    logger.error("Error: No valid JSON found")
    raise Exception("No valid JSON found")


def clean_response(response: str) -> str:
    """
    Clean the response by removing:
    1. Meta tags in format $tagname: tagcontent$
    2. Text surrounded by asterisks only if:
       - No spaces between asterisks and text
       - Text inside is not a number

    :param response: The response string
    :return: The cleaned response string
    """
    # Remove meta tags
    pattern_tags = r"\$\w+:[^$]*\$"
    response = re.sub(pattern_tags, "", response)

    # Remove text surrounded by asterisks that meets conditions
    pattern_asterisks = r"\*(?!\d+\*)([^\s*]+)\*"
    return re.sub(pattern_asterisks, "", response)


def retry(
    max_retries: int = 5,
    initial_retry_delay: float = 1,
    backoff_factor: float = 2,
    jitter_factor: float = 0.1,
):
    """
    A decorator for implementing retry logic with exponential backoff and jitter.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_retry_delay (float): Initial delay between retries in seconds.
        backoff_factor (float): Factor by which the delay increases with each retry.
        jitter_factor (float): Factor for randomness in retry delay.

    Returns:
        Callable: Decorated function with retry logic.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == (max_retries - 1):
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    retry_time = initial_retry_delay * (backoff_factor**attempt)
                    jitter = random.uniform(0, jitter_factor * retry_time)
                    total_delay = retry_time + jitter
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {total_delay:.2f} seconds..."
                    )
                    time.sleep(total_delay)

        return wrapper

    return decorator


def retry_async(
    max_retries: int = 5,
    initial_retry_delay: float = 1,
    backoff_factor: float = 2,
    jitter_factor: float = 0.1,
):
    """
    A decorator for implementing retry logic with exponential backoff and jitter.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_retry_delay (float): Initial delay between retries in seconds.
        backoff_factor (float): Factor by which the delay increases with each retry.
        jitter_factor (float): Factor for randomness in retry delay.

    Returns:
        Callable: Decorated function with retry logic.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == (max_retries - 1):
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    retry_time = initial_retry_delay * (backoff_factor**attempt)
                    jitter = random.uniform(0, jitter_factor * retry_time)
                    total_delay = retry_time + jitter
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {total_delay:.2f} seconds..."
                    )
                    await asyncio.sleep(total_delay)

        return wrapper

    return decorator


def retry_async_generator(
    max_retries: int = 5,
    initial_retry_delay: float = 1,
    backoff_factor: float = 2,
    jitter_factor: float = 0.1,
):
    """
    A decorator for implementing retry logic with exponential backoff and jitter.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_retry_delay (float): Initial delay between retries in seconds.
        backoff_factor (float): Factor by which the delay increases with each retry.
        jitter_factor (float): Factor for randomness in retry delay.

    Returns:
        Callable: Decorated function with retry logic.
    """

    def decorator(
        func: Callable[..., AsyncGenerator[Any, None]],
    ) -> Callable[..., AsyncGenerator[Any, None]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            for attempt in range(max_retries):
                try:
                    async for item in func(*args, **kwargs):
                        yield item
                    return
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    retry_time = initial_retry_delay * (backoff_factor**attempt)
                    jitter = random.uniform(0, jitter_factor * retry_time)
                    total_delay = retry_time + jitter
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {total_delay:.2f} seconds..."
                    )
                    await asyncio.sleep(total_delay)

        return wrapper

    return decorator


def extract_sentences(
    text: str, expected_enumeration: Optional[List[int]] = None
) -> List[str]:
    """
    Extracts sentences from the given text while preserving special patterns and numbered lists.

    This function handles text that may contain special patterns (starting with $)
    such as emotion tags or JSON-like structures. It preserves these patterns within
    the sentences, and correctly splits sentences at punctuation marks, even when
    they appear immediately after a special pattern without a space. It also preserves
    numbered list items as separate sentences when they match the expected enumeration.

    The function is designed to work with both English and Russian text.

    Args:
        text (str): The input text to be processed. May contain special patterns
                    starting with $, as well as normal sentences in English or Russian,
                    and numbered list items.
        expected_enumeration (Optional[List[int]]): A list containing the expected number
                    for the next enumeration item. If provided, it should be a list with
                    one integer element. This list will be updated as enumeration is processed.

    Returns:
        List[str]: A list of extracted sentences. Special patterns are preserved within
                   the sentences they were originally associated with, and numbered list
                   items are treated as separate sentences when they match the expected enumeration.

    Note:
    - The function treats $... patterns (complete or incomplete) as part of the sentence.
    - It correctly splits sentences at punctuation marks, even immediately after special patterns.
    - It handles ellipsis and multiple punctuation marks as single sentence endings.
    - Incomplete sentences or patterns at the end of the text are preserved.
    - Numbered list items are treated as separate sentences only when they match the expected enumeration.
    - The function handles cases where sentence-ending punctuation is not followed by a space.
    - The expected_enumeration list is updated as valid enumeration items are encountered.
    """
    logger.debug(f"Extracting sentences from: {text}")

    # Define patterns
    special_pattern = r"\$[^$]+(?:\$|$)"
    sentence_end_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|:|\))\s"
    numbered_list_pattern = r"\s*\d+\.(?:\s|$)"

    # Find all special patterns and their positions
    special_matches = list(re.finditer(special_pattern, text))

    # Split text into segments (alternating between normal text and special patterns)
    segments = []
    last_end = 0
    for match in special_matches:
        if match.start() > last_end:
            segments.append(text[last_end : match.start()])
        segments.append(match.group())
        last_end = match.end()
    if last_end < len(text):
        segments.append(text[last_end:])

    # Combine segments into sentences
    sentences = []
    current_sentence = ""
    for segment in segments:
        if segment.startswith("$"):
            current_sentence += segment
        else:
            # Split the non-special segment by sentence endings and numbered list items
            parts = re.split(
                f"({sentence_end_pattern})|({numbered_list_pattern})", segment
            )
            parts = [p for p in parts if p is not None]
            for i, part in enumerate(parts):
                if re.match(sentence_end_pattern, part) or (
                    i > 0
                    and not part.strip()
                    and re.search(sentence_end_pattern, parts[i - 1] + part)
                ):
                    # End the current sentence if we encounter sentence-ending punctuation
                    # or if we have a space after sentence-ending punctuation
                    if current_sentence:
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                elif re.match(numbered_list_pattern, part.strip()):
                    # Check if this matches our expected enumeration
                    match = re.match(r"\s*(\d+)\.", part)
                    if match:
                        num = int(match.group(1))
                        if expected_enumeration is None or (
                            expected_enumeration and num == expected_enumeration[0]
                        ):
                            if current_sentence:
                                sentences.append(current_sentence.strip())
                            current_sentence = part
                            if expected_enumeration:
                                expected_enumeration[0] = num + 1
                            else:
                                expected_enumeration = [num + 1]
                        else:
                            current_sentence += part
                    else:
                        current_sentence += part
                else:
                    current_sentence += part

    if current_sentence:  # Add any remaining text as a sentence
        sentences.append(current_sentence.strip())

    # Remove empty sentences
    sentences = [s for s in sentences if s]

    logger.debug(f"Extracted sentences: {sentences}")
    return sentences


def yield_complete_sentences(
    func: Callable[..., AsyncIterator[str]],
) -> Callable[..., AsyncIterator[str]]:
    """
    A decorator that modifies an async generator function to yield complete sentences.

    This decorator wraps around functions that yield text in chunks. It buffers the
    yielded text and only yields complete sentences. Any incomplete sentence at the
    end of the input is yielded as-is.

    Args:
        func (Callable[..., AsyncIterator[str]]): The async generator function to be decorated.
            This function should yield strings (text chunks).

    Returns:
        Callable[..., AsyncIterator[str]]: A wrapped version of the input function that
        yields complete sentences.

    Example:
        @yield_complete_sentences
        async def my_text_generator() -> AsyncIterator[str]:
            yield "Hello, wo"
            yield "rld! How are"
            yield " you today?"

        # Using the decorated function
        async for sentence in my_text_generator():
            print(sentence)

        # Output:
        # Hello, world!
        # How are you today?

    Note:
        This decorator assumes the existence of an `extract_sentences` function
        that can split a string into a list of sentences. Make sure to implement
        this function according to your specific needs.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        current_text = ""
        async for text in func(*args, **kwargs):
            current_text += text
            sentences = extract_sentences(current_text)

            if len(sentences) > 1:
                for sentence in sentences[:-1]:
                    yield sentence
                current_text = sentences[-1]

        sentences = extract_sentences(current_text)
        for sentence in sentences:
            yield sentence

    return wrapper


def combine_audio_files(file_list: List[str], output_filename: str) -> None:
    """
    Combine multiple audio files into a single file.

    Args:
        file_list (List[str]): List of audio file paths to combine.
        output_filename (str): Path to save the combined audio file.
    """
    logger.debug(f"Combining {len(file_list)} audio files into {output_filename}")
    combined = AudioSegment.empty()
    for file in file_list:
        audio = AudioSegment.from_wav(file)
        combined += audio
    logger.debug(f"Exporting combined audio to {output_filename}")
    combined.export(output_filename, format="wav")
    logger.debug(f"Exported combined audio to {output_filename}")


def fix_stress_marks_russian(s: str) -> str:
    """
    Corrects the placement of stress marks in Russian text.

    This function moves any '+' sign that's incorrectly placed before a consonant
    forward to the next vowel. It handles both Cyrillic and Latin alphabets, as well
    as the soft sign (ь) and hard sign (ъ).

    Parameters:
    s (str): The input string containing Russian text with stress marks.

    Returns:
    str: The corrected string with properly placed stress marks.

    Behavior:
    1. Identifies vowels in both Cyrillic and Latin alphabets.
    2. Recognizes the soft sign (ь) and hard sign (ъ) as special characters.
    3. Moves '+' forward if found before a consonant until it precedes a vowel.
    4. Preserves all original characters, only modifying the position of '+' signs.
    5. Keeps '+' at the end of the string if present.

    Examples:
    >>> fix_stress_marks_russian("п+ривет")
    "пр+ивет"
    >>> fix_stress_marks_russian("об+ъявление")
    "объ+явление"
    >>> fix_stress_marks_russian("подъ+езд")
    "подъ+езд"
    >>> fix_stress_marks_russian("семь+я")
    "семь+я"
    >>> fix_stress_marks_russian("+слово")
    "+слово"
    >>> fix_stress_marks_russian("конец+")
    "конец+"

    Limitations:
    - Does not validate linguistic correctness of stress placement.
    - Does not handle other types of diacritical marks or punctuation.
    - Assumes '+' is exclusively used as a stress mark in the input text.

    Note:
    This function is useful for correcting automatically generated or improperly
    formatted stressed Russian text. It can be integrated into larger text processing
    pipelines for Russian language learning materials or linguistic analysis tools.
    """
    # fix stress marks in a string by moving any '+' sign that's incorrectly placed before a consonant forward to the next vowel.
    vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouAEIOU"
    non_consonants = vowels + "ьъЬЪ"  # Include soft and hard signs
    result = []
    stress_pending = False

    for char in s:
        if char == "+":
            stress_pending = True
        elif char in vowels and stress_pending:
            result.append("+")
            result.append(char)
            stress_pending = False
        elif char in non_consonants:
            result.append(char)
        else:  # consonant
            if stress_pending:
                result.append(char)
            else:
                result.append(char)

    # If there's a pending stress at the end, just append it
    if stress_pending:
        result.append("+")

    result = "".join(result)
    if result != s:
        logger.info(f"fixed stress mark: {s} -> {result}")
    return result


def test():
    # tz = get_timezone()
    # print(tz)
    # print(get_current_datetime_english(tz) + " " + get_location())
    # print(get_current_date_time_for_facts(tz))
    #     print(indent_content('''Спасибо за предложение, Антон. Ты прав, возможно, другая формулировка поможет мне лучше усвоить это правило. Давай попробую:
    #
    # "Знак ударения '+' всегда ставится непосредственно перед ударной гласной в слове."
    #
    # Или еще короче:
    #
    # "'+' идет перед ударной гласной."
    #
    # Как тебе такие варианты? Может быть, эти более лаконичные формулировки будут легче запомнить. Спасибо, что помогаешь мне улучшить мою работу с языком. Твой подход к обучению очень ценен.'''))
    expected_enumeration = [1]
    print(extract_sentences("от 0 до 10. Но сначала,", expected_enumeration))
    # print(extract_sentences("abc$x.f"))
    # print(extract_sentences("$x.f$abc. 123"))
    # print(extract_sentences("qr.$x.f$abc.123"))
    # print(extract_sentences("First sentence... Second sentence."))
    # print(extract_sentences("Is this a question?! Yes, it is!"))
    # print(extract_sentences("Hello $world$! This is a $test. And $another one$."))
    # print(extract_sentences("Hello $world$! This is a $test. And $another one$."))
    # print(extract_sentences("$pattern1$. $pattern2$. Normal sentence."))
    # print(extract_sentences("Start $mid1$ middle $mid2$ end."))
    # print(extract_sentences("$incomplete... Next sentence."))
    # print(extract_sentences('$remember: начало факта. Второе предложение. $ Остальной текст.'))
    # print(extract_sentences('This is a sentence. This is another one!'))
    # print(extract_sentences('text) 1. one. 2. two.'))
    # print(extract_sentences('Intro: 1. one. 2. two.'))
    # print(extract_sentences('Швеция - 3:56.92 8. Нидерланды - 3:59.52'))
    # print(extract_sentences('$pattern1$.$pattern2$.'))
    # print(extract_sentences('1. First. And. 2. Second'))
    # print(extract_sentences('1. First point. 2. Second point with multiple sentences. It continues here. 3. Third point.'))
    # print(extract_sentences('This is an $incomplete pattern'))
    pass


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    test()
