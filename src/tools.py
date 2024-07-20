from datetime import datetime
from collections import deque
import re
import geocoder
import pytz
import os
from typing import List, Dict, Union
import logging
import json

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a given text.

    Args:
        text (str): The text to estimate tokens for.

    Returns:
        int: Estimated number of tokens.
    """
    words = len(re.findall(r'\b\w+\b', text))
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
        content = msg['content']
        if isinstance(content, str):
            count += estimate_tokens(content)
        elif isinstance(content, dict):
            if 'text' in content:
                count += estimate_tokens(content['text'])
            elif 'content' in content:
                count += get_token_count(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    count += estimate_tokens(item)
                elif isinstance(item, dict) and 'type' in item and item['type'] == 'text' and 'text' in item:
                    count += estimate_tokens(item['text'])
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

    g = geocoder.ip('me')

    key = os.environ.get('GOOGLE_API_KEY')
    gmaps = googlemaps.Client(key=key)
    timezone = gmaps.timezone(g.latlng, time.time())

    return timezone["timeZoneId"]


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
    # Get current date and time in UTC
    now_utc = datetime.now(pytz.utc)

    # Define the timezone you want to convert to (for example, PST)
    timezone = pytz.timezone(timezone_string)
    now_local = now_utc.astimezone(timezone)

    # Format the date with the month as a word
    months = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня', 7: 'июля', 8: 'августа',
              9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
    date_str = now_local.strftime(f"%d {months[now_local.month]} %Y")

    # Format the time in 12-hour format with AM/PM and timezone
    time_str = now_local.strftime("%I:%M:%S %p, %Z")

    # Get current location
    g = geocoder.ip('me')
    location_parts = [g.city, g.state, g.country]
    location = ', '.join([part for part in location_parts if part]) if any(location_parts) else ''

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
    g = geocoder.ip('me')
    location_parts = [g.city, g.state, g.country]
    location = ', '.join([part for part in location_parts if part]) if any(location_parts) else ''

    return f"In {location}."


def get_current_datetime_english(timezone_string: str) -> str:
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
    tz = pytz.timezone(timezone_string)

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
    months = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня', 7: 'июля', 8: 'августа',
              9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
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
        return f'\n{{{len(special_parts)-1}}}\n'

    content = re.sub(r'\$\w+:.+?\$', replace_special, content)
    # Split content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content.strip())
    formatted_lines = []

    current_line = []
    current_length = 0
    for sentence in sentences:
        words = sentence.split()

        for word in words:
            if word.startswith('{') and word.endswith('}'):
                # This is a placeholder for a special part
                if current_line:
                    formatted_lines.append(' '.join(current_line))
                formatted_lines.append(special_parts[int(word[1:-1])])
                current_line = []
                current_length = 0

            elif current_length + len(word) + (1 if current_line else 0) > max_width:
                formatted_lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + (1 if current_line else 0)

    if current_line:
        formatted_lines.append(' '.join(current_line))

        # Add a blank line after each sentence, except the last one
        if sentence != sentences[-1]:
            formatted_lines.append('')

    # Join all lines with newline and indentation
    return '    ' + '\n    '.join(formatted_lines)


def format_message_history(message_history: deque, max_width=120) -> str:
    """
    Format the message history into a formatted string with a specified maximum width.

    :param message_history: A deque containing the message history.
    :param max_width: An optional parameter specifying the maximum width of the formatted string. Default is 120.
    :return: The formatted message history as a string.

    """
    return "\n\n".join([f'{msg["role"]}:\n{indent_content(msg["content"], max_width)}' for msg in message_history])


def extract_json(text):
    # Pattern to match JSON content within triple backticks
    pattern = r'```json\s*([\s\S]*?)\s*```'

    # Find all matches
    matches = re.findall(pattern, text)

    if not matches:
        return None

    # Get the first match (assuming there's only one JSON block)
    json_str = matches[0]

    try:
        # Parse the JSON string
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        logger.error(f"Error: Invalid JSON format: {e}")
        return None


def test():
    # tz = get_timezone()
    # print(tz)
    # print(get_current_datetime_english(tz) + " " + get_location())
    # print(get_current_date_time_for_facts(tz))
    print(indent_content('''Спасибо за предложение, Антон. Ты прав, возможно, другая формулировка поможет мне лучше усвоить это правило. Давай попробую:

"Знак ударения '+' всегда ставится непосредственно перед ударной гласной в слове."

Или еще короче:

"'+' идет перед ударной гласной."

Как тебе такие варианты? Может быть, эти более лаконичные формулировки будут легче запомнить. Спасибо, что помогаешь мне улучшить мою работу с языком. Твой подход к обучению очень ценен.'''))


if __name__ == '__main__':
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    test()