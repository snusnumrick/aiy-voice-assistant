import re
from typing import List, Dict, Union
import logging

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
        else:
            logger.error(f"unknown message type {content}")

    return count