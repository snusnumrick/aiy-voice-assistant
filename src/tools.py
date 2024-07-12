import re
from typing import List, Dict


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


def get_token_count(messages: List[Dict[str, str]]) -> int:
    """
    Get the total token count for a list of messages.

    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries.

    Returns:
        int: Total estimated token count.
    """
    return sum(estimate_tokens(msg["content"]) for msg in messages)