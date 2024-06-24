from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)


def get_openai_response(text):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ],
        max_tokens=100
    )
    text2return = response.choices[0].message['content'].strip()
    logger.info(f"OpenAI response: {text} -> {text2return}")

    return text2return
