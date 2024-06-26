import re

from openai import OpenAI
import os
import logging
from collections import deque

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

logger = logging.getLogger(__name__)
client = OpenAI()
model = "gpt-4o"

MAX_MESSAGES = 10
message_history = deque(maxlen=MAX_MESSAGES)
message_history.append({"role": "system", "content": "Your name is Robie."})


def get_openai_response(text):
    def estimate_tokens_russian(text):
        # Count words (including numbers)
        words = len(re.findall(r'\b\w+\b', text))

        # Count punctuation marks
        punctuation = len(re.findall(r'[.,!?;:"]', text))

        # Estimate: Each word is roughly 1.5 tokens, each punctuation is 1 token
        return int(words * 1.5 + punctuation)

    def get_token_count(messages):
        return sum(estimate_tokens_russian(msg["content"]) for msg in messages)

    def summarize_and_compress_history():
        summary_prompt = \
            ("Сделайте краткое изложение ключевых моментов этой беседы, "
             "сосредоточившись на самых важных фактах и контексте. Будьте лаконичны:")
        for msg in message_history:
            if msg["role"] != "system":
                summary_prompt += f"\n{msg['role']}: {msg['content']}"

        summary_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=400  # Reduced for more conservative token usage
        )
        summary = summary_response.choices[0].message.content.strip()

        # Clear the message history except for the system message
        system_message = message_history[0]
        message_history.clear()
        message_history.append(system_message)

        # Add the summary as a system message
        message_history.append({"role": "system", "content": f"Краткое изложение беседы: {summary}"})

    # Add user message to history
    message_history.append({"role": "user", "content": text})

    # Check estimated token count and summarize if necessary
    # Using a lower threshold due to potential underestimation
    while get_token_count(message_history) > 2500:
        summarize_and_compress_history()

    response = client.chat.completions.create(
        model=model,
        messages=list(message_history),
        max_tokens=1000  # Adjust as needed
    )
    response_text = response.choices[0].message.content.strip()

    # Add assistant response to history
    message_history.append({"role": "assistant", "content": response_text})

    logger.info(f"OpenAI response: {text} -> {response_text}")
    return response_text


if __name__ == '__main__':
    while True:
        text = input("You: ")
        if text.lower() in ['exit', 'quit', 'bye']:
            break
        response = get_openai_response(text)
        print(f"Robie: {response}")
