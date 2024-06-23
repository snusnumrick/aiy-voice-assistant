import openai
import os

# Load environment variables from .env file
openai.api_key = os.getenv('OPENAI_API_KEY')


def get_openai_response(text):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content'].strip()
