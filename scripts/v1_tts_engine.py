#!/usr/bin/env python3
"""
Example: Yandex TTS using API v1 (per-character pricing)
instead of SDK v3 (per-request pricing)

This saves money when using sentence splitting
"""

import os
import aiohttp
import aiofiles
import asyncio
from typing import Optional
from urllib.parse import urlencode

# API v1 endpoint
V1_ENDPOINT = "https://tts.api.cloud.yandex.net/v1/synthesize"

# API v1 pricing
V1_RATE_PER_CHAR = 0.000011  # $ per character

async def synthesize_v1(
    text: str,
    voice: str = "ermil",
    lang: str = "ru-RU",
    filename: str = "output.wav",
    speed: float = 1.0,
    session: Optional[aiohttp.ClientSession] = None
) -> bool:
    """
    Synthesize speech using Yandex TTS API v1

    Args:
        text: Text to synthesize
        voice: Voice to use (default: ermil)
        lang: Language code (default: ru-RU)
        filename: Output filename
        speed: Speech speed (default: 1.0)
        session: Optional aiohttp session

    Returns:
        True if successful
    """

    # Get API key
    api_key = os.environ.get("YANDEX_API_KEY")
    if not api_key:
        raise ValueError("YANDEX_API_KEY not found in environment")

    # Calculate cost (for logging)
    char_count = len(text)
    cost_usd = char_count * V1_RATE_PER_CHAR

    # Log the usage (you can customize this)
    print(f"TTS v1 | {char_count} chars | ${cost_usd:.6f} | {lang}/{voice}")

    # Prepare request
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "text": text,
        "lang": lang,
        "voice": voice,
        "speed": speed,
        "format": "wav"
    }

    # Make request
    if session:
        async with session.post(V1_ENDPOINT, headers=headers, data=data) as response:
            if response.status == 200:
                async with aiofiles.open(filename, "wb") as f:
                    await f.write(await response.read())
                return True
            else:
                print(f"Error: {response.status} - {await response.text()}")
                return False
    else:
        async with aiohttp.ClientSession() as session:
            async with session.post(V1_ENDPOINT, headers=headers, data=data) as response:
                if response.status == 200:
                    async with aiofiles.open(filename, "wb") as f:
                        await f.write(await response.read())
                    return True
                else:
                    print(f"Error: {response.status} - {await response.text()}")
                    return False


def synthesize_v1_sync(
    text: str,
    voice: str = "ermil",
    lang: str = "ru-RU",
    filename: str = "output.wav",
    speed: float = 1.0
) -> bool:
    """
    Synchronous version of v1 synthesis
    """

    import requests

    api_key = os.environ.get("YANDEX_API_KEY")
    if not api_key:
        raise ValueError("YANDEX_API_KEY not found in environment")

    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "text": text,
        "lang": lang,
        "voice": voice,
        "speed": speed,
        "format": "wav"
    }

    response = requests.post(V1_ENDPOINT, headers=headers, data=data)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False


# Example usage
if __name__ == "__main__":
    # Test with short sentence
    text = "Привет!"
    cost = len(text) * V1_RATE_PER_CHAR
    print(f"Cost for '{text}': ${cost:.6f} (was $0.001333 with v3)")

    # Run async synthesis
    asyncio.run(synthesize_v1(text, filename="test_v1.wav"))
