import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import aiohttp
import requests
from pydub import AudioSegment

from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
from src.server_utils import get_server_url

# Optional dependency for MP3 metadata
try:
    from mutagen.id3 import (
        COMM,
        ID3,
        TALB,
        TCOM,
        TCON,
        TDRC,
        TENC,
        TIT2,
        TLEN,
        TPE1,
        TPE2,
        USLT,
    )

    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False
    logger = logging.getLogger(__name__)
    logger.info("mutagen not installed, MP3 metadata tagging disabled")

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiMusicTool:
    def __init__(self, config: Config, response_player):
        """
        Initialize Gemini music tool.

        Args:
            config: Configuration object
            response_player: ResponsePlayer instance for audio playback
        """
        self.config = config
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.response_player = response_player
        self._cleanup_files = []  # Track temp files for cleanup

        # Discover cubie-server music folder (same as MiniMax tool)
        self.music_dir = self._discover_music_folder()
        if self.music_dir:
            logger.info(f"Using music folder from cubie-server: {self.music_dir}")
        else:
            self.music_dir = Path("/tmp/generated_music")
            logger.warning(f"cubie-server not found, using fallback: {self.music_dir}")
            self.music_dir.mkdir(parents=True, exist_ok=True)

    def _discover_music_folder(self) -> Path:
        """Discover cubie-server and get music folder path"""
        server_url = get_server_url()

        try:
            logger.info(f"Discovering cubie-server at {server_url}...")
            response = requests.get(f"{server_url}/api/config/folders", timeout=5)

            if response.status_code == 200:
                config = response.json()
                music_folder = config.get("music")
                if music_folder and os.path.exists(music_folder):
                    logger.info(f"Server discovered! Music folder: {music_folder}")
                    return Path(music_folder)
                else:
                    logger.warning(f"Server returned config but music folder not found: {music_folder}")
            else:
                logger.warning(f"Server returned status {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Cannot connect to cubie-server: {e}")
        except Exception as e:
            logger.warning(f"Error discovering music folder: {e}")

        return None

    def _refresh_music_library(self):
        """Tell cubie-server to refresh its music library"""
        server_url = get_server_url()

        try:
            response = requests.post(f"{server_url}/api/refresh", timeout=5)
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Music library refreshed! ({result.get('count', 'N/A')} tracks)")
                return True
            else:
                logger.warning(f"Refresh returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not refresh library: {e}")
            return False

    def tool_definition(self) -> Tool:
        """Return tool definition for AI model"""
        return Tool(
            name="generate_music_gemini",
            description="Generate music using Google Gemini Lyria API. ",
            iterative=True,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description=(
                        "A detailed description of the music to generate. The more specific, the better. "
                        "Include any combination of: "
                        "Genre (e.g. 'lo-fi hip hop', 'jazz fusion', 'cinematic orchestral'); "
                        "Instruments (e.g. 'Fender Rhodes piano', 'slide guitar', 'TR-808 drum machine'); "
                        "BPM (e.g. '120 BPM', 'slow tempo around 70 BPM'); "
                        "Key/Scale (e.g. 'in G major', 'D minor'); "
                        "Mood (e.g. 'nostalgic', 'aggressive', 'ethereal', 'dreamy'); "
                        "Structure tags like [Verse], [Chorus], [Bridge], [Intro], [Outro] to control progression. "
                        "Example: 'Lo-fi hip hop, 85 BPM, in C minor, Fender Rhodes piano and vinyl crackle, "
                        "nostalgic and dreamy. [Intro] mellow chords [Verse] soft beat enters [Chorus] fuller sound.'"
                    ),
                ),
                ToolParameter(
                    name="lyrics",
                    type="string",
                    description=(
                        "Optional song lyrics to include in the generation. "
                        "Use \\n to separate lines. You may add structure tags like [Intro], [Verse], [Chorus], "
                        "[Bridge], [Outro] to guide the arrangement. "
                        "Leave empty for instrumental music."
                    ),
                ),
                ToolParameter(
                    name="model",
                    type="string",
                    description=(
                        "Which Lyria model to use: "
                        "'lyria-3-pro-preview' for full-length songs (default), "
                        "'lyria-3-clip-preview' for 30-second clips."
                    ),
                ),
                ToolParameter(
                    name="emotion",
                    type="object",
                    description=(
                        "Optional emotion dictionary: "
                        "{'light': {'color': [R,G,B], 'behavior': 'continuous/blinking/breathing', "
                        "'brightness': 'dark/medium/bright', 'period': seconds}, "
                        "'voice': {'tone': 'plain/happy'}}"
                    ),
                ),
            ],
            required=["prompt"],
            processor=self.generate_music_async,

            # RULE CONTRIBUTIONS
            rule_instructions={
                "russian": (
                    "Когда пользователь просит спеть, сыграть, создать, сгенерировать, сочинить или написать музыку "
                    "с помощью Gemini или Google, используй инструмент generate_music_gemini. "
                    "Триггеры: 'создай музыку gemini', 'сгенерируй песню через google', 'lyria'. "
                    "Всегда уточни у пользователя жанр, настроение и описание желаемой музыки перед генерацией. "
                    "После получения ответа от инструмента сохрани описание и локальный путь к файлу в память."
                ),
                "english": (
                    "When user asks to create, generate, or compose music using Gemini or Google, "
                    "use generate_music_gemini tool. "
                    "Triggers: 'create music with gemini', 'generate song via google', 'lyria'. "
                    "Always ask for genre, mood, and description before generation. "
                    "After receiving the tool response, save the description and local file path to memory."
                ),
            },
        )

    async def generate_music_async(self, parameters: dict) -> str:
        """
        Generate music using Google Gemini Lyria API and queue for playback.

        Process:
        1. POST prompt to Lyria generateContent endpoint
        2. Parse response parts, find inlineData audio
        3. Base64-decode the audio bytes
        4. Save as MP3 locally
        5. Decode MP3 → WAV once (avoids real-time decode on Pi Zero W)
        6. Add WAV to ResponsePlayer queue for interruptible playback
        7. Return success message with local path

        Args:
            parameters: Dict with 'prompt', optional 'model' and 'emotion' keys

        Returns:
            str: Success message with local MP3 path, or error description
        """
        if "prompt" not in parameters:
            return "Error: 'prompt' is required"

        prompt = parameters["prompt"]
        if len(prompt) < 10:
            return "Error: 'prompt' should be at least 10 characters"

        lyrics = parameters.get("lyrics", "")
        if lyrics:
            prompt = f"{prompt}\n\nLyrics:\n{lyrics}"

        model = parameters.get(
            "model",
            self.config.get("gemini_music_model", "lyria-3-pro-preview"),
        )
        emotion = parameters.get("emotion", None)

        if not self.api_key:
            return "Error: GEMINI_API_KEY environment variable is not set"

        url = f"{GEMINI_API_BASE}/{model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO", "TEXT"],
            },
        }
        timeout = self.config.get("gemini_music_timeout", 600)

        logger.info(f"Generating music via Gemini ({model}): prompt='{prompt}'")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    response.raise_for_status()
                    response_text = await response.text()

            if not response_text.strip():
                logger.error("Gemini returned an empty response body")
                return "Error: Music service returned an empty response"

            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                logger.error(f"Gemini returned non-JSON: {response_text[:500]}")
                return "Error: Music service returned an invalid response"

            # Check for API-level errors
            if "error" in result:
                error_msg = result["error"].get("message", str(result["error"]))
                logger.error(f"Gemini API error: {error_msg}")
                return f"Error generating music: {error_msg}"

            # Extract audio bytes from response parts
            # Per docs: do not assume audio is always the first part — iterate all parts
            audio_data: Optional[bytes] = None
            lyrics_text: str = ""

            candidates = result.get("candidates", [])
            if not candidates:
                logger.error(f"No candidates in response: {response_text[:500]}")
                return "Error: No audio candidates in API response"

            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "inlineData" in part:
                    raw_b64 = part["inlineData"].get("data", "")
                    audio_data = base64.b64decode(raw_b64)
                elif "text" in part:
                    lyrics_text = part["text"]

            if not audio_data:
                logger.error(f"No inlineData audio found in parts: {response_text[:500]}")
                return "Error: No audio data received from API"

            logger.info(f"Received {len(audio_data)} bytes of audio ({len(audio_data) / 1024:.1f} KB)")

            # Clean up old temp files
            self._cleanup_temp_files()

            # Save MP3 file
            timestamp = int(time.time())
            safe_prompt = "".join(c if c.isalnum() or c.isspace() else "_" for c in prompt[:30])
            safe_prompt = "_".join(safe_prompt.split())
            mp3_filename = f"gemini_music_{timestamp}_{safe_prompt}.mp3"
            mp3_path = self.music_dir / mp3_filename

            with open(mp3_path, "wb") as f:
                f.write(audio_data)

            logger.info(f"Saved MP3 file: {mp3_path}")
            self._cleanup_files.append(str(mp3_path))

            # Add ID3 metadata tags
            if HAS_MUTAGEN:
                try:
                    audio_tag = ID3(mp3_path)

                    title = prompt[:50] + ("..." if len(prompt) > 50 else "")
                    mp3_audio_seg = AudioSegment.from_mp3(mp3_path)
                    duration_ms = len(mp3_audio_seg)

                    audio_tag.add(TIT2(encoding=3, text=title))
                    audio_tag.add(TPE1(encoding=3, text=["Gemini Lyria"]))
                    audio_tag.add(TPE2(encoding=3, text=["Gemini Lyria"]))
                    audio_tag.add(TCOM(encoding=3, text=["Google Gemini"]))
                    audio_tag.add(TALB(encoding=3, text=["Generated Music"]))
                    audio_tag.add(TCON(encoding=3, text=["AI Generated"]))
                    audio_tag.add(TDRC(encoding=3, text=[time.strftime("%Y")]))
                    audio_tag.add(TLEN(encoding=3, text=[str(duration_ms)]))
                    audio_tag.add(TENC(encoding=3, text=["Cubic"]))
                    if lyrics_text:
                        audio_tag.add(USLT(encoding=3, lang="eng", desc="AI-generated lyrics", text=lyrics_text))
                    audio_tag.add(COMM(encoding=3, desc="Comments", text=[
                        f"Generated by Google Gemini Lyria\n"
                        f"Model: {model}\n"
                        f"Prompt: {prompt}\n"
                        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}"
                    ]))

                    audio_tag.save()
                    logger.info(f"Added metadata tags to {mp3_path}")
                except Exception as e:
                    logger.warning(f"Failed to add metadata tags: {e}")
            else:
                logger.info("Skipping metadata tags (mutagen not available)")

            # Refresh music library
            self._refresh_music_library()

            # Decode MP3 → WAV once (avoids real-time decode on Pi Zero W)
            logger.info("Decoding MP3 to WAV for playback")
            mp3_audio_seg = AudioSegment.from_mp3(mp3_path)
            wav_path = str(mp3_path).replace(".mp3", ".wav")
            mp3_audio_seg.export(wav_path, format="wav")

            # Queue for playback
            self.response_player.add((emotion, wav_path, "generated music"))
            logger.info(f"Added to playback queue: {wav_path}")

            result_msg = f"Music generated and playing. Local MP3: {mp3_path}"
            if lyrics_text:
                result_msg += f"\nGenerated lyrics/description:\n{lyrics_text}"
            return result_msg

        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error from Gemini API: {e.status} {e.message}")
            return f"Error: Gemini API returned HTTP {e.status}: {e.message}"
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            return f"Error: Failed to connect to Gemini music API: {str(e)}"
        except Exception as e:
            logger.error(f"Music generation failed: {e}", exc_info=True)
            return f"Error generating music: {str(e)}"

    def _cleanup_temp_files(self):
        """Keep only the most recent 50 temp files"""
        if len(self._cleanup_files) > 50:
            files_to_remove = self._cleanup_files[:-50]
            for filepath in files_to_remove:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {filepath}: {e}")
            self._cleanup_files = self._cleanup_files[-50:]

    def close(self):
        """Clean up resources"""
        pass


class MockResponsePlayer:
    """Mock ResponsePlayer for testing"""
    def __init__(self):
        self.queue = []

    def add(self, item):
        self.queue.append(item)
        logger.info(f"Added to playback queue: {item[2] if len(item) > 2 else 'audio'}")


async def main():
    """Test function to run GeminiMusicTool directly"""
    logger.info("Starting GeminiMusicTool test...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY=your_api_key")
        return

    class MockConfig:
        def get(self, key, default=None):
            return default

    config = MockConfig()
    response_player = MockResponsePlayer()
    tool = GeminiMusicTool(config, response_player)

    try:
        test_prompt = "Upbeat electronic music with energetic beats and futuristic synthesizer sounds"

        logger.info("Generating test music...")
        result = await tool.generate_music_async({"prompt": test_prompt})

        print(f"\n{'=' * 60}")
        print(f"Result: {result}")
        print(f"{'=' * 60}\n")

        logger.info(f"Playback queue size: {len(response_player.queue)}")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nError: {e}\n")
    finally:
        tool.close()


if __name__ == "__main__":
    import asyncio  # noqa: E402

    from dotenv import load_dotenv  # noqa: E402

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(main())
