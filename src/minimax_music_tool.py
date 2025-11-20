import os
import time
import aiohttp
from pathlib import Path
from typing import Dict
from pydub import AudioSegment
from src.config import Config
from src.ai_models_with_tools import Tool, ToolParameter
import logging

logger = logging.getLogger(__name__)

class MiniMaxMusicTool:
    def __init__(self, config: Config, response_player):
        """
        Initialize MiniMax music tool

        Args:
            config: Configuration object
            response_player: ResponsePlayer instance for audio playback
        """
        self.config = config
        self.api_key = os.environ.get("MINIMAX_API_KEY")
        self.base_url = config.get("minimax_base_url", "https://api.minimax.io")
        self.response_player = response_player
        self._cleanup_files = []  # Track temp files for cleanup
        self.music_dir = Path("/tmp/generated_music")  # Directory for saved MP3 files
        self.music_dir.mkdir(parents=True, exist_ok=True)

    def tool_definition(self) -> Tool:
        """Return tool definition for AI model"""
        return Tool(
            name="generate_music",
            description="Generate music using MiniMax API",
            iterative=True,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description="Music style description and mood, 10 to 2000 characters"
                ),
                ToolParameter(
                    name="lyrics",
                    type="string",
                    description="Song lyrics, 10 to 3000 characters"
                ),
            ],
            required=["prompt", "lyrics"],
            processor=self.generate_music_async,

            # RULE CONTRIBUTIONS
            rule_instructions={
                "russian": (
                    "Когда пользователь просит спеть, сыграть, создать, сгенерировать, сочинить или написать музыку, "
                    "используй инструмент generate_music. "
                    "Триггеры: 'создай музыку', 'сгенерируй песню', 'сочини мелодию', "
                    "'напиши композицию', 'сделай трек'. "
                    "Всегда уточни у пользователя жанр, настроение, описание или характеристики "
                    "желаемой музыки перед генерацией."
                    "Если не попросили иначе и ты поешь саи, добавь в prompt 'Мужской голос среднего возраста, глубокий и теплый тембр'."
                ),
                "english": (
                    "When user asks to sing, play, create, generate, compose, or write music, "
                    "use generate_music tool. "
                    "Triggers: 'create music', 'generate song', 'compose melody', "
                    "'write composition', 'make a track'. "
                    "Always ask for genre, mood, description, or characteristics of desired music "
                    "before generation."
                )
            }
        )

    async def generate_music_async(self, parameters: Dict) -> str:
        """
        Generate music using MiniMax API and play immediately

        Process (Pi Zero W optimized - non-streaming):
        1. Validate parameters
        2. Make a non-streaming API request (stream: false, output_format: url)
        3. Receive audio URL in a single response
        4. Download MP3 from URL
        5. Save complete MP3 file locally
        6. Decode MP3 → WAV once (single operation, no real-time CPU load)
        7. Add WAV to the ResponsePlayer queue (interruptible playback)
        8. Return the success message with download URL (for email) + local path

        Args:
            parameters: Dict with 'prompt' and 'lyrics' keys

        Returns:
            str: Success message with download URL (valid 24h) + local MP3 path
        """
        # Validate parameters
        if "prompt" not in parameters or "lyrics" not in parameters:
            logger.error("Missing 'prompt' or 'lyrics' parameter")
            return "Error: Both 'prompt' and 'lyrics' are required"

        prompt = parameters["prompt"]
        lyrics = parameters["lyrics"]

        if len(prompt) < 10:
            return "Error: 'prompt' should be at least 10 characters"
        if len(lyrics) < 10:
            return "Error: 'lyrics' should be at least 10 characters"



        logger.info(f"Generating music: prompt='{prompt}', lyrics='{lyrics[:50]}...'")

        try:
            # Make a streaming API request to MiniMax
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/music_generation",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "music-2.0",
                        "prompt": prompt,
                        "lyrics": lyrics,
                        "stream": False,  # Pi Zero W: single response, no real-time processing
                        "output_format": "url",  # Get URL for email inclusion + download for playback
                        "audio_setting": {
                            "sample_rate": 16000,  # Minimum valid sample rate for Pi Zero W
                            "bitrate": 32000,      # Minimum valid bitrate (3x smaller files, faster processing)
                            "format": "mp3"
                        }
                    },
                    timeout=300  # 5 minute timeout
                ) as response:
                    response.raise_for_status()

                    # Pi Zero W: Get audio URL in one response (non-streaming)
                    logger.info("Receiving audio URL...")
                    result = await response.json()

                    # Check for API errors
                    if 'error' in result:
                        error_msg = result['error']
                        logger.error(f"API error: {error_msg}")
                        return f"Error generating music: {error_msg}"

                    # Extract URL
                    if 'data' not in result or 'audio' not in result['data']:
                        logger.error("No audio_url in response")
                        return "Error: No audio URL received from API"

                    audio_url = result['data']['audio']
                    logger.info(f"Received audio URL: {audio_url}")

                    # Download MP3 from URL
                    logger.info("Downloading MP3 from URL...")
                    async with session.get(audio_url) as download_response:
                        download_response.raise_for_status()
                        mp3_audio_data = await download_response.read()
                        logger.info(f"Downloaded {len(mp3_audio_data)} bytes ({len(mp3_audio_data) / 1024:.1f} KB)")

                    # Store URL for inclusion in response message
                    audio_url_for_email = audio_url

                    # Clean up old temp files (keep last 50)
                    self._cleanup_temp_files()

                    if not mp3_audio_data:
                        return "Error: No audio data received from API"

                    # Save a complete MP3 file
                    timestamp = int(time.time())
                    # Create a safe filename from lyrics
                    safe_lyrics = "".join(c if c.isalnum() or c.isspace() else "_" for c in lyrics[:30])
                    safe_lyrics = "_".join(safe_lyrics.split())  # Replace spaces with underscores
                    mp3_filename = f"music_{timestamp}_{safe_lyrics}.mp3"
                    mp3_path = self.music_dir / mp3_filename

                    with open(mp3_path, "wb") as f:
                        f.write(mp3_audio_data)

                    logger.info(f"Saved MP3 file: {mp3_path}")

                    # Decode complete MP3 to WAV once (Pi Zero W: avoid real-time decode)
                    logger.info("Decoding complete MP3 to WAV (single decode operation)")
                    mp3_audio = AudioSegment.from_mp3(mp3_path)
                    wav_file = str(mp3_path).replace(".mp3", ".wav")
                    mp3_audio.export(wav_file, format="wav")

                    # Add to ResponsePlayer queue for playback
                    self.response_player.add((None, wav_file, "generated music"))
                    logger.info(f"Added to playback queue: {wav_file}")

                    # Return URL for email inclusion + local path for debugging
                    return f"Music generated and playing. Download URL (valid 24h): {audio_url_for_email}\nLocal MP3: {mp3_path}"

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            return f"Error: Failed to connect to music generation API: {str(e)}"
        except Exception as e:
            logger.error(f"Music generation failed: {e}", exc_info=True)
            return f"Error generating music: {str(e)}"

    def _cleanup_temp_files(self):
        """Keep only the most recent temp files"""
        if len(self._cleanup_files) > 50:
            # Remove oldest files
            files_to_remove = self._cleanup_files[:-50]
            for filepath in files_to_remove:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {filepath}: {e}")
            self._cleanup_files = self._cleanup_files[-50:]

    def close(self):
        """Clean up resources - call when done using the tool"""
        pass


class MockResponsePlayer:
    """Mock ResponsePlayer for testing"""
    def __init__(self):
        self.queue = []

    def add(self, item):
        """Add item to playback queue"""
        self.queue.append(item)
        logger.info(f"Added to playback queue: {item[2] if len(item) > 2 else 'audio'}")


class MockButtonState:
    """Mock ButtonState for testing"""
    def __init__(self):
        self.pressed = False

    def __call__(self):
        """Return current button state"""
        return self.pressed


async def main():
    """Test function to run MiniMaxMusicTool directly"""
    logger.info("Starting MiniMaxMusicTool test...")

    # Check for API key
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("Error: MINIMAX_API_KEY environment variable not set")
        print("Please set it with: export MINIMAX_API_KEY=your_api_key")
        return

    # Create mock dependencies
    class MockConfig:
        def get(self, key, default=None):
            if key == "minimax_base_url":
                return "https://api.minimax.io"
            return default

    config = MockConfig()
    response_player = MockResponsePlayer()
    button_state = MockButtonState()

    # Create tool instance
    tool = MiniMaxMusicTool(config, response_player, button_state)

    try:
        # Test parameters
        test_prompt = "Upbeat electronic music with energetic beats and futuristic sounds"
        test_lyrics = "Dancing through the night, feeling so alive, electronic dreams are calling"

        logger.info("Generating test music...")
        result = await tool.generate_music_async({
            "prompt": test_prompt,
            "lyrics": test_lyrics
        })

        print(f"\n{'='*60}")
        print(f"Result: {result}")
        print(f"{'='*60}\n")

        logger.info(f"Playback queue size: {len(response_player.queue)}")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nError: {e}\n")
    finally:
        tool.close()


if __name__ == "__main__":
    import asyncio

    # Configure logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the async main function
    asyncio.run(main())

