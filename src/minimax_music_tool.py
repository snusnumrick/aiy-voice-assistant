import os
import json
import io
import time
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict
from pydub import AudioSegment
from src.config import Config
from src.ai_models_with_tools import Tool, ToolParameter
import logging

logger = logging.getLogger(__name__)


class MiniMaxMusicTool:
    def __init__(self, config: Config, response_player, button_state):
        """
        Initialize MiniMax music tool

        Args:
            config: Configuration object
            response_player: ResponsePlayer instance for audio playback
            button_state: ButtonState instance for shared button press state
        """
        self.config = config
        self.api_key = os.environ.get("MINIMAX_API_KEY")
        self.base_url = config.get("minimax_base_url", "https://api.minimax.io")
        self.response_player = response_player
        self.button_state = button_state
        self._cleanup_files = []  # Track temp files for cleanup
        self.music_dir = Path("/tmp/generated_music")  # Directory for saved MP3 files
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=2)  # For parallel MP3 decoding
        self.chunks_per_decode = 10  # Buffer 10 chunks before decoding (ensures valid MP3 frames)
        self.max_buffers = 4  # Increased to 4 for smoother parallel processing, prevents decode bottlenecks

    def _decode_buffered_mp3_to_wav(self, buffered_mp3: bytes, buffer_idx: int) -> str:
        """
        Decode buffered MP3 data to WAV file (for thread pool)

        Args:
            buffered_mp3: Buffered MP3 audio data containing multiple chunks
            buffer_idx: Buffer index for filename

        Returns:
            str: Path to the generated WAV file
        """
        # Convert buffered MP3 bytes to WAV
        mp3_audio = AudioSegment.from_mp3(io.BytesIO(buffered_mp3))

        # Export to WAV format in-memory
        wav_io = io.BytesIO()
        mp3_audio.export(wav_io, format="wav")
        wav_bytes = wav_io.getvalue()

        # Save to temporary file
        timestamp = int(time.time() * 1000)  # millisecond timestamp
        wav_file = f"/tmp/music_buffer_{timestamp}_{buffer_idx}.wav"

        with open(wav_file, "wb") as f:
            f.write(wav_bytes)

        return wav_file


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

        Process:
        1. Validate parameters
        2. Make streaming API request
        3. Parse SSE response for hex audio data
        4. Convert MP3 hex → MP3 bytes → WAV bytes → WAV file
        5. Save complete MP3 file
        6. Add WAV to ResponsePlayer queue (interruptible playback)
        7. Return success message with MP3 file path

        Args:
            parameters: Dict with 'prompt' and 'lyrics' keys

        Returns:
            str: Success/error message with MP3 file path
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
            # Make streaming API request to MiniMax
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
                        "stream": True,
                        "audio_setting": {
                            "sample_rate": 16000,
                            "bitrate": 32000,
                            "format": "mp3"
                        }
                    },
                    timeout=300  # 5 minute timeout
                ) as response:
                    response.raise_for_status()

                    chunk_count = 0
                    buffer_count = 0
                    mp3_audio_data = bytearray()  # Store complete MP3 data
                    current_buffer = bytearray()  # Buffer for current decoding batch
                    processing_tasks = []  # Track concurrent decoding tasks

                    # Parse SSE (Server-Sent Events) response with buffered parallel processing
                    try:
                        async for line in response.content:
                            if line:
                                line_str = line.decode('utf-8')

                                # Skip SSE comments
                                if line_str.startswith(':'):
                                    continue

                                # Parse SSE data format: "data: {json}"
                                if line_str.startswith('data: '):
                                    try:
                                        data = json.loads(line_str[6:])['data']

                                        # Log full data structure for debugging
                                        logger.info(f"SSE data keys: {list(data.keys())}")
                                        if 'status' in data:
                                            logger.info(f"SSE status: {data['status']}")

                                        # Check if this is an error message from the API
                                        if 'msg' in data or 'message' in data:
                                            api_msg = data.get('msg') or data.get('message')
                                            logger.error(f"API message: {api_msg}")
                                            logger.error(f"Full API data: {data}")

                                        # Check for hex audio data
                                        if 'audio' in data:
                                            # Skip empty chunks (end markers or heartbeats)
                                            if data['audio'] == '' or len(data['audio']) == 0:
                                                logger.info(f"Skipping empty chunk {chunk_count}")
                                                chunk_count += 1
                                                continue
                                        try:
                                            hex_data = data['audio']
                                            logger.info(f"Received chunk {chunk_count}, hex data length: {len(hex_data)}")

                                            # Check if hex data is too large
                                            if len(hex_data) > 1000000:  # 1MB of hex = ~500KB binary
                                                logger.error(f"Hex data too large: {len(hex_data)} characters")

                                            mp3_bytes = bytes.fromhex(hex_data)
                                            logger.info(f"Decoded to {len(mp3_bytes)} bytes")

                                            mp3_audio_data.extend(mp3_bytes)
                                            current_buffer.extend(mp3_bytes)

                                            chunk_count += 1

                                            # Decode when buffer reaches threshold (ensures valid MP3 frames and reduces final buffer size)
                                            if len(current_buffer) >= 4000:  # Small 4KB buffers for smooth playback, eliminates pauses
                                                # Submit buffer for decoding in thread pool
                                                loop = asyncio.get_event_loop()
                                                buffer_copy = bytes(current_buffer)  # Copy for thread safety
                                                task = loop.run_in_executor(
                                                    self.executor,
                                                    self._decode_buffered_mp3_to_wav,
                                                    buffer_copy,
                                                    buffer_count
                                                )

                                                # Store task with buffer index for tracking
                                                processing_tasks.append((buffer_count, task))

                                                # Clear buffer for next batch
                                                current_buffer = bytearray()
                                                buffer_count += 1

                                                # Limit concurrent processing to prevent memory issues
                                                # Wait for oldest buffer to decode if max buffers reached
                                                if len(processing_tasks) >= self.max_buffers:
                                                    oldest_idx, oldest_task = processing_tasks.pop(0)
                                                    wav_file = await oldest_task
                                                    self._cleanup_files.append(wav_file)
                                                    # Add to ResponsePlayer queue for interruptible playback
                                                    self.response_player.add((None, wav_file, "generated music"))
                                                    logger.info(f"Processed buffer {oldest_idx}")

                                        except Exception as e:
                                            logger.error(f"Error processing audio chunk {chunk_count}: {e}", exc_info=True)
                                            raise

                                        # Check for button press interrupt
                                        if self.button_state():
                                            logger.info("Button pressed during music generation, stopping")
                                            break

                                        # Check for completion
                                        if data.get('is_finish', False):
                                                logger.info(f"Music generation complete: {chunk_count} chunks, {buffer_count} buffers")
                                                break

                                        # Check for errors
                                        if 'error' in data:
                                            error_msg = data['error']
                                            logger.error(f"API error received: {error_msg}")
                                            logger.error(f"Full error data: {data}")
                                            return f"Error generating music: {error_msg}"

                                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                                        logger.warning(f"Failed to parse SSE data: {e}")
                                        continue
                    except ValueError as e:
                        # Handle aiohttp "Chunk too big" error from empty/malformed chunks
                        if "Chunk too big" in str(e):
                            logger.info(f"Reached end of stream (aiohttp chunk limit): {e}")
                        else:
                            logger.error(f"ValueError during streaming: {e}", exc_info=True)
                            raise

                    # Cleanup old temp files (keep last 50)
                    self._cleanup_temp_files()

                    # Decode any remaining buffered data
                    if current_buffer:
                        logger.info(f"Decoding final buffer of {len(current_buffer)} bytes")
                        loop = asyncio.get_event_loop()
                        buffer_copy = bytes(current_buffer)
                        task = loop.run_in_executor(
                            self.executor,
                            self._decode_buffered_mp3_to_wav,
                            buffer_copy,
                            buffer_count
                        )
                        processing_tasks.append((buffer_count, task))
                        buffer_count += 1

                    # Wait for all remaining buffers to finish decoding
                    logger.info(f"Waiting for {len(processing_tasks)} remaining buffers to decode")
                    for idx, task in processing_tasks:
                        wav_file = await task
                        self._cleanup_files.append(wav_file)
                        # Add to ResponsePlayer queue for interruptible playback
                        self.response_player.add((None, wav_file, "generated music"))
                        logger.info(f"Processed final buffer {idx}")

                    if chunk_count > 0 and mp3_audio_data:
                        # Save complete MP3 file
                        timestamp = int(time.time())
                        # Create safe filename from lyrics
                        safe_lyrics = "".join(c if c.isalnum() or c.isspace() else "_" for c in lyrics[:30])
                        safe_lyrics = "_".join(safe_lyrics.split())  # Replace spaces with underscores
                        mp3_filename = f"music_{timestamp}_{safe_lyrics}.mp3"
                        mp3_path = self.music_dir / mp3_filename

                        with open(mp3_path, "wb") as f:
                            f.write(mp3_audio_data)

                        logger.info(f"Saved MP3 file: {mp3_path}")
                        return f"Music generated and playing; saved as a file at {mp3_path}"
                    else:
                        return "Error: No audio data received from API"

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
        self.executor.shutdown(wait=True)
        logger.info("MiniMaxMusicTool executor shut down")


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
    # Configure logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the async main function
    asyncio.run(main())

