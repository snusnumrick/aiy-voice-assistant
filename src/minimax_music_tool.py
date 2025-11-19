import os
import json
import io
import time
import aiohttp
from pathlib import Path
from typing import Dict, Optional
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
        self.base_url = config.get("minimax_base_url", "https://api.minimax.chat")
        self.response_player = response_player
        self.button_state = button_state
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
                    description="Music style description and mood"
                ),
                ToolParameter(
                    name="lyrics",
                    type="string",
                    description="Song lyrics"
                ),
            ],
            required=["prompt", "lyrics"],
            processor=self.generate_music_async,

            # RULE CONTRIBUTIONS
            rule_instructions={
                "russian": (
                    "Когда пользователь просит создать, сгенерировать, сочинить или написать музыку, "
                    "используй инструмент generate_music. "
                    "Триггеры: 'создай музыку', 'сгенерируй песню', 'сочини мелодию', "
                    "'напиши композицию', 'сделай трек'. "
                    "Всегда уточни у пользователя жанр, настроение, описание или характеристики "
                    "желаемой музыки перед генерацией."
                ),
                "english": (
                    "When user asks to create, generate, compose, or write music, "
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
            return "Error: Both 'prompt' and 'lyrics' are required"

        prompt = parameters["prompt"]
        lyrics = parameters["lyrics"]

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
                    mp3_audio_data = bytearray()  # Store complete MP3 data

                    # Parse SSE (Server-Sent Events) response
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

                                    # Check for hex audio data
                                    if 'audio' in data:
                                        hex_data = data['audio']
                                        mp3_bytes = bytes.fromhex(hex_data)
                                        mp3_audio_data.extend(mp3_bytes)

                                        # Convert MP3 bytes to WAV
                                        mp3_audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))

                                        # Export to WAV format in-memory
                                        wav_io = io.BytesIO()
                                        mp3_audio.export(wav_io, format="wav")
                                        wav_bytes = wav_io.getvalue()

                                        # Save to temporary file
                                        timestamp = int(time.time() * 1000)  # millisecond timestamp
                                        wav_file = f"/tmp/music_{timestamp}_{chunk_count}.wav"

                                        with open(wav_file, "wb") as f:
                                            f.write(wav_bytes)

                                        self._cleanup_files.append(wav_file)

                                        # Add to ResponsePlayer queue for interruptible playback
                                        self.response_player.add((None, wav_file, "generated music"))

                                        chunk_count += 1

                                        # Check for button press interrupt
                                        if self.button_state():
                                            logger.info("Button pressed during music generation, stopping")
                                            break

                                    # Check for completion
                                    if data.get('is_finish', False):
                                        logger.info(f"Music generation complete: {chunk_count} chunks")
                                        break

                                    # Check for errors
                                    if 'error' in data:
                                        error_msg = data['error']
                                        logger.error(f"API error: {error_msg}")
                                        return f"Error generating music: {error_msg}"

                                except (json.JSONDecodeError, KeyError, ValueError) as e:
                                    logger.warning(f"Failed to parse SSE data: {e}")
                                    continue

                    # Cleanup old temp files (keep last 50)
                    self._cleanup_temp_files()

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
                        return f"Music generated and playing. MP3 file saved at: {mp3_path}"
                    else:
                        return "Error: No audio data received from API"

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            return f"Error: Failed to connect to music generation API: {str(e)}"
        except Exception as e:
            logger.error(f"Music generation failed: {e}")
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
