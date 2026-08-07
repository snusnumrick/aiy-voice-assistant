import base64
import json
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import aiohttp
import requests

if __name__ == "__main__":
    import sys

    sys.path.append(os.getcwd())

from src.config import Config
from src.server_utils import get_server_url

if TYPE_CHECKING:
    from src.ai_models_with_tools import Tool

logger = logging.getLogger(__name__)

try:
    from pydub import AudioSegment

    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    AudioSegment = None
    logger.info("pydub not installed, audio conversion disabled")

try:
    from mutagen.id3 import COMM, ID3, TALB, TCOM, TCON, TDRC, TENC, TIT2, TLEN, TPE1, TPE2, USLT

    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False
    logger.info("mutagen not installed, MP3 metadata tagging disabled")


class MusicTool(ABC):
    def __init__(self, config: Config, response_player):
        self.config = config
        self.response_player = response_player
        self._cleanup_files = []

        self.music_dir = self._discover_music_folder()
        if self.music_dir:
            logger.info("Using music folder from cubie-server: %s", self.music_dir)
        else:
            self.music_dir = Path("/tmp/generated_music")
            logger.warning("cubie-server not found, using fallback: %s", self.music_dir)
            self.music_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def provider_display_name(self) -> str:
        """Human-readable provider name for logs and tool descriptions."""

    @property
    @abstractmethod
    def required_api_key_env_var(self) -> str:
        """Environment variable name required by this provider."""

    @property
    @abstractmethod
    def filename_prefix(self) -> str:
        """Filename prefix used for generated MP3 files."""

    @property
    @abstractmethod
    def metadata_artist_name(self) -> str:
        """Artist string used for generated MP3 metadata."""

    @property
    def metadata_composer_name(self) -> str:
        return self.metadata_artist_name

    @property
    def metadata_genre(self) -> str:
        return "AI Generated"

    def _discover_music_folder(self) -> Optional[Path]:
        """Discover cubie-server and use its shared music folder when available."""
        server_url = get_server_url()

        try:
            logger.info("Discovering cubie-server at %s", server_url)
            response = requests.get(f"{server_url}/api/config/folders", timeout=5)

            if response.status_code != 200:
                logger.warning("cubie-server returned status %s", response.status_code)
                return None

            config = response.json()
            music_folder = config.get("music")
            if music_folder and os.path.exists(music_folder):
                logger.info("Server discovered. Music folder: %s", music_folder)
                return Path(music_folder)

            logger.warning(
                "cubie-server returned config but music folder is unavailable: %s",
                music_folder,
            )
        except requests.exceptions.RequestException:
            pass
        except Exception as exc:
            logger.warning("Error discovering music folder: %s", exc)

        return None

    def _refresh_music_library(self) -> bool:
        """Tell cubie-server to refresh its music library."""
        server_url = get_server_url()

        try:
            response = requests.post(f"{server_url}/api/refresh", timeout=5)
            if response.status_code == 200:
                result = response.json()
                logger.info("Music library refreshed (%s tracks)", result.get("count", "N/A"))
                return True

            logger.warning("Refresh returned status %s", response.status_code)
        except Exception as exc:
            logger.warning("Could not refresh library: %s", exc)

        return False

    def tool_definition(self) -> "Tool":
        from src.ai_models_with_tools import Tool, ToolParameter

        return Tool(
            name="generate_music",
            description=(
                f"Generate songs, melodies, instrumentals, and music using "
                f"{self.provider_display_name}."
            ),
            iterative=True,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description=(
                        "A detailed description of the music to generate. Prefer: genre or style, "
                        "tempo or BPM, mood, key or scale, instruments, structure, and production "
                        "details. Example: 'Lo-fi hip hop, 85 BPM, in C minor, Fender Rhodes piano "
                        "and vinyl crackle, nostalgic and dreamy. [Intro] mellow chords [Verse] "
                        "soft beat enters [Chorus] fuller sound.'"
                    ),
                ),
                ToolParameter(
                    name="lyrics",
                    type="string",
                    description=(
                        "Optional song lyrics. Use \\n for line breaks. You may add structure tags "
                        "like [Intro], [Verse], [Chorus], [Bridge], and [Outro]. Leave empty for "
                        "instrumental music."
                    ),
                ),
                ToolParameter(
                    name="model",
                    type="string",
                    description="Optional provider-specific model override.",
                ),
                ToolParameter(
                    name="emotion",
                    type="object",
                    description=(
                        "Optional emotion dictionary with format: {'light': {'color': [R,G,B], "
                        "'behavior': 'continuous/blinking/breathing', 'brightness': "
                        "'dark/medium/bright', 'period': seconds}, 'voice': {'tone': 'plain/happy'}}"
                    ),
                ),
            ],
            required=["prompt"],
            processor=self.generate_music_async,
            rule_instructions={
                "russian": (
                    "Когда пользователь просит спеть, сыграть, создать, сгенерировать, сочинить "
                    "или написать музыку, используй инструмент generate_music. Триггеры: "
                    "'создай музыку', 'сгенерируй песню', 'сочини мелодию', 'сделай трек', "
                    "'создай музыку gemini', 'lyria'. Всегда уточни жанр, настроение, описание "
                    "или характеристики желаемой музыки перед генерацией. Если не попросили иначе "
                    "и нужна песня с текстом, добавь в prompt 'Мужской голос среднего возраста, "
                    "глубокий и теплый тембр'. После получения ответа обязательно сохрани текст "
                    "песни и локальный путь к файлу в память, используя формат: $remember: "
                    "Сгенерированная музыка: <описание из prompt>, текст: <lyrics>, файл: <путь>. "
                    "Удали из памяти через 24 часа.$."
                ),
                "english": (
                    "When user asks to sing, play, create, generate, compose, or write music, use "
                    "generate_music tool. Triggers: 'create music', 'generate song', 'compose "
                    "melody', 'make a track', 'create music with gemini', 'lyria'. Always ask for "
                    "genre, mood, description, or characteristics of the desired music before "
                    "generation. After receiving the tool response, save the lyrics and local file "
                    "path to memory using format: $remember: Generated music: <description from "
                    "prompt>, lyrics: <lyrics>, file: <path> Remove from memory in 24 hours$."
                ),
            },
        )

    def library_tool_definition(self) -> "Tool":
        from src.ai_models_with_tools import Tool, ToolParameter

        return Tool(
            name="play_music",
            description=(
                "Browse and play songs already saved in the local music library. Use this instead "
                "of generating new music when the user asks for an old, previous, saved, or "
                "existing song."
            ),
            iterative=True,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "Optional words from the title, description, lyrics, or filename. Leave "
                        "empty for requests like 'play the old song' to play the newest saved song."
                    ),
                ),
                ToolParameter(
                    name="action",
                    type="string",
                    description="Use 'play' to play a match or 'list' to browse matching songs.",
                ),
                ToolParameter(
                    name="emotion",
                    type="object",
                    description="Optional light/voice emotion dictionary for playback.",
                ),
            ],
            required=[],
            processor=self.play_music_async,
            rule_instructions={
                "russian": (
                    "Когда пользователь просит найти, перечислить или включить ранее созданную, "
                    "старую либо сохраненную песню, используй play_music, а не generate_music. "
                    "Для просьбы вроде 'включи старую' оставь query пустым."
                ),
                "english": (
                    "When the user asks to find, list, or play an old, previous, or saved song, "
                    "use play_music rather than generate_music. Leave query empty for requests "
                    "like 'play the old one'."
                ),
            },
        )

    def tool_definitions(self) -> list["Tool"]:
        return [self.tool_definition(), self.library_tool_definition()]

    @staticmethod
    def _music_files(music_dir: Path) -> list[Path]:
        extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
        try:
            files = [
                path
                for path in music_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in extensions
            ]
        except OSError as exc:
            logger.warning("Could not browse music folder %s: %s", music_dir, exc)
            return []
        return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)

    @staticmethod
    def _searchable_music_text(path: Path) -> str:
        parts = [path.stem.replace("_", " ")]
        if HAS_MUTAGEN and path.suffix.lower() == ".mp3":
            try:
                tags = ID3(path)
                parts.extend(str(frame) for frame in tags.values())
            except Exception:
                pass
        return " ".join(parts).casefold()

    def _find_music(self, query: str) -> list[Path]:
        files = self._music_files(self.music_dir)
        words = [word for word in query.casefold().split() if word]
        if not words:
            return files

        scored = []
        for path in files:
            text = self._searchable_music_text(path)
            score = sum(1 for word in words if word in text)
            if score:
                scored.append((score, path.stat().st_mtime, path))
        scored.sort(reverse=True)
        return [path for _score, _mtime, path in scored]

    def _queue_existing_audio(self, path: Path, emotion) -> None:
        playback_path = path
        if path.suffix.lower() != ".wav":
            if not HAS_PYDUB:
                raise RuntimeError("pydub is not installed")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                playback_path = Path(temp_file.name)
            AudioSegment.from_file(path).export(playback_path, format="wav")
            self._cleanup_files.append(str(playback_path))

        self.response_player.add((emotion, str(playback_path), f"saved music: {path.stem}"))
        logger.info("Added saved music to playback queue: %s", path)

    async def play_music_async(self, parameters: dict) -> str:
        """Browse or play music already present in the shared library."""
        query = str(parameters.get("query", "") or "").strip()
        action = str(parameters.get("action", "play") or "play").strip().lower()
        matches = self._find_music(query)
        if not matches:
            return f"No saved music found matching: {query}" if query else "No saved music found"

        if action == "list":
            names = ", ".join(path.stem for path in matches[:10])
            return f"Saved music ({min(len(matches), 10)} shown): {names}"
        if action != "play":
            return "Error: action must be 'play' or 'list'"

        selected = matches[0]
        try:
            self._queue_existing_audio(selected, parameters.get("emotion"))
        except Exception as exc:
            logger.error("Could not play saved music %s: %s", selected, exc, exc_info=True)
            return f"Error playing saved music: {exc}"
        return f"Playing saved music: {selected.stem}"

    @staticmethod
    def _prompt_from_parameters(parameters: dict) -> str:
        prompt = parameters.get("prompt", "")
        return str(prompt).strip()

    @staticmethod
    def _safe_prompt_fragment(prompt: str) -> str:
        safe_prompt = "".join(c if c.isalnum() or c.isspace() else "_" for c in prompt[:30])
        return "_".join(safe_prompt.split()) or "generated_music"

    def _save_generated_audio(self, audio_data: bytes, prompt: str) -> Path:
        timestamp = int(time.time())
        safe_prompt = self._safe_prompt_fragment(prompt)
        mp3_path = self.music_dir / f"{self.filename_prefix}_{timestamp}_{safe_prompt}.mp3"

        with open(mp3_path, "wb") as handle:
            handle.write(audio_data)

        self._cleanup_files.append(str(mp3_path))
        logger.info("Saved MP3 file: %s", mp3_path)
        return mp3_path

    def _add_mp3_metadata(
        self,
        mp3_path: Path,
        prompt: str,
        lyrics: str = "",
        comment_lines: Optional[list[str]] = None,
    ) -> None:
        if not HAS_MUTAGEN:
            logger.info("Skipping metadata tags (mutagen not available)")
            return

        try:
            title = prompt[:50] if len(prompt) <= 50 else f"{prompt[:50]}..."
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            comments = list(comment_lines or [])
            comments.extend([f"Prompt: {prompt}", f"Timestamp: {timestamp}"])

            audio_tag = ID3()
            audio_tag.add(TIT2(encoding=3, text=title))
            audio_tag.add(TPE1(encoding=3, text=[self.metadata_artist_name]))
            audio_tag.add(TPE2(encoding=3, text=[self.metadata_artist_name]))
            audio_tag.add(TCOM(encoding=3, text=[self.metadata_composer_name]))
            audio_tag.add(TALB(encoding=3, text=["Generated Music"]))
            audio_tag.add(TCON(encoding=3, text=[self.metadata_genre]))
            audio_tag.add(TDRC(encoding=3, text=[time.strftime("%Y")]))
            if HAS_PYDUB:
                duration_ms = len(AudioSegment.from_mp3(mp3_path))
                audio_tag.add(TLEN(encoding=3, text=[str(duration_ms)]))
            audio_tag.add(TENC(encoding=3, text=["Cubic"]))
            if lyrics:
                audio_tag.add(USLT(encoding=3, lang="eng", desc="AI-generated lyrics", text=lyrics))
            audio_tag.add(COMM(encoding=3, desc="Comments", text=["\n".join(comments)]))
            audio_tag.save(mp3_path)
            logger.info("Added metadata tags to %s", mp3_path)
        except Exception as exc:
            logger.warning("Failed to add metadata tags: %s", exc)

    def _queue_generated_audio(self, mp3_path: Path, emotion) -> None:
        if not HAS_PYDUB:
            raise RuntimeError("pydub is not installed")

        logger.info("Decoding MP3 to WAV for playback")
        mp3_audio = AudioSegment.from_mp3(mp3_path)
        wav_path = mp3_path.with_suffix(".wav")
        mp3_audio.export(wav_path, format="wav")

        self._cleanup_files.append(str(wav_path))
        self.response_player.add((emotion, str(wav_path), "generated music"))
        logger.info("Added to playback queue: %s", wav_path)

    def _process_generated_audio(
        self,
        audio_data: bytes,
        prompt: str,
        lyrics: str = "",
        emotion=None,
        comment_lines: Optional[list[str]] = None,
    ) -> Path:
        self._cleanup_temp_files()
        mp3_path = self._save_generated_audio(audio_data, prompt)
        self._add_mp3_metadata(mp3_path, prompt=prompt, lyrics=lyrics, comment_lines=comment_lines)
        self._refresh_music_library()
        self._queue_generated_audio(mp3_path, emotion)
        return mp3_path

    def _cleanup_temp_files(self) -> None:
        """Keep only the most recent 50 temp files."""
        if len(self._cleanup_files) <= 50:
            return

        files_to_remove = self._cleanup_files[:-50]
        for filepath in files_to_remove:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as exc:
                logger.warning("Failed to remove temp file %s: %s", filepath, exc)

        self._cleanup_files = self._cleanup_files[-50:]

    def close(self) -> None:
        """Clean up resources."""

    @abstractmethod
    async def generate_music_async(self, parameters: dict) -> str:
        """Generate music and return a user-facing status string."""


class MiniMaxMusicTool(MusicTool):
    def __init__(self, config: Config, response_player):
        super().__init__(config, response_player)
        self.api_key = os.environ.get("MINIMAX_API_KEY")
        self.base_url = config.get("minimax_base_url", "https://api.minimax.io")

    @property
    def provider_display_name(self) -> str:
        return "MiniMax"

    @property
    def required_api_key_env_var(self) -> str:
        return "MINIMAX_API_KEY"

    @property
    def filename_prefix(self) -> str:
        return "music"

    @property
    def metadata_artist_name(self) -> str:
        return "MiniMax AI"

    @property
    def metadata_genre(self) -> str:
        return "Electronic"

    async def generate_music_async(self, parameters: dict) -> str:
        prompt = self._prompt_from_parameters(parameters)
        if not prompt:
            logger.error("Missing 'prompt' parameter")
            return "Error: 'prompt' is required"

        lyrics = str(parameters.get("lyrics", "") or "")
        emotion = parameters.get("emotion")
        if len(prompt) < 10:
            return "Error: 'prompt' should be at least 10 characters"
        if lyrics and len(lyrics) < 10:
            return "Error: 'lyrics' should be at least 10 characters"
        if not self.api_key:
            return "Error: MINIMAX_API_KEY environment variable is not set"

        model = str(
            parameters.get("model")
            or self.config.get("minimax_music_model", "music-2.6-free")
        ).strip()

        logger.info("Generating music with MiniMax: prompt=%r model=%s", prompt, model)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "output_format": "url",
            "audio_setting": {
                "sample_rate": 16000,
                "bitrate": 32000,
                "format": "mp3",
            },
        }
        if lyrics:
            payload["lyrics"] = lyrics
        else:
            payload["is_instrumental"] = True

        timeout = self.config.get("minimax_music_timeout", 600)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url.rstrip('/')}/v1/music_generation",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    response.raise_for_status()
                    response_text = await response.text()

                if not response_text.strip():
                    logger.error("MiniMax returned an empty response body")
                    return "Error: Music service returned an empty response"

                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.error("MiniMax returned non-JSON response: %s", response_text[:500])
                    return "Error: Music service returned an invalid response"

                if result is None:
                    logger.error("MiniMax returned JSON null response")
                    return "Error: Music service returned an empty JSON response"

                if "error" in result:
                    error_msg = result["error"]
                    logger.error("MiniMax API error: %s", error_msg)
                    return f"Error generating music: {error_msg}"

                base_resp = result.get("base_resp")
                if isinstance(base_resp, dict) and base_resp.get("status_code") not in (None, 0):
                    status_msg = base_resp.get("status_msg", "unknown error")
                    logger.error(
                        "MiniMax API returned error status %s: %s",
                        base_resp.get("status_code"),
                        status_msg,
                    )
                    return f"Error generating music: {status_msg}"

                audio_url = ((result.get("data") or {}).get("audio"))
                if not audio_url:
                    logger.error("No audio URL in MiniMax response: %s", response_text[:500])
                    return "Error: No audio URL received from API"

                logger.info("Received audio URL: %s", audio_url)
                async with session.get(audio_url) as download_response:
                    download_response.raise_for_status()
                    mp3_audio_data = await download_response.read()

                if not mp3_audio_data:
                    return "Error: No audio data received from API"

            mp3_path = self._process_generated_audio(
                audio_data=mp3_audio_data,
                prompt=prompt,
                lyrics=lyrics,
                emotion=emotion,
                comment_lines=[
                    "Generated by MiniMax AI",
                    f"Model: {model}",
                ],
            )
            return (
                "Music generated and playing. "
                f"Download URL (valid 24h): {audio_url}\nLocal MP3: {mp3_path}"
            )
        except aiohttp.ClientResponseError as exc:
            logger.error("MiniMax API returned HTTP %s: %s", exc.status, exc.message)
            return f"Error: MiniMax API returned HTTP {exc.status}: {exc.message}"
        except aiohttp.ClientError as exc:
            logger.error("HTTP request failed: %s", exc)
            return f"Error: Failed to connect to music generation API: {exc}"
        except Exception as exc:
            logger.error("Music generation failed: %s", exc, exc_info=True)
            return f"Error generating music: {exc}"


class GeminiMusicTool(MusicTool):
    def __init__(self, config: Config, response_player):
        super().__init__(config, response_player)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.base_url = config.get(
            "gemini_base_url", "https://generativelanguage.googleapis.com"
        )

    @property
    def provider_display_name(self) -> str:
        return "Gemini"

    @property
    def required_api_key_env_var(self) -> str:
        return "GEMINI_API_KEY"

    @property
    def filename_prefix(self) -> str:
        return "gemini_music"

    @property
    def metadata_artist_name(self) -> str:
        return "Gemini Lyria"

    @property
    def metadata_composer_name(self) -> str:
        return "Google Gemini"

    async def generate_music_async(self, parameters: dict) -> str:
        prompt = self._prompt_from_parameters(parameters)
        if not prompt:
            return "Error: 'prompt' is required"
        if len(prompt) < 10:
            return "Error: 'prompt' should be at least 10 characters"
        if not self.api_key:
            return "Error: GEMINI_API_KEY environment variable is not set"

        lyrics = str(parameters.get("lyrics", "") or "")
        if lyrics:
            prompt = f"{prompt}\n\nLyrics:\n{lyrics}"

        model = str(
            parameters.get("model")
            or self.config.get("gemini_music_model", "lyria-3-pro-preview")
        ).strip()
        emotion = parameters.get("emotion")
        timeout = self.config.get("gemini_music_timeout", 600)

        logger.info("Generating music with Gemini: prompt=%r model=%s", prompt, model)

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseModalities": ["AUDIO", "TEXT"],
                    },
                }

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
                logger.error("Gemini returned non-JSON: %s", response_text[:500])
                return "Error: Music service returned an invalid response"

            if "error" in result:
                error_msg = result["error"].get("message", str(result["error"]))
                logger.error("Gemini API error: %s", error_msg)
                return f"Error generating music: {error_msg}"

            audio_data = None
            lyrics_text = ""
            candidates = result.get("candidates", [])
            if not candidates:
                logger.error("No candidates in Gemini response: %s", response_text[:500])
                return "Error: No audio candidates in API response"

            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data and inline_data.get("data"):
                    audio_data = base64.b64decode(inline_data["data"])
                elif "text" in part:
                    lyrics_text = part["text"]

            if not audio_data:
                logger.error("No inlineData audio found in Gemini response: %s", response_text[:500])
                return "Error: No audio data received from API"

            mp3_path = self._process_generated_audio(
                audio_data=audio_data,
                prompt=prompt,
                lyrics=lyrics_text,
                emotion=emotion,
                comment_lines=[
                    "Generated by Google Gemini Lyria",
                    f"Model: {model}",
                ],
            )

            result_msg = f"Music generated and playing. Local MP3: {mp3_path}"
            if lyrics_text:
                result_msg += f"\nGenerated lyrics/description:\n{lyrics_text}"
            return result_msg
        except aiohttp.ClientResponseError as exc:
            logger.error("HTTP error from Gemini API: %s %s", exc.status, exc.message)
            return f"Error: Gemini API returned HTTP {exc.status}: {exc.message}"
        except aiohttp.ClientError as exc:
            logger.error("HTTP request failed: %s", exc)
            return f"Error: Failed to connect to Gemini music API: {exc}"
        except Exception as exc:
            logger.error("Music generation failed: %s", exc, exc_info=True)
            return f"Error generating music: {exc}"


def normalize_music_provider(provider: Optional[str]) -> str:
    normalized = str(provider or "minimax").strip().lower()
    if normalized in {"minimax", "mini max"}:
        return "minimax"
    if normalized in {"gemini", "google", "lyria"}:
        return "gemini"
    raise ValueError(
        f"Unsupported music_tool_provider: {provider}. Expected 'gemini' or 'minimax'."
    )


def create_music_tool(config: Config, response_player) -> MusicTool:
    provider = normalize_music_provider(config.get("music_tool_provider", "minimax"))
    if provider == "gemini":
        return GeminiMusicTool(config, response_player)
    return MiniMaxMusicTool(config, response_player)


class MockResponsePlayer:
    """Mock ResponsePlayer for testing."""

    def __init__(self):
        self.queue = []

    def add(self, item):
        self.queue.append(item)
        logger.info("Added to playback queue: %s", item[2] if len(item) > 2 else "audio")


async def main():
    logger.info("Starting music tool test...")

    config = Config()
    response_player = MockResponsePlayer()
    tool = create_music_tool(config, response_player=response_player)

    required_env_var = tool.required_api_key_env_var
    if not os.environ.get(required_env_var):
        print(f"Error: {required_env_var} environment variable not set")
        print(f"Please set it with: export {required_env_var}=your_api_key")
        return

    try:
        result = await tool.generate_music_async(
            {
                "prompt": "Upbeat electronic music with energetic beats and futuristic sounds",
            }
        )
        print(f"\n{'=' * 60}")
        print(f"Result: {result}")
        print(f"{'=' * 60}\n")
    except Exception as exc:
        logger.error("Music tool test failed: %s", exc, exc_info=True)
        print(f"\nError: {exc}\n")


if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
