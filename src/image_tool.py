import asyncio
import base64
import io
import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
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

try:
    from PIL import Image, ImageFile
    from PIL.PngImagePlugin import PngInfo

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import piexif
from iptcinfo3 import IPTCInfo

logger = logging.getLogger(__name__)

OPENAI_IMAGE_SIZE_BY_ASPECT_RATIO = {
    "1:1": "1024x1024",
    "16:9": "1536x1024",
    "9:16": "1024x1536",
    "3:4": "1152x1536",
    "4:3": "1536x1152",
}


def write_metadata_compatible(
    image_path, title, caption, date_taken, make="AI Camera", model="AI Image Model"
):
    """
    Write basic IPTC and EXIF metadata that can be read by common desktop apps.
    """
    try:
        info = IPTCInfo(image_path, force=True)
        info["object name"] = title.encode("utf-8")
        info["caption/abstract"] = caption.encode("utf-8")

        try:
            date_only = datetime.strptime(date_taken, "%Y:%m:%d %H:%M:%S").strftime("%Y%m%d")
            info["date created"] = date_only.encode("utf-8")
        except ValueError:
            logger.warning("Could not parse image date for IPTC metadata: %s", date_taken)

        info.save_as(image_path + ".tmp")
        os.replace(image_path + ".tmp", image_path)
    except Exception as exc:
        logger.warning("Error writing IPTC data: %s", exc)

    try:
        exif_dict = piexif.load(image_path)
        exif_dict["0th"][piexif.ImageIFD.Make] = make.encode("utf-8")
        exif_dict["0th"][piexif.ImageIFD.Model] = model.encode("utf-8")
        exif_dict["0th"][piexif.ImageIFD.XPTitle] = title.encode("utf-16le")
        exif_dict["0th"][piexif.ImageIFD.XPComment] = caption.encode("utf-16le")
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_taken.encode("utf-8")

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)
    except Exception as exc:
        logger.warning("Error writing EXIF data: %s", exc)


def save_with_metadata(
    image_data: bytes,
    directory: Path,
    filename_prefix: str,
    title: str,
    caption: str,
    camera_model: str,
    mime_type: Optional[str] = None,
) -> Path:
    """
    Save image with metadata when Pillow is available.
    """
    ImageFile.MAXBLOCK = 64 * 1024 * 1024
    preferred_format = _preferred_output_format(mime_type)
    if preferred_format == "PNG":
        return _save_png_with_metadata(
            image_data=image_data,
            directory=directory,
            filename_prefix=filename_prefix,
            title=title,
            caption=caption,
            camera_model=camera_model,
        )

    filename = f"{filename_prefix}.jpg"
    file_path = directory / filename

    try:
        _save_jpeg_with_pillow(image_data, file_path)
    except OSError as exc:
        if not _is_truncated_image_error(exc):
            if _is_jpeg_encoder_error(exc):
                logger.warning("JPEG encoder unavailable; saving as PNG instead: %s", exc)
                return _save_png_with_metadata(
                    image_data=image_data,
                    directory=directory,
                    filename_prefix=filename_prefix,
                    title=title,
                    caption=caption,
                    camera_model=camera_model,
                )
            raise
        logger.warning("Image stream looks truncated; retrying with tolerant PIL mode: %s", exc)
        try:
            _save_jpeg_with_pillow(image_data, file_path, allow_truncated=True)
        except OSError as retry_exc:
            if _is_jpeg_encoder_error(retry_exc):
                logger.warning(
                    "JPEG encoder unavailable after truncated-image retry; saving as PNG instead: %s",
                    retry_exc,
                )
                return _save_png_with_metadata(
                    image_data=image_data,
                    directory=directory,
                    filename_prefix=filename_prefix,
                    title=title,
                    caption=caption,
                    camera_model=camera_model,
                )
            raise

    write_metadata_compatible(
        file_path.as_posix(),
        title,
        caption,
        date_taken=datetime.now(timezone.utc).strftime("%Y:%m:%d %H:%M:%S"),
        model=camera_model,
        make="Cubie AI Assistant",
    )
    return file_path


def _preferred_output_format(mime_type: Optional[str]) -> str:
    normalized = str(mime_type or "").lower()
    if not normalized:
        return "JPEG"
    if "jpeg" in normalized or "jpg" in normalized:
        return "JPEG"
    return "PNG"


def _is_truncated_image_error(exc: OSError) -> bool:
    message = str(exc).lower()
    return (
        "broken data stream" in message
        or "image file is truncated" in message
        or "truncated file read" in message
    )


def _is_jpeg_encoder_error(exc: OSError) -> bool:
    message = str(exc).lower()
    return "encoder error" in message or "wrong jpeg library version" in message


@contextmanager
def _allow_truncated_image_loads():
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        yield
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _save_jpeg_with_pillow(
    image_data: bytes, file_path: Path, allow_truncated: bool = False
) -> None:
    image_context = _allow_truncated_image_loads() if allow_truncated else nullcontext()
    with image_context:
        with Image.open(io.BytesIO(image_data)) as image:
            logger.info(
                "Processing image: format=%s mode=%s size=%s",
                image.format,
                image.mode,
                image.size,
            )

            # Force decoding before save so truncated-stream errors happen here
            # and can be retried with Pillow's tolerant mode.
            image.load()

            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(file_path, "JPEG", quality=95)


def _save_png_with_metadata(
    image_data: bytes,
    directory: Path,
    filename_prefix: str,
    title: str,
    caption: str,
    camera_model: str,
) -> Path:
    file_path = directory / f"{filename_prefix}.png"
    try:
        _save_png_with_pillow(
            image_data=image_data,
            file_path=file_path,
            title=title,
            caption=caption,
            camera_model=camera_model,
        )
    except OSError as exc:
        if not _is_truncated_image_error(exc):
            raise
        logger.warning("PNG stream looks truncated; retrying with tolerant PIL mode: %s", exc)
        _save_png_with_pillow(
            image_data=image_data,
            file_path=file_path,
            title=title,
            caption=caption,
            camera_model=camera_model,
            allow_truncated=True,
        )
    return file_path


def _save_png_with_pillow(
    image_data: bytes,
    file_path: Path,
    title: str,
    caption: str,
    camera_model: str,
    allow_truncated: bool = False,
) -> None:
    image_context = _allow_truncated_image_loads() if allow_truncated else nullcontext()
    with image_context:
        with Image.open(io.BytesIO(image_data)) as image:
            logger.info(
                "Processing image: format=%s mode=%s size=%s",
                image.format,
                image.mode,
                image.size,
            )

            image.load()

            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                image_to_save = image.convert("RGBA")
            elif image.mode not in {"RGB", "L"}:
                image_to_save = image.convert("RGB")
            else:
                image_to_save = image

            png_info = PngInfo()
            if title:
                png_info.add_text("Title", title)
            if caption:
                png_info.add_text("Description", caption)
            png_info.add_text("Software", "Cubie AI Assistant")
            png_info.add_text("Model", camera_model)
            png_info.add_text("Creation Time", datetime.now(timezone.utc).isoformat())

            image_to_save.save(file_path, "PNG", pnginfo=png_info)


class ImageTool(ABC):
    def __init__(self, config: Config):
        self.config = config
        self._cleanup_files = []

        self.pictures_dir = self._discover_pictures_folder()
        if self.pictures_dir:
            logger.info("Using pictures folder from cubie-server: %s", self.pictures_dir)
        else:
            self.pictures_dir = Path("/tmp/generated_pictures")
            logger.warning("cubie-server not found, using fallback: %s", self.pictures_dir)
            self.pictures_dir.mkdir(parents=True, exist_ok=True)

        if not HAS_PIL:
            logger.warning(
                "Pillow (PIL) not found. Image conversion and metadata features are disabled."
            )

    @property
    @abstractmethod
    def provider_display_name(self) -> str:
        """Human-readable provider name for logs and tool descriptions."""

    @property
    @abstractmethod
    def camera_model_name(self) -> str:
        """Camera model string used for generated image metadata."""

    def _discover_pictures_folder(self) -> Optional[Path]:
        """Discover cubie-server and use its shared pictures folder when available."""
        server_url = get_server_url()

        try:
            logger.info("Discovering cubie-server at %s", server_url)
            response = requests.get(f"{server_url}/api/config/folders", timeout=2)

            if response.status_code != 200:
                logger.warning("cubie-server returned status %s", response.status_code)
                return None

            config = response.json()
            pictures_folder = config.get("pictures")
            if pictures_folder and os.path.exists(pictures_folder):
                logger.info("Server discovered. Pictures folder: %s", pictures_folder)
                return Path(pictures_folder)

            logger.warning(
                "cubie-server returned config but pictures folder is unavailable: %s",
                pictures_folder,
            )
        except requests.exceptions.RequestException:
            pass
        except Exception as exc:
            logger.warning("Error discovering pictures folder: %s", exc)

        return None

    def tool_definition(self) -> "Tool":
        from src.ai_models_with_tools import Tool, ToolParameter

        return Tool(
            name="generate_image",
            description=(
                f"Generate images, pictures, and illustrations using {self.provider_display_name}."
            ),
            iterative=True,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description=(
                        "A detailed image prompt. Prefer: scene/background -> main subject -> "
                        "important details -> constraints. Include style or medium, "
                        "framing or viewpoint, lighting or mood, and placement if layout "
                        "matters. For exact in-image text, put the text in quotes and "
                        "specify typography or placement. For photorealism, say "
                        "'photorealistic'. Explicitly list exclusions like 'no logos', "
                        "'no watermark', or 'no extra text'."
                    ),
                ),
                ToolParameter(
                    name="aspect_ratio",
                    type="string",
                    description=(
                        "Aspect ratio of the generated image. Options: '1:1', '16:9', "
                        "'9:16', '3:4', '4:3'. Default is '1:1'."
                    ),
                ),
                ToolParameter(
                    name="title",
                    type="string",
                    description="Optional title for the image metadata.",
                ),
                ToolParameter(
                    name="caption",
                    type="string",
                    description="Optional caption or detailed description for the image metadata.",
                ),
            ],
            required=["prompt"],
            processor=self.generate_image_async,
            rule_instructions={
                "russian": (
                    "Когда пользователь просит нарисовать, создать картинку, изображение, "
                    "сгенерировать фото или иллюстрацию, используй инструмент generate_image. "
                    "Всегда уточни описание, стиль и детали перед генерацией, если они не "
                    "указаны. Создай заголовок (title) и подпись (caption) для сохранения в "
                    "метаданных файла. После получения ответа обязательно сохрани путь к "
                    "файлу в память: $remember: Сгенерировано изображение: <prompt>, файл: "
                    "<путь>. Удали из памяти через 24 часа.$."
                ),
                "english": (
                    "When user asks to draw, paint, create picture, image, generate photo or "
                    "illustration, use generate_image tool. Triggers: 'draw', 'create picture', "
                    "'generate image', 'make photo'. Always ask for description, style and "
                    "details before generation if not specified. Create a title and caption to "
                    "be saved in the file metadata. After receiving response, save file path to "
                    "memory: $remember: Generated image: <prompt>, file: <path> Remove from "
                    "memory in 24 hours$."
                ),
            },
        )

    @staticmethod
    def _prompt_from_parameters(parameters: dict) -> str:
        prompt = parameters.get("prompt", "")
        return str(prompt).strip()

    @staticmethod
    def _safe_prompt_fragment(prompt: str) -> str:
        safe_prompt = "".join(c if c.isalnum() or c.isspace() else "_" for c in prompt[:30])
        return "_".join(safe_prompt.split()) or "generated_image"

    @staticmethod
    def _extension_for_mime_type(mime_type: Optional[str]) -> str:
        normalized = str(mime_type or "").lower()
        if "jpeg" in normalized or "jpg" in normalized:
            return ".jpg"
        if "gif" in normalized:
            return ".gif"
        if "webp" in normalized:
            return ".webp"
        if "bmp" in normalized:
            return ".bmp"
        return ".png"

    @staticmethod
    def openai_size_for_aspect_ratio(aspect_ratio: str) -> str:
        normalized = str(aspect_ratio or "1:1").strip()
        return OPENAI_IMAGE_SIZE_BY_ASPECT_RATIO.get(
            normalized, OPENAI_IMAGE_SIZE_BY_ASPECT_RATIO["1:1"]
        )

    def _save_generated_image(
        self,
        image_data: bytes,
        prompt: str,
        title: str,
        caption: str,
        mime_type: Optional[str] = None,
    ) -> Path:
        timestamp = int(time.time())
        safe_prompt = self._safe_prompt_fragment(prompt)
        filename_prefix = f"image_{timestamp}_{safe_prompt}"
        saved_path = None

        if HAS_PIL:
            try:
                saved_path = save_with_metadata(
                    image_data=image_data,
                    directory=self.pictures_dir,
                    filename_prefix=filename_prefix,
                    title=title,
                    caption=caption,
                    camera_model=self.camera_model_name,
                    mime_type=mime_type,
                )
            except Exception as exc:
                logger.warning("Falling back to raw image save after PIL processing failed: %s", exc)

        if saved_path:
            return saved_path

        ext = self._extension_for_mime_type(mime_type)
        file_path = self.pictures_dir / f"{filename_prefix}{ext}"
        with open(file_path, "wb") as handle:
            handle.write(image_data)
        logger.info("Saved raw image file: %s", file_path)
        return file_path

    @abstractmethod
    async def generate_image_async(self, parameters: dict) -> str:
        """Generate an image and return a user-facing status string."""


class GeminiImageTool(ImageTool):
    def __init__(self, config: Config):
        super().__init__(config)
        self.api_key = os.environ.get("GEMINI_API_KEY") or config.get("gemini_api_key")
        self.base_url = config.get(
            "gemini_base_url", "https://generativelanguage.googleapis.com"
        )

    @property
    def provider_display_name(self) -> str:
        return "Gemini"

    @property
    def camera_model_name(self) -> str:
        return "Gemini AI"

    async def generate_image_async(self, parameters: dict) -> str:
        prompt = self._prompt_from_parameters(parameters)
        if not prompt:
            return "Error: 'prompt' is required"
        if not self.api_key:
            return "Error: GEMINI_API_KEY environment variable is not set"

        title = parameters.get("title", "")
        caption = parameters.get("caption", prompt)
        aspect_ratio = parameters.get("aspect_ratio", "1:1")
        model_name = self.config.get("gemini_image_model", "gemini-2.5-flash-image")
        timeout_sec = int(self.config.get("gemini_image_timeout_sec", 60))

        logger.info(
            "Generating image with Gemini: prompt=%r aspect_ratio=%s model=%s",
            prompt,
            aspect_ratio,
            model_name,
        )

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v1beta/models/{model_name}:generateContent"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                }

                async with session.post(
                    url,
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout_sec,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error("Gemini image API error: %s - %s", response.status, error_text)
                        return (
                            f"Error generating image: API returned {response.status}. "
                            f"{error_text}"
                        )

                    result = await response.json()

            candidates = result.get("candidates") or []
            if not candidates:
                logger.error("Invalid Gemini image response: %s", result)
                return "Error: No image data received from API"

            parts = candidates[0].get("content", {}).get("parts", [])
            b64_data = None
            mime_type = None
            for part in parts:
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data and inline_data.get("data"):
                    b64_data = inline_data["data"]
                    mime_type = inline_data.get("mimeType") or inline_data.get("mime_type")
                    break

            if not b64_data:
                logger.error("No inline image data found in Gemini response: %s", result)
                return "Error: No image data received from API"

            image_data = base64.b64decode(b64_data)
            saved_path = self._save_generated_image(
                image_data=image_data,
                prompt=prompt,
                title=title,
                caption=caption,
                mime_type=mime_type or "image/png",
            )
            return f"Image generated successfully. Saved to: {saved_path}"
        except asyncio.TimeoutError:
            logger.warning("Gemini image generation timed out after %ss", timeout_sec)
            return f"Error generating image: Gemini request timed out after {timeout_sec}s"
        except Exception as exc:
            logger.error("Gemini image generation failed: %s", exc, exc_info=True)
            return f"Error generating image: {exc}"


class OpenAIImageTool(ImageTool):
    def __init__(self, config: Config):
        super().__init__(config)
        self.api_key = os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")
        self.base_url = config.get("openai_base_url", "https://api.openai.com")

    @property
    def provider_display_name(self) -> str:
        return "OpenAI"

    @property
    def camera_model_name(self) -> str:
        return "OpenAI GPT Image 2"

    @staticmethod
    def _mime_type_for_output_format(output_format: str) -> str:
        normalized = str(output_format or "png").strip().lower()
        if normalized in {"jpeg", "jpg"}:
            return "image/jpeg"
        if normalized == "webp":
            return "image/webp"
        return "image/png"

    async def generate_image_async(self, parameters: dict) -> str:
        prompt = self._prompt_from_parameters(parameters)
        if not prompt:
            return "Error: 'prompt' is required"
        if not self.api_key:
            return "Error: OPENAI_API_KEY environment variable is not set"

        title = parameters.get("title", "")
        caption = parameters.get("caption", prompt)
        aspect_ratio = parameters.get("aspect_ratio", "1:1")
        model_name = self.config.get("openai_image_model", "gpt-image-2")
        output_format = str(self.config.get("openai_image_output_format", "png")).strip() or "png"
        timeout_sec = int(self.config.get("openai_image_timeout_sec", 300))

        payload = {
            "model": model_name,
            "prompt": prompt,
            "size": self.openai_size_for_aspect_ratio(aspect_ratio),
            "n": 1,
            "output_format": output_format,
        }

        for config_key, payload_key in (
            ("openai_image_quality", "quality"),
            ("openai_image_background", "background"),
            ("openai_image_moderation", "moderation"),
        ):
            value = self.config.get(config_key)
            if value is not None and str(value).strip():
                payload[payload_key] = value

        output_compression = self.config.get("openai_image_output_compression")
        if output_compression is not None and output_format.lower() in {"jpeg", "jpg", "webp"}:
            payload["output_compression"] = output_compression

        logger.info(
            "Generating image with OpenAI: prompt=%r aspect_ratio=%s size=%s model=%s",
            prompt,
            aspect_ratio,
            payload["size"],
            model_name,
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url.rstrip('/')}/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout_sec,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error("OpenAI image API error: %s - %s", response.status, error_text)
                        return (
                            f"Error generating image: API returned {response.status}. "
                            f"{error_text}"
                        )

                    result = await response.json()

            data = result.get("data") or []
            if not data or not data[0].get("b64_json"):
                logger.error("Invalid OpenAI image response: %s", result)
                return "Error: No image data received from API"

            image_data = base64.b64decode(data[0]["b64_json"])
            saved_path = self._save_generated_image(
                image_data=image_data,
                prompt=prompt,
                title=title,
                caption=caption,
                mime_type=self._mime_type_for_output_format(output_format),
            )
            return f"Image generated successfully. Saved to: {saved_path}"
        except asyncio.TimeoutError:
            logger.warning("OpenAI image generation timed out after %ss", timeout_sec)
            return f"Error generating image: OpenAI request timed out after {timeout_sec}s"
        except Exception as exc:
            logger.error("OpenAI image generation failed: %s", exc, exc_info=True)
            return f"Error generating image: {exc}"


def normalize_image_provider(provider: Optional[str]) -> str:
    normalized = str(provider or "gemini").strip().lower()
    if normalized in {"gemini", "google"}:
        return "gemini"
    if normalized in {"openai", "gpt", "gpt-image", "gpt-image-2"}:
        return "openai"
    raise ValueError(
        f"Unsupported image_tool_provider: {provider}. Expected 'gemini' or 'openai'."
    )


def create_image_tool(config: Config) -> ImageTool:
    provider = normalize_image_provider(config.get("image_tool_provider", "gemini"))
    if provider == "openai":
        return OpenAIImageTool(config)
    return GeminiImageTool(config)


async def main():
    logger.info("Starting image tool test...")

    config = Config()
    tool = create_image_tool(config)

    required_env_var = "OPENAI_API_KEY" if isinstance(tool, OpenAIImageTool) else "GEMINI_API_KEY"
    if not os.environ.get(required_env_var):
        print(f"Error: {required_env_var} environment variable not set")
        print(f"Please set it with: export {required_env_var}=your_api_key")
        return

    try:
        result = await tool.generate_image_async(
            {
                "prompt": "A cute robot painter in a futuristic studio, digital art style",
                "aspect_ratio": "16:9",
                "title": "Robot Artist",
                "caption": "A detailed digital painting of a robot creating art.",
            }
        )
        print(f"\n{'=' * 60}")
        print(f"Result: {result}")
        print(f"{'=' * 60}\n")
    except Exception as exc:
        logger.error("Image tool test failed: %s", exc, exc_info=True)
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
