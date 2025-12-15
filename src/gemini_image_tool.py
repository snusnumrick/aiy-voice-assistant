from datetime import datetime, timezone
import os
import time
import aiohttp
import requests
import base64
import logging
import io
from pathlib import Path
from typing import Dict, Optional

if __name__ == "__main__":
    # add the current directory to the python path
    import sys
    sys.path.append(os.getcwd())
    
from src.config import Config
from src.ai_models_with_tools import Tool, ToolParameter

# Optional dependency for Image processing
try:
    from PIL import Image
    from PIL import ImageFile
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import piexif
from iptcinfo3 import IPTCInfo

logger = logging.getLogger(__name__)

def write_metadata_compatible(
    image_path, title, caption, date_taken, make="AI Camera", model="Gemini Flash"
):
    """
    Writes metadata (including date_taken) into a JPEG so that it is
    readable by the provided 'get_exif_data' function.

    :param image_path: Path to the JPEG file.
    :param title: The title/object name.
    :param caption: The caption/abstract.
    :param date_taken: Date/time string in the format 'YYYY:MM:DD HH:MM:SS'.
    :param make: Camera make string.
    :param model: Camera model string.
    """

    # --- 1. Handle IPTC (Title, Caption, and Date Created) ---
    try:
        info = IPTCInfo(image_path, force=True)

        # Write Title and Caption (as before)
        info["object name"] = title.encode("utf-8")  # (2, 5)
        info["caption/abstract"] = caption.encode("utf-8")  # (2, 120)

        # Write IPTC Date Created (2, 55)
        # IPTC date is only YYYYMMDD (8 characters)
        try:
            date_only = datetime.strptime(date_taken, "%Y:%m:%d %H:%M:%S").strftime(
                "%Y%m%d"
            )
            info["date created"] = date_only.encode("utf-8")
        except ValueError:
            print(
                "Warning: Could not parse date for IPTC (requires 'YYYY:MM:DD HH:MM:SS' format). IPTC date skipped."
            )

        # Save IPTC data
        info.save_as(image_path + ".tmp")
        os.replace(image_path + ".tmp", image_path)
    except Exception as e:
        print(f"Error writing IPTC data: {e}")

    # --- 2. Handle EXIF (Make, Model, XPTitle, XPComment, and DateTimeOriginal) ---
    try:
        # Load existing exif or start fresh
        exif_dict = piexif.load(image_path)

        # Set Make and Model (0th IFD)
        exif_dict["0th"][piexif.ImageIFD.Make] = make.encode("utf-8")
        exif_dict["0th"][piexif.ImageIFD.Model] = model.encode("utf-8")

        # Set Windows-specific tags (XPTitle and XPComment)
        exif_dict["0th"][piexif.ImageIFD.XPTitle] = title.encode("utf-16le")
        exif_dict["0th"][piexif.ImageIFD.XPComment] = caption.encode("utf-16le")

        # Set Primary Date Tag (Exif IFD)
        # DateTimeOriginal is the key your function looks for first.
        # It requires the exact format 'YYYY:MM:DD HH:MM:SS'
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_taken.encode("utf-8")

        # Convert dict back to bytes and insert into image
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)

        print(f"Successfully updated metadata for {image_path}")
    except Exception as e:
        print(f"Error writing EXIF data: {e}")


def save_with_metadata(
    image_data: bytes,
    directory: Path,
    filename_prefix: str,
    title: str,
    caption: str,
) -> Path:
    """
    Save image as JPEG with EXIF metadata using Pillow.
    """

    ImageFile.MAXBLOCK = (
        64 * 1024 * 1024
    )  # Increase buffer size to prevent encoder errors

    with Image.open(io.BytesIO(image_data)) as image:
        logger.info(
            f"Processing image: Format={image.format}, Mode={image.mode}, Size={image.size}"
        )

        # Unconditionally convert to RGB to ensure JPEG compatibility
        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = f"{filename_prefix}.jpg"
        file_path = directory / filename

        image.save(file_path, "JPEG", quality=95)

    write_metadata_compatible(file_path.as_posix(), title, caption,
                              date_taken=datetime.now(timezone.utc).strftime('%Y:%m:%d %H:%M:%S'),
                              model="Gemini AI", make="Cubie AI Assistant")
    return file_path


class GeminiImageTool:
    def __init__(self, config: Config):
        """
        Initialize Gemini Image Tool

        Args:
            config: Configuration object
        """
        self.config = config
        self.api_key = os.environ.get("GEMINI_API_KEY")
        # Allow overriding base URL if needed, though standard is generativelanguage
        self.base_url = config.get("gemini_base_url", "https://generativelanguage.googleapis.com")
        self._cleanup_files = []

        # Discover cubie-server pictures folder
        self.pictures_dir = self._discover_pictures_folder()
        if self.pictures_dir:
            logger.info(f"Using pictures folder from cubie-server: {self.pictures_dir}")
        else:
            # Fallback
            self.pictures_dir = Path("/tmp/generated_pictures")
            logger.warning(f"cubie-server not found, using fallback: {self.pictures_dir}")
            self.pictures_dir.mkdir(parents=True, exist_ok=True)
            
        if not HAS_PIL:
            logger.warning("Pillow (PIL) not found. Image conversion and metadata features will be disabled.")

    def _discover_pictures_folder(self) -> Optional[Path]:
        """Discover cubie-server and get pictures folder path"""
        server_url = 'http://localhost:5001'

        try:
            logger.info(f"🔍 Discovering cubie-server at {server_url}...")
            # Using shorter timeout for discovery
            response = requests.get(f'{server_url}/api/config/folders', timeout=2)

            if response.status_code == 200:
                config = response.json()
                # Look for pictures_folder or images_folder
                pictures_folder = config.get('pictures')
                if pictures_folder and os.path.exists(pictures_folder):
                    logger.info(f"✅ Server discovered! Pictures folder: {pictures_folder}")
                    return Path(pictures_folder)
                else:
                    logger.warning(f"⚠️ Server returned config but pictures folder not found: {pictures_folder}")
            else:
                logger.warning(f"❌ Server returned status {response.status_code}")

        except requests.exceptions.RequestException:
            # Silent fail is fine for discovery
            pass
        except Exception as e:
            logger.warning(f"❌ Error discovering pictures folder: {e}")

        return None

    def tool_definition(self) -> Tool:
        """Return tool definition for AI model"""
        return Tool(
            name="generate_image",
            description="Generate images/pictures/drawings using Gemini API",
            iterative=True,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description='A detailed description of the image to generate. Include style, lighting, mood, and composition details.'
                ),
                ToolParameter(
                    name="aspect_ratio",
                    type="string",
                    description="Aspect ratio of the generated image. Options: '1:1', '16:9', '9:16', '3:4', '4:3'. Default is '1:1'."
                ),
                ToolParameter(
                    name="title",
                    type="string",
                    description="Optional title for the image metadata."
                ),
                ToolParameter(
                    name="caption",
                    type="string",
                    description="Optional caption or detailed description for the image metadata."
                ),
            ],
            required=["prompt"],
            processor=self.generate_image_async,
            
            # RULE CONTRIBUTIONS
            rule_instructions={
                "russian": (
                    "Когда пользователь просит нарисовать, создать картинку, изображение, "
                    "сгенерировать фото или иллюстрацию, используй инструмент generate_image. "
                    "Всегда уточни описание, стиль и детали перед генерацией, если они не указаны. "
                    "Создай заголовок (title) и подпись (caption) для сохранения в метаданных файла. "
                    "После получения ответа обязательно сохрани путь к файлу в память: "
                    "$remember: Сгенерировано изображение: <prompt>, файл: <путь>. Удали из памяти через 24 часа.$."
                ),
                "english": (
                    "When user asks to draw, paint, create picture, image, generate photo or illustration, "
                    "use generate_image tool. "
                    "Triggers: 'draw', 'create picture', 'generate image', 'make photo'. "
                    "Always ask for description, style and details before generation if not specified. "
                    "Create a title and caption to be saved in the file metadata. "
                    "After receiving response, save file path to memory: "
                    "$remember: Generated image: <prompt>, file: <path> Remove from memory in 24 hours$."
                )
            }
        )

    async def generate_image_async(self, parameters: Dict) -> str:
        """
        Generate image using Gemini API

        Args:
            parameters: Dict with 'prompt', optional 'aspect_ratio', 'title', 'caption'

        Returns:
            str: Success message with local path
        """
        if "prompt" not in parameters:
            return "Error: 'prompt' is required"

        prompt = parameters["prompt"]
        aspect_ratio = parameters.get("aspect_ratio", "1:1")
        title = parameters.get("title", "")
        caption = parameters.get("caption", prompt) # Use prompt as default caption
        
        # Determine model name - user specified "gemini-2.5-flash-image" or similar
        # but we allow config override
        model_name = self.config.get("gemini_image_model", "gemini-2.5-flash-image")

        logger.info(f"Generating image: prompt='{prompt}', aspect_ratio='{aspect_ratio}', model='{model_name}'")

        try:
            async with aiohttp.ClientSession() as session:
                # Construct URL for the `generateContent` endpoint as per curl example
                url = f"{self.base_url}/v1beta/models/{model_name}:generateContent"
                
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt}
                            ]
                        }
                    ]
                }
                
                async with session.post(
                    url,
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=60  # Image gen can take time
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        return f"Error generating image: API returned {response.status}. {error_text}"

                    result = await response.json()

                    # Handle response structure as per curl example
                    # Expecting: {"candidates": [{"content": {"parts": [{"inline_data": {"mime_type": "image/png", "data": "..."}}]}}]}
                    if 'candidates' not in result or not result['candidates']:
                        logger.error(f"Invalid response format: {result}")
                        return "Error: No image data received from API"
                    
                    # Extract the first candidate's content part that contains inline_data
                    b64_data = None
                    mime_type = None
                    for part in result['candidates'][0]['content']['parts']:
                        if 'inlineData' in part and 'data' in part['inlineData']:
                            b64_data = part['inlineData']['data']
                            mime_type = part['inlineData'].get('mime_type', 'image/png') # Default to png
                            break
                    
                    if not b64_data:
                        logger.error(f"No inline_data found in response: {result}")
                        return "Error: No image data received from API"

                    # Decode image
                    image_data = base64.b64decode(b64_data)
                    
                    timestamp = int(time.time())
                    safe_prompt = "".join(c if c.isalnum() or c.isspace() else "_" for c in prompt[:30])
                    safe_prompt = "_".join(safe_prompt.split())
                    
                    # Try to use Pillow to convert to JPEG and add metadata
                    saved_path = None
                    if HAS_PIL:
                        try:
                            # Pass mime_type for better image handling
                            saved_path = save_with_metadata(
                                image_data, 
                                self.pictures_dir, 
                                filename_prefix=f"image_{timestamp}_{safe_prompt}",
                                title=title,
                                caption=caption,
                            )
                        except Exception as e:
                            logger.error(f"Failed to process image with PIL: {e}", exc_info=True)
                            # Fallback to raw write will happen below if saved_path is None
                    
                    if not saved_path:
                        # Fallback: save raw bytes with detected mime type extension if possible, else png
                        ext = '.png'
                        if mime_type:
                            if 'jpeg' in mime_type: ext = '.jpeg'
                            elif 'jpg' in mime_type: ext = '.jpg'
                            elif 'gif' in mime_type: ext = '.gif'
                            elif 'webp' in mime_type: ext = '.webp'
                            elif 'bmp' in mime_type: ext = '.bmp'
                        filename = f"image_{timestamp}_{safe_prompt}{ext}"
                        file_path = self.pictures_dir / filename
                        with open(file_path, "wb") as f:
                            f.write(image_data)
                        saved_path = file_path
                        logger.info(f"Saved raw image file (fallback): {file_path}")
                    
                    return f"Image generated successfully. Saved to: {saved_path}"

        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            return f"Error generating image: {str(e)}"


async def main():

    """Test function to run GeminiImageTool directly"""
    logger.info("Starting GeminiImageTool test...")

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY=your_api_key")
        return

    config = Config()

    # Create tool instance
    tool = GeminiImageTool(config)

    try:
        # Test parameters
        test_prompt = "A cute robot painter in a futuristic studio, digital art style"
        logger.info("Generating test image...")
        result = await tool.generate_image_async({
            "prompt": test_prompt,
            "aspect_ratio": "16:9",
            "title": "Robot Artist",
            "caption": "A detailed digital painting of a robot creating art."
        })

        print(f"\n{'='*60}")
        print(f"Result: {result}")
        print(f"{ '='*60}\n")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nError: {e}\n")

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv() # Load environment variables for API key

    # Configure logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the async main function
    asyncio.run(main())
