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
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)

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
                            saved_path = self._save_with_metadata(
                                image_data, 
                                self.pictures_dir, 
                                filename_prefix=f"image_{timestamp}_{safe_prompt}",
                                prompt=prompt,
                                title=title,
                                caption=caption,
                                original_mime_type=mime_type
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

    def _save_with_metadata(self, image_data: bytes, directory: Path, filename_prefix: str, prompt: str, title: str, caption: str, original_mime_type: str) -> Path:
        """
        Save image as JPEG with EXIF metadata using Pillow.
        """
        from PIL import ImageFile
        ImageFile.MAXBLOCK = 64 * 1024 * 1024 # Increase buffer size to prevent encoder errors
        
        image = Image.open(io.BytesIO(image_data))
        logger.info(f"Processing image: Format={image.format}, Mode={image.mode}, Size={image.size}")
        
        # Unconditionally convert to RGB to ensure JPEG compatibility
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        filename = f"{filename_prefix}.jpg"
        file_path = directory / filename
        
        # Common standard tags
        timestamp_str = time.strftime("%Y:%m:%d %H:%M:%S")
        
        def get_standard_exif():
            exif = image.getexif() # For Pillow 6.2.2, get from image
            
            # ImageDescription (0x010E): Must be ASCII. 
            # We force it to safe ASCII string to avoid Pillow crashing when it tries to encode bytes or non-ascii.
            # Full unicode text is preserved in UserComment and XP tags.
            safe_desc = str(caption).encode('ascii', 'replace').decode('ascii')
            exif[int(0x010E)] = safe_desc
            
            exif[int(0x013B)] = "Gemini AI"
            exif[int(0x0131)] = "Cubie AI Assistant"
            exif[int(0x9003)] = str(timestamp_str)
            
            # UserComment (0x9286): UNICODE prefix + UTF-16 (with BOM)
            # This is the standard EXIF way for non-ASCII comments
            try:
                exif[int(0x9286)] = b'UNICODE\x00' + str(caption).encode('utf-16')
            except Exception:
                pass # Skip if encoding fails
                
            return exif
            
        # Attempt 1: Full Metadata (Standard + XP)
        try:
            exif = get_standard_exif()
            # XP tags (UCS-2/UTF-16LE encoding) with double-null terminator
            logger.info(f"Adding XP tags: {title}, {caption}")
            if title:
                exif[int(0x9c9b)] = str(title).encode("utf-16le") + b'\x00\x00'
                logger.info(int(0x9c9b), exif[int(0x9c9b)])
            exif[int(0x9c9c)] = str(caption).encode("utf-16le") + b'\x00\x00'
            exif[int(0x9c9d)] = "Gemini AI".encode("utf-16le") + b'\x00\x00'
            
            image.save(file_path, "JPEG", quality=95, exif=exif.tobytes())
            logger.info(f"Saved JPEG with full metadata (Standard + XP): {file_path}")

            img = Image.open(file_path)
            raw_exif = img._getexif()
            logger.info(int(0x9C9B), raw_exif[int(0x9C9B)])
            return file_path
        except Exception as e:
            logger.warning(f"Failed to save with full metadata: {e}")

        # Attempt 2: Standard Metadata Only
        try:
            exif = get_standard_exif()
            image.save(file_path, "JPEG", quality=95, exif=exif.tobytes())
            logger.info(f"Saved JPEG with standard metadata only: {file_path}")
            return file_path
        except Exception as e:
            logger.warning(f"Failed to save with standard metadata: {e}")

        # Attempt 3: No Metadata (Fallback)
        try:
            # Clear any potential internal state issues by reloading or just saving directly
            image.save(file_path, "JPEG", quality=95)
            logger.info(f"Saved JPEG without metadata (fallback): {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save image as JPEG (High Quality): {type(e).__name__}: {e}")
            
            # Attempt 4: No Metadata, Default Quality (Lower fallback)
            try:
                image.save(file_path, "JPEG")
                logger.info(f"Saved JPEG with default quality: {file_path}")
                return file_path
            except Exception as e2:
                logger.error(f"Failed to save image as JPEG (Default): {type(e2).__name__}: {e2}")

                # Attempt 5: Save as PNG (Pillow re-encode)
                try:
                    png_path = file_path.with_suffix(".png")
                    image.save(png_path, "PNG")
                    logger.info(f"Saved as PNG (re-encoded): {png_path}")
                    return png_path
                except Exception as png_e:
                    logger.error(f"Failed to save as PNG re-encode: {type(png_e).__name__}: {png_e}")
                    
                    # Attempt 6: RAW WRITE (Ultimate Fallback)
                    # We have the original bytes. Just write them.
                    # Guess extension from original mime type or default to png
                    ext = ".png"
                    if original_mime_type:
                        if "jpeg" in original_mime_type or "jpg" in original_mime_type: ext = ".jpg"
                        elif "webp" in original_mime_type: ext = ".webp"
                    
                    raw_path = file_path.with_suffix(ext)
                    with open(raw_path, "wb") as f:
                        f.write(image_data)
                    logger.info(f"Saved raw image bytes (ultimate fallback): {raw_path}")
                    return raw_path

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
