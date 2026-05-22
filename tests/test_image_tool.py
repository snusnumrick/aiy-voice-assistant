import asyncio
import base64
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.config import Config
from src.image_tool import (
    GeminiImageTool,
    ImageTool,
    OpenAIImageTool,
    _is_jpeg_encoder_error,
    _is_truncated_image_error,
    create_image_tool,
    normalize_image_provider,
    save_with_metadata,
)


class TestImageToolFactory(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    @patch("src.image_tool.ImageTool._discover_pictures_folder", return_value=Path("/tmp"))
    def test_factory_defaults_to_gemini(self, _mock_discover):
        tool = create_image_tool(self._config())
        self.assertIsInstance(tool, GeminiImageTool)
        self.assertIsInstance(tool, ImageTool)

    @patch("src.image_tool.ImageTool._discover_pictures_folder", return_value=Path("/tmp"))
    def test_factory_uses_openai_provider(self, _mock_discover):
        tool = create_image_tool(self._config(image_tool_provider="openai"))
        self.assertIsInstance(tool, OpenAIImageTool)
        self.assertIsInstance(tool, ImageTool)

    def test_normalize_image_provider_accepts_gpt_image_alias(self):
        self.assertEqual(normalize_image_provider("gpt-image-2"), "openai")


class TestOpenAIImageTool(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    @patch("src.image_tool.ImageTool._discover_pictures_folder", return_value=Path("/tmp"))
    @patch("src.image_tool.OpenAIImageTool._save_generated_image", return_value=Path("/tmp/out.png"))
    @patch("src.image_tool.aiohttp.ClientSession.post")
    def test_generate_image_uses_openai_generations_endpoint(
        self,
        mock_post,
        _mock_save,
        _mock_discover,
    ):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "data": [{"b64_json": base64.b64encode(b"fake-image").decode("ascii")}]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"}, clear=False):
            tool = OpenAIImageTool(self._config(openai_image_model="gpt-image-2"))
            result = asyncio.run(
                tool.generate_image_async(
                    {
                        "prompt": "Draw a robot on a skateboard",
                        "aspect_ratio": "16:9",
                        "title": "Robot",
                        "caption": "Robot on a skateboard",
                    }
                )
            )

        self.assertEqual(result, "Image generated successfully. Saved to: /tmp/out.png")
        called_url = mock_post.call_args.args[0]
        self.assertEqual(called_url, "https://api.openai.com/v1/images/generations")
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-openai-key",
        )
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "gpt-image-2")
        self.assertEqual(mock_post.call_args.kwargs["json"]["size"], "1536x1024")
        self.assertEqual(mock_post.call_args.kwargs["json"]["output_format"], "png")

    @patch("src.image_tool.ImageTool._discover_pictures_folder", return_value=Path("/tmp"))
    @patch("src.image_tool.aiohttp.ClientSession.post")
    def test_generate_image_timeout_returns_clear_error(self, mock_post, _mock_discover):
        mock_post.return_value.__aenter__.side_effect = asyncio.TimeoutError()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"}, clear=False):
            tool = OpenAIImageTool(self._config(openai_image_timeout_sec=7))
            result = asyncio.run(tool.generate_image_async({"prompt": "Draw a slow image"}))

        self.assertEqual(result, "Error generating image: OpenAI request timed out after 7s")
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 7)


class TestGeminiImageTool(unittest.TestCase):
    def _config(self, **kwargs):
        return Config(
            config_file="__missing_config__.json",
            user_config_file="__missing_user__.json",
            **kwargs,
        )

    @patch("src.image_tool.ImageTool._discover_pictures_folder", return_value=Path("/tmp"))
    @patch("src.image_tool.GeminiImageTool._save_generated_image", return_value=Path("/tmp/gemini.png"))
    @patch("src.image_tool.aiohttp.ClientSession.post")
    def test_generate_image_uses_gemini_model_override(
        self,
        mock_post,
        _mock_save,
        _mock_discover,
    ):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "data": base64.b64encode(b"gemini-image").decode("ascii"),
                                    "mimeType": "image/png",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}, clear=False):
            tool = GeminiImageTool(self._config(gemini_image_model="gemini-3-pro-image-preview"))
            result = asyncio.run(tool.generate_image_async({"prompt": "Draw a mountain lake"}))

        self.assertEqual(result, "Image generated successfully. Saved to: /tmp/gemini.png")
        called_url = mock_post.call_args.args[0]
        self.assertIn("/v1beta/models/gemini-3-pro-image-preview:generateContent", called_url)
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["x-goog-api-key"],
            "test-gemini-key",
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"],
            "Draw a mountain lake",
        )

    @patch("src.image_tool.ImageTool._discover_pictures_folder", return_value=Path("/tmp"))
    @patch("src.image_tool.aiohttp.ClientSession.post")
    def test_generate_image_timeout_returns_clear_error(self, mock_post, _mock_discover):
        mock_post.return_value.__aenter__.side_effect = asyncio.TimeoutError()

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}, clear=False):
            tool = GeminiImageTool(self._config(gemini_image_timeout_sec=9))
            result = asyncio.run(tool.generate_image_async({"prompt": "Draw a slow image"}))

        self.assertEqual(result, "Error generating image: Gemini request timed out after 9s")
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 9)


class TestImageMetadataSaving(unittest.TestCase):
    @patch("src.image_tool.write_metadata_compatible")
    @patch("src.image_tool._save_jpeg_with_pillow")
    def test_save_with_metadata_retries_truncated_image_errors(
        self, mock_save_jpeg, mock_write_metadata
    ):
        mock_save_jpeg.side_effect = [
            OSError("broken data stream when reading image file"),
            None,
        ]

        result = save_with_metadata(
            image_data=b"fake-image",
            directory=Path("/tmp"),
            filename_prefix="retry_test",
            title="Title",
            caption="Caption",
            camera_model="Camera",
        )

        self.assertEqual(result, Path("/tmp/retry_test.jpg"))
        self.assertEqual(mock_save_jpeg.call_count, 2)
        self.assertFalse(mock_save_jpeg.call_args_list[0].kwargs.get("allow_truncated", False))
        self.assertTrue(mock_save_jpeg.call_args_list[1].kwargs["allow_truncated"])
        mock_write_metadata.assert_called_once()

    @patch("src.image_tool._save_jpeg_with_pillow")
    @patch("src.image_tool._save_png_with_metadata", return_value=Path("/tmp/png_choice.png"))
    def test_save_with_metadata_uses_png_for_png_mime_type(
        self, mock_save_png, mock_save_jpeg
    ):
        result = save_with_metadata(
            image_data=b"fake-image",
            directory=Path("/tmp"),
            filename_prefix="png_choice",
            title="Title",
            caption="Caption",
            camera_model="Camera",
            mime_type="image/png",
        )

        self.assertEqual(result, Path("/tmp/png_choice.png"))
        mock_save_png.assert_called_once()
        mock_save_jpeg.assert_not_called()

    @patch("src.image_tool.write_metadata_compatible")
    @patch("src.image_tool._save_png_with_metadata", return_value=Path("/tmp/encoder_fallback.png"))
    @patch("src.image_tool._save_jpeg_with_pillow")
    def test_save_with_metadata_falls_back_to_png_on_jpeg_encoder_error(
        self, mock_save_jpeg, mock_save_png, mock_write_metadata
    ):
        mock_save_jpeg.side_effect = OSError("encoder error -2 when writing image file")

        result = save_with_metadata(
            image_data=b"fake-image",
            directory=Path("/tmp"),
            filename_prefix="encoder_fallback",
            title="Title",
            caption="Caption",
            camera_model="Camera",
            mime_type="image/jpeg",
        )

        self.assertEqual(result, Path("/tmp/encoder_fallback.png"))
        mock_save_png.assert_called_once()
        mock_write_metadata.assert_not_called()

    def test_truncated_error_detection_is_specific(self):
        self.assertTrue(_is_truncated_image_error(OSError("broken data stream when reading image file")))
        self.assertTrue(_is_truncated_image_error(OSError("image file is truncated (4 bytes not processed)")))
        self.assertFalse(_is_truncated_image_error(OSError("permission denied")))

    def test_jpeg_encoder_error_detection_is_specific(self):
        self.assertTrue(_is_jpeg_encoder_error(OSError("encoder error -2 when writing image file")))
        self.assertTrue(_is_jpeg_encoder_error(OSError("Wrong JPEG library version: library is 90")))
        self.assertFalse(_is_jpeg_encoder_error(OSError("permission denied")))


if __name__ == "__main__":
    unittest.main()
