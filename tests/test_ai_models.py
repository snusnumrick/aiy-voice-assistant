import unittest
import asyncio
from unittest.mock import patch, MagicMock
from src.ai_models import ClaudeAIModel
from src.config import Config

class TestClaudeAIModel(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = Config()
        self.model = ClaudeAIModel(self.config)

    @patch('requests.post')
    def test_get_response(self, mock_post):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.content.decode.return_value = '{"content": [{"type": "text", "text": "Test response"}]}'
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        response = self.model.get_response(messages)

        self.assertEqual(response, "Test response")
        mock_post.assert_called_once()

    @patch('aiohttp.ClientSession.post')
    async def test_get_response_async(self, mock_post):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = asyncio.coroutine(lambda: '{"content": [{"type": "text", "text": "Test async response"}]}')
        mock_post.return_value.__aenter__.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        async for response in self.model.get_response_async(messages):
            self.assertEqual(response, "Test async response")

        mock_post.assert_called_once()

if __name__ == '__main__':
    asyncio.run(unittest.main())