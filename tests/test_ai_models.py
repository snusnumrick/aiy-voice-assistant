import unittest
from unittest.mock import patch, AsyncMock
import asyncio
from src.config import Config
from src.ai_models import (
    MessageModel,
    normalize_messages,
    AIModel,
    OpenAIModel,
    ClaudeAIModel,
    OpenRouterModel,
    PerplexityModel,
)


class TestAIModels(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_normalize_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            MessageModel(role="assistant", content="Hi there!"),
        ]
        normalized = normalize_messages(messages)
        self.assertTrue(all(isinstance(msg, dict) for msg in normalized))
        self.assertEqual(normalized, [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ])

    def test_ai_model_abstract_methods(self):
        class ConcreteAIModel(AIModel):
            def get_response(self, messages):
                return "Concrete response"

            async def get_response_async(self, messages):
                yield "Async concrete response"

        model = ConcreteAIModel()
        self.assertEqual(model.get_response([]), "Concrete response")

        async def test_async():
            async for response in model.get_response_async([]):
                self.assertEqual(response, "Async concrete response")

        asyncio.run(test_async())

    @patch('openai.OpenAI')
    @patch('openai.AsyncOpenAI')
    def test_openai_model(self, MockAsyncOpenAI, MockOpenAI):
        # Mock synchronous Responses API
        mock_openai = MockOpenAI.return_value
        class SyncResp:
            text = "OpenAI response"
        mock_openai.responses.create.return_value = SyncResp()

        # Mock asynchronous Responses API
        mock_async_openai = MockAsyncOpenAI.return_value
        class AsyncResp:
            def __init__(self, text):
                self.text = text
        async def mock_acreate(*args, **kwargs):
            return AsyncResp("Async OpenAI response")
        mock_async_openai.responses.create = mock_acreate

        model = OpenAIModel(self.config)

        # Test synchronous response
        response = model.get_response([{"role": "user", "content": "Hello"}])
        self.assertEqual(response, "OpenAI response")

        # Test asynchronous response
        async def test_async():
            async_response = ""
            async for chunk in model.get_response_async([{"role": "user", "content": "Hello"}]):
                async_response += chunk
            self.assertEqual(async_response, "Async OpenAI response")

        asyncio.run(test_async())

    @patch('requests.post')
    @patch('aiohttp.ClientSession.post')
    def test_claude_ai_model(self, mock_aiohttp_post, mock_requests_post):
        mock_requests_post.return_value.content.decode.return_value = '{"content": [{"type": "text", "text": "Claude response"}]}'
        mock_aiohttp_post.return_value.__aenter__.return_value.text.return_value = '{"content": [{"type": "text", "text": "Async Claude response"}]}'

        model = ClaudeAIModel(self.config)
        response = model.get_response([{"role": "user", "content": "Hello"}])
        self.assertEqual(response, "Claude response")

        async def test_async():
            async_response = ""
            async for chunk in model.get_response_async([{"role": "user", "content": "Hello"}]):
                async_response += chunk
            self.assertEqual(async_response, "Async Claude response")

        asyncio.run(test_async())

    def test_open_router_model(self):
        with patch('src.ai_models.OpenAIModel.__init__') as mock_init:
            OpenRouterModel(self.config)
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            self.assertEqual(kwargs['base_url'], 'https://openrouter.ai/api/v1')
            self.assertEqual(kwargs['model_id'], 'anthropic/claude-3.5-sonnet')

    def test_perplexity_model(self):
        with patch('src.ai_models.OpenAIModel.__init__') as mock_init:
            PerplexityModel(self.config)
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            self.assertEqual(kwargs['base_url'], 'https://api.perplexity.ai')
            self.assertEqual(kwargs['model_id'], 'sonar')


if __name__ == '__main__':
    unittest.main()