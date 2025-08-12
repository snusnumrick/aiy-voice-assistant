import unittest
from unittest.mock import patch

from src.config import Config
from src.ai_models import OpenAIModel, ReasoningEffort


class TestReasoningEffort(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    @patch('openai.AsyncOpenAI', None)
    @patch('openai.OpenAI', None)
    @patch('requests.post')
    def test_enum_maps_to_openai_minimal(self, mock_post):
        # Mock REST /responses reply
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {"text": "ok"}

        model = OpenAIModel(
            self.config,
            model_id="gpt-5",
            reasoning_effort=ReasoningEffort.QUICK,
        )
        _ = model.get_response([{"role": "user", "content": "hi"}])

        # Verify payload mapping: quick -> minimal
        args, kwargs = mock_post.call_args
        sent_json = kwargs.get('data') or kwargs.get('json')
        # When using requests with data=json.dumps(payload)
        import json as _json
        if isinstance(sent_json, str):
            payload = _json.loads(sent_json)
        else:
            payload = sent_json
        self.assertIn('reasoning', payload)
        self.assertEqual(payload['reasoning'].get('effort'), 'minimal')

    @patch('openai.AsyncOpenAI', None)
    @patch('openai.OpenAI', None)
    @patch('requests.post')
    def test_string_aliases_and_passthrough(self, mock_post):
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {"text": "ok"}

        # internal string "comprehensive" -> high
        model = OpenAIModel(self.config, model_id="gpt-5", reasoning_effort="comprehensive")
        _ = model.get_response([{"role": "user", "content": "hi"}])
        args, kwargs = mock_post.call_args
        import json as _json
        payload = _json.loads(kwargs['data']) if isinstance(kwargs.get('data'), str) else kwargs.get('json')
        self.assertEqual(payload['reasoning'].get('effort'), 'high')

        # openai-native string "medium" should pass through unchanged
        model = OpenAIModel(self.config, model_id="gpt-5", reasoning_effort="medium")
        _ = model.get_response([{"role": "user", "content": "hi"}])
        args, kwargs = mock_post.call_args
        payload = _json.loads(kwargs['data']) if isinstance(kwargs.get('data'), str) else kwargs.get('json')
        self.assertEqual(payload['reasoning'].get('effort'), 'medium')


    @patch('requests.post')
    def test_per_call_override_sync(self, mock_post):
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {"text": "ok"}

        # Constructor says COMPREHENSIVE -> high, but override to QUICK -> minimal
        model = OpenAIModel(
            self.config,
            model_id="gpt-5",
            reasoning_effort=ReasoningEffort.COMPREHENSIVE,
        )
        _ = model.get_response([{"role": "user", "content": "hi"}], reasoning_effort=ReasoningEffort.QUICK)

        import json as _json
        args, kwargs = mock_post.call_args
        payload = _json.loads(kwargs['data']) if isinstance(kwargs.get('data'), str) else kwargs.get('json')
        self.assertEqual(payload['reasoning'].get('effort'), 'minimal')

    @patch('openai.AsyncOpenAI', None)
    @patch('openai.OpenAI', None)
    @patch('aiohttp.ClientSession.post')
    def test_per_call_override_async(self, mock_aiohttp_post):
        # Mock aiohttp response for /responses
        class _Resp:
            status = 200
            async def json(self):
                return {"text": "ok"}
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                return False
            def raise_for_status(self):
                return None
        mock_aiohttp_post.return_value.__aenter__.return_value = _Resp()

        async def _run():
            model = OpenAIModel(
                self.config,
                model_id="gpt-5",
                reasoning_effort=ReasoningEffort.COMPREHENSIVE,
            )
            out = ""
            async for chunk in model.get_response_async([{"role": "user", "content": "hi"}], reasoning_effort="quick"):
                out += chunk
            return out

        import asyncio
        out = asyncio.run(_run())
        self.assertTrue(isinstance(out, str))

        # Ensure the JSON payload used quick -> minimal
        called_kwargs = mock_aiohttp_post.call_args.kwargs
        payload = called_kwargs.get('json')
        self.assertIsNotNone(payload)
        self.assertIn('reasoning', payload)
        self.assertEqual(payload['reasoning'].get('effort'), 'minimal')


if __name__ == '__main__':
    unittest.main()
