import base64
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from src.tools import NonRetryableError
from src.tts_engine import Language, Tone, YandexTTSEngine


class TestYandexTTSHTTP(unittest.IsolatedAsyncioTestCase):
    def make_engine(self):
        engine = YandexTTSEngine.__new__(YandexTTSEngine)
        engine.api_key = "test-api-key"
        engine.lang_voices = {Language.RUSSIAN: "jane"}
        engine.roles = {Tone.PLAIN: "neutral"}
        engine.yandex_tts_unsafe_mode = False
        engine._synthesize_sync_wrapper = AsyncMock(return_value=b"sdk-audio")
        return engine

    def response(self, status=200):
        response = MagicMock()
        response.status = status
        response.json = AsyncMock(return_value={"result": {"audioChunk": {
            "data": base64.b64encode(b"http-audio").decode()}}})
        response.text = AsyncMock(return_value="Rejected")
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        return context

    async def test_connection_failure_retries_http_before_fallback(self):
        engine = self.make_engine()
        session = MagicMock()
        failed = MagicMock()
        failed.__aenter__ = AsyncMock(side_effect=aiohttp.ClientConnectionError("offline"))
        failed.__aexit__ = AsyncMock(return_value=False)
        session.post.side_effect = [failed, self.response()]
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=False)
        with patch("src.tts_engine.aiohttp.ClientSession", return_value=session_context), patch(
            "src.tools.asyncio.sleep", new=AsyncMock()
        ):
            result = await engine._synthesize_async_http("Hello", Tone.PLAIN, Language.RUSSIAN)
        self.assertEqual(result, b"http-audio")
        self.assertEqual(session.post.call_count, 2)
        headers = session.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Api-Key test-api-key")
        self.assertNotIn("x-folder-id", headers)
        engine._synthesize_sync_wrapper.assert_not_awaited()

    async def test_connection_retries_exhausted_use_sdk(self):
        engine = self.make_engine()
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(side_effect=aiohttp.ClientConnectionError("offline"))
        with patch("src.tts_engine.aiohttp.ClientSession", return_value=session_context) as create, patch(
            "src.tools.asyncio.sleep", new=AsyncMock()
        ):
            result = await engine._synthesize_async_http("Hello", Tone.PLAIN, Language.RUSSIAN)
        self.assertEqual(create.call_count, 3)
        self.assertEqual(result, b"sdk-audio")
        engine._synthesize_sync_wrapper.assert_awaited_once()

    async def test_auth_error_is_not_retried(self):
        engine = self.make_engine()
        session = MagicMock()
        session.post.return_value = self.response(status=401)
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=False)
        with patch("src.tts_engine.aiohttp.ClientSession", return_value=session_context):
            with self.assertRaises(NonRetryableError):
                await engine._request_synthesis_http("Hello", Tone.PLAIN, Language.RUSSIAN)
        session.post.assert_called_once()
