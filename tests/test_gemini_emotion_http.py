"""Offline HTTP, privacy, and recovery regressions for Gemini emotion detection."""
import asyncio
import base64
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from src.emotion_engine import GeminiEmotionEngine, format_annotation


class Response:
    def __init__(self, status=200, body=None, headers=None):
        self.status = status
        self.body = body if body is not None else json.dumps({
            "candidates": [{"content": {"parts": [{"text": '{"joy": 0.8}'}]}}]
        })
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self):
        return self.body


class TestGeminiEmotionHTTP(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.engine = GeminiEmotionEngine({"emotion_detection_timeout": 10})
        self.session = MagicMock()
        self.session.__aenter__ = AsyncMock(return_value=self.session)
        self.session.__aexit__ = AsyncMock(return_value=False)
        self.client = patch("src.emotion_engine.aiohttp.ClientSession", return_value=self.session)
        self.client.start()
        self.addCleanup(self.client.stop)
        self.sleep = patch("src.emotion_engine.asyncio.sleep", new_callable=AsyncMock).start()
        self.addCleanup(patch.stopall)
        self.mark = patch("src.emotion_engine.async_mark_payment_required").start()
        self.clear = patch("src.emotion_engine.async_clear_payment_required").start()

    async def test_402_is_not_retried_and_later_turns_skip_http(self):
        self.session.post.return_value = Response(402, json.dumps({"error": {
            "code": 402, "status": "RESOURCE_EXHAUSTED",
            "message": "Your prepayment credits are depleted.",
        }}))
        with self.assertLogs("src.emotion_engine", level="ERROR") as logs:
            self.assertIsNone(await self.engine._request_scores(b"audio", "audio/wav"))
            self.assertIsNone(await self.engine._request_scores(b"audio2", "audio/wav"))
        self.assertEqual(self.session.post.call_count, 1)
        self.sleep.assert_not_awaited()
        self.mark.assert_called_once()
        self.clear.assert_not_called()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("prepayment credits are depleted", logs.output[0])
        self.assertIn("RESOURCE_EXHAUSTED", logs.output[0])
        self.assertIn("Speech recognition continues", logs.output[0])

    async def test_cooldown_expiry_allows_probe_and_valid_scores_clear_status(self):
        self.session.post.side_effect = [Response(402, '{}'), Response()]
        await self.engine._request_scores(b"audio", "audio/wav")
        self.engine._payment_retry_at = time.monotonic() - 1
        scores = await self.engine._request_scores(b"audio", "audio/wav")
        self.clear.assert_not_called()  # HTTP success alone is not valid emotion recovery.
        result = await self.engine._result_from_scores(scores)
        self.assertEqual(result.top_emotions, [("joy", 0.8)])
        self.clear.assert_called_once()
        self.assertEqual(self.session.post.call_count, 2)

    async def test_non_retryable_http_errors_fail_once(self):
        for status in (400, 401, 403, 404, 422, 501):
            with self.subTest(status=status):
                self.session.post.reset_mock()
                self.session.post.return_value = Response(status, '{}')
                self.assertIsNone(await self.engine._request_scores(b"audio", "audio/wav"))
                self.assertEqual(self.session.post.call_count, 1)
        self.sleep.assert_not_awaited()
        self.mark.assert_not_called()

    async def test_transient_http_errors_retry_and_recover(self):
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.session.post.reset_mock()
                self.session.post.side_effect = [Response(status, '{}'), Response()]
                scores = await self.engine._request_scores(b"audio", "audio/wav")
                self.assertEqual(scores, {"joy": 0.8})
                self.assertEqual(self.session.post.call_count, 2)

    async def test_transient_network_errors_retry(self):
        for error in (asyncio.TimeoutError(), aiohttp.ClientConnectionError(),
                      aiohttp.ClientPayloadError()):
            with self.subTest(error=type(error).__name__):
                self.session.post.reset_mock()
                self.session.post.side_effect = [error, Response()]
                self.assertEqual(await self.engine._request_scores(b"audio", "audio/wav"),
                                 {"joy": 0.8})
                self.assertEqual(self.session.post.call_count, 2)

    async def test_retry_exhaustion_is_bounded_and_graceful(self):
        self.session.post.side_effect = [Response(503, '{}')] * 3
        self.assertIsNone(await self.engine._request_scores(b"audio", "audio/wav"))
        self.assertEqual(self.session.post.call_count, 3)
        self.assertEqual(self.sleep.await_count, 2)
        self.assertEqual(format_annotation(None), "")
        self.clear.assert_not_called()

    async def test_retry_after_is_respected(self):
        self.session.post.side_effect = [Response(429, '{}', {"Retry-After": "3"}), Response()]
        await self.engine._request_scores(b"audio", "audio/wav")
        self.sleep.assert_awaited_once_with(3.0)

    async def test_backoff_does_not_exceed_overall_timeout(self):
        self.engine.timeout = 0.5
        self.session.post.return_value = Response(429, '{}', {"Retry-After": "60"})
        self.assertIsNone(await self.engine._request_scores(b"audio", "audio/wav"))
        self.assertEqual(self.session.post.call_count, 1)
        self.sleep.assert_not_awaited()
        request_timeout = self.session.post.call_args.kwargs["timeout"].total
        self.assertGreater(request_timeout, 0)
        self.assertLessEqual(request_timeout, 0.5)

    async def test_cancellation_propagates_without_retry(self):
        self.session.post.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.engine._request_scores(b"audio", "audio/wav")
        self.assertEqual(self.session.post.call_count, 1)
        self.sleep.assert_not_awaited()

    async def test_unexpected_error_is_not_retried(self):
        self.session.post.side_effect = ValueError("https://private.test/?key=secret")
        with self.assertLogs("src.emotion_engine", level="ERROR") as logs:
            self.assertIsNone(await self.engine._request_scores(b"audio", "audio/wav"))
        self.assertEqual(self.session.post.call_count, 1)
        self.assertIn("ValueError", logs.output[0])
        self.assertNotIn("private.test", logs.output[0])

    async def test_error_body_redacts_credentials_audio_and_omits_extra_fields(self):
        raw_audio = b"some private audio bytes"
        audio_b64 = base64.b64encode(raw_audio).decode()
        message = (f"Payment required. key test-secret-key; data='{audio_b64}'; "
                   "token=other-secret; Authorization: Bearer abc123; "
                   "https://example.test/path?api_key=secret\nnew line")
        self.session.post.return_value = Response(402, json.dumps({"error": {
            "code": 402, "status": "RESOURCE_EXHAUSTED", "message": message,
            "details": {"private": "private-request-payload"},
        }, "audio": "private-other-audio"}))
        with self.assertLogs("src.emotion_engine", level="ERROR") as logs:
            await self.engine._request_scores(raw_audio, "audio/wav")
        text = "\n".join(logs.output)
        for secret in ("test-secret-key", audio_b64, "other-secret", "abc123",
                       "example.test", "private-request-payload", "private-other-audio"):
            self.assertNotIn(secret, text)
        self.assertIn("Payment required", text)

    async def test_unstructured_error_body_is_never_logged(self):
        self.session.post.return_value = Response(402, "private raw response body")
        with self.assertLogs("src.emotion_engine", level="ERROR") as logs:
            await self.engine._request_scores(b"audio", "audio/wav")
        self.assertNotIn("private raw response body", "\n".join(logs.output))
        self.assertIn("unstructured error body omitted", "\n".join(logs.output))

    async def test_invalid_success_payload_does_not_clear_unresolved_status(self):
        self.session.post.return_value = Response(body=json.dumps({
            "candidates": [{"content": {"parts": [{"text": "private invalid text"}]}}]
        }))
        with self.assertLogs("src.emotion_engine", level="ERROR") as logs:
            self.assertIsNone(await self.engine._request_scores(b"audio", "audio/wav"))
        self.clear.assert_not_called()
        self.assertNotIn("private invalid text", "\n".join(logs.output))

    async def test_cooldown_still_drains_audio_stream(self):
        self.engine._payment_retry_at = time.monotonic() + 100
        consumed = []

        async def chunks():
            for chunk in (b"a", b"b"):
                consumed.append(chunk)
                yield chunk

        self.assertIsNone(await self.engine.detect_stream(chunks()))
        self.assertEqual(consumed, [b"a", b"b"])
        self.session.post.assert_not_called()

    async def test_default_and_invalid_cooldown_are_one_hour(self):
        self.assertEqual(self.engine.payment_cooldown, 3600)
        for setting in (0, -1, "bad", float("inf"), float("nan"), None):
            with self.subTest(setting=setting):
                engine = GeminiEmotionEngine({"gemini_emotion_payment_cooldown_sec": setting})
                self.assertEqual(engine.payment_cooldown, 3600)
        engine = GeminiEmotionEngine({"gemini_emotion_payment_cooldown_sec": 60})
        self.assertEqual(engine.payment_cooldown, 60)

    async def test_recovery_probe_is_single_flight_and_one_attempt(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        class SlowResponse(Response):
            async def text(inner_self):
                entered.set()
                await release.wait()
                return inner_self.body

        self.engine._payment_retry_at = time.monotonic() - 1
        self.session.post.return_value = SlowResponse(503, '{}')
        first = asyncio.create_task(self.engine._request_scores(b"audio", "audio/wav"))
        await entered.wait()
        self.assertIsNone(await self.engine._request_scores(b"another", "audio/wav"))
        release.set()
        self.assertIsNone(await first)
        self.assertEqual(self.session.post.call_count, 1)
        self.sleep.assert_not_awaited()
        self.clear.assert_not_called()
        self.assertGreater(self.engine._payment_retry_at, time.monotonic())
        self.assertFalse(self.engine._recovery_probe_inflight)

    async def test_repeated_402_rearms_cooldown(self):
        self.session.post.return_value = Response(402, '{}')
        await self.engine._request_scores(b"audio", "audio/wav")
        self.engine._payment_retry_at = time.monotonic() - 1
        await self.engine._request_scores(b"audio", "audio/wav")
        self.assertEqual(self.session.post.call_count, 2)
        self.assertEqual(self.mark.call_count, 2)
        self.assertGreater(self.engine._payment_retry_at, time.monotonic())
        self.clear.assert_not_called()

    async def test_invalid_score_objects_do_not_confirm_recovery(self):
        for payload in ({"unknown": 0.9}, {"joy": "bad"}, {"joy": float("nan")},
                        {"joy": float("inf")}, [], None):
            with self.subTest(payload=payload):
                self.engine._payment_retry_at = time.monotonic() + 3600
                self.assertIsNone(self.engine._parse_scores(payload))
                self.assertGreater(self.engine._payment_retry_at, time.monotonic())
        self.clear.assert_not_called()

    async def test_retry_after_http_date_respects_budget(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        retry_date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120))
        self.session.post.return_value = Response(429, '{}', {"Retry-After": retry_date})
        self.assertIsNone(await self.engine._request_scores(b"audio", "audio/wav"))
        self.assertEqual(self.session.post.call_count, 1)
        self.sleep.assert_not_awaited()

    async def test_http_402_with_unreadable_body_still_sets_status(self):
        class BrokenResponse(Response):
            async def text(self):
                raise aiohttp.ClientPayloadError("private response")

        self.session.post.return_value = BrokenResponse(402)
        self.assertIsNone(await self.engine._request_scores(b"audio", "audio/wav"))
        self.mark.assert_called_once()
        self.sleep.assert_not_awaited()

    async def test_402_headers_persist_before_any_body_failure_or_cancellation(self):
        for error in (asyncio.TimeoutError(), aiohttp.ClientConnectionError(),
                      UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid'),
                      asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                self.engine._payment_retry_at = 0
                self.session.post.reset_mock()
                self.mark.reset_mock()
                response = Response(402)
                response.text = AsyncMock(side_effect=error)
                self.session.post.return_value = response
                if isinstance(error, asyncio.CancelledError):
                    with self.assertRaises(asyncio.CancelledError):
                        await self.engine._request_scores(b"audio", "audio/wav")
                else:
                    self.assertIsNone(await self.engine._request_scores(b"audio", "audio/wav"))
                self.mark.assert_called_once()
                self.assertGreater(self.engine._payment_retry_at, time.monotonic())
                self.assertEqual(self.session.post.call_count, 1)
        self.sleep.assert_not_awaited()

    async def test_concurrent_402_prevents_another_requests_pending_retry(self):
        backoff_started = asyncio.Event()
        release_backoff = asyncio.Event()

        async def wait_in_backoff(_delay):
            backoff_started.set()
            await release_backoff.wait()

        self.sleep.side_effect = wait_in_backoff
        self.session.post.side_effect = [Response(503, '{}'), Response(402, '{}'), Response()]
        first = asyncio.create_task(self.engine._request_scores(b"first", "audio/wav"))
        await backoff_started.wait()
        self.assertIsNone(await self.engine._request_scores(b"second", "audio/wav"))
        release_backoff.set()
        self.assertIsNone(await first)
        self.assertEqual(self.session.post.call_count, 2)

    async def test_failed_status_write_is_retried_during_cooldown_without_http(self):
        self.mark.side_effect = [False, True]
        self.session.post.return_value = Response(402, '{}')
        await self.engine._request_scores(b"audio", "audio/wav")
        self.assertTrue(self.engine._pending_payment_status)
        await self.engine._request_scores(b"audio", "audio/wav")
        self.assertFalse(self.engine._pending_payment_status)
        self.assertEqual(self.mark.await_count, 2)
        self.assertEqual(self.session.post.call_count, 1)

    async def test_boolean_scores_do_not_clear_payment_status(self):
        self.engine._payment_retry_at = time.monotonic() + 3600
        self.assertIsNone(await self.engine._result_from_scores({"joy": False}))
        self.assertGreater(self.engine._payment_retry_at, time.monotonic())
        self.clear.assert_not_awaited()

    async def test_exact_cooldown_boundary_allows_one_probe(self):
        now = time.monotonic()
        self.engine._payment_retry_at = now
        self.session.post.return_value = Response()
        with patch("src.emotion_engine.time.monotonic", return_value=now):
            self.assertEqual(await self.engine._request_scores(b"audio", "audio/wav"),
                             {"joy": 0.8})
        self.assertEqual(self.session.post.call_count, 1)


    async def test_stale_failed_write_cannot_reopen_status_after_concurrent_recovery(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_write():
            entered.set()
            await release.wait()
            return False

        self.mark.side_effect = delayed_write
        self.session.post.side_effect = [Response(402, '{}'), Response()]
        failed_turn = asyncio.create_task(self.engine._request_scores(b"audio", "audio/wav"))
        await entered.wait()
        self.assertIsNotNone(await self.engine._result_from_scores({"joy": 0.8}))
        release.set()
        self.assertIsNone(await failed_turn)
        self.assertEqual(self.engine._payment_retry_at, 0)
        self.assertFalse(self.engine._pending_payment_status)
        self.assertEqual(await self.engine._request_scores(b"next turn", "audio/wav"),
                         {"joy": 0.8})
        self.mark.assert_awaited_once()


class TestGeminiStatusIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_observed_402_stays_reported_until_valid_recovery(self):
        import tempfile
        from pathlib import Path

        from src import emotion_status

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.post.side_effect = [Response(402, '{}'), Response()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), \
                    patch.object(emotion_status, "DEFAULT_STATUS_PATH", path), \
                    patch("src.emotion_engine.aiohttp.ClientSession", return_value=session):
                engine = GeminiEmotionEngine({})
                self.assertIsNone(await engine._request_scores(b"audio", "audio/wav"))
                self.assertIn("unresolved", emotion_status.payment_required_report())
                self.assertIsNone(await engine._request_scores(b"audio", "audio/wav"))
                self.assertIn("unresolved", emotion_status.payment_required_report())
                self.assertEqual(session.post.call_count, 1)
                engine._payment_retry_at = time.monotonic() - 1
                scores = await engine._request_scores(b"audio", "audio/wav")
                self.assertIn("unresolved", emotion_status.payment_required_report())
                self.assertIsNotNone(await engine._result_from_scores(scores))
                self.assertEqual(emotion_status.payment_required_report(), "")
                self.assertEqual(engine._payment_retry_at, 0)
