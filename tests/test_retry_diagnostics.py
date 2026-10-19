import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from src.tools import NonRetryableError, retry_async, retry_async_generator


class TestRetryDiagnostics(unittest.IsolatedAsyncioTestCase):
    decorators = (retry_async, retry_async_generator)

    def make_operation(self, decorator, action, **options):
        if decorator is retry_async:

            async def request(*args, **kwargs):
                return action(*args, **kwargs)

        else:

            async def request(*args, **kwargs):
                yield action(*args, **kwargs)

        operation = decorator(**options)(request)
        self.assertEqual(operation.__qualname__, request.__qualname__)

        async def invoke(*args, **kwargs):
            if decorator is retry_async:
                return await operation(*args, **kwargs)
            return [item async for item in operation(*args, **kwargs)]

        return invoke, f"{request.__module__}.{request.__qualname__}"

    async def test_success_does_not_log_or_sleep(self):
        for decorator in self.decorators:
            with self.subTest(decorator=decorator.__name__):
                action = Mock(return_value="success")
                invoke, _ = self.make_operation(decorator, action)
                with (
                    patch("src.tools.logger") as logger,
                    patch("src.tools.asyncio.sleep", new_callable=AsyncMock) as sleep,
                ):
                    result = await invoke("audio", key="credential")
                self.assertEqual(result, "success" if decorator is retry_async else ["success"])
                action.assert_called_once_with("audio", key="credential")
                logger.warning.assert_not_called()
                logger.error.assert_not_called()
                sleep.assert_not_awaited()

    async def test_empty_timeout_identifies_operation_and_type(self):
        for decorator in self.decorators:
            with self.subTest(decorator=decorator.__name__):
                action = Mock(side_effect=[asyncio.TimeoutError(), "success"])
                invoke, name = self.make_operation(
                    decorator, action, max_retries=3, initial_retry_delay=2, jitter_factor=0
                )
                with (
                    self.assertLogs("src.tools", level="WARNING") as logs,
                    patch("src.tools.asyncio.sleep", new_callable=AsyncMock) as sleep,
                ):
                    result = await invoke()
                self.assertEqual(result, "success" if decorator is retry_async else ["success"])
                self.assertEqual(action.call_count, 2)
                sleep.assert_awaited_once_with(2)
                self.assertEqual(len(logs.records), 1)
                self.assertEqual(logs.records[0].levelno, logging.WARNING)
                self.assertIn(name, logs.output[0])
                self.assertIn("TimeoutError", logs.output[0])
                self.assertIn("Attempt 1/3", logs.output[0])
                self.assertIn("Retrying in 2.00 seconds", logs.output[0])

    async def test_exhaustion_logs_error_and_reraises_original_exception(self):
        for decorator in self.decorators:
            with self.subTest(decorator=decorator.__name__):
                error = TimeoutError()
                action = Mock(side_effect=error)
                invoke, name = self.make_operation(
                    decorator,
                    action,
                    max_retries=3,
                    initial_retry_delay=2,
                    backoff_factor=3,
                    jitter_factor=0.25,
                )
                with (
                    self.assertLogs("src.tools", level="WARNING") as logs,
                    patch("src.tools.asyncio.sleep", new_callable=AsyncMock) as sleep,
                    patch("src.tools.random.uniform", side_effect=[0.25, 1]) as jitter,
                    self.assertRaises(TimeoutError) as raised,
                ):
                    await invoke()
                self.assertIs(raised.exception, error)
                self.assertEqual(action.call_count, 3)
                self.assertEqual(sleep.await_args_list, [call(2.25), call(7)])
                self.assertEqual(jitter.call_args_list, [call(0, 0.5), call(0, 1.5)])
                self.assertEqual(
                    [record.levelno for record in logs.records],
                    [logging.WARNING, logging.WARNING, logging.ERROR],
                )
                for message in logs.output:
                    self.assertIn(name, message)
                    self.assertIn("TimeoutError", message)
                self.assertIn("failed after 3 attempts", logs.output[-1])

    async def test_permanent_errors_and_cancellation_bypass_retry(self):
        for decorator in self.decorators:
            for error in (NonRetryableError("permanent"), asyncio.CancelledError()):
                with self.subTest(decorator=decorator.__name__, error=type(error).__name__):
                    action = Mock(side_effect=error)
                    invoke, _ = self.make_operation(decorator, action, max_retries=3)
                    with (
                        patch("src.tools.logger") as logger,
                        patch("src.tools.asyncio.sleep", new_callable=AsyncMock) as sleep,
                        self.assertRaises(type(error)) as raised,
                    ):
                        await invoke()
                    self.assertIs(raised.exception, error)
                    action.assert_called_once_with()
                    sleep.assert_not_awaited()
                    logger.warning.assert_not_called()
                    logger.error.assert_not_called()

    async def test_cancellation_during_backoff_bypasses_retry(self):
        for decorator in self.decorators:
            with self.subTest(decorator=decorator.__name__):
                cancellation = asyncio.CancelledError()
                action = Mock(side_effect=TimeoutError())
                invoke, _ = self.make_operation(decorator, action, max_retries=3)
                with (
                    patch("src.tools.logger") as logger,
                    patch(
                        "src.tools.asyncio.sleep", new_callable=AsyncMock, side_effect=cancellation
                    ) as sleep,
                    self.assertRaises(asyncio.CancelledError) as raised,
                ):
                    await invoke()
                self.assertIs(raised.exception, cancellation)
                action.assert_called_once_with()
                sleep.assert_awaited_once()
                logger.warning.assert_called_once()
                logger.error.assert_not_called()

    async def test_generator_preserves_yields_and_restarts_after_failure(self):
        attempts = 0

        @retry_async_generator(max_retries=2, jitter_factor=0)
        async def stream():
            nonlocal attempts
            attempts += 1
            yield "first"
            if attempts == 1:
                raise TimeoutError()
            yield "last"

        with (
            self.assertLogs("src.tools", level="WARNING") as logs,
            patch("src.tools.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            result = [item async for item in stream()]

        self.assertEqual(result, ["first", "first", "last"])
        self.assertEqual(attempts, 2)
        sleep.assert_awaited_once_with(1)
        self.assertIn(f"{stream.__module__}.{stream.__qualname__}", logs.output[0])

    async def test_logs_only_safe_numeric_metadata_without_payloads(self):
        class ProviderError(Exception):
            status = 503
            status_code = 503
            errno = 104

        secret = "https://example.test/?key=private-key Authorization: Bearer secret"
        audio = "private-audio-payload" * 1000
        for decorator in self.decorators:
            with self.subTest(decorator=decorator.__name__):
                error = ProviderError(f"{secret}\n{audio}")
                error.request = {"headers": secret, "audio": audio}
                error.response = {"body": audio}
                action = Mock(side_effect=error)
                invoke, name = self.make_operation(decorator, action, max_retries=2)
                with (
                    self.assertLogs("src.tools", level="WARNING") as logs,
                    patch("src.tools.asyncio.sleep", new_callable=AsyncMock),
                    self.assertRaises(ProviderError),
                ):
                    await invoke(audio, key=secret)
                for message in logs.output:
                    self.assertIn(name, message)
                    self.assertIn("ProviderError status=503 status_code=503 errno=104", message)
                    self.assertNotIn(secret, message)
                    self.assertNotIn("private-key", message)
                    self.assertNotIn("private-audio", message)
                    self.assertNotIn("\n", message)
                    self.assertLess(len(message), 400)
                self.assertTrue(all(record.exc_info is None for record in logs.records))

    async def test_untrusted_metadata_and_broken_exception_text_cannot_break_retries(self):
        class UnprintableError(Exception):
            status_code = "503 with private credentials"
            errno = 10**100

            @property
            def status(self):
                raise ValueError("private data")

            def __str__(self):
                raise AssertionError("Exception text must not be logged")

            def __repr__(self):
                raise AssertionError("Exception repr must not be logged")

        for decorator in self.decorators:
            with self.subTest(decorator=decorator.__name__):
                action = Mock(side_effect=[UnprintableError(), "success"])
                invoke, _ = self.make_operation(decorator, action)
                with (
                    self.assertLogs("src.tools", level="WARNING") as logs,
                    patch("src.tools.asyncio.sleep", new_callable=AsyncMock),
                ):
                    await invoke()
                self.assertEqual(action.call_count, 2)
                self.assertIn("UnprintableError", logs.output[0])
                self.assertNotIn("status", logs.output[0])
                self.assertNotIn("errno", logs.output[0])
                self.assertNotIn("private", logs.output[0])


if __name__ == "__main__":
    unittest.main()
