import asyncio
import importlib
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

from mock_aiy import mock_aiy

sys.modules["aiy"] = mock_aiy
sys.modules["aiy.board"] = mock_aiy.board
sys.modules["aiy.leds"] = mock_aiy.leds
sys.modules["aiy.voice"] = mock_aiy.voice
sys.modules["aiy.voice.audio"] = mock_aiy.voice.audio

audio = importlib.import_module("src.audio")


class TestAudioStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_late_emotion_is_cancelled_when_stt_finishes(self):
        emotion_cancelled = asyncio.Event()

        async def stt():
            return "hello"

        async def emotion():
            try:
                await asyncio.Event().wait()
            finally:
                emotion_cancelled.set()

        text, annotation = await asyncio.wait_for(
            audio._run_stt_and_timely_emotion(stt(), emotion()),
            timeout=0.1,
        )

        self.assertEqual(text, "hello")
        self.assertEqual(annotation, "")
        self.assertTrue(emotion_cancelled.is_set())

    async def test_completed_emotion_is_kept(self):
        emotion_finished = asyncio.Event()

        async def stt():
            await emotion_finished.wait()
            return "hello"

        async def emotion():
            emotion_finished.set()
            return "[User emotion: calm (0.90)]"

        result = await audio._run_stt_and_timely_emotion(stt(), emotion())

        self.assertEqual(result, ("hello", "[User emotion: calm (0.90)]"))

    async def test_soniox_async_generator_does_not_block_for_next_chunk(self):
        service = audio.SonioxSpeechRecognition()
        service.asyncio = asyncio
        chunks = asyncio.Queue()

        async def source():
            while True:
                chunk = await chunks.get()
                if chunk is None:
                    break
                yield chunk

        stream = service._async_generator(source())
        pending_chunk = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        self.assertFalse(pending_chunk.done())

        await chunks.put(b"audio")
        result = await asyncio.wait_for(pending_chunk, timeout=0.1)
        self.assertEqual(result, b"audio")

        await chunks.put(None)
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

    async def test_soniox_receive_error_logs_request_details(self):
        service = audio.SonioxSpeechRecognition()
        service.asyncio = asyncio
        service.json = __import__("json")
        service.response_timeout_sec = 1
        service.ConnectionClosed = Exception
        websocket = MagicMock()
        websocket.recv = AsyncMock(
            return_value=(
                '{"error_code":408,"error_type":"request_timeout",'
                '"error_message":"Request timeout.","request_id":"request-123"}'
            )
        )

        with self.assertLogs("src.audio", level="ERROR") as logs:
            result = await service._receive_transcripts(
                websocket,
                asyncio.Event(),
                [],
                "",
            )

        self.assertEqual(result, "")
        log_text = "\n".join(logs.output)
        self.assertIn("request_timeout", log_text)
        self.assertIn("request-123", log_text)

    async def test_soniox_receive_collects_multiple_final_batches(self):
        service = audio.SonioxSpeechRecognition()
        service.asyncio = asyncio
        service.json = __import__("json")
        service.response_timeout_sec = 1
        service.ConnectionClosed = Exception
        websocket = MagicMock()
        websocket.recv = AsyncMock(
            side_effect=[
                '{"tokens":[{"text":"first ","is_final":true}]}',
                '{"tokens":[{"text":"second","is_final":true}]}',
                '{"tokens":[{"text":"<fin>","is_final":true}]}',
            ]
        )
        send_done = asyncio.Event()
        send_done.set()

        result = await service._receive_transcripts(websocket, send_done, [], "")

        self.assertEqual(result, "first second")
        self.assertEqual(websocket.recv.await_count, 3)
