import asyncio
import importlib
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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

    async def test_soniox_empty_finished_response_logs_diagnostics(self):
        service = audio.SonioxSpeechRecognition()
        service.asyncio = asyncio
        service.json = __import__("json")
        service.response_timeout_sec = 1
        service.ConnectionClosed = Exception
        websocket = MagicMock()
        websocket.recv = AsyncMock(return_value='{"finished":true}')

        with self.assertLogs("src.audio", level="WARNING") as logs:
            result = await service._receive_transcripts(
                websocket,
                asyncio.Event(),
                [],
                "",
            )

        self.assertEqual(result, "")
        log_text = "\n".join(logs.output)
        self.assertIn("reason=finished", log_text)
        self.assertIn("messages=1", log_text)
        self.assertIn("tokens=0", log_text)

    async def test_transcriber_forwards_first_chunk_after_button_wait(self):
        class FakeRecorder:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def record(self, *args, **kwargs):
                yield from (b"first", b"second", b"third", b"fourth")

        class FakeSpeechService:
            supports_async_audio_generator = True

            def __init__(self):
                self.chunks = []

            async def transcribe_stream(self, audio_generator, config, context=None):
                async for chunk in audio_generator:
                    self.chunks.append(chunk)
                return "heard"

        class FakeButton:
            def __init__(self):
                self.states = iter(
                    [
                        audio.ButtonState.DEPRESSED,
                        audio.ButtonState.PRESSED,
                    ]
                )

            @property
            def state(self):
                return next(self.states, audio.ButtonState.PRESSED)

        transcriber = audio.SpeechTranscriber.__new__(audio.SpeechTranscriber)
        transcriber.button = FakeButton()
        transcriber.leds = MagicMock()
        transcriber.config = MagicMock()
        transcriber.config.get.side_effect = lambda key, default=None: {
            "stt_debug_recording_enabled": False,
            "emotion_audio_limit_sec": 0,
            "speaker_audio_limit_sec": 0,
        }.get(key, default)
        transcriber.emotion_engine = None
        transcriber.speaker_engine = None
        transcriber.speech_service = FakeSpeechService()
        transcriber.audio_sample_rate = 16000
        transcriber.audio_recording_chunk_duration_sec = 0.3
        transcriber.max_number_of_chunks = 5
        transcriber.number_of_chuncks_to_record_after_button_depressed = 0
        transcriber.breathing_period_ms = 10000
        transcriber.led_breathing_color = (0, 1, 0)
        transcriber.led_recording_color = (0, 255, 0)
        transcriber.led_breathing_duration = 60
        transcriber.led_processing_color = (0, 1, 0)
        transcriber.led_processing_blink_period_ms = 300
        transcriber.timezone = "UTC"
        transcriber.check_and_schedule_tasks = AsyncMock()

        with (
            patch.object(audio, "Recorder", FakeRecorder),
            self.assertLogs("src.audio", level="INFO") as logs,
        ):
            text, annotations = await transcriber.transcribe_speech()

        self.assertEqual(text, "heard")
        self.assertEqual(annotations, "")
        self.assertEqual(transcriber.speech_service.chunks[0], b"first")
        self.assertIn("Button state transition:", "\n".join(logs.output))

    async def test_transcriber_skips_stt_when_recorder_ends_before_button(self):
        class EmptyRecorder:
            _process = None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def record(self, *args, **kwargs):
                return iter(())

        transcriber = audio.SpeechTranscriber.__new__(audio.SpeechTranscriber)
        transcriber.button = MagicMock()
        type(transcriber.button).state = property(
            lambda _: audio.ButtonState.RELEASED
        )
        transcriber.leds = MagicMock()
        transcriber.config = MagicMock()
        transcriber.config.get.side_effect = lambda key, default=None: {
            "audio_recorder_failure_retry_sec": 0,
            "audio_device_error_indication_sec": 0,
        }.get(key, default)
        transcriber.emotion_engine = None
        transcriber.speaker_engine = None
        transcriber.speech_service = MagicMock()
        transcriber.audio_sample_rate = 16000
        transcriber.audio_recording_chunk_duration_sec = 0.3
        transcriber.max_number_of_chunks = 5
        transcriber.number_of_chuncks_to_record_after_button_depressed = 3
        transcriber.breathing_period_ms = 10000
        transcriber.led_breathing_color = (0, 1, 0)
        transcriber.led_breathing_duration = 60
        transcriber.timezone = "UTC"
        transcriber.check_and_schedule_tasks = AsyncMock()

        with (
            patch.object(audio, "Recorder", EmptyRecorder),
            self.assertLogs("src.audio", level="INFO") as logs,
        ):
            result = await transcriber.transcribe_speech()

        self.assertEqual(result, ("", ""))
        log_text = "\n".join(logs.output)
        self.assertIn("Button state at recorder start", log_text)
        self.assertIn("ended before button press", log_text)
        self.assertNotIn("LED red blinking", log_text)
        self.assertEqual(transcriber.leds.update.call_count, 1)
        transcriber.speech_service.transcribe_stream.assert_not_called()

        transcriber.leds.reset_mock()
        transcriber._wait_for_audio_error_button_press = AsyncMock(return_value=True)
        with (
            patch.object(audio, "Recorder", EmptyRecorder),
            self.assertLogs("src.audio", level="INFO") as pressed_logs,
        ):
            await transcriber.transcribe_speech()

        self.assertIn("LED red blinking", "\n".join(pressed_logs.output))
        transcriber.leds.update.assert_called_with(
            audio.Leds.rgb_pattern((255, 0, 0))
        )

    async def test_audio_error_button_wait_detects_existing_press(self):
        transcriber = audio.SpeechTranscriber.__new__(audio.SpeechTranscriber)
        transcriber.button = MagicMock()
        type(transcriber.button).state = property(
            lambda _: audio.ButtonState.PRESSED
        )

        pressed = await transcriber._wait_for_audio_error_button_press(3.0)

        self.assertTrue(pressed)
        transcriber.button.wait_for_press.assert_not_called()
