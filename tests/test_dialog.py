# ruff: noqa: E402

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

# Import and use the custom mock_aiy
from mock_aiy import mock_aiy

# Use the custom mock_aiy
sys.modules['aiy'] = mock_aiy
sys.modules['aiy.board'] = mock_aiy.board
sys.modules['aiy.leds'] = mock_aiy.leds
sys.modules['aiy.voice'] = mock_aiy.voice
sys.modules['aiy.voice.audio'] = mock_aiy.voice.audio

# Mock Google Cloud libraries comprehensively
mock_google = Mock()
mock_google.cloud = Mock()
mock_google.cloud.speech = Mock()
mock_google.cloud.speech_v1 = Mock()
mock_google.protobuf = Mock()
sys.modules['google'] = mock_google
sys.modules['google.cloud'] = mock_google.cloud
sys.modules['google.cloud.speech'] = mock_google.cloud.speech
sys.modules['google.cloud.speech_v1'] = mock_google.cloud.speech_v1
sys.modules['google.protobuf'] = mock_google.protobuf

# Mock other necessary modules
sys.modules['pydub'] = Mock()
sys.modules['speechkit'] = Mock()
sys.modules['yandex'] = Mock()

# Now it's safe to import from src
from src.conversation_manager import TOOL_FILLER_RESPONSE_KEY
from src.dialog import DialogManager, append_suffix, error_visual, synthesize_with_fallback
from src.tts_engine import Language, Tone, TTSEngine, YandexTTSEngine


class TestDialogModule(unittest.IsolatedAsyncioTestCase):

    def make_tts_engine(self, max_length=-1, success=True):
        engine = Mock(spec=TTSEngine)
        engine.max_text_length.return_value = max_length
        engine.synthesize_async = AsyncMock(return_value=success)
        return engine

    def setUp(self):
        self.button = mock_aiy.board.Button()
        self.leds = mock_aiy.leds.Leds()
        self.tts_engines = Mock()
        self.fallback_tts_engine = Mock()
        self.conversation_manager = Mock()
        self.config = Mock()
        self.timezone = "UTC"

    def test_error_visual(self):
        error_visual(self.leds)
        self.leds.update.assert_called()
        self.assertEqual(self.leds.update.call_count, 2)

    def test_append_suffix(self):
        self.assertEqual(append_suffix("/path/to/file.wav", "_suffix"), "/path/to/file_suffix.wav")
        self.assertEqual(append_suffix("file.wav", "_suffix"), "file_suffix.wav")

    @patch('src.dialog.logger')
    @patch('aiohttp.ClientSession')
    async def test_synthesize_with_fallback_success(self, mock_session, mock_logger):
        tts_engine = self.make_tts_engine()
        fallback_tts_engine = self.make_tts_engine()

        result = await synthesize_with_fallback(mock_session, tts_engine, fallback_tts_engine, "Test text", "test.wav",
                                                Mock(), Mock())

        self.assertTrue(result)
        tts_engine.synthesize_async.assert_called_once()
        fallback_tts_engine.synthesize_async.assert_not_called()

    @patch('src.dialog.logger')
    @patch('aiohttp.ClientSession')
    async def test_synthesize_with_fallback_failure(self, mock_session, mock_logger):
        tts_engine = self.make_tts_engine(success=False)
        fallback_tts_engine = self.make_tts_engine(success=False)

        result = await synthesize_with_fallback(mock_session, tts_engine, fallback_tts_engine, "Test text", "test.wav",
                                                Mock(), Mock())

        self.assertFalse(result)
        tts_engine.synthesize_async.assert_called_once()
        fallback_tts_engine.synthesize_async.assert_called_once()

    async def test_yandex_fallback_splits_long_text_and_combines_in_order(self):
        texts = [
            "я" * 251,
            "Это длинное предложение без запятых. " * 20,
            "я" * 601,
        ]
        for text in texts:
            for primary_raises in (False, True):
                with self.subTest(text=text[:20], primary_raises=primary_raises):
                    primary = self.make_tts_engine(2500, success=False)
                    if primary_raises:
                        primary.synthesize_async.side_effect = RuntimeError("offline")
                    fallback = YandexTTSEngine.__new__(YandexTTSEngine)
                    fallback.yandex_tts_unsafe_mode = False
                    chunk_files = []

                    async def write_chunk(session, chunk, filename, tone, lang):
                        chunk_files.append(Path(filename))
                        Path(filename).write_text(chunk, encoding="utf-8")
                        return True

                    fallback._do_synthesize_async = AsyncMock(side_effect=write_chunk)

                    def combine(files, output):
                        Path(output).write_text(
                            " ".join(Path(file).read_text(encoding="utf-8") for file in files),
                            encoding="utf-8",
                        )

                    with tempfile.TemporaryDirectory() as directory:
                        output = str(Path(directory) / "speech.wav")
                        session = Mock()
                        with patch("src.tts_engine.log_tts_usage"), patch(
                            "src.dialog.combine_audio_files", side_effect=combine
                        ) as combine_audio:
                            result = await synthesize_with_fallback(
                                session, primary, fallback, text, output,
                                Tone.HAPPY, Language.RUSSIAN,
                            )

                        self.assertTrue(result)
                        primary.synthesize_async.assert_awaited_once_with(
                            session, text, output, Tone.HAPPY, Language.RUSSIAN
                        )
                        calls = fallback._do_synthesize_async.await_args_list
                        chunks = [call.args[1] for call in calls]
                        self.assertGreater(len(chunks), 1)
                        self.assertTrue(all(0 < len(chunk) <= 250 for chunk in chunks))
                        self.assertEqual("".join(chunks).replace(" ", ""), text.replace(" ", ""))
                        self.assertEqual(Path(output).read_text(encoding="utf-8"), " ".join(chunks))
                        self.assertTrue(all(call.args[0] is session for call in calls))
                        self.assertTrue(all(call.args[3:] == (Tone.HAPPY, Language.RUSSIAN) for call in calls))
                        combine_audio.assert_called_once_with([str(file) for file in chunk_files], output)
                        self.assertTrue(all(not file.parent.exists() for file in chunk_files))

    async def test_primary_splits_text_to_its_own_limit(self):
        primary = self.make_tts_engine(250)
        fallback = self.make_tts_engine()
        with patch("src.dialog.combine_audio_files") as combine_audio:
            result = await synthesize_with_fallback(
                Mock(), primary, fallback, "я" * 501, "speech.wav",
                Tone.PLAIN, Language.RUSSIAN,
            )

        self.assertTrue(result)
        self.assertEqual(
            [len(call.args[1]) for call in primary.synthesize_async.await_args_list],
            [250, 250, 1],
        )
        fallback.synthesize_async.assert_not_awaited()
        combine_audio.assert_called_once()

    async def test_fallback_preserves_short_and_unlimited_requests(self):
        for max_length, length in ((250, 249), (250, 250), (5000, 601), (-1, 601), (0, 601)):
            with self.subTest(max_length=max_length, length=length):
                primary = self.make_tts_engine(success=False)
                fallback = self.make_tts_engine(max_length)
                session = Mock()
                text = "я" * length
                with patch("src.dialog.combine_audio_files") as combine_audio:
                    result = await synthesize_with_fallback(
                        session, primary, fallback, text, "speech.wav",
                        Tone.PLAIN, Language.RUSSIAN,
                    )

                self.assertTrue(result)
                fallback.synthesize_async.assert_awaited_once_with(
                    session, text, "speech.wav", Tone.PLAIN, Language.RUSSIAN
                )
                combine_audio.assert_not_called()

    async def test_failed_or_cancelled_chunk_is_not_combined_and_temp_files_are_removed(self):
        for failure in (False, RuntimeError("offline"), asyncio.CancelledError()):
            with self.subTest(failure=failure):
                primary = self.make_tts_engine(success=False)
                fallback = self.make_tts_engine(250)
                chunk_files = []

                async def synthesize_chunk(session, text, filename, tone, lang):
                    chunk_files.append(Path(filename))
                    Path(filename).write_bytes(b"audio")
                    if len(chunk_files) == 2:
                        if isinstance(failure, BaseException):
                            raise failure
                        return failure
                    return True

                fallback.synthesize_async.side_effect = synthesize_chunk
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "speech.wav"
                    output.write_bytes(b"existing")
                    with patch("src.dialog.combine_audio_files") as combine_audio:
                        synthesis = synthesize_with_fallback(
                            Mock(), primary, fallback, "я" * 601, str(output),
                            Tone.PLAIN, Language.RUSSIAN,
                        )
                        if isinstance(failure, asyncio.CancelledError):
                            with self.assertRaises(asyncio.CancelledError):
                                await synthesis
                        else:
                            self.assertFalse(await synthesis)

                    combine_audio.assert_not_called()
                    self.assertEqual(len(chunk_files), 2)
                    self.assertTrue(all(not file.parent.exists() for file in chunk_files))
                    self.assertEqual(output.read_bytes(), b"existing")


class TestDialogManager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.button = mock_aiy.board.Button()
        self.leds = mock_aiy.leds.Leds()
        self.tts_engines = Mock()
        self.fallback_tts_engine = Mock()
        self.conversation_manager = Mock()
        self.config = Mock()
        self.timezone = "UTC"
        self.response_player = Mock()

        # Mock SpeechTranscriber
        self.speech_transcriber_patcher = patch('src.dialog.SpeechTranscriber')
        self.combine_audio_files_patcher = patch('src.tools.combine_audio_files', return_value=None)

        self.mock_speech_transcriber = self.speech_transcriber_patcher.start()
        self.mock_combine_audio_files = self.combine_audio_files_patcher.start()

        self.dialog_manager = DialogManager(self.button, self.leds, self.tts_engines, self.fallback_tts_engine,
                                            self.conversation_manager, self.config, self.timezone, self.response_player)

    def tearDown(self):
        self.speech_transcriber_patcher.stop()
        self.combine_audio_files_patcher.stop()

    async def test_process_completed_tasks(self):
        mock_task1 = asyncio.get_running_loop().create_future()
        mock_task1.set_result(True)
        mock_task2 = asyncio.get_running_loop().create_future()
        mock_task2.set_result(True)

        synthesis_tasks = [(mock_task1, {"emo": None, "audio_file_name": "test1.wav", "response_text": "Test 1"}),
                           (mock_task2, {"emo": None, "audio_file_name": "test2.wav", "response_text": "Test 2"})]

        next_response_index = await self.dialog_manager.process_completed_tasks(synthesis_tasks, 0)

        self.assertEqual(next_response_index, 2)
        # mock_task1.result.assert_called()
        # mock_task2.result.assert_called()

    async def test_process_completed_tasks_skips_stale_pre_tool_speech(self):
        task = asyncio.get_running_loop().create_future()
        task.set_result(True)
        response_info = {
            "emo": None,
            "audio_file_name": "stale.wav",
            "response_text": "Let me look.",
            "skip_playback": True,
        }

        next_response_index = await self.dialog_manager.process_completed_tasks(
            [(task, response_info)], 0
        )

        self.assertEqual(next_response_index, 1)
        self.dialog_manager.response_player.add.assert_not_called()

    async def test_process_completed_tasks_accepts_cancelled_filler(self):
        task = asyncio.get_running_loop().create_future()
        task.cancel()

        next_response_index = await self.dialog_manager.process_completed_tasks(
            [
                (
                    task,
                    {
                        "emo": None,
                        "audio_file_name": "cancelled.wav",
                        "response_text": "Let me look.",
                    },
                )
            ],
            0,
        )

        self.assertEqual(next_response_index, 1)
        self.dialog_manager.response_player.add.assert_not_called()

    async def test_discard_pending_tool_fillers_leaves_regular_speech_running(self):
        filler_task = asyncio.get_running_loop().create_future()
        regular_task = asyncio.get_running_loop().create_future()
        filler_info = {TOOL_FILLER_RESPONSE_KEY: True}
        regular_info = {TOOL_FILLER_RESPONSE_KEY: False}

        discarded = self.dialog_manager._discard_pending_tool_fillers(
            [(filler_task, filler_info), (regular_task, regular_info)]
        )

        self.assertEqual(discarded, 1)
        self.assertTrue(filler_task.cancelled())
        self.assertTrue(filler_info["skip_playback"])
        self.assertFalse(regular_task.cancelled())
        self.assertNotIn("skip_playback", regular_info)

    async def test_cleaning_routine(self):
        self.dialog_manager.conversation_manager.process_and_clean = AsyncMock()
        await self.dialog_manager.cleaning_routine()
        self.dialog_manager.conversation_manager.process_and_clean.assert_called_once()

    async def test_handle_successful_synthesis(self):
        response_info = {"emo": None, "audio_file_name": "test.wav", "response_text": "Test"}

        # Test that add() is called on the response player
        self.dialog_manager.handle_successful_synthesis(response_info)
        self.dialog_manager.response_player.add.assert_called_once_with((
            response_info["emo"],
            response_info["audio_file_name"],
            response_info["response_text"]
        ))

    @patch('aiohttp.ClientSession')
    async def test_main_loop_async(self, mock_session):
        self.dialog_manager.conversation_manager.save_dialog = Mock()
        self.dialog_manager.transcriber.transcribe_speech = AsyncMock(return_value="Test input")
        self.dialog_manager.process_ai_response = AsyncMock()

        # Make the main loop run only once
        async def mock_main_loop():
            self.dialog_manager.conversation_manager.save_dialog()
            await self.dialog_manager.transcriber.transcribe_speech(None)
            await self.dialog_manager.process_ai_response(mock_session, "Test input")
            raise Exception("Stop loop")

        self.dialog_manager.main_loop_async = mock_main_loop

        with self.assertRaises(Exception):
            await self.dialog_manager.main_loop_async()

        self.dialog_manager.conversation_manager.save_dialog.assert_called_once()
        self.dialog_manager.transcriber.transcribe_speech.assert_called_once()
        self.dialog_manager.process_ai_response.assert_called_once()

    @patch('aiohttp.ClientSession')
    async def test_process_ai_response(self, mock_session):
        async def mock_get_response(text):
            yield [{"text": "Test response", "emotion": None, "language": "en"}]

        self.dialog_manager.conversation_manager.get_response = mock_get_response
        self.dialog_manager.create_synthesis_task = Mock(return_value=(AsyncMock(), {}))
        self.dialog_manager.process_completed_tasks = AsyncMock(return_value=1)

        with patch("src.dialog.save_to_conversation", new=AsyncMock()):
            await self.dialog_manager.process_ai_response(mock_session, "Test input")

        self.dialog_manager.create_synthesis_task.assert_called_once()
        self.dialog_manager.process_completed_tasks.assert_called_once()
        self.dialog_manager.response_player.begin_response.assert_called_once()
        self.dialog_manager.response_player.finish_response.assert_called_once_with(completed=True)


if __name__ == '__main__':
    unittest.main()
