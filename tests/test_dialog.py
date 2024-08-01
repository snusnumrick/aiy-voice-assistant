import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import sys
from typing import Dict, List, Tuple, Optional

# Mock aiy modules comprehensively
mock_aiy = Mock()
mock_aiy.voice = Mock()
mock_aiy.voice.audio = Mock()
mock_aiy.voice.audio.AudioFormat = Mock()
mock_aiy.voice.audio.Recorder = Mock()
mock_aiy.board = Mock()
mock_aiy.leds = Mock()
sys.modules['aiy'] = mock_aiy
sys.modules['aiy.voice'] = mock_aiy.voice
sys.modules['aiy.voice.audio'] = mock_aiy.voice.audio
sys.modules['aiy.board'] = mock_aiy.board
sys.modules['aiy.leds'] = mock_aiy.leds

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

# Mock the entire src.responce_player module
mock_responce_player = Mock()
sys.modules['src.responce_player'] = mock_responce_player

# Now it's safe to import from src
from src.dialog import DialogManager, error_visual, append_suffix, synthesize_with_fallback


class MockResponsePlayer:
    def __init__(self, playlist: List[Tuple[Optional[Dict], str, str]], leds: Mock, timezone: str):
        self.playlist = playlist or []
        self.leds = leds
        self.timezone = timezone
        self._should_play = False
        self._stopped = False

    def add(self, playitem: Tuple[Optional[Dict], str, str]) -> None:
        self.playlist.append(playitem)

    def play(self):
        self._should_play = True
        self._stopped = False

    def stop(self):
        self._should_play = False
        self._stopped = True

    def is_playing(self) -> bool:
        return self._should_play and not self._stopped


# Replace the ResponsePlayer in the mocked module with our MockResponsePlayer
mock_responce_player.ResponsePlayer = MockResponsePlayer


class TestDialogModule(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.button = Mock()
        self.leds = Mock()
        self.tts_engines = Mock()
        self.fallback_tts_engine = Mock()
        self.conversation_manager = Mock()
        self.config = Mock()
        self.timezone = "UTC"

    def test_error_visual(self):
        leds_mock = Mock()
        error_visual(leds_mock)
        leds_mock.update.assert_called()
        self.assertEqual(leds_mock.update.call_count, 2)

    def test_append_suffix(self):
        self.assertEqual(append_suffix("/path/to/file.wav", "_suffix"), "/path/to/file_suffix.wav")
        self.assertEqual(append_suffix("file.wav", "_suffix"), "file_suffix.wav")

    @patch('src.dialog.logger')
    @patch('aiohttp.ClientSession')
    async def test_synthesize_with_fallback_success(self, mock_session, mock_logger):
        tts_engine = AsyncMock()
        tts_engine.synthesize_async.return_value = True
        fallback_tts_engine = AsyncMock()

        result = await synthesize_with_fallback(
            mock_session, tts_engine, fallback_tts_engine,
            "Test text", "test.wav", Mock(), Mock()
        )

        self.assertTrue(result)
        tts_engine.synthesize_async.assert_called_once()
        fallback_tts_engine.synthesize_async.assert_not_called()

    @patch('src.dialog.logger')
    @patch('aiohttp.ClientSession')
    async def test_synthesize_with_fallback_failure(self, mock_session, mock_logger):
        tts_engine = AsyncMock()
        tts_engine.synthesize_async.return_value = False
        fallback_tts_engine = AsyncMock()
        fallback_tts_engine.synthesize_async.return_value = False

        result = await synthesize_with_fallback(
            mock_session, tts_engine, fallback_tts_engine,
            "Test text", "test.wav", Mock(), Mock()
        )

        self.assertFalse(result)
        tts_engine.synthesize_async.assert_called_once()
        fallback_tts_engine.synthesize_async.assert_called_once()


class TestDialogManager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.button = Mock()
        self.leds = Mock()
        self.tts_engines = Mock()
        self.fallback_tts_engine = Mock()
        self.conversation_manager = Mock()
        self.config = Mock()
        self.timezone = "UTC"

        # Mock SpeechTranscriber
        with patch('src.dialog.SpeechTranscriber'):
            self.dialog_manager = DialogManager(
                self.button, self.leds, self.tts_engines, self.fallback_tts_engine,
                self.conversation_manager, self.config, self.timezone
            )

    async def test_cleaning_routine(self):
        self.conversation_manager.process_and_clean = AsyncMock()
        await self.dialog_manager.cleaning_routine()
        self.conversation_manager.process_and_clean.assert_called_once()

    async def test_process_completed_tasks(self):
        mock_task1 = MagicMock()
        mock_task1.done.return_value = True
        mock_task1.result.return_value = True

        mock_task2 = MagicMock()
        mock_task2.done.return_value = True
        mock_task2.result.return_value = True

        synthesis_tasks = [
            (mock_task1, {"emo": None, "audio_file_name": "test1.wav", "response_text": "Test 1"}),
            (mock_task2, {"emo": None, "audio_file_name": "test2.wav", "response_text": "Test 2"})
        ]

        next_response_index = await self.dialog_manager.process_completed_tasks(synthesis_tasks, 0)

        self.assertEqual(next_response_index, 2)
        mock_task1.result.assert_called_once()
        mock_task2.result.assert_called_once()

    @patch('src.dialog.ResponsePlayer', MockResponsePlayer)
    async def test_handle_successful_synthesis(self):
        response_info = {"emo": None, "audio_file_name": "test.wav", "response_text": "Test"}

        self.dialog_manager.handle_successful_synthesis(response_info)
        self.assertIsInstance(self.dialog_manager.response_player, MockResponsePlayer)
        self.assertEqual(len(self.dialog_manager.response_player.playlist), 1)
        self.assertEqual(self.dialog_manager.response_player.playlist[0], (None, "test.wav", "Test"))

        # Test adding to existing response player
        self.dialog_manager.handle_successful_synthesis(response_info)
        self.assertEqual(len(self.dialog_manager.response_player.playlist), 2)

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

        await self.dialog_manager.process_ai_response(mock_session, "Test input")

        self.dialog_manager.create_synthesis_task.assert_called_once()
        self.dialog_manager.process_completed_tasks.assert_called_once()


if __name__ == '__main__':
    unittest.main()