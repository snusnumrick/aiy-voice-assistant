import unittest
from unittest.mock import Mock, patch
from src.dialog import main_loop


class TestDialog(unittest.TestCase):
    @patch('src.dialog.SpeechTranscriber')
    @patch('src.dialog.synthesize_speech')
    @patch('src.dialog.play_wav_async')
    def test_main_loop(self, mock_play_wav, mock_synthesize_speech, mock_transcriber):
        mock_button = Mock()
        mock_leds = Mock()
        mock_stt_engine = Mock()
        mock_tts_engine = Mock()
        mock_conversation_manager = Mock()
        mock_config = Mock()

        mock_transcriber.return_value.transcribe_speech.return_value = "Hello"
        mock_conversation_manager.get_response.return_value = "Hi there!"

        # Run the main loop once
        with patch('src.dialog.logger'):
            main_loop(mock_button, mock_leds, mock_stt_engine, mock_tts_engine, mock_conversation_manager, mock_config)

        mock_transcriber.return_value.transcribe_speech.assert_called_once()
        mock_conversation_manager.get_response.assert_called_once_with("Hello")
        mock_synthesize_speech.assert_called_once()
        mock_play_wav.assert_called_once()


if __name__ == '__main__':
    unittest.main()
