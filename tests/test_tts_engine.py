import unittest
from unittest.mock import Mock, patch
from src.tts_engine import OpenAITTSEngine, GoogleTTSEngine


class TestTTSEngine(unittest.TestCase):
    def setUp(self):
        self.mock_config = Mock()

    @patch('src.tts_engine.OpenAI')
    def test_openai_tts_engine(self, mock_openai):
        engine = OpenAITTSEngine(self.mock_config)
        engine.synthesize("Hello, world!", "test.wav")
        mock_openai.return_value.audio.speech.create.assert_called_once()

    @patch('src.tts_engine.texttospeech.TextToSpeechClient')
    def test_google_tts_engine(self, mock_tts_client):
        engine = GoogleTTSEngine(self.mock_config)
        with patch("builtins.open", mock_open()) as mock_file:
            engine.synthesize("Hello, world!", "test.wav")
        mock_tts_client.return_value.synthesize_speech.assert_called_once()
        mock_file.assert_called_once_with("test.wav", "wb")


if __name__ == '__main__':
    unittest.main()
