import unittest
from unittest.mock import patch, mock_open
from src.stt_engine import OpenAISTTEngine, GoogleSTTEngine


class TestSTTEngine(unittest.TestCase):
    @patch('src.stt_engine.openai.audio.transcriptions.create')
    def test_openai_stt_engine(self, mock_create):
        mock_create.return_value = "Hello, world!"
        engine = OpenAISTTEngine()
        with patch("builtins.open", mock_open(read_data="audio_data")):
            result = engine.transcribe("test.wav")
        self.assertEqual(result, "Hello, world!")

    @patch('src.stt_engine.speech_recognition.Recognizer')
    def test_google_stt_engine(self, mock_recognizer):
        mock_recognizer.return_value.recognize_google.return_value = "Hello, world!"
        engine = GoogleSTTEngine()
        with patch("src.stt_engine.sr.AudioFile"):
            result = engine.transcribe("test.wav")
        self.assertEqual(result, "Hello, world!")


if __name__ == '__main__':
    unittest.main()
