import unittest
from unittest.mock import Mock, patch
from src.audio import SpeechTranscriber, synthesize_speech


class TestAudio(unittest.TestCase):
    def setUp(self):
        self.mock_button = Mock()
        self.mock_leds = Mock()
        self.mock_config = Mock()

    @patch('src.audio.speech.SpeechClient')
    def test_speech_transcriber_initialization(self, mock_speech_client):
        transcriber = SpeechTranscriber(self.mock_button, self.mock_leds, self.mock_config)
        self.assertIsNotNone(transcriber.speech_client)
        self.assertIsNotNone(transcriber.streaming_config)

    @patch('src.audio.Recorder')
    @patch('src.audio.speech.SpeechClient')
    def test_transcribe_speech(self, mock_speech_client, mock_recorder):
        transcriber = SpeechTranscriber(self.mock_button, self.mock_leds, self.mock_config)
        mock_speech_client.return_value.streaming_recognize.return_value = [
            Mock(results=[Mock(alternatives=[Mock(transcript="Hello")], is_final=True)])
        ]

        result = transcriber.transcribe_speech()
        self.assertEqual(result, "Hello")

    def test_synthesize_speech(self):
        mock_engine = Mock()
        mock_config = Mock()
        synthesize_speech(mock_engine, "Hello, world!", "test.wav", mock_config)
        mock_engine.synthesize.assert_called_once_with("Hello, world!", "test.wav")


if __name__ == '__main__':
    unittest.main()
