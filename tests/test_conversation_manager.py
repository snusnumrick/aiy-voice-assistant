import unittest
from unittest.mock import Mock
from src.conversation_manager import ConversationManager


class TestConversationManager(unittest.TestCase):
    def setUp(self):
        self.mock_config = Mock()
        self.mock_ai_model = Mock()
        self.conversation_manager = ConversationManager(self.mock_config, self.mock_ai_model)

    def test_estimate_tokens(self):
        text = "Hello, world! How are you?"
        tokens = self.conversation_manager.estimate_tokens(text)
        self.assertEqual(tokens, 9)  # 5 words + 4 punctuation marks

    def test_get_response(self):
        self.mock_ai_model.get_response.return_value = "I'm fine, thank you!"
        response = self.conversation_manager.get_response("How are you?")
        self.assertEqual(response, "I'm fine, thank you!")
        self.assertEqual(len(self.conversation_manager.message_history),
                         3)  # system message + user message + AI response

    def test_summarize_and_compress_history(self):
        self.conversation_manager.message_history.extend([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks for asking!"}
        ])
        self.mock_ai_model.get_response.return_value = "A friendly greeting exchange"
        self.conversation_manager.summarize_and_compress_history()
        self.assertEqual(len(self.conversation_manager.message_history), 2)  # system message + summary


if __name__ == '__main__':
    unittest.main()
