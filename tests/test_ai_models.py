import unittest
from unittest.mock import Mock, patch
from src.ai_models import OpenAIModel, ClaudeAIModel


class TestAIModels(unittest.TestCase):
    def setUp(self):
        self.mock_config = Mock()

    @patch('src.ai_models.OpenAI')
    def test_openai_model(self, mock_openai):
        model = OpenAIModel(self.mock_config)
        mock_openai.return_value.chat.completions.create.return_value.choices[0].message.content = "Hello, world!"
        response = model.get_response([{"role": "user", "content": "Hi"}])
        self.assertEqual(response, "Hello, world!")

    @patch('src.ai_models.anthropic.Anthropic')
    def test_claude_model(self, mock_anthropic):
        model = ClaudeAIModel(self.mock_config)
        mock_anthropic.return_value.completions.create.return_value.completion = "Hello, world!"
        response = model.get_response([{"role": "user", "content": "Hi"}])
        self.assertEqual(response, "Hello, world!")


if __name__ == '__main__':
    unittest.main()
