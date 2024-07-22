import os
import unittest

from src.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file
        with open('test_config.json', 'w') as f:
            f.write('{"test_key": "test_value"}')

        # Set up test environment variable
        os.environ['APP_TEST_ENV'] = 'test_env_value'

    def tearDown(self):
        # Remove the temporary config file
        os.remove('test_config.json')

        # Remove test environment variable
        del os.environ['APP_TEST_ENV']

    def test_load_from_file(self):
        config = Config('test_config.json')
        self.assertEqual(config.get('test_key'), 'test_value')

    def test_load_from_env(self):
        config = Config('test_config.json')
        self.assertEqual(config.get('test_env'), 'test_env_value')

    def test_get_default(self):
        config = Config('test_config.json')
        self.assertEqual(config.get('non_existent_key', 'default'), 'default')


if __name__ == '__main__':
    unittest.main()
