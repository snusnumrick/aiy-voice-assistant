import json
import logging
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

class TestLogErrorNotification(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)
        self.log_file = self.log_dir / "assistant.log"
        
        # Create temporary config file
        self.config_file = Path(self.temp_dir) / "config.json"
        with open(self.config_file, 'w') as f:
            json.dump({
                "admin_email": "test@example.com"
            }, f)
        
        # Set up environment
        os.environ['CONFIG_FILE'] = str(self.config_file)
        
    def tearDown(self):
        # Clean up temporary files
        if self.log_file.exists():
            self.log_file.unlink()
        if self.config_file.exists():
            self.config_file.unlink()
        os.rmdir(self.temp_dir)
        
    @patch('src.email_tools.send_email')
    def test_error_detection_and_notification(self, mock_send_email):
        # Create log file with errors
        with open(self.log_file, 'w') as f:
            f.write("2024-01-27 10:00:00 - ERROR - Test error message\n")
            f.write("2024-01-27 10:00:01 - INFO - Normal message\n")
            f.write("2024-01-27 10:00:02 - ERROR - Another error\n")
        
        # Run the check_logs script
        result = os.system(f'CONFIG_FILE={self.config_file} LOG_FILE={self.log_file} ./check_logs.sh')
        self.assertEqual(result, 0, "check_logs.sh failed to run")
        
        # Verify email was sent with correct content
        mock_send_email.assert_called_once()
        args = mock_send_email.call_args[0]
        self.assertEqual(args[0], 'AIY Assistant Log Errors Found')
        self.assertIn('Found 2 errors', args[1])
        self.assertIn('Test error message', args[1])
        self.assertIn('Another error', args[1])

if __name__ == '__main__':
    unittest.main()
