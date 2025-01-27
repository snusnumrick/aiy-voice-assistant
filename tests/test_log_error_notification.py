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
        
    def test_error_detection_and_notification(self):
        # Create log file with errors
        with open(self.log_file, 'w') as f:
            f.write("2024-01-27 10:00:00 - ERROR - Test error message\n")
            f.write("2024-01-27 10:00:01 - INFO - Normal message\n")
            f.write("2024-01-27 10:00:02 - ERROR - Another error\n")
        
        # Mock the email sending function
        with patch('smtplib.SMTP') as mock_smtp:
            # Configure the mock
            mock_smtp_instance = mock_smtp.return_value.__enter__.return_value
            
            # Run the check_logs script
            result = os.system(f'CONFIG_FILE="{self.config_file}" LOG_FILE="{self.log_file}" ./check_logs.sh')
            self.assertEqual(result, 0, "check_logs.sh failed to run")
            
            # Verify SMTP was used
            mock_smtp_instance.starttls.assert_called_once()
            mock_smtp_instance.login.assert_called_once()
            mock_smtp_instance.send_message.assert_called_once()
            
            # Get the email message that was sent
            sent_message = mock_smtp_instance.send_message.call_args[0][0]
            self.assertEqual(sent_message['Subject'], 'AIY Assistant Log Errors Found')
            message_body = sent_message.get_payload()[0].get_payload()
            self.assertIn('Found 2 errors', message_body)
            self.assertIn('Test error message', message_body)
            self.assertIn('Another error', message_body)

if __name__ == '__main__':
    unittest.main()
