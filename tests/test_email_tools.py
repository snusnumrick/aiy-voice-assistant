import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import aiosmtplib
from aiosmtplib.response import SMTPResponse

from src.config import Config
from src.email_tools import SendEmailTool, _format_email_failure


class TestEmailTools(unittest.TestCase):
    def _config(self, **overrides):
        return Config(
            config_file="missing-test-config.json",
            user_config_file="missing-test-user.json",
            **overrides,
        )

    def test_formats_sender_verification_refusal_with_actionable_hint(self):
        refusal = aiosmtplib.SMTPRecipientsRefused(
            [
                aiosmtplib.SMTPRecipientRefused(
                    550,
                    "Verification failed for <cubick@treskunov.net>\n"
                    "Mailbox is full / Blocks limit exceeded / Inode limit exceeded\n"
                    "Sender verify failed",
                    "treskunov@gmail.com",
                )
            ]
        )

        message = _format_email_failure(
            refusal,
            smtp_server="mail.treskunov.net",
            sender="cubick@treskunov.net",
            recipient="treskunov@gmail.com",
            subject="Test",
        )

        self.assertIn("mail.treskunov.net", message)
        self.assertIn("from cubick@treskunov.net to treskunov@gmail.com", message)
        self.assertIn("550 Verification failed", message)
        self.assertIn("Mailbox is full", message)
        self.assertIn("check that mailbox's quota/storage", message)

    def test_send_email_tool_returns_delivery_result_to_model(self):
        tool = SendEmailTool(self._config())
        self.assertTrue(tool.tool_definition().iterative)

        with patch(
            "src.email_tools.send_email_async",
            new=AsyncMock(return_value="Failed to send email via mail.example.test."),
        ):
            result = asyncio.run(
                tool.do_send_email(
                    {
                        "subject": "Hello",
                        "body": "Body",
                        "to": "user@example.test",
                    }
                )
            )

        self.assertEqual(result, "Failed to send email via mail.example.test.")

    def test_formats_async_partial_refusal_response_map(self):
        message = _format_email_failure(
            {"bad@example.test": SMTPResponse(550, "No such mailbox")},
            smtp_server="mail.example.test",
            sender="sender@example.test",
            recipient="bad@example.test",
            subject="Test",
        )

        self.assertIn("bad@example.test: 550 No such mailbox", message)
