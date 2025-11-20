import logging
import os
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict

if __name__ == "__main__":
    # add current directory to python path
    import sys

    sys.path.append(os.getcwd())

from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config

logger = logging.getLogger(__name__)


def send_email(subject: str, body: str, config: Config, sendto: str = None, attachments: list = None):
    """
    Sends an email to the user with the given subject and body using the provided email configuration.
    If the `sendto` parameter is not provided, the email will be sent to the default user_email_address from the config.

    :param subject: A string representing the subject of the email.
    :param body: A string representing the body of the email.
    :param config: A Config object containing the email configuration.
    :param sendto: Optional string representing the recipient's email address. If not specified, falls back to the
                  user's default email address (`user_email_address`) in the configuration.
    :param attachments: Optional list of file paths to attach to the email.
    :return: None

    """
    import smtplib

    assistant_email_address = config.get(
        "assistant_email_address", "cubick@treskunov.net"
    )
    user_email_address = sendto or config.get("user_email_address", "treskunov@gmail.com")
    smtp_server = config.get("smtp_server", "mail.treskunov.net")
    smtp_port = config.get("smtp_port", "26")
    username = config.get("assistant_email_username", "cubick@treskunov.net")

    password = os.environ.get("EMAIL_PASSWORD")

    # Create message
    msg = MIMEMultipart()
    msg["From"] = assistant_email_address
    msg["To"] = user_email_address
    msg["Subject"] = subject

    # Add authentication headers to improve deliverability
    # Extract domain from assistant email address for Message-ID
    sender_domain = assistant_email_address.split('@')[1] if '@' in assistant_email_address else 'localhost'
    msg["Message-ID"] = f"<{uuid.uuid4()}@{sender_domain}>"
    msg["X-Mailer"] = "AIY Voice Assistant"
    msg["X-Priority"] = "3"

    # Attach body to the email
    msg.attach(MIMEText(body, "plain"))

    # Attach files if provided
    if attachments:
        for filepath in attachments:
            try:
                with open(filepath, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)

                    # Safely encode filename for email headers
                    import urllib.parse
                    filename = os.path.basename(filepath)

                    # RFC 2231 encoding for non-ASCII filenames
                    encoded_filename = urllib.parse.quote(filename.encode('utf-8'))

                    # Add Content-Disposition with both filename and filename* parameters
                    part.add_header(
                        'Content-Disposition',
                        f"attachment; filename={filename}; filename*=UTF-8''{encoded_filename}"
                    )

                    msg.attach(part)
            except FileNotFoundError:
                logger.error(f"Attachment file not found: {filepath}")
            except Exception as e:
                logger.error(f"Error attaching file {filepath}: {str(e)}")

    try:
        # Create SMTP session
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Enable TLS
            server.login(username, password)

            # Send email
            server.send_message(msg)
        logger.info(f"sent email about {subject}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


async def send_email_async(subject: str, body: str, config: Config, sendto: str = None, attachments: list = None):
    """
    Sends an email to the user asynchronously with the given subject and body using the provided email configuration.
    Sends an email to the user asynchronously with the given subject and body using the provided email configuration.
    If the `sendto` parameter is not provided, the email will be sent to the default user_email_address from the config.
    :param subject: A string representing the subject of the email.
    :param body: A string representing the body of the email.
    :param config: A Config object containing the email configuration.
    :param sendto: Optional string representing the recipient's email address. If not specified, falls back to the
                   user's default email address (`user_email_address`) in the configuration.
    :param attachments: Optional list of file paths to attach to the email.
    :return: None
    """
    import aiosmtplib

    assistant_email_address = config.get(
        "assistant_email_address", "cubick@treskunov.net"
    )
    user_email_address = sendto or config.get("user_email_address", "treskunov@gmail.com")
    smtp_server = config.get("smtp_server", "mail.treskunov.net")
    smtp_port = config.get("smtp_port", 26)  # Note: Changed to int
    username = config.get("assistant_email_username", "cubick@treskunov.net")

    password = os.environ.get("EMAIL_PASSWORD")

    # Create message
    msg = MIMEMultipart()
    msg["From"] = assistant_email_address
    msg["To"] = user_email_address
    msg["Subject"] = subject

    # Add authentication headers to improve deliverability
    # Extract domain from assistant email address for Message-ID
    sender_domain = assistant_email_address.split('@')[1] if '@' in assistant_email_address else 'localhost'
    msg["Message-ID"] = f"<{uuid.uuid4()}@{sender_domain}>"
    msg["X-Mailer"] = "AIY Voice Assistant"
    msg["X-Priority"] = "3"

    # Attach body to the email
    msg.attach(MIMEText(body, "plain"))

    # Attach files if provided
    if attachments:
        for filepath in attachments:
            try:
                with open(filepath, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)

                    # Safely encode filename for email headers
                    import urllib.parse
                    filename = os.path.basename(filepath)

                    # RFC 2231 encoding for non-ASCII filenames
                    encoded_filename = urllib.parse.quote(filename.encode('utf-8'))

                    # Add Content-Disposition with both filename and filename* parameters
                    part.add_header(
                        'Content-Disposition',
                        f"attachment; filename={filename}; filename*=UTF-8''{encoded_filename}"
                    )

                    msg.attach(part)
            except FileNotFoundError:
                logger.error(f"Attachment file not found: {filepath}")
            except Exception as e:
                logger.error(f"Error attaching file {filepath}: {str(e)}")

    try:
        # Create SMTP client and send email
        async with aiosmtplib.SMTP(hostname=smtp_server, port=smtp_port) as server:
            try:
                await server.starttls()
            except aiosmtplib.SMTPException:
                # If STARTTLS fails, assume the connection is already secure
                pass

            await server.login(username, password)
            await server.send_message(msg)
        logger.info(f"sent email about {subject}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


class SendEmailTool:
    def __init__(self, config: Config):
        self.config = config

    def tool_definition(self) -> Tool:
        return Tool(
            name="send_email_to_user",
            description="Send an email with given subject and body to the user",
            iterative=False,
            parameters=[
                ToolParameter(
                    name="subject", type="string", description="Email subject"
                ),
                ToolParameter(
                    name="body",
                    type="string",
                    description="Email body. Detailed message to convey.",
                ),
                ToolParameter(
                    name="to",
                    type="string",
                    description="Optional recipient's email address. If not provided, the default address will be used.",
                ),
                ToolParameter(
                    name="attachments",
                    type="array",
                    description="Optional list of file paths to attach to the email.",
                ),
            ],
            processor=self.do_send_email,
            required=["subject", "body"],
        )

    async def do_send_email(self, parameters: Dict[str, any]):
        logger.info(f"Sending email {parameters}")
        subject = parameters.get("subject", "")
        body = parameters.get("body", "")
        to = parameters.get("to", None)
        attachments = parameters.get("attachments", None)
        await send_email_async(subject, body, self.config, sendto=to, attachments=attachments)


async def main():
    config = Config()
    await send_email_async("hello", "4", config)


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(main())
