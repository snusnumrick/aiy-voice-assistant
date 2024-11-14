import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict
import asyncio
import aiosmtplib

if __name__ == "__main__":
    # add current directory to python path
    import sys
    sys.path.append(os.getcwd())

from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config

logger = logging.getLogger(__name__)


def send_email(subject: str, body: str, config: Config):
    """
    Sends an email to the user with the given subject and body using the provided email configuration.

    :param subject: A string representing the subject of the email.
    :param body: A string representing the body of the email.
    :param config: A Config object containing the email configuration.

    :return: None

    """
    assistant_email_address = config.get("assistant_email_address", "cubick@treskunov.net")
    user_email_address = config.get("user_email_address", "treskunov@gmail.com")
    smtp_server = config.get("smtp_server", "mail.treskunov.net")
    smtp_port = config.get("smtp_port", "26")
    username = config.get("assistant_email_username", "cubick@treskunov.net")

    password = os.environ.get("EMAIL_PASSWORD")

    # Create message
    msg = MIMEMultipart()
    msg['From'] = assistant_email_address
    msg['To'] = user_email_address
    msg['Subject'] = subject

    # Attach body to the email
    msg.attach(MIMEText(body, 'plain'))

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


async def send_email_async(subject: str, body: str, config: Config):
    """
    Sends an email to the user asynchronously with the given subject and body using the provided email configuration.

    :param subject: A string representing the subject of the email.
    :param body: A string representing the body of the email.
    :param config: A Config object containing the email configuration.

    :return: None
    """
    assistant_email_address = config.get("assistant_email_address", "cubick@treskunov.net")
    user_email_address = config.get("user_email_address", "treskunov@gmail.com")
    smtp_server = config.get("smtp_server", "mail.treskunov.net")
    smtp_port = config.get("smtp_port", 26)  # Note: Changed to int
    username = config.get("assistant_email_username", "cubick@treskunov.net")

    password = os.environ.get("EMAIL_PASSWORD")

    # Create message
    msg = MIMEMultipart()
    msg['From'] = assistant_email_address
    msg['To'] = user_email_address
    msg['Subject'] = subject

    # Attach body to the email
    msg.attach(MIMEText(body, 'plain'))

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
        return Tool(name="send_email_to_user", description="Send an email with given subject and body to the user",
                    iterative=False,
                    parameters=[ToolParameter(name='subject', type='string',
                                              description='Email subject'),
                                ToolParameter(name='body', type='string',
                                              description='Email body. Detailed message to convey.')],
                    processor=self.do_send_email,
                    required=['subject', 'body'])

    async def do_send_email(self, parameters: Dict[str, any]):
        logger.info(f"Sending email {parameters}")
        subject = parameters.get("subject", "")
        body = parameters.get("body", "")
        await send_email_async(subject, body, self.config)


async def main():
    config = Config()
    await send_email_async("hello", "4", config)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(main())

