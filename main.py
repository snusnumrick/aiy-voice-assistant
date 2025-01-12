"""
Main entry point for the AI Voice Assistant.

This script initializes the necessary components and starts the main conversation loop.
It uses the Google AIY Voice Kit for hardware interfacing and various AI services for
speech recognition, synthesis, and conversation management.
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from aiy.board import Board
from aiy.leds import Leds, Color
from dotenv import load_dotenv

from src.ai_models_with_tools import ClaudeAIModelWithTools, OpenAIModelWithTools
from src.config import Config
from src.conversation_manager import ConversationManager
from src.dialog import main_loop_async
from src.email_tools import SendEmailTool
from src.weather_tool import EnhancedWeatherTool
from src.tools import get_timezone
from src.tts_engine import YandexTTSEngine, Language, ElevenLabsTTSEngine
from src.web_search_tool import WebSearchTool
from src.stress_tool import StressTool
from src.code_interpreter_tool import InterpreterTool
from src.volume_control_tool import VolumeControlTool

# Set up signal handling for graceful shutdown
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

# Load environment variables
load_dotenv()


def setup_logging(log_level, log_dir=None):
    """
    Set up logging with the specified log level and daily log files.
    Keeps only the last 5 days of logs.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # If log_dir is specified, set up daily rotating file handler
    if log_dir:
        # Create log directory if it doesn't exist
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Set up file handler for daily rotation
        log_file = log_dir_path / 'assistant.log'
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',
            interval=1,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="AI Voice Assistant")
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level (default: INFO)')
    parser.add_argument('--log-dir', default='logs',
                       help='Directory for storing log files (default: logs)')
    return parser.parse_args()

def main():
    """
    Main function to initialize and run the AI Voice Assistant.
    """
    args = parse_arguments()
    logger = setup_logging(args.log_level, args.log_dir)

    logger.info("Starting AI Voice Assistant")

    config = Config()
    print(config.dict())
    timezone = get_timezone()

    with Board() as board, Leds() as leds:

        search_tool = WebSearchTool(config)
        stress_tool = StressTool(config)
        send_email_tool = SendEmailTool(config)
        interpreter_tool = InterpreterTool(config)
        volume_control_tool = VolumeControlTool(config)
        weather_tool = EnhancedWeatherTool(config)

        tools = [search_tool.tool_definition(), send_email_tool.tool_definition(), stress_tool.tool_definition(),
                 interpreter_tool.tool_definition(), volume_control_tool.tool_definition(),
                 weather_tool.tool_definition()]

        # Initial LED feedback
        leds.update(Leds.rgb_on(Color.WHITE))
        time.sleep(1)
        leds.update(Leds.rgb_off())

        # Initialize components
        elevenlabs_engine = None
        try:
            elevenlabs_engine = ElevenLabsTTSEngine(config)
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs: {e}")
        yandex_engine = None
        try:
            yandex_engine = YandexTTSEngine(config, timezone=timezone)
        except Exception as e:
            logger.error(f"Failed to initialize Yandex: {e}")
        if not elevenlabs_engine and not yandex_engine:
            logger.critical("No TTS engine available. Exiting.")
            return
        tts_engines = {Language.RUSSIAN: yandex_engine if yandex_engine else elevenlabs_engine,
                       Language.ENGLISH: elevenlabs_engine if elevenlabs_engine else yandex_engine,
                       Language.GERMAN: elevenlabs_engine if elevenlabs_engine else yandex_engine}
        fallback_tts_engine = yandex_engine if yandex_engine else elevenlabs_engine
        # ai_model = OpenRouterModel(config)
        ai_model = ClaudeAIModelWithTools(config, tools=tools, timezone=timezone)
        # ai_model = OpenAIModelWithTools(config, tools=tools)
        conversation_manager = ConversationManager(config, ai_model, timezone)

        logger.info("All components initialized. Starting main conversation loop.")

        # Start the main conversation loop
        asyncio.run(main_loop_async(board.button, leds, tts_engines, fallback_tts_engine, conversation_manager, config,
                                    timezone))


if __name__ == '__main__':
    main()
