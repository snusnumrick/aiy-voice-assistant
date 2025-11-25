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

from dotenv import load_dotenv

from aiy.board import Board
from aiy.leds import Color, Leds
from src.ai_models_with_tools import ClaudeAIModelWithTools, OpenAIModelWithTools
from src.responce_player import ResponsePlayer
from src.code_interpreter_tool import InterpreterTool
from src.config import Config
from src.conversation_manager import ConversationManager
from src.dialog import main_loop_async
from src.email_tools import SendEmailTool
from src.weather_tool import EnhancedWeatherTool
from src.stress_tool import StressTool
from src.tools import get_timezone
from src.tts_engine import ElevenLabsTTSEngine, Language, YandexTTSEngine
from src.volume_control_tool import VolumeControlTool
from src.web_search_tool import WebSearchTool
from src.wizard_tool import WizardTool
from src.minimax_music_tool import MiniMaxMusicTool

# Set up signal handling for graceful shutdown
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

# Load environment variables
load_dotenv()


def setup_logging(log_level, log_dir=None):
    """
    Set up logging with the specified log level and daily log files.
    Keeps only the last 5 days of logs.
    Automatically checks for errors before rotation and emails notifications.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # If log_dir is specified, set up daily rotating file handler
    if log_dir:
        # Create log directory if it doesn't exist
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Set up file handler for daily rotation
        log_file = log_dir_path / "assistant.log"
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info("Log rotation configured with error notification enabled")

    return logger


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="AI Voice Assistant")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for storing log files (default: logs)",
    )
    return parser.parse_args()


def main():
    """
    Main function to initialize and run the AI Voice Assistant.
    """
    args = parse_arguments()
    logger = setup_logging(args.log_level, args.log_dir)

    logger.info("Starting AI Voice Assistant")

    config = Config()
    print(config.model_dump())
    timezone = get_timezone()
    # System timezone is now set by run.sh
    use_claude_search = config.get("claude_use_search", False)

    with Board() as board, Leds() as leds:
        # Decide whether to use our custom WebSearchTool or the provider's built-in
        ai_model_api = config.get("ai_model_api", "claude")
        use_openai_search = config.get("openai_use_search", False) if ai_model_api == "openai" else False
        use_builtin_search = use_claude_search if ai_model_api == "claude" else use_openai_search
        if not use_builtin_search:
            search_tool = WebSearchTool(config)
        stress_tool = StressTool(config)
        send_email_tool = SendEmailTool(config)
        interpreter_tool = InterpreterTool(config)
        volume_control_tool = VolumeControlTool(config)
        weather_tool = EnhancedWeatherTool(config)
        wizard_tool = WizardTool(config)

        tools = [
            send_email_tool.tool_definition(),
            stress_tool.tool_definition(),
            interpreter_tool.tool_definition(),
            volume_control_tool.tool_definition(),
            weather_tool.tool_definition(),
        ] + wizard_tool.tool_definitions()
        if not use_builtin_search:
            tools.append(search_tool.tool_definition())

        # Initial LED feedback
        leds.update(Leds.rgb_on(Color.WHITE))
        time.sleep(1)
        leds.update(Leds.rgb_off())

        response_player = ResponsePlayer([], leds, timezone)

        # Create MiniMax music tool (requires response_player and button_state)
        minimax_tool = MiniMaxMusicTool(
            config=config,
            response_player=response_player
        )

        # Add MiniMax music tool to tools list
        tools.append(minimax_tool.tool_definition())

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
        tts_engines = {
            Language.RUSSIAN: yandex_engine if yandex_engine else elevenlabs_engine,
            Language.ENGLISH: elevenlabs_engine if elevenlabs_engine else yandex_engine,
            Language.GERMAN: elevenlabs_engine if elevenlabs_engine else yandex_engine,
        }
        fallback_tts_engine = yandex_engine if yandex_engine else elevenlabs_engine

        ai_model = None
        ai_model_api = config.get("ai_model_api", "claude")
        if  ai_model_api == "openai":
            logger.info("Using OpenAI model")
            ai_model = OpenAIModelWithTools(config, tools=tools)
        if not ai_model:
            logger.info("Using Claude model")
            ai_model = ClaudeAIModelWithTools(config, tools=tools, timezone=timezone)

        conversation_manager = ConversationManager(config, ai_model, timezone, enabled_tools=tools)

        logger.info("All components initialized. Starting main conversation loop.")

        # Start the main conversation loop
        asyncio.run(
            main_loop_async(
                board.button,
                leds,
                tts_engines,
                fallback_tts_engine,
                conversation_manager,
                config,
                timezone,
                response_player,
            )
        )


if __name__ == "__main__":
    main()
