"""
Main entry point for the AI Voice Assistant.

This script initializes the necessary components and starts the main conversation loop.
It uses the Google AIY Voice Kit for hardware interfacing and various AI services for
speech recognition, synthesis, and conversation management.
"""

import asyncio
import logging
import signal
import sys
import time

from aiy.board import Board
from aiy.leds import Leds, Color
from dotenv import load_dotenv

from src.ai_models import OpenRouterModel
from src.ai_models_with_tools import ClaudeAIModelWithTools, Tool, ToolParameter
from src.web_search_tool import WebSearchTool
from src.email_tools import SendEmailTool
from src.config import Config
from src.conversation_manager import ConversationManager
from src.dialog import main_loop_async
from src.tts_engine import YandexTTSEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up signal handling for graceful shutdown
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

# Load environment variables
load_dotenv()


def main():
    """
    Main function to initialize and run the AI Voice Assistant.
    """
    config = Config()

    with Board() as board, Leds() as leds:

        search_tool = WebSearchTool(config, leds)
        send_email_tool = SendEmailTool(config)
        tools = [search_tool.tool_definition(), send_email_tool.tool_definition()]

        # Initial LED feedback
        leds.update(Leds.rgb_on(Color.WHITE))
        time.sleep(1)
        leds.update(Leds.rgb_off())

        # Initialize components
        tts_engine = YandexTTSEngine(config)
        # ai_model = OpenRouterModel(config)
        ai_model = ClaudeAIModelWithTools(config, tools)
        conversation_manager = ConversationManager(config, ai_model)

        # Start the main conversation loop
        # main_loop(board.button, leds, tts_engine, conversation_manager, config)
        asyncio.run(main_loop_async(board.button, leds, tts_engine, conversation_manager, config))


if __name__ == '__main__':
    main()
