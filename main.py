"""
Main entry point for the AI Voice Assistant.

This script initializes the necessary components and starts the main conversation loop.
It uses the Google AIY Voice Kit for hardware interfacing and various AI services for
speech recognition, synthesis, and conversation management.
"""

import logging
import signal
import sys
import time
from dotenv import load_dotenv
from aiy.board import Board
from aiy.leds import Leds, Color

from src.config import Config
from src.stt_engine import OpenAISTTEngine
from src.tts_engine import OpenAITTSEngine, YandexTTSEngine
from src.ai_models import OpenAIModel, ClaudeAIModel, OpenRouterModel
from src.conversation_manager import ConversationManager
from src.dialog import main_loop

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
        # Initial LED feedback
        leds.update(Leds.rgb_on(Color.WHITE))
        time.sleep(1)
        leds.update(Leds.rgb_off())

        # Initialize components
        tts_engine = YandexTTSEngine(config)
        ai_model = OpenRouterModel(config)
        conversation_manager = ConversationManager(config, ai_model)

        # Start the main conversation loop
        main_loop(board.button, leds, tts_engine, conversation_manager, config)


if __name__ == '__main__':
    main()
