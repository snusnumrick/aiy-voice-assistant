import logging
import signal
import sys
from dotenv import load_dotenv
from aiy.board import Board
from aiy.leds import (Leds, Pattern, PrivacyLed, RgbLeds, Color)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

from src.speech import transcribe_speech, synthesize_speech
from src.openai_interaction import get_openai_response


def main():
    load_dotenv()
    with Board() as board:
        with Leds() as leds:

            button = board.button

            while True:
                text = transcribe_speech(button, leds)
                if text:
                    ai_response = get_openai_response(text)
                    logger.info('AI says: %s', ai_response)

                    speech_audio_file = synthesize_speech(ai_response)
                    aiy.audio.play_wave(speech_audio_file)


if __name__ == '__main__':
    main()
