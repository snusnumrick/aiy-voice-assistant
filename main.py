import logging
import signal
import sys
from dotenv import load_dotenv
from aiy.board import Board
from aiy.leds import (Leds, Pattern, PrivacyLed, RgbLeds, Color)
from aiy.voice.audio import FilePlayer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
load_dotenv()

from src.speech import transcribe_speech, synthesize_speech
from src.openai_interaction import get_openai_response


def main():
    with Board() as board:
        with Leds() as leds:

            button = board.button
            player = FilePlayer()

            while True:
                text = transcribe_speech(button, leds)
                if text:
                    ai_response = get_openai_response(text)
                    logger.info('AI says: %s', ai_response)

                    audio_file_name = "speech.wav"
                    synthesize_speech(ai_response, audio_file_name)
                    player.play_wav(audio_file_name)


if __name__ == '__main__':
    main()
