import logging
import signal
import sys
import time
from dotenv import load_dotenv
from aiy.board import Board, Button
from aiy.leds import Leds, Color

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
load_dotenv()

from src.STTEngine import OpenAISTTEngine
from src.dialog import main_loop
from src.config import Config
from src.speech import OpenAITTSEngine


def main():

    from aiy.cloudspeech import CloudSpeechClient
    client = CloudSpeechClient("~/assistant.json")
    while True:
        logging.info('Say something.')
        text = client.recognize(language_code="ru-RU")
        if text is None:
            logging.info('You said nothing.')
            continue
        logging.info('You said: "%s"' % text)

    config = Config()
    with Board() as board, Leds() as leds:

        leds.update(Leds.rgb_on(Color.WHITE))
        time.sleep(1)
        leds.update(Leds.rgb_off())

        main_loop(board.button, leds, OpenAISTTEngine(), OpenAITTSEngine(), config)


if __name__ == '__main__':
    main()
