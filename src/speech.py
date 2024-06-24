import logging
import openai
import os
from aiy.leds import (Leds, Pattern, PrivacyLed, RgbLeds, Color)


logger = logging.getLogger(__name__)


def synthesize_speech(text):
    logger.info('Synthesizing speech for: %s', text)


def transcribe_speech(button, leds):
    logger.info('Press the button and speak')
    button.wait_for_press()
    leds.update(Leds.rgb_on(Color.GREEN))

    logger.info('Listening...')
    # text = recognizer.recognize()
    text = "привет"
    leds.update(Leds.rgb_off())
    if text:
        logger.info('You said: %s', text)
        return text
    else:
        logger.warning('Sorry, I did not hear you.')
        return None
