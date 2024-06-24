import logging
import openai
import os
from aiy.leds import (Leds, Pattern, PrivacyLed, RgbLeds, Color)
from aiy.voice.audio import AudioFormat, Recorder, BytesPlayer
from openai import OpenAI
import tempfile

logger = logging.getLogger(__name__)
openai = OpenAI()
RUN_DIR = '/run/user/%d' % os.getuid()
AUDIO_SAMPLE_RATE_HZ = 16000
AUDIO_FORMAT = AudioFormat(sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
                           num_channels=1,
                           bytes_per_sample=2)


def synthesize_speech(text):
    logger.info('Synthesizing speech for: %s', text)
    with tempfile.NamedTemporaryFile(suffix='.wav', dir=RUN_DIR) as f, \
            openai.audio.speech.with_streaming_response.create(
                model="tts-1-hd",
                voice="onyx",
                input=text
            ) as response, \
            BytesPlayer() as player:
        play = player.play(AUDIO_FORMAT)
        for data in response.iter_bytes:
            logger.info("playing {data}")
            play(data)
        play(None)


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
