import logging
import openai
import os
from aiy.leds import (Leds, Pattern, PrivacyLed, RgbLeds, Color)
from aiy.voice.audio import AudioFormat, play_wav, record_file, Recorder
from openai import OpenAI
import tempfile

logger = logging.getLogger(__name__)
openai = OpenAI()
RUN_DIR = '/run/user/%d' % os.getuid()
AUDIO_SAMPLE_RATE_HZ = 24000
AUDIO_FORMAT = AudioFormat(sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
                           num_channels=1,
                           bytes_per_sample=2)


def synthesize_speech(text, filename):
    logger.debug('Synthesizing speech for: %s', text)
    openai.audio.speech.create(
        model="tts-1-hd",
        voice="onyx",
        response_format="wav",
        input=text
    ).stream_to_file(filename)
    logger.debugb (f"saved at {filename}")



def transcribe_speech(button, leds):
    logger.info('Press the button and speak')
    button.wait_for_press()
    leds.update(Leds.rgb_on(Color.GREEN))
    logger.info('Listening...')
    recording_filename = "recording.wav"
    record_file(AudioFormat.CD, filename=recording_filename, wait=button.wait_for_release, filetype='wav')
    logger.info(f"recorded {recording_filename}")

    with open(recording_filename, 'rb') as f:
        text = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    leds.update(Leds.rgb_off())
    if text:
        logger.info('You said: %s', text)
        return text
    else:
        logger.warning('Sorry, I did not hear you.')
        return None
