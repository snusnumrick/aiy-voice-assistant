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
    logger.debug(f"saved at {filename}")


def transcribe_speech(button, leds):
    timeout2off_lights_sec = 5
    button_was_pressed = False
    recording_filename = "recording.wav"

    # remove recording file if it exists
    if os.path.exists(recording_filename):
        os.remove(recording_filename)

    def button_pressed():
        # declare button_was_pressed as a nonlocal variable
        nonlocal button_was_pressed

        leds.update(Leds.rgb_on(Color.GREEN))
        logger.info('Listening...')
        record_file(AudioFormat.CD, filename=recording_filename, wait=button.wait_for_release, filetype='wav')
        logger.info(f"recorded {recording_filename}")
        button_was_pressed = True

    # Set the function to be called when the button is pressed
    button.when_pressed = button_pressed
    logger.info('Press the button and speak')
    button.wait_for_press(timeout2off_lights_sec)
    leds.update(Leds.rgb_off())
    if not button_was_pressed:
        # if the button was not pressed, continue waiting with lights off
        button.wait_for_press()

    # check if recording file exists
    text = ""
    if not os.path.exists(recording_filename):
        logger.warning('No recording file found')
    else:
        with open(recording_filename, 'rb') as f:
            text = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        if not text:
            logger.warning('Sorry, I did not hear you.')
        else:
            logger.info('You said: %s', text)
    return text
