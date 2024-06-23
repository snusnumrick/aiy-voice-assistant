import logging
from google.cloud import texttospeech
from google.cloud import speech
import aiy.audio
import aiy.cloudspeech

logger = logging.getLogger(__name__)
tts_client = texttospeech.TextToSpeechClient()
speech_client = speech.SpeechClient()
recognizer = aiy.cloudspeech.get_recognizer()


def synthesize_speech(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    output_file = 'output.wav'
    with open(output_file, 'wb') as out:
        out.write(response.audio_content)
        logger.info('Synthesized speech saved to %s', output_file)
    return output_file


def transcribe_speech(button, led):
    recognizer.expect_phrase('Hello')
    logger.info('Press the button and speak')
    button.wait_for_press()
    led_on(led)
    logger.info('Listening...')
    text = recognizer.recognize()
    led_off(led)
    if text:
        logger.info('You said: %s', text)
        return text
    else:
        logger.warning('Sorry, I did not hear you.')
        return None
