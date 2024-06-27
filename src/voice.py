"""
return text from speech
"""

import os
import logging
from subprocess import Popen
from typing import Optional
from abc import ABC, abstractmethod
from aiy.leds import Leds, Color, Pattern
from aiy.voice.audio import record_file, AudioFormat
from aiy.board import Board, Button

if __name__ == '__main__':
    from config import Config
else:
    from .config import Config


logger = logging.getLogger(__name__)

RECORDING_FILENAME = "recording.wav"
TIMEOUT_LIGHTS_OFF_SEC = 60
BREATHING_PERIOD_MS = 10000
DARK_GREEN = (0x00, 0x01, 0x00)
DARK_BLUE = (0x01, 0x00, 0x00)


class STTEngine(ABC):
    """
    Abstract class for Speech-to-Text Engine
    """
    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        pass


class OpenAISTTEngine(STTEngine):
    """
    Implementation of STTEngine using OpenAI API
    """
    def transcribe(self, audio_file: str) -> str:
        import openai
        with open(audio_file, 'rb') as f:
            return openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ru",
                response_format="text"
            )


class GoogleSTTEngine(STTEngine):
    """
    Implementation of STTEngine using Google Speech Recognition API
    """
    def transcribe(self, audio_file: str) -> str:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            logger.error("Could not request results from Google Speech Recognition service")
            return ""


class SpeechTranscriber:
    def __init__(self, button: Button, leds: Leds, tts_engine: STTEngine, config: Config):
        self.button = button
        self.leds = leds
        self.tts_engine = tts_engine
        self.button_is_pressed = False
        self.config = config

    def button_pressed(self):
        self.button_is_pressed = True
        logger.debug('Button pressed')

    def button_released(self):
        self.button_is_pressed = False
        logger.debug('Button released')

    def setup_button_callbacks(self):
        self.button.when_pressed = self.button_pressed
        self.button.when_released = self.button_released

    def wait_for_button_press(self):
        self.leds.pattern = Pattern.breathe(BREATHING_PERIOD_MS)
        self.leds.update(Leds.rgb_pattern(DARK_GREEN))
        self.button.wait_for_press(TIMEOUT_LIGHTS_OFF_SEC)
        self.leds.update(Leds.rgb_off())
        if not self.button_is_pressed:
            logger.debug('No button press detected during timeout. Switching off lights.')
            self.button.wait_for_press()

    def record_audio(self, recording_file_name: str):
        self.leds.update(Leds.rgb_on(Color.GREEN))
        logger.debug('Listening...')

        def wait_to_stop_recording():
            if not self.button_is_pressed:
                return
            self.button.wait_for_release()

        record_file(AudioFormat.CD, filename=recording_file_name, wait=wait_to_stop_recording, filetype='wav')
        self.leds.update(Leds.rgb_off())
        logger.debug(f"Recorded {recording_file_name}")

    def transcribe_audio(self, recording_file_name: str):
        if not os.path.exists(recording_file_name):
            logger.warning('No recording file found')
            return ""

        self.leds.update(Leds.rgb_pattern(DARK_BLUE))
        text = self.tts_engine.transcribe(recording_file_name)
        if not text:
            logger.warning('Sorry, I did not hear you.')
        else:
            logger.debug('You said: %s', text)
        return text

    def transcribe_speech(self, player_process: Optional[Popen] = None) -> str:
        recording_file_name = self.config.get('recording_file_name', 'recording.wav')

        self.setup_button_callbacks()
        logger.info('Press the button and speak')
        self.wait_for_button_press()

        if player_process:
            player_process.terminate()

        if self.button_is_pressed:
            if os.path.exists(recording_file_name):
                os.remove(recording_file_name)
            self.record_audio(recording_file_name)

        return self.transcribe_audio(recording_file_name)


def test():
    config = Config()


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    test()