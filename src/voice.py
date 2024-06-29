"""
return text from speech
"""

import os
import logging
from subprocess import Popen
from typing import Optional
from aiy.leds import Leds, Color, Pattern
from aiy.voice.audio import record_file, AudioFormat
from aiy.board import Board, Button
from aiy.voice.audio import Recorder

if __name__ == '__main__':
    from config import Config
    from STTEngine import STTEngine
else:
    from .config import Config
    from .STTEngine import STTEngine

logger = logging.getLogger(__name__)

RECORDING_FILENAME = "recording.wav"
TIMEOUT_LIGHTS_OFF_SEC = 60
BREATHING_PERIOD_MS = 10000
DARK_GREEN = (0x00, 0x01, 0x00)
DARK_BLUE = (0x01, 0x00, 0x00)


class SpeechTranscriber:
    def __init__(self, button: Button, leds: Leds, stt_engine: STTEngine, config: Config):
        self.button = button
        self.leds = leds
        self.tts_engine = stt_engine
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

        AUDIO_SAMPLE_RATE_HZ = 16000
        AUDIO_FORMAT = AudioFormat(sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
                                   num_channels=1,
                                   bytes_per_sample=2)

        record_file(AUDIO_FORMAT, filename=recording_file_name, wait=wait_to_stop_recording, filetype='wav')
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


class SpeechTranscriber2:
    def __init__(self, button: Button, leds: Leds, config: Config):
        from google.oauth2 import service_account
        from google.cloud import speech

        self.button = button
        self.leds = leds
        self.button_is_pressed = False
        self.config = config
        self.language_code = self.config.get("language_code", "ru-RU")

        service_accout_file = self.config.get('service_account_file', '~/gcloud.json')
        service_accout_file = os.path.expanduser(service_accout_file)
        credentials = service_account.Credentials.from_service_account_file(service_accout_file)
        self.speech_client = speech.SpeechClient(credentials=credentials)

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

    def record_audio(self):
        self.leds.update(Leds.rgb_on(Color.GREEN))
        logger.debug('Listening...')

        def wait_to_stop_recording():
            if not self.button_is_pressed:
                return
            self.button.wait_for_release()

        AUDIO_SAMPLE_RATE_HZ = 16000
        AUDIO_FORMAT = AudioFormat(sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
                                   num_channels=1,
                                   bytes_per_sample=2)

        chunks = []
        with Recorder() as recorder:
            for chunk in recorder.record(AUDIO_FORMAT, chunk_duration_sec=0.1):
                chunks.append(chunk)
                if not self.button_is_pressed:
                    break

        self.leds.update(Leds.rgb_off())
        return chunks

    def transcribe_audio(self, chunks: list):
        from google.cloud import speech
        END_OF_SINGLE_UTTERANCE = speech.types.StreamingRecognizeResponse.END_OF_SINGLE_UTTERANCE
        AUDIO_SAMPLE_RATE_HZ = 16000

        config = speech.types.RecognitionConfig(
            encoding=speech.types.RecognitionConfig.LINEAR16,
            sample_rate_hertz=AUDIO_SAMPLE_RATE_HZ,
            language_code=self.language_code)
        streaming_config = speech.types.StreamingRecognitionConfig(config=config)

        requests = (speech.types.StreamingRecognizeRequest(audio_content=data) for data in chunks)
        responses = self.speech_client.streaming_recognize(config=streaming_config, requests=requests)

        logger.info(f"{len(chunks)} chunks; {len(list(responses))} responses")

        text = ""
        for response in responses:
            logger.info(f"Speech event type: {response.speech_event_type}")
            logger.info(f"Results: {response.results}")
            if response.speech_event_type == END_OF_SINGLE_UTTERANCE:
                for result in response.results:
                    if result.is_final:
                        text += result.alternatives[0].transcript

        self.leds.update(Leds.rgb_pattern(DARK_BLUE))
        if not text:
            logger.warning('Sorry, I did not hear you.')
        else:
            logger.debug('You said: %s', text)
        return text

    def transcribe_speech(self, player_process: Optional[Popen] = None) -> str:
        self.setup_button_callbacks()
        logger.info('Press the button and speak')
        self.wait_for_button_press()

        if player_process:
            player_process.terminate()

        text = ""
        if self.button_is_pressed:
            chunks = self.record_audio()
            text = self.transcribe_audio(chunks)
        return text
