from aiy.board import Button
from aiy.leds import Leds
from .voice import STTEngine, SpeechTranscriber
from .config import Config
from .openai_interaction import get_openai_response
from .speech import synthesize_speech
from aiy.voice.audio import play_wav_async

import os
import logging

logger = logging.getLogger(__name__)


def main_loop(button: Button, leds: Leds, sst_engine: STTEngine, config: Config) -> None:
    transcriber = SpeechTranscriber(button, leds, sst_engine, config)
    player_process = None

    while True:
        try:
            text = transcriber.transcribe_speech(player_process)
            if text:
                ai_response = get_openai_response(text)
                logger.info('AI says: %s', ai_response)

                audio_file_name = config.get('audio_file_name', 'speech.wav')
                logger.info('Synthesizing speech to: %s', audio_file_name)
                synthesize_speech(ai_response, audio_file_name)
                player_process = play_wav_async(audio_file_name)

                # Cleanup
                if config.get('cleanup_audio_files', True):
                    os.remove(audio_file_name)
                    os.remove(config.get('recording_file_name', 'recording.wav'))

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
