from aiy.board import Button
from aiy.leds import Leds
from .voice import STTEngine, SpeechTranscriber2
from .config import Config
from .speech import synthesize_speech, TTSEngine
from .conversation_manager import ConversationManager, OpenAIModel
from aiy.voice.audio import play_wav_async

import os
import logging

logger = logging.getLogger(__name__)


def main_loop(button: Button, leds: Leds, sst_engine: STTEngine, tts_engine: TTSEngine, config: Config) -> None:
    # transcriber = SpeechTranscriber(button, leds, sst_engine, config)
    transcriber = SpeechTranscriber2(button, leds, config)
    player_process = None
    conversation_manager = ConversationManager(config, OpenAIModel(config))

    while True:
        try:
            text = transcriber.transcribe_speech(player_process)
            logger.info('You said: %s', text)
            if text:
                ai_response = conversation_manager.get_response(text)
                logger.info('AI says: %s', ai_response)

                audio_file_name = config.get('audio_file_name', 'speech.wav')
                synthesize_speech(tts_engine, ai_response, audio_file_name, config)
                player_process = play_wav_async(audio_file_name)

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
