"""
Dialog module.

This module contains the main loop for handling the conversation flow,
including speech recognition, AI response generation, and speech synthesis.
"""

import logging
import time

from aiy.board import Button
from aiy.leds import Leds, Color
from aiy.voice.audio import play_wav_async

from .audio import SpeechTranscriber, synthesize_speech
from .config import Config
from .conversation_manager import ConversationManager
from .tts_engine import TTSEngine

logger = logging.getLogger(__name__)


def error_visual(leds: Leds):
    # blink
    leds.update(Leds.rgb_on(Color.RED))
    time.sleep(0.3)
    leds.update(Leds.rgb_off())


def main_loop(button: Button, leds: Leds, tts_engine: TTSEngine, conversation_manager: ConversationManager,
              config: Config) -> None:
    """
    The main conversation loop of the AI assistant.

    This function handles the flow of conversation, including:
    - Listening for user input
    - Transcribing speech to text
    - Generating AI responses
    - Synthesizing speech from the AI responses
    - Playing the synthesized speech

    Args:
        button (Button): The AIY Kit button object.
        leds (Leds): The AIY Kit LED object for visual feedback.
        stt_engine: The speech-to-text engine.
        tts_engine (TTSEngine): The text-to-speech engine.
        conversation_manager (ConversationManager): The conversation manager object.
        config (Config): The application configuration object.
    """
    transcriber = SpeechTranscriber(button, leds, config)
    player_process = None

    while True:
        try:
            # Listen and transcribe user speech
            text = transcriber.transcribe_speech(player_process)
            logger.info('You said: %s', text)

            if text:
                # Generate AI response
                ai_response = conversation_manager.get_response(text)
                logger.info('AI says: %s', " ".join([t for e, t in ai_response]))

                if ai_response:
                    # Synthesize and play AI response
                    audio_file_name = config.get('audio_file_name', 'speech.wav')
                    response_text = " ".join([t for e, t in ai_response])

                    while True:

                        if synthesize_speech(tts_engine, response_text, audio_file_name, config):
                            break
                        # error happened, retry

                        error_visual(leds)

                        # retry
                        continue

                    logger.info(f"Playing audio file: {audio_file_name}")
                    player_process = play_wav_async(audio_file_name)

        except Exception as e:
            logger.error(f"An error occurred in the main loop: {str(e)}",
                         exc_info=True)  # Implement appropriate error handling and recovery here
            error_visual(leds)
