"""
Dialog module.

This module contains the main loop for handling the conversation flow,
including speech recognition, AI response generation, and speech synthesis.
"""

import logging
import time
import os

from aiy.board import Button
from aiy.leds import Leds, Color
from aiy.voice.audio import play_wav_async

from .audio import SpeechTranscriber, synthesize_speech
from .config import Config
from .conversation_manager import ConversationManager
from .tts_engine import TTSEngine
from .responce_player import ResponsePlayer

logger = logging.getLogger(__name__)


def error_visual(leds: Leds):
    # blink
    leds.update(Leds.rgb_on(Color.RED))
    time.sleep(0.3)
    leds.update(Leds.rgb_off())


def append_suffix(file_name: str, suffix: str) -> str:
    base_name = os.path.basename(file_name)
    dir_name = os.path.dirname(file_name)

    # split the base_name into name without extension and extension
    name, extension = os.path.splitext(base_name)

    # create new name with suffix and original extension
    new_name = f"{name}{suffix}{extension}"

    # Join directory name and new base name to get full path
    new_path = os.path.join(dir_name, new_name)

    return new_path


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
    responce_player = None

    while True:
        try:
            # Listen and transcribe user speech
            text = transcriber.transcribe_speech(responce_player)
            logger.info('You said: %s', text)

            if text:
                # Generate AI response
                ai_response = conversation_manager.get_response(text)
                logger.info('AI says: %s', " ".join([t for e, t in ai_response]))

                if ai_response:
                    # Synthesize and play AI response
                    audio_file_name = config.get('audio_file_name', 'speech.wav')
                    # response_text = " ".join([t for e, t in ai_response])

                    playlist = []
                    for n, (emo, response_text) in enumerate(ai_response):
                        while True:

                            if synthesize_speech(tts_engine, response_text, audio_file_name, config):
                                break

                            # error happened, retry
                            error_visual(leds)

                            # retry
                            continue

                        playlist.append((emo, audio_file_name))
                        audio_file_name = append_suffix(audio_file_name, str(n+1))

                    responce_player = ResponsePlayer(playlist)
                    responce_player.play()
                    # player_process = play_wav_async(audio_file_name)

        except Exception as e:
            logger.error(f"An error occurred in the main loop: {str(e)}",
                         exc_info=True)  # Implement appropriate error handling and recovery here
            error_visual(leds)
