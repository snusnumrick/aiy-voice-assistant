"""
Dialog module.

This module contains the main loop for handling the conversation flow,
including speech recognition, AI response generation, and speech synthesis.
"""

import asyncio
import logging
import os
import time

from aiy.board import Button
from aiy.leds import Leds, Color

from .audio import SpeechTranscriber, synthesize_speech, synthesize_speech_async
from .config import Config
from .conversation_manager import ConversationManager
from .responce_player import ResponsePlayer
from .tts_engine import TTSEngine

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

    original_audio_file_name = config.get('audio_file_name', 'speech.wav')

    while True:
        try:
            # Listen and transcribe user speech
            text = transcriber.transcribe_speech(responce_player)
            logger.info('You said: %s', text)

            if text:
                # Generate AI response
                ai_response = conversation_manager.get_response(text)
                logger.debug('AI response: %s', ai_response)
                logger.info('AI says: %s', " ".join([t for e, t in ai_response]))

                if ai_response:
                    # Synthesize and play AI response
                    audio_file_name = original_audio_file_name

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
                        audio_file_name = append_suffix(original_audio_file_name, str(n + 1))

                    responce_player = ResponsePlayer(playlist, leds)
                    responce_player.play()  # player_process = play_wav_async(audio_file_name)

        except Exception as e:
            logger.error(f"An error occurred in the main loop: {str(e)}",
                         exc_info=True)  # Implement appropriate error handling and recovery here
            error_visual(leds)


async def main_loop_async(button: Button, leds: Leds, tts_engine: TTSEngine, conversation_manager: ConversationManager,
                          config: Config) -> None:
    transcriber = SpeechTranscriber(button, leds, config)
    response_player = None
    original_audio_file_name = config.get('audio_file_name', 'speech.wav')

    while True:
        try:
            text = transcriber.transcribe_speech(response_player)
            logger.info('You said: %s', text)

            if text:
                ai_response = conversation_manager.get_response(text)
                logger.debug('AI response: %s', ai_response)
                logger.info('AI says: %s', " ".join([t for e, t in ai_response]))

                if ai_response:
                    tasks = []
                    for n, (emo, response_text) in enumerate(ai_response):
                        audio_file_name = append_suffix(original_audio_file_name, str(n + 1))
                        task = asyncio.create_task(
                            synthesize_speech_async(tts_engine, response_text, audio_file_name, config))
                        tasks.append((emo, audio_file_name, task))

                    results = [None] * len(tasks)
                    for i, (emo, audio_file_name, task) in enumerate(tasks):
                        try:
                            success = await task
                            if success:
                                results[i] = (emo, audio_file_name)
                            else:
                                logger.error(f"Failed to synthesize speech for file: {audio_file_name}")
                                error_visual(leds)
                        except Exception as e:
                            logger.error(f"Error synthesizing speech for file {audio_file_name}: {str(e)}")
                            error_visual(leds)

                    # Remove any None entries from the results (in case of errors)
                    playlist = [item for item in results if item is not None]

                    response_player = ResponsePlayer(playlist, leds)
                    response_player.play()

        except Exception as e:
            logger.error(f"An error occurred in the main loop: {str(e)}", exc_info=True)
            error_visual(leds)
