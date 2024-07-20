"""
Dialog module.

This module contains the main loop for handling the conversation flow,
including speech recognition, AI response generation, and speech synthesis.
"""

import asyncio
import logging
import os
import time
import traceback

import aiohttp

from aiy.board import Button
from aiy.leds import Leds, Color

from .audio import SpeechTranscriber, synthesize_speech, synthesize_speech_async
from .config import Config
from .conversation_manager import ConversationManager
from .responce_player import ResponsePlayer
from .tts_engine import TTSEngine, Tone

logger = logging.getLogger(__name__)


def error_visual(leds: Leds):
    """
        Display a visual error indicator using the LED.
        Blinks the LED red once to indicate an error has occurred.
    """
    leds.update(Leds.rgb_on(Color.RED))
    time.sleep(0.3)
    leds.update(Leds.rgb_off())


def append_suffix(file_name: str, suffix: str) -> str:
    """
        Append a suffix to a file name while preserving its extension and directory path.

        Args:
            file_name (str): The original file name (can include path).
            suffix (str): The suffix to append before the file extension.

        Returns:
            str: The new file name with the suffix appended.
    """
    base_name = os.path.basename(file_name)
    dir_name = os.path.dirname(file_name)

    # split the base_name into name without extension and extension
    name, extension = os.path.splitext(base_name)

    # create new name with suffix and original extension
    new_name = f"{name}{suffix}{extension}"

    # Join directory name and new base name to get full path
    new_path = os.path.join(dir_name, new_name)

    return new_path


async def main_loop_async(button: Button, leds: Leds, tts_engine: TTSEngine, conversation_manager: ConversationManager,
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

    async def cleaning_routine():
        logger.info('Cleaning up...')
        await conversation_manager.process_and_clean()

    # Initialize components
    transcriber = SpeechTranscriber(button, leds, config, cleaning=cleaning_routine)
    response_player = None
    original_audio_file_name = config.get('audio_file_name', 'speech.wav')

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # for viz purposes
                conversation_manager.save_dialog()

                # Step 1: Listen and transcribe user speech
                text = await transcriber.transcribe_speech(response_player)
                logger.info('You said: %s', text)

                if text:
                    # Step 2: Generate AI response
                    async for ai_response in conversation_manager.get_response(text):

                        logger.debug('AI response: %s', ai_response)
                        logger.info('AI says: %s', " ".join([t for e, t in ai_response]))

                        if ai_response:
                            # Step 3: Asynchronously synthesize speech for each response
                            tasks = []
                            for n, (emo, response_text) in enumerate(ai_response):
                                audio_file_name = append_suffix(original_audio_file_name, str(n+1))
                                tone = Tone.PLAIN
                                if 'voice' in emo and 'tone' in emo['voice'] and emo['voice']['tone'] == "happy":
                                    tone = Tone.HAPPY
                                logger.debug('Tone: %s', tone)
                                task = asyncio.create_task(tts_engine.synthesize_async(session, response_text,
                                                                                       audio_file_name, tone))
                                tasks.append((emo, audio_file_name, task))

                            # Step 4: Wait for all synthesis tasks to complete and build playlist
                            playlist = []
                            for emo, audio_file_name, task in tasks:
                                try:
                                    result = await task
                                    logger.debug(f"Synthesis result for {audio_file_name}: {result}")
                                    if isinstance(result, bool):
                                        if result:
                                            playlist.append((emo, audio_file_name))
                                        else:
                                            logger.error(f"Speech synthesis failed for file: {audio_file_name}")
                                            error_visual(leds)
                                    else:
                                        logger.error(f"Unexpected result type from synthesize_async: {type(result)}")
                                        error_visual(leds)
                                except Exception as e:
                                    logger.error(f"Error synthesizing speech for file {audio_file_name}: {str(e)}")
                                    logger.error(traceback.format_exc())
                                    error_visual(leds)

                            # Step 5: Play synthesized responses
                            if playlist:
                                if response_player:
                                    response_player.stop()
                                    time.sleep(0.5)
                                response_player = ResponsePlayer(playlist, leds)
                                response_player.play()

            except Exception as e:
                logger.error(f"An error occurred in the main loop: {str(e)}")
                logger.error(traceback.format_exc())
                error_visual(leds)
