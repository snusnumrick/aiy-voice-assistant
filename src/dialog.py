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
from typing import Dict

import aiohttp
from aiy.board import Button
from aiy.leds import Leds, Color

from .audio import SpeechTranscriber
from .config import Config
from .conversation_manager import ConversationManager
from .responce_player import ResponsePlayer
from .tts_engine import TTSEngine, Tone, Language
from .tools import time_string_ms

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


async def main_loop_async(button: Button, leds: Leds, tts_engines: Dict[Language, TTSEngine],
                          conversation_manager: ConversationManager,
                          config: Config, timezone: str) -> None:
    """
    The main conversation loop of the AI assistant with interleaved AI response and speech synthesis.

    This function handles the flow of conversation, including:
    - Listening for user input
    - Transcribing speech to text
    - Generating AI responses and immediately synthesizing speech for each response
    - Playing all synthesized speech responses together

    Args:
        button (Button): The AIY Kit button object.
        leds (Leds): The AIY Kit LED object for visual feedback.
        tts_engines (Dict[Language, TTSEngine]): Dictionary of TTS engines for each supported language.
        conversation_manager (ConversationManager): The conversation manager object.
        config (Config): The application configuration object.
        timezone (str): The timezone to use for the conversation loop.
    """

    async def cleaning_routine():
        await conversation_manager.process_and_clean()

    # Initialize components
    transcriber = SpeechTranscriber(button, leds, config, cleaning=cleaning_routine, timezone=timezone)
    response_player = None
    original_audio_file_name = config.get('audio_file_name', 'speech.wav')

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # for viz purposes
                conversation_manager.save_dialog()

                # Step 1: Listen and transcribe user speech
                text = await transcriber.transcribe_speech(response_player)
                logger.info(f'({time_string_ms(timezone)}) You said: %s', text)

                if text:
                    # Step 2 & 3: Generate AI responses, synthesize speech, and build playlist
                    playlist = []
                    response_count = 0
                    synthesis_tasks = []

                    async for ai_response in conversation_manager.get_response(text):
                        for response in ai_response:
                            response_count += 1
                            logger.info(f'({time_string_ms(timezone)}) AI response: {response["text"]}')

                            emo = response["emotion"]
                            response_text = response["text"]
                            lang_code = response["language"]
                            audio_file_name = append_suffix(original_audio_file_name, str(response_count))
                            tone = Tone.PLAIN if 'voice' not in emo or 'tone' not in emo['voice'] or emo['voice'][
                                'tone'] != "happy" else Tone.HAPPY
                            lang = {"ru": Language.RUSSIAN, "en": Language.ENGLISH, "de": Language.GERMAN}.get(
                                lang_code, Language.RUSSIAN)

                            # Select the appropriate TTS engine based on language
                            tts_engine = tts_engines.get(lang, tts_engines[Language.RUSSIAN])
                            logger.debug(f"Tone: {tone}, language = {lang}, tts_engine = {tts_engine}")

                            # Start speech synthesis
                            synthesis_task = asyncio.create_task(
                                tts_engine.synthesize_async(session, response_text, audio_file_name, tone, lang))
                            synthesis_tasks.append((emo, audio_file_name, synthesis_task))

                        # Process completed synthesis tasks
                        done, pending = await asyncio.wait(
                            [task for _, _, task in synthesis_tasks],
                            timeout=0.1,
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        for completed_task in done:
                            for i, (emo, audio_file_name, task) in enumerate(synthesis_tasks):
                                if task == completed_task:
                                    try:
                                        result = await task
                                        logger.info(
                                            f"({time_string_ms(timezone)}) Synthesis result for {audio_file_name}: {result}")
                                        if result:
                                            playlist.append((emo, audio_file_name))
                                        else:
                                            logger.error(f"Speech synthesis failed for file: {audio_file_name}")
                                            error_visual(leds)
                                    except Exception as e:
                                        logger.error(f"Error synthesizing speech for file {audio_file_name}: {str(e)}")
                                        logger.error(traceback.format_exc())
                                        error_visual(leds)
                                    synthesis_tasks.pop(i)
                                    break

                    # Wait for any remaining synthesis tasks
                    if synthesis_tasks:
                        done, _ = await asyncio.wait([task for _, _, task in synthesis_tasks])
                        for completed_task in done:
                            for emo, audio_file_name, task in synthesis_tasks:
                                if task == completed_task:
                                    try:
                                        result = await task
                                        logger.info(
                                            f"({time_string_ms(timezone)}) Synthesis result for {audio_file_name}: {result}")
                                        if result:
                                            playlist.append((emo, audio_file_name))
                                        else:
                                            logger.error(f"Speech synthesis failed for file: {audio_file_name}")
                                            error_visual(leds)
                                    except Exception as e:
                                        logger.error(f"Error synthesizing speech for file {audio_file_name}: {str(e)}")
                                        logger.error(traceback.format_exc())
                                        error_visual(leds)

                    # Step 4: Play all synthesized responses
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