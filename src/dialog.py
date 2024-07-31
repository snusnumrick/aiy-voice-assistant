"""
Dialog module.

This module contains the main loop for handling the conversation flow,
including speech recognition, AI response generation, and speech synthesis.
"""

import logging
import os
import time
import traceback
import asyncio
from typing import Dict, List, Tuple

import aiohttp
from aiy.board import Button
from aiy.leds import Leds, Color

from .audio import SpeechTranscriber
from .config import Config
from .conversation_manager import ConversationManager
from .responce_player import ResponsePlayer
from .tools import time_string_ms
from .tts_engine import TTSEngine, Tone, Language

logger = logging.getLogger(__name__)


def error_visual(leds: Leds):
    """
        Display a visual error indicator using the LED.
        Blinks the LED red once to indicate an error has occurred.
    """
    logger.info("error... LED blinking")
    leds.update(Leds.rgb_on(Color.RED))
    time.sleep(0.3)
    leds.update(Leds.rgb_off())
    logger.info("error... LED off")


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


async def main_loop_async(button: Button, leds: Leds,
                          tts_engines: Dict[Language, TTSEngine], fallback_tts_engine: TTSEngine,
                          conversation_manager: ConversationManager, config: Config, timezone: str) -> None:
    """
    The main conversation loop of the AI assistant with truly interleaved AI response generation and speech synthesis.

    This function handles the flow of conversation, including:
    - Listening for user input
    - Transcribing speech to text
    - Generating AI responses and immediately synthesizing speech for each response
    - Playing all synthesized speech responses in order

    Args:
        button (Button): The AIY Kit button object.
        leds (Leds): The AIY Kit LED object for visual feedback.
        tts_engines (Dict[Language, TTSEngine]): Dictionary of TTS engines for each supported language.
        fallback_tts_engine (TTSEngine): Fallback TTS engine.
        conversation_manager (ConversationManager): The conversation manager object.
        config (Config): The application configuration object.
        timezone (str): The timezone to use for the conversation loop.
    """

    async def cleaning_routine():
        await conversation_manager.process_and_clean()

    transcriber = SpeechTranscriber(button, leds, config, cleaning=cleaning_routine, timezone=timezone)
    response_player = None
    original_audio_file_name = config.get('audio_file_name', 'speech.wav')

    async def synthesize_with_fallback(session, tts_engine, fallback_tts_engine, response_text: str,
                                       audio_file_name: str, tone: Tone, lang: Language) -> bool:
        try:
            result = await tts_engine.synthesize_async(session, response_text, audio_file_name, tone,
                                                       lang)
            if result:
                logger.debug(f"({time_string_ms(timezone)}) Synthesis {audio_file_name} completed")
                return True
            else:
                logger.warning(
                    f"Primary TTS engine failed for file: {audio_file_name}. Trying fallback engine.")
                fallback_result = await fallback_tts_engine.synthesize_async(session, response_text,
                                                                             audio_file_name, tone,
                                                                             lang)
                if fallback_result:
                    logger.info(f"Fallback TTS engine succeeded for file: {audio_file_name}")
                    return True
                else:
                    logger.error(
                        f"Both primary and fallback TTS engines failed for file: {audio_file_name}")
                    return False
        except Exception as e:
            logger.error(f"Error synthesizing speech for file {audio_file_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                conversation_manager.save_dialog()

                text = await transcriber.transcribe_speech(response_player)
                logger.info(f'({time_string_ms(timezone)}) You said: %s', text)
                response_player = None

                if text:
                    response_count = 0
                    synthesis_tasks: List[Tuple[asyncio.Task, dict]] = []
                    next_response_index = 0

                    async def process_completed_tasks():
                        nonlocal next_response_index, response_player
                        while next_response_index < len(synthesis_tasks):
                            task, response_info = synthesis_tasks[next_response_index]
                            if task.done():
                                try:
                                    result = task.result()
                                    if result:
                                        if response_player is None:
                                            response_player = ResponsePlayer([(response_info["emo"],
                                                                               response_info[
                                                                                   "audio_file_name"],
                                                                               response_info[
                                                                                   "response_text"])],
                                                                             leds, timezone)
                                            response_player.play()
                                        else:
                                            response_player.add((response_info["emo"],
                                                                 response_info["audio_file_name"],
                                                                 response_info["response_text"]))
                                    else:
                                        logger.error(
                                            f"Speech synthesis failed for file: {response_info['audio_file_name']}")
                                        error_visual(leds)
                                except Exception as e:
                                    logger.error(f"Error occurred during synthesis: {str(e)}")
                                    error_visual(leds)
                                next_response_index += 1
                            else:
                                break

                    button_pressed = False

                    def set_button_pressed():
                        nonlocal button_pressed
                        button_pressed = True

                    button.when_pressed = lambda: set_button_pressed()
                    logger.info("set button callback")

                    async for ai_response in conversation_manager.get_response(text):
                        if button_pressed:
                            logger.info("button pressed, stopping processing")
                            if response_player:
                                response_player.stop()
                            break

                        logger.info(f"ai response: {ai_response}")
                        for response in ai_response:
                            response_count += 1
                            logger.debug(f'({time_string_ms(timezone)}) AI says: {response["text"]}')

                            emo = response["emotion"]
                            response_text = response["text"]
                            lang_code = response["language"]
                            audio_file_name = append_suffix(original_audio_file_name,
                                                            str(response_count))
                            tone = Tone.PLAIN if (emo is None) or ('voice' not in emo) or (
                                    'tone' not in emo['voice']) or (emo['voice'][
                                                                        'tone'] != "happy") else Tone.HAPPY
                            lang = {"ru": Language.RUSSIAN, "en": Language.ENGLISH,
                                    "de": Language.GERMAN}.get(
                                lang_code, Language.RUSSIAN)

                            tts_engine = tts_engines.get(lang, tts_engines[Language.RUSSIAN])
                            logger.debug(f"Tone: {tone}, language = {lang}, tts_engine = {tts_engine}")

                            logger.debug(
                                f"({time_string_ms(timezone)}) Created task for synthesis {audio_file_name} "
                                f"from {response_text[:50]}")
                            synthesis_task = asyncio.create_task(
                                synthesize_with_fallback(session, tts_engine, fallback_tts_engine,
                                                         response_text, audio_file_name, tone, lang))
                            synthesis_tasks.append((synthesis_task,
                                                    {"emo": emo, "audio_file_name": audio_file_name,
                                                     "response_text": response_text}))

                            # Process any completed tasks
                            await process_completed_tasks()

                            # Yield control to allow other tasks to run
                            await asyncio.sleep(0)

                    # Wait for any remaining tasks to complete
                    while next_response_index < len(synthesis_tasks):
                        if button_pressed:
                            logger.info("button pressed (2), stopping processing")
                            if response_player:
                                response_player.stop()
                            break
                        await process_completed_tasks()
                        await asyncio.sleep(0.1)

                    # Ensure all audio has finished playing
                    while (response_player is not None) and response_player.is_playing():
                        await asyncio.sleep(0.1)

                    response_player = None

            except Exception as e:
                logger.error(f"An error occurred in the main loop: {str(e)}")
                logger.error(traceback.format_exc())
                error_visual(leds)
