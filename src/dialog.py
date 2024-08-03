"""
Dialog module.

This module contains the main loop for handling the conversation flow,
including speech recognition, AI response generation, and speech synthesis.
It defines the DialogManager class which orchestrates the entire conversation
process, managing the interaction between the user input, AI response, and
speech synthesis.

Classes:
    DialogManager: Manages the conversation flow and speech synthesis.

Functions:
    error_visual: Displays a visual error indicator using the LED.
    append_suffix: Appends a suffix to a file name while preserving its extension.
    synthesize_with_fallback: Attempts to synthesize speech with a fallback option.
    main_loop_async: Entry point for the conversation loop.

Dependencies:
    - aiohttp for asynchronous HTTP requests
    - AIY libraries for hardware interaction (Button, LEDs)
    - Custom modules for speech transcription, conversation management, and TTS
"""

import asyncio
import logging
import os
import time
import traceback
from typing import Dict, List, Tuple

import aiohttp
from aiy.board import Button
from aiy.leds import Leds, Color

from .audio import SpeechTranscriber
from .config import Config
from .conversation_manager import ConversationManager
from .responce_player import ResponsePlayer
from .tools import time_string_ms, save_to_conversation
from .tts_engine import TTSEngine, Tone, Language

logger = logging.getLogger(__name__)


def error_visual(leds: Leds) -> None:
    """
    Display a visual error indicator using the LED.

    Args:
        leds (Leds): The AIY Kit LED object for visual feedback.
    """
    logger.info("Error... LED blinking")
    leds.update(Leds.rgb_on(Color.RED))
    time.sleep(0.3)
    leds.update(Leds.rgb_off())
    logger.info("Error... LED off")


def append_suffix(file_name: str, suffix: str) -> str:
    """
    Append a suffix to a file name while preserving its extension and directory path.

    Args:
        file_name (str): The original file name (can include path).
        suffix (str): The suffix to append before the file extension.

    Returns:
        str: The new file name with the suffix appended.
    """
    dir_name, base_name = os.path.split(file_name)
    name, extension = os.path.splitext(base_name)
    new_name = f"{name}{suffix}{extension}"
    return os.path.join(dir_name, new_name)


async def synthesize_with_fallback(session: aiohttp.ClientSession, tts_engine: TTSEngine,
                                   fallback_tts_engine: TTSEngine, response_text: str, audio_file_name: str, tone: Tone,
                                   lang: Language) -> bool:
    """
    Attempt to synthesize speech, falling back to a secondary engine if necessary.

    Args:
        session (aiohttp.ClientSession): The aiohttp session for making requests.
        tts_engine (TTSEngine): The primary TTS engine to use.
        fallback_tts_engine (TTSEngine): The fallback TTS engine to use if the primary fails.
        response_text (str): The text to synthesize.
        audio_file_name (str): The name of the audio file to save the synthesized speech.
        tone (Tone): The tone to use for speech synthesis.
        lang (Language): The language to use for speech synthesis.

    Returns:
        bool: True if synthesis was successful, False otherwise.
    """
    logger.info(f"synthesize_with_fallback. text: {response_text}, file: {audio_file_name}, tone: {tone}, lang: {lang}")
    try:
        if await tts_engine.synthesize_async(session, response_text, audio_file_name, tone, lang):
            logger.debug(f"Synthesis {audio_file_name} completed")
            return True

        logger.warning(f"Primary TTS engine failed for file: {audio_file_name}. Trying fallback engine.")
        if await fallback_tts_engine.synthesize_async(session, response_text, audio_file_name, tone, lang):
            logger.info(f"Fallback TTS engine succeeded for file: {audio_file_name}")
            return True

        logger.error(f"Both primary and fallback TTS engines failed for file: {audio_file_name}")
        return False
    except Exception as e:
        logger.error(f"Error synthesizing speech for file {audio_file_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return False


class DialogManager:
    """
    Manages the conversation flow, including speech recognition, AI response generation, and speech synthesis.

    This class orchestrates the entire conversation process, handling user input,
    generating AI responses, and synthesizing speech output.

    Attributes:
        button (Button): The AIY Kit button object.
        leds (Leds): The AIY Kit LED object for visual feedback.
        tts_engines (Dict[Language, TTSEngine]): Dictionary of TTS engines for each supported language.
        fallback_tts_engine (TTSEngine): Fallback TTS engine.
        conversation_manager (ConversationManager): Manages the conversation state and generates responses.
        config (Config): The application configuration object.
        timezone (str): The timezone to use for the conversation loop.
        transcriber (SpeechTranscriber): Handles speech-to-text conversion.
        response_player (ResponsePlayer): Plays synthesized speech responses.
        original_audio_file_name (str): Base name for audio files.
    """

    def __init__(self, button: Button, leds: Leds, tts_engines: Dict[Language, TTSEngine],
                 fallback_tts_engine: TTSEngine, conversation_manager: ConversationManager, config: Config,
                 timezone: str):
        """
        Initialize the DialogManager with necessary components.

        Args:
            button (Button): The AIY Kit button object.
            leds (Leds): The AIY Kit LED object for visual feedback.
            tts_engines (Dict[Language, TTSEngine]): Dictionary of TTS engines for each supported language.
            fallback_tts_engine (TTSEngine): Fallback TTS engine.
            conversation_manager (ConversationManager): Manages the conversation state and generates responses.
            config (Config): The application configuration object.
            timezone (str): The timezone to use for the conversation loop.
        """
        self.button = button
        self.leds = leds
        self.tts_engines = tts_engines
        self.fallback_tts_engine = fallback_tts_engine
        self.conversation_manager = conversation_manager
        self.config = config
        self.timezone = timezone
        self.transcriber = SpeechTranscriber(button, leds, config, cleaning=self.cleaning_routine, timezone=timezone)
        self.response_player = None
        self.original_audio_file_name = config.get('audio_file_name', 'speech.wav')

    async def cleaning_routine(self):
        """Perform cleanup tasks for the conversation manager."""
        await self.conversation_manager.process_and_clean()

    async def process_completed_tasks(self, synthesis_tasks: List[Tuple[asyncio.Task, dict]],
                                      next_response_index: int) -> int:
        """
        Process completed speech synthesis tasks and update the response player.

        Args:
            synthesis_tasks (List[Tuple[asyncio.Task, dict]]): List of synthesis tasks and their info.
            next_response_index (int): Index of the next task to process.

        Returns:
            int: Updated index of the next task to process.
        """
        while next_response_index < len(synthesis_tasks):
            task, response_info = synthesis_tasks[next_response_index]
            if task.done():
                try:
                    completion_time = time.time()
                    creation_time = getattr(task, 'creation_time', None)
                    if creation_time:
                        logger.debug(
                            f"Task {next_response_index} took {completion_time - creation_time:.2f} seconds to complete")

                    if await asyncio.wait_for(task, timeout=10.0):  # 10 second timeout
                        self.handle_successful_synthesis(response_info)
                        next_response_index += 1
                    else:
                        logger.error(f"Speech synthesis failed for file: {response_info['audio_file_name']}")
                        error_visual(self.leds)
                        next_response_index += 1
                except asyncio.TimeoutError:
                    logger.error(f"Synthesis task {next_response_index} timed out")
                    next_response_index += 1
                except Exception as e:
                    logger.error(f"Error occurred during synthesis: {str(e)}")
                    error_visual(self.leds)
                    next_response_index += 1
            else:
                # If the next task isn't done, we stop processing to maintain order
                break
        return next_response_index

    def handle_successful_synthesis(self, response_info: dict):
        """
        Handle successful speech synthesis by updating the response player.

        Args:
            response_info (dict): Information about the synthesized response.
        """
        if self.response_player is None:
            self.response_player = ResponsePlayer(
                [(response_info["emo"], response_info["audio_file_name"], response_info["response_text"])], self.leds,
                self.timezone)
        else:
            self.response_player.add(
                (response_info["emo"], response_info["audio_file_name"], response_info["response_text"]))
        logger.debug(f"Added to merge queue: {response_info['audio_file_name']}")

    async def main_loop_async(self):
        """
        The main conversation loop of the AI assistant.

        This method handles the flow of conversation, including listening for user input,
        generating AI responses, synthesizing speech, and playing responses.
        """
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    self.conversation_manager.save_dialog()
                    text = await self.transcriber.transcribe_speech(self.response_player)
                    logger.info(f'({time_string_ms(self.timezone)}) You said: {text}')
                    self.response_player = None

                    if text:
                        await asyncio.gather(
                            save_to_conversation("user", text, self.timezone),
                            self.process_ai_response(session, text),
                        )

                except Exception as e:
                    logger.error(f"An error occurred in the main loop: {str(e)}")
                    logger.error(traceback.format_exc())
                    error_visual(self.leds)

    async def process_ai_response(self, session: aiohttp.ClientSession, text: str):
        """
        Process the AI response to user input.

        This method generates AI responses, synthesizes speech, and manages the response playback.

        Args:
            session (aiohttp.ClientSession): The aiohttp session for making requests.
            text (str): The transcribed user input.
        """
        response_count = 0
        synthesis_tasks = []
        next_response_index = 0
        button_pressed = False

        def set_button_pressed():
            nonlocal button_pressed
            button_pressed = True

        self.button.when_pressed = set_button_pressed
        logger.info("Set button callback")

        ai_message = ""
        async for ai_response in self.conversation_manager.get_response(text):

            for response in ai_response:
                if button_pressed:
                    logger.info("Button pressed, stopping processing")
                    if self.response_player:
                        self.response_player.stop()
                    break

                response_count += 1
                logger.info(f'({time_string_ms(self.timezone)}) AI: {response["text"]}')
                ai_message += response["text"]

                synthesis_task = self.create_synthesis_task(session, response, response_count)
                synthesis_tasks.append(synthesis_task)

                # Process completed tasks, but only play if it's the next in order
                next_response_index = await self.process_completed_tasks(synthesis_tasks, next_response_index)

                await asyncio.sleep(0)

        save_conversation_task = asyncio.create_task(save_to_conversation("assistant", ai_message, self.timezone))

        # Process any remaining tasks
        while next_response_index < len(synthesis_tasks):
            if button_pressed:
                logger.info("Button pressed (2), stopping processing")
                if self.response_player:
                    self.response_player.stop()
                break
            next_response_index = await self.process_completed_tasks(synthesis_tasks, next_response_index)
            await asyncio.sleep(0.1)

        # while (self.response_player is not None) and self.response_player.is_playing():
        #     await asyncio.sleep(0.1)
        #
        # self.response_player = None

        # Wait for the save_to_conversation task to finish
        await save_conversation_task

    def create_synthesis_task(self, session: aiohttp.ClientSession, response: dict, response_count: int) -> Tuple[
        asyncio.Task, dict]:
        """
        Create a speech synthesis task for an AI response.

        Args:
            session (aiohttp.ClientSession): The aiohttp session for making requests.
            response (dict): The AI response to synthesize.
            response_count (int): The count of responses processed so far.

        Returns:
            Tuple[asyncio.Task, dict]: A tuple containing the synthesis task and response info.
        """
        emo = response["emotion"]
        response_text = response["text"]
        lang_code = response["language"]
        audio_file_name = append_suffix(self.original_audio_file_name, str(response_count))
        tone = Tone.HAPPY if emo and 'voice' in emo and 'tone' in emo['voice'] and emo['voice'][
            'tone'] == "happy" else Tone.PLAIN
        lang = {"ru": Language.RUSSIAN, "en": Language.ENGLISH, "de": Language.GERMAN}.get(lang_code, Language.RUSSIAN)

        tts_engine = self.tts_engines.get(lang, self.tts_engines[Language.RUSSIAN])
        logger.debug(f"Tone: {tone}, language = {lang}, tts_engine = {tts_engine}")

        logger.debug(f"Creating synthesis task for response {response_count}")
        synthesis_task = asyncio.create_task(
            synthesize_with_fallback(session, tts_engine, self.fallback_tts_engine, response_text, audio_file_name,
                                     tone, lang))
        setattr(synthesis_task, 'creation_time', time.time())
        logger.debug(f"Synthesis task created for response {response_count}")

        return synthesis_task, {"emo": emo, "audio_file_name": audio_file_name, "response_text": response_text}


async def main_loop_async(button: Button, leds: Leds, tts_engines: Dict[Language, TTSEngine],
                          fallback_tts_engine: TTSEngine, conversation_manager: ConversationManager, config: Config,
                          timezone: str) -> None:
    """
    The main entry point for the conversation loop of the AI assistant.

    This function initializes the DialogManager and starts the main conversation loop.

    Args:
        button (Button): The AIY Kit button object.
        leds (Leds): The AIY Kit LED object for visual feedback.
        tts_engines (Dict[Language, TTSEngine]): Dictionary of TTS engines for each supported language.
        fallback_tts_engine (TTSEngine): Fallback TTS engine.
        conversation_manager (ConversationManager): Manages the conversation state and generates responses.
        config (Config): The application configuration object.
        timezone (str): The timezone to use for the conversation loop.
    """
    dialog_manager = DialogManager(button, leds, tts_engines, fallback_tts_engine, conversation_manager, config,
                                   timezone)
    await dialog_manager.main_loop_async()
