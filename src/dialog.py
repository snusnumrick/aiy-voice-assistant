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
import tempfile
from aiy.board import Button
from aiy.leds import Leds, Color

from .audio import SpeechTranscriber
from .config import Config
from .conversation_manager import ConversationManager
from .responce_player import ResponsePlayer
from .shared_state import ButtonState
from .tools import time_string_ms, save_to_conversation
from .tts_engine import TTSEngine, Tone, Language

logger = logging.getLogger(__name__)


def error_visual(leds: Leds) -> None:
    """
    Display a visual error indicator using the LED.

    Args:
        leds (Leds): The AIY Kit LED object for visual feedback.
    """
    logger.debug("Error... LED blinking")
    leds.update(Leds.rgb_on(Color.RED))
    time.sleep(0.3)
    leds.update(Leds.rgb_off())
    logger.debug("Error... LED off")


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


async def synthesize_with_fallback(
    session: aiohttp.ClientSession,
    tts_engine: TTSEngine,
    fallback_tts_engine: TTSEngine,
    response_text: str,
    audio_file_name: str,
    tone: Tone,
    lang: Language,
) -> bool:
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
    logger.debug(
        f"synthesize_with_fallback. text: {response_text}, file: {audio_file_name}, tone: {tone}, lang: {lang}"
    )
    try:
        if await tts_engine.synthesize_async(
            session, response_text, audio_file_name, tone, lang
        ):
            logger.debug(f"Synthesis {audio_file_name} completed")
            return True

        logger.warning(
            f"Primary TTS engine failed for file: {audio_file_name}. Trying fallback engine."
        )
        if await fallback_tts_engine.synthesize_async(
            session, response_text, audio_file_name, tone, lang
        ):
            logger.info(f"Fallback TTS engine succeeded for file: {audio_file_name}")
            return True

        logger.error(
            f"Both primary and fallback TTS engines failed for file: {audio_file_name}"
        )
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
        button_state (ButtonState): Shared button press state for checking across components.
        transcriber (SpeechTranscriber): Handles speech-to-text conversion.
        response_player (ResponsePlayer): Plays synthesized speech responses.
    """

    def __init__(
        self,
        button: Button,
        leds: Leds,
        tts_engines: Dict[Language, TTSEngine],
        fallback_tts_engine: TTSEngine,
        conversation_manager: ConversationManager,
        config: Config,
        timezone: str,
        button_state: ButtonState,
        response_player: ResponsePlayer,
    ):
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
            button_state (ButtonState): Shared button press state for checking across components.
            response_player (ResponsePlayer): The audio response player for playback.
        """
        self.button = button
        self.leds = leds
        self.tts_engines = tts_engines
        self.fallback_tts_engine = fallback_tts_engine
        self.conversation_manager = conversation_manager
        self.config = config
        self.timezone = timezone
        self.button_state = button_state
        self.response_player = response_player
        self.transcriber = SpeechTranscriber(
            button, leds, config, cleaning=self.cleaning_routine, timezone=timezone
        )

    async def cleaning_routine(self):
        """Perform cleanup tasks for the conversation manager."""
        await self.conversation_manager.process_and_clean()

    async def process_completed_tasks(
        self, synthesis_tasks: List[Tuple[asyncio.Task, dict]], next_response_index: int
    ) -> int:
        """
        Process completed speech synthesis tasks and update the response player.

        Args:
            synthesis_tasks (List[Tuple[asyncio.Task, dict]]): List of synthesis tasks and their info.
            next_response_index (int): Index of the next task to process.

        Returns:
            int: Updated index of the next task to process.
        """
        logger.debug(
            f"process_completed_tasks {next_response_index}/{len(synthesis_tasks)}"
        )
        while next_response_index < len(synthesis_tasks):
            task, response_info = synthesis_tasks[next_response_index]
            if task.done():
                logger.debug(f"task {next_response_index} done")
                try:
                    completion_time = time.time()
                    creation_time = getattr(task, "creation_time", None)
                    if creation_time:
                        logger.debug(
                            f"Task {next_response_index} took {completion_time - creation_time:.2f} seconds to complete"
                        )

                    if await asyncio.wait_for(task, timeout=10.0):  # 10 second timeout
                        logger.debug(f"Task {next_response_index} completed")
                        self.handle_successful_synthesis(response_info)
                        next_response_index += 1
                    else:
                        logger.error(
                            f"Speech synthesis failed for file: {response_info['audio_file_name']}"
                        )
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
        self.response_player.add(
            (
                response_info["emo"],
                response_info["audio_file_name"],
                response_info["response_text"],
            )
        )
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
                    text = await self.transcriber.transcribe_speech(
                        self.response_player
                    )
                    logger.info(f"({time_string_ms(self.timezone)}) You said: {text}")
                    # Clear queues for next conversation while keeping player active
                    self.response_player.clear_queues()

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
        ai_message = ""
        ai_responses_complete = False

        def _set_button_pressed():
            self.button_state.press()
            logger.debug("Button press detected, initiating shutdown sequence")

        def _set_ai_responses_complete():
            nonlocal ai_responses_complete
            ai_responses_complete = True
            logger.debug("AI response generation marked as complete")

        self.button.when_pressed = _set_button_pressed
        logger.debug("Button callback set for interruption")

        conversation_response_generator = self.conversation_manager.get_response(text)
        logger.debug(f"Initialized conversation response generator for input: {text}")

        save_conversation_task = None

        async def _process_ai_responses():
            nonlocal response_count, ai_message, synthesis_tasks, save_conversation_task
            try:
                logger.debug("Starting AI response processing loop")
                async for ai_response in conversation_response_generator:
                    if self.button_state():
                        logger.info(
                            "Button press detected, stopping AI response processing"
                        )
                        break
                    logger.debug(
                        f"Received AI response chunk with {len(ai_response)} responses"
                    )
                    for response in ai_response:
                        response_count += 1
                        logger.debug(
                            f'({time_string_ms(self.timezone)}) Processing AI response {response_count}: {response["text"][:50]}...'
                        )

                        # Accumulate AI message
                        if ai_message:
                            ai_message += " "
                        ai_message += response["text"]

                        # Create and add synthesis task
                        synthesis_task = self.create_synthesis_task(
                            session, response, response_count
                        )
                        synthesis_tasks.append(synthesis_task)
                        logger.debug(
                            f"Created synthesis task {response_count}, total tasks now: {len(synthesis_tasks)}"
                        )

                logger.debug("AI response generation completed normally")
            except Exception as e:
                logger.error(f"Error in AI response generation: {str(e)}")
                logger.error(traceback.format_exc())
                error_visual(self.leds)
            finally:
                _set_ai_responses_complete()  # Mark AI responses as complete
                logger.debug("AI response processing loop exited")

        async def process_synthesis_tasks():
            nonlocal synthesis_tasks
            try:
                logger.debug("Starting synthesis task processing loop")
                while not self.button_state():
                    if ai_responses_complete and not synthesis_tasks:
                        logger.debug(
                            "AI responses complete and no more synthesis tasks, exiting loop"
                        )
                        break
                    if synthesis_tasks:
                        # logger.debug(f"Processing batch of {len(synthesis_tasks)} synthesis tasks")
                        next_response_index = await self.process_completed_tasks(
                            synthesis_tasks, 0
                        )
                        # processed_tasks = synthesis_tasks[:next_response_index]
                        synthesis_tasks = synthesis_tasks[next_response_index:]
                        # logger.debug(f"Processed {len(processed_tasks)} tasks, {len(synthesis_tasks)} remaining")
                    # else:
                    #     logger.debug("No synthesis tasks to process, waiting...")
                    await asyncio.sleep(0.1)
                if self.button_state():
                    logger.info(
                        "Button pressed, immediately exiting synthesis task processing"
                    )
                logger.debug("Synthesis task processing loop completed")
            except Exception as e:
                logger.error(f"Error in synthesis task processing: {str(e)}")
                logger.error(traceback.format_exc())
            finally:
                logger.debug("Synthesis task processing loop exited")

        try:
            logger.debug(
                "Initiating concurrent processing of AI responses and synthesis tasks"
            )
            ai_task = asyncio.create_task(_process_ai_responses())
            synthesis_task = asyncio.create_task(process_synthesis_tasks())

            # Wait for both tasks to complete
            await asyncio.gather(ai_task, synthesis_task)
            logger.debug("Both AI response and synthesis task processing completed")

            # Handle conversation saving
            if ai_message:
                if save_conversation_task and not save_conversation_task.done():
                    logger.debug("Cancelling previous save_to_conversation task")
                    save_conversation_task.cancel()

                save_conversation_task = asyncio.create_task(
                    save_to_conversation("assistant", ai_message, self.timezone)
                )
                logger.debug("Initiated new save_to_conversation task")

        except Exception as e:
            logger.error(f"Unexpected error in process_ai_response: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            logger.debug("Entering final cleanup phase")
            # Cancel AI and synthesis tasks if they're still running
            for task in [ai_task, synthesis_task]:
                if task and not task.done():
                    logger.info(f"Cancelling task: {task}")
                    task.cancel()

            # Wait for tasks to be cancelled
            if ai_task or synthesis_task:
                logger.debug("Waiting for tasks to be cancelled")
                await asyncio.gather(ai_task, synthesis_task, return_exceptions=True)
                logger.debug("All tasks cancelled")

            # Only stop the response player if button was pressed
            if self.button_state():
                if self.response_player:
                    logger.debug("Button pressed, stopping response player")
                    self.response_player.stop()
            else:
                logger.debug("Button not pressed, leaving response player active")

            # Ensure final conversation state is saved
            if save_conversation_task:
                logger.debug("Finalizing conversation save")
                try:
                    await save_conversation_task
                    logger.debug("Final conversation save completed successfully")
                except asyncio.CancelledError:
                    logger.warning("Final save_to_conversation task was cancelled")

            logger.debug(
                f"process_ai_response completed. Processed {response_count} responses in total."
            )

    def create_synthesis_task(
        self, session: aiohttp.ClientSession, response: dict, response_count: int
    ) -> Tuple[asyncio.Task, dict]:
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
        audio_file_name = tempfile.mktemp(suffix=".wav")
        tone = (
            Tone.HAPPY
            if emo
            and "voice" in emo
            and "tone" in emo["voice"]
            and emo["voice"]["tone"] == "happy"
            else Tone.PLAIN
        )
        lang = {
            "ru": Language.RUSSIAN,
            "en": Language.ENGLISH,
            "de": Language.GERMAN,
        }.get(lang_code, Language.RUSSIAN)

        tts_engine = self.tts_engines.get(lang, self.tts_engines[Language.RUSSIAN])
        logger.debug(f"Tone: {tone}, language = {lang}, tts_engine = {tts_engine}")

        logger.debug(f"Creating synthesis task for response {response_count}")
        synthesis_task = asyncio.create_task(
            synthesize_with_fallback(
                session,
                tts_engine,
                self.fallback_tts_engine,
                response_text,
                audio_file_name,
                tone,
                lang,
            )
        )
        setattr(synthesis_task, "creation_time", time.time())
        logger.debug(f"Synthesis task created for response {response_count}")

        return synthesis_task, {
            "emo": emo,
            "audio_file_name": audio_file_name,
            "response_text": response_text,
        }


async def main_loop_async(
    button: Button,
    leds: Leds,
    tts_engines: Dict[Language, TTSEngine],
    fallback_tts_engine: TTSEngine,
    conversation_manager: ConversationManager,
    config: Config,
    timezone: str,
    button_state: ButtonState,
    response_player: ResponsePlayer,
) -> None:
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
        button_state (ButtonState): Shared button press state for checking across components.
        response_player (ResponsePlayer): The audio response player for playback.
    """
    dialog_manager = DialogManager(
        button,
        leds,
        tts_engines,
        fallback_tts_engine,
        conversation_manager,
        config,
        timezone,
        button_state,
        response_player,
    )
    await dialog_manager.main_loop_async()
