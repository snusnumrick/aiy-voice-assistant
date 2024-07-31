"""
This module provides functionality for controlling LED behavior and playing audio responses
based on emotional states. It includes utilities for adjusting RGB colors, changing LED
patterns, and managing a playlist of audio files with corresponding LED behaviors.
"""
import json
import logging
import queue
import re
import tempfile
import threading
from subprocess import Popen
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from aiy.leds import Leds, Pattern
from aiy.voice.audio import play_wav_async

from src.tools import combine_audio_files, time_string_ms

logger = logging.getLogger(__name__)


def emotions_prompt() -> str:
    """
        Returns a string containing instructions for expressing emotions using LED commands.

        Returns:
            str: A prompt explaining how to use LED commands to express emotions.
        """
    return ('Express emotions with light and tone of voice (always place before relevant text): '
            '$emotion:{"light":{"color":[R,G,B] (0-255),"behavior":"continuous/blinking/breathing",'
            '"brightness":"dark/medium/bright","period":X (sec)}}, {"voice":{"tone":"plain/happy"}}}$. '
            'All fields are mandatory. '
            'Empty emotion or emotion with empty light turns off light. '
            'Empty emotion or emotion with empty voice reset tone to plain.')


def extract_emotions(text: str) -> List[Tuple[Optional[dict], str]]:
    """
        This function parses the given text and extracts 'emotion' dictionaries (if any) and the associated text following them.
        The structured data is returned as a list of tuples, each containing the dictionary and the corresponding text.

        An emotion dictionary is expected to be enclosed inside '$emotion:' and '$' markers in the input text.
        Any text not preceded by an emotion marker is associated with an empty dictionary.

        :param text: str, Input text which includes 'emotion' dictionaries and text.
        :return: List[Tuple[Dict, str]]. Each tuple contains:
            - dict: The parsed 'emotion' dictionary or an empty dictionary if no dictionary was found.
            - str: The associated text following the dictionary or preceding the next dictionary.
        """

    pattern = re.compile(r'(.*?)\$emotion:\s*(\{.*?\})?\$(.*?)(?=\$emotion:|$)', re.DOTALL)

    results = []
    pos = 0
    while pos < len(text):
        match = pattern.search(text, pos)
        if not match:
            remaining_text = text[pos:].strip()
            if remaining_text:
                results.append((None, remaining_text))
            break
        preceding_text = match.group(1).strip()
        if preceding_text:
            results.append((None, preceding_text))

        emotion_dict_str = match.group(2) if match.group(2) else '{}'
        associated_text = match.group(3).strip()
        try:
            emotion_dict = json.loads(emotion_dict_str)
            results.append((emotion_dict, associated_text))
        except json.JSONDecodeError:
            results.append((None, associated_text))
        pos = match.end()

    return results


def language_prompt() -> str:
    return "If you reply or part of it uses different language than before, use $lang: ru/en/de$. "


def extract_language(text: str, default_lang="ru") -> List[Tuple[str, str]]:
    # Regular expression to match language codes and subsequent text
    pattern = r'(?:^(.*?))?(?:\$lang:\s*(\w+)\$(.*?))?(?=\$lang:|$)'

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Process matches into the desired format
    result = []
    for default, lang, segment in matches:
        if default.strip():
            result.append((default_lang, default.strip()))
        if lang:
            result.append((lang, segment.strip()))

    return result


def adjust_rgb_brightness(rgb: List[int], brightness: str) -> Tuple[int, int, int]:
    """
    Adjusts the brightness of an RGB color.

    Args:
        rgb (List[int]): A list of three integers representing RGB values (0-255).
        brightness (str): A string indicating the desired brightness level ('low', 'medium', 'high').

    Returns:
        Tuple[int, int, int]: Adjusted RGB values.
    """
    import colorsys

    # Define brightness factors
    brightness_factors = {'low': 0.4, 'medium': 0.7, 'high': 1.0}

    # Get the brightness factor, default to medium if invalid input
    factor = brightness_factors.get(brightness.lower(), 0.7)

    # Convert RGB to HSV
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Adjust the V (value) component
    v = min(1.0, v * factor)

    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    # Convert back to 0-255 range and return as integers
    r, g, b = (int(x * 255) for x in (r, g, b))
    return r, g, b


def change_light_behavior(behaviour: dict, leds: Leds) -> None:
    """
    Changes the LED behavior based on the provided behaviour dictionary.

    Args:
        behaviour (dict): A dictionary containing LED behavior parameters.
        leds (Leds): An instance of the Leds class to control.
    """
    logger.info(f"changing LED behavior: {behaviour}")
    if not behaviour:
        leds.update(Leds.rgb_off())
    else:
        color = adjust_rgb_brightness(behaviour['color'], behaviour['brightness'])
        if behaviour["behavior"] == "breathing":
            leds.pattern = Pattern.breathe(behaviour["period"] * 1000)
            leds.update(Leds.rgb_pattern(color))
            logger.debug(
                f"breathing {behaviour['color']} {behaviour['brightness']} ({color}) with {behaviour['period']} period")
        elif behaviour["behavior"] == "blinking":
            leds.pattern = Pattern.blink(behaviour["period"] * 1000)
            leds.update(Leds.rgb_pattern(color))
            logger.debug(
                f"blinking {behaviour['color']} {behaviour['brightness']} ({color}) with {behaviour['period']} period")
        else:
            leds.update(Leds.rgb_on(color))
            logger.debug(f"solid {behaviour['color']} {behaviour['brightness']} ({color}) color")


@dataclass
class MergeItem:
    light: dict
    filename: str
    text: str


class ResponsePlayer:
    """
    A class for playing a sequence of audio files with corresponding LED behaviors.

    This class manages a playlist of audio files and their associated LED behaviors.
    It handles merging of audio files, queueing of playback items, and coordination
    of audio playback with LED control in a thread-safe manner. It uses condition
    variables for efficient thread synchronization.

    Attributes:
        timezone (str): The timezone used for logging timestamps.
        playlist (queue.Queue): A queue of audio files and their associated LED behaviors ready for playback.
        wav_list (list): A list of WAV files to be merged.
        current_process (Optional[Popen]): The currently playing audio process.
        _should_play (bool): Flag indicating whether playback should continue.
        play_thread (Optional[threading.Thread]): Thread for audio playback.
        leds (Leds): An instance of the Leds class for controlling LED behavior.
        merge_thread (Optional[threading.Thread]): Thread for merging audio files.
        merge_queue (queue.Queue): A queue of audio files and LED behaviors to be merged.
        _playback_completed (threading.Event): Event to signal when playback is completed.
        current_light (Optional[Dict]): The current LED behavior.
        lock (threading.Lock): Lock for ensuring thread-safe operations.
        _stopped (bool): Flag indicating whether the player is in a stopped state.
        condition (threading.Condition): Condition variable for efficient thread synchronization.
    """

    def __init__(self, playlist: List[Tuple[Optional[Dict], str, str]], leds: Leds, timezone: str):
        """
        Initialize the ResponsePlayer.

        Args:
            playlist (List[Tuple[Optional[Dict], str]]): Initial playlist of audio files and LED behaviors.
            leds (Leds): An instance of the Leds class for controlling LED behavior.
            timezone (str): The timezone used for logging timestamps.
        """
        self.timezone = timezone
        self.leds = leds
        self.playlist = queue.Queue()
        self.merge_queue = queue.Queue()
        self.current_process: Optional[Popen] = None
        self.play_thread: Optional[threading.Thread] = None
        self.merge_thread: Optional[threading.Thread] = None
        self._playback_completed = threading.Event()
        self.lock = threading.RLock()  # Use RLock for nested locking
        self.condition = threading.Condition(self.lock)
        self._should_play = False
        self._stopped = False
        self.current_light = None
        self.wav_list: List[Tuple[str, str]] = []

        for item in playlist:
            self.add(item)

    def add(self, playitem: Tuple[Optional[Dict], str, str]) -> None:
        """
        Add a new item to the merge queue and start merging if necessary.

        If the player is in a stopped state, this method will ignore the add request.
        When an item is added, it notifies the playback thread to wake up and process the new item.

        Args:
            playitem (Tuple[Optional[Dict], str]): A tuple containing the LED behavior (or None) and the audio file path.
        """
        with self.lock:
            if self._stopped:
                logger.warning(f"Ignoring add request for {playitem} as player is stopped.")
                return

            emo, file, text = playitem
            light = None if emo is None else emo.get('light', None)
            m_item = MergeItem(light=light, filename=file, text=text)
            logger.debug(f"({time_string_ms(self.timezone)}) Adding {m_item} to merge queue.")
            self.merge_queue.put(m_item)

        with self.condition:
            self.condition.notify()

        if self.merge_thread is None or not self.merge_thread.is_alive():
            self.merge_thread = threading.Thread(target=self._merge_audio_files)
            self.merge_thread.start()
        if not self._should_play:
            self.play()

    def _merge_audio_files(self):
        """
        Merge audio files with the same LED behavior.

        This method runs in a separate thread and continuously processes items from the merge queue.
        It groups audio files with the same LED behavior and calls _process_wav_list to merge them.
        """
        logger.debug("Starting merge process")
        while True:
            with self.lock:
                if not self._should_play and self.merge_queue.empty():
                    break
                try:
                    mi: MergeItem = self.merge_queue.get_nowait()
                    logger.info(
                        f"({time_string_ms(self.timezone)}) merging {mi.light} {mi.filename} {self.current_light} {self.wav_list}")
                    if self.current_light is None:
                        self.current_light = mi.light
                        logger.info(f"set current light (2) to {self.current_light}")
                        self.wav_list.append((mi.filename, mi.text))
                    elif mi.light is None or mi.light == self.current_light:
                        self.wav_list.append((mi.filename, mi.text))
                    else:
                        self._process_wav_list(force=True)
                        self.current_light = mi.light
                        logger.info(f"set current light (3) to {self.current_light}")
                        self.wav_list.append((mi.filename, mi.text))
                except queue.Empty:
                    if self.wav_list:
                        self._process_wav_list()
            with self.condition:
                if self.merge_queue.empty() and not self._should_play:
                    self.condition.wait(timeout=1.0)
        logger.info("Merge process ended")

    def _process_wav_list(self, force=False):
        """
        Process the current list of WAV files.

        This method is called by _merge_audio_files to combine multiple WAV files with the same LED behavior
        into a single file, or add a single WAV file directly to the playlist.
        """
        with self.lock:
            logger.info(f"process wav list {self.wav_list}")

            if not self.wav_list:
                return

            # if not force:
            #     last_text = self.wav_list[-1][1]
            #     if last_text and not last_text[-1] in ".!?:":
            #         logger.info(f"wav list does not end with sentence ending: {self.wav_list} - waiting for completion")
            #         return

            logger.debug(
                f"({time_string_ms(self.timezone)}) merging {self.current_light} {self.wav_list} {self.playlist}")

            if len(self.wav_list) == 1:
                self.playlist.put((self.current_light, self.wav_list[0][0]))
            else:
                output_filename = tempfile.mktemp(suffix=".wav")
                combine_audio_files([w[0] for w in self.wav_list], output_filename)
                self.playlist.put((self.current_light, output_filename))

            logger.info(
                f"Processed and added merged audio to playlist: {self.current_light}, {self.wav_list}")
            self.wav_list = []

    def play(self):
        """
        Start the playback process.

        This method initiates the playback thread if it's not already running and resets the stopped state.
        """
        logger.debug("Starting playback")
        with self.condition:
            if not self._should_play:
                self._should_play = True
                self._stopped = False
                self._playback_completed.clear()
                self.play_thread = threading.Thread(target=self._play_sequence)
                self.play_thread.start()

    def _play_sequence(self):
        """
        Play the sequence of audio files with their corresponding LED behaviors.

        This method runs in a separate thread and continuously processes items from the playlist queue.
        It handles playing audio files and controlling LED behavior. It uses a condition variable
        to efficiently wait for new items to be added to the playlist or for a stop signal.
        """
        logger.debug("_play_sequence started")
        while True:
            with self.condition:
                while self._should_play and self.playlist.empty() and self.merge_queue.empty():
                    logger.debug(f"({time_string_ms(self.timezone)}) wait for condition")
                    self.condition.wait()
                if not self._should_play:
                    logger.info(f"should_play: False")
                    break
                try:
                    light, audio_file = self.playlist.get_nowait()
                    logger.info(f"({time_string_ms(self.timezone)}) got from playlist {light}, {audio_file}")
                except queue.Empty:
                    # If playlist is empty, process wav_list and continue
                    self._process_wav_list()
                    continue

            logger.info(f"({time_string_ms(self.timezone)}) Playing {audio_file} with light {light}, with current light {self.current_light}")

            if light is not None and light != self.current_light:
                change_light_behavior(light, self.leds)
            self.current_light = light
            logger.info(f"set current light (1) to {self.current_light}")

            self.current_process = play_wav_async(audio_file)
            self.current_process.wait()
            self.current_process = None

            # Switch off LED
            self.leds.update(Leds.rgb_off())

            logger.debug(f"Finished playing {audio_file}")

        logger.debug("_play_sequence ended")
        self.current_process = None
        self._playback_completed.set()

    def stop(self):
        """
        Stop the playback and merging processes, clear all queues, and ignore further add calls.

        This method:
        1. Sets the stop flags
        2. Notifies all waiting threads to wake up
        3. Terminates any current playback
        4. Clears all queues (merge_queue, playlist, wav_list)
        5. Waits for the playback and merge threads to complete
        6. Sets the player to a stopped state, ignoring further add calls

        The use of condition variables ensures that waiting threads are immediately notified of the stop request.
        """

        def clear_queue(q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        logger.info("Stopping playback and clearing all queues")
        with self.condition:
            self._should_play = False
            self._stopped = True
            self.condition.notify_all()

        if self.current_process:
            self.current_process.terminate()

        with self.lock:
            clear_queue(self.merge_queue)
            clear_queue(self.playlist)
            self.wav_list.clear()

        self._playback_completed.wait(timeout=5.0)
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
        if self.merge_thread and self.merge_thread.is_alive():
            self.merge_thread.join(timeout=1.0)

        logger.debug("Playback stopped, all queues cleared, and player set to stopped state")

    def is_playing(self) -> bool:
        """
        Check if audio is currently playing or queued for playback.

        Returns:
            bool: True if audio is playing or queued, False otherwise.
        """
        with self.lock:
            return (self._should_play and not self._stopped and (
                    not self.playlist.empty() or not self.merge_queue.empty() or self.current_process is not None))
