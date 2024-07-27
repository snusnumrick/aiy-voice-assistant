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
import time
from subprocess import Popen
from typing import List, Tuple, Dict, Optional

from aiy.leds import Leds, Pattern
from aiy.voice.audio import play_wav_async

from src.tools import combine_audio_files

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


class ResponsePlayer:
    """
    A class for playing a sequence of audio files with corresponding LED behaviors.
    """

    def __init__(self, playlist: List[Tuple[Optional[Dict], str]], leds: Leds):
        logger.info(f"Initializing ResponsePlayer with playlist: {playlist}")
        self.playlist = queue.Queue()
        self.wav_list = []
        self.current_process: Optional[Popen] = None
        self._should_play = False
        self.play_thread: Optional[threading.Thread] = None
        self.leds = leds
        self.currently_playing = -1
        self.merge_thread: Optional[threading.Thread] = None
        self.merge_queue = queue.Queue()
        self._playback_completed = threading.Event()
        self.current_light = None
        self.lock = threading.Lock()
        for item in playlist:
            self.add(item)

    def add(self, playitem: Tuple[Optional[Dict], str]) -> None:
        emo, file = playitem
        light = None if emo is None else emo.get('light', {})
        light_item = (emo, file)
        logger.info(f"Adding {light_item} to merge queue.")
        self.merge_queue.put(light_item)
        if self.merge_thread is None or not self.merge_thread.is_alive():
            self.merge_thread = threading.Thread(target=self._merge_audio_files)
            self.merge_thread.start()
        if not self._should_play:
            self.play()

    def _merge_audio_files(self):
        logger.info("Starting merge process")
        while self._should_play or not self.merge_queue.empty():
            try:
                light, wav = self.merge_queue.get(timeout=1.0)  # Wait for 1 second for new items
                logger.info(f"merging {light} {wav} {self.current_light} {self.wav_list}")
                if self.current_light is None:
                    self.current_light = light if light is not None else {}
                    self.wav_list = [wav]
                    logger.info(f"1 {self.current_light} {self.wav_list}")
                elif light is None or light == self.current_light:
                    self.wav_list.append(wav)
                    logger.info(f"2 {self.current_light} {self.wav_list}")
                else:
                    self._process_merged_audio()
                    self.current_light = light
                    self.wav_list = [wav]
                    logger.info(f"3 {self.current_light} {self.wav_list}")
            except queue.Empty:
                if self.wav_list:
                    self._process_merged_audio()
                    self.wav_list = []
                    self.current_light = None
                    logger.info(f"4 {self.current_light} {self.wav_list}")
        logger.info("Merge process ended")

    def _process_merged_audio(self):
        with self.lock:
            logger.info(f"merging {self.current_light} {self.wav_list} {self.playlist}")
            if len(self.wav_list) == 1:
                self.playlist.put((self.current_light, self.wav_list[0]))
            else:
                output_filename = tempfile.mktemp(suffix=".wav")
                combine_audio_files(self.wav_list, output_filename)
                self.playlist.put((self.current_light, output_filename))
            logger.info(f"Processed and added merged audio to playlist: {self.current_light}, {self.wav_list},  {self.playlist}")

    def play(self):
        logger.info("Starting playback")
        if not self._should_play:
            self._should_play = True
            self._playback_completed.clear()
            self.play_thread = threading.Thread(target=self._play_sequence)
            self.play_thread.start()

    def _play_sequence(self):
        logger.info("_play_sequence started")
        while self._should_play:
            try:
                light, audio_file = self.playlist.get(timeout=0.1)
                logger.info(f"Playing {audio_file} with light {light}")

                if light is not None:
                    change_light_behavior(light, self.leds)

                self.current_process = play_wav_async(audio_file)
                self.currently_playing += 1

                # Wait for the audio to finish
                self.current_process.wait()
                self.current_process = None

                # Switch off LED
                self.leds.update(Leds.rgb_off())

                logger.info(f"Finished playing {audio_file}")
            except queue.Empty:
                self._process_merged_audio()
                # If both queues are empty, wait a bit before checking again
                if self.playlist.empty() and self.merge_queue.empty():
                    if not self._should_play:
                        break
                    time.sleep(0.1)

        logger.info("_play_sequence ended")
        self.current_process = None
        self._playback_completed.set()

    def stop(self):
        logger.info("Stopping playback")
        self._should_play = False
        if self.current_process:
            self.current_process.terminate()
        self._playback_completed.wait(timeout=5.0)  # Wait up to 5 seconds for playback to complete
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
        if self.merge_thread and self.merge_thread.is_alive():
            self.merge_thread.join(timeout=1.0)

    def is_playing(self) -> bool:
        logger.debug(f"{self._should_play} {self.playlist.empty()} {self.merge_queue.empty()} {self.current_process }")
        return self._should_play and (
                    not self.playlist.empty() or not self.merge_queue.empty() or self.current_process is not None)