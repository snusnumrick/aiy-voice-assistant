"""
This module provides functionality for controlling LED behavior and playing audio responses
based on emotional states. It includes utilities for adjusting RGB colors, changing LED
patterns, and managing a playlist of audio files with corresponding LED behaviors.
"""
import json
import logging
import threading
from subprocess import Popen
import re
from typing import List, Tuple, Dict, Optional

from aiy.leds import Leds, Pattern
from aiy.voice.audio import play_wav_async

logger = logging.getLogger(__name__)


def emotions_prompt() -> str:
    """
        Returns a string containing instructions for expressing emotions using LED commands.

        Returns:
            str: A prompt explaining how to use LED commands to express emotions.
        """
    # return ("У тебя есть эмоции (или ты можешь симулировать их). "
    #         "Ты можешь выразить их управляя цветным фонариком с рассеяным светом. "
    #         "Чтобы отдать команду фонарику, вставь в текст твоего сообшения "
    #         '$emotion: {"color": [R, G, B], "behavior": "continuous/blinking/breathing", '
    #         '"brightness": "dark/medium/bright", "cycle": X}$, '
    #         "где R, G, B - компоненты цвета от 0 до 255; "
    #         "X - продолжительность цикла в секундах для режимов blinking и breathing. "
    #         "Сообщать об эмоциях надо перед текстом, к которому они относятся. "
    #         "Отправить пустую эмоцию для выключения фонарика. ")
    return ('Express emotions with light (always place before relevant text): '
            '$emotion:{"color":[R,G,B] (0-255),"behavior":"continuous/blinking/breathing",'
            '"brightness":"dark/medium/bright","cycle":X}$. Empty emotion turns off light. ')


def extract_emotions(text: str) -> List[Tuple[dict, str]]:
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
                results.append(({}, remaining_text))
            break
        preceding_text = match.group(1).strip()
        if preceding_text:
            results.append(({}, preceding_text))

        emotion_dict_str = match.group(2) if match.group(2) else '{}'
        associated_text = match.group(3).strip()
        try:
            emotion_dict = json.loads(emotion_dict_str)
            results.append((emotion_dict, associated_text))
        except json.JSONDecodeError:
            results.append(({}, associated_text))
        pos = match.end()

    return results


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
            leds.pattern = Pattern.breathe(behaviour["cycle"] * 1000)
            leds.update(Leds.rgb_pattern(color))
            logger.debug(
                f"breathing {behaviour['color']} {behaviour['brightness']} ({color}) with {behaviour['cycle']} period")
        elif behaviour["behavior"] == "blinking":
            leds.pattern = Pattern.blink(behaviour["cycle"] * 1000)
            leds.update(Leds.rgb_pattern(color))
            logger.debug(
                f"blinking {behaviour['color']} {behaviour['brightness']} ({color}) with {behaviour['cycle']} period")
        else:
            leds.update(Leds.rgb_on(color))
            logger.debug(f"solid {behaviour['color']} {behaviour['brightness']} ({color}) color")


class ResponsePlayer:
    """
    A class for playing a sequence of audio files with corresponding LED behaviors.
    """

    def __init__(self, playlist: List[Tuple[Dict, str]], leds: Leds):
        """
        Initializes the ResponsePlayer.

        Args:
            playlist (List[Tuple[Dict, str]]): A list of tuples containing LED behavior and audio file path.
            leds (Leds): An instance of the Leds class to control.
        """

        self.playlist = playlist
        self.current_process: Optional[Popen] = None
        self._is_playing = False
        self.play_thread: Optional[threading.Thread] = None
        self.leds = leds

    def play(self):
        """Starts playing the playlist in a separate thread."""

        self._is_playing = True
        self.play_thread = threading.Thread(target=self._play_sequence)
        self.play_thread.start()

    def _play_sequence(self):
        """Internal method to play the sequence of audio files and control LED behavior."""

        for light_behavior, audio_file in self.playlist:
            if not self._is_playing:
                break

            change_light_behavior(light_behavior, self.leds)
            self.current_process = play_wav_async(audio_file)

            # Wait for the audio to finish
            self.current_process.wait()

            # switch off led
            self.leds.update(Leds.rgb_off())

            if not self._is_playing:
                break

        self._is_playing = False
        self.current_process = None

    def stop(self):
        """Stops the currently playing audio and LED sequence."""

        self._is_playing = False
        if self.current_process:
            self.current_process.terminate()
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join()

    def is_playing(self) -> bool:
        """
        Checks if the player is currently playing.

        Returns:
            bool: True if playing, False otherwise.
        """

        return self._is_playing
