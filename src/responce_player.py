from typing import List, Tuple, Dict, Optional
from collections import deque
from subprocess import Popen
import threading
import time
import logging

from aiy.voice.audio import play_wav_async
from aiy.leds import Leds, Color, Pattern

logger = logging.getLogger(__name__)


def adjust_rgb_brightness(rgb: List[int], brightness: str) -> Tuple[int, int, int]:
    import colorsys

    # Define brightness factors
    brightness_factors = {
        'low': 0.4,
        'medium': 0.7,
        'high': 1.0
    }

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


def apply_behaviour(behaviour: dict):

    with Leds() as leds:
        if not behaviour:
            leds.update(Leds.rgb_off())
        else:
            color = adjust_rgb_brightness(behaviour['color'], behaviour['brightness'])
            if behaviour["behaviour"] == "breathing":
                leds.pattern = Pattern.breathe(behaviour["cycle"] * 1000)
                leds.update(Leds.rgb_pattern(color))
            elif behaviour["behaviour"] == "blinking":
                leds.pattern = Pattern.blink(behaviour["cycle"] * 1000)
                leds.update(Leds.rgb_pattern(color))
            else:
                leds.update(Leds.rgb_on(color))

class ResponsePlayer:
    def __init__(self, playlist: List[Tuple[Dict, str]]):
        self.playlist = playlist
        self.current_process: Optional[Popen] = None
        self.is_playing = False
        self.play_thread: Optional[threading.Thread] = None

    def play(self):
        self.is_playing = True
        self.play_thread = threading.Thread(target=self._play_sequence)
        self.play_thread.start()

    def _play_sequence(self):
        for light_behavior, audio_file in self.playlist:
            if not self.is_playing:
                break

            logger.info(f"playing {audio_file} with {light_behavior}")
            apply_behaviour(light_behavior)
            self.current_process = play_wav_async(audio_file)

            # Wait for the audio to finish
            while self.current_process.poll() is None:
                if not self.is_playing:
                    self.current_process.terminate()
                    break
                time.sleep(0.1)

        self.is_playing = False
        self.current_process = None

    def stop(self):
        self.is_playing = False
        if self.current_process:
            self.current_process.terminate()
        if self.play_thread:
            self.play_thread.join()

    def is_playing(self) -> bool:
        return self.is_playing
