from typing import List, Tuple, Dict, Optional
from collections import deque
from subprocess import Popen
import threading
import time
import logging

from aiy.voice.audio import play_wav_async

logger = logging.getLogger(__name__)


def change_light_behavior(behavior: Dict):
    pass


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
            change_light_behavior(light_behavior)
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
