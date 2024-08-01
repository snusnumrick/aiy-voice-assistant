from typing import Dict, List, Tuple, Optional
from unittest.mock import Mock


class MockResponsePlayer:
    def __init__(self, playlist: List[Tuple[Optional[Dict], str, str]], leds: Mock, timezone: str):
        self.playlist = playlist or []
        self.leds = leds
        self.timezone = timezone
        self._should_play = False
        self._stopped = False

    def add(self, playitem: Tuple[Optional[Dict], str, str]) -> None:
        self.playlist.append(playitem)

    def play(self):
        self._should_play = True
        self._stopped = False

    def stop(self):
        self._should_play = False
        self._stopped = True

    def is_playing(self) -> bool:
        return self._should_play and not self._stopped


# If you want to create a mock for the entire module:
mock_response_player = Mock()
mock_response_player.ResponsePlayer = MockResponsePlayer
