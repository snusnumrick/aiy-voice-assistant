"""Reminder announcer module.

Plays a bell sound, sets LED pattern, and injects reminder into context.
"""

import logging
import math
import os
import wave
from typing import Dict, Optional

from src.config import Config
from src.conversation_manager import ConversationManager
from src.responce_player import ResponsePlayer

logger = logging.getLogger(__name__)


class ReminderAnnouncer:
    """Handle due reminders with bell + LED and context injection."""

    def __init__(
        self,
        config: Config,
        response_player: ResponsePlayer,
        conversation_manager: ConversationManager,
    ) -> None:
        self.config = config
        self.response_player = response_player
        self.conversation_manager = conversation_manager
        self.bell_file = config.get("reminder_bell_file", "assets/bell.wav")
        self.default_light = config.get(
            "reminder_default_light",
            {
                "color": [255, 180, 0],
                "behavior": "blinking",
                "brightness": "bright",
                "period": 0.4,
            },
        )
        self._bell_duration_sec = self._load_bell_duration_sec(self.bell_file)

    def _load_bell_duration_sec(self, path: str) -> Optional[float]:
        if not os.path.exists(path):
            logger.warning(f"Reminder bell file not found: {path}")
            return None
        try:
            with wave.open(path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate <= 0:
                    return None
                return frames / float(rate)
        except Exception as e:
            logger.warning(f"Failed to read bell duration: {e}")
            return None

    def _resolve_repeat(self, reminder: Dict[str, any]) -> int:
        if "bell_repeat" in reminder:
            try:
                return max(1, int(reminder.get("bell_repeat")))
            except Exception:
                return 1
        if "bell_duration_sec" in reminder and self._bell_duration_sec:
            try:
                duration = float(reminder.get("bell_duration_sec"))
                if duration <= 0:
                    return 1
                return max(1, int(math.ceil(duration / self._bell_duration_sec)))
            except Exception:
                return 1
        return 1

    async def notify(self, reminder: Dict[str, any]) -> None:
        self.conversation_manager.add_pending_reminder_fact(reminder)

        if not os.path.exists(self.bell_file):
            logger.warning("Bell file missing; skipping bell playback")
            return

        light = reminder.get("light") if isinstance(reminder.get("light"), dict) else None
        if not light:
            light = self.default_light
        emo = {"light": light}

        repeat = self._resolve_repeat(reminder)
        for _ in range(repeat):
            self.response_player.add((emo, self.bell_file, "reminder_bell"))
