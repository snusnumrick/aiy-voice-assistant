"""Reminder announcer module.

Plays a bell sound, sets LED pattern, and injects reminder into context.
"""

import asyncio
import logging
import math
import os
import wave
from typing import Any, Dict, Optional

import aiohttp

from src.config import Config
from src.conversation_manager import ConversationManager
from src.responce_player import ResponsePlayer
from src.tts_engine import Language, Tone, TTSEngine

logger = logging.getLogger(__name__)


class ReminderAnnouncer:
    """Handle due reminders with bell + LED and context injection."""

    def __init__(
        self,
        config: Config,
        response_player: ResponsePlayer,
        conversation_manager: ConversationManager,
        tts_engines: Dict[Language, TTSEngine],
        fallback_tts_engine: TTSEngine,
    ) -> None:
        self.config = config
        self.response_player = response_player
        self.conversation_manager = conversation_manager
        self.tts_engines = tts_engines
        self.fallback_tts_engine = fallback_tts_engine
        self.bell_file = config.get("reminder_bell_file", "assets/bell.wav")
        self.silence_file = config.get(
            "reminder_silence_file", "assets/silence_250ms.wav"
        )
        self.speech_delay_sec = float(config.get("reminder_speech_delay_sec", 0.25))
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
        self._silence_cache: Dict[float, str] = {}
        self.ready_breathing_period_ms = float(
            config.get("ready_breathing_period_ms", 10000)
        )
        self.ready_breathing_color = config.get("ready_breathing_color", (0, 1, 0))
        self.ready_breathing_duration = float(
            config.get("ready_breathing_duration", 60)
        )

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

    def _resolve_repeat(self, reminder: Dict[str, Any]) -> int:
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

    def _resolve_language(self, reminder: Dict[str, Any]) -> Language:
        lang_code = reminder.get("language") or self.config.get(
            "reminders_language", "ru"
        )
        return {
            "ru": Language.RUSSIAN,
            "en": Language.ENGLISH,
            "de": Language.GERMAN,
        }.get(str(lang_code).lower(), Language.RUSSIAN)

    def _resolve_tone(self, reminder: Dict[str, Any]) -> Tone:
        emotion = reminder.get("emotion")
        if isinstance(emotion, dict):
            voice = emotion.get("voice")
            if isinstance(voice, dict) and voice.get("tone") == "happy":
                return Tone.HAPPY
        return Tone.PLAIN

    def _resolve_light(self, reminder: Dict[str, Any]) -> dict:
        emotion = reminder.get("emotion")
        if isinstance(emotion, dict):
            light = emotion.get("light")
            if isinstance(light, dict):
                return light
        light = reminder.get("light")
        if isinstance(light, dict):
            return light
        return self.default_light

    def _resolve_speak_text(self, reminder: Dict[str, Any]) -> Optional[str]:
        speak_text = reminder.get("speak_text")
        if isinstance(speak_text, str) and speak_text.strip():
            return speak_text.strip()
        message = reminder.get("message")
        if isinstance(message, str) and message.strip():
            prefix = self.config.get("reminders_prefix", "Напоминаю: ")
            return f"{prefix}{message.strip()}"
        return None

    def _breathing_light(self) -> dict:
        color = self.ready_breathing_color
        if isinstance(color, tuple):
            color = list(color)
        return {
            "color": color,
            "behavior": "breathing",
            "brightness": "medium",
            "period": self.ready_breathing_period_ms / 1000.0,
        }

    def _get_silence_file(self, duration_sec: float) -> Optional[str]:
        duration_sec = max(0.0, float(duration_sec))
        if duration_sec <= 0:
            return None
        if duration_sec in self._silence_cache:
            return self._silence_cache[duration_sec]
        path = f"/tmp/silence_{int(duration_sec * 1000)}ms.wav"
        try:
            sample_rate = 44100
            frames = int(sample_rate * duration_sec)
            with wave.open(path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"\x00\x00" * frames)
            self._silence_cache[duration_sec] = path
            return path
        except Exception as e:
            logger.warning(f"Failed to create silence file: {e}")
            return None

    def _queue_ready_breathing(self) -> None:
        if self.ready_breathing_duration <= 0:
            return
        silence_file = self._get_silence_file(self.ready_breathing_duration)
        if silence_file:
            self.response_player.add(
                ({"light": self._breathing_light()}, silence_file, "reminder_breathing")
            )
        if os.path.exists(self.silence_file):
            self.response_player.add(({}, self.silence_file, "reminder_reset"))

    async def _synthesize_speech(self, text: str, tone: Tone, lang: Language) -> Optional[str]:
        audio_file_name = f"/tmp/reminder_{int(asyncio.get_event_loop().time() * 1000)}.wav"
        tts_engine = self.tts_engines.get(lang, self.tts_engines[Language.RUSSIAN])
        try:
            async with aiohttp.ClientSession() as session:
                ok = await tts_engine.synthesize_async(
                    session, text, audio_file_name, tone, lang
                )
                if not ok:
                    ok = await self.fallback_tts_engine.synthesize_async(
                        session, text, audio_file_name, tone, lang
                    )
                if ok:
                    return audio_file_name
        except Exception as e:
            logger.error(f"Reminder TTS failed: {e}")
        return None

    async def notify(self, reminder: Dict[str, Any]) -> None:
        self.conversation_manager.add_pending_reminder_fact(reminder)

        if not os.path.exists(self.bell_file):
            logger.warning("Bell file missing; skipping bell playback")
            return

        light = self._resolve_light(reminder)
        emo = {"light": light}
        repeat = self._resolve_repeat(reminder)
        for _ in range(repeat):
            self.response_player.add((emo, self.bell_file, "reminder_bell"))

        speak_text = self._resolve_speak_text(reminder)
        if not speak_text:
            self._queue_ready_breathing()
            return

        lang = self._resolve_language(reminder)
        tone = self._resolve_tone(reminder)
        speech_task = asyncio.create_task(
            self._synthesize_speech(speak_text, tone, lang)
        )

        if os.path.exists(self.silence_file):
            self.response_player.add((None, self.silence_file, "reminder_silence"))
        elif self.speech_delay_sec > 0:
            await asyncio.sleep(self.speech_delay_sec)

        speech_file = await speech_task
        if speech_file:
            self.response_player.add(({"light": light}, speech_file, speak_text))
        self._queue_ready_breathing()
