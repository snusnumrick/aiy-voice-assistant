"""
Reminder management module.

Loads reminders from a JSON file, checks for due items, and persists updates.
"""

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReminderManager:
    """Load, check, and update reminders stored in a JSON file."""

    def __init__(self, file_path: str, timezone: Optional[dt.tzinfo] = None):
        self.file_path = Path(file_path)
        self.timezone = timezone

    def _load(self) -> List[Dict[str, Any]]:
        if not self.file_path.exists():
            return []
        try:
            with self.file_path.open(encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            logger.error(f"Failed to load reminders from {self.file_path}: {e}")
        return []

    def _save(self, reminders: List[Dict[str, Any]]) -> None:
        try:
            with self.file_path.open("w", encoding="utf-8") as f:
                json.dump(reminders, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reminders to {self.file_path}: {e}")

    def _ensure_done(self, reminder: Dict[str, Any]) -> None:
        if "done" not in reminder:
            reminder["done"] = False

    def _parse_time(self, value: str) -> Optional[dt.datetime]:
        try:
            parsed = dt.datetime.fromisoformat(value)
        except Exception:
            logger.warning(f"Invalid reminder time format: {value}")
            return None

        if parsed.tzinfo is None:
            if self.timezone:
                try:
                    if hasattr(self.timezone, "localize"):
                        return self.timezone.localize(parsed)
                    return parsed.replace(tzinfo=self.timezone)
                except Exception:
                    return parsed
            return parsed

        if self.timezone:
            try:
                return parsed.astimezone(self.timezone)
            except Exception:
                return parsed
        return parsed

    def _format_time(self, value: dt.datetime) -> str:
        return value.isoformat()

    def _next_daily_time(self, value: dt.datetime) -> dt.datetime:
        return value + dt.timedelta(days=1)

    def _next_id(self, reminders: List[Dict[str, Any]]) -> str:
        numeric_ids = []
        for reminder in reminders:
            rid = reminder.get("id")
            if isinstance(rid, str) and rid.isdigit():
                numeric_ids.append(int(rid))
        if numeric_ids:
            return str(max(numeric_ids) + 1)
        return str(int(dt.datetime.now().timestamp() * 1000))

    def list_reminders(self, include_done: bool = True) -> List[Dict[str, Any]]:
        reminders = self._load()
        if include_done:
            return reminders
        return [r for r in reminders if isinstance(r, dict) and not r.get("done", False)]

    def add_reminder(
        self,
        message: str,
        when: dt.datetime,
        recurring: bool = False,
        speak_text: Optional[str] = None,
        language: Optional[str] = None,
        emotion: Optional[Dict[str, Any]] = None,
        light: Optional[Dict[str, Any]] = None,
        bell_repeat: Optional[int] = None,
        bell_duration_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        reminders = self._load()
        reminder = {
            "id": self._next_id(reminders),
            "time": self._format_time(when),
            "message": message,
            "recurring": bool(recurring),
            "done": False,
        }
        if isinstance(speak_text, str):
            reminder["speak_text"] = speak_text
        if isinstance(language, str):
            reminder["language"] = language
        if isinstance(emotion, dict):
            reminder["emotion"] = emotion
        if isinstance(light, dict):
            reminder["light"] = light
        if bell_repeat is not None:
            reminder["bell_repeat"] = int(bell_repeat)
        if bell_duration_sec is not None:
            reminder["bell_duration_sec"] = float(bell_duration_sec)
        reminders.append(reminder)
        self._save(reminders)
        return reminder

    def update_reminder(
        self, reminder_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        reminders = self._load()
        updated_reminder = None
        for reminder in reminders:
            if str(reminder.get("id")) != str(reminder_id):
                continue
            if "time" in updates and isinstance(updates["time"], str):
                reminder["time"] = updates["time"]
            if "message" in updates and isinstance(updates["message"], str):
                reminder["message"] = updates["message"]
            if "recurring" in updates:
                reminder["recurring"] = bool(updates["recurring"])
            if "done" in updates:
                reminder["done"] = bool(updates["done"])
            if "speak_text" in updates and isinstance(updates["speak_text"], str):
                reminder["speak_text"] = updates["speak_text"]
            if "language" in updates and isinstance(updates["language"], str):
                reminder["language"] = updates["language"]
            if "emotion" in updates and isinstance(updates["emotion"], dict):
                reminder["emotion"] = updates["emotion"]
            if "light" in updates and isinstance(updates["light"], dict):
                reminder["light"] = updates["light"]
            if "bell_repeat" in updates and updates["bell_repeat"] is not None:
                reminder["bell_repeat"] = int(updates["bell_repeat"])
            if "bell_duration_sec" in updates and updates["bell_duration_sec"] is not None:
                reminder["bell_duration_sec"] = float(updates["bell_duration_sec"])
            updated_reminder = reminder
            break
        if updated_reminder is not None:
            self._save(reminders)
        return updated_reminder

    def delete_reminder(self, reminder_id: str) -> bool:
        reminders = self._load()
        before = len(reminders)
        reminders = [
            r for r in reminders if str(r.get("id")) != str(reminder_id)
        ]
        if len(reminders) != before:
            self._save(reminders)
            return True
        return False

    def clear_reminders(self) -> None:
        self._save([])

    def check_due(self, now: dt.datetime) -> List[Dict[str, Any]]:
        reminders = self._load()
        due: List[Dict[str, Any]] = []
        updated = False

        for reminder in reminders:
            if not isinstance(reminder, dict):
                continue
            self._ensure_done(reminder)
            if reminder.get("done"):
                continue

            time_value = reminder.get("time")
            if not isinstance(time_value, str):
                continue

            reminder_time = self._parse_time(time_value)
            if reminder_time is None:
                continue

            if reminder_time <= now:
                due.append(reminder)
                if reminder.get("recurring"):
                    next_time = self._next_daily_time(reminder_time)
                    reminder["time"] = self._format_time(next_time)
                    reminder["done"] = False
                else:
                    reminder["done"] = True
                updated = True

        if updated:
            self._save(reminders)

        return due
