"""
Reminder management module.

Loads reminders from a JSON file, validates them, checks for due items, and
persists updates.
"""

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

from .models import ReminderDraft, ReminderRecord, ReminderUpdate

logger = logging.getLogger(__name__)


class ReminderManager:
    """Load, check, and update reminders stored in a JSON file."""

    def __init__(self, file_path: str, timezone: Optional[dt.tzinfo] = None):
        self.file_path = Path(file_path)
        self.timezone = timezone

    def _load(self) -> list[ReminderRecord]:
        if not self.file_path.exists():
            return []
        try:
            with self.file_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load reminders from {self.file_path}: {e}")
            return []

        reminders: list[ReminderRecord] = []
        if not isinstance(data, list):
            return reminders

        for raw_reminder in data:
            if not isinstance(raw_reminder, dict):
                continue
            try:
                reminders.append(ReminderRecord.from_data(raw_reminder))
            except Exception as e:
                logger.warning(f"Skipping invalid reminder from {self.file_path}: {e}")

        return reminders

    def _save(self, reminders: list[ReminderRecord]) -> None:
        payload = [reminder.to_storage() for reminder in reminders]
        try:
            with self.file_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reminders to {self.file_path}: {e}")

    def _normalize_datetime(self, value: dt.datetime) -> dt.datetime:
        if value.tzinfo is None:
            if self.timezone:
                try:
                    if hasattr(self.timezone, "localize"):
                        return self.timezone.localize(value)
                    return value.replace(tzinfo=self.timezone)
                except Exception:
                    return value
            return value

        if self.timezone:
            try:
                return value.astimezone(self.timezone)
            except Exception:
                return value
        return value

    def _next_daily_time(self, value: dt.datetime) -> dt.datetime:
        return value + dt.timedelta(days=1)

    def _next_id(self, reminders: list[ReminderRecord]) -> str:
        numeric_ids = []
        for reminder in reminders:
            if reminder.id.isdigit():
                numeric_ids.append(int(reminder.id))
        if numeric_ids:
            return str(max(numeric_ids) + 1)
        return str(int(dt.datetime.now().timestamp() * 1000))

    def list_reminders(self, include_done: bool = True) -> list[ReminderRecord]:
        reminders = self._load()
        if include_done:
            return reminders
        return [reminder for reminder in reminders if not reminder.done]

    def add_reminder(self, reminder: ReminderDraft) -> ReminderRecord:
        reminders = self._load()
        normalized_reminder = ReminderDraft.from_data(
            {
                **reminder.to_data(exclude_none=True),
                "when": self._normalize_datetime(reminder.when),
            }
        )
        created = normalized_reminder.to_record(self._next_id(reminders))
        reminders.append(created)
        self._save(reminders)
        return created

    def update_reminder(
        self, reminder_id: str, updates: ReminderUpdate
    ) -> Optional[ReminderRecord]:
        reminders = self._load()
        update_data = updates.to_data(exclude_none=True)
        updated_reminder = None
        for reminder in reminders:
            if reminder.id != str(reminder_id):
                continue
            for key, value in update_data.items():
                if key == "time" and isinstance(value, dt.datetime):
                    setattr(reminder, key, self._normalize_datetime(value))
                else:
                    setattr(reminder, key, value)
            updated_reminder = reminder
            break
        if updated_reminder is not None:
            self._save(reminders)
        return updated_reminder

    def delete_reminder(self, reminder_id: str) -> bool:
        reminders = self._load()
        before = len(reminders)
        reminders = [reminder for reminder in reminders if reminder.id != str(reminder_id)]
        if len(reminders) != before:
            self._save(reminders)
            return True
        return False

    def clear_reminders(self) -> None:
        self._save([])

    def check_due(self, now: dt.datetime) -> list[ReminderRecord]:
        now = self._normalize_datetime(now)
        reminders = self._load()
        due: list[ReminderRecord] = []
        updated = False

        for reminder in reminders:
            if reminder.done:
                continue

            reminder_time = self._normalize_datetime(reminder.time)
            if reminder_time <= now:
                due.append(reminder)
                if reminder.recurring:
                    next_time = self._next_daily_time(reminder_time)
                    reminder.time = next_time
                    reminder.done = False
                else:
                    reminder.done = True
                updated = True

        if updated:
            self._save(reminders)

        return due
