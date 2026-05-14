import datetime as dt
import json
import os
import tempfile
import unittest

import pytz
from pydantic import ValidationError

from src.reminder import ReminderDraft, ReminderManager, ReminderUpdateRequest


class TestReminderManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.reminders_file = os.path.join(self.temp_dir.name, "reminders.json")
        self.timezone = pytz.timezone("UTC")
        self.manager = ReminderManager(self.reminders_file, timezone=self.timezone)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_reminder_normalizes_naive_time_to_timezone(self):
        reminder = self.manager.add_reminder(
            ReminderDraft.from_data(
                {
                    "message": "Call mom",
                    "when": "2026-04-23T09:30:00",
                    "fact_text": "Напомнил: позвонить маме.",
                }
            )
        )

        self.assertEqual(reminder.time.isoformat(), "2026-04-23T09:30:00+00:00")

        with open(self.reminders_file, encoding="utf-8") as f:
            payload = json.load(f)

        self.assertEqual(payload[0]["time"], "2026-04-23T09:30:00+00:00")
        self.assertEqual(payload[0]["fact_text"], "Напомнил: позвонить маме.")

    def test_load_skips_invalid_reminder_records(self):
        with open(self.reminders_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"id": "1", "time": "not-a-date", "message": "broken"},
                    {"id": "2", "time": "2026-04-23T09:30:00+00:00", "message": "ok"},
                ],
                f,
            )

        reminders = self.manager.list_reminders()

        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0].id, "2")
        self.assertEqual(reminders[0].message, "ok")

    def test_check_due_marks_non_recurring_reminder_done(self):
        self.manager.add_reminder(
            ReminderDraft.from_data(
                {
                    "message": "Stretch",
                    "when": "2026-04-23T09:30:00+00:00",
                }
            )
        )

        due = self.manager.check_due(dt.datetime(2026, 4, 23, 9, 31, tzinfo=self.timezone))

        self.assertEqual(len(due), 1)
        self.assertEqual(due[0].message, "Stretch")
        self.assertTrue(self.manager.list_reminders()[0].done)

    def test_cleanup_expired_reminders_removes_done_entries_only(self):
        with open(self.reminders_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": "1",
                        "time": "2026-04-23T09:30:00+00:00",
                        "message": "already fired",
                        "done": True,
                    },
                    {
                        "id": "2",
                        "time": "2026-04-23T09:30:00+00:00",
                        "message": "missed but still pending",
                        "done": False,
                    },
                    {
                        "id": "3",
                        "time": "2026-04-24T09:30:00+00:00",
                        "message": "future reminder",
                    },
                ],
                f,
            )

        removed = self.manager.cleanup_expired_reminders()

        self.assertEqual(removed, 1)
        self.assertEqual(
            [reminder.id for reminder in self.manager.list_reminders()],
            ["2", "3"],
        )


class TestReminderRequests(unittest.TestCase):
    def test_update_request_rejects_invalid_time_format(self):
        with self.assertRaises(ValidationError):
            ReminderUpdateRequest.from_data(
                {
                    "id": "123",
                    "time": "definitely-not-a-time",
                }
            )
