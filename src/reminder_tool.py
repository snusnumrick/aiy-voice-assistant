"""
Reminder tool module.

Provides tools to set, list, update, and delete reminders stored in reminders.json.
"""

import datetime as dt
import logging
from typing import Any, Dict, Optional

import pytz

from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
from src.reminders import ReminderManager
from src.tools import get_timezone

logger = logging.getLogger(__name__)


class ReminderTool:
    """Tool for managing reminders via tool calls."""

    def __init__(self, config: Config, timezone: str):
        self.config = config
        self.reminders_file = config.get("reminders_file", "reminders.json")
        self.tz_name = timezone
        self.timezone: dt.tzinfo = pytz.timezone(timezone)

    def tool_definitions(self) -> list[Tool]:
        manage_tool = Tool(
            name="manage_reminders",
            description="""
Manage reminders stored in reminders.json.

Actions:
- set_reminder_at: Set a reminder at an absolute ISO timestamp.
- set_reminder_in: Set a reminder relative to now (amount + unit).
- update_reminder: Update message/time/recurring/done for a reminder by id.
- delete_reminder: Delete a reminder by id.
- clear_reminders: Remove all reminders.

Use set_reminder_at for absolute timestamps, and set_reminder_in for relative durations.
Return concise confirmations and IDs.

IMPORTANT: Only create reminders when explicitly requested by the user.
Never create reminders for internal AI tasks, housekeeping, or self-management
(e.g., deleting facts, cleaning history, processing data).
            """,
            iterative=False,
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description=(
                        "Action to perform: set_reminder_at, set_reminder_in, "
                        "update_reminder, delete_reminder, clear_reminders"
                    ),
                ),
                ToolParameter(
                    name="message",
                    type="string",
                    description="Reminder message (required for set/update).",
                ),
                ToolParameter(
                    name="speak_text",
                    type="string",
                    description='Text to speak when reminder fires (optional). Example: "Напомнил: позвонить маме"',
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description='Language code for spoken reminder (ru/en/de). Example: "ru"',
                ),
                ToolParameter(
                    name="fact_text",
                    type="string",
                    description='Text to store in facts when reminder fires (optional). Example: "Напомнил: позвонить маме."',
                ),
                ToolParameter(
                    name="time",
                    type="string",
                    description="ISO timestamp for set/update. Example: 2026-02-05T14:30:00+03:00",
                ),
                ToolParameter(
                    name="amount",
                    type="integer",
                    description="Relative amount for set_reminder_in (e.g., 10).",
                ),
                ToolParameter(
                    name="unit",
                    type="string",
                    description="Relative unit for set_reminder_in: seconds, minutes, hours, days.",
                ),
                ToolParameter(
                    name="recurring",
                    type="boolean",
                    description="Whether the reminder repeats daily.",
                ),
                ToolParameter(
                    name="light",
                    type="object",
                    description=(
                        "Optional light dict: {'color':[R,G,B],'behavior':'continuous/blinking/breathing',"
                        "'brightness':'dark/medium/bright','period':seconds}"
                    ),
                ),
                ToolParameter(
                    name="emotion",
                    type="object",
                    description=(
                        "Optional emotion dict in the same format as $emotion; "
                        "used for both LED behavior and voice tone."
                    ),
                ),
                ToolParameter(
                    name="bell_repeat",
                    type="integer",
                    description="Repeat bell N times (optional).",
                ),
                ToolParameter(
                    name="bell_duration_sec",
                    type="number",
                    description="Total bell duration in seconds (optional).",
                ),
                ToolParameter(
                    name="id",
                    type="string",
                    description="Reminder id for update/delete.",
                ),
                ToolParameter(
                    name="done",
                    type="boolean",
                    description="Set done flag when updating.",
                ),
            ],
            processor=self.manage_reminders,
            required=["action"],
        )
        list_tool = Tool(
            name="list_reminders",
            description="List reminders stored in reminders.json.",
            iterative=True,
            parameters=[
                ToolParameter(
                    name="include_done",
                    type="boolean",
                    description="Include completed reminders. Example: true",
                )
            ],
            processor=self.list_reminders,
            required=[],
        )
        return [manage_tool, list_tool]

    async def manage_reminders(self, parameters: Dict[str, Any]) -> str:
        action = parameters.get("action")
        manager = ReminderManager(self.reminders_file, timezone=self.timezone)

        if action == "set_reminder_at":
            message = parameters.get("message")
            speak_text = parameters.get("speak_text")
            language = parameters.get("language")
            fact_text = parameters.get("fact_text")
            time_value = parameters.get("time")
            recurring = bool(parameters.get("recurring", False))
            light = parameters.get("light")
            emotion = parameters.get("emotion")
            if isinstance(emotion, dict) and isinstance(emotion.get("light"), dict):
                light = emotion.get("light")
            bell_repeat = parameters.get("bell_repeat")
            bell_duration_sec = parameters.get("bell_duration_sec")
            if not message or not time_value:
                return "Missing message or time."
            when = manager._parse_time(str(time_value))
            if when is None:
                return "Invalid time format."
            reminder = manager.add_reminder(
                str(message),
                when,
                recurring,
                speak_text=str(speak_text) if speak_text is not None else None,
                language=str(language) if language is not None else None,
                emotion=emotion if isinstance(emotion, dict) else None,
                fact_text=str(fact_text) if fact_text is not None else None,
                light=light if isinstance(light, dict) else None,
                bell_repeat=bell_repeat,
                bell_duration_sec=bell_duration_sec,
            )
            return f"Reminder set: id={reminder['id']}, time={reminder['time']}"

        if action == "set_reminder_in":
            message = parameters.get("message")
            speak_text = parameters.get("speak_text")
            language = parameters.get("language")
            fact_text = parameters.get("fact_text")
            amount = parameters.get("amount")
            unit = parameters.get("unit")
            recurring = bool(parameters.get("recurring", False))
            light = parameters.get("light")
            emotion = parameters.get("emotion")
            if isinstance(emotion, dict) and isinstance(emotion.get("light"), dict):
                light = emotion.get("light")
            bell_repeat = parameters.get("bell_repeat")
            bell_duration_sec = parameters.get("bell_duration_sec")
            if not message or amount is None or not unit:
                return "Missing message, amount, or unit."
            try:
                amount_int = int(amount)
            except Exception:
                return "Invalid amount."
            unit = str(unit).lower()
            seconds = {
                "second": 1,
                "seconds": 1,
                "minute": 60,
                "minutes": 60,
                "hour": 3600,
                "hours": 3600,
                "day": 86400,
                "days": 86400,
            }.get(unit)
            if seconds is None:
                return "Invalid unit. Use seconds, minutes, hours, or days."
            now = dt.datetime.now(self.timezone)
            when = now + dt.timedelta(seconds=amount_int * seconds)
            reminder = manager.add_reminder(
                str(message),
                when,
                recurring,
                speak_text=str(speak_text) if speak_text is not None else None,
                language=str(language) if language is not None else None,
                emotion=emotion if isinstance(emotion, dict) else None,
                fact_text=str(fact_text) if fact_text is not None else None,
                light=light if isinstance(light, dict) else None,
                bell_repeat=bell_repeat,
                bell_duration_sec=bell_duration_sec,
            )
            return f"Reminder set: id={reminder['id']}, time={reminder['time']}"

        if action == "update_reminder":
            reminder_id = parameters.get("id")
            if not reminder_id:
                return "Missing reminder id."
            updates: Dict[str, Optional[Any]] = {}
            if "message" in parameters:
                updates["message"] = parameters.get("message")
            if "speak_text" in parameters:
                updates["speak_text"] = parameters.get("speak_text")
            if "language" in parameters:
                updates["language"] = parameters.get("language")
            if "fact_text" in parameters:
                updates["fact_text"] = parameters.get("fact_text")
            if "time" in parameters:
                updates["time"] = parameters.get("time")
            if "recurring" in parameters:
                updates["recurring"] = parameters.get("recurring")
            if "done" in parameters:
                updates["done"] = parameters.get("done")
            if "light" in parameters:
                updates["light"] = parameters.get("light")
            if "emotion" in parameters:
                emotion = parameters.get("emotion")
                if isinstance(emotion, dict) and isinstance(emotion.get("light"), dict):
                    updates["light"] = emotion.get("light")
                updates["emotion"] = emotion
            if "bell_repeat" in parameters:
                updates["bell_repeat"] = parameters.get("bell_repeat")
            if "bell_duration_sec" in parameters:
                updates["bell_duration_sec"] = parameters.get("bell_duration_sec")
            if not updates:
                return "No updates provided."
            updated = manager.update_reminder(str(reminder_id), updates)
            if not updated:
                return "Reminder not found."
            return f"Reminder updated: id={updated.get('id')}"

        if action == "delete_reminder":
            reminder_id = parameters.get("id")
            if not reminder_id:
                return "Missing reminder id."
            if manager.delete_reminder(str(reminder_id)):
                return f"Reminder deleted: id={reminder_id}"
            return "Reminder not found."

        if action == "clear_reminders":
            manager.clear_reminders()
            return "All reminders cleared."

        return "Unknown action."

    async def list_reminders(self, parameters: Dict[str, Any]) -> str:
        manager = ReminderManager(self.reminders_file, timezone=self.timezone)
        include_done = parameters.get("include_done", True)
        reminders = manager.list_reminders(include_done=bool(include_done))
        if not reminders:
            return "No reminders."
        lines = []
        for reminder in reminders:
            rid = reminder.get("id")
            time_value = reminder.get("time")
            message = reminder.get("message")
            recurring = reminder.get("recurring")
            done = reminder.get("done", False)
            lines.append(
                f"id={rid} time={time_value} recurring={recurring} done={done} message={message}"
            )
        return "\n".join(lines)
