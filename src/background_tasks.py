"""
Background tasks management module.

This module provides functionality for scheduling and managing background tasks
such as system maintenance, and cleaning.
"""

import datetime
import logging
from typing import Callable, Optional, Awaitable

from .reminders import ReminderManager

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Manages scheduled background tasks."""

    def __init__(self, config, timezone: Optional[str] = None):
        """
        Initialize the BackgroundTaskManager.

        Args:
            config: The application configuration object.
            timezone: Optional timezone string.
        """
        self.config = config
        self.timezone = timezone
        self.cleaning_routine: Optional[Callable] = None
        self.last_clean_date: Optional[datetime.date] = None
        self.cleaning_time_start = datetime.time(
            hour=self.config.get("cleaning_time_start_hour", 3)
        )
        self.cleaning_time_stop = datetime.time(
            hour=self.config.get("cleaning_time_stop_hour", 4)
        )
        self.reminder_notifier: Optional[Callable[[dict], Awaitable[None]]] = None
        self.reminders_enabled = self.config.get("reminders_enabled", True)
        self.reminder_check_interval_sec = self.config.get(
            "reminders_check_interval_sec", 60
        )
        self.last_reminder_check_ts: Optional[float] = None
        reminders_file = self.config.get("reminders_file", "reminders.json")
        self.reminder_manager = ReminderManager(reminders_file)

    def set_cleaning_routine(self, routine: Callable) -> None:
        """Set the cleaning routine to be executed during maintenance."""
        self.cleaning_routine = routine

    def set_reminder_notifier(self, notifier: Callable[[dict], Awaitable[None]]) -> None:
        """Set the reminder notification callback."""
        self.reminder_notifier = notifier

    async def check_and_run_tasks(self) -> None:
        """Check and run scheduled tasks if needed."""
        import pytz

        # Get the local timezone or use configured timezone
        if self.timezone:
            tz = pytz.timezone(self.timezone)
        else:
            # Use local timezone
            tz = datetime.datetime.now().astimezone().tzinfo

        # Get current time in the configured timezone
        now = datetime.datetime.now(tz)

        # Check if cleaning is needed
        if (
            self.cleaning_routine
            and self.cleaning_time_start <= now.time() < self.cleaning_time_stop
            and self.last_clean_date != now.date()
        ):
            logger.debug(f"Running cleaning routine on {now.date()} (timezone: {tz})")
            await self.cleaning_routine()
            self.last_clean_date = now.date()

        # Check reminders (rate-limited)
        if self.reminders_enabled and self.reminder_notifier:
            now_ts = now.timestamp()
            if (
                self.last_reminder_check_ts is None
                or now_ts - self.last_reminder_check_ts >= self.reminder_check_interval_sec
            ):
                self.last_reminder_check_ts = now_ts
                self.reminder_manager.timezone = tz
                due = self.reminder_manager.check_due(now)
                for reminder in due:
                    try:
                        await self.reminder_notifier(reminder)
                    except Exception as e:
                        logger.error(f"Error notifying reminder {reminder}: {e}")
