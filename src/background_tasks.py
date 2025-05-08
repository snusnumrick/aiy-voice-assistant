"""
Background tasks management module.

This module provides functionality for scheduling and managing background tasks
such as system maintenance, and cleaning.
"""

import datetime
import logging
from typing import Optional, Callable

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

    def set_cleaning_routine(self, routine: Callable) -> None:
        """Set the cleaning routine to be executed during maintenance."""
        self.cleaning_routine = routine

    async def check_and_run_tasks(self) -> None:
        """Check and run scheduled tasks if needed."""
        now = datetime.datetime.now()

        # Check if cleaning is needed
        if (
            self.cleaning_routine
            and self.cleaning_time_start <= now.time() < self.cleaning_time_stop
            and self.last_clean_date != now.date()
        ):
            logger.debug(f"Running cleaning routine on {now.date()}")
            await self.cleaning_routine()
            self.last_clean_date = now.date()
