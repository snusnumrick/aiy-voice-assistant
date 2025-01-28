"""
Background tasks management module.

This module provides functionality for scheduling and managing background tasks
such as system maintenance, cleaning, and Tailscale VPN state management.
"""

import datetime
import logging
import subprocess
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class TailscaleManager:
    """Manages Tailscale VPN state based on time of day."""

    def __init__(self, config):
        self.config = config
        self.enable_hour = self.config.get("tailscale_enable_hour", 22)  # 10 PM
        self.disable_hour = self.config.get("tailscale_disable_hour", 7)  # 7 AM
        self.last_state: Optional[bool] = None

    async def check_and_update_state(self) -> None:
        """Check current time and update Tailscale state if needed."""
        now = datetime.datetime.now()
        current_hour = now.hour

        # Determine desired state based on time
        should_be_enabled = (
            current_hour >= self.enable_hour or current_hour < self.disable_hour
        )

        # Check if state needs to be changed
        if self.last_state != should_be_enabled:
            try:
                if should_be_enabled:
                    logger.info("Enabling Tailscale")
                    subprocess.run(
                        ["sudo", "tailscale", "up"], check=True, capture_output=True
                    )
                else:
                    logger.info("Disabling Tailscale")
                    subprocess.run(
                        ["sudo", "tailscale", "down"], check=True, capture_output=True
                    )
                self.last_state = should_be_enabled
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Failed to {'enable' if should_be_enabled else 'disable'} Tailscale: {e}"
                )
            except Exception as e:
                logger.error(f"Unexpected error managing Tailscale: {e}")


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
        self.tailscale_manager = TailscaleManager(config)
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

        # Check Tailscale state
        await self.tailscale_manager.check_and_update_state()

        # Check if cleaning is needed
        if (
            self.cleaning_routine
            and self.cleaning_time_start <= now.time() < self.cleaning_time_stop
            and self.last_clean_date != now.date()
        ):
            logger.debug(f"Running cleaning routine on {now.date()}")
            await self.cleaning_routine()
            self.last_clean_date = now.date()
