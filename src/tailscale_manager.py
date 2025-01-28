"""
Tailscale management module.

This module provides functionality for scheduling Tailscale enabling/disabling
based on time of day.
"""

import datetime
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class TailscaleManager:
    """
    Manages Tailscale VPN state based on time of day.
    Enables Tailscale during night hours and disables during day for better CPU performance.
    """

    def __init__(self, config):
        """
        Initialize the TailscaleManager.

        Args:
            config: The application configuration object.
        """
        self.config = config
        self.enable_hour = self.config.get("tailscale_enable_hour", 22)  # 10 PM
        self.disable_hour = self.config.get("tailscale_disable_hour", 7)  # 7 AM
        self.last_check_date: Optional[datetime.date] = None
        self.last_state: Optional[bool] = None

    async def check_and_update_state(self) -> None:
        """
        Check current time and update Tailscale state if needed.
        Enables Tailscale during night hours and disables during day.
        """
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

        self.last_check_date = now.date()
