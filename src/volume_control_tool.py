"""
VolumeControlTool Module

This module provides a tool for controlling speaker volume in response to user commands.
It allows for increasing, decreasing, or setting the volume to a specific level using
the 'amixer' command-line utility.

Classes:
    VolumeControlTool: A tool for controlling speaker volume.

Dependencies:
    - subprocess: For running shell commands
    - logging: For logging errors and information
    - typing: For type hinting
    - src.ai_models_with_tools: For Tool and ToolParameter classes
    - src.config: For Config class
"""

import logging
import subprocess
from typing import Dict, List, Optional
from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config

logger = logging.getLogger(__name__)

class VolumeControlTool:
    """
    A tool for controlling the speaker volume adaptively.

    This class provides methods to adjust the speaker volume using available system controls.
    It automatically detects and selects an appropriate volume control (e.g., 'Master' or 'Speaker')
    and can increase, decrease, or set the volume to a specific level.

    Attributes:
        config (Config): Configuration object containing settings.
        min_volume (int): Minimum allowable volume level.
        max_volume (int): Maximum allowable volume level.
        step (int): Default step size for volume adjustments.
        available_controls (List[str]): List of available volume controls on the system.
        current_control (Optional[str]): The currently selected volume control.
    """

    def __init__(self, config: Config):
        """
        Initialize the VolumeControlTool.

        Args:
            config (Config): Configuration object containing settings for the tool.
        """
        self.config = config
        self.min_volume = config.get("min_volume", 0)
        self.max_volume = config.get("max_volume", 100)
        self.step = config.get("volume_step", 10)
        self.available_controls = self._get_available_controls()
        self.current_control = self._select_control()
        logger.info(f"Using volume control: {self.current_control}")

    def _get_available_controls(self) -> List[str]:
        """
        Retrieve a list of available volume controls on the system.

        Returns:
            List[str]: A list of available volume control names.
        """
        try:
            result = subprocess.run(['amixer', 'scontrols'], capture_output=True, text=True)
            controls = [line.split("'")[1] for line in result.stdout.splitlines() if "'" in line]
            return controls
        except Exception as e:
            logger.error(f"Failed to get available controls: {str(e)}")
            return []

    def _select_control(self) -> Optional[str]:
        """
        Select an appropriate volume control from the available options.

        Prefers 'Master' or 'Speaker' controls if available, otherwise selects the first available control.

        Returns:
            Optional[str]: The name of the selected control, or None if no controls are available.
        """
        preferred_controls = ['Master', 'Speaker']
        for control in preferred_controls:
            if control in self.available_controls:
                return control
        if self.available_controls:
            return self.available_controls[0]
        return None

    def tool_definition(self) -> Tool:
        """
        Define the tool interface for the AI model.

        Returns:
            Tool: A Tool object describing the volume control functionality.
        """
        return Tool(
            name="control_speaker_volume",
            description="""
Control the speaker volume based on user requests. Use this tool when the user asks to change the volume,
even if they don't explicitly mention 'volume'. Common requests include:
- Increase volume: "говори погромче" (speak louder), "сделай громче" (make it louder)
- Decrease volume: "говори потише" (speak softer), "сделай тише" (make it quieter)
- Set specific volume: "установи громкость на 50 процентов" (set volume to 50 percent)

For non-specific requests:
- Use 'increase' or 'decrease' with the default step size.
- For specific volume requests, use the 'set' action with the requested percentage.

Always use this tool when the user's request implies a volume change.
Returns the new volume level after adjustment.
            """,
            iterative=True,
            parameters=[
                ToolParameter(
                    name='action',
                    type='string',
                    description='Action to perform: "increase", "decrease", or "set"'
                ),
                ToolParameter(
                    name='value',
                    type='integer',
                    description='Volume value (0-100) for "set" action, or step size for increase/decrease. Use default step size if not specified.'
                )
            ],
            processor=self.adjust_volume,
            required=['action']
        )

    async def adjust_volume(self, parameters: Dict[str, any]) -> str:
        """
        Adjust the volume based on the provided parameters.

        This method interprets the action and value from the parameters and adjusts
        the volume accordingly. It handles increasing, decreasing, and setting the volume.

        Args:
            parameters (Dict[str, any]): A dictionary containing 'action' and optionally 'value'.

        Returns:
            str: A message indicating the result of the volume adjustment.

        Raises:
            Exception: If there's an error in adjusting the volume.
        """
        if not self.current_control:
            return "Volume control is not available on this system."

        action = parameters.get("action", "").lower()
        value = parameters.get("value", self.step)

        try:
            current_volume = self.get_current_volume()

            if action == "increase":
                new_volume = min(current_volume + value, self.max_volume)
            elif action == "decrease":
                new_volume = max(current_volume - value, self.min_volume)
            elif action == "set":
                new_volume = max(min(value, self.max_volume), self.min_volume)
            else:
                return f"Invalid action: {action}. Use 'increase', 'decrease', or 'set'. Current volume: {current_volume}%"

            self.set_volume(new_volume)
            return f"Volume adjusted to {new_volume}%"
        except Exception as e:
            logger.error(f"An error occurred while adjusting volume: {str(e)}")
            return f"Failed to adjust volume: {str(e)}"

    def get_current_volume(self) -> int:
        """
        Get the current volume level using the selected control.

        Returns:
            int: The current volume level as a percentage (0-100).

        Raises:
            Exception: If there's an error in fetching the current volume.
        """
        try:
            result = subprocess.run(['amixer', 'get', self.current_control], capture_output=True, text=True)
            volume = int(result.stdout.split('[')[1].split('%')[0])
            return volume
        except Exception as e:
            logger.error(f"Failed to get current volume: {str(e)}")
            return 50  # Return a default value if unable to get the current volume

    def set_volume(self, volume: int):
        """
        Set the volume to a specific level using the selected control.

        Args:
            volume (int): The volume level to set (0-100).

        Raises:
            subprocess.CalledProcessError: If the 'amixer' command fails.
        """
        try:
            subprocess.run(['amixer', 'set', self.current_control, f'{volume}%'], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set volume: {str(e)}")
            raise

# Example usage:
# config = Config()
# volume_tool = VolumeControlTool(config)
# result = asyncio.run(volume_tool.adjust_volume({"action": "increase", "value": 10}))
# print(result)