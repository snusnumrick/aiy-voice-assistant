"""
Shared state module.

This module provides shared state objects that can be accessed across different
components of the application, such as button press state.
"""

import logging

logger = logging.getLogger(__name__)


class ButtonState:
    """
    Shared button press state that can be accessed across components.

    This class provides a thread-safe way to track and check button press state
    across DialogManager and tools. Tools can check button state during long
    operations (e.g., streaming API calls) to stop early when user presses button.

    Attributes:
        button_pressed (bool): Flag indicating if button is currently pressed.
    """

    def __init__(self):
        """Initialize the button state."""
        self.button_pressed = False
        logger.info("ButtonState initialized")

    def press(self):
        """
        Mark button as pressed.

        This should be called when hardware button is pressed.
        """
        self.button_pressed = True
        logger.info("Button marked as pressed")

    def reset(self):
        """
        Reset button state to not pressed.

        This should be called after processing a button press.
        """
        self.button_pressed = False
        logger.info("Button state reset")

    def __call__(self) -> bool:
        """
        Check if button is pressed.

        Returns:
            bool: True if button is pressed, False otherwise.
        """
        return self.button_pressed

    def __repr__(self):
        """Return string representation of button state."""
        return f"ButtonState(pressed={self.button_pressed})"
