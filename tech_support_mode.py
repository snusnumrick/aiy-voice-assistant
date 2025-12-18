#!/usr/bin/env python3
"""
Tech Support Mode - Early Startup VPN Activation

This script runs at the very beginning of the startup sequence (before network check)
to provide remote users with a way to enable Tailscale VPN for troubleshooting.

Usage:
    python tech_support_mode.py

Functionality:
- Blinks yellow LED for 5 seconds
- If button is pressed: starts 5-second hold timer
  - If button released within 5 seconds: accidental press, ignore
  - If button held for full 5 seconds: intentional, exits with code 100
- If not pressed: exits normally with code 0 to continue startup
- Run.sh checks the exit code and enables Tailscale immediately

Exit Codes:
    0 - Normal startup should continue
    100 - Tech support mode activated, run.sh should enable VPN
"""

import logging
import sys
import time

from aiy.board import Board, ButtonState
from aiy.leds import Leds, Color, Pattern

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_tech_support_mode():
    """
    Check if tech support mode should be activated.

    Returns:
        bool: True if tech support mode is activated, False otherwise
    """
    logger.info("=" * 60)
    logger.info("TECH SUPPORT MODE CHECK")
    logger.info("=" * 60)
    logger.info("If you need remote support, press and HOLD the button.")
    logger.info("You must hold it for 5 seconds to activate.")
    logger.info("Yellow LED will blink for 5 seconds...")
    logger.info("=" * 60)

    try:
        with Board() as board, Leds() as leds:
            # Set up blinking yellow LED pattern
            leds.pattern = Pattern.blink(500)  # Blink every 500ms
            leds.update(Leds.rgb_pattern(Color.YELLOW))

            logger.info("Monitoring button for 5 seconds...")

            # Monitor button state during blinking period
            tech_support_activated = False
            button_held = False
            check_interval = 0.1  # Check every 100ms
            max_wait_time = 5.0  # Wait up to 5 seconds
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                if board.button.state == ButtonState.PRESSED:
                    button_held = True
                    logger.info("Button detected as PRESSED")

                    # Start 5-second timer to verify intentional press
                    # If button is NOT released within 5 seconds, it's intentional
                    logger.info("Hold button for 5 seconds to activate tech support mode...")

                    # Switch to fast blinking to indicate hold period
                    leds.pattern = Pattern.blink(200)  # Fast blink every 200ms
                    leds.update(Leds.rgb_pattern(Color.YELLOW))

                    hold_start_time = time.time()
                    button_released = False

                    # Monitor for button release during the 5-second hold period
                    while (time.time() - hold_start_time) < 5.0:
                        if board.button.state == ButtonState.DEPRESSED:
                            button_released = True
                            logger.info("Button released - accidental press detected")

                            # Switch back to slow blinking
                            leds.pattern = Pattern.blink(500)
                            leds.update(Leds.rgb_pattern(Color.YELLOW))
                            break
                        time.sleep(check_interval)

                    # Check if button was held for full 5 seconds (intentional)
                    if not button_released and (time.time() - hold_start_time) >= 5.0:
                        logger.info("Button held for 5 seconds - intentional press detected")
                        tech_support_activated = True

                    # Exit the main monitoring loop
                    break
                time.sleep(check_interval)
                elapsed_time += check_interval

            if tech_support_activated:
                # Tech support mode activated
                logger.info("")
                logger.info("█" * 60)
                logger.info("█ TECH SUPPORT MODE ACTIVATED")
                logger.info("█" * 60)

                # Stop blinking and show solid yellow
                # Don't set pattern to None, as it causes an error when LED system tries to access period_ms
                # Instead, just update the LED channels to solid yellow
                leds.update(Leds.rgb_on(Color.YELLOW))

                # Wait for button to be released
                logger.info("Waiting for button release...")
                try:
                    board.button.wait_for_release(timeout=10)
                except Exception as e:
                    logger.warning(f"Timeout waiting for button release: {e}")

                logger.info("")
                logger.info("run.sh will now enable Tailscale VPN.")
                logger.info("Device will be accessible via VPN for remote support.")
                logger.info("█" * 60)

                # Exit with code 100 to indicate tech support mode was activated
                # run.sh will check this and enable Tailscale immediately
                sys.exit(100)
            else:
                # Normal startup - turn off LEDs and return False
                logger.info("Tech support mode not activated - continuing with normal startup")
                leds.update(Leds.rgb_off())

                # Brief pause for visual feedback
                time.sleep(0.5)

                # Exit normally - startup should continue
                sys.exit(0)

    except Exception as e:
        logger.error(f"Error in tech support mode check: {e}")
        # On error, turn off LEDs and continue with normal startup
        try:
            with Leds() as leds:
                leds.update(Leds.rgb_off())
        except:
            pass
        # On error, exit normally and let startup continue
        sys.exit(0)


def main():
    """Main entry point for tech support mode script."""
    logger.info("Tech Support Mode Handler Starting")

    # Check if tech support mode should be activated
    # This function will sys.exit() with appropriate code
    check_tech_support_mode()


if __name__ == "__main__":
    main()
