#!/usr/bin/env python3
"""
Tech Support Mode - Early Startup VPN Activation

This script runs at the very beginning of the startup sequence (before network check)
to provide remote users with a way to enable Tailscale VPN for troubleshooting.

Usage:
    python tech_support_mode.py

Functionality:

Activation (during initial monitoring):
- LED: Slow yellow breathing (breathe pattern)
- Monitor for button press for up to 5 seconds
- If button pressed: start 5-second hold verification
  - If released within 5 seconds: accidental press, continue monitoring
  - If held for full 5 seconds: activate tech support mode

Tech Support Mode (VPN Active):
- LED: Solid yellow (indicates VPN is active)
- Tailscale VPN enabled
- Monitors for button press to allow cancellation
- Press and HOLD button for 5 seconds to confirm cancellation
  - If released within 5 seconds: deactivation canceled, VPN remains active
  - If held for full 5 seconds: deactivate VPN and exit normally
- Or press Ctrl+C to disable VPN and exit

Behavior:
- Normal mode: Script exits with code 0, LEDs turned off, startup continues
- Tech support mode: Script enables Tailscale, keeps LED on, runs indefinitely
- Deactivation: Press and hold button for 5 seconds to disable VPN and exit normally
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
        # Initialize Board and LEDs
        board = Board()
        leds = Leds()

        try:
            # Set up blinking yellow LED pattern
            leds.pattern = Pattern.breathe(500)
            leds.update(Leds.rgb_pattern(Color.YELLOW))
            logger.info("yellow 500 breath")

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
                    leds.pattern = Pattern.blink(500)
                    leds.update(Leds.rgb_pattern(Color.YELLOW))
                    logger.info("yellow 500 blink")

                    hold_start_time = time.time()
                    button_released = False

                    # Monitor for button release during the 5-second hold period
                    while (time.time() - hold_start_time) < 5.0:
                        if board.button.state == ButtonState.DEPRESSED:
                            button_released = True
                            logger.info("Button released - accidental press detected")

                            # Switch back to slow blinking
                            leds.pattern = Pattern.breathe(500)
                            leds.update(Leds.rgb_pattern(Color.YELLOW))
                            logger.info("yellow 500 breath")
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
                logger.info("yellow solid")

                # Wait for button to be released
                logger.info("Waiting for button release...")
                try:
                    board.button.wait_for_release(timeout=10)
                except Exception as e:
                    logger.warning(f"Timeout waiting for button release: {e}")

                logger.info("")
                logger.info("Enabling Tailscale VPN...")
                logger.info("█" * 60)

                # Enable Tailscale VPN
                try:
                    import subprocess
                    logger.info("Executing: sudo tailscale up")
                    result = subprocess.run(
                        ["sudo", "tailscale", "up"],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.info("Tailscale VPN enabled successfully!")
                    if result.stdout:
                        logger.info(f"Output: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to enable Tailscale: {e}")
                    if e.stderr:
                        logger.error(f"Error output: {e.stderr}")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")

                logger.info("")
                logger.info("Tailscale VPN is now active.")
                logger.info("Device is accessible via VPN for remote support.")
                logger.info("LED will remain SOLID YELLOW to indicate VPN is active.")
                logger.info("Press and HOLD the button for 5 seconds to cancel and return to normal startup.")
                logger.info("Or press Ctrl+C to disable VPN and exit.")
                logger.info("█" * 60)

                # Run indefinitely - keep LED on and process alive
                # This ensures the LED stays yellow and the script doesn't exit
                vpn_active = True
                deactivation_in_progress = False

                try:
                    while vpn_active:
                        time.sleep(1)  # Check every second for button press or timeout

                        # Check if button is pressed during VPN active state (start deactivation)
                        if board.button.state == ButtonState.PRESSED and not deactivation_in_progress:
                            logger.info("")
                            logger.info("Button pressed - starting deactivation")
                            logger.info("Hold button for 5 seconds to confirm cancellation...")

                            # Switch to fast blinking during deactivation hold
                            leds.pattern = Pattern.blink(200)
                            leds.update(Leds.rgb_pattern(Color.YELLOW))

                            deactivation_in_progress = True
                            hold_start_time = time.time()
                            deactivation_confirmed = False

                            # Monitor for button release during the 5-second deactivation hold
                            while deactivation_in_progress:
                                elapsed = time.time() - hold_start_time

                                if elapsed >= 5.0:
                                    # Button held for full 5 seconds - confirmed deactivation
                                    deactivation_confirmed = True
                                    break

                                if board.button.state == ButtonState.DEPRESSED:
                                    # Button released early - cancel deactivation
                                    logger.info("Button released - deactivation canceled")
                                    logger.info("Resuming VPN active state")

                                    # Switch back to solid yellow
                                    leds.update(Leds.rgb_on(Color.YELLOW))
                                    deactivation_in_progress = False
                                    break

                                time.sleep(0.1)

                            # Check if deactivation was confirmed
                            if deactivation_confirmed:
                                logger.info("Button held for 5 seconds - deactivation confirmed")
                                logger.info("Disabling Tailscale VPN...")
                                try:
                                    subprocess.run(["sudo", "tailscale", "down"], check=False)
                                except:
                                    pass
                                logger.info("Tailscale VPN disabled.")

                                logger.info("Cleaning up resources...")
                                board.close()
                                leds.reset()

                                logger.info("Returning to normal startup...")
                                logger.info("Exiting...")
                                # Exit normally to allow startup to continue
                                sys.exit(0)

                        # Log status every 60 seconds to show we're still running
                        if int(time.time()) % 60 == 0:
                            logger.info(f"Tailscale VPN active - {time.ctime()}")

                except KeyboardInterrupt:
                    logger.info("")
                    logger.info("Shutting down Tailscale VPN...")
                    try:
                        subprocess.run(["sudo", "tailscale", "down"], check=False)
                    except:
                        pass
                    logger.info("Tailscale VPN disabled.")
                    logger.info("Cleaning up resources...")
                    # Properly close Board and LEDs before exit
                    board.close()
                    leds.reset()
                    logger.info("Exiting...")
                    # Use os._exit to force immediate exit and avoid threading cleanup issues
                    import os
                    os._exit(0)
            else:
                # Normal startup - turn off LEDs
                logger.info("Tech support mode not activated - continuing with normal startup")
                leds.update(Leds.rgb_off())
                logger.info("LED turned OFF")

                # Brief pause for visual feedback
                time.sleep(0.5)

                # Cleanup resources
                board.close()
                leds.reset()

                # Exit normally - startup should continue
                sys.exit(0)
        except Exception as inner_e:
            # Cleanup on error
            board.close()
            leds.reset()
            raise inner_e

    except Exception as e:
        logger.error(f"Error in tech support mode check: {e}")
        # On error, turn off LEDs and continue with normal startup
        try:
            with Leds() as leds:
                leds.update(Leds.rgb_off())
                logger.info("LED turned OFF")
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
