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
- Runs system diagnostics in background (network, VPN, SSH)
- If issues detected: shows LED diagnostic pattern
- Monitor for button press for up to 5 seconds
- If button pressed: start 5-second hold verification
  - If released within 5 seconds: accidental press, continue monitoring
  - If held for full 5 seconds: activate tech support mode

Tech Support Mode (VPN Active):
- LED: Solid yellow (indicates VPN is active)
- Tailscale VPN enabled
- SSH Readiness Diagnostics: Automatically checks 3 barriers to SSH:
  1. Network connectivity (ping test)
  2. VPN (Tailscale) connection status
  3. SSH service status (systemctl check)
- LED Diagnostic Patterns:
  * GREEN solid = All 3 barriers OK (SSH ready!)
  * RED 1 blink = Network problem
  * RED 2 blinks = VPN problem
  * RED 3 blinks = SSH service problem
- Monitors for button press to allow cancellation
- Press and HOLD button for 5 seconds to confirm cancellation
  - If released within 5 seconds: deactivation canceled, VPN remains active
  - If held for full 5 seconds: deactivate VPN and exit normally
- Or press Ctrl+C to disable VPN and exit

Behavior:
- Normal mode: Runs diagnostics, shows LED pattern if issues found, exits with code 0, LEDs turned off, startup continues
- Tech support mode: Script enables Tailscale, runs diagnostics, shows LED pattern, runs indefinitely
- Deactivation: Press and hold button for 5 seconds to disable VPN and exit normally
"""

import logging
import math
import struct
import subprocess
import sys
import time

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_beep_tone(frequency=800, duration_ms=200, sample_rate=44100, volume=0.5):
    """
    Generate a simple beep tone as WAV bytes.

    Args:
        frequency: Frequency in Hz (default: 800)
        duration_ms: Duration in milliseconds (default: 200)
        sample_rate: Sample rate in Hz (default: 44100)
        volume: Volume level 0.0-1.0 (default: 0.5)

    Returns:
        bytes: WAV audio data
    """
    # Calculate number of samples
    num_samples = int(sample_rate * duration_ms / 1000.0)

    # Generate sine wave samples
    samples = []
    for i in range(num_samples):
        # Generate sine wave
        sample_value = int(volume * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        # Convert to 16-bit signed integer (little endian)
        samples.append(struct.pack('<h', sample_value))

    # Create WAV data
    wav_data = b''.join(samples)

    # Add WAV header
    # Use simple header construction to avoid AudioFormat import
    num_bytes = len(wav_data)

    header = struct.pack('<4sI4s4sIHHIIHH4sI',
                        b'RIFF',           # ChunkID
                        36 + num_bytes,    # ChunkSize
                        b'WAVE',           # Format
                        b'fmt ',           # Subchunk1ID
                        16,                # Subchunk1Size (PCM)
                        1,                 # AudioFormat (PCM)
                        1,                 # NumChannels (mono)
                        sample_rate,       # SampleRate
                        sample_rate * 2,   # ByteRate (sample_rate * num_channels * bytes_per_sample)
                        2,                 # BlockAlign (num_channels * bytes_per_sample)
                        16,                # BitsPerSample
                        b'data',           # Subchunk2ID
                        num_bytes)         # Subchunk2Size

    return header + wav_data


def play_audio_cue(cue_type):
    """
    Play audio cue based on type.

    Args:
        cue_type: Type of cue ('start', 'pressed', 'confirmed', 'success', 'error')
    """
    # Import audio module only when needed (lazy loading for faster startup)
    from aiy.voice.audio import play_wav_async

    try:
        if cue_type == 'start':
            # Attention-getting sound: 3 short beeps
            for _ in range(3):
                play_wav_async(generate_beep_tone(frequency=1000, duration_ms=100))
                time.sleep(0.1)
        elif cue_type == 'pressed':
            # Single beep when button is detected as pressed
            play_wav_async(generate_beep_tone(frequency=800, duration_ms=150))
        elif cue_type == 'confirmed':
            # Confirmation sound: rising tone
            for freq in [600, 800, 1000]:
                play_wav_async(generate_beep_tone(frequency=freq, duration_ms=100))
                time.sleep(0.05)
        elif cue_type == 'success':
            # Success sound: two ascending beeps
            play_wav_async(generate_beep_tone(frequency=800, duration_ms=150))
            time.sleep(0.1)
            play_wav_async(generate_beep_tone(frequency=1000, duration_ms=150))
        elif cue_type == 'error':
            # Error sound: descending tone
            for freq in [800, 600, 400]:
                play_wav_async(generate_beep_tone(frequency=freq, duration_ms=150))
                time.sleep(0.05)
    except Exception as e:
        logger.warning(f"Audio cue playback failed: {e}")


def check_ssh_barriers(check_vpn=True):
    """
    Check the 3 critical barriers to SSH connectivity.

    Args:
        check_vpn: If True, check VPN status. If False, skip VPN check (for normal startup).

    Returns:
        tuple: (network_ok, vpn_ok, ssh_ok)
    """
    network_ok = False
    vpn_ok = False
    ssh_ok = False

    logger.info("")
    logger.info("=" * 60)
    logger.info("SSH READINESS DIAGNOSTICS")
    logger.info("=" * 60)

    # Check 1: Network connectivity
    logger.info("Checking network connectivity...")
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", "8.8.8.8"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            network_ok = True
            logger.info("  ✓ Network: OK")
        else:
            logger.info("  ✗ Network: FAILED")
    except Exception as e:
        logger.info(f"  ✗ Network: ERROR - {e}")

    # Check 2: VPN (Tailscale) status
    if check_vpn:
        logger.info("Checking VPN (Tailscale) status...")
        try:
            result = subprocess.run(
                ["tailscale", "status"],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout.strip()

            # Check if VPN is connected (status shows nodes)
            if result.returncode == 0 and output and "Tailscale is stopped" not in output:
                vpn_ok = True
                logger.info("  ✓ VPN: Connected")
                # Count nodes (each line is a node)
                node_count = len([line for line in output.split('\n') if line.strip()])
                logger.info(f"    Nodes on network: {node_count}")
            else:
                vpn_ok = False
                if "Tailscale is stopped" in output:
                    logger.info("  ✗ VPN: Tailscale is stopped")
                    logger.info("    Run 'sudo tailscale up' to enable")
                elif not output:
                    logger.info("  ✗ VPN: Not connected")
                    logger.info("    No nodes found - Tailscale may not be authenticated")
                else:
                    logger.info("  ✗ VPN: Not connected")
                if result.stderr:
                    logger.info(f"    {result.stderr.strip()}")
        except Exception as e:
            logger.info(f"  ✗ VPN: ERROR - {e}")
    else:
        logger.info("Skipping VPN check (not in tech support mode)")
        vpn_ok = False  # Don't count as failure when not checking

    # Check 3: SSH service status
    logger.info("Checking SSH service status...")
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "ssh"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip() == "active":
            ssh_ok = True
            logger.info("  ✓ SSH Service: Running")
        else:
            logger.info("  ✗ SSH Service: Not running")
    except Exception as e:
        logger.info(f"  ✗ SSH Service: ERROR - {e}")

    logger.info("=" * 60)

    # Determine overall status
    if network_ok and vpn_ok and ssh_ok:
        logger.info("SSH READY: All barriers passed - SSH connection possible!")
    elif not network_ok:
        logger.info("SSH BLOCKED: Network connectivity issue - fix network first")
    elif not vpn_ok and check_vpn:
        logger.info("SSH BLOCKED: VPN not connected - check Tailscale")
    elif not ssh_ok:
        logger.info("SSH ISSUE: SSH service not running - start with: sudo systemctl start ssh")
    else:
        logger.info("SSH READY: Core barriers passed")

    logger.info("=" * 60)
    logger.info("")

    return network_ok, vpn_ok, ssh_ok


def show_diagnostic_led_pattern(leds, network_ok, vpn_ok, ssh_ok):
    """
    Show LED pattern based on diagnostic results.

    Patterns:
    - 1 red blink = Network problem
    - 2 red blinks = VPN problem
    - 3 red blinks = SSH service problem
    - Green solid = All good

    Args:
        leds: Leds object
        network_ok: Network connectivity status
        vpn_ok: VPN connection status
        ssh_ok: SSH service status
    """
    if network_ok and vpn_ok and ssh_ok:
        # All good - Green solid
        logger.info("LED: GREEN solid (All systems ready)")
        leds.update(Leds.rgb_on(Color.GREEN))
    elif not network_ok:
        # 1 blink - Network problem
        logger.info("LED: RED 1 blink (Network problem)")
        for _ in range(15):  # Show for 15 seconds
            leds.update(Leds.rgb_on(Color.RED))
            time.sleep(0.2)
            leds.update(Leds.rgb_off())
            time.sleep(1.0)
    elif not vpn_ok:
        # 2 blinks - VPN problem
        logger.info("LED: RED 2 blinks (VPN problem)")
        for _ in range(15):
            for _ in range(2):
                leds.update(Leds.rgb_on(Color.RED))
                time.sleep(0.2)
                leds.update(Leds.rgb_off())
                time.sleep(0.3)
            time.sleep(0.7)
    elif not ssh_ok:
        # 3 blinks - SSH service problem
        logger.info("LED: RED 3 blinks (SSH service problem)")
        for _ in range(15):
            for _ in range(3):
                leds.update(Leds.rgb_on(Color.RED))
                time.sleep(0.2)
                leds.update(Leds.rgb_off())
                time.sleep(0.3)
            time.sleep(0.4)


def check_tech_support_mode():
    """
    Check if tech support mode should be activated.

    Returns:
        bool: True if tech support mode is activated, False otherwise
    """
    # Import hardware modules only when needed (lazy loading for faster startup)
    from aiy.board import Board, ButtonState
    from aiy.leds import Leds, Color, Pattern

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
            # Play audio cue to alert user that monitoring is starting
            play_audio_cue('start')
            logger.info("Audio cue played - start monitoring")

            # Set up blinking yellow LED pattern
            leds.pattern = Pattern.breathe(500)
            leds.update(Leds.rgb_pattern(Color.YELLOW))
            logger.info("yellow 500 breath")

            logger.info("Monitoring button for 5 seconds...")
            logger.info("Waiting for button press (hold for 5 seconds to activate tech support mode)...")
            logger.info("Note: System diagnostics will run AFTER tech support mode is activated")

            logger.info("Press and HOLD button to activate tech support mode...")

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
                    play_audio_cue('pressed')

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
                        play_audio_cue('confirmed')
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

                logger.info("Tailscale VPN is now active.")
                logger.info("Running SSH readiness diagnostics...")
                logger.info("█" * 60)

                # Run diagnostics to check SSH barriers (with VPN check enabled)
                network_ok, vpn_ok, ssh_ok = check_ssh_barriers(check_vpn=True)

                # Show LED pattern based on diagnostic results
                show_diagnostic_led_pattern(leds, network_ok, vpn_ok, ssh_ok)

                # Play audio cue based on diagnostic results
                if network_ok and vpn_ok and ssh_ok:
                    play_audio_cue('success')
                else:
                    play_audio_cue('error')

                # Pause to show diagnostic results before switching to VPN yellow
                time.sleep(3)

                # Switch back to solid yellow for VPN active state
                leds.update(Leds.rgb_on(Color.YELLOW))
                logger.info("")
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
                                play_audio_cue('confirmed')
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
