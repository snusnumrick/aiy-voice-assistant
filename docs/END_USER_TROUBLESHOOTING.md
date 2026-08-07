# End-User Troubleshooting

This guide covers the problems a person using the voice assistant can safely handle without
changing its software.

## LED guide

| LED behavior | Meaning | What to do |
| --- | --- | --- |
| Slowly breathing green | Ready | Press and hold the button while speaking. |
| Solid bright green | Recording | Keep holding the button and finish speaking. |
| Blinking dark green | Processing | Release the button and wait for the response. |
| Brief red blinking after a button press | Microphone or sound device is unavailable | Power-cycle the assistant using the steps below. |
| No light after a period of inactivity | Normal power-saving behavior | Press the button once to interact. |

The red warning appears only after someone tries to use the button. The assistant continues
checking the microphone in the background and returns to normal green behavior if it recovers.

## Red light after pressing the button

1. Disconnect the assistant's **power cable**.
2. Wait 10 seconds.
3. Reconnect the power cable.
4. Allow the assistant to finish starting.
5. When the LED breathes green, press and hold the button and try speaking again.

Do not remove or reseat the AIY Voice Bonnet while the Raspberry Pi is powered. If the Bonnet
needs to be physically checked, shut the Pi down and disconnect power first.

If red blinking returns after a full power cycle, contact the person who maintains the device.
Tell them that the microphone or ALSA sound card may be unavailable.

## The assistant does not react to the button

- Confirm that the assistant has power.
- Wait for the breathing green ready light.
- Press and hold the button rather than tapping it quickly.
- If the LED never becomes solid green, power-cycle the assistant.
- If the problem remains, the button, its wiring, or the sound device needs administrator
  diagnostics.

## The assistant records but does not answer

- Make sure the LED becomes solid green while the button is held.
- Speak close to the microphone and release the button after finishing.
- Wait while the LED blinks green during processing.
- Try once more after checking the internet connection.
- Power-cycle the assistant if it repeatedly returns to the ready state without speaking.

## The assistant answers but there is no sound

- Ask it to increase the volume, or have an administrator check the configured speaker volume.
- Power-cycle the assistant.
- If red appears when the button is pressed, follow the sound-device recovery steps above.
- If recording works without a red warning, the problem is likely on the playback or speaker
  side rather than the microphone side.

## The home Wi-Fi password changed

When the saved home Wi-Fi password no longer works, the assistant cannot reconnect to the
internet. Its Wi-Fi recovery service creates a temporary setup access point so the network
credentials can be updated without SSH.

1. If the assistant has just been powered on or restarted, allow up to three minutes for Wi-Fi
   recovery mode to start.
2. Open Wi-Fi settings on a phone, tablet, or computer near the assistant.
3. Connect to the temporary network named **`MusicServer`** using password **`music123`**.
4. If the phone warns that this network has no internet, choose the option to stay connected.
5. Wait for the **WiFi Setup** page to open automatically. This is a local page served by the
   assistant, so it does not need internet access.
6. Select the home Wi-Fi network, enter its **new Wi-Fi password**, and press **Connect**.
   WPA/WPA2 passwords must contain 8–63 characters.
7. When the page says that a reboot is required, press **Reboot Now**. Do not disconnect power
   while the reboot is in progress.
8. The `MusicServer` network will disappear while the assistant reboots and reconnects to the
   home Wi-Fi. Reconnect the phone or computer to its usual network.
9. Allow up to three minutes for startup. When the assistant returns to its normal ready
   indication, press the button and try a simple request that requires the internet.

Only enter the home network name and Wi-Fi password in this setup page. The page should never
ask for the assistant's API keys, email password, SSH password, or other account credentials.

If the temporary setup network does not appear:

- Confirm that three minutes have passed since startup.
- Power-cycle the assistant, wait up to three minutes, and check the available networks again.
- Move the phone or computer closer to the assistant.
- Temporarily disable VPN and automatic switching to mobile data or another Wi-Fi network.
- Ask the administrator to verify that the `autohotspot` service is running.

If the setup network connects but the page does not open:

- Accept any `network has no internet` warning and remain connected.
- Turn mobile data and VPN off temporarily, then reconnect to the setup network.
- Open the system's `sign in to network` or captive-portal notification if one appears.
- In a browser, try **`http://cubie:5000/setup-wifi`**.
- On devices where `.local` names work, also try **`http://cubie.local:5000/setup-wifi`**.
- If neither address opens, ask the administrator for the current hotspot IP. The normal
  fallback is `192.168.4.1`, but an installation can retain a different address.

If the old home Wi-Fi network still exists but only its password changed, make sure the correct
network name is selected before saving. After a successful update, the recovery access point
should turn off automatically.

## Administrator checks

Connect over SSH and confirm that Linux detects the AIY Voice Bonnet:

```bash
arecord -l
aplay -l
```

A working device should list `aiyvoicebonnet` for both capture and playback. If the commands
report `no soundcards found`, reboot the device and check again:

```bash
sudo reboot
```

To test microphone capture, first stop the assistant so it does not hold the audio device:

```bash
sudo systemctl stop aiy.service
timeout 8s arecord -vv -D default -t wav -c 1 -f S16_LE -r 16000 -d 5 /tmp/mic-test.wav
ls -l /tmp/mic-test.wav
aplay /tmp/mic-test.wav
sudo systemctl start aiy.service
```

The WAV should be larger than 44 bytes and should contain audible microphone input. A 44-byte
WAV contains only an empty header.

Useful log messages include:

```text
Button state at recorder start: DEPRESSED
Button state transition: DEPRESSED -> PRESSED
Recording audio... LED solid
Button state transition: PRESSED -> DEPRESSED
STT audio stream complete: chunks=... bytes=... duration=...
Microphone stream ended before button press; arecord_exit_code=...
```

For temporary recording diagnostics, enable `stt_debug_recording_enabled` in `user.json`.
Recordings are written under `logs/` and may contain private speech. Disable the option after
troubleshooting.
