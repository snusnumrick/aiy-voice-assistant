# ruff: noqa: E402

import logging
import queue
import sys
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# Set up logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import and use the custom mock_aiy
from mock_aiy import mock_aiy

# Use the custom mock_aiy
sys.modules['aiy'] = mock_aiy
sys.modules['aiy.board'] = mock_aiy.board
sys.modules['aiy.leds'] = mock_aiy.leds
sys.modules['aiy.voice'] = mock_aiy.voice
sys.modules['aiy.voice.audio'] = mock_aiy.voice.audio

# Mock optional runtime dependencies imported by src.tools
mock_pydub = MagicMock()
mock_pydub.AudioSegment = MagicMock()
sys.modules['aiofiles'] = MagicMock()
sys.modules['geocoder'] = MagicMock()
sys.modules['pydub'] = mock_pydub

# Now it's safe to import from src
from src.responce_player import (
    MergeItem,
    ResponsePlayer,
    adjust_light_for_audio_duration,
    adjust_rgb_brightness,
    emotions_prompt,
    extract_emotions,
    extract_language,
    get_wav_duration_seconds,
    language_prompt,
)


class TestResponsePlayer(unittest.TestCase):
    def setUp(self):
        self.leds = MagicMock()
        self.timezone = "UTC"
        self.playlist = []
        self.player = ResponsePlayer(self.playlist, self.leds, self.timezone)

    def test_init(self):
        self.assertIsInstance(self.player.playlist, queue.Queue)
        self.assertIsInstance(self.player.merge_queue, queue.Queue)
        self.assertEqual(self.player.timezone, self.timezone)
        self.assertEqual(self.player.leds, self.leds)
        self.assertFalse(self.player._should_play)
        self.assertFalse(self.player._stopped)

    @patch('src.responce_player.threading.Thread')
    def test_add(self, mock_thread):
        playitem = ({"light": {"color": [255, 0, 0], "brightness": "medium", "behavior": "continuous"}}, "test.wav",
                    "Test text")
        self.player.add(playitem)
        self.assertEqual(self.player.merge_queue.qsize(), 1)
        expected_calls = [call(target=self.player._merge_audio_files), call().start(),
            call(target=self.player._play_sequence), call().start()]
        mock_thread.assert_has_calls(expected_calls, any_order=True)

    def test_change_light_behavior(self):
        # Reset the mock to clear the call from the constructor
        self.leds.reset_mock()

        behaviour = {"color": [255, 0, 0], "behavior": "continuous", "brightness": "medium", "period": 1}
        with patch.object(self.player, 'current_light', None):
            self.player.change_light_behavior(behaviour)
            self.leds.update.assert_called_once()

        # Check that the LED was updated with the correct color
        expected_color = (178, 0, 0)  # This is the adjusted color for [255, 0, 0] with "medium" brightness
        self.leds.update.assert_called_with(expected_color)

    @patch('src.responce_player.combine_audio_files')
    @patch('tempfile.mktemp')
    def test_process_wav_list(self, mock_mktemp, mock_combine):
        mock_mktemp.return_value = "temp.wav"
        self.player.wav_list = [("file1.wav", "Text 1"), ("file2.wav", "Text 2")]
        self.player.wav_list_light = {"color": [255, 0, 0], "brightness": "medium", "behavior": "continuous"}
        self.player._process_wav_list()
        mock_combine.assert_called_once()
        self.assertEqual(self.player.playlist.qsize(), 1)

    def test_merge_audio_files_keeps_light_reset_separate(self):
        colored_light = {
            "color": [0, 0, 255],
            "brightness": "medium",
            "behavior": "continuous",
            "period": 1,
        }
        self.player.merge_queue.put(MergeItem(light=colored_light, filename="blue.wav", text="Blue"))
        self.player.merge_queue.put(MergeItem(light=None, filename="plain.wav", text="Plain"))

        self.player._merge_audio_files()

        self.assertEqual(self.player.playlist.qsize(), 2)
        self.assertEqual(self.player.playlist.get_nowait(), (colored_light, "blue.wav"))
        self.assertEqual(self.player.playlist.get_nowait(), (None, "plain.wav"))

    @patch('src.responce_player.play_wav_async')
    def test_play_sequence(self, mock_play_async):
        self.player.playlist.put(({"color": [255, 0, 0], "brightness": "medium", "behavior": "continuous"}, "test.wav"))
        self.player._should_play = True

        def run_play_sequence():
            logger.debug("Starting _play_sequence")
            self.player._play_sequence()
            logger.debug("Finished _play_sequence")

        play_thread = threading.Thread(target=run_play_sequence)
        play_thread.start()

        # Wait for a short time to allow _play_sequence to start
        time.sleep(0.1)

        # Check that play_wav_async was called
        mock_play_async.assert_called_once_with("test.wav")

        # Add another item to the playlist
        self.player.playlist.put(
            ({"color": [0, 255, 0], "brightness": "medium", "behavior": "continuous"}, "test2.wav"))

        # Notify the condition variable to wake up the waiting thread
        with self.player.condition:
            self.player.condition.notify()

        # Wait a short time for the second item to be processed
        time.sleep(0.1)

        # Stop the player
        self.player._should_play = False

        # Notify the condition variable again to wake up the waiting thread
        with self.player.condition:
            self.player.condition.notify()

        # Wait for the play_thread to finish
        play_thread.join(timeout=1.0)

        self.assertFalse(play_thread.is_alive(), "play_sequence did not finish in time")

        # Check that play_wav_async was called twice
        self.assertEqual(mock_play_async.call_count, 2)

    def test_stop(self):
        self.player._should_play = True
        self.player.stop()
        self.assertFalse(self.player._should_play)
        self.assertTrue(self.player._stopped)

    def test_is_playing(self):
        self.player._should_play = True
        self.player._stopped = False
        self.player.playlist.put(("light", "audio.wav"))
        self.assertTrue(self.player.is_playing())


class TestHelperFunctions(unittest.TestCase):
    def test_extract_emotions(self):
        text = '$emotion:{"light":{"color":[255,0,0],"behavior":"continuous","brightness":"medium","period":1}}$ Happy text $emotion:{"light":{"color":[0,255,0],"behavior":"continuous","brightness":"medium","period":1}}$ Sad text'
        result = extract_emotions(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], "Happy text")
        self.assertEqual(result[1][1], "Sad text")

    def test_extract_language(self):
        text = "Russian text $lang:en$ English text $lang:de$ German text"
        result = extract_language(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ("ru", "Russian text"))
        self.assertEqual(result[1], ("en", "English text"))
        self.assertEqual(result[2], ("de", "German text"))

    def test_extract_language_whitespace_and_missing_dollar(self):
        # With spaces after ':' and before '$', and missing trailing '$' in the first tag
        text = "$lang: ru Привет! $lang: en $ Hello! $lang:de$Hallo!"
        result = extract_language(text)
        # Expect three segments in ru, en, de
        self.assertEqual(result[0], ("ru", "Привет!"))
        self.assertEqual(result[1], ("en", "Hello!"))
        self.assertEqual(result[2], ("de", "Hallo!"))

    def test_adjust_rgb_brightness(self):
        rgb = [255, 128, 64]
        result = adjust_rgb_brightness(rgb, "low")
        self.assertEqual(len(result), 3)
        self.assertTrue(all(0 <= x <= 255 for x in result))

    def test_adjust_rgb_brightness_accepts_prompt_aliases(self):
        rgb = [255, 128, 64]
        self.assertEqual(
            adjust_rgb_brightness(rgb, "bright"),
            adjust_rgb_brightness(rgb, "high"),
        )
        self.assertEqual(
            adjust_rgb_brightness(rgb, "dark"),
            adjust_rgb_brightness(rgb, "low"),
        )

    def test_get_wav_duration_seconds(self):
        wav_path = Path("tests/test_duration.wav")
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)

        try:
            self.assertAlmostEqual(get_wav_duration_seconds(str(wav_path)), 1.0, places=2)
        finally:
            wav_path.unlink(missing_ok=True)

    def test_adjust_light_for_audio_duration_shortens_slow_pattern(self):
        light = {
            "color": [255, 80, 0],
            "behavior": "breathing",
            "brightness": "dark",
            "period": 3,
        }

        adjusted = adjust_light_for_audio_duration(light, 1.25)

        self.assertIsNot(adjusted, light)
        self.assertEqual(adjusted["period"], 1.25)
        self.assertEqual(adjusted["color"], light["color"])

    def test_adjust_light_for_audio_duration_leaves_continuous_light_unchanged(self):
        light = {
            "color": [255, 0, 0],
            "behavior": "continuous",
            "brightness": "medium",
            "period": 1,
        }
        self.assertEqual(adjust_light_for_audio_duration(light, 0.5), light)

    def test_emotions_prompt(self):
        prompt = emotions_prompt()
        self.assertIsInstance(prompt, str)
        self.assertTrue("Express emotions with light" in prompt)

    def test_language_prompt(self):
        prompt = language_prompt()
        self.assertIsInstance(prompt, str)
        self.assertTrue("$lang:" in prompt)


if __name__ == '__main__':
    unittest.main()
