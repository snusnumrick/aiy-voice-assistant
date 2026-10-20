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
    def test_music_waits_for_spoken_response(self, mock_thread):
        self.player.begin_response()
        self.player.add_music((None, 'song.wav', 'song'))
        self.assertFalse(self.player.is_playing())
        mock_thread.assert_not_called()

        self.player.add((None, 'intro.wav', 'Включаю!'))
        self.player.finish_response()

        self.assertEqual(self.player.merge_queue.get_nowait().filename, 'intro.wav')
        self.assertEqual(self.player.merge_queue.get_nowait().filename, 'song.wav')
        self.player.finish_response()
        self.assertTrue(self.player.merge_queue.empty())

    @patch('src.responce_player.threading.Thread')
    def test_interrupted_response_discards_music(self, mock_thread):
        self.player.begin_response()
        self.player.add_music((None, 'song.wav', 'song'))
        self.player.finish_response(completed=False)
        self.player.finish_response()
        self.assertTrue(self.player.merge_queue.empty())
        mock_thread.assert_not_called()

    @patch('src.responce_player.threading.Thread')
    def test_music_outside_response_plays_immediately(self, mock_thread):
        self.player.add_music((None, 'song.wav', 'song'))
        self.assertEqual(self.player.merge_queue.get_nowait().filename, 'song.wav')

    @patch('src.responce_player.threading.Thread')
    def test_response_after_stop_still_holds_music(self, mock_thread):
        self.player._stopped = True
        self.player.begin_response()
        self.player.add_music((None, 'song.wav', 'song'))
        self.assertTrue(self.player.merge_queue.empty())
        self.player.finish_response()
        self.assertEqual(self.player.merge_queue.get_nowait().filename, 'song.wav')

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
        expected_color = (178, 0, 0)  # Adjusted [255, 0, 0] at medium brightness.
        with patch.object(self.player, 'current_light', None):
            with patch('src.responce_player.Leds.rgb_on', return_value=expected_color) as mock_rgb_on:
                self.player.change_light_behavior(behaviour)
                self.leds.update.assert_called_once()
                mock_rgb_on.assert_called_once_with(expected_color)

        # Check that the LED was updated with the correct color
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

    def test_song_timelines_are_not_merged_with_speech_or_each_other(self):
        track = [{"time": 0, "color": [255, 0, 0], "brightness": 0.5}]
        self.player.merge_queue.put(MergeItem(None, "intro.wav", "Intro"))
        self.player.merge_queue.put(MergeItem(None, "song.wav", "Song", track))
        self.player.merge_queue.put(MergeItem(None, "next.wav", "Next song", track))
        with patch("src.responce_player.combine_audio_files") as combine:
            self.player._merge_audio_files()
        combine.assert_not_called()
        self.assertEqual(self.player.playlist.get_nowait(), (None, "intro.wav"))
        self.assertEqual(self.player.playlist.get_nowait(), (None, "song.wav", track))
        self.assertEqual(self.player.playlist.get_nowait(), (None, "next.wav", track))

    @patch('src.responce_player.threading.Thread')
    def test_song_timeline_waits_for_intro_and_is_discarded_on_interruption(self, _thread):
        track = [{"time": 0, "color": [255, 0, 0], "brightness": 0.5}]
        self.player.begin_response()
        self.player.add_music(({"light_track": track}, "song.wav", "Song"))
        self.assertTrue(self.player.merge_queue.empty())
        self.player.add((None, "intro.wav", "Intro"))
        self.player.finish_response()
        self.assertEqual(self.player.merge_queue.get_nowait().filename, "intro.wav")
        self.assertEqual(self.player.merge_queue.get_nowait().light_track, track)
        self.player.begin_response()
        self.player.add_music(({"light_track": track}, "song.wav", "Song"))
        self.player.finish_response(completed=False)
        self.assertTrue(self.player.merge_queue.empty())

    def test_timeline_schedules_numeric_brightness_and_stops_promptly(self):
        first_frame = threading.Event()
        process = MagicMock()
        process.poll.return_value = None
        track = [
            {"time": 0, "color": [200, 100, 0], "brightness": 0.5},
            {"time": 30, "color": [0, 0, 255], "brightness": 1},
        ]
        self.player._should_play = True
        self.leds.reset_mock()
        self.leds.update.side_effect = lambda _: first_frame.set()
        done = threading.Event()
        with patch("src.responce_player.Leds.rgb_on", side_effect=lambda color: color):
            runner = threading.Thread(target=self.player._play_light_track,
                                      args=(track, process, time.monotonic(), done))
            runner.start()
            self.assertTrue(first_frame.wait(timeout=1))
            self.player._light_stop.set()
            runner.join(timeout=1)
        self.assertFalse(runner.is_alive())
        self.leds.update.assert_called_once_with((100, 50, 0))

    @patch("src.responce_player.play_wav_async")
    def test_timeline_starts_with_audio_and_restores_previous_light(self, play_async):
        track = [
            {"time": 0, "color": [200, 100, 0], "brightness": 0.5},
            {"time": 0.02, "color": [0, 200, 0], "brightness": 0.5},
        ]
        previous = {"color": [0, 0, 255], "brightness": "bright", "behavior": "continuous"}
        self.player.change_light_behavior(previous)
        second_frame = threading.Event()
        process = MagicMock()
        process.poll.return_value = None
        play_async.return_value = process
        colors = []

        def update(color):
            colors.append(color)
            if color == (0, 100, 0):
                second_frame.set()

        def wait():
            self.assertTrue(second_frame.wait(timeout=1))

        process.wait.side_effect = wait
        self.leds.update.side_effect = update
        self.player._should_play = True
        self.player.playlist.put((None, "song.wav", track))
        original_change = self.player.change_light_behavior

        def restore(light):
            original_change(light)
            if light == previous and (0, 100, 0) in colors:
                self.player._should_play = False

        with patch("src.responce_player.Leds.rgb_on", side_effect=lambda color: color), \
                patch.object(self.player, "change_light_behavior", side_effect=restore):
            self.player._play_sequence()
        play_async.assert_called_once_with("song.wav")
        self.assertIn((100, 50, 0), colors)
        self.assertEqual(colors[-1], (0, 0, 255))
        self.assertTrue(self.player._playback_completed.is_set())

    def test_completed_audio_does_not_keep_running_lights(self):
        process = MagicMock()
        process.poll.return_value = 0
        self.leds.reset_mock()
        self.player._should_play = True
        self.player._play_light_track(
            [{"time": 0, "color": [255, 0, 0], "brightness": 1}],
            process, time.monotonic(), threading.Event(),
        )
        self.leds.update.assert_not_called()

    def test_late_scheduler_skips_obsolete_frames(self):
        process = MagicMock()
        process.poll.return_value = None
        self.player._should_play = True
        track = [{"time": at, "color": color, "brightness": 1} for at, color in (
            (0, [255, 0, 0]), (0.05, [0, 255, 0]), (0.1, [0, 0, 255]),
        )]
        self.leds.reset_mock()
        with patch("src.responce_player.time.monotonic", return_value=10.12), \
                patch("src.responce_player.Leds.rgb_on", side_effect=lambda color: color):
            self.player._play_light_track(track, process, 10, threading.Event())
        self.leds.update.assert_called_once_with((0, 0, 255))

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
    def test_emotion_extractor_logs_one_error_instead_of_raw_payload(self):
        with self.assertLogs("src.response_metadata", level="ERROR") as captured:
            self.assertEqual(extract_emotions('$emotion:{“voice”:{“tone”:“plain”}} Reply.'), [(None, 'Reply.')])
        self.assertEqual([record.getMessage() for record in captured.records], [
            "Malformed service metadata removed: tag=emotion reason=missing_delimiter,invalid_json,typographic_quotes",
        ])

    def test_emotion_prompt_contains_a_valid_json_example(self):
        import json

        from src.response_metadata import iter_service_tags

        tag = next(iter_service_tags(emotions_prompt()))
        value = json.loads(tag.payload)
        self.assertEqual(set(value), {"light", "voice"})
        self.assertEqual(value["light"]["color"], [255, 160, 80])

    def test_invalid_emotion_json_is_hidden_but_following_speech_is_preserved(self):
        variants = [
            '$emotion:{"voice":{"tone":"plain"}} Neutral reply.',
            '$emotion:{“voice”:{“tone”:“plain”}} Neutral reply.',
            '$emotion:{"light":invalid}$ Neutral reply.',
            '$emotion:{"light":{}}}, "voice":{"tone":"plain"}} Neutral reply.',
        ]
        for text in variants:
            with self.subTest(text=text):
                result = extract_emotions(text)
                self.assertEqual([speech for _, speech in result], ["Neutral reply."])

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
