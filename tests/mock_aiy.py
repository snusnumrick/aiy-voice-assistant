from unittest.mock import MagicMock

class MockButton:
    def __init__(self):
        self.when_pressed = None
        self.when_released = None

    def wait_for_press(self):
        pass

class MockLeds:
    def __init__(self):
        self.pattern = None
        self.brightness = None

    def update(self, pattern):
        self.pattern = pattern

    @staticmethod
    def rgb_on(color):
        return color

    @staticmethod
    def rgb_off():
        return (0, 0, 0)

    @staticmethod
    def rgb_pattern(color):
        return color

class MockBoard:
    def __init__(self):
        self.button = MockButton()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MockPattern:
    @staticmethod
    def breathe(period_ms):
        return "breathe"

    @staticmethod
    def blink(period_ms):
        return "blink"

mock_aiy = MagicMock()
mock_aiy.board.Board = MockBoard
mock_aiy.leds.Leds = MockLeds
mock_aiy.leds.Pattern = MockPattern
mock_aiy.leds.Color = MagicMock()
mock_aiy.voice.audio.AudioFormat = MagicMock()
mock_aiy.voice.audio.play_wav_async = MagicMock()
mock_aiy.voice.audio.Recorder = MagicMock()