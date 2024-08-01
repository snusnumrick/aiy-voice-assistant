from unittest.mock import MagicMock, Mock

class MockButton:
    def __init__(self):
        self.when_pressed = None
        self.when_released = None

    def wait_for_press(self, timeout=None):
        pass

    def wait_for_release(self, timeout=None):
        pass

class MockLeds(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern = None
        self.brightness = None
        self.update = MagicMock()  # This ensures update has all the necessary mock methods

    @staticmethod
    def rgb_on(color):
        return color

    @staticmethod
    def rgb_off():
        return (0, 0, 0)

    @staticmethod
    def rgb_pattern(color):
        return color

class MockPattern:
    @staticmethod
    def blink(period_ms):
        return "blink"

    @staticmethod
    def breathe(period_ms):
        return "breathe"

class MockColor:
    RED = (0xFF, 0x00, 0x00)
    GREEN = (0x00, 0xFF, 0x00)
    BLUE = (0x00, 0x00, 0xFF)
    YELLOW = (0xFF, 0xFF, 0x00)
    PURPLE = (0xFF, 0x00, 0xFF)
    CYAN = (0x00, 0xFF, 0xFF)
    WHITE = (0xFF, 0xFF, 0xFF)

class MockBoard:
    def __init__(self):
        self._button = MockButton()
        self._led = MockLeds()

    @property
    def button(self):
        return self._button

    @property
    def led(self):
        return self._led

class MockAudioFormat:
    def __init__(self, sample_rate_hz, num_channels, bytes_per_sample):
        self.sample_rate_hz = sample_rate_hz
        self.num_channels = num_channels
        self.bytes_per_sample = bytes_per_sample

    @property
    def bytes_per_second(self):
        return self.sample_rate_hz * self.num_channels * self.bytes_per_sample

# Create mock objects
mock_aiy = MagicMock()
mock_aiy.board = Mock()
mock_aiy.board.Button = MockButton
mock_aiy.board.Led = MockLeds

mock_aiy.leds = Mock()
mock_aiy.leds.Leds = MockLeds
mock_aiy.leds.Pattern = MockPattern
mock_aiy.leds.Color = MockColor

mock_aiy.voice = Mock()
mock_aiy.voice.audio = Mock()
mock_aiy.voice.audio.AudioFormat = MockAudioFormat
mock_aiy.voice.audio.play_wav_async = MagicMock()

# Assign mock objects
import sys
sys.modules['aiy'] = mock_aiy
sys.modules['aiy.board'] = mock_aiy.board
sys.modules['aiy.leds'] = mock_aiy.leds
sys.modules['aiy.voice'] = mock_aiy.voice
sys.modules['aiy.voice.audio'] = mock_aiy.voice.audio