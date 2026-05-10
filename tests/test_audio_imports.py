import importlib
import sys
from importlib.abc import MetaPathFinder

from mock_aiy import mock_aiy


class GoogleSpeechImportProbe(MetaPathFinder):
    def __init__(self):
        self.seen = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "google.cloud.speech":
            self.seen = True
            raise ImportError("blocked google speech import")
        return None


def test_audio_module_import_does_not_load_google_speech():
    sys.modules["aiy"] = mock_aiy
    sys.modules["aiy.board"] = mock_aiy.board
    sys.modules["aiy.leds"] = mock_aiy.leds
    sys.modules["aiy.voice"] = mock_aiy.voice
    sys.modules["aiy.voice.audio"] = mock_aiy.voice.audio
    sys.modules.pop("src.audio", None)
    sys.modules.pop("google.cloud.speech", None)

    probe = GoogleSpeechImportProbe()
    sys.meta_path.insert(0, probe)
    try:
        importlib.import_module("src.audio")
    finally:
        sys.meta_path.remove(probe)

    assert not probe.seen
