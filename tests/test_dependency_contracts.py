from pathlib import Path

import pytoml


def test_google_cloud_clients_are_explicit_dependencies():
    pyproject = pytoml.loads(Path("pyproject.toml").read_text())
    dependencies = pyproject["tool"]["poetry"]["dependencies"]

    assert "google-cloud-speech" in dependencies
    assert "google-cloud-texttospeech" in dependencies
    assert "google-cloud" not in dependencies
