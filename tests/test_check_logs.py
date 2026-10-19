import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.emotion_status import clear_payment_required, mark_payment_required

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report_project(tmp_path):
    """Run the real report script in an isolated project with a fake email sender."""
    root = tmp_path / "test-project"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (root / "src").mkdir()
    (root / "logs").mkdir()
    shutil.copy2(PROJECT_ROOT / "scripts/check_logs.sh", scripts / "check_logs.sh")
    shutil.copy2(PROJECT_ROOT / "src/emotion_status.py", root / "src/emotion_status.py")
    (root / "config.json").write_text(json.dumps({"admin_email": "admin@example.com"}))
    sender = scripts / "send_email.sh"
    sender.write_text('#!/bin/bash\nprintf "%s\\0" "$@" >> "$EMAIL_CAPTURE"\n')
    sender.chmod(0o700)
    return root


def run_report(root):
    environment = os.environ.copy()
    environment.pop("CONFIG_FILE", None)
    environment.pop("LOG_FILE", None)
    environment["EMAIL_CAPTURE"] = str(root / "emails.bin")
    result = subprocess.run(
        ["bash", str(root / "scripts/check_logs.sh")],
        cwd=root.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    capture = root / "emails.bin"
    fields = capture.read_bytes().decode().split("\0")[:-1] if capture.exists() else []
    return result, [fields[index : index + 3] for index in range(0, len(fields), 3)]


@pytest.mark.parametrize("log_state", ["missing", "empty", "clean", "rotated"])
def test_unresolved_status_is_emailed_on_every_run_without_new_errors(report_project, log_state):
    root = report_project
    log = root / "logs/assistant.log"
    if log_state == "empty":
        log.touch()
    elif log_state == "clean":
        log.write_text("INFO Speech recognition succeeded\n")
    elif log_state == "rotated":
        (root / "logs/assistant.log.2026-10-18").write_text("ERROR Gemini HTTP 402\n")
        log.write_text("INFO New day\n")
    status = root / "logs/gemini_emotion_status.json"
    assert mark_payment_required(status)
    original = status.read_bytes()

    _, emails = run_report(root)
    assert len(emails) == 1
    subject, body, recipient = emails[0]
    assert subject == "AIY Assistant Log Errors Found"
    assert recipient == "admin@example.com"
    assert "unresolved historical HTTP 402" in body
    assert "Recovery is unverified" in body
    assert "check Gemini project billing" in body

    _, emails = run_report(root)
    assert len(emails) == 2
    assert emails[0] == emails[1]
    assert status.read_bytes() == original


def test_status_and_current_errors_are_combined_in_one_email(report_project):
    root = report_project
    (root / "logs/assistant.log").write_text(
        "INFO startup\nERROR First real error\nERROR notification enabled\nTraceback failure\n"
    )
    assert mark_payment_required(root / "logs/gemini_emotion_status.json")
    _, emails = run_report(root)
    assert len(emails) == 1
    body = emails[0][1]
    assert "Found 2 errors in the log file:" in body
    assert "First real error" in body
    assert "Traceback failure" in body
    # Preserve the original -A 3 context behavior: this line is excluded from
    # the count but may still appear as context after a real error.
    assert "ERROR notification enabled" in body
    assert "unresolved historical HTTP 402" in body


@pytest.mark.parametrize("log_contents", [None, "", "INFO healthy\n", "ERROR notification enabled\n"])
def test_no_alert_when_no_status_and_no_real_errors(report_project, log_contents):
    if log_contents is not None:
        (report_project / "logs/assistant.log").write_text(log_contents)
    _, emails = run_report(report_project)
    assert emails == []


def test_current_error_report_is_unchanged_without_status(report_project):
    (report_project / "logs/assistant.log").write_text("ERROR something failed\n")
    _, emails = run_report(report_project)
    assert emails == [
        [
            "AIY Assistant Log Errors Found",
            "Found 1 errors in the log file:\n\nERROR something failed",
            "admin@example.com",
        ]
    ]


def test_verified_recovery_stops_persistent_alert_but_preserves_current_errors(report_project):
    root = report_project
    status = root / "logs/gemini_emotion_status.json"
    assert mark_payment_required(status)
    assert clear_payment_required(status)
    _, emails = run_report(root)
    assert emails == []
    (root / "logs/assistant.log").write_text("ERROR unrelated error\n")
    _, emails = run_report(root)
    assert len(emails) == 1
    assert "unrelated error" in emails[0][1]
    assert "Gemini" not in emails[0][1]


def test_corrupt_status_is_reported_even_without_current_log(report_project):
    (report_project / "logs/gemini_emotion_status.json").write_text("corrupt marker")
    _, emails = run_report(report_project)
    assert len(emails) == 1
    assert "status could not be read" in emails[0][1]
    assert "recovery is unverified" in emails[0][1]
    assert "corrupt marker" not in emails[0][1]


def test_failed_status_cli_is_reported_instead_of_silencing_alert(report_project):
    (report_project / "src/emotion_status.py").write_text("raise RuntimeError('broken helper')\n")
    _, emails = run_report(report_project)
    assert len(emails) == 1
    assert "status could not be checked" in emails[0][1]
    assert "recovery is unverified" in emails[0][1]
