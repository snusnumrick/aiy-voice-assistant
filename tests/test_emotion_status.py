import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src import emotion_status


@pytest.fixture
def status_path(tmp_path):
    return tmp_path / "logs" / "gemini_emotion_status.json"


def test_default_location_is_project_root_not_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert emotion_status.DEFAULT_STATUS_PATH == (
        Path(emotion_status.__file__).resolve().parents[1] / "logs/gemini_emotion_status.json"
    )


def test_mark_records_only_diagnostic_fields_and_utc_timestamps(status_path):
    assert emotion_status.mark_payment_required(status_path)
    data = json.loads(status_path.read_text())
    assert set(data) == {"schema_version", "status", "http_status", "first_seen", "last_seen"}
    assert data["schema_version"] == 1
    assert data["status"] == "payment_required"
    assert data["http_status"] == 402
    assert data["first_seen"] == data["last_seen"]
    assert datetime.fromisoformat(data["first_seen"]).utcoffset() == timedelta(0)
    assert list(status_path.parent.iterdir()) == [status_path]


def test_repeated_failure_preserves_first_seen_and_updates_last_seen(status_path):
    first = "2026-10-18T12:00:00+00:00"
    last = "2026-10-19T12:00:00+00:00"
    with patch.object(emotion_status, "_now_iso", side_effect=[first, last]):
        assert emotion_status.mark_payment_required(status_path)
        assert emotion_status.mark_payment_required(status_path)
    data = json.loads(status_path.read_text())
    assert data["first_seen"] == first
    assert data["last_seen"] == last
    report = emotion_status.payment_required_report(status_path)
    assert first in report
    assert last in report
    assert "unresolved historical HTTP 402" in report
    assert "Recovery is unverified" in report
    assert "check Gemini project billing" in report
    # Reporting does not acknowledge or expire the failure.
    assert emotion_status.payment_required_report(status_path) == report
    assert json.loads(status_path.read_text()) == data


def test_clear_requires_explicit_call_and_is_idempotent(status_path):
    assert emotion_status.payment_required_report(status_path) == ""
    assert emotion_status.mark_payment_required(status_path)
    assert emotion_status.payment_required_report(status_path)
    assert emotion_status.clear_payment_required(status_path)
    assert not status_path.exists()
    assert emotion_status.payment_required_report(status_path) == ""
    assert emotion_status.clear_payment_required(status_path)


def test_failed_atomic_replacement_preserves_original_and_logs_error(status_path, caplog):
    assert emotion_status.mark_payment_required(status_path)
    original = status_path.read_bytes()
    with patch.object(emotion_status.os, "replace", side_effect=PermissionError("private detail")):
        with caplog.at_level(logging.ERROR):
            assert not emotion_status.mark_payment_required(status_path)
    assert status_path.read_bytes() == original
    assert list(status_path.parent.iterdir()) == [status_path]
    assert "Failed to persist" in caplog.text
    assert "PermissionError" in caplog.text
    assert "private detail" not in caplog.text


def test_unwritable_directory_does_not_raise(status_path, caplog):
    with patch.object(Path, "mkdir", side_effect=PermissionError()):
        with caplog.at_level(logging.ERROR):
            assert not emotion_status.mark_payment_required(status_path)
    assert "Failed to persist" in caplog.text


def test_failed_clear_keeps_unresolved_marker(status_path, caplog):
    assert emotion_status.mark_payment_required(status_path)
    with patch.object(Path, "unlink", side_effect=PermissionError()):
        with caplog.at_level(logging.ERROR):
            assert not emotion_status.clear_payment_required(status_path)
    assert "Failed to clear" in caplog.text
    assert "unresolved historical HTTP 402" in emotion_status.payment_required_report(status_path)


@pytest.mark.parametrize("contents", ["invalid json", "[]", '{"status": "resolved"}'])
def test_corrupt_status_remains_visible_without_echoing_contents(status_path, caplog, contents):
    status_path.parent.mkdir()
    status_path.write_text(contents)
    with caplog.at_level(logging.ERROR):
        report = emotion_status.payment_required_report(status_path)
    assert "ERROR:" in report
    assert "recovery is unverified" in report
    assert "check Gemini project billing" in report
    assert contents not in report
    assert "Failed to read" in caplog.text
    assert status_path.read_text() == contents


def test_unreadable_status_remains_visible(status_path, caplog):
    with patch.object(Path, "open", side_effect=PermissionError()):
        with caplog.at_level(logging.ERROR):
            report = emotion_status.payment_required_report(status_path)
    assert "could not be read" in report
    assert "Failed to read" in caplog.text


def test_invalid_timestamp_is_not_included_in_report(status_path):
    assert emotion_status.mark_payment_required(status_path)
    data = json.loads(status_path.read_text())
    data["first_seen"] = "untrusted private content"
    status_path.write_text(json.dumps(data))
    report = emotion_status.payment_required_report(status_path)
    assert "untrusted private content" not in report
    assert "recovery is unverified" in report


def test_concurrent_marks_keep_valid_atomic_status(status_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(emotion_status.mark_payment_required, [status_path] * 12))
    assert json.loads(status_path.read_text())["http_status"] == 402
    assert list(status_path.parent.iterdir()) == [status_path]


def test_default_helper_path_can_be_overridden_without_configuration(status_path, monkeypatch):
    monkeypatch.setattr(emotion_status, "DEFAULT_STATUS_PATH", status_path)
    assert emotion_status.mark_payment_required()
    assert "HTTP 402" in emotion_status.payment_required_report()
    assert emotion_status.clear_payment_required()



def test_async_status_io_does_not_block_stt(status_path, monkeypatch):
    import asyncio
    import threading

    entered = threading.Event()
    release = threading.Event()
    original_write = emotion_status._write_status

    def slow_write(path, data):
        entered.set()
        assert release.wait(2)
        original_write(path, data)

    monkeypatch.setattr(emotion_status, "_write_status", slow_write)

    async def scenario():
        update = asyncio.create_task(emotion_status.async_mark_payment_required(status_path))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            # The disk write is still blocked, but STT can finish independently.
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.2)
            assert not update.done()
        finally:
            release.set()
        assert await update

    asyncio.run(scenario())
    assert emotion_status.payment_required_report(status_path)


def test_cancelled_mark_finishes_and_ordered_recovery_clears_it(status_path, monkeypatch):
    import asyncio
    import threading

    entered = threading.Event()
    release = threading.Event()
    original_write = emotion_status._write_status

    def slow_write(path, data):
        entered.set()
        assert release.wait(2)
        original_write(path, data)

    monkeypatch.setattr(emotion_status, "_write_status", slow_write)

    async def scenario():
        update = asyncio.create_task(emotion_status.async_mark_payment_required(status_path))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
            # Clear queues behind the still-running mark, so it cannot be lost.
            recovery = asyncio.create_task(emotion_status.async_clear_payment_required(status_path))
        finally:
            release.set()
        assert await recovery

    asyncio.run(scenario())
    assert emotion_status.payment_required_report(status_path) == ""


def test_cancelled_emotion_task_cannot_lose_observed_payment_status(status_path, monkeypatch):
    import asyncio
    import threading

    entered = threading.Event()
    release = threading.Event()
    original_write = emotion_status._write_status

    def slow_write(path, data):
        entered.set()
        assert release.wait(2)
        original_write(path, data)

    monkeypatch.setattr(emotion_status, "_write_status", slow_write)

    async def scenario():
        update = asyncio.create_task(emotion_status.async_mark_payment_required(status_path))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        finally:
            release.set()
        # The executor barrier waits for the shielded write, without a recovery clear.
        await asyncio.get_running_loop().run_in_executor(emotion_status._executor, lambda: None)

    asyncio.run(scenario())
    assert "unresolved historical HTTP 402" in emotion_status.payment_required_report(status_path)
