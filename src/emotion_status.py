"""Keep an unresolved Gemini emotion HTTP 402 visible across log rotation."""

import asyncio
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_STATUS_PATH = Path(__file__).resolve().parents[1] / "logs/gemini_emotion_status.json"
_lock = threading.Lock()
# Preserve mark/clear order without blocking speech recognition on SD-card I/O.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion-status")
StatusPath = Optional[Union[str, Path]]


def _status_path(path: StatusPath) -> Path:
    return DEFAULT_STATUS_PATH if path is None else Path(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("status") != "payment_required"
        or data.get("http_status") != 402
    ):
        raise ValueError("Invalid Gemini emotion status")
    for field in ("first_seen", "last_seen"):
        timestamp = datetime.fromisoformat(data[field])
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("Gemini emotion status timestamps must be UTC")
        # Normalize validated timestamps before using them in an email report.
        data[field] = timestamp.isoformat()
    return data


def _write_status(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def mark_payment_required(path: StatusPath = None) -> bool:
    """Record an observed HTTP 402 without allowing storage errors to break STT."""
    try:
        with _lock:
            target = _status_path(path)
            previous = _read_status(target)
            now = _now_iso()
            _write_status(
                target,
                {
                    "schema_version": 1,
                    "status": "payment_required",
                    "http_status": 402,
                    "first_seen": previous["first_seen"] if previous else now,
                    "last_seen": now,
                },
            )
        return True
    except Exception as exc:
        logger.error("Failed to persist Gemini emotion HTTP 402 status (%s)", type(exc).__name__)
        return False


def clear_payment_required(path: StatusPath = None) -> bool:
    """Clear only after a validated successful Gemini emotion response."""
    try:
        with _lock:
            _status_path(path).unlink(missing_ok=True)
        return True
    except Exception as exc:
        logger.error("Failed to clear Gemini emotion HTTP 402 status (%s)", type(exc).__name__)
        return False


async def _update_in_background(operation, path: StatusPath) -> bool:
    """Finish queued state writes even if the emotion task is cancelled by STT."""
    try:
        future = asyncio.get_running_loop().run_in_executor(_executor, operation, path)
        return await asyncio.shield(future)
    except Exception as exc:
        logger.error("Failed to schedule Gemini emotion status update (%s)", type(exc).__name__)
        return False


async def async_mark_payment_required(path: StatusPath = None) -> bool:
    return await _update_in_background(mark_payment_required, path)


async def async_clear_payment_required(path: StatusPath = None) -> bool:
    return await _update_in_background(clear_payment_required, path)


def payment_required_report(path: StatusPath = None) -> str:
    """Return a safe email alert until recovery is verified, or an empty string."""
    try:
        with _lock:
            status = _read_status(_status_path(path))
    except Exception as exc:
        logger.error("Failed to read Gemini emotion HTTP 402 status (%s)", type(exc).__name__)
        return (
            "ERROR: Gemini emotion HTTP 402 status could not be read; recovery is unverified.\n"
            "Action: check Gemini project billing and the persistent emotion status file, "
            "then verify a successful Gemini emotion response."
        )
    if status is None:
        return ""
    return (
        "ERROR: Gemini emotion analysis has an unresolved historical HTTP 402 (Payment Required).\n"
        "Recovery is unverified; this alert remains even when no new requests are made "
        "or application logs are rotated.\n"
        f"First seen (UTC): {status['first_seen']}\n"
        f"Last seen (UTC): {status['last_seen']}\n"
        "Action: check Gemini project billing and payment settings, then verify a successful "
        "Gemini emotion response."
    )


if __name__ == "__main__":
    report = payment_required_report()
    if report:
        print(report)
