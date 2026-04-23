"""
Persistent tool usage statistics for LLM cost optimizations.

The tracker is intentionally simple:
- JSON file storage
- per-tool counts and last_used timestamp
- process-local lock for safe concurrent writes
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ToolUsageStats:
    """Tracks usage counters for tool invocations."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _empty(self) -> dict:
        return {
            "schema_version": 1,
            "total_calls": 0,
            "tools": {},
            "updated_at": self._now_iso(),
        }

    def _load(self) -> dict:
        if not self.path or not os.path.exists(self.path):
            return self._empty()
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return self._empty()
            if "tools" not in raw or not isinstance(raw["tools"], dict):
                raw["tools"] = {}
            raw.setdefault("total_calls", 0)
            raw.setdefault("schema_version", 1)
            raw.setdefault("updated_at", self._now_iso())
            return raw
        except Exception as e:
            logger.warning(f"Failed to load tool usage stats from {self.path}: {e}")
            return self._empty()

    def _save_locked(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def record_call(self, tool_name: str) -> None:
        """Record one tool invocation."""
        if not tool_name:
            return
        with self._lock:
            tools = self._data.setdefault("tools", {})
            row = tools.get(tool_name, {"count": 0, "last_used": None})
            row["count"] = int(row.get("count", 0)) + 1
            row["last_used"] = self._now_iso()
            tools[tool_name] = row
            self._data["total_calls"] = int(self._data.get("total_calls", 0)) + 1
            self._data["updated_at"] = self._now_iso()
            try:
                self._save_locked()
            except Exception as e:
                logger.warning(f"Failed to save tool usage stats: {e}")

    def get_count(self, tool_name: str) -> int:
        with self._lock:
            row = self._data.get("tools", {}).get(tool_name, {})
            return int(row.get("count", 0))

    def total_calls(self) -> int:
        with self._lock:
            return int(self._data.get("total_calls", 0))

    def top_tools(self, limit: int = 5) -> list[str]:
        with self._lock:
            tools = self._data.get("tools", {})
            sorted_names = sorted(
                tools.keys(),
                key=lambda k: int((tools.get(k) or {}).get("count", 0)),
                reverse=True,
            )
            return sorted_names[: max(0, int(limit))]

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))


def get_tool_usage_stats(config) -> ToolUsageStats | None:
    """
    Return a process-shared ToolUsageStats instance stored on config.
    """
    if not config.get("optimize_tool_usage_tracking_enabled", False):
        return None
    existing = getattr(config, "_tool_usage_stats_instance", None)
    if existing:
        return existing
    path = config.get("optimize_tool_usage_stats_file", "tool_usage_stats.json")
    tracker = ToolUsageStats(path)
    setattr(config, "_tool_usage_stats_instance", tracker)
    return tracker
