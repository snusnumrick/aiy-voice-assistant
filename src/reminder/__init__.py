"""Reminder subsystem package."""

from importlib import import_module
from typing import Any

__all__ = [
    "ReminderAnnouncer",
    "ReminderCreateAtRequest",
    "ReminderCreateInRequest",
    "ReminderDraft",
    "ReminderManager",
    "ReminderRecord",
    "ReminderTool",
    "ReminderUpdate",
    "ReminderUpdateRequest",
]

_EXPORTS = {
    "ReminderAnnouncer": (".announcer", "ReminderAnnouncer"),
    "ReminderCreateAtRequest": (".models", "ReminderCreateAtRequest"),
    "ReminderCreateInRequest": (".models", "ReminderCreateInRequest"),
    "ReminderDraft": (".models", "ReminderDraft"),
    "ReminderManager": (".manager", "ReminderManager"),
    "ReminderRecord": (".models", "ReminderRecord"),
    "ReminderTool": (".tool", "ReminderTool"),
    "ReminderUpdate": (".models", "ReminderUpdate"),
    "ReminderUpdateRequest": (".models", "ReminderUpdateRequest"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
