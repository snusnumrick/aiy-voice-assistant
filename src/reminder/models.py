"""
Reminder schema models.

This module formalizes reminder payloads used by tool calls, persistence, and
notification handling.
"""

import datetime as dt
from typing import Any, Optional, TypeVar

try:
    from pydantic import BaseModel, ConfigDict

    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel

    PYDANTIC_V2 = False


ReminderModelT = TypeVar("ReminderModelT", bound="ReminderModel")


class ReminderModel(BaseModel):
    """Compatibility base model for reminder payloads."""

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")
    else:

        class Config:
            extra = "allow"

    @classmethod
    def from_data(cls: type[ReminderModelT], data: Any) -> ReminderModelT:
        if PYDANTIC_V2:
            return cls.model_validate(data)
        return cls.parse_obj(data)

    def to_data(self, **kwargs: Any) -> dict[str, Any]:
        if PYDANTIC_V2:
            return self.model_dump(**kwargs)
        return self.dict(**kwargs)


class ReminderDraft(ReminderModel):
    """Validated reminder payload before it is assigned an id."""

    message: str
    when: dt.datetime
    recurring: bool = False
    speak_text: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[dict[str, Any]] = None
    fact_text: Optional[str] = None
    light: Optional[dict[str, Any]] = None
    bell_repeat: Optional[int] = None
    bell_duration_sec: Optional[float] = None

    def to_record(self, reminder_id: str) -> "ReminderRecord":
        data = self.to_data(exclude_none=True)
        data["id"] = str(reminder_id)
        data["time"] = data.pop("when")
        data["done"] = False
        return ReminderRecord.from_data(data)


class ReminderRecord(ReminderModel):
    """Validated reminder record as stored on disk and passed at runtime."""

    id: str
    time: dt.datetime
    message: str
    recurring: bool = False
    done: bool = False
    speak_text: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[dict[str, Any]] = None
    fact_text: Optional[str] = None
    light: Optional[dict[str, Any]] = None
    bell_repeat: Optional[int] = None
    bell_duration_sec: Optional[float] = None

    def to_storage(self) -> dict[str, Any]:
        data = self.to_data(exclude_none=True)
        data["time"] = self.time.isoformat()
        return data


class ReminderUpdate(ReminderModel):
    """Validated partial reminder update payload."""

    time: Optional[dt.datetime] = None
    message: Optional[str] = None
    recurring: Optional[bool] = None
    done: Optional[bool] = None
    speak_text: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[dict[str, Any]] = None
    fact_text: Optional[str] = None
    light: Optional[dict[str, Any]] = None
    bell_repeat: Optional[int] = None
    bell_duration_sec: Optional[float] = None

    def has_updates(self) -> bool:
        return bool(self.to_data(exclude_none=True))


class ReminderCreateAtRequest(ReminderModel):
    """Tool request payload for absolute reminder creation."""

    message: str
    time: dt.datetime
    recurring: bool = False
    speak_text: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[dict[str, Any]] = None
    fact_text: Optional[str] = None
    light: Optional[dict[str, Any]] = None
    bell_repeat: Optional[int] = None
    bell_duration_sec: Optional[float] = None

    def to_draft(self) -> ReminderDraft:
        data = self.to_data(exclude_none=True)
        data["when"] = data.pop("time")
        return ReminderDraft.from_data(data)


class ReminderCreateInRequest(ReminderModel):
    """Tool request payload for relative reminder creation."""

    message: str
    amount: int
    unit: str
    recurring: bool = False
    speak_text: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[dict[str, Any]] = None
    fact_text: Optional[str] = None
    light: Optional[dict[str, Any]] = None
    bell_repeat: Optional[int] = None
    bell_duration_sec: Optional[float] = None

    def to_draft(self, now: dt.datetime) -> ReminderDraft:
        seconds = {
            "second": 1,
            "seconds": 1,
            "minute": 60,
            "minutes": 60,
            "hour": 3600,
            "hours": 3600,
            "day": 86400,
            "days": 86400,
        }.get(str(self.unit).lower())
        if seconds is None:
            raise ValueError("Invalid unit. Use seconds, minutes, hours, or days.")
        data = self.to_data(exclude_none=True)
        data.pop("amount", None)
        data.pop("unit", None)
        data["when"] = now + dt.timedelta(seconds=self.amount * seconds)
        return ReminderDraft.from_data(data)


class ReminderUpdateRequest(ReminderModel):
    """Tool request payload for reminder updates."""

    id: str
    time: Optional[dt.datetime] = None
    message: Optional[str] = None
    recurring: Optional[bool] = None
    done: Optional[bool] = None
    speak_text: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[dict[str, Any]] = None
    fact_text: Optional[str] = None
    light: Optional[dict[str, Any]] = None
    bell_repeat: Optional[int] = None
    bell_duration_sec: Optional[float] = None

    def to_update(self) -> ReminderUpdate:
        data = self.to_data(exclude_none=True)
        data.pop("id", None)
        return ReminderUpdate.from_data(data)
