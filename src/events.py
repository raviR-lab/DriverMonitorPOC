"""
Structured event types for the Driver Monitoring System.
Every alert flowing to the UI / admin / log is a DMSEvent instance.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    DROWSINESS = "DROWSINESS"
    YAWN = "YAWN"
    DISTRACTION = "DISTRACTION"
    PHONE_USAGE = "PHONE_USAGE"
    NO_FACE = "NO_FACE"
    CONTINUOUS_DRIVE = "CONTINUOUS_DRIVE"
    DRIVER_QUESTION = "DRIVER_QUESTION"
    DRIVER_ANSWER = "DRIVER_ANSWER"
    SYSTEM = "SYSTEM"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Audience(str, Enum):
    DRIVER = "DRIVER"
    TRIP = "TRIP"
    ADMIN = "ADMIN"


@dataclass
class DMSEvent:
    event_type: EventType
    severity: Severity
    audience: Audience
    message: str
    metrics: dict = field(default_factory=dict)
    trip_stats: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    frame_index: Optional[int] = None
    session_id: Optional[str] = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["severity"] = self.severity.value
        d["audience"] = self.audience.value
        return d
