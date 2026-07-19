"""
Template-based alert generation.
Replaces LLM at inference time. Driver-facing, TTS-ready, ≤15 words.
Swap text output → TTS later by replacing _speak().
"""
from __future__ import annotations
from typing import Optional
from src.events import DMSEvent, EventType, Severity, Audience


ALERT_TEMPLATES = {
    EventType.DROWSINESS: [
        "Drowsiness detected. Please pull over and rest.",
        "You look drowsy. Find a safe place to stop.",
        "Eyes are closing. Take a break now.",
    ],
    EventType.YAWN: [
        "Frequent yawning. Consider taking a break.",
        "You seem tired. A short rest is recommended.",
    ],
    EventType.DISTRACTION: [
        "Eyes on the road, please.",
        "You are looking away. Refocus on driving.",
        "Attention drifting. Return your gaze to the road.",
    ],
    EventType.PHONE_USAGE: [
        "Put the phone down and focus on driving.",
        "Phone use while driving is unsafe. Please put it away.",
    ],
    EventType.NO_FACE: [
        "Driver not detected. Please face the camera.",
    ],
    EventType.CONTINUOUS_DRIVE: [
        "You have been driving for a while. Consider taking a break.",
        "Long drive detected. A short rest will improve your alertness.",
    ],
}


def _pick(event_type: EventType) -> str:
    templates = ALERT_TEMPLATES.get(event_type)
    if not templates:
        return f"Safety event: {event_type.value}"
    idx = hash(event_type) % len(templates)
    return templates[idx]


def severity_for(event_type: EventType) -> Severity:
    if event_type in (EventType.DROWSINESS, EventType.PHONE_USAGE, EventType.NO_FACE):
        return Severity.CRITICAL
    if event_type in (EventType.DISTRACTION, EventType.CONTINUOUS_DRIVE):
        return Severity.MEDIUM
    return Severity.LOW


def audience_for(event_type: EventType) -> Audience:
    if event_type == EventType.DRIVER_QUESTION or event_type == EventType.DRIVER_ANSWER:
        return Audience.TRIP
    if event_type in (EventType.DROWSINESS, EventType.PHONE_USAGE, EventType.NO_FACE,
                       EventType.DISTRACTION, EventType.CONTINUOUS_DRIVE,
                       EventType.YAWN):
        return Audience.DRIVER
    return Audience.TRIP


def make_event(event_type: EventType, **kwargs) -> DMSEvent:
    return DMSEvent(
        event_type=event_type,
        severity=severity_for(event_type),
        audience=audience_for(event_type),
        message=_pick(event_type),
        **kwargs,
    )


def make_admin_event(event_type: EventType, **kwargs) -> DMSEvent:
    return DMSEvent(
        event_type=event_type,
        severity=severity_for(event_type),
        audience=Audience.ADMIN,
        message=f"Admin: {event_type.value} event flagged.",
        **kwargs,
    )


def speak(event: DMSEvent) -> None:
    """
    Stub for in-cabin TTS.
    On Renesas board: pipe 'event.message' to espeak / flite / pyttsx3.
    For now: print to stdout so it shows up in the headless console log.
    """
    if event.audience == Audience.DRIVER:
        print(f"[TTS] {event.message}")
