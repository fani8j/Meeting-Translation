from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1


class CaptionState(str, Enum):
    PROVISIONAL = "provisional"
    COMMITTED = "committed"
    CORRECTED = "corrected"


class ServiceState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    LISTENING = "listening"
    TRANSLATING = "translating"
    CATCHING_UP = "catching_up"
    PAUSED = "paused"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass(frozen=True)
class Correction:
    before: str
    after: str
    reason: str


@dataclass(frozen=True)
class CaptionEvent:
    session_id: str
    segment_id: str
    revision: int
    state: CaptionState
    audio_start_ms: int
    audio_end_ms: int
    emitted_at_ms: int
    mandarin: str
    english: str = ""
    asr_model: str = ""
    corrections: tuple[Correction, ...] = ()
    event: str = field(default="caption", init=False)
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["corrections"] = [asdict(item) for item in self.corrections]
        return payload


@dataclass(frozen=True)
class HealthEvent:
    state: ServiceState
    message: str = ""
    backlog_seconds: float = 0.0
    active_asr_model: str = ""
    gpu_memory_mb: int = 0
    current_delay_ms: int = 0
    dropped_frames: int = 0
    microphone_enabled: bool = True
    event: str = field(default="health", init=False)
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


@dataclass(frozen=True)
class AudioLevelEvent:
    desktop_level: float
    microphone_level: float
    microphone_available: bool
    event: str = field(default="audio_level", init=False)
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionEvent:
    session_id: str
    action: str
    emitted_at_ms: int
    details: dict[str, Any] = field(default_factory=dict)
    event: str = field(default="session", init=False)
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_id() -> str:
    return str(uuid.uuid4())


def wall_time_ms() -> int:
    return time.time_ns() // 1_000_000


def parse_event(payload: dict[str, Any]) -> CaptionEvent | HealthEvent | AudioLevelEvent | SessionEvent:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {payload.get('schema_version')}")
    event_type = payload.get("event")
    if event_type == "caption":
        return CaptionEvent(
            session_id=payload["session_id"],
            segment_id=payload["segment_id"],
            revision=int(payload["revision"]),
            state=CaptionState(payload["state"]),
            audio_start_ms=int(payload["audio_start_ms"]),
            audio_end_ms=int(payload["audio_end_ms"]),
            emitted_at_ms=int(payload["emitted_at_ms"]),
            mandarin=payload["mandarin"],
            english=payload.get("english", ""),
            asr_model=payload.get("asr_model", ""),
            corrections=tuple(Correction(**item) for item in payload.get("corrections", [])),
        )
    if event_type == "health":
        return HealthEvent(
            state=ServiceState(payload["state"]),
            message=payload.get("message", ""),
            backlog_seconds=float(payload.get("backlog_seconds", 0)),
            active_asr_model=payload.get("active_asr_model", ""),
            gpu_memory_mb=int(payload.get("gpu_memory_mb", 0)),
            current_delay_ms=int(payload.get("current_delay_ms", 0)),
            dropped_frames=int(payload.get("dropped_frames", 0)),
            microphone_enabled=bool(payload.get("microphone_enabled", True)),
        )
    if event_type == "audio_level":
        return AudioLevelEvent(
            desktop_level=float(payload.get("desktop_level", 0)),
            microphone_level=float(payload.get("microphone_level", 0)),
            microphone_available=bool(payload.get("microphone_available", False)),
        )
    if event_type == "session":
        return SessionEvent(
            session_id=payload["session_id"],
            action=payload["action"],
            emitted_at_ms=int(payload["emitted_at_ms"]),
            details=dict(payload.get("details", {})),
        )
    raise ValueError(f"Unknown event type: {event_type}")
