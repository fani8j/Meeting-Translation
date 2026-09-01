from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config/default.toml"
DEFAULT_GLOSSARY = PROJECT_ROOT / "config/glossary.json"


@dataclass(frozen=True)
class ModelConfig:
    primary_asr: str
    fallback_asr: str
    translator: str
    translation_quantization: str


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    frame_ms: int
    vad_aggressiveness: int
    silence_close_ms: int
    preroll_ms: int
    partial_interval_ms: int
    max_phrase_ms: int
    overlap_ms: int
    desktop_source: str
    microphone_source: str
    include_microphone: bool


@dataclass(frozen=True)
class OverloadConfig:
    enter_backlog_seconds: float
    exit_backlog_seconds: float
    recovery_hold_seconds: float


@dataclass(frozen=True)
class OverlayConfig:
    width_ratio: float
    bottom_margin: int
    english_px: int
    mandarin_px: int
    opacity: float
    locked: bool


@dataclass(frozen=True)
class StorageConfig:
    session_root: Path
    save_transcript: bool
    record_audio: bool


@dataclass(frozen=True)
class TransportConfig:
    socket_path: Path


@dataclass(frozen=True)
class AppConfig:
    models: ModelConfig
    audio: AudioConfig
    overload: OverloadConfig
    overlay: OverlayConfig
    hotkeys: dict[str, str]
    storage: StorageConfig
    transport: TransportConfig
    source_path: Path
    glossary_path: Path


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing configuration section: {name}")
    return section


def load_config(path: Path | None = None, glossary_path: Path | None = None) -> AppConfig:
    source = (path or DEFAULT_CONFIG).resolve()
    with source.open("rb") as handle:
        data = tomllib.load(handle)
    models = _section(data, "models")
    models = dict(models)
    translator_override = os.environ.get("MEETING_TRANSLATION_TRANSLATOR")
    if translator_override:
        models["translator"] = translator_override
    audio = _section(data, "audio")
    overload = _section(data, "overload")
    overlay = _section(data, "overlay")
    storage = _section(data, "storage")
    transport = _section(data, "transport")
    session_root = Path(storage["session_root"])
    if not session_root.is_absolute():
        session_root = PROJECT_ROOT / session_root
    return AppConfig(
        models=ModelConfig(**models),
        audio=AudioConfig(**audio),
        overload=OverloadConfig(**overload),
        overlay=OverlayConfig(**overlay),
        hotkeys=dict(_section(data, "hotkeys")),
        storage=StorageConfig(
            session_root=session_root,
            save_transcript=bool(storage["save_transcript"]),
            record_audio=bool(storage["record_audio"]),
        ),
        transport=TransportConfig(socket_path=Path(transport["socket_path"])),
        source_path=source,
        glossary_path=(glossary_path or DEFAULT_GLOSSARY).resolve(),
    )
