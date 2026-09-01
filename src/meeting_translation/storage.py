from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class SessionLog:
    def __init__(self, root: Path, session_id: str, sample_rate: int, record_audio: bool) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = root / f"{stamp}_{session_id}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events_path = self.directory / "events.jsonl"
        self._events = self.events_path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._audio = (
            sf.SoundFile(
                self.directory / "meeting_audio.flac",
                mode="w",
                samplerate=sample_rate,
                channels=1,
                subtype="PCM_16",
            )
            if record_audio
            else None
        )

    def append(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._events.write(encoded + "\n")
            self._events.flush()

    def append_audio(self, samples: np.ndarray) -> None:
        if self._audio is None:
            return
        with self._lock:
            self._audio.write(np.asarray(samples, dtype=np.float32))
            self._audio.flush()

    def close(self) -> None:
        with self._lock:
            if not self._events.closed:
                self._events.flush()
                self._events.close()
            if self._audio is not None:
                self._audio.flush()
                self._audio.close()
                self._audio = None
