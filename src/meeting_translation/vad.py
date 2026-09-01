from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import webrtcvad

from .audio import AudioFrame
from .domain import new_id


@dataclass(frozen=True)
class PhraseUpdate:
    kind: Literal["partial", "final"]
    segment_id: str
    audio: np.ndarray
    start_ms: int
    end_ms: int


class PhraseBuffer:
    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        silence_close_ms: int = 720,
        preroll_ms: int = 300,
        partial_interval_ms: int = 1_600,
        max_phrase_ms: int = 8_000,
        overlap_ms: int = 300,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD frame size must be 10, 20, or 30 ms")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_close_ms = silence_close_ms
        self.partial_interval_ms = partial_interval_ms
        self.max_phrase_ms = max_phrase_ms
        self.overlap_frames = max(1, overlap_ms // frame_ms)
        self.preroll = deque(maxlen=max(1, preroll_ms // frame_ms))
        self.vad = webrtcvad.Vad(aggressiveness)
        self.active_frames: list[AudioFrame] = []
        self.segment_id = ""
        self.silence_ms = 0
        self.last_partial_ms = 0

    @property
    def active(self) -> bool:
        return bool(self.active_frames)

    def push(self, frame: AudioFrame) -> list[PhraseUpdate]:
        expected = self.sample_rate * self.frame_ms // 1000
        if len(frame.samples) != expected:
            raise ValueError(f"Expected {expected} samples, received {len(frame.samples)}")
        pcm16 = np.clip(frame.samples * 32767, -32768, 32767).astype("<i2").tobytes()
        speech = self.vad.is_speech(pcm16, self.sample_rate)
        updates: list[PhraseUpdate] = []

        if not self.active_frames:
            self.preroll.append(frame)
            if speech:
                self.segment_id = new_id()
                self.active_frames = list(self.preroll)
                self.preroll.clear()
                self.silence_ms = 0
                self.last_partial_ms = frame.end_ms
            return updates

        self.active_frames.append(frame)
        self.silence_ms = 0 if speech else self.silence_ms + self.frame_ms
        duration_ms = self.active_frames[-1].end_ms - self.active_frames[0].start_ms

        if frame.end_ms - self.last_partial_ms >= self.partial_interval_ms:
            updates.append(self._snapshot("partial"))
            self.last_partial_ms = frame.end_ms

        if self.silence_ms >= self.silence_close_ms:
            updates.append(self._snapshot("final", trim_silence=True))
            self._reset()
        elif duration_ms >= self.max_phrase_ms:
            updates.append(self._snapshot("final"))
            overlap = self.active_frames[-self.overlap_frames:]
            self.segment_id = new_id()
            self.active_frames = overlap.copy()
            self.silence_ms = 0
            self.last_partial_ms = frame.end_ms
        return updates

    def flush(self) -> PhraseUpdate | None:
        if not self.active_frames:
            return None
        update = self._snapshot("final", trim_silence=True)
        self._reset()
        return update

    def _snapshot(self, kind: Literal["partial", "final"], trim_silence: bool = False) -> PhraseUpdate:
        frames = self.active_frames
        if trim_silence and self.silence_ms:
            trailing = min(len(frames) - 1, self.silence_ms // self.frame_ms)
            if trailing:
                frames = frames[:-trailing]
        audio = np.concatenate([frame.samples for frame in frames]).astype(np.float32, copy=False)
        return PhraseUpdate(
            kind=kind,
            segment_id=self.segment_id,
            audio=audio.copy(),
            start_ms=frames[0].start_ms,
            end_ms=frames[-1].end_ms,
        )

    def _reset(self) -> None:
        trailing = self.active_frames[-self.preroll.maxlen:] if self.active_frames else []
        self.preroll.clear()
        self.preroll.extend(trailing)
        self.active_frames = []
        self.segment_id = ""
        self.silence_ms = 0
        self.last_partial_ms = 0
