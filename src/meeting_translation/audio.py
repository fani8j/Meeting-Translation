from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray
    start_ms: int
    end_ms: int
    source: str


class ReplaySource:
    def __init__(
        self,
        path: Path,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
        speed: float = 1.0,
        limit_seconds: float | None = None,
    ) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.speed = speed
        self.limit_seconds = limit_seconds

    async def frames(self) -> AsyncIterator[AudioFrame]:
        audio, source_rate = sf.read(self.path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if source_rate != self.sample_rate:
            audio = _linear_resample(audio, source_rate, self.sample_rate)
        if self.limit_seconds is not None:
            audio = audio[: int(self.limit_seconds * self.sample_rate)]
        frame_samples = self.sample_rate * self.frame_ms // 1000
        started = time.monotonic()
        for offset in range(0, len(audio) - frame_samples + 1, frame_samples):
            start_ms = offset * 1000 // self.sample_rate
            end_ms = (offset + frame_samples) * 1000 // self.sample_rate
            if self.speed > 0:
                target = started + (end_ms / 1000) / self.speed
                await asyncio.sleep(max(0.0, target - time.monotonic()))
            yield AudioFrame(
                samples=np.asarray(audio[offset:offset + frame_samples], dtype=np.float32),
                start_ms=start_ms,
                end_ms=end_ms,
                source="replay",
            )


class PulseSource:
    def __init__(self, device: str, sample_rate: int = 16_000, frame_ms: int = 30) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._process: asyncio.subprocess.Process | None = None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        frame_samples = self.sample_rate * self.frame_ms // 1000
        frame_bytes = frame_samples * 2
        self._process = await asyncio.create_subprocess_exec(
            "parec",
            "--device", self.device,
            "--format=s16le",
            "--rate", str(self.sample_rate),
            "--channels=1",
            "--raw",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert self._process.stdout is not None
        started = time.monotonic()
        sequence = 0
        try:
            while True:
                try:
                    raw = await self._process.stdout.readexactly(frame_bytes)
                except asyncio.IncompleteReadError:
                    break
                start_ms = sequence * self.frame_ms
                sequence += 1
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                yield AudioFrame(samples=samples, start_ms=start_ms, end_ms=sequence * self.frame_ms, source=self.device)
        finally:
            del started
            await self.close()

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=2)
            if process.returncode is None:
                process.kill()
                await process.wait()


class LevelNormalizer:
    def __init__(self, target_rms: float = 0.12) -> None:
        self.target_rms = target_rms
        self.running_rms = target_rms
        self.current_rms = 0.0

    def apply(self, samples: np.ndarray) -> np.ndarray:
        self.current_rms = float(np.sqrt(np.dot(samples, samples) / max(1, samples.size) + 1e-8))
        if self.current_rms > 0.002:
            self.running_rms = 0.97 * self.running_rms + 0.03 * self.current_rms
        gain = np.clip(self.target_rms / max(self.running_rms, 0.01), 0.5, 3.0)
        return np.asarray(samples * gain, dtype=np.float32)


class DualSourceMixer:
    def __init__(self, desktop: PulseSource, microphone: PulseSource | None) -> None:
        self.desktop = desktop
        self.microphone = microphone
        self.desktop_normalizer = LevelNormalizer()
        self.microphone_normalizer = LevelNormalizer(target_rms=0.10)
        self.microphone_enabled = microphone is not None
        self.microphone_failed = False
        self.microphone_started = False
        self.desktop_level = 0.0
        self.microphone_level = 0.0
        self._microphone_iter = microphone.frames().__aiter__() if microphone is not None else None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        async for desktop_frame in self.desktop.frames():
            microphone_frame = None
            microphone_iter = self._microphone_iter
            if microphone_iter is not None and not self.microphone_failed:
                try:
                    if self.microphone_started:
                        microphone_frame = await anext(microphone_iter)
                    else:
                        microphone_frame = await asyncio.wait_for(anext(microphone_iter), timeout=5.0)
                        self.microphone_started = True
                except (StopAsyncIteration, asyncio.TimeoutError, OSError, RuntimeError):
                    if microphone_iter is self._microphone_iter:
                        self.microphone_failed = True
                        self.microphone_enabled = False
            desktop_audio = self.desktop_normalizer.apply(desktop_frame.samples)
            self.desktop_level = self.desktop_normalizer.current_rms
            if self.microphone_enabled and microphone_frame is not None:
                microphone_audio = self.microphone_normalizer.apply(microphone_frame.samples)
                self.microphone_level = self.microphone_normalizer.current_rms
            else:
                microphone_audio = np.zeros_like(desktop_audio)
                self.microphone_level = 0.0
            mixed = np.tanh(desktop_audio + microphone_audio).astype(np.float32)
            yield AudioFrame(
                samples=mixed,
                start_ms=desktop_frame.start_ms,
                end_ms=desktop_frame.end_ms,
                source="mixed",
            )

    async def reconnect_microphone(self, microphone: PulseSource) -> None:
        previous = self.microphone
        self.microphone = microphone
        self._microphone_iter = microphone.frames().__aiter__()
        self.microphone_normalizer = LevelNormalizer(target_rms=0.10)
        self.microphone_started = False
        self.microphone_failed = False
        self.microphone_enabled = True
        if previous is not None:
            await previous.close()

    async def close(self) -> None:
        await asyncio.gather(
            self.desktop.close(),
            self.microphone.close() if self.microphone is not None else asyncio.sleep(0),
        )


def _linear_resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if not len(audio):
        return np.asarray([], dtype=np.float32)
    target_length = round(len(audio) * target_rate / source_rate)
    source_positions = np.linspace(0, len(audio) - 1, num=len(audio))
    target_positions = np.linspace(0, len(audio) - 1, num=target_length)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)
