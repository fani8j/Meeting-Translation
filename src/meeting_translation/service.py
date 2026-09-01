from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import time
from pathlib import Path
from typing import Any

from .audio import DualSourceMixer, PulseSource, ReplaySource
from .config import AppConfig, load_config
from .domain import (
    AudioLevelEvent,
    CaptionEvent,
    CaptionState,
    HealthEvent,
    ServiceState,
    SessionEvent,
    new_id,
    wall_time_ms,
)
from .storage import SessionLog
from .text import Glossary, StabilityTracker
from .transport import JsonlServer
from .vad import PhraseBuffer, PhraseUpdate
from .worker import GPUWorkerClient, InferenceResult, WorkerSettings


class CaptionService:
    def __init__(
        self,
        config: AppConfig,
        replay_path: Path | None = None,
        replay_speed: float = 1.0,
        replay_limit: float | None = None,
    ) -> None:
        self.config = config
        self.replay_path = replay_path
        self.replay_speed = replay_speed
        self.replay_limit = replay_limit
        self.glossary = Glossary(config.glossary_path)
        self.transport = JsonlServer(config.transport.socket_path, self.handle_command)
        self.worker: GPUWorkerClient | None = None
        self.session_id = ""
        self.session_log: SessionLog | None = None
        self.audio_task: asyncio.Task | None = None
        self.result_task: asyncio.Task | None = None
        self.health_task: asyncio.Task | None = None
        self.stop_event = asyncio.Event()
        self.session_done = asyncio.Event()
        self.worker_ready = asyncio.Event()
        self.paused = False
        self.microphone_enabled = config.audio.include_microphone
        self.microphone_failure_announced = False
        self.desktop_source = config.audio.desktop_source
        self.microphone_source = config.audio.microphone_source if config.audio.include_microphone else ""
        self.record_audio = config.storage.record_audio
        self.latest_audio_ms = 0
        self.last_result_audio_ms = 0
        self.active_model = config.models.primary_asr
        self.degraded = False
        self.recovery_started: float | None = None
        self.outstanding = 0
        self.asr_outstanding = 0
        self.request_revisions: dict[str, int] = {}
        self.event_revisions: dict[str, int] = {}
        self.final_segments: dict[str, dict[str, Any]] = {}
        self.latest_phrases: dict[str, PhraseUpdate] = {}
        self.stability = StabilityTracker()
        self.restart_count = 0
        self.mixer: DualSourceMixer | None = None
        self.last_level_event_at = 0.0
        self.session_started_at_ms = 0
        self.committed_segments: set[str] = set()
        self.translated_segments: set[str] = set()
        self.max_delay_ms = 0
        self.microphone_outage_started_ms: int | None = None
        self.microphone_outages: list[dict[str, int]] = []
        self._state = ServiceState.IDLE

    async def run(self, auto_start: bool = False) -> None:
        await self.transport.start()
        await self.publish_health(ServiceState.IDLE, "Ready")
        if auto_start:
            await self.start_session()
            await self.session_done.wait()
            await self.shutdown()
            return
        await self.stop_event.wait()
        await self.shutdown()

    async def handle_command(self, message: dict[str, Any]) -> None:
        command = message.get("command")
        if command == "start":
            self.record_audio = bool(message.get("record_audio", self.record_audio))
            desktop_source = message.get("desktop_source")
            microphone_source = message.get("microphone_source")
            if isinstance(desktop_source, str) and desktop_source:
                self.desktop_source = desktop_source
            if isinstance(microphone_source, str):
                self.microphone_source = microphone_source
                self.microphone_enabled = bool(microphone_source)
            await self.start_session()
        elif command == "stop":
            await self.stop_session()
        elif command == "pause":
            self.paused = not self.paused
            await self.publish_health(ServiceState.PAUSED if self.paused else ServiceState.LISTENING)
        elif command == "toggle_microphone":
            if not self.microphone_source:
                self.microphone_enabled = False
                await self.publish_health(self._state, "No microphone source selected")
                return
            self.microphone_enabled = not self.microphone_enabled
            if self.mixer is not None:
                self.mixer.microphone_enabled = self.microphone_enabled
            await self.publish_health(self._state, "Microphone enabled" if self.microphone_enabled else "Microphone muted")
        elif command == "reconnect_microphone":
            source = message.get("microphone_source")
            if not isinstance(source, str) or not source:
                await self.publish_health(self._state, "Select an available microphone before reconnecting")
                return
            self.microphone_source = source
            self.microphone_enabled = True
            self.microphone_failure_announced = False
            if self.mixer is not None:
                await self.mixer.reconnect_microphone(
                    PulseSource(source, self.config.audio.sample_rate, self.config.audio.frame_ms)
                )
            if self.microphone_outage_started_ms is not None:
                recovered_at = wall_time_ms()
                self.microphone_outages.append({
                    "started_at_ms": self.microphone_outage_started_ms,
                    "ended_at_ms": recovered_at,
                })
                self.microphone_outage_started_ms = None
            if self.session_id:
                await self.publish(SessionEvent(
                    session_id=self.session_id,
                    action="microphone_recovered",
                    emitted_at_ms=wall_time_ms(),
                    details={"microphone_source": source},
                ).to_dict())
            await self.publish_health(self._state, "Microphone reconnected")
        elif command == "retry":
            if self.session_id and self._state == ServiceState.ERROR:
                self.restart_count = 0
                await self._restart_worker("Manual retry")
            else:
                await self.publish_health(self._state, "Service connection verified")
        elif command == "record_audio":
            if self.session_id:
                await self.publish_health(self._state, "Audio recording can only be selected before starting a session")
            else:
                self.record_audio = bool(message.get("enabled", False))
        elif command == "shutdown":
            self.stop_event.set()
        elif command == "status":
            await self.publish_health(self._state)

    async def start_session(self) -> None:
        if self.session_id:
            return
        self.session_done.clear()
        self.session_started_at_ms = 0
        self.committed_segments.clear()
        self.translated_segments.clear()
        self.max_delay_ms = 0
        self.microphone_outage_started_ms = None
        self.microphone_outages.clear()
        self.microphone_failure_announced = False
        self.worker_ready.clear()
        self.session_id = new_id()
        self.session_log = SessionLog(
            self.config.storage.session_root,
            self.session_id,
            self.config.audio.sample_rate,
            self.record_audio,
        )
        self.worker = self._new_worker()
        self.worker.start()
        self.result_task = asyncio.create_task(self._consume_results())
        self.health_task = asyncio.create_task(self._health_loop())
        await self.publish_health(ServiceState.PREPARING, "Loading speech model")
        try:
            await asyncio.wait_for(self.worker_ready.wait(), timeout=300)
        except asyncio.TimeoutError:
            await self.publish_health(ServiceState.ERROR, "Speech model did not become ready")
            await self.stop_session()
            return
        self.session_started_at_ms = wall_time_ms()
        event = SessionEvent(
            session_id=self.session_id,
            action="started",
            emitted_at_ms=self.session_started_at_ms,
            details={
                "desktop_source": self.desktop_source,
                "microphone_source": self.microphone_source,
                "record_audio": self.record_audio,
            },
        )
        await self.publish(event.to_dict())
        self.audio_task = asyncio.create_task(self._capture_loop())
        await self.publish_health(ServiceState.LISTENING, "Listening")

    async def stop_session(self) -> None:
        if not self.session_id:
            return
        await self.publish_health(ServiceState.STOPPING, "Finishing captions and saving transcript")
        if self.audio_task is not None:
            self.audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.audio_task
            self.audio_task = None
        if self.mixer is not None:
            await self.mixer.close()
            self.mixer = None
        deadline = time.monotonic() + 180
        while self.outstanding > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if self.result_task is not None:
            self.result_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.result_task
            self.result_task = None
        if self.health_task is not None:
            self.health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.health_task
            self.health_task = None
        if self.worker is not None:
            await asyncio.to_thread(self.worker.close)
            self.worker = None
        ended_at_ms = wall_time_ms()
        effective_started_at_ms = self.session_started_at_ms or ended_at_ms
        if self.microphone_outage_started_ms is not None:
            self.microphone_outages.append({
                "started_at_ms": self.microphone_outage_started_ms,
                "ended_at_ms": ended_at_ms,
            })
            self.microphone_outage_started_ms = None
        session_directory = self.session_log.directory if self.session_log is not None else None
        transcript_path = self.session_log.events_path if self.session_log is not None else None
        audio_path = session_directory / "meeting_audio.flac" if session_directory is not None and self.record_audio else None
        summary = {
            "started_at_ms": effective_started_at_ms,
            "ended_at_ms": ended_at_ms,
            "duration_ms": max(0, ended_at_ms - effective_started_at_ms),
            "last_audio_ms": self.latest_audio_ms,
            "committed_captions": len(self.committed_segments),
            "translated_captions": len(self.translated_segments),
            "max_delay_ms": self.max_delay_ms,
            "microphone_outages": list(self.microphone_outages),
            "transcript_path": str(transcript_path) if transcript_path is not None else "",
            "audio_path": str(audio_path) if audio_path is not None else "",
            "outstanding": self.outstanding,
        }
        event = SessionEvent(
            session_id=self.session_id,
            action="ended",
            emitted_at_ms=ended_at_ms,
            details=summary,
        )
        if self.session_log is not None:
            self.session_log.append(event.to_dict())
            self.session_log.close()
            self.session_log = None
        await self.transport.broadcast(event.to_dict())
        self.session_id = ""
        self.outstanding = 0
        self.latest_phrases.clear()
        self.final_segments.clear()
        self.request_revisions.clear()
        self.event_revisions.clear()
        self.stability = StabilityTracker()
        await self.publish_health(ServiceState.IDLE, "Session complete")
        self.session_done.set()

    async def shutdown(self) -> None:
        await self.stop_session()
        await self.transport.close()

    def _new_worker(self) -> GPUWorkerClient:
        return GPUWorkerClient(WorkerSettings(
            primary_asr=self.config.models.primary_asr,
            fallback_asr=self.config.models.fallback_asr,
            translator=self.config.models.translator,
            prompt_terms=self.glossary.prompt_terms,
            sample_rate=self.config.audio.sample_rate,
        ))

    async def _capture_loop(self) -> None:
        cfg = self.config.audio
        phrase_buffer = PhraseBuffer(
            sample_rate=cfg.sample_rate,
            frame_ms=cfg.frame_ms,
            aggressiveness=cfg.vad_aggressiveness,
            silence_close_ms=cfg.silence_close_ms,
            preroll_ms=cfg.preroll_ms,
            partial_interval_ms=cfg.partial_interval_ms,
            max_phrase_ms=cfg.max_phrase_ms,
            overlap_ms=cfg.overlap_ms,
        )
        if self.replay_path:
            source = ReplaySource(
                self.replay_path,
                sample_rate=cfg.sample_rate,
                frame_ms=cfg.frame_ms,
                speed=self.replay_speed,
                limit_seconds=self.replay_limit,
            )
            frames = source.frames()
        else:
            self.mixer = DualSourceMixer(
                PulseSource(self.desktop_source, cfg.sample_rate, cfg.frame_ms),
                PulseSource(self.microphone_source, cfg.sample_rate, cfg.frame_ms)
                if self.microphone_source else None,
            )
            self.mixer.microphone_enabled = self.microphone_enabled
            frames = self.mixer.frames()
        try:
            async for frame in frames:
                if self.paused:
                    continue
                self.latest_audio_ms = frame.end_ms
                if self.mixer is not None:
                    now = time.monotonic()
                    if now - self.last_level_event_at >= 0.1:
                        self.last_level_event_at = now
                        await self.transport.broadcast(AudioLevelEvent(
                            desktop_level=self.mixer.desktop_level,
                            microphone_level=self.mixer.microphone_level,
                            microphone_available=not self.mixer.microphone_failed and self.mixer.microphone is not None,
                        ).to_dict())
                if self.session_log is not None:
                    self.session_log.append_audio(frame.samples)
                for update in phrase_buffer.push(frame):
                    self._submit_phrase(update)
            final = phrase_buffer.flush()
            if final is not None:
                self._submit_phrase(final)
            deadline = time.monotonic() + 180
            while self.outstanding > 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.publish_health(ServiceState.ERROR, f"Audio capture failed: {error}")
        finally:
            if self.replay_path and self.session_id:
                asyncio.create_task(self.stop_session())

    def _submit_phrase(self, update: PhraseUpdate) -> None:
        if self.worker is None:
            return
        revision = self.request_revisions.get(update.segment_id, 0) + 1
        self.request_revisions[update.segment_id] = revision
        self.latest_phrases[update.segment_id] = update
        self.worker.submit_asr(
            segment_id=update.segment_id,
            revision=revision,
            final=update.kind == "final",
            model_name=self.active_model,
            audio=update.audio,
            start_ms=update.start_ms,
            end_ms=update.end_ms,
        )
        self.outstanding += 1
        self.asr_outstanding += 1

    async def _consume_results(self) -> None:
        assert self.worker is not None
        while True:
            result = await asyncio.to_thread(self.worker.get_result, 0.2)
            if result is None:
                if not self.worker.alive():
                    await self._restart_worker("GPU worker exited")
                continue
            if result.kind == "ready":
                self.worker_ready.set()
                await self.publish_health(ServiceState.LISTENING, "Listening", result.gpu_memory_mb)
                continue
            self.outstanding = max(0, self.outstanding - 1)
            if result.kind == "asr" or result.model_name in {
                self.config.models.primary_asr,
                self.config.models.fallback_asr,
            }:
                self.asr_outstanding = max(0, self.asr_outstanding - 1)
            if result.kind == "superseded":
                continue
            if result.kind == "error":
                await self.publish_health(ServiceState.ERROR, result.error.splitlines()[-1], result.gpu_memory_mb)
                if self.restart_count == 0:
                    await self._restart_worker("Inference failed")
                continue
            if result.kind == "asr":
                await self._handle_asr(result)
            else:
                await self._handle_translation(result)

    async def _handle_asr(self, result: InferenceResult) -> None:
        latest_revision = self.request_revisions.get(result.segment_id, 0)
        if result.revision < latest_revision and result.final:
            return
        self.last_result_audio_ms = max(self.last_result_audio_ms, result.audio_end_ms)
        stable = self.stability.update(result.segment_id, result.text, result.final)
        if stable is None:
            return
        text, _ = stable
        corrections = ()
        if result.final:
            text, corrections = self.glossary.apply(text)
        revision = self.event_revisions.get(result.segment_id, 0) + 1
        self.event_revisions[result.segment_id] = revision
        event = CaptionEvent(
            session_id=self.session_id,
            segment_id=result.segment_id,
            revision=revision,
            state=CaptionState.COMMITTED if result.final else CaptionState.PROVISIONAL,
            audio_start_ms=result.audio_start_ms,
            audio_end_ms=result.audio_end_ms,
            emitted_at_ms=wall_time_ms(),
            mandarin=text,
            asr_model=result.model_name,
            corrections=corrections,
        )
        await self.publish(event.to_dict())
        if result.final:
            self.committed_segments.add(result.segment_id)
        if result.final and self.worker is not None:
            self.final_segments[result.segment_id] = {
                "mandarin": text,
                "corrections": corrections,
                "start_ms": result.audio_start_ms,
                "end_ms": result.audio_end_ms,
            }
            self.worker.submit_translation(
                result.segment_id, revision, text, result.audio_start_ms, result.audio_end_ms
            )
            self.outstanding += 1

    async def _handle_translation(self, result: InferenceResult) -> None:
        segment = self.final_segments.get(result.segment_id)
        if segment is None:
            return
        revision = self.event_revisions.get(result.segment_id, 0) + 1
        self.event_revisions[result.segment_id] = revision
        event = CaptionEvent(
            session_id=self.session_id,
            segment_id=result.segment_id,
            revision=revision,
            state=CaptionState.COMMITTED,
            audio_start_ms=segment["start_ms"],
            audio_end_ms=segment["end_ms"],
            emitted_at_ms=wall_time_ms(),
            mandarin=segment["mandarin"],
            english=result.text,
            asr_model=self.active_model,
            corrections=segment["corrections"],
        )
        await self.publish(event.to_dict())
        self.translated_segments.add(result.segment_id)

    def _asr_backlog_seconds(self) -> float:
        if self.asr_outstanding == 0:
            return 0.0
        return max(0.0, (self.latest_audio_ms - self.last_result_audio_ms) / 1000)

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(1)
            if self.mixer is not None and self.mixer.microphone_failed and not self.microphone_failure_announced:
                self.microphone_failure_announced = True
                self.microphone_enabled = False
                lost_at_ms = wall_time_ms()
                self.microphone_outage_started_ms = self.microphone_outage_started_ms or lost_at_ms
                await self.publish(SessionEvent(
                    session_id=self.session_id,
                    action="microphone_lost",
                    emitted_at_ms=lost_at_ms,
                    details={"microphone_source": self.microphone_source},
                ).to_dict())
                await self.publish_health(self._state, "Microphone unavailable; remote audio continues")
            backlog = self._asr_backlog_seconds()
            now = time.monotonic()
            if not self.degraded and backlog >= self.config.overload.enter_backlog_seconds:
                self.degraded = True
                self.active_model = self.config.models.fallback_asr
                self.recovery_started = None
                await self.publish_health(ServiceState.DEGRADED, "Catching up with faster ASR")
            elif self.degraded:
                if backlog <= self.config.overload.exit_backlog_seconds:
                    self.recovery_started = self.recovery_started or now
                    if now - self.recovery_started >= self.config.overload.recovery_hold_seconds:
                        self.degraded = False
                        self.active_model = self.config.models.primary_asr
                        self.recovery_started = None
                        await self.publish_health(ServiceState.LISTENING, "Primary ASR restored")
                else:
                    self.recovery_started = None
            await self.publish_health(ServiceState.DEGRADED if self.degraded else ServiceState.LISTENING)

    async def _restart_worker(self, reason: str) -> None:
        if self.restart_count >= 1:
            await self.publish_health(ServiceState.ERROR, f"{reason}; restart limit reached")
            return
        self.restart_count += 1
        self.worker_ready.clear()
        if self.worker is not None:
            await asyncio.to_thread(self.worker.close)
        self.worker = self._new_worker()
        self.worker.start()
        self.outstanding = 0
        self.asr_outstanding = 0
        for update in tuple(self.latest_phrases.values()):
            self._submit_phrase(update)
        await self.publish_health(ServiceState.PREPARING, f"{reason}; restarting inference")

    async def publish_health(self, state: ServiceState, message: str = "", gpu_memory_mb: int = 0) -> None:
        self._state = state
        backlog = self._asr_backlog_seconds()
        self.max_delay_ms = max(self.max_delay_ms, round(backlog * 1000))
        event = HealthEvent(
            state=state,
            message=message,
            backlog_seconds=backlog,
            active_asr_model=self.active_model,
            gpu_memory_mb=gpu_memory_mb,
            current_delay_ms=round(backlog * 1000),
            microphone_enabled=self.microphone_enabled,
        )
        await self.publish(event.to_dict())

    async def publish(self, payload: dict[str, Any]) -> None:
        if self.session_log is not None:
            try:
                self.session_log.append(payload)
            except OSError as error:
                self.session_log = None
                payload = HealthEvent(state=ServiceState.ERROR, message=f"Persistence disabled: {error}").to_dict()
        await self.transport.broadcast(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--replay-limit", type=float)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--record-audio", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.glossary)
    service = CaptionService(config, args.replay, args.replay_speed, args.replay_limit)
    service.record_audio = args.record_audio or config.storage.record_audio
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, service.stop_event.set)
    await service.run(auto_start=args.auto_start)


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
