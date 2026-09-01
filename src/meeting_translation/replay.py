from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def wait_for_socket(path: Path, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                reader, writer = await asyncio.open_unix_connection(path)
                writer.close()
                await writer.wait_closed()
                del reader
                return
            except OSError:
                pass
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Caption service socket did not appear: {path}")


async def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, args.glossary)
    command = [
        sys.executable,
        "-m",
        "meeting_translation.service",
        "--replay", str(args.audio),
        "--replay-speed", str(args.speed),
    ]
    if args.limit is not None:
        command.extend(["--replay-limit", str(args.limit)])
    if args.config:
        command.extend(["--config", str(args.config)])
    if args.glossary:
        command.extend(["--glossary", str(args.glossary)])
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    log_path = args.output.with_suffix(".service.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as service_log:
        process = subprocess.Popen(command, stdout=service_log, stderr=subprocess.STDOUT, env=environment)
        try:
            await wait_for_socket(config.transport.socket_path)
            reader, writer = await asyncio.open_unix_connection(config.transport.socket_path)
            writer.write(b'{"command":"start"}\n')
            await writer.drain()
            events: list[dict[str, Any]] = []
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=max(300, (args.limit or 30) / max(args.speed, 0.01) + 300))
                if not line:
                    break
                payload = json.loads(line)
                events.append(payload)
                if payload.get("event") == "session" and payload.get("action") == "ended":
                    break
            writer.write(b'{"command":"shutdown"}\n')
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    started = next(event for event in events if event.get("event") == "session" and event.get("action") == "started")
    started_ms = int(started["emitted_at_ms"])
    provisional: list[float] = []
    committed: list[float] = []
    english: list[float] = []
    caption_events = [event for event in events if event.get("event") == "caption"]
    seen_commit: set[str] = set()
    seen_english: set[str] = set()
    for event in caption_events:
        expected_wall = started_ms + int(event["audio_end_ms"] / args.speed)
        latency = max(0.0, int(event["emitted_at_ms"]) - expected_wall) / 1000
        if event["state"] == "provisional":
            provisional.append(latency)
        elif event["segment_id"] not in seen_commit and not event.get("english"):
            seen_commit.add(event["segment_id"])
            committed.append(latency)
        if event.get("english") and event["segment_id"] not in seen_english:
            seen_english.add(event["segment_id"])
            english.append(latency)
    health = [event for event in events if event.get("event") == "health"]
    degraded_entries = 0
    previous_state = ""
    for event in health:
        state = str(event.get("state", ""))
        if state == "degraded" and previous_state != "degraded":
            degraded_entries += 1
        previous_state = state
    metrics = {
        "audio": str(args.audio),
        "speed": args.speed,
        "limit_seconds": args.limit,
        "caption_events": len(caption_events),
        "segments_committed": len(seen_commit),
        "segments_translated": len(seen_english),
        "provisional_latency_seconds": {
            "p50": percentile(provisional, 0.50),
            "p95": percentile(provisional, 0.95),
        },
        "commit_latency_seconds": {
            "p50": percentile(committed, 0.50),
            "p95": percentile(committed, 0.95),
        },
        "english_latency_seconds": {
            "p50": percentile(english, 0.50),
            "p95": percentile(english, 0.95),
        },
        "max_backlog_seconds": max((float(event.get("backlog_seconds", 0)) for event in health), default=0),
        "max_gpu_memory_mb": max((int(event.get("gpu_memory_mb", 0)) for event in health), default=0),
        "degraded_transitions": degraded_entries,
        "errors": [event.get("message", "") for event in health if event.get("state") == "error"],
        "service_log": str(log_path),
    }
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--assert-targets", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.speed <= 0:
        raise SystemExit("Replay speed must be positive for latency measurement")
    metrics = asyncio.run(run_replay(args))
    print(json.dumps(metrics, indent=2))
    if args.assert_targets:
        failures = []
        provisional_p95 = metrics["provisional_latency_seconds"]["p95"]
        commit_p95 = metrics["commit_latency_seconds"]["p95"]
        english_p95 = metrics["english_latency_seconds"]["p95"]
        if provisional_p95 is None or provisional_p95 > 3:
            failures.append(f"provisional p95={provisional_p95}")
        if commit_p95 is None or commit_p95 > 2:
            failures.append(f"commit p95={commit_p95}")
        if english_p95 is None or english_p95 > 6:
            failures.append(f"English p95={english_p95}")
        if metrics["errors"]:
            failures.append(f"errors={metrics['errors']}")
        if failures:
            raise SystemExit("Latency targets failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
