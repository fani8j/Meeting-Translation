from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .config import load_config
from .ui import DesktopController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--replay-limit", type=float)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--record-audio", action="store_true")
    parser.add_argument("--no-backend", action="store_true")
    parser.add_argument("--quit-after", type=float)
    return parser


def launch_backend(args: argparse.Namespace) -> subprocess.Popen:
    command = [sys.executable, "-m", "meeting_translation.service"]
    if args.config:
        command.extend(["--config", str(args.config)])
    if args.glossary:
        command.extend(["--glossary", str(args.glossary)])
    if args.replay:
        command.extend(["--replay", str(args.replay), "--replay-speed", str(args.replay_speed)])
    if args.replay_limit is not None:
        command.extend(["--replay-limit", str(args.replay_limit)])
    if args.auto_start:
        command.append("--auto-start")
    if args.record_audio:
        command.append("--record-audio")
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    return subprocess.Popen(command, env=environment)


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.glossary)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Meeting Translation")
    app.setQuitOnLastWindowClosed(False)
    backend = None if args.no_backend else launch_backend(args)
    controller = DesktopController(app, config, backend)
    if args.quit_after:
        QTimer.singleShot(round(args.quit_after * 1000), controller.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
