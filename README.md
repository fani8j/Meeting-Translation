# Meeting Translation

A Linux desktop overlay for live Mandarin meeting captions with an auditable English translation. It captures desktop meeting audio and, optionally, microphone audio; segments speech with voice-activity detection; preserves a Mandarin source line; applies a project glossary; and stores each session as append-only JSONL.

## Requirements

- Linux desktop session with PulseAudio or PipeWire PulseAudio compatibility.
- A CUDA-capable NVIDIA GPU. The configured Qwen ASR and translation models run on `cuda:0`.
- Python 3.10 or newer.
- [`uv`](https://docs.astral.sh/uv/) to create the local environment.
- Headphones when microphone capture is enabled, to avoid acoustic feedback.

The first model load downloads the configured models unless they are already available in the local Hugging Face cache.

## Setup

```bash
cd /path/to/transcribe_program
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -e .
```

The launcher keeps runtime state beside the source tree:

- `.venv/` — Python environment
- `models/huggingface/` — Hugging Face model cache
- `models/torch/` — PyTorch cache
- `cache/` — `uv` cache
- `sessions/` — local caption history

Those paths, recorded sessions, and replay media are deliberately excluded from version control.

## Run live captions

```bash
./run_caption.sh
```

Before starting a session, select the desktop-audio monitor and optional microphone in the preflight dialog. The default configuration uses PulseAudio's `@DEFAULT_MONITOR@` for remote meeting audio. If the saved microphone source no longer exists, inspect available sources and update `audio.microphone_source` in `config/default.toml`:

```bash
pactl list short sources
```

Default global shortcuts:

| Shortcut | Action |
| --- | --- |
| `Ctrl+Alt+C` | Start or stop session |
| `Ctrl+Alt+Space` | Pause or resume |
| `Ctrl+Alt+O` | Show or hide overlay |
| `Ctrl+Alt+H` | Show history |
| `Ctrl+Alt+M` | Mute or unmute microphone |

Session events are written to `sessions/<timestamp>_<session-id>/events.jsonl`. Audio recording is disabled by default; enable it before a session when required.

## Replay and benchmark audio

Launch the desktop application against a recording:

```bash
./run_caption.sh \
  --replay /path/to/meeting_16k_mono.wav \
  --replay-limit 600 \
  --auto-start
```

Produce replay metrics without the desktop UI:

```bash
./run_caption.sh --metrics \
  /path/to/meeting_16k_mono.wav \
  replay_metrics.json \
  --limit 600
```

The metrics command also writes `replay_metrics.service.log` next to the JSON output.

## Configuration

- `config/default.toml` configures models, audio capture, segmentation, overlay appearance, hotkeys, overload behavior, storage, and the local transport socket.
- `config/glossary.json` defines verified terms and aliases applied to committed captions.
- `MEETING_TRANSLATION_PYTHON` overrides the interpreter used by `run_caption.sh`.
- `MEETING_TRANSLATION_TRANSLATOR` overrides only the configured translator model. This supports isolated experimental translators while keeping ASR, audio, storage, and UI configuration unchanged.

For the lower-memory Hy-MT2 experiment, point the two overrides at an isolated environment and local model snapshot. Keep the standard Qwen translator as the default until representative meeting captions have been reviewed by a human.

## Test

```bash
QT_QPA_PLATFORM=offscreen \
PYTHONNOUSERSITE=1 \
.venv/bin/python -m unittest discover -s tests -v
```

## Project layout

```text
src/meeting_translation/  application, audio pipeline, GPU worker, UI, and replay code
config/              runtime defaults and glossary
tests/               unit and integration coverage
run_caption.sh       local launcher for the application and metrics replay
PLAN.md              detailed architecture and operating runbook
```
