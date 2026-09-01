# Live Caption Overlay Plan

## Product decision

Build a Linux-first, system-wide desktop caption overlay using Python and PySide6. It captures both remote meeting audio and the user's microphone, assuming headphones prevent acoustic feedback. English is the primary caption; Mandarin remains visible as a smaller auditable source line.

The first release uses Qwen stability plus a verified terminology glossary. FireRed does not run during the meeting. It remains an optional post-meeting checker because the tested dual-model pipeline increased latency and did not beat Qwen overall.

## First-release scope

### Included

- Transparent, always-on-top, click-through PySide6 overlay.
- Bottom-center placement with adjustable offset and width.
- Two caption lines: large English, smaller Mandarin.
- Two recent caption blocks plus a searchable history drawer.
- Tray controls and configurable global hotkeys.
- Online-meeting audio from the desktop monitor source and microphone.
- Separate capture streams mixed on a common monotonic timeline.
- Qwen3-ASR-1.7B primary recognition.
- Automatic Qwen3-ASR-0.6B fallback when the inference backlog grows.
- Glossary-guided Qwen3-4B English translation.
- Provisional, committed, and corrected caption states.
- Timestamped append-only JSONL session log.
- Transcript saved by default; full audio recording requires an explicit per-meeting toggle.
- Linux implementation behind portable audio, hotkey, and window adapters.

### Excluded

- Live FireRed consensus.
- Speaker diarization or participant-name inference.
- Cloud transcription or uploading meeting audio.
- Windows support in the first release.
- Automatic meeting summaries in the live critical path.
- Software acoustic echo cancellation; headphones are required when microphone capture is enabled.

## User experience

### Overlay

- Default location: bottom center, above typical meeting control bars.
- Maximum width: 70% of the active display.
- Background: translucent dark panel with high-contrast text.
- English: primary line, approximately 30–36 px depending on display scale.
- Mandarin: secondary line, approximately 18–22 px.
- Provisional text: muted color.
- Committed text: full contrast.
- Corrected terminology: briefly highlighted; committed text is never replaced silently.
- Status indicators are compact: listening, translating, catching up, paused, degraded, or error.

### Controls

Tray menu:

- Start or end meeting session.
- Pause or resume capture.
- Show or hide overlay.
- Open history drawer.
- Select desktop and microphone devices.
- Enable full-audio recording for this meeting.
- Open glossary settings.
- Open the current session folder.

Default configurable hotkeys:

- `Ctrl+Alt+C`: start or end session.
- `Ctrl+Alt+Space`: pause or resume.
- `Ctrl+Alt+O`: show or hide overlay.
- `Ctrl+Alt+H`: open or close history.
- `Ctrl+Alt+M`: temporarily mute microphone capture.

## Runtime architecture

```text
PulseAudio/PipeWire monitor ─┐
                            ├─> Audio timeline/mixer ─> VAD/phrase buffer
Microphone capture ─────────┘                              │
                                                          v
                                              Priority inference scheduler
                                               ├─ Qwen3-ASR-1.7B
                                               ├─ Qwen3-ASR-0.6B fallback
                                               └─ Qwen3-4B 4-bit translator
                                                          │
                                                          v
                                                Caption state machine
                                               ├─ provisional
                                               ├─ committed
                                               └─ corrected
                                                  │              │
                                                  v              v
                                            PySide6 UI       JSONL session log
```

### Process boundaries

1. **Desktop process**
   - Owns PySide6 windows, tray icon, history drawer, and global hotkeys.
   - Never loads GPU models.
   - Remains responsive if inference fails.

2. **Caption service process**
   - Owns audio capture, resampling, mixing, VAD, phrase construction, model scheduling, glossary handling, and persistence.
   - Publishes typed caption and health events to the desktop process.

3. **GPU worker process**
   - Owns all CUDA model instances to prevent repeated initialization and allocator fragmentation.
   - Uses a priority queue: active ASR first and visible-caption translation second.
   - Coalesces queued revisions of the same active phrase so obsolete partials do not consume GPU time; committed revisions and translation tasks are never discarded.
   - Preloads Qwen3-ASR-1.7B and the double-quantized 4-bit Qwen3-4B translator. Qwen3-ASR-0.6B loads only when overload mode first needs it, reducing normal-session GPU allocation by approximately 1.5 GB without changing model output.
   - Caps ASR and caption translation generation at 512 new tokens. This matches the ASR model limit and remains far above the output required by an eight-second caption phrase.

Communication uses versioned newline-delimited JSON messages over a local Unix socket. A transport adapter can replace the Unix socket with localhost TCP on Windows later.

## Core contracts

### Caption event

```json
{
  "schema_version": 1,
  "session_id": "uuid",
  "segment_id": "uuid",
  "revision": 2,
  "state": "provisional|committed|corrected",
  "audio_start_ms": 12400,
  "audio_end_ms": 16950,
  "emitted_at_ms": 17600,
  "mandarin": "...",
  "english": "...",
  "asr_model": "Qwen3-ASR-1.7B",
  "corrections": [
    {"before": "D135i", "after": "D435i", "reason": "verified_glossary"}
  ]
}
```

### Health event

Carries capture state, queue backlog, active model, GPU memory, current delay, dropped-frame count, and the last recoverable error.

### Session log

Append-only JSONL records session metadata, caption revisions, health transitions, device changes, pause intervals, model versions, glossary version, and optional audio-file references. Revisions never destroy earlier text.

## Audio pipeline

- Discover sources through a Linux audio adapter; initial known devices are the desktop monitor and Logitech C930e microphone.
- Capture each source independently with monotonic timestamps.
- Convert both streams to 16 kHz mono float32.
- Apply per-source level normalization and a limiter before mixing.
- Do not attempt echo cancellation; the product records the headphone requirement in session readiness checks.
- Run CPU VAD on 20–30 ms frames.
- Emit provisional inference after roughly 1.5–2 seconds of speech.
- Close a phrase after approximately 700 ms silence or at an 8-second maximum phrase length.
- Preserve a small overlap when a maximum-length phrase is split.

## Caption stability and translation

1. Run Qwen on the current phrase buffer.
2. Compare consecutive results using stable-prefix matching.
3. Publish a provisional Mandarin revision only when the stable prefix advances materially.
4. Commit Mandarin when VAD closes the phrase or when the prefix remains stable across two decodes.
5. Apply conservative glossary correction only when phonetic similarity and local context support the replacement.
6. Translate committed Mandarin, not unstable token-by-token partials.
7. Publish English as soon as translation completes.
8. Never rewrite a committed caption without emitting a corrected revision and a visible short highlight.

## Overload and failure behavior

### Degraded mode

- Enter degraded mode when queued audio exceeds 5 seconds or when projected final-caption delay exceeds 7 seconds.
- Switch new phrases from Qwen3-ASR-1.7B to Qwen3-ASR-0.6B.
- Keep displaying a compact `Catching up` indicator.
- Return to 1.7B only after backlog remains below 2 seconds for 15 seconds.
- Record every model transition in the session log.
- Never drop committed audio silently.

### Recoverable failures

- Audio device disappears: freeze captions, show the failed device, and retry discovery.
- Microphone disappears: continue remote-only capture and show a warning.
- Translator fails: continue Mandarin captions and queue committed phrases for later translation.
- ASR worker fails: restart it once, preserve buffered audio, and expose the restart state.
- UI process fails: caption service continues logging; reconnecting the UI restores recent state.
- Disk write fails: continue live captions, disable persistence, and show a persistent warning.

## Proposed repository structure

```text
transcribe_program/
├── PLAN.md
├── pyproject.toml
├── config/
│   ├── default.toml
│   └── glossary.json
├── src/meeting_translation/
│   ├── app.py
│   ├── domain/
│   │   ├── events.py
│   │   └── state.py
│   ├── audio/
│   │   ├── base.py
│   │   ├── pulse.py
│   │   ├── mixer.py
│   │   └── vad.py
│   ├── inference/
│   │   ├── worker.py
│   │   ├── scheduler.py
│   │   ├── qwen_asr.py
│   │   ├── stability.py
│   │   └── translation.py
│   ├── glossary/
│   │   └── matcher.py
│   ├── transport/
│   │   ├── protocol.py
│   │   └── unix_socket.py
│   ├── storage/
│   │   └── session_log.py
│   └── ui/
│       ├── overlay.py
│       ├── history.py
│       ├── tray.py
│       └── settings.py
├── scripts/
│   └── replay_meeting.py
└── tests/
```

## Delivery phases

### Phase 1 — Live-latency proof

- Build the shared audio-frame and caption-event contracts.
- Feed the existing meeting recording through the pipeline at real-time speed.
- Implement VAD, rolling phrases, provisional Qwen decoding, stability tracking, and committed Mandarin events.
- Measure first-caption latency, commit latency, backlog, and VRAM before building UI polish.

Exit condition: ten-minute replay completes without backlog growth and meets provisional/commit latency targets.

### Phase 2 — Translation and persistence

- Add the 4-bit Qwen3-4B translator under the priority scheduler.
- Add glossary correction and versioning.
- Add append-only session JSONL.
- Add optional audio recording.

Exit condition: bilingual output covers the replay without blocking ASR, and every visible revision is represented in the log.

### Phase 3 — Desktop overlay

- Implement transparent always-on-top overlay, click-through mode, display selection, tray controls, and hotkeys.
- Implement two-caption presentation and status indicators.
- Implement the searchable history drawer.

Exit condition: overlay remains usable above a browser meeting and does not intercept clicks while locked.

### Phase 4 — Live audio integration

- Implement desktop monitor and microphone adapters.
- Add device readiness checks and separate source meters.
- Validate headphone-based dual capture in a real online meeting.

Exit condition: remote and local speech are both captioned without feedback or duplicate remote speech.

### Phase 5 — Resilience and fallback

- Add 0.6B overload fallback, model transition policy, worker restart, device loss recovery, and disk-failure behavior.
- Run a complete 45-minute real-time replay and a live meeting smoke test.

Exit condition: no audio is silently lost; every degradation is visible and logged.

## Acceptance criteria

### Latency

- First provisional Mandarin caption: p50 ≤ 2.0 seconds and p95 ≤ 3.0 seconds after speech becomes intelligible.
- Committed Mandarin: p95 ≤ 2.0 seconds after phrase end.
- English caption: p95 ≤ 6.0 seconds after phrase end.
- No continuously growing backlog during a 45-minute real-time replay.

### Reliability

- 45-minute replay completes without process crash, CUDA out-of-memory, or lost committed segments.
- Peak GPU memory remains below 15 GB.
- Overlay stays responsive while the GPU worker is busy or restarting.
- Audio-device loss and translator failure degrade visibly without terminating the session.

### Caption behavior

- Provisional and committed states are visually distinguishable.
- Committed revisions are never silently overwritten.
- English and Mandarin retain the same segment and timestamp identity.
- Glossary corrections record the original text, replacement, reason, and glossary version.
- Qwen3-ASR-0.6B fallback activates and recovers at the specified thresholds.

### Interface

- Overlay remains above Zoom, Teams, and browser meetings on the target Linux desktop.
- Locked overlay is click-through.
- Tray actions and global hotkeys work while another application has focus.
- History search returns Mandarin, English, model names, and technical terms.
- Transcript saving defaults on; audio recording defaults off and is visibly indicated when enabled.

## UX implementation roadmap

The interface work is ordered by operational risk rather than appearance. Audio confidence, session safety, and recoverability ship before cosmetic refinement.

### UX phase A — Audio confidence and meeting safety

1. Replace PulseAudio identifiers with device descriptions while retaining the exact source name in tooltips and command payloads.
2. Show live meeting-audio and microphone level meters driven by captured frames, not configured PulseAudio volume.
3. Collapse source setup after a session starts; keep Audio expansion and microphone mute available.
4. Rename ambiguous actions to `Hide overlay` and `Quit`; quitting an active session requires an explicit end-session confirmation.
5. Preserve disconnected selections, label them as disconnected, disable invalid microphone actions, and require a valid meeting-audio source before Start.
6. Add a preflight panel for meeting audio, microphone, model readiness, and transcript persistence.
7. Promote READY, LISTENING, PAUSED, MIC MUTED, RECONNECTING, and ERROR to explicit text-and-color states.

Exit condition: users can identify both sources, see incoming signal, and cannot unknowingly start or terminate a broken session.

### UX phase B — Meeting focus and caption stability

1. Separate setup controls from frequent meeting actions.
2. Keep mute state visible beside the selected microphone and in compact mode.
3. Preserve fixed caption space and fade revisions so provisional updates do not move surrounding content.
4. Strengthen provisional, committed, translated, and corrected caption styling without relying on color alone.
5. Add focused keyboard shortcuts with discoverable tooltips.
6. Increase frequent-action targets and isolate destructive actions.
7. Turn capture, translation, and device failures into messages with direct recovery actions.

Exit condition: normal meeting operation needs no setup controls and captions remain spatially stable.

### UX phase C — Personalization and accessibility

1. Theme selector popups with high contrast, selection highlighting, and device-type indicators.
2. Add constrained background/text opacity controls that preserve readable contrast.
3. Add Compact, Standard, Presentation, and High Contrast layout presets.
4. Remember placement per monitor and provide explicit monitor and edge-placement actions.

Exit condition: the overlay remains readable and reachable across display and accessibility configurations.

### UX phase D — Review and session completion

1. Extend history with timestamp navigation, segment/full-transcript copy, state labels, and correction markers.
2. Show a post-session summary with duration, transcript path, committed-caption count, capture gaps, translation gaps, and open/copy actions.

Exit condition: users can recover meeting context and locate the saved result without browsing the filesystem.

## First implementation task

Build the real-time replay harness and latency instrumentation before the visual overlay. The existing benchmark proves offline throughput but does not prove incremental Qwen behavior, phrase stability, translation scheduling, or live delay. The replay harness resolves those risks before UI code depends on them.

## Implemented runbook

Use the launcher from the self-contained program folder. Its Python environment and Qwen model cache are stored beside the source:

```bash
cd /home/farhan/Documents/transcribe_program
./run_caption.sh
```

The isolated Qwen-ASR plus Hy-MT2 candidate uses the same application code with an alternate `uv` environment, model cache, and translator override:

```bash
MEETING_TRANSLATION_PYTHON="/media/farhan/My Passport/bstc_external/.venv-hybrid/bin/python" \
MEETING_TRANSLATION_TRANSLATOR="/media/farhan/My Passport/bstc_external/models/hub/models--tencent--Hy-MT2-1.8B/snapshots/9a341cd1b679d3efd23b46e847b01745a71ed792" \
HF_HOME="/home/farhan/Documents/transcribe_program/models/huggingface" \
HF_HUB_OFFLINE=1 \
./run_caption.sh
```

`MEETING_TRANSLATION_TRANSLATOR` changes only the translator model; ASR, audio, storage, and interface configuration remain unchanged. The default configuration continues to select Qwen3-4B until the full-meeting candidate is accepted.

Before Start, a preflight dialog verifies the caption-service connection, selected PulseAudio sources, current signal activity, writable transcript storage, model load policy, and recording choice. Both source lists refresh whenever opened and show readable hardware descriptions; exact PulseAudio identifiers remain in tooltips. A disconnected microphone does not stop remote-audio capture. `Refresh devices` and `Reconnect microphone` restore the selected device explicitly, and microphone loss/recovery intervals are recorded in the session log. Caption-service failures expose a `Retry service` action.

Starting a session applies the persisted compact policy: immediately, after five seconds, or never. Both audio meters and microphone mute remain available while selectors are collapsed. `Space`, `M`, `H`, and `Esc` provide focused-window pause, microphone, history, and collapse controls; the existing global shortcuts remain available. `Hide overlay` leaves transcription running. Lock mode shows a recovery hint and tray notification. `Quit` exits immediately while idle; during a session it asks, stops capture, finishes queued captions, closes transcript/audio files, and only then exits.

`Appearance` provides Compact, Standard, Presentation, and High Contrast presets; opacity choices; multi-display placement; compact timing; and a persisted reduced-motion preference. Caption blocks reserve stable space and label revisions as `LIVE`, `FINAL`, or `CORRECTED`. Controls have accessible names, keyboard focus, minimum 32-pixel targets, non-color lifecycle labels, labeled audio meters, and visible muted/destructive states.

History supports text search, `LIVE`/`FINAL`/`CORRECTED` filtering, timestamp navigation, optional model names, correction before/after/reason inspection, language-specific copying, and session export. Normal Stop opens a summary with duration, committed/translated counts, maximum delay, microphone interruptions, transcript/recording paths, and open/copy actions.

- `Ctrl+Alt+C`: start or end the session.
- `Ctrl+Alt+Space`: pause or resume.
- `Ctrl+Alt+O`: show or hide the overlay.
- `Ctrl+Alt+H`: open or close history.
- `Ctrl+Alt+M`: mute or unmute microphone capture.

Replay the normalized meeting recording at real-time speed:

```bash
./run_caption.sh \
  --replay "/path/to/meeting_16k_mono.wav" \
  --replay-limit 600 \
  --auto-start
```

Run the measured headless replay and enforce the latency targets:

```bash
./run_caption.sh --metrics \
  "/path/to/meeting_16k_mono.wav" \
  replay_metrics.json \
  --limit 600 \
  --speed 1 \
  --assert-targets
```

`config/default.toml` uses PulseAudio's `@DEFAULT_MONITOR@` for remote meeting audio and the Logitech C930e source for the local microphone. Use headphones to prevent acoustic echo. If PulseAudio assigns a different microphone source name after reconnecting the webcam, update `audio.microphone_source` from `pactl list short sources`.

Session events are appended to `sessions/<timestamp>_<session-id>/events.jsonl`. Full meeting audio is off by default and is written beside the event log only when enabled before the session starts.
