import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from meeting_translation.audio import AudioFrame, DualSourceMixer, ReplaySource
from meeting_translation.vad import PhraseBuffer


class EnergyVad:
    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        del sample_rate
        samples = np.frombuffer(pcm, dtype="<i2")
        return bool(np.max(np.abs(samples)) > 1000)


class FakePulse:
    frame_ms = 30

    def __init__(self, frames):
        self._frames = frames

    async def frames(self):
        for frame in self._frames:
            yield frame

    async def close(self):
        return None


class AudioVadTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_preserves_audio_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            sf.write(path, np.ones(1600, dtype=np.float32) * 0.1, 16000)
            frames = [frame async for frame in ReplaySource(path, speed=100).frames()]
            self.assertEqual(len(frames), 3)
            self.assertEqual((frames[0].start_ms, frames[-1].end_ms), (0, 90))

    async def test_desktop_capture_continues_when_microphone_disappears(self):
        frames = [
            AudioFrame(np.full(480, 0.1, dtype=np.float32), index * 30, (index + 1) * 30, "desktop")
            for index in range(3)
        ]
        mixer = DualSourceMixer(FakePulse(frames), FakePulse([]))
        mixed = [frame async for frame in mixer.frames()]
        self.assertEqual(len(mixed), len(frames))
        self.assertTrue(mixer.microphone_failed)
        self.assertTrue(all(frame.source == "mixed" for frame in mixed))
        self.assertGreater(mixer.desktop_level, 0)
        self.assertEqual(mixer.microphone_level, 0)

    async def test_microphone_can_reconnect_without_stopping_desktop_capture(self):
        desktop_frames = [
            AudioFrame(np.full(480, 0.1, dtype=np.float32), index * 30, (index + 1) * 30, "desktop")
            for index in range(2)
        ]
        microphone_frames = [
            AudioFrame(np.full(480, 0.05, dtype=np.float32), index * 30, (index + 1) * 30, "microphone")
            for index in range(2)
        ]
        mixer = DualSourceMixer(FakePulse(desktop_frames), None)
        mixer.microphone_failed = True
        await mixer.reconnect_microphone(FakePulse(microphone_frames))
        mixed = [frame async for frame in mixer.frames()]
        self.assertEqual(len(mixed), 2)
        self.assertFalse(mixer.microphone_failed)
        self.assertTrue(mixer.microphone_enabled)
        self.assertGreater(mixer.microphone_level, 0)

    async def test_phrase_emits_partial_and_final_after_silence(self):
        buffer = PhraseBuffer(
            partial_interval_ms=60,
            silence_close_ms=60,
            preroll_ms=30,
            max_phrase_ms=1000,
        )
        buffer.vad = EnergyVad()
        updates = []
        for index, amplitude in enumerate([0.0, 0.3, 0.3, 0.3, 0.0, 0.0]):
            samples = np.full(480, amplitude, dtype=np.float32)
            frame = AudioFrame(samples, index * 30, (index + 1) * 30, "test")
            updates.extend(buffer.push(frame))
        self.assertGreaterEqual([update.kind for update in updates].count("partial"), 1)
        self.assertEqual(updates[-1].kind, "final")
        self.assertGreater(len(updates[-1].audio), 0)
        self.assertEqual(updates[-1].end_ms, 120)


if __name__ == "__main__":
    unittest.main()
