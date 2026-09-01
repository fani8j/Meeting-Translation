import json
import tempfile
import unittest
from pathlib import Path

from meeting_translation.domain import CaptionEvent, CaptionState, Correction, parse_event
from meeting_translation.text import Glossary, StabilityTracker


class DomainTextTests(unittest.TestCase):
    def test_caption_event_round_trip_preserves_revision_and_corrections(self):
        original = CaptionEvent(
            session_id="session",
            segment_id="segment",
            revision=3,
            state=CaptionState.CORRECTED,
            audio_start_ms=100,
            audio_end_ms=900,
            emitted_at_ms=1200,
            mandarin="D435i",
            english="Intel RealSense D435i",
            asr_model="qwen",
            corrections=(Correction("D135i", "D435i", "verified_glossary"),),
        )
        recovered = parse_event(original.to_dict())
        self.assertEqual(recovered, original)

    def test_stability_requires_repeat_then_commits_final_text(self):
        tracker = StabilityTracker()
        self.assertIsNone(tracker.update("s", "我们使用 SMC", final=False))
        stable = tracker.update("s", "我们使用 SMC LEHZ32", final=False)
        self.assertEqual(stable, ("我们使用 SMC", 1))
        final = tracker.update("s", "我们使用 SMC LEHZ32", final=True)
        self.assertEqual(final, ("我们使用 SMC LEHZ32", 2))

    def test_glossary_records_each_applied_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glossary.json"
            path.write_text(json.dumps({
                "version": 4,
                "entries": [{"canonical": "D435i", "aliases": ["D135i"]}],
            }), encoding="utf-8")
            glossary = Glossary(path)
            corrected, changes = glossary.apply("camera D135i")
            self.assertEqual(corrected, "camera D435i")
            self.assertEqual(changes[0].reason, "verified_glossary")
            self.assertEqual(glossary.version, 4)


if __name__ == "__main__":
    unittest.main()
