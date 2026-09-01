import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from meeting_translation.config import load_config
from meeting_translation.service import CaptionService
from meeting_translation.storage import SessionLog
from meeting_translation.transport import JsonlServer
from meeting_translation.worker import (
    InferenceTask,
    WorkerSettings,
    _coalesce_pending,
    _contains_cjk,
    _remove_unsupported_generation_inputs,
    _startup_asr_models,
)


class TransportStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_receives_command_and_broadcasts_event(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "caption.sock"
            received = []

            async def handle(message):
                received.append(message)

            server = JsonlServer(socket_path, handle)
            await server.start()
            reader, writer = await asyncio.open_unix_connection(socket_path)
            writer.write(b'{"command":"pause"}\n')
            await writer.drain()
            await asyncio.sleep(0.02)
            await server.broadcast({"event": "health", "state": "paused"})
            payload = json.loads(await reader.readline())
            self.assertEqual(received, [{"command": "pause"}])
            self.assertEqual(payload["state"], "paused")
            writer.close()
            await writer.wait_closed()
            await server.close()
            self.assertFalse(socket_path.exists())

    async def test_session_log_is_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            log = SessionLog(Path(directory), "session", 16000, record_audio=False)
            log.append({"event": "caption", "revision": 1})
            log.append({"event": "caption", "revision": 2})
            path = log.events_path
            log.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["revision"] for row in rows], [1, 2])

    async def test_start_command_applies_selected_audio_sources(self):
        service = CaptionService(load_config())
        service.start_session = AsyncMock()
        await service.handle_command({
            "command": "start",
            "desktop_source": "meeting.monitor",
            "microphone_source": "",
        })
        self.assertEqual(service.desktop_source, "meeting.monitor")
        self.assertEqual(service.microphone_source, "")
        self.assertFalse(service.microphone_enabled)
        service.start_session.assert_awaited_once()

    async def test_reconnect_microphone_resets_failure_and_logs_recovery(self):
        service = CaptionService(load_config())
        service.session_id = "session"
        service.microphone_outage_started_ms = 100
        service.mixer = MagicMock()
        service.mixer.reconnect_microphone = AsyncMock()
        service.publish = AsyncMock()
        service.publish_health = AsyncMock()
        await service.handle_command({
            "command": "reconnect_microphone",
            "microphone_source": "returned.microphone",
        })
        service.mixer.reconnect_microphone.assert_awaited_once()
        recovery = service.publish.await_args.args[0]
        self.assertEqual(recovery["action"], "microphone_recovered")
        self.assertEqual(service.microphone_source, "returned.microphone")
        self.assertIsNone(service.microphone_outage_started_ms)
        self.assertEqual(len(service.microphone_outages), 1)

    async def test_stop_closes_storage_before_broadcasting_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            service = CaptionService(load_config())
            service.session_id = "session"
            service.session_started_at_ms = 100
            service.session_log = SessionLog(Path(directory), "session", 16000, record_audio=False)
            service.committed_segments = {"one", "two"}
            service.translated_segments = {"one"}
            service.max_delay_ms = 1200
            service.publish_health = AsyncMock()

            async def verify_closed(payload):
                self.assertTrue(service.session_log is None)
                self.assertEqual(payload["action"], "ended")

            service.transport.broadcast = AsyncMock(side_effect=verify_closed)
            await service.stop_session()
            summary = service.transport.broadcast.await_args.args[0]["details"]
            self.assertEqual(summary["committed_captions"], 2)
            self.assertEqual(summary["translated_captions"], 1)
            self.assertEqual(summary["max_delay_ms"], 1200)
            transcript_path = Path(summary["transcript_path"])
            self.assertTrue(transcript_path.exists())
            last_event = json.loads(transcript_path.read_text().splitlines()[-1])
            self.assertEqual(last_event["action"], "ended")

    def test_worker_coalesces_superseded_partial_revisions(self):
        tasks = [
            InferenceTask(0, 1, "asr", "segment-a", 1, audio=np.zeros(10, dtype=np.float32)),
            InferenceTask(1, 2, "translate", "segment-finished", 1, text="完成"),
            InferenceTask(0, 3, "asr", "segment-a", 2, audio=np.ones(20, dtype=np.float32)),
            InferenceTask(0, 4, "asr", "segment-b", 1, audio=np.ones(10, dtype=np.float32)),
        ]
        retained, dropped = _coalesce_pending(tasks)
        retained_asr = {
            (task.segment_id, task.revision) for task in retained if task.kind == "asr"
        }
        self.assertEqual(retained_asr, {("segment-a", 2), ("segment-b", 1)})
        self.assertEqual([(task.segment_id, task.revision) for task in dropped], [("segment-a", 1)])
        self.assertEqual(sum(task.kind == "translate" for task in retained), 1)

    def test_worker_eagerly_loads_only_primary_asr(self):
        settings = WorkerSettings(
            primary_asr="primary",
            fallback_asr="fallback",
            translator="translator",
            prompt_terms="",
        )
        self.assertEqual(_startup_asr_models(settings), ("primary",))

    def test_translator_can_be_selected_without_changing_default_config(self):
        with patch.dict("os.environ", {"MEETING_TRANSLATION_TRANSLATOR": "tencent/Hy-MT2-1.8B"}):
            configured = load_config()
        self.assertEqual(configured.models.translator, "tencent/Hy-MT2-1.8B")

    def test_translation_inputs_remove_unsupported_token_types(self):
        inputs = {"input_ids": [1, 2], "token_type_ids": [0, 0]}
        returned = _remove_unsupported_generation_inputs(inputs)
        self.assertIs(returned, inputs)
        self.assertEqual(inputs, {"input_ids": [1, 2]})

    def test_translation_output_detects_untranslated_chinese_script(self):
        self.assertTrue(_contains_cjk("The 本体 has not arrived."))
        self.assertFalse(_contains_cjk("The main unit has not arrived."))


if __name__ == "__main__":
    unittest.main()
