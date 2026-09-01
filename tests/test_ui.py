import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QSettings
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel, QMessageBox

from meeting_translation.config import load_config
from meeting_translation.ui import DesktopController, HistoryDialog, OverlayWindow, PreflightDialog, PulseDevice, RefreshingComboBox


class OverlayUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        QCoreApplication.setOrganizationName("MeetingTranslationTests")
        QCoreApplication.setApplicationName("CaptionUiTests")
        cls.settings_directory = tempfile.TemporaryDirectory()
        QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, cls.settings_directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.settings_directory.cleanup()

    def tearDown(self):
        for widget in QApplication.topLevelWidgets():
            widget.close()

    def test_audio_sources_refresh_when_selector_opens(self):
        first = [
            PulseDevice("speaker.one.monitor", "Conference Display — Meeting output", True),
            PulseDevice("microphone.one", "Webcam Microphone", False),
        ]
        second = [
            PulseDevice("speaker.two.monitor", "Headphones — Meeting output", True),
            PulseDevice("microphone.two", "Headset Microphone", False),
        ]
        with patch("meeting_translation.ui._pulse_sources", return_value=first):
            window = OverlayWindow(load_config())
        self.assertIsInstance(window.desktop_selector, RefreshingComboBox)
        with patch("meeting_translation.ui._pulse_sources", return_value=second):
            window.desktop_selector.refresh_requested.emit()
        desktop_values = [window.desktop_selector.itemData(index) for index in range(window.desktop_selector.count())]
        microphone_values = [window.microphone_selector.itemData(index) for index in range(window.microphone_selector.count())]
        self.assertIn("speaker.two.monitor", desktop_values)
        self.assertIn("microphone.two", microphone_values)

    def test_microphone_mute_is_integrated_and_quit_emits(self):
        devices = [PulseDevice("microphone.one", "Webcam Microphone", False)]
        with patch("meeting_translation.ui._pulse_sources", return_value=devices):
            window = OverlayWindow(load_config())
        window.microphone_selector.setCurrentIndex(window.microphone_selector.findData("microphone.one"))
        self.assertEqual(window.microphone_button.text(), "Mute")
        self.assertEqual(window.microphone_selector.currentText(), "Webcam Microphone")
        closed = []
        window.close_requested.connect(lambda: closed.append(True))
        window.quit_button.click()
        self.assertEqual(closed, [True])

    def test_session_compacts_audio_setup_but_keeps_meters_and_mute(self):
        devices = [
            PulseDevice("speaker.monitor", "Built-in Audio — Meeting output", True),
            PulseDevice("microphone.one", "Webcam Microphone", False),
        ]
        with patch("meeting_translation.ui._pulse_sources", return_value=devices):
            window = OverlayWindow(load_config())
        window.microphone_selector.setCurrentIndex(window.microphone_selector.findData("microphone.one"))
        window.set_session_active(True)
        self.assertTrue(window.desktop_selector.isHidden())
        self.assertTrue(window.microphone_selector.isHidden())
        self.assertFalse(window.audio_expanded)
        self.assertFalse(window.microphone_button.isHidden())
        window.set_audio_levels(0.1, 0.01)
        self.assertGreater(window.desktop_meter.value(), window.microphone_meter.value())

    def test_disconnected_sources_are_visible_and_safe(self):
        configured = load_config()
        with patch("meeting_translation.ui._pulse_sources", return_value=[]):
            window = OverlayWindow(configured)
        microphone_index = window.microphone_selector.findData(configured.audio.microphone_source)
        self.assertGreaterEqual(microphone_index, 0)
        window.microphone_selector.setCurrentIndex(microphone_index)
        self.assertIn("Disconnected", window.microphone_selector.currentText())
        self.assertFalse(window.microphone_button.isEnabled())
        self.assertEqual(window.selected_sources()[1], "")
    def test_appearance_presets_update_visual_state_and_persist(self):
        with patch("meeting_translation.ui._pulse_sources", return_value=[]):
            window = OverlayWindow(load_config())
        window.apply_layout_preset("High Contrast")
        self.assertEqual(window.layout_preset, "High Contrast")
        self.assertTrue(window.high_contrast)
        self.assertEqual(window.opacity, 0.98)
        self.assertEqual(window.settings.value("layout_preset"), "High Contrast")
        window.set_opacity(0.75)
        self.assertEqual(window.opacity, 0.75)
        self.assertEqual(float(window.settings.value("opacity")), 0.75)
        window.save_preferences()
        with patch("meeting_translation.ui._pulse_sources", return_value=[]):
            reopened = OverlayWindow(load_config())
        self.assertEqual(reopened.layout_preset, "High Contrast")
        self.assertTrue(reopened.high_contrast)
        self.assertEqual(reopened.opacity, 0.75)
        screen = window.current_screen()
        window.place_on_screen(screen, "top")
        top_y = window.y()
        window.place_on_screen(screen, "bottom")
        self.assertGreaterEqual(window.y(), top_y)
        self.assertGreaterEqual(window.quit_button.minimumHeight(), 32)

    def test_caption_lifecycle_is_visible_without_geometry_collapse(self):
        with patch("meeting_translation.ui._pulse_sources", return_value=[]):
            window = OverlayWindow(load_config())
        self.assertFalse(window.previous.isHidden())
        self.assertGreater(window.previous.minimumHeight(), 0)
        window.update_event({
            "segment_id": "one", "revision": 1, "state": "provisional",
            "english": "", "mandarin": "測試", "corrections": [],
        })
        self.assertEqual(window.current.state_badge.text(), "LIVE")
        self.assertEqual(window.previous.state_badge.text(), "")
        window.update_event({
            "segment_id": "one", "revision": 2, "state": "committed",
            "english": "Test", "mandarin": "測試", "corrections": [{"before": "a", "after": "b"}],
        })
        self.assertEqual(window.current.state_badge.text(), "CORRECTED")

    def test_status_chip_and_muted_button_are_visually_explicit(self):
        devices = [PulseDevice("microphone.one", "Webcam Microphone", False)]
        with patch("meeting_translation.ui._pulse_sources", return_value=devices):
            window = OverlayWindow(load_config())
        window.microphone_selector.setCurrentIndex(window.microphone_selector.findData("microphone.one"))
        window.microphone_button.setText("Unmute")
        window._sync_audio_state()
        window.set_health({"state": "paused", "microphone_enabled": False})
        self.assertTrue(window.microphone_button.property("muted"))
        self.assertIn("PAUSED", window.status.text())
        self.assertIn("MIC OFF", window.status.text())

    def test_compact_policy_reduced_motion_and_accessibility_persist(self):
        with patch("meeting_translation.ui._pulse_sources", return_value=[]):
            window = OverlayWindow(load_config())
        window.set_compact_policy("never")
        window.set_reduced_motion(True)
        window.set_session_active(True)
        self.assertTrue(window.audio_expanded)
        self.assertEqual(window.desktop_selector.accessibleName(), "Meeting audio source")
        self.assertEqual(window.status.accessibleName(), "Caption service status")
        window.save_preferences()
        with patch("meeting_translation.ui._pulse_sources", return_value=[]):
            reopened = OverlayWindow(load_config())
        self.assertEqual(reopened.compact_policy, "never")
        self.assertTrue(reopened.reduced_motion)
        reopened.set_compact_policy("immediate")

    def test_history_filters_copies_and_explains_corrections(self):
        history = HistoryDialog()
        history.update_event({
            "segment_id": "one", "revision": 1, "state": "committed",
            "audio_start_ms": 62000, "english": "Purchase order", "mandarin": "採購單",
            "asr_model": "Qwen", "corrections": [
                {"before": "purchase older", "after": "purchase order", "reason": "verified_glossary"}
            ],
        })
        history.state_filter.setCurrentText("Corrected")
        history.show_models.setChecked(True)
        self.assertIn("Original:", history.browser.toPlainText())
        self.assertIn("verified glossary", history.browser.toPlainText())
        self.assertEqual(history.navigator.currentText(), "01:02")
        history.copy_text("english")
        self.assertIn("Purchase order", QApplication.clipboard().text())

    def test_preflight_blocks_start_for_fatal_readiness_failure(self):
        dialog = PreflightDialog(
            [("Caption service", "Disconnected", False)],
            can_start=False,
        )
        button_box = dialog.findChild(QDialogButtonBox)
        self.assertIsNotNone(button_box)
        self.assertFalse(button_box.button(QDialogButtonBox.Ok).isEnabled())

    def test_active_quit_cancel_preserves_session_and_confirm_waits_for_stop(self):
        controller = SimpleNamespace(
            session_active=True,
            overlay=SimpleNamespace(status=QLabel()),
            commands=MagicMock(),
            pending_quit=False,
            quit=MagicMock(),
        )
        with patch("meeting_translation.ui.QMessageBox.question", return_value=QMessageBox.Cancel):
            DesktopController.request_quit(controller)
        controller.commands.send.assert_not_called()
        controller.quit.assert_not_called()
        self.assertFalse(controller.pending_quit)

        with patch("meeting_translation.ui.QMessageBox.question", return_value=QMessageBox.Yes):
            DesktopController.request_quit(controller)
        controller.commands.send.assert_called_once_with("stop")
        controller.quit.assert_not_called()
        self.assertTrue(controller.pending_quit)
        self.assertEqual(controller.overlay.status.text(), "STOPPING")



if __name__ == "__main__":
    unittest.main()
