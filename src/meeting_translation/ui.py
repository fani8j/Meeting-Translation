from __future__ import annotations

import html
import json
import math
import os
import socket
import subprocess
import sys
import time
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QEvent, QObject, QPoint, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QDesktopServices, QFont, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QDialog,
    QFrame,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizeGrip,
    QSystemTrayIcon,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig


class SocketReader(QThread):
    message = Signal(dict)
    connection = Signal(bool, str)

    def __init__(self, socket_path: Path) -> None:
        super().__init__()
        self.socket_path = socket_path
        self._running = True

    def run(self) -> None:
        while self._running:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                client.connect(str(self.socket_path))
                client.settimeout(0.5)
                self.connection.emit(True, "Caption service connected")
                buffer = b""
                while self._running:
                    try:
                        chunk = client.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line:
                            self.message.emit(json.loads(line))
            except (FileNotFoundError, ConnectionRefusedError, OSError) as error:
                self.connection.emit(False, str(error))
                time.sleep(0.5)
            finally:
                client.close()

    def stop(self) -> None:
        self._running = False
        self.wait(1500)


class CommandClient:
    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path

    def send(self, command: str, **values: Any) -> bool:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(1)
            client.connect(str(self.socket_path))
            client.sendall(json.dumps({"command": command, **values}).encode("utf-8") + b"\n")
            return True
        except OSError:
            return False
        finally:
            client.close()

@dataclass(frozen=True)
class PulseDevice:
    name: str
    description: str
    is_monitor: bool


class AudioMeter(QProgressBar):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(True)
        self.setFormat("LEVEL")
        self.setFixedSize(78, 12)
        self.setToolTip(f"Live {label} input level")
        self.setStyleSheet(
            "QProgressBar { color: #F5F7FA; background: #4A5568; border: 1px solid #C0CAD8; "
            "border-radius: 5px; text-align: center; font-size: 7px; font-weight: 800; }"
            "QProgressBar::chunk { background: #3FAF8F; border-radius: 4px; }"
        )

    def set_level(self, rms: float) -> None:
        if rms <= 0:
            value = 0
        else:
            value = round(max(0.0, min(1.0, (20 * math.log10(rms) + 60) / 60)) * 100)
        self.setValue(value)


class RefreshingComboBox(QComboBox):
    refresh_requested = Signal()

    def showPopup(self) -> None:
        self.refresh_requested.emit()
        super().showPopup()



class CaptionBlock(QFrame):
    def __init__(self, english_px: int, mandarin_px: int) -> None:
        super().__init__()
        self.english_px = english_px
        self.mandarin_px = mandarin_px
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 6, 20, 6)
        layout.setSpacing(3)
        self.state_badge = QLabel()
        self.state_badge.setAlignment(Qt.AlignCenter)
        self.state_badge.setFixedWidth(92)
        self.english = QLabel()
        self.english.setWordWrap(True)
        self.english.setAlignment(Qt.AlignCenter)
        self.mandarin = QLabel()
        self.mandarin.setWordWrap(True)
        self.mandarin.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.state_badge, 0, Qt.AlignHCenter)
        layout.addWidget(self.english)
        layout.addWidget(self.mandarin)
        self.setMinimumHeight(max(82, english_px * 2 + mandarin_px + 22))

    def set_scale(self, scale: float) -> None:
        self.english.setFont(QFont("Noto Sans", max(12, round(self.english_px * scale)), QFont.DemiBold))
        self.mandarin.setFont(QFont("Noto Sans CJK TC", max(9, round(self.mandarin_px * scale))))

    def set_caption(self, english: str, mandarin: str, state: str, corrected: bool) -> None:
        self.english.setText(english or "Translation pending")
        self.mandarin.setText(mandarin)
        provisional = state == "provisional"
        badge = "CORRECTED" if corrected else ("LIVE" if provisional else "FINAL")
        badge_color = "#FFE082" if corrected else ("#8FA6C4" if provisional else "#7FD1B9")
        english_color = "#B8C0CC" if provisional else "#FFFFFF"
        mandarin_color = "#8994A4" if provisional else "#D3DAE6"
        if corrected:
            english_color = "#FFE082"
            mandarin_color = "#FFE082"
        self.state_badge.setText(badge)
        self.state_badge.setStyleSheet(
            f"color: {badge_color}; background: rgba(255,255,255,18); border: 1px solid {badge_color}; "
            "border-radius: 8px; padding: 2px 7px; font-size: 9px; font-weight: 800; letter-spacing: 2px;"
        )
        self.english.setStyleSheet(f"color: {english_color}; background: transparent;")
        self.mandarin.setStyleSheet(f"color: {mandarin_color}; background: transparent;")

    def clear(self) -> None:
        self.state_badge.clear()
        self.english.clear()
        self.mandarin.clear()


class OverlayWindow(QWidget):
    toggle_session_requested = Signal()
    pause_requested = Signal()
    history_requested = Signal()
    microphone_requested = Signal()
    refresh_requested = Signal()
    reconnect_microphone_requested = Signal()
    retry_requested = Signal()
    hide_requested = Signal()
    lock_changed = Signal(bool)
    close_requested = Signal()

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.settings = QSettings("MeetingTranslation", "LiveCaptionOverlay")
        self._locked = config.overlay.locked
        self._drag_origin: QPoint | None = None
        self.font_scale = float(self.settings.value("font_scale", 1.0))
        self.font_scale = min(1.5, max(0.55, self.font_scale))
        self.opacity = min(0.98, max(0.60, float(self.settings.value("opacity", config.overlay.opacity))))
        self.high_contrast = str(self.settings.value("high_contrast", "false")).lower() == "true"
        self.layout_preset = str(self.settings.value("layout_preset", "Standard"))
        self.compact_policy = str(self.settings.value("compact_policy", "immediate"))
        self.reduced_motion = str(self.settings.value("reduced_motion", "true")).lower() == "true"
        self._compact_timer = QTimer(self)
        self._compact_timer.setSingleShot(True)
        self._compact_timer.timeout.connect(lambda: self.set_audio_expanded(False) if self.session_active else None)
        self.captions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.setMinimumSize(760, 340)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)

        self.panel = QFrame(self)
        self.panel.setObjectName("captionPanel")
        self.panel.setStyleSheet(self._panel_stylesheet())
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.addWidget(self.panel)
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(4)

        self.controls = QFrame()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(4, 2, 4, 4)
        controls_layout.setSpacing(5)
        button_row = QHBoxLayout()
        button_row.setSpacing(5)
        self.drag_handle = QLabel("Drag")
        self.drag_handle.setCursor(QCursor(Qt.SizeAllCursor))
        self.drag_handle.setObjectName("secondaryLabel")
        self.drag_handle.installEventFilter(self)
        button_row.addWidget(self.drag_handle)
        self.session_button = self._button("Start", self.toggle_session_requested.emit)
        self.pause_button = self._button("Pause", self.pause_requested.emit)
        self.history_button = self._button("History", self.history_requested.emit)
        self.quit_button = self._button("Quit", self.close_requested.emit)
        smaller = self._button("A−", lambda: self.adjust_font(-0.1))
        self.appearance_button = self._button("Appearance", self.show_appearance_menu)
        larger = self._button("A+", lambda: self.adjust_font(0.1))
        self.hide_button = self._button("Hide overlay", self.hide_requested.emit)
        lock = self._button("Lock", self._lock_from_overlay)
        self.audio_button = self._button("Audio ▴", lambda: self.set_audio_expanded(not self.audio_expanded))
        self.audio_expanded = True
        self.session_active = False
        for button in (
            self.session_button,
            self.pause_button,
            self.history_button,
            self.audio_button,
            self.appearance_button,
        ):
            button_row.addWidget(button)
        self.appearance_button.setToolTip("Layout, opacity, and display placement")
        button_row.addStretch(1)
        controls_layout.addLayout(button_row)
        utility_row = QHBoxLayout()
        utility_row.setSpacing(5)
        utility_row.addStretch(1)
        for button in (smaller, larger, self.hide_button, lock):
            utility_row.addWidget(button)
        utility_row.addSpacing(12)
        utility_row.addWidget(self.quit_button)
        controls_layout.addLayout(utility_row)

        source_row = QHBoxLayout()
        self.desktop_label = QLabel("Meeting audio")
        self.desktop_label.setObjectName("secondaryLabel")
        self.desktop_selector = RefreshingComboBox()
        self.desktop_meter = AudioMeter("meeting audio")
        self.microphone_label = QLabel("Microphone")
        self.microphone_label.setObjectName("secondaryLabel")
        self.microphone_selector = RefreshingComboBox()
        self.microphone_meter = AudioMeter("microphone")
        self.microphone_button = self._button("Mute", self.microphone_requested.emit)
        source_row.addWidget(self.desktop_label)
        source_row.addWidget(self.desktop_selector, 3)
        source_row.addWidget(self.desktop_meter)
        source_row.addWidget(self.microphone_label)
        source_row.addWidget(self.microphone_selector, 2)
        source_row.addWidget(self.microphone_meter)
        source_row.addWidget(self.microphone_button)
        controls_layout.addLayout(source_row)
        self.audio_warning = QLabel()
        self.audio_warning.setStyleSheet("color: #FFB74D; font-size: 11px; font-weight: 600;")
        self.audio_warning.setVisible(False)
        recovery_row = QHBoxLayout()
        self.refresh_button = self._button("Refresh devices", self.refresh_requested.emit)
        self.reconnect_button = self._button("Reconnect microphone", self.reconnect_microphone_requested.emit)
        self.retry_button = self._button("Retry service", self.retry_requested.emit)
        for button in (self.refresh_button, self.reconnect_button, self.retry_button):
            recovery_row.addWidget(button)
        recovery_row.addStretch(1)
        controls_layout.addLayout(recovery_row)
        self.recovery_buttons = (self.refresh_button, self.reconnect_button, self.retry_button)
        for button in self.recovery_buttons:
            button.setVisible(False)
        controls_layout.addWidget(self.audio_warning)
        self.locked_hint = QLabel("Overlay locked — use the tray menu to unlock")
        self.locked_hint.setAlignment(Qt.AlignCenter)
        self.locked_hint.setStyleSheet(
            "color:#FFE082;background:rgba(0,0,0,170);border:1px solid #FFE082;"
            "border-radius:8px;padding:4px;font-weight:700;"
        )
        self.locked_hint.setVisible(False)
        panel_layout.addWidget(self.locked_hint)
        panel_layout.addWidget(self.controls)
        self._populate_audio_sources()
        self.desktop_selector.refresh_requested.connect(self._populate_audio_sources)
        self.microphone_selector.refresh_requested.connect(self._populate_audio_sources)
        self.desktop_selector.currentIndexChanged.connect(self._sync_audio_state)
        self.microphone_selector.currentIndexChanged.connect(self._sync_audio_state)
        self.desktop_selector.setToolTip("Device list refreshes whenever this menu opens")
        self.microphone_selector.setToolTip("Device list refreshes whenever this menu opens")
        self.microphone_button.setToolTip("Mute or unmute the selected microphone")
        self.hide_button.setToolTip("Hide the overlay; transcription continues")
        self.quit_button.setObjectName("quitButton")
        self.microphone_button.setObjectName("microphoneButton")
        self.quit_button.setToolTip("End transcription and quit the application")
        self.panel.setStyleSheet(self._panel_stylesheet())

        self.status = QLabel("READY")
        self.status.setObjectName("statusChip")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setMinimumHeight(24)
        panel_layout.addWidget(self.status, 0, Qt.AlignHCenter)
        self.previous = CaptionBlock(config.overlay.english_px - 6, config.overlay.mandarin_px - 3)
        self.previous.clear()
        self.current = CaptionBlock(config.overlay.english_px, config.overlay.mandarin_px)
        panel_layout.addWidget(self.previous, 1)
        panel_layout.addWidget(self.current, 2)
        self.current.set_caption("Waiting for speech…", "等待語音…", "committed", False)
        self.adjust_font(0)
        self.set_health({"state": "idle", "microphone_enabled": True})

        grip_row = QHBoxLayout()
        grip_row.addStretch(1)
        self.size_grip = QSizeGrip(self)
        self.size_grip.setToolTip("Drag to resize subtitle panel")
        grip_row.addWidget(self.size_grip)
        panel_layout.addLayout(grip_row)
        self._configure_accessibility()
        self._install_focused_shortcuts()

    def _configure_accessibility(self) -> None:
        controls = {
            self.session_button: "Start or stop caption session",
            self.pause_button: "Pause or resume captions",
            self.history_button: "Open caption history",
            self.audio_button: "Show or hide audio controls",
            self.appearance_button: "Open appearance settings",
            self.desktop_selector: "Meeting audio source",
            self.microphone_selector: "Microphone source",
            self.microphone_button: "Mute or unmute microphone",
            self.status: "Caption service status",
            self.current.english: "Current English caption",
            self.current.mandarin: "Current Mandarin caption",
            self.previous.english: "Previous English caption",
            self.previous.mandarin: "Previous Mandarin caption",
            self.desktop_meter: "Meeting audio signal level",
            self.microphone_meter: "Microphone signal level",
        }
        for widget, name in controls.items():
            widget.setAccessibleName(name)
        self.status.setAccessibleDescription("Current lifecycle state and caption delay")
        QWidget.setTabOrder(self.session_button, self.pause_button)
        QWidget.setTabOrder(self.pause_button, self.history_button)
        QWidget.setTabOrder(self.history_button, self.audio_button)
        QWidget.setTabOrder(self.audio_button, self.appearance_button)
        QWidget.setTabOrder(self.appearance_button, self.desktop_selector)
        QWidget.setTabOrder(self.desktop_selector, self.microphone_selector)
        QWidget.setTabOrder(self.microphone_selector, self.microphone_button)

    def _install_focused_shortcuts(self) -> None:
        shortcuts = (
            ("Space", self.pause_requested.emit, "Pause or resume captions"),
            ("M", self.microphone_requested.emit, "Mute or unmute microphone"),
            ("H", self.history_requested.emit, "Open caption history"),
            ("Escape", lambda: self.set_audio_expanded(False), "Collapse audio controls"),
        )
        self._focused_shortcuts: list[QShortcut] = []
        for sequence, callback, description in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.setWhatsThis(description)
            shortcut.activated.connect(callback)
            self._focused_shortcuts.append(shortcut)

        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
            self.resize(max(760, self.width()), max(self.sizeHint().height(), self.height()))
        else:
            self.reposition()

    def _button(self, text: str, callback: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)
        button.setFocusPolicy(Qt.StrongFocus)
        button.setMinimumHeight(32)
        button.clicked.connect(callback)
        return button
    def _panel_stylesheet(self) -> str:
        panel_rgb = "0, 0, 0" if self.high_contrast else "14, 18, 26"
        border = "rgba(255,255,255,110)" if self.high_contrast else "rgba(255,255,255,30)"
        control = "#111111" if self.high_contrast else "#283141"
        control_border = "#FFFFFF" if self.high_contrast else "#465268"
        popup = "#080808" if self.high_contrast else "#202938"
        quit_background = "#8B1E1E" if self.high_contrast else "#4A2528"
        quit_color = "#FFFFFF" if self.high_contrast else "#FFD8D5"
        secondary = "#FFFFFF" if self.high_contrast else "#AEB8C7"
        return (
            f"QFrame#captionPanel {{ background: rgba({panel_rgb}, {round(self.opacity * 255)}); "
            f"border: 1px solid {border}; border-radius: 16px; }}"
            f"QLabel#secondaryLabel {{ color: {secondary}; font-size: 11px; font-weight: 700; padding: 5px; }}"
            f"QPushButton {{ color: #F5F7FA; background: {control}; border: 1px solid {control_border}; "
            "border-radius: 7px; padding: 5px 10px; font-weight: 650; }"
            "QPushButton:hover { background: #3D4D66; }"
            "QPushButton:pressed { background: #1C2431; }"
            "QPushButton:disabled { color: #747E8E; background: #1A202B; border-color: #30394A; }"
            f"QPushButton#quitButton {{ color: {quit_color}; background: {quit_background}; border-color: #FF8A80; }}"
            "QPushButton#quitButton:hover { background: #A52A2A; }"
            "QPushButton#microphoneButton[muted=\"true\"] { color: #10151D; background: #FFB74D; border-color: #FFD180; }"
            f"QComboBox {{ color: #F5F7FA; background: {popup}; border: 1px solid {control_border}; "
            "border-radius: 6px; padding: 5px 9px; min-height: 20px; }"
            f"QComboBox QAbstractItemView {{ color: #F5F7FA; background: {popup}; "
            "selection-color: #FFFFFF; selection-background-color: #3D5A80; "
            f"border: 1px solid {control_border}; outline: none; }}"
        )

    def show_appearance_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { color: #F5F7FA; background: #202938; border: 1px solid #465268; padding: 6px; }"
            "QMenu::item { padding: 7px 22px; border-radius: 4px; }"
            "QMenu::item:selected { background: #3D5A80; }"
            "QMenu::separator { height: 1px; background: #465268; margin: 5px 8px; }"
        )
        presets = menu.addMenu("Layout preset")
        for name in ("Compact", "Standard", "Presentation", "High Contrast"):
            action = presets.addAction(name)
            action.setCheckable(True)
            action.setChecked(self.layout_preset == name)
            action.triggered.connect(lambda checked=False, preset=name: self.apply_layout_preset(preset))
        opacity = menu.addMenu("Background opacity")
        for value in (60, 75, 86, 95):
            action = opacity.addAction(f"{value}%")
            action.setCheckable(True)
            action.setChecked(round(self.opacity * 100) == value)
            action.triggered.connect(lambda checked=False, percent=value: self.set_opacity(percent / 100))
        displays = menu.addMenu("Move to display")
        for index, screen in enumerate(QApplication.screens(), start=1):
            action = displays.addAction(f"Display {index}: {screen.name()}")
            action.triggered.connect(lambda checked=False, target=screen: self.place_on_screen(target, "bottom"))
        menu.addSeparator()
        menu.addAction("Place at top", lambda: self.place_on_screen(self.current_screen(), "top"))
        compact = menu.addMenu("Compact after Start")
        policies = (
            ("Immediately", "immediate"),
            ("After 5 seconds", "after_5"),
            ("Never", "never"),
        )
        for label, policy in policies:
            action = compact.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.compact_policy == policy)
            action.triggered.connect(lambda checked=False, value=policy: self.set_compact_policy(value))
        reduced_motion = menu.addAction("Reduce motion")
        reduced_motion.setCheckable(True)
        reduced_motion.setChecked(self.reduced_motion)
        reduced_motion.triggered.connect(self.set_reduced_motion)
        menu.addAction("Place at bottom", lambda: self.place_on_screen(self.current_screen(), "bottom"))
        menu.exec(QCursor.pos())
    def set_compact_policy(self, policy: str) -> None:
        if policy not in {"immediate", "after_5", "never"}:
            return
        self.compact_policy = policy
        self.settings.setValue("compact_policy", policy)

    def set_reduced_motion(self, enabled: bool) -> None:
        self.reduced_motion = enabled
        self.settings.setValue("reduced_motion", enabled)


    def set_opacity(self, opacity: float) -> None:
        self.opacity = min(0.98, max(0.60, opacity))
        self.panel.setStyleSheet(self._panel_stylesheet())
        self.settings.setValue("opacity", self.opacity)

    def apply_layout_preset(self, preset: str) -> None:
        specifications = {
            "Compact": (0.82, 0.78, False, 800, 360),
            "Standard": (1.00, 0.86, False, 980, 420),
            "Presentation": (1.25, 0.92, False, 1280, 520),
            "High Contrast": (1.10, 0.98, True, 1100, 460),
        }
        if preset not in specifications:
            return
        scale, opacity, high_contrast, width, height = specifications[preset]
        self.layout_preset = preset
        self.high_contrast = high_contrast
        self.font_scale = scale
        self.previous.set_scale(scale)
        self.current.set_scale(scale)
        self.opacity = opacity
        self.panel.setStyleSheet(self._panel_stylesheet())
        available = self.current_screen().availableGeometry()
        target_width = min(width, available.width() - 40)
        target_height = max(self.sizeHint().height(), min(height, available.height() - 40))
        self.resize(target_width, target_height)
        self.place_on_screen(self.current_screen(), "bottom")
        self.settings.setValue("layout_preset", preset)
        self.settings.setValue("high_contrast", high_contrast)
        self.settings.setValue("font_scale", scale)
        self.settings.setValue("opacity", opacity)
        self.save_preferences()
    def current_screen(self):
        return QApplication.screenAt(self.frameGeometry().center()) or QApplication.primaryScreen()

    def place_on_screen(self, screen, edge: str) -> None:
        if screen is None:
            return
        available = screen.availableGeometry()
        x = available.x() + (available.width() - self.width()) // 2
        if edge == "top":
            y = available.y() + 20
        else:
            y = available.y() + available.height() - self.height() - self.config.overlay.bottom_margin
        self.move(x, max(available.y() + 20, y))
        self.save_preferences()

    def _populate_audio_sources(self) -> None:
        selected_desktop = self.desktop_selector.currentData()
        selected_microphone = self.microphone_selector.currentData()
        if selected_desktop is None:
            selected_desktop = self.config.audio.desktop_source
        if selected_microphone is None and self.config.audio.include_microphone:
            selected_microphone = self.config.audio.microphone_source
        sources = _pulse_sources()
        monitors = [source for source in sources if source.is_monitor]
        microphones = [source for source in sources if not source.is_monitor]

        self.desktop_selector.blockSignals(True)
        self.desktop_selector.clear()
        self._add_source_item(
            self.desktop_selector,
            PulseDevice("@DEFAULT_MONITOR@", "Default output", True),
            connected=True,
        )
        for source in monitors:
            self._add_source_item(self.desktop_selector, source, connected=True)
        desktop_index = self.desktop_selector.findData(selected_desktop)
        if selected_desktop and desktop_index < 0 and selected_desktop != "@DEFAULT_MONITOR@":
            self._add_source_item(
                self.desktop_selector,
                PulseDevice(str(selected_desktop), _friendly_source(str(selected_desktop)), True),
                connected=False,
            )
            desktop_index = self.desktop_selector.count() - 1
        self.desktop_selector.setCurrentIndex(max(0, desktop_index))
        self.desktop_selector.blockSignals(False)

        self.microphone_selector.blockSignals(True)
        self.microphone_selector.clear()
        self.microphone_selector.addItem("No microphone", "")
        self.microphone_selector.setItemData(0, True, Qt.UserRole + 1)
        for source in microphones:
            self._add_source_item(self.microphone_selector, source, connected=True)
        microphone_index = self.microphone_selector.findData(selected_microphone)
        if selected_microphone and microphone_index < 0:
            self._add_source_item(
                self.microphone_selector,
                PulseDevice(str(selected_microphone), _friendly_source(str(selected_microphone)), False),
                connected=False,
            )
            microphone_index = self.microphone_selector.count() - 1
        self.microphone_selector.setCurrentIndex(max(0, microphone_index))
        self.microphone_selector.blockSignals(False)
        self._sync_audio_state()

    @staticmethod
    def _add_source_item(selector: QComboBox, source: PulseDevice, connected: bool) -> None:
        label = source.description if connected else f"{source.description} — Disconnected"
        selector.addItem(label, source.name)
        index = selector.count() - 1
        selector.setItemData(index, source.name, Qt.ToolTipRole)
        selector.setItemData(index, connected, Qt.UserRole + 1)

    @staticmethod
    def _selection_connected(selector: QComboBox) -> bool:
        return bool(selector.currentData(Qt.UserRole + 1))

    def _sync_audio_state(self) -> None:
        desktop_connected = self._selection_connected(self.desktop_selector)
        microphone_selected = bool(self.microphone_selector.currentData())
        microphone_connected = self._selection_connected(self.microphone_selector)
        self.session_button.setEnabled(self.session_active or desktop_connected)
        self.microphone_button.setEnabled(microphone_selected and microphone_connected)
        if not microphone_selected:
            self.microphone_button.setText("No mic")
        elif not microphone_connected:
            self.microphone_button.setText("Unavailable")
        elif self.microphone_button.text() not in {"Mute", "Unmute"}:
            self.microphone_button.setText("Mute")
        self.microphone_button.setProperty("muted", self.microphone_button.text() == "Unmute")
        self.microphone_button.style().unpolish(self.microphone_button)
        self.microphone_button.style().polish(self.microphone_button)
        warnings = []
        if not desktop_connected:
            warnings.append("Meeting audio disconnected — select an available output before starting.")
        if microphone_selected and not microphone_connected:
            warnings.append("Microphone disconnected — session will continue without it.")
        self.audio_warning.setText("  ".join(warnings))
        self.audio_warning.setVisible(self.audio_expanded and bool(warnings))
        self.refresh_button.setVisible(self.audio_expanded and bool(warnings))
        self.reconnect_button.setVisible(
            self.audio_expanded and microphone_selected and not microphone_connected and self.session_active
        )

    def selected_sources(self) -> tuple[str, str]:
        microphone = str(self.microphone_selector.currentData() or "")
        if not self._selection_connected(self.microphone_selector):
            microphone = ""
        return str(self.desktop_selector.currentData() or "@DEFAULT_MONITOR@"), microphone

    def set_audio_expanded(self, expanded: bool) -> None:
        self.audio_expanded = expanded
        self.desktop_selector.setVisible(expanded)
        self.microphone_selector.setVisible(expanded)
        self.audio_warning.setVisible(expanded and bool(self.audio_warning.text()))
        self.refresh_button.setVisible(expanded and bool(self.audio_warning.text()))
        self.reconnect_button.setVisible(
            expanded
            and self.session_active
            and bool(self.microphone_selector.currentData())
            and not self._selection_connected(self.microphone_selector)
        )
        self.retry_button.setVisible(expanded and "service" in self.audio_warning.text().casefold())
        self.audio_button.setText("Audio ▴" if expanded else "Audio ▾")

    def set_audio_levels(self, desktop_level: float, microphone_level: float) -> None:
        self.desktop_meter.set_level(desktop_level)
        self.microphone_meter.set_level(microphone_level)
    def set_microphone_available(self, available: bool) -> None:
        if not self.microphone_selector.currentData():
            return
        self.microphone_selector.setItemData(
            self.microphone_selector.currentIndex(), available, Qt.UserRole + 1
        )
        self._sync_audio_state()


    def set_session_active(self, active: bool) -> None:
        self.session_active = active
        self.session_button.setText("Stop" if active else "Start")
        self.desktop_selector.setEnabled(not active)
        self.microphone_selector.setEnabled(not active)
        self._compact_timer.stop()
        if not active:
            self.set_audio_levels(0, 0)
            self._populate_audio_sources()
            self.set_audio_expanded(True)
        elif self.compact_policy == "immediate":
            self.set_audio_expanded(False)
        elif self.compact_policy == "after_5":
            self.set_audio_expanded(True)
            self._compact_timer.start(5000)
        else:
            self.set_audio_expanded(True)
        self._sync_audio_state()

    def adjust_font(self, delta: float) -> None:
        self.font_scale = min(1.5, max(0.55, self.font_scale + delta))
        self.previous.set_scale(self.font_scale)
        self.current.set_scale(self.font_scale)
        self.settings.setValue("font_scale", self.font_scale)

    def reposition(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = min(max(760, round(available.width() * self.config.overlay.width_ratio)), available.width() - 40)
        self.resize(width, max(300, self.sizeHint().height()))
        self.place_on_screen(screen, "bottom")

    def save_preferences(self) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("font_scale", self.font_scale)
        self.settings.setValue("opacity", self.opacity)
        self.settings.setValue("compact_policy", self.compact_policy)
        self.settings.setValue("reduced_motion", self.reduced_motion)
        self.settings.setValue("high_contrast", self.high_contrast)
        self.settings.setValue("layout_preset", self.layout_preset)

    def _lock_from_overlay(self) -> None:
        self.apply_lock(True)
        self.lock_changed.emit(True)

    def apply_lock(self, locked: bool) -> None:
        self._locked = locked
        self.locked_hint.setVisible(locked)
        self.controls.setVisible(not locked)
        self.size_grip.setVisible(not locked)
        self.setWindowFlag(Qt.WindowTransparentForInput, locked)
        if self.isVisible():
            self.show()
        self.save_preferences()

    def update_event(self, payload: dict[str, Any]) -> None:
        segment_id = payload["segment_id"]
        existing = self.captions.get(segment_id)
        if existing is None or int(payload["revision"]) >= int(existing["revision"]):
            self.captions[segment_id] = payload
            self.captions.move_to_end(segment_id)
        while len(self.captions) > 100:
            self.captions.popitem(last=False)
        visible = list(self.captions.values())[-2:]
        current = visible[-1]
        self.current.set_caption(
            current.get("english", ""), current.get("mandarin", ""), current["state"], bool(current.get("corrections"))
        )
        if len(visible) == 2:
            previous = visible[0]
            self.previous.set_caption(
                previous.get("english", ""), previous.get("mandarin", ""), previous["state"], False
            )
        else:
            self.previous.clear()

    def set_health(self, payload: dict[str, Any]) -> None:
        state = payload.get("state", "idle")
        label = {
            "idle": "READY",
            "preparing": "LOADING",
            "listening": "LISTENING",
            "translating": "TRANSLATING",
            "paused": "PAUSED",
            "degraded": "CATCHING UP",
            "error": "ATTENTION",
            "stopping": "STOPPING",
        }.get(state, state.upper())
        if not payload.get("microphone_enabled", True):
            label += "  ·  MIC OFF"
        delay = int(payload.get("current_delay_ms", 0))
        self.status.setText(f"{label}  ·  {delay / 1000:.1f}s" if delay else label)
        colors = {
            "idle": ("#7FD1B9", "rgba(127,209,185,28)"),
            "preparing": ("#90CAF9", "rgba(144,202,249,28)"),
            "listening": ("#7FD1B9", "rgba(127,209,185,28)"),
            "translating": ("#CE93D8", "rgba(206,147,216,30)"),
            "paused": ("#FFB74D", "rgba(255,183,77,32)"),
            "degraded": ("#FFB74D", "rgba(255,183,77,32)"),
            "error": ("#FF8A80", "rgba(255,138,128,35)"),
            "stopping": ("#B0BEC5", "rgba(176,190,197,26)"),
        }
        foreground, background = colors.get(state, ("#D3DAE6", "rgba(211,218,230,25)"))
        self.status.setStyleSheet(
            f"color: {foreground}; background: {background}; border: 1px solid {foreground}; "
            "border-radius: 10px; padding: 3px 10px; font-size: 10px; font-weight: 800; letter-spacing: 2px;"
        )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.drag_handle and not self._locked:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
            if event.type() == QEvent.MouseMove and self._drag_origin is not None:
                self.move(event.globalPosition().toPoint() - self._drag_origin)
                return True
            if event.type() == QEvent.MouseButtonRelease:
                self._drag_origin = None
                self.save_preferences()
                return True
        return super().eventFilter(watched, event)


class HistoryDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Meeting Translation History")
        self.resize(820, 720)
        self.items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search Mandarin, English, models, or technical terms…")
        self.search.setAccessibleName("Search caption history")
        self.search.textChanged.connect(self.render)
        self.state_filter = QComboBox()
        self.state_filter.addItems(["All", "Live", "Final", "Corrected"])
        self.state_filter.setAccessibleName("Caption state filter")
        self.state_filter.currentIndexChanged.connect(self.render)
        self.navigator = QComboBox()
        self.navigator.setAccessibleName("Jump to caption timestamp")
        self.navigator.currentIndexChanged.connect(self._jump_to_selected)
        self.show_models = QCheckBox("Show model")
        self.show_models.toggled.connect(self.render)
        controls.addWidget(self.search, 3)
        controls.addWidget(self.state_filter)
        controls.addWidget(self.navigator)
        controls.addWidget(self.show_models)
        layout.addLayout(controls)
        actions = QHBoxLayout()
        for label, callback in (
            ("Copy all", lambda: self.copy_text("all")),
            ("Copy Mandarin", lambda: self.copy_text("mandarin")),
            ("Copy English", lambda: self.copy_text("english")),
            ("Export session", self.export_session),
        ):
            button = QPushButton(label)
            button.setMinimumHeight(32)
            button.clicked.connect(callback)
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.browser = QTextBrowser()
        self.browser.setAccessibleName("Caption history results")
        layout.addWidget(self.browser)

    def update_event(self, payload: dict[str, Any]) -> None:
        current = self.items.get(payload["segment_id"])
        if current is None or int(payload["revision"]) >= int(current["revision"]):
            self.items[payload["segment_id"]] = payload
        self.render()

    def _visible_items(self) -> list[dict[str, Any]]:
        query = self.search.text().casefold().strip()
        selected_state = self.state_filter.currentText().casefold()
        visible: list[dict[str, Any]] = []
        for payload in self.items.values():
            searchable = " ".join([
                payload.get("mandarin", ""), payload.get("english", ""), payload.get("asr_model", "")
            ])
            if query and query not in searchable.casefold():
                continue
            state = payload.get("state", "")
            corrected = bool(payload.get("corrections"))
            if selected_state == "live" and state != "provisional":
                continue
            if selected_state == "final" and state == "provisional":
                continue
            if selected_state == "corrected" and not corrected:
                continue
            visible.append(payload)
        return visible

    def render(self) -> None:
        visible = self._visible_items()
        sections: list[str] = []
        selected_segment = self.navigator.currentData()
        self.navigator.blockSignals(True)
        self.navigator.clear()
        for payload in visible:
            segment_id = str(payload["segment_id"])
            timestamp = _format_ms(int(payload.get("audio_start_ms", 0)))
            self.navigator.addItem(timestamp, segment_id)
            corrections = "".join(
                "<div style='color:#FFE082;margin-top:5px'>"
                f"Original: {html.escape(str(item.get('before', '')))} → "
                f"Corrected: {html.escape(str(item.get('after', '')))} · "
                f"Reason: {html.escape(str(item.get('reason', '')).replace('_', ' '))}</div>"
                for item in payload.get("corrections", [])
            )
            model = (
                f"<div style='color:#8994A4;font-size:10px'>Model: "
                f"{html.escape(payload.get('asr_model', ''))}</div>"
                if self.show_models.isChecked() else ""
            )
            sections.append(
                f"<div id='{html.escape(segment_id)}' style='margin:12px 4px;padding:12px;"
                "background:#202632;border-radius:8px'>"
                f"<div style='color:#7FD1B9;font-size:11px'>{timestamp} · "
                f"{html.escape(payload.get('state',''))}</div>"
                f"<div style='color:white;font-size:18px;margin-top:6px'>"
                f"{html.escape(payload.get('english','') or 'Translating…')}</div>"
                f"<div style='color:#C8D0DC;font-size:14px;margin-top:4px'>"
                f"{html.escape(payload.get('mandarin',''))}</div>{corrections}{model}</div>"
            )
        if selected_segment:
            index = self.navigator.findData(selected_segment)
            self.navigator.setCurrentIndex(max(0, index))
        self.navigator.blockSignals(False)
        self.browser.setHtml("".join(sections) or "<p>No matching captions.</p>")

    def _jump_to_selected(self) -> None:
        segment_id = self.navigator.currentData()
        if segment_id:
            self.browser.scrollToAnchor(str(segment_id))

    def copy_text(self, language: str) -> None:
        lines: list[str] = []
        for payload in self._visible_items():
            timestamp = _format_ms(int(payload.get("audio_start_ms", 0)))
            if language in {"all", "english"} and payload.get("english"):
                lines.append(f"[{timestamp}] {payload['english']}")
            if language in {"all", "mandarin"} and payload.get("mandarin"):
                lines.append(f"[{timestamp}] {payload['mandarin']}")
        QApplication.clipboard().setText("\n".join(lines))

    def export_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export caption history", "captions.txt", "Text files (*.txt)")
        if not path:
            return
        lines: list[str] = []
        for payload in self._visible_items():
            timestamp = _format_ms(int(payload.get("audio_start_ms", 0)))
            lines.append(f"[{timestamp}] {payload.get('english', '')}\n{payload.get('mandarin', '')}")
        Path(path).write_text("\n\n".join(lines) + "\n", encoding="utf-8")

class PreflightDialog(QDialog):
    def __init__(self, checks: list[tuple[str, str, bool]], can_start: bool) -> None:
        super().__init__()
        self.setWindowTitle("Preflight readiness")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        title = QLabel("Check audio and session readiness before starting")
        title.setStyleSheet("font-size:16px;font-weight:700;")
        layout.addWidget(title)
        for label, detail, ready in checks:
            row = QHBoxLayout()
            state = QLabel("READY" if ready else "ATTENTION")
            state.setStyleSheet(f"color:{'#2E7D32' if ready else '#C62828'};font-weight:800;")
            row.addWidget(QLabel(label), 1)
            row.addWidget(QLabel(detail), 2)
            row.addWidget(state)
            layout.addLayout(row)
        note = QLabel("A source can be connected but silent. Play meeting audio and speak once before continuing.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Start session")
        buttons.button(QDialogButtonBox.Ok).setEnabled(can_start)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class PostSessionSummaryDialog(QDialog):
    def __init__(self, summary: dict[str, Any]) -> None:
        super().__init__()
        self.summary = summary
        self.setWindowTitle("Session complete")
        self.setMinimumWidth(560)
        duration_ms = int(summary.get("duration_ms", 0))
        outages = summary.get("microphone_outages", [])
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Transcript and recordings are flushed and closed.</b>"))
        details = (
            f"Duration: {_format_ms(duration_ms)}<br>"
            f"Committed captions: {int(summary.get('committed_captions', 0))}<br>"
            f"Translated captions: {int(summary.get('translated_captions', 0))}<br>"
            f"Maximum caption delay: {int(summary.get('max_delay_ms', 0)) / 1000:.1f} seconds<br>"
            f"Microphone interruptions: {len(outages)}<br>"
            f"Transcript: {html.escape(str(summary.get('transcript_path', '')))}<br>"
            f"Recording: {html.escape(str(summary.get('audio_path', '') or 'Not recorded'))}"
        )
        report = QLabel(details)
        report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        report.setWordWrap(True)
        layout.addWidget(report)
        actions = QHBoxLayout()
        open_button = QPushButton("Open transcript")
        open_button.setEnabled(bool(summary.get("transcript_path")))
        open_button.clicked.connect(self.open_transcript)
        copy_button = QPushButton("Copy summary")
        copy_button.clicked.connect(self.copy_summary)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        for button in (open_button, copy_button, close_button):
            button.setMinimumHeight(32)
            actions.addWidget(button)
        layout.addLayout(actions)

    def open_transcript(self) -> None:
        path = str(self.summary.get("transcript_path", ""))
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def copy_summary(self) -> None:
        outages = self.summary.get("microphone_outages", [])
        text = (
            f"Duration: {_format_ms(int(self.summary.get('duration_ms', 0)))}\n"
            f"Committed captions: {int(self.summary.get('committed_captions', 0))}\n"
            f"Translated captions: {int(self.summary.get('translated_captions', 0))}\n"
            f"Maximum caption delay: {int(self.summary.get('max_delay_ms', 0)) / 1000:.1f} seconds\n"
            f"Microphone interruptions: {len(outages)}\n"
            f"Transcript: {self.summary.get('transcript_path', '')}\n"
            f"Recording: {self.summary.get('audio_path', '') or 'Not recorded'}"
        )
        QApplication.clipboard().setText(text)

class HotkeySignals(QObject):
    toggle_session = Signal()
    pause = Signal()
    toggle_overlay = Signal()
    toggle_history = Signal()
    toggle_microphone = Signal()


class HotkeyManager:
    def __init__(self, mappings: dict[str, str], signals: HotkeySignals) -> None:
        self.listener = None
        self.error = ""
        try:
            from pynput.keyboard import GlobalHotKeys
            actions: dict[str, Callable[[], None]] = {
                mappings["toggle_session"]: signals.toggle_session.emit,
                mappings["pause"]: signals.pause.emit,
                mappings["toggle_overlay"]: signals.toggle_overlay.emit,
                mappings["toggle_history"]: signals.toggle_history.emit,
                mappings["toggle_microphone"]: signals.toggle_microphone.emit,
            }
            self.listener = GlobalHotKeys(actions)
            self.listener.start()
        except Exception as error:
            self.error = str(error)

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class DesktopController(QObject):
    def __init__(self, app: QApplication, config: AppConfig, backend: subprocess.Popen | None = None) -> None:
        super().__init__()
        self.app = app
        self.config = config
        self.backend = backend
        self.commands = CommandClient(config.transport.socket_path)
        self.overlay = OverlayWindow(config)
        self.history = HistoryDialog()
        self.session_active = False
        self.connected = False
        self.pending_quit = False
        self.summary_dialog: PostSessionSummaryDialog | None = None
        self.reader = SocketReader(config.transport.socket_path)
        self.reader.message.connect(self.on_message)
        self.reader.connection.connect(self.on_connection)
        self.reader.start()
        self.hotkey_signals = HotkeySignals()
        self.hotkey_signals.toggle_session.connect(self.toggle_session)
        self.hotkey_signals.pause.connect(lambda: self.commands.send("pause"))
        self.hotkey_signals.toggle_overlay.connect(self.toggle_overlay)
        self.hotkey_signals.toggle_history.connect(self.toggle_history)
        self.hotkey_signals.toggle_microphone.connect(lambda: self.commands.send("toggle_microphone"))
        self.hotkeys = HotkeyManager(config.hotkeys, self.hotkey_signals)
        self.tray = self._build_tray()
        self.overlay.toggle_session_requested.connect(self.toggle_session)
        self.overlay.pause_requested.connect(self.toggle_pause)
        self.overlay.history_requested.connect(self.toggle_history)
        self.overlay.microphone_requested.connect(self.toggle_microphone)
        self.overlay.hide_requested.connect(self.toggle_overlay)
        self.overlay.close_requested.connect(self.request_quit)
        self.overlay.lock_changed.connect(self._set_lock_from_overlay)
        self.overlay.refresh_requested.connect(self.refresh_devices)
        self.overlay.reconnect_microphone_requested.connect(self.reconnect_microphone)
        self.overlay.retry_requested.connect(lambda: self.commands.send("retry"))
        self.overlay.show()
        self.tray.show()

    def _build_tray(self) -> QSystemTrayIcon:
        tray = QSystemTrayIcon(_tray_icon(), self.app)
        tray.setToolTip("Meeting Translation")
        menu = QMenu()
        self.session_action = QAction("Start session", menu)
        self.session_action.triggered.connect(self.toggle_session)
        menu.addAction(self.session_action)
        pause = QAction("Pause / resume", menu)
        pause.triggered.connect(self.toggle_pause)
        menu.addAction(pause)
        show_overlay = QAction("Show / hide overlay", menu)
        show_overlay.triggered.connect(self.toggle_overlay)
        menu.addAction(show_overlay)
        history = QAction("Caption history", menu)
        history.triggered.connect(self.toggle_history)
        menu.addAction(history)
        self.lock_action = QAction("Lock overlay", menu)
        self.lock_action.setCheckable(True)
        self.lock_action.setChecked(self.config.overlay.locked)
        self.lock_action.toggled.connect(self._set_lock_from_tray)
        menu.addAction(self.lock_action)
        self.record_action = QAction("Record full audio for next session", menu)
        self.record_action.setCheckable(True)
        self.record_action.setChecked(self.config.storage.record_audio)
        self.record_action.toggled.connect(lambda enabled: self.commands.send("record_audio", enabled=enabled))
        menu.addAction(self.record_action)
        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self.request_quit)
        menu.addAction(quit_action)
        tray.setContextMenu(menu)
        tray.activated.connect(lambda reason: self.toggle_overlay() if reason == QSystemTrayIcon.Trigger else None)
        return tray

    def on_message(self, payload: dict[str, Any]) -> None:
        if payload.get("event") == "caption":
            self.overlay.update_event(payload)
            self.history.update_event(payload)
        elif payload.get("event") == "health":
            self.overlay.set_health(payload)
            self.overlay.pause_button.setText("Resume" if payload.get("state") == "paused" else "Pause")
            microphone_enabled = bool(payload.get("microphone_enabled", True))
            self.overlay.microphone_button.setText("Mute" if microphone_enabled else "Unmute")
            self.overlay._sync_audio_state()
            if payload.get("state") == "idle" and self.session_active:
                self.session_active = False
                self.session_action.setText("Start session")
                self.overlay.set_session_active(False)
            if payload.get("state") == "error":
                message = str(payload.get("message") or "Caption service needs attention.")
                self.overlay.audio_warning.setText(message)
                self.overlay.audio_warning.setVisible(self.overlay.audio_expanded)
                self.overlay.retry_button.setVisible(self.overlay.audio_expanded)
        elif payload.get("event") == "session":
            action = payload.get("action")
            self.session_active = action == "started"
            self.session_action.setText("End session" if self.session_active else "Start session")
            self.overlay.set_session_active(self.session_active)
            if action == "ended":
                if self.pending_quit:
                    self.quit()
                else:
                    self.summary_dialog = PostSessionSummaryDialog(dict(payload.get("details", {})))
                    self.summary_dialog.show()
        elif payload.get("event") == "audio_level":
            self.overlay.set_audio_levels(
                float(payload.get("desktop_level", 0)),
                float(payload.get("microphone_level", 0)),
            )
            self.overlay.set_microphone_available(bool(payload.get("microphone_available", False)))

    def on_connection(self, connected: bool, message: str) -> None:
        self.connected = connected
        if not connected:
            self.overlay.status.setText("CONNECTING")
            self.overlay.audio_warning.setText("Caption service disconnected — retry when the backend is available.")
            self.overlay.audio_warning.setVisible(self.overlay.audio_expanded)
            self.overlay.retry_button.setVisible(self.overlay.audio_expanded)
        else:
            if "service disconnected" in self.overlay.audio_warning.text().casefold():
                self.overlay.audio_warning.clear()
                self.overlay.audio_warning.setVisible(False)
                self.overlay.retry_button.setVisible(False)
            if self.hotkeys.error:
                self.tray.showMessage("Global hotkeys unavailable", self.hotkeys.error)

    def toggle_session(self) -> None:
        if self.session_active:
            self.commands.send("stop")
            return
        if not self._run_preflight():
            return
        desktop_source, microphone_source = self.overlay.selected_sources()
        if self.commands.send(
            "start",
            record_audio=self.record_action.isChecked(),
            desktop_source=desktop_source,
            microphone_source=microphone_source,
        ):
            self.session_active = True
            self.overlay.set_session_active(True)

    def _run_preflight(self) -> bool:
        self.overlay._populate_audio_sources()
        desktop_source, microphone_source = self.overlay.selected_sources()
        desktop_connected = self.overlay._selection_connected(self.overlay.desktop_selector)
        microphone_connected = (
            not microphone_source or self.overlay._selection_connected(self.overlay.microphone_selector)
        )
        desktop_signal = _probe_audio_signal(desktop_source) if desktop_connected else False
        microphone_signal = _probe_audio_signal(microphone_source) if microphone_source and microphone_connected else False
        storage_ready = True
        storage_detail = "Transcript destination is writable"
        try:
            self.config.storage.session_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.config.storage.session_root):
                pass
        except OSError as error:
            storage_ready = False
            storage_detail = f"Cannot write transcript: {error}"
        checks = [
            (
                "Meeting audio",
                "Connected — signal detected" if desktop_signal else (
                    "Connected — no signal detected yet" if desktop_connected else "Selected source is disconnected"
                ),
                desktop_connected,
            ),
            (
                "Microphone",
                "Not selected — meeting audio only" if not microphone_source else (
                    "Connected — signal detected" if microphone_signal else (
                        "Connected — no signal detected yet" if microphone_connected else "Selected source is disconnected"
                    )
                ),
                microphone_connected,
            ),
            ("Caption service", "Connected" if self.connected else "Disconnected", self.connected),
            ("Transcript", storage_detail, storage_ready),
            ("Models", "Load when session starts", True),
            (
                "Full audio recording",
                "Enabled" if self.record_action.isChecked() else "Disabled",
                True,
            ),
        ]
        dialog = PreflightDialog(checks, desktop_connected and self.connected and storage_ready)
        return dialog.exec() == QDialog.Accepted

    def refresh_devices(self) -> None:
        self.overlay._populate_audio_sources()

    def reconnect_microphone(self) -> None:
        self.overlay._populate_audio_sources()
        _, microphone_source = self.overlay.selected_sources()
        if not microphone_source:
            self.overlay.audio_warning.setText("Microphone is still unavailable; reconnect it and refresh devices.")
            self.overlay.audio_warning.setVisible(True)
            return
        self.commands.send("reconnect_microphone", microphone_source=microphone_source)

    def _set_lock_from_overlay(self, locked: bool) -> None:
        self.lock_action.setChecked(locked)
        if locked:
            self.tray.showMessage("Overlay locked", "Use the tray menu and clear “Lock overlay” to recover controls.")

    def _set_lock_from_tray(self, locked: bool) -> None:
        self.overlay.apply_lock(locked)
        if locked:
            self.tray.showMessage("Overlay locked", "Use this tray menu to unlock the overlay.")

    def toggle_pause(self) -> None:
        self.commands.send("pause")

    def toggle_microphone(self) -> None:
        self.commands.send("toggle_microphone")

    def toggle_overlay(self) -> None:
        self.overlay.setVisible(not self.overlay.isVisible())

    def toggle_history(self) -> None:
        self.history.setVisible(not self.history.isVisible())
        if self.history.isVisible():
            self.history.raise_()
            self.history.activateWindow()

    def request_quit(self) -> None:
        if self.session_active:
            choice = QMessageBox.question(
                self.overlay,
                "End transcription and quit?",
                "An active caption session is running. Finish captions, flush files, and then quit?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if choice != QMessageBox.Yes:
                return
            self.pending_quit = True
            self.overlay.status.setText("STOPPING")
            self.commands.send("stop")
            return
        self.quit()

    def quit(self) -> None:
        self.overlay.save_preferences()
        self.commands.send("shutdown")
        self.reader.stop()
        self.hotkeys.stop()
        if self.backend is not None:
            try:
                self.backend.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend.terminate()
                self.backend.wait(timeout=2)
        self.tray.hide()
        self.app.quit()


def _probe_audio_signal(source: str, duration_seconds: float = 0.35) -> bool:
    if not source:
        return False
    try:
        process = subprocess.Popen(
            [
                "parec",
                "--raw",
                "--format=s16le",
                "--rate=16000",
                "--channels=1",
                f"--device={source}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    data = b""
    try:
        data, _ = process.communicate(timeout=duration_seconds)
    except subprocess.TimeoutExpired as error:
        data = error.output or b""
        process.terminate()
        try:
            remainder, _ = process.communicate(timeout=1)
            data += remainder or b""
        except subprocess.TimeoutExpired:
            process.kill()
            remainder, _ = process.communicate()
            data += remainder or b""
    return any(data)


def _pulse_sources() -> list[PulseDevice]:
    try:
        completed = subprocess.run(
            ["pactl", "--format=json", "list", "sources"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, list):
            devices = []
            for source in payload:
                if not isinstance(source, dict) or not isinstance(source.get("name"), str):
                    continue
                name = source["name"]
                description = str(source.get("description") or _friendly_source(name))
                devices.append(PulseDevice(name, _device_description(description, name), name.endswith(".monitor")))
            return devices
    return _pulse_sources_short()


def _pulse_sources_short() -> list[PulseDevice]:
    try:
        completed = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    devices = []
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            name = fields[1]
            devices.append(PulseDevice(name, _friendly_source(name), name.endswith(".monitor")))
    return devices


def _device_description(description: str, name: str) -> str:
    description = description.removeprefix("Monitor of ")
    description = description.removesuffix(" Analog Stereo")
    if description.endswith(" Digital Stereo (HDMI)"):
        description = description.removesuffix(" Digital Stereo (HDMI)") + " (HDMI)"
    if name.endswith(".monitor"):
        return f"{description} — Meeting output"
    return description


def _friendly_source(source: str) -> str:
    if source == "@DEFAULT_MONITOR@":
        return "Default meeting output"
    name = source.removesuffix(".monitor")
    name = name.replace("alsa_output.", "").replace("alsa_input.", "")
    name = name.replace("bluez_sink.", "Bluetooth ")
    return name.replace("_", " ")


def _tray_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#7FD1B9"))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 8, 56, 44, 12, 12)
    painter.setPen(QColor("#10151D"))
    painter.setFont(QFont("Sans", 22, QFont.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "CC")
    painter.end()
    return QIcon(pixmap)


def _format_ms(value: int) -> str:
    total = value // 1000
    return f"{total // 60:02d}:{total % 60:02d}"
