from __future__ import annotations

import html as html_lib
import json
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.desktop_studio.ui_job_defaults import (
    DESKTOP_DEFAULT_AUDIO_CONCURRENCY,
    DESKTOP_DEFAULT_COMIC_IMAGE_MODEL,
    DESKTOP_DEFAULT_COMIC_MAX_PANELS,
    DESKTOP_DEFAULT_HIGHLIGHT_MAX_SEGMENTS,
    DESKTOP_DEFAULT_HIGHLIGHT_TARGET_SECONDS,
    DESKTOP_DEFAULT_STORY_MODEL,
    DESKTOP_DEFAULT_SUMMARY_MODEL,
    DESKTOP_DEFAULT_VIDEO_CONCURRENCY,
    desktop_concurrency_options,
)
from apps.desktop_studio.ui_strings import TASK_LABELS, TASK_NAMES
from life_report.session_pipeline import SessionPipelineConfig, default_session_output_dir, run_session_pipeline
from life_report.settings_store import (
    ApiSettings,
    apply_api_settings,
    load_api_settings,
    missing_for_comic,
    missing_for_provider,
    save_api_settings,
)

from PySide6.QtCore import QTimer, QUrl, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - depends on local Qt install.
    QWebEngineView = None

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
except ImportError:  # pragma: no cover - depends on local Qt install.
    QAudioOutput = None
    QMediaPlayer = None
    QVideoWidget = None


def _append_text_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _comic_log_path(run_dir: Path) -> Path:
    return run_dir / "comic" / "comic_generation.log"


def _highlight_log_path(run_dir: Path) -> Path:
    return run_dir / "highlight_video" / "highlight_generation.log"


def _append_exception_logs(run_dir: Path | None, message: str, *, comic: bool = False, highlight: bool = False) -> None:
    if not run_dir:
        return
    _append_text_log(run_dir / "studio_gui.log", message)
    if comic:
        _append_text_log(_comic_log_path(run_dir), message)
    if highlight:
        _append_text_log(_highlight_log_path(run_dir), message)


def _comic_result_from_run_dir(run_dir: Path) -> dict:
    comic_dir = run_dir / "comic"
    storyline_path = comic_dir / "comic_storyline.json"
    reference_plan_path = comic_dir / "comic_reference_plan.json"
    html_path = comic_dir / "daily_comic.html"
    panel_count = 0
    reference_count = 0
    if storyline_path.exists():
        try:
            storyline = json.loads(storyline_path.read_text(encoding="utf-8"))
            panel_count = len(storyline.get("panels", []))
        except Exception:
            panel_count = 0
    if reference_plan_path.exists():
        try:
            reference_plan = json.loads(reference_plan_path.read_text(encoding="utf-8"))
            reference_count = len(reference_plan.get("selected_references", []))
        except Exception:
            reference_count = 0
    return {
        "daily_comic_html_path": str(html_path),
        "panel_count": panel_count,
        "reference_image_count": reference_count,
    }


def _highlight_result_from_run_dir(run_dir: Path) -> dict:
    highlight_dir = run_dir / "highlight_video"
    html_path = highlight_dir / "highlight_video.html"
    plan_path = highlight_dir / "highlight_plan.json"
    segment_count = 0
    estimated_duration_sec = 0
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            segments = plan.get("segments", [])
            segment_count = len(segments)
            estimated_duration_sec = round(
                sum(float(item.get("duration_sec", 0) or 0) for item in segments),
                2,
            )
        except Exception:
            segment_count = 0
            estimated_duration_sec = 0
    return {
        "highlight_video_html_path": str(html_path),
        "segment_count": segment_count,
        "estimated_duration_sec": estimated_duration_sec,
    }


class ReportWorker(QThread):
    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, config: SessionPipelineConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            self.finished.emit(run_session_pipeline(self.config, progress=self.progress.emit))
        except Exception:  # pragma: no cover - UI safety boundary.
            self.failed.emit(traceback.format_exc())


class ComicWorker(QThread):
    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, run_dir: Path, provider: str, image_provider: str, text_model: str, image_model: str) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.provider = provider
        self.image_provider = image_provider
        self.text_model = text_model
        self.image_model = image_model

    def run(self) -> None:
        log_path = _comic_log_path(self.run_dir)
        command = [
            sys.executable,
            "-m",
            "life_report.cli",
            "build-comic-products",
            "--run-dir",
            str(self.run_dir),
            "--provider",
            self.provider,
            "--text-model",
            self.text_model,
            "--image-model",
            self.image_model,
            "--max-reference-images",
            str(DESKTOP_DEFAULT_COMIC_MAX_PANELS),
            "--max-panels",
            str(DESKTOP_DEFAULT_COMIC_MAX_PANELS),
        ]
        if self.image_provider:
            command.extend(["--image-provider", self.image_provider])
        process = None
        try:
            _append_text_log(log_path, f"[{datetime.now().isoformat(timespec='seconds')}] START comic generation")
            _append_text_log(log_path, "COMMAND: " + " ".join(command))
            self.progress.emit("正在生成漫画分镜和参考图...")
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            stdout = process.stdout
            if stdout is not None:
                for line in stdout:
                    if self.isInterruptionRequested():
                        process.terminate()
                        _append_text_log(log_path, "INTERRUPTED: comic generation cancelled from GUI")
                        raise RuntimeError("漫画生成已取消。")
                    message = line.strip()
                    if message:
                        _append_text_log(log_path, message)
                        self.progress.emit(message)
            return_code = process.wait()
            if return_code != 0:
                _append_text_log(log_path, f"FAILED: comic generation subprocess exit code {return_code}")
                raise RuntimeError(f"漫画生成子进程失败，退出码 {return_code}")
            _append_text_log(log_path, "DONE: comic generation completed")
            self.finished.emit(_comic_result_from_run_dir(self.run_dir))
        except Exception:  # pragma: no cover - UI safety boundary.
            if process and process.poll() is None:
                process.terminate()
            error = traceback.format_exc()
            _append_text_log(log_path, error)
            self.failed.emit(error)


class HighlightVideoWorker(QThread):
    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, run_dir: Path, provider: str, text_model: str) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.provider = provider
        self.text_model = text_model

    def run(self) -> None:
        log_path = _highlight_log_path(self.run_dir)
        command = [
            sys.executable,
            "-m",
            "life_report.cli",
            "build-highlight-video",
            "--run-dir",
            str(self.run_dir),
            "--provider",
            self.provider,
            "--text-model",
            self.text_model,
            "--target-seconds",
            str(DESKTOP_DEFAULT_HIGHLIGHT_TARGET_SECONDS),
            "--max-segments",
            str(DESKTOP_DEFAULT_HIGHLIGHT_MAX_SEGMENTS),
        ]
        process = None
        try:
            _append_text_log(log_path, f"[{datetime.now().isoformat(timespec='seconds')}] START highlight generation")
            _append_text_log(
                log_path,
                "COMMAND: " + " ".join(command),
            )
            self.progress.emit("正在生成高光视频故事线和剪辑计划...")
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            stdout = process.stdout
            if stdout is not None:
                for line in stdout:
                    if self.isInterruptionRequested():
                        process.terminate()
                        _append_text_log(log_path, "INTERRUPTED: highlight generation cancelled from GUI")
                        raise RuntimeError("高光视频生成已取消。")
                    message = line.strip()
                    if message:
                        _append_text_log(log_path, message)
                        self.progress.emit(message)
            return_code = process.wait()
            if return_code != 0:
                _append_text_log(log_path, f"FAILED: highlight generation subprocess exit code {return_code}")
                raise RuntimeError(f"高光视频生成子进程失败，退出码 {return_code}")
            _append_text_log(log_path, "DONE: highlight generation completed")
            self.finished.emit(_highlight_result_from_run_dir(self.run_dir))
        except Exception:  # pragma: no cover - UI safety boundary.
            if process and process.poll() is None:
                process.terminate()
            error = traceback.format_exc()
            _append_text_log(log_path, error)
            self.failed.emit(error)


class PhoneLifeAgentWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PhoneLifeAgent 工作台")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 740)
        self.setStyleSheet(
            """
            QMainWindow, QWidget#Root {
                background: #f4f7fb;
                color: #182231;
                font-family: "Avenir Next", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
            }
            QLabel { color: #243244; font-size: 13px; }
            QLabel#AppTitle {
                color: #111b2a;
                font-size: 25px;
                font-weight: 900;
            }
            QLabel#MutedLabel {
                color: #64748b;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#StepChip {
                color: #52667d;
                background: #eef5fc;
                border: 1px solid #d4e1ef;
                border-radius: 12px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: 700;
            }
            QWidget#Header {
                background: transparent;
            }
            QWidget#ActionBar {
                background: #ffffff;
                border: 1px solid #dde5ef;
                border-radius: 16px;
            }
            QGroupBox {
                border: 1px solid #dce4ee;
                border-radius: 16px;
                margin-top: 18px;
                padding: 18px 14px 14px 14px;
                font-size: 14px;
                font-weight: 800;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                color: #111827;
                background: #f4f7fb;
            }
            QPushButton {
                border: 1px solid #cfd9e6;
                border-radius: 11px;
                padding: 9px 13px;
                background: #ffffff;
                color: #273449;
                font-size: 13px;
                font-weight: 750;
            }
            QPushButton:hover {
                background: #f5f9ff;
                border-color: #9dbff5;
            }
            QPushButton:pressed { background: #e8f1ff; }
            QPushButton:disabled {
                color: #9aa8b8;
                background: #eef2f7;
                border-color: #d6dee9;
            }
            QPushButton#PrimaryButton {
                background: #0f72f2;
                color: #ffffff;
                border: 1px solid #0f72f2;
                border-radius: 14px;
                padding: 12px 20px;
                font-size: 15px;
                font-weight: 900;
            }
            QPushButton#PrimaryButton:hover { background: #075fd0; border-color: #075fd0; }
            QPushButton#PrimaryButton:pressed { background: #064faa; }
            QPushButton#PrimaryButton:disabled {
                background: #dfe6ef;
                color: #91a0b1;
                border-color: #dfe6ef;
            }
            QPushButton#SecondaryButton {
                background: #f7fafc;
                color: #3c4b60;
                border-radius: 14px;
                padding: 12px 18px;
                font-size: 14px;
                font-weight: 800;
            }
            QPushButton#SecondaryButton:disabled {
                color: #9aa8b8;
                background: #eef2f7;
                border-color: #d6dee9;
            }
            QPushButton#SettingsButton {
                border-radius: 12px;
                padding: 9px 16px;
                background: #ffffff;
            }
            QLineEdit, QComboBox, QListWidget, QTextEdit {
                border: 1px solid #d1dbe8;
                border-radius: 12px;
                padding: 8px;
                background: #ffffff;
                selection-background-color: #dbeafe;
                font-size: 13px;
            }
            QListWidget, QTextEdit {
                background: #fbfdff;
            }
            QListWidget::item {
                padding: 7px;
                border-radius: 8px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background: #dbeafe;
                color: #102a43;
            }
            QTextEdit {
                line-height: 140%;
            }
            QProgressBar {
                border: 1px solid #d5deea;
                border-radius: 8px;
                height: 12px;
                text-align: center;
                color: transparent;
                background: #eef3f8;
            }
            QProgressBar::chunk {
                border-radius: 8px;
                background: #0f72f2;
            }
            QTabWidget::pane {
                border: 1px solid #dce4ee;
                border-radius: 16px;
                top: -1px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #eef3f8;
                color: #3b485a;
                border: 1px solid #d3dce8;
                padding: 8px 18px;
                min-width: 110px;
                font-size: 13px;
                font-weight: 750;
            }
            QTabBar::tab:first { border-top-left-radius: 12px; border-bottom-left-radius: 12px; }
            QTabBar::tab:last { border-top-right-radius: 12px; border-bottom-right-radius: 12px; }
            QTabBar::tab:selected {
                background: #0f72f2;
                color: #ffffff;
                border-color: #0f72f2;
            }
            QTabBar::tab:hover:!selected { background: #e4edf8; }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QSplitter::handle {
                background: transparent;
                width: 10px;
            }
            """
        )
        self.worker: ReportWorker | None = None
        self.comic_worker: ComicWorker | None = None
        self.highlight_worker: HighlightVideoWorker | None = None
        self.last_report_html: Path | None = None
        self.last_run_dir: Path | None = None
        self.last_comic_html: Path | None = None
        self.last_highlight_html: Path | None = None
        self.last_highlight_video: Path | None = None
        self.run_started_at: float | None = None
        self.last_progress_message = ""
        self.last_log_at = 0.0
        self.task_started_at: dict[str, float] = {}
        self.api_settings = ApiSettings()
        self.summary_model = DESKTOP_DEFAULT_SUMMARY_MODEL
        self.story_model = DESKTOP_DEFAULT_STORY_MODEL
        self.task_names = list(TASK_NAMES)
        self.task_labels = dict(TASK_LABELS)
        self.task_status = {name: "pending" for name in self.task_names}

        self.session_list = QListWidget()
        self.session_status_label = QLabel("还没有选择采集数据")
        self.session_status_label.setObjectName("MutedLabel")
        self.session_status_label.setWordWrap(True)
        self.output_label = QLabel(str(default_session_output_dir(Path("run"))))
        self.output_label.setWordWrap(True)
        self.output_label.setMinimumHeight(40)
        self.output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["aliyun", "mock"])
        self.provider_combo.currentTextChanged.connect(self.update_settings_status)
        self.use_amap_checkbox = QCheckBox("使用高德增强定位")
        self.use_amap_checkbox.setChecked(True)
        self.use_amap_checkbox.stateChanged.connect(self.update_settings_status)
        self.force_rebuild_checkbox = QCheckBox("强制重新生成")
        self.audio_concurrency_combo = QComboBox()
        self.audio_concurrency_combo.addItems(desktop_concurrency_options())
        self.audio_concurrency_combo.setCurrentText(str(DESKTOP_DEFAULT_AUDIO_CONCURRENCY))
        self.video_concurrency_combo = QComboBox()
        self.video_concurrency_combo.addItems(desktop_concurrency_options())
        self.video_concurrency_combo.setCurrentText(str(DESKTOP_DEFAULT_VIDEO_CONCURRENCY))
        self.settings_button = QPushButton("设置")
        self.settings_button.setObjectName("SettingsButton")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.api_status_label = QLabel("")
        self.api_status_label.setObjectName("MutedLabel")
        self.api_status_label.setWordWrap(True)
        self.settings_status = QLabel("")
        self.settings_status.setWordWrap(True)
        self.current_step_label = QLabel("空闲")
        self.current_step_label.setObjectName("MutedLabel")
        self.current_step_label.setWordWrap(True)
        self.total_elapsed_label = QLabel("总耗时：0s")
        self.total_elapsed_label.setObjectName("MutedLabel")
        self.total_elapsed_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.task_names))
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.task_list = QListWidget()
        self.task_list.setMinimumHeight(270)
        self.task_list.setFocusPolicy(Qt.NoFocus)
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.setInterval(1000)
        self.heartbeat_timer.timeout.connect(self.on_heartbeat)

        if QWebEngineView is not None:
            self.preview = QWebEngineView()
        else:
            self.preview = QTextEdit()
            self.preview.setReadOnly(True)
        self.set_empty_preview()

        self.tabs = QTabWidget()
        self.open_story_tab_button = QPushButton("打开故事")
        self.open_story_tab_button.setObjectName("SecondaryButton")
        self.open_story_tab_button.clicked.connect(self.open_report)
        self.open_story_tab_button.setEnabled(False)
        self.comic_status_label = QLabel("请先生成故事，完成后会自动生成漫画。")
        self.comic_status_label.setObjectName("MutedLabel")
        self.comic_status_label.setWordWrap(True)
        self.generate_comic_button = QPushButton("生成漫画")
        self.generate_comic_button.setObjectName("SecondaryButton")
        self.generate_comic_button.clicked.connect(lambda: self.generate_comic())
        self.generate_comic_button.setEnabled(False)
        self.open_comic_button = QPushButton("打开漫画")
        self.open_comic_button.setObjectName("SecondaryButton")
        self.open_comic_button.clicked.connect(self.open_comic)
        self.open_comic_button.setEnabled(False)
        self.highlight_status_label = QLabel("请先生成故事，完成后会自动生成高光视频。")
        self.highlight_status_label.setObjectName("MutedLabel")
        self.highlight_status_label.setWordWrap(True)
        self.generate_highlight_button = QPushButton("生成高光视频")
        self.generate_highlight_button.setObjectName("SecondaryButton")
        self.generate_highlight_button.clicked.connect(lambda: self.generate_highlight_video())
        self.generate_highlight_button.setEnabled(False)
        self.play_highlight_button = QPushButton("播放/停止")
        self.play_highlight_button.setObjectName("SecondaryButton")
        self.play_highlight_button.clicked.connect(self.toggle_highlight_playback)
        self.play_highlight_button.setEnabled(False)
        self.open_highlight_button = QPushButton("打开高光视频")
        self.open_highlight_button.setObjectName("SecondaryButton")
        self.open_highlight_button.clicked.connect(self.open_highlight_video)
        self.open_highlight_button.setEnabled(False)
        if QWebEngineView is not None:
            self.comic_preview = QWebEngineView()
        else:
            self.comic_preview = QTextEdit()
            self.comic_preview.setReadOnly(True)
        self.highlight_player = None
        self.highlight_audio_output = None
        self.highlight_video_widget = None
        self.highlight_video_container = None
        self.highlight_placeholder = None
        if QVideoWidget is not None and QMediaPlayer is not None and QAudioOutput is not None:
            self.highlight_preview = QStackedWidget()
            self.highlight_placeholder = QTextEdit()
            self.highlight_placeholder.setReadOnly(True)
            self.highlight_video_widget = QVideoWidget()
            self.highlight_video_container = QWidget()
            video_layout = QVBoxLayout()
            video_layout.setContentsMargins(0, 0, 0, 0)
            video_layout.setSpacing(8)
            video_layout.addWidget(self.highlight_video_widget, stretch=1)
            video_layout.addWidget(self.play_highlight_button, alignment=Qt.AlignCenter)
            self.highlight_video_container.setLayout(video_layout)
            self.highlight_player = QMediaPlayer(self)
            self.highlight_audio_output = QAudioOutput(self)
            self.highlight_audio_output.setVolume(0.85)
            self.highlight_player.setAudioOutput(self.highlight_audio_output)
            self.highlight_player.setVideoOutput(self.highlight_video_widget)
            self.highlight_preview.addWidget(self.highlight_placeholder)
            self.highlight_preview.addWidget(self.highlight_video_container)
        elif QWebEngineView is not None:
            self.highlight_preview = QWebEngineView()
        else:
            self.highlight_preview = QTextEdit()
            self.highlight_preview.setReadOnly(True)
        self.set_empty_comic_preview()
        self.set_empty_highlight_preview()
        self.story_tab = self.build_story_tab()
        self.comic_tab = self.build_comic_tab()
        self.highlights_tab = self.build_highlight_tab()

        self.generate_button = QPushButton("生成生活故事")
        self.generate_button.setObjectName("PrimaryButton")
        self.generate_button.setMinimumHeight(48)
        self.generate_button.clicked.connect(self.generate_report)

        self.build_layout()
        self.load_settings()

    def build_layout(self) -> None:
        left_widget = self.build_left_panel()
        center_widget = self.build_center_panel()
        right_widget = self.build_right_panel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 820, 320])
        splitter.setChildrenCollapsible(False)

        root = QVBoxLayout()
        root.setContentsMargins(22, 18, 22, 22)
        root.setSpacing(14)
        root.addWidget(splitter, stretch=1)
        root.addWidget(self.progress_bar)

        container = QWidget()
        container.setObjectName("Root")
        container.setLayout(root)
        self.setCentralWidget(container)
        self.refresh_task_list()

    def build_left_panel(self) -> QWidget:
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        session_group = QGroupBox("设置")
        session_layout = QVBoxLayout()
        session_layout.setSpacing(12)
        session_layout.setContentsMargins(16, 16, 16, 16)
        session_layout.addWidget(QLabel("1. 选择 iPhone 采集数据"))
        add_button = QPushButton("选择采集文件夹")
        add_button.clicked.connect(self.add_session)
        session_layout.addWidget(add_button)
        session_layout.addWidget(self.session_status_label)
        session_layout.addSpacing(14)
        session_layout.addWidget(QLabel("2. 输出位置"))
        output_button = QPushButton("修改输出位置")
        output_button.clicked.connect(self.choose_output_dir)
        session_layout.addWidget(output_button)
        session_layout.addWidget(self.output_label)
        session_layout.addSpacing(14)
        session_layout.addWidget(QLabel("3. 开始生成"))
        session_layout.addWidget(self.generate_button)
        subtitle = QLabel("故事完成后会自动生成漫画和高光视频")
        subtitle.setObjectName("MutedLabel")
        subtitle.setWordWrap(True)
        session_layout.addWidget(subtitle)
        session_layout.addWidget(self.current_step_label)
        session_layout.addWidget(self.total_elapsed_label)
        session_layout.addStretch(1)
        settings_row = QHBoxLayout()
        settings_row.setContentsMargins(0, 0, 0, 0)
        settings_row.setSpacing(10)
        settings_row.addWidget(self.api_status_label, stretch=1)
        settings_row.addWidget(self.settings_button)
        session_layout.addLayout(settings_row)
        session_group.setLayout(session_layout)
        left.addWidget(session_group, stretch=1)

        left.addStretch(1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(280)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(left_widget)
        scroll.setMinimumWidth(290)
        return scroll

    def build_center_panel(self) -> QWidget:
        self.tabs.addTab(self.story_tab, "故事")
        self.tabs.addTab(self.comic_tab, "漫画")
        self.tabs.addTab(self.highlights_tab, "高光视频")
        return self.tabs

    def build_story_tab(self) -> QWidget:
        root = QVBoxLayout()
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(10)
        controls.addWidget(self.open_story_tab_button)
        controls.addStretch(1)
        root.addLayout(controls)
        root.addWidget(self.preview, stretch=1)
        widget = QWidget()
        widget.setLayout(root)
        return widget

    def build_comic_tab(self) -> QWidget:
        root = QVBoxLayout()
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(10)
        controls.addWidget(self.generate_comic_button)
        controls.addWidget(self.open_comic_button)
        controls.addWidget(self.comic_status_label, stretch=1)
        root.addLayout(controls)
        root.addWidget(self.comic_preview, stretch=1)
        widget = QWidget()
        widget.setLayout(root)
        return widget

    def build_highlight_tab(self) -> QWidget:
        root = QVBoxLayout()
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(10)
        controls.addWidget(self.generate_highlight_button)
        controls.addWidget(self.open_highlight_button)
        controls.addWidget(self.highlight_status_label, stretch=1)
        root.addLayout(controls)
        root.addWidget(self.highlight_preview, stretch=1)
        widget = QWidget()
        widget.setLayout(root)
        return widget

    def set_empty_preview(self) -> None:
        html = """
        <html>
        <body style="margin:0;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,'Avenir Next',sans-serif;color:#172033;">
          <div style="height:100vh;display:flex;align-items:center;justify-content:center;">
            <div style="width:420px;text-align:center;border:1px solid #e1e7ef;border-radius:24px;padding:40px 36px;background:#fbfdff;">
              <div style="font-size:17px;font-weight:800;margin-bottom:10px;">故事预览</div>
              <div style="font-size:13px;line-height:1.7;color:#64748b;">选择采集数据后点击生成生活故事，结果会显示在这里。</div>
            </div>
          </div>
        </body>
        </html>
        """
        if QWebEngineView is not None:
            self.preview.setHtml(html)
        else:
            self.preview.setHtml(html)

    def set_story_running_preview(self, message: str = "正在生成生活故事") -> None:
        preview_html = f"""
        <html>
        <body style="margin:0;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,'Avenir Next',sans-serif;color:#172033;">
          <div style="height:100vh;display:flex;align-items:center;justify-content:center;">
            <div style="width:520px;text-align:center;border:1px solid #d7e3f2;border-radius:24px;padding:42px 40px;background:#fbfdff;">
              <div style="font-size:18px;font-weight:800;margin-bottom:12px;">正在生成故事</div>
              <div style="font-size:13px;line-height:1.7;color:#64748b;margin-bottom:6px;">当前正在：{html_lib.escape(message)}</div>
              <div style="font-size:12px;line-height:1.7;color:#7a8da6;">右侧进度和日志会持续显示详细阶段。</div>
            </div>
          </div>
        </body>
        </html>
        """
        if QWebEngineView is not None:
            self.preview.setHtml(preview_html)
        else:
            self.preview.setHtml(preview_html)

    def set_empty_comic_preview(self) -> None:
        html = """
        <html>
        <body style="margin:0;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,'Avenir Next',sans-serif;color:#172033;">
          <div style="height:100vh;display:flex;align-items:center;justify-content:center;">
            <div style="width:420px;text-align:center;border:1px solid #e1e7ef;border-radius:24px;padding:40px 36px;background:#fbfdff;">
              <div style="font-size:17px;font-weight:800;margin-bottom:10px;">漫画</div>
              <div style="font-size:13px;line-height:1.7;color:#64748b;">漫画结果会显示在这里。</div>
            </div>
          </div>
        </body>
        </html>
        """
        if QWebEngineView is not None:
            self.comic_preview.setHtml(html)
        else:
            self.comic_preview.setHtml(html)

    def set_empty_highlight_preview(self) -> None:
        html = """
        <html>
        <body style="margin:0;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,'Avenir Next',sans-serif;color:#172033;">
          <div style="height:100vh;display:flex;align-items:center;justify-content:center;">
            <div style="width:420px;text-align:center;border:1px solid #e1e7ef;border-radius:24px;padding:40px 36px;background:#fbfdff;">
              <div style="font-size:17px;font-weight:800;margin-bottom:10px;">高光视频</div>
              <div style="font-size:13px;line-height:1.7;color:#64748b;">高光视频结果会显示在这里。</div>
            </div>
          </div>
        </body>
        </html>
        """
        self._set_highlight_placeholder_html(html)

    def set_highlight_ready_preview(self, html_path: Path, video_path: Path | None = None) -> None:
        if video_path and video_path.exists() and self.highlight_player is not None and self.highlight_video_widget is not None:
            self.last_highlight_video = video_path
            self.highlight_player.setSource(QUrl.fromLocalFile(str(video_path.resolve())))
            self.highlight_preview.setCurrentWidget(self.highlight_video_container)
            self.play_highlight_button.setEnabled(True)
            return
        html_text = f"""
        <html>
        <body style="margin:0;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,'Avenir Next',sans-serif;color:#172033;">
          <div style="height:100vh;display:flex;align-items:center;justify-content:center;">
            <div style="width:460px;text-align:center;border:1px solid #e1e7ef;border-radius:24px;padding:40px 36px;background:#fbfdff;">
              <div style="font-size:17px;font-weight:800;margin-bottom:10px;">高光视频已生成</div>
              <div style="font-size:13px;line-height:1.8;color:#64748b;">
                内置预览可能受 Qt 视频解码限制影响。点击上方“打开高光视频”，会用系统浏览器播放。
              </div>
              <div style="font-size:12px;line-height:1.7;color:#8a99aa;margin-top:18px;word-break:break-all;">
                {html_lib.escape(str(video_path or html_path))}
              </div>
            </div>
          </div>
        </body>
        </html>
        """
        self._set_highlight_placeholder_html(html_text)

    def _set_highlight_placeholder_html(self, html_text: str) -> None:
        if self.highlight_player is not None:
            self.highlight_player.stop()
        if self.highlight_placeholder is not None:
            self.highlight_placeholder.setHtml(html_text)
            self.highlight_preview.setCurrentWidget(self.highlight_placeholder)
        elif QWebEngineView is not None:
            self.highlight_preview.setHtml(html_text)
        else:
            self.highlight_preview.setHtml(html_text)

    def build_right_panel(self) -> QWidget:
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(16, 16, 16, 16)
        progress_layout.addWidget(self.task_list)
        progress_group.setLayout(progress_layout)
        progress_group.setMinimumHeight(340)
        right.addWidget(progress_group)

        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(self.log)
        log_group.setLayout(log_layout)
        right.addWidget(log_group, stretch=1)

        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setMinimumWidth(310)
        return right_widget

    def add_session(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择采集数据文件夹",
            str(Path.home()),
        )
        if directory:
            self.session_list.clear()
            self.session_list.addItem(directory)
            self.session_status_label.setText(Path(directory).name)
            self.append_log(f"已选择采集数据：{directory}")

    def remove_selected(self) -> None:
        for item in self.session_list.selectedItems():
            self.append_log(f"已移除采集数据：{item.text()}")
            self.session_list.takeItem(self.session_list.row(item))

    def choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择输出位置",
            str(REPO_ROOT / "outputs"),
        )
        if directory:
            self.output_label.setText(directory)

    def selected_sessions(self) -> list[Path]:
        return [Path(self.session_list.item(i).text()) for i in range(self.session_list.count())]

    def current_settings(self) -> ApiSettings:
        return self.api_settings

    def load_settings(self) -> None:
        self.api_settings = load_api_settings(REPO_ROOT)
        self.update_settings_status()

    def open_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("API 设置")
        layout = QVBoxLayout()
        form = QFormLayout()
        dashscope_key_input = QLineEdit(self.api_settings.dashscope_api_key)
        dashscope_key_input.setEchoMode(QLineEdit.Password)
        dashscope_base_url_input = QLineEdit(self.api_settings.dashscope_openai_base_url)
        amap_key_input = QLineEdit(self.api_settings.amap_api_key)
        amap_key_input.setEchoMode(QLineEdit.Password)
        ark_key_input = QLineEdit(self.api_settings.ark_api_key)
        ark_key_input.setEchoMode(QLineEdit.Password)
        summary_model_input = QLineEdit(self.summary_model)
        story_model_input = QLineEdit(self.story_model)
        provider_combo = QComboBox()
        provider_combo.addItems(["aliyun", "mock"])
        provider_combo.setCurrentText(self.provider_combo.currentText())
        audio_concurrency_combo = QComboBox()
        audio_concurrency_combo.addItems(desktop_concurrency_options())
        audio_concurrency_combo.setCurrentText(self.audio_concurrency_combo.currentText())
        video_concurrency_combo = QComboBox()
        video_concurrency_combo.addItems(desktop_concurrency_options())
        video_concurrency_combo.setCurrentText(self.video_concurrency_combo.currentText())
        use_amap_checkbox = QCheckBox("使用高德增强定位")
        use_amap_checkbox.setChecked(self.use_amap_checkbox.isChecked())
        force_rebuild_checkbox = QCheckBox("强制重新生成")
        force_rebuild_checkbox.setChecked(self.force_rebuild_checkbox.isChecked())
        form.addRow("阿里 DashScope Key", dashscope_key_input)
        dashscope_base_url_input.setPlaceholderText("留空则使用默认 DashScope 地址")
        form.addRow("阿里 Base URL（选填）", dashscope_base_url_input)
        form.addRow("高德 Key", amap_key_input)
        form.addRow("Seedream/Ark Key", ark_key_input)
        form.addRow("摘要模型", summary_model_input)
        form.addRow("故事模型", story_model_input)
        form.addRow("模型服务", provider_combo)
        form.addRow("音频并发数", audio_concurrency_combo)
        form.addRow("视频并发数", video_concurrency_combo)
        layout.addWidget(use_amap_checkbox)
        layout.addWidget(force_rebuild_checkbox)
        layout.addLayout(form)
        status_label = QLabel(self.settings_status.text())
        status_label.setWordWrap(True)
        layout.addWidget(status_label)

        buttons = QHBoxLayout()
        save_button = QPushButton("保存")
        cancel_button = QPushButton("取消")
        buttons.addWidget(save_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)
        dialog.setLayout(layout)

        def save_from_dialog() -> None:
            settings = ApiSettings(
                dashscope_api_key=dashscope_key_input.text().strip(),
                dashscope_openai_base_url=dashscope_base_url_input.text().strip(),
                amap_api_key=amap_key_input.text().strip(),
                ark_api_key=ark_key_input.text().strip(),
            )
            self.provider_combo.setCurrentText(provider_combo.currentText())
            self.audio_concurrency_combo.setCurrentText(audio_concurrency_combo.currentText())
            self.video_concurrency_combo.setCurrentText(video_concurrency_combo.currentText())
            self.use_amap_checkbox.setChecked(use_amap_checkbox.isChecked())
            self.force_rebuild_checkbox.setChecked(force_rebuild_checkbox.isChecked())
            self.summary_model = summary_model_input.text().strip() or DESKTOP_DEFAULT_SUMMARY_MODEL
            self.story_model = story_model_input.text().strip() or DESKTOP_DEFAULT_STORY_MODEL
            if self.save_settings(settings):
                dialog.accept()

        save_button.clicked.connect(save_from_dialog)
        cancel_button.clicked.connect(dialog.reject)
        dialog.exec()
        self.update_settings_status()

    def save_settings(self, settings: ApiSettings) -> bool:
        try:
            destination = save_api_settings(settings, REPO_ROOT)
            self.api_settings = settings
            apply_api_settings(settings)
            self.append_log(f"已保存 API 设置：{destination}")
            self.update_settings_status()
            return True
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return False

    def update_settings_status(self) -> None:
        settings = self.current_settings()
        missing = missing_for_provider(settings, self.provider_combo.currentText(), self.use_amap_checkbox.isChecked())
        configured = []
        if settings.dashscope_api_key:
            configured.append("阿里 DashScope Key")
        configured.append("阿里 Base URL（默认）" if not settings.dashscope_openai_base_url else "阿里 Base URL")
        if settings.amap_api_key:
            configured.append("高德 Key")
        if settings.ark_api_key:
            configured.append("Seedream/Ark Key")
        status = "已配置：" + ("，".join(configured) if configured else "无")
        if missing:
            status += "\n当前运行缺少：" + "，".join(missing)
        self.settings_status.setText(status)
        compact_status = "API 就绪" if not missing else "API 缺失"
        self.api_status_label.setText(compact_status)

    def generate_report(self) -> None:
        sessions = self.selected_sessions()
        if not sessions:
            QMessageBox.warning(self, "缺少采集数据", "请先选择一个 LifeLogger 采集数据文件夹。")
            return
        if len(sessions) != 1:
            QMessageBox.warning(self, "只能选择一个采集数据", "完整流程当前一次只处理一个 LifeLogger 采集数据文件夹。")
            return
        settings = self.current_settings()
        missing = missing_for_provider(settings, self.provider_combo.currentText(), self.use_amap_checkbox.isChecked())
        if missing:
            QMessageBox.warning(self, "缺少 API 设置", "请先配置：" + "，".join(missing))
            return
        apply_api_settings(settings)

        output_dir = Path(self.output_label.text()).expanduser()
        self.generate_button.setEnabled(False)
        self.open_story_tab_button.setEnabled(False)
        self.generate_comic_button.setEnabled(False)
        self.open_comic_button.setEnabled(False)
        self.generate_highlight_button.setEnabled(False)
        self.play_highlight_button.setEnabled(False)
        self.open_highlight_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.current_step_label.setText("准备开始...")
        self.total_elapsed_label.setText("总耗时：0s")
        self.set_story_running_preview("准备处理采集数据")
        self.run_started_at = time.time()
        self.last_progress_message = "准备开始..."
        self.heartbeat_timer.start()
        self.comic_status_label.setText("等待故事结果。")
        self.highlight_status_label.setText("等待故事结果。")
        self.task_status = {name: "pending" for name in self.task_names}
        self.task_started_at = {}
        self.refresh_task_list()
        self.append_log("正在生成故事...")

        config = SessionPipelineConfig(
            session_path=sessions[0],
            output_dir=output_dir,
            provider=self.provider_combo.currentText(),
            use_amap=self.use_amap_checkbox.isChecked(),
            summary_model=self.summary_model,
            story_model=self.story_model,
            audio_concurrency=int(self.audio_concurrency_combo.currentText()),
            video_concurrency=int(self.video_concurrency_combo.currentText()),
            force_rebuild=self.force_rebuild_checkbox.isChecked(),
        )
        self.worker = ReportWorker(config)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_report_finished)
        self.worker.failed.connect(self.on_report_failed)
        self.worker.start()

    def append_log(self, message: str) -> None:
        now = time.time()
        if message == self.last_progress_message and now - self.last_log_at < 2.0:
            return
        self.last_log_at = now
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.log.append(line)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        if self.last_run_dir:
            _append_text_log(self.last_run_dir / "studio_gui.log", line)

    def on_progress(self, message: str) -> None:
        if message.startswith("TASK|"):
            self.on_task_progress(message)
            return
        message = self._display_message(message)
        match = re.search(r"(\d+)/7", message)
        if match:
            self.progress_bar.setValue(int(match.group(1)))
        self.last_progress_message = message
        self.current_step_label.setText(message)
        self.set_story_running_preview(message)
        self.append_log(message)

    def on_task_progress(self, message: str) -> None:
        parts = message.split("|", 3)
        if len(parts) < 4:
            return
        _, name, status, detail = parts
        self.task_status[name] = status if not detail else f"{status} {detail}"
        if status == "running":
            self.task_started_at.setdefault(name, time.time())
        if status in {"done", "skipped", "failed"}:
            self.task_started_at.pop(name, None)
        label = self.task_labels.get(name, name)
        display_status = self._ui_status(self.task_status[name])
        self.last_progress_message = f"{label}：{display_status}"
        completed = sum(1 for value in self.task_status.values() if value.startswith(("done", "skipped")))
        self.progress_bar.setValue(completed)
        self.current_step_label.setText(f"{label}：{display_status}")
        if name in self.task_names[:7]:
            self.set_story_running_preview(self.task_labels[name])
        self.refresh_task_list()
        self.append_log(f"{label}：{display_status}")

    def refresh_task_list(self) -> None:
        self.task_list.clear()
        completed = 0
        for name in self.task_names:
            label = self.task_labels.get(name, name)
            status = self.task_status.get(name, "pending")
            if status.startswith(("done", "skipped")):
                completed += 1
            item_text = f"{label}: {self._ui_status(status)}"
            if status.startswith("running") and name in self.task_started_at:
                item_text += f" · {int(time.time() - self.task_started_at[name])}s"
            self.task_list.addItem(item_text)
            item = self.task_list.item(self.task_list.count() - 1)
            if status.startswith("running"):
                item.setForeground(QColor("#087f3f"))
                item.setBackground(QColor("#e6f7ec"))
                self.task_list.scrollToItem(item)
        self.progress_bar.setValue(completed)

    def on_report_finished(self, result: dict) -> None:
        output_dir = Path(result["output_dir"])
        self.last_run_dir = output_dir
        self.last_report_html = Path(result["story_html_path"])
        self.progress_bar.setValue(7)
        self.current_step_label.setText("故事完成")
        self._update_total_elapsed()
        for name in self.task_names[:7]:
            if self.task_status.get(name) in {"pending", "running"}:
                self.task_status[name] = "done"
        self.refresh_task_list()
        self.append_log(f"故事已输出：{output_dir}")
        for name in result["files"]:
            self.append_log(f"  - {name}")

        QTimer.singleShot(0, self.load_story_preview)

        self.generate_button.setEnabled(True)
        self.open_story_tab_button.setEnabled(True)
        self.generate_comic_button.setEnabled(True)
        self.generate_highlight_button.setEnabled(True)
        self.comic_status_label.setText("故事已完成，准备生成漫画...")
        self.highlight_status_label.setText("故事已完成，准备生成高光视频...")
        QTimer.singleShot(700, self.auto_generate_comic)
        QTimer.singleShot(700, self.auto_generate_highlight_video)

    def auto_generate_comic(self) -> None:
        self.append_log("AUTO START COMIC")
        try:
            self.generate_comic(show_missing_warning=False)
        except Exception:
            error = traceback.format_exc()
            self.append_log("AUTO START COMIC FAILED")
            self.append_log(error)
            _append_exception_logs(self.last_run_dir, error, comic=True)
            self.task_status["Comic"] = "failed"
            self.task_started_at.pop("Comic", None)
            self.refresh_task_list()

    def auto_generate_highlight_video(self) -> None:
        self.append_log("AUTO START HIGHLIGHT")
        try:
            self.generate_highlight_video(show_missing_warning=False)
        except Exception:
            error = traceback.format_exc()
            self.append_log("AUTO START HIGHLIGHT FAILED")
            self.append_log(error)
            _append_exception_logs(self.last_run_dir, error, highlight=True)
            self.task_status["Highlight Video"] = "failed"
            self.task_started_at.pop("Highlight Video", None)
            self.refresh_task_list()

    def on_report_failed(self, message: str) -> None:
        self.heartbeat_timer.stop()
        self.current_step_label.setText("生成失败")
        self._update_total_elapsed()
        for name, status in list(self.task_status.items()):
            if status == "running":
                self.task_status[name] = "failed"
        self.refresh_task_list()
        self.append_log("生成失败，完整错误：")
        self.append_log(message)
        QMessageBox.critical(self, "生成失败", message.splitlines()[-1] if message.splitlines() else message)
        self.generate_button.setEnabled(True)
        if self.last_run_dir:
            self.generate_comic_button.setEnabled(True)
            self.generate_highlight_button.setEnabled(True)

    def on_heartbeat(self) -> None:
        workers_running = any(
            worker and worker.isRunning() for worker in [self.worker, self.comic_worker, self.highlight_worker]
        )
        if not workers_running or self.run_started_at is None:
            self.heartbeat_timer.stop()
            return
        elapsed = int(time.time() - self.run_started_at)
        self.current_step_label.setText(f"{self.last_progress_message} · {elapsed}s")
        self.total_elapsed_label.setText(f"总耗时：{elapsed}s")
        self.refresh_task_list()

    def load_story_preview(self) -> None:
        if not self.last_report_html or not self.last_report_html.exists():
            return
        if QWebEngineView is not None:
            self.preview.load(QUrl.fromLocalFile(str(self.last_report_html.resolve())))
        else:
            self.preview.setText(self.last_report_html.read_text(encoding="utf-8"))
        self.tabs.setCurrentIndex(0)

    def open_report(self) -> None:
        if self.last_report_html and self.last_report_html.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_report_html.resolve())))

    def generate_comic(self, show_missing_warning: bool = True) -> None:
        try:
            if self.comic_worker and self.comic_worker.isRunning():
                self.comic_status_label.setText("漫画正在生成中。")
                self.append_log("漫画启动跳过：已有运行中的 comic worker")
                return
            if not self.last_run_dir:
                self.append_log("漫画启动失败：last_run_dir 为空")
                if show_missing_warning:
                    QMessageBox.warning(self, "缺少故事结果", "请先生成故事，再生成漫画。")
                return
            settings = self.current_settings()
            provider = self.provider_combo.currentText()
            image_provider = "ark" if provider == "aliyun" else "mock"
            missing = missing_for_comic(settings, provider=provider, image_provider=image_provider)
            if missing:
                self.comic_status_label.setText("漫画未开始，缺少：" + "，".join(missing))
                self.append_log("漫画未开始，缺少：" + "，".join(missing))
                if show_missing_warning:
                    QMessageBox.warning(self, "缺少 API 设置", "请先配置：" + "，".join(missing))
                return
            apply_api_settings(settings)

            self.generate_comic_button.setEnabled(False)
            self.open_comic_button.setEnabled(False)
            self.task_status["Comic"] = "running"
            self.task_started_at.setdefault("Comic", time.time())
            self.refresh_task_list()
            self.comic_status_label.setText("正在生成漫画...")
            self.current_step_label.setText("漫画：进行中")
            self.append_log("正在生成漫画...")
            self.append_log(
                f"漫画启动参数：provider={provider} image_provider={image_provider} "
                f"text_model={self.story_model} image_model={DESKTOP_DEFAULT_COMIC_IMAGE_MODEL}"
            )
            self.heartbeat_timer.start()

            self.append_log("创建 comic worker")
            self.comic_worker = ComicWorker(
                run_dir=self.last_run_dir,
                provider=provider,
                image_provider=image_provider,
                text_model=self.story_model,
                image_model=DESKTOP_DEFAULT_COMIC_IMAGE_MODEL,
            )
            self.comic_worker.started.connect(lambda: self.append_log("comic worker started"))
            self.comic_worker.progress.connect(self.on_comic_progress)
            self.comic_worker.finished.connect(self.on_comic_finished)
            self.comic_worker.failed.connect(self.on_comic_failed)
            self.append_log("调用 comic worker.start()")
            self.comic_worker.start()
            self.append_log(f"comic worker.start() 已返回，isRunning={self.comic_worker.isRunning()}")
        except Exception:
            error = traceback.format_exc()
            self.append_log("漫画启动异常：")
            self.append_log(error)
            _append_exception_logs(self.last_run_dir, error, comic=True)
            self.comic_status_label.setText("漫画启动失败")
            self.task_status["Comic"] = "failed"
            self.task_started_at.pop("Comic", None)
            self.refresh_task_list()
            self.generate_comic_button.setEnabled(True)
            if show_missing_warning:
                QMessageBox.critical(self, "漫画启动失败", error.splitlines()[-1] if error.splitlines() else error)

    def on_comic_progress(self, message: str) -> None:
        message = self._display_message(message)
        self.comic_status_label.setText(message)
        self.current_step_label.setText("漫画：进行中")
        self.refresh_task_list()
        self.append_log(message)

    def on_comic_finished(self, result: dict) -> None:
        self.last_comic_html = Path(result["daily_comic_html_path"])
        self.comic_status_label.setText(
            f"漫画已完成 · {result.get('panel_count')} 个分镜 · {result.get('reference_image_count')} 张参考图"
        )
        self.task_status["Comic"] = "done"
        self.task_started_at.pop("Comic", None)
        self.refresh_task_list()
        self.current_step_label.setText("漫画：完成")
        self._update_total_elapsed()
        self.append_log(f"漫画已输出：{self.last_comic_html.parent}")
        self.append_log(f"  - {self.last_comic_html.name}")
        if QWebEngineView is not None:
            self.comic_preview.load(QUrl.fromLocalFile(str(self.last_comic_html.resolve())))
        else:
            self.comic_preview.setText(self.last_comic_html.read_text(encoding="utf-8"))
        self.tabs.setCurrentIndex(1)
        self.generate_comic_button.setEnabled(True)
        self.open_comic_button.setEnabled(True)
        if self.highlight_worker and self.highlight_worker.isRunning():
            self.heartbeat_timer.start()

    def on_comic_failed(self, message: str) -> None:
        self.comic_status_label.setText("漫画生成失败")
        self.task_status["Comic"] = "failed"
        self.task_started_at.pop("Comic", None)
        self.refresh_task_list()
        self.current_step_label.setText("漫画：失败")
        self._update_total_elapsed()
        self.append_log("漫画生成失败，完整错误：")
        self.append_log(message)
        QMessageBox.critical(self, "漫画生成失败", message.splitlines()[-1] if message.splitlines() else message)
        self.generate_comic_button.setEnabled(True)

    def open_comic(self) -> None:
        if self.last_comic_html and self.last_comic_html.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_comic_html.resolve())))

    def generate_highlight_video(self, show_missing_warning: bool = True) -> None:
        try:
            if self.highlight_worker and self.highlight_worker.isRunning():
                self.highlight_status_label.setText("高光视频正在生成中。")
                self.append_log("高光视频启动跳过：已有运行中的 highlight worker")
                return
            if not self.last_run_dir:
                self.append_log("高光视频启动失败：last_run_dir 为空")
                if show_missing_warning:
                    QMessageBox.warning(self, "缺少故事结果", "请先生成故事，再生成高光视频。")
                return
            provider = self.provider_combo.currentText()
            if provider != "aliyun":
                message = "高光视频需要使用 aliyun 服务"
                self.highlight_status_label.setText(message)
                self.append_log(message)
                if show_missing_warning:
                    QMessageBox.warning(self, "服务不支持", message)
                return
            settings = self.current_settings()
            missing = missing_for_provider(settings, provider, use_amap=False)
            if missing:
                self.highlight_status_label.setText("高光视频未开始，缺少：" + "，".join(missing))
                self.append_log("高光视频未开始，缺少：" + "，".join(missing))
                if show_missing_warning:
                    QMessageBox.warning(self, "缺少 API 设置", "请先配置：" + "，".join(missing))
                return
            apply_api_settings(settings)

            if self.highlight_player is not None:
                self.highlight_player.stop()
            self.generate_highlight_button.setEnabled(False)
            self.play_highlight_button.setEnabled(False)
            self.open_highlight_button.setEnabled(False)
            self.task_status["Highlight Video"] = "running"
            self.task_started_at.setdefault("Highlight Video", time.time())
            self.refresh_task_list()
            self.highlight_status_label.setText("正在生成高光视频...")
            self.current_step_label.setText("高光视频：进行中")
            self.append_log("正在生成高光视频...")
            self.append_log(
                f"高光启动参数：provider={provider} text_model={self.story_model} "
                f"target_seconds={DESKTOP_DEFAULT_HIGHLIGHT_TARGET_SECONDS} "
                f"max_segments={DESKTOP_DEFAULT_HIGHLIGHT_MAX_SEGMENTS}"
            )
            self.heartbeat_timer.start()

            self.append_log("创建 highlight worker")
            self.highlight_worker = HighlightVideoWorker(
                run_dir=self.last_run_dir,
                provider=provider,
                text_model=self.story_model,
            )
            self.highlight_worker.started.connect(lambda: self.append_log("highlight worker started"))
            self.highlight_worker.progress.connect(self.on_highlight_progress)
            self.highlight_worker.finished.connect(self.on_highlight_finished)
            self.highlight_worker.failed.connect(self.on_highlight_failed)
            self.append_log("调用 highlight worker.start()")
            self.highlight_worker.start()
            self.append_log(f"highlight worker.start() 已返回，isRunning={self.highlight_worker.isRunning()}")
        except Exception:
            error = traceback.format_exc()
            self.append_log("高光视频启动异常：")
            self.append_log(error)
            _append_exception_logs(self.last_run_dir, error, highlight=True)
            self.highlight_status_label.setText("高光视频启动失败")
            self.task_status["Highlight Video"] = "failed"
            self.task_started_at.pop("Highlight Video", None)
            self.refresh_task_list()
            self.generate_highlight_button.setEnabled(True)
            if show_missing_warning:
                QMessageBox.critical(self, "高光视频启动失败", error.splitlines()[-1] if error.splitlines() else error)

    def on_highlight_progress(self, message: str) -> None:
        message = self._display_message(message)
        self.highlight_status_label.setText(message)
        self.current_step_label.setText("高光视频：进行中")
        self.refresh_task_list()
        self.append_log(message)

    def on_highlight_finished(self, result: dict) -> None:
        self.last_highlight_html = Path(result["highlight_video_html_path"])
        video_path = self.last_highlight_html.with_name("highlight_video.mp4")
        self.last_highlight_video = video_path if video_path.exists() else None
        self.highlight_status_label.setText(
            f"高光视频已完成 · {result.get('segment_count')} 段 · {result.get('estimated_duration_sec')}s"
        )
        self.task_status["Highlight Video"] = "done"
        self.task_started_at.pop("Highlight Video", None)
        self.refresh_task_list()
        self.current_step_label.setText("高光视频：完成")
        self._update_total_elapsed()
        self.append_log(f"高光视频已输出：{self.last_highlight_html.parent}")
        self.append_log(f"  - {self.last_highlight_html.name}")
        self.set_highlight_ready_preview(self.last_highlight_html, self.last_highlight_video)
        self.tabs.setCurrentIndex(2)
        self.generate_highlight_button.setEnabled(True)
        self.open_highlight_button.setEnabled(True)
        if self.comic_worker and self.comic_worker.isRunning():
            self.heartbeat_timer.start()

    def on_highlight_failed(self, message: str) -> None:
        self.highlight_status_label.setText("高光视频生成失败")
        self.task_status["Highlight Video"] = "failed"
        self.task_started_at.pop("Highlight Video", None)
        self.refresh_task_list()
        self.current_step_label.setText("高光视频：失败")
        self._update_total_elapsed()
        self.append_log("高光视频生成失败，完整错误：")
        self.append_log(message)
        QMessageBox.critical(self, "高光视频生成失败", message.splitlines()[-1] if message.splitlines() else message)
        self.generate_highlight_button.setEnabled(True)
        self.play_highlight_button.setEnabled(False)

    def open_highlight_video(self) -> None:
        highlight_html = self.last_highlight_html
        if (not highlight_html or not highlight_html.exists()) and self.last_run_dir:
            candidate = self.last_run_dir / "highlight_video" / "highlight_video.html"
            if candidate.exists():
                highlight_html = candidate
                self.last_highlight_html = candidate
        if highlight_html and highlight_html.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(highlight_html.resolve())))

    def toggle_highlight_playback(self) -> None:
        if self.highlight_player is None:
            return
        if self.last_highlight_video and self.last_highlight_video.exists():
            if self.highlight_player.source().isEmpty():
                self.highlight_player.setSource(QUrl.fromLocalFile(str(self.last_highlight_video.resolve())))
            self.highlight_preview.setCurrentWidget(self.highlight_video_container)
        if self.highlight_player.playbackState() == QMediaPlayer.PlayingState:
            self.highlight_player.stop()
        else:
            self.highlight_player.play()

    def closeEvent(self, event: QCloseEvent) -> None:
        running = [
            name
            for name, worker in [
                ("故事", self.worker),
                ("漫画", self.comic_worker),
                ("高光视频", self.highlight_worker),
            ]
            if worker and worker.isRunning()
        ]
        if not running:
            event.accept()
            return
        choice = QMessageBox.question(
            self,
            "任务仍在运行",
            "以下任务仍在运行：" + "，".join(running) + "。是否停止并退出？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            event.ignore()
            return
        self.stop_workers()
        event.accept()

    def stop_workers(self) -> None:
        for worker in [self.highlight_worker, self.comic_worker, self.worker]:
            if worker and worker.isRunning():
                worker.requestInterruption()
                worker.terminate()
                worker.wait(3000)

    def _ui_status(self, status: str) -> str:
        if status.startswith("running"):
            return "进行中"
        if status.startswith("done"):
            return "完成"
        if status.startswith("skipped"):
            return "已跳过"
        if status.startswith("failed"):
            return "失败"
        return "等待"

    def _display_message(self, message: str) -> str:
        replacements = {
            "Location": "理解定位",
            "Motion": "理解运动",
            "Audio Understanding": "理解音频",
            "Audio Summary": "整理音频",
            "Video Understanding": "理解视频",
            "Video Summary": "整理视频",
            "Final Story": "生成故事",
            "Comic": "漫画",
            "Highlight Video": "高光视频",
            "Story": "故事",
            "Starting": "准备开始",
            "running": "进行中",
            "done": "完成",
            "skipped": "已跳过",
            "failed": "失败",
            "building": "构建中",
            "calling": "调用模型中",
            "writing": "写入中",
            "Generating": "正在生成",
            "Generate": "生成",
            "Report written to": "故事已输出",
            "written to": "已输出",
        }
        display = message
        for source, target in replacements.items():
            display = display.replace(source, target)
        return display

    def _update_total_elapsed(self) -> None:
        if self.run_started_at is None:
            self.total_elapsed_label.setText("总耗时：0s")
            return
        elapsed = int(time.time() - self.run_started_at)
        self.total_elapsed_label.setText(f"总耗时：{elapsed}s")


def main() -> int:
    app = QApplication(sys.argv)
    window = PhoneLifeAgentWindow()
    app.aboutToQuit.connect(window.stop_workers)

    def handle_sigint(_signum: int, _frame: object) -> None:
        window.stop_workers()
        app.quit()

    signal.signal(signal.SIGINT, handle_sigint)
    window.show()
    try:
        return app.exec()
    finally:
        window.stop_workers()


if __name__ == "__main__":
    raise SystemExit(main())
