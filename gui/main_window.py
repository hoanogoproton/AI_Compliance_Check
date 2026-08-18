from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QPushButton, QMessageBox
)

from gui.widgets.stepper import Stepper, STEP_NAMES
from gui.steps.step_video import StepVideo
from gui.steps.step_config import StepConfig
from gui.steps.step_zones import StepZones
from gui.steps.step_run import StepRun
from gui.steps.step_results import StepResults


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand-to-Head Detection")
        self.resize(1280, 860)

        self.settings = QSettings("HandHead", "HandHeadDetection")

        self.current_video: str | None = None
        self.config_path: str = str(Path("config.yaml").resolve())
        self.config_data: dict | None = None
        self.current_step: int = 0

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.stepper = Stepper()
        root.addWidget(self.stepper)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 16, 20, 0)
        content_layout.setSpacing(12)

        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack, 1)

        nav = QHBoxLayout()
        self.back_btn = QPushButton("\u2190 Back")
        self.back_btn.setObjectName("NavButton")
        self.next_btn = QPushButton("Next \u2192")
        self.next_btn.setObjectName("PrimaryButton")
        nav.addWidget(self.back_btn)
        nav.addStretch()
        nav.addWidget(self.next_btn)
        content_layout.addLayout(nav)

        root.addWidget(content, 1)
        self.setCentralWidget(central)

        self.step_video = StepVideo(self)
        self.step_config = StepConfig(self)
        self.step_zones = StepZones(self)
        self.step_run = StepRun(self)
        self.step_results = StepResults(self)

        for w in [self.step_video, self.step_config, self.step_zones, self.step_run, self.step_results]:
            self.stack.addWidget(w)

        self.stepper.step_selected.connect(self._on_step_selected)
        self.back_btn.clicked.connect(self._go_back)
        self.next_btn.clicked.connect(self._go_next)

        self.step_video.video_selected.connect(self._on_video_selected)
        self.step_run.pipeline_finished.connect(self._on_pipeline_finished)

        self._sync_ui()

    # --- navigation ---
    def _on_step_selected(self, index: int):
        self._navigate_to(index)

    def _navigate_to(self, index: int):
        if index == self.current_step:
            return
        if index < self.current_step:
            self._set_step(index)
            return
        for i in range(self.current_step, index):
            if not self._validate_step(i):
                return
        self._set_step(index)

    def _set_step(self, index: int):
        self.current_step = index
        self.stepper.set_active(index)
        self.stack.setCurrentIndex(index)
        if index == 2 and self.current_video:
            self.step_zones.load_video(self.current_video)
        if index == 3:
            self.step_run.refresh_preflight()
        self._sync_ui()

    def _go_next(self):
        self._navigate_to(self.current_step + 1)

    def _go_back(self):
        self._navigate_to(max(0, self.current_step - 1))

    def _validate_step(self, index: int) -> bool:
        if index == 0:
            if not self.current_video:
                self.show_error("Validation", "Please select a video first.")
                return False
        elif index == 1:
            if not self._has_enabled_behavior():
                self.show_error("Validation", "Please configure and enable at least one behavior.")
                return False
        elif index == 3:
            if not self.current_video:
                self.show_error("Validation", "Please select a video first.")
                return False
            if not self._has_enabled_behavior():
                self.show_error("Validation", "Please enable at least one behavior before running.")
                return False
        return True

    def _has_enabled_behavior(self) -> bool:
        if not self.config_data:
            return False
        behaviors = self.config_data.get("behaviors", [])
        return any(b.get("enabled", False) for b in behaviors)

    # --- events ---
    def _on_video_selected(self, path: str):
        self.current_video = path
        self._sync_ui()

    def _on_pipeline_finished(self, output_dir: str):
        self.stepper.set_completed(3, True)
        self.step_results.load_results(output_dir)
        self.stepper.set_completed(4, True)
        self._set_step(4)

    def on_zones_changed(self):
        self.step_config.refresh_zones()
        self._sync_ui()

    def _sync_ui(self):
        self.stepper.set_completed(0, self.current_video is not None)
        self.stepper.set_completed(1, self._has_enabled_behavior())
        zones = (self.config_data or {}).get("zones", {})
        self.stepper.set_completed(2, bool(zones))

        self.back_btn.setEnabled(self.current_step > 0)
        self.next_btn.setEnabled(self.current_step < len(STEP_NAMES) - 1)

        video_name = Path(self.current_video).name if self.current_video else "no video"
        config_name = Path(self.config_path).name if self.config_path else "no config"
        self.statusBar().showMessage(f"Video: {video_name}   |   Config: {config_name}")

    def show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str):
        QMessageBox.information(self, title, message)
