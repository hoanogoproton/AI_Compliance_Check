from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QPushButton, QMessageBox
)

from gui.widgets.stepper import Stepper, STEP_NAMES
from gui.steps.step_video import StepVideo
from gui.steps.step_crop import StepCrop
from gui.steps.step_config import StepConfig
from gui.steps.step_zones import StepZones
from gui.steps.step_run import StepRun
from gui.steps.step_results import StepResults
from detection.config_loader import load_config


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI compliance check")
        self.resize(1280, 860)

        self.settings = QSettings("HandHead", "AI compliance check")

        self.current_video: str | None = None
        self.config_path: str = str(Path("config.yaml").resolve())
        self.config_data: dict | None = None
        self.current_step: int = 0

        self.csv_data: list[dict] | None = None
        self.csv_path: str | None = None
        self.csv_output_dir: str | None = None
        self._selecting_from_csv: bool = False

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
        self.step_crop = StepCrop(self)
        self.step_config = StepConfig(self)
        self.step_zones = StepZones(self)
        self.step_run = StepRun(self)
        self.step_results = StepResults(self)

        for w in [self.step_video, self.step_crop, self.step_config, self.step_zones, self.step_run, self.step_results]:
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
        if index == 1 and self.current_video:
            self.step_crop.load_video(self.current_video)
        if index == 3 and self.current_video:
            self.step_zones.load_video(self.current_video)
        if index == 4:
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
        elif index == 2:
            if not self._has_enabled_behavior():
                self.show_error("Validation", "Please configure and enable at least one behavior.")
                return False
        elif index == 4:
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

    def load_config_from_csv(self, config_path: str, output_dir: str | None = None):
        config = load_config(config_path)
        if output_dir:
            if "output" not in config:
                config["output"] = {}
            config["output"]["dir"] = output_dir
        self.config_path = config_path
        self.config_data = config
        self.csv_output_dir = output_dir or config.get("output", {}).get("dir", "./outputs")
        self.step_config._populate_from_config(config)
        self.step_config.default_badge.setVisible(False)
        self.stepper.set_completed(2, self._has_enabled_behavior())
        self._sync_ui()

    # --- events ---
    def _on_video_selected(self, path: str):
        self.current_video = path
        if not self._selecting_from_csv:
            self._sync_ui()

    def _on_pipeline_finished(self, output_dir: str):
        self.stepper.set_completed(4, True)
        self.step_results.load_results(output_dir)
        self.stepper.set_completed(5, True)
        self._set_step(5)

    def on_zones_changed(self):
        self.step_config.refresh_zones()
        self._sync_ui()

    def _sync_ui(self):
        self.stepper.set_completed(0, self.current_video is not None)
        self.stepper.set_completed(1, True)
        self.stepper.set_completed(2, self._has_enabled_behavior())
        zones = (self.config_data or {}).get("zones", {})
        self.stepper.set_completed(3, bool(zones))

        self.back_btn.setEnabled(self.current_step > 0)
        self.next_btn.setEnabled(self.current_step < len(STEP_NAMES) - 1)

        video_name = Path(self.current_video).name if self.current_video else "no video"
        config_name = Path(self.config_path).name if self.config_path else "no config"
        parts = [f"Video: {video_name}", f"Config: {config_name}"]
        if self.csv_data is not None:
            parts.append(f"CSV: {Path(self.csv_path).name if self.csv_path else ''} ({len(self.csv_data)} entries)")
        self.statusBar().showMessage("   |   ".join(parts))

    def show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str):
        QMessageBox.information(self, title, message)
