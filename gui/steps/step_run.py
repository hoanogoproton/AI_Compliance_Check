import subprocess
import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QPlainTextEdit, QGroupBox
)

from gui.workers.pipeline_worker import PipelineWorker

ZONE_BEHAVIORS = ("leave_zone", "hand_in_zone")


class StepRun(QWidget):
    pipeline_finished = Signal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._worker: PipelineWorker | None = None
        self._last_output_dir: str | None = None

        layout = QVBoxLayout(self)

        title = QLabel("Step 4: Run Pipeline")
        title.setObjectName("StepTitle")
        layout.addWidget(title)

        self.preflight_group = QGroupBox("Pre-flight Check")
        preflight_layout = QVBoxLayout()
        self.check_labels: dict[str, QLabel] = {}
        for key in ["video", "config", "behaviors", "zones"]:
            lbl = QLabel()
            preflight_layout.addWidget(lbl)
            self.check_labels[key] = lbl
        self.preflight_group.setLayout(preflight_layout)
        layout.addWidget(self.preflight_group)

        run_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self._run_pipeline)
        self.run_btn.setMinimumHeight(44)
        run_layout.addWidget(self.run_btn)

        self.open_output_btn = QPushButton("Open Output Folder")
        self.open_output_btn.clicked.connect(self._open_output)
        self.open_output_btn.setEnabled(False)
        run_layout.addWidget(self.open_output_btn)
        layout.addLayout(run_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_group = QGroupBox("Log")
        self.log_group.setCheckable(True)
        self.log_group.setChecked(False)
        log_layout = QVBoxLayout()
        self.log_area = QPlainTextEdit()
        self.log_area.setObjectName("LogArea")
        self.log_area.setReadOnly(True)
        log_layout.addWidget(self.log_area)
        self.log_group.setLayout(log_layout)
        layout.addWidget(self.log_group, 1)

        self.refresh_preflight()

    def _compute_checks(self) -> dict:
        config = self.main_window.config_data
        video_ok = self.main_window.current_video is not None
        config_ok = config is not None
        behaviors_ok = bool(config) and any(
            b.get("enabled", False) for b in config.get("behaviors", [])
        )
        zones_ok = self._zones_ok(config)
        return {
            "video": video_ok,
            "config": config_ok,
            "behaviors": behaviors_ok,
            "zones": zones_ok,
        }

    def _zones_ok(self, config: dict | None) -> bool:
        if not config:
            return False
        zones = config.get("zones", {})
        for b in config.get("behaviors", []):
            if not b.get("enabled", False):
                continue
            if b.get("name") in ZONE_BEHAVIORS:
                zone = (b.get("params") or {}).get("zone")
                if not zone or zone not in zones:
                    return False
        return True

    def _preflight_passed(self, checks: dict) -> bool:
        return all(checks.values())

    def refresh_preflight(self):
        checks = self._compute_checks()
        labels = {
            "video": "Video selected",
            "config": "Config loaded",
            "behaviors": "At least one behavior enabled",
            "zones": "Zones defined (required for zone behaviors)",
        }
        for key, text in labels.items():
            ok = checks.get(key, False)
            lbl = self.check_labels[key]
            marker = "\u2713" if ok else "\u2717"
            lbl.setText(f"  {marker}  {text}")
            lbl.setObjectName("PreflightOk" if ok else "PreflightBad")
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

        self.run_btn.setEnabled(self._preflight_passed(checks))

    def _run_pipeline(self):
        self.refresh_preflight()
        if not self._preflight_passed(self._compute_checks()):
            self.main_window.show_error("Pre-flight Failed", "Resolve the checklist items before running.")
            return

        video = self.main_window.current_video
        config = self.main_window.config_data

        model_path = config.get("model", {}).get("path", "yolo11n-pose.pt")
        conf = config.get("model", {}).get("conf", 0.3)
        iou = config.get("model", {}).get("iou", 0.5)
        output_dir = config.get("output", {}).get("dir", "./outputs")
        visualize = config.get("output", {}).get("visualize", False)
        context_seconds = config.get("output", {}).get("context_seconds", 5)
        crop_padding = config.get("output", {}).get("crop_padding", 20)
        debug_keypoints = config.get("output", {}).get("debug_keypoints", False)

        self._last_output_dir = output_dir

        self.run_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.log_area.clear()
        self.log_group.setChecked(True)
        self.log("Pipeline starting...")

        self._worker = PipelineWorker()
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self.log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._worker.start(
            video_path=video,
            model_path=model_path,
            output_dir=output_dir,
            conf=conf,
            iou=iou,
            visualize=visualize,
            context_seconds=context_seconds,
            crop_padding=crop_padding,
            debug_keypoints=debug_keypoints,
            config_path=self.main_window.config_path,
        )

    def _on_progress(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self.progress_bar.setValue(pct)
        else:
            self.progress_bar.setRange(0, 0)

    def log(self, message: str):
        self.log_area.appendPlainText(message)

    def _on_finished(self, result: dict):
        self.log("Pipeline completed successfully.")
        self.progress_bar.setValue(100)
        self.open_output_btn.setEnabled(True)
        self._cleanup_worker()
        output_dir = result.get("output_dir", "")
        self.pipeline_finished.emit(output_dir)

    def _on_error(self, error_msg: str):
        self.log(f"ERROR: {error_msg}")
        self.main_window.show_error("Pipeline Error", error_msg)
        self._cleanup_worker()
        self.progress_bar.setValue(0)

    def _cleanup_worker(self):
        self.run_btn.setEnabled(True)
        self._worker = None

    def _open_output(self):
        if self._last_output_dir and Path(self._last_output_dir).exists():
            subprocess.Popen(["explorer", os.path.normpath(self._last_output_dir)])
