"""
Standalone PySide6 GUI for collecting keypoint-classifier training data.

Workflow:
    1. Load a video.
    2. Run YOLO-pose detection over a frame range (caches per-track keypoints).
    3. Scrub the video, pick a track, set a start/end frame range.
    4. Choose a behavior (or mark as a negative example), fill camera/session
       metadata, and "Add sample".
    5. Samples are written as `samples/sample_XXXX.npz` + appended to
       `metadata.csv` inside the chosen dataset directory.

Run:
    python dataset_tool.py
    python dataset_tool.py --model yolo26n-pose.pt --dataset ./dataset/
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt, QSize, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QProgressBar,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QHeaderView,
    QSplitter,
    QScrollArea,
    QSlider,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.config import KEYPOINT_CONFIDENCE_THRESHOLD
from detection.detector import TrackedPerson, process_frame
from gui.theme import apply_theme
from gui.widgets.video_player import VideoPlayer

FEATURE_VERSION = "v3.0.0"
DEFAULT_MODEL = "yolo26n-pose.pt"

BEHAVIOR_OPTIONS = [
    "move_1_step"
]

METADATA_COLUMNS = [
    "sample_id",
    "video_id",
    "camera_id",
    "recording_session",
    "annotator_id",
    "annotation_status",
    "track_id",
    "start_frame",
    "end_frame",
    "label",
    "behavior",
    "pose_model_version",
    "tracker_config_hash",
    "quality_score",
    "duration_sec",
    "fps",
]

SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


class DetectionWorker(QThread):
    """Runs YOLO-pose detection over a frame range in a background thread."""

    progress = Signal(int, int)
    frame_done = Signal(int, list)
    error = Signal(str)
    finished_ok = Signal()

    def __init__(
        self,
        model_path: str,
        video_path: str,
        conf: float,
        iou: float,
        start_frame: int,
        end_frame: int,
    ):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.conf = conf
        self.iou = iou
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from detection.model import load_pose_model

            model = load_pose_model(self.model_path)
        except Exception as exc:
            self.error.emit(f"Failed to load model: {exc}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"Could not open video: {self.video_path}")
            return

        total = max(1, self.end_frame - self.start_frame + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        frame_idx = self.start_frame
        count = 0

        while frame_idx <= self.end_frame and not self._cancel:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                people = process_frame(
                    model,
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                )
            except Exception as exc:
                self.error.emit(f"Detection error at frame {frame_idx}: {exc}")
                break

            safe_people = []
            for person in people:
                safe_people.append(
                    TrackedPerson(
                        track_id=person.track_id,
                        bbox=tuple(float(value) for value in person.bbox),
                        keypoints=np.array(
                            person.keypoints,
                            dtype=np.float32,
                            copy=True,
                        ),
                        conf=float(person.conf),
                    )
                )

            self.frame_done.emit(frame_idx, safe_people)

            count += 1
            self.progress.emit(count, total)
            frame_idx += 1

        cap.release()

        if not self._cancel:
            self.finished_ok.emit()


class DatasetToolWindow(QMainWindow):
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        dataset_dir: str = "./dataset/",
    ):
        super().__init__()

        self.setWindowTitle("Keypoint Classifier — Dataset Tool")
        self.resize(1400, 860)

        self.model_path = model_path
        self.dataset_dir = Path(dataset_dir)

        self.video_path: str | None = None
        self.fps = 30.0
        self.frame_w = 0
        self.frame_h = 0

        self.yolo_cache: dict[int, dict[int, dict]] = {}
        self.current_track: int | None = None
        self.range_start = 0
        self.range_end = 0

        self.worker: DetectionWorker | None = None
        self._lock = threading.Lock()

        # Frame RGB hiện tại, dùng để redraw khi zoom hoặc đổi track.
        self._current_rgb: np.ndarray | None = None
        self._current_frame_idx = -1

        # Pixmap gốc có overlay, trước khi scale theo zoom.
        self._overlay_pixmap: QPixmap | None = None
        self._zoom = 1.0

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)

        # ==========================================================
        # LEFT: Large video/overlay display + detection controls
        # ==========================================================
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.video_player = VideoPlayer(show_video=False)
        self.video_player.frame_changed.connect(self._on_frame_changed)
        self.video_player.setMaximumHeight(120)

        # Main image area. QScrollArea enables scrolling when zoomed.
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setStyleSheet("background-color: #161616;")

        self.image_scroll = QScrollArea()
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_scroll.setMinimumSize(640, 440)
        self.image_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #161616;
                border: 1px solid #333333;
            }
        """)

        # Main display gets most of the left panel's height.
        left_layout.addWidget(self.image_scroll, 8)

        # ---------------- Zoom controls ----------------
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)

        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedWidth(34)
        self.zoom_out_btn.setToolTip("Thu nhỏ")

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 400)
        self.zoom_slider.setSingleStep(10)
        self.zoom_slider.setPageStep(25)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setToolTip("Phóng to / thu nhỏ video và overlay")

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(34)
        self.zoom_in_btn.setToolTip("Phóng to")

        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.setToolTip("Đưa khung hình về kích thước vừa màn hình")

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.zoom_out_btn.clicked.connect(self._zoom_out)
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_fit_btn.clicked.connect(self._zoom_fit)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)

        zoom_row.addWidget(QLabel("Zoom"))
        zoom_row.addWidget(self.zoom_out_btn)
        zoom_row.addWidget(self.zoom_slider, 1)
        zoom_row.addWidget(self.zoom_in_btn)
        zoom_row.addWidget(self.zoom_fit_btn)
        zoom_row.addWidget(self.zoom_label)

        left_layout.addLayout(zoom_row)

        # VideoPlayer is only used for navigation/control.
        left_layout.addWidget(self.video_player, 0)

        # ---------------- Detection controls ----------------
        det_box = QGroupBox("Detection")
        det_form = QHBoxLayout(det_box)

        self.model_edit = QLineEdit(self.model_path)

        self.model_btn = QPushButton("Browse...")
        self.model_btn.clicked.connect(self._pick_model)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.4)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.2)

        det_form.addWidget(QLabel("Model:"))
        det_form.addWidget(self.model_edit, 1)
        det_form.addWidget(self.model_btn)
        det_form.addWidget(QLabel("conf"))
        det_form.addWidget(self.conf_spin)
        det_form.addWidget(QLabel("iou"))
        det_form.addWidget(self.iou_spin)

        left_layout.addWidget(det_box)

        det_run = QHBoxLayout()

        self.run_range_btn = QPushButton("Detect range")
        self.run_range_btn.setObjectName("PrimaryButton")
        self.run_range_btn.clicked.connect(self._run_detect_range)

        self.run_all_btn = QPushButton("Detect whole video")
        self.run_all_btn.clicked.connect(self._run_detect_all)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_detect)

        det_run.addWidget(self.run_range_btn)
        det_run.addWidget(self.run_all_btn)
        det_run.addWidget(self.stop_btn)

        left_layout.addLayout(det_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)

        splitter.addWidget(left)

        # ==========================================================
        # RIGHT: annotation panel
        # ==========================================================
        right = QWidget()
        right.setMinimumWidth(400)

        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # ---------------- Video / Dataset ----------------
        video_box = QGroupBox("Video / Dataset")
        vb = QVBoxLayout(video_box)

        vrow = QHBoxLayout()

        self.load_video_btn = QPushButton("Load video...")
        self.load_video_btn.clicked.connect(self._pick_video)

        self.video_label = QLabel("no video")

        vrow.addWidget(self.load_video_btn)
        vrow.addWidget(self.video_label, 1)
        vb.addLayout(vrow)

        drow = QHBoxLayout()

        self.dataset_edit = QLineEdit(str(self.dataset_dir))

        self.dataset_btn = QPushButton("Browse...")
        self.dataset_btn.clicked.connect(self._pick_dataset)

        drow.addWidget(QLabel("Dataset dir:"))
        drow.addWidget(self.dataset_edit, 1)
        drow.addWidget(self.dataset_btn)

        vb.addLayout(drow)
        right_layout.addWidget(video_box)

        # ---------------- Tracks ----------------
        track_box = QGroupBox("Tracks (at current frame)")
        tb = QVBoxLayout(track_box)

        self.track_list = QListWidget()
        self.track_list.currentItemChanged.connect(self._on_track_changed)
        tb.addWidget(self.track_list)

        self.track_info = QLabel("")
        self.track_info.setStyleSheet("color: #8a8a8a;")
        tb.addWidget(self.track_info)

        right_layout.addWidget(track_box)

        # ---------------- Sample range ----------------
        range_box = QGroupBox("Sample range")
        rf = QFormLayout(range_box)

        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 10_000_000)

        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, 10_000_000)

        self.start_set_btn = QPushButton("Set to current")
        self.start_set_btn.clicked.connect(
            lambda: self._set_range(self.start_spin)
        )

        self.end_set_btn = QPushButton("Set to current")
        self.end_set_btn.clicked.connect(
            lambda: self._set_range(self.end_spin)
        )

        rf.addRow("Start frame", self._row(self.start_spin, self.start_set_btn))
        rf.addRow("End frame", self._row(self.end_spin, self.end_set_btn))

        right_layout.addWidget(range_box)

        # ---------------- Annotation ----------------
        ann_box = QGroupBox("Annotation")
        af = QFormLayout(ann_box)

        self.behavior_combo = QComboBox()
        self.behavior_combo.addItems(BEHAVIOR_OPTIONS)

        self.negative_cb = QCheckBox("Negative example (label=0)")

        self.camera_edit = QLineEdit("unknown")
        self.session_edit = QLineEdit("default")
        self.annotator_edit = QLineEdit("default")

        af.addRow("Behavior", self.behavior_combo)
        af.addRow("", self.negative_cb)
        af.addRow("Camera ID", self.camera_edit)
        af.addRow("Recording session", self.session_edit)
        af.addRow("Annotator ID", self.annotator_edit)

        right_layout.addWidget(ann_box)

        self.add_btn = QPushButton("Add sample")
        self.add_btn.setObjectName("PrimaryButton")
        self.add_btn.clicked.connect(self._add_sample)

        right_layout.addWidget(self.add_btn)

        # ---------------- Existing samples ----------------
        samples_box = QGroupBox("Samples in dataset")
        sb = QVBoxLayout(samples_box)

        self.samples_table = QTableWidget(0, len(METADATA_COLUMNS))
        self.samples_table.setHorizontalHeaderLabels(METADATA_COLUMNS)
        self.samples_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

        sb.addWidget(self.samples_table)
        right_layout.addWidget(samples_box, 1)

        splitter.addWidget(right)

        # Video side gets more width.
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([900, 430])

        splitter.splitterMoved.connect(
            lambda *_: QTimer.singleShot(0, self._update_display_scale)
        )

        root.addWidget(splitter)
        self.setCentralWidget(central)

        self._refresh_samples_table()
        self._sync_ui()

    # ==========================================================
    # Generic helpers
    # ==========================================================
    @staticmethod
    def _row(*widgets) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        for item in widgets:
            layout.addWidget(item)

        layout.addStretch()
        return widget

    def _sync_ui(self):
        has_video = self.video_path is not None
        has_track = self.current_track is not None

        self.run_range_btn.setEnabled(has_video and self.worker is None)
        self.run_all_btn.setEnabled(has_video and self.worker is None)
        self.stop_btn.setEnabled(self.worker is not None)
        self.add_btn.setEnabled(has_video and has_track)

        self.zoom_out_btn.setEnabled(has_video)
        self.zoom_slider.setEnabled(has_video)
        self.zoom_in_btn.setEnabled(has_video)
        self.zoom_fit_btn.setEnabled(has_video)

    # ==========================================================
    # File pickers
    # ==========================================================
    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select pose model",
            "",
            "PyTorch (*.pt)",
        )

        if path:
            self.model_path = path
            self.model_edit.setText(path)

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )

        if path:
            self.load_video(path)

    def _pick_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select dataset directory",
            str(self.dataset_dir),
        )

        if path:
            self.dataset_dir = Path(path)
            self.dataset_edit.setText(path)
            self._refresh_samples_table()

    # ==========================================================
    # Video loading
    # ==========================================================
    def load_video(self, video_path: str):
        self.video_path = video_path
        self.video_label.setText(Path(video_path).name)

        self.video_player.load(video_path)

        self.fps = self.video_player.fps()
        total = self.video_player.total_frames()

        self.start_spin.setMaximum(max(total - 1, 0))
        self.end_spin.setMaximum(max(total - 1, 0))

        self.range_start = 0
        self.range_end = max(total - 1, 0)

        self.start_spin.setValue(0)
        self.end_spin.setValue(max(total - 1, 0))

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        with self._lock:
            self.yolo_cache.clear()

        self.current_track = None
        self._current_rgb = None
        self._current_frame_idx = -1
        self._overlay_pixmap = None

        self.track_list.clear()
        self.track_info.clear()
        self.image_label.clear()

        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(100)
        self.zoom_slider.blockSignals(False)

        self._zoom = 1.0
        self.zoom_label.setText("100%")

        self._sync_ui()

    # ==========================================================
    # Detection
    # ==========================================================
    def _run_detect_range(self):
        start = int(self.start_spin.value())
        end = int(self.end_spin.value())

        if end < start:
            QMessageBox.warning(
                self,
                "Range",
                "End frame must be >= start frame.",
            )
            return

        self._start_worker(start, end)

    def _run_detect_all(self):
        total = self.video_player.total_frames()
        self._start_worker(0, max(total - 1, 0))

    def _start_worker(self, start: int, end: int):
        if self.worker is not None:
            return

        if not self.video_path:
            return

        self.progress_bar.setValue(0)

        try:
            conf = float(self.conf_spin.value())
            iou = float(self.iou_spin.value())
        except Exception:
            conf, iou = 0.4, 0.2

        model_path = self.model_edit.text().strip() or DEFAULT_MODEL

        self.worker = DetectionWorker(
            model_path=model_path,
            video_path=self.video_path,
            conf=conf,
            iou=iou,
            start_frame=start,
            end_frame=end,
        )

        self.worker.progress.connect(self._on_detect_progress)
        self.worker.frame_done.connect(self._on_detect_frame)
        self.worker.error.connect(self._on_detect_error)
        self.worker.finished_ok.connect(self._on_detect_finished)
        self.worker.finished.connect(self._on_thread_finished)

        self._sync_ui()
        self.worker.start()

    def _stop_detect(self):
        if self.worker is not None:
            self.worker.cancel()

    def _on_detect_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))

    def _on_detect_frame(self, frame_idx: int, people: list):
        track_map: dict[int, dict] = {}

        for person in people:
            track_map[person.track_id] = {
                "keypoints": person.keypoints,
                "bbox": np.array(person.bbox, dtype=np.float32),
                "conf": person.conf,
            }

        with self._lock:
            self.yolo_cache[frame_idx] = track_map

        current_frame = self.video_player.current_frame()

        if frame_idx == current_frame:
            self._refresh_track_list()
            self._redraw_overlay(current_frame)

    def _on_detect_error(self, message: str):
        QMessageBox.critical(self, "Detection error", message)

    def _on_detect_finished(self):
        self.progress_bar.setValue(100)

    def _on_thread_finished(self):
        self.worker = None
        self._sync_ui()

    # ==========================================================
    # Frame navigation and overlay drawing
    # ==========================================================
    def _on_frame_changed(self, frame_idx: int, rgb: np.ndarray):
        self._current_frame_idx = int(frame_idx)
        self._current_rgb = np.ascontiguousarray(rgb.copy())

        self._refresh_track_list()
        self._redraw_overlay(frame_idx)

    def _refresh_track_list(self):
        current_frame = self.video_player.current_frame()

        with self._lock:
            tracks = sorted(self.yolo_cache.get(current_frame, {}).keys())

        previous_track = self.current_track

        self.track_list.blockSignals(True)
        self.track_list.clear()

        for track_id in tracks:
            self.track_list.addItem(str(track_id))

        if previous_track in tracks:
            self.track_list.setCurrentRow(tracks.index(previous_track))
        else:
            self.current_track = None

        self.track_list.blockSignals(False)

        if self.current_track is None and tracks:
            self.current_track = tracks[0]
            self.track_list.setCurrentRow(0)

        self._update_track_info()
        self._sync_ui()

    def _on_track_changed(self, current_item, _previous_item):
        if current_item is None:
            self.current_track = None
        else:
            try:
                self.current_track = int(current_item.text())
            except ValueError:
                self.current_track = None

        self._update_track_info()
        self._redraw_overlay(self.video_player.current_frame())
        self._sync_ui()

    def _update_track_info(self):
        current_frame = self.video_player.current_frame()
        text = ""

        if self.current_track is not None:
            with self._lock:
                data = self.yolo_cache.get(
                    current_frame,
                    {},
                ).get(self.current_track)

            if data is not None:
                bbox = data["bbox"]
                text = (
                    f"track {self.current_track}  "
                    f"conf={data['conf']:.2f}  "
                    f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},"
                    f"{bbox[2]:.0f},{bbox[3]:.0f})"
                )

        self.track_info.setText(text)

    def _redraw_overlay(self, frame_idx: int, rgb: np.ndarray | None = None):
        """
        Draw bbox and keypoints onto the current frame.

        The latest frame is retained in self._current_rgb so changing the
        selected track or moving the zoom slider redraws immediately.
        """
        if rgb is not None:
            self._current_rgb = np.ascontiguousarray(rgb.copy())
            self._current_frame_idx = int(frame_idx)

        if self._current_rgb is None:
            return

        if int(frame_idx) != self._current_frame_idx:
            return

        overlay = self._current_rgb.copy()

        with self._lock:
            frame_tracks = self.yolo_cache.get(frame_idx, {})

        for track_id, data in frame_tracks.items():
            keypoints = data["keypoints"]
            selected = track_id == self.current_track

            color = (0, 255, 0) if selected else (255, 165, 0)

            self._draw_track(
                overlay,
                keypoints,
                data["bbox"],
                color,
                track_id,
                selected,
            )

        height, width, channels = overlay.shape

        qimage = QImage(
            overlay.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()

        self._overlay_pixmap = QPixmap.fromImage(qimage)
        self._update_display_scale()

    @staticmethod
    def _draw_track(
        image: np.ndarray,
        keypoints: np.ndarray,
        bbox: np.ndarray,
        color: tuple[int, int, int],
        track_id: int,
        selected: bool,
    ):
        x1, y1, x2, y2 = [int(value) for value in bbox]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2 if selected else 1,
        )

        cv2.putText(
            image,
            f"id={track_id}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        for i, j in SKELETON_CONNECTIONS:
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]

            if (
                ci > KEYPOINT_CONFIDENCE_THRESHOLD
                and cj > KEYPOINT_CONFIDENCE_THRESHOLD
            ):
                cv2.line(
                    image,
                    (int(xi), int(yi)),
                    (int(xj), int(yj)),
                    (255, 255, 255),
                    1,
                )

        for index in range(keypoints.shape[0]):
            x, y, confidence = keypoints[index]

            if confidence <= KEYPOINT_CONFIDENCE_THRESHOLD:
                continue

            if index in (9, 10):
                point_color = (0, 165, 255)
            elif index in (0, 1, 2, 3, 4):
                point_color = (255, 0, 0)
            else:
                point_color = color

            cv2.circle(
                image,
                (int(x), int(y)),
                3,
                point_color,
                -1,
            )

    # ==========================================================
    # Zoom
    # ==========================================================
    def _zoom_out(self):
        self.zoom_slider.setValue(
            max(
                self.zoom_slider.minimum(),
                self.zoom_slider.value() - 25,
            )
        )

    def _zoom_in(self):
        self.zoom_slider.setValue(
            min(
                self.zoom_slider.maximum(),
                self.zoom_slider.value() + 25,
            )
        )

    def _zoom_fit(self):
        self.zoom_slider.setValue(100)

    def _on_zoom_changed(self, value: int):
        self._zoom = value / 100.0
        self.zoom_label.setText(f"{value}%")
        self._update_display_scale()

    def _update_display_scale(self):
        """
        Scale overlay image.

        At 100%, the image is fitted inside the available viewport.
        Above 100%, the image grows and QScrollArea provides scrollbars.
        """
        if self._overlay_pixmap is None or self._overlay_pixmap.isNull():
            return

        viewport_size = self.image_scroll.viewport().size()
        source_size = self._overlay_pixmap.size()

        if viewport_size.width() <= 1 or viewport_size.height() <= 1:
            return

        if source_size.width() <= 0 or source_size.height() <= 0:
            return

        fit_scale = min(
            viewport_size.width() / source_size.width(),
            viewport_size.height() / source_size.height(),
        )

        scale = max(0.01, fit_scale * self._zoom)

        target_size = QSize(
            max(1, int(source_size.width() * scale)),
            max(1, int(source_size.height() * scale)),
        )

        scaled_pixmap = self._overlay_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

    # ==========================================================
    # Frame range
    # ==========================================================
    def _set_range(self, spin: QSpinBox):
        spin.setValue(self.video_player.current_frame())

        if spin is self.start_spin:
            self.range_start = int(spin.value())
        else:
            self.range_end = int(spin.value())

    # ==========================================================
    # Sample writing
    # ==========================================================
    def _next_sample_id(self) -> int:
        existing = set()

        samples_dir = self.dataset_dir / "samples"
        if samples_dir.exists():
            for path in samples_dir.glob("sample_*.npz"):
                try:
                    existing.add(int(path.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue

        metadata_path = self.dataset_dir / "metadata.csv"
        if metadata_path.exists():
            with open(metadata_path, "r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)

                for row in reader:
                    try:
                        existing.add(int(row.get("sample_id", "")))
                    except ValueError:
                        continue

        if not existing:
            return 0

        return max(existing) + 1

    def _add_sample(self):
        if not self.video_path:
            QMessageBox.warning(
                self,
                "No video",
                "Load a video first.",
            )
            return

        if self.current_track is None:
            QMessageBox.warning(
                self,
                "No track",
                "Select a track first.",
            )
            return

        start = int(self.start_spin.value())
        end = int(self.end_spin.value())

        if end < start:
            QMessageBox.warning(
                self,
                "Range",
                "End frame must be >= start frame.",
            )
            return

        if end - start + 1 < 2:
            QMessageBox.warning(
                self,
                "Range",
                "Select at least 2 frames.",
            )
            return

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "samples").mkdir(parents=True, exist_ok=True)

        track_id = self.current_track
        behavior = self.behavior_combo.currentText()
        is_negative = self.negative_cb.isChecked()

        fps = self.fps if self.fps > 0 else 30.0

        keypoint_sequence = []
        bbox_sequence = []
        frame_index_sequence = []
        timestamp_sequence = []
        valid_sequence = []
        track_confidence_sequence = []
        pose_confidence_sequence = []

        for frame_idx in range(start, end + 1):
            with self._lock:
                cache = self.yolo_cache.get(
                    frame_idx,
                    {},
                ).get(track_id)

            if cache is not None:
                keypoints = np.asarray(
                    cache["keypoints"],
                    dtype=np.float32,
                )

                keypoint_sequence.append(keypoints)
                bbox_sequence.append(
                    np.asarray(cache["bbox"], dtype=np.float32)
                )
                valid_sequence.append(True)
                track_confidence_sequence.append(float(cache["conf"]))
                pose_confidence_sequence.append(
                    float(np.mean(keypoints[:, 2]))
                )
            else:
                keypoint_sequence.append(
                    np.zeros((17, 3), dtype=np.float32)
                )
                bbox_sequence.append(
                    np.array([0, 0, 0, 0], dtype=np.float32)
                )
                valid_sequence.append(False)
                track_confidence_sequence.append(0.0)
                pose_confidence_sequence.append(0.0)

            frame_index_sequence.append(frame_idx)
            timestamp_sequence.append(frame_idx / fps)

        if not any(valid_sequence):
            QMessageBox.warning(
                self,
                "No detection",
                "No cached detections for this track in the range. "
                "Run detection over the range first.",
            )
            return

        keypoints_arr = np.stack(keypoint_sequence, axis=0)
        bboxes_arr = np.stack(bbox_sequence, axis=0)
        frame_indices_arr = np.array(frame_index_sequence)
        timestamps_arr = np.array(timestamp_sequence, dtype=np.float32)
        valid_mask_arr = np.array(valid_sequence, dtype=bool)
        track_conf_arr = np.array(
            track_confidence_sequence,
            dtype=np.float32,
        )
        pose_conf_arr = np.array(
            pose_confidence_sequence,
            dtype=np.float32,
        )

        sample_id = self._next_sample_id()

        video_stem = Path(self.video_path).stem
        video_id = (
            f"{video_stem}_"
            f"{abs(hash(self.video_path)) & 0xFFFFFFFF:08x}"
        )

        camera_id = self.camera_edit.text().strip() or "unknown"
        recording_session = self.session_edit.text().strip() or "default"
        annotator_id = self.annotator_edit.text().strip() or "default"

        pose_model_version = Path(
            self.model_edit.text().strip() or DEFAULT_MODEL
        ).stem

        quality_score = float(np.mean(valid_mask_arr))

        sample_path = (
            self.dataset_dir
            / "samples"
            / f"sample_{sample_id:04d}.npz"
        )

        np.savez_compressed(
            sample_path,
            keypoints=keypoints_arr,
            bboxes=bboxes_arr,
            frame_indices=frame_indices_arr,
            timestamps=timestamps_arr,
            frame_size=np.array([self.frame_w, self.frame_h]),
            valid_mask=valid_mask_arr,
            track_confidence=track_conf_arr,
            pose_confidence=pose_conf_arr,
            label=0 if is_negative else 1,
            behavior=behavior,
            video_id=video_id,
            camera_id=camera_id,
            track_id=track_id,
            start_frame=start,
            end_frame=end,
            fps=fps,
            duration_sec=(end - start + 1) / fps,
            pose_model_version=pose_model_version,
            tracker_version="bytetrack",
            feature_version=FEATURE_VERSION,
        )

        annotation_status = (
            "valid" if quality_score >= 0.5 else "uncertain"
        )

        metadata_path = self.dataset_dir / "metadata.csv"
        write_header = not metadata_path.exists()

        with open(
            metadata_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            if write_header:
                writer.writerow(METADATA_COLUMNS)

            writer.writerow(
                [
                    sample_id,
                    video_id,
                    camera_id,
                    recording_session,
                    annotator_id,
                    annotation_status,
                    track_id,
                    start,
                    end,
                    0 if is_negative else 1,
                    behavior,
                    pose_model_version,
                    "bytetrack",
                    quality_score,
                    (end - start + 1) / fps,
                    fps,
                ]
            )

        sample_kind = "negative" if is_negative else "positive"

        self.statusBar().showMessage(
            f"Saved sample_{sample_id:04d} "
            f"(track {track_id}, {behavior}, {sample_kind}, "
            f"frames {start}-{end})",
            8000,
        )

        self._refresh_samples_table()

    # ==========================================================
    # Metadata table
    # ==========================================================
    def _refresh_samples_table(self):
        self.samples_table.setRowCount(0)

        metadata_path = self.dataset_dir / "metadata.csv"
        if not metadata_path.exists():
            return

        try:
            with open(
                metadata_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                reader = csv.DictReader(file)
                rows = list(reader)
        except OSError:
            return

        self.samples_table.setRowCount(len(rows))

        for row_index, row in enumerate(rows):
            for column_index, column_name in enumerate(METADATA_COLUMNS):
                value = str(row.get(column_name, ""))
                self.samples_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(value),
                )

    # ==========================================================
    # Qt events
    # ==========================================================
    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "image_scroll"):
            QTimer.singleShot(0, self._update_display_scale)

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(3000)

        super().closeEvent(event)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keypoint Classifier Dataset Tool"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to YOLO pose model (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--dataset",
        default="./dataset/",
        help="Dataset output directory (default: ./dataset/)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Keypoint Dataset Tool")
    app.setOrganizationName("HandHead")

    apply_theme(app)

    window = DatasetToolWindow(
        model_path=args.model,
        dataset_dir=args.dataset,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()