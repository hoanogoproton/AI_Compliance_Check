import csv
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel,
    QFileDialog, QComboBox, QSizePolicy, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView
)


class StepVideo(QWidget):
    video_selected = Signal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.settings = QSettings("HandHead", "HandHeadDetection")

        layout = QVBoxLayout(self)

        title = QLabel("Step 1: Select Video")
        title.setObjectName("StepTitle")
        layout.addWidget(title)

        subtitle = QLabel("Choose a surveillance video to analyze.")
        subtitle.setObjectName("Subtitle")
        layout.addWidget(subtitle)

        file_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a video file...")
        self.path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_video)
        file_layout.addWidget(self.path_edit)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        csv_layout = QHBoxLayout()
        self.csv_load_btn = QPushButton("Load CSV")
        self.csv_load_btn.clicked.connect(self._load_csv)
        self.csv_entry_count = QLabel("")
        csv_layout.addWidget(self.csv_load_btn)
        csv_layout.addWidget(self.csv_entry_count)
        csv_layout.addStretch()
        layout.addLayout(csv_layout)

        self.csv_table = QTableWidget()
        self.csv_table.setColumnCount(3)
        self.csv_table.setHorizontalHeaderLabels(["Video", "Config", "Output"])
        self.csv_table.horizontalHeader().setStretchLastSection(True)
        self.csv_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.csv_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.csv_table.setSelectionMode(QTableWidget.SingleSelection)
        self.csv_table.setVisible(False)
        self.csv_table.itemSelectionChanged.connect(self._csv_row_selected)
        layout.addWidget(self.csv_table)

        recent_layout = QHBoxLayout()
        recent_layout.addWidget(QLabel("Recent:"))
        self.recent_combo = QComboBox()
        self.recent_combo.setMinimumWidth(300)
        self.recent_combo.currentIndexChanged.connect(self._recent_selected)
        recent_layout.addWidget(self.recent_combo, 1)
        layout.addLayout(recent_layout)

        info_group = QGroupBox("Video Info")
        info_form = QFormLayout()
        self.duration_label = QLabel("-")
        self.fps_label = QLabel("-")
        self.resolution_label = QLabel("-")
        self.codec_label = QLabel("-")
        info_form.addRow("Duration:", self.duration_label)
        info_form.addRow("FPS:", self.fps_label)
        info_form.addRow("Resolution:", self.resolution_label)
        info_form.addRow("Codec:", self.codec_label)
        info_group.setLayout(info_form)
        layout.addWidget(info_group)

        self.thumbnail = QLabel()
        self.thumbnail.setObjectName("Thumbnail")
        self.thumbnail.setAlignment(Qt.AlignCenter)
        self.thumbnail.setMinimumSize(480, 320)
        self.thumbnail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.thumbnail.setText("No video selected")
        layout.addWidget(self.thumbnail, 1)

        self._load_recent()

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if path:
            self._set_video(path)

    def _set_video(self, path: str):
        self.path_edit.setText(path)
        self._add_recent(path)
        self._show_thumbnail(path)
        self._update_info(path)
        self.video_selected.emit(path)

    def _recent_selected(self, index: int):
        if index < 0:
            return
        path = self.recent_combo.currentData()
        if path and Path(path).exists():
            self._set_video(path)

    def _update_info(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()

        if fps <= 0:
            fps = 30.0
        seconds = total / fps if fps else 0
        hh = int(seconds) // 3600
        mm = (int(seconds) % 3600) // 60
        ss = int(seconds) % 60
        self.duration_label.setText(f"{hh:02d}:{mm:02d}:{ss:02d}")
        self.fps_label.setText(f"{fps:.2f}")
        self.resolution_label.setText(f"{w} x {h}")
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
        self.codec_label.setText(codec or "-")

    def _show_thumbnail(self, path: str):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.thumbnail.setText("Could not read video")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.thumbnail.width(), self.thumbnail.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.thumbnail.setPixmap(scaled)

    def _add_recent(self, path: str):
        recent = self.settings.value("recent_videos", [])
        if not isinstance(recent, list):
            recent = []
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:10]
        self.settings.setValue("recent_videos", recent)
        self._load_recent()

    def _load_recent(self):
        self.recent_combo.blockSignals(True)
        self.recent_combo.clear()
        recent = self.settings.value("recent_videos", [])
        if isinstance(recent, list):
            for path in recent:
                if Path(path).exists():
                    self.recent_combo.addItem(Path(path).name, path)
        self.recent_combo.blockSignals(False)

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV Batch", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self._parse_csv(path)

    def _parse_csv(self, path: str):
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != ["video", "config", "output_dir"]:
                    self.main_window.show_error(
                        "CSV Error",
                        "CSV must have exactly these columns: video,config,output_dir"
                    )
                    return
                rows = []
                for row in reader:
                    video = (row.get("video") or "").strip()
                    config = (row.get("config") or "").strip()
                    output_dir = (row.get("output_dir") or "").strip()
                    if not video or not config:
                        continue
                    rows.append({"video": video, "config": config, "output_dir": output_dir})
                if not rows:
                    self.main_window.show_error("CSV Error", "CSV file is empty or has no valid rows.")
                    return
        except Exception as e:
            self.main_window.show_error("CSV Error", str(e))
            return

        self.main_window.csv_data = rows
        self.main_window.csv_path = path

        self.csv_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.csv_table.setItem(i, 0, QTableWidgetItem(row["video"]))
            self.csv_table.setItem(i, 1, QTableWidgetItem(row["config"]))
            self.csv_table.setItem(i, 2, QTableWidgetItem(row["output_dir"]))
        self.csv_table.setVisible(True)
        self.csv_entry_count.setText(f"{len(rows)} entries loaded")
        self.csv_load_btn.setText("Reload CSV")

    def _csv_row_selected(self):
        selected = self.csv_table.selectedItems()
        if not selected:
            return
        row_idx = selected[0].row()
        if row_idx >= len(self.main_window.csv_data or []):
            return
        row = self.main_window.csv_data[row_idx]
        video_path = row["video"]
        config_path = row["config"]
        output_dir = row.get("output_dir") or None

        mw = self.main_window
        mw._selecting_from_csv = True
        try:
            mw.load_config_from_csv(config_path, output_dir)
            self._set_video(video_path)
        except Exception as e:
            mw.show_error("CSV Row Error", str(e))
        finally:
            mw._selecting_from_csv = False
