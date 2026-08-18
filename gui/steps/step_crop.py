import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QGroupBox, QFormLayout, QSizePolicy, QSplitter,
)

from gui.widgets.crop_canvas import CropCanvas


class StepCrop(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._video_path: str | None = None

        layout = QVBoxLayout(self)

        title = QLabel("Step 2: Define Crop Region")
        title.setObjectName("StepTitle")
        layout.addWidget(title)

        subtitle = QLabel("Drag on the video to define the crop region. All processing will use this cropped area.")
        subtitle.setObjectName("Subtitle")
        layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = CropCanvas()
        left_layout.addWidget(self.canvas, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        controls_group = QGroupBox("Crop Coordinates")
        controls_form = QFormLayout()
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 99999)
        self.x_spin.valueChanged.connect(self._spinbox_changed)
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 99999)
        self.y_spin.valueChanged.connect(self._spinbox_changed)
        self.w_spin = QSpinBox()
        self.w_spin.setRange(1, 99999)
        self.w_spin.valueChanged.connect(self._spinbox_changed)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(1, 99999)
        self.h_spin.valueChanged.connect(self._spinbox_changed)
        controls_form.addRow("X:", self.x_spin)
        controls_form.addRow("Y:", self.y_spin)
        controls_form.addRow("Width:", self.w_spin)
        controls_form.addRow("Height:", self.h_spin)
        controls_group.setLayout(controls_form)
        right_layout.addWidget(controls_group)

        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Crop")
        self.apply_btn.setObjectName("PrimaryButton")
        self.apply_btn.clicked.connect(self._apply_crop)
        self.clear_btn = QPushButton("Clear Crop")
        self.clear_btn.clicked.connect(self._clear_crop)
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.clear_btn)
        right_layout.addLayout(btn_layout)

        self.preview_label = QLabel()
        self.preview_label.setObjectName("PreviewLabel")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setText("Crop preview will appear here")
        self.preview_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
        right_layout.addWidget(self.preview_label, 1)

        splitter.addWidget(right)
        splitter.setSizes([500, 300])
        layout.addWidget(splitter, 1)

        self.canvas.crop_changed.connect(self._on_canvas_crop_changed)
        self._spinbox_block = False

    def load_video(self, video_path: str):
        self._video_path = video_path
        self.canvas.load_first_frame(video_path)
        existing = self.main_window.config_data.get("crop") if self.main_window.config_data else None
        if existing:
            x, y, w, h = existing
            self.canvas.set_crop_rect(x, y, w, h)
            self._update_spinboxes(x, y, w, h)
            self._update_preview(x, y, w, h)
        else:
            self.canvas.clear_crop()
            self._update_spinboxes(0, 0, 0, 0)
            self.preview_label.clear()
            self.preview_label.setText("No crop region defined")

    def _on_canvas_crop_changed(self, x: int, y: int, w: int, h: int):
        if w == 0 and h == 0:
            self._update_spinboxes(0, 0, 0, 0)
            self.preview_label.setText("No crop region defined")
        else:
            self._update_spinboxes(x, y, w, h)
            self._update_preview(x, y, w, h)

    def _update_spinboxes(self, x: int, y: int, w: int, h: int):
        self._spinbox_block = True
        self.x_spin.setValue(x)
        self.y_spin.setValue(y)
        self.w_spin.setValue(w)
        self.h_spin.setValue(h)
        self._spinbox_block = False

    def _spinbox_changed(self):
        if self._spinbox_block:
            return
        x = self.x_spin.value()
        y = self.y_spin.value()
        w = self.w_spin.value()
        h = self.h_spin.value()
        if w > 0 and h > 0:
            self.canvas.set_crop_rect(x, y, w, h)
            self._update_preview(x, y, w, h)

    def _update_preview(self, x: int, y: int, w: int, h: int):
        if not self._video_path:
            return
        cap = cv2.VideoCapture(self._video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        fh, fw = frame.shape[:2]
        x_end = min(x + w, fw)
        y_end = min(y + h, fh)
        if x_end > x and y_end > y:
            crop = frame[y:y_end, x:x_end]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h_c, w_c, ch = rgb.shape
            bytes_per_line = ch * w_c
            qt_image = QImage(rgb.data, w_c, h_c, bytes_per_line, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qt_image)
            scaled = pixmap.scaled(
                self.preview_label.width(), self.preview_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)

    def _apply_crop(self):
        x = self.x_spin.value()
        y = self.y_spin.value()
        w = self.w_spin.value()
        h = self.h_spin.value()
        if w > 0 and h > 0:
            if self.main_window.config_data is None:
                self.main_window.config_data = {}
            self.main_window.config_data["crop"] = [x, y, w, h]
            self.main_window.show_info("Crop Applied", f"Crop region set to [{x}, {y}, {w}, {h}]")

    def _clear_crop(self):
        self.canvas.clear_crop()
        self._update_spinboxes(0, 0, 0, 0)
        self.preview_label.clear()
        self.preview_label.setText("No crop region defined")
        if self.main_window.config_data:
            self.main_window.config_data.pop("crop", None)