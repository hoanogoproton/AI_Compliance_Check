import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QLabel, QSizePolicy, QVBoxLayout, QWidget, QSlider, QHBoxLayout, QPushButton
)


class VideoPlayer(QWidget):
    frame_changed = Signal(int, object)

    def __init__(self, parent=None, show_video: bool = True):
        super().__init__(parent)
        self._video_path: str | None = None
        self._cap: cv2.VideoCapture | None = None
        self._total_frames = 0
        self._fps = 30.0
        self._current_frame = 0
        self._playing = False
        self._show_video = show_video

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(320, 240)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._label.setStyleSheet("background-color: black;")
        layout.addWidget(self._label)
        if not show_video:
            self._label.hide()

        controls = QHBoxLayout()
        controls.setSpacing(4)

        self._step_back_btn = QPushButton("\u23EE")
        self._step_back_btn.setToolTip("Previous frame")
        self._step_back_btn.clicked.connect(self.step_backward)
        controls.addWidget(self._step_back_btn)

        self._play_btn = QPushButton("\u25B6")
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self._play_btn)

        self._step_fwd_btn = QPushButton("\u23ED")
        self._step_fwd_btn.setToolTip("Next frame")
        self._step_fwd_btn.clicked.connect(self.step_forward)
        controls.addWidget(self._step_fwd_btn)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.valueChanged.connect(self._seek)
        controls.addWidget(self._slider, 1)

        self._frame_label = QLabel("")
        self._frame_label.setMinimumWidth(150)
        self._frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        controls.addWidget(self._frame_label)

        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

    def load(self, video_path: str):
        self.stop()
        if self._cap:
            self._cap.release()
        self._video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            self._label.setText(f"Could not open: {video_path}")
            return
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0
        self._slider.setMaximum(max(self._total_frames - 1, 0))
        self._current_frame = 0
        self._show_frame(0)

    def play(self):
        if self._cap is None or self._total_frames == 0:
            return
        self._playing = True
        self._play_btn.setText("\u23F8")
        interval = int(1000.0 / self._fps)
        self._timer.start(interval)

    def stop(self):
        self._playing = False
        self._timer.stop()
        self._play_btn.setText("\u25B6")

    def _toggle_play(self):
        if self._playing:
            self.stop()
        else:
            self.play()

    def is_playing(self) -> bool:
        return self._playing

    def current_frame(self) -> int:
        return self._current_frame

    def total_frames(self) -> int:
        return self._total_frames

    def fps(self) -> float:
        return self._fps

    def video_path(self) -> str | None:
        return self._video_path

    def step_forward(self):
        if self._cap is None:
            return
        self.stop()
        if self._current_frame < self._total_frames - 1:
            self._current_frame += 1
            self._show_frame(self._current_frame)

    def step_backward(self):
        if self._cap is None:
            return
        self.stop()
        if self._current_frame > 0:
            self._current_frame -= 1
            self._show_frame(self._current_frame)

    def _next_frame(self):
        if self._cap is None or not self._playing:
            return
        if self._current_frame >= self._total_frames - 1:
            self.stop()
            return
        self._current_frame += 1
        self._show_frame(self._current_frame)

    def _seek(self, frame_idx: int):
        if self._cap is None:
            return
        self._current_frame = frame_idx
        self._show_frame(frame_idx)

    def _show_frame(self, frame_idx: int):
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._show_video:
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qt_image)
            scaled = pixmap.scaled(
                self._label.width(), self._label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._label.setPixmap(scaled)

        self._slider.blockSignals(True)
        self._slider.setValue(frame_idx)
        self._slider.blockSignals(False)

        self._update_frame_label()
        self.frame_changed.emit(frame_idx, rgb)

    def _update_frame_label(self):
        secs = self._current_frame / self._fps if self._fps > 0 else 0
        mm = int(secs) // 60
        ss = int(secs) % 60
        self._frame_label.setText(
            f"Frame {self._current_frame} / {self._total_frames}   {mm:02d}:{ss:02d}"
        )

    def closeEvent(self, event):
        self.stop()
        if self._cap:
            self._cap.release()
        super().closeEvent(event)
