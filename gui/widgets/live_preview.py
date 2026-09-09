"""Live realtime preview widget for frames pushed by the detection pipeline.

The pipeline (see ``detection/pipeline.py``) emits throttled annotated frames
(bounding boxes, keypoints, face landmarks, zones) as RGB numpy arrays; this
widget converts them to a QImage/QPixmap and scales them to fit while keeping
the aspect ratio. It re-renders the last frame when resized.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class LivePreviewWidget(QWidget):
    """Displays RGB numpy frames (HxWx3, uint8) pushed by the pipeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(320, 220)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._label.setStyleSheet("background-color: black; color: #9aa0a6;")
        self._label.setText("Chưa có video đang xử lý")
        layout.addWidget(self._label)

        self._last_image: QImage | None = None

    def set_frame(self, frame_rgb: np.ndarray) -> None:
        """Render one RGB frame; the buffer is copied so callers may reuse it."""
        if frame_rgb is None or frame_rgb.ndim != 3 or frame_rgb.size == 0:
            return
        h, w, ch = frame_rgb.shape
        if ch != 3 or h <= 0 or w <= 0:
            return
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._last_image = image
        self._render()

    def clear(self, message: str = "") -> None:
        """Drop the last frame and optionally show a placeholder message."""
        self._last_image = None
        self._label.setPixmap(QPixmap())
        self._label.setText(message)

    def _render(self) -> None:
        if self._last_image is None:
            return
        pixmap = QPixmap.fromImage(self._last_image)
        scaled = pixmap.scaled(
            self._label.width(), self._label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        super().resizeEvent(event)
        self._render()