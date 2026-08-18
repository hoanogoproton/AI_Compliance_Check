import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QPointF, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QImage, QPolygonF, QMouseEvent, QKeyEvent
from PySide6.QtWidgets import QWidget, QSizePolicy


class ZoneCanvas(QWidget):
    points_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._points: list[list[int]] = []
        self._hover_pos: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray):
        if image is None:
            return
        image = np.ascontiguousarray(image)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qt_image)
        self.setMinimumSize(1, 1)
        self.update()

    def set_frame(self, image: np.ndarray):
        self.set_image(image)

    def has_image(self) -> bool:
        return self._pixmap is not None

    def load_first_frame(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.set_image(rgb)

    def points(self) -> list[list[int]]:
        return self._points

    def set_points(self, pts: list[list[int]]):
        self._points = [list(p) for p in pts]
        self.points_changed.emit(self._points)
        self.update()

    def clear_points(self):
        self._points.clear()
        self.points_changed.emit(self._points)
        self.update()

    def _to_image_coords(self, pos) -> tuple[int, int]:
        if self._pixmap is None:
            return (0, 0)
        scaled = self._pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        ox = (self.width() - scaled.width()) // 2
        oy = (self.height() - scaled.height()) // 2
        ix = (pos.x() - ox) * self._pixmap.width() // max(scaled.width(), 1)
        iy = (pos.y() - oy) * self._pixmap.height() // max(scaled.height(), 1)
        return (max(0, min(ix, self._pixmap.width() - 1)),
                max(0, min(iy, self._pixmap.height() - 1)))

    def mousePressEvent(self, event: QMouseEvent):
        if self._pixmap is None:
            return
        if event.button() == Qt.LeftButton:
            ix, iy = self._to_image_coords(event.position())
            self._points.append([ix, iy])
            self.points_changed.emit(self._points)
            self.update()
        elif event.button() == Qt.RightButton:
            if self._points:
                self._points.pop()
                self.points_changed.emit(self._points)
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._pixmap is not None:
            self._hover_pos = (int(event.position().x()), int(event.position().y()))
            self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace:
            if self._points:
                self._points.pop()
                self.points_changed.emit(self._points)
                self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg = QColor(30, 30, 30)
        painter.fillRect(self.rect(), bg)

        if self._pixmap is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            painter.end()
            return

        scaled = self._pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        ox = (self.width() - scaled.width()) // 2
        oy = (self.height() - scaled.height()) // 2
        painter.drawPixmap(ox, oy, scaled)

        if not self._points:
            painter.end()
            return

        scale_x = scaled.width() / self._pixmap.width()
        scale_y = scaled.height() / self._pixmap.height()

        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)

        for i, (x, y) in enumerate(self._points):
            sx = int(x * scale_x + ox)
            sy = int(y * scale_y + oy)
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            painter.drawEllipse(sx - 4, sy - 4, 8, 8)
            painter.setPen(Qt.white)
            painter.drawText(sx + 8, sy - 8, str(i + 1))
            painter.setPen(pen)

        if len(self._points) >= 2:
            poly = QPolygonF()
            for x, y in self._points:
                sx = x * scale_x + ox
                sy = y * scale_y + oy
                poly.append(QPointF(sx, sy))
            painter.setBrush(QBrush(QColor(0, 255, 0, 40)))
            painter.setPen(pen)
            painter.drawPolygon(poly)

        if self._hover_pos and self._points:
            hx, hy = self._hover_pos
            painter.setPen(QPen(QColor(255, 255, 0, 180), 1, Qt.DashLine))
            last = self._points[-1]
            lsx = last[0] * scale_x + ox
            lsy = last[1] * scale_y + oy
            painter.drawLine(int(lsx), int(lsy), hx, hy)

        painter.end()