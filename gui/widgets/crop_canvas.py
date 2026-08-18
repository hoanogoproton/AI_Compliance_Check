import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QImage, QMouseEvent
from PySide6.QtWidgets import QWidget, QSizePolicy


HANDLE_SIZE = 8


class CropCanvas(QWidget):
    crop_changed = Signal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._crop_rect: tuple[int, int, int, int] | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_end: tuple[int, int] | None = None
        self._resize_handle: int | None = None
        self._is_dragging: bool = False

    def set_image(self, image: np.ndarray):
        if image is None:
            return
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qt_image)
        self.setMinimumSize(1, 1)
        self.update()

    def has_image(self) -> bool:
        return self._pixmap is not None

    def load_first_frame(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.set_image(rgb)

    def crop_rect(self) -> tuple[int, int, int, int] | None:
        return self._crop_rect

    def set_crop_rect(self, x: int, y: int, w: int, h: int):
        self._crop_rect = (x, y, w, h)
        self._drag_start = None
        self._drag_end = None
        self.update()

    def clear_crop(self):
        self._crop_rect = None
        self._drag_start = None
        self._drag_end = None
        self.crop_changed.emit(0, 0, 0, 0)
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

    def _to_widget_coords(self, ix: int, iy: int) -> tuple[int, int]:
        if self._pixmap is None:
            return (0, 0)
        scaled = self._pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        ox = (self.width() - scaled.width()) // 2
        oy = (self.height() - scaled.height()) // 2
        sx = ix * scaled.width() // max(self._pixmap.width(), 1) + ox
        sy = iy * scaled.height() // max(self._pixmap.height(), 1) + oy
        return (sx, sy)

    def _get_handle_at(self, pos) -> int | None:
        if self._crop_rect is None:
            return None
        x, y, w, h = self._crop_rect
        cx, cy = x + w // 2, y + h // 2
        ext_x, ext_y = x + w, y + h
        corners = [
            (0, x, y), (1, ext_x, y), (2, x, ext_y), (3, ext_x, ext_y),
        ]
        ix, iy = self._to_image_coords(pos)
        for idx, hx, hy in corners:
            if abs(ix - hx) <= HANDLE_SIZE and abs(iy - hy) <= HANDLE_SIZE:
                return idx
        if x <= ix <= ext_x and y <= iy <= ext_y:
            return 4
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if self._pixmap is None or event.button() != Qt.LeftButton:
            return
        handle = self._get_handle_at(event.position())
        if handle is not None:
            self._resize_handle = handle
            self._drag_start = self._to_image_coords(event.position())
            self._is_dragging = True
        else:
            self._resize_handle = None
            self._drag_start = self._to_image_coords(event.position())
            self._drag_end = self._drag_start
            self._is_dragging = True

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._is_dragging:
            handle = self._get_handle_at(event.position())
            if handle is not None:
                self.setCursor(Qt.SizeAllCursor if handle == 4 else Qt.CrossCursor)
            else:
                self.setCursor(Qt.CrossCursor)
            return
        pos = self._to_image_coords(event.position())
        if self._resize_handle is not None and self._crop_rect:
            x, y, w, h = self._crop_rect
            ex, ey = x + w, y + h
            hx, hy = pos
            if self._resize_handle == 0:
                x, y = hx, hy
            elif self._resize_handle == 1:
                ex, y = hx, hy
            elif self._resize_handle == 2:
                x, ey = hx, hy
            elif self._resize_handle == 3:
                ex, ey = hx, hy
            elif self._resize_handle == 4:
                dx = hx - self._drag_start[0]
                dy = hy - self._drag_start[1]
                x += dx
                y += dy
                ex += dx
                ey += dy
                self._drag_start = (hx, hy)
            if ex < x:
                x, ex = ex, x
            if ey < y:
                y, ey = ey, y
            pw = self._pixmap.width()
            ph = self._pixmap.height()
            x = max(0, x)
            y = max(0, y)
            ex = min(pw, ex)
            ey = min(ph, ey)
            self._crop_rect = (x, y, ex - x, ey - y)
        else:
            self._drag_end = pos
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self._is_dragging or event.button() != Qt.LeftButton:
            return
        self._is_dragging = False
        if self._resize_handle is None and self._drag_start and self._drag_end:
            x1 = min(self._drag_start[0], self._drag_end[0])
            y1 = min(self._drag_start[1], self._drag_end[1])
            x2 = max(self._drag_start[0], self._drag_end[0])
            y2 = max(self._drag_start[1], self._drag_end[1])
            diff_x = x2 - x1
            diff_y = y2 - y1
            if diff_x > 10 and diff_y > 10:
                self._crop_rect = (x1, y1, diff_x, diff_y)
        self._drag_start = None
        self._drag_end = None
        self._resize_handle = None
        if self._crop_rect:
            x, y, w, h = self._crop_rect
            self.crop_changed.emit(x, y, w, h)
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

        scale_x = scaled.width() / self._pixmap.width()
        scale_y = scaled.height() / self._pixmap.height()

        rect_to_draw = self._crop_rect
        if self._drag_start and self._drag_end and self._resize_handle is None:
            x1 = min(self._drag_start[0], self._drag_end[0])
            y1 = min(self._drag_start[1], self._drag_end[1])
            x2 = max(self._drag_start[0], self._drag_end[0])
            y2 = max(self._drag_start[1], self._drag_end[1])
            rect_to_draw = (x1, y1, x2 - x1, y2 - y1)

        if rect_to_draw:
            rx, ry, rw, rh = rect_to_draw
            sx = rx * scale_x + ox
            sy = ry * scale_y + oy
            sw = rw * scale_x
            sh = rh * scale_y

            painter.setPen(QPen(QColor(0, 180, 255), 2))
            painter.setBrush(QBrush(QColor(0, 180, 255, 40)))
            painter.drawRect(int(sx), int(sy), int(sw), int(sh))

            for hx, hy in [(rx, ry), (rx + rw, ry), (rx, ry + rh), (rx + rw, ry + rh)]:
                hsx = hx * scale_x + ox - HANDLE_SIZE // 2
                hsy = hy * scale_y + oy - HANDLE_SIZE // 2
                painter.setBrush(QBrush(QColor(0, 180, 255)))
                painter.setPen(QPen(Qt.white, 1))
                painter.drawRect(int(hsx), int(hsy), HANDLE_SIZE, HANDLE_SIZE)

            dim_text = f"{rw} x {rh} px"
            painter.setPen(Qt.white)
            painter.drawText(int(sx), int(sy) - 8, dim_text)

        painter.end()