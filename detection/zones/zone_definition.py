from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Zone:
    name: str
    label: str
    points: list[list[int]]

    @property
    def polygon(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32)

    def contains_point(self, x: float, y: float) -> bool:
        if len(self.points) < 3:
            return False
        pts = self.polygon.astype(np.float32)
        result = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
        return result >= 0

    @staticmethod
    def _segments_intersect(
        a: tuple[float, float], b: tuple[float, float],
        c: tuple[float, float], d: tuple[float, float],
    ) -> bool:
        def cross(o: tuple[float, float], p: tuple[float, float], q: tuple[float, float]) -> float:
            return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

        def on_segment(o: tuple[float, float], p: tuple[float, float], q: tuple[float, float]) -> bool:
            return (
                min(o[0], p[0]) <= q[0] <= max(o[0], p[0])
                and min(o[1], p[1]) <= q[1] <= max(o[1], p[1])
            )

        d1 = cross(c, d, a)
        d2 = cross(c, d, b)
        d3 = cross(a, b, c)
        d4 = cross(a, b, d)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
            (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
        ):
            return True
        if d1 == 0 and on_segment(c, d, a):
            return True
        if d2 == 0 and on_segment(c, d, b):
            return True
        if d3 == 0 and on_segment(a, b, c):
            return True
        if d4 == 0 and on_segment(a, b, d):
            return True
        return False

    def intersects_bbox(self, bbox: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        for cx, cy in rect:
            if self.contains_point(cx, cy):
                return True

        pts = self.points
        n = len(pts)
        for px, py in pts:
            if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
                return True

        for i in range(n):
            a = tuple(pts[i])
            b = tuple(pts[(i + 1) % n])
            for j in range(4):
                if self._segments_intersect(a, b, rect[j], rect[(j + 1) % 4]):
                    return True
        return False

    def contains_bbox_center(self, bbox: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return self.contains_point(cx, cy)

    def to_dict(self) -> dict:
        return {"label": self.label, "points": self.points}

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "Zone":
        return cls(name=name, label=data.get("label", name), points=data["points"])
