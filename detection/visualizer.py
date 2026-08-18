import cv2
import numpy as np

from detection.config import COCO_KEYPOINTS, KEYPOINT_CONFIDENCE_THRESHOLD

KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(
    frame: np.ndarray,
    kpts: np.ndarray,
    bbox: tuple[float, float, float, float],
    track_id: int,
    detected: bool,
    behavior_name: str = "",
) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 0, 255) if detected else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id}"
    if detected and behavior_name:
        label += f" {behavior_name}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for pt_idx in range(kpts.shape[0]):
        x, y, c = kpts[pt_idx]
        if c > KEYPOINT_CONFIDENCE_THRESHOLD:
            color_kp = (0, 255, 255)
            if pt_idx in (9, 10):
                color_kp = (0, 165, 255)
            elif pt_idx in (0, 1, 2, 3, 4):
                color_kp = (255, 0, 0)
            cv2.circle(frame, (int(x), int(y)), 4, color_kp, -1)
    for i, j in KEYPOINT_CONNECTIONS:
        if kpts[i, 2] > 0.3 and kpts[j, 2] > 0.3:
            xi, yi = int(kpts[i, 0]), int(kpts[i, 1])
            xj, yj = int(kpts[j, 0]), int(kpts[j, 1])
            cv2.line(frame, (xi, yi), (xj, yj), (255, 255, 255), 2)
    return frame


def draw_zone(frame: np.ndarray, zone, active: bool | str) -> np.ndarray:
    if len(zone.points) < 3:
        return frame
    if isinstance(active, str):
        color = {
            "active": (0, 0, 255),
            "inactive": (0, 255, 0),
        }.get(active, (0, 255, 0))
    else:
        color = (0, 0, 255) if active else (0, 255, 0)
    pts = zone.polygon.reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.polylines(frame, [pts], True, color, 2)
    xs = [p[0] for p in zone.points]
    ys = [p[1] for p in zone.points]
    cx = int(sum(xs) / len(xs))
    cy = int(min(ys)) - 10
    cv2.putText(frame, zone.label, (cx - 30, max(cy, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
