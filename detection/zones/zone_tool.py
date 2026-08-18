import sys

import cv2
import numpy as np
import yaml

from detection.zones.zone_checker import save_zones
from detection.zones.zone_definition import Zone


_POINTS: list[list[int]] = []
_DRAG_START = None
_DRAG_END = None


def _mouse_callback(event, x, y, flags, param):
    global _POINTS, _DRAG_START, _DRAG_END
    if event == cv2.EVENT_LBUTTONDOWN:
        _POINTS.append([x, y])
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        _DRAG_START = _POINTS[-1] if _POINTS else None
        _DRAG_END = [x, y]


def define_zones(video_path: str, config_path: str, crop_region: tuple | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read first frame")
        sys.exit(1)

    if crop_region:
        x, y, w, h = crop_region
        fh, fw = frame.shape[:2]
        x_end = min(x + w, fw)
        y_end = min(y + h, fh)
        frame = frame[y:y_end, x:x_end]

    global _POINTS, _DRAG_START, _DRAG_END
    _POINTS = []
    _DRAG_START = None
    _DRAG_END = None

    cv2.namedWindow("Define Zone")
    cv2.setMouseCallback("Define Zone", _mouse_callback)

    print("=== Zone Definition Tool ===")
    print("Left click: add point")
    print("Right click: undo last point")
    print("Enter: finish zone and save")
    print("ESC: exit without saving")

    while True:
        display = frame.copy()
        for pt in _POINTS:
            cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
        if len(_POINTS) > 1:
            pts_arr = np.array(_POINTS, dtype=np.int32)
            cv2.polylines(display, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2)
        if _DRAG_START and _DRAG_END:
            cv2.line(display, tuple(_DRAG_START), tuple(_DRAG_END), (0, 255, 0), 2)
        cv2.imshow("Define Zone", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:
            if len(_POINTS) < 3:
                print("Need at least 3 points for a polygon.")
                continue
            zone_name = input("Zone name: ").strip()
            zone_label = input("Zone label (display name): ").strip() or zone_name
            zone = Zone(name=zone_name, label=zone_label, points=_POINTS)
            save_zones({zone_name: zone}, config_path)
            print(f"Zone '{zone_name}' saved to {config_path}")
            break
        elif key == 27:
            print("Exiting zone tool.")
            break
        elif key == 2:
            if _POINTS:
                removed = _POINTS.pop()
                print(f"Removed point: {removed}")

    cv2.destroyAllWindows()
    sys.exit(0)
