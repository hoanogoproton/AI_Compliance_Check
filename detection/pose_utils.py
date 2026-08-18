import numpy as np

from detection.config import HEAD_KEYPOINT_CONFIDENCE_THRESHOLD, KEYPOINT_CONFIDENCE_THRESHOLD


def get_keypoint(kpts: np.ndarray, idx: int) -> tuple[float, float, float]:
    return float(kpts[idx, 0]), float(kpts[idx, 1]), float(kpts[idx, 2])


def compute_shoulder_width(kpts: np.ndarray) -> float:
    lx, ly, lc = get_keypoint(kpts, 5)
    rx, ry, rc = get_keypoint(kpts, 6)
    if lc < KEYPOINT_CONFIDENCE_THRESHOLD or rc < KEYPOINT_CONFIDENCE_THRESHOLD:
        return 0.0
    return np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)


def compute_head_center(kpts: np.ndarray) -> tuple[float, float] | None:
    le_x, le_y, le_c = get_keypoint(kpts, 3)
    re_x, re_y, re_c = get_keypoint(kpts, 4)
    if le_c >= HEAD_KEYPOINT_CONFIDENCE_THRESHOLD and re_c >= HEAD_KEYPOINT_CONFIDENCE_THRESHOLD:
        return ((le_x + re_x) / 2, (le_y + re_y) / 2)
    li_x, li_y, li_c = get_keypoint(kpts, 1)
    ri_x, ri_y, ri_c = get_keypoint(kpts, 2)
    if li_c >= HEAD_KEYPOINT_CONFIDENCE_THRESHOLD and ri_c >= HEAD_KEYPOINT_CONFIDENCE_THRESHOLD:
        return ((li_x + ri_x) / 2, (li_y + ri_y) / 2)
    nx, ny, nc = get_keypoint(kpts, 0)
    if nc >= KEYPOINT_CONFIDENCE_THRESHOLD:
        return (nx, ny)
    return None


def get_wrist_positions(kpts: np.ndarray) -> list[tuple[float, float, float]]:
    result = []
    lx, ly, lc = get_keypoint(kpts, 9)
    rx, ry, rc = get_keypoint(kpts, 10)
    if lc >= KEYPOINT_CONFIDENCE_THRESHOLD:
        result.append((lx, ly, lc))
    if rc >= KEYPOINT_CONFIDENCE_THRESHOLD:
        result.append((rx, ry, rc))
    return result
