import math

import numpy as np

from detection.config import HEAD_KEYPOINT_CONFIDENCE_THRESHOLD, KEYPOINT_CONFIDENCE_THRESHOLD


def get_keypoint(kpts: np.ndarray, idx: int) -> tuple[float, float, float]:
    return float(kpts[idx, 0]), float(kpts[idx, 1]), float(kpts[idx, 2])


def compute_shoulder_width(kpts: np.ndarray, kpt_conf_threshold: float | None = None) -> float:
    threshold = kpt_conf_threshold if kpt_conf_threshold is not None else KEYPOINT_CONFIDENCE_THRESHOLD
    lx, ly, lc = get_keypoint(kpts, 5)
    rx, ry, rc = get_keypoint(kpts, 6)
    if lc < threshold or rc < threshold:
        return 0.0
    return np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)


def compute_head_center(kpts: np.ndarray, head_conf_threshold: float | None = None) -> tuple[float, float] | None:
    threshold = head_conf_threshold if head_conf_threshold is not None else HEAD_KEYPOINT_CONFIDENCE_THRESHOLD
    le_x, le_y, le_c = get_keypoint(kpts, 3)
    re_x, re_y, re_c = get_keypoint(kpts, 4)
    if le_c >= threshold and re_c >= threshold:
        return ((le_x + re_x) / 2, (le_y + re_y) / 2)
    li_x, li_y, li_c = get_keypoint(kpts, 1)
    ri_x, ri_y, ri_c = get_keypoint(kpts, 2)
    if li_c >= threshold and ri_c >= threshold:
        return ((li_x + ri_x) / 2, (li_y + ri_y) / 2)
    nx, ny, nc = get_keypoint(kpts, 0)
    if nc >= threshold:
        return (nx, ny)
    return None


def compute_head_yaw_offset(kpts: np.ndarray) -> tuple[float, float] | None:
    nx, ny, nc = get_keypoint(kpts, 0)
    if nc < KEYPOINT_CONFIDENCE_THRESHOLD:
        return None
    head_center = compute_head_center(kpts)
    if head_center is None:
        return None
    hx, hy = head_center
    le_x, le_y, le_c = get_keypoint(kpts, 3)
    re_x, re_y, re_c = get_keypoint(kpts, 4)
    if le_c >= HEAD_KEYPOINT_CONFIDENCE_THRESHOLD and re_c >= HEAD_KEYPOINT_CONFIDENCE_THRESHOLD:
        head_width = abs(re_x - le_x)
    else:
        shoulder_width = compute_shoulder_width(kpts)
        head_width = shoulder_width * 0.5
    if head_width < 1e-3:
        return None
    return ((nx - hx) / head_width, head_width)


def get_wrist_positions(kpts: np.ndarray) -> list[tuple[float, float, float]]:
    result = []
    lx, ly, lc = get_keypoint(kpts, 9)
    rx, ry, rc = get_keypoint(kpts, 10)
    if lc >= KEYPOINT_CONFIDENCE_THRESHOLD:
        result.append((lx, ly, lc))
    if rc >= KEYPOINT_CONFIDENCE_THRESHOLD:
        result.append((rx, ry, rc))
    return result


def compute_shoulder_angle(kpts: np.ndarray) -> float | None:
    lx, ly, lc = get_keypoint(kpts, 5)
    rx, ry, rc = get_keypoint(kpts, 6)
    if lc < KEYPOINT_CONFIDENCE_THRESHOLD or rc < KEYPOINT_CONFIDENCE_THRESHOLD:
        return None
    angle_rad = math.atan2(ry - ly, rx - lx)
    angle_deg = math.degrees(angle_rad) % 180.0
    return angle_deg


def compute_hip_angle(kpts: np.ndarray) -> float | None:
    lx, ly, lc = get_keypoint(kpts, 11)
    rx, ry, rc = get_keypoint(kpts, 12)
    if lc < KEYPOINT_CONFIDENCE_THRESHOLD or rc < KEYPOINT_CONFIDENCE_THRESHOLD:
        return None
    angle_rad = math.atan2(ry - ly, rx - lx)
    angle_deg = math.degrees(angle_rad) % 180.0
    return angle_deg


def compute_body_orientation(kpts: np.ndarray) -> float | None:
    angle = compute_shoulder_angle(kpts)
    if angle is not None:
        return angle
    return compute_hip_angle(kpts)


def angular_difference_deg(a: float, b: float) -> float:
    diff = abs(a - b)
    return min(diff, 180.0 - diff)


def ear_dominance_ratio(kpts: np.ndarray, min_conf: float = 0.1) -> float | None:
    le_x, le_y, le_c = get_keypoint(kpts, 3)
    re_x, re_y, re_c = get_keypoint(kpts, 4)
    if le_c < min_conf and re_c < min_conf:
        return None
    return le_c / (le_c + re_c + 1e-6)


def face_visibility_score(
    kpts: np.ndarray,
    min_conf: float = KEYPOINT_CONFIDENCE_THRESHOLD,
) -> float:
    face_indices = [0, 1, 2, 3, 4]
    if kpts.shape[0] <= max(face_indices):
        return 0.0
    visible = sum(
        1 for idx in face_indices
        if float(kpts[idx, 2]) >= min_conf
    )
    return visible / len(face_indices)


def is_body_trackable(
    kpts: np.ndarray,
    min_conf: float = KEYPOINT_CONFIDENCE_THRESHOLD,
    min_visible: int = 2,
) -> bool:
    body_indices = [5, 6, 11, 12]
    if kpts.shape[0] <= max(body_indices):
        return False
    visible = sum(
        1 for idx in body_indices
        if float(kpts[idx, 2]) >= min_conf
    )
    return visible >= min_visible
