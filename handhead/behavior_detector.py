import numpy as np

from handhead.config import DISTANCE_THRESHOLD_RATIO, KEYPOINT_CONFIDENCE_THRESHOLD, VERTICAL_OFFSET_RATIO
from handhead.pose_utils import compute_head_center, compute_shoulder_width, get_keypoint


def is_hand_to_head(
    kpts: np.ndarray, threshold_ratio: float = DISTANCE_THRESHOLD_RATIO
) -> tuple[bool, float, str]:
    head_center = compute_head_center(kpts)
    if head_center is None:
        return (False, 0.0, "none")
    shoulder_width = compute_shoulder_width(kpts)
    if shoulder_width < 20.0:
        return (False, 0.0, "none")
    max_allowed = shoulder_width * threshold_ratio
    hx, hy = head_center
    detected = False
    best_conf = 0.0
    sides = []
    for wrist_idx, side_name in [(9, "left"), (10, "right")]:
        wx, wy, wc = get_keypoint(kpts, wrist_idx)
        if wc < KEYPOINT_CONFIDENCE_THRESHOLD:
            continue
        if wy > hy + shoulder_width * VERTICAL_OFFSET_RATIO:
            continue
        dist = np.sqrt((wx - hx) ** 2 + (wy - hy) ** 2)
        if dist < max_allowed:
            detected = True
            conf = 1.0 - (dist / max_allowed)
            if conf > best_conf:
                best_conf = conf
            sides.append(side_name)
    if not detected:
        return (False, 0.0, "none")
    if len(sides) == 2:
        return (True, best_conf, "both")
    return (True, best_conf, sides[0])