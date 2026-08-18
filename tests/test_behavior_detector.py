import numpy as np

from detection.behavior_detector import is_hand_to_head
from detection.pose_utils import (
    compute_head_center,
    compute_shoulder_width,
    get_keypoint,
    get_wrist_positions,
)


def _make_kpts(arr_17x3):
    return np.array(arr_17x3, dtype=np.float32)


def test_get_keypoint():
    kpts = _make_kpts([[i * 10, i * 10 + 1, 0.9] for i in range(17)])
    x, y, c = get_keypoint(kpts, 5)
    assert x == 50.0
    assert y == 51.0
    assert round(c, 6) == 0.9


def test_compute_shoulder_width():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[5] = [0, 0, 0.9]
    kpts[6] = [100, 0, 0.9]
    assert compute_shoulder_width(kpts) == 100.0


def test_compute_shoulder_width_low_conf():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[5] = [0, 0, 0.2]
    kpts[6] = [100, 0, 0.9]
    assert compute_shoulder_width(kpts) == 0.0


def test_compute_head_center_ears():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[3] = [80, 50, 0.9]
    kpts[4] = [120, 50, 0.9]
    cx, cy = compute_head_center(kpts)
    assert cx == 100.0
    assert cy == 50.0


def test_compute_head_center_fallback_nose():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[0] = [100, 60, 0.8]
    cx, cy = compute_head_center(kpts)
    assert cx == 100.0
    assert cy == 60.0


def test_compute_head_center_none():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    assert compute_head_center(kpts) is None


def test_get_wrist_positions():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[9] = [10, 20, 0.9]
    kpts[10] = [30, 40, 0.1]
    wrists = get_wrist_positions(kpts)
    assert len(wrists) == 1
    wx, wy, wc = wrists[0]
    assert wx == 10.0
    assert wy == 20.0
    assert round(wc, 6) == 0.9


def test_is_hand_to_head_detected_left():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[3] = [80, 50, 0.9]
    kpts[4] = [120, 50, 0.9]
    kpts[5] = [60, 80, 0.9]
    kpts[6] = [140, 80, 0.9]
    kpts[9] = [95, 55, 0.9]
    detected, conf, side = is_hand_to_head(kpts, threshold_ratio=0.25)
    assert detected
    assert side == "left"


def test_is_hand_to_head_not_detected():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[3] = [80, 50, 0.9]
    kpts[4] = [120, 50, 0.9]
    kpts[5] = [60, 80, 0.9]
    kpts[6] = [140, 80, 0.9]
    kpts[9] = [400, 400, 0.9]
    detected, conf, side = is_hand_to_head(kpts, threshold_ratio=0.25)
    assert not detected
    assert side == "none"


def test_is_hand_to_head_too_small():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[5] = [99, 0, 0.9]
    kpts[6] = [100, 0, 0.9]
    detected, conf, side = is_hand_to_head(kpts)
    assert not detected


def test_is_hand_to_head_both():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[3] = [80, 50, 0.9]
    kpts[4] = [120, 50, 0.9]
    kpts[5] = [60, 80, 0.9]
    kpts[6] = [140, 80, 0.9]
    kpts[9] = [95, 55, 0.9]
    kpts[10] = [105, 55, 0.9]
    detected, conf, side = is_hand_to_head(kpts, threshold_ratio=0.25)
    assert detected
    assert side == "both"
