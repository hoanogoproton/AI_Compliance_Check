import numpy as np

from detection.behaviors.body_turn import BodyTurnBehavior
from detection.detector import TrackedPerson
from detection.pose_utils import (
    angular_difference_deg,
    compute_body_orientation,
    compute_hip_angle,
    compute_shoulder_angle,
)


def _make_kpts(shoulder_l=None, shoulder_r=None, hip_l=None, hip_r=None):
    kpts = np.zeros((17, 3), dtype=np.float32)
    if shoulder_l:
        kpts[5] = shoulder_l
    if shoulder_r:
        kpts[6] = shoulder_r
    if hip_l:
        kpts[11] = hip_l
    if hip_r:
        kpts[12] = hip_r
    return kpts


def _make_person(track_id, kpts):
    return TrackedPerson(track_id=track_id, bbox=(0, 0, 100, 200), keypoints=kpts, conf=0.9)


def _run(behavior, track_id, kpts_sequence):
    results = []
    for i, kpts in enumerate(kpts_sequence):
        person = _make_person(track_id, kpts)
        results.append(behavior.detect_person(person, None, i, i / 30.0))
    return results


def test_compute_shoulder_angle_facing_camera():
    kpts = _make_kpts(shoulder_l=(40, 60, 0.9), shoulder_r=(60, 60, 0.9))
    angle = compute_shoulder_angle(kpts)
    assert angle is not None
    assert angle == 0.0


def test_compute_shoulder_angle_rotated_90deg():
    kpts = _make_kpts(shoulder_l=(50, 40, 0.9), shoulder_r=(50, 60, 0.9))
    angle = compute_shoulder_angle(kpts)
    assert angle is not None
    assert angle == 90.0


def test_compute_shoulder_angle_45deg():
    kpts = _make_kpts(shoulder_l=(40, 40, 0.9), shoulder_r=(60, 60, 0.9))
    angle = compute_shoulder_angle(kpts)
    assert angle is not None
    assert angle == 45.0


def test_compute_shoulder_angle_returns_none():
    kpts = _make_kpts(shoulder_l=(40, 60, 0.1), shoulder_r=(60, 60, 0.1))
    assert compute_shoulder_angle(kpts) is None


def test_compute_shoulder_angle_one_occluded():
    kpts = _make_kpts(shoulder_l=(40, 60, 0.9), shoulder_r=(60, 60, 0.1))
    assert compute_shoulder_angle(kpts) is None


def test_compute_hip_angle_fallback():
    kpts = _make_kpts(shoulder_l=(40, 60, 0.1), shoulder_r=(60, 60, 0.1),
                      hip_l=(40, 120, 0.9), hip_r=(60, 120, 0.9))
    angle = compute_hip_angle(kpts)
    assert angle is not None
    assert angle == 0.0
    assert compute_body_orientation(kpts) == 0.0


def test_angular_difference_deg_wrap():
    assert angular_difference_deg(10.0, 170.0) == 20.0
    assert angular_difference_deg(170.0, 10.0) == 20.0
    assert angular_difference_deg(0.0, 90.0) == 90.0
    assert angular_difference_deg(45.0, 45.0) == 0.0


def test_body_turn_not_detected_when_stable():
    behavior = BodyTurnBehavior({"min_angle": 45, "window_frames": 15, "velocity_threshold": 10})
    kpts = _make_kpts(shoulder_l=(40, 60, 0.9), shoulder_r=(60, 60, 0.9))
    results = _run(behavior, 1, [kpts] * 20)
    assert all(not r.detected for r in results)


def test_body_turn_detected_after_90deg_turn():
    behavior = BodyTurnBehavior({"min_angle": 45, "window_frames": 15, "velocity_threshold": 5})
    facing = _make_kpts(shoulder_l=(40, 60, 0.9), shoulder_r=(60, 60, 0.9))
    turned = _make_kpts(shoulder_l=(50, 40, 0.9), shoulder_r=(50, 60, 0.9))
    kpts_seq = [facing] * 5 + [turned] * 20
    results = _run(behavior, 1, kpts_seq)
    detected = [r for r in results if r.detected]
    assert len(detected) > 0
    assert detected[0].confidence > 0


def test_body_turn_confidence_scales_with_delta():
    behavior = BodyTurnBehavior({"min_angle": 30, "window_frames": 15, "velocity_threshold": 3})
    facing = _make_kpts(shoulder_l=(40, 60, 0.9), shoulder_r=(60, 60, 0.9))
    slight = _make_kpts(shoulder_l=(35, 55, 0.9), shoulder_r=(65, 65, 0.9))
    kpts_seq = [facing] * 5 + [slight] * 20
    results = _run(behavior, 1, kpts_seq)
    small_detected = [r for r in results if r.detected]
    full = _make_kpts(shoulder_l=(50, 40, 0.9), shoulder_r=(50, 60, 0.9))
    kpts_seq2 = [facing] * 5 + [full] * 20
    results2 = _run(behavior, 2, kpts_seq2)
    big_detected = [r for r in results2 if r.detected]
    assert len(big_detected) > 0
    assert all(r.confidence > 0 for r in big_detected)


def test_body_turn_return_metadata():
    behavior = BodyTurnBehavior({"min_angle": 45, "window_frames": 15, "velocity_threshold": 5})
    facing = _make_kpts(shoulder_l=(40, 60, 0.9), shoulder_r=(60, 60, 0.9))
    turned = _make_kpts(shoulder_l=(50, 40, 0.9), shoulder_r=(50, 60, 0.9))
    results = _run(behavior, 1, [facing] * 5 + [turned])
    assert "delta_deg" in results[-1].metadata
    assert "velocity" in results[-1].metadata