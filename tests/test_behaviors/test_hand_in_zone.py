import numpy as np

from detection.behaviors.hand_in_zone import HandInZoneBehavior
from detection.detector import TrackedPerson
from detection.zones.zone_definition import Zone


def _make_person(track_id, kpts):
    return TrackedPerson(track_id=track_id, bbox=(0, 0, 100, 200), keypoints=kpts, conf=0.9)


def _make_kpts(arr_17x3):
    return np.array(arr_17x3, dtype=np.float32)


def test_hand_in_zone_left():
    zone = Zone(name="desk", label="Desk", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    behavior = HandInZoneBehavior({"hand": "any"}, zone=zone)
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[9] = [50, 50, 0.9]  # left wrist inside zone
    kpts[10] = [200, 200, 0.9]  # right wrist outside
    person = _make_person(1, kpts)
    result = behavior.detect_person(person, None, 0, 0.0)
    assert result.detected
    assert result.metadata["side"] == "left"


def test_hand_in_zone_right():
    zone = Zone(name="desk", label="Desk", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    behavior = HandInZoneBehavior({"hand": "any"}, zone=zone)
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[9] = [200, 200, 0.9]
    kpts[10] = [50, 50, 0.9]
    person = _make_person(1, kpts)
    result = behavior.detect_person(person, None, 0, 0.0)
    assert result.detected
    assert result.metadata["side"] == "right"


def test_hand_in_zone_not_detected():
    zone = Zone(name="desk", label="Desk", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    behavior = HandInZoneBehavior({"hand": "any"}, zone=zone)
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[9] = [200, 200, 0.9]
    kpts[10] = [300, 300, 0.9]
    person = _make_person(1, kpts)
    result = behavior.detect_person(person, None, 0, 0.0)
    assert not result.detected
    assert result.metadata["side"] == "none"


def test_hand_in_zone_no_zone_raises():
    try:
        HandInZoneBehavior({"hand": "any"}, zone=None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
