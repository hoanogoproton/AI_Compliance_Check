import numpy as np

from detection.behaviors.leave_zone import LeaveZoneBehavior
from detection.detector import TrackedPerson
from detection.zones.zone_definition import Zone


def _make_person(track_id, bbox):
    kpts = np.zeros((17, 3), dtype=np.float32)
    return TrackedPerson(track_id=track_id, bbox=bbox, keypoints=kpts, conf=0.9)


def test_leave_zone_detected():
    zone = Zone(name="test_zone", label="Test Zone", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    behavior = LeaveZoneBehavior({"min_stay_frames": 2}, zone=zone)
    person = _make_person(1, (40, 40, 60, 60))

    # First frame inside
    r = behavior.detect_person(person, None, 0, 0.0)
    assert not r.detected
    assert r.metadata["inside"]

    # Second frame inside (reaches min_stay)
    r = behavior.detect_person(person, None, 1, 0.033)
    assert not r.detected

    # Third frame outside
    person2 = _make_person(1, (200, 200, 220, 220))
    r = behavior.detect_person(person2, None, 2, 0.066)
    assert r.detected
    assert r.metadata["side"] == "outside"


def test_leave_zone_never_inside():
    zone = Zone(name="test_zone", label="Test Zone", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    behavior = LeaveZoneBehavior({"min_stay_frames": 2}, zone=zone)
    person = _make_person(1, (200, 200, 220, 220))
    r = behavior.detect_person(person, None, 0, 0.0)
    assert not r.detected
    assert not r.metadata["inside"]


def test_leave_zone_no_zone_raises():
    try:
        LeaveZoneBehavior({"min_stay_frames": 2}, zone=None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_leave_zone_process_frame_emits_event():
    zone = Zone(name="test_zone", label="Test Zone", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    behavior = LeaveZoneBehavior({"min_stay_frames": 2}, zone=zone)

    # Two frames inside to reach min_stay
    inside = _make_person(1, (40, 40, 60, 60))
    events = behavior.process_frame([inside], None, 0, 0.0)
    assert events == []
    events = behavior.process_frame([inside], None, 1, 0.033)
    assert events == []
    assert behavior._track_inside[1]

    # Person leaves -> event emitted immediately
    outside = _make_person(1, (200, 200, 220, 220))
    events = behavior.process_frame([outside], None, 2, 0.066)
    assert len(events) == 1
    assert events[0].track_id == 1
    assert events[0].behavior_name == "leave_zone"
    assert not behavior._track_inside[1]

    # No more events while person stays outside
    events = behavior.process_frame([outside], None, 3, 0.099)
    assert events == []
