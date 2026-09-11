import numpy as np

from detection.behaviors.danger_zone import DangerZoneBehavior
from detection.detector import TrackedPerson
from detection.zones.zone_definition import Zone


def _make_person(track_id, bbox):
    kpts = np.zeros((17, 3), dtype=np.float32)
    return TrackedPerson(track_id=track_id, bbox=bbox, keypoints=kpts, conf=0.9)


def _make_zone():
    return Zone(name="test_zone", label="Test Zone", points=[[0, 0], [100, 0], [100, 100], [0, 100]])


def test_danger_zone_detect_on_enter():
    zone = _make_zone()
    behavior = DangerZoneBehavior({"min_enter_frames": 2, "min_exit_frames": 2}, zones=[zone])
    person = _make_person(1, (40, 40, 60, 60))

    r = behavior.detect_person(person, None, 0, 0.0)
    assert not r.detected
    assert r.metadata["inside"]

    r = behavior.detect_person(person, None, 1, 0.033)
    assert r.detected
    assert r.metadata["side"] == "inside"
    assert r.metadata["zone"] == "test_zone"
    assert r.metadata["triggered_zones"] == ["test_zone"]
    assert r.metadata["inside"]


def test_danger_zone_process_frame_emits_event():
    zone = _make_zone()
    behavior = DangerZoneBehavior({"min_enter_frames": 2, "min_exit_frames": 2}, zones=[zone])
    inside = _make_person(1, (40, 40, 60, 60))

    events = behavior.process_frame([inside], None, 0, 0.0)
    assert events == []
    events = behavior.process_frame([inside], None, 1, 0.033)
    assert len(events) == 1
    assert events[0].track_id == 1
    assert events[0].behavior_name == "danger_zone"
    assert events[0].metadata["side"] == "inside"
    assert events[0].start_frame == 1
    assert events[0].end_frame == 1
    assert events[0].frames == [1]


def test_danger_zone_never_inside():
    zone = _make_zone()
    behavior = DangerZoneBehavior({"min_enter_frames": 1, "min_exit_frames": 2}, zones=[zone])
    outside = _make_person(1, (200, 200, 220, 220))
    for f in range(5):
        events = behavior.process_frame([outside], None, f, f * 0.033)
        assert events == []
    assert not behavior._track_in_zone[1]
    assert not behavior._alerted[1]


def test_danger_zone_no_zone_raises():
    try:
        DangerZoneBehavior({"min_enter_frames": 2}, zones=None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_danger_zone_enter_debounce():
    zone = _make_zone()
    behavior = DangerZoneBehavior({"min_enter_frames": 3, "min_exit_frames": 2}, zones=[zone])
    person = _make_person(1, (40, 40, 60, 60))

    for f in range(2):
        events = behavior.process_frame([person], None, f, f * 0.033)
        assert events == []
    assert behavior._in_zone_counter[1] == 2

    events = behavior.process_frame([person], None, 2, 0.066)
    assert len(events) == 1
    assert events[0].track_id == 1


def test_danger_zone_single_alert_per_entry():
    zone = _make_zone()
    behavior = DangerZoneBehavior({"min_enter_frames": 1, "min_exit_frames": 2}, zones=[zone])
    person = _make_person(1, (40, 40, 60, 60))

    events = behavior.process_frame([person], None, 0, 0.0)
    assert len(events) == 1

    for f in range(1, 8):
        events = behavior.process_frame([person], None, f, f * 0.033)
        assert events == []
    assert behavior._alerted[1]


def test_danger_zone_rearm_after_exit():
    zone = _make_zone()
    behavior = DangerZoneBehavior({"min_enter_frames": 1, "min_exit_frames": 2}, zones=[zone])
    inside = _make_person(1, (40, 40, 60, 60))
    outside = _make_person(1, (200, 200, 220, 220))

    events = behavior.process_frame([inside], None, 0, 0.0)
    assert len(events) == 1

    events = behavior.process_frame([outside], None, 1, 0.033)
    assert events == []
    assert behavior._alerted[1]

    events = behavior.process_frame([outside], None, 2, 0.066)
    assert events == []
    assert not behavior._alerted[1]
    assert not behavior._inside_zones.get(1)

    events = behavior.process_frame([inside], None, 3, 0.099)
    assert len(events) == 1
    assert events[0].track_id == 1
    assert events[0].behavior_name == "danger_zone"
    assert events[0].metadata["zone"] == "test_zone"


def test_danger_zone_flicker_no_duplicate():
    zone = _make_zone()
    behavior = DangerZoneBehavior({"min_enter_frames": 1, "min_exit_frames": 3}, zones=[zone])
    inside = _make_person(1, (40, 40, 60, 60))
    outside = _make_person(1, (200, 200, 220, 220))

    events = behavior.process_frame([inside], None, 0, 0.0)
    assert len(events) == 1

    events = behavior.process_frame([outside], None, 1, 0.033)
    assert events == []
    assert behavior._alerted[1]

    events = behavior.process_frame([outside], None, 2, 0.066)
    assert events == []
    assert behavior._alerted[1]

    events = behavior.process_frame([inside], None, 3, 0.099)
    assert events == []
    assert behavior._alerted[1]


def test_danger_zone_alert_flash_window():
    zone = _make_zone()
    behavior = DangerZoneBehavior(
        {"min_enter_frames": 1, "min_exit_frames": 2, "alert_flash_frames": 2}, zones=[zone]
    )
    inside = _make_person(1, (40, 40, 60, 60))
    outside = _make_person(1, (200, 200, 220, 220))

    behavior.process_frame([inside], None, 0, 0.0)
    assert behavior.is_person_in_alert(1, 0)
    assert behavior.is_person_in_alert(1, 2)

    behavior.process_frame([outside], None, 1, 0.033)
    assert behavior.is_person_in_alert(1, 1)

    behavior.process_frame([outside], None, 2, 0.066)
    assert behavior.is_person_in_alert(1, 2)
    assert not behavior.is_person_in_alert(1, 3)


def test_danger_zone_current_triggered_zones():
    zone = _make_zone()
    behavior = DangerZoneBehavior(
        {"min_enter_frames": 1, "min_exit_frames": 2, "alert_flash_frames": 3}, zones=[zone]
    )
    assert behavior.current_triggered_zones == set()

    inside = _make_person(1, (40, 40, 60, 60))
    behavior.process_frame([inside], None, 0, 0.0)
    assert behavior.current_triggered_zones == {"test_zone"}

    outside = _make_person(1, (200, 200, 220, 220))
    behavior.process_frame([outside], None, 1, 0.033)
    assert behavior.current_triggered_zones == {"test_zone"}

    behavior.process_frame([outside], None, 4, 0.132)
    assert behavior.current_triggered_zones == set()


def test_danger_zone_multiple_zones_accumulate():
    zone_a = Zone(name="zone_a", label="A", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    zone_b = Zone(name="zone_b", label="B", points=[[50, 50], [200, 50], [200, 200], [50, 200]])
    behavior = DangerZoneBehavior({"min_enter_frames": 2, "min_exit_frames": 5}, zones=[zone_a, zone_b])

    person = _make_person(1, (60, 60, 80, 80))
    r = behavior.detect_person(person, None, 0, 0.0)
    assert not r.detected

    r = behavior.detect_person(person, None, 1, 0.033)
    assert r.detected
    assert r.metadata["zone"] == "zone_a"
    assert r.metadata["triggered_zones"] == ["zone_a", "zone_b"]
    assert r.metadata["inside"]