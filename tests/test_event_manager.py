from handhead.event_manager import StatefulEventManager


def _d(tid, detected, conf=0.8, side="right"):
    return {tid: (detected, conf, side)}


def test_idle_to_active_on_detection():
    mgr = StatefulEventManager()
    result = mgr.update(0, 0.0, _d(1, True))
    assert result == []


def test_accumulate_gap_then_cooldown():
    mgr = StatefulEventManager()
    for i in range(6):
        mgr.update(i, i / 30.0, _d(1, True))
    all_completed = []
    for i in range(6, 18):
        result = mgr.update(i, i / 30.0, _d(1, False))
        all_completed.extend(result)
    assert len(all_completed) == 1
    assert all_completed[0].track_id == 1
    assert all_completed[0].start_frame == 0
    assert all_completed[0].end_frame == 5


def test_gap_within_max_tolerated():
    mgr = StatefulEventManager()
    mgr.update(0, 0.0, _d(1, True))
    for i in range(1, 8):
        mgr.update(i, i / 30.0, _d(1, True))
    result = mgr.update(8, 8 / 30.0, _d(1, False))
    assert result == []


def test_discard_short_event():
    mgr = StatefulEventManager()
    mgr.update(0, 0.0, _d(1, True))
    for i in range(1, 4):
        detected = i < 3
        mgr.update(i, i / 30.0, _d(1, detected))
    result = mgr.update(5, 5 / 30.0, _d(1, False))
    assert result == []


def test_multiple_tracks_independent():
    mgr = StatefulEventManager()
    mgr.update(0, 0.0, {1: (True, 0.8, "right"), 2: (True, 0.9, "left")})
    for i in range(1, 6):
        mgr.update(i, i / 30.0, {1: (True, 0.8, "right"), 2: (True, 0.9, "left")})
    all_completed = []
    for i in range(6, 18):
        result = mgr.update(i, i / 30.0, {1: (False, 0, "none"), 2: (True, 0.9, "left")})
        all_completed.extend(result)
    all_completed.extend(mgr.finalize())
    track1_events = [e for e in all_completed if e.track_id == 1]
    assert len(track1_events) == 1


def test_finalize_flushes_active_events():
    mgr = StatefulEventManager()
    for i in range(10):
        mgr.update(i, i / 30.0, _d(1, True))
    remaining = mgr.finalize()
    assert len(remaining) == 1
    assert remaining[0].track_id == 1
    assert len(remaining[0].frames) >= 5