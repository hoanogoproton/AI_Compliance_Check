from detection.behaviors.head_shake import YawShakeTracker


def test_steady_yaw_no_reversals():
    tracker = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=8.0,
        smoothing_alpha=0.3, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    for i in range(50):
        tracker.update(0.0, i)
    assert tracker.reversal_count == 0


def test_small_noise_below_threshold():
    tracker = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=8.0,
        smoothing_alpha=1.0, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    import random
    random.seed(42)
    for i in range(50):
        noise = random.uniform(-2.0, 2.0)
        tracker.update(noise, i)
    assert tracker.reversal_count == 0


def test_yaw_sequence_three_reversals():
    tracker = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=8.0,
        smoothing_alpha=0.3, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    for i, yaw in enumerate([0, 30, -25, 28, -30]):
        tracker.update(yaw, i)
    assert tracker.reversal_count == 3


def test_yaw_below_amplitude_threshold():
    tracker = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=20.0,
        smoothing_alpha=1.0, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    for i, yaw in enumerate([0, 5, -4, 5]):
        tracker.update(yaw, i)
    assert tracker.reversal_count == 0


def test_single_direction_no_reversal():
    tracker = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=8.0,
        smoothing_alpha=0.3, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    for i in range(10):
        tracker.update(20.0, i)
    assert tracker.reversal_count == 0


def test_gap_exceeds_max_resets_direction():
    tracker = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=8.0,
        smoothing_alpha=1.0, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    tracker.update(0.0, 0)
    tracker.update(15.0, 1)
    tracker.update(-15.0, 2)
    reversals_before = tracker.reversal_count
    # gap > max_gap_frames (5): skip frames 3-9, resume at 10
    for i in range(10, 20):
        tracker.update(0.0, i)
    reversals_after = tracker.reversal_count
    assert reversals_after == reversals_before


def test_missing_face_within_gap_preserves_state():
    tracker = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=8.0,
        smoothing_alpha=1.0, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    tracker.update(0.0, 0)
    tracker.update(15.0, 1)
    tracker.update(None, 2)
    tracker.update(None, 3)
    tracker.update(-15.0, 4)
    assert tracker.reversal_count == 1


def test_multiple_trackers_independent():
    t1 = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=8.0,
        smoothing_alpha=1.0, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    t2 = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=8.0,
        smoothing_alpha=1.0, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    t1.update(0.0, 0)
    t1.update(20.0, 1)
    t1.update(-20.0, 2)
    t2.update(0.0, 0)
    t2.update(0.0, 1)
    assert t1.reversal_count == 1
    assert t2.reversal_count == 0


def test_old_reversals_pruned():
    tracker = YawShakeTracker(
        window_frames=5, yaw_amplitude_threshold=8.0,
        smoothing_alpha=1.0, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    tracker.update(0.0, 0)
    tracker.update(20.0, 1)
    tracker.update(-20.0, 2)
    assert tracker.reversal_count == 1
    # advance past window with zero yaw (no new reversals)
    for i in range(10, 20):
        tracker.update(0.0, i)
    assert tracker.reversal_count == 0


def test_reversal_count_eq_min_reversals():
    tracker = YawShakeTracker(
        window_frames=35, yaw_amplitude_threshold=8.0,
        smoothing_alpha=1.0, ema_deadband=0.5, max_gap_frames=5,
        min_reversals=4,
    )
    yaws = [0.0, 20.0, -20.0, 20.0, -20.0, 20.0, -20.0, 20.0, -20.0]
    for i, yaw in enumerate(yaws):
        tracker.update(yaw, i)
    assert tracker.reversal_count >= 4