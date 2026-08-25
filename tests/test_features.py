import numpy as np

from features import (
    DEFAULT_TARGET_LEN,
    compute_sequence_scale,
    extract_features,
    extract_aggregated_features,
    extract_sequence_features,
    feature_dim,
    get_root_center,
    resample_sequence,
)


def _synth_seq(T=24, seed=0, shoulder_visible=True):
    rng = np.random.default_rng(seed)
    kp = np.zeros((T, 17, 3), dtype=np.float32)
    kp[:, 5] = [100, 200, 0.9] if shoulder_visible else [100, 200, 0.0]
    kp[:, 6] = [160, 200, 0.9] if shoulder_visible else [160, 200, 0.0]
    kp[:, 11] = [110, 320, 0.9]
    kp[:, 12] = [150, 320, 0.9]
    kp[:, :, :2] += rng.normal(0, 1.0, (T, 17, 2)).astype(np.float32)
    bb = np.tile([80.0, 130, 260, 360], (T, 1)).astype(np.float32)
    ts = np.arange(T, dtype=np.float32) / 30.0
    vm = np.ones(T, dtype=bool)
    return kp, bb, ts, vm


def test_feature_dims_match():
    kp, bb, ts, vm = _synth_seq()
    for mode in ("temporal", "agg"):
        f = extract_features(kp, bb, ts, vm, target_len=32, mode=mode)
        assert f.ndim == 1
        assert f.shape[0] == feature_dim(32, mode), (mode, f.shape[0], feature_dim(32, mode))


def test_temporal_dim_formula():
    assert feature_dim(32, "temporal") == 32 * 97 + 32 * 6 + 4 + 6
    assert feature_dim(32, "agg") == 4 * 97 + 4 * 6 + 4 + 6


def test_resample_sequence_length():
    arr = np.arange(24, dtype=np.float32).reshape(24, 1)
    r = resample_sequence(arr, target_len=32)
    assert r.shape == (32, 1)
    r2 = resample_sequence(np.arange(10, dtype=np.float32), target_len=5)
    assert r2.shape == (5,)


def test_scale_priority_shoulder():
    kp, bb, ts, vm = _synth_seq(shoulder_visible=True)
    scale, source = compute_sequence_scale(kp, bb, vm)
    assert source == "shoulder_width"
    assert 50.0 < scale < 70.0


def test_scale_fallback_to_bbox():
    kp, bb, ts, vm = _synth_seq(shoulder_visible=False)
    # zero hip conf too so torso fails -> bbox_height
    kp[:, 11, 2] = 0.0
    kp[:, 12, 2] = 0.0
    scale, source = compute_sequence_scale(kp, bb, vm)
    assert source == "bbox_height"
    assert scale > 20.0


def test_root_center_priority():
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[11] = [10, 20, 0.9]
    kp[12] = [30, 20, 0.9]
    rc = get_root_center(kp, None)
    assert rc == (20.0, 20.0)
    # falls back to bbox when hips/shoulders invisible
    kp[11, 2] = 0.0
    kp[12, 2] = 0.0
    rc2 = get_root_center(kp, np.array([0, 0, 100, 200], dtype=np.float32))
    assert rc2 == (50.0, 100.0)


def test_constant_features_for_identical_sequence():
    kp, bb, ts, vm = _synth_seq()
    f1 = extract_sequence_features(kp, bb, ts, vm, target_len=16)
    f2 = extract_sequence_features(kp, bb, ts, vm, target_len=16)
    assert np.allclose(f1, f2)


def test_aggregation_smaller_than_temporal():
    kp, bb, ts, vm = _synth_seq()
    agg = extract_aggregated_features(kp, bb, ts, vm, target_len=32)
    temp = extract_sequence_features(kp, bb, ts, vm, target_len=32)
    assert agg.shape[0] < temp.shape[0]
