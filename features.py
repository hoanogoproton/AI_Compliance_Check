"""Feature extraction v3 for keypoint-based behavior classifiers.

Produces a fixed-size, scale- and time-normalized feature vector from a
sequence of YOLO-pose keypoints + bounding boxes. Designed to feed a simple
MLP (one model per behavior, one-vs-rest labeling).

Public entry points:
    - compute_sequence_scale(keypoints_seq, bboxes_seq, valid_mask) -> (scale, source)
    - get_root_center(kpts, bbox) -> (x, y) | None
    - extract_per_frame_features(...) -> np.ndarray (F,)
    - compute_normalized_velocity(...) -> np.ndarray (T, 6)
    - resample_sequence(arr, target_len) -> np.ndarray
    - extract_sequence_features(...) -> np.ndarray (D,)   # temporal mode
    - extract_aggregated_features(...) -> np.ndarray (D,) # aggregation mode
"""

from __future__ import annotations

import numpy as np

from detection.pose_utils import (
    compute_head_center,
    compute_head_yaw_offset,
    compute_shoulder_angle,
    compute_shoulder_width,
    face_visibility_score,
    get_keypoint,
)

SCALE_SOURCE_KEYS = [
    "shoulder_width",
    "torso_length",
    "bbox_height",
    "bbox_diagonal",
    "fallback",
    "fallback_none",
]

DEFAULT_TARGET_LEN = 32
_PER_FRAME_DIM = 97
_VELOCITY_DIM = 6


def compute_sequence_scale(
    keypoints_seq: np.ndarray,
    bboxes_seq: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[float, str]:
    """Compute a single stable scale for the whole sequence.

    Priority chain: shoulder_width -> torso_length -> bbox_height ->
    bbox_diagonal -> 1.0. Uses median over valid frames to stay robust to
    transient occlusion / bad detections.
    """
    if valid_mask is None:
        valid_idx = np.arange(keypoints_seq.shape[0])
    else:
        valid_idx = np.where(np.asarray(valid_mask, dtype=bool))[0]
    if len(valid_idx) == 0:
        return 1.0, "fallback_none"

    # Level 1: shoulder width median
    sw_list = []
    for t in valid_idx:
        sw = compute_shoulder_width(keypoints_seq[t], kpt_conf_threshold=0.3)
        if sw > 10.0:
            sw_list.append(sw)
    if len(sw_list) >= 3:
        return float(np.median(sw_list)), "shoulder_width"

    # Level 2: torso length (hip center -> shoulder center)
    torso_list = []
    for t in valid_idx:
        k = keypoints_seq[t]
        ls = k[5]; rs = k[6]; lh = k[11]; rh = k[12]
        if ls[2] > 0.3 and rs[2] > 0.3 and lh[2] > 0.3 and rh[2] > 0.3:
            shoulder_center_y = (ls[1] + rs[1]) / 2
            hip_center_y = (lh[1] + rh[1]) / 2
            torso = abs(shoulder_center_y - hip_center_y)
            if torso > 10.0:
                torso_list.append(torso)
    if len(torso_list) >= 3:
        return float(np.median(torso_list)), "torso_length"

    # Level 3: bbox height median
    bh_list = [float(bboxes_seq[t, 3] - bboxes_seq[t, 1]) for t in valid_idx]
    bh_median = float(np.median(bh_list)) if bh_list else 0.0
    if bh_median > 20.0:
        return bh_median, "bbox_height"

    # Level 4: bbox diagonal median
    diag_list = [
        float(np.sqrt((bboxes_seq[t, 2] - bboxes_seq[t, 0]) ** 2
                      + (bboxes_seq[t, 3] - bboxes_seq[t, 1]) ** 2))
        for t in valid_idx
    ]
    diag_median = float(np.median(diag_list)) if diag_list else 0.0
    if diag_median > 20.0:
        return diag_median, "bbox_diagonal"

    return 1.0, "fallback"


def get_root_center(
    keypoints: np.ndarray, bbox: np.ndarray | None
) -> tuple[float, float] | None:
    """Stable root center: mid-hip -> mid-shoulder -> bbox center."""
    lhip, rhip = keypoints[11], keypoints[12]
    if lhip[2] > 0.3 and rhip[2] > 0.3:
        return ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)
    lsho, rsho = keypoints[5], keypoints[6]
    if lsho[2] > 0.3 and rsho[2] > 0.3:
        return ((lsho[0] + rsho[0]) / 2, (lsho[1] + rsho[1]) / 2)
    if bbox is not None:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return None


def extract_per_frame_features(
    kpts: np.ndarray,
    bbox: np.ndarray | None,
    root_center: tuple[float, float] | None,
    inv_scale: float,
    head_center: tuple[float, float] | None,
) -> np.ndarray:
    """Per-frame feature vector (~97 dims)."""
    features: list[float] = []

    # Keypoint features: [x_norm, y_norm, conf, visible] x 17 = 68
    for i in range(17):
        x, y, c = kpts[i]
        if root_center is not None and c > 0.1:
            xn = (x - root_center[0]) * inv_scale
            yn = (y - root_center[1]) * inv_scale
        else:
            xn, yn = 0.0, 0.0
        visible = 1.0 if c > 0.3 else 0.0
        features += [xn, yn, float(c), visible]

    # Wrist-specific (for hand_to_head) x 2 wrists x 7 = 14
    for wrist_idx in [9, 10]:
        wx, wy, wc = kpts[wrist_idx]
        if head_center is not None and wc > 0.3:
            dx = (wx - head_center[0]) * inv_scale
            dy = (wy - head_center[1]) * inv_scale
            dist = float(np.sqrt(dx * dx + dy * dy))
        else:
            dx, dy, dist = 0.0, 0.0, -1.0
        if root_center is not None and wc > 0.3:
            rx = (wx - root_center[0]) * inv_scale
            ry = (wy - root_center[1]) * inv_scale
            rdist = float(np.sqrt(rx * rx + ry * ry))
        else:
            rx, ry, rdist = 0.0, 0.0, -1.0
        features += [dx, dy, dist, rx, ry, rdist, float(wc)]

    # Semantic features
    yaw = compute_head_yaw_offset(kpts)
    features.append(yaw[0] if yaw is not None else 0.0)

    angle = compute_shoulder_angle(kpts)
    features.append(angle if angle is not None else -1.0)

    features.append(float(face_visibility_score(kpts)))

    # Confidence stats per body group: [mean, std, visible_ratio] x 3 = 9
    for indices in ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16]):
        confs = [float(kpts[i, 2]) for i in indices]
        visible = sum(1 for c in confs if c > 0.3)
        features += [float(np.mean(confs)), float(np.std(confs)), visible / len(indices)]

    # Absolute size from bbox (3)
    if bbox is not None:
        bw = float(bbox[2] - bbox[0])
        bh = float(bbox[3] - bbox[1])
        features += [bw, bh, float(np.sqrt(bw * bw + bh * bh))]
    else:
        features += [0.0, 0.0, 0.0]

    return np.array(features, dtype=np.float32)


def compute_normalized_velocity(
    keypoints_seq: np.ndarray,
    timestamps: np.ndarray,
    inv_scale: float,
    wrist_indices: list[int] | None = None,
    root_center_seq: list[tuple[float, float] | None] | None = None,
) -> np.ndarray:
    """Velocity normalized by (scale * delta_time). Returns (T, 6).

    Columns: [lw_dx, lw_dy, rw_dx, rw_dy, root_dx, root_dy]. First row is zero.
    """
    if wrist_indices is None:
        wrist_indices = [9, 10]
    T = keypoints_seq.shape[0]
    vel = np.zeros((T, _VELOCITY_DIM), dtype=np.float32)
    for t in range(1, T):
        dt = float(timestamps[t] - timestamps[t - 1])
        if dt <= 0:
            dt = 1.0 / 30.0
        scale_dt = inv_scale / dt
        for i, wi in enumerate(wrist_indices):
            vel[t, i * 2] = (keypoints_seq[t, wi, 0] - keypoints_seq[t - 1, wi, 0]) * scale_dt
            vel[t, i * 2 + 1] = (keypoints_seq[t, wi, 1] - keypoints_seq[t - 1, wi, 1]) * scale_dt
        if (
            root_center_seq is not None
            and root_center_seq[t - 1] is not None
            and root_center_seq[t] is not None
        ):
            vel[t, 4] = (root_center_seq[t][0] - root_center_seq[t - 1][0]) * scale_dt
            vel[t, 5] = (root_center_seq[t][1] - root_center_seq[t - 1][1]) * scale_dt
    return vel


def resample_sequence(arr: np.ndarray, target_len: int = DEFAULT_TARGET_LEN) -> np.ndarray:
    """Resample (T, F) or (T,) to (target_len, F) / (target_len,) via linear interp."""
    T = arr.shape[0]
    if T == target_len:
        return arr
    if T < 2:
        if arr.ndim == 2:
            return np.tile(arr, (target_len, 1)) if T == 1 else np.zeros((target_len, arr.shape[1]), dtype=arr.dtype)
        return np.tile(arr, target_len) if T == 1 else np.zeros(target_len, dtype=arr.dtype)
    orig_idx = np.linspace(0, T - 1, T)
    target_idx = np.linspace(0, T - 1, target_len)
    if arr.ndim == 2:
        resampled = np.zeros((target_len, arr.shape[1]), dtype=arr.dtype)
        for f in range(arr.shape[1]):
            resampled[:, f] = np.interp(target_idx, orig_idx, arr[:, f])
        return resampled
    return np.interp(target_idx, orig_idx, arr).astype(arr.dtype)


def _build_global_features(
    scale: float,
    scale_source: str,
    timestamps: np.ndarray,
    valid_ratio: float,
) -> np.ndarray:
    global_features: list[float] = [scale]
    if len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
        eff_fps = (len(timestamps) - 1) / float(timestamps[-1] - timestamps[0])
        duration = float(timestamps[-1] - timestamps[0])
    else:
        eff_fps = 30.0
        duration = 0.0
    global_features += [eff_fps, duration, valid_ratio]
    for sk in SCALE_SOURCE_KEYS:
        global_features.append(1.0 if scale_source == sk else 0.0)
    return np.array(global_features, dtype=np.float32)


def extract_sequence_features(
    keypoints_seq: np.ndarray,
    bboxes_seq: np.ndarray,
    timestamps: np.ndarray,
    valid_mask: np.ndarray | None = None,
    target_len: int = DEFAULT_TARGET_LEN,
) -> np.ndarray:
    """Temporal mode: resample to fixed length then flatten + velocity + global.

    Returns a 1D float32 vector of dim
    target_len * _PER_FRAME_DIM + target_len * _VELOCITY_DIM + len(global).
    """
    T = keypoints_seq.shape[0]
    if T == 0:
        raise ValueError("Empty keypoint sequence")

    if valid_mask is None:
        valid_mask = np.ones(T, dtype=bool)

    scale, scale_source = compute_sequence_scale(keypoints_seq, bboxes_seq, valid_mask)
    inv_scale = 1.0 / scale if scale > 0 else 1.0

    root_centers = [get_root_center(keypoints_seq[t], bboxes_seq[t]) for t in range(T)]
    head_centers = [compute_head_center(keypoints_seq[t]) for t in range(T)]

    per_frame = np.stack(
        [
            extract_per_frame_features(
                keypoints_seq[t], bboxes_seq[t], root_centers[t], inv_scale, head_centers[t]
            )
            for t in range(T)
        ],
        axis=0,
    )

    per_frame_resampled = resample_sequence(per_frame, target_len)

    vel_raw = compute_normalized_velocity(keypoints_seq, timestamps, inv_scale, root_center_seq=root_centers)
    vel_resampled = resample_sequence(vel_raw, target_len)

    valid_ratio = float(np.mean(valid_mask)) if T > 0 else 0.0
    global_features = _build_global_features(scale, scale_source, timestamps, valid_ratio)

    temporal = np.concatenate([per_frame_resampled.ravel(), vel_resampled.ravel()])
    return np.concatenate([temporal, global_features]).astype(np.float32)


def extract_aggregated_features(
    keypoints_seq: np.ndarray,
    bboxes_seq: np.ndarray,
    timestamps: np.ndarray,
    valid_mask: np.ndarray | None = None,
    target_len: int = DEFAULT_TARGET_LEN,
) -> np.ndarray:
    """Aggregation mode: summarize per-frame + velocity with stats + global.

    Lower-dimensional alternative to temporal mode for small datasets.
    """
    T = keypoints_seq.shape[0]
    if T == 0:
        raise ValueError("Empty keypoint sequence")
    if valid_mask is None:
        valid_mask = np.ones(T, dtype=bool)

    scale, scale_source = compute_sequence_scale(keypoints_seq, bboxes_seq, valid_mask)
    inv_scale = 1.0 / scale if scale > 0 else 1.0

    root_centers = [get_root_center(keypoints_seq[t], bboxes_seq[t]) for t in range(T)]
    head_centers = [compute_head_center(keypoints_seq[t]) for t in range(T)]

    per_frame = np.stack(
        [
            extract_per_frame_features(
                keypoints_seq[t], bboxes_seq[t], root_centers[t], inv_scale, head_centers[t]
            )
            for t in range(T)
        ],
        axis=0,
    )
    vel_raw = compute_normalized_velocity(keypoints_seq, timestamps, inv_scale, root_center_seq=root_centers)

    def _stats(x: np.ndarray) -> np.ndarray:
        # mean, std, min, max along time
        return np.concatenate(
            [x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0)]
        ).astype(np.float32)

    valid_ratio = float(np.mean(valid_mask)) if T > 0 else 0.0
    global_features = _build_global_features(scale, scale_source, timestamps, valid_ratio)

    return np.concatenate([_stats(per_frame), _stats(vel_raw), global_features]).astype(np.float32)


def extract_features(
    keypoints_seq: np.ndarray,
    bboxes_seq: np.ndarray,
    timestamps: np.ndarray,
    valid_mask: np.ndarray | None = None,
    target_len: int = DEFAULT_TARGET_LEN,
    mode: str = "temporal",
) -> np.ndarray:
    """Dispatch helper used by training and inference."""
    if mode == "agg" or mode == "aggregation":
        return extract_aggregated_features(keypoints_seq, bboxes_seq, timestamps, valid_mask, target_len)
    return extract_sequence_features(keypoints_seq, bboxes_seq, timestamps, valid_mask, target_len)


def feature_dim(target_len: int = DEFAULT_TARGET_LEN, mode: str = "temporal") -> int:
    """Return the feature dimension for a given mode/target_len without computing."""
    n_global = 4 + len(SCALE_SOURCE_KEYS)
    if mode == "agg" or mode == "aggregation":
        return 4 * _PER_FRAME_DIM + 4 * _VELOCITY_DIM + n_global
    return target_len * _PER_FRAME_DIM + target_len * _VELOCITY_DIM + n_global


__all__ = [
    "compute_sequence_scale",
    "get_root_center",
    "extract_per_frame_features",
    "compute_normalized_velocity",
    "resample_sequence",
    "extract_sequence_features",
    "extract_aggregated_features",
    "extract_features",
    "feature_dim",
    "SCALE_SOURCE_KEYS",
    "DEFAULT_TARGET_LEN",
]
