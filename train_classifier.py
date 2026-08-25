"""Train a one-vs-rest keypoint classifier for a single behavior.

Usage:
    python train_classifier.py --dataset ./dataset/ --behavior hand_to_head --output ./models/
    python train_classifier.py --dataset ./dataset/ --behavior body_turn --output ./models/ --mode agg

Each behavior gets its own model directory:
    models/<behavior>/
        classifier.pkl   # sklearn MLPClassifier
        scaler.pkl       # StandardScaler
        metadata.json    # threshold, feature_dim, val metrics, ...
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

from detection.config import (
    CLASSIFIER_DEFAULT_MODE,
    CLASSIFIER_DEFAULT_THRESHOLD,
    CLASSIFIER_MIN_DURATION_SEC,
    CLASSIFIER_MIN_SEQUENCE_FRAMES,
    CLASSIFIER_MIN_TRACK_QUALITY,
    CLASSIFIER_TARGET_LEN,
)
from features import DEFAULT_TARGET_LEN, extract_features, feature_dim

FEATURE_VERSION = "v3.0.0"
POSE_MODEL_VERSION_DEFAULT = "yolo26m-pose"

METADATA_COLUMNS = [
    "sample_id", "video_id", "camera_id", "recording_session", "annotator_id",
    "annotation_status", "track_id", "start_frame", "end_frame", "label", "behavior",
    "pose_model_version", "tracker_config_hash", "quality_score", "duration_sec", "fps",
]


def _load_metadata(dataset_dir: Path) -> list[dict]:
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    with open(meta_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        raise ValueError("metadata.csv is empty")
    # Coerce numeric fields
    for r in rows:
        for k in ("label", "track_id", "start_frame", "end_frame"):
            if k in r and r[k] != "":
                r[k] = int(r[k])
        for k in ("quality_score", "duration_sec", "fps"):
            if k in r and r[k] != "":
                r[k] = float(r[k])
    return rows


def _load_sample(dataset_dir: Path, sample_id: int | str) -> dict | None:
    sample_path = dataset_dir / "samples" / f"sample_{int(sample_id):04d}.npz"
    if not sample_path.exists():
        return None
    with np.load(sample_path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    return out


def _str_field(row: dict, key: str, default: str = "") -> str:
    val = row.get(key, default)
    if val is None or val == "":
        return default
    return str(val)


def _is_negative_behavior(row: dict, behavior: str) -> bool:
    """One-vs-rest: positive iff row.behavior == behavior (and label==1)."""
    row_behavior = _str_field(row, "behavior")
    row_label = row.get("label", 0)
    try:
        row_label = int(row_label)
    except (TypeError, ValueError):
        row_label = 0
    # A sample is positive only if it matches the target behavior and is
    # annotated as a positive (label==1) example. Everything else is negative.
    return not (row_behavior == behavior and row_label == 1)


def _quality_ok(sample: dict, min_track_quality: float, min_duration: float) -> bool:
    valid_mask = sample.get("valid_mask")
    if valid_mask is not None:
        ratio = float(np.mean(np.asarray(valid_mask, dtype=bool)))
        if ratio < min_track_quality:
            return False
    timestamps = sample.get("timestamps")
    if timestamps is not None and len(timestamps) > 1:
        duration = float(timestamps[-1] - timestamps[0])
        if duration < min_duration:
            return False
    return True


def _augment_sample(
    keypoints_seq: np.ndarray,
    bboxes_seq: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Light data augmentation: random scale, keypoint jitter, random drop frame."""
    T = keypoints_seq.shape[0]
    kpts = keypoints_seq.copy()
    bboxes = bboxes_seq.copy()

    # Random scale 0.7x - 1.5x applied to normalized-relative features is a no-op
    # here (features are scale-normalized internally), but we still jitter
    # the absolute bbox/keypoint magnitudes to teach robustness.
    scale_factor = float(rng.uniform(0.7, 1.5))
    kpts[:, :, :2] = kpts[:, :, :2] * scale_factor
    bboxes = bboxes * scale_factor

    # Random noise +-2px on keypoint coordinates
    noise = rng.uniform(-2.0, 2.0, size=kpts[:, :, :2].shape).astype(kpts.dtype)
    kpts[:, :, :2] = kpts[:, :, :2] + noise

    # Random drop frame (simulate occlusion) with prob 0.1, zeroing keypoints
    if T > 4:
        drop_mask = rng.random(T) < 0.1
        if drop_mask.any():
            kpts[drop_mask] = 0.0
            bboxes[drop_mask] = 0.0
    return kpts, bboxes


def build_dataset(
    dataset_dir: Path,
    behavior: str,
    target_len: int,
    mode: str,
    min_track_quality: float,
    min_duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Return (X, y, video_ids, kept_rows). One-vs-rest labeling."""
    rows = _load_metadata(dataset_dir)
    samples_dir = dataset_dir / "samples"

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    vid_list: list[str] = []
    kept_rows: list[dict] = []

    skipped = 0
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is None or sample_id == "":
            continue
        try:
            sid = int(sample_id)
        except (TypeError, ValueError):
            continue
        sample_path = samples_dir / f"sample_{sid:04d}.npz"
        if not sample_path.exists():
            skipped += 1
            continue
        with np.load(sample_path, allow_pickle=True) as data:
            keypoints_seq = data["keypoints"]
            bboxes_seq = data["bboxes"]
            timestamps = data["timestamps"]
            valid_mask = data["valid_mask"] if "valid_mask" in data.files else None

        sample_dict = {
            "keypoints": keypoints_seq,
            "bboxes": bboxes_seq,
            "timestamps": timestamps,
            "valid_mask": valid_mask,
        }
        if not _quality_ok(sample_dict, min_track_quality, min_duration):
            skipped += 1
            continue

        is_negative = _is_negative_behavior(row, behavior)
        label = 0 if is_negative else 1

        try:
            feat = extract_features(
                keypoints_seq, bboxes_seq, timestamps, valid_mask,
                target_len=target_len, mode=mode,
            )
        except Exception:  # noqa: BLE001
            skipped += 1
            continue

        X_list.append(feat)
        y_list.append(label)
        vid_list.append(_str_field(row, "video_id", f"unknown_{sid}"))
        kept_rows.append(row)

    if not X_list:
        raise ValueError(
            f"No usable samples for behavior '{behavior}' in {dataset_dir} "
            f"(skipped {skipped} samples)"
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    video_ids = np.array(vid_list, dtype=object)
    print(f"[Dataset] behavior='{behavior}' kept={len(y)} skipped={skipped} "
          f"positives={int(y.sum())} negatives={int((y == 0).sum())}")
    return X, y, video_ids, kept_rows


def _group_split_by_video(
    video_ids: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split unique video_ids into train/val, returning sample index arrays.

    Stratified by majority class within each video to keep class balance stable.
    """
    unique_vids = np.unique(video_ids)
    if len(unique_vids) < 2:
        # Not enough videos to split without leakage; fall back to random split.
        idx = np.arange(len(video_ids))
        train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=seed,
                                              stratify=y if len(np.unique(y)) > 1 else None)
        return train_idx, val_idx

    # Score each video by its positive ratio for a stratified group split.
    vid_positive_ratio = []
    for vid in unique_vids:
        mask = video_ids == vid
        vid_positive_ratio.append(float(np.mean(y[mask] == 1)))
    vid_labels = np.array([1 if r >= 0.5 else 0 for r in vid_positive_ratio])

    try:
        train_vids, val_vids = train_test_split(
            unique_vids, test_size=test_size, random_state=seed,
            stratify=vid_labels if len(np.unique(vid_labels)) > 1 else None,
        )
    except ValueError:
        train_vids, val_vids = train_test_split(unique_vids, test_size=test_size, random_state=seed)

    train_idx = np.array([i for i, v in enumerate(video_ids) if v in set(train_vids.tolist())])
    val_idx = np.array([i for i, v in enumerate(video_ids) if v in set(val_vids.tolist())])
    return train_idx, val_idx


def _tune_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_precision: float) -> tuple[float, dict]:
    """Pick threshold maximizing F1 subject to precision >= min_precision."""
    if len(np.unique(y_true)) < 2:
        # Only one class present in val; fall back to default threshold.
        return CLASSIFIER_DEFAULT_THRESHOLD, {"threshold": CLASSIFIER_DEFAULT_THRESHOLD, "note": "single_class_val"}

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    best_f1 = -1.0
    best_thr = CLASSIFIER_DEFAULT_THRESHOLD
    # precision_recall_curve returns thresholds with len = len(precision) - 1
    for i in range(len(thresholds)):
        p = precision[i]
        r = recall[i]
        if p < min_precision:
            continue
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thresholds[i])
    metrics = {
        "threshold": best_thr,
        "val_f1": float(best_f1) if best_f1 >= 0 else 0.0,
        "note": "tuned",
    }
    return best_thr, metrics


def train(
    dataset_dir: Path,
    behavior: str,
    output_dir: Path,
    mode: str = CLASSIFIER_DEFAULT_MODE,
    target_len: int = CLASSIFIER_TARGET_LEN,
    min_track_quality: float = CLASSIFIER_MIN_TRACK_QUALITY,
    min_duration: float = CLASSIFIER_MIN_DURATION_SEC,
    min_sequence_frames: int = CLASSIFIER_MIN_SEQUENCE_FRAMES,
    test_size: float = 0.2,
    seed: int = 42,
    min_precision: float = 0.85,
    augment: bool = True,
    hidden: tuple[int, ...] | None = None,
    alpha: float = 0.01,
    max_iter: int = 2000,
) -> dict:
    X, y, video_ids, kept_rows = build_dataset(
        dataset_dir, behavior, target_len, mode, min_track_quality, min_duration,
    )

    if len(X) < min_sequence_frames:
        raise ValueError(f"Too few samples ({len(X)}) to train a model")

    feat_dim = X.shape[1]
    train_idx, val_idx = _group_split_by_video(video_ids, y, test_size=test_size, seed=seed)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"[Split] train={len(y_train)} (pos={int(y_train.sum())}) "
          f"val={len(y_val)} (pos={int(y_val.sum())})")

    # Augmentation: create extra positive (and a few negative) copies to fight
    # class imbalance, in addition to class_weight='balanced'.
    if augment:
        rng = np.random.default_rng(seed)
        aug_X, aug_y = [], []
        # Oversample positives up to ~ parity with negatives when very imbalanced.
        pos_idx = np.where(y_train == 1)[0]
        neg_count = int(np.sum(y_train == 0))
        target_pos = max(len(pos_idx), neg_count)
        if len(pos_idx) > 0 and target_pos > len(pos_idx):
            for _ in range(target_pos - len(pos_idx)):
                src = int(pos_idx[rng.integers(0, len(pos_idx))])
                row = kept_rows[int(train_idx[src])]
                sid = int(row["sample_id"]) if row is not None else -1
                sample_path = dataset_dir / "samples" / f"sample_{sid:04d}.npz"
                if not sample_path.exists():
                    continue
                with np.load(sample_path, allow_pickle=True) as data:
                    kp = data["keypoints"].copy()
                    bb = data["bboxes"].copy()
                    ts = data["timestamps"]
                    vm = data["valid_mask"] if "valid_mask" in data.files else None
                kp_aug, bb_aug = _augment_sample(kp, bb, rng)
                try:
                    feat = extract_features(kp_aug, bb_aug, ts, vm, target_len=target_len, mode=mode)
                except Exception:  # noqa: BLE001
                    continue
                aug_X.append(feat)
                aug_y.append(1)
        if aug_X:
            X_train = np.concatenate([X_train, np.stack(aug_X)], axis=0)
            y_train = np.concatenate([y_train, np.array(aug_y, dtype=np.int32)], axis=0)
            print(f"[Augment] added {len(aug_X)} augmented positive samples "
                  f"-> train={len(y_train)} (pos={int(y_train.sum())})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    if hidden is None:
        if mode in ("agg", "aggregation"):
            hidden = (256, 128, 64)
        else:
            hidden = (512, 256, 128)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=seed,
    )
    clf.fit(X_train_s, y_train)

    # Predict probabilities on val
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_val_s)[:, 1]
    elif hasattr(clf, "decision_function"):
        d = clf.decision_function(X_val_s)
        y_proba = 1.0 / (1.0 + np.exp(-d))
    else:
        y_proba = clf.predict(X_val_s).astype(float)

    threshold, thr_meta = _tune_threshold(y_val, y_proba, min_precision)

    y_pred = (y_proba >= threshold).astype(int)
    val_f1 = float(f1_score(y_val, y_pred, zero_division=0))
    val_precision = float(precision_score(y_val, y_pred, zero_division=0))
    val_recall = float(recall_score(y_val, y_pred, zero_division=0))

    # Camera profiles + scale source distribution
    camera_profiles = sorted({_str_field(r, "camera_id", "unknown") for r in [kept_rows[i] for i in val_idx]})

    print(f"[Result] threshold={threshold:.3f} val_f1={val_f1:.3f} "
          f"precision={val_precision:.3f} recall={val_recall:.3f}")

    # Save model
    model_dir = output_dir / behavior
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "behavior": behavior,
        "model_version": f"mlp-{mode}",
        "feature_version": FEATURE_VERSION,
        "feature_dim": int(feat_dim),
        "mode": mode,
        "target_len": int(target_len),
        "threshold": float(threshold),
        "min_sequence_frames": int(min_sequence_frames),
        "min_track_quality": float(min_track_quality),
        "min_duration_sec": float(min_duration),
        "hidden_layers": list(hidden),
        "alpha": float(alpha),
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_train_pos": int(np.sum(y_train)),
        "n_val_pos": int(np.sum(y_val)),
        "camera_profiles": camera_profiles,
        "pose_model_version": POSE_MODEL_VERSION_DEFAULT,
        "tracker_version": "bytetrack",
        "seed": int(seed),
        "threshold_meta": thr_meta,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Saved] {model_dir}/classifier.pkl, scaler.pkl, metadata.json")
    return metadata


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a per-behavior keypoint classifier")
    p.add_argument("--dataset", required=True, help="Dataset directory containing metadata.csv + samples/")
    p.add_argument("--behavior", required=True, help="Behavior name (e.g. hand_to_head, body_turn)")
    p.add_argument("--output", default="./models/", help="Output models root directory")
    p.add_argument("--mode", default=CLASSIFIER_DEFAULT_MODE, choices=["temporal", "agg"],
                   help="Feature mode: temporal (resample+flatten) or agg (statistics)")
    p.add_argument("--target-len", type=int, default=CLASSIFIER_TARGET_LEN)
    p.add_argument("--min-track-quality", type=float, default=CLASSIFIER_MIN_TRACK_QUALITY)
    p.add_argument("--min-duration", type=float, default=CLASSIFIER_MIN_DURATION_SEC)
    p.add_argument("--min-sequence-frames", type=int, default=CLASSIFIER_MIN_SEQUENCE_FRAMES)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-precision", type=float, default=0.85)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--hidden", type=int, nargs="+", default=None,
                   help="Hidden layer sizes, e.g. --hidden 512 256 128")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--max-iter", type=int, default=2000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    hidden = tuple(args.hidden) if args.hidden else None
    # Sanity: feature_dim consistency
    expected_dim = feature_dim(args.target_len, args.mode)
    print(f"[Config] behavior={args.behavior} mode={args.mode} target_len={args.target_len} "
          f"feature_dim={expected_dim}")
    train(
        dataset_dir=dataset_dir,
        behavior=args.behavior,
        output_dir=output_dir,
        mode=args.mode,
        target_len=args.target_len,
        min_track_quality=args.min_track_quality,
        min_duration=args.min_duration,
        min_sequence_frames=args.min_sequence_frames,
        test_size=args.test_size,
        seed=args.seed,
        min_precision=args.min_precision,
        augment=not args.no_augment,
        hidden=hidden,
        alpha=args.alpha,
        max_iter=args.max_iter,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
