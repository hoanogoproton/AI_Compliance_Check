"""
Inference script: classify behavioral tracks in a video using a trained MLP classifier.

Usage:
    python inference.py --video path/to/video.mp4
    python inference.py --video path/to/video.mp4 --model ./models/move_1_step --visualize
    python inference.py --video path/to/video.mp4 --model ./models/move_1_step --output ./results

Pipeline:
    1. Load YOLO-pose model + ByteTrack
    2. Process every frame, cache per-track keypoints
    3. Slide a window over each track's keypoint sequence
    4. Extract features, run classifier, merge consecutive positives
    5. Print / save suspicious segments
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        if iterable is not None:
            return iterable
        class _Fake:
            @staticmethod
            def update(n=1):
                pass
            @staticmethod
            def close():
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return _Fake()

from detection.config import (
    CLASSIFIER_DEFAULT_MODE,
    CLASSIFIER_DEFAULT_THRESHOLD,
    CLASSIFIER_MIN_SEQUENCE_FRAMES,
    CLASSIFIER_TARGET_LEN,
)
from detection.detector import TrackedPerson, process_frame
from detection.model import load_pose_model
from detection.visualizer import draw_skeleton
from features import extract_features, feature_dim


def load_classifier(model_dir: Path) -> dict:
    """Load classifier.pkl, scaler.pkl, metadata.json from model_dir."""
    model_dir = Path(model_dir)

    classifier_path = model_dir / "classifier.pkl"
    scaler_path = model_dir / "scaler.pkl"
    metadata_path = model_dir / "metadata.json"

    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier not found: {classifier_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with open(classifier_path, "rb") as f:
        clf = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    threshold = float(metadata.get("threshold", CLASSIFIER_DEFAULT_THRESHOLD))
    mode = metadata.get("mode", CLASSIFIER_DEFAULT_MODE)
    target_len = int(metadata.get("target_len", CLASSIFIER_TARGET_LEN))
    min_seq = int(metadata.get("min_sequence_frames", CLASSIFIER_MIN_SEQUENCE_FRAMES))

    print(f"[Classifier] loaded: behavior={metadata.get('behavior', '?')} "
          f"mode={mode} threshold={threshold:.3f} dim={metadata.get('feature_dim', '?')}")

    return {
        "model": clf,
        "scaler": scaler,
        "threshold": threshold,
        "mode": mode,
        "target_len": target_len,
        "min_sequence_frames": min_seq,
        "metadata": metadata,
    }


def predict_proba(model, X: np.ndarray) -> float:
    """Return P(positive class)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0, -1])
    if hasattr(model, "decision_function"):
        d = float(model.decision_function(X)[0])
        return 1.0 / (1.0 + float(np.exp(-d)))
    return float(model.predict(X)[0])


def classify_window(
    keypoints_seq: np.ndarray,
    bboxes_seq: np.ndarray,
    timestamps: np.ndarray,
    valid_mask: np.ndarray,
    clf_cfg: dict,
) -> float:
    """Extract features and return classifier probability."""
    features = extract_features(
        keypoints_seq, bboxes_seq, timestamps, valid_mask,
        target_len=clf_cfg["target_len"],
        mode=clf_cfg["mode"],
    )
    features_scaled = clf_cfg["scaler"].transform([features])
    return predict_proba(clf_cfg["model"], features_scaled)


def run_inference(
    video_path: str,
    model_path: str,
    clf_cfg: dict,
    conf: float = 0.4,
    iou: float = 0.5,
    window_stride: int = 1,
    visualize: bool = False,
    output_dir: str | None = None,
) -> list[dict]:
    """
    Process video and return suspicious track segments.
    
    Returns list of dicts:
        {
            "track_id": int,
            "start_frame": int,
            "end_frame": int,
            "start_time": float,
            "end_time": float,
            "max_prob": float,
            "mean_prob": float,
        }
    """
    pose_model = load_pose_model(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video] {total_frames} frames, {fps:.2f} fps, {frame_w}x{frame_h}")

    # Cache: frame_idx -> track_id -> {"keypoints", "bbox", "conf"}
    frame_cache: dict[int, dict[int, dict]] = {}

    # Process every frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar = tqdm(total=total_frames, desc="Detecting", unit="frame")
    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        people = process_frame(pose_model, frame_rgb, conf=conf, iou=iou)
        track_map: dict[int, dict] = {}
        for person in people:
            track_map[person.track_id] = {
                "keypoints": np.asarray(person.keypoints, dtype=np.float32),
                "bbox": np.array(person.bbox, dtype=np.float32),
                "conf": person.conf,
            }
        frame_cache[frame_idx] = track_map
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"[Cache] {len(frame_cache)} frames, {sum(len(v) for v in frame_cache.values())} track entries")

    # Build per-track frame sequences
    track_frames: dict[int, list[int]] = defaultdict(list)
    for f_idx in sorted(frame_cache.keys()):
        for tid in frame_cache[f_idx]:
            track_frames[tid].append(f_idx)

    # Sliding-window classification per track
    min_seq = clf_cfg["min_sequence_frames"]
    target_len = clf_cfg["target_len"]
    stride = window_stride

    results: list[dict] = []
    suspicious_segments: list[dict] = []

    for tid in sorted(track_frames.keys()):
        frames = sorted(track_frames[tid])
        if len(frames) < min_seq:
            continue

        n_windows = max(0, (len(frames) - target_len) // stride + 1)
        if n_windows == 0:
            n_windows = 1

        window_results: list[tuple[int, int, float]] = []
        for w_idx in range(n_windows):
            start = w_idx * stride
            end = min(start + target_len, len(frames))
            if end - start < min_seq:
                break
            win_frames = frames[start:end]
            T = len(win_frames)
            kpts_seq = np.zeros((T, 17, 3), dtype=np.float32)
            bbox_seq = np.zeros((T, 4), dtype=np.float32)
            ts_seq = np.zeros(T, dtype=np.float32)
            valid_seq = np.zeros(T, dtype=bool)
            for i, f in enumerate(win_frames):
                ts_seq[i] = f / fps
                data = frame_cache.get(f, {}).get(tid)
                if data is None:
                    continue
                kpts_seq[i] = np.asarray(data["keypoints"], dtype=np.float32)
                bbox_seq[i] = np.asarray(data["bbox"], dtype=np.float32)
                valid_seq[i] = True

            try:
                prob = classify_window(kpts_seq, bbox_seq, ts_seq, valid_seq, clf_cfg)
            except Exception as e:
                print(f"[Warn] track {tid} window {w_idx}: {e}")
                continue

            window_results.append((win_frames[0], win_frames[-1], prob))

            if prob >= clf_cfg["threshold"]:
                results.append({
                    "track_id": tid,
                    "start_frame": int(win_frames[0]),
                    "end_frame": int(win_frames[-1]),
                    "start_time": round(win_frames[0] / fps, 3),
                    "end_time": round(win_frames[-1] / fps, 3),
                    "probability": round(prob, 4),
                })

        # Merge consecutive positive windows into segments
        if window_results:
            merged = _merge_windows(window_results, clf_cfg["threshold"], fps)
            for seg in merged:
                seg["track_id"] = tid
                suspicious_segments.append(seg)

    # Print results
    if results:
        print(f"\n=== Suspicious windows (prob >= {clf_cfg['threshold']}) ===")
        for r in results:
            print(f"  track={r['track_id']:3d}  frames={r['start_frame']:5d}-{r['end_frame']:5d}  "
                  f"time={r['start_time']:6.2f}s-{r['end_time']:6.2f}s  prob={r['probability']:.3f}")

    if suspicious_segments:
        print(f"\n=== Suspicious segments (merged) ===")
        for seg in suspicious_segments:
            print(f"  track={seg['track_id']:3d}  frames={seg['start_frame']:5d}-{seg['end_frame']:5d}  "
                  f"time={seg['start_time']:6.2f}s-{seg['end_time']:6.2f}s  "
                  f"max_prob={seg['max_prob']:.3f}")

    if not results:
        print(f"[Result] No suspicious windows found (threshold={clf_cfg['threshold']})")

    # Save output JSON
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        video_stem = Path(video_path).stem
        output = {
            "video": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "model": model_path,
            "behavior": clf_cfg["metadata"].get("behavior", "unknown"),
            "threshold": clf_cfg["threshold"],
            "results": results,
            "segments": suspicious_segments,
        }
        result_file = out_path / f"{video_stem}_results.json"
        with open(result_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[Saved] {result_file}")

    # Annotated video
    if visualize:
        _write_annotated_video(
            video_path, frame_cache, results, clf_cfg, fps, total_frames,
            output_dir or "./outputs", Path(video_path).stem,
        )

    return suspicious_segments


def _merge_windows(
    windows: list[tuple[int, int, float]], threshold: float, fps: float = 30.0,
) -> list[dict]:
    """Merge consecutive positive windows into segments."""
    segments: list[dict] = []
    current: list[tuple[int, int, float]] = []
    for w in windows:
        if w[2] >= threshold:
            current.append(w)
        else:
            if current:
                seg = {
                    "start_frame": current[0][0],
                    "end_frame": current[-1][1],
                    "start_time": round(current[0][0] / fps, 3),
                    "end_time": round(current[-1][1] / fps, 3),
                    "max_prob": round(max(p for _, _, p in current), 4),
                    "mean_prob": round(float(np.mean([p for _, _, p in current])), 4),
                }
                segments.append(seg)
                current = []
    if current:
        segments.append({
            "start_frame": current[0][0],
            "end_frame": current[-1][1],
            "start_time": round(current[0][0] / fps, 3),
            "end_time": round(current[-1][1] / fps, 3),
            "max_prob": round(max(p for _, _, p in current), 4),
            "mean_prob": round(float(np.mean([p for _, _, p in current])), 4),
        })
    return segments


def _write_annotated_video(
    video_path: str,
    frame_cache: dict,
    results: list[dict],
    clf_cfg: dict,
    fps: float,
    total_frames: int,
    output_dir: str,
    video_stem: str,
):
    """Write annotated video with bounding boxes and suspicion probability."""
    out_path = Path(output_dir) / f"{video_stem}_annotated.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build set of (track_id, frame_range) for suspicious windows
    suspicious: dict[int, set[range]] = defaultdict(set)
    for r in results:
        suspicious[r["track_id"]].add(range(r["start_frame"], r["end_frame"] + 1))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Warn] Could not open video for annotation: {video_path}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    pbar = tqdm(total=total_frames, desc="Annotating", unit="frame")
    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        track_data = frame_cache.get(frame_idx, {})
        for tid, data in track_data.items():
            is_susp = False
            if tid in suspicious:
                for rng in suspicious[tid]:
                    if frame_idx in rng:
                        is_susp = True
                        break
            color = (0, 0, 255) if is_susp else (0, 255, 0)
            bbox = tuple(float(v) for v in data["bbox"])
            draw_skeleton(frame, data["keypoints"], bbox, tid, is_susp, "suspicious" if is_susp else "")
        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    print(f"[Saved] annotated video: {out_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference: classify behavioral tracks in video using trained MLP classifier."
    )
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--model", default="./models/move_1_step",
                   help="Path to model directory containing classifier.pkl, scaler.pkl, metadata.json")
    p.add_argument("--pose-model", default="yolo26m-pose.pt",
                   help="YOLO pose model path or name (default: yolo26m-pose.pt)")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Detection confidence threshold (default: 0.4)")
    p.add_argument("--iou", type=float, default=0.5,
                   help="NMS IoU threshold (default: 0.5)")
    p.add_argument("--stride", type=int, default=1,
                   help="Sliding window stride in frames (default: 1)")
    p.add_argument("--output", default=None,
                   help="Output directory for results JSON and annotated video")
    p.add_argument("--visualize", action="store_true",
                   help="Generate annotated video with bounding boxes and skeleton")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        return 1

    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"Error: model directory not found: {model_dir}")
        return 1

    clf_cfg = load_classifier(model_dir)

    expected_dim = feature_dim(clf_cfg["target_len"], clf_cfg["mode"])
    print(f"[Config] behavior={clf_cfg['metadata'].get('behavior', '?')} "
          f"mode={clf_cfg['mode']} target_len={clf_cfg['target_len']} "
          f"expected_dim={expected_dim}")

    run_inference(
        video_path=str(video_path),
        model_path=args.pose_model,
        clf_cfg=clf_cfg,
        conf=args.conf,
        iou=args.iou,
        window_stride=args.stride,
        visualize=args.visualize,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())