import argparse
import csv
from pathlib import Path

import cv2

from detection.config_loader import load_config
from detection.detector import process_frame
from detection.model import load_pose_model


def main():
    parser = argparse.ArgumentParser(
        description="Export all ByteTrack coordinates per track ID per frame to CSV."
    )
    parser.add_argument("--video", required=True, help="Path to input video (MP4).")
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <video_stem>_tracks.csv).")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        return 1

    model_path = "yolo11n-pose.pt"
    conf = 0.3
    iou = 0.5
    crop_region = None

    if args.config:
        config = load_config(args.config)
        model_path = config.get("model", {}).get("path", model_path)
        conf = config.get("model", {}).get("conf", conf)
        iou = config.get("model", {}).get("iou", iou)
        raw_crop = config.get("crop")
        if raw_crop:
            crop_region = tuple(raw_crop)

    csv_path = args.output or f"{video_path.stem}_tracks.csv"
    model = load_pose_model(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 999999

    kp_columns = []
    for i in range(17):
        kp_columns.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"])

    fieldnames = ["frame_idx", "timestamp", "track_id", "x1", "y1", "x2", "y2", "conf"] + kp_columns

    print(f"Processing {video_path} ({total_frames} frames, {fps:.2f} fps)...")
    print(f"Model: {model_path}  conf={conf}  iou={iou}")
    if crop_region:
        print(f"Crop: {crop_region}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row_count = 0
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if crop_region:
                x, y, w, h = crop_region
                h_f, w_f = frame.shape[:2]
                if y >= h_f or x >= w_f:
                    break
                x_end = min(x + w, w_f)
                y_end = min(y + h, h_f)
                frame = frame[y:y_end, x:x_end]
                if frame.size == 0:
                    break

            timestamp = frame_idx / fps
            people = process_frame(model, frame, conf=conf, iou=iou)

            for person in people:
                x1, y1, x2, y2 = person.bbox
                row = {
                    "frame_idx": frame_idx,
                    "timestamp": round(timestamp, 4),
                    "track_id": person.track_id,
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "conf": round(person.conf, 4),
                }
                for kp_i in range(17):
                    row[f"kp{kp_i}_x"] = round(person.keypoints[kp_i][0], 2)
                    row[f"kp{kp_i}_y"] = round(person.keypoints[kp_i][1], 2)
                    row[f"kp{kp_i}_conf"] = round(person.keypoints[kp_i][2], 4)
                writer.writerow(row)
                row_count += 1

            if frame_idx % 500 == 0:
                print(f"  Frame {frame_idx}/{total_frames} — tracks so far: {row_count}")

    cap.release()
    print(f"\nDone — {row_count} tracking rows written to {csv_path}")

    return 0


if __name__ == "__main__":
    exit(main())