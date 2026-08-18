import argparse
import csv
from pathlib import Path

from detection.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Hand-to-Head Detection MVP — detect hand-to-head behavior in surveillance video."
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to input video file (MP4).",
    )
    parser.add_argument(
        "--batch-csv",
        default=None,
        help="Path to CSV file mapping videos to configs (video,config,output_dir columns).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (optional — defaults to hand_to_head only).",
    )
    parser.add_argument(
        "--define-zones",
        action="store_true",
        help="Launch zone definition tool for this video.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="YOLO pose model path or name (default: yolo11n-pose.pt).",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Output directory for results (default: ./outputs).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate annotated video with bounding boxes and skeleton.",
    )
    parser.add_argument(
        "--context-seconds",
        type=int,
        default=5,
        help="Seconds of context before/after each event clip (default: 5).",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=20,
        help="Padding pixels around bounding box crop (default: 20).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI application.",
    )
    parser.add_argument(
        "--debug-keypoints",
        action="store_true",
        help="Export additional debug clips with keypoint skeleton overlay.",
    )
    args = parser.parse_args()

    if args.gui:
        from gui.app import main as gui_main
        gui_main()
        return 0

    if args.batch_csv:
        return _run_batch_csv(args)
    elif args.video:
        return _run_single(args)
    else:
        parser.print_help()
        return 1


def _run_single(args):
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        return 1

    if args.define_zones:
        from detection.zones.zone_tool import define_zones
        config_path = args.config or "config.yaml"
        define_zones(str(video_path.resolve()), config_path)
        return 0

    run_pipeline(
        video_path=str(video_path.resolve()),
        model_path=args.model,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        visualize=args.visualize,
        context_seconds=args.context_seconds,
        crop_padding=args.crop_padding,
        debug_keypoints=args.debug_keypoints,
        config_path=args.config,
    )
    return 0


def _run_batch_csv(args):
    csv_path = Path(args.batch_csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["video", "config", "output_dir"]:
            print("Error: CSV must have columns: video,config,output_dir")
            return 1
        rows = []
        for row in reader:
            video = (row.get("video") or "").strip()
            config = (row.get("config") or "").strip()
            output_dir = (row.get("output_dir") or "./outputs").strip()
            if not video or not config:
                continue
            rows.append({"video": video, "config": config, "output_dir": output_dir})

    if not rows:
        print("Error: CSV file is empty or has no valid rows.")
        return 1

    total = len(rows)
    for i, row in enumerate(rows, 1):
        video_path = Path(row["video"])
        config_path = row["config"]
        output_dir = row["output_dir"]
        print(f"\n[{i}/{total}] Processing: {video_path.name}")
        print(f"  Config: {config_path}")
        print(f"  Output: {output_dir}")

        if not video_path.exists():
            print(f"  SKIP: video not found — {video_path}")
            continue
        if not Path(config_path).exists():
            print(f"  SKIP: config not found — {config_path}")
            continue

        try:
            run_pipeline(
                video_path=str(video_path.resolve()),
                model_path=args.model,
                output_dir=output_dir,
                conf=args.conf,
                iou=args.iou,
                visualize=args.visualize,
                context_seconds=args.context_seconds,
                crop_padding=args.crop_padding,
                debug_keypoints=args.debug_keypoints,
                config_path=config_path,
            )
            print(f"  OK: {video_path.name} finished")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nBatch complete: {total} videos processed.")
    return 0


if __name__ == "__main__":
    exit(main())
