import argparse
from pathlib import Path

from detection.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Hand-to-Head Detection MVP — detect hand-to-head behavior in surveillance video."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file (MP4).",
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


if __name__ == "__main__":
    exit(main())
