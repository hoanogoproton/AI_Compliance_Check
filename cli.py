import argparse
from pathlib import Path

from handhead.pipeline import run_pipeline


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
        "--debug-keypoints",
        action="store_true",
        help="Export additional debug clips with keypoint skeleton overlay.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        return 1

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
    )
    return 0


if __name__ == "__main__":
    exit(main())