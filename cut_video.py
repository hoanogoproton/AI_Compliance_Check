import subprocess
import sys
from pathlib import Path


def cut_video(input_path: str, start_sec: float, end_sec: float, output_path: str = None):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")

    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError("end_sec must be greater than start_sec")

    if output_path is None:
        stem = input_path.stem
    else:
        stem = Path(output_path).stem

    ffmpeg = "ffmpeg"

    output = Path("outputs") / f"{stem}_cut.mp4"
    # stream copy — no re-encode, preserves fps/quality/codec
    cmd = [
        ffmpeg,
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    print(f"Cut saved to: {output}")
    return output


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cut_video.py <video_path> <start_sec> [end_sec]")
        print("  end_sec defaults to end of video if omitted")
        sys.exit(1)

    video_path = sys.argv[1]
    start_sec = float(sys.argv[2])
    end_sec = float(sys.argv[3]) if len(sys.argv) >= 4 else None

    if end_sec is None:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        end_sec = total_frames / fps

    cut_video(video_path, start_sec, end_sec)