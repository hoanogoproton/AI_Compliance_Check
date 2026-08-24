"""
Diagnostic script to isolate OpenCV "Unknown C++ exception" error.

Tests two hypotheses:
1. Unicode path issue (video in non-ASCII directory)
2. Video codec/corruption issue (seek + read)
"""
import sys
import shutil
import tempfile
from pathlib import Path
import cv2

VIDEO_PATH = r"D:\Project Demo\DEMO - Nhan dien nguoi\videos\CA927-FB-RAI7-No3-20260822-093621.mp4"
CONFIG_PATH = r"D:\Project Demo\DEMO - Nhan dien nguoi\config\config_4.yaml"


def test_1_basic_open():
    """Test if VideoCapture can open the file."""
    print("=== Test 1: Basic VideoCapture open ===")
    cap = cv2.VideoCapture(VIDEO_PATH)
    print(f"  isOpened: {cap.isOpened()}")
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
        print(f"  FPS: {fps}, Frames: {total}, Size: {w}x{h}, Codec: {codec}")
        cap.release()
        return True
    else:
        print("  FAILED: Could not open video at original path")
        return False


def test_2_read_first_frame():
    """Test reading a few frames."""
    print("\n=== Test 2: Read first 10 frames ===")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("  SKIP (could not open)")
        return False
    for i in range(10):
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"  FAILED at frame {i}: ret=False")
                cap.release()
                return False
            print(f"  Frame {i}: OK ({frame.shape})", end="")
            if frame.size == 0:
                print(" [EMPTY FRAME]")
            else:
                print()
        except Exception as e:
            print(f"  EXCEPTION at frame {i}: {type(e).__name__}: {e}")
            cap.release()
            return False
    cap.release()
    print("  SUCCESS: 10 frames read without error")
    return True


def test_3_seek_and_read():
    """Test seeking to various positions (simulates exporter behavior)."""
    print("\n=== Test 3: Seek + read (simulates export_single_event) ===")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("  SKIP (could not open)")
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Seek to roughly 25%, 50%, 75% positions
    seek_positions = [int(total * 0.25), int(total * 0.5), int(total * 0.75)]
    print(f"  Total frames: {total}")
    for pos in seek_positions:
        if pos <= 0:
            continue
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"  FAILED at seek pos {pos}: could not open")
            continue
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret and frame.size > 0:
                print(f"  Seek to {pos}: OK ({frame.shape})")
            else:
                print(f"  Seek to {pos}: FAILED (ret={ret}, size={frame.size if ret else 'N/A'})")
        except Exception as e:
            print(f"  Seek to {pos}: EXCEPTION: {type(e).__name__}: {e}")
        cap.release()
    print("  SUCCESS (seek tests completed)")
    return True


def test_4_unicode_path_workaround():
    """Copy video to a non-Unicode path and test."""
    print("\n=== Test 4: Copy video to ASCII-only path and retry ===")
    ascii_path = r"C:\Users\hoangnv\AppData\Local\Temp\kilo\test_video.mp4"
    try:
        shutil.copy2(VIDEO_PATH, ascii_path)
        print(f"  Copied to: {ascii_path}")
        cap = cv2.VideoCapture(ascii_path)
        print(f"  isOpened: {cap.isOpened()}")
        if cap.isOpened():
            for i in range(5):
                ret, frame = cap.read()
                if not ret:
                    print(f"  FAILED at frame {i}")
                    cap.release()
                    return False
                print(f"  Frame {i}: OK ({frame.shape})")
            cap.release()
            print("  SUCCESS: Video works from ASCII-only path")
            Path(ascii_path).unlink(missing_ok=True)
            return True
        else:
            print("  FAILED: Still cannot open from ASCII path (codec/corruption issue)")
            Path(ascii_path).unlink(missing_ok=True)
            return False
    except Exception as e:
        print(f"  ERROR during copy/test: {e}")
        return False


def main():
    print("=" * 60)
    print("OpenCV Diagnostic for 'Unknown C++ exception'")
    print("=" * 60)
    print(f"Video: {VIDEO_PATH}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Python version: {sys.version}")
    print()

    t1 = test_1_basic_open()
    t2 = test_2_read_first_frame()
    t3 = test_3_seek_and_read()
    t4 = test_4_unicode_path_workaround()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (basic open):    {'PASS' if t1 else 'FAIL'}")
    print(f"  Test 2 (read frames):   {'PASS' if t2 else 'FAIL'}")
    print(f"  Test 3 (seek + read):   {'PASS' if t3 else 'FAIL'}")
    print(f"  Test 4 (ASCII path):    {'PASS' if t4 else 'FAIL'}")

    if t1 and t2 and t3 and not t4:
        print("\n  → Unicode path is likely the issue.")
        print("    The video works fine from an ASCII-only path but fails from the")
        print("    Vietnamese-character path.")
    elif not t1 and t4:
        print("\n  → Strong evidence of Unicode path issue.")
        print("    Video opens from ASCII path but not from original path.")
    elif not t1 and not t4:
        print("\n  → Likely a codec/corruption issue.")
        print("    Video fails to open from ANY path. Check codec support.")
    elif t1 and t2 and not t3:
        print("\n  → Seeking triggers the error.")
        print("    The issue is with cap.set(CAP_PROP_POS_FRAMES) / seeking in this video.")
    else:
        print("\n  → All tests passed under simple conditions.")
        print("    The error may require the full pipeline (model + threading + export).")


if __name__ == "__main__":
    main()