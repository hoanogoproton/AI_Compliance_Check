"""Realtime camera tests without real hardware.

* ``export_event_clip_from_buffer`` is exercised with a synthetic rolling
  buffer (JPEG frames + per-frame track metadata) and must honor the same
  metadata contract as ``export_single_event``.
* ``run_realtime_camera`` runs against a plain video file (one pass to EOF)
  with the YOLO model and detector replaced by trivial stand-ins — the same
  trick as tests/test_pipeline_abort.py — to validate the grabber thread, the
  preview bridge, the status heartbeat, the open-failure path and stopping.
"""

import threading

import cv2
import numpy as np
import pytest

import detection.realtime as rt_mod
from detection.event_manager import Event
from detection.exporter import export_event_clip_from_buffer


def _frame_with_box(width=64, height=48):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[16:32, 24:40] = 255
    return frame


def test_export_event_clip_from_buffer(tmp_path):
    fps = 25.0
    buffer = []
    frame_data_cache = {}
    bbox = (24, 16, 40, 32)
    kps = np.zeros((17, 3), dtype=np.float32)
    for idx in range(60):
        ok, jpeg = cv2.imencode(".jpg", _frame_with_box())
        assert ok
        buffer.append((idx, jpeg.tobytes()))
        frame_data_cache[idx] = {1: {"bbox": bbox, "keypoints": kps, "behaviors": {}}}

    event = Event(
        track_id=1, start_frame=50, end_frame=58,
        start_time=50 / fps, end_time=58 / fps, max_confidence=0.9,
        frames=list(range(50, 59)), hand_sides=["right"],
        behavior_name="hand_to_head",
    )
    meta = export_event_clip_from_buffer(
        event, 1, buffer, frame_data_cache, tmp_path, fps,
        context_seconds=2, padding=10, camera_name="172.17.108.15_ch1",
    )
    assert meta is not None
    assert meta["behavior"] == "hand_to_head"
    assert meta["track_id"] == 1
    assert meta["hand_side"] == "right"
    # 2s of context reaches the buffer start (frame 0); the future is capped
    # at the newest buffered frame (59).
    assert meta["clip_start_frame"] == 0
    assert meta["clip_end_frame"] == 59
    clip = tmp_path / meta["clip_path"]
    assert clip.name.startswith("172.17.108.15_ch1_")
    assert clip.exists() and clip.stat().st_size > 0
    cap = cv2.VideoCapture(str(clip))
    assert cap.isOpened()
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    assert n_frames >= 1
    # Cropped around the track with the minimum padding; the crop can exceed
    # the 64px frame width (black padding), but stays sane for a 64x48 input.
    assert 0 < w < 4 * 64


def test_export_event_clip_missing_end_frame_returns_none(tmp_path):
    meta = export_event_clip_from_buffer(
        Event(track_id=1, start_frame=0, end_frame=99, start_time=0.0,
              end_time=1.0, max_confidence=0.5),
        1, [(0, b"x")], {}, tmp_path, 25.0,
    )
    assert meta is None


@pytest.fixture()
def tiny_video(tmp_path):
    """A small 30-frame black mp4 used as a stand-in live source."""
    path = tmp_path / "cam_file.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def fake_model(monkeypatch):
    """Replace the YOLO model and detector with trivial stand-ins."""
    monkeypatch.setattr(rt_mod, "load_pose_model", lambda path: object())
    monkeypatch.setattr(rt_mod, "process_frame", lambda model, frame, conf=0.3, iou=0.5: [])


def test_run_realtime_camera_file_source_completes(tmp_path, tiny_video, fake_model):
    previews = []
    events = []

    def frame_cb(idx, rgb, info):
        assert rgb.ndim == 3 and rgb.shape[2] == 3  # RGB preview frame
        previews.append(idx)

    rt_mod.run_realtime_camera(
        source=str(tiny_video),
        camera_name="testcam",
        output_dir=str(tmp_path / "out"),
        fps_fallback=10.0,
        send_email=False,
        event_callback=events.append,
        frame_callback=frame_cb,
    )
    assert len(previews) > 0                      # preview frames were pushed
    assert any(e["type"] == "camera_status" for e in events)
    assert any(e.get("camera") == "testcam" for e in events)


def test_run_realtime_camera_abort_stops(tmp_path, tiny_video, fake_model):
    abort = threading.Event()
    seen = []

    def frame_cb(idx, rgb, info):
        seen.append(idx)
        if len(seen) >= 3:
            abort.set()

    rt_mod.run_realtime_camera(
        source=str(tiny_video),
        camera_name="testcam",
        output_dir=str(tmp_path / "out"),
        fps_fallback=10.0,
        frame_callback=frame_cb,
        abort_event=abort,
    )
    # Aborted well before the 30-frame file was exhausted.
    assert 0 < len(seen) < 30


def test_run_realtime_camera_open_failure_emits_error(tmp_path, fake_model):
    events = []
    rt_mod.run_realtime_camera(
        source=str(tmp_path / "missing.mp4"),
        camera_name="tc",
        output_dir=str(tmp_path / "out"),
        event_callback=events.append,
    )
    assert any(
        e["type"] == "camera_error" and not e.get("reconnecting") for e in events
    )