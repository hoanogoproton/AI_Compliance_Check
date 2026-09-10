"""Pipeline-level abort tests: run_pipeline must stop mid-video.

The watcher passes its stop Event as ``abort_event``; when it is set the
pipeline must stop reading / inferring / writing promptly, remove the
incomplete annotated video and raise ``PipelineAborted`` instead of running
to completion. The YOLO model and detector are replaced with trivial
stand-ins so the test does not need any weights.
"""
import threading

import cv2
import numpy as np
import pytest

import detection.pipeline as pipeline_mod
from detection.pipeline import PipelineAborted, run_pipeline


@pytest.fixture()
def tiny_video(tmp_path):
    """A small 60-frame black mp4 so the pipeline has real frames to read."""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for _ in range(60):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def fake_model(monkeypatch):
    """Replace the YOLO model and detector with trivial stand-ins."""
    monkeypatch.setattr(pipeline_mod, "load_pose_model", lambda path: object())
    monkeypatch.setattr(
        pipeline_mod, "process_frame", lambda model, frame, conf=0.3, iou=0.5: []
    )


def test_run_pipeline_aborts_mid_video(tmp_path, tiny_video, fake_model):
    abort_event = threading.Event()
    seen = []

    def progress_cb(frame, total):
        seen.append(frame)
        if frame >= 3:
            abort_event.set()

    with pytest.raises(PipelineAborted):
        run_pipeline(
            video_path=str(tiny_video),
            output_dir=str(tmp_path / "out"),
            visualize=False,
            progress_callback=progress_cb,
            abort_event=abort_event,
        )

    # Aborted well before the end of the 60-frame video.
    assert 0 < len(seen) < 60


def test_run_pipeline_abort_removes_partial_annotated_video(
    tmp_path, tiny_video, fake_model, monkeypatch
):
    monkeypatch.setattr(
        pipeline_mod,
        "create_video_writer",
        lambda path, codec, fps, size: cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        ),
    )
    abort_event = threading.Event()

    def progress_cb(frame, total):
        if frame >= 2:
            abort_event.set()

    out_dir = tmp_path / "out"
    with pytest.raises(PipelineAborted):
        run_pipeline(
            video_path=str(tiny_video),
            output_dir=str(out_dir),
            visualize=True,
            progress_callback=progress_cb,
            abort_event=abort_event,
        )

    # The partially written annotated video was removed on abort.
    assert not (out_dir / "tiny_annotated_video.mp4").exists()


def test_run_pipeline_without_abort_event_completes(tmp_path, tiny_video, fake_model):
    """Backward compatibility: no abort_event -> the video runs to completion."""
    out_dir = tmp_path / "out"
    run_pipeline(
        video_path=str(tiny_video),
        output_dir=str(out_dir),
        visualize=False,
    )
    # No abort -> metadata written as usual.
    assert (out_dir / "tiny_metadata.json").exists()