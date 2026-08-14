import queue
import threading
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from handhead.behavior_detector import is_hand_to_head
from handhead.detector import process_frame
from handhead.event_manager import StatefulEventManager
from handhead.exporter import export_outputs, export_single_event, write_metadata_files
from handhead.model import load_pose_model
from handhead.video_utils import create_video_writer
from handhead.visualizer import draw_skeleton

MAX_CACHED_FRAMES = 600


def _reader_worker(cap, read_queue, total_frames):
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        read_queue.put((frame_idx, frame))
    read_queue.put(None)


def _inference_worker(
    model, read_queue, write_queue, event_manager, frame_data_cache, conf, iou, fps
):
    while True:
        item = read_queue.get()
        if item is None:
            write_queue.put(None)
            break
        frame_idx, frame = item
        timestamp = frame_idx / fps

        people = process_frame(model, frame, conf=conf, iou=iou)
        detections = {}
        frame_data = {}
        for person in people:
            detected, conf_val, side = is_hand_to_head(person.keypoints)
            detections[person.track_id] = (detected, conf_val, side)
            frame_data[person.track_id] = {
                "bbox": person.bbox,
                "keypoints": person.keypoints,
                "detected": bool(detected),
            }

        frame_data_cache[frame_idx] = frame_data
        if len(frame_data_cache) > MAX_CACHED_FRAMES:
            oldest = min(frame_data_cache)
            del frame_data_cache[oldest]

        new_events = event_manager.update(frame_idx, timestamp, detections)
        write_queue.put((frame_idx, frame, people, detections, new_events))


def run_pipeline(
    video_path: str,
    model_path: str = "yolo11n-pose.pt",
    output_dir: str = "./outputs",
    conf: float = 0.3,
    iou: float = 0.5,
    visualize: bool = False,
    context_seconds: int = 5,
    crop_padding: int = 20,
    debug_keypoints: bool = False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_stem = Path(video_path).stem
    model = load_pose_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    if total_frames <= 0:
        total_frames = 999999

    event_manager = StatefulEventManager()
    frame_data_cache = {}
    events = []
    metadata_events = []
    event_counter = 0

    writer = None
    writer_size = None
    if visualize:
        ret, first_frame = cap.read()
        if ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            h, w = first_frame.shape[:2]
            writer_size = (w // 2 * 2, h // 2 * 2)
            writer = create_video_writer(
                output_path / f"{video_stem}_annotated_video.mp4", "mp4v", fps, writer_size
            )

    read_queue = queue.Queue(maxsize=30)
    write_queue = queue.Queue(maxsize=30)

    reader = threading.Thread(
        target=_reader_worker, args=(cap, read_queue, total_frames), daemon=True
    )
    inference = threading.Thread(
        target=_inference_worker,
        args=(model, read_queue, write_queue, event_manager, frame_data_cache, conf, iou, fps),
        daemon=True,
    )

    reader.start()
    inference.start()

    progress = tqdm(total=total_frames, desc="Processing frames")
    while True:
        item = write_queue.get()
        if item is None:
            break
        frame_idx, frame, people, detections, new_events = item
        events.extend(new_events)

        for ev in new_events:
            event_counter += 1
            meta = export_single_event(
                ev,
                event_counter,
                video_path,
                frame_data_cache,
                output_path,
                fps,
                context_seconds=context_seconds,
                padding=crop_padding,
                debug_keypoints=debug_keypoints,
                video_stem=video_stem,
            )
            metadata_events.append(meta)

        if visualize:
            for person in people:
                det_bool, _, _ = detections.get(person.track_id, (False, 0.0, "none"))
                frame = draw_skeleton(
                    frame, person.keypoints, person.bbox, person.track_id, det_bool
                )
            if writer is not None:
                if frame.shape[1] != writer_size[0] or frame.shape[0] != writer_size[1]:
                    frame = cv2.resize(frame, writer_size)
                writer.write(frame)
        progress.update(1)
    progress.close()

    inference.join()
    reader.join()

    remaining = event_manager.finalize()
    events.extend(remaining)

    for ev in remaining:
        if ev.end_time == 0.0:
            ev.end_time = total_frames / fps
        event_counter += 1
        meta = export_single_event(
            ev,
            event_counter,
            video_path,
            frame_data_cache,
            output_path,
            fps,
            context_seconds=context_seconds,
            padding=crop_padding,
            debug_keypoints=debug_keypoints,
            video_stem=video_stem,
        )
        metadata_events.append(meta)

    cap.release()
    if writer is not None:
        writer.release()

    write_metadata_files(metadata_events, video_path, output_path, fps, total_frames, video_stem)

    print(f"\nPipeline complete.")
    print(f"  Total frames processed: {len(frame_data_cache)}")
    print(f"  Events detected: {len(metadata_events)}")
    print(f"  Output directory: {output_path.resolve()}")
    for ev in metadata_events:
        print(
            f"    Event {ev['event_id']}: Track {ev['track_id']}, "
            f"Frames {ev['start_frame']}-{ev['end_frame']}, "
            f"Side: {ev['hand_side']}, Confidence: {ev['max_confidence']}"
        )