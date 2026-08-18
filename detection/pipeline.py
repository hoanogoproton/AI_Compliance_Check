import queue
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from detection.behavior_detector import get_registry
from detection.behaviors.base import BaseBehavior
# Trigger registration of all behaviors
import detection.behaviors  # noqa: F401
from detection.config_loader import load_config
from detection.detector import process_frame
from detection.event_manager import Event, StatefulEventManager
from detection.exporter import export_single_event, write_metadata_files
from detection.model import load_pose_model
from detection.video_utils import create_video_writer
from detection.visualizer import draw_skeleton, draw_zone
from detection.zones.zone_checker import load_zones

MAX_CACHED_FRAMES = 600


def _build_behaviors(config: dict) -> list[BaseBehavior]:
    zones = load_zones(config)
    registry = get_registry()
    behaviors: list[BaseBehavior] = []
    for bcfg in config["behaviors"]:
        if not bcfg.get("enabled", True):
            continue
        cls = registry.get(bcfg["name"])
        if cls is None:
            raise ValueError(f"Unknown behavior: {bcfg['name']}")
        zone_name = bcfg.get("params", {}).get("zone")
        zone = zones.get(zone_name) if zone_name else None
        if zone_name and zone is None:
            raise ValueError(f"Zone '{zone_name}' not found in config for behavior '{bcfg['name']}'")
        behavior = cls(bcfg["params"], zone=zone) if zone is not None else cls(bcfg["params"])
        behaviors.append(behavior)
    return behaviors


def _reader_worker(cap, read_queue, total_frames):
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        read_queue.put((frame_idx, frame))
    read_queue.put(None)


def _inference_worker(
    model, read_queue, write_queue, behaviors, frame_data_cache, conf, iou, fps
):
    while True:
        item = read_queue.get()
        if item is None:
            write_queue.put(None)
            break
        frame_idx, frame = item
        timestamp = frame_idx / fps

        people = process_frame(model, frame, conf=conf, iou=iou)
        all_new_events = []
        frame_data = {}
        for person in people:
            person_behaviors = {}
            for behavior in behaviors:
                new_events = behavior.process_frame([person], frame, frame_idx, timestamp)
                for ev in new_events:
                    ev.behavior_name = behavior.name
                all_new_events.extend(new_events)
                tid = person.track_id
                if behavior.name == "leave_zone":
                    if behavior._track_inside.get(tid, False):
                        is_detected = False
                    elif behavior.is_person_in_flash(tid, frame_idx):
                        is_detected = True
                    else:
                        is_detected = False
                else:
                    ts = behavior.event_manager._tracks.get(tid)
                    is_detected = bool(ts and ts.state == "ACTIVE")
                person_behaviors[behavior.name] = {"detected": is_detected}
            frame_data[person.track_id] = {
                "bbox": person.bbox,
                "keypoints": person.keypoints,
                "behaviors": person_behaviors,
            }

        frame_data_cache[frame_idx] = frame_data
        if len(frame_data_cache) > MAX_CACHED_FRAMES:
            oldest = min(frame_data_cache)
            del frame_data_cache[oldest]

        zone_active = {}
        for behavior in behaviors:
            if hasattr(behavior, 'zone') and behavior.zone:
                zn = behavior.zone.name
                if behavior.name == "leave_zone":
                    flash_frames = behavior.params.get("leave_flash_frames", 20)
                    is_active = any(
                        frame_idx - leave_frame <= flash_frames
                        for leave_frame in behavior._last_leave_frame.values()
                    )
                    state = "active" if is_active else "inactive"
                else:
                    active = any(
                        fd["behaviors"].get(behavior.name, {}).get("detected")
                        for fd in frame_data.values()
                    )
                    state = "active" if active else "inactive"
                zone_active[zn] = state

        write_queue.put((frame_idx, frame, people, all_new_events, zone_active))


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
    config_path: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_stem = Path(video_path).stem

    if config_path:
        config = load_config(config_path)
        model_path = config.get("model", {}).get("path", model_path)
        conf = config.get("model", {}).get("conf", conf)
        iou = config.get("model", {}).get("iou", iou)
        visualize = config.get("output", {}).get("visualize", visualize)
        context_seconds = config.get("output", {}).get("context_seconds", context_seconds)
        crop_padding = config.get("output", {}).get("crop_padding", crop_padding)
        debug_keypoints = config.get("output", {}).get("debug_keypoints", debug_keypoints)
        behaviors = _build_behaviors(config)
        zones_export = []
        for b in behaviors:
            if hasattr(b, 'zone') and b.zone:
                zones_export.append({"zone": b.zone, "behavior_name": b.name})
    else:
        from detection.behaviors.hand_to_head import HandToHeadBehavior
        from detection.config import (
            CONFIRMATION_FRAMES, DISTANCE_THRESHOLD_RATIO, HEAD_KEYPOINT_CONFIDENCE_THRESHOLD,
            KEYPOINT_CONFIDENCE_THRESHOLD, MAX_GAP_FRAMES, MIN_EVENT_FRAMES, VERTICAL_OFFSET_RATIO,
        )
        behavior = HandToHeadBehavior({
            "distance_threshold_ratio": DISTANCE_THRESHOLD_RATIO,
            "vertical_offset_ratio": VERTICAL_OFFSET_RATIO,
            "keypoint_conf_threshold": KEYPOINT_CONFIDENCE_THRESHOLD,
            "head_keypoint_conf_threshold": HEAD_KEYPOINT_CONFIDENCE_THRESHOLD,
            "confirmation_frames": CONFIRMATION_FRAMES,
            "max_gap_frames": MAX_GAP_FRAMES,
            "min_event_frames": MIN_EVENT_FRAMES,
        })
        behaviors = [behavior]
        zones_export = []

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
        args=(model, read_queue, write_queue, behaviors, frame_data_cache, conf, iou, fps),
        daemon=True,
    )

    reader.start()
    inference.start()

    progress = tqdm(total=total_frames, desc="Processing frames") if log_callback is None else None
    if log_callback:
        log_callback(f"Pipeline started for {video_path}")
        log_callback(f"Model: {model_path}, Output: {output_dir}")
        log_callback(f"Behaviors: {[b.name for b in behaviors]}")
    while True:
        item = write_queue.get()
        if item is None:
            break
        frame_idx, frame, people, new_events, zone_active = item
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
                zone_export_info=zones_export,
            )
            metadata_events.append(meta)
            if log_callback:
                log_callback(
                    f"  Event {meta['event_id']}: Track {meta['track_id']}, "
                    f"Behavior: {meta.get('behavior', 'N/A')}, "
                    f"Frames {meta['start_frame']}-{meta['end_frame']}, "
                    f"Side: {meta['hand_side']}, Confidence: {meta['max_confidence']}"
                )

        if visualize:
            for person in people:
                fd = frame_data_cache.get(frame_idx, {}).get(person.track_id, {})
                behaviors_data = fd.get("behaviors", {})
                active_behavior_name = ""
                for bname, bdata in behaviors_data.items():
                    if bdata.get("detected"):
                        active_behavior_name = bname
                        break
                frame = draw_skeleton(
                    frame, person.keypoints, person.bbox, person.track_id,
                    bool(active_behavior_name), active_behavior_name,
                )
            for behavior in behaviors:
                if hasattr(behavior, 'zone') and behavior.zone:
                    is_active = zone_active.get(behavior.zone.name, False)
                    frame = draw_zone(frame, behavior.zone, is_active)
            if writer is not None:
                if frame.shape[1] != writer_size[0] or frame.shape[0] != writer_size[1]:
                    frame = cv2.resize(frame, writer_size)
                writer.write(frame)
        if progress is not None:
            progress.update(1)
        if progress_callback:
            progress_callback(frame_idx + 1, total_frames)
    if progress is not None:
        progress.close()

    inference.join()
    reader.join()

    for behavior in behaviors:
        remaining = behavior.event_manager.finalize()
        for ev in remaining:
            ev.behavior_name = behavior.name
        events.extend(remaining)

    for ev in events:
        if isinstance(ev, Event) and ev.end_time == 0.0:
            ev.end_time = total_frames / fps

    for ev in events:
        if ev not in metadata_events:
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
                zone_export_info=zones_export,
            )
            metadata_events.append(meta)

    cap.release()
    if writer is not None:
        writer.release()

    write_metadata_files(metadata_events, video_path, output_path, fps, total_frames, video_stem)

    summary = (
        f"\nPipeline complete.\n"
        f"  Total frames processed: {len(frame_data_cache)}\n"
        f"  Events detected: {len(metadata_events)}\n"
        f"  Output directory: {output_path.resolve()}\n"
    )
    for ev in metadata_events:
        summary += (
            f"    Event {ev['event_id']}: Track {ev['track_id']}, "
            f"Behavior: {ev.get('behavior', 'N/A')}, "
            f"Frames {ev['start_frame']}-{ev['end_frame']}, "
            f"Side: {ev['hand_side']}, Confidence: {ev['max_confidence']}\n"
        )
    if log_callback:
        log_callback(summary)
    else:
        print(summary)
