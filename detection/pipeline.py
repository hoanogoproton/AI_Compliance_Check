import json
import pickle
import queue
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from detection.behavior_detector import get_registry
from detection.behaviors.base import BaseBehavior
# Trigger registration of all behaviors
import detection.behaviors  # noqa: F401
from detection.config import CLASSIFIER_DEFAULT_THRESHOLD, CLASSIFIER_MIN_SEQUENCE_FRAMES
from detection.config_loader import load_config
from detection.detector import process_frame
from detection.event_manager import Event, StatefulEventManager
from detection.exporter import export_single_event, write_metadata_files
from detection.model import load_pose_model
from detection.video_utils import create_video_writer
from detection.visualizer import draw_skeleton, draw_zone
from detection.zones.zone_checker import load_zones
from features import extract_features

MAX_CACHED_FRAMES = 100000


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
        zone_names = bcfg.get("params", {}).get("zones")
        if isinstance(zone_names, str):
            zone_names = [zone_names]
        if not zone_names:
            single = bcfg.get("params", {}).get("zone")
            if single:
                zone_names = [single]
        resolved_zones: list = []
        if zone_names:
            for zn in zone_names:
                z = zones.get(zn)
                if z is None:
                    raise ValueError(f"Zone '{zn}' not found in config for behavior '{bcfg['name']}'")
                resolved_zones.append(z)
        behavior = cls(bcfg["params"], zones=resolved_zones)
        behaviors.append(behavior)
    return behaviors


def _load_classifier_models(config: dict) -> dict:
    """Load one-vs-rest classifier models from config["classifier"]["behaviors"].

    Returns behavior_name -> {"model", "scaler", "threshold", "metadata",
    "mode", "target_len", "min_sequence_frames"}. Behaviors whose model files
    are missing are skipped (with a console warning), so a config can be used
    before any model has been trained.
    """
    classifier_cfg = config.get("classifier") or {}
    behaviors_cfg = classifier_cfg.get("behaviors") or {}
    models: dict[str, dict] = {}
    for behavior_name, bcfg in behaviors_cfg.items():
        model_path = bcfg.get("model_path")
        scaler_path = bcfg.get("scaler_path")
        metadata_path = bcfg.get("metadata_path")
        if not model_path or not Path(model_path).exists():
            print(f"[Classifier] Skipping '{behavior_name}': model not found at {model_path}", flush=True)
            continue
        if not scaler_path or not Path(scaler_path).exists():
            print(f"[Classifier] Skipping '{behavior_name}': scaler not found at {scaler_path}", flush=True)
            continue
        metadata = {}
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        threshold = float(bcfg.get("threshold", metadata.get("threshold", CLASSIFIER_DEFAULT_THRESHOLD)))
        models[behavior_name] = {
            "model": model,
            "scaler": scaler,
            "threshold": threshold,
            "metadata": metadata,
            "mode": metadata.get("mode", "temporal"),
            "target_len": int(metadata.get("target_len", 32)),
            "min_sequence_frames": int(metadata.get("min_sequence_frames", CLASSIFIER_MIN_SEQUENCE_FRAMES)),
        }
        print(
            f"[Classifier] Loaded '{behavior_name}' (threshold={threshold:.3f}, "
            f"mode={models[behavior_name]['mode']}, dim={metadata.get('feature_dim', '?')})",
            flush=True,
        )
    return models


def _build_event_sequence(
    ev: Event, frame_data_cache: dict, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Build (keypoints_seq, bboxes_seq, timestamps, valid_mask) for an event.

    Uses the contiguous frame range [start_frame, end_frame]. Frames whose
    track data is missing from the cache are emitted as zeros with
    valid_mask=False so the feature extractor can rely on the mask.
    """
    if ev.end_frame < ev.start_frame or fps <= 0:
        return None
    frames = list(range(ev.start_frame, ev.end_frame + 1))
    T = len(frames)
    keypoints_seq = np.zeros((T, 17, 3), dtype=np.float32)
    bboxes_seq = np.zeros((T, 4), dtype=np.float32)
    timestamps = np.zeros(T, dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)
    for i, f in enumerate(frames):
        timestamps[i] = f / fps
        person_data = frame_data_cache.get(f, {}).get(ev.track_id)
        if person_data is None:
            continue
        kp = person_data.get("keypoints")
        if kp is None:
            continue
        keypoints_seq[i] = np.asarray(kp, dtype=np.float32)
        bboxes_seq[i] = np.asarray(person_data.get("bbox", (0, 0, 0, 0)), dtype=np.float32)
        valid_mask[i] = True
    return keypoints_seq, bboxes_seq, timestamps, valid_mask


def _predict_proba(model, X: np.ndarray) -> float:
    """Return P(positive class). Falls back to decision_function sigmoid."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0, -1])
    if hasattr(model, "decision_function"):
        d = float(model.decision_function(X)[0])
        return 1.0 / (1.0 + float(np.exp(-d)))
    return float(model.predict(X)[0])


def _apply_classifier_filter(
    all_new_events: list[Event],
    classifier_models: dict,
    frame_data_cache: dict,
    fps: float,
) -> list[Event]:
    """Phase 1.5: per-behavior classifier filter (one model per behavior)."""
    if not classifier_models or not all_new_events:
        return all_new_events
    filtered: list[Event] = []
    for ev in all_new_events:
        cm = classifier_models.get(ev.behavior_name)
        if cm is None:
            filtered.append(ev)
            continue
        seq = _build_event_sequence(ev, frame_data_cache, fps)
        if seq is None:
            filtered.append(ev)
            continue
        keypoints_seq, bboxes_seq, timestamps, valid_mask = seq
        if len(keypoints_seq) < cm["min_sequence_frames"]:
            filtered.append(ev)
            continue
        try:
            features = extract_features(
                keypoints_seq,
                bboxes_seq,
                timestamps,
                valid_mask,
                target_len=cm["target_len"],
                mode=cm["mode"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"[Classifier] feature error ({ev.behavior_name} track {ev.track_id}): {e}", flush=True)
            filtered.append(ev)
            continue
        features_scaled = cm["scaler"].transform([features])
        prob = _predict_proba(cm["model"], features_scaled)
        if prob >= cm["threshold"]:
            ev.max_confidence = float(prob)
            ev.metadata["classifier_confidence"] = float(prob)
            ev.metadata["classifier_threshold"] = float(cm["threshold"])
            ev.metadata["classifier_mode"] = cm["mode"]
            filtered.append(ev)
        else:
            ev.metadata["classifier_confidence"] = float(prob)
            ev.metadata["classifier_threshold"] = float(cm["threshold"])
            ev.metadata["classifier_dropped"] = True
    return filtered


def _reader_worker(cap, read_queue, total_frames, crop_region):
    for frame_idx in range(total_frames):
        try:
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
        except Exception as e:
            print(f"[ReaderWorker] OpenCV error at frame {frame_idx}: {e}", flush=True)
            break
        read_queue.put((frame_idx, frame))
    read_queue.put(None)


def _inference_worker(
    model, read_queue, write_queue, behaviors, frame_data_cache, conf, iou, fps,
    classifier_models,
):
    while True:
        item = read_queue.get()
        if item is None:
            write_queue.put(None)
            break
        frame_idx, frame = item
        timestamp = frame_idx / fps

        try:
            people = process_frame(model, frame, conf=conf, iou=iou)
        except Exception as e:
            print(f"[InferenceWorker] process_frame error at frame {frame_idx}: {e}", flush=True)
            traceback.print_exc()
            # Put an error sentinel to signal main loop
            write_queue.put(("ERROR", frame_idx, str(e)))
            continue
        all_new_events = []
        frame_data = {}

        # Phase 1: Process each behavior ONCE with ALL people,
        # so the state machine sees all track detections atomically per frame.
        for behavior in behaviors:
            new_events = behavior.process_frame(people, frame, frame_idx, timestamp)
            for ev in new_events:
                ev.behavior_name = behavior.name
            all_new_events.extend(new_events)

        # Phase 1.5: Per-behavior classifier filter (one model per behavior).
        all_new_events = _apply_classifier_filter(
            all_new_events, classifier_models, frame_data_cache, fps
        )

        # Phase 2: Build per-person frame_data for visualization / debugging.
        for person in people:
            person_behaviors = {}
            tid = person.track_id
            for behavior in behaviors:
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
            if hasattr(behavior, 'zones') and behavior.zones:
                for z in behavior.zones:
                    zn = z.name
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
                        if hasattr(behavior, 'current_triggered_zones'):
                            active = zn in behavior.current_triggered_zones
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
            if hasattr(b, 'zones') and b.zones:
                for z in b.zones:
                    zones_export.append({"zone": z, "behavior_name": b.name})
        classifier_models = _load_classifier_models(config)
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
        classifier_models = {}

    model = load_pose_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    crop_region = None
    if config_path:
        raw_crop = config.get("crop")
        if raw_crop:
            crop_region = tuple(raw_crop)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    if total_frames <= 0:
        total_frames = 999999

    frame_data_cache = {}
    events: list[Event] = []
    metadata_events = []
    event_counter = 0
    exported_event_ids: set[int] = set()

    writer = None
    writer_size = None
    if visualize:
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if crop_region:
            cx, cy, cw, ch = crop_region
            cw = min(cw, raw_w - cx)
            ch = min(ch, raw_h - cy)
            w, h = cw, ch
        else:
            w, h = raw_w, raw_h
        writer_size = (w // 2 * 2, h // 2 * 2)
        writer = create_video_writer(
            output_path / f"{video_stem}_annotated_video.mp4", "mp4v", fps, writer_size
        )

    read_queue = queue.Queue(maxsize=30)
    write_queue = queue.Queue(maxsize=30)

    reader = threading.Thread(
        target=_reader_worker, args=(cap, read_queue, total_frames, crop_region), daemon=True
    )
    inference = threading.Thread(
        target=_inference_worker,
        args=(model, read_queue, write_queue, behaviors, frame_data_cache, conf, iou, fps,
              classifier_models),
        daemon=True,
    )

    reader.start()
    inference.start()

    try:
        progress = tqdm(total=total_frames, desc="Processing frames") if log_callback is None else None
        if log_callback:
            log_callback(f"Pipeline started for {video_path}")
            log_callback(f"Model: {model_path}, Output: {output_dir}")
            log_callback(f"Behaviors: {[b.name for b in behaviors]}")
        while True:
            item = write_queue.get()
            if item is None:
                break
            # Check for error sentinel from inference worker
            if isinstance(item, tuple) and len(item) == 3 and item[0] == "ERROR":
                _error_frame, error_msg = item[1], item[2]
                if log_callback:
                    log_callback(f"ERROR at frame {_error_frame}: {error_msg}")
                if progress is not None:
                    progress.close()
                raise RuntimeError(f"Inference error at frame {_error_frame}: {error_msg}")
            frame_idx, frame, people, new_events, zone_active = item

            for ev in new_events:
                event_counter += 1
                try:
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
                        crop_region=crop_region,
                    )
                except Exception as e:
                    if log_callback:
                        log_callback(f"  Event export error (event {event_counter}): {e}")
                    continue
                metadata_events.append(meta)
            if new_events:
                exported_event_ids.add(id(ev))
                if log_callback:
                    log_callback(
                        f"  Event {meta['event_id']}: Track {meta['track_id']}, "
                        f"Behavior: {meta.get('behavior', 'N/A')}, "
                        f"Frames {meta['start_frame']}-{meta['end_frame']}, "
                        f"Side: {meta['hand_side']}, Confidence: {meta['max_confidence']}"
                    )

            # --- update progress for every processed frame ---
            if progress is not None:
                progress.update(1)
            if progress_callback:
                progress_callback(frame_idx + 1, total_frames)

            # --- annotate and write output video for every frame ---
            if visualize:
                try:
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
                        if hasattr(behavior, 'zones') and behavior.zones:
                            for z in behavior.zones:
                                is_active = zone_active.get(z.name, False)
                                frame = draw_zone(frame, z, is_active)
                    # write once per frame after all annotations are done
                    if writer is not None and writer.isOpened():
                        if frame.size == 0:
                            continue
                        if frame.shape[1] != writer_size[0] or frame.shape[0] != writer_size[1]:
                            frame = cv2.resize(frame, writer_size)
                        writer.write(frame)
                except Exception as e:
                    if log_callback:
                        log_callback(f"  Visualization error at frame {frame_idx}: {e}")
                    else:
                        print(f"  Visualization error at frame {frame_idx}: {e}", flush=True)

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
            if id(ev) in exported_event_ids:
                continue
            event_counter += 1
            try:
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
                    crop_region=crop_region,
                )
            except Exception as e:
                if log_callback:
                    log_callback(f"  Post-pipeline export error (event {event_counter}): {e}")
                continue
            metadata_events.append(meta)

    finally:
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
