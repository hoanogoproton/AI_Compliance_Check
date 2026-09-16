"""Realtime camera pipeline: the file pipeline's detection stack on live streams.

``run_realtime_camera`` processes one camera source (an RTSP/HTTP stream URL,
or a plain video-file path — handy for tests) with the same detection stack as
``detection.pipeline.run_pipeline``: YOLO pose tracking, the per-behavior state
machines, the classifier filter and identical annotated-frame visuals. What
differs is everything a live source implies:

* A grabber thread keeps only the newest frame, so when inference is slower
  than the stream the loop never falls behind (stale frames are dropped and
  counted, never queued up).
* There is no total frame count: progress is a ``camera_status`` heartbeat
  (measured FPS, people count, event total) instead of a progress bar.
* A rolling buffer of recent frames (JPEG-encoded to keep RAM low) provides
  the ``context_seconds`` of footage BEFORE an event, so the clip can be
  exported the moment the event completes (``export_event_clip_from_buffer``).
* Lost connections are re-opened with a backoff and ``camera_error`` events
  (with ``reconnecting=True``); ``abort_event`` stops the loop between frames.

Events emitted through ``event_callback`` (dicts, forwarded by the watcher to
the GUI): ``camera_status``, ``camera_event``, ``camera_error``.
``frame_callback(frame_idx, rgb, info)`` receives throttled annotated preview
frames and ``log_callback`` human-readable lines.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import email_notifier

from detection.config_loader import load_config
from detection.detector import process_frame
from detection.exporter import export_event_clip_from_buffer, write_metadata_files
from detection.model import load_pose_model
from detection.pipeline import (
    _annotate_frame,
    _apply_classifier_filter,
    _build_behaviors,
    _compute_zone_active,
    _load_classifier_models,
)

STREAM_SCHEMES = ("rtsp://", "rtsps://", "rtsph://", "http://", "https://")
DEFAULT_FPS_FALLBACK = 25.0   # used when the stream does not report a fps
PREVIEW_MIN_INTERVAL = 0.1    # min seconds between preview frames (~10 FPS)
STATUS_EVERY = 5.0            # seconds between camera_status heartbeats
STALL_TIMEOUT = 10.0          # no frame for this long -> treat as lost stream
GRAB_QUEUE_SIZE = 1           # keep only the newest frame
JPEG_QUALITY = 85
BUFFER_MARGIN_FRAMES = 10     # frames kept on top of context_seconds * fps
RECONNECT_DELAYS = (2.0, 5.0, 10.0, 30.0)


def is_stream_source(source: str) -> bool:
    """True when ``source`` is a network stream (reconnect semantics apply)."""
    return str(source).strip().lower().startswith(STREAM_SCHEMES)


def _open_capture(source: str):
    """Open a capture, forcing the FFmpeg backend + TCP for network streams.

    Returns an opened ``cv2.VideoCapture`` or ``None``.
    """
    src = str(source).strip()
    if is_stream_source(src):
        # Read with the FFmpeg backend; TCP transport avoids the tearing and
        # packet-loss artifacts of RTSP over UDP. The env var is re-read on
        # every capture open, so setdefault still allows an external override.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(src)
    return cap if cap.isOpened() else None


def _crop_frame(frame: np.ndarray, crop_region) -> np.ndarray | None:
    """Apply the config crop (same semantics as the pipeline reader worker)."""
    if not crop_region:
        return frame
    h_f, w_f = frame.shape[:2]
    x, y, w, h = crop_region
    if y >= h_f or x >= w_f:
        return None
    x_end = min(x + w, w_f)
    y_end = min(y + h, h_f)
    cropped = frame[y:y_end, x:x_end]
    return cropped if cropped.size else None


def _grabber_worker(cap, out_q: queue.Queue, abort_event, stats: dict) -> None:
    """Read frames as fast as the source delivers; keep only the newest.

    When the consumer is slower than the stream, the stale queued frame is
    replaced (counted in ``stats["dropped"]``) so processing always works on
    the most recent image and lag never accumulates. Ends by queueing a
    ``None`` sentinel when the stream ends, errors or abort is requested.
    """
    try:
        while not (abort_event is not None and abort_event.is_set()):
            ret, frame = cap.read()
            if not ret:
                break
            if out_q.full():
                try:
                    out_q.get_nowait()
                except queue.Empty:
                    pass
                stats["dropped"] += 1
            out_q.put(frame)
    except Exception:  # noqa: BLE001 — any capture error just ends the stream
        pass
    finally:
        out_q.put(None)


def _setup_detection(config: dict, fps: float) -> tuple:
    """Build behaviors / classifier / face pipeline / zones from the config.

    Mirrors ``run_pipeline``'s setup (including the no-config default of a
    single hand_to_head behavior) so both pipelines behave identically.
    Returns ``(behaviors, classifier_models, face_pipeline, zones_export,
    crop_region)``.
    """
    if config:
        behaviors = _build_behaviors(config, fps=fps)
        zones_export = []
        for b in behaviors:
            if hasattr(b, "zones") and b.zones:
                for z in b.zones:
                    zones_export.append({"zone": z, "behavior_name": b.name})
        classifier_models = _load_classifier_models(config)
        crop_region = tuple(config["crop"]) if config.get("crop") else None
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
        crop_region = None

    face_pipeline = None
    if config:
        face_cfg = config.get("face_detection", {})
        has_head_shake = any(
            b.get("name") == "head_shake" and b.get("enabled", True)
            for b in config.get("behaviors", [])
        )
        if has_head_shake:
            from detection.face_utils import FacePipeline
            face_pipeline = FacePipeline(
                min_detection_confidence=face_cfg.get("min_detection_confidence", 0.5),
                max_reprojection_error=face_cfg.get("max_reprojection_error", 10.0),
            )
    return behaviors, classifier_models, face_pipeline, zones_export, crop_region


def _send_event_email(camera_name: str, output_dir: Path, meta: dict, log) -> None:
    """Fire-and-forget email for one realtime event (never affects the run)."""
    try:
        subject = f"CẢNH BÁO REALTIME - {camera_name}: {meta.get('behavior', '')}".replace("|", "/")
        body = email_notifier.build_results_html(camera_name, str(output_dir), [meta])
        email_notifier.send_results_email_async(subject, body, log=log)
    except Exception as e:  # noqa: BLE001 — email must never break the camera
        log(f"[{camera_name}] WARNING: không gửi được email sự kiện: {e}")


def run_realtime_camera(
    source: str,
    config_path: str | None = None,
    camera_name: str = "camera",
    model_path: str = "yolo11n-pose.pt",
    output_dir: str = "./outputs/realtime",
    conf: float = 0.3,
    iou: float = 0.5,
    fps_fallback: float = DEFAULT_FPS_FALLBACK,
    context_seconds: int = 5,
    crop_padding: int = 20,
    debug_keypoints: bool = False,
    send_email: bool = True,
    max_retries: int = 5,
    event_callback: Callable[[dict], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    frame_callback: Callable[[int, np.ndarray, dict], None] | None = None,
    abort_event: threading.Event | None = None,
) -> None:
    """Process one live camera source until the stream ends or stop is asked.

    A network stream that drops is re-opened with backoff (up to
    ``max_retries`` consecutive failures before giving up — the watcher then
    restarts this worker on a later poll). A plain video-file source runs one
    pass to end-of-file. Detection parameters mirror ``run_pipeline`` and are
    overridden the same way by the config (model, conf/iou, output.*).

    Emits ``camera_status`` / ``camera_event`` / ``camera_error`` dicts via
    ``event_callback``, annotated preview frames via ``frame_callback`` and
    progress lines via ``log_callback``.
    """
    cb_holder = [event_callback]

    def log(message: str) -> None:
        if log_callback is not None:
            log_callback(message)
        else:
            print(message, flush=True)

    def emit(event_type: str, **fields) -> None:
        cb = cb_holder[0]
        if cb is None:
            return
        try:
            cb({"type": event_type, **fields})
        except Exception as e:  # noqa: BLE001 — a dead GUI must not kill us
            cb_holder[0] = None
            log(f"[{camera_name}] WARNING: event callback disabled after error: {e}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path) if config_path else None
    if config:
        model_path = config.get("model", {}).get("path", model_path)
        conf = config.get("model", {}).get("conf", conf)
        iou = config.get("model", {}).get("iou", iou)
        context_seconds = config.get("output", {}).get("context_seconds", context_seconds)
        crop_padding = config.get("output", {}).get("crop_padding", crop_padding)
        debug_keypoints = config.get("output", {}).get("debug_keypoints", debug_keypoints)

    is_stream = is_stream_source(source)
    cap = _open_capture(source)
    if cap is None:
        log(f"[{camera_name}] ERROR: không mở được nguồn: {source}")
        emit("camera_error", camera=camera_name, source=str(source),
             error="Không mở được nguồn video", reconnecting=False)
        return

    reported = cap.get(cv2.CAP_PROP_FPS)
    fps = float(reported) if reported and 0 < reported <= 240 else float(fps_fallback)

    model = load_pose_model(model_path)
    behaviors, classifier_models, face_pipeline, zones_export, crop_region = _setup_detection(
        config, fps
    )
    log(
        f"[{camera_name}] Bắt đầu realtime | model={model_path} | fps={fps:g} | "
        f"behaviors={[b.name for b in behaviors]}"
    )

    stats = {"dropped": 0}
    buffer_capacity = max(2, int(context_seconds * fps) + BUFFER_MARGIN_FRAMES)
    frame_buffer: deque = deque(maxlen=buffer_capacity)
    cache_keep = buffer_capacity + BUFFER_MARGIN_FRAMES
    frame_data_cache: dict = {}

    frame_idx = -1
    events_total = 0
    event_counter = 0
    metadata_events: list[dict] = []
    consecutive_failures = 0
    measured_fps = 0.0
    last_frame_t: float | None = None
    next_preview_t = 0.0
    next_status_t = 0.0

    def _export_event(ev) -> dict | None:
        """Export one completed event from the rolling buffer + notify."""
        nonlocal event_counter, events_total
        event_counter += 1
        try:
            meta = export_event_clip_from_buffer(
                ev, event_counter, frame_buffer, frame_data_cache, output_path, fps,
                context_seconds=context_seconds,
                padding=crop_padding,
                debug_keypoints=debug_keypoints,
                camera_name=camera_name,
                zone_export_info=zones_export,
            )
        except Exception as e:  # noqa: BLE001 — one bad export must not stop the stream
            log(f"[{camera_name}] Lỗi export sự kiện {event_counter}: {e}")
            event_counter -= 1
            return None
        if meta is None:
            # The event's end frame already left the rolling buffer.
            event_counter -= 1
            return None
        events_total += 1
        meta["camera_name"] = camera_name
        meta["source"] = str(source)
        meta["detected_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_events.append(meta)
        log(
            f"[{camera_name}] SỰ KIỆN {meta['event_id']}: {meta['behavior']} "
            f"(track {meta['track_id']}, {meta['start_time_sec']:g}s–{meta['end_time_sec']:g}s) "
            f"→ {meta['clip_path']}"
        )
        emit(
            "camera_event",
            camera=camera_name,
            event=meta,
            behavior=meta.get("behavior", ""),
            events_total=events_total,
            clip=meta.get("clip_path", ""),
        )
        if send_email:
            _send_event_email(camera_name, output_path, meta, log)
        # Crash-safe audit trail: one JSON line per event, appended live.
        try:
            with open(output_path / f"{camera_name}_events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        except OSError as e:
            log(f"[{camera_name}] WARNING: không ghi được events.jsonl: {e}")
        return meta

    def _finalize_events() -> None:
        """Flush the behavior state machines (plain-file end of stream)."""
        for behavior in behaviors:
            for ev in behavior.event_manager.finalize():
                ev.behavior_name = behavior.name
                _export_event(ev)

    def _start_grabber(current_cap) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=GRAB_QUEUE_SIZE)
        threading.Thread(
            target=_grabber_worker,
            args=(current_cap, q, abort_event, stats),
            name=f"grab-{camera_name}",
            daemon=True,
        ).start()
        return q

    grab_q = _start_grabber(cap)

    while True:
        stream_ended = False
        while True:
            if abort_event is not None and abort_event.is_set():
                break
            try:
                frame = grab_q.get(timeout=STALL_TIMEOUT)
            except queue.Empty:
                log(
                    f"[{camera_name}] Không nhận được frame trong {STALL_TIMEOUT:g}s — "
                    f"mất luồng, thử kết nối lại…"
                )
                emit("camera_error", camera=camera_name, source=str(source),
                     error=f"Không nhận được frame trong {STALL_TIMEOUT:g}s",
                     reconnecting=True)
                break
            if frame is None:
                stream_ended = True
                break

            frame_idx += 1
            timestamp = frame_idx / fps
            cropped = _crop_frame(frame, crop_region)
            if cropped is None:
                frame_idx -= 1
                continue
            frame = cropped

            try:
                people = process_frame(model, frame, conf=conf, iou=iou)
            except Exception as e:  # noqa: BLE001 — skip the frame, keep the stream
                log(f"[{camera_name}] process_frame lỗi tại frame {frame_idx}: {e}")
                frame_idx -= 1
                time.sleep(0.01)
                continue

            if face_pipeline is not None:
                face_map = face_pipeline.run(frame, people)
                for person in people:
                    person.face_data = face_map.get(person.track_id)

            all_new_events: list = []
            for behavior in behaviors:
                new_evs = behavior.process_frame(people, frame, frame_idx, timestamp)
                for ev in new_evs:
                    ev.behavior_name = behavior.name
                all_new_events.extend(new_evs)
            if classifier_models and all_new_events:
                all_new_events = _apply_classifier_filter(
                    all_new_events, classifier_models, frame_data_cache, fps
                )

            # Per-person frame_data, same shape as the file pipeline (Phase 2)
            # so annotation, classifier sequences and clip export all work.
            frame_data: dict = {}
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
                    elif behavior.name == "danger_zone":
                        is_detected = behavior.is_person_in_alert(tid, frame_idx)
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
            while len(frame_data_cache) > cache_keep:
                del frame_data_cache[min(frame_data_cache)]

            # Rolling buffer: JPEG keeps RAM low (~0.1-0.4 MB per frame).
            ok_enc, jpeg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok_enc:
                frame_buffer.append((frame_idx, jpeg.tobytes()))

            zone_active = _compute_zone_active(behaviors, frame_data, frame_idx)

            for ev in all_new_events:
                _export_event(ev)

            # Annotated realtime preview (throttled, like the file pipeline).
            if frame_callback is not None:
                now = time.monotonic()
                if now >= next_preview_t:
                    next_preview_t = now + PREVIEW_MIN_INTERVAL
                    try:
                        annotated = _annotate_frame(
                            frame.copy(), frame_idx, people,
                            frame_data_cache, behaviors, zone_active,
                        )
                        frame_callback(
                            frame_idx,
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            {"people": len(people), "events": events_total},
                        )
                    except Exception as e:  # noqa: BLE001
                        log(f"[{camera_name}] Lỗi chú thích frame {frame_idx}: {e}")

            # Status heartbeat for the GUI (progress bar replacement).
            now = time.monotonic()
            if last_frame_t is not None:
                dt = now - last_frame_t
                if dt > 0:
                    inst = 1.0 / dt
                    measured_fps = inst if measured_fps <= 0 else 0.8 * measured_fps + 0.2 * inst
            last_frame_t = now
            if now >= next_status_t:
                next_status_t = now + STATUS_EVERY
                emit(
                    "camera_status",
                    camera=camera_name,
                    fps=round(measured_fps, 1),
                    people=len(people),
                    events_total=events_total,
                    dropped=stats["dropped"],
                    frame=frame_idx,
                )

        # Inner loop exited: stream ended, stalled or stop was requested.
        cap.release()
        if abort_event is not None and abort_event.is_set():
            log(f"[{camera_name}] Đã dừng theo yêu cầu.")
            break
        if stream_ended and not is_stream:
            # Plain video file: one pass is the whole stream — flush the
            # behavior state machines so trailing events are exported too.
            _finalize_events()
            break
        consecutive_failures += 1
        if consecutive_failures > max_retries:
            log(f"[{camera_name}] Bỏ luồng sau {max_retries} lần kết nối lại liên tiếp.")
            emit("camera_error", camera=camera_name, source=str(source),
                 error=f"Bỏ luồng sau {max_retries} lần kết nối lại", reconnecting=False)
            break
        delay = RECONNECT_DELAYS[min(consecutive_failures - 1, len(RECONNECT_DELAYS) - 1)]
        log(
            f"[{camera_name}] Kết nối lại sau {delay:g}s "
            f"(lần {consecutive_failures}/{max_retries})…"
        )
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline and not (
            abort_event is not None and abort_event.is_set()
        ):
            time.sleep(0.2)
        if abort_event is not None and abort_event.is_set():
            break
        new_cap = _open_capture(source)
        if new_cap is None:
            log(f"[{camera_name}] Kết nối lại thất bại.")
            emit("camera_error", camera=camera_name, source=str(source),
                 error="Kết nối lại thất bại", reconnecting=True)
            continue
        cap = new_cap
        consecutive_failures = 0
        grab_q = _start_grabber(cap)

    # Run summary, mirroring the file pipeline's metadata files so the output
    # folder of a camera looks like the output folder of a processed video.
    try:
        write_metadata_files(
            metadata_events, str(source), output_path, fps, frame_idx + 1, camera_name
        )
    except Exception as e:  # noqa: BLE001
        log(f"[{camera_name}] WARNING: không ghi được metadata: {e}")

    log(
        f"[{camera_name}] Kết thúc: {events_total} sự kiện, "
        f"{frame_idx + 1} frame, {stats['dropped']} frame bỏ (chậm xử lý)."
    )