import csv
import json
from pathlib import Path

import cv2
import numpy as np

from detection.config import KEYPOINT_CONFIDENCE_THRESHOLD, SMOOTHING_MIN_PADDING
from detection.video_utils import create_video_writer
from detection.visualizer import draw_zone

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

CENTER_ALPHA = 0.25
SIZE_ALPHA = 0.4


def _compute_zone_crop_rect(zones, padding=80, frame_w=0, frame_h=0):
    all_xs = []
    all_ys = []
    for zone in zones:
        for px, py in zone.points:
            all_xs.append(px)
            all_ys.append(py)
    if not all_xs:
        return None
    x1 = max(0, int(min(all_xs)) - padding)
    y1 = max(0, int(min(all_ys)) - padding)
    x2 = int(max(all_xs)) + padding
    y2 = int(max(all_ys)) + padding
    if frame_w:
        x2 = min(frame_w, x2)
    if frame_h:
        y2 = min(frame_h, y2)
    w = (x2 - x1) // 2 * 2
    h = (y2 - y1) // 2 * 2
    return (x1, y1, x1 + w, y1 + h)


def _draw_debug_overlay(crop, bbox, kpts, offset_x, offset_y, frame_idx, detected):
    bx1, by1, bx2, by2 = map(int, bbox)
    bx1 -= offset_x
    by1 -= offset_y
    bx2 -= offset_x
    by2 -= offset_y
    bbox_color = (0, 0, 255) if detected else (0, 255, 0)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), bbox_color, 3)
    if detected:
        cv2.putText(crop, "DETECTED", (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
    for pt_idx in range(kpts.shape[0]):
        x, y, c = kpts[pt_idx]
        if c > KEYPOINT_CONFIDENCE_THRESHOLD:
            px = int(x - offset_x)
            py = int(y - offset_y)
            if 0 <= px < crop.shape[1] and 0 <= py < crop.shape[0]:
                color = (0, 255, 255)
                if pt_idx in (9, 10):
                    color = (0, 165, 255)
                elif pt_idx in (0, 1, 2, 3, 4):
                    color = (255, 0, 0)
                cv2.circle(crop, (px, py), 5, color, -1)
    for i, j in SKELETON_CONNECTIONS:
        xi, yi, ci = kpts[i]
        xj, yj, cj = kpts[j]
        if ci > KEYPOINT_CONFIDENCE_THRESHOLD and cj > KEYPOINT_CONFIDENCE_THRESHOLD:
            p1 = (int(xi - offset_x), int(yi - offset_y))
            p2 = (int(xj - offset_x), int(yj - offset_y))
            cv2.line(crop, p1, p2, (255, 255, 255), 2)
    cv2.putText(crop, f"frame {frame_idx}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return crop


def export_outputs(
    video_path: str,
    events: list,
    all_frames_data: dict,
    output_dir: Path,
    fps: float,
    context_seconds: int = 5,
    padding: int = 20,
    debug_keypoints: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    context_frames = int(context_seconds * fps)

    event_infos = []
    metadata_events = []
    for event in events:
        event_id = len(metadata_events) + 1
        first_frame = max(0, event.start_frame - context_frames)
        last_frame = min(total_frames - 1, event.end_frame + context_frames)
        clip_path = output_dir / f"event_{event_id}_track_{event.track_id}.mp4"

        duration = 0.0
        if event.end_time > event.start_time:
            duration = event.end_time - event.start_time
        elif fps > 0:
            duration = len(event.frames) / fps

        hand_side = "none"
        if event.hand_sides:
            hand_counts = {}
            for s in event.hand_sides:
                hand_counts[s] = hand_counts.get(s, 0) + 1
            hand_side = max(hand_counts, key=hand_counts.get)

        metadata_events.append({
            "event_id": event_id,
            "track_id": event.track_id,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "start_time_sec": round(event.start_time, 3),
            "end_time_sec": round(event.end_time, 3),
            "duration_sec": round(duration, 3),
            "max_confidence": round(event.max_confidence, 3),
            "hand_side": hand_side,
            "clip_path": str(clip_path.name),
            "clip_start_frame": first_frame,
            "clip_end_frame": last_frame,
            "clip_start_time_sec": round(first_frame / fps, 3) if fps > 0 else 0.0,
            "clip_end_time_sec": round(last_frame / fps, 3) if fps > 0 else 0.0,
        })

        event_infos.append((event, first_frame, last_frame, clip_path, event_id))

    if not event_infos:
        cap.release()
        metadata = {"source_video": str(video_path), "fps": fps, "total_frames": total_frames, "events": []}
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        Path(output_dir / "metadata.csv").write_text("")
        return []

    global_first = min(info[1] for info in event_infos)
    global_last = max(info[2] for info in event_infos)

    cap.set(cv2.CAP_PROP_POS_FRAMES, global_first)

    writers = {}
    debug_writers = {}
    last_known_bboxes = {}
    smoothed_centers = {}
    smoothed_sizes = {}
    fixed_crop_sizes = {}
    frame_h, frame_w = 0, 0

    try:
        for f_idx in range(global_first, global_last + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_h == 0:
                frame_h, frame_w = frame.shape[:2]

            for event, first_frame, last_frame, clip_path, event_id in event_infos:
                if not (first_frame <= f_idx <= last_frame):
                    continue

                frame_data = all_frames_data.get(f_idx, {})
                person_data = frame_data.get(event.track_id)
                if person_data is not None:
                    last_known_bboxes[event_id] = person_data["bbox"]

                bbox = last_known_bboxes.get(event_id)
                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1

                # Strong EMA on center -> steady-cam effect
                prev_cx, prev_cy = smoothed_centers.get(event_id, (cx, cy))
                scx = CENTER_ALPHA * cx + (1.0 - CENTER_ALPHA) * prev_cx
                scy = CENTER_ALPHA * cy + (1.0 - CENTER_ALPHA) * prev_cy
                smoothed_centers[event_id] = (scx, scy)

                # Lighter EMA on size to avoid sudden crop resize
                prev_bw, prev_bh = smoothed_sizes.get(event_id, (bw, bh))
                sbw = SIZE_ALPHA * bw + (1.0 - SIZE_ALPHA) * prev_bw
                sbh = SIZE_ALPHA * bh + (1.0 - SIZE_ALPHA) * prev_bh
                smoothed_sizes[event_id] = (sbw, sbh)

                # Lock crop size on first valid frame
                crop_pad = max(padding, SMOOTHING_MIN_PADDING)
                if event_id not in fixed_crop_sizes:
                    cw = int(sbw + 2 * crop_pad)
                    ch = int(sbh + 2 * crop_pad)
                    fixed_crop_sizes[event_id] = (cw, ch)
                target_w, target_h = fixed_crop_sizes[event_id]

                half_w = target_w // 2
                half_h = target_h // 2
                cx1 = int(scx - half_w)
                cy1 = int(scy - half_h)
                cx2 = cx1 + target_w
                cy2 = cy1 + target_h

                # Clamp to frame boundaries
                cx1 = max(0, cx1)
                cy1 = max(0, cy1)
                cx2 = min(frame_w, cx2)
                cy2 = min(frame_h, cy2)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                # Pad with black if cropped region smaller than target
                if crop.shape[1] != target_w or crop.shape[0] != target_h:
                    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    ph = min(crop.shape[0], target_h)
                    pw = min(crop.shape[1], target_w)
                    padded[:ph, :pw] = crop[:ph, :pw]
                    crop = padded

                if event_id not in writers:
                    writers[event_id] = create_video_writer(clip_path, "mp4v", fps, (target_w, target_h))
                if writers[event_id].isOpened():
                    writers[event_id].write(crop)

                # Debug clip with keypoints overlay
                if debug_keypoints and person_data is not None:
                    kpts = person_data["keypoints"]
                    detected = bool(person_data.get("detected", False))
                    debug_crop = crop.copy()
                    _draw_debug_overlay(debug_crop, bbox, kpts, cx1, cy1, f_idx, detected)
                    if event_id not in debug_writers:
                        debug_path = clip_path.with_stem(clip_path.stem + "_debug")
                        debug_writers[event_id] = create_video_writer(debug_path, "mp4v", fps, (target_w, target_h))
                    if debug_writers[event_id].isOpened():
                        debug_writers[event_id].write(debug_crop)
    except Exception as e:
        raise RuntimeError(f"Export error: {e}") from e
    finally:
        for w in writers.values():
            w.release()
        for w in debug_writers.values():
            w.release()
        cap.release()

    metadata = {"source_video": str(video_path), "fps": fps, "total_frames": total_frames, "events": metadata_events}
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if metadata_events:
        with open(output_dir / "metadata.csv", "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=metadata_events[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(metadata_events)
    else:
        Path(output_dir / "metadata.csv").write_text("")
    return metadata_events


def write_metadata_files(metadata_events, video_path, output_dir, fps, total_frames, video_stem=""):
    meta_name = f"{video_stem}_metadata" if video_stem else "metadata"
    metadata = {"source_video": str(video_path), "fps": fps, "total_frames": total_frames, "events": metadata_events}
    with open(output_dir / f"{meta_name}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if metadata_events:
        with open(output_dir / f"{meta_name}.csv", "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=metadata_events[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(metadata_events)
    else:
        Path(output_dir / f"{meta_name}.csv").write_text("")


def export_single_event(
    event,
    event_id,
    video_path,
    all_frames_data,
    output_dir,
    fps,
    context_seconds=5,
    padding=20,
    debug_keypoints=False,
    video_stem="",
    zone_export_info=None,
    crop_region=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    context_frames = int(context_seconds * fps)

    first_frame = max(0, event.start_frame - context_frames)
    last_frame = min(total_frames - 1, event.end_frame + context_frames)
    behavior_name = getattr(event, "behavior_name", "")
    behavior_prefix = f"{behavior_name}_" if behavior_name else ""
    clip_path = output_dir / f"{video_stem}_{behavior_prefix}event_{event_id}_track_{event.track_id}.mp4" if video_stem else output_dir / f"{behavior_prefix}event_{event_id}_track_{event.track_id}.mp4"

    hand_side = "none"
    if event.hand_sides:
        hand_counts = {}
        for s in event.hand_sides:
            hand_counts[s] = hand_counts.get(s, 0) + 1
        hand_side = max(hand_counts, key=hand_counts.get)

    duration = 0.0
    if event.end_time > event.start_time:
        duration = event.end_time - event.start_time
    elif fps > 0:
        duration = len(event.frames) / fps

    metadata_event = {
        "event_id": event_id,
        "behavior": behavior_name,
        "track_id": event.track_id,
        "start_frame": event.start_frame,
        "end_frame": event.end_frame,
        "start_time_sec": round(event.start_time, 3),
        "end_time_sec": round(event.end_time, 3),
        "duration_sec": round(duration, 3),
        "max_confidence": round(event.max_confidence, 3),
        "hand_side": hand_side,
        "clip_path": str(clip_path.name),
        "clip_start_frame": first_frame,
        "clip_end_frame": last_frame,
        "clip_start_time_sec": round(first_frame / fps, 3) if fps > 0 else 0.0,
        "clip_end_time_sec": round(last_frame / fps, 3) if fps > 0 else 0.0,
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    writer = None
    debug_writer = None
    last_known_bbox = None
    smoothed_center = None
    smoothed_size = None
    fixed_crop_size = None
    frame_h, frame_w = 0, 0
    zone_crop_rect = None

    try:
        for f_idx in range(first_frame, last_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_h == 0:
                frame_h, frame_w = frame.shape[:2]
                if zone_export_info:
                    zone_crop_rect = _compute_zone_crop_rect([zi["zone"] for zi in zone_export_info], frame_w=frame_w, frame_h=frame_h)

            if crop_region:
                cx, cy, cw, ch = crop_region
                x_end = min(cx + cw, frame_w)
                y_end = min(cy + ch, frame_h)
                frame = frame[cy:y_end, cx:x_end]

            frame_data = all_frames_data.get(f_idx, {})
            person_data = frame_data.get(event.track_id)
            if person_data is not None:
                last_known_bbox = person_data["bbox"]

            if last_known_bbox is None:
                continue

            frame_annotated = frame.copy()
            for zi in (zone_export_info or []):
                is_active = any(
                    fd.get("behaviors", {}).get(zi["behavior_name"], {}).get("detected", False)
                    for fd in frame_data.values()
                )
                frame_annotated = draw_zone(frame_annotated, zi["zone"], is_active)

            x1, y1, x2, y2 = last_known_bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            if smoothed_center is None:
                smoothed_center = (cx, cy)
            prev_cx, prev_cy = smoothed_center
            scx = CENTER_ALPHA * cx + (1.0 - CENTER_ALPHA) * prev_cx
            scy = CENTER_ALPHA * cy + (1.0 - CENTER_ALPHA) * prev_cy
            smoothed_center = (scx, scy)

            if smoothed_size is None:
                smoothed_size = (bw, bh)
            prev_bw, prev_bh = smoothed_size
            sbw = SIZE_ALPHA * bw + (1.0 - SIZE_ALPHA) * prev_bw
            sbh = SIZE_ALPHA * bh + (1.0 - SIZE_ALPHA) * prev_bh
            smoothed_size = (sbw, sbh)

            crop_pad = max(padding, SMOOTHING_MIN_PADDING)
            if fixed_crop_size is None:
                cw = int(sbw + 2 * crop_pad)
                ch = int(sbh + 2 * crop_pad)
                fixed_crop_size = (cw, ch)
            target_w, target_h = fixed_crop_size

            half_w = target_w // 2
            half_h = target_h // 2
            cx1 = int(scx - half_w)
            cy1 = int(scy - half_h)
            cx2 = cx1 + target_w
            cy2 = cy1 + target_h

            cx1 = max(0, cx1)
            cy1 = max(0, cy1)
            cx2 = min(frame_w, cx2)
            cy2 = min(frame_h, cy2)

            crop = frame_annotated[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            if crop.shape[1] != target_w or crop.shape[0] != target_h:
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                ph = min(crop.shape[0], target_h)
                pw = min(crop.shape[1], target_w)
                padded[:ph, :pw] = crop[:ph, :pw]
                crop = padded

            if writer is None:
                writer = create_video_writer(clip_path, "mp4v", fps, (target_w, target_h))
            if writer.isOpened():
                writer.write(crop)

            if debug_keypoints and person_data is not None:
                kpts = person_data["keypoints"]
                detected = bool(person_data.get("detected", False))
                if zone_export_info and zone_crop_rect:
                    zx1, zy1, zx2, zy2 = zone_crop_rect
                    debug_crop = frame_annotated[zy1:zy2, zx1:zx2]
                    if debug_crop.size == 0:
                        continue
                    _draw_debug_overlay(debug_crop, last_known_bbox, kpts, zx1, zy1, f_idx, detected)
                    dw, dh = debug_crop.shape[1] // 2 * 2, debug_crop.shape[0] // 2 * 2
                    if debug_crop.shape[1] != dw or debug_crop.shape[0] != dh:
                        debug_crop = cv2.resize(debug_crop, (dw, dh))
                else:
                    debug_crop = crop.copy()
                    _draw_debug_overlay(debug_crop, last_known_bbox, kpts, cx1, cy1, f_idx, detected)
                if debug_writer is None:
                    debug_path = clip_path.with_stem(clip_path.stem + "_debug")
                    debug_writer = create_video_writer(debug_path, "mp4v", fps, (debug_crop.shape[1], debug_crop.shape[0]))
                if debug_writer.isOpened():
                    debug_writer.write(debug_crop)
    except Exception as e:
        raise RuntimeError(f"Export error for event {event_id} at frame {f_idx}: {e}") from e
    finally:
        if writer is not None:
            writer.release()
        if debug_writer is not None:
            debug_writer.release()
        cap.release()

    return metadata_event
