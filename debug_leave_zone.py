"""
Diagnostic script: Debug leave_zone false positive on ban sub 3.mp4 + config_3.yaml.
Runs detection frame-by-frame, traces leave_zone state machine,
and prints detailed diagnostics when leave events fire.

Root cause hypothesis: bounding box jitter caused by unstable YOLO tracking
makes the bbox momentarily shift outside the zone, triggering an immediate
leave event despite the person still being physically in the zone.
"""
import os
os.environ["YOLO_VERBOSE"] = "false"

import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
ULTRALYTICS_LOGGER.setLevel(logging.ERROR)
from detection.detector import process_frame
from detection.behaviors.leave_zone import LeaveZoneBehavior
from detection.config_loader import load_config
from detection.zones.zone_definition import Zone

VIDEO_PATH = r"D:\Project Demo\DEMO - Nhan dien nguoi\videos\ban sub 3.mp4"
CONFIG_PATH = r"D:\Project Demo\DEMO - Nhan dien nguoi\config_3.yaml"

config = load_config(CONFIG_PATH)
model = YOLO(config["model"]["path"])
crop = config.get("crop", None)
conf = config["model"]["conf"]
iou = config["model"]["iou"]

print("=" * 90)
print("LEAVE_ZONE FALSE POSITIVE DIAGNOSTIC")
print("=" * 90)
print(f"Video: {VIDEO_PATH}")
print(f"Crop region:  {crop}")
print(f"Model conf:   {conf}, iou: {iou}")

zone_dict = config.get("zones", {})
zone_name = "Leave"
zdata = zone_dict.get(zone_name)
if zdata is None:
    print(f"ERROR: Zone '{zone_name}' not found in config zones!")
    sys.exit(1)

zone = Zone.from_dict(zone_name, zdata)
print(f"Zone '{zone_name}': label={zone.label}")
print(f"  Points: {zone.points}")
print(f"  Polygon shape: {zone.polygon.shape}")

lv_params = None
for bcfg in config["behaviors"]:
    if bcfg["name"] == "leave_zone" and bcfg.get("enabled", True):
        lv_params = bcfg["params"]
        break

if lv_params is None:
    print("ERROR: leave_zone not enabled in config!")
    sys.exit(1)

min_stay_frames = lv_params.get("min_stay_frames", 10)
leave_flash_frames = lv_params.get("leave_flash_frames", 20)
max_missing = lv_params.get("max_missing_frames", 15)

print(f"  min_stay_frames:    {min_stay_frames}")
print(f"  leave_flash_frames: {leave_flash_frames}")
print(f"  max_missing_frames: {max_missing}")

behavior = LeaveZoneBehavior(lv_params, zones=[zone])

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\nFPS: {fps}, Total frames: {total_frames}")

CONTEXT_BEFORE = 15
CONTEXT_AFTER = 5

leave_events = []
frame_log = []

print("\n" + "=" * 90)
print("PROCESSING FRAMES...")
print("=" * 90)

for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    frame_proc = frame.copy()
    if crop:
        x, y, w, h = crop
        h_f, w_f = frame.shape[:2]
        x_end = min(x + w, w_f)
        y_end = min(y + h, h_f)
        frame_proc = frame[y:y_end, x:x_end]

    timestamp = frame_idx / fps
    people = process_frame(model, frame_proc, conf=conf, iou=iou)
    track_ids = [p.track_id for p in people]

    state_before = {}
    for tid in track_ids:
        state_before[tid] = {
            "inside": behavior._track_inside.get(tid, None),
            "counter": behavior._track_inside_counter.get(tid, 0),
            "inside_zones": sorted(behavior._track_inside_zones.get(tid, set())),
            "last_leave": behavior._last_leave_frame.get(tid, None),
            "in_flash": behavior.is_person_in_flash(tid, frame_idx),
        }

    events = behavior.process_frame(people, frame_proc, frame_idx, timestamp)

    for tid in track_ids:
        p = next((p for p in people if p.track_id == tid), None)
        if p is None:
            continue
        bbox = p.bbox
        x1, y1, x2, y2 = bbox
        is_inside = zone.intersects_bbox(bbox)
        center_inside = zone.contains_bbox_center(bbox)
        any_corner = any(zone.contains_point(cx, cy) for cx, cy in [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        ])
        corners_in = sum(1 for cx, cy in [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        ] if zone.contains_point(cx, cy))

        frame_log.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "track_id": tid,
            "bbox": bbox,
            "bbox_w": x2 - x1,
            "bbox_h": y2 - y1,
            "bbox_center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "is_inside": is_inside,
            "center_inside": center_inside,
            "corners_in": corners_in,
            "track_inside": behavior._track_inside.get(tid, False),
            "counter": behavior._track_inside_counter.get(tid, 0),
            "state_before": state_before.get(tid, {}),
        })

    for ev in events:
        leave_events.append({
            "event": ev,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
        })

cap.release()

print(f"\nProcessing complete. {len(leave_events)} leave events detected.")
print("=" * 90)

if not leave_events:
    print("\nNo leave_zone events detected. The false positive may not reproduce")
    print("with current config. Try running the pipeline to check outputs.")
    sys.exit(0)

for i, ev_info in enumerate(leave_events):
    ev = ev_info["event"]
    ev_frame = ev_info["frame_idx"]
    ev_ts = ev_info["timestamp"]

    print(f"\n{'#' * 90}")
    print(f"LEAVE EVENT #{i + 1}")
    print(f"{'#' * 90}")
    print(f"  Trigger frame: {ev_frame} ({ev_ts:.2f}s)")
    print(f"  Track ID:      {ev.track_id}")
    print(f"  Metadata:      {ev.metadata}")

    start_log = max(0, ev_frame - CONTEXT_BEFORE)
    end_log = min(total_frames, ev_frame + CONTEXT_AFTER)

    relevant_logs = [
        fl for fl in frame_log
        if fl["track_id"] == ev.track_id and start_log <= fl["frame_idx"] <= end_log
    ]

    if not relevant_logs:
        print(f"  WARNING: No frame log found for track {ev.track_id} in window [{start_log}, {end_log}]")
        continue

    print(f"\n  Frame-by-frame trace (frames {start_log}-{end_log} for track {ev.track_id}):")
    print(f"  {'Frame':>6s} {'Time (s)':>8s} {'bbox_x1':>7s} {'bbox_y1':>7s} {'bbox_x2':>7s} {'bbox_y2':>7s}"
          f" {'W':>6s} {'H':>6s} {'is_inside':>9s} {'cntr_in':>7s} {'crn_in':>6s}"
          f" {'track_in':>8s} {'counter':>7s} {'NOTE'}")

    prev_inside = None
    for fl in relevant_logs:
        fid = fl["frame_idx"]
        ts = fl["timestamp"]
        x1, y1, x2, y2 = fl["bbox"]
        bw = fl["bbox_w"]
        bh = fl["bbox_h"]
        inside = fl["is_inside"]
        cntr = fl["center_inside"]
        crn = fl["corners_in"]
        trk_in = fl["track_inside"]
        cnt = fl["counter"]

        note = ""
        if fid == ev_frame:
            note = " <<< LEAVE EVENT FIRED"
        elif prev_inside is True and inside is False:
            note = " bbox jumped OUTSIDE (jitter)"
        elif prev_inside is False and inside is True:
            note = " bbox jumped INSIDE"
        elif inside and cnt >= min_stay_frames and not fl.get("state_before", {}).get("inside"):
            note = " ENTERED zone"

        prev_inside = inside

        print(f"  {fid:6d} {ts:8.2f} {x1:7.1f} {y1:7.1f} {x2:7.1f} {y2:7.1f}"
              f" {bw:6.1f} {bh:6.1f} {str(inside):>9s} {str(cntr):>7s} {crn:6d}"
              f" {str(trk_in):>8s} {cnt:7d}{note}")

    print(f"\n  DIAGNOSTIC SUMMARY for event #{i + 1}:")
    outside_frames = [fl for fl in relevant_logs if not fl["is_inside"]]
    inside_frames = [fl for fl in relevant_logs if fl["is_inside"]]

    print(f"    Window size: {len(relevant_logs)} frames")
    print(f"    Frames INSIDE zone:  {len(inside_frames)}")
    print(f"    Frames OUTSIDE zone: {len(outside_frames)}")

    if outside_frames:
        consecutive_outs = []
        prev_fid = None
        run_start = None
        for fl in sorted(outside_frames, key=lambda x: x["frame_idx"]):
            fid = fl["frame_idx"]
            if prev_fid is None or fid != prev_fid + 1:
                if run_start is not None:
                    consecutive_outs.append((run_start, prev_fid, prev_fid - run_start + 1))
                run_start = fid
            prev_fid = fid
        if run_start is not None:
            consecutive_outs.append((run_start, prev_fid, prev_fid - run_start + 1))

        print(f"    Consecutive out-of-zone sequences:")
        for start_f, end_f, length in consecutive_outs:
            print(f"      Frames {start_f}-{end_f}: {length} consecutive frames OUTSIDE")

        max_consecutive = max(length for _, _, length in consecutive_outs) if consecutive_outs else 0
        print(f"    Max consecutive outside frames: {max_consecutive}")

        triggers = [fl for fl in outside_frames if fl["frame_idx"] == ev_frame]
        if triggers:
            tf = triggers[0]
            print(f"    Trigger frame bbox: ({tf['bbox'][0]:.1f}, {tf['bbox'][1]:.1f}, "
                  f"{tf['bbox'][2]:.1f}, {tf['bbox'][3]:.1f})")
            print(f"    Trigger frame corners inside: {tf['corners_in']}/4")
            print(f"    Trigger frame center inside: {tf['center_inside']}")

            fl_before = [fl for fl in relevant_logs if fl["frame_idx"] == ev_frame - 1]
            if fl_before:
                fb = fl_before[0]
                dx = fb["bbox_center"][0] - tf["bbox_center"][0]
                dy = fb["bbox_center"][1] - tf["bbox_center"][1]
                print(f"    Bbox movement from prev frame: dx={dx:.1f}px, dy={dy:.1f}px")

    else:
        max_consecutive = 0
    severity = "LIKELY FALSE POSITIVE" if max_consecutive <= 3 else "POSSIBLE GENUINE LEAVE"
    print(f"\n    VERDICT: {severity}")
    print(f"    If max_consecutive_outside <= 2-3 frames, bbox jitter is confirmed.")
    print(f"    Solution: add 'min_leave_frames' parameter for debounce/hysteresis.")

print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)