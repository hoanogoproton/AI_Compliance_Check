"""
Diagnostic script: Debug hand_to_head miss at second 59 for track ID 2.
Runs detection on frames around second 59 and prints frame-by-frame diagnostics.
"""
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from detection.detector import process_frame
from detection.pose_utils import compute_head_center, compute_shoulder_width, get_keypoint
from detection.behaviors.hand_to_head import HandToHeadBehavior
from detection.config_loader import load_config

# --- Config ---
VIDEO_PATH = r"D:\Project Demo\DEMO - Nhan dien nguoi\videos\Video camera DAHUA.mp4"
CONFIG_PATH = r"D:\Project Demo\DEMO - Nhan dien nguoi\config.yaml"
TARGET_SECOND = 59
TARGET_TRACK_ID = 2
FRAME_WINDOW_BEFORE = 10  # frames to inspect before
FRAME_WINDOW_AFTER = 15   # frames to inspect after

# --- Load config and model ---
config = load_config(CONFIG_PATH)
model = YOLO(config["model"]["path"])
crop = config.get("crop", None)  # [453, 100, 483, 360]
conf = config["model"]["conf"]
iou = config["model"]["iou"]

# Build hand_to_head behavior
ht_params = None
for bcfg in config["behaviors"]:
    if bcfg["name"] == "hand_to_head" and bcfg.get("enabled", True):
        ht_params = bcfg["params"]
        break

if ht_params is None:
    print("ERROR: hand_to_head not enabled in config!")
    sys.exit(1)

print("=" * 80)
print(f"CONFIG PARAMS for hand_to_head:")
print(f"  distance_threshold_ratio = {ht_params['distance_threshold_ratio']}")
print(f"  vertical_offset_ratio    = {ht_params['vertical_offset_ratio']}")
print(f"  keypoint_conf_threshold  = {ht_params['keypoint_conf_threshold']}")
print(f"  head_keypoint_conf_threshold = {ht_params['head_keypoint_conf_threshold']}")
print(f"  confirmation_frames      = {ht_params['confirmation_frames']}")
print(f"  max_gap_frames           = {ht_params['max_gap_frames']}")
print(f"  min_event_frames         = {ht_params['min_event_frames']}")

behavior = HandToHeadBehavior(ht_params)
behavior.event_manager.confirmation_frames = 1   # Override: report per-frame
behavior.event_manager.max_gap_frames = 100
behavior.event_manager.min_event_frames = 1

# --- Open video ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: FPS={fps:.4f}, Total frames={total_frames}, Crop={crop}")

# Calculate frame index at target second
target_frame = int(TARGET_SECOND * fps)
start_frame = max(0, target_frame - FRAME_WINDOW_BEFORE)
end_frame = min(total_frames - 1, target_frame + FRAME_WINDOW_AFTER)
print(f"Target second {TARGET_SECOND}s => frame ~{target_frame}")
print(f"Inspecting frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")
print("=" * 80)

# --- Process frames ---
COCO_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
}

for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < start_frame:
        continue
    if frame_idx > end_frame:
        break

    # Apply crop
    frame_proc = frame.copy()
    if crop:
        x, y, w, h = crop
        h_f, w_f = frame.shape[:2]
        x_end = min(x + w, w_f)
        y_end = min(y + h, h_f)
        frame_proc = frame[y:y_end, x:x_end]

    timestamp = frame_idx / fps
    people = process_frame(model, frame_proc, conf=conf, iou=iou)

    # Find track ID 2
    target_person = None
    for p in people:
        if p.track_id == TARGET_TRACK_ID:
            target_person = p
            break

    if target_person is None:
        # Scan for any person
        all_ids = [p.track_id for p in people]
        if all_ids:
            print(f"[Frame {frame_idx} | t={timestamp:.2f}s] Track IDs present: {all_ids} (ID {TARGET_TRACK_ID} NOT found)")
        continue

    p = target_person
    kpts = p.keypoints

    # === Run detection with detailed diagnostics ===
    kpt_conf_th = ht_params["keypoint_conf_threshold"]
    head_conf_th = ht_params["head_keypoint_conf_threshold"]
    dist_ratio = ht_params["distance_threshold_ratio"]
    vert_offset = ht_params["vertical_offset_ratio"]

    # Print all keypoints for this person
    print(f"\n{'─' * 70}")
    print(f"[Frame {frame_idx} | t={timestamp:.2f}s] Track ID={p.track_id}, bbox={p.bbox}, conf={p.conf:.4f}")
    print(f"  Keypoints (x, y, conf):")
    for kp_idx in range(17):
        kx, ky, kc = get_keypoint(kpts, kp_idx)
        name = COCO_NAMES.get(kp_idx, f"kp_{kp_idx}")
        flag = ""
        if kp_idx in (3, 4, 1, 2, 0):
            flag += f" [head check: conf>={head_conf_th}? {'PASS' if kc >= head_conf_th else 'FAIL'}]"
        elif kp_idx in (5, 6):
            flag += f" [shoulder conf>={kpt_conf_th}? {'PASS' if kc >= kpt_conf_th else 'FAIL'}]"
        elif kp_idx in (9, 10):
            flag += f" [wrist conf>={kpt_conf_th}? {'PASS' if kc >= kpt_conf_th else 'FAIL'}]"
        print(f"    kp[{kp_idx:2d}] {name:15s}: x={kx:7.1f}, y={ky:7.1f}, conf={kc:.4f}{flag}")

    # Step 1: Head center
    head_center = compute_head_center(kpts, head_conf_threshold=head_conf_th)
    print(f"\n  [Step 1] Head center: {head_center}")
    if head_center is None:
        # Show which cascade levels failed
        le_x, le_y, le_c = get_keypoint(kpts, 3)
        re_x, re_y, re_c = get_keypoint(kpts, 4)
        li_x, li_y, li_c = get_keypoint(kpts, 1)
        ri_x, ri_y, ri_c = get_keypoint(kpts, 2)
        nx, ny, nc = get_keypoint(kpts, 0)
        print(f"    -> Ears fallback:  left_ear conf={le_c:.4f} (need>={head_conf_th}), right_ear conf={re_c:.4f} (need>={head_conf_th}) => {'PASS' if le_c>=head_conf_th and re_c>=head_conf_th else 'FAIL'}")
        print(f"    -> Eyes fallback:  left_eye conf={li_c:.4f} (need>={head_conf_th}), right_eye conf={ri_c:.4f} (need>={head_conf_th}) => {'PASS' if li_c>=head_conf_th and ri_c>=head_conf_th else 'FAIL'}")
        print(f"    -> Nose fallback:  nose conf={nc:.4f} (need>=0.5 -- BUG: uses global KEYPOINT_CONFIDENCE_THRESHOLD instead of head_conf_th={head_conf_th}) => {'PASS' if nc>=0.5 else 'FAIL'}")
        print(f"  >>> MISS: head_center is None -> hand_to_head returns NOT detected")
        continue

    hx, hy = head_center

    # Step 2: Shoulder width
    shoulder_width = compute_shoulder_width(kpts, kpt_conf_threshold=kpt_conf_th)
    print(f"  [Step 2] Shoulder width: {shoulder_width:.2f} px (min required: 20.0)")
    if shoulder_width < 20.0:
        ls_x, ls_y, ls_c = get_keypoint(kpts, 5)
        rs_x, rs_y, rs_c = get_keypoint(kpts, 6)
        print(f"    -> Left shoulder:  ({ls_x:.1f}, {ls_y:.1f}, conf={ls_c:.4f})")
        print(f"    -> Right shoulder: ({rs_x:.1f}, {rs_y:.1f}, conf={rs_c:.4f})")
        print(f"  >>> MISS: shoulder_width < 20.0 -> detection skipped")
        continue

    # Step 3: Distance threshold
    max_allowed = shoulder_width * dist_ratio
    print(f"  [Step 3] Max allowed wrist-head distance: {max_allowed:.2f} px (shoulder_width={shoulder_width:.2f} * ratio={dist_ratio})")

    # Step 4: Check each wrist
    print(f"  [Step 4] Wrist checks (head_center=({hx:.1f}, {hy:.1f})):")
    detected_this_frame = False
    total_detected = False
    for wrist_idx, side_name in [(9, "left"), (10, "right")]:
        wx, wy, wc = get_keypoint(kpts, wrist_idx)
        print(f"    {side_name} wrist (kp[{wrist_idx}]): x={wx:.1f}, y={wy:.1f}, conf={wc:.4f}")

        # Check 1: Confidence
        if wc < kpt_conf_th:
            print(f"      -> FAIL: conf {wc:.4f} < threshold {kpt_conf_th}")
            continue

        # Check 2: Vertical offset
        vert_limit = hy + shoulder_width * vert_offset
        print(f"      -> Vertical check: wrist y={wy:.1f}, head_y={hy:.1f}, limit=head_y + {shoulder_width:.2f}*{vert_offset} = {vert_limit:.2f}")
        if wy > vert_limit:
            print(f"      -> FAIL: wrist too low (wy={wy:.1f} > limit={vert_limit:.2f})")
            continue

        # Check 3: Distance
        dist = np.sqrt((wx - hx) ** 2 + (wy - hy) ** 2)
        print(f"      -> Distance: {dist:.2f} px (max allowed: {max_allowed:.2f})")
        if dist < max_allowed:
            conf_val = 1.0 - (dist / max_allowed)
            print(f"      -> PASS! confidence={conf_val:.4f}")
            detected_this_frame = True
        else:
            surplus = dist - max_allowed
            print(f"      -> FAIL: distance {dist:.2f} > max {max_allowed:.2f} (exceeds by {surplus:.2f} px, {surplus/max_allowed*100:.1f}%)")

    if detected_this_frame:
        print(f"  >>> DETECTED: hand_to_head = TRUE for track {p.track_id}")
    else:
        print(f"  >>> MISS: hand_to_head = FALSE for track {p.track_id}")

cap.release()
print("\n" + "=" * 80)
print("Diagnostic complete.")