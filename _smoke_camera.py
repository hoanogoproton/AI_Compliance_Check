"""Smoke test cho camera realtime (không cần camera thật).

A) WatchRunner + dòng camera rtsp:// chết -> phải phát camera_error/camera_stopped.
B) run_realtime_camera với một file mp4 nhỏ (nguồn file = 1 pass, EOF) dùng
   model thật -> phải chạy trọn vòng lặp và ghi metadata tổng kết.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

tmp = ROOT / "outputs" / "_smoke_cam"
tmp.mkdir(parents=True, exist_ok=True)

vid = tmp / "tiny.mp4"
writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
for _ in range(40):
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
writer.release()

# --- A: WatchRunner với URL chết (không cần model — open fail trước) ----------
import watch_folders as wf  # noqa: E402

csv_file = tmp / "watch.csv"
csv_file.write_text(
    "folder,config\n"
    f"rtsp://127.0.0.1:9/nope?channel=1&subtype=0,{ROOT / 'config' / 'config_4.yaml'}\n",
    encoding="utf-8",
)
events: list[dict] = []
runner = wf.WatchRunner(
    csv_path=csv_file,
    poll_interval=0.01,
    output_dir=tmp / "out",
    journal_path=tmp / "state.json",
    send_email=False,
    camera_max_retries=1,
    max_cycles=1,
    event_callback=events.append,
)
code = runner.run()
# The camera thread may still be opening the (unreachable) stream — FFmpeg's
# built-in stream timeout is ~30s — so wait for it to finish erroring out.
for t in list(runner._camera_threads.values()):
    t.join(timeout=60)
types = [e["type"] for e in events]
print("\n[A] events:", types)
assert "camera_start" in types, types
assert "camera_error" in types, types
assert "camera_stopped" in types, types
# Journal has a "cameras" section (empty until the first camera_event).
assert "cameras" in runner.journal, runner.journal.keys()
print("[A] OK — runner bắt đầu/dừng camera, journal có mục 'cameras':", runner.journal["cameras"])

# --- B: run_realtime_camera với model thật trên file nhỏ ----------------------
import detection.realtime as rt  # noqa: E402

evs: list[dict] = []
rt.run_realtime_camera(
    source=str(vid),
    camera_name="smoke",
    output_dir=str(tmp / "out2"),
    fps_fallback=10.0,
    send_email=False,
    event_callback=evs.append,
    log_callback=print,
)
print("\n[B] events:", [e["type"] for e in evs])
meta_json = tmp / "out2" / "smoke_metadata.json"
assert meta_json.exists(), "metadata tổng kết phải được ghi khi worker kết thúc"
data = json.loads(meta_json.read_text(encoding="utf-8"))
assert "events" in data and "fps" in data
print("[B] OK — metadata:", meta_json.name, "| events:", len(data["events"]))
print("\nSMOKE TEST PASSED")
