from typing import Any

import numpy as np
from dataclasses import dataclass


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray
    conf: float
    face_data: Any | None = None


def process_frame(model, frame, conf=0.3, iou=0.5) -> list[TrackedPerson]:
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml", conf=conf, iou=iou
    )
    people = []
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return people
    boxes = results[0].boxes
    kpts_data = results[0].keypoints.data
    for i in range(len(boxes)):
        track_id = int(boxes.id[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        conf_val = float(boxes.conf[i].item())
        kpts = kpts_data[i].cpu().numpy()
        people.append(
            TrackedPerson(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                keypoints=kpts,
                conf=conf_val,
            )
        )
    return people
