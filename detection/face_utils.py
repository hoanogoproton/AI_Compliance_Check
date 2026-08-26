from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class FaceData:
    track_id: int
    yaw: float | None
    pitch: float | None
    roll: float | None
    landmarks: list | None
    detection_confidence: float
    reprojection_error: float | None
    pose_valid: bool


_3D_FACE_LANDMARKS = np.array([
    [-0.170, -0.140, -0.100],   #  1: left eye (person's left)
    [ 0.000,  0.000,  0.000],   # 33: nose tip (origin)
    [-0.130,  0.080, -0.050],   # 61: left mouth corner
    [ 0.170, -0.140, -0.100],   # 199: right eye (person's right)
    [ 0.130,  0.080, -0.050],   # 263: right mouth corner
    [ 0.000,  0.230, -0.020],   # 291: chin
], dtype=np.float64)

_MP_INDICES = [1, 33, 61, 199, 263, 291]


def get_head_roi(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, int],
    head_ratio: float = 0.3,
    expand_ratio: float = 0.2,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame_shape
    roi_h = (y2 - y1) * head_ratio
    roi_y1 = y1
    roi_y2 = y1 + roi_h
    expand_x = (x2 - x1) * expand_ratio
    expand_y = roi_h * expand_ratio
    roi_x1 = max(0, int(x1 - expand_x))
    roi_y1 = max(0, int(roi_y1 - expand_y))
    roi_x2 = min(w_frame, int(x2 + expand_x))
    roi_y2 = min(h_frame, int(roi_y2 + expand_y))
    if roi_x2 - roi_x1 < 32 or roi_y2 - roi_y1 < 32:
        return None
    return roi_x1, roi_y1, roi_x2, roi_y2


class FacePipeline:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        max_reprojection_error: float = 10.0,
        model_dir: str = "models",
    ):
        self.min_detection_confidence = min_detection_confidence
        self.max_reprojection_error = max_reprojection_error

        model_dir_path = Path(model_dir)
        lm_model = model_dir_path / "face_landmarker.task"
        if not lm_model.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at {lm_model}. "
                f"Download it from the MediaPipe Model Zoo and place it in {model_dir}/"
            )

        base = python.BaseOptions(model_asset_path=str(lm_model))
        lm_opts = vision.FaceLandmarkerOptions(
            base_options=base,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(lm_opts)

    def _estimate_pose(self, landmarks, roi_x1, roi_y1, roi_w, roi_h):
        img_pts = np.zeros((6, 2), dtype=np.float64)
        for i, idx in enumerate(_MP_INDICES):
            lm = landmarks[idx]
            img_pts[i] = [lm.x * roi_w + roi_x1, lm.y * roi_h + roi_y1]
        focal_length = 1.0 * max(roi_w, roi_h)
        camera_matrix = np.array([
            [focal_length, 0, roi_w / 2.0],
            [0, focal_length, roi_h / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            _3D_FACE_LANDMARKS, img_pts, camera_matrix, dist_coeffs
        )
        if not success:
            return None, None
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
            pitch = np.arctan2(-rmat[2, 0], sy)
            roll = np.arctan2(rmat[2, 1], rmat[2, 2])
        else:
            yaw = np.arctan2(-rmat[1, 2], rmat[1, 1])
            pitch = np.arctan2(-rmat[2, 0], sy)
            roll = 0.0
        projected, _ = cv2.projectPoints(
            _3D_FACE_LANDMARKS, rvec, tvec, camera_matrix, dist_coeffs
        )
        reproj_error = cv2.norm(
            img_pts - projected.squeeze(), cv2.NORM_L2
        ) / len(img_pts)
        return (float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))), float(reproj_error)

    def run_on_head_roi(self, frame: np.ndarray, bbox: tuple, track_id: int) -> FaceData:
        roi = get_head_roi(bbox, frame.shape[:2])
        if roi is None:
            return FaceData(
                track_id=track_id, yaw=None, pitch=None, roll=None,
                landmarks=None, detection_confidence=0.0,
                reprojection_error=None, pose_valid=False,
            )
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        roi_rgb = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2RGB)
        mp_image = MpImage(ImageFormat.SRGB, data=roi_rgb)

        lm_result = self._face_landmarker.detect(mp_image)
        if not lm_result or not lm_result.face_landmarks:
            return FaceData(
                track_id=track_id, yaw=None, pitch=None, roll=None,
                landmarks=None, detection_confidence=0.0,
                reprojection_error=None, pose_valid=False,
            )

        detection_confidence = 1.0
        landmarks = lm_result.face_landmarks[0]
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        pose_result, reproj_error = self._estimate_pose(landmarks, roi_x1, roi_y1, roi_w, roi_h)
        if pose_result is None:
            return FaceData(
                track_id=track_id, yaw=None, pitch=None, roll=None,
                landmarks=None, detection_confidence=0.0,
                reprojection_error=None, pose_valid=False,
            )
        yaw, pitch, roll = pose_result
        pose_valid = reproj_error <= self.max_reprojection_error
        return FaceData(
            track_id=track_id, yaw=yaw, pitch=pitch, roll=roll,
            landmarks=landmarks, detection_confidence=detection_confidence,
            reprojection_error=reproj_error, pose_valid=pose_valid,
        )

    def run(self, frame: np.ndarray, people: list) -> dict[int, FaceData]:
        face_map: dict[int, FaceData] = {}
        for person in people:
            fd = self.run_on_head_roi(frame, person.bbox, person.track_id)
            face_map[person.track_id] = fd
        return face_map