COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

HEAD_KEYPOINT_IDS = [0, 1, 2, 3, 4]
WRIST_KEYPOINT_IDS = [9, 10]

SHOULDER_KEYPOINT_IDS = [5, 6]

KEYPOINT_CONFIDENCE_THRESHOLD = 0.5
HEAD_KEYPOINT_CONFIDENCE_THRESHOLD = 0.5

DISTANCE_THRESHOLD_RATIO = 0.9
VERTICAL_OFFSET_RATIO = 0.2
CONFIRMATION_FRAMES = 30
MAX_GAP_FRAMES = 10
MIN_EVENT_FRAMES = 30

HEAD_TURN_THRESHOLD_RATIO = 0.25
HEAD_TURN_WINDOW_FRAMES = 90
HEAD_TURN_MAX_TURNS = 3

SMOOTHING_MIN_PADDING = 40
