from ultralytics import YOLO


def load_pose_model(model_path: str = "yolo26m-pose.pt") -> YOLO:
    return YOLO(model_path)