import threading
from PySide6.QtCore import QObject, Signal
from detection.pipeline import run_pipeline


class PipelineWorker(QObject):
    progress = Signal(int, int)
    log = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._thread: threading.Thread | None = None

    def start(self, video_path, model_path, output_dir, conf, iou, visualize,
              context_seconds, crop_padding, debug_keypoints, config_path):
        self._thread = threading.Thread(
            target=self._run,
            args=(video_path, model_path, output_dir, conf, iou, visualize,
                  context_seconds, crop_padding, debug_keypoints, config_path),
            daemon=True,
        )
        self._thread.start()

    def _run(self, video_path, model_path, output_dir, conf, iou, visualize,
             context_seconds, crop_padding, debug_keypoints, config_path):
        try:
            run_pipeline(
                video_path=video_path,
                model_path=model_path,
                output_dir=output_dir,
                conf=conf,
                iou=iou,
                visualize=visualize,
                context_seconds=context_seconds,
                crop_padding=crop_padding,
                debug_keypoints=debug_keypoints,
                config_path=config_path,
                progress_callback=self.progress.emit,
                log_callback=self.log.emit,
            )
            self.finished.emit({"output_dir": output_dir, "video_path": video_path})
        except Exception as e:
            self.error.emit(str(e))