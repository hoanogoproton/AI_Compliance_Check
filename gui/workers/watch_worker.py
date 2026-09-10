import threading
import traceback

from PySide6.QtCore import QObject, Signal

from watch_folders import WatchRunner


class WatchWorker(QObject):
    """Runs WatchRunner in a daemon thread, bridging its event dicts to Qt."""

    event = Signal(dict)
    frame_ready = Signal(str, int, object, dict)
    finished = Signal(int)

    def __init__(self):
        super().__init__()
        self._thread: threading.Thread | None = None
        self._runner: WatchRunner | None = None
        self._stop_requested = threading.Event()

    def start(self, **runner_kwargs) -> None:
        # Bridge realtime preview frames emitted by run_pipeline to the GUI
        # thread (auto queued connection). The window decides whether to
        # render them, so this stays cheap even when the preview is off.
        runner_kwargs.setdefault(
            "frame_callback",
            lambda key, frame_idx, frame_rgb, info: self.frame_ready.emit(
                key, frame_idx, frame_rgb, info
            ),
        )
        self._thread = threading.Thread(
            target=self._run, kwargs=runner_kwargs, daemon=True
        )
        self._thread.start()

    def _run(self, **runner_kwargs) -> None:
        code = 0
        try:
            self._runner = WatchRunner(event_callback=self.event.emit, **runner_kwargs)
            if self._stop_requested.is_set():
                # Stop was requested before the runner even existed.
                self._runner.stop()
            code = self._runner.run()
        except Exception:
            code = 1
            self.event.emit({
                "type": "log",
                "message": f"Watcher thread crashed:\n{traceback.format_exc()}",
            })
        finally:
            self._runner = None
            self.finished.emit(code)

    def stop(self) -> None:
        """Ask the runner to stop: the current video is aborted immediately."""
        self._stop_requested.set()
        if self._runner is not None:
            self._runner.stop()
