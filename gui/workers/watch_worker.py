import threading
import traceback

from PySide6.QtCore import QObject, Signal

from watch_folders import WatchRunner


class WatchWorker(QObject):
    """Runs WatchRunner in a daemon thread, bridging its event dicts to Qt."""

    event = Signal(dict)
    finished = Signal(int)

    def __init__(self):
        super().__init__()
        self._thread: threading.Thread | None = None
        self._runner: WatchRunner | None = None

    def start(self, **runner_kwargs) -> None:
        self._thread = threading.Thread(
            target=self._run, kwargs=runner_kwargs, daemon=True
        )
        self._thread.start()

    def _run(self, **runner_kwargs) -> None:
        code = 0
        try:
            self._runner = WatchRunner(event_callback=self.event.emit, **runner_kwargs)
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
        """Ask the runner to stop: the current video finishes first."""
        if self._runner is not None:
            self._runner.stop()
