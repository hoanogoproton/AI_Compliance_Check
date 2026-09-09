"""Chạy fetch_all_from_csv (get_video.py) trong daemon thread, cầu nối Qt signals."""

from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from get_video import fetch_all_from_csv


class FetchWorker(QObject):
    """Tải video "giờ trôi qua" cho mọi camera trong CSV trên nền background.

    Mỗi dòng log/sự kiện được phát qua signal ``event`` (dict) và tín hiệu
    ``finished(int)`` bắn khi tiến trình kết thúc — cả hai đều được Qt xếp
    hàng (queued) về GUI thread.
    """

    event = Signal(dict)
    finished = Signal(int)

    def __init__(self):
        super().__init__()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(
        self,
        *,
        csv_path: str,
        begin_dt: datetime,
        end_dt: datetime,
        headless: bool = False,
        state_path: str | Path | None = None,
    ) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            kwargs={
                "csv_path": csv_path,
                "begin_dt": begin_dt,
                "end_dt": end_dt,
                "headless": headless,
                "state_path": state_path,
            },
            daemon=True,
        )
        self._thread.start()

    def _run(
        self,
        *,
        csv_path: str,
        begin_dt: datetime,
        end_dt: datetime,
        headless: bool,
        state_path: str | Path | None,
    ) -> None:
        code = 0
        try:
            summary = fetch_all_from_csv(
                csv_path,
                begin_dt,
                end_dt,
                headless=headless,
                should_stop=self._stop_event.is_set,
                log=lambda message: self.event.emit(
                    {"type": "log", "message": message}
                ),
                state_path=state_path,
            )
            self.event.emit({"type": "fetch_done", **summary})
        except Exception:
            code = 1
            self.event.emit({
                "type": "log",
                "message": f"[tải video] Lỗi tiến trình:\n{traceback.format_exc()}",
            })
        finally:
            self._thread = None
            self.finished.emit(code)

    def stop(self) -> None:
        """Yêu cầu dừng (có hiệu lực giữa các camera; camera hiện tại chạy nốt)."""
        self._stop_event.set()