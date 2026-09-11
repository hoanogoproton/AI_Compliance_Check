"""Realtime Vietnamese GUI for the folder watcher (watch_folders.py).

Shows a live status table (one row per video), per-video progress bars, an
annotated realtime video preview (bounding boxes, keypoints, face landmarks,
zones) of the video currently being processed, a timestamped log panel and
Start/Stop controls. The watcher runs in a daemon thread owned by
``WatchWorker``; its event dicts and preview frames arrive here via queued Qt
signals, so every handler below runs on the GUI thread.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from get_video import previous_hour_window
from gui.workers.fetch_worker import FetchWorker
from gui.workers.watch_worker import WatchWorker
from gui.widgets.live_preview import LivePreviewWidget
from watch_folders import load_watch_csv

DEFAULT_CSV = "watch_folders.csv"
DEFAULT_OUTPUT_DIR = "./outputs"

# Table columns
COL_VIDEO = 0
COL_FOLDER = 1
COL_CONFIG = 2
COL_STATUS = 3
COL_PROGRESS = 4
COL_EVENTS = 5
COL_NOTE = 6

ST_WAITING = "Đang chờ"
ST_PROCESSING = "Đang xử lý"
ST_DONE = "Hoàn tất"
ST_ERROR = "Lỗi"
ST_SKIPPED = "Bỏ qua"
ST_STOPPED = "Đã dừng"

NOTE_IGNORED_INITIAL = "Đã có trong thư mục khi bắt đầu"
NOTE_DELETE_FAILED = "Không xóa được tệp"
NOTE_STOPPED_MIDWAY = (
    "Dừng giữa chừng theo yêu cầu — video giữ lại, sẽ xử lý lại ở lần khởi động kế tiếp"
)


class WatchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Theo dõi thư mục — Nhận diện hành vi")
        self.resize(1080, 700)

        self._worker: WatchWorker | None = None
        self._rows: dict[str, int] = {}       # video key -> table row
        self._bars: dict[str, QProgressBar] = {}
        self._counts = {"processed": 0, "deleted": 0, "failed": 0}
        self._close_after_stop = False
        self._current_video: str | None = None   # name of the video in progress
        self._last_frame_t: float | None = None  # monotonic ts of last preview frame
        self._ema_frame_dt: float | None = None  # smoothed preview frame interval (s)
        # Tự động tải video "giờ trôi qua" từ Surveillance Station
        self._fetch_worker: FetchWorker | None = None
        self._fetch_reschedule = False
        self._fetch_timer = QTimer(self)
        self._fetch_timer.setSingleShot(True)
        self._fetch_timer.timeout.connect(self._on_fetch_tick)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        title = QLabel("Theo dõi thư mục tự động")
        title.setObjectName("StepTitle")
        root.addWidget(title)

        subtitle = QLabel(
            "Tự động chạy nhận diện cho mọi video mới xuất hiện trong các thư mục "
            "khai báo trong tệp CSV."
        )
        subtitle.setObjectName("Subtitle")
        root.addWidget(subtitle)

        # -- controls ---------------------------------------------------------
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Tệp CSV:"))
        self.csv_edit = QLineEdit(DEFAULT_CSV if Path(DEFAULT_CSV).exists() else "")
        self.csv_edit.setPlaceholderText("Đường dẫn tới watch_folders.csv")
        row1.addWidget(self.csv_edit, 2)
        self.csv_btn = QPushButton("Chọn...")
        self.csv_btn.clicked.connect(self._browse_csv)
        row1.addWidget(self.csv_btn)

        row1.addWidget(QLabel("Chu kỳ quét:"))
        self.poll_spin = QDoubleSpinBox()
        self.poll_spin.setRange(1.0, 3600.0)
        self.poll_spin.setValue(10.0)
        self.poll_spin.setSingleStep(1.0)
        self.poll_spin.setDecimals(1)
        self.poll_spin.setSuffix(" giây")
        self.poll_spin.setMinimumWidth(110)
        row1.addWidget(self.poll_spin)

        self.visualize_check = QCheckBox("Xuất video chú thích")
        self.visualize_check.setChecked(True)
        row1.addWidget(self.visualize_check)

        self.preview_check = QCheckBox("Xem trước realtime")
        self.preview_check.setChecked(True)
        self.preview_check.setToolTip(
            "Hiển thị video đang xử lý kèm bounding box, keypoint, zone…\n"
            "Có thể bật/tắt ngay cả khi đang chạy."
        )
        self.auto_fetch_check = QCheckBox("Tự động tải video giờ trước")
        self.auto_fetch_check.setChecked(True)
        self.auto_fetch_check.setToolTip(
            "Khi trình theo dõi đang chạy: tải ngay video của giờ vừa kết thúc từ\n"
            "Surveillance Station cho mọi camera trong tệp CSV (tên camera = tên\n"
            "thư mục), rồi tự lặp lại mỗi khi sang giờ mới.\n"
            "Ví dụ khởi động lúc 14:20 ngày 9/9/2026 → tải video 13:00-14:00."
        )
        self.auto_fetch_check.toggled.connect(self._on_auto_fetch_toggled)
        row1.addWidget(self.auto_fetch_check)

        self.email_check = QCheckBox("Gửi email kết quả")
        self.email_check.setChecked(True)
        self.email_check.setToolTip(
            "Sau khi mỗi video xử lý xong, gửi email danh sách sự kiện tới\n"
            "dịch vụ thư của nhà máy. Chỉ đổi được trước khi bấm Khởi động."
        )
        row1.addWidget(self.email_check)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Thư mục output:"))
        self.out_edit = QLineEdit(DEFAULT_OUTPUT_DIR)
        row2.addWidget(self.out_edit, 2)
        self.out_btn = QPushButton("Chọn...")
        self.out_btn.clicked.connect(self._browse_output)
        row2.addWidget(self.out_btn)

        self.fetch_now_btn = QPushButton("Tải video giờ trước ngay")
        self.fetch_now_btn.setToolTip(
            "Tải ngay video của giờ vừa kết thúc cho mọi camera trong tệp CSV."
        )
        self.fetch_now_btn.clicked.connect(self._on_fetch_now)
        row2.addWidget(self.fetch_now_btn)

        row2.addStretch(1)

        self.stop_btn = QPushButton("Dừng")
        self.stop_btn.setObjectName("NavButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        row2.addWidget(self.stop_btn)

        self.start_btn = QPushButton("Khởi động")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self._on_start)
        row2.addWidget(self.start_btn)
        root.addLayout(row2)

        # -- preview + table + log ---------------------------------------------
        splitter = QSplitter(Qt.Vertical)

        preview_group = QGroupBox("Xem trước — video đang xử lý")
        preview_layout = QVBoxLayout()
        caption_row = QHBoxLayout()
        self.preview_caption = QLabel("Chưa có video đang xử lý")
        self.preview_caption.setObjectName("Subtitle")
        caption_row.addWidget(self.preview_caption, 1)
        self.preview_fps_label = QLabel("")
        caption_row.addWidget(self.preview_fps_label)
        preview_layout.addLayout(caption_row)
        self.preview = LivePreviewWidget()
        preview_layout.addWidget(self.preview, 1)
        preview_group.setLayout(preview_layout)
        splitter.addWidget(preview_group)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Video", "Thư mục", "Config", "Trạng thái", "Tiến độ", "Sự kiện", "Ghi chú"]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self.table.setColumnWidth(COL_VIDEO, 230)
        self.table.setColumnWidth(COL_FOLDER, 140)
        self.table.setColumnWidth(COL_CONFIG, 140)
        self.table.setColumnWidth(COL_STATUS, 100)
        self.table.setColumnWidth(COL_PROGRESS, 190)
        self.table.setColumnWidth(COL_EVENTS, 70)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        splitter.addWidget(self.table)

        log_group = QGroupBox("Nhật ký")
        log_layout = QVBoxLayout()
        self.log_area = QPlainTextEdit()
        self.log_area.setObjectName("LogArea")
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumBlockCount(5000)
        log_layout.addWidget(self.log_area)
        log_group.setLayout(log_layout)
        splitter.addWidget(log_group)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([300, 360, 170])
        root.addWidget(splitter, 1)

        self._update_summary()

    # -- controls ----------------------------------------------------------

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn tệp CSV", self.csv_edit.text() or ".",
            "CSV (*.csv);;Tất cả tệp (*)",
        )
        if path:
            self.csv_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục output", self.out_edit.text() or "."
        )
        if path:
            self.out_edit.setText(path)

    def _on_start(self):
        csv_path = self.csv_edit.text().strip() or DEFAULT_CSV
        try:
            load_watch_csv(csv_path)
        except (FileNotFoundError, ValueError) as e:
            QMessageBox.critical(self, "CSV không hợp lệ", str(e))
            return

        self._close_after_stop = False
        self._rows.clear()
        self._bars.clear()
        self._counts = {"processed": 0, "deleted": 0, "failed": 0}
        self.table.setRowCount(0)
        self.log_area.clear()
        self._update_summary()

        self._reset_preview()
        self._set_running(True)
        self._append_log("Đang khởi động trình theo dõi...")

        self._worker = WatchWorker()
        self._worker.event.connect(self._on_event)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start(
            csv_path=csv_path,
            poll_interval=self.poll_spin.value(),
            output_dir=self.out_edit.text().strip() or DEFAULT_OUTPUT_DIR,
            visualize=self.visualize_check.isChecked(),
            send_email=self.email_check.isChecked(),
        )

        if self.auto_fetch_check.isChecked():
            # Tải ngay khung "giờ trôi qua" rồi tự hẹn các chu kỳ đầu giờ sau.
            self._start_fetch(reschedule=True)

    def _on_stop(self):
        if self._worker is None:
            return
        self.stop_btn.setEnabled(False)
        self._append_log(
            "Đã yêu cầu dừng — video đang xử lý sẽ bị dừng NGAY và giữ lại trên đĩa "
            "(sẽ được xử lý tiếp ở lần khởi động kế tiếp)."
        )
        self._fetch_timer.stop()
        if self._fetch_worker is not None:
            self._fetch_worker.stop()
            self._append_log(
                "[tải video] Sẽ dừng sau camera hiện tại. Video đã tải nhưng chưa "
                "xử lý sẽ bị coi là 'đã có sẵn' ở lần khởi động kế tiếp."
            )
        self._worker.stop()

    def _on_finished(self, code: int):
        self._worker = None
        self._set_running(False)
        self._fetch_timer.stop()
        if code == 0:
            self._append_log("Trình theo dõi đã dừng.")
        else:
            self._append_log(f"Trình theo dõi đã dừng với mã lỗi {code}.")
        if self._close_after_stop and not self.isVisible():
            # Worker is gone now, so closeEvent accepts and the app quits.
            self.close()

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for widget in (
            self.csv_edit, self.csv_btn, self.out_edit, self.out_btn,
            self.poll_spin, self.visualize_check, self.email_check,
        ):
            widget.setEnabled(not running)

    # -- auto fetch: video "giờ trôi qua" từ Surveillance Station ------------

    def _on_fetch_now(self):
        if self._fetch_worker is not None:
            QMessageBox.information(
                self, "Đang tải",
                "Tiến trình tải video vẫn đang chạy, vui lòng đợi.",
            )
            return
        if self._worker is None:
            self._append_log(
                "[tải video] Lưu ý: trình theo dõi chưa chạy — video tải về chỉ "
                "được xử lý tự động nếu watcher đang chạy khi file xuất hiện."
            )
        self._start_fetch(reschedule=False)

    def _on_auto_fetch_toggled(self, checked: bool):
        if checked and self._worker is not None:
            self._append_log("[tải video] Đã bật tự động tải theo giờ.")
            self._start_fetch(reschedule=True)
        elif not checked:
            self._fetch_timer.stop()
            self._append_log("[tải video] Đã tắt tự động tải theo giờ.")

    def _on_fetch_tick(self):
        # QTimer bắn mỗi đầu giờ; chỉ tải khi watcher vẫn đang chạy.
        if not self.auto_fetch_check.isChecked() or self._worker is None:
            return
        if self._fetch_worker is not None:
            self._append_log(
                "[tải video] Lần tải trước chưa xong — sẽ tự thử lại khi xong."
            )
            return
        self._start_fetch(reschedule=True)

    def _start_fetch(self, reschedule: bool):
        if self._fetch_worker is not None:
            self._append_log("[tải video] Đang có tiến trình tải — bỏ qua yêu cầu mới.")
            return
        begin_dt, end_dt = previous_hour_window()
        csv_path = self.csv_edit.text().strip() or DEFAULT_CSV
        out_dir = self.out_edit.text().strip() or DEFAULT_OUTPUT_DIR
        self._append_log(
            f"[tải video] Tải video {begin_dt:%H:%M}-{end_dt:%H:%M} ngày "
            f"{end_dt:%d/%m/%Y} cho các camera trong {csv_path}..."
        )
        self._fetch_reschedule = reschedule
        self._fetch_worker = FetchWorker()
        self._fetch_worker.event.connect(self._on_fetch_event)
        self._fetch_worker.finished.connect(self._on_fetch_finished)
        self._fetch_worker.start(
            csv_path=csv_path,
            begin_dt=begin_dt,
            end_dt=end_dt,
            state_path=Path(out_dir) / "fetch_state.json",
        )

    def _on_fetch_event(self, ev: dict):
        etype = ev.get("type")
        if etype == "log":
            self._append_log(ev.get("message", ""))
        elif etype == "fetch_done":
            self._append_log(
                f"[tải video] Kết quả: {ev.get('ok', 0)} thành công, "
                f"{ev.get('skipped', 0)} đã tải trước đó, {ev.get('failed', 0)} lỗi."
            )

    def _on_fetch_finished(self, code: int):
        self._fetch_worker = None
        if code != 0:
            self._append_log("[tải video] Tiến trình tải kết thúc với lỗi.")
        if (
            self._fetch_reschedule
            and self.auto_fetch_check.isChecked()
            and self._worker is not None
        ):
            self._schedule_next_fetch()

    def _schedule_next_fetch(self):
        now = datetime.now()
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        next_hour = hour_start + timedelta(hours=1)
        # +10 giây để Surveillance Station kịp ghi xong video cuối giờ.
        delay_ms = int((next_hour - now).total_seconds() * 1000) + 10_000
        self._fetch_timer.start(max(1000, min(delay_ms, 2**31 - 1)))
        self._append_log(
            f"[tải video] Hẹn chu kỳ kế tiếp lúc {next_hour:%H:%M} "
            f"(sẽ tải {hour_start:%H:%M}-{next_hour:%H:%M})."
        )

    # -- events (all on the GUI thread via queued connection) ---------------

    def _on_event(self, ev: dict):
        etype = ev.get("type")
        if etype == "log":
            self._append_log(ev.get("message", ""))
        elif etype == "startup":
            ignored = ev.get("ignored_initial", 0)
            if ignored:
                self._append_log(
                    f"Nhật ký cũ: {ignored} video đã có từ lần chạy trước sẽ bị bỏ qua."
                )
        elif etype == "folder_seen":
            ignored = ev.get("ignored", 0)
            if ignored:
                self._append_log(
                    f"Thư mục {ev.get('folder_raw', ev.get('folder', ''))}: "
                    f"bỏ qua {ignored} video đã có sẵn."
                )
                for name in ev.get("videos", []):
                    key = f"ignored::{ev.get('folder', '')}::{name}"
                    row = self._row_for(key, name, ev.get("folder_raw", ""))
                    self._set_cell(row, COL_STATUS, ST_SKIPPED)
                    self._set_cell(row, COL_NOTE, NOTE_IGNORED_INITIAL)
            else:
                self._append_log(
                    f"Đang theo dõi: {ev.get('folder_raw', ev.get('folder', ''))}"
                )
        elif etype == "video_new":
            row = self._row_for(
                ev["key"], ev.get("video_name", ""), ev.get("folder", ""), ev.get("config", "")
            )
            self._set_cell(row, COL_STATUS, ST_WAITING)
        elif etype == "video_start":
            row = self._row_for(
                ev["key"], ev.get("video_name", ""), ev.get("folder", ""), ev.get("config", "")
            )
            self._set_cell(row, COL_STATUS, ST_PROCESSING)
            self._current_video = ev.get("video_name", "")
            self.preview_caption.setText(f"{self._current_video} — đang xử lý…")
        elif etype == "video_progress":
            key = ev["key"]
            row = self._row_for(key, ev.get("video_name", ""))
            self._set_cell(row, COL_STATUS, ST_PROCESSING)
            total = ev.get("total", 0)
            frame = ev.get("frame", 0)
            if total > 0:
                self._bars[key].setValue(min(100, int(frame * 100 / total)))
        elif etype == "video_done":
            key = ev["key"]
            row = self._row_for(key, ev.get("video_name", ""))
            self._current_video = None
            self.preview_caption.setText(f"{ev.get('video_name', '')} — hoàn tất")
            deleted = bool(ev.get("deleted"))
            if deleted:
                self._set_cell(row, COL_STATUS, ST_DONE)
                self._set_cell(row, COL_NOTE, "Đã xóa video gốc")
            else:
                self._set_cell(row, COL_STATUS, ST_SKIPPED)
                self._set_cell(
                    row, COL_NOTE, f"{NOTE_DELETE_FAILED}: {ev.get('delete_error', '')}"
                )
            events = ev.get("events")
            self._set_cell(row, COL_EVENTS, "?" if events is None else str(events))
            self._bars[key].setValue(100)
            self._counts["processed"] += 1
            if deleted:
                self._counts["deleted"] += 1
            self._update_summary()
        elif etype == "video_error":
            key = ev["key"]
            row = self._row_for(key, ev.get("video_name", ""))
            self._current_video = None
            self.preview_caption.setText(f"{ev.get('video_name', '')} — lỗi")
            self._set_cell(row, COL_STATUS, ST_ERROR)
            gave_up = bool(ev.get("gave_up"))
            attempt = (
                f"Thất bại {ev.get('attempts')}/{ev.get('max_retries')} lần — đã từ bỏ"
                if gave_up
                else f"Lần {ev.get('attempts')}/{ev.get('max_retries')} — sẽ thử lại"
            )
            self._set_cell(row, COL_NOTE, f"{attempt}: {ev.get('error', '')}")
            if gave_up:
                self._counts["failed"] += 1
                self._update_summary()
        elif etype == "video_stopped":
            key = ev["key"]
            row = self._row_for(key, ev.get("video_name", ""))
            self._current_video = None
            caption = f"{ev.get('video_name', '')} — đã dừng giữa chừng"
            self.preview.clear(caption)
            self.preview_caption.setText(caption)
            self.preview_fps_label.setText("")
            self._set_cell(row, COL_STATUS, ST_STOPPED)
            self._set_cell(row, COL_NOTE, NOTE_STOPPED_MIDWAY)
            self._bars[key].setValue(0)
        elif etype == "stopped":
            self._counts = {
                "processed": ev.get("processed", 0),
                "deleted": ev.get("deleted", 0),
                "failed": ev.get("failed", 0),
            }
            self._update_summary()

    # -- realtime preview ------------------------------------------------------

    def _on_frame_ready(self, key: str, frame_idx: int, frame_rgb, info: dict):
        """Render one annotated frame pushed by the pipeline (GUI thread)."""
        if not self.preview_check.isChecked():
            self._last_frame_t = None  # restart FPS smoothing when re-enabled
            return
        now = time.monotonic()
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if dt > 0:
                self._ema_frame_dt = (
                    dt if self._ema_frame_dt is None
                    else 0.7 * self._ema_frame_dt + 0.3 * dt
                )
        self._last_frame_t = now
        self.preview.set_frame(frame_rgb)

        caption = f"{self._current_video or 'Đang xử lý'} — Frame {frame_idx + 1}"
        people = info.get("people")
        events = info.get("events")
        if people is not None:
            caption += f" • Người: {people}"
        if events is not None:
            caption += f" • Sự kiện: {events}"
        self.preview_caption.setText(caption)
        if self._ema_frame_dt:
            self.preview_fps_label.setText(f"{1.0 / self._ema_frame_dt:.1f} FPS")
        else:
            self.preview_fps_label.setText("")

    def _reset_preview(self):
        self._current_video = None
        self._last_frame_t = None
        self._ema_frame_dt = None
        self.preview.clear("Chưa có video đang xử lý")
        self.preview_caption.setText("Chưa có video đang xử lý")
        self.preview_fps_label.setText("")

    # -- table / log helpers -------------------------------------------------

    def _row_for(self, key: str, video_name: str, folder: str = "", config: str = "") -> int:
        row = self._rows.get(key)
        if row is not None:
            return row
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._set_cell(row, COL_VIDEO, video_name)
        self._set_cell(row, COL_FOLDER, folder)
        self._set_cell(row, COL_CONFIG, config)
        self._set_cell(row, COL_STATUS, ST_WAITING)
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFormat("%p%")
        self.table.setCellWidget(row, COL_PROGRESS, bar)
        self._bars[key] = bar
        self._set_cell(row, COL_EVENTS, "")
        self._set_cell(row, COL_NOTE, "")
        self._rows[key] = row
        return row

    def _set_cell(self, row: int, column: int, text: str):
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, column, item)
        item.setText(text)

    def _append_log(self, message: str):
        self.log_area.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")

    def _update_summary(self):
        c = self._counts
        self.statusBar().showMessage(
            f"Đã xử lý {c['processed']} (đã xóa {c['deleted']}) | Lỗi {c['failed']}"
        )

    # -- shutdown --------------------------------------------------------------

    def closeEvent(self, event):
        if self._worker is None:
            self._fetch_timer.stop()
            event.accept()
            return
        reply = QMessageBox.question(
            self,
            "Đang chạy",
            "Trình theo dõi đang chạy. Dừng và thoát?\n"
            "(Video đang xử lý sẽ bị dừng NGAY và giữ lại trên đĩa — lần khởi "
            "động kế tiếp nó sẽ được xử lý tiếp; tiến trình tải video đang "
            "chạy sẽ bị ngắt.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        # Keep the event loop alive (window hidden) so the current video can
        # finish and its journal entry is written before the process exits.
        self._close_after_stop = True
        # Ngừng hẹn/tải video: không khởi động chu kỳ mới khi đang thoát.
        self.auto_fetch_check.setChecked(False)
        self._fetch_timer.stop()
        if self._fetch_worker is not None:
            self._fetch_worker.stop()
        self._worker.stop()
        self.hide()
        event.ignore()
