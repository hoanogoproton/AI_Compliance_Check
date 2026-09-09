"""Realtime Vietnamese GUI for the folder watcher (watch_folders.py).

Shows a live status table (one row per video), per-video progress bars, a
timestamped log panel and Start/Stop controls. The watcher runs in a daemon
thread owned by ``WatchWorker``; its event dicts arrive here via a queued Qt
signal, so every handler below runs on the GUI thread.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
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

from gui.workers.watch_worker import WatchWorker
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

NOTE_IGNORED_INITIAL = "Đã có trong thư mục khi bắt đầu"
NOTE_DELETE_FAILED = "Không xóa được tệp"


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
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Thư mục output:"))
        self.out_edit = QLineEdit(DEFAULT_OUTPUT_DIR)
        row2.addWidget(self.out_edit, 2)
        self.out_btn = QPushButton("Chọn...")
        self.out_btn.clicked.connect(self._browse_output)
        row2.addWidget(self.out_btn)

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

        # -- table + log -------------------------------------------------------
        splitter = QSplitter(Qt.Vertical)

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
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 190])
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

        self._set_running(True)
        self._append_log("Đang khởi động trình theo dõi...")

        self._worker = WatchWorker()
        self._worker.event.connect(self._on_event)
        self._worker.finished.connect(self._on_finished)
        self._worker.start(
            csv_path=csv_path,
            poll_interval=self.poll_spin.value(),
            output_dir=self.out_edit.text().strip() or DEFAULT_OUTPUT_DIR,
            visualize=self.visualize_check.isChecked(),
        )

    def _on_stop(self):
        if self._worker is None:
            return
        self.stop_btn.setEnabled(False)
        self._append_log("Đã yêu cầu dừng — video hiện tại sẽ chạy xong trước khi thoát.")
        self._worker.stop()

    def _on_finished(self, code: int):
        self._worker = None
        self._set_running(False)
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
            self.poll_spin, self.visualize_check,
        ):
            widget.setEnabled(not running)

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
        elif etype == "stopped":
            self._counts = {
                "processed": ev.get("processed", 0),
                "deleted": ev.get("deleted", 0),
                "failed": ev.get("failed", 0),
            }
            self._update_summary()

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
            event.accept()
            return
        reply = QMessageBox.question(
            self,
            "Đang chạy",
            "Trình theo dõi đang chạy. Dừng và thoát?\n"
            "(Video đang xử lý sẽ chạy xong trước khi thoát.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        # Keep the event loop alive (window hidden) so the current video can
        # finish and its journal entry is written before the process exits.
        self._close_after_stop = True
        self._worker.stop()
        self.hide()
        event.ignore()
