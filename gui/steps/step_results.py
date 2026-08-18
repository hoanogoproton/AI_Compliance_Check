import json
import subprocess
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QSplitter, QGroupBox, QPlainTextEdit, QAbstractItemView
)

from gui.widgets.video_player import VideoPlayer


class StepResults(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._metadata: dict | None = None
        self._output_dir: str | None = None

        layout = QVBoxLayout(self)

        title = QLabel("Step 5: Results")
        title.setObjectName("StepTitle")
        layout.addWidget(title)

        load_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Results")
        self.load_btn.clicked.connect(self._load_results_dialog)
        self.open_output_btn = QPushButton("Open Output Folder")
        self.open_output_btn.clicked.connect(self._open_output)
        self.open_output_btn.setEnabled(False)
        self.dir_label = QLabel("(no results loaded)")
        self.dir_label.setObjectName("Subtitle")
        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.open_output_btn)
        load_layout.addWidget(self.dir_label, 1)
        layout.addLayout(load_layout)

        splitter = QSplitter(Qt.Vertical)

        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "ID", "Behavior", "Track", "Start Frame", "End Frame",
            "Hand", "Confidence", "Duration (s)"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        table_layout.addWidget(self.table)

        self.open_clip_btn = QPushButton("Open Clip in Explorer")
        self.open_clip_btn.clicked.connect(self._open_clip)
        self.open_clip_btn.setEnabled(False)
        table_layout.addWidget(self.open_clip_btn)

        splitter.addWidget(table_container)

        player_container = QWidget()
        player_layout = QVBoxLayout(player_container)
        player_layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        controls.addWidget(self.play_btn)
        self.clip_label = QLabel("Select an event to preview")
        controls.addWidget(self.clip_label, 1)
        player_layout.addLayout(controls)

        self.video_player = VideoPlayer()
        player_layout.addWidget(self.video_player, 1)

        meta_group = QGroupBox("Raw Metadata")
        meta_layout = QVBoxLayout()
        self.meta_text = QPlainTextEdit()
        self.meta_text.setObjectName("LogArea")
        self.meta_text.setReadOnly(True)
        self.meta_text.setMaximumHeight(150)
        meta_layout.addWidget(self.meta_text)
        meta_group.setLayout(meta_layout)
        player_layout.addWidget(meta_group)

        splitter.addWidget(player_container)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, 1)

    def _load_results_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.load_results(path)

    def load_results(self, output_dir: str):
        self._output_dir = output_dir
        self.dir_label.setText(f"Results: {output_dir}")
        self.open_output_btn.setEnabled(True)

        json_path = Path(output_dir)
        meta_files = list(json_path.glob("*_metadata.json"))
        if not meta_files:
            meta_files = list(json_path.glob("metadata.json"))
        if not meta_files:
            self.main_window.show_error("Error", f"No metadata JSON found in {output_dir}")
            return

        with open(meta_files[0]) as f:
            self._metadata = json.load(f)

        self.meta_text.setPlainText(json.dumps(self._metadata, indent=2))
        self._populate_table()

    def _populate_table(self):
        if not self._metadata:
            return
        events = self._metadata.get("events", [])
        self.table.setRowCount(len(events))

        for row, ev in enumerate(events):
            self.table.setItem(row, 0, QTableWidgetItem(str(ev.get("event_id", ""))))
            self.table.setItem(row, 1, QTableWidgetItem(ev.get("behavior", "")))
            self.table.setItem(row, 2, QTableWidgetItem(str(ev.get("track_id", ""))))
            self.table.setItem(row, 3, QTableWidgetItem(str(ev.get("start_frame", ""))))
            self.table.setItem(row, 4, QTableWidgetItem(str(ev.get("end_frame", ""))))
            self.table.setItem(row, 5, QTableWidgetItem(ev.get("hand_side", "")))
            self.table.setItem(row, 6, QTableWidgetItem(str(ev.get("max_confidence", ""))))
            self.table.setItem(row, 7, QTableWidgetItem(str(ev.get("duration_sec", ""))))

    def _on_selection_changed(self):
        rows = self.table.selectedItems()
        if not rows or not self._metadata:
            self.open_clip_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.clip_label.setText("Select an event to preview")
            return

        row = rows[0].row()
        events = self._metadata.get("events", [])
        if row >= len(events):
            return

        ev = events[row]
        clip_path = ev.get("clip_path", "")
        if self._output_dir:
            full_path = Path(self._output_dir) / clip_path
            if full_path.exists():
                self.video_player.load(str(full_path))
                self.play_btn.setEnabled(True)
                self.play_btn.setText("Play")
                self.clip_label.setText(f"Event {ev['event_id']}: {clip_path}")
            else:
                self.clip_label.setText(f"Clip not found: {clip_path}")
                self.play_btn.setEnabled(False)

        self.open_clip_btn.setEnabled(True)
        self.open_clip_btn.clip_path = str(Path(self._output_dir) / clip_path) if self._output_dir else ""

    def _toggle_play(self):
        if self.video_player.is_playing():
            self.video_player.stop()
            self.play_btn.setText("Play")
        else:
            self.video_player.play()
            self.play_btn.setText("Pause")

    def _open_clip(self):
        path = getattr(self.open_clip_btn, "clip_path", "")
        if path and Path(path).exists():
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])

    def _open_output(self):
        if self._output_dir and Path(self._output_dir).exists():
            subprocess.Popen(["explorer", os.path.normpath(self._output_dir)])
