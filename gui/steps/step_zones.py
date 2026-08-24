from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QFormLayout
)

from detection.zones.zone_checker import save_zones
from detection.zones.zone_definition import Zone
from gui.widgets.zone_canvas import ZoneCanvas
from gui.widgets.video_player import VideoPlayer


class StepZones(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._zones: dict[str, Zone] = {}
        self._crop_region: tuple[int, int, int, int] | None = None
        self._editing_zone: str | None = None

        layout = QVBoxLayout(self)

        title = QLabel("Step 3: Define Zones (optional)")
        title.setObjectName("StepTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Pause the video and left-click to add zone points. "
            "Right-click or Backspace removes the last point."
        )
        subtitle.setObjectName("Subtitle")
        layout.addWidget(subtitle)

        body = QHBoxLayout()
        body.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(8)
        self.canvas = ZoneCanvas()
        left.addWidget(self.canvas, 1)
        self.player = VideoPlayer(show_video=False)
        self.player.frame_changed.connect(self._on_frame)
        left.addWidget(self.player)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(8)

        zone_form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("zone_name")
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Display label")
        zone_form.addRow("Name:", self.name_edit)
        zone_form.addRow("Label:", self.label_edit)
        right.addLayout(zone_form)

        btn_row = QHBoxLayout()
        self.save_zone_btn = QPushButton("Save Zone")
        self.save_zone_btn.setObjectName("PrimaryButton")
        self.save_zone_btn.clicked.connect(self._save_zone)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_edit)
        btn_row.addWidget(self.save_zone_btn)
        btn_row.addWidget(self.clear_btn)
        right.addLayout(btn_row)

        edit_hint = QLabel()
        edit_hint.setObjectName("Subtitle")
        edit_hint.setVisible(False)
        right.addWidget(edit_hint)
        self._edit_hint = edit_hint

        right.addWidget(QLabel("Existing Zones:"))
        self.zone_list = QListWidget()
        self.zone_list.itemClicked.connect(self._zone_selected)
        right.addWidget(self.zone_list, 1)

        self.delete_zone_btn = QPushButton("Delete Selected Zone")
        self.delete_zone_btn.clicked.connect(self._delete_zone)
        right.addWidget(self.delete_zone_btn)

        body.addLayout(right, 1)
        layout.addLayout(body, 1)

        self.canvas.points_changed.connect(self._on_points_changed)

    def load_video(self, video_path: str):
        self._load_zones_from_config()
        raw_crop = (self.main_window.config_data or {}).get("crop")
        self._crop_region = tuple(raw_crop) if raw_crop else None
        self.player.set_crop_region(self._crop_region)
        self.player.load(video_path)

    def _on_frame(self, frame_idx: int, rgb):
        self.canvas.set_frame(rgb)

    def _on_points_changed(self, points):
        self.save_zone_btn.setEnabled(len(points) >= 3)

    def _load_zones_from_config(self):
        self._zones = {}
        data = self.main_window.config_data
        if data:
            for zname, zdata in data.get("zones", {}).items():
                if "points" in zdata:
                    self._zones[zname] = Zone.from_dict(zname, zdata)
        self._refresh_zone_list()

    def _persist_zones(self):
        if self.main_window.config_data is None:
            self.main_window.config_data = {}
        self.main_window.config_data["zones"] = {n: z.to_dict() for n, z in self._zones.items()}

        config_path = self.main_window.config_path
        if config_path and Path(config_path).exists():
            save_zones(self._zones, config_path)

    def _save_zone(self):
        name = self.name_edit.text().strip()
        label = self.label_edit.text().strip() or name
        points = list(self.canvas.points())

        if not name:
            self.main_window.show_error("Error", "Zone name is required")
            return
        if len(points) < 3:
            self.main_window.show_error("Error", "A zone needs at least 3 points")
            return

        is_update = self._editing_zone is not None
        old_name = self._editing_zone

        if is_update and old_name != name:
            self._zones.pop(old_name, None)

        self._zones[name] = Zone(name=name, label=label, points=points)
        self._persist_zones()

        self._clear_edit()
        self._refresh_zone_list()
        self.main_window.on_zones_changed()

        verb = "Updated" if is_update else "Saved"
        self.main_window.show_info(verb, f"Zone '{name}' {verb.lower()}")

    def _delete_zone(self):
        item = self.zone_list.currentItem()
        if not item:
            return
        name = item.text().split(" (")[0]
        reply = QMessageBox.question(
            self, "Delete Zone", f"Delete zone '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self._zones.pop(name, None)
        if self._editing_zone == name:
            self._clear_edit()
        self._persist_zones()
        self._refresh_zone_list()
        self.main_window.on_zones_changed()

    def _zone_selected(self, item: QListWidgetItem):
        name = item.text().split(" (")[0]
        zone = self._zones.get(name)
        if not zone:
            return
        self._editing_zone = name
        self.name_edit.setText(name)
        self.label_edit.setText(zone.label)
        self.canvas.set_points(zone.points)
        self.save_zone_btn.setText("Update Zone")
        self.save_zone_btn.setEnabled(len(zone.points) >= 3)
        self._edit_hint.setText(f"Editing zone '{name}' — modify points or fields, then click Update")
        self._edit_hint.setVisible(True)

    def _clear_edit(self):
        self._editing_zone = None
        self.canvas.clear_points()
        self.name_edit.clear()
        self.label_edit.clear()
        self.save_zone_btn.setText("Save Zone")
        self.save_zone_btn.setEnabled(False)
        self._edit_hint.setVisible(False)

    def _refresh_zone_list(self):
        self.zone_list.clear()
        for name, zone in self._zones.items():
            pt_count = len(zone.points)
            self.zone_list.addItem(f"{name} ({pt_count} points)")
