import tempfile
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QScrollArea, QComboBox, QFrame, QListWidget, QLineEdit
)

from detection.config_loader import load_config
from detection.behavior_detector import get_registry

import detection.behaviors  # noqa: F401  (trigger behavior registration)

BEHAVIOR_PARAMS = {
    "hand_to_head": [
        ("distance_threshold_ratio", float, 0.1, 2.0, 0.9),
        ("vertical_offset_ratio", float, 0.0, 1.0, 0.2),
        ("keypoint_conf_threshold", float, 0.0, 1.0, 0.5),
        ("head_keypoint_conf_threshold", float, 0.0, 1.0, 0.5),
        ("confirmation_frames", int, 1, 300, 30),
        ("max_gap_frames", int, 0, 300, 10),
        ("min_event_frames", int, 1, 300, 30),
    ],
    "leave_zone": [
        ("zones", "zone_multi", None, None, None),
        ("min_stay_frames", int, 1, 300, 10),
        ("leave_flash_frames", int, 1, 300, 20),
    ],
    "hand_in_zone": [
        ("zones", "zone_multi", None, None, None),
        ("hand", str, None, None, None),
        ("min_duration_frames", int, 1, 300, 5),
        ("confirmation_frames", int, 1, 300, 30),
        ("max_gap_frames", int, 0, 300, 10),
        ("min_event_frames", int, 1, 300, 30),
    ],
    "head_turn": [
        ("turn_threshold_ratio", float, 0.05, 1.0, 0.25),
        ("window_frames", int, 1, 600, 90),
        ("max_turns", int, 1, 50, 3),
        ("confirmation_frames", int, 1, 300, 30),
        ("max_gap_frames", int, 0, 300, 10),
        ("min_event_frames", int, 1, 300, 30),
    ],
    "hand_shake_object": [
        ("zones", "zone_multi", None, None, None),
        ("window_frames", int, 1, 300, 40),
        ("min_reversals", int, 1, 30, 3),
        ("min_displacement_ratio", float, 0.001, 1.0, 0.01),
        ("keypoint_conf_threshold", float, 0.0, 1.0, 0.5),
    ],
    "body_turn": [
        ("min_angle", float, 1.0, 180.0, 45.0),
        ("window_frames", int, 1, 300, 15),
        ("velocity_threshold", float, 0.5, 90.0, 10.0),
        ("confirmation_frames", int, 1, 300, 30),
        ("max_gap_frames", int, 0, 300, 10),
        ("min_event_frames", int, 1, 300, 30),
    ],
    "hand_snatch_object": [
        ("zones", "zone_multi", None, None, None),
        ("min_grasp_frames", int, 1, 300, 3),
        ("snatch_velocity_ratio", float, 0.01, 2.0, 0.15),
        ("approach_window", int, 1, 300, 10),
        ("velocity_baseline_ratio", float, 1.0, 10.0, 2.0),
        ("keypoint_conf_threshold", float, 0.0, 1.0, 0.5),
        ("confirmation_frames", int, 1, 300, 30),
        ("max_gap_frames", int, 0, 300, 10),
        ("min_event_frames", int, 1, 300, 15),
    ],
}


class ZoneMultiSelect(QWidget):
    def __init__(self, zone_names: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._all_names = list(zone_names) if zone_names else []
        self._checked: set[str] = set()

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._button = QPushButton("(none)")
        self._button.clicked.connect(self._show_popup)
        self._layout.addWidget(self._button)

        self._popup: QFrame | None = None
        self._checkboxes: list[QCheckBox] = []

    def _show_popup(self):
        if self._popup is not None:
            self._popup.close()
            self._popup = None
            return

        popup = QFrame(self, Qt.WindowType.Popup)
        popup.setFrameStyle(QFrame.Panel | QFrame.Raised)
        popup_layout = QVBoxLayout(popup)
        popup_layout.setContentsMargins(8, 8, 8, 8)
        popup_layout.setSpacing(2)

        self._checkboxes = []
        for name in self._all_names:
            cb = QCheckBox(name)
            cb.setChecked(name in self._checked)
            popup_layout.addWidget(cb)
            cb.clicked.connect(self._update_button_text)
            self._checkboxes.append(cb)

        close_btn = QPushButton("Done")
        close_btn.clicked.connect(lambda: self._close_popup(popup))
        popup_layout.addWidget(close_btn)

        popup.move(self._button.mapToGlobal(QPoint(0, self._button.height())))
        popup.show()
        self._popup = popup

    def _close_popup(self, popup):
        self._checked = {cb.text() for cb in self._checkboxes if cb.isChecked()}
        self._update_button_text()
        popup.close()
        self._popup = None

    def _update_button_text(self):
        if self._checkboxes:
            selected = [cb.text() for cb in self._checkboxes if cb.isChecked()]
        else:
            selected = list(self._checked)
        if not selected:
            self._button.setText("(none)")
        else:
            self._button.setText("Zones: " + ", ".join(selected))

    def get_selected_zones(self) -> list[str]:
        if self._checkboxes:
            return [cb.text() for cb in self._checkboxes if cb.isChecked()]
        return sorted(self._checked)

    def set_selected_zones(self, names: list[str]):
        self._checked = set(names)
        for cb in self._checkboxes:
            cb.setChecked(cb.text() in self._checked)
        self._update_button_text()

    def update_zone_names(self, names: list[str]):
        self._all_names = list(names)
        if self._popup is not None:
            self._popup.close()
            self._popup = None
        self._checkboxes = []
        self._checked = {z for z in self._checked if z in set(names)}
        self._update_button_text()


class BehaviorEditor(QFrame):
    def __init__(self, name: str, params: dict, zone_names: list[str], parent=None):
        super().__init__(parent)
        self.name = name
        self._zone_names = zone_names
        self._widgets = {}

        self.setObjectName("BehaviorCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>{name}</b>"))
        self.enabled_cb = QCheckBox("Enabled")
        self.enabled_cb.setChecked(params.get("enabled", True))
        header.addStretch()
        header.addWidget(self.enabled_cb)
        layout.addLayout(header)

        form = QFormLayout()
        param_defs = BEHAVIOR_PARAMS.get(name, [])

        known_names = {p[0] for p in param_defs}
        param_values = {k: v for k, v in params.get("params", {}).items() if k in known_names}

        for pname, ptype, pmin, pmax, pdefault in param_defs:

            if ptype == "zone_multi":
                w = ZoneMultiSelect(self._zone_names)
                selected = param_values.get("zones", param_values.get("zone"))
                if isinstance(selected, list):
                    w.set_selected_zones(selected)
                elif isinstance(selected, str) and selected:
                    w.set_selected_zones([selected])
                form.addRow("zones:", w)
                self._widgets[pname] = w
                continue

            value = param_values.get(pname, pdefault)

            if ptype == str and pname == "hand":
                cb = QComboBox()
                cb.addItems(["any", "left", "right", "both"])
                if value in ["any", "left", "right", "both"]:
                    cb.setCurrentText(value)
                form.addRow(f"{pname}:", cb)
                self._widgets[pname] = cb

            elif ptype == float:
                sp = QDoubleSpinBox()
                sp.setRange(pmin, pmax)
                sp.setSingleStep(0.05)
                sp.setValue(float(value if value is not None else pdefault))
                form.addRow(f"{pname}:", sp)
                self._widgets[pname] = sp

            elif ptype == int:
                sp = QSpinBox()
                sp.setRange(pmin, pmax)
                sp.setValue(int(value if value is not None else pdefault))
                form.addRow(f"{pname}:", sp)
                self._widgets[pname] = sp

        layout.addLayout(form)

    def get_data(self) -> dict:
        params = {}
        param_defs = BEHAVIOR_PARAMS.get(self.name, [])
        for pname, ptype, pmin, pmax, pdefault in param_defs:
            w = self._widgets.get(pname)
            if ptype == "zone_multi":
                if isinstance(w, ZoneMultiSelect):
                    selected = w.get_selected_zones()
                    if selected:
                        params[pname] = selected
            elif isinstance(w, QDoubleSpinBox):
                params[pname] = w.value()
            elif isinstance(w, QSpinBox):
                params[pname] = w.value()
            elif isinstance(w, QComboBox):
                val = w.currentText()
                params[pname] = val

        return {
            "name": self.name,
            "enabled": self.enabled_cb.isChecked(),
            "params": params,
        }

    def update_zone_names(self, names: list[str]):
        self._zone_names = names
        for pname, w in self._widgets.items():
            if isinstance(w, ZoneMultiSelect):
                w.update_zone_names(names)


class StepConfig(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._behavior_editors: list[BehaviorEditor] = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self._layout = QVBoxLayout(container)

        title = QLabel("Step 2: Configure Detection")
        title.setObjectName("StepTitle")
        self._layout.addWidget(title)

        load_save_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Config")
        self.load_btn.clicked.connect(self._load_config)
        self.save_btn = QPushButton("Save Config")
        self.save_btn.setObjectName("PrimaryButton")
        self.save_btn.clicked.connect(self._save_config)
        load_save_layout.addWidget(self.load_btn)
        load_save_layout.addWidget(self.save_btn)
        load_save_layout.addStretch()
        self._layout.addLayout(load_save_layout)

        self.default_badge = QLabel("Using default config (hand_to_head only)")
        self.default_badge.setObjectName("DefaultBadge")
        self.default_badge.setVisible(False)
        self._layout.addWidget(self.default_badge)

        model_group = QGroupBox("Model Settings")
        model_group.setCheckable(True)
        model_group.setChecked(False)
        model_form = QFormLayout()
        model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit("yolo11n-pose.pt")
        model_browse_btn = QPushButton("Browse...")
        model_browse_btn.clicked.connect(self._browse_model)
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_browse_btn)
        model_form.addRow("Path:", model_path_layout)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.3)
        model_form.addRow("Confidence:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.5)
        model_form.addRow("IoU:", self.iou_spin)
        model_group.setLayout(model_form)
        self._layout.addWidget(model_group)

        self.behavior_group = QGroupBox("Behaviors")
        self.behavior_layout = QVBoxLayout()
        self.behavior_layout.setSpacing(6)
        self.behavior_group.setLayout(self.behavior_layout)
        self._layout.addWidget(self.behavior_group)

        output_group = QGroupBox("Output Settings")
        output_group.setCheckable(True)
        output_group.setChecked(False)
        output_form = QFormLayout()
        out_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit("./outputs")
        out_browse_btn = QPushButton("Browse...")
        out_browse_btn.clicked.connect(self._browse_output_dir)
        out_dir_layout.addWidget(self.output_dir_edit)
        out_dir_layout.addWidget(out_browse_btn)
        output_form.addRow("Dir:", out_dir_layout)

        self.visualize_cb = QCheckBox("Generate annotated video")
        self.visualize_cb.setChecked(True)
        output_form.addRow("", self.visualize_cb)

        self.context_spin = QSpinBox()
        self.context_spin.setRange(0, 60)
        self.context_spin.setValue(5)
        output_form.addRow("Context (sec):", self.context_spin)

        self.crop_spin = QSpinBox()
        self.crop_spin.setRange(0, 200)
        self.crop_spin.setValue(20)
        output_form.addRow("Crop padding:", self.crop_spin)

        self.debug_kp_cb = QCheckBox("Debug keypoints")
        output_form.addRow("", self.debug_kp_cb)
        output_group.setLayout(output_form)
        self._layout.addWidget(output_group)

        self.zone_list = QListWidget()
        self.zone_group = QGroupBox("Zones (read-only)")
        zone_layout = QVBoxLayout()
        zone_layout.addWidget(self.zone_list)
        self.zone_group.setLayout(zone_layout)
        self._layout.addWidget(self.zone_group)

        self._layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._load_default_behaviors()

    def _zone_names(self) -> list[str]:
        if not self.main_window.config_data:
            return []
        return list(self.main_window.config_data.get("zones", {}).keys())

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Models (*.pt);;ONNX Models (*.onnx);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if not path:
            return
        try:
            config = load_config(path)
        except Exception as e:
            self.main_window.show_error("Config Error", str(e))
            return

        self.main_window.config_path = path
        self.main_window.config_data = config
        self._populate_from_config(config)
        self.default_badge.setVisible(False)
        self.main_window.stepper.set_completed(1, True)

    def _populate_from_config(self, config: dict):
        model_cfg = config.get("model", {})
        self.model_path_edit.setText(model_cfg.get("path", "yolo11n-pose.pt"))
        self.conf_spin.setValue(float(model_cfg.get("conf", 0.3)))
        self.iou_spin.setValue(float(model_cfg.get("iou", 0.5)))

        out_cfg = config.get("output", {})
        self.output_dir_edit.setText(out_cfg.get("dir", "./outputs"))
        self.visualize_cb.setChecked(bool(out_cfg.get("visualize", True)))
        self.context_spin.setValue(int(out_cfg.get("context_seconds", 5)))
        self.crop_spin.setValue(int(out_cfg.get("crop_padding", 20)))
        self.debug_kp_cb.setChecked(bool(out_cfg.get("debug_keypoints", False)))

        self.zone_list.clear()
        zones = config.get("zones", {})
        for zname, zdata in zones.items():
            pt_count = len(zdata.get("points", []))
            self.zone_list.addItem(f"{zname} ({pt_count} points)")

        self._rebuild_behavior_editors(config.get("behaviors", []), list(zones.keys()))

    def _rebuild_behavior_editors(self, behaviors: list[dict], zone_names: list[str]):
        for editor in self._behavior_editors:
            self.behavior_layout.removeWidget(editor)
            editor.deleteLater()
        self._behavior_editors.clear()

        for bcfg in behaviors:
            editor = BehaviorEditor(bcfg["name"], bcfg, zone_names)
            self._behavior_editors.append(editor)
            self.behavior_layout.addWidget(editor)

    def _load_default_behaviors(self):
        try:
            config = load_config(str(Path("config.yaml").resolve()))
            self._populate_from_config(config)
            self.main_window.config_data = config
            self.default_badge.setVisible(False)
        except Exception:
            self.default_badge.setVisible(True)
            from detection.config import (
                CONFIRMATION_FRAMES, DISTANCE_THRESHOLD_RATIO, HEAD_KEYPOINT_CONFIDENCE_THRESHOLD,
                KEYPOINT_CONFIDENCE_THRESHOLD, MAX_GAP_FRAMES, MIN_EVENT_FRAMES, VERTICAL_OFFSET_RATIO,
            )
            cfg = {
                "model": {"path": "yolo11n-pose.pt", "conf": 0.3, "iou": 0.5},
                "output": {"dir": "./outputs", "visualize": True, "context_seconds": 5, "crop_padding": 20, "debug_keypoints": False},
                "zones": {},
                "behaviors": [{
                    "name": "hand_to_head", "enabled": True,
                    "params": {
                        "distance_threshold_ratio": DISTANCE_THRESHOLD_RATIO,
                        "vertical_offset_ratio": VERTICAL_OFFSET_RATIO,
                        "keypoint_conf_threshold": KEYPOINT_CONFIDENCE_THRESHOLD,
                        "head_keypoint_conf_threshold": HEAD_KEYPOINT_CONFIDENCE_THRESHOLD,
                        "confirmation_frames": CONFIRMATION_FRAMES,
                        "max_gap_frames": MAX_GAP_FRAMES,
                        "min_event_frames": MIN_EVENT_FRAMES,
                    },
                }],
            }
            self._populate_from_config(cfg)
            self.main_window.config_data = cfg

    def _save_config(self) -> bool:
        behaviors = [e.get_data() for e in self._behavior_editors]
        enabled = [b for b in behaviors if b.get("enabled")]
        if not enabled:
            self.main_window.show_error("Validation Error", "At least one behavior must be enabled.")
            return False

        registry = get_registry()
        for b in behaviors:
            name = b.get("name", "")
            if not name or name not in registry:
                self.main_window.show_error("Validation Error", f"Unknown behavior: '{name}'")
                return False

        config = {
            "crop": (self.main_window.config_data or {}).get("crop"),
            "model": {
                "path": self.model_path_edit.text(),
                "conf": self.conf_spin.value(),
                "iou": self.iou_spin.value(),
            },
            "output": {
                "dir": self.output_dir_edit.text(),
                "visualize": self.visualize_cb.isChecked(),
                "context_seconds": self.context_spin.value(),
                "crop_padding": self.crop_spin.value(),
                "debug_keypoints": self.debug_kp_cb.isChecked(),
            },
            "zones": (self.main_window.config_data or {}).get("zones", {}),
            "behaviors": behaviors,
        }

        config_path = self.main_window.config_path
        if not config_path or not Path(config_path).exists():
            config_path, _ = QFileDialog.getSaveFileName(
                self, "Save Config", "config.yaml",
                "YAML Files (*.yaml *.yml);;All Files (*)"
            )
            if not config_path:
                return False

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                tmp_path = f.name
            load_config(tmp_path)
            Path(tmp_path).unlink()
        except Exception as e:
            if tmp_path:
                Path(tmp_path).unlink()
            self.main_window.show_error("Validation Error", str(e))
            return False

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.main_window.config_path = config_path
        self.main_window.config_data = config
        self.main_window.stepper.set_completed(1, True)
        self.main_window.show_info("Saved", f"Config saved to {config_path}")
        return True

    def refresh_zones(self):
        self.zone_list.clear()
        zones = (self.main_window.config_data or {}).get("zones", {})
        for zname, zdata in zones.items():
            pt_count = len(zdata.get("points", []))
            self.zone_list.addItem(f"{zname} ({pt_count} points)")

        names = list(zones.keys())
        for editor in self._behavior_editors:
            editor.update_zone_names(names)

    def update_zone_names(self):
        self.refresh_zones()