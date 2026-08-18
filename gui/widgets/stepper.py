from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel

STEP_NAMES = ["Video", "Crop", "Config", "Zones", "Run", "Results"]


class Stepper(QWidget):
    step_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(220)

        self._buttons: list[QPushButton] = []
        self._completed: list[bool] = [False] * len(STEP_NAMES)
        self._active = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(2)

        title = QLabel("Steps")
        title.setObjectName("SidebarTitle")
        layout.addWidget(title)

        for i, name in enumerate(STEP_NAMES):
            btn = QPushButton()
            btn.setObjectName("StepButton")
            btn.setProperty("stepState", "pending")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, idx=i: self.step_selected.emit(idx))
            layout.addWidget(btn)
            self._buttons.append(btn)

        layout.addStretch()
        self._refresh_all()

    def _refresh_button(self, index: int):
        btn = self._buttons[index]
        if index == self._active:
            state = "active"
        elif self._completed[index]:
            state = "completed"
        else:
            state = "pending"
        btn.setProperty("stepState", state)

        if state == "completed":
            marker = "\u2713"
        elif state == "active":
            marker = "\u25CF"
        else:
            marker = "\u25CB"
        btn.setText(f"  {marker}  {STEP_NAMES[index]}")

        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _refresh_all(self):
        for i in range(len(self._buttons)):
            self._refresh_button(i)

    def set_active(self, index: int):
        self._active = index
        self._refresh_all()

    def set_completed(self, index: int, completed: bool = True):
        self._completed[index] = completed
        self._refresh_button(index)

    def set_enabled(self, index: int, enabled: bool = True):
        self._buttons[index].setEnabled(enabled)

    def completed(self, index: int) -> bool:
        return self._completed[index]
