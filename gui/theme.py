"""Global dark theme for the Hand-to-Head Detection GUI."""

ACCENT = "#0078d4"
ACCENT_HOVER = "#1a8de0"
ACCENT_PRESSED = "#005a9e"

BG = "#1e1e1e"
SURFACE = "#2d2d30"
SURFACE_ALT = "#252526"
SURFACE_RAISED = "#333337"
BORDER = "#3f3f46"
TEXT = "#d4d4d4"
TEXT_DIM = "#8a8a8a"
SUCCESS = "#16c784"
WARNING = "#ffa500"
DANGER = "#e81123"

APP_STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: {BG};
}}

#Sidebar {{
    background-color: {SURFACE_ALT};
    border-right: 1px solid {BORDER};
}}

#SidebarTitle {{
    color: {TEXT_DIM};
    padding: 0 16px 12px 16px;
    font-weight: bold;
    letter-spacing: 1px;
}}

#StepButton {{
    text-align: left;
    padding: 12px 16px;
    border: none;
    border-left: 3px solid transparent;
    background-color: transparent;
    color: {TEXT_DIM};
    font-size: 14px;
}}

#StepButton:hover {{
    background-color: {SURFACE};
    color: {TEXT};
}}

#StepButton[stepState="active"] {{
    background-color: {SURFACE};
    border-left: 3px solid {ACCENT};
    color: {TEXT};
    font-weight: bold;
}}

#StepButton[stepState="completed"] {{
    color: {TEXT};
}}

#StepButton:disabled {{
    color: #555555;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
    background-color: {SURFACE};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: {TEXT};
}}

QGroupBox::indicator {{
    width: 14px;
    height: 14px;
}}

QPushButton {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    color: {TEXT};
}}

QPushButton:hover {{
    background-color: #3d3d41;
}}

QPushButton:pressed {{
    background-color: #2a2a2d;
}}

QPushButton:disabled {{
    color: #6a6a6a;
    background-color: #2a2a2d;
    border-color: #333333;
}}

#PrimaryButton {{
    background-color: {ACCENT};
    border: none;
    color: #ffffff;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 4px;
    font-size: 15px;
}}

#PrimaryButton:hover {{
    background-color: {ACCENT_HOVER};
}}

#PrimaryButton:pressed {{
    background-color: {ACCENT_PRESSED};
}}

#PrimaryButton:disabled {{
    background-color: #3a3a3d;
    color: #777777;
}}

#NavButton {{
    min-width: 90px;
    padding: 8px 18px;
}}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {TEXT};
    selection-background-color: {ACCENT};
}}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}

QLineEdit:disabled, QComboBox:disabled {{
    color: {TEXT_DIM};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}

QListWidget, QTableWidget, QTreeWidget {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    alternate-background-color: {SURFACE};
}}

QListWidget::item:selected, QTableWidget::item:selected {{
    background-color: {ACCENT};
    color: #ffffff;
}}

QTableWidget {{
    gridline-color: {BORDER};
}}

QHeaderView::section {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 6px;
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {SURFACE_RAISED};
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {TEXT};
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}

QProgressBar {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT};
    height: 18px;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {SURFACE_RAISED};
    border-radius: 6px;
    min-height: 30px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {SURFACE_RAISED};
    border-radius: 6px;
    min-width: 30px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QStatusBar {{
    background-color: {SURFACE_ALT};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}

QCheckBox::indicator:unchecked {{
    border: 1px solid {BORDER};
    background: {SURFACE_ALT};
    border-radius: 3px;
}}

QCheckBox::indicator:checked {{
    border: 1px solid {ACCENT};
    background: {ACCENT};
    border-radius: 3px;
}}

QToolTip {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px;
}}

#StepTitle {{
    font-size: 18px;
    font-weight: bold;
}}

#Subtitle {{
    color: {TEXT_DIM};
}}

#PreflightOk {{
    color: {SUCCESS};
    font-weight: bold;
}}

#PreflightBad {{
    color: {DANGER};
    font-weight: bold;
}}

#DefaultBadge {{
    color: {WARNING};
    font-weight: bold;
}}

#BehaviorCard {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {SURFACE};
}}

#Thumbnail {{
    background-color: #161616;
    color: #888888;
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

#LogArea {{
    background-color: #161616;
    color: #d4d4d4;
    font-family: Consolas, monospace;
}}
"""


def apply_theme(app):
    app.setStyleSheet(APP_STYLESHEET)
