import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication

from gui.theme import apply_theme
from gui.watch_window import WatchWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Folder watcher")
    app.setOrganizationName("HandHead")
    apply_theme(app)

    window = WatchWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
