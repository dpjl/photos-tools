"""main.py — Point d'entrée de l'application."""
import sys
import os

# S'assurer que le dossier app/ est dans sys.path pour tous les imports relatifs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.main_window import MainApp


def main():
    # Haute résolution (HiDPI)
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("Restauration Photo")
    app.setOrganizationName("Plustek Photo")

    window = MainApp()

    # Ouvrir une image passée en argument (optionnel)
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            import cv2
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                window._original      = img
                window._original_path = path
                from steps import ALL_STEPS
                step_names = {s.id: s.short_name for s in ALL_STEPS}
                window._thumb_strip.rebuild([s.id for s in ALL_STEPS], step_names)
                window._thumb_strip.update_image("original", img)
                window._view_a_widget.set_image(img)
                window._status_bar.showMessage(
                    f"{os.path.basename(path)}  ({img.shape[1]}×{img.shape[0]} px)"
                )

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
