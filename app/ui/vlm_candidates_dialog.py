"""ui/vlm_candidates_dialog.py — Aperçu des candidats avant affinage VLM.

Montre quelles composantes du masque seront envoyées au VLM : lignes (cyan),
grosses taches (orange), et petites zones gardées d'office (rouge pâle).
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton, QWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class VLMCandidatesDialog(QDialog):
    """Aperçu statique des candidats (image RGB déjà annotée)."""

    def __init__(self, parent, overlay_rgb: np.ndarray,
                 n_lines: int, n_blobs: int, n_kept: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Candidats à l'affinage VLM")
        self.setMinimumSize(720, 560)
        self.resize(1000, 760)
        self.setStyleSheet("background:#14141f;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        total = n_lines + n_blobs
        header = QLabel(
            f"<b>{total}</b> zone(s) seront envoyées au VLM&nbsp;:&nbsp;"
            f"<span style='color:#33d6d6'>{n_lines} ligne(s)</span> (cyan), "
            f"<span style='color:#ffa733'>{n_blobs} tache(s)</span> (orange).&nbsp;&nbsp;"
            f"<span style='color:#d66'>{n_kept} petite(s) zone(s) gardée(s)</span> "
            f"d'office (rouge pâle)."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color:#cdd; font-size:12px; padding:10px 12px; background:#1a1a2e;")
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; }")
        rgb = np.ascontiguousarray(overlay_rgb)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pic = QLabel()
        pic.setPixmap(QPixmap.fromImage(qimg))
        pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        holder = QWidget(); hl = QVBoxLayout(holder)
        hl.setContentsMargins(8, 8, 8, 8); hl.addWidget(pic)
        scroll.setWidget(holder)
        root.addWidget(scroll, stretch=1)

        close = QPushButton("Fermer")
        close.clicked.connect(self.accept)
        close.setStyleSheet(
            "QPushButton { background:#1e3a52; color:#b8e0f7; border-radius:4px;"
            " padding:7px 16px; font-size:12px; }"
            "QPushButton:hover { background:#2a5577; }"
        )
        bar = QHBoxLayout(); bar.setContentsMargins(12, 8, 12, 12)
        bar.addStretch(); bar.addWidget(close)
        root.addLayout(bar)
