"""ui/vlm_conversation_dialog.py — Fenêtre d'inspection de la conversation VLM.

Affiche, pour chaque composante candidate, les deux images réellement envoyées
au modèle, sa décision (défaut gardé / décor retiré), la raison et la réponse
brute.  Indispensable pour comprendre et améliorer le comportement du VLM.
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QWidget,
    QFrame, QPushButton, QTextEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


def _rgb_to_pixmap(rgb: np.ndarray, max_w: int) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if w > max_w:
        pix = pix.scaledToWidth(max_w, Qt.TransformationMode.SmoothTransformation)
    return pix


class VLMConversationDialog(QDialog):
    """Affiche le RefineResult d'un affinage VLM."""

    def __init__(self, parent, result) -> None:
        super().__init__(parent)
        self.setWindowTitle("Conversation VLM — affinage du masque")
        self.setMinimumSize(820, 640)
        self.resize(980, 820)
        self.setStyleSheet("background:#14141f;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── En-tête : modèle + résumé ────────────────────────────────────────
        kept = len(result.exchanges) - result.removed
        header = QLabel(
            f"<b>Modèle :</b> {result.model_id}　|　"
            f"<b>{result.n_components}</b> zones, "
            f"<b>{len(result.exchanges)}</b> lignes analysées　→　"
            f"<span style='color:#e57373'>{result.removed} retirées (décor)</span>, "
            f"<span style='color:#81c784'>{kept} gardées (défaut)</span>　|　"
            f"{result.elapsed:.0f}s"
        )
        header.setWordWrap(True)
        header.setStyleSheet("color:#cdd; font-size:12px; padding:10px 12px; background:#1a1a2e;")
        root.addWidget(header)

        # ── Prompt (repliable) ───────────────────────────────────────────────
        self._prompt_box = QTextEdit()
        self._prompt_box.setReadOnly(True)
        self._prompt_box.setPlainText(result.prompt)
        self._prompt_box.setFixedHeight(120)
        self._prompt_box.setStyleSheet(
            "QTextEdit { background:#0f0f18; color:#9ab; border:none;"
            " font-family:Consolas,monospace; font-size:10px; padding:6px; }"
        )
        self._prompt_box.setVisible(False)

        prompt_btn = QPushButton("▸ Voir le prompt envoyé")
        prompt_btn.setStyleSheet(
            "QPushButton { text-align:left; background:#1a1a2e; color:#7a9ab0;"
            " border:none; padding:5px 12px; font-size:11px; }"
            "QPushButton:hover { color:#cde; }"
        )
        prompt_btn.clicked.connect(self._toggle_prompt)
        self._prompt_btn = prompt_btn
        root.addWidget(prompt_btn)
        root.addWidget(self._prompt_box)

        # ── Liste des échanges ───────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; }")
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(12, 12, 12, 12)
        col.setSpacing(10)

        if not result.exchanges:
            empty = QLabel("Aucune composante allongée à analyser sur cette image.")
            empty.setStyleSheet("color:#889; padding:20px;")
            col.addWidget(empty)
        for ex in result.exchanges:
            col.addWidget(self._make_card(ex))
        col.addStretch()

        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        # ── Bouton fermer ────────────────────────────────────────────────────
        close = QPushButton("Fermer")
        close.clicked.connect(self.accept)
        close.setStyleSheet(
            "QPushButton { background:#1e3a52; color:#b8e0f7; border-radius:4px;"
            " padding:7px 16px; font-size:12px; }"
            "QPushButton:hover { background:#2a5577; }"
        )
        bar = QHBoxLayout()
        bar.setContentsMargins(12, 8, 12, 12)
        bar.addStretch()
        bar.addWidget(close)
        root.addLayout(bar)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _toggle_prompt(self) -> None:
        vis = not self._prompt_box.isVisible()
        self._prompt_box.setVisible(vis)
        self._prompt_btn.setText(("▾ " if vis else "▸ ") + "Voir le prompt envoyé")

    def _make_card(self, ex) -> QFrame:
        removed = ex.removed
        accent = "#e57373" if removed else "#81c784"
        if getattr(ex, "kind", "line") == "cast":
            verdict = "NATUREL — retiré" if removed else "CAST — gardé"
        else:
            verdict = "DÉCOR — retiré" if removed else "DÉFAUT — gardé"

        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background:#1a1a2e; border-radius:6px;"
            f" border-left:3px solid {accent}; }}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(6)

        conf = f" · confiance {ex.confidence:.0%}" if ex.confidence is not None else ""
        kind = {"line": "ligne", "blob": "tache", "cast": "zone rouge"}.get(
            getattr(ex, "kind", "line"), "tache")
        title = QLabel(
            f"<b>#{ex.comp_id}</b> · {kind} · {ex.length}×{getattr(ex,'thickness','?')}px · "
            f"<span style='color:{accent}; font-weight:700'>{verdict}</span>{conf}"
        )
        title.setStyleSheet("color:#cdd; font-size:12px; background:transparent;")
        lay.addWidget(title)

        # Deux images côte à côte
        imgs = QHBoxLayout()
        imgs.setSpacing(10)
        for rgb, cap in ((ex.global_rgb, "Vue globale"), (ex.zoom_rgb, "Zoom local")):
            box = QVBoxLayout(); box.setSpacing(2)
            pic = QLabel(); pic.setPixmap(_rgb_to_pixmap(rgb, 380))
            pic.setStyleSheet("background:#0c0c14; border:1px solid #2a2a4a;")
            lbl = QLabel(cap); lbl.setStyleSheet("color:#778; font-size:9px; background:transparent;")
            box.addWidget(pic); box.addWidget(lbl)
            imgs.addLayout(box)
        imgs.addStretch()
        lay.addLayout(imgs)

        if ex.reason:
            reason = QLabel(f"<b>Raison :</b> {ex.reason}")
            reason.setWordWrap(True)
            reason.setStyleSheet("color:#bcd; font-size:11px; background:transparent;")
            lay.addWidget(reason)

        raw = QLabel(f"Réponse brute : {ex.raw}")
        raw.setWordWrap(True)
        raw.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        raw.setStyleSheet(
            "color:#67788a; font-size:9px; font-family:Consolas,monospace;"
            " background:transparent;"
        )
        lay.addWidget(raw)
        return card
