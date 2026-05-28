"""ui/json_diff_panel.py — Panneau de diff JSON pour les onglets de comparaison batch.

Affiche un diff unifié colorisé entre deux représentations JSON :
  - vert  (fond)  : lignes présentes dans « Actuel » mais pas sur disque (+)
  - rouge (fond)  : lignes présentes sur disque mais pas dans « Actuel » (-)
  - bleu          : marqueurs @@ de contexte
  - gris          : lignes communes
"""

from __future__ import annotations

import difflib
import json
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit,
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


class JsonDiffPanel(QWidget):
    """Diff unifié entre deux dicts JSON.

    Usage ::

        panel = JsonDiffPanel()
        panel.set_refresh_fn(lambda: panel.update_diff(current, disk, "Actuel", "Disque"))
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#0e0e1a;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── En-tête ───────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setFixedHeight(36)
        hdr.setStyleSheet("background:#1e1e38; border-bottom:1px solid #2a2a4a;")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(12, 0, 8, 0)
        hdr_lay.setSpacing(8)

        self._info_lbl = QLabel("Sélectionnez une image pour afficher le diff")
        self._info_lbl.setStyleSheet("color:#888; font-size:11px;")
        hdr_lay.addWidget(self._info_lbl, stretch=1)

        self._stats_lbl = QLabel("")
        self._stats_lbl.setStyleSheet(
            "color:#9de; font-size:10px; font-weight:600; min-width:80px;"
        )
        self._stats_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hdr_lay.addWidget(self._stats_lbl)

        self._refresh_btn = QPushButton("⟳  Actualiser")
        self._refresh_btn.setFixedHeight(26)
        self._refresh_btn.setStyleSheet(
            "QPushButton { background:#2a3050; color:#9de; border-radius:4px;"
            "  padding:2px 10px; font-size:11px; }"
            "QPushButton:hover { background:#3a4070; }"
        )
        hdr_lay.addWidget(self._refresh_btn)
        root.addWidget(hdr)

        # ── Légende ───────────────────────────────────────────────────────────
        legend = QWidget()
        legend.setFixedHeight(22)
        legend.setStyleSheet("background:#14142a;")
        leg_lay = QHBoxLayout(legend)
        leg_lay.setContentsMargins(12, 2, 12, 2)
        leg_lay.setSpacing(20)
        for text, color in [
            ("██ Ajouté dans actuel", "#2ecc71"),
            ("██ Supprimé de actuel", "#e74c3c"),
            ("██ Contexte", "#444"),
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{color}; font-size:10px;")
            leg_lay.addWidget(lbl)
        leg_lay.addStretch()
        root.addWidget(legend)

        # ── Zone diff ─────────────────────────────────────────────────────────
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        font = QFont("Monospace")
        font.setPointSize(9)
        self._text.setFont(font)
        self._text.setStyleSheet(
            "QTextEdit { background:#0e0e1a; border:none; color:#ccc; padding:0; }"
        )
        root.addWidget(self._text, stretch=1)

        # ── Connexions ────────────────────────────────────────────────────────
        self._refresh_fn = None
        self._refresh_btn.clicked.connect(self._do_refresh)

    def set_refresh_fn(self, fn) -> None:
        self._refresh_fn = fn

    def _do_refresh(self) -> None:
        if self._refresh_fn:
            self._refresh_fn()

    # ── API publique ─────────────────────────────────────────────────────────

    def update_diff(
        self,
        current:     Optional[dict],
        disk:        Optional[dict],
        left_label:  str = "Actuel",
        right_label: str = "Disque",
    ) -> None:
        """Met à jour le diff affiché.

        Parameters
        ----------
        current     : dict courant en mémoire (non encore sauvegardé)
        disk        : dict chargé depuis le disque (None si absent)
        left_label  : label pour le côté « current »
        right_label : label pour le côté « disk »
        """
        if current is None:
            self._info_lbl.setText("Aucune image sélectionnée")
            self._stats_lbl.setText("")
            self._text.setPlainText("")
            return

        self._info_lbl.setText(f"● {left_label}   vs   ● {right_label}")

        current_str = json.dumps(current, ensure_ascii=False, indent=2)

        if disk is None:
            self._stats_lbl.setText("⚠ absent")
            self._text.setHtml(
                "<div style='font-family:monospace; font-size:9pt; padding:12px;'>"
                "<p style='color:#f39c12;'>⚠ Fichier non trouvé sur le disque.</p>"
                "<p style='color:#666; margin-top:8px;'>État actuel (non sauvegardé) :</p>"
                "<pre style='color:#9de; margin:0;'>"
                + self._escape(current_str)
                + "</pre></div>"
            )
            return

        disk_str = json.dumps(disk, ensure_ascii=False, indent=2)

        if current_str == disk_str:
            self._stats_lbl.setText("✓ identique")
            self._text.setHtml(
                "<div style='padding:16px; font-size:11pt;'>"
                "<p style='color:#2ecc71;'>✓ Identique au fichier sur le disque.</p>"
                "</div>"
            )
            return

        current_lines = current_str.splitlines(keepends=True)
        disk_lines    = disk_str.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            disk_lines, current_lines,
            fromfile=f"disque — {right_label}",
            tofile=f"actuel — {left_label}",
            n=3,
        ))

        added   = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        total   = len(current_lines)
        self._stats_lbl.setText(f"+{added} / -{removed} / {total} lignes")

        self._text.setHtml(self._diff_to_html(diff))
        # Scroll au premier changement
        cursor = self._text.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        self._text.setTextCursor(cursor)

    # ── Rendu ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _escape(text: str) -> str:
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    def _diff_to_html(self, diff_lines: list) -> str:
        parts = ["<div style='font-family:monospace; font-size:9pt; padding:2px 0;'>"]
        for line in diff_lines:
            line_e = self._escape(line.rstrip("\n"))
            if line.startswith("+++") or line.startswith("---"):
                color, bg = "#aaa", "#1a1a2e"
            elif line.startswith("+"):
                color, bg = "#b8e8b8", "#162616"
            elif line.startswith("-"):
                color, bg = "#e8b8b8", "#261616"
            elif line.startswith("@@"):
                color, bg = "#7ab4d4", "#162030"
            else:
                color, bg = "#666", "#0e0e1a"
            parts.append(
                f"<div style='background:{bg}; color:{color};"
                f" white-space:pre; margin:0; padding:0 6px; line-height:1.4;'>"
                f"{line_e}</div>"
            )
        parts.append("</div>")
        return "".join(parts)
