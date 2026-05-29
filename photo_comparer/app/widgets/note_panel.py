"""NotePanel — panneau de notes par photo, affiché à la demande en bas de la fenêtre."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class NotePanel(QWidget):
    """Bande compacte avec un champ de texte libre lié au groupe photo courant.

    Signaux
    -------
    note_changed(group_key, text) — émis ~600 ms après la dernière frappe.
    """

    note_changed = Signal(str, str)  # group_key, note_text

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setFixedHeight(115)
        self.setStyleSheet(
            "NotePanel { background:#18182e; border-top:1px solid #3a3a6a; }"
        )

        self._group_key: str = ""
        self._dirty: bool = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 6)
        outer.setSpacing(3)

        # Header row
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self._title_lbl = QLabel("Notes")
        self._title_lbl.setStyleSheet(
            "color:#8ac4ff; font-size:11px; font-weight:bold; background:transparent;"
        )
        self._save_lbl = QLabel("")
        self._save_lbl.setStyleSheet(
            "color:#4CAF50; font-size:10px; background:transparent;"
        )
        self._save_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        header.addWidget(self._title_lbl)
        header.addStretch()
        header.addWidget(self._save_lbl)
        outer.addLayout(header)

        # Text area
        self._edit = QPlainTextEdit()
        self._edit.setPlaceholderText("Tapez une note pour ce groupe de photos…")
        self._edit.setStyleSheet(
            "QPlainTextEdit { background:#111128; color:#ccc;"
            " border:1px solid #3a3a6a; border-radius:3px;"
            " padding:3px; font-size:11px; }"
        )
        outer.addWidget(self._edit)

        # Auto-save debounce
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(600)
        self._timer.timeout.connect(self._do_save)
        self._edit.textChanged.connect(self._on_text_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_group(self, key: str, text: str):
        """Bascule vers un nouveau groupe ; sauvegarde les changements en attente."""
        self._flush()
        self._group_key = key
        self._title_lbl.setText(f"Notes — {key}" if key else "Notes")
        self._edit.blockSignals(True)
        self._edit.setPlainText(text)
        self._edit.blockSignals(False)
        self._dirty = False
        self._save_lbl.setText("")

    def flush(self):
        """Force-sauvegarde les changements en attente (appeler avant fermeture)."""
        self._flush()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_text_changed(self):
        self._dirty = True
        self._save_lbl.setText("")
        self._timer.start()

    def _do_save(self):
        if not self._dirty or not self._group_key:
            return
        self.note_changed.emit(self._group_key, self._edit.toPlainText())
        self._dirty = False
        self._save_lbl.setText("✓ Enregistré")

    def _flush(self):
        if self._dirty and self._group_key:
            self._timer.stop()
            self._do_save()
