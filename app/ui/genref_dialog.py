"""ui/genref_dialog.py — Générateur de référence IA pour l'étape « genref ».

Flux de travail :
  1. Choisir le style de prompt (court / détaillé / manuel) et la graine.
  2. « Rédiger le prompt » : le VLM liste les éléments et leurs couleurs ;
     le texte est affiché et ÉDITABLE (style manuel = édition libre).
  3. « Générer » : FLUX Kontext produit la référence (~2-3 min, barre de
     progression), puis la LUT est ajustée automatiquement (flot optique).
  4. Les trois vues (entrée / référence / résultat LUT) sont affichées.
  5. Accepter → les paramètres du panneau sont alignés sur l'entrée générée
     et l'aperçu se met à jour.

Toutes les entrées générées sont conservées sur disque (core/genref) ; la
liste déroulante en haut permet de revoir celles qui existent déjà pour
cette image.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QPlainTextEdit, QProgressBar, QWidget, QFrame, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from core import genref


# ══════════════════════════════════════════════════════════════════════════
# Workers (threads de fond)
# ══════════════════════════════════════════════════════════════════════════

class _PromptWorker(QThread):
    """Rédaction du prompt par le VLM (chargement du modèle inclus)."""

    done   = pyqtSignal(dict)    # {"prompt", "items", "cast"}
    failed = pyqtSignal(str)

    def __init__(self, img_bgr: np.ndarray, style: str, parent=None):
        super().__init__(parent)
        self._img = img_bgr
        self._style = style

    def run(self):
        try:
            self.done.emit(genref.write_prompt(self._img, self._style))
        except Exception as exc:
            self.failed.emit(str(exc))


class _GenerateWorker(QThread):
    """Génération FLUX + ajustement LUT (phases avec progression)."""

    phase    = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    done     = pyqtSignal(object)          # GenRefEntry
    failed   = pyqtSignal(str)

    def __init__(self, img_bgr: np.ndarray, style: str, seed: int,
                 prompt: str, parent=None):
        super().__init__(parent)
        self._img = img_bgr
        self._style = style
        self._seed = seed
        self._prompt = prompt

    def run(self):
        try:
            # Libérer la VRAM des modèles du pipeline avant FLUX (~10 Go)
            try:
                from core.model_memory import unload_all_models
                from steps import ALL_STEPS
                unload_all_models(ALL_STEPS)
            except Exception:
                pass
            entry = genref.build_entry(
                self._img, self._style, self._seed,
                prompt=self._prompt or None,
                on_phase=self.phase.emit,
                on_progress=self.progress.emit,
            )
            self.done.emit(entry)
        except Exception as exc:
            self.failed.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════
# Dialogue
# ══════════════════════════════════════════════════════════════════════════

_VIEW_W, _VIEW_H = 380, 260


def _to_pixmap(img_bgr: np.ndarray, w: int = _VIEW_W, h: int = _VIEW_H) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                  rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        w, h, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation)


class GenRefDialog(QDialog):
    """Dialogue de génération / inspection des références IA."""

    def __init__(self, parent, input_bgr: np.ndarray,
                 style: str = "court", seed: int = 42):
        super().__init__(parent)
        self.setWindowTitle("Référence IA — génération et LUT")
        self.setModal(True)
        self.resize(1240, 640)

        self._img = input_bgr
        self._digest = genref.image_digest(input_bgr)
        self._entry: Optional[genref.GenRefEntry] = None
        self._prompt_worker: Optional[_PromptWorker] = None
        self._gen_worker: Optional[_GenerateWorker] = None

        self._build_ui(style, seed)
        self._refresh_existing()
        self._load_entry_if_cached()

    # ── Construction ─────────────────────────────────────────────────────

    def _build_ui(self, style: str, seed: int):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Ligne 1 : paramètres + entrées existantes ────────────────────
        params_row = QHBoxLayout()

        params_row.addWidget(self._mk_label("Style :"))
        self._style_combo = QComboBox()
        self._style_combo.addItems(genref.PROMPT_STYLES)
        if style in genref.PROMPT_STYLES:
            self._style_combo.setCurrentText(style)
        self._style_combo.currentTextChanged.connect(self._on_params_changed)
        params_row.addWidget(self._style_combo)

        params_row.addSpacing(12)
        params_row.addWidget(self._mk_label("Graine :"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 9999)
        self._seed_spin.setValue(int(seed))
        self._seed_spin.valueChanged.connect(self._on_params_changed)
        params_row.addWidget(self._seed_spin)

        params_row.addSpacing(24)
        params_row.addWidget(self._mk_label("Déjà générées :"))
        self._existing_combo = QComboBox()
        self._existing_combo.setMinimumWidth(180)
        self._existing_combo.activated.connect(self._on_existing_selected)
        params_row.addWidget(self._existing_combo)

        params_row.addStretch()
        root.addLayout(params_row)

        # ── Ligne 2 : prompt (éditable) + boutons d'action ───────────────
        prompt_box = QVBoxLayout()
        prompt_header = QHBoxLayout()
        prompt_header.addWidget(self._mk_label("Prompt FLUX (éditable) :"))
        self._cast_lbl = QLabel("")
        self._cast_lbl.setStyleSheet("color: #7ec8c8; font-size: 11px;")
        prompt_header.addWidget(self._cast_lbl)
        prompt_header.addStretch()

        self._write_btn = QPushButton("🤖  Rédiger le prompt (VLM)")
        self._write_btn.setStyleSheet(self._btn_style())
        self._write_btn.clicked.connect(self._on_write_prompt)
        prompt_header.addWidget(self._write_btn)

        self._gen_btn = QPushButton("✨  Générer la référence (~2-3 min GPU)")
        self._gen_btn.setStyleSheet(self._btn_style(primary=True))
        self._gen_btn.clicked.connect(self._on_generate)
        prompt_header.addWidget(self._gen_btn)
        prompt_box.addLayout(prompt_header)

        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setPlaceholderText(
            "Cliquer « Rédiger le prompt (VLM) » pour une proposition "
            "automatique, ou écrire un prompt en anglais (style « manuel »). "
            "Astuce : nommer la couleur du voile (« remove the magenta color "
            "cast ») renforce beaucoup la correction."
        )
        self._prompt_edit.setFixedHeight(86)
        self._prompt_edit.setStyleSheet(
            "QPlainTextEdit { background: #14141f; color: #cdd;"
            " border: 1px solid #2a2a4a; border-radius: 4px; font-size: 11px; }"
        )
        prompt_box.addWidget(self._prompt_edit)
        root.addLayout(prompt_box)

        # ── Ligne 3 : progression ────────────────────────────────────────
        prog_row = QHBoxLayout()
        self._phase_lbl = QLabel("")
        self._phase_lbl.setStyleSheet("color: #f0b64c; font-size: 11px;")
        prog_row.addWidget(self._phase_lbl)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(14)
        self._progress.setVisible(False)
        prog_row.addWidget(self._progress, stretch=1)
        root.addLayout(prog_row)

        # ── Ligne 4 : les trois vues ─────────────────────────────────────
        views_row = QHBoxLayout()
        self._view_in = self._mk_view("Entrée de l'étape")
        self._view_ref = self._mk_view("Référence FLUX (palette seulement)")
        self._view_out = self._mk_view("Résultat LUT (force 1.0)")
        for v in (self._view_in, self._view_ref, self._view_out):
            views_row.addWidget(v["box"])
        root.addLayout(views_row, stretch=1)

        self._view_in["img"].setPixmap(_to_pixmap(self._img))

        # ── Ligne 5 : infos + OK/Annuler ─────────────────────────────────
        bottom = QHBoxLayout()
        self._info_lbl = QLabel("")
        self._info_lbl.setStyleSheet("color: #888; font-size: 10px;")
        bottom.addWidget(self._info_lbl, stretch=1)

        self._cancel_btn = QPushButton("Fermer")
        self._cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(self._cancel_btn)

        self._ok_btn = QPushButton("Utiliser cette référence")
        self._ok_btn.setStyleSheet(self._btn_style(primary=True))
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self.accept)
        bottom.addWidget(self._ok_btn)
        root.addLayout(bottom)

        self.setStyleSheet("QDialog { background: #1a1a2c; }")

    @staticmethod
    def _mk_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #aab; font-size: 11px;")
        return lbl

    @staticmethod
    def _btn_style(primary: bool = False) -> str:
        bg, hover = ("#2a6496", "#3a84b6") if primary else ("#1e3a52", "#2a5577")
        return (f"QPushButton {{ background: {bg}; color: #d8ecf7;"
                f" border-radius: 4px; padding: 6px 14px; font-size: 11px; }}"
                f"QPushButton:hover {{ background: {hover}; }}"
                f"QPushButton:disabled {{ background: #222230; color: #556; }}")

    @staticmethod
    def _mk_view(title: str) -> dict:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lbl = QLabel(title)
        lbl.setStyleSheet("color: #88a; font-size: 10px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        img = QLabel("—")
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setMinimumSize(_VIEW_W, _VIEW_H)
        img.setStyleSheet(
            "background: #101018; color: #444; border: 1px solid #26263a;"
            " border-radius: 4px;"
        )
        lay.addWidget(img, stretch=1)
        return {"box": box, "img": img, "title": lbl}

    # ── État courant ─────────────────────────────────────────────────────

    def current_style(self) -> str:
        return self._style_combo.currentText()

    def current_seed(self) -> int:
        return int(self._seed_spin.value())

    # ── Entrées existantes ───────────────────────────────────────────────

    def _refresh_existing(self):
        self._existing_combo.blockSignals(True)
        self._existing_combo.clear()
        self._existing_combo.addItem("—")
        for e in genref.list_entries(self._digest):
            state = "✓ LUT" if e.beta is not None else "prompt seul"
            self._existing_combo.addItem(f"{e.style} · graine {e.seed} ({state})",
                                         userData=(e.style, e.seed))
        self._existing_combo.blockSignals(False)

    def _on_existing_selected(self, index: int):
        data = self._existing_combo.itemData(index)
        if not data:
            return
        style, seed = data
        self._style_combo.setCurrentText(style)
        self._seed_spin.setValue(seed)
        self._load_entry_if_cached()

    def _on_params_changed(self, *_):
        self._load_entry_if_cached()

    def _load_entry_if_cached(self):
        """Affiche l'entrée du cache correspondant à (style, graine) si elle existe."""
        entry = genref.load_entry(self._digest, self.current_style(),
                                  self.current_seed())
        self._show_entry(entry)

    # ── Rédaction du prompt ──────────────────────────────────────────────

    def _on_write_prompt(self):
        if self._busy():
            return
        style = self.current_style()
        if style == "manuel":
            self._phase_lbl.setText("Style « manuel » : écrire le prompt "
                                    "directement dans la zone de texte.")
            return
        self._set_busy(True, "Rédaction du prompt (VLM, chargement du modèle "
                             "à la première utilisation)…")
        self._prompt_worker = _PromptWorker(self._img, style, self)
        self._prompt_worker.done.connect(self._on_prompt_done)
        self._prompt_worker.failed.connect(self._on_failed)
        self._prompt_worker.start()

    def _on_prompt_done(self, info: dict):
        self._set_busy(False)
        self._prompt_edit.setPlainText(info.get("prompt", ""))
        cast = info.get("cast", "")
        if cast:
            self._cast_lbl.setText(f"voile détecté : {cast}")
        self._phase_lbl.setText("Prompt rédigé — éditable avant génération.")

    # ── Génération ───────────────────────────────────────────────────────

    def _on_generate(self):
        if self._busy():
            return
        prompt = self._prompt_edit.toPlainText().strip()
        style = self.current_style()
        if style == "manuel" and not prompt:
            QMessageBox.information(
                self, "Prompt manquant",
                "Le style « manuel » nécessite un prompt écrit à la main.")
            return
        self._set_busy(True, "Préparation…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)      # indéterminé jusqu'au 1er pas FLUX
        self._gen_worker = _GenerateWorker(
            self._img, style, self.current_seed(), prompt, self)
        self._gen_worker.phase.connect(self._on_phase)
        self._gen_worker.progress.connect(self._on_progress)
        self._gen_worker.done.connect(self._on_generated)
        self._gen_worker.failed.connect(self._on_failed)
        self._gen_worker.start()

    def _on_phase(self, label: str):
        self._phase_lbl.setText(label)

    def _on_progress(self, i: int, n: int):
        self._progress.setRange(0, n)
        self._progress.setValue(i)

    def _on_generated(self, entry):
        self._set_busy(False)
        self._progress.setVisible(False)
        self._phase_lbl.setText("Référence générée et LUT ajustée ✓")
        self._show_entry(entry)
        self._refresh_existing()
        # Invalider le cache mémoire de l'étape (nouvelle LUT sur disque)
        parent = self.parent()
        step = getattr(parent, "_steps_by_id", {}).get("genref") \
            if parent is not None else None
        if step is not None and hasattr(step, "invalidate_cache"):
            step.invalidate_cache()

    def _on_failed(self, msg: str):
        self._set_busy(False)
        self._progress.setVisible(False)
        self._phase_lbl.setText("")
        QMessageBox.warning(self, "Échec", f"L'opération a échoué :\n{msg}")

    # ── Affichage d'une entrée ───────────────────────────────────────────

    def _show_entry(self, entry):
        self._entry = entry
        if entry is None:
            self._view_ref["img"].setText("—")
            self._view_ref["img"].setPixmap(QPixmap())
            self._view_out["img"].setText("—")
            self._view_out["img"].setPixmap(QPixmap())
            self._info_lbl.setText(
                "Aucune référence en cache pour ce style et cette graine.")
            self._ok_btn.setEnabled(False)
            return
        if entry.prompt:
            self._prompt_edit.setPlainText(entry.prompt)
        if entry.cast:
            self._cast_lbl.setText(f"voile détecté : {entry.cast}")
        if entry.ref_bgr is not None:
            self._view_ref["img"].setPixmap(_to_pixmap(entry.ref_bgr))
        if entry.beta is not None:
            out = genref.apply_lut(self._img, entry.beta)
            self._view_out["img"].setPixmap(_to_pixmap(out))
            self._info_lbl.setText(
                f"LUT prête (30 coefficients) · confiance flot "
                f"{entry.conf_mean:.2f} · {entry.style} / graine {entry.seed}")
            self._ok_btn.setEnabled(True)
        else:
            self._ok_btn.setEnabled(False)

    # ── Divers ───────────────────────────────────────────────────────────

    def _busy(self) -> bool:
        for w in (self._prompt_worker, self._gen_worker):
            if w is not None and w.isRunning():
                return True
        return False

    def _set_busy(self, busy: bool, label: str = ""):
        self._write_btn.setEnabled(not busy)
        self._gen_btn.setEnabled(not busy)
        self._style_combo.setEnabled(not busy)
        self._seed_spin.setEnabled(not busy)
        self._existing_combo.setEnabled(not busy)
        if label:
            self._phase_lbl.setText(label)

    def closeEvent(self, event):
        if self._busy():
            QMessageBox.information(
                self, "Génération en cours",
                "Attendre la fin de la génération avant de fermer.")
            event.ignore()
            return
        super().closeEvent(event)
