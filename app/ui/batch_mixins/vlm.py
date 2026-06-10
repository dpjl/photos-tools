"""batch_mixins/vlm.py — Affinage du masque par VLM local + gestion VRAM.

Construit le panneau VLM (choix du modèle, token HF, seuils de candidats), lance
l'affinage en tâche de fond, ouvre les fenêtres d'inspection (candidats /
conversation), et gère la jauge VRAM + le déchargement global des modèles.

Mixin de BatchWindow : utilise les attributs de la fenêtre (``self._mask_panel``,
``self._statusbar``, les widgets ``self._vlm_*`` et ``self._vram_bar``, l'état
``self._vlm_worker`` / ``self._vlm_last_result`` initialisé à la construction).
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSlider, QProgressBar, QInputDialog, QLineEdit, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class _VLMRefineWorker(QThread):
    """Affine le masque avec un VLM local hors du thread UI."""

    progress = pyqtSignal(int, int)   # (fait, total)
    done     = pyqtSignal(object)     # RefineResult
    failed   = pyqtSignal(str)

    def __init__(self, image_bgr: np.ndarray, mask: np.ndarray, model_id: str,
                 min_len: int, min_thick: int) -> None:
        super().__init__()
        self._image = image_bgr
        self._mask = mask
        self._model_id = model_id
        self._min_len = min_len
        self._min_thick = min_thick

    def run(self) -> None:
        try:
            from core.vlm_refine import VLMRefiner
            res = VLMRefiner.get().refine(
                self._image, self._mask, self._model_id,
                min_len=self._min_len, min_thick=self._min_thick,
                on_progress=lambda k, t: self.progress.emit(k, t),
            )
            self.done.emit(res)
        except Exception as exc:  # noqa: BLE001 — remonté à l'UI via signal
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class VlmMixin:
    """Affinage VLM du masque + jauge/déchargement VRAM."""

    def _build_vlm_controls(self) -> QWidget:
        """Panneau : modèle + token + seuils candidats + affinage + inspection."""
        from core.vlm_refine import VLM_MODELS, DEFAULT_MODEL

        box = QFrame()
        box.setStyleSheet("QFrame { background:#201a26; border-radius:4px; }")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 5, 6, 6)
        lay.setSpacing(3)

        def _lbl(text: str) -> QLabel:
            q = QLabel(text)
            q.setStyleSheet("color:#a89ab0; font-size:9px; background:transparent;")
            return q

        # Style commun des boutons secondaires (compacts)
        sec_css = (
            "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
            "  padding:4px 4px; font-size:10px; }"
            "QPushButton:hover { background:#2a2a50; color:#ccc; }"
            "QPushButton:disabled { color:#445; }"
        )

        title = QLabel("Affinage IA (VLM)")
        title.setStyleSheet("color:#c9b; font-size:10px; font-weight:700; background:transparent;")
        lay.addWidget(title)

        # Modèle + bouton token (gated repos comme Gemma)
        row = QHBoxLayout(); row.setSpacing(4)
        self._vlm_model_combo = QComboBox()
        for label, repo, fits in VLM_MODELS:
            self._vlm_model_combo.addItem(label, repo)
        idx = self._vlm_model_combo.findData(DEFAULT_MODEL)
        if idx >= 0:
            self._vlm_model_combo.setCurrentIndex(idx)
        self._vlm_model_combo.setStyleSheet(
            "QComboBox { background:#1a1426; color:#cbd; border:1px solid #3a2a4a;"
            "  border-radius:4px; padding:3px 6px; font-size:10px; }"
            "QComboBox QAbstractItemView { background:#1a1426; color:#cbd;"
            "  selection-background-color:#3a2a5a; }"
        )
        row.addWidget(self._vlm_model_combo, stretch=1)
        btn_token = QPushButton("🔑")
        btn_token.setFixedWidth(28)
        btn_token.setToolTip("Renseigner un token HuggingFace (pour les modèles\n"
                             "restreints comme Gemma). Enregistré de façon permanente.")
        btn_token.setStyleSheet(
            "QPushButton { background:#1a1426; color:#cb9; border:1px solid #3a2a4a;"
            "  border-radius:4px; padding:3px; font-size:11px; }"
            "QPushButton:hover { background:#2a2040; }"
        )
        btn_token.clicked.connect(self._set_hf_token)
        row.addWidget(btn_token)
        lay.addLayout(row)

        # Seuils de sélection des candidats
        self._vlm_line_lbl = _lbl("Lignes ≥ 45 px")
        lay.addWidget(self._vlm_line_lbl)
        self._vlm_line_slider = QSlider(Qt.Orientation.Horizontal)
        self._vlm_line_slider.setRange(20, 200); self._vlm_line_slider.setValue(45)
        self._vlm_line_slider.valueChanged.connect(
            lambda v: self._vlm_line_lbl.setText(f"Lignes ≥ {v} px"))
        lay.addWidget(self._vlm_line_slider)

        self._vlm_blob_lbl = _lbl("Taches ≥ 12 px")
        lay.addWidget(self._vlm_blob_lbl)
        self._vlm_blob_slider = QSlider(Qt.Orientation.Horizontal)
        self._vlm_blob_slider.setRange(6, 60); self._vlm_blob_slider.setValue(12)
        self._vlm_blob_slider.valueChanged.connect(
            lambda v: self._vlm_blob_lbl.setText(f"Taches ≥ {v} px"))
        lay.addWidget(self._vlm_blob_slider)

        btn_cand = QPushButton("👁  Voir les candidats")
        btn_cand.setToolTip("Affiche, sans lancer le VLM, les zones qui seront\n"
                            "analysées (lignes + grosses taches).")
        btn_cand.setStyleSheet(sec_css)
        btn_cand.clicked.connect(self._show_vlm_candidates)
        lay.addWidget(btn_cand)

        self._vlm_refine_btn = QPushButton("✨  Affiner (VLM)")
        self._vlm_refine_btn.setToolTip(
            "Envoie les zones suspectes (lignes + grosses taches) à un VLM\n"
            "local qui retire celles qui sont en réalité des éléments du décor.\n"
            "1er usage : téléchargement du modèle (~8 Go)."
        )
        self._vlm_refine_btn.setStyleSheet(
            "QPushButton { background:#2a1e38; color:#c9a6e8; border:1px solid #5a3a7a;"
            "  border-radius:4px; padding:5px 6px; font-size:10px; }"
            "QPushButton:hover { background:#3a2a50; color:#dcb6ff; }"
            "QPushButton:disabled { color:#544; border-color:#332; }"
        )
        self._vlm_refine_btn.clicked.connect(self._vlm_refine)
        lay.addWidget(self._vlm_refine_btn)

        self._vlm_progress = QProgressBar()
        self._vlm_progress.setVisible(False)
        self._vlm_progress.setTextVisible(True)
        self._vlm_progress.setFixedHeight(14)
        self._vlm_progress.setStyleSheet(
            "QProgressBar { background:#16121e; border:1px solid #3a2a4a;"
            "  border-radius:4px; color:#dcb; font-size:9px; text-align:center; }"
            "QProgressBar::chunk { background:#7a4caa; border-radius:3px; }"
        )
        lay.addWidget(self._vlm_progress)

        # Bouton d'application de la revue (supprime les zones violettes = décor)
        self._vlm_apply_btn = QPushButton("✅  Appliquer")
        self._vlm_apply_btn.setEnabled(False)
        self._vlm_apply_btn.setToolTip(
            "Supprime du masque toutes les zones marquées DÉCOR (violet).\n"
            "Double-cliquez une zone pour changer violet ↔ vert avant d'appliquer."
        )
        self._vlm_apply_btn.setStyleSheet(
            "QPushButton { background:#1e3320; color:#9de0a8; border:1px solid #3a6a45;"
            "  border-radius:4px; padding:5px 6px; font-size:10px; }"
            "QPushButton:hover { background:#274530; color:#bff0c8; }"
            "QPushButton:disabled { color:#455; border-color:#233; }"
        )
        self._vlm_apply_btn.clicked.connect(self._vlm_apply_review)
        lay.addWidget(self._vlm_apply_btn)

        # Boutons secondaires côte à côte (conversation / décharger)
        row2 = QHBoxLayout(); row2.setSpacing(4)
        self._vlm_convo_btn = QPushButton("\U0001f4ac  Conversation")
        self._vlm_convo_btn.setEnabled(False)
        self._vlm_convo_btn.setToolTip("Voir l'échange complet avec le VLM.")
        self._vlm_convo_btn.setStyleSheet(sec_css)
        self._vlm_convo_btn.clicked.connect(self._show_vlm_conversation)
        row2.addWidget(self._vlm_convo_btn)

        self._vlm_unload_btn = QPushButton("🗑  Décharger")
        self._vlm_unload_btn.setEnabled(False)
        self._vlm_unload_btn.setToolTip("Libère la mémoire GPU occupée par le modèle VLM.")
        self._vlm_unload_btn.setStyleSheet(
            "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
            "  padding:4px 4px; font-size:10px; }"
            "QPushButton:hover { background:#3a2030; color:#e99; }"
            "QPushButton:disabled { color:#445; }"
        )
        self._vlm_unload_btn.clicked.connect(self._unload_vlm)
        row2.addWidget(self._vlm_unload_btn)
        lay.addLayout(row2)

        return box

    def _set_hf_token(self) -> None:
        """Saisie + enregistrement permanent d'un token HuggingFace."""
        from core.vlm_refine import set_hf_token, has_hf_token
        existing = "déjà enregistré" if has_hf_token() else "aucun"
        token, ok = QInputDialog.getText(
            self, "Token HuggingFace",
            f"Token HF (pour les modèles restreints, ex. Gemma).\n"
            f"État actuel : {existing}.\nColler le token (hf_…) :",
            QLineEdit.EchoMode.Password,
        )
        if ok and token.strip():
            try:
                set_hf_token(token)
                self._statusbar.showMessage("Token HuggingFace enregistré.")
            except Exception as exc:
                self._statusbar.showMessage(f"Échec enregistrement token : {exc}")

    def _vlm_candidate_params(self) -> "tuple[int, int]":
        return self._vlm_line_slider.value(), self._vlm_blob_slider.value()

    def _show_vlm_candidates(self) -> None:
        """Aperçu (sans VLM) des zones qui seront analysées."""
        from core.vlm_refine import candidates_overlay
        from ui.vlm_candidates_dialog import VLMCandidatesDialog
        img  = self._mask_panel.get_display_image()
        mask = self._mask_panel.get_mask()
        if mask is None or not mask.any():
            self._statusbar.showMessage("Le masque est vide — lancez d'abord la détection.")
            return
        min_len, min_thick = self._vlm_candidate_params()
        rgb, nl, nb, nk = candidates_overlay(img, mask, min_len, min_thick)
        VLMCandidatesDialog(self, rgb, nl, nb, nk).exec()

    def _unload_vlm(self) -> None:
        from core.vlm_refine import VLMRefiner
        VLMRefiner.get().unload()
        self._vlm_unload_btn.setEnabled(False)
        self._update_vram_gauge()
        self._statusbar.showMessage("VLM déchargé — VRAM libérée.")

    # ── Jauge VRAM / déchargement global ──────────────────────────────────────

    def _update_vram_gauge(self) -> None:
        from core.model_memory import gpu_memory
        info = gpu_memory()
        if info is None:
            self._vram_bar.setMaximum(100)
            self._vram_bar.setValue(0)
            self._vram_bar.setFormat("VRAM n/a")
            return
        used, total = info
        pct = used / total if total else 0.0
        self._vram_bar.setMaximum(max(1, total))
        self._vram_bar.setValue(used)
        self._vram_bar.setFormat(f"VRAM {used/1024:.1f}/{total/1024:.1f} Go")
        chunk = "#5a8f5a" if pct < 0.6 else ("#b89a3a" if pct < 0.85 else "#c05a5a")
        self._vram_bar.setStyleSheet(
            "QProgressBar { background:#14142a; border:1px solid #2a2a4a;"
            " border-radius:4px; color:#dde; font-size:9px; text-align:center; }"
            f"QProgressBar::chunk {{ background:{chunk}; border-radius:3px; }}"
        )

    def _unload_all_models(self) -> None:
        if (self._worker is not None and self._worker.isRunning()) or \
           (self._full_preview_worker is not None and self._full_preview_worker.isRunning()) or \
           (self._vlm_worker is not None):
            self._statusbar.showMessage("Impossible de décharger pendant un traitement.")
            return
        from core.model_memory import unload_all_models
        freed = unload_all_models(list(self._steps_by_id.values()))
        self._vlm_unload_btn.setEnabled(False)
        self._update_vram_gauge()
        self._statusbar.showMessage(f"{len(freed)} modèle(s) déchargé(s) — VRAM libérée.")

    # ── Affinage ──────────────────────────────────────────────────────────────

    def _vlm_refine(self) -> None:
        """Lance l'affinage VLM du masque courant (en tâche de fond)."""
        if self._vlm_worker is not None:
            return
        img  = self._mask_panel.get_display_image()
        mask = self._mask_panel.get_mask()
        if img is None or img.size <= 3:
            self._statusbar.showMessage("Aucune image chargée.")
            return
        if mask is None or not mask.any():
            self._statusbar.showMessage("Le masque est vide — lancez d'abord la détection.")
            return

        model_id = self._vlm_model_combo.currentData()
        min_len, min_thick = self._vlm_candidate_params()
        self._vlm_refine_btn.setEnabled(False)
        self._vlm_progress.setValue(0)
        self._vlm_progress.setFormat("préparation…")
        self._vlm_progress.setVisible(True)
        self._statusbar.showMessage(
            "Affinage VLM… (1er usage : téléchargement du modèle, peut être long)"
        )
        worker = _VLMRefineWorker(img, mask, model_id, min_len, min_thick)
        worker.progress.connect(self._on_vlm_progress)
        worker.done.connect(self._on_vlm_done)
        worker.failed.connect(self._on_vlm_failed)
        worker.finished.connect(self._on_vlm_finished)
        self._vlm_worker = worker
        worker.start()

    def _on_vlm_progress(self, k: int, total: int) -> None:
        if total > 0:
            self._vlm_progress.setMaximum(total)
            self._vlm_progress.setValue(k)
            self._vlm_progress.setFormat("%v / %m  candidats")
            self._statusbar.showMessage(f"Affinage VLM : analyse {k}/{total}…")
        else:
            self._vlm_progress.setMaximum(0)  # indéterminé (chargement modèle)
            self._vlm_progress.setFormat("chargement du modèle…")

    def _on_vlm_done(self, result) -> None:
        self._vlm_last_result = result
        self._vlm_convo_btn.setEnabled(True)
        self._vlm_unload_btn.setEnabled(True)
        # Entrer en mode REVUE (rien n'est supprimé tout de suite) : les candidats
        # s'affichent en violet (décor) / vert (défaut) sur le masque.
        self._mask_panel.enter_review(
            result.review_labels, result.review_categories, result.review_reasons)
        self._vlm_apply_btn.setEnabled(True)
        n_scene, n_def = self._mask_panel.review_counts()
        self._statusbar.showMessage(
            f"VLM ({result.model_id.split('/')[-1]}) en {result.elapsed:.0f}s : "
            f"{n_scene} décor (violet), {n_def} défaut (vert). "
            f"Survolez pour la raison, double-clic pour corriger, puis « Appliquer »."
        )

    def _vlm_apply_review(self) -> None:
        """Applique la revue : supprime les zones décor (violet) du masque."""
        if not self._mask_panel.in_review():
            self._vlm_apply_btn.setEnabled(False)
            return
        n = self._mask_panel.apply_review()
        self._vlm_apply_btn.setEnabled(False)
        self._statusbar.showMessage(f"{n} zone(s) décor supprimée(s) du masque.")

    def _on_vlm_failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "erreur inconnue"
        self._statusbar.showMessage(f"Erreur VLM : {first}")
        # Trace complète dans une boîte de dialogue (utile pour déboguer un modèle).
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Affinage VLM — erreur")
        box.setText("L'affinage par le VLM a échoué.")
        box.setInformativeText(first)
        box.setDetailedText(message)
        box.exec()

    def _on_vlm_finished(self) -> None:
        self._vlm_refine_btn.setEnabled(True)
        self._vlm_progress.setVisible(False)
        self._vlm_worker = None

    def _show_vlm_conversation(self) -> None:
        if self._vlm_last_result is None:
            return
        from ui.vlm_conversation_dialog import VLMConversationDialog
        VLMConversationDialog(self, self._vlm_last_result).exec()
