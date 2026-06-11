"""batch_mixins/redzones.py — Détection des zones rouges (onglet Zones rouges).

Construit le panneau de réglages, lance l'analyse chromatique (rembg + cartes
d'écart au chroma attendu) en tâche de fond, recombine le masque en live selon
les curseurs, propose l'extension en bandes des bords et la vérification VLM
(cast résiduel ↔ couleur naturelle) avec revue interactive.

Flux recommandé : Détecter → Vérifier (VLM) → Appliquer → Étendre en bandes.

Mixin de BatchWindow : toutes les méthodes utilisent les attributs de la
fenêtre (``self._redzone_panel``, ``self._statusbar``, l'état ``self._redzone_*``
initialisé dans la construction de l'UI).
"""

from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QSlider,
    QPushButton, QProgressBar, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Valeurs par défaut des réglages (pour le bouton « Défaut »)
_REDZONE_SENS_DEFAULT = 12   # écart chromatique minimal (chroma_min)
_REDZONE_AREA_DEFAULT = 15   # taille minimale ×100 px


class _RedZoneAnalyzeWorker(QThread):
    """Calcule l'analyse chromatique (rembg + cartes d'écart) hors du thread UI.

    Seule l'analyse (lente : segmentation des personnes + référence par
    luminance) tourne ici ; le seuillage (rapide) est appliqué dans le thread
    UI pour permettre un réglage live des curseurs.
    """

    done   = pyqtSignal(object)   # RedZoneAnalysis
    failed = pyqtSignal(str)

    def __init__(self, image_bgr: np.ndarray, bg_only: bool) -> None:
        super().__init__()
        self._image = image_bgr
        self._bg_only = bg_only

    def run(self) -> None:
        try:
            from core.redzone import analyze_red_zones, background_mask
            bg = None
            if self._bg_only:
                from core.rembg_session import get_shared_session
                bg = background_mask(self._image, get_shared_session())
            self.done.emit(analyze_red_zones(self._image, bg))
        except Exception as exc:  # noqa: BLE001 — remonté à l'UI via signal
            self.failed.emit(str(exc))


class _RedZoneVlmWorker(QThread):
    """Classe les zones rouges (cast ↔ naturel) avec le VLM hors du thread UI."""

    progress = pyqtSignal(int, int)   # (fait, total)
    done     = pyqtSignal(object)     # RefineResult
    failed   = pyqtSignal(str)

    def __init__(self, image_bgr: np.ndarray, mask: np.ndarray, model_id: str) -> None:
        super().__init__()
        self._image = image_bgr
        self._mask = mask
        self._model_id = model_id

    def run(self) -> None:
        try:
            from core.vlm_refine import VLMRefiner
            res = VLMRefiner.get().refine_redzones(
                self._image, self._mask, self._model_id,
                on_progress=lambda k, t: self.progress.emit(k, t),
            )
            self.done.emit(res)
        except Exception as exc:  # noqa: BLE001 — remonté à l'UI via signal
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class RedZonesMixin:
    """Détection automatique des zones rouges pour l'onglet Zones rouges."""

    # ── Panneau de réglages ──────────────────────────────────────────────────

    def _build_redzone_controls(self) -> QWidget:
        """Panneau de réglages de la détection des zones rouges (live)."""
        box = QFrame()
        box.setStyleSheet("QFrame { background:#241a1a; border-radius:4px; }")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 5, 6, 6)
        lay.setSpacing(3)

        def _lbl(text: str) -> QLabel:
            q = QLabel(text)
            q.setStyleSheet("color:#b09a9a; font-size:9px; background:transparent;")
            return q

        # Sensibilité = écart chromatique minimal au chroma attendu
        self._redzone_sens_lbl = _lbl(f"Sensibilité : {_REDZONE_SENS_DEFAULT}")
        lay.addWidget(self._redzone_sens_lbl)
        self._redzone_sens_slider = QSlider(Qt.Orientation.Horizontal)
        self._redzone_sens_slider.setRange(6, 30)
        self._redzone_sens_slider.setValue(_REDZONE_SENS_DEFAULT)
        self._redzone_sens_slider.setToolTip(
            "Écart de couleur minimal au chroma « normal » des surfaces de\n"
            "même luminosité. Plus bas = plus sensible (plus de zones)."
        )
        self._redzone_sens_slider.valueChanged.connect(self._on_redzone_param_changed)
        lay.addWidget(self._redzone_sens_slider)

        # Taille minimale d'une zone (×100 px)
        self._redzone_area_lbl = _lbl(f"Taille ≥ {_REDZONE_AREA_DEFAULT * 100} px")
        lay.addWidget(self._redzone_area_lbl)
        self._redzone_area_slider = QSlider(Qt.Orientation.Horizontal)
        self._redzone_area_slider.setRange(3, 60)
        self._redzone_area_slider.setValue(_REDZONE_AREA_DEFAULT)
        self._redzone_area_slider.valueChanged.connect(self._on_redzone_param_changed)
        lay.addWidget(self._redzone_area_slider)

        # Fond seulement (exclut les personnes via rembg) — relance l'analyse
        self._redzone_bgonly_cb = QCheckBox("Fond seulement (IA)")
        self._redzone_bgonly_cb.setChecked(True)
        self._redzone_bgonly_cb.setToolTip(
            "Exclut les personnes (segmentation U2-Net) de la détection.\n"
            "Décocher pour détecter aussi sur les personnes."
        )
        self._redzone_bgonly_cb.setStyleSheet(
            "QCheckBox { color:#cbb; font-size:10px; background:transparent; }"
        )
        self._redzone_bgonly_cb.toggled.connect(self._on_redzone_bgonly_changed)
        lay.addWidget(self._redzone_bgonly_cb)

        # Extension en bandes : action explicite (après revue VLM de préférence)
        btn_bands = QPushButton("▭  Étendre en bandes (bords)")
        btn_bands.setToolTip(
            "Si un bord de la photo est largement touché, couvre la bande\n"
            "entière le long de ce bord (les bandes de détérioration courent\n"
            "souvent sur toute la longueur). À utiliser de préférence APRÈS\n"
            "la vérification VLM, pour ne pas fusionner défauts et zones\n"
            "naturelles avant leur classification."
        )
        btn_bands.setStyleSheet(
            "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
            "  padding:4px 6px; font-size:10px; }"
            "QPushButton:hover { background:#2a2a50; color:#ccc; }"
        )
        btn_bands.clicked.connect(self._redzone_extend_bands)
        lay.addWidget(btn_bands)

        # Bouton : restaurer les valeurs par défaut
        btn_default = QPushButton("↺  Défaut")
        btn_default.setToolTip("Réinitialise les réglages de détection ci-dessus.")
        btn_default.setStyleSheet(
            "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
            "  padding:4px 6px; font-size:10px; }"
            "QPushButton:hover { background:#2a2a50; color:#ccc; }"
        )
        btn_default.clicked.connect(self._reset_redzone_params)
        lay.addWidget(btn_default)

        return box

    def _build_redzone_vlm_controls(self) -> QWidget:
        """Panneau VLM : vérification cast ↔ naturel + revue interactive."""
        box = QFrame()
        box.setStyleSheet("QFrame { background:#201a26; border-radius:4px; }")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 5, 6, 6)
        lay.setSpacing(3)

        title = QLabel("Vérification IA (VLM)")
        title.setStyleSheet(
            "color:#c9b; font-size:10px; font-weight:700; background:transparent;"
        )
        lay.addWidget(title)

        self._redzone_vlm_btn = QPushButton("✨  Vérifier (VLM)")
        self._redzone_vlm_btn.setToolTip(
            "Montre chaque zone au VLM local (modèle choisi dans l'onglet\n"
            "Masque) qui distingue un cast résiduel d'une couleur naturelle\n"
            "(bois, sol, tissu rouge…). Les zones « naturelles » passent en\n"
            "violet : double-clic pour corriger, puis « Appliquer »."
        )
        self._redzone_vlm_btn.setStyleSheet(
            "QPushButton { background:#2a1e38; color:#c9a6e8; border:1px solid #5a3a7a;"
            "  border-radius:4px; padding:5px 6px; font-size:10px; }"
            "QPushButton:hover { background:#3a2a50; color:#dcb6ff; }"
            "QPushButton:disabled { color:#544; border-color:#332; }"
        )
        self._redzone_vlm_btn.clicked.connect(self._redzone_vlm_verify)
        lay.addWidget(self._redzone_vlm_btn)

        self._redzone_vlm_progress = QProgressBar()
        self._redzone_vlm_progress.setVisible(False)
        self._redzone_vlm_progress.setTextVisible(True)
        self._redzone_vlm_progress.setFixedHeight(14)
        self._redzone_vlm_progress.setStyleSheet(
            "QProgressBar { background:#16121e; border:1px solid #3a2a4a;"
            "  border-radius:4px; color:#dcb; font-size:9px; text-align:center; }"
            "QProgressBar::chunk { background:#7a4caa; border-radius:3px; }"
        )
        lay.addWidget(self._redzone_vlm_progress)

        self._redzone_apply_btn = QPushButton("✅  Appliquer")
        self._redzone_apply_btn.setEnabled(False)
        self._redzone_apply_btn.setToolTip(
            "Supprime du masque toutes les zones marquées NATURELLES (violet).\n"
            "Double-cliquez une zone pour changer violet ↔ vert avant d'appliquer."
        )
        self._redzone_apply_btn.setStyleSheet(
            "QPushButton { background:#1e3320; color:#9de0a8; border:1px solid #3a6a45;"
            "  border-radius:4px; padding:5px 6px; font-size:10px; }"
            "QPushButton:hover { background:#274530; color:#bff0c8; }"
            "QPushButton:disabled { color:#455; border-color:#233; }"
        )
        self._redzone_apply_btn.clicked.connect(self._redzone_apply_review)
        lay.addWidget(self._redzone_apply_btn)

        self._redzone_convo_btn = QPushButton("\U0001f4ac  Conversation")
        self._redzone_convo_btn.setEnabled(False)
        self._redzone_convo_btn.setToolTip("Voir l'échange complet avec le VLM.")
        self._redzone_convo_btn.setStyleSheet(
            "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
            "  padding:4px 4px; font-size:10px; }"
            "QPushButton:hover { background:#2a2a50; color:#ccc; }"
            "QPushButton:disabled { color:#445; }"
        )
        self._redzone_convo_btn.clicked.connect(self._show_redzone_vlm_conversation)
        lay.addWidget(self._redzone_convo_btn)

        return box

    # ── Détection automatique ────────────────────────────────────────────────

    def _redzone_auto_detect(self) -> None:
        """Lance la détection et applique le résultat (analyse en tâche de fond)."""
        if self._redzone_worker is not None:
            return  # analyse déjà en cours
        # Détection sur l'image ACTUELLEMENT affichée (= preview courant) : les
        # zones rouges résiduelles n'existent qu'après les corrections globales.
        img = self._redzone_panel.get_display_image()
        if img is None or img.size <= 3:
            self._statusbar.showMessage("Aucune image chargée.")
            return

        # Une nouvelle détection invalide une revue VLM en cours.
        self._redzone_apply_btn.setEnabled(False)
        # Point de référence : masque manuel actuel + image analysée.
        self._redzone_panel.push_undo()    # une seule entrée undo pour l'auto-détection
        self._redzone_base_mask = self._redzone_panel.get_mask()
        self._redzone_base_image = img
        self._redzone_analysis = None
        self._run_redzone_analysis(report=True)

    def _run_redzone_analysis(self, report: bool) -> None:
        """Calcule (ou recalcule) l'analyse chromatique pour l'image de référence."""
        if self._redzone_worker is not None or self._redzone_base_image is None:
            return

        self._redzone_pending_report = report
        self._redzone_auto_btn.setEnabled(False)
        self._statusbar.showMessage(
            "Analyse des zones rouges en cours… "
            "(segmentation des personnes + référence couleur)"
        )
        worker = _RedZoneAnalyzeWorker(
            self._redzone_base_image, self._redzone_bgonly_cb.isChecked()
        )
        worker.done.connect(self._on_redzone_analyzed)
        worker.failed.connect(self._on_redzone_analyze_failed)
        worker.finished.connect(self._on_redzone_worker_finished)
        self._redzone_worker = worker
        worker.start()

    def _on_redzone_analyzed(self, analysis) -> None:
        self._redzone_analysis = analysis
        self._recompute_redzone_mask(report=self._redzone_pending_report)

    def _on_redzone_analyze_failed(self, message: str) -> None:
        self._statusbar.showMessage(f"Erreur détection zones rouges : {message}")

    def _on_redzone_worker_finished(self) -> None:
        self._redzone_auto_btn.setEnabled(True)
        self._redzone_worker = None

    def _on_redzone_param_changed(self) -> None:
        """Réglage live : recombine le masque depuis le cache, sans relancer l'analyse."""
        self._update_redzone_labels()
        if self._redzone_analysis is not None:
            self._recompute_redzone_mask(report=False)

    def _on_redzone_bgonly_changed(self) -> None:
        """« Fond seulement » fait partie de l'analyse → relance si déjà détecté."""
        if self._redzone_base_image is not None:
            self._redzone_analysis = None
            self._run_redzone_analysis(report=False)

    def _update_redzone_labels(self) -> None:
        """Resynchronise les libellés des réglages avec les valeurs des curseurs."""
        self._redzone_sens_lbl.setText(
            f"Sensibilité : {self._redzone_sens_slider.value()}"
        )
        self._redzone_area_lbl.setText(
            f"Taille ≥ {self._redzone_area_slider.value() * 100} px"
        )

    def _recompute_redzone_mask(self, report: bool) -> None:
        """Recombine masque manuel + détection selon les réglages courants."""
        from core.redzone import mask_from_analysis
        base_mask = self._redzone_base_mask
        if base_mask is None or self._redzone_analysis is None:
            return

        detected = mask_from_analysis(
            self._redzone_analysis,
            chroma_min=float(self._redzone_sens_slider.value()),
            min_area=self._redzone_area_slider.value() * 100,
        )

        # Réaligner sur l'espace du masque si le preview a d'autres dimensions
        # (ex. recadrage / upscale dans le pipeline).
        if detected.shape[:2] != base_mask.shape[:2]:
            detected = cv2.resize(
                detected, (base_mask.shape[1], base_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = cv2.bitwise_or(base_mask, detected)
        self._redzone_panel.set_mask(mask, push_undo=False)

        if report:
            n = int(cv2.connectedComponents(detected)[0]) - 1
            if n <= 0:
                self._statusbar.showMessage("Aucune zone rouge détectée automatiquement.")
            else:
                self._statusbar.showMessage(
                    f"{n} zone(s) rouge(s) candidate(s) — réglez la sensibilité à "
                    f"droite, puis « Vérifier (VLM) » pour écarter les couleurs "
                    f"naturelles."
                )

    def _reset_redzone_params(self) -> None:
        """Restaure les réglages de détection à leurs valeurs par défaut."""
        widgets = (self._redzone_sens_slider, self._redzone_area_slider,
                   self._redzone_bgonly_cb)
        for w in widgets:
            w.blockSignals(True)
        self._redzone_sens_slider.setValue(_REDZONE_SENS_DEFAULT)
        self._redzone_area_slider.setValue(_REDZONE_AREA_DEFAULT)
        bg_changed = not self._redzone_bgonly_cb.isChecked()
        self._redzone_bgonly_cb.setChecked(True)
        for w in widgets:
            w.blockSignals(False)

        self._update_redzone_labels()
        if self._redzone_base_image is not None:
            if bg_changed:
                self._redzone_analysis = None
                self._run_redzone_analysis(report=False)
            elif self._redzone_analysis is not None:
                self._recompute_redzone_mask(report=False)

    # ── Extension en bandes ──────────────────────────────────────────────────

    def _redzone_extend_bands(self) -> None:
        """Étend le masque courant en bandes le long des bords largement touchés."""
        from core.redzone import extend_border_bands
        mask = self._redzone_panel.get_mask()
        if mask is None or not mask.any():
            self._statusbar.showMessage("Le masque est vide — lancez d'abord la détection.")
            return
        extended = extend_border_bands(mask)
        added = int((extended > 0).sum() - (mask > 0).sum())
        if added <= 0:
            self._statusbar.showMessage(
                "Aucun bord suffisamment touché pour une extension en bande."
            )
            return
        self._redzone_panel.set_mask(extended)   # push_undo=True → annulable
        self._statusbar.showMessage(
            f"Bande(s) de bord étendue(s) : +{added} px de masque (Ctrl+Z pour annuler)."
        )

    # ── Vérification VLM ─────────────────────────────────────────────────────

    def _redzone_vlm_verify(self) -> None:
        """Lance la classification VLM des zones du masque (en tâche de fond)."""
        if self._redzone_vlm_worker is not None:
            return
        img = self._redzone_panel.get_display_image()
        mask = self._redzone_panel.get_mask()
        if img is None or img.size <= 3:
            self._statusbar.showMessage("Aucune image chargée.")
            return
        if mask is None or not mask.any():
            self._statusbar.showMessage("Le masque est vide — lancez d'abord la détection.")
            return

        # Modèle choisi dans le panneau VLM de l'onglet Masque (état partagé).
        model_id = self._vlm_model_combo.currentData()
        self._redzone_vlm_btn.setEnabled(False)
        self._redzone_vlm_progress.setValue(0)
        self._redzone_vlm_progress.setFormat("préparation…")
        self._redzone_vlm_progress.setVisible(True)
        self._statusbar.showMessage(
            "Vérification VLM des zones rouges… "
            "(1er usage : téléchargement du modèle, peut être long)"
        )
        worker = _RedZoneVlmWorker(img, mask, model_id)
        worker.progress.connect(self._on_redzone_vlm_progress)
        worker.done.connect(self._on_redzone_vlm_done)
        worker.failed.connect(self._on_redzone_vlm_failed)
        worker.finished.connect(self._on_redzone_vlm_finished)
        self._redzone_vlm_worker = worker
        worker.start()

    def _on_redzone_vlm_progress(self, k: int, total: int) -> None:
        if total > 0:
            self._redzone_vlm_progress.setMaximum(total)
            self._redzone_vlm_progress.setValue(k)
            self._redzone_vlm_progress.setFormat("%v / %m  zones")
            self._statusbar.showMessage(f"Vérification VLM : zone {k}/{total}…")
        else:
            self._redzone_vlm_progress.setMaximum(0)  # indéterminé (chargement)
            self._redzone_vlm_progress.setFormat("chargement du modèle…")

    def _on_redzone_vlm_done(self, result) -> None:
        self._redzone_vlm_last_result = result
        self._redzone_convo_btn.setEnabled(True)
        if hasattr(self, "_vlm_unload_btn"):
            self._vlm_unload_btn.setEnabled(True)
        # Mode REVUE : cast (vert, gardé) / naturel (violet, à retirer).
        self._redzone_panel.enter_review(
            result.review_labels, result.review_categories, result.review_reasons)
        self._redzone_apply_btn.setEnabled(True)
        n_nat, n_cast = self._redzone_panel.review_counts()
        self._statusbar.showMessage(
            f"VLM ({result.model_id.split('/')[-1]}) en {result.elapsed:.0f}s : "
            f"{n_cast} cast (vert), {n_nat} naturelle(s) (violet). "
            f"Survolez pour la raison, double-clic pour corriger, puis « Appliquer »."
        )

    def _redzone_apply_review(self) -> None:
        """Applique la revue : supprime les zones naturelles (violet) du masque."""
        if not self._redzone_panel.in_review():
            self._redzone_apply_btn.setEnabled(False)
            return
        n = self._redzone_panel.apply_review()
        self._redzone_apply_btn.setEnabled(False)
        self._statusbar.showMessage(
            f"{n} zone(s) naturelle(s) retirée(s) du masque — "
            f"« Étendre en bandes » si un bord est touché."
        )

    def _on_redzone_vlm_failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "erreur inconnue"
        self._statusbar.showMessage(f"Erreur VLM : {first}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Vérification VLM — erreur")
        box.setText("La vérification par le VLM a échoué.")
        box.setInformativeText(first)
        box.setDetailedText(message)
        box.exec()

    def _on_redzone_vlm_finished(self) -> None:
        self._redzone_vlm_btn.setEnabled(True)
        self._redzone_vlm_progress.setVisible(False)
        self._redzone_vlm_worker = None

    def _show_redzone_vlm_conversation(self) -> None:
        if self._redzone_vlm_last_result is None:
            return
        from ui.vlm_conversation_dialog import VLMConversationDialog
        VLMConversationDialog(self, self._redzone_vlm_last_result).exec()

    # ── Invalidation (navigation entre images) ───────────────────────────────

    def _invalidate_redzone_cache(self) -> None:
        """Oublie l'analyse liée à l'image précédente (navigation/restauration)."""
        self._redzone_analysis = None
        self._redzone_base_mask = None
        self._redzone_base_image = None
        if hasattr(self, "_redzone_apply_btn"):
            self._redzone_apply_btn.setEnabled(False)
