"""batch_mixins/artifacts.py — Détection automatique d'artefacts (onglet Masque).

Construit le panneau de réglages, lance la détection (rayures via réseau + points
colorés) en tâche de fond et recombine le masque en live selon les réglages.

Mixin de BatchWindow : toutes les méthodes utilisent les attributs de la fenêtre
(``self._mask_panel``, ``self._statusbar``, l'état ``self._artifact_*`` initialisé
dans la construction de l'UI).
"""

from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QLabel, QCheckBox, QSlider, QPushButton,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


# Niveaux de « Couverture » → échelles d'inférence fusionnées (max multi-échelle).
# L'échelle haute (1600) donne la précision/les détails fins ; l'échelle basse
# assure la continuité des longues rayures faibles. Plus la basse est petite,
# plus on capte de rayures/poussières (mais plus grossièrement).
_ARTIFACT_COVER_LEVELS = [
    ("Fine",      (1600,)),       # haute résolution seule — précis, peut fragmenter
    ("Standard",  (1600, 640)),   # fusion équilibrée — défaut
    ("Élevée",    (1600, 500)),
    ("Maximale",  (1600, 360)),
]
_ARTIFACT_COVER_DEFAULT = 1   # « Standard » — rayures complètes + propre

# Valeurs par défaut des réglages de détection d'artefacts (pour le bouton « Défaut »)
_ARTIFACT_THRESH_DEFAULT = 50   # seuil IA ×100
_ARTIFACT_COLOR_DEFAULT  = 40   # sensibilité couleur (écart chromatique)
_ARTIFACT_DILATE_DEFAULT = 1    # dilatation (px)


class _ArtifactDetectWorker(QThread):
    """Calcule la carte de probabilité de rayures hors du thread UI.

    Seule l'inférence neuronale (lente, + téléchargement au 1er appel) tourne
    ici ; le seuillage, la dilatation et les points colorés (rapides) sont
    appliqués dans le thread UI pour permettre un réglage live.
    """

    done   = pyqtSignal(object)   # liste de cartes de proba par échelle (ou None)
    failed = pyqtSignal(str)

    def __init__(self, image_bgr: np.ndarray, scales: tuple) -> None:
        super().__init__()
        self._image = image_bgr
        self._scales = scales

    def run(self) -> None:
        try:
            from core.artifact_detect import ScratchDetector
            # On renvoie les cartes PAR échelle : la fusion (max ↔ consensus) se
            # fait ensuite dans l'UI, en live, sans relancer l'inférence.
            self.done.emit(
                ScratchDetector.get().detect_prob_scales(self._image, self._scales)
            )
        except Exception as exc:  # noqa: BLE001 — remonté à l'UI via signal
            self.failed.emit(str(exc))


class ArtifactsMixin:
    """Détection automatique d'artefacts pour l'onglet Masque."""

    def _build_artifact_controls(self) -> QWidget:
        """Panneau de réglages de la détection d'artefacts (live)."""
        box = QFrame()
        box.setStyleSheet("QFrame { background:#16201a; border-radius:4px; }")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 5, 6, 6)
        lay.setSpacing(3)

        def _lbl(text: str) -> QLabel:
            q = QLabel(text)
            q.setStyleSheet("color:#9ab; font-size:9px; background:transparent;")
            return q

        cb_css = "QCheckBox { color:#bcd; font-size:10px; background:transparent; }"

        # Cases : rayures (IA) / points colorés
        self._artifact_scratch_cb = QCheckBox("Rayures/plis (IA)")
        self._artifact_spots_cb   = QCheckBox("Points colorés")
        for cb in (self._artifact_scratch_cb, self._artifact_spots_cb):
            cb.setChecked(True)
            cb.setStyleSheet(cb_css)
            cb.toggled.connect(self._on_artifact_param_changed)
            lay.addWidget(cb)

        # Consensus multi-échelle : exige un accord entre résolutions → retire
        # les faux positifs linéaires (arêtes d'objets, motifs). Live (max↔moyenne).
        self._artifact_consensus_cb = QCheckBox("Consensus (anti-FP)")
        self._artifact_consensus_cb.setChecked(False)
        self._artifact_consensus_cb.setToolTip(
            "Exige qu'une rayure soit détectée à plusieurs résolutions.\n"
            "Retire les faux positifs type arêtes/motifs, mais capte un peu\n"
            "moins les rayures et poussières les plus faibles."
        )
        self._artifact_consensus_cb.setStyleSheet(cb_css)
        self._artifact_consensus_cb.toggled.connect(self._on_artifact_param_changed)
        lay.addWidget(self._artifact_consensus_cb)

        # Couverture = résolution d'inférence (bas = plus couvrant, relance l'IA)
        self._artifact_cover_lbl = _lbl("Couv. : Standard")
        lay.addWidget(self._artifact_cover_lbl)
        self._artifact_cover_slider = QSlider(Qt.Orientation.Horizontal)
        self._artifact_cover_slider.setRange(0, 3)   # 0=Fine … 3=Maximale
        self._artifact_cover_slider.setValue(_ARTIFACT_COVER_DEFAULT)
        self._artifact_cover_slider.valueChanged.connect(self._on_artifact_cover_label)
        self._artifact_cover_slider.sliderReleased.connect(self._on_artifact_cover_changed)
        lay.addWidget(self._artifact_cover_slider)

        # Seuil IA (0.05–0.95)
        self._artifact_thresh_lbl = _lbl("Seuil : 0.50")
        lay.addWidget(self._artifact_thresh_lbl)
        self._artifact_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._artifact_thresh_slider.setRange(5, 95)
        self._artifact_thresh_slider.setValue(_ARTIFACT_THRESH_DEFAULT)
        self._artifact_thresh_slider.valueChanged.connect(self._on_artifact_param_changed)
        lay.addWidget(self._artifact_thresh_slider)

        # Sensibilité couleur (écart chromatique, plus bas = plus sensible)
        self._artifact_color_lbl = _lbl("Couleur : 40")
        lay.addWidget(self._artifact_color_lbl)
        self._artifact_color_slider = QSlider(Qt.Orientation.Horizontal)
        self._artifact_color_slider.setRange(15, 90)
        self._artifact_color_slider.setValue(_ARTIFACT_COLOR_DEFAULT)
        self._artifact_color_slider.valueChanged.connect(self._on_artifact_param_changed)
        lay.addWidget(self._artifact_color_slider)

        # Dilatation (px)
        self._artifact_dilate_lbl = _lbl("Dilat. : 1 px")
        lay.addWidget(self._artifact_dilate_lbl)
        self._artifact_dilate_slider = QSlider(Qt.Orientation.Horizontal)
        self._artifact_dilate_slider.setRange(0, 6)
        self._artifact_dilate_slider.setValue(_ARTIFACT_DILATE_DEFAULT)
        self._artifact_dilate_slider.valueChanged.connect(self._on_artifact_param_changed)
        lay.addWidget(self._artifact_dilate_slider)

        # Bouton : restaurer les valeurs par défaut
        btn_default = QPushButton("↺  Défaut")
        btn_default.setToolTip("Réinitialise tous les réglages de détection ci-dessus.")
        btn_default.setStyleSheet(
            "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
            "  padding:4px 6px; font-size:10px; }"
            "QPushButton:hover { background:#2a2a50; color:#ccc; }"
        )
        btn_default.clicked.connect(self._reset_artifact_params)
        lay.addWidget(btn_default)

        return box

    def _reset_artifact_params(self) -> None:
        """Restaure les réglages de détection d'artefacts à leurs valeurs par défaut."""
        widgets = (
            self._artifact_scratch_cb, self._artifact_spots_cb,
            self._artifact_consensus_cb,
            self._artifact_cover_slider, self._artifact_thresh_slider,
            self._artifact_color_slider, self._artifact_dilate_slider,
        )
        for w in widgets:
            w.blockSignals(True)
        self._artifact_scratch_cb.setChecked(True)
        self._artifact_spots_cb.setChecked(True)
        self._artifact_consensus_cb.setChecked(False)
        self._artifact_cover_slider.setValue(_ARTIFACT_COVER_DEFAULT)
        self._artifact_thresh_slider.setValue(_ARTIFACT_THRESH_DEFAULT)
        self._artifact_color_slider.setValue(_ARTIFACT_COLOR_DEFAULT)
        self._artifact_dilate_slider.setValue(_ARTIFACT_DILATE_DEFAULT)
        for w in widgets:
            w.blockSignals(False)

        # Resynchroniser les libellés
        self._on_artifact_cover_label()
        self._update_artifact_labels()

        # Si une détection a déjà tourné, relancer avec les réglages par défaut
        # (couverture incluse → réinférence) ; sinon il n'y a rien à recomposer.
        if self._artifact_base_image is not None:
            self._run_artifact_inference(report=False)

    def _artifact_scales(self) -> tuple:
        """Échelles d'inférence correspondant au niveau de couverture choisi."""
        return _ARTIFACT_COVER_LEVELS[self._artifact_cover_slider.value()][1]

    def _mask_auto_detect(self) -> None:
        """Lance la détection et applique le résultat (réseau en tâche de fond)."""
        if self._artifact_worker is not None:
            return  # détection déjà en cours
        # On détecte sur l'image ACTUELLEMENT affichée (= preview courant), pas sur
        # l'originale figée : ainsi un preview SCUNet/couleur change la détection.
        img = self._mask_panel.get_display_image()
        if img is None or img.size <= 3:
            self._statusbar.showMessage("Aucune image chargée.")
            return

        # Une nouvelle détection invalide une revue VLM en cours.
        self._vlm_apply_btn.setEnabled(False)
        # Point de référence : masque manuel actuel + image analysée.
        self._mask_panel.push_undo()           # une seule entrée undo pour toute l'auto-détection
        self._artifact_base_mask = self._mask_panel.get_mask()
        self._artifact_base_image = img
        self._artifact_probs = None
        self._artifact_spots = None
        self._artifact_spots_dev = -1.0
        self._run_artifact_inference(report=True)

    def _run_artifact_inference(self, report: bool) -> None:
        """Calcule (ou recalcule) la carte IA pour l'image de référence courante."""
        if self._artifact_worker is not None or self._artifact_base_image is None:
            return

        if not self._artifact_scratch_cb.isChecked():
            # Pas d'IA demandée → uniquement points colorés (immédiat).
            self._artifact_probs = None
            self._recompute_artifact_mask(report=report)
            return

        self._artifact_pending_report = report
        self._mask_auto_btn.setEnabled(False)
        self._statusbar.showMessage(
            "Détection des artefacts en cours… "
            "(1er appel : téléchargement du modèle, peut être long)"
        )
        worker = _ArtifactDetectWorker(self._artifact_base_image, self._artifact_scales())
        worker.done.connect(self._on_artifacts_detected)
        worker.failed.connect(self._on_artifacts_failed)
        worker.finished.connect(self._on_artifact_worker_finished)
        self._artifact_worker = worker
        worker.start()

    def _on_artifact_cover_label(self) -> None:
        """Met à jour le libellé de couverture pendant le glissement (sans relancer)."""
        name = _ARTIFACT_COVER_LEVELS[self._artifact_cover_slider.value()][0]
        self._artifact_cover_lbl.setText(f"Couv. : {name}")

    def _on_artifact_cover_changed(self) -> None:
        """Relâchement du curseur de couverture → relance l'inférence (≈0,4 s)."""
        self._on_artifact_cover_label()
        if self._artifact_base_image is not None:
            self._run_artifact_inference(report=False)

    def _on_artifacts_detected(self, probs) -> None:
        self._artifact_probs = probs
        self._recompute_artifact_mask(report=self._artifact_pending_report)

    def _on_artifacts_failed(self, message: str) -> None:
        self._statusbar.showMessage(f"Erreur détection artefacts : {message}")

    def _on_artifact_worker_finished(self) -> None:
        self._mask_auto_btn.setEnabled(True)
        self._artifact_worker = None

    def _update_artifact_labels(self) -> None:
        """Resynchronise les libellés des réglages avec les valeurs des curseurs."""
        thr = self._artifact_thresh_slider.value() / 100.0
        self._artifact_thresh_lbl.setText(f"Seuil : {thr:.2f}")
        self._artifact_color_lbl.setText(
            f"Couleur : {self._artifact_color_slider.value()}"
        )
        self._artifact_dilate_lbl.setText(
            f"Dilat. : {self._artifact_dilate_slider.value()} px"
        )

    def _on_artifact_param_changed(self) -> None:
        """Réglage live : recombine le masque depuis le cache, sans relancer l'IA."""
        self._update_artifact_labels()
        # Recombiner seulement si une détection a déjà été lancée sur cette image
        if self._artifact_base_image is not None:
            self._recompute_artifact_mask(report=False)

    def _recompute_artifact_mask(self, report: bool) -> None:
        """Recombine masque manuel + détections selon les réglages courants."""
        from core.artifact_detect import (
            mask_from_prob, dilate_mask, detect_color_spots, fuse_probs,
        )
        base_mask = self._artifact_base_mask
        base_img  = self._artifact_base_image
        if base_mask is None or base_img is None:
            return

        # Les détections sont calculées dans l'espace de l'image analysée (preview).
        detected = np.zeros(base_img.shape[:2], dtype=np.uint8)
        if self._artifact_scratch_cb.isChecked() and self._artifact_probs:
            thr  = self._artifact_thresh_slider.value() / 100.0
            # Fusion max ↔ consensus (moyenne) — live, sans relancer l'IA.
            prob = fuse_probs(self._artifact_probs, self._artifact_consensus_cb.isChecked())
            detected = cv2.bitwise_or(detected, mask_from_prob(prob, thr))
        if self._artifact_spots_cb.isChecked():
            dev = float(self._artifact_color_slider.value())
            if self._artifact_spots is None or self._artifact_spots_dev != dev:
                self._artifact_spots = detect_color_spots(base_img, chroma_dev=dev)
                self._artifact_spots_dev = dev
            detected = cv2.bitwise_or(detected, self._artifact_spots)

        detected = dilate_mask(detected, self._artifact_dilate_slider.value())

        # Réaligner sur l'espace du masque si le preview a d'autres dimensions
        # (ex. recadrage / upscale dans le pipeline).
        if detected.shape[:2] != base_mask.shape[:2]:
            detected = cv2.resize(
                detected, (base_mask.shape[1], base_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = cv2.bitwise_or(base_mask, detected)
        self._mask_panel.set_mask(mask, push_undo=False)

        if report:
            n = int(cv2.connectedComponents(detected)[0]) - 1
            if n <= 0:
                self._statusbar.showMessage("Aucun artefact détecté automatiquement.")
            else:
                self._statusbar.showMessage(
                    f"{n} zone(s) d'artefact détectée(s) — réglez seuil/sensibilité à droite."
                )
