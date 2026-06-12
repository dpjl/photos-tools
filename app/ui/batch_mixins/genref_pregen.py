"""batch_mixins/genref_pregen.py — Pré-génération FLUX pour tout le batch.

Bouton « ✨ Pré-gén. FLUX » dans la toolbar (à côté de « ⚡ Lancer le
batch ») : pour chaque photo (sélection si présente, sinon tout le batch),
produit EXACTEMENT ce que ferait le flux manuel de l'onglet « Réf. IA » —
entrée = pipeline jusqu'à l'étape genref, prompt rédigé par le VLM (style
choisi au lancement, une fois pour toutes), génération FLUX, version
sidecar enregistrée à côté de la source et activée dans la recette. Que
l'étape soit cochée ou non est sans importance.

Optimisation des bascules VLM ↔ FLUX (exclusifs en VRAM) : deux phases —
  A. pour chaque photo : calcul de l'entrée + rédaction du prompt (le VLM
     reste chargé pour toute la phase) ;
  B. pour chaque photo : génération FLUX (pipeline gardé chargé via
     keep_loaded, déchargé une seule fois à la fin de la série).
Soit un seul aller-retour au lieu de deux par photo.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QCheckBox, QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal

from core import genref, genref_store
from core.batch import BatchImageConfig, save_recipe


# ══════════════════════════════════════════════════════════════════════════
# Worker
# ══════════════════════════════════════════════════════════════════════════

class _PregenWorker(QThread):
    """Pré-génération en deux phases (VLM puis FLUX) sur une liste d'images."""

    progress = pyqtSignal(int, int, str)     # (i, n, libellé)
    photo_done = pyqtSignal(str, int)        # (file_path, index version)
    photo_skipped = pyqtSignal(str, str)     # (file_path, raison)
    finished_ok = pyqtSignal(int, int, int)  # (générées, ignorées, échecs)
    failed = pyqtSignal(str)

    def __init__(self, targets: list[BatchImageConfig], style: str, seed: int,
                 skip_existing: bool, steps_by_id: dict, inject_fn,
                 gen_steps: int = 28, parent=None):
        super().__init__(parent)
        self._targets = list(targets)
        self._style = style
        self._seed = int(seed)
        self._skip_existing = skip_existing
        self._steps_by_id = steps_by_id
        self._inject = inject_fn          # _inject_instance_state (pas de UI)
        self._gen_steps = gen_steps
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    # ── Pipeline jusqu'à l'étape genref (équivalent du preview d'entrée) ──

    def _compute_input(self, cfg: BatchImageConfig,
                       img: np.ndarray) -> np.ndarray:
        self._inject(cfg)
        out = img
        ctx: dict = {}
        for sid in cfg.step_order:
            if sid == "genref":
                break
            if not cfg.step_enabled.get(sid, True):
                continue
            step = self._steps_by_id.get(sid)
            if step is None:
                continue
            params = cfg.step_params.get(sid) or step.default_params()
            try:
                result, extras = step.process(out, params, ctx)
                out = result
                ctx.update(extras)
            except Exception:
                pass   # même tolérance que PipelineWorker : étape sautée
        return out

    @staticmethod
    def _shrink(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
        """Réduit l'entrée stockée en RAM. Sans perte pour l'usage aval :
        la génération FLUX travaille à ≈1 Mpx et le prompt VLM à 896 px."""
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return img
        s = max_side / max(h, w)
        return cv2.resize(img, (int(w * s), int(h * s)),
                          interpolation=cv2.INTER_AREA)

    @staticmethod
    def _unload_gpu(steps: bool = True) -> None:
        """Vide la VRAM entre les passes (une seule famille de modèles à la
        fois : étapes du pipeline, puis VLM, puis FLUX)."""
        try:
            from core.model_memory import unload_all_models
            from steps import ALL_STEPS
            unload_all_models(ALL_STEPS if steps else [])
        except Exception:
            pass

    def run(self):
        try:
            n = len(self._targets)
            generated = skipped = errors = 0
            self.failures: list[tuple[str, str]] = []

            def _fail(cfg, reason: str) -> None:
                self.failures.append((cfg.filename, reason))
                self.photo_skipped.emit(cfg.file_path, reason)

            # ── Passe A1 : entrées (modèles d'étapes seuls sur le GPU) ───
            self._unload_gpu()   # session précédente (FLUX, VLM, etc.)
            staged: list[tuple[BatchImageConfig, np.ndarray]] = []
            for i, cfg in enumerate(self._targets):
                if self._cancel:
                    break
                if self._skip_existing and genref_store.list_versions(
                        cfg.file_path):
                    self.photo_skipped.emit(cfg.file_path, "version existante")
                    skipped += 1
                    continue
                self.progress.emit(i + 1, n, f"{cfg.filename} — entrée "
                                             "(pipeline jusqu'à l'étape)")
                img = cv2.imread(cfg.file_path, cv2.IMREAD_COLOR)
                if img is None:
                    _fail(cfg, "image illisible")
                    errors += 1
                    continue
                try:
                    input_img = self._shrink(self._compute_input(cfg, img))
                except Exception as exc:
                    _fail(cfg, f"entrée : {type(exc).__name__}: {exc}")
                    errors += 1
                    continue
                staged.append((cfg, input_img))

            # ── Passe A2 : prompts (VLM seul sur le GPU) ─────────────────
            self._unload_gpu()
            prepared: list[tuple[BatchImageConfig, np.ndarray, dict]] = []
            for i, (cfg, input_img) in enumerate(staged):
                if self._cancel:
                    break
                self.progress.emit(i + 1, len(staged),
                                   f"{cfg.filename} — prompt (VLM)")
                try:
                    info = genref.write_prompt(input_img, self._style)
                except Exception as exc:
                    _fail(cfg, f"prompt : {type(exc).__name__}: {exc}")
                    errors += 1
                    continue
                prepared.append((cfg, input_img, info))

            # ── Passe B : générations (FLUX seul sur le GPU) ─────────────
            self._unload_gpu()

            engine = genref.GenRefEngine.get()
            try:
                for j, (cfg, input_img, info) in enumerate(prepared):
                    if self._cancel:
                        break
                    name = cfg.filename
                    self.progress.emit(j + 1, len(prepared),
                                       f"{name} — génération FLUX")
                    try:
                        ref = engine.generate(
                            input_img, info["prompt"], seed=self._seed,
                            steps=self._gen_steps, keep_loaded=True)
                        version = genref_store.save_version(
                            cfg.file_path, ref, info["prompt"], self._style,
                            self._seed, cast=info.get("cast", ""),
                            items=info.get("items", []))
                        cfg.genref_version = version.index
                        save_recipe(cfg)
                        generated += 1
                        self.photo_done.emit(cfg.file_path, version.index)
                    except Exception as exc:
                        self.failures.append(
                            (name, f"génération : {type(exc).__name__}: {exc}"))
                        self.photo_skipped.emit(cfg.file_path,
                                                f"génération : {exc}")
                        errors += 1
            finally:
                engine.unload()

            self.finished_ok.emit(generated, skipped, errors)
        except Exception as exc:
            self.failed.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════
# Dialogue de lancement
# ══════════════════════════════════════════════════════════════════════════

class _PregenDialog(QDialog):
    """Options de la pré-génération : style, graine, photos déjà traitées."""

    def __init__(self, parent, n_targets: int):
        super().__init__(parent)
        self.setWindowTitle("Pré-génération FLUX")
        self.setModal(True)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        intro = QLabel(
            f"Génère pour {n_targets} photo(s) une référence IA complète\n"
            "(prompt VLM + image FLUX), enregistrée à côté de chaque source\n"
            "et activée dans sa recette — comme le flux manuel de l'onglet\n"
            "« Réf. IA ». Compter ~1 à 2 min par photo."
        )
        intro.setStyleSheet("color:#aab; font-size:11px;")
        lay.addWidget(intro)

        row = QHBoxLayout()
        lbl = QLabel("Style de prompt :")
        lbl.setStyleSheet("color:#aab; font-size:11px;")
        row.addWidget(lbl)
        self._style_combo = QComboBox()
        # « manuel » exclu : la pré-génération est entièrement automatique
        self._style_combo.addItems(["court", "détaillé"])
        row.addWidget(self._style_combo, stretch=1)
        lbl2 = QLabel("Graine :")
        lbl2.setStyleSheet("color:#aab; font-size:11px;")
        row.addWidget(lbl2)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 9999)
        self._seed_spin.setValue(42)
        row.addWidget(self._seed_spin)
        lay.addLayout(row)

        self._skip_cb = QCheckBox(
            "Ignorer les photos ayant déjà au moins une version")
        self._skip_cb.setChecked(True)
        self._skip_cb.setStyleSheet("color:#bcd; font-size:11px;")
        lay.addWidget(self._skip_cb)

        btns = QHBoxLayout()
        btns.addStretch()
        cancel = QPushButton("Annuler")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        ok = QPushButton("✨  Lancer la pré-génération")
        ok.setStyleSheet(
            "QPushButton { background:#2a6496; color:#d8ecf7;"
            " border-radius:4px; padding:6px 14px; }"
            "QPushButton:hover { background:#3a84b6; }")
        ok.clicked.connect(self.accept)
        ok.setDefault(True)
        btns.addWidget(ok)
        lay.addLayout(btns)
        self.setStyleSheet("QDialog { background:#1a1a2c; }")

    def options(self) -> tuple[str, int, bool]:
        return (self._style_combo.currentText(),
                int(self._seed_spin.value()),
                self._skip_cb.isChecked())


# ══════════════════════════════════════════════════════════════════════════
# Mixin
# ══════════════════════════════════════════════════════════════════════════

class GenRefPregenMixin:
    """Bouton toolbar + orchestration de la pré-génération FLUX."""

    def _build_pregen_button(self) -> QPushButton:
        self._pregen_worker: Optional[_PregenWorker] = None
        self._pregen_btn = QPushButton("✨  Pré-gén. FLUX")
        self._pregen_btn.setToolTip(
            "Pré-génère les références IA (prompt VLM + image FLUX) de\n"
            "chaque photo de la sélection (ou de tout le batch), comme le\n"
            "ferait le flux manuel de l'onglet « Réf. IA » — versions\n"
            "enregistrées à côté des sources et activées dans les recettes.\n"
            "L'état coché/décoché de l'étape est sans importance."
        )
        self._style_btn(self._pregen_btn, accent=True)
        self._pregen_btn.clicked.connect(self._on_pregen_clicked)
        return self._pregen_btn

    # ── Lancement / annulation ────────────────────────────────────────────

    def _on_pregen_clicked(self) -> None:
        if self._pregen_worker is not None and self._pregen_worker.isRunning():
            self._pregen_worker.cancel()
            self._pregen_btn.setText("⏳ Arrêt en cours…")
            self._pregen_btn.setEnabled(False)
            return

        if self._worker is not None and self._worker.isRunning():
            self._statusbar.showMessage(
                "Pré-génération indisponible pendant un traitement batch.")
            return
        if (self._full_preview_worker is not None
                and self._full_preview_worker.isRunning()):
            self._statusbar.showMessage(
                "Pré-génération indisponible pendant un preview complet.")
            return

        self._save_current_state()
        targets = self._propagation_targets()
        if not targets:
            self._statusbar.showMessage("Aucune photo dans le batch.")
            return

        dlg = _PregenDialog(self, len(targets))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        style, seed, skip_existing = dlg.options()

        self._pregen_worker = _PregenWorker(
            targets, style, seed, skip_existing,
            self._steps_by_id, self._inject_instance_state, parent=self)
        self._pregen_worker.progress.connect(self._on_pregen_progress)
        self._pregen_worker.photo_done.connect(self._on_pregen_photo_done)
        self._pregen_worker.photo_skipped.connect(self._on_pregen_skipped)
        self._pregen_worker.finished_ok.connect(self._on_pregen_finished)
        self._pregen_worker.failed.connect(self._on_pregen_failed)

        self._pregen_btn.setText("⏹  Annuler pré-gén.")
        self._selection_btn.setEnabled(False)
        self._batch_btn.setEnabled(False)
        self._pregen_worker.start()

    # ── Suivi ─────────────────────────────────────────────────────────────

    def _on_pregen_progress(self, i: int, n: int, label: str) -> None:
        self._statusbar.showMessage(f"Pré-génération FLUX {i}/{n} — {label}")

    def _on_pregen_photo_done(self, file_path: str, index: int) -> None:
        self._strip.set_customized(file_path, True)
        cfg = self._config_for(file_path)
        # La version a été activée dans la recette ; si c'est l'image
        # affichée, rafraîchir l'injection et l'onglet Réf. IA.
        if cfg is not None and cfg is self._current_cfg:
            self._inject_instance_state(cfg)
            if hasattr(self, "_genref_refresh_versions"):
                self._genref_refresh_versions()
                self._genref_refresh_triptych()
            self._schedule_preview_update()

    def _on_pregen_skipped(self, file_path: str, reason: str) -> None:
        import os
        self._statusbar.showMessage(
            f"Pré-génération : {os.path.basename(file_path)} ignorée "
            f"({reason})")

    def _on_pregen_finished(self, generated: int, skipped: int,
                            errors: int) -> None:
        from ui.notifications import Level
        self._pregen_reset_buttons()
        msg = f"{generated} générée(s), {skipped} ignorée(s), {errors} échec(s)"
        self._statusbar.showMessage(f"Pré-génération FLUX terminée — {msg}")
        if hasattr(self, "_notif"):
            level = Level.SUCCESS if errors == 0 else Level.WARNING
            self._notif.notify("Pré-génération FLUX terminée", msg,
                               level=level, duration=8000)
        # Bilan détaillé des échecs : les messages de la barre d'état ne font
        # que passer — sans ce récapitulatif l'utilisateur ne sait pas
        # pourquoi rien n'a été généré.
        failures = list(getattr(self._pregen_worker, "failures", []) or [])
        if failures:
            shown = failures[:8]
            lines = [f"• {name} — {reason}" for name, reason in shown]
            if len(failures) > len(shown):
                lines.append(f"… +{len(failures) - len(shown)} autre(s)")
            QMessageBox.warning(
                self, "Pré-génération : échecs",
                f"{msg}\n\nDétail des échecs :\n" + "\n".join(lines))

    def _on_pregen_failed(self, msg: str) -> None:
        self._pregen_reset_buttons()
        QMessageBox.warning(self, "Échec",
                            f"La pré-génération a échoué :\n{msg}")

    def _pregen_reset_buttons(self) -> None:
        self._pregen_btn.setText("✨  Pré-gén. FLUX")
        self._pregen_btn.setEnabled(True)
        self._selection_btn.setEnabled(True)
        self._batch_btn.setEnabled(True)
