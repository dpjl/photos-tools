"""ui/mixins/batch.py — Traitement par lot et gestion des dossiers récents."""

from __future__ import annotations

import os

from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QSettings


class BatchMixin:

    def _on_batch_requested(self) -> None:
        """Ouvre la fenêtre batch avec un snapshot de la configuration courante."""
        from core.batch import BatchSession, BatchImageConfig  # noqa: F401
        from ui.batch_window import BatchWindow  # noqa: F401

        # Si déjà ouverte, la ramener au premier plan
        if self._batch_window is not None:
            self._batch_window.raise_()
            self._batch_window.activateWindow()
            return

        # Choisir le dossier source
        start_dir = (
            os.path.dirname(self._original_path)
            if self._original_path
            else ""
        )
        folder = QFileDialog.getExistingDirectory(
            self, "Dossier d'images à traiter en batch", start_dir
        )
        if not folder:
            return
        self._open_batch_folder(folder)

    def _open_batch_folder(self, folder: str) -> None:
        """Ouvre un batch depuis un dossier donné (partagé par menu et récents)."""
        from core.batch import BatchSession, BatchImageConfig
        from ui.batch_window import BatchWindow

        if self._batch_window is not None:
            self._batch_window.raise_()
            self._batch_window.activateWindow()
            return

        if not os.path.isdir(folder):
            self._status_bar.showMessage(f"Dossier introuvable : {folder}")
            return

        self._add_recent_batch(folder)

        # Snapshot de la config courante
        defaults = BatchImageConfig(
            file_path    = "",
            step_order   = list(self._ctrl.step_list.get_order()),
            step_enabled = dict(self._ctrl.step_list.get_enabled()),
            step_params  = {
                k: dict(v)
                for k, v in self._ctrl.step_list.get_all_params().items()
            },
        )

        session = BatchSession()
        session.load_folder(folder, defaults)

        if not session.images:
            self._status_bar.showMessage("Aucune image trouvée dans ce dossier.")
            return

        self._batch_window = BatchWindow(self, session)
        self._batch_window.closed.connect(self._on_batch_closed)
        self._ctrl.set_running(True)
        self._batch_window.show()

    def _add_recent_batch(self, path: str) -> None:
        """Préfixe path dans la liste des batchs récents (max 8)."""
        s = QSettings("PlusTekPhoto", "RestaurationPhoto")
        recents: list = s.value("recent_batches", [], type=list)
        recents = [p for p in recents if p != path]
        recents.insert(0, path)
        s.setValue("recent_batches", recents[:8])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        s = QSettings("PlusTekPhoto", "RestaurationPhoto")
        recents: list = s.value("recent_batches", [], type=list)
        for path in recents:
            folder_name = os.path.basename(path.rstrip("/\\")) or path
            act = QAction(folder_name, self)
            act.setToolTip(path)
            act.triggered.connect(lambda checked=False, p=path: self._open_batch_folder(p))
            self._recent_menu.addAction(act)
        if recents:
            self._recent_menu.addSeparator()
            clear_act = QAction("Effacer les récents", self)
            clear_act.triggered.connect(self._clear_recent_batches)
            self._recent_menu.addAction(clear_act)
        self._recent_menu.setEnabled(bool(recents))

    def _clear_recent_batches(self) -> None:
        QSettings("PlusTekPhoto", "RestaurationPhoto").remove("recent_batches")
        self._rebuild_recent_menu()

    def _on_batch_closed(self) -> None:
        self._batch_window = None
        self._ctrl.set_running(False)
