"""ui/param_compare.py — Comparaison compacte des paramètres entre exports.

Widget affichant les différences de paramètres entre l'export sélectionné
et les autres exports disponibles, dans un format compact adapté au panneau latéral.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt

from core.export_manager import ExportEntry
from steps import ALL_STEPS

# Lookup step name depuis ALL_STEPS
_STEP_NAMES: dict[str, str] = {}
try:
    _STEP_NAMES = {s.id: s.name for s in ALL_STEPS}
except Exception:
    pass


class ParamCompareWidget(QWidget):
    """Vue compacte de comparaison des paramètres entre exports.

    Par défaut : affiche uniquement les paramètres qui diffèrent.
    Bouton toggle : afficher tous les paramètres.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._show_all = False
        self._entry: Optional[ExportEntry] = None
        self._all_entries: list[ExportEntry] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Bouton toggle
        self._toggle_btn = QPushButton("Afficher tous les paramètres")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #68a; border: none;"
            "  font-size: 10px; text-align: left; padding: 2px 0; }"
            "QPushButton:hover { color: #8ac; }"
            "QPushButton:checked { color: #acd; }"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)
        lay.addWidget(self._toggle_btn)

        # Conteneur des étapes
        self._steps_container = QWidget()
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.setContentsMargins(0, 0, 0, 0)
        self._steps_layout.setSpacing(2)
        lay.addWidget(self._steps_container)
        lay.addStretch()

    def set_data(
        self,
        entry: Optional[ExportEntry],
        all_entries: list[ExportEntry],
    ):
        """Met à jour la comparaison."""
        self._entry = entry
        self._all_entries = all_entries
        self._rebuild()

    def _on_toggle(self):
        self._show_all = self._toggle_btn.isChecked()
        self._toggle_btn.setText(
            "Masquer les paramètres communs" if self._show_all
            else "Afficher tous les paramètres"
        )
        self._rebuild()

    def _rebuild(self):
        # Vider
        while self._steps_layout.count():
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._entry or not self._entry.recipe_data:
            return

        recipe = self._entry.recipe_data
        step_order = recipe.get("step_order", [])
        step_enabled = recipe.get("step_enabled", {})
        step_params = recipe.get("step_params", {})

        # Collecter les recipes de TOUS les exports (y compris le courant)
        all_recipes = []
        for e in self._all_entries:
            if e.recipe_data:
                all_recipes.append((e.index, e.recipe_data))

        # Recipes des AUTRES exports (pour les diffs)
        other_recipes = [(i, r) for i, r in all_recipes if i != self._entry.index]

        # Pré-calculer : l'étape est-elle active dans au moins un export ?
        step_active_anywhere: dict[str, bool] = {}
        for sid in step_order:
            active = False
            for _, r in all_recipes:
                if r.get("step_enabled", {}).get(sid, False):
                    active = True
                    break
            step_active_anywhere[sid] = active

        # ── Paramètres globaux (wb_pick, wb_patch_radius) ────────────────
        self._build_global_params(recipe, other_recipes)

        for sid in step_order:
            enabled = step_enabled.get(sid, True)
            params = step_params.get(sid, {})

            # Calculer les diffs pour cette étape
            enabled_diffs = []
            param_diffs: dict[str, list[tuple[int, object]]] = {}

            for o_idx, o_recipe in other_recipes:
                o_enabled = o_recipe.get("step_enabled", {}).get(sid)
                if o_enabled is not None and o_enabled != enabled:
                    enabled_diffs.append((o_idx, o_enabled))

                o_params = o_recipe.get("step_params", {}).get(sid, {})
                all_keys = set(params.keys()) | set(o_params.keys())
                for key in all_keys:
                    v = params.get(key)
                    o_v = o_params.get(key)
                    if v != o_v:
                        if key not in param_diffs:
                            param_diffs[key] = []
                        param_diffs[key].append((o_idx, o_v))

            has_any_diff = bool(enabled_diffs or param_diffs)
            is_active = step_active_anywhere.get(sid, False)

            # En mode "diffs only" : masquer les étapes sans diff ET
            # masquer les étapes inactives partout (même avec des diffs de params)
            if not self._show_all:
                if not is_active or not has_any_diff:
                    continue

            # ── Widget étape ──
            step_widget = QFrame()
            step_widget.setStyleSheet(
                "QFrame { background: #1e1e32; border-radius: 3px;"
                "  border-left: 3px solid %s; padding: 4px; margin: 1px 0; }"
                % ("#3a8fd4" if has_any_diff else "#2a2a4a")
            )
            slayout = QVBoxLayout(step_widget)
            slayout.setContentsMargins(8, 4, 4, 4)
            slayout.setSpacing(2)

            # En-tête étape
            step_name = _STEP_NAMES.get(sid, sid)
            badge = "✓" if enabled else "✗"
            badge_color = "#6e6" if enabled else "#c66"
            diff_icon = " ⚡" if has_any_diff else ""

            header = QLabel(
                f'<span style="color:{badge_color}">{badge}</span> '
                f'<b>{step_name}</b>'
                f'<span style="color:#f80">{diff_icon}</span>'
            )
            header.setStyleSheet("font-size: 11px; color: #ccc; background: transparent;")
            slayout.addWidget(header)

            # Diff enabled
            if enabled_diffs:
                for o_idx, o_val in enabled_diffs:
                    o_str = "✓" if o_val else "✗"
                    dl = QLabel(
                        f'  <span style="color:#888">Export {o_idx:03d}:</span>'
                        f' <span style="color:#f80">{o_str}</span>'
                    )
                    dl.setStyleSheet("font-size: 10px; background: transparent;")
                    slayout.addWidget(dl)

            # Paramètres
            for key in sorted(params.keys()):
                val = params[key]
                diffs = param_diffs.get(key, [])
                has_diff = bool(diffs)

                if not self._show_all and not has_diff:
                    continue

                val_color = "#f90" if has_diff else "#999"
                plabel = QLabel(
                    f'  <span style="color:#777">{key}:</span>'
                    f' <span style="color:{val_color}">{_format_val(val)}</span>'
                )
                plabel.setWordWrap(True)
                plabel.setStyleSheet("font-size: 10px; background: transparent;")
                slayout.addWidget(plabel)

                if has_diff:
                    for o_idx, o_val in diffs[:3]:
                        dl = QLabel(
                            f'    <span style="color:#555">#{o_idx:03d}:</span>'
                            f' <span style="color:#886">{_format_val(o_val)}</span>'
                        )
                        dl.setStyleSheet("font-size: 9px; background: transparent;")
                        slayout.addWidget(dl)

            self._steps_layout.addWidget(step_widget)

    def _build_global_params(self, recipe: dict, other_recipes: list):
        """Affiche les paramètres globaux (wb_pick, wb_patch_radius) avec diff."""
        global_keys = [
            ("wb_pick", "Position balance blancs"),
            ("wb_patch_radius", "Rayon patch WB"),
        ]
        diffs_found = []
        for key, label in global_keys:
            val = recipe.get(key)
            others = []
            has_diff = False
            for o_idx, o_recipe in other_recipes:
                o_val = o_recipe.get(key)
                if o_val != val:
                    has_diff = True
                    others.append((o_idx, o_val))
            if has_diff or self._show_all:
                diffs_found.append((key, label, val, others, has_diff))

        if not diffs_found:
            return

        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background: #1e1e32; border-radius: 3px;"
            "  border-left: 3px solid #3a8fd4; padding: 4px; margin: 1px 0; }"
        )
        flayout = QVBoxLayout(frame)
        flayout.setContentsMargins(8, 4, 4, 4)
        flayout.setSpacing(2)

        header = QLabel('<b>Paramètres globaux</b> <span style="color:#f80">⚡</span>')
        header.setStyleSheet("font-size: 11px; color: #ccc; background: transparent;")
        flayout.addWidget(header)

        for key, label, val, others, has_diff in diffs_found:
            val_color = "#f90" if has_diff else "#999"
            plabel = QLabel(
                f'  <span style="color:#777">{label}:</span>'
                f' <span style="color:{val_color}">{_format_val(val)}</span>'
            )
            plabel.setWordWrap(True)
            plabel.setStyleSheet("font-size: 10px; background: transparent;")
            flayout.addWidget(plabel)
            if has_diff:
                for o_idx, o_val in others[:3]:
                    dl = QLabel(
                        f'    <span style="color:#555">#{o_idx:03d}:</span>'
                        f' <span style="color:#886">{_format_val(o_val)}</span>'
                    )
                    dl.setStyleSheet("font-size: 9px; background: transparent;")
                    flayout.addWidget(dl)

        self._steps_layout.addWidget(frame)


def _format_val(val) -> str:
    """Formate une valeur de paramètre pour l'affichage."""
    if isinstance(val, float):
        return f"{val:.3g}"
    if val is None:
        return "—"
    return str(val)
