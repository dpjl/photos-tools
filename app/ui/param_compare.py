"""ui/param_compare.py — Paramètres d'export avec sections pliables."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from core.export_manager import ExportEntry
from steps import ALL_STEPS


_STEP_NAMES = {s.id: s.name for s in ALL_STEPS}
_STEP_SHORT_NAMES = {s.id: s.short_name for s in ALL_STEPS}


class _FoldSection(QFrame):
    """Petite section pliable, compacte par défaut."""

    def __init__(
        self,
        title: str,
        summary: str,
        accent: str,
        expanded: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._expanded = expanded
        self.setStyleSheet(
            "QFrame { background:#1e1e32; border-radius:4px;"
            f" border-left:3px solid {accent}; }}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(7, 4, 5, 5)
        root.setSpacing(3)

        self._header_btn = QPushButton()
        self._header_btn.setCheckable(True)
        self._header_btn.setChecked(expanded)
        self._header_btn.clicked.connect(self._toggle)
        self._header_btn.setStyleSheet(
            "QPushButton { background:transparent; border:none; color:#dde;"
            " text-align:left; padding:0; font-size:10px; font-weight:700; }"
            "QPushButton:hover { color:#fff; }"
        )
        root.addWidget(self._header_btn)

        self._summary_lbl = QLabel(summary)
        self._summary_lbl.setWordWrap(True)
        self._summary_lbl.setStyleSheet(
            "color:#8792a8; font-size:9px; background:transparent;"
        )
        root.addWidget(self._summary_lbl)

        self._body = QWidget()
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(0, 2, 0, 0)
        self._body_lay.setSpacing(2)
        root.addWidget(self._body)
        self._title = title
        self._sync()

    def add_row(self, label: str, value: str, changed: bool = False) -> None:
        row = QLabel(
            f'<span style="color:#778">{label}</span> '
            f'<span style="color:{"#f0a13a" if changed else "#aab"}">{value}</span>'
        )
        row.setWordWrap(True)
        row.setStyleSheet("font-size:10px; background:transparent;")
        self._body_lay.addWidget(row)

    def add_diff(self, export_index: int, value: str) -> None:
        row = QLabel(
            f'<span style="color:#556">#{export_index:03d}</span> '
            f'<span style="color:#8d8a64">{value}</span>'
        )
        row.setWordWrap(True)
        row.setStyleSheet("font-size:9px; background:transparent; margin-left:8px;")
        self._body_lay.addWidget(row)

    def add_note(self, text: str) -> None:
        row = QLabel(text)
        row.setWordWrap(True)
        row.setStyleSheet("font-size:9px; color:#778; background:transparent;")
        self._body_lay.addWidget(row)

    def _toggle(self) -> None:
        self._expanded = self._header_btn.isChecked()
        self._sync()

    def _sync(self) -> None:
        self._header_btn.setText(("▾ " if self._expanded else "▸ ") + self._title)
        self._body.setVisible(self._expanded)


class ParamCompareWidget(QWidget):
    """Vue concise des paramètres entre exports.

    Les étapes sont repliées par défaut : le résumé indique activation, nombre de
    paramètres et différences. Le détail garde les mêmes informations que l'ancien
    affichage, accessible au besoin.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._show_all = False
        self._entry: Optional[ExportEntry] = None
        self._all_entries: list[ExportEntry] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        self._toggle_btn = QPushButton("Afficher toutes les étapes")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setStyleSheet(
            "QPushButton { background:#15152a; color:#86a4be; border:1px solid #2a2a4a;"
            " border-radius:4px; font-size:10px; text-align:left; padding:5px 7px; }"
            "QPushButton:hover { color:#b9d7ee; border-color:#3a4e6c; }"
            "QPushButton:checked { color:#d4e7f7; }"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)
        lay.addWidget(self._toggle_btn)

        self._steps_container = QWidget()
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.setContentsMargins(0, 0, 0, 0)
        self._steps_layout.setSpacing(5)
        lay.addWidget(self._steps_container)
        lay.addStretch()

    def set_data(
        self,
        entry: Optional[ExportEntry],
        all_entries: list[ExportEntry],
    ):
        self._entry = entry
        self._all_entries = all_entries
        self._rebuild()

    def _on_toggle(self):
        self._show_all = self._toggle_btn.isChecked()
        self._toggle_btn.setText(
            "Masquer les étapes sans différence"
            if self._show_all else "Afficher toutes les étapes"
        )
        self._rebuild()

    def _rebuild(self):
        while self._steps_layout.count():
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._entry or not self._entry.recipe_data:
            return

        recipe = self._entry.recipe_data
        step_order = list(recipe.get("step_order", []))
        step_enabled = recipe.get("step_enabled", {})
        step_params = recipe.get("step_params", {})

        all_recipes = [
            (e.index, e.recipe_data)
            for e in self._all_entries
            if e.recipe_data
        ]
        other_recipes = [
            (i, r)
            for i, r in all_recipes
            if i != self._entry.index
        ]

        self._build_global_params(recipe, other_recipes)

        for sid in step_order:
            enabled = step_enabled.get(sid, True)
            params = step_params.get(sid, {})
            enabled_diffs, param_diffs = _diffs_for_step(
                sid, enabled, params, other_recipes
            )
            has_diff = bool(enabled_diffs or param_diffs)
            active_anywhere = _step_active_anywhere(sid, all_recipes)
            if not self._show_all and (not active_anywhere or not has_diff):
                continue

            section = _FoldSection(
                title=_step_title(sid, enabled),
                summary=_step_summary(enabled, params, enabled_diffs, param_diffs),
                accent="#3a8fd4" if has_diff else "#303052",
                expanded=False,
            )
            if enabled_diffs:
                section.add_row("Activation", "différente", changed=True)
                for o_idx, o_val in enabled_diffs:
                    section.add_diff(o_idx, "activée" if o_val else "désactivée")

            keys = sorted(params.keys())
            for key in keys:
                diffs = param_diffs.get(key, [])
                if not self._show_all and not diffs:
                    continue
                section.add_row(key, _format_val(params.get(key)), changed=bool(diffs))
                for o_idx, o_val in diffs[:4]:
                    section.add_diff(o_idx, _format_val(o_val))
                if len(diffs) > 4:
                    section.add_note(f"+{len(diffs) - 4} autre(s) export(s)")
            self._steps_layout.addWidget(section)

    def _build_global_params(self, recipe: dict, other_recipes: list):
        items = []
        for key, label in (
            ("crop_rect", "Recadrage"),
            ("wb_pick", "Point WB"),
            ("wb_patch_radius", "Rayon WB"),
        ):
            value = recipe.get(key)
            diffs = [
                (idx, other.get(key))
                for idx, other in other_recipes
                if other.get(key) != value
            ]
            if self._show_all or diffs:
                items.append((label, value, diffs))
        if not items:
            return

        diff_count = sum(1 for _, _, diffs in items if diffs)
        section = _FoldSection(
            title="Paramètres globaux",
            summary=f"{diff_count} différence(s) · {len(items)} valeur(s)",
            accent="#3a8fd4" if diff_count else "#303052",
            expanded=False,
        )
        for label, value, diffs in items:
            section.add_row(label, _format_val(value), changed=bool(diffs))
            for idx, other_value in diffs[:4]:
                section.add_diff(idx, _format_val(other_value))
        self._steps_layout.addWidget(section)


def _diffs_for_step(sid: str, enabled: bool, params: dict, other_recipes: list):
    enabled_diffs = []
    param_diffs: dict[str, list[tuple[int, object]]] = {}
    for o_idx, o_recipe in other_recipes:
        o_enabled = o_recipe.get("step_enabled", {}).get(sid)
        if o_enabled is not None and o_enabled != enabled:
            enabled_diffs.append((o_idx, o_enabled))
        o_params = o_recipe.get("step_params", {}).get(sid, {})
        for key in sorted(set(params.keys()) | set(o_params.keys())):
            val = params.get(key)
            other = o_params.get(key)
            if val != other:
                param_diffs.setdefault(key, []).append((o_idx, other))
    return enabled_diffs, param_diffs


def _step_active_anywhere(sid: str, all_recipes: list[tuple[int, dict]]) -> bool:
    return any(recipe.get("step_enabled", {}).get(sid, False) for _, recipe in all_recipes)


def _step_title(sid: str, enabled: bool) -> str:
    state = "✓" if enabled else "✗"
    name = _STEP_SHORT_NAMES.get(sid) or _STEP_NAMES.get(sid, sid)
    return f"{state} {name}"


def _step_summary(
    enabled: bool,
    params: dict,
    enabled_diffs: list,
    param_diffs: dict,
) -> str:
    changed = len(param_diffs) + (1 if enabled_diffs else 0)
    state = "activée" if enabled else "désactivée"
    if changed:
        return f"{state} · {changed} différence(s) · {len(params)} paramètre(s)"
    return f"{state} · {len(params)} paramètre(s)"


def _format_val(val) -> str:
    if isinstance(val, float):
        return f"{val:.3g}"
    if isinstance(val, bool):
        return "oui" if val else "non"
    if val is None:
        return "—"
    text = str(val)
    if len(text) > 90:
        return text[:87] + "..."
    return text
