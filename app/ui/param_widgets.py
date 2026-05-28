"""ui/param_widgets.py — Widgets paramétriques pour les étapes du pipeline."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QSlider,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QSizePolicy,
    QAbstractSpinBox, QPushButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor

# Correctif PyQt6/Windows : les boutons ↑↓ du spinbox peuvent être masqués
# par la zone d'édition quand la largeur est contrainte. On force leur géométrie.
_SPINBOX_BTN_STYLE = (
    "QAbstractSpinBox::up-button   { subcontrol-origin: border;"
    "  subcontrol-position: top right; width: 18px; }"
    "QAbstractSpinBox::down-button { subcontrol-origin: border;"
    "  subcontrol-position: bottom right; width: 18px; }"
)

# Styles pour les boutons de sélection de profil (type "choice" avec as_buttons=True)
_BTN_CHOICE_STYLE = (
    "QPushButton { border-radius: 3px; padding: 3px 6px; font-size: 10px;"
    "  border: 1px solid #333; background: #252535; color: #aaa; }"
    "QPushButton:checked { background: #2a6496; color: #fff; border: 1px solid #3a84b6; }"
    "QPushButton:hover   { background: #2a2a45; }"
    "QPushButton:checked:hover { background: #3a74a6; }"
)

_BTN_PERSO_STYLE = (
    "QPushButton { border-radius: 3px; padding: 3px 6px; font-size: 10px;"
    "  border: 1px solid #333; background: #252535; color: #888; font-style: italic; }"
    "QPushButton:checked { background: #3a2a4a; color: #cc99ff;"
    "  border: 1px solid #5a4a6a; font-style: italic; }"
    "QPushButton:hover   { background: #2a2a40; }"
)

_BTN_PROPAGATE_STYLE = (
    "QPushButton { background: #1e3050; color: #7ab; border-radius: 3px;"
    "  border: 1px solid #2a4060; font-size: 10px; padding: 0; }"
    "QPushButton:hover { background: #2a4470; color: #adf; }"
)


class ParamRow(QWidget):
    """Ligne de paramètre : label | widget(s) [+ bouton propagation batch optionnel].

    Types supportés :
      float/int  → label | slider | spinbox
      choice     → label | combobox  (ou grille de boutons si as_buttons=True)
      bool       → label | checkbox

    Le bouton ⬇ de propagation est toujours créé mais masqué par défaut.
    Il devient visible au survol quand set_propagate_visible(True) est appelé.
    """

    value_changed       = pyqtSignal(str, object)  # (key, value)
    propagate_requested = pyqtSignal(str, object)  # (key, value) — mode batch

    def __init__(self, param_def: dict, parent=None):
        super().__init__(parent)
        self._def        = param_def
        self._key        = param_def["key"]
        self._type       = param_def["type"]
        self._block      = False
        self._as_buttons = (
            self._type == "choice" and param_def.get("as_buttons", False)
        )
        self._propagate_enabled = False

        # Widgets secondaires (None si non utilisés)
        self._combo:     QComboBox | None                   = None
        self._checkbox:  QCheckBox | None                   = None
        self._slider:    QSlider | None                     = None
        self._spin:      QSpinBox | QDoubleSpinBox | None   = None
        self._btn_group: QButtonGroup | None                = None
        self._btns:      dict[str, QPushButton]             = {}

        # Construction du layout selon le type.
        # Les deux méthodes retournent le layout auquel sera ajouté le bouton ⬇.
        if self._as_buttons:
            self._prop_row = self._build_buttons_layout(param_def)
        else:
            self._prop_row = self._build_normal_layout(param_def)

        # Bouton propagation (commun à tous les types, toujours en fin de ligne)
        self._propagate_btn = QPushButton("⬇")
        self._propagate_btn.setFixedSize(18, 18)
        self._propagate_btn.setToolTip("Propager ce paramètre aux images sélectionnées")
        self._propagate_btn.setStyleSheet(_BTN_PROPAGATE_STYLE)
        # Largeur 0 par défaut : ne prend pas de place tant que batch_mode n'est pas actif
        self._propagate_btn.setVisible(False)
        self._propagate_btn.setMaximumWidth(0)
        self._propagate_btn.clicked.connect(
            lambda: self.propagate_requested.emit(self._key, self.value())
        )
        self._prop_row.addWidget(self._propagate_btn)

        # Timer pour masquage différé (évite le clignotement sur les widgets enfants)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(120)
        self._hide_timer.timeout.connect(self._check_hide_propagate)

        self._scale = (
            10 ** _precision(param_def.get("step", 1))
            if self._type not in ("choice", "bool") else 1
        )

    # ── Construction des layouts ──────────────────────────────────────────────

    def _build_normal_layout(self, param_def: dict) -> QHBoxLayout:
        """Layout horizontal : [label 190px] | [widget central] | [⬇ caché].
        Retourne le layout auquel sera ajouté le bouton propagation.
        """
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(param_def["label"])
        lbl.setFixedWidth(120)
        lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        layout.addWidget(lbl)

        if self._type == "choice":
            self._combo = QComboBox()
            for c in param_def["choices"]:
                self._combo.addItem(c)
            self._combo.setCurrentText(str(param_def["default"]))
            self._combo.currentTextChanged.connect(self._on_combo_changed)
            layout.addWidget(self._combo)

        elif self._type == "bool":
            self._checkbox = QCheckBox()
            self._checkbox.setChecked(bool(param_def["default"]))
            self._checkbox.stateChanged.connect(self._on_checkbox_changed)
            layout.addWidget(self._checkbox)

        else:
            mn   = param_def["min"]
            mx   = param_def["max"]
            step = param_def["step"]
            dflt = param_def["default"]
            prec = _precision(step)
            scl  = 10 ** prec

            self._slider = QSlider(Qt.Orientation.Horizontal)
            self._slider.setRange(round(mn * scl), round(mx * scl))
            self._slider.setValue(round(dflt * scl))
            self._slider.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._slider.valueChanged.connect(self._on_slider_changed)
            layout.addWidget(self._slider)

            if self._type == "int":
                self._spin = QSpinBox()
                self._spin.setRange(int(mn), int(mx))
                self._spin.setValue(int(dflt))
                self._spin.setSingleStep(int(step))
                self._spin.setFixedWidth(58)
                self._spin.valueChanged.connect(self._on_spin_changed_int)
            else:
                self._spin = QDoubleSpinBox()
                self._spin.setRange(mn, mx)
                self._spin.setDecimals(prec)
                self._spin.setSingleStep(step)
                self._spin.setValue(dflt)
                self._spin.setFixedWidth(68)
                self._spin.valueChanged.connect(self._on_spin_changed_float)
            self._spin.setStyleSheet(_SPINBOX_BTN_STYLE)
            layout.addWidget(self._spin)

        return layout

    def _build_buttons_layout(self, param_def: dict) -> QHBoxLayout:
        """Layout vertical : [label + ⬇] / [grille de boutons].
        Pour les paramètres "choice" avec as_buttons=True (ex. sélection de profil).
        Retourne la ligne label (QHBoxLayout) à laquelle sera ajouté le bouton ⬇.
        """
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)

        # Ligne label (le bouton propagation sera ajouté après à ce layout)
        lbl_row = QHBoxLayout()
        lbl_row.setContentsMargins(0, 0, 0, 0)
        lbl_row.setSpacing(4)
        lbl = QLabel(param_def["label"])
        lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        lbl_row.addWidget(lbl)
        lbl_row.addStretch()
        outer.addLayout(lbl_row)

        # Grille de boutons (3 colonnes)
        choices  = param_def["choices"]
        btn_grid = QGridLayout()
        btn_grid.setContentsMargins(0, 0, 0, 0)
        btn_grid.setSpacing(3)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        for i, c in enumerate(choices):
            btn = QPushButton(c)
            btn.setCheckable(True)
            btn.setChecked(c == str(param_def["default"]))
            btn.setStyleSheet(
                _BTN_PERSO_STYLE if c == "personnalisé" else _BTN_CHOICE_STYLE
            )
            self._btns[c] = btn
            self._btn_group.addButton(btn)
            btn_grid.addWidget(btn, *divmod(i, 3))

        outer.addLayout(btn_grid)
        self._btn_group.buttonToggled.connect(self._on_btn_toggled)

        return lbl_row

    # ── Valeur courante ──────────────────────────────────────────────────────

    def value(self):
        if self._as_buttons:
            checked = self._btn_group.checkedButton() if self._btn_group else None
            return checked.text() if checked else self._def["default"]
        if self._type == "choice":
            return self._combo.currentText()
        if self._type == "bool":
            return self._checkbox.isChecked()
        if self._type == "int":
            return self._spin.value()
        return round(self._spin.value(), 6)

    def set_value(self, val):
        self._block = True
        try:
            if self._as_buttons:
                btn = self._btns.get(str(val))
                if btn:
                    btn.setChecked(True)
            elif self._type == "choice":
                self._combo.setCurrentText(str(val))
            elif self._type == "bool":
                self._checkbox.setChecked(bool(val))
            elif self._type == "int":
                self._spin.setValue(int(val))
                self._slider.setValue(int(val))
            else:
                self._spin.setValue(float(val))
                self._slider.setValue(round(float(val) * self._scale))
        finally:
            self._block = False

    # ── Mode batch : bouton propagation ──────────────────────────────────────

    def set_propagate_visible(self, visible: bool) -> None:
        """Active (hover-visible) ou désactive complètement le bouton propagation."""
        self._propagate_enabled = visible
        if visible:
            self._propagate_btn.setMaximumWidth(18)
        else:
            self._hide_timer.stop()
            self._propagate_btn.setVisible(False)
            self._propagate_btn.setMaximumWidth(0)

    def enterEvent(self, event) -> None:
        if self._propagate_enabled:
            self._hide_timer.stop()
            self._propagate_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._propagate_enabled:
            self._hide_timer.start()
        super().leaveEvent(event)

    def _check_hide_propagate(self) -> None:
        """Cache le bouton si la souris a réellement quitté le widget (et ses enfants)."""
        pos = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(pos):
            self._propagate_btn.setVisible(False)

    # ── Slots internes ──────────────────────────────────────────────────────

    def _on_btn_toggled(self, btn, checked: bool) -> None:
        if not self._block and checked:
            self.value_changed.emit(self._key, btn.text())

    def _on_slider_changed(self, int_val: int):
        if self._block:
            return
        self._block = True
        real_val = int_val / self._scale
        if self._type == "int":
            real_val = int(real_val)
        self._spin.setValue(real_val)
        self._block = False
        self.value_changed.emit(self._key, real_val)

    def _on_spin_changed_float(self, val: float):
        if self._block:
            return
        self._block = True
        self._slider.setValue(round(val * self._scale))
        self._block = False
        self.value_changed.emit(self._key, val)

    def _on_spin_changed_int(self, val: int):
        if self._block:
            return
        self._block = True
        self._slider.setValue(val)
        self._block = False
        self.value_changed.emit(self._key, val)

    def _on_combo_changed(self, text: str):
        if not self._block:
            self.value_changed.emit(self._key, text)

    def _on_checkbox_changed(self, state: int):
        if not self._block:
            self.value_changed.emit(self._key, bool(state))


def _precision(step: float) -> int:
    """Retourne le nombre de décimales nécessaires pour représenter step."""
    if step >= 1:
        return 0
    s = f"{step:.10f}".rstrip("0")
    return len(s.split(".")[-1]) if "." in s else 0
