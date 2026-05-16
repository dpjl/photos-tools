"""ui/param_widgets.py — Widgets paramétriques pour les étapes du pipeline."""

from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox,
    QSpinBox, QComboBox, QCheckBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal


class ParamRow(QWidget):
    """Ligne de paramètre : label | slider | spinbox (float/int) ou combobox (choice)."""

    value_changed = pyqtSignal(str, object)  # (key, value)

    def __init__(self, param_def: dict, parent=None):
        super().__init__(parent)
        self._def   = param_def
        self._key   = param_def["key"]
        self._type  = param_def["type"]
        self._block = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Label
        lbl = QLabel(param_def["label"])
        lbl.setFixedWidth(190)
        lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        layout.addWidget(lbl)

        if self._type == "choice":
            self._combo    = QComboBox()
            self._checkbox = None
            self._slider   = None
            self._spin     = None
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
            self._combo  = None
            self._slider = None
            self._spin   = None

        else:
            # Slider + spinbox
            mn    = param_def["min"]
            mx    = param_def["max"]
            step  = param_def["step"]
            dflt  = param_def["default"]
            prec  = _precision(step)
            scale = 10 ** prec

            self._slider = QSlider(Qt.Orientation.Horizontal)
            self._slider.setRange(round(mn * scale), round(mx * scale))
            self._slider.setValue(round(dflt * scale))
            self._slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._slider.valueChanged.connect(self._on_slider_changed)
            layout.addWidget(self._slider)

            if self._type == "int":
                self._spin = QSpinBox()
                self._spin.setRange(int(mn), int(mx))
                self._spin.setValue(int(dflt))
                self._spin.setSingleStep(int(step))
                self._spin.setFixedWidth(60)
                self._spin.valueChanged.connect(self._on_spin_changed_int)
            else:
                self._spin = QDoubleSpinBox()
                self._spin.setRange(mn, mx)
                self._spin.setDecimals(prec)
                self._spin.setSingleStep(step)
                self._spin.setValue(dflt)
                self._spin.setFixedWidth(70)
                self._spin.valueChanged.connect(self._on_spin_changed_float)
            layout.addWidget(self._spin)
            self._combo = None

        self._scale = 10 ** _precision(param_def.get("step", 1)) if self._type != "choice" else 1

    # ── Valeur courante ──────────────────────────────────────────────────────

    def value(self):
        if self._type == "choice":
            return self._combo.currentText()
        if self._type == "bool":
            return self._checkbox.isChecked()
        if self._type == "int":
            return self._spin.value()
        return round(self._spin.value(), 6)

    def set_value(self, val):
        self._block = True
        if self._type == "choice":
            self._combo.setCurrentText(str(val))
        elif self._type == "bool":
            self._checkbox.setChecked(bool(val))
        elif self._type == "int":
            self._spin.setValue(int(val))
            self._slider.setValue(int(val))
        else:
            self._spin.setValue(float(val))
            self._slider.setValue(round(float(val) * self._scale))
        self._block = False

    # ── Slots internes ──────────────────────────────────────────────────────

    def _on_slider_changed(self, int_val: int):
        if self._block:
            return
        self._block = True
        real_val = int_val / self._scale
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
