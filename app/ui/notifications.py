"""ui/notifications.py — Système de toasts (notifications flottantes).

Les toasts sont des **fenêtres top-level sans cadre** (Qt.Tool), indépendantes
de la hiérarchie de widgets de l'application. Elles ne peuvent donc jamais
être enterrées par des repaints de l'UI principale.

Usage ::

    mgr = NotificationManager(main_window)
    mgr.notify("Titre", "Corps", level=Level.SUCCESS)
    # Toast mis à jour si la clé est encore visible :
    mgr.notify("Batch en cours", "3/10", key="batch")
    mgr.notify("Batch en cours", "4/10", key="batch")  # → update in-place
    mgr.dismiss_key("batch")                           # → fermeture immédiate
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from PyQt6.QtWidgets import QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer, QObject, QEvent


# ══════════════════════════════════════════════════════════════════════════════
# Constantes
# ══════════════════════════════════════════════════════════════════════════════

class Level(str, Enum):
    INFO    = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR   = "error"


# (accent, icône, fond, bordure)
_LEVEL_STYLE: dict[Level, tuple[str, str, str, str]] = {
    Level.INFO:    ("#4a9de8", "ℹ",  "#111830", "#2a4a7a"),
    Level.SUCCESS: ("#27c96e", "✓",  "#0a1f14", "#1e5a38"),
    Level.WARNING: ("#f5a623", "⚠",  "#1e1600", "#5a3e00"),
    Level.ERROR:   ("#e84040", "✗",  "#1e0808", "#5a1a1a"),
}

_TOAST_W    = 320   # largeur fixe des toasts
_MARGIN     = 16    # marge vis-à-vis du coin de la fenêtre
_SPACING    = 8     # espace vertical entre toasts
_MAX_TOASTS = 6     # toasts simultanés maximum


# ══════════════════════════════════════════════════════════════════════════════
# Widget individuel — fenêtre top-level indépendante
# ══════════════════════════════════════════════════════════════════════════════

class ToastWidget(QFrame):
    """Notification flottante indépendante de la hiérarchie de widgets."""

    def __init__(
        self,
        title:    str,
        body:     Optional[str],
        level:    Level,
        duration: int,
    ) -> None:
        # Pas de parent Qt → fenêtre indépendante
        super().__init__(None)
        accent, icon, bg, border = _LEVEL_STYLE[level]
        self._duration = duration

        # ── Flags fenêtre ──────────────────────────────────────────────────
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        # Ne pas voler le focus ni apparaître dans la barre des tâches
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self.setFixedWidth(_TOAST_W)
        self.setObjectName("ToastFrame")
        self.setStyleSheet(
            f"QFrame#ToastFrame {{"
            f"  background:{bg};"
            f"  border-radius:6px;"
            f"  border:2px solid {border};"
            f"  border-left:none;"
            f"}}"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(0)

        # Barre accent gauche
        bar = QFrame()
        bar.setFixedWidth(5)
        bar.setStyleSheet(f"background:{accent}; border-radius:4px 0 0 4px;")
        lay.addWidget(bar)

        # Icône
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(
            f"color:{accent}; font-size:16px; border:none; background:transparent;"
            f" padding:10px 6px 10px 10px;"
        )
        icon_lbl.setFixedWidth(32)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(icon_lbl)

        # Zone texte
        text_w = QWidget()
        text_w.setStyleSheet("background:transparent; border:none;")
        text_lay = QVBoxLayout(text_w)
        text_lay.setContentsMargins(0, 10, 0, 10)
        text_lay.setSpacing(3)

        self._title_lbl = QLabel(title)
        self._title_lbl.setWordWrap(True)
        self._title_lbl.setMaximumWidth(_TOAST_W - 72)
        self._title_lbl.setStyleSheet(
            "color:#e8e8f0; font-size:12px; font-weight:700;"
            " border:none; background:transparent;"
        )
        text_lay.addWidget(self._title_lbl)

        self._body_lbl = QLabel(body or "")
        self._body_lbl.setWordWrap(True)
        self._body_lbl.setMaximumWidth(_TOAST_W - 72)
        self._body_lbl.setStyleSheet(
            "color:#9ab0c8; font-size:10px; border:none; background:transparent;"
        )
        self._body_lbl.setVisible(bool(body))
        text_lay.addWidget(self._body_lbl)

        lay.addWidget(text_w, stretch=1)

        # Bouton fermer
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QPushButton { background:transparent; color:#556; border:none;"
            "  font-size:11px; padding:0; }"
            "QPushButton:hover { color:#bcd; }"
        )
        close_btn.setToolTip("Fermer")
        close_btn.clicked.connect(self._dismiss)
        lay.addWidget(close_btn)

        self.adjustSize()

        # Timer d'auto-fermeture
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(duration)
        self._timer.timeout.connect(self._dismiss)
        self._timer.start()

    # ── API publique ────────────────────────────────────────────────────────

    def update_content(self, title: str, body: Optional[str]) -> None:
        """Met à jour titre/corps et relance le timer."""
        self._title_lbl.setText(title)
        if body:
            self._body_lbl.setText(body)
            self._body_lbl.setVisible(True)
        else:
            self._body_lbl.setVisible(False)
        self.adjustSize()
        self._timer.stop()
        self._timer.start(self._duration)

    # ── Gestion ────────────────────────────────────────────────────────────

    def _dismiss(self) -> None:
        self._timer.stop()
        self.close()

    # Pause au survol
    def enterEvent(self, event) -> None:
        self._timer.stop()

    def leaveEvent(self, event) -> None:
        self._timer.start(max(2000, self._duration // 2))


# ══════════════════════════════════════════════════════════════════════════════
# Gestionnaire
# ══════════════════════════════════════════════════════════════════════════════

class NotificationManager(QObject):
    """Gère une pile de toasts ancrée en bas à droite de la fenêtre principale.

    Les toasts sont des fenêtres indépendantes (Qt.Tool) : ils ne peuvent
    jamais être masqués par des opérations UI internes à l'application.

    Paramètres clés :
        anchor  — QWidget dont le coin bas-droit sert de référence de position.
                  Habituellement le QMainWindow principal.

    Usage ::

        mgr = NotificationManager(main_window)
        mgr.notify("Titre", "Corps", level=Level.SUCCESS, duration=4000)
        mgr.notify("En cours", "3/10", key="batch", duration=5000)
        mgr.notify("En cours", "4/10", key="batch")   # update in-place
        mgr.dismiss_key("batch")                       # fermeture immédiate
    """

    def __init__(self, anchor: QWidget) -> None:
        super().__init__(anchor)
        self._anchor  = anchor
        self._toasts: list[ToastWidget]      = []
        self._keyed:  dict[str, ToastWidget] = {}
        # Repositionner automatiquement lors des redimensionnements/déplacements
        anchor.installEventFilter(self)

    # ── API publique ────────────────────────────────────────────────────────

    def notify(
        self,
        title:    str,
        body:     Optional[str] = None,
        *,
        level:    Level        = Level.INFO,
        duration: int          = 4000,
        key:      Optional[str] = None,
    ) -> None:
        """Affiche un nouveau toast ou met à jour un toast existant (si ``key`` fourni)."""
        # ── Mise à jour in-place si la clé est active ──────────────────────
        if key and key in self._keyed:
            t = self._keyed[key]
            try:
                if not t.isHidden() and t.isVisible():
                    t.update_content(title, body)
                    return
            except RuntimeError:
                pass
            del self._keyed[key]

        # ── Éviction des plus anciens (sans clé sticky) ────────────────────
        non_sticky = [t for t in self._toasts if t not in self._keyed.values()]
        while len(self._toasts) >= _MAX_TOASTS and non_sticky:
            oldest = non_sticky.pop(0)
            if oldest in self._toasts:
                self._toasts.remove(oldest)
            try:
                oldest._dismiss()
            except RuntimeError:
                pass

        # ── Créer le nouveau toast ─────────────────────────────────────────
        toast = ToastWidget(title, body, level, duration)
        toast.destroyed.connect(lambda obj=None, t=toast: self._on_destroyed(t))
        self._toasts.append(toast)
        if key:
            self._keyed[key] = toast

        self._reposition()
        toast.show()

    def dismiss_key(self, key: str) -> None:
        """Ferme immédiatement le toast identifié par ``key``."""
        if key in self._keyed:
            t = self._keyed.pop(key)
            try:
                t._dismiss()
            except RuntimeError:
                pass

    def reposition(self) -> None:
        """Repositionne manuellement tous les toasts (utile après un redim. explicite)."""
        self._reposition()

    # ── Event filter sur l'ancre ────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Move):
            self._reposition()
        elif event.type() in (QEvent.Type.Hide, QEvent.Type.Close):
            # Cacher tous les toasts quand la fenêtre principale disparaît
            for t in self._toasts:
                try:
                    t.hide()
                except RuntimeError:
                    pass
        elif event.type() == QEvent.Type.Show:
            for t in self._toasts:
                try:
                    if not t.isHidden():
                        t.show()
                except RuntimeError:
                    pass
            self._reposition()
        return False  # ne pas consommer l'événement

    # ── Interne ─────────────────────────────────────────────────────────────

    def _on_destroyed(self, toast: ToastWidget) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
        dead_keys = [k for k, t in self._keyed.items() if t is toast]
        for k in dead_keys:
            del self._keyed[k]
        try:
            self._reposition()
        except RuntimeError:
            pass

    def _reposition(self) -> None:
        """Recalcule les positions écran de tous les toasts visibles."""
        visible = [t for t in self._toasts if not t.isHidden()]
        if not visible:
            return
        try:
            # Coin bas-droit de l'ancre en coordonnées écran
            br = self._anchor.mapToGlobal(self._anchor.rect().bottomRight())
            x_right  = br.x() - _MARGIN
            y_bottom = br.y() - _MARGIN
        except RuntimeError:
            return

        y = y_bottom
        for toast in reversed(visible):
            try:
                y -= toast.height() + _SPACING
                toast.move(x_right - _TOAST_W, max(0, y))
            except RuntimeError:
                pass
