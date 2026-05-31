"""core/rembg_session.py — Session U2-Net partagée pour rembg.

Évite de charger le modèle U2-Net (~170 Mo) deux fois en mémoire
lorsque les étapes LightLeak et Cast argentique sont toutes deux activées.
"""

from __future__ import annotations

_shared_session = None


def get_shared_session():
    """Retourne la session U2-Net partagée (chargée paresseusement)."""
    global _shared_session
    if _shared_session is None:
        from rembg import new_session
        _shared_session = new_session("u2net")
    return _shared_session
