"""core/artifact_detect.py — Détection automatique d'artefacts pour le masque.

Génère un masque (uint8, 255 = à retoucher) des défauts physiques les plus
grossiers d'une photo scannée, destiné à alimenter l'inpainting :

  · **Rayures, traits, plis, craquelures, poussières** → réseau de neurones
    dédié de Microsoft « Bringing Old Photos Back to Life » (U-Net entraîné
    spécifiquement sur ce type de défaut). C'est ce qui donne la précision :
    le réseau distingue une vraie rayure d'une texture (tissu, grille…) là où
    les filtres classiques (crêtes / morphologie) confondent les deux.

  · **Points colorés étrangers** (taches d'encre, moisissure…) → détecteur
    chromatique classique *conservateur* : petite tache compacte dont la
    couleur s'écarte fortement de son voisinage local, à forte saturation.

Le détecteur neuronal est un singleton à chargement paresseux (cf.
``LamaInpainter``) ; les poids sont téléchargés à la première utilisation.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from config import (
    BOPBTL_DETECTION_DIR,
    BOPBTL_DETECTION_URL,
    BOPBTL_DETECTION_WEIGHTS,
)

# ── Paramètres par défaut ─────────────────────────────────────────────────────

# Inférence neuronale
_SCRATCH_PROB_THRESH = 0.5          # seuil sur la probabilité de rayure
_SCRATCH_MAX_SIDE    = 1600         # côté max pour l'inférence (précision vs mémoire)
_SCRATCH_SCALES      = (1600, 640)  # fusion multi-échelle (haute=précision, basse=continuité)
_SCRATCH_DILATE      = 1            # dilatation du masque (px) pour couvrir le bord

# Points colorés étrangers
_SPOT_MIN_AREA   = 6
_SPOT_MAX_AREA   = 600
_SPOT_SAT_MIN    = 90      # saturation HSV minimale
_SPOT_CHROMA_DEV = 40.0    # écart chromatique local (Lab) minimal
_SPOT_MIN_EXTENT = 0.45    # compacité minimale (aire / aire boîte)


# ══════════════════════════════════════════════════════════════════════════════
# Détecteur neuronal de rayures (BOPBTL)
# ══════════════════════════════════════════════════════════════════════════════

class ScratchDetector:
    """Détection de rayures / plis par U-Net — singleton, chargement paresseux."""

    _instance: "ScratchDetector | None" = None

    @classmethod
    def get(cls) -> "ScratchDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        import torch
        self._torch = torch
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch = self._torch

        if not os.path.exists(BOPBTL_DETECTION_WEIGHTS):
            from torch.hub import download_url_to_file
            os.makedirs(BOPBTL_DETECTION_DIR, exist_ok=True)
            download_url_to_file(
                BOPBTL_DETECTION_URL, BOPBTL_DETECTION_WEIGHTS, progress=True,
            )

        # Le code réseau vit à côté des poids (même convention que SCUNet).
        if BOPBTL_DETECTION_DIR not in sys.path:
            sys.path.insert(0, BOPBTL_DETECTION_DIR)
        from bopbtl_networks import UNet

        model = UNet(in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6)
        checkpoint = torch.load(
            BOPBTL_DETECTION_WEIGHTS, map_location="cpu", weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        self._model = model.to(self._device).eval()

    def detect_prob(
        self,
        image_bgr: np.ndarray,
        max_side: int = _SCRATCH_MAX_SIDE,
    ) -> np.ndarray:
        """Carte de probabilité de rayure (float32 [0,1], taille de l'image).

        C'est la sortie « brute » du réseau : le seuillage et la dilatation se
        font ensuite à moindre coût (cf. :func:`mask_from_prob`), ce qui permet
        de régler le seuil en direct sans relancer l'inférence.
        """
        torch = self._torch
        self._ensure_loaded()

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Échelle d'inférence : pleine résolution (bornée), arrondie au multiple
        # de 16 exigé par le réseau.
        scale = min(1.0, max_side / max(h, w))
        ih = _round16(h * scale)
        iw = _round16(w * scale)

        t = (torch.from_numpy(gray).float() / 255.0 - 0.5) / 0.5
        t = t[None, None]
        t = torch.nn.functional.interpolate(
            t, size=(ih, iw), mode="bilinear", align_corners=False,
        ).to(self._device)

        with torch.no_grad():
            prob = torch.sigmoid(self._model(t))
        prob = torch.nn.functional.interpolate(
            prob.cpu(), size=(h, w), mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        return prob.astype(np.float32)

    def detect_prob_scales(
        self,
        image_bgr: np.ndarray,
        scales: "tuple[int, ...]" = _SCRATCH_SCALES,
    ) -> "list[np.ndarray]":
        """Carte de probabilité par résolution (chacune ramenée à la taille image)."""
        return [self.detect_prob(image_bgr, ms) for ms in scales]

    def detect_prob_multiscale(
        self,
        image_bgr: np.ndarray,
        scales: "tuple[int, ...]" = _SCRATCH_SCALES,
        consensus: bool = False,
    ) -> np.ndarray:
        """Carte de probabilité fusionnée sur plusieurs résolutions.

        Une rayure longue mais faible se fragmente à haute résolution (elle
        passe sous le seuil par endroits) alors qu'elle ressort en continu à
        basse résolution.  La fusion multi-échelle combine une échelle haute
        (précision/détails fins) et une échelle basse (continuité) :

          · **max** (défaut) : une détection à *n'importe quelle* échelle suffit
            → couverture maximale ;
          · **consensus** (moyenne) : exige un *accord* entre échelles → retire
            les faux positifs spécifiques à une seule échelle (arêtes d'objets,
            motifs) tout en gardant les vraies rayures (robustes inter-échelles).
        """
        return fuse_probs(self.detect_prob_scales(image_bgr, scales), consensus)

    def detect(
        self,
        image_bgr: np.ndarray,
        threshold: float = _SCRATCH_PROB_THRESH,
        scales: "tuple[int, ...]" = _SCRATCH_SCALES,
    ) -> np.ndarray:
        """Retourne un masque uint8 (255 = rayure) à la taille de ``image_bgr``."""
        return mask_from_prob(self.detect_prob_multiscale(image_bgr, scales), threshold)


def fuse_probs(probs: "list[np.ndarray]", consensus: bool = False) -> "np.ndarray | None":
    """Fusionne des cartes de proba multi-échelle : max (couverture) ou moyenne (consensus)."""
    if not probs:
        return None
    if len(probs) == 1:
        return probs[0]
    stack = np.stack(probs, axis=0)
    return stack.mean(axis=0) if consensus else stack.max(axis=0)


def mask_from_prob(prob: np.ndarray, threshold: float = _SCRATCH_PROB_THRESH) -> np.ndarray:
    """Binarise une carte de probabilité en masque uint8 (255 = rayure)."""
    return (prob >= threshold).astype(np.uint8) * 255


def dilate_mask(mask: np.ndarray, dilate: int) -> np.ndarray:
    """Dilate un masque de ``dilate`` pixels (no-op si dilate ≤ 0 ou masque vide)."""
    if dilate <= 0 or not mask.any():
        return mask
    k = 2 * dilate + 1
    return cv2.dilate(mask, np.ones((k, k), np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# Détecteur classique de points colorés étrangers
# ══════════════════════════════════════════════════════════════════════════════

def detect_color_spots(
    image_bgr: np.ndarray,
    min_area: int = _SPOT_MIN_AREA,
    max_area: int = _SPOT_MAX_AREA,
    sat_min: int = _SPOT_SAT_MIN,
    chroma_dev: float = _SPOT_CHROMA_DEV,
) -> np.ndarray:
    """Masque des petites taches colorées s'écartant fortement du voisinage.

    Conservateur par construction (forte saturation + fort écart chromatique
    local + petite taille + compacité) afin de ne pas marquer des détails
    légitimes de l'image.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    _, a, b = cv2.split(lab)

    # Couleur locale « attendue » = moyenne des canaux a/b sur une grande fenêtre.
    win = 31
    a_loc = cv2.blur(a, (win, win))
    b_loc = cv2.blur(b, (win, win))
    deviation = np.sqrt((a - a_loc) ** 2 + (b - b_loc) ** 2)

    sat = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    candidate = ((deviation > chroma_dev) & (sat > sat_min)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    out = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if not (min_area <= area <= max_area):
            continue
        if area / max(1, bw * bh) < _SPOT_MIN_EXTENT:   # compacité
            continue
        if max(bw, bh) > 3 * min(bw, bh):               # rejette les traînées
            continue
        out[labels == i] = 255
    return out


# ══════════════════════════════════════════════════════════════════════════════
# API publique
# ══════════════════════════════════════════════════════════════════════════════

def detect_artifacts(
    image_bgr: np.ndarray,
    detect_scratches: bool = True,
    detect_spots: bool = True,
    threshold: float = _SCRATCH_PROB_THRESH,
    dilate: int = _SCRATCH_DILATE,
    scales: "tuple[int, ...]" = _SCRATCH_SCALES,
) -> np.ndarray:
    """Masque combiné des artefacts détectés (uint8, 255 = à retoucher).

    image_bgr : H x W x 3 uint8 BGR
    Retourne un masque H x W ; vide (tout à 0) si rien n'est détecté.
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if detect_scratches:
        mask = cv2.bitwise_or(mask, ScratchDetector.get().detect(image_bgr, threshold, scales))
    if detect_spots:
        mask = cv2.bitwise_or(mask, detect_color_spots(image_bgr))

    return dilate_mask(mask, dilate)


def _round16(x: float) -> int:
    """Arrondi au multiple de 16 le plus proche (≥ 16), exigé par le U-Net."""
    return max(16, int(round(x / 16.0)) * 16)
