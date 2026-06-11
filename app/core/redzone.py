"""core/redzone.py — Détection et correction des zones rouges résiduelles.

Les photos argentiques détériorées présentent des voiles rosés/magenta
localisés (patch sur un mur, bande le long d'un bord, tache sur un tissu)
que les corrections couleur globales ne peuvent pas retirer sans casser
les couleurs légitimes.

Deux briques, partageant la même référence couleur :

· **Détection** (:func:`analyze_red_zones` + :func:`mask_from_analysis`) :
  pour chaque pixel du fond, on compare son chroma (a*, b*) au chroma
  « attendu », défini comme la médiane des pixels sains de même luminance
  (médiane par tranche de L*). Un écart fort orienté vers le rouge/magenta
  est un candidat. Volontairement haut rappel : l'affinage VLM
  (cf. core/vlm_refine.refine_redzones) et la revue manuelle apportent la
  précision.

· **Correction** (:func:`remove_red_cast`) : suppression du cast basse
  fréquence en chroma seulement. Le champ de cast = (chroma réel lissé
  edge-aware) − (chroma attendu par tranche de luminance) est soustrait
  des canaux a*/b* sous un masque sur-dilaté et fondu. La luminance, le
  grain et la texture sont conservés ; comme le champ tend vers zéro hors
  défaut, la sur-dilatation est sans risque et assure la transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ── Paramètres par défaut (validés sur les photos d'étude 1985-0042/0043) ─────

# Détection
CHROMA_MIN_DEFAULT = 12.0    # écart chromatique minimal au chroma attendu
MIN_AREA_DEFAULT   = 1500    # aire minimale d'une zone (px)
_ANGLE_LO          = -50.0   # secteur rouge/magenta du résidu (degrés)
_ANGLE_HI          = 70.0
_CLOSE_PX          = 12      # fermeture morphologique (combler les trous)
_OPEN_PX           = 4       # ouverture (retirer les miettes)

# Extension en bande : si une part suffisante d'un bord est couverte par des
# zones du masque, le défaut court probablement sur toute la longueur du bord
# (cas typique des bandes de détérioration) → on couvre la bande entière.
# Action explicite de l'UI (après revue VLM), pas appliquée à la détection :
# elle fusionnerait défauts et zones naturelles en une seule composante.
_BAND_FRAC_MIN   = 0.12      # fraction du bord couverte pour déclencher
_BAND_PROBE_PX   = 40        # profondeur de sonde le long du bord
_BAND_DEPTH_MIN  = 40        # profondeur minimale d'une bande (px)
_BAND_DEPTH_CAP  = 0.25      # profondeur max (fraction de la dimension)

# Référence par luminance (commune détection / correction)
_NBINS   = 12                # tranches de luminance
_SIGMA_R = 10.0              # similarité L* (unités L 0–255)

# Correction
DILATE_DEFAULT  = 60         # sur-dilatation du masque (px)
FEATHER_DEFAULT = 20         # fondu gaussien du masque (σ px)
_ACTUAL_SIGMA   = 12         # lissage spatial du chroma réel (px)
_SCALE          = 4          # facteur de sous-échantillonnage des calculs


# ══════════════════════════════════════════════════════════════════════════════
# Référence couleur par tranche de luminance
# ══════════════════════════════════════════════════════════════════════════════

def _bin_centers(guide: np.ndarray, nbins: int) -> tuple[np.ndarray, float]:
    g0, g1 = float(guide.min()), float(guide.max())
    if g1 - g0 < 1e-3:
        g1 = g0 + 1.0
    centers = np.linspace(g0, g1, nbins)
    return centers, (g1 - g0) / max(1, nbins - 1)


def _bin_medians(
    guide:   np.ndarray,
    values:  np.ndarray,
    sel:     np.ndarray,
    centers: np.ndarray,
    halfbin: float,
    min_px:  int = 50,
) -> np.ndarray:
    """Médiane de ``values`` par tranche de luminance, sur les pixels ``sel``.

    Les tranches sans assez de pixels sont interpolées depuis les voisines.
    """
    meds  = np.zeros(len(centers), np.float32)
    valid = np.zeros(len(centers), bool)
    for j, cj in enumerate(centers):
        s = sel & (np.abs(guide - cj) < halfbin)
        if s.sum() > min_px:
            meds[j] = np.median(values[s])
            valid[j] = True
    if 2 <= valid.sum() < len(centers):
        idx = np.arange(len(centers), dtype=np.float32)
        meds = np.interp(idx, idx[valid], meds[valid]).astype(np.float32)
    return meds


def _expected_channel(
    guide:   np.ndarray,
    meds:    np.ndarray,
    centers: np.ndarray,
    sigma_r: float,
) -> np.ndarray:
    """Valeur attendue par pixel : slicing lisse des médianes selon L*."""
    wr_sum = np.zeros_like(guide)
    out    = np.zeros_like(guide)
    for j, cj in enumerate(centers):
        wr = np.exp(-0.5 * ((guide - cj) / sigma_r) ** 2)
        wr_sum += wr
        out += wr * meds[j]
    return out / np.maximum(wr_sum, 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Détection
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RedZoneAnalysis:
    """Cartes pré-calculées (coûteuses) pour le seuillage live dans l'UI."""

    magnitude: np.ndarray   # float32 H×W — écart chromatique au chroma attendu
    angle:     np.ndarray   # float32 H×W — orientation du résidu (degrés)
    bg:        np.ndarray   # bool H×W — fond (hors personnes)


def background_mask(image_bgr: np.ndarray, session) -> np.ndarray:
    """Masque booléen du fond (personnes exclues via U2-Net rembg)."""
    from PIL import Image
    from rembg import remove
    pil_in = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    alpha  = np.array(remove(pil_in, session=session))[:, :, 3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(alpha, k) < 128


def analyze_red_zones(
    image_bgr: np.ndarray,
    bg:        np.ndarray | None = None,
    nbins:     int = _NBINS,
    sigma_r:   float = _SIGMA_R,
) -> RedZoneAnalysis:
    """Calcule l'écart chromatique de chaque pixel au chroma attendu.

    ``bg`` : masque booléen du fond (cf. :func:`background_mask`), ou None
    pour analyser toute l'image.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    if bg is None:
        bg = np.ones(L.shape, bool)

    centers, halfbin = _bin_centers(L, nbins)
    expected_a = _expected_channel(L, _bin_medians(L, a, bg, centers, halfbin),
                                   centers, sigma_r)
    expected_b = _expected_channel(L, _bin_medians(L, b, bg, centers, halfbin),
                                   centers, sigma_r)

    da, db = a - expected_a, b - expected_b
    return RedZoneAnalysis(
        magnitude=np.hypot(da, db).astype(np.float32),
        angle=np.degrees(np.arctan2(db, da)).astype(np.float32),
        bg=bg,
    )


def mask_from_analysis(
    analysis:   RedZoneAnalysis,
    chroma_min: float = CHROMA_MIN_DEFAULT,
    min_area:   int = MIN_AREA_DEFAULT,
) -> np.ndarray:
    """Seuillage + nettoyage morphologique → masque uint8 (255 = zone rouge).

    Rapide (pas de modèle) : appelé en live quand l'utilisateur règle les
    curseurs de sensibilité.
    """
    cand = (
        (analysis.magnitude >= chroma_min)
        & (analysis.angle >= _ANGLE_LO) & (analysis.angle <= _ANGLE_HI)
        & analysis.bg
    ).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _CLOSE_PX + 1,) * 2)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _OPEN_PX + 1,) * 2)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    out = np.zeros_like(cand)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _band_depth(covered: np.ndarray) -> int:
    """Profondeur de bande à remplir le long du bord HAUT de ``covered``.

    Retourne 0 si le bord n'est pas suffisamment touché. La profondeur est
    la médiane, par colonne touchée, de l'étendue du masque depuis le bord —
    mesurée dans une zone bornée pour qu'une bande perpendiculaire touchant
    le même coin ne fausse pas la statistique.
    """
    h, w = covered.shape
    cap   = max(_BAND_DEPTH_MIN, int(_BAND_DEPTH_CAP * h))
    probe = min(_BAND_PROBE_PX, h)

    hit = covered[:probe, :].any(axis=0)
    if hit.sum() < _BAND_FRAC_MIN * w:
        return 0

    strip = covered[:cap, :]
    depths = cap - np.argmax(strip[::-1, :], axis=0)   # dernière ligne couverte
    depths = depths[hit & strip.any(axis=0)]
    if len(depths) == 0:
        return 0
    return min(max(int(np.median(depths)), _BAND_DEPTH_MIN), cap)


def extend_border_bands(mask: np.ndarray) -> np.ndarray:
    """Étend le masque en bande le long des bords largement touchés.

    Les bandes de détérioration courent souvent sur toute la longueur d'un
    bord mais ne dépassent le seuil de détection que par endroits : si une
    fraction suffisante d'un bord est couverte, on remplit la bande entière
    sur une profondeur représentative des zones détectées.
    """
    h, w = mask.shape[:2]
    out = mask.copy()
    covered = mask > 0

    d = _band_depth(covered)                  # haut
    if d:
        out[:d, :] = 255
    d = _band_depth(covered[::-1, :])         # bas
    if d:
        out[h - d:, :] = 255
    d = _band_depth(covered.T)                # gauche
    if d:
        out[:, :d] = 255
    d = _band_depth(covered[:, ::-1].T)       # droite
    if d:
        out[:, w - d:] = 255

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Correction
# ══════════════════════════════════════════════════════════════════════════════

def remove_red_cast(
    image_bgr:     np.ndarray,
    mask:          np.ndarray,
    person_alpha:  np.ndarray | None = None,
    strength:      float = 1.0,
    luma_strength: float = 0.0,
    dilate_px:     int = DILATE_DEFAULT,
    feather_sigma: float = FEATHER_DEFAULT,
    nbins:         int = _NBINS,
    sigma_r:       float = _SIGMA_R,
    actual_sigma:  float = _ACTUAL_SIGMA,
    scale:         int = _SCALE,
) -> np.ndarray:
    """Supprime le cast couleur basse fréquence dans les zones masquées.

    Pour chaque canal corrigé :
      attendu = médiane des pixels sains par tranche de luminance ;
      réel    = lissage spatial edge-aware (bilatéral guidé par L*) ;
      sortie  = canal − masque_doux × (réel − attendu) × force.

    ``person_alpha`` : alpha 0–1 des personnes (rembg) — elles ne sont
    jamais modifiées, et leurs pixels sont exclus du calcul de l'attendu.
    """
    h, w = image_bgr.shape[:2]
    mask_bin = (mask > 127).astype(np.uint8)
    if not mask_bin.any():
        return image_bgr.copy()

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_soft = (
        1.0 - np.clip(person_alpha.astype(np.float32), 0.0, 1.0)
        if person_alpha is not None
        else np.ones((h, w), np.float32)
    )

    # Masque doux : plein au centre, fondu aux bords, jamais sur les personnes
    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2
        )
        mask_d = cv2.dilate(mask_bin, k)
    else:
        mask_d = mask_bin
    soft = cv2.GaussianBlur(mask_d.astype(np.float32), (0, 0), max(1.0, feather_sigma))
    soft = np.clip(soft, 0.0, 1.0) * cv2.GaussianBlur(bg_soft, (0, 0), 4)

    # Calculs basse fréquence en résolution réduite
    scale = max(1, min(scale, min(h, w) // 64 or 1))
    sw, sh = max(1, w // scale), max(1, h // scale)
    labs   = cv2.resize(lab, (sw, sh), interpolation=cv2.INTER_AREA)
    guide  = labs[:, :, 0]
    mask_s = cv2.resize(mask_d, (sw, sh), interpolation=cv2.INTER_NEAREST)
    mask_s = cv2.dilate(mask_s, np.ones((5, 5), np.uint8))
    bg_s   = cv2.resize(bg_soft, (sw, sh), interpolation=cv2.INTER_AREA)
    clean  = (mask_s == 0) & (bg_s > 0.5)

    centers, halfbin = _bin_centers(guide, nbins)
    strengths = (luma_strength, strength, strength)

    out = lab.copy()
    for c, s in enumerate(strengths):
        if s <= 0:
            continue
        chs = labs[:, :, c]

        meds = _bin_medians(guide, chs, clean, centers, halfbin)
        expected_s = _expected_channel(guide, meds, centers, sigma_r)

        # Réel lissé edge-aware : moyenne bilatérale guidée par L* (le champ
        # de cast ne traverse pas les frontières entre surfaces).
        num = np.zeros_like(chs)
        den = np.zeros_like(chs)
        sigma_sp = max(1.0, actual_sigma / scale)
        for cj in centers:
            wr = np.exp(-0.5 * ((guide - cj) / sigma_r) ** 2)
            num += wr * cv2.GaussianBlur(chs * wr, (0, 0), sigma_sp)
            den += wr * cv2.GaussianBlur(wr, (0, 0), sigma_sp)
        actual_s = num / np.maximum(den, 1e-6)

        expected = cv2.resize(expected_s, (w, h), interpolation=cv2.INTER_LINEAR)
        actual   = cv2.resize(actual_s, (w, h), interpolation=cv2.INTER_LINEAR)
        out[:, :, c] = lab[:, :, c] - (actual - expected) * s

    out = np.clip(out, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    # Fondu appliqué en BGR contre l'original : les pixels hors masque restent
    # STRICTEMENT identiques (pas d'aller-retour LAB↔BGR qui décale de ±1).
    soft3 = soft[:, :, None]
    result = (image_bgr.astype(np.float32) * (1.0 - soft3)
              + out_bgr.astype(np.float32) * soft3)
    return np.clip(result, 0, 255).astype(np.uint8)
