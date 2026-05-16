#!/usr/bin/env python3
"""
restore_core.py — Fonctions de traitement photo, importables sans effets de bord.

Ce module est partagé entre :
  - restore_simple.py  (pipeline CLI)
  - restore_gui.py     (interface graphique PyQt6)

Aucun code exécutable au niveau module (pas de lecture de sys.argv, pas
de chargement de modèles à l'import).
"""

import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Chemins des modèles IA — dans le même répertoire que ce fichier
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GFPGAN_MODEL_PATH = os.path.join(_SCRIPT_DIR, "GFPGANv1.4.pth")
SCUNET_MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")
SCUNET_MODEL      = "gan"   # 'gan' (perceptuel, défaut) ou 'psnr' (conservateur)


# ===========================================================================
# CORRECTION DES BANDES COLORÉES — MASQUE SPATIAL (méthode historique)
# ===========================================================================

def correct_color_bands(img, strength=1.0, lum_strength=0.0, small_w=60,
                        ref_margin=0.22,
                        right_start=0.50, right_mid=0.78, right_end=0.95,
                        top_end=0.25, left_end=0.0,
                        lum_right_start=0.82, lum_top_end=0.20):
    """Supprime les bandes colorées dues au vieillissement du film argentique.

    Masque spatial à deux étages pour le bord droit :
      - x de right_start à right_mid : correction douce (0 → 50 %)
      - x de right_mid   à right_end : correction forte (50 → 100 %)
    Le centre de l'image sert de référence chromatique et reste inchangé.
    """
    if strength <= 0 and lum_strength <= 0:
        return img

    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    sh = max(4, int(small_w * h / w))

    a_s = cv2.resize(a, (small_w, sh), interpolation=cv2.INTER_AREA)
    b_s = cv2.resize(b, (small_w, sh), interpolation=cv2.INTER_AREA)
    L_s = cv2.resize(L, (small_w, sh), interpolation=cv2.INTER_AREA)

    cy1 = max(1, int(sh * ref_margin));      cy2 = min(sh-1, int(sh * (1-ref_margin)))
    cx1 = max(1, int(small_w * ref_margin)); cx2 = min(small_w-1, int(small_w * (1-ref_margin)))
    a_ref = float(np.median(a_s[cy1:cy2, cx1:cx2]))
    b_ref = float(np.median(b_s[cy1:cy2, cx1:cx2]))
    L_ref = float(np.median(L_s[cy1:cy2, cx1:cx2]))

    xs = np.linspace(0, 1, small_w)
    ys = np.linspace(0, 1, sh)
    XX, YY = np.meshgrid(xs, ys)

    m_gentle = np.clip((XX - right_start) / (right_mid - right_start + 1e-6), 0, 1) * 0.5
    m_strong = np.clip((XX - right_mid)   / (right_end  - right_mid  + 1e-6), 0, 1) * 0.5 + 0.5
    mc = np.where(XX < right_mid, m_gentle, np.maximum(m_gentle, m_strong))
    mc = np.clip(mc, 0, 1)

    if top_end > 0:
        mc = np.maximum(mc, np.clip((top_end - YY) / top_end, 0, 1))
    if left_end > 0:
        mc = np.maximum(mc, np.clip((left_end - XX) / left_end, 0, 1))

    a_corr_s = (a_ref - a_s) * strength * mc
    b_corr_s = (b_ref - b_s) * strength * mc

    ml_r = np.clip((XX - lum_right_start) / (1.0 - lum_right_start + 1e-6), 0, 1)
    ml_t = np.clip((lum_top_end - YY) / (lum_top_end + 1e-6), 0, 1)
    L_excess = np.maximum(L_s - L_ref, 0)
    L_corr_s = -L_excess * np.maximum(ml_r, ml_t * 1.15) * lum_strength

    k = max(3, w // (small_w * 2)) * 2 + 1

    def _up(m):
        return cv2.GaussianBlur(cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC), (k, k), k // 3)

    return cv2.cvtColor(
        cv2.merge([
            np.clip(L + _up(L_corr_s), 0, 255).astype(np.uint8),
            np.clip(a + _up(a_corr_s), 0, 255).astype(np.uint8),
            np.clip(b + _up(b_corr_s), 0, 255).astype(np.uint8),
        ]),
        cv2.COLOR_LAB2BGR,
    )


# ===========================================================================
# CORRECTION DES BANDES COLORÉES — SEGMENTATION U2-Net (rembg)
# ===========================================================================

def correct_color_rembg(img, strength=1.3, sigma_fill=120,
                        ref_x1=0.03, ref_x2=0.20,
                        fg_blur_k=21, fg_blur_sigma=7,
                        strength_a=1.3, strength_b=0.4):
    """Correction des bandes colorées avec segmentation IA (U2-Net via rembg).

    - strength_a  : intensité canal a* (rouge)
    - strength_b  : intensité canal b* (jaune)
    - sigma_fill  : sigma du flou gaussien de diffusion du champ de cast (px)
    - ref_x1/x2   : fraction de largeur de la zone de référence gauche
    - fg_blur_k/sigma : lissage du masque de premier plan
    """
    try:
        from rembg import remove as _rembg_remove, new_session as _rembg_session
        from PIL import Image as _PILImage
    except ImportError:
        print("  [rembg non installé] fallback masque spatial")
        return correct_color_bands(img, strength=min(strength, 1.0))

    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    # ── 1. Masque foreground (U2-Net) ──────────────────────────────────────
    pil = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    session = _rembg_session("u2net")
    fg_pil = _rembg_remove(pil, session=session, only_mask=True,
                           alpha_matting=True,
                           alpha_matting_foreground_threshold=240,
                           alpha_matting_background_threshold=10,
                           alpha_matting_erode_size=10)
    fg_mask = np.array(fg_pil).astype(np.float32) / 255.0
    bg_mask = fg_mask < 0.3

    # ── 2. Référence : fond gauche (zone non dégradée) ─────────────────────
    rx1, rx2 = int(w * ref_x1), int(w * ref_x2)
    ref_zone_bg = bg_mask[int(h * .10):int(h * .90), rx1:rx2]
    ref_a = np.array(a[int(h * .10):int(h * .90), rx1:rx2])
    ref_b = np.array(b[int(h * .10):int(h * .90), rx1:rx2])
    a_ref = float(np.median(ref_a[ref_zone_bg]))

    # b* : référence PAR LIGNE
    from scipy.ndimage import gaussian_filter1d as _gf1d
    b_ref_row = np.full(h, np.nan)
    for y in range(h):
        left_bg = bg_mask[y, rx1:rx2]
        if left_bg.sum() >= 3:
            b_ref_row[y] = np.median(b[y, rx1:rx2][left_bg])
    valid_rows = ~np.isnan(b_ref_row)
    if valid_rows.sum() > 2:
        b_ref_row = np.interp(np.arange(h),
                              np.where(valid_rows)[0],
                              b_ref_row[valid_rows])
        b_ref_row = _gf1d(b_ref_row.astype(np.float32), sigma=20)
    else:
        b_ref_row[:] = float(np.median(ref_b[ref_zone_bg]))

    # ── 3. Champ de cast 2D estimé par diffusion ───────────────────────────
    bg_y, bg_x = np.where(bg_mask)
    a_dev = np.zeros((h, w), np.float32)
    b_dev = np.zeros((h, w), np.float32)
    va    = np.zeros((h, w), np.float32)
    vb    = np.zeros((h, w), np.float32)
    a_dev[bg_y, bg_x] = a[bg_y, bg_x] - a_ref
    b_ref_2d = np.tile(b_ref_row[:, np.newaxis], (1, w))
    b_dev[bg_y, bg_x] = b[bg_y, bg_x] - b_ref_2d[bg_y, bg_x]
    va[bg_y, bg_x] = 1.0
    vb[bg_y, bg_x] = 1.0

    eps = 1e-6
    a_cast = cv2.GaussianBlur(a_dev * va, (0, 0), sigma_fill) / \
             (cv2.GaussianBlur(va, (0, 0), sigma_fill) + eps)
    b_cast = cv2.GaussianBlur(b_dev * vb, (0, 0), sigma_fill) / \
             (cv2.GaussianBlur(vb, (0, 0), sigma_fill) + eps)

    a_field = np.minimum(0.0, -a_cast) * strength_a
    b_field = np.minimum(0.0, -b_cast) * strength_b

    # ── 4. Application fond seulement ─────────────────────────────────────
    a_new = np.clip(a + a_field, 0, 255).astype(np.uint8)
    b_new = np.clip(b + b_field, 0, 255).astype(np.uint8)
    img_corrected = cv2.cvtColor(
        cv2.merge([np.clip(L, 0, 255).astype(np.uint8), a_new, b_new]),
        cv2.COLOR_LAB2BGR)

    fg_soft = cv2.GaussianBlur(fg_mask, (fg_blur_k, fg_blur_k), fg_blur_sigma)
    fg_3d = np.stack([fg_soft] * 3, axis=-1)
    result = (img.astype(np.float32) * fg_3d +
              img_corrected.astype(np.float32) * (1.0 - fg_3d))
    return np.clip(result, 0, 255).astype(np.uint8)


# ===========================================================================
# CORRECTION COULEUR CLASSIQUE
# ===========================================================================

def stretch_histogram(img_bgr, low_pct=1, high_pct=99, roi_inset=0.06):
    """Étirement de l'histogramme par canal pour supprimer la dominante orange."""
    img_f = img_bgr.astype(np.float32)
    h, w  = img_f.shape[:2]
    iy1, iy2 = int(h * roi_inset), int(h * (1 - roi_inset))
    ix1, ix2 = int(w * roi_inset), int(w * (1 - roi_inset))
    roi = img_f[iy1:iy2, ix1:ix2]
    out = np.zeros_like(img_f)
    for i in range(3):
        lo = np.percentile(roi[:, :, i], low_pct)
        hi = np.percentile(roi[:, :, i], high_pct)
        if hi > lo:
            out[:, :, i] = np.clip((img_f[:, :, i] - lo) / (hi - lo) * 255, 0, 255)
        else:
            out[:, :, i] = img_f[:, :, i]
    return out.astype(np.uint8)


def apply_clahe(img_bgr, clip_limit=2.5, tile_size=(8, 8)):
    """CLAHE sur le canal L en espace LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def boost_saturation(img_bgr, factor=1.35):
    """Amplification de la saturation en espace LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([
        l,
        np.clip(128 + (a - 128) * factor, 0, 255),
        np.clip(128 + (b - 128) * factor, 0, 255),
    ]).astype(np.uint8), cv2.COLOR_LAB2BGR)


def color_correction(img_bgr):
    """Pipeline couleur complet : histogramme → CLAHE → saturation."""
    return boost_saturation(apply_clahe(stretch_histogram(img_bgr)))


# ===========================================================================
# SCUNET — DÉBRUITAGE INTELLIGENT
# ===========================================================================

def _download_scunet_files():
    """Télécharge le code réseau et les poids SCUNet si absents."""
    import urllib.request
    os.makedirs(SCUNET_MODELS_DIR, exist_ok=True)
    files = {
        "network_scunet.py":          "https://raw.githubusercontent.com/cszn/SCUNet/main/models/network_scunet.py",
        "scunet_color_real_gan.pth":  "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth",
        "scunet_color_real_psnr.pth": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    }
    for fname, url in files.items():
        dst = os.path.join(SCUNET_MODELS_DIR, fname)
        if os.path.exists(dst):
            continue
        print(f"  -> Téléchargement {fname}...")
        def _prog(c, bs, ts):
            if ts > 0:
                print(f"\r     {min(100, int(c*bs*100/ts))}%", end="", flush=True)
        urllib.request.urlretrieve(url, dst, reporthook=_prog)
        print()


def apply_scunet(img_bgr, mode=None):
    """Embellissement par débruitage aveugle SCUNet.

    mode : 'gan'  -> perceptuel, texture naturelle (défaut)
           'psnr' -> conservateur, aucune hallucination
    """
    if mode is None:
        mode = SCUNET_MODEL

    print(f"  -> Préparation SCUNet ({mode})...")
    try:
        _download_scunet_files()
    except Exception as e:
        print(f"  Impossible de télécharger les modèles SCUNet : {e}")
        return None

    net_path   = os.path.join(SCUNET_MODELS_DIR, "network_scunet.py")
    model_path = os.path.join(SCUNET_MODELS_DIR, f"scunet_color_real_{mode}.pth")

    if not os.path.exists(net_path) or not os.path.exists(model_path):
        print(f"  Fichiers SCUNet manquants après téléchargement.")
        return None

    try:
        import torch
        if SCUNET_MODELS_DIR not in sys.path:
            sys.path.insert(0, SCUNET_MODELS_DIR)
        from network_scunet import SCUNet

        model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        model.eval()

        img_f = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
        t_in  = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).float().unsqueeze(0)

        print(f"  -> Inférence SCUNet sur {img_bgr.shape[1]}x{img_bgr.shape[0]} px...")
        with torch.no_grad():
            t_out = model(t_in)

        out_np  = t_out.squeeze().numpy()
        out_np  = np.transpose(out_np, (1, 2, 0)).clip(0, 1)
        return (out_np[:, :, ::-1] * 255).round().astype(np.uint8)

    except Exception as e:
        print(f"  Erreur SCUNet : {e}")
        import traceback; traceback.print_exc()
        return None


# ===========================================================================
# GFPGAN — RESTAURATION DES VISAGES
# ===========================================================================

def _patch_basicsr_registry():
    """Rend ARCH_REGISTRY idempotent pour éviter les conflits d'import."""
    try:
        from basicsr.utils.registry import Registry
        _orig = Registry._do_register

        def _idempotent(self, name, obj, suffix=None):
            reg_name = name + '_' + suffix if isinstance(suffix, str) else name
            if reg_name not in self._obj_map:
                _orig(self, name, obj, suffix)

        Registry._do_register = _idempotent
    except Exception:
        pass


def download_gfpgan_model(model_path):
    """Télécharge GFPGANv1.4 si absent."""
    if os.path.exists(model_path):
        return True
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    print(f"  -> Téléchargement GFPGANv1.4 (~333 MB)...")
    try:
        import urllib.request
        def _prog(c, bs, ts):
            if ts > 0:
                print(f"\r     {min(100, int(c*bs*100/ts))}%", end="", flush=True)
        urllib.request.urlretrieve(url, model_path, reporthook=_prog)
        print()
        return True
    except Exception as e:
        print(f"\n  Échec téléchargement : {e}")
        return False


def apply_gfpgan(img_bgr, upscale=1, weight=0.5):
    """Restauration des visages avec GFPGAN v1.4.

    - upscale : facteur d'agrandissement (1 = taille originale)
    - weight  : force de restauration, 0.0 (original) → 1.0 (pleine restauration)
                0.5 est le défaut GFPGAN. Réduire pour éviter l'effet plastique.

    Retourne (image_restaurée, liste_de_bbox_visages) ou None si erreur.
    """
    _patch_basicsr_registry()
    try:
        from gfpgan import GFPGANer
    except Exception as e:
        print(f"  GFPGAN non disponible : {e}")
        return None

    if not download_gfpgan_model(GFPGAN_MODEL_PATH):
        return None

    try:
        restorer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        print(f"  -> Détection et restauration des visages (weight={weight})...")
        _, _, result = restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight,
        )
        face_bboxes = []
        try:
            dets = getattr(restorer.face_helper, 'face_det_results',
                   getattr(restorer.face_helper, 'det_faces', []))
            for det in dets:
                bbox = det[0] if (isinstance(det, (list, tuple)) and len(det) == 2
                                  and not isinstance(det[0], (int, float))) else det
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                face_bboxes.append((x1, y1, x2, y2))
        except Exception:
            pass
        return result, face_bboxes

    except Exception as e:
        print(f"  Erreur GFPGAN : {e}")
        import traceback; traceback.print_exc()
        return None


def blend_faces_from_gfpgan(gfpgan_img, scunet_img, face_bboxes, expand=0.4):
    """Restitue les visages GFPGAN sur le fond SCUNet via masque gaussien.

    Évite l'effet plastique de SCUNet sur des visages déjà restaurés par GFPGAN.
    - expand : extension relative de la boîte (0.4 = +40 % de la taille du visage)
    """
    if not face_bboxes:
        return scunet_img

    h, w = scunet_img.shape[:2]
    face_mask = np.zeros((h, w), np.float32)

    for (x1, y1, x2, y2) in face_bboxes:
        fw, fh = x2 - x1, y2 - y1
        pad = int(max(fw, fh) * expand)
        ex1, ey1 = max(0, x1 - pad), max(0, y1 - pad)
        ex2, ey2 = min(w, x2 + pad), min(h, y2 + pad)
        patch = np.zeros((h, w), np.float32)
        patch[ey1:ey2, ex1:ex2] = 1.0
        sigma = max(10, int(max(fw, fh) * 0.20))
        patch = cv2.GaussianBlur(patch, (0, 0), sigma)
        face_mask = np.maximum(face_mask, patch)

    face_mask = np.clip(face_mask, 0, 1)
    m3 = np.stack([face_mask] * 3, axis=-1)
    result = (gfpgan_img.astype(np.float32) * m3 +
              scunet_img.astype(np.float32) * (1.0 - m3))
    return np.clip(result, 0, 255).astype(np.uint8)


# ===========================================================================
# GRILLE DE COMPARAISON
# ===========================================================================

def make_grid(images_dict, max_width=2200):
    """Crée une grille de comparaison avec étiquettes de colonnes."""
    n = len(images_dict)
    h0, w0 = next(iter(images_dict.values())).shape[:2]
    col_w  = max_width // n
    col_h  = int(h0 * col_w / w0)
    label_h = 44
    cols = []
    for label, img in images_dict.items():
        resized = cv2.resize(img, (col_w, col_h), interpolation=cv2.INTER_LANCZOS4)
        header  = np.full((label_h, col_w, 3), 40, dtype=np.uint8)
        cv2.putText(header, label, (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cols.append(np.vstack([header, resized]))
    return np.hstack(cols)
