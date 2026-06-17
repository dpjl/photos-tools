"""core/genref.py — Couleur par référence IA : moteur de l'étape « genref ».

Principe (issu de l'étude do-not-commit/flux-ref) :
  1. Le VLM local rédige un prompt de restauration par photo : il liste les
     éléments visibles avec leur couleur réaliste attendue, et la couleur du
     voile global est détectée automatiquement et nommée dans le prompt
     (déterminant pour la force de l'édition Kontext).
  2. FLUX.1 Kontext (quantifié 4-bit, ~1 Mpx) génère une « référence » :
     la même scène avec des couleurs restaurées. Cette image n'est JAMAIS
     utilisée directement (détails régénérés) — elle ne sert que de palette.
  3. Un flot optique jetable (DIS, cohérence aller-retour) fournit des paires
     de couleurs fiables original→référence, sur lesquelles on ajuste une
     application polynomiale globale (10 termes quadratiques × 3 canaux =
     30 coefficients « beta », régularisée vers l'identité).
  4. L'application de beta à l'image est instantanée et linéaire en beta :
     interpoler beta vers l'identité = curseur « force » exact.

Cache disque : config.GENREF_CACHE_DIR/<digest>/<style>_<graine>/
  prompt.json (prompt + items + cast) · ref.png (référence) · beta.json
Une entrée est réutilisable d'une session à l'autre (digest de l'image
d'entrée de l'étape).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config

# Styles de prompt proposés par l'étape. « manuel » = texte édité par
# l'utilisateur dans le dialogue (pas de passage VLM imposé).
PROMPT_STYLES = ["court", "détaillé", "manuel"]

_ASK_COURT = (
    "This is a faded 1980s family photo whose colors must be restored. "
    "List the 4 or 5 MAIN visible elements with the realistic color they "
    "should have (e.g. 'green plant leaves', 'white wall'). Short factual "
    "phrases only. Answer ONLY as a JSON list of strings."
)

_ASK_DETAILLE = (
    "This is a faded 1980s family photo whose colors must be restored. "
    "List the 8 to 10 main visible elements with the realistic color they "
    "should have (e.g. 'green plant leaves', 'white wall', 'light blue "
    "baby outfit'). Be specific and factual, no commentary. "
    "Answer ONLY as a JSON list of short strings."
)

_TEMPLATE = (
    "Completely remove the {cast} color cast from this faded 1980s "
    "photograph. Restore bright, saturated, true-to-life colors: {items}. "
    "Natural warm skin tones, neutral whites. Modern digital camera look "
    "with deep contrast. Keep every person, object, pose and the framing "
    "exactly identical."
)


# ══════════════════════════════════════════════════════════════════════════
# Utilitaires couleur (OKLab) — autonomes, pas de dépendance UI
# ══════════════════════════════════════════════════════════════════════════

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def _srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
    f = img_u8.astype(np.float32) / 255.0
    return np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    out = np.where(lin <= 0.0031308, lin * 12.92,
                   1.055 * np.power(np.maximum(lin, 1e-8), 1.0 / 2.4) - 0.055)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _bgr_to_oklab(img_u8: np.ndarray) -> np.ndarray:
    lin = _srgb_to_linear(img_u8)[:, :, ::-1]
    lms = lin @ _M1.T
    return (np.cbrt(np.maximum(lms, 0.0)) @ _M2.T).astype(np.float32)


def _oklab_to_bgr(lab: np.ndarray) -> np.ndarray:
    lms = (lab @ _M2_INV.T) ** 3
    rgb = np.clip(lms @ _M1_INV.T, 0.0, 1.0)
    return _linear_to_srgb(rgb[:, :, ::-1])


def image_digest(img: np.ndarray) -> str:
    """Empreinte stable de l'image (mêmes octets → même entrée de cache)."""
    arr = np.ascontiguousarray(img)
    h = hashlib.blake2b(digest_size=12)
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def cast_name(img_bgr: np.ndarray) -> str:
    """Nomme la dominante globale (teinte du décalage ab médian en OKLab).

    Nommer la couleur du voile dans le prompt change radicalement la force
    de l'édition Kontext (vérifié expérimentalement)."""
    lab = _bgr_to_oklab(img_bgr)
    a = float(np.median(lab[:, :, 1]))
    b = float(np.median(lab[:, :, 2]))
    if (a * a + b * b) ** 0.5 < 0.008:
        return "slight"
    hue = float(np.degrees(np.arctan2(b, a))) % 360
    for lim, name in ((20, "pink-magenta"), (45, "orange-red"), (75, "orange"),
                      (105, "yellow"), (150, "yellow-green"),
                      (200, "green-cyan"), (250, "blue"), (290, "purple"),
                      (330, "magenta"), (360, "pink-magenta")):
        if hue <= lim:
            return name
    return "color"


# ══════════════════════════════════════════════════════════════════════════
# Entrée de cache
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class GenRefEntry:
    """Une référence générée + sa LUT, pour une (image, style, graine)."""
    digest:  str
    style:   str
    seed:    int
    prompt:  str
    items:   list = field(default_factory=list)
    cast:    str = ""
    beta:    np.ndarray | None = None      # (10, 3) float32
    ref_bgr: np.ndarray | None = None      # référence FLUX (≈1 Mpx)
    conf_mean: float = 0.0                 # confiance moyenne du flot

    @property
    def key(self) -> str:
        return f"{self.style}_{self.seed}"


def _entry_dir(digest: str, style: str, seed: int) -> str:
    return os.path.join(config.GENREF_CACHE_DIR, digest, f"{style}_{seed}")


def save_entry(entry: GenRefEntry) -> None:
    d = _entry_dir(entry.digest, entry.style, entry.seed)
    os.makedirs(d, exist_ok=True)
    meta = {"prompt": entry.prompt, "items": entry.items, "cast": entry.cast,
            "style": entry.style, "seed": entry.seed,
            "conf_mean": entry.conf_mean}
    with open(os.path.join(d, "prompt.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if entry.ref_bgr is not None:
        cv2.imwrite(os.path.join(d, "ref.png"), entry.ref_bgr)
    if entry.beta is not None:
        with open(os.path.join(d, "beta.json"), "w") as f:
            json.dump(entry.beta.round(8).tolist(), f)


def load_entry(digest: str, style: str, seed: int,
               with_ref: bool = True) -> GenRefEntry | None:
    d = _entry_dir(digest, style, seed)
    meta_path = os.path.join(d, "prompt.json")
    beta_path = os.path.join(d, "beta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    entry = GenRefEntry(digest=digest, style=style, seed=seed,
                        prompt=meta.get("prompt", ""),
                        items=meta.get("items", []),
                        cast=meta.get("cast", ""),
                        conf_mean=float(meta.get("conf_mean", 0.0)))
    if os.path.exists(beta_path):
        try:
            with open(beta_path) as f:
                entry.beta = np.array(json.load(f), np.float32)
        except Exception:
            entry.beta = None
    if with_ref:
        ref = cv2.imread(os.path.join(d, "ref.png"))
        entry.ref_bgr = ref
    return entry


def list_entries(digest: str) -> list[GenRefEntry]:
    """Toutes les entrées en cache pour cette image (sans charger les refs)."""
    base = os.path.join(config.GENREF_CACHE_DIR, digest)
    out: list[GenRefEntry] = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        style, seed_s = parts
        try:
            seed = int(seed_s)
        except ValueError:
            continue
        entry = load_entry(digest, style, seed, with_ref=False)
        if entry is not None:
            out.append(entry)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Rédaction du prompt (VLM local)
# ══════════════════════════════════════════════════════════════════════════

def write_prompt(img_bgr: np.ndarray, style: str) -> dict:
    """Rédige le prompt FLUX pour cette image : items listés par le VLM
    (longueur selon le style) + couleur du voile détectée automatiquement.

    Retourne {"prompt", "items", "cast"}. Charge le VLM (lourd) au besoin.
    """
    import re
    from core.vlm_refine import VLMRefiner

    # Le VLM (~9 Go) et FLUX résident ne tiennent pas ensemble en VRAM :
    # libérer FLUX d'abord (rechargé automatiquement à la génération).
    GenRefEngine.release_for_vlm()

    cast = cast_name(img_bgr)
    ask = _ASK_COURT if style == "court" else _ASK_DETAILLE

    h, w = img_bgr.shape[:2]
    s = 896 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * s), int(h * s)))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    txt = VLMRefiner.get().ask([rgb], ask, max_new_tokens=300)
    items: list[str] = []
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if m:
        try:
            items = [str(x) for x in json.loads(m.group(0))]
        except Exception:
            items = []

    prompt = _TEMPLATE.format(
        cast=cast,
        items=", ".join(items) if items else "all objects with plausible colors",
    )
    return {"prompt": prompt, "items": items, "cast": cast}


def translate_to_prompt_en(text_fr: str) -> str:
    """Traduit un texte français en anglais optimisé pour un prompt FLUX.

    Style impératif et concret (vocabulaire couleurs/objets), sans
    commentaire — le résultat est destiné à être collé dans le prompt.
    Charge le VLM (lourd) au besoin.
    """
    from core.vlm_refine import VLMRefiner

    GenRefEngine.release_for_vlm()   # VLM et FLUX exclusifs en VRAM

    ask = (
        "Translate the following French text into English, optimized as a "
        "fragment of an image-editing instruction prompt for the FLUX.1 "
        "Kontext model: imperative style, concise, concrete color and "
        "object vocabulary, no commentary, no quotes. "
        "Answer ONLY with the translation.\n\n"
        f"French text:\n{text_fr.strip()}"
    )
    out = VLMRefiner.get().ask([], ask, max_new_tokens=300).strip()
    # Retirer d'éventuels guillemets d'encadrement
    if len(out) >= 2 and out[0] in "\"'«" and out[-1] in "\"'»":
        out = out[1:-1].strip()
    return out


# ══════════════════════════════════════════════════════════════════════════
# Génération de la référence (FLUX.1 Kontext, singleton déchargeable)
# ══════════════════════════════════════════════════════════════════════════

class GenRefEngine:
    """Pipeline FLUX Kontext chargé paresseusement, résident en VRAM.

    Le pipeline fp4 complet (~13,4 Go) tient dans 16 Go de VRAM : on évite
    ainsi le mode cpu_offload qui gardait ~14 Go résidents en RAM système
    en permanence — cause des OOM kill (« Killed ») observés pendant
    l'inférence sur une machine à 32 Go. En VRAM, un dépassement éventuel
    devient une erreur CUDA interceptable au lieu d'un SIGKILL noyau.

    Conséquence : VLM (≈9 Go) et FLUX ne cohabitent plus sur le GPU — la
    bascule est automatique dans les deux sens (release_for_vlm() avant un
    usage VLM ; _ensure() décharge le VLM). Recharger l'un ou l'autre coûte
    ~10-20 s, négligeable devant une génération.

    Si la VRAM ne suffit pas au chargement, repli automatique sur
    enable_model_cpu_offload (ancien mode, plus gourmand en RAM).
    """

    _instance: "GenRefEngine | None" = None

    @classmethod
    def get(cls) -> "GenRefEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._pipe = None
        self._gpu_resident = False

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def unload(self) -> None:
        self._pipe = None
        self._gpu_resident = False
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def release_for_vlm(cls) -> None:
        """Libère le GPU avant un usage VLM (prompt, traduction).

        Sans effet si le pipeline n'est pas chargé. Il sera rechargé
        automatiquement à la prochaine génération.
        """
        inst = cls._instance
        if inst is not None and inst._pipe is not None:
            inst.unload()

    @staticmethod
    def _available_ram_gb() -> float | None:
        """RAM disponible (Go) via /proc/meminfo, None si indisponible."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024 * 1024)
        except Exception:
            pass
        return None

    # Pic de RAM transitoire pendant la lecture/transfert des poids vers le
    # GPU. En dessous, le chargement risquerait l'OOM kill (SIGKILL noyau,
    # non interceptable) — on lève une erreur propre à la place.
    _MIN_RAM_GB = 8.0

    def _ensure(self):
        if self._pipe is not None:
            return self._pipe
        import gc
        import torch
        from diffusers import FluxKontextPipeline

        # Libère le VLM : FLUX + VLM ne tiennent pas ensemble sur 16 Go.
        try:
            from core.vlm_refine import VLMRefiner
            if VLMRefiner._instance is not None:
                VLMRefiner._instance.unload()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Garde-fou RAM : mieux vaut une erreur propre qu'un SIGKILL.
        avail = self._available_ram_gb()
        if avail is not None and avail < self._MIN_RAM_GB:
            raise RuntimeError(
                f"RAM disponible insuffisante pour charger FLUX "
                f"({avail:.1f} Go libres, {self._MIN_RAM_GB:.0f} Go requis). "
                "Fermer des applications ou utiliser le bouton 🧹 pour "
                "décharger les modèles, puis réessayer."
            )

        pipe = FluxKontextPipeline.from_pretrained(
            config.FLUX_KONTEXT_REPO, torch_dtype=torch.bfloat16,
            cache_dir=config.HF_HUB_DIR)
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe.to("cuda")
            self._gpu_resident = True
        except torch.cuda.OutOfMemoryError:
            # VRAM insuffisante (autre charge GPU ?) → repli cpu_offload.
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()
            self._gpu_resident = False
        self._pipe = pipe
        return pipe

    def generate(self, img_bgr: np.ndarray, prompt: str, seed: int = 42,
                 steps: int = 28, guidance: float = 4.0,
                 on_progress=None, keep_loaded: bool = False) -> np.ndarray:
        """Génère la référence (~1 Mpx, ratio préservé). on_progress(i, n).

        keep_loaded : garder le pipeline en VRAM après la génération —
        réservé aux enchaînements (pré-génération batch) ; l'appelant DOIT
        appeler unload() à la fin de la série.
        """
        import torch
        from PIL import Image

        pipe = self._ensure()
        h, w = img_bgr.shape[:2]
        scale = (1024 * 1024 / (w * h)) ** 0.5
        tw, th = int(w * scale / 16) * 16, int(h * scale / 16) * 16
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        def _cb(p, step_index, timestep, kwargs):
            if on_progress is not None:
                on_progress(step_index + 1, steps)
            return kwargs

        try:
            try:
                out = pipe(image=pil, prompt=prompt, guidance_scale=guidance,
                           width=tw, height=th, num_inference_steps=steps,
                           generator=torch.Generator("cpu").manual_seed(int(seed)),
                           callback_on_step_end=_cb).images[0]
            except torch.cuda.OutOfMemoryError:
                if not self._gpu_resident:
                    raise
                # VRAM insuffisante en mode résident : repli cpu_offload puis
                # une nouvelle tentative (plus lent mais sûr côté VRAM).
                self.unload()
                from diffusers import FluxKontextPipeline
                pipe = FluxKontextPipeline.from_pretrained(
                    config.FLUX_KONTEXT_REPO, torch_dtype=torch.bfloat16,
                    cache_dir=config.HF_HUB_DIR)
                pipe.set_progress_bar_config(disable=True)
                pipe.enable_model_cpu_offload()
                self._pipe = pipe
                self._gpu_resident = False
                out = pipe(image=pil, prompt=prompt, guidance_scale=guidance,
                           width=tw, height=th, num_inference_steps=steps,
                           generator=torch.Generator("cpu").manual_seed(int(seed)),
                           callback_on_step_end=_cb).images[0]
        finally:
            # Mode résident : libérer SYSTÉMATIQUEMENT la VRAM après chaque
            # génération, sinon les étapes suivantes du pipeline (SCUNet,
            # GFPGAN…) tombent en CUDA OOM. Sans danger côté RAM : les poids
            # ne vivent qu'en VRAM, le rechargement (~15 s) relit le disque.
            # En repli cpu_offload, on GARDE le pipeline chargé : c'est le
            # cycle déchargement/rechargement en RAM qui causait les
            # OOM kill (« Killed »).
            if self._gpu_resident and not keep_loaded:
                # La référence locale doit disparaître AVANT le gc.collect()
                # de unload(), sinon les tenseurs restent vivants.
                del pipe
                self.unload()
        ref = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        # Resize cosmétique vers les dimensions d'entrée : l'arrondi /16 de
        # FLUX change le ratio (~0,3 %) et la référence ne se superposait pas
        # à l'original. Sans effet sur la LUT : _warp_reference redimensionne
        # déjà la référence aux dimensions de l'original (même interpolation).
        if ref.shape[:2] != (h, w):
            ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_CUBIC)
        return ref


# ══════════════════════════════════════════════════════════════════════════
# Ajustement de la LUT (flot optique + moindres carrés pondérés)
# ══════════════════════════════════════════════════════════════════════════

def _gray_norm(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _warp_reference(orig: np.ndarray, ref: np.ndarray,
                    fb_thresh: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Recale la référence sur l'original (flot DIS) ; renvoie
    (référence warpée, confiance 0..1 par cohérence aller-retour)."""
    H, W = orig.shape[:2]
    ref_r = cv2.resize(ref, (W, H), interpolation=cv2.INTER_CUBIC)
    g1, g2 = _gray_norm(orig), _gray_norm(ref_r)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    flow_fw = dis.calc(g1, g2, None)
    flow_bw = dis.calc(g2, g1, None)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = xx + flow_fw[:, :, 0]
    map_y = yy + flow_fw[:, :, 1]
    warped = cv2.remap(ref_r, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)

    bw_at = cv2.remap(flow_bw, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
    fb_err = np.sqrt((flow_fw[:, :, 0] + bw_at[:, :, 0]) ** 2 +
                     (flow_fw[:, :, 1] + bw_at[:, :, 1]) ** 2)
    conf = np.exp(-fb_err / fb_thresh)
    l1 = g1.astype(np.float32) / 255.0
    l2 = cv2.remap(g2, map_x, map_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
    conf *= np.exp(-np.abs(l1 - l2) / 0.18)
    conf = cv2.GaussianBlur(conf.astype(np.float32), (0, 0), 3.0)
    return warped, np.clip(conf, 0.0, 1.0)


def _poly_features(bgr: np.ndarray) -> np.ndarray:
    x = bgr.astype(np.float32) / 255.0
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    return np.column_stack([np.ones(len(x), np.float32),
                            b, g, r, b * g, b * r, g * r,
                            b * b, g * g, r * r])


def identity_beta() -> np.ndarray:
    beta = np.zeros((10, 3), np.float32)
    beta[1, 0] = beta[2, 1] = beta[3, 2] = 1.0
    return beta


def fit_lut(orig: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """Ajuste beta (10×3) : couleurs originales → couleurs de la référence,
    pondéré par la confiance d'alignement, régularisé vers l'identité et
    ancré sur la grille RGB (stabilité hors du gamut observé)."""
    warped, conf = _warp_reference(orig, ref)

    h, w = orig.shape[:2]
    scale = min(1.0, 560.0 / max(h, w))
    size = (int(w * scale), int(h * scale))
    src = cv2.resize(orig, size, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    tgt = cv2.resize(warped, size, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    wei = cv2.resize(conf, size, interpolation=cv2.INTER_AREA).reshape(-1)

    if len(src) > 40000:
        idx = np.linspace(0, len(src) - 1, 40000).astype(np.int64)
        src, tgt, wei = src[idx], tgt[idx], wei[idx]

    x = _poly_features(src)
    y = tgt.astype(np.float32) / 255.0
    wts = np.clip(wei, 0.02, 1.0)

    grid = np.linspace(0, 255, 7, dtype=np.float32)
    bb, gg, rr = np.meshgrid(grid, grid, grid, indexing="ij")
    anchors = np.column_stack([bb.ravel(), gg.ravel(), rr.ravel()])
    xa = _poly_features(anchors)
    ya = anchors / 255.0
    wa = np.full(len(anchors), 0.15, np.float32)

    x_all = np.vstack([x, xa])
    y_all = np.vstack([y, ya])
    w_all = np.concatenate([wts, wa])
    sw = np.sqrt(w_all)[:, None]
    xw, yw = x_all * sw, y_all * sw
    reg = np.diag([12, 18, 18, 18, 32, 32, 32, 36, 36, 36]).astype(np.float32)
    beta = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw + reg @ identity_beta())
    return beta.astype(np.float32), float(conf.mean())


# ══════════════════════════════════════════════════════════════════════════
# Application de la LUT (instantanée — utilisée par le preview)
# ══════════════════════════════════════════════════════════════════════════

def apply_lut(img: np.ndarray, beta: np.ndarray,
              force: float = 1.0, saturation: float = 1.0) -> np.ndarray:
    """Applique beta avec force et saturation.

    force : interpolation linéaire beta_identité ↔ beta (exacte : la sortie
    est linéaire en beta), >1 = extrapolation douce.
    saturation : modulation de la chroma OKLab du résultat (luminance intacte).
    """
    if force <= 0.0:
        out = img.copy()
    else:
        beta_eff = identity_beta() * (1.0 - force) + beta * force
        h, w = img.shape[:2]
        vals = _poly_features(img.reshape(-1, 3)) @ beta_eff
        out = np.clip(vals.reshape(h, w, 3) * 255.0, 0, 255).astype(np.uint8)

    if abs(saturation - 1.0) > 1e-3:
        lab = _bgr_to_oklab(out)
        lab[:, :, 1] *= saturation
        lab[:, :, 2] *= saturation
        out = _oklab_to_bgr(lab)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Orchestration complète (utilisée par le dialogue de génération)
# ══════════════════════════════════════════════════════════════════════════

def build_entry(img_bgr: np.ndarray, style: str, seed: int,
                prompt: str | None = None,
                on_phase=None, on_progress=None) -> GenRefEntry:
    """Construit (ou complète) une entrée : prompt → référence → LUT.

    prompt : si fourni (style « manuel » ou prompt édité), pas de passage VLM.
    on_phase(label) et on_progress(i, n) : retours pour l'UI.
    Sauvegarde l'entrée sur disque à chaque jalon.
    """
    digest = image_digest(img_bgr)
    entry = load_entry(digest, style, seed) or GenRefEntry(
        digest=digest, style=style, seed=int(seed), prompt="")

    if prompt:
        if entry.prompt and prompt.strip() != entry.prompt.strip():
            # Prompt modifié par l'utilisateur → la référence et la LUT en
            # cache ne correspondent plus : tout régénérer.
            entry.ref_bgr = None
            entry.beta = None
        entry.prompt = prompt
    if not entry.prompt:
        if on_phase is not None:
            on_phase("Rédaction du prompt (VLM)…")
        info = write_prompt(img_bgr, style)
        entry.prompt = info["prompt"]
        entry.items = info["items"]
        entry.cast = info["cast"]
        save_entry(entry)

    if entry.ref_bgr is None:
        if on_phase is not None:
            on_phase("Génération de la référence (FLUX)…")
        t0 = time.time()
        entry.ref_bgr = GenRefEngine.get().generate(
            img_bgr, entry.prompt, seed=int(seed), on_progress=on_progress)
        save_entry(entry)

    if entry.beta is None:
        if on_phase is not None:
            on_phase("Ajustement de la LUT (flot optique)…")
        entry.beta, entry.conf_mean = fit_lut(img_bgr, entry.ref_bgr)
        save_entry(entry)

    return entry
