"""core/vlm_refine.py — Affinage des masques d'artefacts par un VLM local.

Le détecteur de rayures (BOPBTL) produit parfois des faux positifs sur des
structures *réelles* de la scène : lignes (arêtes de meuble, motifs de papier
peint, câbles…) mais aussi **grosses taches** (peau, main, reflets, détails
d'objet).  Un VLM généraliste local, à qui l'on montre la zone dans son
contexte, sait distinguer un vrai défaut d'un élément du décor.

Recette (validée sur GPU) :

  · on n'envoie au VLM que les composantes **suspectes** :
      - les **lignes** (allongées et longues), seules sujettes aux FP linéaires ;
      - les **grosses taches** (épaisseur ≥ seuil) : une vraie poussière est fine,
        une grosse zone compacte est suspecte (peau, objet…).
    Les petites poussières compactes sont gardées d'office (clairement des défauts).
  · pour chaque candidat, **deux images** : vue *globale* (photo entière, zone
    peinte magenta + flèche) pour la localisation, et *zoom local HD* pour le
    détail.  C'est ce double point de vue qui rend la décision fiable.
  · question binaire **défaut / décor**, réponse en **JSON strict**.

Toute la conversation est conservée pour inspection dans l'UI.
Chargement paresseux (singleton), modèle déchargeable pour libérer la VRAM.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config  # fixe HF_HOME avant tout import de transformers/huggingface_hub  # noqa: F401

# ── Catalogue de modèles (juin 2026, pour un GPU 16 Go) ───────────────────────
# (libellé, repo HuggingFace, tient_en_bf16_sur_16Go)
# Gemma 4 (sorti le 2026-04) est multimodal (image) ; pour 16 Go, les variantes
# « effectives » E4B (meilleure qualité qui tient) et E2B (plus légère) ; les
# 26B MoE / 31B Dense ne tiennent pas en bf16. Modèles gated → token HF requis.
VLM_MODELS: list[tuple[str, str, bool]] = [
    ("Qwen3-VL 4B (défaut)",      "Qwen/Qwen3-VL-4B-Instruct", True),
    ("Qwen3-VL 8B (VRAM limite)", "Qwen/Qwen3-VL-8B-Instruct", False),
    ("Gemma 4 E4B (VRAM limite)", "google/gemma-4-E4B-it",     False),
    ("Gemma 4 E2B",              "google/gemma-4-E2B-it",     True),
    ("Gemma 3 4B",               "google/gemma-3-4b-it",      True),
]
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

# Sélection des composantes envoyées au VLM
_MIN_LEN_DEFAULT   = 45     # longueur mini d'une ligne (px)
_MIN_THICK_DEFAULT = 12     # épaisseur mini d'une « grosse tache » (px)
_MIN_RATIO         = 2.2    # allongement mini d'une ligne (longueur / épaisseur)

# Deux prompts dédiés (lignes vs taches) : un prompt unique tirait les lignes
# vers "scene" dès qu'une catégorie large (peau, reflet, couture...) pouvait
# s'appliquer, y compris pour de longues rayures bien réelles. Le prompt
# "ligne" est celui validé sur 0031/0033 (cf. do-not-commit/_vlm3) : il ne
# laisse que des catégories *linéaires* pour "scene", avec une règle stricte
# sur les traits qui traversent librement plusieurs surfaces.
_LINE_PROMPT = (
    "Old scanned color photo (1985). Two views of the SAME candidate mark:\n"
    "- IMAGE 1: whole photo; the mark is painted magenta with a yellow arrow.\n"
    "- IMAGE 2: high-resolution zoom; the mark is outlined in magenta.\n"
    "Decide if the marked thing is damage on the photo or a real part of the scene.\n"
    "- \"defect\": scratch, scuff, crease, crack, dust/hair, stain ON the photo. "
    "Often a bright/pale thin streak crossing freely over surfaces, unrelated to objects.\n"
    "- \"scene\": furniture/door/window edge, wallpaper pattern line, cable, fabric seam.\n"
    "Be careful: a long thin streak that crosses over several different objects/"
    "surfaces and does NOT coincide with any real edge is a DEFECT (scratch).\n"
    "Answer ONLY JSON: {\"label\":\"defect|scene\",\"confidence\":0-1,\"reason\":\"few words\"}"
)

# Prompt « zones rouges » : distingue un voile rosé/magenta de détérioration
# (cast résiduel) de la couleur naturelle d'un objet chaud (bois, terre cuite,
# tissu rouge, peau). Validé sur les photos d'étude 1985-0042/0043
# (cf. do-not-commit/red-zones-study).
_CAST_PROMPT = (
    "Old scanned color photograph (1985) being restored. Two views of the SAME "
    "candidate region:\n"
    "- IMAGE 1: the whole photo; the region is outlined in magenta with a yellow arrow.\n"
    "- IMAGE 2: a high-resolution zoom; the region is outlined in magenta.\n"
    "This region looks reddish/pink/magenta. Decide whether this color is:\n"
    "- \"cast\": an abnormal pink/red/magenta color stain or light-leak residue caused by "
    "film/photo deterioration, which should be neutral (grey/white/blue/etc.) like the rest "
    "of the same object or surface, OR\n"
    "- \"natural\": the genuine natural color of the depicted material (e.g. terracotta tile "
    "floor, wood, skin, reddish fabric, warm light reflection consistent with the scene).\n"
    "Compare the region to the SAME object/surface elsewhere in IMAGE 1: if elsewhere it is a "
    "different (e.g. neutral grey/blue) color than here, this is a \"cast\".\n"
    "Answer ONLY JSON: {\"label\":\"cast|natural\",\"confidence\":0-1,\"reason\":\"few words\"}"
)

# Le prompt "tache" garde des catégories de décor plus larges (peau, reflet,
# détail d'objet) car les grosses taches compactes correspondent souvent à
# ce genre d'éléments réels.
_BLOB_PROMPT = (
    "Old scanned color photograph (1985). Two views of the SAME candidate region:\n"
    "- IMAGE 1: the whole photo; the region is painted magenta with a yellow arrow.\n"
    "- IMAGE 2: a high-resolution zoom; the region is outlined in magenta.\n"
    "This region is a compact spot/blob detected as a possible defect.\n"
    "Decide if it is damage ON the photo (dust speck, hair, stain, blotch, scuff), "
    "or a real part of the depicted scene (skin, face, eye, hand, button, jewelry, "
    "fabric/object detail, wallpaper pattern, highlight or reflection).\n"
    "- If the spot lies ON an object and matches its material, color and shape "
    "=> scene.\n"
    "- If it is an isolated spot/stain unrelated to any object, with an odd color "
    "or texture not matching its surroundings => defect.\n"
    "Answer ONLY with JSON: "
    "{\"label\":\"defect\" or \"scene\",\"confidence\":0.0-1.0,\"reason\":\"few words\"}"
)


# ── Authentification HuggingFace (modèles gated comme Gemma) ───────────────────

def set_hf_token(token: str) -> None:
    """Enregistre le token HF de façon permanente (sous HF_HOME) + session courante.

    Valide le token auprès du Hub puis l'écrit sur disque (relu automatiquement
    aux sessions suivantes).  Lève une exception si le token est invalide.
    """
    from huggingface_hub import login
    token = token.strip()
    login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def has_hf_token() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token
        return bool(get_token())
    except Exception:
        return False


# ── Sélection des candidats (pur, sans modèle) ────────────────────────────────

@dataclass
class _Components:
    n: int
    labels: np.ndarray
    stats: np.ndarray
    cents: np.ndarray


def _connected(mask: np.ndarray) -> _Components:
    n, labels, stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    return _Components(n, labels, stats, cents)


def select_candidates(
    mask: np.ndarray,
    min_len: int = _MIN_LEN_DEFAULT,
    min_thick: int = _MIN_THICK_DEFAULT,
) -> tuple[list[tuple[int, str]], _Components]:
    """Composantes à soumettre au VLM : (id, kind) avec kind ∈ {"line","blob"}.

    · "line" : allongée et longue (faux positif linéaire potentiel) ;
    · "blob" : compacte mais épaisse (grosse tache suspecte) ;
    · les petites poussières fines ne sont pas candidates (gardées d'office).
    Triées par aire décroissante (les plus suspectes d'abord).
    """
    c = _connected(mask)
    out: list[tuple[int, str]] = []
    for i in range(1, c.n):
        w = int(c.stats[i, cv2.CC_STAT_WIDTH])
        h = int(c.stats[i, cv2.CC_STAT_HEIGHT])
        length, thick = max(w, h), max(1, min(w, h))
        if length >= min_len and length >= _MIN_RATIO * thick:
            out.append((i, "line"))
        elif thick >= min_thick:
            out.append((i, "blob"))
    out.sort(key=lambda t: -int(c.stats[t[0], cv2.CC_STAT_AREA]))
    return out, c


def candidates_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    min_len: int = _MIN_LEN_DEFAULT,
    min_thick: int = _MIN_THICK_DEFAULT,
) -> tuple[np.ndarray, int, int, int]:
    """Image RGB d'aperçu des candidats : lignes (cyan), taches (orange),
    petites zones gardées (rouge pâle). Retourne (rgb, n_lignes, n_taches, n_gardés)."""
    if mask.shape[:2] != image_bgr.shape[:2]:
        image_bgr = cv2.resize(image_bgr, (mask.shape[1], mask.shape[0]))
    cands, c = select_candidates(mask, min_len, min_thick)
    cand_ids = {i for i, _ in cands}
    over = image_bgr.copy()
    # petites zones gardées (non candidates) en rouge pâle
    kept = ((mask > 0) & ~np.isin(c.labels, list(cand_ids))).astype(bool)
    over[kept] = (0, 0, 200)
    n_line = n_blob = 0
    for i, kind in cands:
        col = (255, 255, 0) if kind == "line" else (0, 165, 255)   # cyan / orange (BGR)
        over[c.labels == i] = col
        n_line += kind == "line"; n_blob += kind == "blob"
    blended = cv2.addWeighted(image_bgr, 0.45, over, 0.55, 0)
    n_kept = int(c.n - 1 - len(cands))
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), n_line, n_blob, n_kept


# ── Vues envoyées au VLM ──────────────────────────────────────────────────────

def _global_view(bgr, labels, lbl, centroid) -> np.ndarray:
    h, w = bgr.shape[:2]
    g = bgr.copy()
    comp = (labels == lbl).astype(np.uint8)
    g[cv2.dilate(comp, np.ones((5, 5), np.uint8)) > 0] = (255, 0, 255)
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.arrowedLine(g, (min(w - 1, cx + 60), max(0, cy - 60)), (cx, cy),
                    (0, 255, 255), 3, tipLength=0.3)
    s = 1000 / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)))
    return cv2.cvtColor(g, cv2.COLOR_BGR2RGB)


def _global_view_outline(bgr, labels, lbl, centroid,
                         color=(255, 0, 255)) -> np.ndarray:
    """Vue globale avec la zone détourée (couleurs réelles visibles).

    Contrairement à :func:`_global_view` (zone peinte), le contenu de la zone
    reste visible : indispensable quand le VLM doit juger sa COULEUR.
    """
    h, w = bgr.shape[:2]
    g = bgr.copy()
    comp = (labels == lbl).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(g, cnts, -1, color, 3)
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.arrowedLine(g, (min(w - 1, cx + 60), max(0, cy - 60)), (cx, cy),
                    (0, 255, 255), 3, tipLength=0.3)
    s = 1000 / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)))
    return cv2.cvtColor(g, cv2.COLOR_BGR2RGB)


def _zoom_view(bgr, labels, lbl, stat, color=(255, 0, 255)) -> np.ndarray:
    h, w = bgr.shape[:2]
    x, y, bw, bh = int(stat[0]), int(stat[1]), int(stat[2]), int(stat[3])
    pad = max(70, int(0.6 * max(bw, bh)))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    crop = bgr[y1:y2, x1:x2].copy()
    comp = (labels[y1:y2, x1:x2] == lbl).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cnts, -1, color, 1)
    s = 900 / max(crop.shape[:2])
    if s > 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _parse_json(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# ── Échanges / résultat ───────────────────────────────────────────────────────

@dataclass
class CandidateExchange:
    comp_id:    int
    kind:       str                 # "line" | "blob"
    length:     int
    thickness:  int
    global_rgb: np.ndarray
    zoom_rgb:   np.ndarray
    raw:        str
    label:      str                 # "defect" | "scene" | "?"
    confidence: float | None
    reason:     str
    removed:    bool


@dataclass
class RefineResult:
    model_id:     str
    prompt:       str
    refined_mask: np.ndarray                          # masque avec 'scene' retiré (si appliqué direct)
    review_labels: np.ndarray                         # int32, espace masque : id de composante candidate (0 ailleurs)
    review_categories: dict                           # {id: "scene" | "defect"} (décision VLM initiale)
    review_reasons: dict = field(default_factory=dict)  # {id: texte explicatif (verdict + raison)}
    exchanges:    list[CandidateExchange] = field(default_factory=list)
    n_components: int = 0
    removed:      int = 0
    elapsed:      float = 0.0


# ── Moteur VLM (singleton, chargement paresseux) ──────────────────────────────

class VLMRefiner:
    _instance: "VLMRefiner | None" = None

    @classmethod
    def get(cls) -> "VLMRefiner":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        import torch
        self._torch = torch
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._proc = None
        self._model_id: str | None = None

    @property
    def loaded_model(self) -> str | None:
        return self._model_id

    def unload(self) -> None:
        self._model = None
        self._proc = None
        self._model_id = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _ensure(self, model_id: str) -> None:
        if self._model is not None and self._model_id == model_id:
            return
        self.unload()
        from transformers import AutoProcessor
        self._proc = AutoProcessor.from_pretrained(model_id)
        self._model = self._load_model(model_id)
        self._model_id = model_id

    def _load_model(self, model_id: str):
        """Charge le VLM via l'Auto-classe adaptée (selon la génération du modèle).

        Qwen3-VL passe par ``AutoModelForImageTextToText`` ; Gemma 4 (multimodal)
        par ``AutoModelForMultimodalLM``.  On essaie les deux pour rester générique.
        """
        import transformers
        torch = self._torch
        kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa", device_map=self._device)
        last_err = None
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(model_id, **kw).eval()
            except Exception as exc:  # mauvaise Auto-classe pour ce modèle → on essaie la suivante
                last_err = exc
        raise last_err if last_err else RuntimeError("Aucune Auto-classe VLM disponible")

    def _ask(self, global_rgb: np.ndarray, zoom_rgb: np.ndarray, prompt: str) -> str:
        from PIL import Image
        torch = self._torch
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(global_rgb)},
            {"type": "image", "image": Image.fromarray(zoom_rgb)},
            {"type": "text", "text": prompt},
        ]}]
        inp = self._proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inp, max_new_tokens=120, do_sample=False)
        return self._proc.decode(
            out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def refine(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        model_id: str = DEFAULT_MODEL,
        min_len: int = _MIN_LEN_DEFAULT,
        min_thick: int = _MIN_THICK_DEFAULT,
        on_progress=None,
    ) -> RefineResult:
        """Classe les composantes suspectes et retire celles « décor »."""
        t0 = time.time()
        if mask.shape[:2] != image_bgr.shape[:2]:
            image_bgr = cv2.resize(image_bgr, (mask.shape[1], mask.shape[0]))

        cands, c = select_candidates(mask, min_len, min_thick)
        self._ensure(model_id)

        refined = mask.copy()
        exchanges: list[CandidateExchange] = []
        for k, (lbl, kind) in enumerate(cands):
            if on_progress is not None:
                on_progress(k, len(cands))
            g = _global_view(image_bgr, c.labels, lbl, c.cents[lbl])
            z = _zoom_view(image_bgr, c.labels, lbl, c.stats[lbl])
            prompt = _LINE_PROMPT if kind == "line" else _BLOB_PROMPT
            raw = self._ask(g, z, prompt)
            data = _parse_json(raw)
            label = str(data.get("label", "?")).lower()
            conf = data.get("confidence")
            removed = (label == "scene")
            if removed:
                refined[c.labels == lbl] = 0
            w = int(c.stats[lbl, cv2.CC_STAT_WIDTH]); h = int(c.stats[lbl, cv2.CC_STAT_HEIGHT])
            exchanges.append(CandidateExchange(
                comp_id=int(lbl), kind=kind, length=max(w, h), thickness=min(w, h),
                global_rgb=g, zoom_rgb=z, raw=raw, label=label,
                confidence=float(conf) if isinstance(conf, (int, float)) else None,
                reason=str(data.get("reason", "")), removed=removed,
            ))
        if on_progress is not None:
            on_progress(len(cands), len(cands))

        # Carte de revue : id de chaque composante candidate + catégorie + raison VLM.
        review_labels = np.zeros(mask.shape[:2], dtype=np.int32)
        review_categories: dict[int, str] = {}
        review_reasons: dict[int, str] = {}
        for e in exchanges:
            review_labels[c.labels == e.comp_id] = e.comp_id
            review_categories[e.comp_id] = "scene" if e.label == "scene" else "defect"
            verdict = {"scene": "décor", "defect": "défaut"}.get(e.label, "?")
            conf = f" {e.confidence:.0%}" if e.confidence is not None else ""
            kind = "ligne" if e.kind == "line" else "tache"
            review_reasons[e.comp_id] = (
                f"[{kind}] VLM : {verdict}{conf}\n{e.reason}" if e.reason
                else f"[{kind}] VLM : {verdict}{conf}"
            )

        prompt = f"--- Lignes ---\n{_LINE_PROMPT}\n\n--- Taches ---\n{_BLOB_PROMPT}"
        return RefineResult(
            model_id=model_id, prompt=prompt, refined_mask=refined,
            review_labels=review_labels, review_categories=review_categories,
            review_reasons=review_reasons,
            exchanges=exchanges, n_components=c.n - 1,
            removed=sum(e.removed for e in exchanges), elapsed=time.time() - t0,
        )

    def refine_redzones(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        model_id: str = DEFAULT_MODEL,
        on_progress=None,
    ) -> RefineResult:
        """Classe chaque zone rouge : cast résiduel (gardé) ou couleur naturelle.

        Toutes les composantes du masque sont soumises au VLM (la détection
        des zones rouges ne produit que de grandes zones, toutes ambiguës a
        priori : un voile magenta et un sol en terre cuite ont le même chroma).
        Le résultat réutilise la mécanique de revue des artefacts :
        « defect » = cast (vert, gardé), « scene » = naturel (violet, retiré).
        """
        t0 = time.time()
        if mask.shape[:2] != image_bgr.shape[:2]:
            image_bgr = cv2.resize(image_bgr, (mask.shape[1], mask.shape[0]))

        c = _connected(mask)
        cands = sorted(range(1, c.n),
                       key=lambda i: -int(c.stats[i, cv2.CC_STAT_AREA]))
        self._ensure(model_id)

        refined = mask.copy()
        exchanges: list[CandidateExchange] = []
        for k, lbl in enumerate(cands):
            if on_progress is not None:
                on_progress(k, len(cands))
            # Zone détourée (pas peinte) : le VLM doit voir sa vraie couleur.
            g = _global_view_outline(image_bgr, c.labels, lbl, c.cents[lbl])
            z = _zoom_view(image_bgr, c.labels, lbl, c.stats[lbl])
            raw = self._ask(g, z, _CAST_PROMPT)
            data = _parse_json(raw)
            label = str(data.get("label", "?")).lower()
            conf = data.get("confidence")
            removed = (label == "natural")
            if removed:
                refined[c.labels == lbl] = 0
            w = int(c.stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(c.stats[lbl, cv2.CC_STAT_HEIGHT])
            exchanges.append(CandidateExchange(
                comp_id=int(lbl), kind="cast", length=max(w, h),
                thickness=min(w, h), global_rgb=g, zoom_rgb=z, raw=raw,
                label=label,
                confidence=float(conf) if isinstance(conf, (int, float)) else None,
                reason=str(data.get("reason", "")), removed=removed,
            ))
        if on_progress is not None:
            on_progress(len(cands), len(cands))

        review_labels = np.zeros(mask.shape[:2], dtype=np.int32)
        review_categories: dict[int, str] = {}
        review_reasons: dict[int, str] = {}
        for e in exchanges:
            review_labels[c.labels == e.comp_id] = e.comp_id
            review_categories[e.comp_id] = "scene" if e.removed else "defect"
            verdict = {"natural": "couleur naturelle", "cast": "cast résiduel"}.get(
                e.label, "?")
            conf = f" {e.confidence:.0%}" if e.confidence is not None else ""
            review_reasons[e.comp_id] = (
                f"VLM : {verdict}{conf}\n{e.reason}" if e.reason
                else f"VLM : {verdict}{conf}"
            )

        return RefineResult(
            model_id=model_id, prompt=_CAST_PROMPT, refined_mask=refined,
            review_labels=review_labels, review_categories=review_categories,
            review_reasons=review_reasons,
            exchanges=exchanges, n_components=c.n - 1,
            removed=sum(e.removed for e in exchanges), elapsed=time.time() - t0,
        )
