"""Routes REST de l'API de comparaison photo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import imaging
from .core.json_diff import compute_aligned_diff
from .session import IMAGE_EXTENSIONS, is_sidecar_image, session

router = APIRouter(prefix="/api")


# ----------------------------------------------------------------------
# Session & répertoires
# ----------------------------------------------------------------------

@router.get("/session")
def get_session() -> dict:
    return session.state()


class DirData(BaseModel):
    path: str
    enabled: bool = True


class SetDirsBody(BaseModel):
    dirs: list[DirData]
    output_dir: str = ""


@router.post("/session/dirs")
def set_dirs(body: SetDirsBody) -> dict:
    session.set_directories(
        [d.model_dump() for d in body.dirs], body.output_dir or None
    )
    return session.state()


@router.get("/fs/list")
def fs_list(path: str = Query("")) -> dict:
    """Liste les sous-dossiers d'un dossier serveur + comptage d'images.

    ``path`` vide → racines disponibles (home + racine système).
    """
    if not path:
        roots = []
        home = Path.home()
        roots.append({"name": f"~ ({home})", "path": str(home)})
        roots.append({"name": "/", "path": "/"})
        return {"path": "", "parent": None, "entries": roots, "image_count": 0}

    p = Path(path)
    if not p.is_dir():
        raise HTTPException(status_code=404, detail="dossier introuvable")

    entries = []
    image_count = 0
    try:
        for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if child.is_dir():
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "image_count": _count_images(child),
                })
            elif (
                child.suffix.lower() in IMAGE_EXTENSIONS
                and not is_sidecar_image(child.name)
            ):
                image_count += 1
    except PermissionError:
        raise HTTPException(status_code=403, detail="accès refusé")

    parent = str(p.parent) if p.parent != p else None
    return {
        "path": str(p),
        "parent": parent,
        "entries": entries,
        "image_count": image_count,
    }


def _count_images(directory: Path) -> Optional[int]:
    try:
        return sum(
            1 for f in directory.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
            and not is_sidecar_image(f.name)
        )
    except (PermissionError, OSError):
        return None


# ----------------------------------------------------------------------
# Groupes
# ----------------------------------------------------------------------

@router.get("/groups")
def get_groups() -> dict:
    return {"groups": session.groups()}


@router.get("/group/{key}")
def get_group(key: str) -> dict:
    detail = session.group_detail(key)
    if detail is None:
        raise HTTPException(status_code=404, detail="groupe introuvable")
    return detail


# ----------------------------------------------------------------------
# Images & vignettes
# ----------------------------------------------------------------------

@router.get("/image")
def get_image(key: str = Query(...), dir: int = Query(...)):
    group = session.get_group(key)
    if group is None:
        raise HTTPException(status_code=404, detail="groupe introuvable")
    path = group.get_photo(dir)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="image absente")

    data, file_path, media_type = imaging.resolve(path)
    if not media_type:
        raise HTTPException(status_code=500, detail="image illisible")
    headers = {"Cache-Control": "public, max-age=86400"}
    if data is not None:
        return Response(content=data, media_type=media_type, headers=headers)
    return FileResponse(file_path, media_type=media_type, headers=headers)


@router.get("/thumb")
def get_thumb(key: str = Query(...), dir: Optional[int] = Query(None)):
    group = session.get_group(key)
    if group is None:
        raise HTTPException(status_code=404, detail="groupe introuvable")
    # Répertoire demandé si présent pour ce groupe, sinon repli sur la 1re version.
    photo = group.get_photo(dir) if dir is not None else None
    if photo is None:
        photo = group.any_photo()
    if photo is None:
        raise HTTPException(status_code=404, detail="aucune image")
    from app.utils.thumbnail_cache import generate_thumbnail
    thumb = generate_thumbnail(photo)
    if thumb is None:
        raise HTTPException(status_code=500, detail="vignette illisible")
    return FileResponse(
        thumb, media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ----------------------------------------------------------------------
# Mode JSON
# ----------------------------------------------------------------------

@router.get("/json/{key}")
def get_json(key: str) -> dict:
    group = session.get_group(key)
    if group is None:
        raise HTTPException(status_code=404, detail="groupe introuvable")

    contents: dict[int, str] = {}
    meta: dict[int, dict] = {}
    for i in range(session.dm.dir_count()):
        path = group.jsons.get(i) or session._derive_json(group, i)
        if path is None or not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            parsed = json.loads(raw)
            contents[i] = json.dumps(parsed, sort_keys=True, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            contents[i] = raw
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        meta[i] = {"filename": path.name, "size": size}

    aligned = compute_aligned_diff(contents)

    tiles = []
    for i in range(session.dm.dir_count()):
        if i in aligned:
            padded, diff_pos, gap_pos = aligned[i]
            tiles.append({
                "dir_index": i,
                "dir_name": session.dm.dir_names[i],
                "present": True,
                "filename": meta[i]["filename"],
                "size": meta[i]["size"],
                "lines": ["" if ln is None else ln for ln in padded],
                "diff": sorted(diff_pos),
                "gap": sorted(gap_pos),
                "is_best": session.best_selections.get(key) == i,
            })
        else:
            tiles.append({
                "dir_index": i,
                "dir_name": session.dm.dir_names[i],
                "present": False,
                "is_best": session.best_selections.get(key) == i,
            })
    return {"key": key, "tiles": tiles}


# ----------------------------------------------------------------------
# Sélection « meilleure » & export
# ----------------------------------------------------------------------

class BestBody(BaseModel):
    key: str
    dir_index: Optional[int] = None


@router.post("/best")
def set_best(body: BestBody) -> dict:
    session.set_best(body.key, body.dir_index)
    return {"ok": True, "best": session.best_selections.get(body.key)}


class ExportBody(BaseModel):
    key: str
    dir_index: int
    overwrite: bool = False


@router.post("/export")
def export(body: ExportBody):
    result = session.export(body.key, body.dir_index, body.overwrite)
    if not result.get("ok"):
        status = 409 if result.get("conflict") else 400
        return JSONResponse(result, status_code=status)
    return result


# ----------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------

@router.get("/notes")
def get_notes() -> dict:
    return {"notes": session.notes}


class NoteBody(BaseModel):
    text: str = ""


@router.put("/notes/{key}")
def put_note(key: str, body: NoteBody) -> dict:
    session.set_note(key, body.text)
    return {"ok": True}
