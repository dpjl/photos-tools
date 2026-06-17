"""État serveur (mono-utilisateur) : enveloppe ``DirectoryManager`` et persiste la
configuration, les notes et les sélections « meilleure » — équivalent serveur de la
logique de ``MainWindow`` / ``ComparisonView``.

La config et les notes réutilisent les mêmes fichiers que l'app de bureau
(``~/.photo_comparer_config.json`` / ``~/.photo_comparer_notes.json``) afin que les
deux interfaces partagent l'état.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Rendre le cœur de l'app de bureau importable (photo_comparer/app/...).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.models.directory_manager import (  # noqa: E402
    DirectoryManager,
    IMAGE_EXTENSIONS,
    is_sidecar_image,
)
from app.models.photo_group import PhotoGroup  # noqa: E402
from app.utils import export_store  # noqa: E402

CONFIG_PATH = Path.home() / ".photo_comparer_config.json"
NOTES_PATH = Path.home() / ".photo_comparer_notes.json"


class Session:
    def __init__(self) -> None:
        self.dm = DirectoryManager()
        self.all_dir_data: List[dict] = []  # [{"path": str, "enabled": bool}, ...]
        self.notes: Dict[str, str] = {}
        self.best_selections: Dict[str, int] = {}  # group_key → dir_index
        self.export_records: Dict[str, export_store.ExportRecord] = {}
        self.mode: str = "images"
        self._key_to_index: Dict[str, int] = {}

        self._load_notes()
        self._load_config()

    # ------------------------------------------------------------------
    # Répertoires
    # ------------------------------------------------------------------

    def set_directories(self, all_dir_data: List[dict], output_dir: Optional[str]) -> None:
        self.all_dir_data = all_dir_data
        dirs = [
            d["path"]
            for d in all_dir_data
            if d.get("enabled", True) and Path(d["path"]).is_dir()
        ]
        self.dm.set_directories(dirs, output_dir or None)
        self._reindex()
        self._load_export_records()
        self._save_config()

    def _reindex(self) -> None:
        self._key_to_index = {g.key: i for i, g in enumerate(self.dm.groups)}

    # ------------------------------------------------------------------
    # Groupes
    # ------------------------------------------------------------------

    def groups(self) -> List[dict]:
        total = self.dm.dir_count()
        out = []
        for g in self.dm.groups:
            out.append({
                "key": g.key,
                "photo_count": g.photo_count(),
                "total_dirs": total,
                "copied": g.key in self.export_records,
            })
        return out

    def get_group(self, key: str) -> Optional[PhotoGroup]:
        idx = self._key_to_index.get(key)
        return self.dm.get_group_at(idx) if idx is not None else None

    def group_detail(self, key: str) -> Optional[dict]:
        group = self.get_group(key)
        if group is None:
            return None
        best = self.best_selections.get(key)
        tiles = []
        for i in range(self.dm.dir_count()):
            path = group.get_photo(i)
            tiles.append({
                "dir_index": i,
                "dir_name": self.dm.dir_names[i],
                "present": path is not None,
                "filename": path.name if path else None,
                "size": _safe_size(path) if path else None,
                "mtime": _safe_mtime(path) if path else None,
                "is_best": best == i,
                "has_json": i in group.jsons or self._derive_json(group, i) is not None,
            })
        record = self.export_records.get(key)
        return {
            "key": key,
            "tiles": tiles,
            "best": best,
            "note": self.notes.get(key, ""),
            "export": _record_to_dict(record) if record else None,
        }

    # ------------------------------------------------------------------
    # Sélection « meilleure »
    # ------------------------------------------------------------------

    def set_best(self, key: str, dir_index: Optional[int]) -> None:
        if dir_index is None:
            self.best_selections.pop(key, None)
        else:
            self.best_selections[key] = dir_index

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, key: str, dir_index: int, overwrite: bool) -> dict:
        group = self.get_group(key)
        if group is None:
            return {"ok": False, "error": "groupe introuvable"}
        if self.dm.output_dir is None:
            return {"ok": False, "error": "aucun répertoire de sortie défini"}
        src = group.get_photo(dir_index)
        if src is None:
            return {"ok": False, "error": "image absente pour ce répertoire"}

        out_dir = self.dm.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / src.name

        if dest.exists() and not overwrite:
            return {"ok": False, "conflict": True, "filename": dest.name}

        try:
            shutil.copy2(str(src), str(dest))
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        dir_name = (
            self.dm.dir_names[dir_index]
            if 0 <= dir_index < len(self.dm.dir_names)
            else "?"
        )
        record = export_store.ExportRecord(
            group_key=key,
            source_path=str(src),
            dir_name=dir_name,
            output_filename=dest.name,
        )
        export_store.save(out_dir, record)
        self.export_records[key] = record
        self.best_selections[key] = dir_index
        return {"ok": True, "export": _record_to_dict(record)}

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def set_note(self, key: str, text: str) -> None:
        if text:
            self.notes[key] = text
        else:
            self.notes.pop(key, None)
        self._save_notes()

    # ------------------------------------------------------------------
    # Sérialisation de la session
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "dir_data": self.all_dir_data,
            "output_dir": str(self.dm.output_dir) if self.dm.output_dir else "",
            "dir_names": self.dm.dir_names,
            "mode": self.mode,
            "group_count": self.dm.group_count(),
        }

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _derive_json(self, group: PhotoGroup, dir_index: int) -> Optional[Path]:
        """Dérive le chemin du JSON associé si non rattaché au scan (parité avec
        ``ComparisonView._load_json_group``)."""
        path = group.jsons.get(dir_index)
        if path is not None:
            return path
        img = group.get_photo(dir_index)
        if img is None:
            return None
        from app.utils.prefix_extractor import extract_prefix
        try:
            candidates = [
                f for f in img.parent.iterdir()
                if f.name.lower().endswith(".result.json")
                and extract_prefix(f.name) == group.key
            ]
        except OSError:
            return None
        return candidates[0] if candidates else None

    def _load_export_records(self) -> None:
        self.export_records = export_store.load_all(self.dm.output_dir)
        for key, record in self.export_records.items():
            dir_idx = self._find_dir_index(record.source_path)
            if dir_idx is not None:
                self.best_selections.setdefault(key, dir_idx)

    def _find_dir_index(self, source_path: str) -> Optional[int]:
        try:
            src_parent = Path(source_path).resolve().parent
            for i, d in enumerate(self.dm.directories):
                if src_parent == d.resolve():
                    return i
        except Exception:
            pass
        return None

    def _load_config(self) -> None:
        try:
            if not CONFIG_PATH.exists():
                return
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if "dir_data" in cfg:
                self.all_dir_data = cfg["dir_data"]
            else:
                self.all_dir_data = [
                    {"path": d, "enabled": True} for d in cfg.get("input_dirs", [])
                ]
            output_dir = cfg.get("output_dir", "")
            dirs = [
                d["path"]
                for d in self.all_dir_data
                if d.get("enabled", True) and Path(d["path"]).is_dir()
            ]
            if not dirs:
                return
            self.dm.set_directories(dirs, output_dir or None)
            self._reindex()
            self._load_export_records()
        except Exception:
            pass

    def _save_config(self) -> None:
        try:
            cfg = {
                "dir_data": self.all_dir_data,
                "output_dir": str(self.dm.output_dir) if self.dm.output_dir else "",
            }
            CONFIG_PATH.write_text(
                json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def _load_notes(self) -> None:
        try:
            if NOTES_PATH.exists():
                self.notes = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
        except Exception:
            self.notes = {}

    def _save_notes(self) -> None:
        try:
            NOTES_PATH.write_text(
                json.dumps(self.notes, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass


def _safe_size(path: Optional[Path]) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _safe_mtime(path: Optional[Path]) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _record_to_dict(record: export_store.ExportRecord) -> dict:
    return {
        "group_key": record.group_key,
        "source_path": record.source_path,
        "dir_name": record.dir_name,
        "output_filename": record.output_filename,
    }


# Singleton global.
session = Session()
