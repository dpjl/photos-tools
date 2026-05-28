"""Scans a set of directories and groups matching photos by prefix key."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .photo_group import PhotoGroup
from ..utils.prefix_extractor import extract_prefix

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


class DirectoryManager:
    def __init__(self):
        self.directories: List[Path] = []
        self.dir_names: List[str] = []
        self.output_dir: Optional[Path] = None
        self.groups: List[PhotoGroup] = []
        self._group_map: Dict[str, PhotoGroup] = {}

    # ------------------------------------------------------------------
    def set_directories(
        self,
        dirs: List[str | Path],
        output_dir: str | Path | None = None,
    ):
        self.directories = [Path(d) for d in dirs if str(d).strip()]
        self.dir_names = [p.name for p in self.directories]
        self.output_dir = Path(output_dir) if output_dir else None
        self._scan()

    # ------------------------------------------------------------------
    def _scan(self):
        self._group_map = {}
        for dir_idx, directory in enumerate(self.directories):
            if not directory.is_dir():
                continue
            for fp in directory.iterdir():
                if fp.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                key = extract_prefix(fp.name)
                if key not in self._group_map:
                    self._group_map[key] = PhotoGroup(key=key)
                # Keep first file found per dir per key (in case of duplicates)
                if dir_idx not in self._group_map[key].photos:
                    self._group_map[key].photos[dir_idx] = fp

        # Second pass: attach JSON files to existing image groups
        for dir_idx, directory in enumerate(self.directories):
            if not directory.is_dir():
                continue
            for fp in directory.iterdir():
                if fp.suffix.lower() != ".json":
                    continue
                key = extract_prefix(fp.name)
                if key not in self._group_map:
                    continue  # no matching image group — skip
                if dir_idx not in self._group_map[key].jsons:
                    self._group_map[key].jsons[dir_idx] = fp

        self.groups = sorted(self._group_map.values())

    # ------------------------------------------------------------------
    def get_group_at(self, index: int) -> Optional[PhotoGroup]:
        if 0 <= index < len(self.groups):
            return self.groups[index]
        return None

    def group_count(self) -> int:
        return len(self.groups)

    def dir_count(self) -> int:
        return len(self.directories)
