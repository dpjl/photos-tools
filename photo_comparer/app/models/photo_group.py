"""Photo group data model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PhotoGroup:
    """One logical photo (e.g. '1995-0077') that may exist in multiple dirs."""

    key: str                          # e.g. '1995-0077'
    photos: Dict[int, Path] = field(default_factory=dict)  # dir_index → path
    jsons: Dict[int, Path] = field(default_factory=dict)   # dir_index → associated JSON path

    # ------------------------------------------------------------------
    def get_photo(self, dir_index: int) -> Optional[Path]:
        return self.photos.get(dir_index)

    def has_photo(self, dir_index: int) -> bool:
        return dir_index in self.photos

    def any_photo(self) -> Optional[Path]:
        """Return the first available photo path (lowest dir index)."""
        for path in self.photos.values():
            return path
        return None

    def photo_count(self) -> int:
        return len(self.photos)

    # Allow sorting by key
    def __lt__(self, other: PhotoGroup) -> bool:
        return self.key < other.key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhotoGroup):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)
