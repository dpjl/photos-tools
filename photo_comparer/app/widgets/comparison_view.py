"""ComparisonView — adaptive grid of PhotoTiles with synchronized viewport.

Layout
------
  2 dirs → 1×2   4 dirs → 2×2   6 dirs → 2×3   10 dirs → 2×5
  Formula: cols = ceil(sqrt(N)), rows = ceil(N/cols)

Zoom/Pan sync
-------------
When any tile's viewer emits viewport_changed(cx, cy, zoom), all other
*visible* tiles with a loaded image receive apply_viewport(cx, cy, zoom).
Because cx/cy are normalized to the image, the same relative region is
shown regardless of each file's resolution.

Solo mode
---------
Double-clicking a tile hides all others; double-clicking again (or Escape)
restores the grid.
"""
from __future__ import annotations

import json as _json
import math
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

try:
    from shiboken6 import isValid as _qt_isvalid
except ImportError:
    def _qt_isvalid(obj) -> bool:
        return True

from pathlib import Path
from typing import Dict

from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from ..models.directory_manager import DirectoryManager
from ..models.photo_group import PhotoGroup
from ..utils import export_store
from ..utils.export_store import ExportRecord
from ..utils.image_loader import ImageLoader
from .exported_bar import ExportedBar
from .photo_tile import PhotoTile


class ComparisonView(QWidget):
    best_changed = Signal(int, object)   # dir_index (or -1), Path or None
    group_loaded = Signal(str)           # group key

    def __init__(self, dir_manager: DirectoryManager, parent: QWidget = None):
        super().__init__(parent)
        self._dir_manager = dir_manager
        self._tiles: List[PhotoTile] = []
        self._current_group: Optional[PhotoGroup] = None
        self._best_tile_index: Optional[int] = None
        self._solo_index: Optional[int] = None
        self._load_gen: int = 0          # cancels stale async results
        self._grid_cols: int = 0
        self._grid_rows: int = 0
        self._pending_meta: dict = {}             # key → (PhotoTile, Path)
        self._best_selections: dict = {}          # group_key → dir_index
        self._export_records: Dict[str, ExportRecord] = {}  # group_key → record

        self._mode: str = "images"   # "images" or "json"
        self._syncing_json_scroll: bool = False

        self._loader = ImageLoader(self)
        self._loader.image_loaded.connect(self._on_image_loaded)

        # Layout: outer VBox → tile grid  +  exported bar
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._grid_widget = QWidget()
        self._layout = QGridLayout(self._grid_widget)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        outer.addWidget(self._grid_widget, 1)   # grid stretches

        self._exported_bar = ExportedBar(self)
        self._exported_bar.restore_requested.connect(self._on_restore_requested)
        outer.addWidget(self._exported_bar, 0)  # bar is fixed-height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_directories(self):
        """Rebuild tiles when directories change (call after set_directories)."""
        self._clear_tiles()
        n = self._dir_manager.dir_count()
        if n == 0:
            return

        rows, cols = self._compute_grid(n)
        self._grid_rows, self._grid_cols = rows, cols

        for i, dir_name in enumerate(self._dir_manager.dir_names):
            tile = PhotoTile(i, dir_name, self)
            tile.solo_requested.connect(self._on_solo_requested)
            tile.best_selected.connect(self._on_best_selected)
            tile.viewer.viewport_changed.connect(
                lambda cx, cy, z, src=i: self._sync_viewport(cx, cy, z, src)
            )
            tile._json_view.verticalScrollBar().valueChanged.connect(
                lambda val, src=i: self._sync_json_scroll(val, src)
            )
            row, col = divmod(i, cols)
            self._layout.addWidget(tile, row, col)
            self._tiles.append(tile)

        for c in range(cols):
            self._layout.setColumnStretch(c, 1)
        for r in range(rows):
            self._layout.setRowStretch(r, 1)

    def load_group(self, group: PhotoGroup):
        self._current_group = group
        self._best_tile_index = None
        self._solo_index = None
        self._show_all_tiles()

        self._load_gen += 1
        current_gen = self._load_gen
        self._pending_meta.clear()

        stored_best = self._best_selections.get(group.key)

        if self._mode == "json":
            self._load_json_group(group, stored_best)
        else:
            for i, tile in enumerate(self._tiles):
                is_best = (stored_best == i)
                tile.set_as_best(is_best)
                path = group.get_photo(i)
                if path is not None:
                    file_size = self._safe_stat(path)
                    tile.set_loading(file_size=file_size)
                    key = f"{current_gen}:{i}"
                    self._pending_meta[key] = (tile, path)
                    self._loader.load(key, path)
                else:
                    tile.set_absent()

        # Restore best state for toolbar button
        if stored_best is not None and 0 <= stored_best < len(self._tiles):
            self._best_tile_index = stored_best
            path = group.get_photo(stored_best)
            self.best_changed.emit(stored_best, path)

        # Show/hide the exported-photo bar
        record = self._export_records.get(group.key)
        if record:
            self._exported_bar.show_record(record)
        else:
            self._exported_bar.clear_record()

        self.group_loaded.emit(group.key)

    def set_mode(self, mode: str):
        """Switch between 'images' and 'json' display modes."""
        if mode == self._mode:
            return
        self._mode = mode
        if self._current_group is not None:
            self.load_group(self._current_group)

    def exit_solo(self):
        if self._solo_index is not None:
            self._solo_index = None
            self._show_all_tiles()

    def is_solo(self) -> bool:
        return self._solo_index is not None

    # --- Zoom controls (triggered from toolbar) ---

    def zoom_in(self):
        self._zoom_first_visible(+1)

    def zoom_out(self):
        self._zoom_first_visible(-1)

    def zoom_fit(self):
        for tile in self._tiles:
            if tile.isVisible():
                tile.viewer.fit_image()

    def zoom_100(self):
        # zoom_to_actual on first, then sync propagates
        for tile in self._tiles:
            if tile.isVisible() and tile._has_image:
                tile.viewer.zoom_to_actual()
                break

    @property
    def best_path(self) -> Optional[Path]:
        if self._best_tile_index is None or self._current_group is None:
            return None
        return self._current_group.get_photo(self._best_tile_index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @Slot(str, object)
    def _on_image_loaded(self, key: str, image):
        """Called on the GUI thread by ImageLoader.image_loaded."""
        meta = self._pending_meta.pop(key, None)
        if meta is None:
            return   # stale key (dirs changed or group navigated away)
        tile, path = meta
        gen = int(key.split(":")[0])
        if gen != self._load_gen:
            return
        if not _qt_isvalid(tile):
            return
        file_size = self._safe_stat(path)
        tile.set_pixmap(QPixmap.fromImage(image), path.name, file_size=file_size)

    def _clear_tiles(self):
        self._load_gen += 1               # invalidate any in-flight loads
        self._pending_meta.clear()
        self._best_selections.clear()
        self._export_records = {}
        if hasattr(self, '_exported_bar'):
            self._exported_bar.clear_record()
        for tile in self._tiles:
            self._layout.removeWidget(tile)
            tile.deleteLater()
        self._tiles = []
        self._best_tile_index = None
        self._solo_index = None
        self._current_group = None
        # Reset old stretch factors
        for c in range(self._grid_cols):
            self._layout.setColumnStretch(c, 0)
        for r in range(self._grid_rows):
            self._layout.setRowStretch(r, 0)
        self._grid_cols = 0
        self._grid_rows = 0

    # ------------------------------------------------------------------
    # JSON mode
    # ------------------------------------------------------------------

    def _load_json_group(self, group: "PhotoGroup", stored_best: Optional[int]):
        """Read JSON files for every tile and display with diff highlighting."""
        contents: Dict[int, str] = {}
        sizes: Dict[int, int] = {}
        filenames: Dict[int, str] = {}

        for i in range(len(self._tiles)):
            path = group.jsons.get(i)
            if path is None:
                # Try to derive JSON path from image path (same dir, same stem prefix)
                img_path = group.get_photo(i)
                if img_path is not None:
                    # Look for any <prefix>*.json in the same directory
                    from ..utils.prefix_extractor import extract_prefix
                    candidates = [
                        f for f in img_path.parent.iterdir()
                        if f.suffix.lower() == ".json"
                        and extract_prefix(f.name) == group.key
                    ]
                    if candidates:
                        path = candidates[0]
            if path is not None and path.is_file():
                try:
                    raw = path.read_text(encoding="utf-8", errors="replace")
                    sizes[i] = self._safe_stat(path)
                    filenames[i] = path.name
                    # Reformat JSON for consistent comparison
                    try:
                        parsed = _json.loads(raw)
                        contents[i] = _json.dumps(
                            parsed, sort_keys=True, indent=2, ensure_ascii=False
                        )
                    except (_json.JSONDecodeError, ValueError):
                        contents[i] = raw  # fallback: raw text
                except OSError:
                    pass

        aligned = self._compute_aligned_diff(contents)

        for i, tile in enumerate(self._tiles):
            is_best = (stored_best == i)
            tile.set_as_best(is_best)
            if i in aligned:
                padded, diff_pos, gap_pos = aligned[i]
                tile._file_label.setText(filenames.get(i, ""))
                tile.set_json(padded, sizes.get(i), diff_pos, gap_pos)
            else:
                tile.set_json_absent()

    @staticmethod
    def _compute_aligned_diff(tile_contents: Dict[int, str]) -> dict:
        """LCS-based multi-tile alignment.

        Returns {tile_idx: (padded_lines, diff_positions, gap_positions)}.
        padded_lines  : list[str | None]  — None marks an alignment gap
        diff_positions: set[int]          — row indices where this tile differs
        gap_positions : set[int]          — row indices where this tile has a gap
        All padded_lines lists share the same length, enabling sync scrolling.
        """
        if not tile_contents:
            return {}

        sorted_idx = sorted(tile_contents.keys())

        if len(sorted_idx) == 1:
            idx = sorted_idx[0]
            lines = tile_contents[idx].splitlines()
            return {idx: (lines, set(), set())}

        ref_idx = sorted_idx[0]
        ref_lines = tile_contents[ref_idx].splitlines()
        n_ref = len(ref_lines)

        # For each non-ref tile compute:
        #   at[r]          : what the tile has at ref position r (None = deleted)
        #   ins_before[r]  : lines inserted in this tile before ref position r
        #   after_extra[r] : extra tile lines after ref position r
        #                    (from replace blocks where tile has more lines than ref)
        tile_info: Dict[int, tuple] = {}
        for idx in sorted_idx[1:]:
            other = tile_contents[idx].splitlines()
            sm = SequenceMatcher(None, ref_lines, other, autojunk=False)
            at: dict = {}
            ins_before: dict = {}
            after_extra: dict = {}

            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                la, lb = i2 - i1, j2 - j1
                if tag == "equal":
                    for k in range(la):
                        at[i1 + k] = other[j1 + k]
                elif tag == "replace":
                    paired = min(la, lb)
                    for k in range(paired):
                        at[i1 + k] = other[j1 + k]
                    for k in range(paired, la):     # extra ref lines → gap in tile
                        at[i1 + k] = None
                    if lb > la:                     # extra tile lines → gap in ref
                        after_extra.setdefault(i2 - 1, []).extend(other[j1 + la: j2])
                elif tag == "delete":
                    for k in range(la):
                        at[i1 + k] = None
                elif tag == "insert":
                    ins_before.setdefault(i1, []).extend(other[j1:j2])

            tile_info[idx] = (at, ins_before, after_extra)

        # Build virtual rows: each row is {tile_idx: str_or_None}
        virtual_rows: list = []

        def _empty_row() -> dict:
            return {i: None for i in sorted_idx}

        for r in range(n_ref + 1):
            # Insertions before ref pos r — one virtual row per line, per tile
            for idx in sorted_idx[1:]:
                for line in tile_info[idx][1].get(r, []):
                    row = _empty_row()
                    row[idx] = line
                    virtual_rows.append(row)

            if r < n_ref:
                # Main row: one line from each tile aligned to ref pos r
                row = _empty_row()
                row[ref_idx] = ref_lines[r]
                for idx in sorted_idx[1:]:
                    row[idx] = tile_info[idx][0].get(r)   # None if deleted
                virtual_rows.append(row)

                # After-extra rows (tile has more lines than ref in a replace block)
                for idx in sorted_idx[1:]:
                    for line in tile_info[idx][2].get(r, []):
                        row = _empty_row()
                        row[idx] = line
                        virtual_rows.append(row)

        # Compute per-tile padded lists, diff positions, gap positions
        result: dict = {}
        for idx in sorted_idx:
            padded: list = []
            diff_pos: set = set()
            gap_pos: set = set()
            for row_i, row in enumerate(virtual_rows):
                val = row[idx]
                padded.append(val)
                if val is None:
                    gap_pos.add(row_i)
                else:
                    # Highlight if any tile disagrees at this row
                    if len(set(row.values())) > 1:
                        diff_pos.add(row_i)
            result[idx] = (padded, diff_pos, gap_pos)

        return result

    @staticmethod
    def _safe_stat(path) -> Optional[int]:
        """Return file size in bytes, or None on error."""
        try:
            return path.stat().st_size
        except OSError:
            return None

    @staticmethod
    def _compute_grid(n: int):
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    def _sync_json_scroll(self, value: int, source_idx: int):
        """Synchronise the JSON text-view scrollbars across all tiles."""
        if self._syncing_json_scroll:
            return
        self._syncing_json_scroll = True
        for i, tile in enumerate(self._tiles):
            if i != source_idx:
                tile._json_view.verticalScrollBar().setValue(value)
        self._syncing_json_scroll = False

    def _sync_viewport(self, cx: float, cy: float, zoom: float, source_idx: int):
        for i, tile in enumerate(self._tiles):
            if i != source_idx and tile.isVisible() and tile._has_image:
                tile.viewer.apply_viewport(cx, cy, zoom)

    def _show_all_tiles(self):
        for tile in self._tiles:
            tile.setVisible(True)

    def _zoom_first_visible(self, direction: int):
        for tile in self._tiles:
            if tile.isVisible() and tile._has_image:
                tile.viewer.adjust_zoom(direction)
                break   # sync propagates to all others

    # ------------------------------------------------------------------
    # Export record management (called from MainWindow)
    # ------------------------------------------------------------------

    def load_export_records(self, output_dir) -> set:
        """Scan output_dir for sidecars; populate records + best_selections.

        Returns the set of group keys that have an export record (for nav
        panel badge update).
        """
        self._export_records = export_store.load_all(output_dir)
        for group_key, record in self._export_records.items():
            dir_idx = self._find_dir_index(record.source_path)
            if dir_idx is not None:
                self._best_selections[group_key] = dir_idx
        return set(self._export_records.keys())

    def add_export_record(self, record: ExportRecord) -> None:
        """Register a freshly-made export and update the bar if relevant."""
        self._export_records[record.group_key] = record
        dir_idx = self._find_dir_index(record.source_path)
        if dir_idx is not None:
            self._best_selections[record.group_key] = dir_idx
        if self._current_group and self._current_group.key == record.group_key:
            self._exported_bar.show_record(record)

    def _find_dir_index(self, source_path: str) -> Optional[int]:
        """Map an absolute source_path back to its input directory index."""
        try:
            src_parent = Path(source_path).resolve().parent
            for i, d in enumerate(self._dir_manager.directories):
                if src_parent == d.resolve():
                    return i
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_restore_requested(self, source_path: str):
        """Re-light the green border for the tile matching source_path."""
        dir_idx = self._find_dir_index(source_path)
        if dir_idx is None or self._current_group is None:
            return
        if self._best_tile_index != dir_idx:
            self._on_best_selected(dir_idx)

    def _on_solo_requested(self, tile_index: int):
        if self._solo_index == tile_index:
            self.exit_solo()
        else:
            self._solo_index = tile_index
            for i, tile in enumerate(self._tiles):
                tile.setVisible(i == tile_index)

    def _on_best_selected(self, tile_index: int):
        if self._best_tile_index == tile_index:
            # Deselect
            self._tiles[tile_index].set_as_best(False)
            self._best_tile_index = None
            if self._current_group:
                self._best_selections.pop(self._current_group.key, None)
            self.best_changed.emit(-1, None)
        else:
            if self._best_tile_index is not None:
                self._tiles[self._best_tile_index].set_as_best(False)
            self._best_tile_index = tile_index
            self._tiles[tile_index].set_as_best(True)
            if self._current_group:
                self._best_selections[self._current_group.key] = tile_index
            path = (
                self._current_group.get_photo(tile_index)
                if self._current_group
                else None
            )
            self.best_changed.emit(tile_index, path)
