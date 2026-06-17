"""Alignement LCS multi-tuiles pour le mode JSON — logique pure (sans Qt).

Extraction fidèle de ``ComparisonView._compute_aligned_diff`` afin d'être
réutilisable côté serveur. Le comportement doit rester identique à celui de
l'application PySide6 (``app/widgets/comparison_view.py``).
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple

AlignedTile = Tuple[List[str], Set[int], Set[int]]  # (padded_lines, diff_pos, gap_pos)


def compute_aligned_diff(tile_contents: Dict[int, str]) -> Dict[int, AlignedTile]:
    """Alignement LCS multi-tuiles.

    Retourne ``{tile_idx: (padded_lines, diff_positions, gap_positions)}``.
    ``padded_lines``  : list[str | None] — None marque un trou d'alignement.
    ``diff_positions``: set[int]          — index de lignes où cette tuile diffère.
    ``gap_positions`` : set[int]          — index de lignes où cette tuile a un trou.
    Toutes les listes ``padded_lines`` partagent la même longueur (scroll synchronisé).
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
                for k in range(paired, la):
                    at[i1 + k] = None
                if lb > la:
                    after_extra.setdefault(i2 - 1, []).extend(other[j1 + la: j2])
            elif tag == "delete":
                for k in range(la):
                    at[i1 + k] = None
            elif tag == "insert":
                ins_before.setdefault(i1, []).extend(other[j1:j2])

        tile_info[idx] = (at, ins_before, after_extra)

    virtual_rows: list = []

    def _empty_row() -> dict:
        return {i: None for i in sorted_idx}

    for r in range(n_ref + 1):
        for idx in sorted_idx[1:]:
            for line in tile_info[idx][1].get(r, []):
                row = _empty_row()
                row[idx] = line
                virtual_rows.append(row)

        if r < n_ref:
            row = _empty_row()
            row[ref_idx] = ref_lines[r]
            for idx in sorted_idx[1:]:
                row[idx] = tile_info[idx][0].get(r)
            virtual_rows.append(row)

            for idx in sorted_idx[1:]:
                for line in tile_info[idx][2].get(r, []):
                    row = _empty_row()
                    row[idx] = line
                    virtual_rows.append(row)

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
                if len(set(row.values())) > 1:
                    diff_pos.add(row_i)
        result[idx] = (padded, diff_pos, gap_pos)

    return result
