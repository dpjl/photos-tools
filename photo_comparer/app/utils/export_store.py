"""Persist 'best photo' export metadata alongside each exported file.

A sidecar file  <filename>.best.json  is written next to every exported
photo in the output directory.  This lets the app restore the green-border
selection and ✓ nav badges across sessions without any central database.

Sidecar schema
--------------
{
    "group_key":       "1995-0077",
    "source_path":     "/abs/path/to/source/1995-0077.jpg",
    "dir_name":        "[5b] 1995 - RT exported",
    "output_filename": "1995-0077.jpg"
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

SIDECAR_SUFFIX = ".best.json"


@dataclass
class ExportRecord:
    group_key: str
    source_path: str        # absolute path (str) of the original source file
    dir_name: str           # display name of the source directory
    output_filename: str    # filename inside the output directory


def save(output_dir: Path, record: ExportRecord) -> None:
    """Write (or overwrite) the sidecar next to the exported image."""
    try:
        sidecar = output_dir / (record.output_filename + SIDECAR_SUFFIX)
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "group_key":       record.group_key,
                    "source_path":     record.source_path,
                    "dir_name":        record.dir_name,
                    "output_filename": record.output_filename,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except OSError:
        pass  # non-fatal — user can still work without persistence


def load_all(output_dir: Optional[Path]) -> Dict[str, ExportRecord]:
    """Scan *output_dir* for sidecars; return {group_key: ExportRecord}."""
    records: Dict[str, ExportRecord] = {}
    if not output_dir or not output_dir.is_dir():
        return records
    for sidecar in output_dir.glob(f"*{SIDECAR_SUFFIX}"):
        try:
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            r = ExportRecord(**data)
            records[r.group_key] = r
        except Exception:
            pass  # corrupt / outdated sidecar — skip silently
    return records
