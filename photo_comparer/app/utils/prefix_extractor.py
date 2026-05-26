"""Extract a grouping key from a photo filename.

Default pattern: the first YYYY-NNNN segment (e.g. '1995-0077').
Works on names like:
  1995-0077.jpg               → '1995-0077'
  1995-0077-topaz-upscale.jpg → '1995-0077'
  1995-0077.jpg.pp3           → '1995-0077'
"""
import re
from pathlib import Path

_DEFAULT_RE = re.compile(r'^(\d{4}-\d{4})')


def clean_stem(filename: str) -> str:
    """Return the base stem, stripping compound extensions like .jpg.pp3."""
    stem = Path(filename).name
    # Strip all extensions iteratively
    while True:
        new = Path(stem).stem
        if new == stem:
            break
        stem = new
    return stem


def extract_prefix(filename: str, pattern: re.Pattern = _DEFAULT_RE) -> str:
    """Return the group key for *filename*.

    Falls back to the full cleaned stem if the pattern doesn't match.
    """
    stem = clean_stem(filename)
    m = pattern.match(stem)
    return m.group(1) if m else stem
