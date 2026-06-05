"""core/config_diff.py — Comparaison de configurations de pipeline."""

from __future__ import annotations

import json
from typing import Any

_MISSING = object()


def _normal(value: Any) -> Any:
    """Normalise les valeurs JSON-like avant comparaison."""
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    return value


def changed_step_ids(
    current_order: list[str],
    current_enabled: dict[str, bool],
    current_params: dict[str, dict],
    reference_order: list[str],
    reference_enabled: dict[str, bool],
    reference_params: dict[str, dict],
) -> set[str]:
    """Retourne les étapes différentes entre deux snapshots de pipeline."""
    return set(
        changed_step_details(
            current_order,
            current_enabled,
            current_params,
            reference_order,
            reference_enabled,
            reference_params,
        )
    )


def changed_step_details(
    current_order: list[str],
    current_enabled: dict[str, bool],
    current_params: dict[str, dict],
    reference_order: list[str],
    reference_enabled: dict[str, bool],
    reference_params: dict[str, dict],
    param_labels: dict[str, dict[str, str]] | None = None,
) -> dict[str, list[str]]:
    """Retourne les raisons de changement, groupées par étape."""
    common_step_ids = set(current_order) & set(reference_order)
    details: dict[str, list[str]] = {}

    current_common_order = [sid for sid in current_order if sid in common_step_ids]
    reference_common_order = [sid for sid in reference_order if sid in common_step_ids]
    cur_common_pos = {sid: i for i, sid in enumerate(current_common_order)}
    ref_common_pos = {sid: i for i, sid in enumerate(reference_common_order)}

    for sid in current_common_order:
        reasons: list[str] = []
        if cur_common_pos.get(sid) != ref_common_pos.get(sid):
            reasons.append(
                f"Ordre : position {ref_common_pos[sid] + 1} -> "
                f"position {cur_common_pos[sid] + 1}"
            )
        if current_enabled.get(sid, True) != reference_enabled.get(sid, True):
            reasons.append(
                "Activation : "
                f"{_format_enabled(reference_enabled.get(sid, True))} -> "
                f"{_format_enabled(current_enabled.get(sid, True))}"
            )

        cur_step_params = current_params.get(sid, {})
        ref_step_params = reference_params.get(sid, {})
        labels = (param_labels or {}).get(sid, {})
        for key in sorted(set(cur_step_params) | set(ref_step_params)):
            cur_val = cur_step_params.get(key, _MISSING)
            ref_val = ref_step_params.get(key, _MISSING)
            if cur_val is _MISSING or ref_val is _MISSING:
                continue
            if _normal(cur_val) == _normal(ref_val):
                continue
            label = labels.get(key, key)
            reasons.append(
                f"{label} : {_format_value(ref_val)} -> {_format_value(cur_val)}"
            )

        if reasons:
            details[sid] = reasons

    for sid in current_order:
        if sid in common_step_ids:
            continue
        if current_enabled.get(sid, True):
            details[sid] = ["Nouvelle étape activée"]

    return details


def _format_enabled(enabled: bool) -> str:
    return "activée" if enabled else "désactivée"


def _format_value(value: Any) -> str:
    if value is None:
        return "non défini"
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    else:
        text = str(value)
    if len(text) > 70:
        return text[:67] + "..."
    return text
