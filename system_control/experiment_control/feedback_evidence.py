"""Summaries for feedback evidence that affects hardware-safe auto-apply."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DIRECT_PHASE_KEYS = ("wch1_phase_correction_map", "phase_point_correction_map")
SENSITIVITY_KEYS = ("wch1_phase_sensitivity", "phase_point_sensitivity")


@dataclass(frozen=True)
class Wch1FeedbackEvidence:
    strict_hardware_mode: bool
    ok: bool
    evidence: str
    detail: str
    direct_map_shape: Optional[Tuple[int, ...]] = None
    sensitivity_shape: Optional[Tuple[int, ...]] = None


def summarize_wch1_feedback_evidence(
    context: Optional[Dict[str, Any]],
    system_type: str,
) -> Wch1FeedbackEvidence:
    """Describe whether strict W_CH1 phase feedback has hardware evidence."""

    context = dict(context or {})
    strict = bool(context.get("strict_hardware_mode", False))
    size = 16 if str(system_type).lower().startswith("4") else 15
    n_modes = 4 if size == 16 else 3

    direct_shape, direct_invalid = _first_matching_shape(
        _context_candidates(context, DIRECT_PHASE_KEYS),
        (size, size),
    )
    sensitivity_shape, sensitivity_invalid = _first_matching_shape(
        _context_candidates(context, SENSITIVITY_KEYS),
        (size, size, n_modes),
    )

    prefix = "strict on" if strict else "strict off"
    if direct_shape is not None:
        return Wch1FeedbackEvidence(
            strict_hardware_mode=strict,
            ok=True,
            evidence="direct_phase_map",
            detail=f"{prefix}; direct W_CH1 phase correction map present ({_shape_text(direct_shape)})",
            direct_map_shape=direct_shape,
            sensitivity_shape=sensitivity_shape,
        )
    if sensitivity_shape is not None:
        return Wch1FeedbackEvidence(
            strict_hardware_mode=strict,
            ok=True,
            evidence="sensitivity_tensor",
            detail=f"{prefix}; W_CH1 phase sensitivity tensor present ({_shape_text(sensitivity_shape)})",
            direct_map_shape=direct_shape,
            sensitivity_shape=sensitivity_shape,
        )

    invalid = direct_invalid + sensitivity_invalid
    invalid_note = f"; ignored invalid shapes: {', '.join(invalid[:4])}" if invalid else ""
    if strict:
        return Wch1FeedbackEvidence(
            strict_hardware_mode=True,
            ok=False,
            evidence="missing",
            detail=(
                "strict on; no direct W_CH1 phase map or sensitivity tensor present "
                f"(expected {size}x{size} or {size}x{size}x{n_modes})"
                f"{invalid_note}"
            ),
        )
    return Wch1FeedbackEvidence(
        strict_hardware_mode=False,
        ok=True,
        evidence="pattern_fallback_allowed",
        detail=f"strict off; W_CH1 pattern fallback allowed when direct evidence is absent{invalid_note}",
    )


def _context_candidates(context: Dict[str, Any], keys: Sequence[str]) -> Iterable[Any]:
    for key in keys:
        yield context.get(key)

    diagnosis = (
        context.get("classifier_diagnostics")
        or context.get("diagnosis")
        or context.get("quantum_diagnosis")
        or {}
    )
    if isinstance(diagnosis, dict):
        for key in keys:
            yield diagnosis.get(key)

    retrieval = context.get("retrieval_metrics", {}) or {}
    nested = retrieval.get("diagnostics") if isinstance(retrieval, dict) else None
    if isinstance(nested, dict):
        nested_diag = (
            nested.get("classifier_diagnostics")
            or nested.get("diagnosis")
            or nested.get("quantum_diagnosis")
            or {}
        )
        if isinstance(nested_diag, dict):
            for key in keys:
                yield nested_diag.get(key)


def _first_matching_shape(
    candidates: Iterable[Any],
    expected_shape: Tuple[int, ...],
) -> Tuple[Optional[Tuple[int, ...]], List[str]]:
    invalid: List[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            shape = tuple(np.asarray(candidate, dtype=float).shape)
        except Exception:
            invalid.append("unreadable")
            continue
        if shape == expected_shape:
            return shape, invalid
        invalid.append(_shape_text(shape) or "scalar")
    return None, invalid


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(value)) for value in shape)
