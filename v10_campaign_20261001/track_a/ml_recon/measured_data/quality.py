"""Robust measured-count quality diagnostics and hard-fault isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


HARD_FAULT_MULTIPLIER = 10.0
SATURATION_LIMIT = 2**32 - 1
ROBUST_Z_SCALE = 0.6744897501960817


@dataclass(frozen=True)
class QualityDiagnostics:
    quality_status: str
    exclusion_reason: str | None
    hard_fault_kind: str | None
    signal_robust_z_score: float
    idler_robust_z_score: float
    saturation_flag: bool
    low_low_collapse_flag: bool
    signal_high_idler_low_flag: bool


def robust_median_and_mad(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("quality reference values must be finite and non-empty")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, max(mad, 1.0)


def classify_count_quality(
    signal_total: int,
    idler_total: int,
    *,
    signal_reference: Sequence[float],
    idler_reference: Sequence[float],
) -> QualityDiagnostics:
    """Classify only the audited hard faults; retain ordinary noise."""
    signal_median, signal_mad = robust_median_and_mad(signal_reference)
    idler_median, idler_mad = robust_median_and_mad(idler_reference)
    signal_z = ROBUST_Z_SCALE * (float(signal_total) - signal_median) / signal_mad
    idler_z = ROBUST_Z_SCALE * (float(idler_total) - idler_median) / idler_mad
    signal_high = float(signal_total) > HARD_FAULT_MULTIPLIER * signal_median
    idler_low = float(idler_total) < idler_median / HARD_FAULT_MULTIPLIER
    saturation = max(int(signal_total), int(idler_total)) >= SATURATION_LIMIT
    low_low = (
        float(signal_total) < signal_median / HARD_FAULT_MULTIPLIER
        and idler_low
    )
    signal_high_idler_low = signal_high and idler_low
    hard_fault_kind: str | None = None
    if signal_high_idler_low:
        hard_fault_kind = "signal_high_idler_low"
    elif signal_high:
        hard_fault_kind = "signal_high"
    elif idler_low:
        hard_fault_kind = "idler_low"
    return QualityDiagnostics(
        quality_status="excluded_hard_fault" if hard_fault_kind else "trusted",
        exclusion_reason=(f"hard_fault:{hard_fault_kind}" if hard_fault_kind else None),
        hard_fault_kind=hard_fault_kind,
        signal_robust_z_score=float(signal_z),
        idler_robust_z_score=float(idler_z),
        saturation_flag=bool(saturation),
        low_low_collapse_flag=bool(low_low),
        signal_high_idler_low_flag=bool(signal_high_idler_low),
    )


def classify_population(
    totals: Iterable[tuple[int, int]],
) -> List[QualityDiagnostics]:
    values = list(totals)
    signal = [item[0] for item in values]
    idler = [item[1] for item in values]
    return [
        classify_count_quality(
            signal_total,
            idler_total,
            signal_reference=signal,
            idler_reference=idler,
        )
        for signal_total, idler_total in values
    ]


__all__ = [
    "HARD_FAULT_MULTIPLIER",
    "QualityDiagnostics",
    "classify_count_quality",
    "classify_population",
    "robust_median_and_mad",
]
