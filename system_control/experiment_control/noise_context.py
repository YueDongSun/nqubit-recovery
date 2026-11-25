"""Feature-schema-v2 noise context with explicit missing-value masks.

There is deliberately no tomography ``g_si`` feature in this module.  An
external DLCZ source reference may be recorded as provenance only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .pair_click_noise import PairClickNoiseResult, fit_pair_click_noise


NOISE_CONTEXT_SCHEMA_VERSION = 2


@dataclass
class NoiseContext:
    schema_version: int = NOISE_CONTEXT_SCHEMA_VERSION
    multi_click_counts: Dict[str, int] = field(default_factory=dict)
    multi_click_fractions: Dict[str, float | None] = field(default_factory=dict)
    multi2_pair_counts: Dict[str, int] = field(default_factory=dict)
    multi2_pair_fractions: Dict[str, float | None] = field(default_factory=dict)
    pair_model: Dict[str, Any] = field(default_factory=dict)
    totals: Dict[str, int] = field(default_factory=dict)
    point_variance_quantiles: Dict[str, float | None] = field(default_factory=dict)
    point_background_quantiles: Dict[str, float | None] = field(default_factory=dict)
    point_leakage_quantiles: Dict[str, float | None] = field(default_factory=dict)
    retrieval_statistics: Dict[str, float | None] = field(default_factory=dict)
    calibration_confidence_statistics: Dict[str, float | None] = field(default_factory=dict)
    external_source_characterization: Dict[str, Any] = field(default_factory=dict)
    missing_value_masks: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_noise_context(
    snapshot: Any,
    *,
    point_variance: Optional[np.ndarray] = None,
    point_background: Optional[np.ndarray] = None,
    point_leakage: Optional[np.ndarray] = None,
    calibration_confidence: Optional[np.ndarray] = None,
    retrieval_rates: Optional[np.ndarray] = None,
    pair_model_mode: str = "disabled",
    external_dlcz_gsi_reference: Optional[float] = None,
) -> NoiseContext:
    diagnostics = getattr(snapshot, "diagnostics", {}) or {}
    multi = {
        name: diagnostics.get(f"{name}_count", getattr(snapshot, f"{name}_count", None))
        for name in ("multi2", "multi3", "multi4")
    }
    multi_counts = {name: int(value) if value is not None else 0 for name, value in multi.items()}
    total_multi = sum(multi_counts.values())
    pairs = diagnostics.get("multi2_pairs") or getattr(snapshot, "multi2_pair_counts", None) or {}
    pair_counts = {name: int(pairs.get(name, 0)) for name in ("12", "13", "14", "23", "24", "34")}
    pair_total = sum(pair_counts.values())
    model = fit_pair_click_noise(pair_counts, mode=pair_model_mode)
    totals = {
        "signal": int(np.sum(getattr(snapshot, "signal_bins", np.asarray([])))),
        "idler": int(np.sum(getattr(snapshot, "idler_bins", np.asarray([])))),
        "coin": int(np.sum(getattr(snapshot, "coin_bins", np.asarray([])))),
    }
    return NoiseContext(
        multi_click_counts=multi_counts,
        multi_click_fractions={name: (value / total_multi if total_multi else None) for name, value in multi_counts.items()},
        multi2_pair_counts=pair_counts,
        multi2_pair_fractions={name: (value / pair_total if pair_total else None) for name, value in pair_counts.items()},
        pair_model=model.to_json_dict(),
        totals=totals,
        point_variance_quantiles=_quantiles(point_variance),
        point_background_quantiles=_quantiles(point_background),
        point_leakage_quantiles=_quantiles(point_leakage),
        retrieval_statistics=_quantiles(retrieval_rates),
        calibration_confidence_statistics=_quantiles(calibration_confidence),
        external_source_characterization=(
            {"external_dlcz_gsi_reference": float(external_dlcz_gsi_reference), "measured_per_tomography_run": False}
            if external_dlcz_gsi_reference is not None else {}
        ),
        missing_value_masks={
            "point_variance": point_variance is None,
            "point_background": point_background is None,
            "point_leakage": point_leakage is None,
            "retrieval_rates": retrieval_rates is None,
            "calibration_confidence": calibration_confidence is None,
            "pair_counts": not bool(pairs),
            "external_source_characterization": external_dlcz_gsi_reference is None,
        },
    )


def _quantiles(values: Optional[np.ndarray]) -> Dict[str, float | None]:
    if values is None:
        return {"q05": None, "q50": None, "q95": None, "mean": None}
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"q05": None, "q50": None, "q95": None, "mean": None}
    return {
        "q05": float(np.quantile(array, 0.05)), "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)), "mean": float(np.mean(array)),
    }
