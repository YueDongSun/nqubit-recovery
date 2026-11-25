"""Split-conformal residual calibration on an independent calibration split."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


DEFAULT_COVERAGES = (0.50, 0.90, 0.95)


def conformal_quantiles(
    calibration_residuals: Any,
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> Dict[str, float]:
    residuals = np.abs(np.asarray(calibration_residuals, dtype=float).reshape(-1))
    if residuals.size == 0:
        raise ValueError("conformal calibration split must be non-empty")
    result = {}
    count = residuals.size
    ordered = np.sort(residuals)
    for coverage in coverages:
        if not 0.0 < float(coverage) < 1.0:
            raise ValueError("coverage must lie in (0, 1)")
        rank = min(int(np.ceil((count + 1) * float(coverage))) - 1, count - 1)
        result[f"{int(round(100 * coverage))}"] = float(ordered[max(rank, 0)])
    return result


def conformal_report(
    calibration_residuals: Any,
    evaluation_residuals: Any,
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> Dict[str, Any]:
    thresholds = conformal_quantiles(calibration_residuals, coverages=coverages)
    evaluation = np.abs(np.asarray(evaluation_residuals, dtype=float).reshape(-1))
    if evaluation.size == 0:
        raise ValueError("conformal evaluation split must be non-empty")
    empirical = {}
    sharpness = {}
    errors = {}
    for coverage in coverages:
        key = f"{int(round(100 * coverage))}"
        threshold = thresholds[key]
        observed = float(np.mean(evaluation <= threshold))
        empirical[key] = observed
        sharpness[key] = 2.0 * threshold
        errors[key] = observed - float(coverage)
    return {
        "method": "split_conformal_absolute_residual",
        "quantile_thresholds": thresholds,
        "coverage": empirical,
        "sharpness": sharpness,
        "calibration_error": errors,
        "calibration_sample_count": int(np.asarray(calibration_residuals).size),
        "evaluation_sample_count": int(evaluation.size),
    }


__all__ = ["DEFAULT_COVERAGES", "conformal_quantiles", "conformal_report"]
