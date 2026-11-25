"""Convert grouped bootstrap fits into unit-bearing coefficient posteriors."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .contracts import PhysicalCoefficientPosteriorV1


def build_coefficient_posteriors(
    bootstrap_coefficients: Any,
    *,
    coefficient_names: Sequence[str],
    units: Sequence[str],
    scope: str,
    evidence: Mapping[str, Mapping[str, float]] | None = None,
    confounded_with: Mapping[str, Sequence[str]] | None = None,
    inclusion_threshold: float = 0.0,
    provenance: str = "grouped_bootstrap_heldout_likelihood",
) -> List[PhysicalCoefficientPosteriorV1]:
    values = np.asarray(bootstrap_coefficients, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(coefficient_names) or len(units) != len(coefficient_names):
        raise ValueError("bootstrap coefficients, names, and units disagree")
    evidence = evidence or {}
    confounded_with = confounded_with or {}
    result = []
    for index, name in enumerate(coefficient_names):
        column = values[:, index]
        nonzero = np.abs(column) > float(inclusion_threshold)
        median = float(np.median(column))
        nonzero_sign = np.sign(column[nonzero])
        sign_stability = 0.0 if nonzero_sign.size == 0 else float(
            max(np.mean(nonzero_sign > 0), np.mean(nonzero_sign < 0))
        )
        item_evidence = dict(evidence.get(str(name), {}))
        confounded = tuple(str(value) for value in confounded_with.get(str(name), ()))
        inclusion = float(np.mean(nonzero))
        result.append(PhysicalCoefficientPosteriorV1(
            coefficient_name=str(name),
            scope=str(scope),
            value=median,
            std=float(np.std(column, ddof=1)) if column.size > 1 else 0.0,
            unit=str(units[index]),
            bootstrap_inclusion_probability=inclusion,
            sign_stability=sign_stability,
            heldout_delta_nll=float(item_evidence.get("heldout_delta_nll", 0.0)),
            intervention_score=float(item_evidence.get("intervention_score", 0.0)),
            probe_score=float(item_evidence.get("probe_score", 0.0)),
            identifiable=bool(inclusion >= 0.5 and sign_stability >= 0.7 and not confounded),
            confounded_with=confounded,
            provenance=provenance,
        ))
    return result


__all__ = ["build_coefficient_posteriors"]
