"""Rate-exposure Poisson/NB2 likelihoods and lightweight hierarchical offsets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


EPSILON = 1e-12


def poisson_nll(count: Any, mean: Any, *, reduction: str = "mean") -> Any:
    y = np.asarray(count, dtype=float)
    mu = np.maximum(np.asarray(mean, dtype=float), EPSILON)
    value = mu - y * np.log(mu) + np.vectorize(math.lgamma)(y + 1.0)
    return _reduce(value, reduction)


def nb2_nll(
    count: Any,
    mean: Any,
    dispersion: Any,
    *,
    reduction: str = "mean",
) -> Any:
    """Negative-binomial NLL with Var[Y] = mu + alpha * mu^2."""
    y = np.asarray(count, dtype=float)
    mu = np.maximum(np.asarray(mean, dtype=float), EPSILON)
    alpha = np.maximum(np.asarray(dispersion, dtype=float), 0.0)
    y, mu, alpha = np.broadcast_arrays(y, mu, alpha)
    poisson_mask = alpha <= 1e-10
    size = 1.0 / np.maximum(alpha, 1e-10)
    log_probability = (
        np.vectorize(math.lgamma)(y + size)
        - np.vectorize(math.lgamma)(size)
        - np.vectorize(math.lgamma)(y + 1.0)
        + size * (np.log(size) - np.log(size + mu))
        + y * (np.log(mu) - np.log(size + mu))
    )
    value = -log_probability
    if np.any(poisson_mask):
        poisson_value = np.asarray(poisson_nll(y, mu, reduction="none"))
        value = np.where(poisson_mask, poisson_value, value)
    return _reduce(value, reduction)


def rate_exposure_mean(log_rate: Any, exposure_s: Any) -> np.ndarray:
    exposure = np.maximum(np.asarray(exposure_s, dtype=float), 0.0)
    return np.exp(np.asarray(log_rate, dtype=float)) * exposure


def nb2_fisher_log_rate(mean: Any, dispersion: Any) -> np.ndarray:
    mu = np.maximum(np.asarray(mean, dtype=float), 0.0)
    alpha = np.maximum(np.asarray(dispersion, dtype=float), 0.0)
    return mu / (1.0 + alpha * mu)


def _reduce(value: np.ndarray, reduction: str) -> Any:
    if reduction == "none":
        return value
    if reduction == "sum":
        return float(np.sum(value))
    if reduction == "mean":
        return float(np.mean(value))
    raise ValueError(f"unsupported reduction: {reduction}")


@dataclass(frozen=True)
class HierarchicalNB2Result:
    global_log_rate: float
    group_log_rate_offsets: np.ndarray
    dispersion: float
    nll: float


def fit_hierarchical_intercept(
    count: Any,
    exposure_s: Any,
    group_index: Any,
    *,
    shrinkage: float = 10.0,
) -> HierarchicalNB2Result:
    y = np.asarray(count, dtype=float).reshape(-1)
    exposure = np.maximum(np.asarray(exposure_s, dtype=float).reshape(-1), EPSILON)
    groups = np.asarray(group_index, dtype=int).reshape(-1)
    if not (y.size == exposure.size == groups.size):
        raise ValueError("count, exposure, and group index must match")
    global_rate = float(np.sum(y) / np.sum(exposure))
    offsets = np.zeros(int(groups.max(initial=-1)) + 1, dtype=float)
    residuals = []
    for group in range(offsets.size):
        mask = groups == group
        group_count = float(np.sum(y[mask]))
        group_exposure = float(np.sum(exposure[mask]))
        rate = (group_count + shrinkage * global_rate) / max(group_exposure + shrinkage, EPSILON)
        offsets[group] = math.log(max(rate, EPSILON)) - math.log(max(global_rate, EPSILON))
        residuals.extend((y[mask] - rate * exposure[mask]).tolist())
    mean_count = max(float(np.mean(y)), EPSILON)
    variance = float(np.var(residuals)) if residuals else mean_count
    dispersion = max(0.0, (variance - mean_count) / (mean_count * mean_count))
    log_rate = math.log(max(global_rate, EPSILON)) + offsets[groups]
    mean = rate_exposure_mean(log_rate, exposure)
    return HierarchicalNB2Result(
        global_log_rate=math.log(max(global_rate, EPSILON)),
        group_log_rate_offsets=offsets,
        dispersion=dispersion,
        nll=float(nb2_nll(y, mean, dispersion)),
    )


__all__ = [
    "HierarchicalNB2Result",
    "fit_hierarchical_intercept",
    "nb2_fisher_log_rate",
    "nb2_nll",
    "poisson_nll",
    "rate_exposure_mean",
]
