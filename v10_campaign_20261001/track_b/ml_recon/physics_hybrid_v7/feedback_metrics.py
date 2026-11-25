"""Strict count and calibration metrics for V7 feedback models.

The legacy V7 ``production_pipeline`` mixed operation-distribution
probabilities with a binary event ECE and used a Poisson-like expression under
an NB2 label.  This module keeps the statistical objects separate:

* count means are evaluated with exact Poisson or NB2 likelihoods;
* operation shares are evaluated with an exact conditional multinomial;
* binary calibration is defined only for explicit events such as ``C > 0``;
* temperature scaling always re-normalizes the operation distribution.

The functions are intentionally model-agnostic so both Model A and independent
baselines use the same evaluation contract.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from scipy.special import gammaln
    from scipy.stats import nbinom, poisson
except ImportError:  # pragma: no cover
    gammaln = None
    nbinom = None
    poisson = None


EPSILON = 1e-8


def _masked_mean(values: Any, mask: Optional[Any]) -> Any:
    if mask is None:
        return values.mean()
    weight = mask.to(dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def torch_nb2_log_probability(
    observed: Any,
    mean: Any,
    dispersion: Any,
) -> Any:
    """Exact NB2 log probability with ``Var[Y] = mu + alpha*mu**2``."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch NB2 likelihood requires torch")
    y = observed.to(dtype=mean.dtype)
    mu = mean.clamp_min(EPSILON)
    alpha = dispersion.to(dtype=mean.dtype).clamp_min(1e-7)
    while alpha.ndim < mu.ndim:
        alpha = alpha.unsqueeze(-1)
    shape = 1.0 / alpha
    return (
        torch.lgamma(y + shape)
        - torch.lgamma(shape)
        - torch.lgamma(y + 1.0)
        + shape * (torch.log(shape) - torch.log(shape + mu))
        + y * (torch.log(mu) - torch.log(shape + mu))
    )


def torch_nb2_nll(
    observed: Any,
    mean: Any,
    dispersion: Any,
    mask: Optional[Any] = None,
) -> Any:
    return _masked_mean(-torch_nb2_log_probability(observed, mean, dispersion), mask)


def torch_poisson_nll(
    observed: Any,
    mean: Any,
    mask: Optional[Any] = None,
) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch Poisson likelihood requires torch")
    y = observed.to(dtype=mean.dtype)
    mu = mean.clamp_min(EPSILON)
    values = mu - y * torch.log(mu) + torch.lgamma(y + 1.0)
    return _masked_mean(values, mask)


def torch_conditional_multinomial_nll(
    observed: Any,
    probability: Any,
    mask: Optional[Any] = None,
) -> Any:
    """Exact conditional multinomial NLL, including combinatorial terms."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch multinomial likelihood requires torch")
    y = observed.to(dtype=probability.dtype)
    weight = torch.ones_like(y) if mask is None else mask.to(dtype=y.dtype)
    masked_counts = y * weight
    probability = probability.clamp_min(EPSILON) * weight
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(EPSILON)
    total = masked_counts.sum(dim=-1)
    log_probability = (
        torch.lgamma(total + 1.0)
        - torch.lgamma(masked_counts + 1.0).sum(dim=-1)
        + (masked_counts * torch.log(probability.clamp_min(EPSILON))).sum(dim=-1)
    )
    valid = (weight.sum(dim=-1) > 0) & (total > 0)
    return _masked_mean(-log_probability, valid)


def torch_density_fidelity(predicted: Any, target: Any) -> Any:
    """Uhlmann fidelity for small complex density matrices."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("density fidelity requires torch")
    predicted = 0.5 * (predicted + predicted.mH)
    target = 0.5 * (target + target.mH)
    target_values, target_vectors = torch.linalg.eigh(target)
    target_values = target_values.real.clamp_min(0.0)
    sqrt_target = (
        target_vectors
        * torch.sqrt(target_values).unsqueeze(-2)
    ) @ target_vectors.mH
    middle = sqrt_target @ predicted @ sqrt_target
    middle = 0.5 * (middle + middle.mH)
    middle_values = torch.linalg.eigvalsh(middle).real.clamp_min(0.0)
    return torch.square(torch.sqrt(middle_values).sum(dim=-1)).clamp(0.0, 1.0)


def torch_masked_density_infidelity(
    predicted: Any,
    target: Any,
    available_mask: Any,
) -> Any:
    fidelity = torch_density_fidelity(predicted, target)
    return _masked_mean(1.0 - fidelity, available_mask)


def categorical_temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probability must have shape [sample, category]")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probability must be finite and non-negative")
    row_sum = values.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("every probability row must have positive mass")
    normalized = values / row_sum
    logits = np.log(np.clip(normalized, EPSILON, 1.0)) / max(float(temperature), 1e-4)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def exact_conditional_multinomial_nll(
    observed: np.ndarray,
    probability: np.ndarray,
) -> float:
    _require_scipy()
    counts = np.asarray(observed, dtype=np.float64)
    values = categorical_temperature_scale(probability, 1.0)
    if counts.shape != values.shape or np.any(counts < 0.0):
        raise ValueError("observed and probability must have matching non-negative shapes")
    totals = counts.sum(axis=1)
    log_probability = (
        gammaln(totals + 1.0)
        - gammaln(counts + 1.0).sum(axis=1)
        + (counts * np.log(np.clip(values, EPSILON, 1.0))).sum(axis=1)
    )
    valid = totals > 0.0
    return float(np.mean(-log_probability[valid])) if np.any(valid) else float("nan")


def fit_categorical_temperature(
    probability: np.ndarray,
    observed: np.ndarray,
    *,
    minimum: float = 0.25,
    maximum: float = 8.0,
    grid_size: int = 96,
) -> Tuple[float, float]:
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in np.geomspace(float(minimum), float(maximum), int(grid_size)):
        scaled = categorical_temperature_scale(probability, float(temperature))
        value = exact_conditional_multinomial_nll(observed, scaled)
        if value < best_nll:
            best_temperature = float(temperature)
            best_nll = float(value)
    return best_temperature, best_nll


def _nb2_numpy_log_probability(
    observed: np.ndarray,
    mean: np.ndarray,
    dispersion: np.ndarray,
) -> np.ndarray:
    _require_scipy()
    y, mu, alpha = np.broadcast_arrays(
        np.asarray(observed, dtype=np.float64),
        np.asarray(mean, dtype=np.float64),
        np.asarray(dispersion, dtype=np.float64),
    )
    mu = np.clip(mu, EPSILON, None)
    alpha = np.clip(alpha, 0.0, None)
    poisson_mask = alpha <= 1e-7
    result = y * np.log(mu) - mu - gammaln(y + 1.0)
    if np.any(~poisson_mask):
        shape = 1.0 / alpha[~poisson_mask]
        selected_mean = mu[~poisson_mask]
        selected_y = y[~poisson_mask]
        result[~poisson_mask] = (
            gammaln(selected_y + shape)
            - gammaln(shape)
            - gammaln(selected_y + 1.0)
            + shape * (np.log(shape) - np.log(shape + selected_mean))
            + selected_y * (np.log(selected_mean) - np.log(shape + selected_mean))
        )
    return result


def _binary_calibration(
    event: np.ndarray,
    predicted_probability: np.ndarray,
    *,
    bins: int,
) -> Dict[str, float]:
    outcome = np.asarray(event, dtype=np.float64).reshape(-1)
    probability = np.clip(np.asarray(predicted_probability, dtype=np.float64).reshape(-1), EPSILON, 1.0 - EPSILON)
    if outcome.shape != probability.shape:
        raise ValueError("event and predicted_probability shapes differ")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for index in range(int(bins)):
        if index == int(bins) - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(outcome[selected])) - float(np.mean(probability[selected]))
            )
    intercept, slope = _logistic_calibration_fit(outcome, probability)
    return {
        "brier": float(np.mean(np.square(outcome - probability))),
        "ece": float(ece),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "event_rate": float(np.mean(outcome)),
        "predicted_event_rate": float(np.mean(probability)),
    }


def _logistic_calibration_fit(
    outcome: np.ndarray,
    probability: np.ndarray,
) -> Tuple[float, float]:
    logit = np.log(probability) - np.log1p(-probability)
    design = np.stack((np.ones_like(logit), logit), axis=1)
    coefficient = np.asarray([0.0, 1.0], dtype=np.float64)
    ridge = np.diag([1e-8, 1e-6])
    for _ in range(32):
        linear = np.clip(design @ coefficient, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weight = np.clip(fitted * (1.0 - fitted), 1e-6, None)
        gradient = design.T @ (outcome - fitted) - ridge @ coefficient
        information = (design.T * weight) @ design + ridge
        step = np.linalg.solve(information, gradient)
        coefficient += step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return float(coefficient[0]), float(coefficient[1])


def _event_probabilities(mean: np.ndarray, dispersion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu, alpha = np.broadcast_arrays(
        np.asarray(mean, dtype=np.float64),
        np.asarray(dispersion, dtype=np.float64),
    )
    poisson_mask = alpha <= 1e-7
    probability_zero = np.exp(-mu)
    probability_one = probability_zero * mu
    if np.any(~poisson_mask):
        selected_alpha = alpha[~poisson_mask]
        selected_mean = mu[~poisson_mask]
        shape = 1.0 / selected_alpha
        probability_zero[~poisson_mask] = np.power(
            shape / (shape + selected_mean), shape,
        )
        probability_one[~poisson_mask] = probability_zero[~poisson_mask] * (
            selected_mean / (1.0 + selected_alpha * selected_mean)
        )
    return 1.0 - probability_zero, 1.0 - probability_zero - probability_one


def standard_count_calibration_report(
    observed: np.ndarray,
    mean: np.ndarray,
    dispersion: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    bins: int = 10,
    interval_mass: float = 0.90,
    seed: int = 0,
) -> Dict[str, Any]:
    """Return exact NLL, binary calibration, randomized PIT and coverage."""

    _require_scipy()
    y, mu, alpha = np.broadcast_arrays(
        np.asarray(observed, dtype=np.float64),
        np.asarray(mean, dtype=np.float64),
        np.asarray(dispersion, dtype=np.float64),
    )
    valid = np.ones_like(y, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    valid &= np.isfinite(y) & np.isfinite(mu) & np.isfinite(alpha) & (y >= 0.0) & (mu > 0.0) & (alpha >= 0.0)
    y_flat = y[valid]
    mu_flat = mu[valid]
    alpha_flat = alpha[valid]
    if y_flat.size == 0:
        raise ValueError("no valid count observations")
    log_probability = _nb2_numpy_log_probability(y_flat, mu_flat, alpha_flat)
    event_positive, event_two = _event_probabilities(mu_flat, alpha_flat)
    lower_probability = (1.0 - float(interval_mass)) / 2.0
    upper_probability = 1.0 - lower_probability
    lower = np.zeros_like(mu_flat)
    upper = np.zeros_like(mu_flat)
    cdf_lower = np.zeros_like(mu_flat)
    cdf_at = np.zeros_like(mu_flat)
    poisson_mask = alpha_flat <= 1e-7
    if np.any(poisson_mask):
        lower[poisson_mask] = poisson.ppf(lower_probability, mu_flat[poisson_mask])
        upper[poisson_mask] = poisson.ppf(upper_probability, mu_flat[poisson_mask])
        cdf_lower[poisson_mask] = poisson.cdf(y_flat[poisson_mask] - 1.0, mu_flat[poisson_mask])
        cdf_at[poisson_mask] = poisson.cdf(y_flat[poisson_mask], mu_flat[poisson_mask])
    if np.any(~poisson_mask):
        shape = 1.0 / alpha_flat[~poisson_mask]
        probability = shape / (shape + mu_flat[~poisson_mask])
        lower[~poisson_mask] = nbinom.ppf(lower_probability, shape, probability)
        upper[~poisson_mask] = nbinom.ppf(upper_probability, shape, probability)
        cdf_lower[~poisson_mask] = nbinom.cdf(y_flat[~poisson_mask] - 1.0, shape, probability)
        cdf_at[~poisson_mask] = nbinom.cdf(y_flat[~poisson_mask], shape, probability)
    rng = np.random.default_rng(int(seed))
    randomized_pit = cdf_lower + rng.random(y_flat.size) * np.clip(cdf_at - cdf_lower, 0.0, 1.0)
    pit_histogram, _ = np.histogram(randomized_pit, bins=int(bins), range=(0.0, 1.0))
    expected_bin = y_flat.size / float(bins)
    pit_l1 = float(np.sum(np.abs(pit_histogram - expected_bin)) / max(y_flat.size, 1))
    return {
        "sample_count": int(y_flat.size),
        "nb2_nll": float(-np.mean(log_probability)),
        "poisson_nll": float(-np.mean(_nb2_numpy_log_probability(y_flat, mu_flat, np.zeros_like(alpha_flat)))),
        "mean_observed": float(np.mean(y_flat)),
        "mean_predicted": float(np.mean(mu_flat)),
        "interval_mass": float(interval_mass),
        "interval_coverage": float(np.mean((y_flat >= lower) & (y_flat <= upper))),
        "interval_mean_width": float(np.mean(upper - lower)),
        "randomized_pit_mean": float(np.mean(randomized_pit)),
        "randomized_pit_variance": float(np.var(randomized_pit)),
        "randomized_pit_l1": pit_l1,
        "randomized_pit_histogram": pit_histogram.astype(int).tolist(),
        "event_c_gt_0": _binary_calibration(y_flat > 0.0, event_positive, bins=int(bins)),
        "event_c_ge_2": _binary_calibration(y_flat >= 2.0, event_two, bins=int(bins)),
    }


def probability_invariants(probability: np.ndarray, *, atol: float = 1e-6) -> Mapping[str, bool]:
    values = np.asarray(probability, dtype=np.float64)
    return {
        "finite": bool(np.all(np.isfinite(values))),
        "nonnegative": bool(np.all(values >= -float(atol))),
        "row_sum_one": bool(values.ndim == 2 and np.allclose(values.sum(axis=1), 1.0, atol=float(atol))),
    }


def _require_scipy() -> None:
    if gammaln is None or nbinom is None or poisson is None:  # pragma: no cover
        raise RuntimeError("strict V7 calibration metrics require scipy")


__all__ = [
    "categorical_temperature_scale",
    "exact_conditional_multinomial_nll",
    "fit_categorical_temperature",
    "probability_invariants",
    "standard_count_calibration_report",
    "torch_conditional_multinomial_nll",
    "torch_density_fidelity",
    "torch_masked_density_infidelity",
    "torch_nb2_log_probability",
    "torch_nb2_nll",
    "torch_poisson_nll",
]
