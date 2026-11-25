"""D3-primary measurement-Hellinger and secondary state-infidelity matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Sequence

import numpy as np

from .contracts import sha256_of


legacy_severity_matching_kind = "legacy_bracket_midpoint_v1"
evaluated_severity_matching_kind = "evaluated_midpoint_v2"


def validate_severity_matching_kind(severity_matching_kind: str) -> None:
    if type(severity_matching_kind) is not str or severity_matching_kind not in (
            legacy_severity_matching_kind, evaluated_severity_matching_kind):
        raise ValueError("unregistered severity_matching_kind")


def hellinger_squared(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    q = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    p /= max(p.sum(), 1e-15)
    q /= max(q.sum(), 1e-15)
    return float(np.clip(1.0 - np.sqrt(p * q).sum(), 0.0, 1.0))


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    a = np.asarray(rho, dtype=np.complex128)
    b = np.asarray(sigma, dtype=np.complex128)
    values, vectors = np.linalg.eigh(0.5 * (a + a.conj().T))
    sqrt_a = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.conj().T
    middle = sqrt_a @ b @ sqrt_a
    roots = np.sqrt(np.maximum(np.linalg.eigvalsh(0.5 * (middle + middle.conj().T)), 0.0))
    return float(np.clip(roots.sum() ** 2, 0.0, 1.0))


def match_hellinger_severity(clean_probability: np.ndarray,
                             channel: Callable[[float], np.ndarray], eta: float,
                             lower: float = 0.0, upper: float = 1.0,
                             tolerance: float = 1e-7, iterations: int = 80, *,
                             severity_matching_kind: str = legacy_severity_matching_kind) -> Dict[str, object]:
    validate_severity_matching_kind(severity_matching_kind)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0,1]")
    low, high = float(lower), float(upper)
    high_distance = hellinger_squared(clean_probability, channel(high))
    if high_distance + tolerance < eta:
        raise ValueError("channel severity bracket cannot reach requested Hellinger eta")
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        distance = hellinger_squared(clean_probability, channel(mid))
        if distance < eta:
            low = mid
        else:
            high = mid
        if abs(distance - eta) <= tolerance:
            if severity_matching_kind == evaluated_severity_matching_kind:
                # Return the evaluated point which met tolerance, not the next midpoint.
                low = high = mid
            break
    severity = 0.5 * (low + high)
    noisy = channel(severity)
    return {"physical_channel_parameter": severity,
            "eta_Hellinger": hellinger_squared(clean_probability, noisy),
            "measurement_noisy": np.asarray(noisy)}


@dataclass(frozen=True)
class D3HellingerFitRecord:
    channel_name: str
    fitted_parameter: float
    train_target_mean: float
    train_achieved_mean: float
    heldout_target_mean: float
    heldout_achieved_mean: float
    heldout_absolute_error: float
    train_sample_ids: tuple[str, ...]
    heldout_sample_ids: tuple[str, ...]
    fit_status: str
    fit_sha: str


def fit_d3_hellinger_parameter(
    *,
    channel_name: str,
    train_clean: Sequence[np.ndarray],
    train_target_eta: Sequence[float],
    heldout_clean: Sequence[np.ndarray],
    heldout_target_eta: Sequence[float],
    channel: Callable[[np.ndarray, float], np.ndarray],
    train_sample_ids: Sequence[str],
    heldout_sample_ids: Sequence[str],
    lower: float = 0.0,
    upper: float = 1.0,
) -> D3HellingerFitRecord:
    """Fit one physical channel parameter on train and audit held-out D3 error."""
    train_clean = tuple(np.asarray(value, dtype=np.float64) for value in train_clean)
    heldout_clean = tuple(np.asarray(value, dtype=np.float64) for value in heldout_clean)
    train_eta = np.asarray(train_target_eta, dtype=np.float64)
    heldout_eta = np.asarray(heldout_target_eta, dtype=np.float64)
    if (len(train_clean) != len(train_eta) or len(heldout_clean) != len(heldout_eta)
            or len(train_sample_ids) != len(train_eta)
            or len(heldout_sample_ids) != len(heldout_eta)
            or not len(train_eta) or not len(heldout_eta)):
        raise ValueError("D3 train/heldout samples, targets, and IDs must align and be non-empty")
    if np.any((train_eta < 0.0) | (train_eta > 1.0)) \
            or np.any((heldout_eta < 0.0) | (heldout_eta > 1.0)):
        raise ValueError("D3 target Hellinger distances must lie in [0,1]")
    if set(train_sample_ids) & set(heldout_sample_ids):
        raise ValueError("D3 train and held-out sample IDs must be disjoint")

    def achieved(clean_rows: Sequence[np.ndarray], parameter: float) -> np.ndarray:
        return np.asarray([
            hellinger_squared(clean, channel(clean, parameter)) for clean in clean_rows
        ], dtype=np.float64)

    def objective(parameter: float) -> float:
        residual = achieved(train_clean, parameter) - train_eta
        return float(np.mean(np.square(residual)))

    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
        parameter = float(result.x)
        fit_status = "converged" if bool(result.success) else "numerical_failure"
    except ImportError:
        grid = np.linspace(lower, upper, 1001)
        losses = np.asarray([objective(float(value)) for value in grid])
        parameter = float(grid[int(np.argmin(losses))])
        fit_status = "grid_fallback"

    train_achieved = achieved(train_clean, parameter)
    heldout_achieved = achieved(heldout_clean, parameter)
    record_without_sha = {
        "channel_name": channel_name,
        "fitted_parameter": parameter,
        "train_target_mean": float(train_eta.mean()),
        "train_achieved_mean": float(train_achieved.mean()),
        "heldout_target_mean": float(heldout_eta.mean()),
        "heldout_achieved_mean": float(heldout_achieved.mean()),
        "heldout_absolute_error": float(np.mean(np.abs(heldout_achieved - heldout_eta))),
        "train_sample_ids": tuple(train_sample_ids),
        "heldout_sample_ids": tuple(heldout_sample_ids),
        "fit_status": fit_status,
    }
    return D3HellingerFitRecord(**record_without_sha, fit_sha=sha256_of(record_without_sha))
