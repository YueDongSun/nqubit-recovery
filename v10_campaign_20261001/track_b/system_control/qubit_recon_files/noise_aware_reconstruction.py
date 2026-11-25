"""Noise-aware, signed-estimator Monte Carlo tomography.

The legacy reconstruction remains the default production path.  This module is
used in shadow mode initially and never clips signed operation estimates before
the inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from ml_recon.operation_point_physics import AODPointResponseModel
from system_control.experiment_control.operation_calibration import OperationCalibrationStore
from system_control.experiment_control.operation_points import OperationPointSpec, build_operation_registry
from system_control.qubit_recon_files.system_specs import get_spec


NOISE_SAMPLING_MODES = {"gaussian", "poisson", "hybrid"}
PROJECTION_MODES = {"legacy_basis", "hardware_response"}


def signed_operation_estimator(coin: Iterable[float], signal: Iterable[float], idler: Iterable[float], read_count: float) -> tuple[np.ndarray, np.ndarray]:
    """Return y=coin/signal-idler/read and count-propagated heteroscedastic variance."""
    c = np.asarray(coin, dtype=float).reshape(-1)
    s = np.asarray(signal, dtype=float).reshape(-1)
    i = np.asarray(idler, dtype=float).reshape(-1)
    read = max(float(read_count), 1.0)
    safe_signal = np.maximum(s, 1.0)
    y = c / safe_signal - i / read
    # Delta method under independent Poisson counters.  Invalid/no-signal
    # points receive a large finite uncertainty rather than being clipped.
    variance = c / safe_signal ** 2 + c ** 2 / safe_signal ** 3 + i / read ** 2
    variance = np.maximum(variance, 1e-12)
    variance = np.where(s > 0, variance, np.maximum(variance, 1.0))
    return y, variance


def reconstruct_noise_aware(
    system_type: str,
    *,
    signal_bins: Optional[Iterable[float]] = None,
    idler_bins: Optional[Iterable[float]] = None,
    coin_bins: Optional[Iterable[float]] = None,
    read_count: Optional[float] = None,
    signed_values: Optional[Iterable[float]] = None,
    estimator_variance: Optional[Iterable[float]] = None,
    registry: Optional[Sequence[OperationPointSpec]] = None,
    calibration: Optional[OperationCalibrationStore] = None,
    point_model: Optional[AODPointResponseModel] = None,
    noise_sampling_mode: str = "gaussian",
    projection_mode: str = "legacy_basis",
    mc_samples: int = 32,
    regularization: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Reconstruct a PSD, trace-one W-subspace state with MC uncertainty.

    With ``mc_samples=0``, uniform variance, no calibration, and non-negative
    values, this uses the same normalized linear inversion as the baseline.
    """
    if noise_sampling_mode not in NOISE_SAMPLING_MODES:
        raise ValueError(f"noise_sampling_mode must be one of {sorted(NOISE_SAMPLING_MODES)}")
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"projection_mode must be one of {sorted(PROJECTION_MODES)}")
    spec = get_spec(system_type)
    n_ops = int(spec["total_operations"])
    if signed_values is None:
        if signal_bins is None or idler_bins is None or coin_bins is None or read_count is None:
            raise ValueError("provide signed_values or signal_bins/idler_bins/coin_bins/read_count")
        values, propagated_variance = signed_operation_estimator(coin_bins, signal_bins, idler_bins, read_count)
    else:
        values = np.asarray(signed_values, dtype=float).reshape(-1)
        propagated_variance = np.ones(values.shape, dtype=float) if estimator_variance is None else np.asarray(estimator_variance, dtype=float).reshape(-1)
    values = _fit_length(values, n_ops)
    propagated_variance = np.maximum(_fit_length(propagated_variance, n_ops), 1e-12)
    registry = list(registry) if registry is not None else build_operation_registry(system_type)
    if len(registry) != n_ops:
        raise ValueError(f"registry has {len(registry)} points, expected {n_ops}")
    model = point_model or AODPointResponseModel()
    responses = [model.evaluate(point, calibration) for point in registry]
    if projection_mode == "legacy_basis":
        operators = np.asarray([_legacy_basis_projector(point) for point in registry], dtype=complex)
    else:
        operators = np.asarray([response.measurement_projector for response in responses], dtype=complex)
    projected, normal_matrix, condition = _invert(values, operators, propagated_variance, regularization)
    rho_linear = projected[0]
    rho_physical = _physical(rho_linear)
    samples = [rho_physical]
    random = rng or np.random.default_rng(20260710)
    for _ in range(max(0, int(mc_samples))):
        draw = _sample_values(values, propagated_variance, noise_sampling_mode, random)
        rho_draw, _, _ = _invert(draw, operators, propagated_variance, regularization)
        samples.append(_physical(rho_draw[0]))
    posterior = np.asarray(samples)
    mean = _physical(np.mean(posterior, axis=0))
    median = _physical(np.median(posterior.real, axis=0) + 1j * np.median(posterior.imag, axis=0))
    fidelities = np.asarray([_w_fidelity(rho) for rho in posterior], dtype=float)
    leakage_samples = _leakage_samples(responses, len(posterior), random)
    adjusted = fidelities * (1.0 - leakage_samples)
    full = _embed_full(system_type, rho_physical)
    full_mean = _embed_full(system_type, mean)
    full_median = _embed_full(system_type, median)
    weights = 1.0 / propagated_variance
    return {
        "system_type": system_type,
        "rho_linear": rho_linear,
        "rho_physical": rho_physical,
        "rho_full": full,
        "posterior_mean_rho": mean,
        "posterior_mean_rho_full": full_mean,
        "posterior_median_rho": median,
        "posterior_median_rho_full": full_median,
        "elementwise_std": np.std(posterior, axis=0),
        "signed_estimators": values,
        "estimator_variance": propagated_variance,
        "conditional_fidelity_median": float(np.median(fidelities)),
        "conditional_fidelity_ci": _interval(fidelities),
        "leakage_estimate_median": float(np.median(leakage_samples)),
        "leakage_ci": _interval(leakage_samples),
        "leakage_adjusted_fidelity_median": float(np.median(adjusted)),
        "leakage_adjusted_fidelity_ci": _interval(adjusted),
        "condition_number": float(condition),
        "effective_rank": int(np.linalg.matrix_rank(normal_matrix)),
        "weight_source": "count_propagation",
        "calibration_source": "operation_calibration" if calibration is not None else "missing_neutral",
        "phase_sign": -1.0,
        "basis_phase_convention": "read_aod_negative_order",
        "projection_mode": projection_mode,
        "efficiency_semantics": "probability_efficiency_converted_to_field_amplitude_with_sqrt",
        "point_diagnostics": {
            "efficiency": [response.expected_point_efficiency for response in responses],
            "background": [response.expected_background_rate for response in responses],
            "leakage": [response.expected_leakage_rate for response in responses],
            "variance": propagated_variance,
            "phase_residual": [
                (getattr(calibration.operation(index), "phase_residual_rad", []) if calibration is not None else [])
                for index in range(n_ops)
            ],
            "amplitude_residual": [
                (getattr(calibration.operation(index), "amplitude_residual", []) if calibration is not None else [])
                for index in range(n_ops)
            ],
            "calibration_confidence": [response.uncertainty_diagnostics.get("calibration_confidence_mean") for response in responses],
            "model_uncertainty": np.sqrt(propagated_variance),
        },
        "diagnostics": {
            "noise_sampling_mode": noise_sampling_mode,
            "mc_samples": int(mc_samples),
            "negative_signed_estimator_count": int(np.sum(values < 0)),
            "point_response_missing_masks": [response.missing_value_masks for response in responses],
            "uniform_weights": bool(np.allclose(propagated_variance, propagated_variance[0])),
            "projection_mode": projection_mode,
            "efficiency_semantics": "probability_efficiency_converted_to_field_amplitude_with_sqrt",
        },
    }


def _legacy_basis_projector(point: OperationPointSpec) -> np.ndarray:
    """Return the exact basis-only operator used by legacy reconstruction."""
    amplitudes = np.asarray(point.basis_amplitudes, dtype=float)
    phases = -np.asarray(point.basis_phases_rad, dtype=float)
    state = amplitudes * np.exp(1j * phases)
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        return np.zeros((state.size, state.size), dtype=complex)
    state = state / norm
    return np.outer(state, state.conj())


def _fit_length(values: np.ndarray, size: int) -> np.ndarray:
    if values.size == size:
        return values.astype(float, copy=True)
    output = np.zeros(size, dtype=float)
    output[:min(size, values.size)] = values[:size]
    return output


def _invert(values: np.ndarray, operators: np.ndarray, variance: np.ndarray, regularization: float) -> tuple[tuple[np.ndarray], np.ndarray, float]:
    scale = float(np.max(np.abs(values))) if np.max(np.abs(values)) > 0 else 1.0
    targets = values / scale
    m = operators.reshape(operators.shape[0], -1)
    weights = 1.0 / np.maximum(variance, 1e-12)
    sqrt_w = np.sqrt(weights / np.max(weights))
    weighted_m = m * sqrt_w[:, None]
    weighted_y = targets * sqrt_w
    normal = weighted_m.conj().T @ weighted_m
    if regularization > 0:
        solution = np.linalg.solve(normal + regularization * np.eye(normal.shape[0]), weighted_m.conj().T @ weighted_y)
    else:
        solution, _, _, _ = np.linalg.lstsq(weighted_m, weighted_y, rcond=None)
    rho = solution.reshape(operators.shape[1], operators.shape[2])
    rho = 0.5 * (rho + rho.conj().T)
    return (rho,), normal, float(np.linalg.cond(normal))


def _physical(rho: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(hermitian)
    vals = np.clip(np.real(vals), 0.0, None)
    if vals.sum() <= 0:
        vals[0] = 1.0
    projected = vecs @ np.diag(vals) @ vecs.conj().T
    return projected / np.trace(projected)


def _sample_values(values: np.ndarray, variance: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "gaussian":
        return rng.normal(values, np.sqrt(variance))
    if mode == "poisson":
        # Signed estimators cannot themselves be Poisson.  This is a
        # variance-matched Skellam approximation, retaining the sign.
        lam = np.maximum(variance / 2.0, 1e-9)
        return values + rng.poisson(lam) - rng.poisson(lam)
    return 0.5 * (rng.normal(values, np.sqrt(variance)) + _sample_values(values, variance, "poisson", rng))


def _w_fidelity(rho: np.ndarray) -> float:
    vector = np.ones(rho.shape[0], dtype=complex) / np.sqrt(rho.shape[0])
    return float(np.clip(np.real(vector.conj() @ rho @ vector), 0.0, 1.0))


def _leakage_samples(responses: Sequence[Any], size: int, rng: np.random.Generator) -> np.ndarray:
    values = [response.expected_leakage_rate for response in responses if response.expected_leakage_rate is not None]
    if not values:
        return np.zeros(size, dtype=float)
    mean = float(np.clip(np.mean(values), 0.0, 1.0))
    # Calibration uncertainty only broadens an explicitly calibrated leakage.
    return np.clip(rng.normal(mean, max(0.005, 0.1 * mean), size=size), 0.0, 1.0)


def _interval(values: np.ndarray) -> Dict[str, float]:
    return {"low": float(np.quantile(values, 0.025)), "high": float(np.quantile(values, 0.975))}


def _embed_full(system_type: str, rho: np.ndarray) -> np.ndarray:
    spec = get_spec(system_type)
    output = np.zeros((int(spec["full_dim"]), int(spec["full_dim"])), dtype=complex)
    indices = spec["full_indices"]
    for i, row in enumerate(indices):
        for j, col in enumerate(indices):
            output[row, col] = rho[i, j]
    return output
