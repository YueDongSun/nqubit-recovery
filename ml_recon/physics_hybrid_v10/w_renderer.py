"""Variable-N W-state renderer with correctable tangent truth."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .control_tangent import (
    aitchison_amplitude_coordinates,
    apply_tangent_action,
    phase_coordinates,
    tangent_dimension,
)
from .synthetic_manifest import V10CounterBasedRng, V10SyntheticSampleKey
from .w_subspace import pure_w_state


def nonlinear_corrective_oracle(
    current: np.ndarray,
    target: np.ndarray,
    *,
    maximum_action_norm: float = 4.0,
) -> Dict[str, object]:
    """Solve the finite-action inverse problem against the nonlinear W map."""
    current = np.asarray(current, dtype=np.complex128).reshape(-1)
    target = np.asarray(target, dtype=np.complex128).reshape(-1)
    if current.shape != target.shape or current.size < 3:
        raise ValueError("current and target W vectors must have the same N>=3 shape")
    current /= np.linalg.norm(current)
    target /= np.linalg.norm(target)
    N = current.size
    initial = np.concatenate((
        aitchison_amplitude_coordinates(np.abs(target))
        - aitchison_amplitude_coordinates(np.abs(current)),
        phase_coordinates(np.angle(target)) - phase_coordinates(np.angle(current)),
    ))
    initial = np.clip(initial, -maximum_action_norm, maximum_action_norm)

    def residual(action: np.ndarray) -> np.ndarray:
        candidate = apply_tangent_action(current, action[:N - 1], action[N - 1:])
        overlap = np.vdot(target, candidate)
        aligned = candidate * np.exp(-1j * np.angle(overlap + 1e-15))
        difference = aligned - target
        return np.concatenate((difference.real, difference.imag))

    method = "coordinate_inverse"
    action = initial
    try:
        from scipy.optimize import least_squares

        fitted = least_squares(
            residual,
            initial,
            bounds=(-maximum_action_norm, maximum_action_norm),
            max_nfev=200,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        action = fitted.x
        method = "nonlinear_least_squares"
    except ImportError:
        action = np.clip(initial, -maximum_action_norm, maximum_action_norm)
    corrected = apply_tangent_action(current, action[:N - 1], action[N - 1:])
    infidelity = 1.0 - float(np.abs(np.vdot(target, corrected)) ** 2)
    return {
        "delta_amplitude": action[:N - 1],
        "delta_phase": action[N - 1:],
        "corrected_coefficients": corrected,
        "achieved_infidelity": max(0.0, infidelity),
        "oracle_method": method,
    }


class V10WRenderer:
    def render(self, key: V10SyntheticSampleKey, rng: V10CounterBasedRng) -> Dict[str, object]:
        state_rng = rng.stream("state")
        action_rng = rng.stream("action")
        amplitudes = np.exp(state_rng.normal(scale=0.25, size=key.N))
        phases = state_rng.uniform(-np.pi, np.pi, size=key.N)
        coefficients = amplitudes * np.exp(1j * phases)
        coefficients /= np.linalg.norm(coefficients)
        delta_a = action_rng.normal(scale=0.25, size=key.N - 1)
        delta_p = action_rng.normal(scale=0.60, size=key.N - 1)
        state = pure_w_state(coefficients)
        current = apply_tangent_action(coefficients, delta_a, delta_p)
        oracle = nonlinear_corrective_oracle(current, coefficients)
        return {"sample_id": key.sample_id, "N": key.N,
                "clean_w_coefficients": coefficients, "rho_clean": state.R1,
                "current_w_coefficients": current,
                "injected_delta_amplitude": delta_a,
                "injected_delta_phase": delta_p,
                "correctable_delta_amplitude": oracle["delta_amplitude"],
                "correctable_delta_phase": oracle["delta_phase"],
                "oracle_corrected_coefficients": oracle["corrected_coefficients"],
                "oracle_achieved_infidelity": oracle["achieved_infidelity"],
                "oracle_method": oracle["oracle_method"],
                "control_tangent_dimension": tangent_dimension(key.N)}
