"""Pure numerical phase lift for Track B's opt-in target teacher."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ml_recon.physics_hybrid_v10.control_tangent import (
    aitchison_amplitude_coordinates,
    apply_tangent_action,
    helmert_matrix,
)


TRACKB_PHASE_LIFT_WINDING_BOUND = 1
TRACKB_PHASE_LIFT_QUOTIENT_TIE_ATOL = 1e-12
TRACKB_PHASE_LIFT_QUOTIENT_TIE_RTOL = 1e-12
TRACKB_PHASE_LIFT_NUMERICAL_TOLERANCE = 1e-9
TRACKB_PHASE_LIFT_CUT_TIE_TOLERANCE = 1e-12
TRACKB_PHASE_LIFT_MINIMUM_AMPLITUDE = 1e-12
CONTROL_AMPLITUDE_BOUND = 0.35
CONTROL_PHASE_BOUND = 0.70
TRACKB_RECEDING_HORIZON_DAMPING = 0.35
_SUPPORTED_MODE_COUNTS = (3, 4)


@dataclass(frozen=True)
class TrackBPhaseLiftResult:
    """Selected phase lift in the orthonormal Helmert chart."""

    relative_phase: np.ndarray
    winding: np.ndarray
    quotient_coordinates: np.ndarray
    quotient_norm_squared: float


def _mode_pair(before: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    before_modes = np.asarray(before, dtype=np.complex128)
    target_modes = np.asarray(target, dtype=np.complex128)
    if before_modes.ndim != 1 or target_modes.ndim != 1:
        raise ValueError("before and target must be one-dimensional mode vectors")
    if before_modes.shape != target_modes.shape:
        raise ValueError("before and target must have the same mode count")
    if before_modes.size not in _SUPPORTED_MODE_COUNTS:
        raise ValueError("phase lift supports only N=3 or N=4")
    if not np.all(np.isfinite(before_modes)) or not np.all(np.isfinite(target_modes)):
        raise ValueError("before and target modes must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        before_amplitudes = np.abs(before_modes)
        target_amplitudes = np.abs(target_modes)
        before_norm = np.linalg.norm(before_amplitudes)
        target_norm = np.linalg.norm(target_amplitudes)
    if (not np.all(np.isfinite(before_amplitudes))
            or not np.all(np.isfinite(target_amplitudes))
            or not np.isfinite(before_norm) or not np.isfinite(target_norm)):
        raise ValueError("before and target amplitude arithmetic must remain finite")
    if (np.any(before_amplitudes <= TRACKB_PHASE_LIFT_MINIMUM_AMPLITUDE)
            or np.any(target_amplitudes <= TRACKB_PHASE_LIFT_MINIMUM_AMPLITUDE)):
        raise ValueError("zero or near-zero mode amplitude has undefined phase")
    return before_modes, target_modes


def trackb_relative_phase_vector(before: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return anchored principal relative phasors, invariant to global phase."""
    before_modes, target_modes = _mode_pair(before, target)
    before_phasors = before_modes / np.abs(before_modes)
    target_phasors = target_modes / np.abs(target_modes)
    relative_phasors = target_phasors * np.conjugate(before_phasors)
    anchored_phasors = relative_phasors * np.conjugate(relative_phasors[0])
    relative_phase = np.angle(anchored_phasors)
    cut_distance = np.abs(np.abs(relative_phase) - np.pi)
    relative_phase[cut_distance <= TRACKB_PHASE_LIFT_CUT_TIE_TOLERANCE] = np.pi
    relative_phase[0] = 0.0
    return relative_phase


def trackb_phase_lift_candidates(
    relative_phase: np.ndarray,
    N: int,
    winding_bound: int = TRACKB_PHASE_LIFT_WINDING_BOUND,
) -> list[tuple[float, tuple[int, ...], np.ndarray]]:
    """Enumerate anchored windings in lexicographic order for the quotient norm."""
    if N not in _SUPPORTED_MODE_COUNTS:
        raise ValueError("phase lift supports only N=3 or N=4")
    values = np.asarray(relative_phase, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("relative phase must be a one-dimensional vector")
    if values.size != N or not np.all(np.isfinite(values)):
        raise ValueError("relative phase must be a finite length-N vector")
    if np.any(np.abs(values) > np.pi + TRACKB_PHASE_LIFT_CUT_TIE_TOLERANCE):
        raise ValueError("relative phase entries must be principal angles in [-pi, pi]")
    values = values.copy()
    cut_distance = np.abs(np.abs(values) - np.pi)
    values[cut_distance <= TRACKB_PHASE_LIFT_CUT_TIE_TOLERANCE] = np.pi
    if abs(values[0]) > TRACKB_PHASE_LIFT_NUMERICAL_TOLERANCE:
        raise ValueError("relative phase must be anchored at mode zero")
    if type(winding_bound) is not int or winding_bound < 0:
        raise ValueError("winding bound must be a non-negative integer")

    helmert = helmert_matrix(N)
    candidates = []
    for tail in itertools.product(range(-winding_bound, winding_bound + 1), repeat=N - 1):
        winding = (0,) + tail
        lifted = values + 2.0 * np.pi * np.asarray(winding, dtype=np.float64)
        quotient = helmert @ lifted
        candidates.append((float(quotient @ quotient), winding, quotient))
    return candidates


def _canonical_principal_phase(relative_phase: np.ndarray) -> np.ndarray:
    values = np.asarray(relative_phase, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("relative phase must be a one-dimensional vector")
    values = values.copy()
    cut_distance = np.abs(np.abs(values) - np.pi)
    values[cut_distance <= TRACKB_PHASE_LIFT_CUT_TIE_TOLERANCE] = np.pi
    return values


def select_trackb_phase_lift(
    relative_phase: np.ndarray,
    N: int,
    winding_bound: int = TRACKB_PHASE_LIFT_WINDING_BOUND,
) -> TrackBPhaseLiftResult:
    """Minimize ``||H(d + 2*pi*k)||²`` with a deterministic numeric tie rule."""
    candidates = trackb_phase_lift_candidates(relative_phase, N, winding_bound)
    best_norm = min(candidate[0] for candidate in candidates)
    tie_limit = (best_norm + TRACKB_PHASE_LIFT_QUOTIENT_TIE_ATOL
                 + TRACKB_PHASE_LIFT_QUOTIENT_TIE_RTOL * abs(best_norm))
    chosen = min((candidate for candidate in candidates if candidate[0] <= tie_limit),
                 key=lambda candidate: candidate[1])
    return TrackBPhaseLiftResult(
        relative_phase=_canonical_principal_phase(relative_phase),
        winding=np.asarray(chosen[1], dtype=np.int64),
        quotient_coordinates=np.asarray(chosen[2], dtype=np.float64),
        quotient_norm_squared=float(chosen[2] @ chosen[2]),
    )


def trackb_receding_horizon_action_v2(
    before: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the V2 bounded logical action and resulting normalized state."""
    before_modes, target_modes = _mode_pair(before, target)
    mode_count = int(before_modes.size)
    helmert = helmert_matrix(mode_count)
    amplitude_residual = (
        aitchison_amplitude_coordinates(np.abs(target_modes))
        - aitchison_amplitude_coordinates(np.abs(before_modes))
    )
    if not np.all(np.isfinite(amplitude_residual)):
        raise ValueError("amplitude residual must remain finite")
    relative_phase = trackb_relative_phase_vector(before_modes, target_modes)
    phase_lift = select_trackb_phase_lift(relative_phase, mode_count)
    amplitude_step = TRACKB_RECEDING_HORIZON_DAMPING * np.clip(
        amplitude_residual, -CONTROL_AMPLITUDE_BOUND, CONTROL_AMPLITUDE_BOUND)
    phase_step = TRACKB_RECEDING_HORIZON_DAMPING * np.clip(
        phase_lift.quotient_coordinates, -CONTROL_PHASE_BOUND, CONTROL_PHASE_BOUND)
    after = apply_tangent_action(before_modes, amplitude_step, phase_step)
    action = np.stack((helmert.T @ amplitude_step, helmert.T @ phase_step), axis=-1)
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(after)):
        raise ValueError("phase action and resulting state must remain finite")
    with np.errstate(over="ignore", invalid="ignore"):
        after_norm = np.linalg.norm(after)
    if not np.isfinite(after_norm) or abs(float(after_norm) - 1.0) > TRACKB_PHASE_LIFT_NUMERICAL_TOLERANCE:
        raise ValueError("resulting state must remain normalized")
    return action, after
