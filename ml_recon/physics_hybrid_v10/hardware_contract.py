"""Read-only V10 action validation; execution remains in system_control."""

from __future__ import annotations

from typing import Dict

import numpy as np

STEP_AMP = 0.005
STEP_PHASE = 0.02
CUMULATIVE_AMP = 0.03
CUMULATIVE_PHASE = 0.10
EMERGENCY_PHASE = 0.20


def validate_relative_action(delta_log_amp: np.ndarray, delta_phase: np.ndarray,
                             cumulative_log_amp: np.ndarray,
                             cumulative_phase: np.ndarray,
                             emergency: bool = False) -> Dict[str, object]:
    da = np.asarray(delta_log_amp, dtype=np.float64)
    dp = np.asarray(delta_phase, dtype=np.float64)
    ca = np.asarray(cumulative_log_amp, dtype=np.float64)
    cp = np.asarray(cumulative_phase, dtype=np.float64)
    reasons = []
    if da.shape != dp.shape or da.shape != ca.shape or da.shape != cp.shape:
        reasons.append("shape_mismatch")
    if abs(da.mean()) > 1e-9 or abs(dp.mean()) > 1e-9:
        reasons.append("relative_action_not_zero_mean")
    if np.max(np.abs(da), initial=0.0) > STEP_AMP + 1e-12:
        reasons.append("amplitude_step_bound")
    phase_cap = EMERGENCY_PHASE if emergency else STEP_PHASE
    if np.max(np.abs(dp), initial=0.0) > phase_cap + 1e-12:
        reasons.append("phase_step_bound")
    if np.max(np.abs(ca + da), initial=0.0) > CUMULATIVE_AMP + 1e-12:
        reasons.append("amplitude_cumulative_bound")
    if np.max(np.abs(cp + dp), initial=0.0) > CUMULATIVE_PHASE + 1e-12:
        reasons.append("phase_cumulative_bound")
    return {"passed": not reasons, "reasons": reasons,
            "materialization_authorized": False, "upload_authorized": False}


def exponential_amplitude(base_amplitude: np.ndarray, delta_log_amp: np.ndarray) -> np.ndarray:
    return np.asarray(base_amplitude, dtype=np.float64) * np.exp(
        np.asarray(delta_log_amp, dtype=np.float64))
