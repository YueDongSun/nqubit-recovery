"""Gauge-invariant target identities for repeated measured W states."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


TARGET_QUANTIZATION_DECIMALS = 9
Q4_RECOVERY_INTERVENTION_ID = "q4_recovery_plus90"
Q4_RECOVERY_DELTA = (0.0, 0.0, 0.0, math.pi / 2.0)


def wrap_phase(value: np.ndarray) -> np.ndarray:
    """Wrap radians to the half-open interval [-pi, pi)."""
    wrapped = (np.asarray(value, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
    wrapped[np.isclose(wrapped, math.pi)] = -math.pi
    return wrapped


def canonical_target_state(
    target_state_params: Mapping[str, Any],
    *,
    decimals: int = TARGET_QUANTIZATION_DECIMALS,
) -> Dict[str, Any]:
    """Canonicalize amplitudes and phases without using a state name."""
    amplitudes = np.asarray(target_state_params.get("amplitudes", []), dtype=float).reshape(-1)
    if amplitudes.size == 0 or not np.all(np.isfinite(amplitudes)):
        raise ValueError("target amplitudes must be finite and non-empty")
    norm = float(np.linalg.norm(amplitudes))
    if norm <= 0.0:
        raise ValueError("target amplitudes must have non-zero norm")
    amplitudes = amplitudes / norm
    if "phases_rad" in target_state_params:
        phases = np.asarray(target_state_params["phases_rad"], dtype=float).reshape(-1)
    else:
        phases = np.deg2rad(
            np.asarray(target_state_params.get("phases_deg", []), dtype=float).reshape(-1)
        )
    if phases.shape != amplitudes.shape or not np.all(np.isfinite(phases)):
        raise ValueError("target phases must be finite and match amplitudes")
    phases = wrap_phase(phases - phases[0])
    quantized_amplitudes = np.round(amplitudes, decimals=decimals)
    quantized_phases = np.round(phases, decimals=decimals)
    return {
        "amplitudes": [float(value) for value in quantized_amplitudes],
        "phases_rad": [float(value) for value in quantized_phases],
        "quantization_decimals": int(decimals),
    }


def target_hash(target_state_params: Mapping[str, Any]) -> str:
    canonical = canonical_target_state(target_state_params)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def intervention_identity(
    *,
    system_type: str,
    target_state_params: Mapping[str, Any],
    plan_reference: str = "",
) -> Tuple[str, str | None, list[float]]:
    """Return base-family ID and an explicit known intervention, if present."""
    canonical = canonical_target_state(target_state_params)
    is_recovery = system_type == "4qubit" and "20260802_170634" in str(plan_reference)
    if not is_recovery:
        return target_hash(canonical), None, []
    phases = np.asarray(canonical["phases_rad"], dtype=float)
    phases[-1] = phases[-1] - Q4_RECOVERY_DELTA[-1]
    base_payload = {
        "amplitudes": canonical["amplitudes"],
        "phases_rad": phases.tolist(),
    }
    return target_hash(base_payload), Q4_RECOVERY_INTERVENTION_ID, list(Q4_RECOVERY_DELTA)


def safe_target_payload(target_state_params: Mapping[str, Any]) -> Dict[str, Any]:
    """Retain physical target controls while stripping label-like path fields."""
    canonical = canonical_target_state(target_state_params)
    allowed = {
        key: value
        for key, value in target_state_params.items()
        if key in {
            "coverage_index", "state_role", "block_id", "design_version",
            "schedule_index", "anchor_id", "pair_split_key",
        }
    }
    allowed.update(canonical)
    return allowed


__all__ = [
    "Q4_RECOVERY_DELTA",
    "Q4_RECOVERY_INTERVENTION_ID",
    "TARGET_QUANTIZATION_DECIMALS",
    "canonical_target_state",
    "intervention_identity",
    "safe_target_payload",
    "target_hash",
    "wrap_phase",
]
