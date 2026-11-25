"""Same-state nominal projective derivatives, without causal FD authority.

This additive artifact retains each ensemble member rather than differentiating
a mixture of different input states. Its empirical covariance is not calibrated
measurement noise, a posterior covariance, or qualified CSM support.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np

from .contracts import canonical_json_bytes, parse_canonical_json_object, require_git_oid
from .coordinate_abi import coordinate_layout_sha
from .effective_inference_artifact import Effective_point_inference
from .native_action_sensitivity import _projective_state
from .native_state_ensemble import Native_state_ensemble, verify_native_state_companions


state_sensitivity_schema = "TrackBNativeStateProjectiveSensitivityV1"
state_sensitivity_scope = "SAME_STATE_NOMINAL_PROJECTIVE_DERIVATIVES_NOT_CAUSAL_CSM_OR_FD"
finite_difference_step = 1.0e-4
maximum_state_sensitivity_bytes = 4 * 1024 * 1024
sensitivity_fields = frozenset({
    "schema", "scope", "producer_commit", "producer_tree", "ensemble", "ensemble_sha", "state_key_sha",
    "action_coordinate_layout_sha", "finite_difference_step", "base_point", "baseline_draws", "jacobian_draws",
    "empirical_state_covariance", "covariance_status", "causal_status", "qualified_csm_available",
    "model_forward_executed", "optimizer_constructed", "parameters_updated", "selection_evidence",
    "scientific_gate_issued", "hardware_action_authorized", "research_only",
})


def _nominal_values(ensemble: Native_state_ensemble) -> dict[str, Any]:
    payload = ensemble.payload()
    mode_count = 3 if payload["context"]["system_type"] == "3qubit" else 4
    zero = np.zeros(2 * (mode_count - 1), dtype=np.float64)
    baselines, jacobians = [], []
    for member in payload["members"]:
        state = np.asarray(member["state_coordinates"], dtype=np.float64)
        baselines.append(_projective_state(state, zero, mode_count).tolist())
        columns = []
        for coordinate in range(zero.size):
            plus, minus = zero.copy(), zero.copy()
            plus[coordinate] = finite_difference_step
            minus[coordinate] = -finite_difference_step
            columns.append((_projective_state(state, plus, mode_count)
                            - _projective_state(state, minus, mode_count)) / (2 * finite_difference_step))
        jacobian = np.stack(columns, axis=1)
        if not np.isfinite(jacobian).all():
            raise FloatingPointError("Nonfinite same-state nominal projective sensitivity")
        jacobians.append(jacobian.tolist())
    return {"ensemble": payload, "ensemble_sha": ensemble.sha256(), "state_key_sha": payload["state_key_sha"],
        "action_coordinate_layout_sha": coordinate_layout_sha(mode_count), "finite_difference_step": finite_difference_step,
        "base_point": zero.tolist(), "baseline_draws": baselines, "jacobian_draws": jacobians,
        "empirical_state_covariance": payload["moments"]["state_coordinates"]["covariance"]}


@dataclass(frozen=True)
class Native_state_sensitivity:
    payload_bytes: bytes

    def payload(self) -> dict[str, Any]:
        if type(self.payload_bytes) is not bytes or len(self.payload_bytes) > maximum_state_sensitivity_bytes:
            raise ValueError("Native state sensitivity requires bounded canonical bytes")
        value = parse_canonical_json_object(self.payload_bytes, schema=state_sensitivity_schema, exact_fields=sensitivity_fields)
        for name in ("producer_commit", "producer_tree"):
            require_git_oid(value[name], name)
        if value["scope"] != state_sensitivity_scope \
                or value["covariance_status"] != "MODEL_ENSEMBLE_DISPERSION_NOT_CALIBRATED_NOISE" \
                or value["causal_status"] != "NOT_IDENTIFIED_FROM_STATIC_OBSERVATIONS" \
                or value["research_only"] is not True \
                or any(value[name] is not False for name in ("qualified_csm_available", "model_forward_executed",
                    "optimizer_constructed", "parameters_updated", "selection_evidence", "scientific_gate_issued",
                    "hardware_action_authorized")):
            raise ValueError("Native state sensitivity scope or authority differs")
        ensemble = Native_state_ensemble(canonical_json_bytes(value["ensemble"]))
        expected = _nominal_values(ensemble)
        if canonical_json_bytes({name: value[name] for name in expected}) != canonical_json_bytes(expected):
            raise ValueError("Native state sensitivity does not reproduce its retained ensemble")
        return value

    def sha256(self) -> str:
        self.payload()
        return hashlib.sha256(self.payload_bytes).hexdigest()


def build_native_state_sensitivity(
    ensemble: Native_state_ensemble, point_artifacts: Sequence[Effective_point_inference], *,
    producer_commit: str, producer_tree: str,
) -> Native_state_sensitivity:
    verify_native_state_companions(ensemble, point_artifacts)
    require_git_oid(producer_commit, "state_sensitivity_producer_commit")
    require_git_oid(producer_tree, "state_sensitivity_producer_tree")
    result = Native_state_sensitivity(canonical_json_bytes({
        "schema": state_sensitivity_schema, "scope": state_sensitivity_scope,
        "producer_commit": producer_commit, "producer_tree": producer_tree, **_nominal_values(ensemble),
        "covariance_status": "MODEL_ENSEMBLE_DISPERSION_NOT_CALIBRATED_NOISE",
        "causal_status": "NOT_IDENTIFIED_FROM_STATIC_OBSERVATIONS", "qualified_csm_available": False,
        "model_forward_executed": False, "optimizer_constructed": False, "parameters_updated": False,
        "selection_evidence": False, "scientific_gate_issued": False, "hardware_action_authorized": False,
        "research_only": True,
    }))
    result.payload()
    return result


def verify_native_state_sensitivity_companions(
    sensitivity: Native_state_sensitivity, point_artifacts: Sequence[Effective_point_inference],
) -> None:
    if type(sensitivity) is not Native_state_sensitivity:
        raise TypeError("Native state sensitivity verification requires the exact typed artifact")
    payload = sensitivity.payload()
    verify_native_state_companions(Native_state_ensemble(canonical_json_bytes(payload["ensemble"])), point_artifacts)
