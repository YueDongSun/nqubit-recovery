"""Versioned identities and admission checks for Track B synthetic teachers."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ml_recon.physics_hybrid_v10.noise_matching import (
    evaluated_severity_matching_kind, legacy_severity_matching_kind, validate_severity_matching_kind,
)

from ..hashing import sha256_json
from ..version import GENERATOR_SHA
from .trackb_phase_lift import (
    CONTROL_AMPLITUDE_BOUND,
    CONTROL_PHASE_BOUND,
    TRACKB_PHASE_LIFT_CUT_TIE_TOLERANCE,
    TRACKB_PHASE_LIFT_MINIMUM_AMPLITUDE,
    TRACKB_PHASE_LIFT_NUMERICAL_TOLERANCE,
    TRACKB_PHASE_LIFT_QUOTIENT_TIE_ATOL,
    TRACKB_PHASE_LIFT_QUOTIENT_TIE_RTOL,
    TRACKB_PHASE_LIFT_WINDING_BOUND,
    TRACKB_RECEDING_HORIZON_DAMPING,
)


TRACKB_TEACHER_POLICY_V1 = "receding_horizon_target_oracle_v1"
TRACKB_TEACHER_POLICY_V2 = "receding_horizon_target_oracle_v2"
TRACKB_TEACHER_POLICY_V2_CELL_IDS = {
    3: "trackb_feedback_history_v2_n3",
    4: "trackb_feedback_history_v2_n4",
}
TRACKB_TEN_ROUND_COUNT = 10


def _trackb_teacher_policy_v2_descriptor() -> dict[str, Any]:
    return {
        "policy_id": TRACKB_TEACHER_POLICY_V2,
        "supported_N": [3, 4],
        "phase_lift": {
            "kind": "minimum_quotient_lattice_lift",
            "relative_phase": "arg(r_i*conj(r_0)) with r_i=target_i/abs(target_i)*conj(before_i/abs(before_i))",
            "equivalence_lattice": "H*(2*pi*Z^N), anchored winding k[0]=0",
            "metric": "squared_euclidean_norm_in_orthonormal_helmert_coordinates",
            "winding_bound": TRACKB_PHASE_LIFT_WINDING_BOUND,
            "tie_rule": "norm within atol+rtol*abs(minimum), then lexicographically smallest anchored winding",
            "tie_atol": TRACKB_PHASE_LIFT_QUOTIENT_TIE_ATOL,
            "tie_rtol": TRACKB_PHASE_LIFT_QUOTIENT_TIE_RTOL,
            "cut_rule": "angles within cut tolerance of either principal cut map to +pi",
            "cut_tie_tolerance": TRACKB_PHASE_LIFT_CUT_TIE_TOLERANCE,
            "minimum_mode_amplitude": TRACKB_PHASE_LIFT_MINIMUM_AMPLITUDE,
            "global_phase_action_tolerance": TRACKB_PHASE_LIFT_NUMERICAL_TOLERANCE,
        },
        "action": {
            "coordinates": "helmert_relative_amplitude_and_phase",
            "amplitude_bound": CONTROL_AMPLITUDE_BOUND,
            "phase_bound": CONTROL_PHASE_BOUND,
            "damping": TRACKB_RECEDING_HORIZON_DAMPING,
            "layout": "logical_interleaved_amp_phase",
            "token_output_dtype": "float32",
        },
        "history_rounds": TRACKB_TEN_ROUND_COUNT,
        "cell_ids": {str(n): cell for n, cell in TRACKB_TEACHER_POLICY_V2_CELL_IDS.items()},
    }


TRACKB_TEACHER_POLICY_V2_DESCRIPTOR = _trackb_teacher_policy_v2_descriptor()
TRACKB_TEACHER_POLICY_V2_SHA256 = sha256_json(TRACKB_TEACHER_POLICY_V2_DESCRIPTOR)


def validate_trackb_v2_cell(cell: Mapping[str, Any]) -> int:
    """Require a complete, internally consistent opt-in V2 Track B cell."""
    try:
        N = cell["N"]
        rounds = cell["history_rounds"]
    except KeyError as exc:
        raise ValueError("Track B V2 cell requires N and ten-round history") from exc
    if type(N) is not int or type(rounds) is not int:
        raise ValueError("Track B V2 N and history_rounds must be JSON integers")
    expected_cell = TRACKB_TEACHER_POLICY_V2_CELL_IDS.get(N)
    if expected_cell is None:
        raise ValueError("Track B V2 supports only N=3 or N=4")
    if cell.get("dataset_family") != "feedback_history":
        raise ValueError("Track B V2 policy is valid only for feedback_history")
    if cell.get("history_policy") != TRACKB_TEACHER_POLICY_V2:
        raise ValueError("Track B V2 cell has a missing or foreign history policy")
    if cell.get("history_policy_sha256") != TRACKB_TEACHER_POLICY_V2_SHA256:
        raise ValueError("Track B V2 cell has a missing or foreign policy SHA")
    if rounds != TRACKB_TEN_ROUND_COUNT:
        raise ValueError("Track B V2 policy requires exactly ten history rounds")
    if cell.get("cell_id") != expected_cell:
        raise ValueError("Track B V2 cell ID does not match its N")
    if cell.get("system_type") != f"{N}qubit":
        raise ValueError("Track B V2 cell system_type does not match its N")
    matching_kind = cell.get("severity_matching_kind")
    if type(matching_kind) is not str or matching_kind != evaluated_severity_matching_kind:
        raise ValueError("Track B V2 requires evaluated_midpoint_v2 severity_matching_kind")
    return N


def validate_trackb_v2_sample_key(
    key: Any,
    *,
    N: int,
    dataset_family: str,
    cell_id: str,
) -> None:
    """Bind V2 use to a valid canonical key from the registered generator."""
    try:
        key.validate()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Track B V2 sample key is invalid") from exc
    if type(getattr(key, "N", None)) is not int or key.N != N:
        raise ValueError("Track B V2 sample key N does not match its cell")
    if (getattr(key, "dataset_family", None) != dataset_family
            or getattr(key, "cell_id", None) != cell_id):
        raise ValueError("Track B V2 sample key does not match its N, family, and cell ID")
    if getattr(key, "generator_sha", None) != GENERATOR_SHA:
        raise ValueError("Track B V2 sample key generator SHA is not registered")
    if getattr(key, "split", None) not in ("train", "validation"):
        raise ValueError("Track B V2 sample key split must be train or validation")
    if type(getattr(key, "sample_id", None)) is not str:
        raise ValueError("Track B V2 sample key ID must be text")
    sample_pattern = re.compile(
        rf"v2_4:feedback_history:{re.escape(cell_id)}:(train|validation):N{N}:([0-9]{{12,}})"
    )
    match = sample_pattern.fullmatch(key.sample_id)
    if match is None or match.group(1) != key.split:
        raise ValueError("Track B V2 sample key ID or split is not canonical")
    sample_index = int(match.group(2))
    if key.sample_id != (
            f"v2_4:{dataset_family}:{cell_id}:{key.split}:N{N}:{sample_index:012d}"):
        raise ValueError("Track B V2 sample key ID is not canonical")


def validate_trackb_policy_namespace(
    key: Any,
    *,
    history_policy: str,
    history_policy_sha256: str | None,
) -> None:
    """Prevent reserved V2 sample namespaces from reaching legacy render paths."""
    cell_id = getattr(key, "cell_id", "")
    sample_id = getattr(key, "sample_id", "")
    has_v2_intent = (
        isinstance(cell_id, str) and cell_id.startswith("trackb_feedback_history_v2")
    ) or (
        isinstance(sample_id, str)
        and sample_id.startswith("v2_4:feedback_history:trackb_feedback_history_v2")
    ) or history_policy == TRACKB_TEACHER_POLICY_V2 or history_policy_sha256 is not None
    if not has_v2_intent:
        return
    if (history_policy != TRACKB_TEACHER_POLICY_V2
            or history_policy_sha256 != TRACKB_TEACHER_POLICY_V2_SHA256):
        raise ValueError("Track B V2 sample namespace requires its exact policy and SHA")
    N = getattr(key, "N", None)
    expected_cell_id = TRACKB_TEACHER_POLICY_V2_CELL_IDS.get(N)
    if expected_cell_id is None:
        raise ValueError("Track B V2 supports only N=3 or N=4")
    validate_trackb_v2_sample_key(
        key, N=N, dataset_family="feedback_history", cell_id=expected_cell_id)


def reject_trackb_v2_intent(cell: Mapping[str, Any] | None) -> None:
    """Reject reserved V2 cells/policy claims outside the exact V2 contract."""
    if cell is None:
        return
    matching_kind = cell.get("severity_matching_kind", legacy_severity_matching_kind)
    validate_severity_matching_kind(matching_kind)
    cell_id = str(cell.get("cell_id", ""))
    policy = cell.get("history_policy")
    sha = cell.get("history_policy_sha256")
    intent = (cell_id.startswith("trackb_feedback_history_v2")
              or policy == TRACKB_TEACHER_POLICY_V2 or sha is not None
              or matching_kind == evaluated_severity_matching_kind)
    if intent:
        validate_trackb_v2_cell(cell)
