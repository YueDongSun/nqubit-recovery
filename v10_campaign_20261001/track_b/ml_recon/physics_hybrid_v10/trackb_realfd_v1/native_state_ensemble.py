"""Per-state cross-model moments, never a calibrated actionful posterior.

The original cohort artifact is unchanged. Full point companions remain the
authority for provenance; parsing this derived artifact is not replay proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np

from .contracts import canonical_json_bytes, canonical_sha256, parse_canonical_json_object, require_git_oid, require_sha256
from .effective_inference_artifact import Effective_point_inference
from .effective_point_values import _validate_psd
from .native_posterior import _ensemble_mean_covariance, _finite_vector, _fresh_unavailable_quantities, _row_draw_vectors
from .native_state_context import select_native_state_row, validate_native_state_context


state_ensemble_schema = "TrackBNativeStateEnsembleV1"
state_ensemble_scope = "SAME_SAMPLE_SESSION_VIEW_MODEL_ENSEMBLE_NOT_CALIBRATED_POSTERIOR"
maximum_state_ensemble_members = 8
maximum_state_ensemble_bytes = 2 * 1024 * 1024
ensemble_fields = frozenset({
    "schema", "scope", "producer_commit", "producer_tree", "context", "state_key_sha", "members",
    "moments", "draw_count", "selected_row_count", "unavailable", "model_forward_executed",
    "optimizer_constructed", "parameters_updated", "selection_evidence", "scientific_gate_issued",
    "hardware_action_authorized", "causal_gain_available", "research_only",
})
member_fields = frozenset({
    "point_file_sha", "training_identity_sha", "checkpoint_file_sha", "model_state_sha",
    "point_protocol_sha", "point_row_sha", "point_row_view_sha", "batch_index", "batch_input_sha",
    "batch_query_mask_sha", "state_coordinates", "native_global_parameters", "effective_response_parameters",
})


def state_unavailable_quantities() -> dict[str, dict[str, Any]]:
    values = _fresh_unavailable_quantities()
    values["posterior_draws"]["status"] = "MODEL_ENSEMBLE_COORDINATES_NOT_CALIBRATED_POSTERIOR_DRAWS"
    return values


def _state_moments(members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    moments = {}
    for name in ("state_coordinates", "native_global_parameters", "effective_response_parameters"):
        mean, covariance = _ensemble_mean_covariance([np.asarray(member[name], dtype=np.float64) for member in members], name)
        moments[name] = {"mean": list(mean), "covariance": [list(row) for row in covariance]}
    return moments


def validate_native_state_ensemble(payload: dict[str, Any]) -> None:
    if type(payload) is not dict or set(payload) != ensemble_fields or payload["schema"] != state_ensemble_schema \
            or payload["scope"] != state_ensemble_scope:
        raise ValueError("Native state ensemble schema/scope differs")
    for name in ("producer_commit", "producer_tree"):
        require_git_oid(payload[name], name)
    validate_native_state_context(payload["context"])
    if payload["state_key_sha"] != canonical_sha256(payload["context"]):
        raise ValueError("Native state ensemble key differs from its context")
    members = payload["members"]
    count = payload["draw_count"]
    if type(count) is not int or not 2 <= count <= maximum_state_ensemble_members \
            or type(members) is not list or len(members) != count \
            or type(payload["selected_row_count"]) is not int or payload["selected_row_count"] != 1:
        raise ValueError("Native state ensemble requires bounded members of exactly one input")
    mode_count = 3 if payload["context"]["system_type"] == "3qubit" else 4
    for member in members:
        if type(member) is not dict or set(member) != member_fields:
            raise ValueError("Native state member fields differ")
        for name in member_fields - {"batch_index", "state_coordinates", "native_global_parameters", "effective_response_parameters"}:
            require_sha256(member[name], name)
        if type(member["batch_index"]) is not int or member["batch_index"] < 0:
            raise ValueError("Native state member batch index differs")
        for name, size in (("state_coordinates", 2 * mode_count * mode_count), ("native_global_parameters", 12),
                           ("effective_response_parameters", 5)):
            if type(member[name]) is not list or len(_finite_vector(member[name], name)) != size:
                raise ValueError("Native state member coordinate dimensions differ")
        coordinates = np.asarray(member["state_coordinates"]).reshape(2, mode_count, mode_count)
        _validate_psd({"state_real": coordinates[0].tolist(), "state_imag": coordinates[1].tolist()}, "state", mode_count)
        native = np.asarray(member["native_global_parameters"], dtype=np.float64)
        if np.any(native[:6] <= 0) or np.any((native[6:] < 0) | (native[6:] > 1)) \
                or abs(float(native[6:].sum()) - 1.0) > 1e-12:
            raise ValueError("Native state global parameters violate the original native head domains")
    for name in ("point_file_sha", "training_identity_sha", "checkpoint_file_sha"):
        if len({member[name] for member in members}) != count:
            raise ValueError("Native state ensemble repeats a point, training identity or checkpoint")
    if canonical_json_bytes(payload["moments"]) != canonical_json_bytes(_state_moments(members)):
        raise ValueError("Native state moments do not reproduce the retained same-input members")
    if payload["unavailable"] != state_unavailable_quantities() or payload["research_only"] is not True \
            or any(payload[name] is not False for name in ("model_forward_executed", "optimizer_constructed",
                "parameters_updated", "selection_evidence", "scientific_gate_issued", "hardware_action_authorized", "causal_gain_available")):
        raise ValueError("Native state uncertainty or authority fields differ")


@dataclass(frozen=True)
class Native_state_ensemble:
    payload_bytes: bytes

    def payload(self) -> dict[str, Any]:
        if type(self.payload_bytes) is not bytes or len(self.payload_bytes) > maximum_state_ensemble_bytes:
            raise ValueError("Native state ensemble payload is not bounded bytes")
        value = parse_canonical_json_object(self.payload_bytes, schema=state_ensemble_schema, exact_fields=ensemble_fields)
        validate_native_state_ensemble(value)
        return value

    def sha256(self) -> str:
        self.payload()
        return hashlib.sha256(self.payload_bytes).hexdigest()


def build_native_state_ensemble(
    point_artifacts: Sequence[Effective_point_inference], *, sample_id: str, expected_state_key_sha: str,
    producer_commit: str, producer_tree: str,
) -> Native_state_ensemble:
    """Aggregate one explicit sample/view across distinct training identities."""
    require_sha256(expected_state_key_sha, "expected_state_key_sha")
    require_git_oid(producer_commit, "state_ensemble_producer_commit")
    require_git_oid(producer_tree, "state_ensemble_producer_tree")
    artifacts = tuple(point_artifacts)
    if not 2 <= len(artifacts) <= maximum_state_ensemble_members \
            or any(type(artifact) is not Effective_point_inference for artifact in artifacts):
        raise ValueError("Native state ensemble requires bounded exact point artifacts")
    members, reference = [], None
    for artifact in artifacts:
        payload = artifact.payload()
        context, row, provenance = select_native_state_row(payload, sample_id)
        if canonical_sha256(context) != expected_state_key_sha or reference is not None and context != reference:
            raise ValueError("Native state sample/session/view/source binding differs across members")
        reference = context
        state, nuisance, response = _row_draw_vectors({"rows": [row]})
        model = payload["model_artifact"]
        members.append({**provenance, "point_file_sha": artifact.file_sha,
            "training_identity_sha": model["training_identity_sha"], "checkpoint_file_sha": model["training_checkpoint_file_sha"],
            "model_state_sha": model["model_state_sha"], "state_coordinates": state[0].tolist(),
            "native_global_parameters": nuisance[0].tolist(), "effective_response_parameters": response[0].tolist()})
    # Cross-seed ordering is also canonical; row positions never pair states.
    members.sort(key=lambda member: member["training_identity_sha"])
    result = Native_state_ensemble(canonical_json_bytes({
        "schema": state_ensemble_schema, "scope": state_ensemble_scope,
        "producer_commit": producer_commit, "producer_tree": producer_tree,
        "context": reference, "state_key_sha": expected_state_key_sha, "members": members,
        "moments": _state_moments(members), "draw_count": len(members), "selected_row_count": 1,
        "unavailable": state_unavailable_quantities(), "model_forward_executed": False,
        "optimizer_constructed": False, "parameters_updated": False, "selection_evidence": False,
        "scientific_gate_issued": False, "hardware_action_authorized": False, "causal_gain_available": False,
        "research_only": True,
    }))
    result.payload()
    return result


def verify_native_state_companions(ensemble: Native_state_ensemble, points: Sequence[Effective_point_inference]) -> None:
    """Rebuild a derived state artifact from exact complete point companions."""
    if type(ensemble) is not Native_state_ensemble:
        raise TypeError("Native state verification requires an exact typed ensemble")
    payload = ensemble.payload()
    rebuilt = build_native_state_ensemble(points, sample_id=payload["context"]["sample_id"],
        expected_state_key_sha=payload["state_key_sha"], producer_commit=payload["producer_commit"],
        producer_tree=payload["producer_tree"])
    if rebuilt.payload_bytes != ensemble.payload_bytes:
        raise ValueError("Native state artifact does not reproduce its actual point companions")
