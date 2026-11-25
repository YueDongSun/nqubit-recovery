"""Source-bound ensemble moments for the effective Track B observation model.

The effective point producer is deterministic for one checkpoint.  This
module turns several independently produced point artifacts into an explicit
ensemble-moment artifact without confusing row variability with posterior
uncertainty.  It deliberately does not estimate microscopic quantities, OOD
confidence, or causal action response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    require_git_oid,
    require_sha256,
)
from .effective_inference_artifact import (
    Effective_point_inference,
    validate_effective_point_artifact,
)
from .effective_family_context_v3 import MODEL_SCHEMA as family_model_schema
from .effective_observation_v2 import MODEL_SCHEMA as effective_model_schema


NATIVE_POSTERIOR_SCHEMA = "TrackBNativeEnsemblePosteriorV1"
NATIVE_POSTERIOR_SCOPE = "MODEL_ENSEMBLE_MOMENTS_NOT_CALIBRATED_POSTERIOR"
_UNAVAILABLE_NATIVE_POSTERIOR_SIGNATURE = (
    ("microscopic_efficiency", "NOT_IDENTIFIED"),
    ("detector_dark_background", "NOT_IDENTIFIED"),
    ("source_multiplicity", "NOT_IDENTIFIED"),
    ("state_posterior_covariance", "ENSEMBLE_DISPERSION_NOT_CALIBRATED_STATE_POSTERIOR"),
    ("posterior_draws", "ENSEMBLE_MEMBERS_RETAINED_AS_MOMENTS_ONLY"),
    ("cross_operation_sampling_covariance", "NOT_QUALIFIED"),
    ("ood_confidence", "NOT_CALIBRATED"),
    ("causal_action_jacobian", "NOT_IDENTIFIED_FROM_STATIC_OBSERVATIONS"),
)

# Retain the historical export for callers, but never use this mutable
# compatibility mapping as the validation baseline.
UNAVAILABLE_NATIVE_POSTERIOR_QUANTITIES = {
    name: {"status": status, "value": None}
    for name, status in _UNAVAILABLE_NATIVE_POSTERIOR_SIGNATURE
}


def _fresh_unavailable_quantities() -> dict[str, dict[str, object]]:
    return {
        name: {"status": status, "value": None}
        for name, status in _UNAVAILABLE_NATIVE_POSTERIOR_SIGNATURE
    }


def _finite_matrix(values: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, ...], ...]:
    try:
        raw = np.asarray(values, dtype=object)
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite square matrix") from error
    if raw.ndim != 2 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] \
            or not all(type(value) is not bool and isinstance(value, Real) for value in raw.reshape(-1)) \
            or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite square matrix")
    if not np.allclose(matrix, matrix.T, atol=1.0e-10, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    if float(np.linalg.eigvalsh(matrix).min(initial=0.0)) < -1.0e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        raw = tuple(values)
        if any(type(value) is bool or not isinstance(value, Real) for value in raw):
            raise ValueError
        result = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be nonempty and finite") from error
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def _protocol_without_training_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(protocol)
    value.pop("training_identity_sha", None)
    return value


def _row_draw_vectors(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = payload["rows"]
    state_values = []
    nuisance_values = []
    response_values = []
    for row in rows:
        values = row["values"]
        state_values.append(np.concatenate((
            np.asarray(values["effective_psd_real"], dtype=np.float64).reshape(-1),
            np.asarray(values["effective_psd_imag"], dtype=np.float64).reshape(-1),
        )))
        nuisance_values.append(np.concatenate((
            np.asarray(values["global_multi_mean"], dtype=np.float64),
            np.asarray(values["global_multi_dispersion"], dtype=np.float64),
            np.asarray(values["global_pair_probability"], dtype=np.float64),
        )))
        response_values.append(np.concatenate((
            np.asarray(values["effective_offsets"], dtype=np.float64),
            np.asarray(values["effective_contrasts"], dtype=np.float64),
        )))
    result = tuple(np.asarray(value, dtype=np.float64) for value in (
        state_values, nuisance_values, response_values,
    ))
    if any(array.ndim != 2 or not np.all(np.isfinite(array)) for array in result):
        raise ValueError("native posterior point rows must contain finite rank-2 vectors")
    return result


def _ensemble_mean_covariance(draws: Sequence[np.ndarray], name: str) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} requires at least two finite ensemble draws")
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    if values.shape[1] == 1:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    covariance = 0.5 * (np.asarray(covariance, dtype=np.float64) + np.asarray(covariance, dtype=np.float64).T)
    return _finite_vector(mean.tolist(), f"{name}_mean"), _finite_matrix(covariance.tolist(), f"{name}_covariance")


def _require_same_point_protocol(payloads: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], tuple[str, ...]]:
    first = payloads[0]
    protocol = first["inference_protocol"]
    common = _protocol_without_training_identity(protocol)
    protocol_shas = []
    for payload in payloads:
        current = payload["inference_protocol"]
        protocol_shas.append(str(payload["inference_protocol_sha"]))
        if _protocol_without_training_identity(current) != common:
            raise ValueError("native posterior ensemble point protocols differ")
        if payload["source_commit"] != first["source_commit"] or payload["source_tree"] != first["source_tree"]:
            raise ValueError("native posterior ensemble reader source identities differ")
        if payload["model_schema"] != first["model_schema"]:
            raise ValueError("native posterior ensemble model schema differs")
        model = payload["model_artifact"]
        first_model = first["model_artifact"]
        for name in ("kind", "model_config_sha", "producer_commit", "producer_tree",
                     "training_source_commit", "training_source_tree"):
            if model[name] != first_model[name]:
                raise ValueError(f"native posterior ensemble model {name} differs")
    return common, tuple(protocol_shas)


@dataclass(frozen=True)
class TrackBNativeEnsemblePosteriorV1:
    """Explicit model-ensemble moments with no hidden calibration claims."""

    system_type: str
    source_commit: str
    source_tree: str
    model_schema: str
    inference_protocol_common_sha: str
    inference_protocol_sha: tuple[str, ...]
    point_artifact_file_shas: tuple[str, ...]
    training_identity_shas: tuple[str, ...]
    training_source_commits: tuple[str, ...]
    training_source_trees: tuple[str, ...]
    state_coordinate_mean: tuple[float, ...]
    state_coordinate_covariance: tuple[tuple[float, ...], ...]
    nuisance_mean: tuple[float, ...]
    nuisance_covariance: tuple[tuple[float, ...], ...]
    measurement_rf_residual_mean: tuple[float, ...]
    row_count: int
    forward_evaluations: tuple[int, ...]
    draw_count: int
    scope: str = NATIVE_POSTERIOR_SCOPE
    uncertainty_status: str = "ENSEMBLE_DISPERSION_NOT_CALIBRATED_STATE_POSTERIOR"
    ood_status: str = "NOT_CALIBRATED"
    causal_status: str = "NOT_IDENTIFIED_FROM_STATIC_OBSERVATIONS"
    unavailable: Mapping[str, Mapping[str, Any]] = field(
        default_factory=_fresh_unavailable_quantities
    )
    research_only: bool = True
    scientific_gate_issued: bool = False
    hardware_action_authorized: bool = False
    causal_gain_available: bool = False
    schema: str = field(default=NATIVE_POSTERIOR_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("native posterior system type is unsupported")
        if self.model_schema not in (effective_model_schema, family_model_schema):
            raise ValueError("native posterior model schema is not registered")
        require_git_oid(self.source_commit, "native_posterior_source_commit")
        require_git_oid(self.source_tree, "native_posterior_source_tree")
        require_sha256(self.inference_protocol_common_sha, "inference_protocol_common_sha")
        if self.scope != NATIVE_POSTERIOR_SCOPE or self.uncertainty_status != "ENSEMBLE_DISPERSION_NOT_CALIBRATED_STATE_POSTERIOR":
            raise ValueError("native posterior scope/status is not registered")
        if self.ood_status != "NOT_CALIBRATED" or self.causal_status != "NOT_IDENTIFIED_FROM_STATIC_OBSERVATIONS":
            raise ValueError("native posterior unavailable statuses are not registered")
        if type(self.draw_count) is not int or self.draw_count < 2:
            raise ValueError("native posterior requires at least two ensemble draws")
        if len(self.point_artifact_file_shas) != self.draw_count \
                or len(self.training_identity_shas) != self.draw_count \
                or len(self.inference_protocol_sha) != self.draw_count \
                or len(self.training_source_commits) != self.draw_count \
                or len(self.training_source_trees) != self.draw_count \
                or len(self.forward_evaluations) != self.draw_count:
            raise ValueError("native posterior ensemble identities are incomplete")
        if len(set(self.point_artifact_file_shas)) != self.draw_count \
                or len(set(self.training_identity_shas)) != self.draw_count:
            raise ValueError("native posterior ensemble identities must be unique")
        for values, name in (
            (self.point_artifact_file_shas, "point_artifact_file_sha"),
            (self.training_identity_shas, "training_identity_sha"),
            (self.inference_protocol_sha, "inference_protocol_sha"),
        ):
            for value in values:
                require_sha256(value, name)
        for values, name in (
            (self.training_source_commits, "training_source_commit"),
            (self.training_source_trees, "training_source_tree"),
        ):
            for value in values:
                require_git_oid(value, name)
        if any(type(value) is not int or value < 1 for value in self.forward_evaluations):
            raise ValueError("native posterior forward evaluation counts are invalid")
        if type(self.row_count) is not int or self.row_count < 1:
            raise ValueError("native posterior row count is invalid")
        mode_count = 3 if self.system_type == "3qubit" else 4
        state_dimension = 2 * mode_count * mode_count
        nuisance_dimension = 12
        state_mean = _finite_vector(self.state_coordinate_mean, "state_coordinate_mean")
        state_covariance = _finite_matrix(self.state_coordinate_covariance, "state_coordinate_covariance")
        nuisance_mean = _finite_vector(self.nuisance_mean, "nuisance_mean")
        nuisance_covariance = _finite_matrix(self.nuisance_covariance, "nuisance_covariance")
        response_mean = _finite_vector(self.measurement_rf_residual_mean, "measurement_rf_residual_mean")
        if len(state_mean) != state_dimension or len(state_covariance) != state_dimension:
            raise ValueError("native posterior state dimensions do not match the registered system")
        if len(nuisance_mean) != nuisance_dimension or len(nuisance_covariance) != nuisance_dimension:
            raise ValueError("native posterior nuisance dimensions are not registered")
        if len(response_mean) != 5:
            raise ValueError("native posterior response layout must contain 3 offsets and 2 contrasts")
        if any(len(row) != state_dimension for row in state_covariance) \
                or any(len(row) != nuisance_dimension for row in nuisance_covariance):
            raise ValueError("native posterior covariance dimensions do not match their means")
        if set(self.unavailable) != {name for name, _status in _UNAVAILABLE_NATIVE_POSTERIOR_SIGNATURE}:
            raise ValueError("native posterior unavailable quantity names changed")
        for name, expected_status in _UNAVAILABLE_NATIVE_POSTERIOR_SIGNATURE:
            entry = self.unavailable[name]
            if not isinstance(entry, Mapping) or set(entry) != {"status", "value"} \
                    or type(entry["status"]) is not str or entry["status"] != expected_status \
                    or entry["value"] is not None:
                raise ValueError("native posterior unavailable quantities changed")
        object.__setattr__(self, "unavailable", MappingProxyType({
            name: MappingProxyType({"status": status, "value": None})
            for name, status in _UNAVAILABLE_NATIVE_POSTERIOR_SIGNATURE
        }))
        if any(type(value) is not bool for value in (
            self.research_only, self.scientific_gate_issued,
            self.hardware_action_authorized, self.causal_gain_available,
        )) or not self.research_only or self.scientific_gate_issued \
                or self.hardware_action_authorized or self.causal_gain_available:
            raise ValueError("native posterior authority flags are invalid")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "model_schema": self.model_schema,
            "scope": self.scope,
            "inference_protocol_common_sha": self.inference_protocol_common_sha,
            "inference_protocol_sha": list(self.inference_protocol_sha),
            "point_artifact_file_shas": list(self.point_artifact_file_shas),
            "training_identity_shas": list(self.training_identity_shas),
            "training_source_commits": list(self.training_source_commits),
            "training_source_trees": list(self.training_source_trees),
            "state_coordinate_mean": list(self.state_coordinate_mean),
            "state_coordinate_covariance": [list(row) for row in self.state_coordinate_covariance],
            "nuisance_mean": list(self.nuisance_mean),
            "nuisance_covariance": [list(row) for row in self.nuisance_covariance],
            "measurement_rf_residual_mean": list(self.measurement_rf_residual_mean),
            "row_count": self.row_count,
            "forward_evaluations": list(self.forward_evaluations),
            "draw_count": self.draw_count,
            "uncertainty_status": self.uncertainty_status,
            "ood_status": self.ood_status,
            "causal_status": self.causal_status,
            "unavailable": {
                name: {"status": entry["status"], "value": None}
                for name, entry in self.unavailable.items()
            },
            "research_only": self.research_only,
            "scientific_gate_issued": self.scientific_gate_issued,
            "hardware_action_authorized": self.hardware_action_authorized,
            "causal_gain_available": self.causal_gain_available,
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return canonical_sha256(self.canonical_json_bytes())

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBNativeEnsemblePosteriorV1":
        fields = frozenset({
            "schema", "system_type", "source_commit", "source_tree", "model_schema", "scope",
            "inference_protocol_common_sha", "inference_protocol_sha", "point_artifact_file_shas",
            "training_identity_shas", "training_source_commits", "training_source_trees",
            "state_coordinate_mean", "state_coordinate_covariance", "nuisance_mean",
            "nuisance_covariance", "measurement_rf_residual_mean", "row_count", "forward_evaluations",
            "draw_count", "uncertainty_status", "ood_status", "causal_status", "unavailable",
            "research_only", "scientific_gate_issued", "hardware_action_authorized", "causal_gain_available",
        })
        decoded = parse_canonical_json_object(payload, schema=NATIVE_POSTERIOR_SCHEMA, exact_fields=fields)
        result = cls(
            system_type=decoded["system_type"], source_commit=decoded["source_commit"],
            source_tree=decoded["source_tree"], model_schema=decoded["model_schema"],
            scope=decoded["scope"], inference_protocol_common_sha=decoded["inference_protocol_common_sha"],
            inference_protocol_sha=tuple(decoded["inference_protocol_sha"]),
            point_artifact_file_shas=tuple(decoded["point_artifact_file_shas"]),
            training_identity_shas=tuple(decoded["training_identity_shas"]),
            training_source_commits=tuple(decoded["training_source_commits"]),
            training_source_trees=tuple(decoded["training_source_trees"]),
            state_coordinate_mean=tuple(decoded["state_coordinate_mean"]),
            state_coordinate_covariance=tuple(tuple(row) for row in decoded["state_coordinate_covariance"]),
            nuisance_mean=tuple(decoded["nuisance_mean"]),
            nuisance_covariance=tuple(tuple(row) for row in decoded["nuisance_covariance"]),
            measurement_rf_residual_mean=tuple(decoded["measurement_rf_residual_mean"]),
            row_count=decoded["row_count"], forward_evaluations=tuple(decoded["forward_evaluations"]),
            draw_count=decoded["draw_count"], uncertainty_status=decoded["uncertainty_status"],
            ood_status=decoded["ood_status"], causal_status=decoded["causal_status"],
            unavailable=decoded["unavailable"], research_only=decoded["research_only"],
            scientific_gate_issued=decoded["scientific_gate_issued"],
            hardware_action_authorized=decoded["hardware_action_authorized"],
            causal_gain_available=decoded["causal_gain_available"],
        )
        if result.canonical_json_bytes() != payload:
            raise ValueError("native posterior payload is not canonical after typed parsing")
        return result


def build_native_ensemble_posterior(
    point_artifacts: Sequence[Effective_point_inference],
    *,
    expected_source_commit: str | None = None,
    expected_source_tree: str | None = None,
) -> TrackBNativeEnsemblePosteriorV1:
    """Build ensemble moments from actual validated point artifacts."""
    artifacts = tuple(point_artifacts)
    if len(artifacts) < 2 or any(type(value) is not Effective_point_inference for value in artifacts):
        raise ValueError("native posterior requires at least two exact point artifacts")
    payloads = tuple(value.payload() for value in artifacts)
    first = payloads[0]
    if expected_source_commit is not None or expected_source_tree is not None:
        if expected_source_commit is None or expected_source_tree is None:
            raise ValueError("native posterior expected source identity must be complete")
        require_git_oid(expected_source_commit, "expected_native_posterior_source_commit")
        require_git_oid(expected_source_tree, "expected_native_posterior_source_tree")
        if (first["source_commit"], first["source_tree"]) != (expected_source_commit, expected_source_tree):
            raise ValueError("native posterior point source identity differs from the expected reader")
    if first["inference_protocol"]["outer_role"] not in {"validation", "calibration"}:
        raise ValueError("native posterior requires an explicit non-training development fold")
    common_protocol, protocol_shas = _require_same_point_protocol(payloads)
    row_vectors = tuple(_row_draw_vectors(payload) for payload in payloads)
    row_count = row_vectors[0][0].shape[0]
    if any(value[0].shape[0] != row_count for value in row_vectors):
        raise ValueError("native posterior point row counts differ")
    state_mean, state_covariance = _ensemble_mean_covariance(
        tuple(value[0].mean(axis=0) for value in row_vectors), "state_coordinate",
    )
    nuisance_mean, nuisance_covariance = _ensemble_mean_covariance(
        tuple(value[1].mean(axis=0) for value in row_vectors), "nuisance",
    )
    response_draws = np.asarray(tuple(value[2].mean(axis=0) for value in row_vectors), dtype=np.float64)
    response_mean = _finite_vector(response_draws.mean(axis=0).tolist(), "measurement_rf_residual_mean")
    source_commits = tuple(payload["model_artifact"]["training_source_commit"] for payload in payloads)
    source_trees = tuple(payload["model_artifact"]["training_source_tree"] for payload in payloads)
    result = TrackBNativeEnsemblePosteriorV1(
        system_type=first["system_type"], source_commit=first["source_commit"], source_tree=first["source_tree"],
        model_schema=first["model_schema"], inference_protocol_common_sha=canonical_sha256(common_protocol),
        inference_protocol_sha=protocol_shas,
        point_artifact_file_shas=tuple(value.file_sha for value in artifacts),
        training_identity_shas=tuple(payload["model_artifact"]["training_identity_sha"] for payload in payloads),
        training_source_commits=source_commits, training_source_trees=source_trees,
        state_coordinate_mean=state_mean, state_coordinate_covariance=state_covariance,
        nuisance_mean=nuisance_mean, nuisance_covariance=nuisance_covariance,
        measurement_rf_residual_mean=response_mean, row_count=row_count,
        forward_evaluations=tuple(payload["forward_evaluations"] for payload in payloads),
        draw_count=len(artifacts),
    )
    result.to_dict()
    return result


def validate_native_ensemble_posterior(payload: Mapping[str, Any]) -> None:
    """Validate a serialized ensemble-moment artifact without executing a model."""
    artifact = TrackBNativeEnsemblePosteriorV1.from_canonical_json_bytes(canonical_json_bytes(payload))
    if artifact.to_dict() != dict(payload):
        raise ValueError("native posterior typed fields differ from their serialized payload")


__all__ = [
    "NATIVE_POSTERIOR_SCHEMA", "NATIVE_POSTERIOR_SCOPE", "UNAVAILABLE_NATIVE_POSTERIOR_QUANTITIES",
    "TrackBNativeEnsemblePosteriorV1", "build_native_ensemble_posterior",
    "validate_native_ensemble_posterior",
]
