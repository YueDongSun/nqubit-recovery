"""Nominal projective state sensitivity for the Track B CSM boundary.

The point ensemble contains effective PSD estimates, not executed action
transitions.  This module therefore computes a source-bound *nominal* global-W
projective sensitivity diagnostic only.  Its output is intentionally
unqualified for causal FD selection and hardware action.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from numbers import Real
from typing import Any, Sequence

import numpy as np

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    require_git_oid,
    require_sha256,
)
from ..csm_contract import SingularSupportContract, WhiteningThresholdContract
from .coordinate_abi import coordinate_layout_sha, grouped_to_mode_action
from .effective_inference_artifact import Effective_point_inference
from ..forward_instance import CSMForwardInstanceContract, build_forward_instance_contract
from .forward_draw_set import BoundTrackBForwardDrawSetV1, forward_callable_identity_sha
from .source_artifacts import FORWARD_SOURCE_ARTIFACT_NAMES, TrackBActionfulSourceArtifactBundleV1
from .native_posterior import (
    TrackBNativeEnsemblePosteriorV1,
    _row_draw_vectors,
    build_native_ensemble_posterior,
)
from .real_csm import CSM_GEOMETRY_STATUSES, RealCSMSupportV1, array_identity_sha256


NATIVE_SENSITIVITY_SCHEMA = "TrackBNativeProjectiveSensitivityV1"
NATIVE_SENSITIVITY_SCOPE = "NOMINAL_PROJECTIVE_STATE_SENSITIVITY_NOT_CAUSAL_ACTION_RESPONSE"
CAUSAL_STATUS = "NOT_IDENTIFIED_FROM_STATIC_OBSERVATIONS"


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def _strict_numeric_matrix(values: object, name: str) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object)
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric matrix values") from error
    if raw.ndim != 2 or not all(
        type(value) is not bool and isinstance(value, Real) for value in raw.reshape(-1)
    ) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite numeric matrix values")
    return matrix


def _projective_state(state_vector: np.ndarray, action: np.ndarray, mode_count: int) -> np.ndarray:
    values = _finite_vector(state_vector, "state_vector")
    expected = 2 * mode_count * mode_count
    if values.size != expected or action.shape != (2 * (mode_count - 1),):
        raise ValueError("projective state/action dimensions do not match the registered system")
    split = mode_count * mode_count
    rho = values[:split].reshape(mode_count, mode_count) \
        + 1j * values[split:].reshape(mode_count, mode_count)
    rho = 0.5 * (rho + rho.conj().T)
    mode_action = grouped_to_mode_action(
        action[:mode_count - 1], action[mode_count - 1:], mode_count,
    )
    mode_amplitudes = np.asarray([pair[0] for pair in mode_action], dtype=np.float64)
    mode_phases = np.asarray([pair[1] for pair in mode_action], dtype=np.float64)
    # A common real shift cancels during projective normalization.  Removing
    # it keeps finite, otherwise valid large coordinates from overflowing.
    mode_amplitudes -= float(mode_amplitudes.max())
    weights = np.exp(mode_amplitudes + 1j * mode_phases)
    transformed = weights[:, None] * rho * weights.conj()[None, :]
    trace = float(np.trace(transformed).real)
    if not math.isfinite(trace) or trace <= 0.0:
        raise ValueError("projective state normalization is not positive")
    transformed /= trace
    return np.concatenate((transformed.real.reshape(-1), transformed.imag.reshape(-1)))


def native_projective_forward(
    action_point: np.ndarray, nuisance_point: np.ndarray, draw_parameter: Any,
) -> np.ndarray:
    """Evaluate the registered nominal projective state forward for one draw."""
    action = _finite_vector(action_point, "action_point")
    nuisance = _finite_vector(nuisance_point, "nuisance_point")
    if nuisance.size != 1:
        raise ValueError("nominal projective forward requires one explicit nuisance coordinate")
    if not isinstance(draw_parameter, dict) or type(draw_parameter.get("mode_count")) is not int:
        raise ValueError("nominal projective draw parameter is invalid")
    return _projective_state(
        np.asarray(draw_parameter.get("state_vector"), dtype=np.float64),
        action,
        int(draw_parameter["mode_count"]),
    )


def native_projective_covariance(
    baseline_output: np.ndarray, draw_parameter: Any,
) -> np.ndarray:
    """Return the registered covariance attached to one nominal draw."""
    baseline = _finite_vector(baseline_output, "baseline_output")
    if not isinstance(draw_parameter, dict):
        raise ValueError("nominal projective draw parameter is invalid")
    covariance = np.asarray(draw_parameter.get("covariance"), dtype=np.float64)
    if covariance.shape != (baseline.size, baseline.size) or not np.all(np.isfinite(covariance)):
        raise ValueError("nominal projective covariance shape is invalid")
    return 0.5 * (covariance + covariance.T)


def _covariance_from_draws(draws: np.ndarray) -> np.ndarray:
    if draws.ndim != 2 or draws.shape[0] < 2 or not np.all(np.isfinite(draws)):
        raise ValueError("sensitivity covariance requires at least two finite draws")
    covariance = np.cov(draws, rowvar=False, ddof=1)
    if draws.shape[1] == 1:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    covariance = 0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def _forward_identity(point_artifacts: Sequence[Effective_point_inference], mode_count: int) -> str:
    return canonical_sha256({
        "schema": "TrackBNativeNominalProjectiveForwardV1",
        "point_artifact_file_shas": [value.file_sha for value in point_artifacts],
        "system_type": "3qubit" if mode_count == 3 else "4qubit",
        "action_chart": "TRACKB_GROUPED_HELMERT_RELATIVE_LOG_AMPLITUDE_AND_PHASE",
        "action_application": "DIAGONAL_PROJECTIVE_STATE_TRANSFORM_ON_EFFECTIVE_PSD",
    })


def _read_source_json_value(
    source_artifact_bundle: TrackBActionfulSourceArtifactBundleV1,
    name: str,
) -> Any:
    """Read one source-bound canonical JSON value, including array roots."""
    payload = source_artifact_bundle.read_bytes(name)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"native projective source artifact {name} is not canonical JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"native projective source artifact {name} is not canonical JSON")
    return value


def _strict_numeric_vector(values: object, name: str) -> np.ndarray:
    """Read a finite JSON vector without accepting coercible strings."""
    try:
        raw = np.asarray(values, dtype=object)
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric vector") from error
    if raw.ndim != 1 or not raw.size or not all(
        type(value) is not bool and isinstance(value, Real) for value in raw.reshape(-1)
    ) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite numeric vector")
    return vector


def build_native_projective_forward_instance(
    posterior: TrackBNativeEnsemblePosteriorV1,
    source_artifact_bundle: TrackBActionfulSourceArtifactBundleV1,
    *,
    finite_difference_step: float,
) -> CSMForwardInstanceContract:
    """Reconstruct the native projective instance from its source preimages."""
    if type(posterior) is not TrackBNativeEnsemblePosteriorV1:
        raise TypeError("native projective instance requires a typed posterior")
    if type(source_artifact_bundle) is not TrackBActionfulSourceArtifactBundleV1:
        raise TypeError("native projective instance requires a typed source bundle")
    step = float(finite_difference_step)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("native projective instance finite-difference step must be positive and finite")
    source_artifact_bundle.assert_current(FORWARD_SOURCE_ARTIFACT_NAMES)
    operation_descriptor = source_artifact_bundle.read_canonical_json_object("operation_descriptor")
    session_context = source_artifact_bundle.read_canonical_json_object("session_context")
    exposure_context = source_artifact_bundle.read_canonical_json_object("exposure_context")
    base_point = _read_source_json_value(source_artifact_bundle, "base_point")
    _read_source_json_value(source_artifact_bundle, "nuisance_point")
    mode_count = 3 if posterior.system_type == "3qubit" else 4
    return build_forward_instance_contract(
        sample_id=f"trackb_native_projective_{posterior.system_type}_ensemble",
        current_state={"schema": "TrackBNativeEffectiveStateV1", "posterior_sha": posterior.sha256()},
        operation_point=operation_descriptor,
        intervention={"schema": "TrackBNativeProjectiveInterventionV1", "base_point": base_point},
        intervention_parameters={
            "schema": "TrackBNativeProjectiveInterventionParametersV1",
            "finite_difference_step": step,
        },
        feedback_measurement_schedule=session_context,
        shot_contract=exposure_context,
        u0=base_point,
        manifold={"schema": "TrackBNativeProjectiveManifoldV1", "mode_count": mode_count},
        forward_code=forward_callable_identity_sha(native_projective_forward),
        covariance=forward_callable_identity_sha(native_projective_covariance),
    )


def _require_native_projective_draw_parameter_binding(
    posterior: TrackBNativeEnsemblePosteriorV1,
    point_artifacts: Sequence[Effective_point_inference],
    forward_draw_set: BoundTrackBForwardDrawSetV1,
) -> None:
    """Bind each source draw state and covariance to the point ensemble."""
    source_bundle = forward_draw_set.source_artifact_bundle
    if source_bundle is None:
        raise ValueError("native projective draw binding requires a source bundle")
    source_payload = source_bundle.read_canonical_json_object("draw_parameters")
    if set(source_payload) != {"schema", "draw_ids", "draw_parameters"} \
            or source_payload["schema"] != "TrackBDrawParametersV1":
        raise ValueError("native projective draw parameter source schema is invalid")
    source_ids = source_payload["draw_ids"]
    source_parameters = source_payload["draw_parameters"]
    if type(source_ids) is not list or tuple(source_ids) != tuple(posterior.training_identity_shas) \
            or tuple(forward_draw_set.draw_ids) != tuple(posterior.training_identity_shas):
        raise ValueError("native projective draw parameter IDs differ from the point ensemble")
    if type(source_parameters) is not list or len(source_parameters) != len(point_artifacts):
        raise ValueError("native projective draw parameter count differs from the point ensemble")
    point_states = tuple(
        _row_draw_vectors(artifact.payload())[0].mean(axis=0)
        for artifact in point_artifacts
    )
    expected_covariance = np.asarray(posterior.state_coordinate_covariance, dtype=np.float64)
    mode_count = 3 if posterior.system_type == "3qubit" else 4
    for index, (parameter, expected_state) in enumerate(zip(source_parameters, point_states, strict=True)):
        if type(parameter) is not dict or set(parameter) != {"mode_count", "state_vector", "covariance"} \
                or type(parameter["mode_count"]) is not int or parameter["mode_count"] != mode_count:
            raise ValueError(f"native projective draw parameter {index} schema is invalid")
        actual_state = _strict_numeric_vector(parameter["state_vector"], "native draw state_vector")
        if actual_state.shape != expected_state.shape or not np.array_equal(actual_state, expected_state):
            raise ValueError("native projective draw state vector does not match its point artifact")
        actual_covariance = _strict_numeric_matrix(parameter["covariance"], "native draw covariance")
        if actual_covariance.shape != expected_covariance.shape \
                or not np.array_equal(actual_covariance, expected_covariance):
            raise ValueError("native projective draw covariance does not match its posterior")
    source_base = _strict_numeric_vector(_read_source_json_value(source_bundle, "base_point"), "native base_point")
    source_nuisance = _strict_numeric_vector(
        _read_source_json_value(source_bundle, "nuisance_point"), "native nuisance_point",
    )
    if array_identity_sha256(source_base) != forward_draw_set.base_point_sha \
            or array_identity_sha256(source_nuisance) != forward_draw_set.nuisance_point_sha:
        raise ValueError("native projective source points differ from the live draw-set inputs")


@dataclass(frozen=True)
class TrackBNativeProjectiveSensitivityV1:
    """A nominal CSM diagnostic with a hard noncausal boundary."""

    system_type: str
    source_commit: str
    source_tree: str
    posterior_sha: str
    point_artifact_file_shas: tuple[str, ...]
    training_identity_shas: tuple[str, ...]
    action_coordinate_layout_sha: str
    forward_sha: str
    covariance_sha: str
    jacobian_draws: tuple[tuple[tuple[float, ...], ...], ...]
    covariance_draws: tuple[tuple[tuple[float, ...], ...], ...]
    nuisance_jacobian_draws: tuple[tuple[tuple[float, ...], ...], ...]
    csm: dict[str, Any]
    forward_draw_set_sha: str | None
    forward_source_binding_sha: str | None
    forward_source_artifact_shas: tuple[tuple[str, str], ...]
    finite_difference_step: float
    draw_count: int
    observation_dimension: int
    action_dimension: int
    scope: str = NATIVE_SENSITIVITY_SCOPE
    causal_status: str = CAUSAL_STATUS
    input_identity_verified: bool = False
    causal_gain_available: bool = False
    hardware_action_authorized: bool = False
    scientific_gate_issued: bool = False
    research_only: bool = True
    schema: str = field(default=NATIVE_SENSITIVITY_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("native projective sensitivity system type is unsupported")
        require_git_oid(self.source_commit, "sensitivity_source_commit")
        require_git_oid(self.source_tree, "sensitivity_source_tree")
        require_sha256(self.posterior_sha, "sensitivity_posterior_sha")
        require_sha256(self.action_coordinate_layout_sha, "action_coordinate_layout_sha")
        require_sha256(self.forward_sha, "sensitivity_forward_sha")
        require_sha256(self.covariance_sha, "sensitivity_covariance_sha")
        if self.scope != NATIVE_SENSITIVITY_SCOPE or self.causal_status != CAUSAL_STATUS:
            raise ValueError("native projective sensitivity status is not registered")
        if type(self.draw_count) is not int or self.draw_count < 2:
            raise ValueError("native projective sensitivity requires at least two draws")
        if len(self.point_artifact_file_shas) != self.draw_count \
                or len(self.training_identity_shas) != self.draw_count \
                or len(self.jacobian_draws) != self.draw_count \
                or len(self.covariance_draws) != self.draw_count \
                or len(self.nuisance_jacobian_draws) != self.draw_count:
            raise ValueError("native projective sensitivity draw identities are incomplete")
        for value in (*self.point_artifact_file_shas, *self.training_identity_shas):
            require_sha256(value, "sensitivity_draw_identity")
        if type(self.finite_difference_step) not in (float, int) \
                or not math.isfinite(float(self.finite_difference_step)) \
                or float(self.finite_difference_step) <= 0.0:
            raise ValueError("sensitivity finite-difference step is invalid")
        if type(self.observation_dimension) is not int or self.observation_dimension <= 0 \
                or type(self.action_dimension) is not int or self.action_dimension <= 0:
            raise ValueError("sensitivity dimensions are invalid")
        mode_count = 3 if self.system_type == "3qubit" else 4
        if self.observation_dimension != 2 * mode_count * mode_count \
                or self.action_dimension != 2 * (mode_count - 1) \
                or self.action_coordinate_layout_sha != coordinate_layout_sha(mode_count):
            raise ValueError("sensitivity system dimensions or coordinate layout are not registered")
        for draws, name in (
            (self.jacobian_draws, "jacobian_draws"),
            (self.covariance_draws, "covariance_draws"),
            (self.nuisance_jacobian_draws, "nuisance_jacobian_draws"),
        ):
            for value in draws:
                _strict_numeric_matrix(value, name)
        if any(np.asarray(value).shape != (self.observation_dimension, self.action_dimension)
               for value in self.jacobian_draws):
            raise ValueError("sensitivity Jacobian shape is invalid")
        if any(np.asarray(value).shape != (self.observation_dimension, self.observation_dimension)
               for value in self.covariance_draws):
            raise ValueError("sensitivity covariance shape is invalid")
        if any(np.asarray(value).shape != (self.observation_dimension, 1)
               for value in self.nuisance_jacobian_draws):
            raise ValueError("sensitivity nuisance shape is invalid")
        if self.causal_gain_available or self.hardware_action_authorized \
                or self.scientific_gate_issued \
                or not self.research_only:
            raise ValueError("nominal projective sensitivity cannot assert action authority")
        if any(type(getattr(self, name)) is not bool for name in (
            "input_identity_verified", "causal_gain_available", "hardware_action_authorized",
            "scientific_gate_issued", "research_only",
        )):
            raise TypeError("sensitivity authority and identity fields must be exact booleans")
        self._validate_nested_csm()
        if self.input_identity_verified:
            if self.forward_draw_set_sha is None or self.csm.get("input_identity_verified") is not True \
                    or self.csm.get("forward_draw_set_sha") != self.forward_draw_set_sha:
                raise ValueError("verified sensitivity must bind its sealed forward draw set")
            require_sha256(self.forward_draw_set_sha, "sensitivity_forward_draw_set_sha")
            if self.forward_source_binding_sha is None:
                raise ValueError("verified sensitivity must bind its forward source artifacts")
            require_sha256(self.forward_source_binding_sha, "sensitivity_forward_source_binding_sha")
            if tuple(name for name, _sha in self.forward_source_artifact_shas) != tuple(sorted(FORWARD_SOURCE_ARTIFACT_NAMES)):
                raise ValueError("verified sensitivity forward source artifact names are incomplete")
            for _name, sha in self.forward_source_artifact_shas:
                require_sha256(sha, "sensitivity_forward_source_artifact_sha")
        if any(type(item) is not tuple or len(item) != 2
               or type(item[0]) is not str or type(item[1]) is not str
               for item in self.forward_source_artifact_shas):
            raise TypeError("sensitivity source artifact identities must be name/SHA pairs")
        if not self.input_identity_verified and (
                self.forward_draw_set_sha is not None or self.forward_source_binding_sha is not None
                or self.forward_source_artifact_shas or self.csm.get("input_identity_verified") is not False
        ):
            raise ValueError("unverified sensitivity cannot carry a sealed forward draw set")

    def _validate_nested_csm(self) -> None:
        """Recompute nested CSM geometry and cross-check its sealed identities."""
        expected_fields = frozenset({
            "schema", "gram_mean", "gram_variance", "support_basis", "rank_distribution",
            "condition_distribution", "recoverable_basis_stability", "direction_stability",
            "confidence", "forward_sha", "jacobian_sha", "covariance_sha",
            "operation_descriptor_sha", "rf_descriptor_sha", "target_independent",
            "nuisance_jacobian_sha", "whitening_contract_sha", "singular_support_contract_sha",
            "whitening_contract", "singular_support_contract", "draw_count",
            "forward_draw_set_sha", "forward_instance_sha", "forward_draw_ids",
            "forward_draw_ids_sha", "nuisance_profiled", "input_identity_verified",
            "geometry_status",
        })
        if type(self.csm) is not dict or frozenset(self.csm) != expected_fields \
                or self.csm.get("schema") != "RealCSMSupportV1":
            raise ValueError("sensitivity CSM payload fields are invalid")
        for name in ("target_independent", "nuisance_profiled", "input_identity_verified"):
            if type(self.csm[name]) is not bool:
                raise TypeError(f"nested CSM {name} must be an exact boolean")
        if self.csm["target_independent"] is not True or self.csm["nuisance_profiled"] is not True:
            raise ValueError("nested CSM must remain target-independent and nuisance-profiled")
        if self.csm["input_identity_verified"] != self.input_identity_verified \
                or self.csm["forward_sha"] != self.forward_sha \
                or self.csm["draw_count"] != self.draw_count:
            raise ValueError("nested CSM top-level identities disagree")
        if type(self.csm["draw_count"]) is not int \
                or any(type(value) is not int or value < 0 for value in self.csm["rank_distribution"]):
            raise TypeError("nested CSM count fields must use exact integer types")
        if self.csm["geometry_status"] not in CSM_GEOMETRY_STATUSES:
            raise ValueError("nested CSM geometry status is not registered")
        for name in ("forward_draw_set_sha", "forward_instance_sha"):
            if self.csm[name] is not None:
                require_sha256(self.csm[name], f"nested_csm_{name}")
        if self.csm["forward_draw_set_sha"] != self.forward_draw_set_sha:
            raise ValueError("nested CSM forward draw-set identity disagrees")
        if self.input_identity_verified:
            if self.csm["geometry_status"] == "LEGACY_SINGLE_DRAW_UNVERIFIED" \
                    or self.csm["forward_instance_sha"] is None \
                    or self.csm["forward_draw_ids"] != list(self.training_identity_shas):
                raise ValueError("verified nested CSM draw identities are incomplete")
            expected_ids_sha = canonical_sha256({
                "schema": "TrackBForwardDrawIDsV1",
                "draw_ids": list(self.training_identity_shas),
            })
            if self.csm["forward_draw_ids_sha"] != expected_ids_sha:
                raise ValueError("nested CSM forward draw ID identity disagrees")
            source_artifact_shas = dict(self.forward_source_artifact_shas)
            if self.csm["operation_descriptor_sha"] != source_artifact_shas["operation_descriptor"] \
                    or self.csm["rf_descriptor_sha"] != source_artifact_shas["rf_descriptor"]:
                raise ValueError("nested CSM descriptor identities disagree with source artifacts")
        elif any(self.csm[name] is not None for name in (
            "forward_instance_sha", "forward_draw_ids", "forward_draw_ids_sha",
        )):
            raise ValueError("unverified nested CSM cannot carry sealed draw identities")

        whitening_payload = self.csm["whitening_contract"]
        singular_payload = self.csm["singular_support_contract"]
        try:
            whitening = None if whitening_payload is None else WhiteningThresholdContract(**whitening_payload)
            singular = None if singular_payload is None else SingularSupportContract(**singular_payload)
        except (TypeError, ValueError) as error:
            raise ValueError("nested CSM threshold contracts are invalid") from error
        for contract, payload, name in (
            (whitening, whitening_payload, "whitening"),
            (singular, singular_payload, "singular_support"),
        ):
            if (contract is None) != (payload is None):
                raise ValueError("nested CSM threshold contract presence is inconsistent")
            if contract is not None:
                contract.validate()
                if contract.to_dict() != payload or contract.sha256() != self.csm[f"{name}_contract_sha"]:
                    raise ValueError(f"nested CSM {name} contract identity disagrees")
        if whitening is None or singular is None:
            raise ValueError("nested CSM must carry whitening and singular-support contracts")
        for name in ("gram_mean", "gram_variance", "support_basis"):
            _strict_numeric_matrix(self.csm[name], f"nested_csm_{name}")
        for name in ("recoverable_basis_stability", "direction_stability", "confidence"):
            if type(self.csm[name]) is bool or not isinstance(self.csm[name], Real) \
                    or not math.isfinite(float(self.csm[name])):
                raise ValueError(f"nested CSM {name} must be a finite numeric value")
        if any(value is not None and (
            type(value) is bool or not isinstance(value, Real) or not math.isfinite(float(value))
        ) for value in self.csm["condition_distribution"]):
            raise ValueError("nested CSM condition distribution has invalid numeric types")
        try:
            expected = RealCSMSupportV1.from_posterior_draws(
                self.jacobian_draws,
                self.covariance_draws,
                forward_sha=self.forward_sha,
                operation_descriptor_sha=self.csm["operation_descriptor_sha"],
                rf_descriptor_sha=self.csm["rf_descriptor_sha"],
                nuisance_jacobian_draws=self.nuisance_jacobian_draws,
                whitening=whitening,
                singular_support=singular,
            ).to_dict()
        except (TypeError, ValueError) as error:
            raise ValueError("nested CSM geometry cannot be recomputed from outer arrays") from error
        for name in (
            "gram_mean", "gram_variance", "support_basis", "rank_distribution",
            "condition_distribution", "recoverable_basis_stability", "direction_stability",
            "confidence", "forward_sha", "jacobian_sha", "covariance_sha",
            "operation_descriptor_sha", "rf_descriptor_sha", "nuisance_jacobian_sha",
            "whitening_contract_sha", "singular_support_contract_sha", "whitening_contract",
            "singular_support_contract", "draw_count", "nuisance_profiled", "target_independent",
        ):
            if self.csm[name] != expected[name]:
                raise ValueError(f"nested CSM field {name} disagrees with outer arrays")
        expected_status = "LEGACY_SINGLE_DRAW_UNVERIFIED"
        if self.input_identity_verified:
            expected_status = (
                "POSTERIOR_DRAW_AGGREGATED_NUISANCE_PROFILED"
                if np.asarray(expected["support_basis"], dtype=np.float64).shape[1]
                else "POSTERIOR_DRAW_AGGREGATED_NO_RECOVERABLE_SUPPORT"
            )
        if self.csm["geometry_status"] != expected_status:
            raise ValueError("nested CSM geometry status disagrees with its recomputed support")
        if self.input_identity_verified and self.covariance_sha != self.csm["covariance_sha"]:
            raise ValueError("verified sensitivity covariance identity disagrees with nested CSM")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "schema": self.schema, "system_type": self.system_type,
            "source_commit": self.source_commit, "source_tree": self.source_tree,
            "posterior_sha": self.posterior_sha,
            "point_artifact_file_shas": list(self.point_artifact_file_shas),
            "training_identity_shas": list(self.training_identity_shas),
            "action_coordinate_layout_sha": self.action_coordinate_layout_sha,
            "forward_sha": self.forward_sha, "covariance_sha": self.covariance_sha,
            "jacobian_draws": [[list(row) for row in draw] for draw in self.jacobian_draws],
            "covariance_draws": [[list(row) for row in draw] for draw in self.covariance_draws],
            "nuisance_jacobian_draws": [[list(row) for row in draw] for draw in self.nuisance_jacobian_draws],
            "csm": self.csm, "forward_draw_set_sha": self.forward_draw_set_sha,
            "forward_source_binding_sha": self.forward_source_binding_sha,
            "forward_source_artifact_shas": [list(item) for item in self.forward_source_artifact_shas],
            "finite_difference_step": self.finite_difference_step,
            "draw_count": self.draw_count, "observation_dimension": self.observation_dimension,
            "action_dimension": self.action_dimension, "scope": self.scope,
            "causal_status": self.causal_status, "input_identity_verified": self.input_identity_verified,
            "causal_gain_available": self.causal_gain_available,
            "hardware_action_authorized": self.hardware_action_authorized,
            "scientific_gate_issued": self.scientific_gate_issued,
            "research_only": self.research_only,
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return canonical_sha256(self.canonical_json_bytes())

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBNativeProjectiveSensitivityV1":
        fields = frozenset({
            "schema", "system_type", "source_commit", "source_tree", "posterior_sha",
            "point_artifact_file_shas", "training_identity_shas", "action_coordinate_layout_sha",
            "forward_sha", "covariance_sha", "jacobian_draws", "covariance_draws",
            "nuisance_jacobian_draws", "csm", "forward_draw_set_sha", "forward_source_binding_sha",
            "forward_source_artifact_shas", "finite_difference_step", "draw_count",
            "observation_dimension", "action_dimension", "scope", "causal_status",
            "input_identity_verified", "causal_gain_available", "hardware_action_authorized",
            "scientific_gate_issued", "research_only",
        })
        decoded = parse_canonical_json_object(
            payload, schema=NATIVE_SENSITIVITY_SCHEMA, exact_fields=fields,
        )
        result = cls(
            system_type=decoded["system_type"], source_commit=decoded["source_commit"],
            source_tree=decoded["source_tree"], posterior_sha=decoded["posterior_sha"],
            point_artifact_file_shas=tuple(decoded["point_artifact_file_shas"]),
            training_identity_shas=tuple(decoded["training_identity_shas"]),
            action_coordinate_layout_sha=decoded["action_coordinate_layout_sha"],
            forward_sha=decoded["forward_sha"], covariance_sha=decoded["covariance_sha"],
            jacobian_draws=tuple(tuple(tuple(row) for row in draw) for draw in decoded["jacobian_draws"]),
            covariance_draws=tuple(tuple(tuple(row) for row in draw) for draw in decoded["covariance_draws"]),
            nuisance_jacobian_draws=tuple(
                tuple(tuple(row) for row in draw) for draw in decoded["nuisance_jacobian_draws"]
            ),
            csm=decoded["csm"], forward_draw_set_sha=decoded["forward_draw_set_sha"],
            forward_source_binding_sha=decoded["forward_source_binding_sha"],
            forward_source_artifact_shas=tuple(tuple(item) for item in decoded["forward_source_artifact_shas"]),
            finite_difference_step=decoded["finite_difference_step"],
            draw_count=decoded["draw_count"], observation_dimension=decoded["observation_dimension"],
            action_dimension=decoded["action_dimension"], scope=decoded["scope"],
            causal_status=decoded["causal_status"], input_identity_verified=decoded["input_identity_verified"],
            causal_gain_available=decoded["causal_gain_available"],
            hardware_action_authorized=decoded["hardware_action_authorized"],
            scientific_gate_issued=decoded["scientific_gate_issued"],
            research_only=decoded["research_only"],
        )
        if result.canonical_json_bytes() != payload:
            raise ValueError("native sensitivity payload is not canonical after typed parsing")
        return result


def build_native_projective_sensitivity(
    posterior: TrackBNativeEnsemblePosteriorV1,
    point_artifacts: Sequence[Effective_point_inference],
    *,
    finite_difference_step: float = 1.0e-4,
    expected_source_commit: str | None = None,
    expected_source_tree: str | None = None,
) -> TrackBNativeProjectiveSensitivityV1:
    """Compute nominal projective sensitivity and unverified CSM geometry."""
    artifacts = tuple(point_artifacts)
    if posterior.point_artifact_file_shas != tuple(value.file_sha for value in artifacts):
        raise ValueError("sensitivity points do not match the native posterior artifact")
    if posterior.draw_count != len(artifacts):
        raise ValueError("sensitivity point draw count differs from native posterior")
    if expected_source_commit is not None and posterior.source_commit != expected_source_commit:
        raise ValueError("sensitivity reader source commit differs")
    if expected_source_tree is not None and posterior.source_tree != expected_source_tree:
        raise ValueError("sensitivity reader source tree differs")
    reconstructed = build_native_ensemble_posterior(
        artifacts,
        expected_source_commit=posterior.source_commit,
        expected_source_tree=posterior.source_tree,
    )
    if reconstructed.canonical_json_bytes() != posterior.canonical_json_bytes():
        raise ValueError("sensitivity posterior does not reproduce the supplied point artifacts")
    if not math.isfinite(float(finite_difference_step)) or float(finite_difference_step) <= 0.0:
        raise ValueError("sensitivity finite-difference step must be positive and finite")
    payloads = tuple(value.payload() for value in artifacts)
    mode_count = 3 if posterior.system_type == "3qubit" else 4
    state_draws = np.asarray(tuple(_row_draw_vectors(payload)[0].mean(axis=0) for payload in payloads))
    observation_dimension = 2 * mode_count * mode_count
    action_dimension = 2 * (mode_count - 1)
    if state_draws.shape != (len(artifacts), observation_dimension):
        raise ValueError("point state vectors do not match the native system")
    zero = np.zeros(action_dimension, dtype=np.float64)
    jacobians = []
    for state in state_draws:
        baseline = _projective_state(state, zero, mode_count)
        jacobian = np.empty((observation_dimension, action_dimension), dtype=np.float64)
        for coordinate in range(action_dimension):
            plus = zero.copy(); plus[coordinate] = float(finite_difference_step)
            minus = zero.copy(); minus[coordinate] = -float(finite_difference_step)
            jacobian[:, coordinate] = (
                _projective_state(state, plus, mode_count)
                - _projective_state(state, minus, mode_count)
            ) / (2.0 * float(finite_difference_step))
        # The transform is projective, so its zero-action output explicitly
        # renormalizes the serialized point.  This avoids treating a small
        # JSON/FP64 trace round-off as a changed physical baseline.
        if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(jacobian)):
            raise ValueError("nominal projective sensitivity baseline or Jacobian is nonfinite")
        jacobians.append(jacobian)
    covariance = _covariance_from_draws(state_draws)
    covariance_draws = tuple(covariance.copy() for _ in artifacts)
    nuisance_draws = tuple(np.zeros((observation_dimension, 1), dtype=np.float64) for _ in artifacts)
    forward_sha = _forward_identity(artifacts, mode_count)
    covariance_sha = canonical_sha256({
        "schema": "TrackBNativeProjectiveCovarianceV1",
        "covariance": covariance.tolist(),
    })
    operation_descriptor_sha = canonical_sha256({
        "schema": "TrackBNativeOperationDescriptorV1",
        "inference_protocol_common_sha": posterior.inference_protocol_common_sha,
    })
    rf_descriptor_sha = canonical_sha256({
        "schema": "TrackBNativeRFDescriptorV1",
        "inference_protocol_common_sha": posterior.inference_protocol_common_sha,
    })
    csm = RealCSMSupportV1.from_posterior_draws(
        jacobians, covariance_draws, forward_sha=forward_sha,
        operation_descriptor_sha=operation_descriptor_sha,
        rf_descriptor_sha=rf_descriptor_sha,
        nuisance_jacobian_draws=nuisance_draws,
    )
    result = TrackBNativeProjectiveSensitivityV1(
        system_type=posterior.system_type, source_commit=posterior.source_commit,
        source_tree=posterior.source_tree, posterior_sha=posterior.sha256(),
        point_artifact_file_shas=posterior.point_artifact_file_shas,
        training_identity_shas=posterior.training_identity_shas,
        action_coordinate_layout_sha=coordinate_layout_sha(mode_count),
        forward_sha=forward_sha, covariance_sha=covariance_sha,
        jacobian_draws=tuple(tuple(tuple(float(value) for value in row) for row in draw) for draw in jacobians),
        covariance_draws=tuple(tuple(tuple(float(value) for value in row) for row in draw) for draw in covariance_draws),
        nuisance_jacobian_draws=tuple(tuple(tuple(float(value) for value in row) for row in draw) for draw in nuisance_draws),
        csm=csm.to_dict(), finite_difference_step=float(finite_difference_step),
        forward_draw_set_sha=None,
        forward_source_binding_sha=None,
        forward_source_artifact_shas=(),
        draw_count=len(artifacts), observation_dimension=observation_dimension,
        action_dimension=action_dimension,
    )
    result.to_dict()
    return result


def build_bound_native_projective_sensitivity(
    posterior: TrackBNativeEnsemblePosteriorV1,
    point_artifacts: Sequence[Effective_point_inference],
    forward_draw_set: BoundTrackBForwardDrawSetV1,
    *,
    finite_difference_step: float = 1.0e-4,
    expected_source_commit: str | None = None,
    expected_source_tree: str | None = None,
) -> TrackBNativeProjectiveSensitivityV1:
    """Rebind nominal sensitivity to an authoritative same-forward draw set."""
    if type(forward_draw_set) is not BoundTrackBForwardDrawSetV1:
        raise TypeError("bound native sensitivity requires an exact forward draw set")
    forward_draw_set.assert_authoritative()
    forward_draw_set.assert_actionful_source_bound()
    base = build_native_projective_sensitivity(
        posterior, point_artifacts, finite_difference_step=finite_difference_step,
        expected_source_commit=expected_source_commit, expected_source_tree=expected_source_tree,
    )
    if tuple(forward_draw_set.draw_ids) != tuple(posterior.training_identity_shas):
        raise ValueError("nominal sensitivity draw IDs differ from posterior training identities")
    if forward_draw_set.source_artifact_bundle is None \
            or forward_draw_set.forward_source_binding_sha is None:
        raise ValueError("bound native sensitivity requires a file-backed forward source bundle")
    _require_native_projective_draw_parameter_binding(posterior, point_artifacts, forward_draw_set)
    expected_forward_instance = build_native_projective_forward_instance(
        posterior,
        forward_draw_set.source_artifact_bundle,
        finite_difference_step=finite_difference_step,
    )
    if forward_draw_set.forward_instance != expected_forward_instance:
        raise ValueError("sealed forward instance does not match its native companion preimages")
    if forward_draw_set.covariance_builder_sha != expected_forward_instance.covariance_sha:
        raise ValueError("sealed covariance callable does not match its native forward instance")
    if (forward_draw_set.source_artifact_bundle.source_commit,
            forward_draw_set.source_artifact_bundle.source_tree) != (
                posterior.source_commit, posterior.source_tree,
    ):
        raise ValueError("sealed forward source identity differs from posterior source")
    if expected_source_commit is not None or expected_source_tree is not None:
        if expected_source_commit is None or expected_source_tree is None:
            raise ValueError("bound native sensitivity expected source identity must be complete")
        if (forward_draw_set.source_artifact_bundle.source_commit,
                forward_draw_set.source_artifact_bundle.source_tree) != (
                    expected_source_commit, expected_source_tree,
        ):
            raise ValueError("sealed forward source checkout identity differs")
    if any(
        not np.array_equal(left, np.asarray(right, dtype=np.float64))
        for left, right in zip(forward_draw_set.jacobian_draws, base.jacobian_draws, strict=True)
    ):
        raise ValueError("sealed forward Jacobians differ from nominal sensitivity")
    if forward_draw_set.finite_difference_step != float(finite_difference_step):
        raise ValueError("sealed forward finite-difference step differs")
    csm = RealCSMSupportV1.from_posterior_draws(
        forward_draw_set.jacobian_draws,
        forward_draw_set.covariance_draws,
        forward_sha=forward_draw_set.forward_sha,
        operation_descriptor_sha=forward_draw_set.operation_descriptor_sha or "",
        rf_descriptor_sha=forward_draw_set.rf_descriptor_sha or "",
        nuisance_jacobian_draws=forward_draw_set.nuisance_jacobian_draws,
        forward_draw_set=forward_draw_set,
    )
    sealed_jacobians = tuple(
        tuple(tuple(float(value) for value in row) for row in draw)
        for draw in forward_draw_set.jacobian_draws
    )
    sealed_covariances = tuple(
        tuple(tuple(float(value) for value in row) for row in draw)
        for draw in forward_draw_set.covariance_draws
    )
    sealed_nuisances = tuple(
        tuple(tuple(float(value) for value in row) for row in draw)
        for draw in forward_draw_set.nuisance_jacobian_draws
    )
    return replace(
        base,
        forward_sha=forward_draw_set.forward_sha,
        jacobian_draws=sealed_jacobians,
        covariance_draws=sealed_covariances,
        nuisance_jacobian_draws=sealed_nuisances,
        covariance_sha=csm.covariance_sha,
        csm=csm.to_dict(),
        forward_draw_set_sha=forward_draw_set.sha256(),
        forward_source_binding_sha=forward_draw_set.forward_source_binding_sha,
        forward_source_artifact_shas=tuple(
            (artifact.name, artifact.sha256)
            for artifact in forward_draw_set.source_artifact_bundle.artifacts
            if artifact.name in FORWARD_SOURCE_ARTIFACT_NAMES
        ),
        input_identity_verified=True,
    )


def verify_native_projective_sensitivity_companions(
    sensitivity: TrackBNativeProjectiveSensitivityV1 | bytes,
    posterior: TrackBNativeEnsemblePosteriorV1,
    point_artifacts: Sequence[Effective_point_inference],
    forward_draw_set: BoundTrackBForwardDrawSetV1,
    *,
    expected_source_commit: str,
    expected_source_tree: str,
) -> TrackBNativeProjectiveSensitivityV1:
    """Verify a serialized sensitivity against its producer companions.

    ``from_canonical_json_bytes`` is intentionally a structural consumer
    parser.  It cannot access the source files, replay callables, or point
    payloads needed to prove a verified claim.  This companion-aware boundary
    replays those inputs and compares the complete canonical producer result,
    including the forward instance, all draw arrays, nested CSM, and source
    artifact binding.
    """
    if isinstance(sensitivity, bytes):
        parsed = TrackBNativeProjectiveSensitivityV1.from_canonical_json_bytes(sensitivity)
    elif type(sensitivity) is TrackBNativeProjectiveSensitivityV1:
        parsed = sensitivity
    else:
        raise TypeError("sensitivity companion verification requires a typed artifact or bytes")
    if type(posterior) is not TrackBNativeEnsemblePosteriorV1:
        raise TypeError("sensitivity companion verification requires a typed posterior")
    if type(forward_draw_set) is not BoundTrackBForwardDrawSetV1:
        raise TypeError("sensitivity companion verification requires a typed forward draw set")
    require_git_oid(expected_source_commit, "expected_sensitivity_source_commit")
    require_git_oid(expected_source_tree, "expected_sensitivity_source_tree")
    if not parsed.input_identity_verified:
        raise ValueError("companion verification requires an identity-verified sensitivity")
    if (parsed.source_commit, parsed.source_tree) != (expected_source_commit, expected_source_tree):
        raise ValueError("sensitivity source identity differs from the expected companions")
    if (posterior.source_commit, posterior.source_tree) != (expected_source_commit, expected_source_tree):
        raise ValueError("posterior source identity differs from the expected companions")
    if not point_artifacts or any(type(value) is not Effective_point_inference for value in point_artifacts):
        raise TypeError("sensitivity companion verification requires exact point artifacts")
    forward_draw_set.forward_instance.validate()
    forward_draw_set.assert_authoritative()
    forward_draw_set.assert_actionful_source_bound()
    if forward_draw_set.source_artifact_bundle is None or (
            forward_draw_set.source_artifact_bundle.source_commit,
            forward_draw_set.source_artifact_bundle.source_tree,
    ) != (expected_source_commit, expected_source_tree):
        raise ValueError("forward source companion identity differs from the expected source")
    expected = build_bound_native_projective_sensitivity(
        posterior,
        tuple(point_artifacts),
        forward_draw_set,
        finite_difference_step=parsed.finite_difference_step,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
    )
    if parsed.canonical_json_bytes() != expected.canonical_json_bytes():
        raise ValueError("sensitivity does not reproduce its companion-bound producer result")
    return parsed


__all__ = [
    "CAUSAL_STATUS", "NATIVE_SENSITIVITY_SCHEMA", "NATIVE_SENSITIVITY_SCOPE",
    "TrackBNativeProjectiveSensitivityV1", "build_native_projective_sensitivity",
    "build_bound_native_projective_sensitivity", "build_native_projective_forward_instance",
    "native_projective_covariance",
    "native_projective_forward", "verify_native_projective_sensitivity_companions",
]
