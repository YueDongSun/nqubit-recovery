"""Source-bound unequal-concentration point overlay.

This is a deterministic research overlay on an already validated native point
artifact. It does not execute a model, construct an optimizer, write a
checkpoint, estimate posterior uncertainty, or authorize an action. Keeping
the overlay separate from TrackBEffectivePointInferenceV1 prevents a
distribution-scoring coordinate from being mistaken for a new model output.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import torch

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    require_git_oid,
    require_sha256,
)
from .effective_inference_artifact import Effective_point_inference
from .effective_point_values import numerical_array
from .unequal_concentration_main_effects import (
    Unequal_concentration_config,
    branch_count,
    expand_main_effects,
    family_names,
    unequal_nested_four_cell_moments,
)


unequal_point_schema = "TrackBUnequalConcentrationPointOverlayV1"
unequal_point_values_schema = "TrackBUnequalConcentrationPointValuesV1"
unequal_point_scope = "FROZEN_NATIVE_POINT_DISTRIBUTION_OVERLAY_NOT_MODEL_OUTPUT"
maximum_exact_read_integer = 2**53
unavailable_overlay_quantities = {
    "posterior_covariance": {"status": "NOT_ESTIMATED", "value": None},
    "posterior_draws": {"status": "NOT_ESTIMATED", "value": None},
    "causal_action_jacobian": {"status": "NOT_IDENTIFIED_FROM_STATIC_POINT", "value": None},
}
point_overlay_fields = frozenset({
    "schema", "source_commit", "source_tree", "system_type", "model_schema",
    "base_point_artifact_sha", "base_model_artifact_sha",
    "base_inference_protocol_sha", "base_source_commit", "base_source_tree",
    "base_forward_evaluations", "configuration", "configuration_sha", "theta",
    "theta_sha", "rows", "scope", "unavailable", "model_forward_evaluations",
    "optimizer_constructed", "model_parameters_updated", "means_frozen",
    "scientific_gate_issued", "hardware_action_authorized",
})
configuration_fields = frozenset({"schema", "maximum_log_delta"})
point_overlay_row_fields = frozenset({
    "sample_id", "batch_index", "input_sha", "operation_ids",
    "operation_families", "cycle_indices", "operation_mask",
    "visible_operation_mask", "query_mask", "base_values_sha", "values",
})
point_overlay_value_fields = frozenset({
    "schema", "native_read_count", "base_four_cell_concentration",
    "log_delta", "concentration_by_branch", "four_cell_mean",
    "four_cell_covariance",
})


def _finite_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON float")
    return value


def _checked_configuration(value: Any) -> Unequal_concentration_config:
    if type(value) is not dict or set(value) != configuration_fields:
        raise ValueError("unequal concentration configuration fields differ")
    config = Unequal_concentration_config(
        maximum_log_delta=_finite_float(value["maximum_log_delta"], "maximum_log_delta"),
        schema=value["schema"],
    )
    config.validate()
    return config


def _configuration_dict(config: Unequal_concentration_config) -> dict[str, Any]:
    config.validate()
    return {"schema": config.schema, "maximum_log_delta": float(config.maximum_log_delta)}


def _checked_theta(
    value: Any, config: Unequal_concentration_config,
) -> torch.Tensor:
    if (
        type(value) is not list or len(value) != 6
        or any(type(item) is not float or not math.isfinite(item) for item in value)
    ):
        raise ValueError("unequal concentration theta must contain six finite JSON floats")
    theta = torch.tensor(value, dtype=torch.float64)
    expand_main_effects(theta, config=config)
    return theta


def _require_registered_row(row: Mapping[str, Any]) -> tuple[tuple[str, ...], torch.Tensor]:
    if type(row) is not dict or set(row) != point_overlay_row_fields:
        raise ValueError("unequal concentration point row fields differ")
    operations = row["operation_ids"]
    families = row["operation_families"]
    cycles = row["cycle_indices"]
    if (
        type(operations) is not list
        or not operations
        or any(type(value) is not str or not value for value in operations)
        or len(set(operations)) != len(operations)
    ):
        raise ValueError("unequal concentration requires unique operation ids")
    if (
        type(families) is not list
        or len(families) != len(operations)
        or any(type(value) is not str for value in families)
        or set(families) != set(family_names)
    ):
        raise ValueError("unequal concentration requires all four registered families")
    if (
        type(cycles) is not list
        or len(cycles) != len(operations)
        or any(type(value) is not int or value < 0 for value in cycles)
    ):
        raise ValueError("unequal concentration cycle metadata is invalid")
    masks = []
    for name in ("operation_mask", "visible_operation_mask", "query_mask"):
        value = row[name]
        if (
            type(value) is not list
            or len(value) != len(operations)
            or any(type(item) is not bool for item in value)
        ):
            raise ValueError("unequal concentration masks must be native booleans")
        masks.append(value)
    available, visible, query = masks
    if not any(visible) or not any(query) or any(
        q and not a or v != (a and not q)
        for a, v, q in zip(available, visible, query, strict=True)
    ):
        raise ValueError("unequal concentration masks disagree")
    return tuple(families), torch.tensor([available], dtype=torch.bool)


def _base_prediction(
    base_values: Mapping[str, Any], operation_count: int,
) -> dict[str, torch.Tensor]:
    probability_names = (
        "signal_probability", "coin_given_signal_probability",
        "idler_given_no_signal_probability",
    )
    return {
        **{
            name: torch.tensor([
                numerical_array(base_values[name], (operation_count,), name).tolist(),
            ],
                               dtype=torch.float64)
            for name in probability_names
        },
        "four_cell_concentration": torch.tensor(
            [float(base_values["four_cell_concentration"])], dtype=torch.float64,
        ),
    }


def _row_overlay_tensors(
    base_row: Mapping[str, Any], theta: torch.Tensor,
    config: Unequal_concentration_config,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    families, operation_mask = _require_registered_row(base_row)
    base_values = base_row["values"]
    if type(base_values) is not dict:
        raise ValueError("unequal concentration base point values are missing")
    prediction = _base_prediction(base_values, len(families))
    expansion = expand_main_effects(theta, config=config)
    family_indices = torch.tensor(
        [family_names.index(value) for value in families], dtype=torch.long,
    )
    log_delta = expansion["log_delta"].index_select(0, family_indices)
    concentration = prediction["four_cell_concentration"][:, None, None] * log_delta.exp()[None]
    concentration = torch.where(operation_mask[..., None], concentration, torch.zeros_like(concentration))
    return log_delta, concentration, {
        "prediction": prediction,
        "operation_mask": operation_mask,
        "families": families,
    }


def _build_row_values(
    base_row: Mapping[str, Any], theta: torch.Tensor,
    config: Unequal_concentration_config,
) -> dict[str, Any]:
    log_delta, concentration, context = _row_overlay_tensors(base_row, theta, config)
    base_values = base_row["values"]
    base_concentration = float(base_values["four_cell_concentration"])
    # Unavailable operations are serialized as zero, but use the parent
    # concentration for the internal moment call before masking them out.
    mask = context["operation_mask"]
    safe_concentration = torch.where(
        mask[..., None], concentration, torch.full_like(concentration, base_concentration),
    )
    prediction = context["prediction"]
    moments = unequal_nested_four_cell_moments(
        read_trials=torch.tensor([int(base_values["native_read_count"])], dtype=torch.int64),
        signal_probability=prediction["signal_probability"],
        coin_given_signal_probability=prediction["coin_given_signal_probability"],
        idler_given_no_signal_probability=prediction["idler_given_no_signal_probability"],
        concentration_by_branch=safe_concentration,
    )
    zero_mean = torch.zeros_like(moments["four_cell_mean"])
    zero_covariance = torch.zeros_like(moments["four_cell_covariance"])
    mean = torch.where(mask[..., None], moments["four_cell_mean"], zero_mean)
    covariance = torch.where(mask[..., None, None], moments["four_cell_covariance"], zero_covariance)
    return {
        "schema": unequal_point_values_schema,
        "native_read_count": int(base_values["native_read_count"]),
        "base_four_cell_concentration": base_concentration,
        "log_delta": log_delta.tolist(),
        "concentration_by_branch": concentration[0].tolist(),
        "four_cell_mean": mean[0].tolist(),
        "four_cell_covariance": covariance[0].tolist(),
    }


def _build_payload(
    base_payload: Mapping[str, Any], base_artifact_sha: str, *,
    source_commit: str, source_tree: str, theta: torch.Tensor,
    config: Unequal_concentration_config,
) -> dict[str, Any]:
    require_git_oid(source_commit, "overlay_source_commit")
    require_git_oid(source_tree, "overlay_source_tree")
    base_rows = base_payload["rows"]
    rows = []
    for base_row in base_rows:
        overlay_base_row = {
            **{name: base_row[name] for name in (
                "sample_id", "batch_index", "input_sha", "operation_ids",
                "operation_families", "cycle_indices", "operation_mask",
                "visible_operation_mask", "query_mask",
            )},
            "base_values_sha": canonical_sha256(base_row["values"]),
            "values": base_row["values"],
        }
        _require_registered_row(overlay_base_row)
        rows.append({
            name: base_row[name]
            for name in (
                "sample_id", "batch_index", "input_sha", "operation_ids",
                "operation_families", "cycle_indices", "operation_mask",
                "visible_operation_mask", "query_mask",
            )
        } | {
            "base_values_sha": overlay_base_row["base_values_sha"],
            "values": _build_row_values(overlay_base_row, theta, config),
        })
    configuration = _configuration_dict(config)
    theta_values = [float(value) for value in theta.tolist()]
    return {
        "schema": unequal_point_schema,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "system_type": base_payload["system_type"],
        "model_schema": base_payload["model_schema"],
        "base_point_artifact_sha": base_artifact_sha,
        "base_model_artifact_sha": base_payload["model_artifact"]["file_sha"],
        "base_inference_protocol_sha": base_payload["inference_protocol_sha"],
        "base_source_commit": base_payload["source_commit"],
        "base_source_tree": base_payload["source_tree"],
        "base_forward_evaluations": base_payload["forward_evaluations"],
        "configuration": configuration,
        "configuration_sha": canonical_sha256(configuration),
        "theta": theta_values,
        "theta_sha": canonical_sha256(theta_values),
        "rows": rows,
        "scope": unequal_point_scope,
        "unavailable": unavailable_overlay_quantities,
        "model_forward_evaluations": 0,
        "optimizer_constructed": False,
        "model_parameters_updated": False,
        "means_frozen": True,
        "scientific_gate_issued": False,
        "hardware_action_authorized": False,
    }


def _validate_overlay_values(
    values: Any, base_values: Mapping[str, Any] | None, base_row: Mapping[str, Any] | None,
    operation_count: int, theta: torch.Tensor, config: Unequal_concentration_config,
) -> None:
    if (
        type(values) is not dict
        or set(values) != point_overlay_value_fields
        or values["schema"] != unequal_point_values_schema
    ):
        raise ValueError("unequal concentration point value fields or schema differ")
    read = values["native_read_count"]
    if type(read) is not int or read < 0 or read > maximum_exact_read_integer:
        raise ValueError("unequal concentration native READ count is invalid")
    base_concentration = _finite_float(
        values["base_four_cell_concentration"], "base_four_cell_concentration",
    )
    if base_concentration <= 0:
        raise ValueError("unequal concentration parent concentration must be positive")
    log_delta = numerical_array(
        values["log_delta"], (operation_count, branch_count), "log_delta",
    )
    concentration = numerical_array(
        values["concentration_by_branch"], (operation_count, branch_count),
        "concentration_by_branch",
    )
    mean = numerical_array(
        values["four_cell_mean"], (operation_count, 4), "four_cell_mean",
    )
    covariance = numerical_array(
        values["four_cell_covariance"], (operation_count, 4, 4), "four_cell_covariance",
    )
    if (
        not bool(torch.isfinite(log_delta).all())
        or not bool(torch.isfinite(concentration).all())
        or not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(covariance).all())
    ):
        raise ValueError("unequal concentration point values must be finite")
    covariance_scale = covariance.abs().amax(dim=(-1, -2)).clamp_min(1.0)
    support_tolerance = torch.maximum(
        torch.full_like(covariance_scale, 1e-10),
        covariance_scale * (64.0 * torch.finfo(torch.float64).eps),
    )
    covariance_symmetry = (covariance - covariance.transpose(-1, -2)).abs()
    row_sum = covariance.sum(-1).abs()
    column_sum = covariance.sum(-2).abs()
    if (
        bool((concentration < 0).any())
        or bool((mean < -1e-10).any())
        or bool((covariance_symmetry > support_tolerance[:, None, None]).any())
        or bool((row_sum > support_tolerance[:, None]).any())
        or bool((column_sum > support_tolerance[:, None]).any())
    ):
        raise ValueError("unequal concentration point moments are invalid")
    if bool((torch.linalg.eigvalsh(covariance) < -support_tolerance[:, None]).any()):
        raise ValueError("unequal concentration covariance is not positive semidefinite")
    read_tensor = torch.full((operation_count,), float(read), dtype=torch.float64)
    allowed_totals = torch.stack((torch.zeros_like(read_tensor), read_tensor), -1)
    mean_total_tolerance = max(1e-8, float(read) * 64.0 * torch.finfo(torch.float64).eps)
    if bool((torch.abs(mean.sum(-1)[:, None] - allowed_totals).amin(-1) > mean_total_tolerance).any()):
        raise ValueError("unequal concentration point means do not preserve fixed READ totals")
    if base_values is None:
        return
    if (
        read != base_values["native_read_count"]
        or base_concentration != base_values["four_cell_concentration"]
    ):
        raise ValueError("unequal concentration point does not preserve parent scalar values")
    expected_base_row = {
        **{name: base_row[name] for name in (
            "sample_id", "batch_index", "input_sha", "operation_ids",
            "operation_families", "cycle_indices", "operation_mask",
            "visible_operation_mask", "query_mask",
        )},
        "base_values_sha": canonical_sha256(base_values),
        "values": base_values,
    }
    expected = _build_row_values(expected_base_row, theta, config)
    for name, shape in (
        ("log_delta", (operation_count, branch_count)),
        ("concentration_by_branch", (operation_count, branch_count)),
        ("four_cell_mean", (operation_count, 4)),
        ("four_cell_covariance", (operation_count, 4, 4)),
    ):
        actual = numerical_array(values[name], shape, name)
        reference = numerical_array(expected[name], shape, name)
        if not torch.allclose(actual, reference, rtol=1e-12, atol=1e-10):
            raise ValueError(f"unequal concentration {name} does not match the frozen parent mapping")


def validate_unequal_concentration_point_artifact(
    payload: Mapping[str, Any], *, base_payload: Mapping[str, Any] | None = None,
) -> None:
    """Validate structure, and when supplied, exact binding to the parent point."""
    if (
        type(payload) is not dict
        or set(payload) != point_overlay_fields
        or payload["schema"] != unequal_point_schema
        or payload["scope"] != unequal_point_scope
    ):
        raise ValueError("unequal concentration point overlay schema or scope differs")
    require_git_oid(payload["source_commit"], "overlay_source_commit")
    require_git_oid(payload["source_tree"], "overlay_source_tree")
    require_sha256(payload["base_point_artifact_sha"], "base_point_artifact_sha")
    require_sha256(payload["base_model_artifact_sha"], "base_model_artifact_sha")
    require_sha256(payload["base_inference_protocol_sha"], "base_inference_protocol_sha")
    require_git_oid(payload["base_source_commit"], "base_source_commit")
    require_git_oid(payload["base_source_tree"], "base_source_tree")
    config = _checked_configuration(payload["configuration"])
    if canonical_sha256(payload["configuration"]) != payload["configuration_sha"]:
        raise ValueError("unequal concentration configuration identity differs")
    require_sha256(payload["configuration_sha"], "configuration_sha")
    theta = _checked_theta(payload["theta"], config)
    if canonical_sha256(payload["theta"]) != payload["theta_sha"]:
        raise ValueError("unequal concentration theta identity differs")
    require_sha256(payload["theta_sha"], "theta_sha")
    if (
        payload["system_type"] not in ("3qubit", "4qubit")
        or type(payload["model_schema"]) is not str
        or not payload["model_schema"]
    ):
        raise ValueError("unequal concentration model/system identity is invalid")
    if payload["unavailable"] != unavailable_overlay_quantities:
        raise ValueError("unequal concentration unknown quantities must remain explicit")
    for name, expected in (
        ("model_forward_evaluations", 0), ("optimizer_constructed", False),
        ("model_parameters_updated", False), ("means_frozen", True),
        ("scientific_gate_issued", False), ("hardware_action_authorized", False),
    ):
        if payload[name] is not expected:
            raise ValueError(f"unequal concentration authority flag {name} is invalid")
    if type(payload["base_forward_evaluations"]) is not int or payload["base_forward_evaluations"] < 1:
        raise ValueError("unequal concentration parent forward count is invalid")
    rows = payload["rows"]
    if type(rows) is not list or not rows:
        raise ValueError("unequal concentration point overlay requires rows")
    if base_payload is not None:
        if type(base_payload) is not dict or base_payload.get("schema") != "TrackBEffectivePointInferenceV1":
            raise ValueError("unequal concentration parent is not a native point artifact")
        if payload["base_point_artifact_sha"] != canonical_sha256(base_payload):
            raise ValueError("unequal concentration parent point identity differs")
        for name in ("system_type", "model_schema", "base_source_commit", "base_source_tree"):
            base_name = {
                "base_source_commit": "source_commit", "base_source_tree": "source_tree",
            }.get(name, name)
            if payload[name] != base_payload[base_name]:
                raise ValueError("unequal concentration parent model/source identity differs")
        if (
            payload["base_model_artifact_sha"] != base_payload["model_artifact"]["file_sha"]
            or payload["base_inference_protocol_sha"] != base_payload["inference_protocol_sha"]
            or payload["base_forward_evaluations"] != base_payload["forward_evaluations"]
        ):
            raise ValueError("unequal concentration parent execution identity differs")
        if len(rows) != len(base_payload["rows"]):
            raise ValueError("unequal concentration row count differs from parent point")
    for index, row in enumerate(rows):
        families, _ = _require_registered_row(row)
        base_row = None
        base_values = None
        if base_payload is not None:
            base_row = base_payload["rows"][index]
            for name in (
                "sample_id", "batch_index", "input_sha", "operation_ids",
                "operation_families", "cycle_indices", "operation_mask",
                "visible_operation_mask", "query_mask",
            ):
                if row[name] != base_row[name]:
                    raise ValueError("unequal concentration row provenance differs from parent point")
            base_values = base_row["values"]
            if row["base_values_sha"] != canonical_sha256(base_values):
                raise ValueError("unequal concentration base row value identity differs")
        else:
            require_sha256(row["base_values_sha"], "base_values_sha")
        _validate_overlay_values(
            row["values"], base_values, base_row, len(row["operation_ids"]), theta, config,
        )
        active = torch.tensor(row["operation_mask"], dtype=torch.bool)
        concentration = numerical_array(
            row["values"]["concentration_by_branch"],
            (len(row["operation_ids"]), branch_count),
            "concentration_by_branch",
        )
        if bool((concentration[~active] != 0).any()):
            raise ValueError("unavailable unequal concentration operations must remain zero")
        if families != tuple(row["operation_families"]):
            raise ValueError("unequal concentration family metadata changed during validation")


@dataclass(frozen=True)
class Unequal_concentration_point_overlay:
    """Immutable canonical overlay; never an interchangeable checkpoint."""

    payload_bytes: bytes
    file_sha: str

    def payload(self) -> dict[str, Any]:
        require_sha256(self.file_sha, "unequal_point_artifact_file_sha")
        if (
            type(self.payload_bytes) is not bytes
            or hashlib.sha256(self.payload_bytes).hexdigest() != self.file_sha
        ):
            raise ValueError("unequal concentration overlay byte identity differs")
        payload = parse_canonical_json_object(
            self.payload_bytes, schema=unequal_point_schema, exact_fields=point_overlay_fields,
        )
        validate_unequal_concentration_point_artifact(payload)
        return payload


def build_unequal_concentration_point_overlay(
    base_artifact: Effective_point_inference, *,
    source_commit: str, source_tree: str, theta: torch.Tensor,
    config: Unequal_concentration_config = Unequal_concentration_config(),
) -> Unequal_concentration_point_overlay:
    """Derive the overlay from a validated parent point without a model call."""
    if type(base_artifact) is not Effective_point_inference:
        raise TypeError("unequal concentration overlay requires a native point artifact")
    if (
        not isinstance(theta, torch.Tensor)
        or theta.device.type != "cpu"
        or theta.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("unequal concentration theta must be a CPU floating tensor")
    config.validate()
    base_payload = base_artifact.payload()
    require_sha256(base_artifact.file_sha, "base_point_artifact_sha")
    payload = _build_payload(
        base_payload, base_artifact.file_sha, source_commit=source_commit,
        source_tree=source_tree, theta=theta.to(torch.float64), config=config,
    )
    encoded = canonical_json_bytes(payload)
    artifact = Unequal_concentration_point_overlay(encoded, hashlib.sha256(encoded).hexdigest())
    validate_unequal_concentration_point_artifact(payload, base_payload=base_payload)
    return artifact


def load_unequal_concentration_point_overlay_bytes(
    payload: bytes, base_artifact: Effective_point_inference, *,
    expected_file_sha: str, expected_source_commit: str, expected_source_tree: str,
    expected_base_point_artifact_sha: str, expected_base_model_artifact_sha: str,
    expected_base_inference_protocol_sha: str,
) -> Unequal_concentration_point_overlay:
    """Load only when the parent native point and all external IDs agree."""
    if type(base_artifact) is not Effective_point_inference:
        raise TypeError("unequal concentration overlay requires a native point parent")
    require_sha256(expected_file_sha, "expected_unequal_point_artifact_sha")
    require_git_oid(expected_source_commit, "expected_overlay_source_commit")
    require_git_oid(expected_source_tree, "expected_overlay_source_tree")
    for name, value in (
        ("expected_base_point_artifact_sha", expected_base_point_artifact_sha),
        ("expected_base_model_artifact_sha", expected_base_model_artifact_sha),
        ("expected_base_inference_protocol_sha", expected_base_inference_protocol_sha),
    ):
        require_sha256(value, name)
    base_payload = base_artifact.payload()
    artifact = Unequal_concentration_point_overlay(payload, expected_file_sha)
    observed = artifact.payload()
    validate_unequal_concentration_point_artifact(observed, base_payload=base_payload)
    if (
        observed["source_commit"] != expected_source_commit
        or observed["source_tree"] != expected_source_tree
        or observed["base_point_artifact_sha"] != expected_base_point_artifact_sha
        or observed["base_model_artifact_sha"] != expected_base_model_artifact_sha
        or observed["base_inference_protocol_sha"] != expected_base_inference_protocol_sha
    ):
        raise ValueError("unequal concentration overlay differs from external identities")
    if base_artifact.file_sha != expected_base_point_artifact_sha:
        raise ValueError("supplied native point parent differs from expected identity")
    return artifact


def verify_unequal_concentration_point_overlay(
    artifact: Unequal_concentration_point_overlay,
    base_artifact: Effective_point_inference, *,
    theta: torch.Tensor, source_commit: str, source_tree: str,
    config: Unequal_concentration_config = Unequal_concentration_config(),
) -> dict[str, Any]:
    """Independently rederive the pure mapping; no neural forward is repeated."""
    if type(artifact) is not Unequal_concentration_point_overlay:
        raise TypeError("overlay verification requires a typed unequal point artifact")
    original = artifact.payload()
    base_payload = base_artifact.payload()
    validate_unequal_concentration_point_artifact(original, base_payload=base_payload)
    repeated = build_unequal_concentration_point_overlay(
        base_artifact, source_commit=source_commit, source_tree=source_tree,
        theta=theta, config=config,
    ).payload()
    compare_fields = tuple(name for name in point_overlay_fields if name not in {"source_commit", "source_tree"})
    if any(
        canonical_json_bytes(original[name]) != canonical_json_bytes(repeated[name])
        for name in compare_fields
    ):
        raise ValueError("independent unequal concentration mapping does not reproduce the overlay")
    return {
        "schema": "TrackBUnequalConcentrationPointReplayVerificationV1",
        "overlay_file_sha": artifact.file_sha,
        "parent_point_artifact_sha": original["base_point_artifact_sha"],
        "original_source_commit": original["source_commit"],
        "original_source_tree": original["source_tree"],
        "replay_source_commit": source_commit,
        "replay_source_tree": source_tree,
        "row_count": len(original["rows"]),
        "mapping_values_identical": True,
        "model_forward_evaluations": 0,
        "optimizer_constructed": False,
        "scientific_gate_issued": False,
        "hardware_action_authorized": False,
    }
