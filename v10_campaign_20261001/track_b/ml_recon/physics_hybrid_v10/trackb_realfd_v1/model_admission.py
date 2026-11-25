"""Explicit, read-only model comparison admission for Track B.

This module freezes a reproducible baseline/candidate protocol without
starting optimization, choosing a winner, or granting scientific or hardware
authority.  It binds the two model variants to one source, data admission,
fold, runtime, and stopping contract so a later pilot cannot silently compare
different experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import canonical_json_bytes, canonical_sha256, require_git_oid, require_sha256


PROTOCOL_SCHEMA = "TrackBModelComparisonProtocolV1"
VARIANT_SCHEMA = "TrackBModelVariantSpecV1"
SUPPORTED_VARIANTS = ("legacy_v1", "effective_family_context_v3")
COMPARISON_METRICS = (
    "native_nll",
    "four_cell_mae",
    "global_native_error",
    "multi2_error",
    "multi3_error",
    "multi4_error",
)
RECOMMENDED_SYSTEM_TYPE = "3qubit"
RECOMMENDED_TRAIN_ROW_COUNT = 440
RECOMMENDED_TRAIN_BATCH_COUNT = 110
RECOMMENDED_TRAIN_QUERY_COUNT = 8800
RECOMMENDED_DEVELOPMENT_ROW_COUNTS = (146, 153)
RECOMMENDED_BATCH_SIZE = 4
RECOMMENDED_OPTIMIZER_STEPS = 110
RECOMMENDED_QUERY_OPERATIONS_PER_ROW = 20


@dataclass(frozen=True)
class TrackBModelVariantSpecV1:
    """The immutable ABI and runtime requirements for one model arm."""

    variant_name: str
    model_schema: str
    config_schema: str
    checkpoint_schema: str
    preflight_schema: str
    training_receipt_schema: str
    requires_global_initialization: bool
    schema: str = VARIANT_SCHEMA
    runtime_device: str = "cpu"
    runtime_precision: str = "fp32"
    optimizer_step_limit: int = 2000

    def validate(self) -> None:
        if self.schema != VARIANT_SCHEMA or self.variant_name not in SUPPORTED_VARIANTS:
            raise ValueError("unknown Track B model variant schema or name")
        for field_name in (
            "model_schema", "config_schema", "checkpoint_schema", "preflight_schema",
            "training_receipt_schema",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.runtime_device != "cpu" or self.runtime_precision != "fp32":
            raise ValueError("Track B comparison is restricted to CPU/FP32")
        if type(self.optimizer_step_limit) is not int or not 0 < self.optimizer_step_limit <= 2000:
            raise ValueError("Track B optimizer step limit is outside the registered bound")
        if type(self.requires_global_initialization) is not bool:
            raise TypeError("requires_global_initialization must be a bool")
        expected = _EXPECTED_VARIANT_ABI[self.variant_name]
        actual = (
            self.model_schema, self.config_schema, self.checkpoint_schema,
            self.preflight_schema, self.training_receipt_schema,
            self.requires_global_initialization,
        )
        if actual != expected:
            raise ValueError("variant name is not bound to its registered ABI")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "variant_name": self.variant_name,
            "model_schema": self.model_schema,
            "config_schema": self.config_schema,
            "checkpoint_schema": self.checkpoint_schema,
            "preflight_schema": self.preflight_schema,
            "training_receipt_schema": self.training_receipt_schema,
            "requires_global_initialization": self.requires_global_initialization,
            "runtime_device": self.runtime_device,
            "runtime_precision": self.runtime_precision,
            "optimizer_step_limit": self.optimizer_step_limit,
        }


LEGACY_V1_VARIANT = TrackBModelVariantSpecV1(
    variant_name="legacy_v1",
    model_schema="TrainableMaskedObservationPlateV1",
    config_schema="TrainableMaskedObservationPlateConfigV1",
    checkpoint_schema="TrackBRealTrainingCheckpointV1",
    preflight_schema="TrackBRealPilotPreflightReceiptV1",
    training_receipt_schema="TrackBRealTrainingReceiptV1",
    requires_global_initialization=False,
)

EFFECTIVE_FAMILY_CONTEXT_V3_VARIANT = TrackBModelVariantSpecV1(
    variant_name="effective_family_context_v3",
    model_schema="TrackBEffectiveFamilyContextObservationV3",
    config_schema="TrackBEffectiveFamilyContextConfigV3",
    checkpoint_schema="TrackBEffectiveFamilyContextCheckpointV3",
    preflight_schema="TrackBEffectiveFamilyContextPreflightReceiptV3",
    training_receipt_schema="TrackBEffectiveFamilyContextTrainingReceiptV3",
    requires_global_initialization=True,
)

_EXPECTED_VARIANT_ABI = MappingProxyType({
    "legacy_v1": (
        "TrainableMaskedObservationPlateV1", "TrainableMaskedObservationPlateConfigV1",
        "TrackBRealTrainingCheckpointV1", "TrackBRealPilotPreflightReceiptV1",
        "TrackBRealTrainingReceiptV1", False,
    ),
    "effective_family_context_v3": (
        "TrackBEffectiveFamilyContextObservationV3", "TrackBEffectiveFamilyContextConfigV3",
        "TrackBEffectiveFamilyContextCheckpointV3", "TrackBEffectiveFamilyContextPreflightReceiptV3",
        "TrackBEffectiveFamilyContextTrainingReceiptV3", True,
    ),
})

REGISTERED_MODEL_VARIANTS = MappingProxyType({
    LEGACY_V1_VARIANT.variant_name: LEGACY_V1_VARIANT,
    EFFECTIVE_FAMILY_CONTEXT_V3_VARIANT.variant_name: EFFECTIVE_FAMILY_CONTEXT_V3_VARIANT,
})


@dataclass(frozen=True)
class TrackBModelComparisonProtocolV1:
    """A frozen comparison plan; it is not an execution authorization."""

    source_commit: str
    source_tree: str
    dataset_build_id: str
    admission_manifest_sha: str
    outer_fold_sha: str
    operation_registry_sha: str
    command_table_sha: str
    operation_metadata_sidecar_sha: str
    system_type: str
    train_row_count: int
    train_batch_count: int
    train_query_count: int
    development_row_counts: tuple[int, ...]
    seed: int
    batch_size: int
    optimizer_steps: int
    baseline_variant: str = "legacy_v1"
    candidate_variant: str = "effective_family_context_v3"
    initialization_contract: str = "source_bound_variant_specific_m0"
    selection_policy: str = "operator_review_required_no_automatic_best_model"
    selected_model: None = None
    execution_status: str = "NOT_EXECUTED"
    scientific_gate_issued: bool = False
    hardware_action_authorized: bool = False
    heldout_selection_allowed: bool = False
    schema: str = PROTOCOL_SCHEMA
    runtime_device: str = "cpu"
    runtime_precision: str = "fp32"
    comparison_metrics: tuple[str, ...] = COMPARISON_METRICS

    def validate(self) -> None:
        if self.schema != PROTOCOL_SCHEMA:
            raise ValueError("unknown Track B model comparison protocol schema")
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        for field_name in (
            "admission_manifest_sha", "outer_fold_sha", "operation_registry_sha",
            "command_table_sha", "operation_metadata_sidecar_sha",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if type(self.dataset_build_id) is not str or not self.dataset_build_id:
            raise ValueError("dataset_build_id must be non-empty")
        if self.system_type != RECOMMENDED_SYSTEM_TYPE:
            raise ValueError("the frozen comparison protocol is defined only for the 3qubit profile")
        for field_name in ("train_row_count", "train_batch_count", "train_query_count", "batch_size", "optimizer_steps"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for name in (self.baseline_variant, self.candidate_variant):
            if name not in REGISTERED_MODEL_VARIANTS:
                raise ValueError("comparison contains an unregistered model variant")
        if self.optimizer_steps > min(
            REGISTERED_MODEL_VARIANTS[name].optimizer_step_limit
            for name in (self.baseline_variant, self.candidate_variant)
        ):
            raise ValueError("optimizer_steps exceeds a registered arm limit")
        if type(self.development_row_counts) is not tuple or not self.development_row_counts:
            raise ValueError("development_row_counts must be a non-empty tuple")
        if any(type(value) is not int or value <= 0 for value in self.development_row_counts):
            raise ValueError("development row counts must be positive integers")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.runtime_device != "cpu" or self.runtime_precision != "fp32":
            raise ValueError("Track B comparison is restricted to CPU/FP32")
        if self.baseline_variant == self.candidate_variant:
            raise ValueError("baseline and candidate must be distinct registered variants")
        if (self.baseline_variant, self.candidate_variant) != (
            "legacy_v1", "effective_family_context_v3",
        ):
            raise ValueError("the frozen protocol requires legacy_v1 baseline versus family-context V3 candidate")
        if (
            self.train_row_count != RECOMMENDED_TRAIN_ROW_COUNT
            or self.train_batch_count != RECOMMENDED_TRAIN_BATCH_COUNT
            or self.train_query_count != RECOMMENDED_TRAIN_QUERY_COUNT
            or self.development_row_counts != RECOMMENDED_DEVELOPMENT_ROW_COUNTS
            or self.batch_size != RECOMMENDED_BATCH_SIZE
            or self.optimizer_steps != RECOMMENDED_OPTIMIZER_STEPS
        ):
            raise ValueError("protocol counts and budget differ from the frozen 3qubit profile")
        if self.train_batch_count != self.train_row_count // self.batch_size:
            raise ValueError("train batch count is inconsistent with the frozen fold and batch size")
        if self.train_query_count != self.train_row_count * RECOMMENDED_QUERY_OPERATIONS_PER_ROW:
            raise ValueError("train query count is inconsistent with the frozen row/query contract")
        for name in (self.baseline_variant, self.candidate_variant):
            REGISTERED_MODEL_VARIANTS[name].validate()
        if self.initialization_contract != "source_bound_variant_specific_m0":
            raise ValueError("comparison requires the registered variant-specific source-bound M0 contract")
        if self.selection_policy != "operator_review_required_no_automatic_best_model":
            raise ValueError("automatic model selection is not permitted")
        if self.selected_model is not None:
            raise ValueError("selected_model must remain null until comparison review")
        if self.execution_status != "NOT_EXECUTED":
            raise ValueError("protocol artifacts cannot attest execution")
        for field_name in ("scientific_gate_issued", "hardware_action_authorized", "heldout_selection_allowed"):
            if getattr(self, field_name) is not False:
                raise ValueError(f"{field_name} must remain false in the frozen protocol")
        if type(self.comparison_metrics) is not tuple \
                or any(type(value) is not str for value in self.comparison_metrics):
            raise TypeError("comparison_metrics must be an immutable tuple of strings")
        if self.comparison_metrics != COMPARISON_METRICS:
            raise ValueError("comparison metrics changed from the registered protocol")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "dataset_build_id": self.dataset_build_id,
            "admission_manifest_sha": self.admission_manifest_sha,
            "outer_fold_sha": self.outer_fold_sha,
            "operation_registry_sha": self.operation_registry_sha,
            "command_table_sha": self.command_table_sha,
            "operation_metadata_sidecar_sha": self.operation_metadata_sidecar_sha,
            "system_type": self.system_type,
            "train_row_count": self.train_row_count,
            "train_batch_count": self.train_batch_count,
            "train_query_count": self.train_query_count,
            "development_row_counts": list(self.development_row_counts),
            "seed": self.seed,
            "batch_size": self.batch_size,
            "optimizer_steps": self.optimizer_steps,
            "baseline_variant": self.baseline_variant,
            "candidate_variant": self.candidate_variant,
            "initialization_contract": self.initialization_contract,
            "selection_policy": self.selection_policy,
            "selected_model": self.selected_model,
            "execution_status": self.execution_status,
            "scientific_gate_issued": self.scientific_gate_issued,
            "hardware_action_authorized": self.hardware_action_authorized,
            "heldout_selection_allowed": self.heldout_selection_allowed,
            "runtime_device": self.runtime_device,
            "runtime_precision": self.runtime_precision,
            "comparison_metrics": list(self.comparison_metrics),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def build_recommended_trackb_model_protocol(
    *, source_commit: str, source_tree: str, dataset_build_id: str,
    admission_manifest_sha: str, outer_fold_sha: str, operation_registry_sha: str,
    command_table_sha: str, operation_metadata_sidecar_sha: str,
    system_type: str = RECOMMENDED_SYSTEM_TYPE, seed: int = 20261001,
    batch_size: int = RECOMMENDED_BATCH_SIZE,
    optimizer_steps: int = RECOMMENDED_OPTIMIZER_STEPS,
) -> TrackBModelComparisonProtocolV1:
    """Build the recommended frozen comparison without reading or writing data."""
    if system_type != RECOMMENDED_SYSTEM_TYPE or batch_size != RECOMMENDED_BATCH_SIZE \
            or optimizer_steps != RECOMMENDED_OPTIMIZER_STEPS:
        raise ValueError("the recommended builder accepts only the frozen 3qubit batch-4/110-step profile")
    protocol = TrackBModelComparisonProtocolV1(
        source_commit=source_commit, source_tree=source_tree, dataset_build_id=dataset_build_id,
        admission_manifest_sha=admission_manifest_sha, outer_fold_sha=outer_fold_sha,
        operation_registry_sha=operation_registry_sha, command_table_sha=command_table_sha,
        operation_metadata_sidecar_sha=operation_metadata_sidecar_sha, system_type=system_type,
        train_row_count=RECOMMENDED_TRAIN_ROW_COUNT,
        train_batch_count=RECOMMENDED_TRAIN_BATCH_COUNT,
        train_query_count=RECOMMENDED_TRAIN_QUERY_COUNT,
        development_row_counts=RECOMMENDED_DEVELOPMENT_ROW_COUNTS, seed=seed, batch_size=batch_size,
        optimizer_steps=optimizer_steps,
    )
    protocol.validate()
    return protocol


def load_trackb_model_protocol(payload: Mapping[str, Any]) -> TrackBModelComparisonProtocolV1:
    """Strictly load a protocol mapping; no execution side effect is allowed."""
    expected = set(TrackBModelComparisonProtocolV1.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError("Track B model protocol fields are not exact")
    development_row_counts = payload["development_row_counts"]
    comparison_metrics = payload["comparison_metrics"]
    if type(development_row_counts) is not list \
            or any(type(value) is not int for value in development_row_counts):
        raise ValueError("development_row_counts must be a JSON array of integers")
    if type(comparison_metrics) is not list \
            or any(type(value) is not str for value in comparison_metrics):
        raise ValueError("comparison_metrics must be a JSON array of strings")
    protocol = TrackBModelComparisonProtocolV1(
        source_commit=payload["source_commit"], source_tree=payload["source_tree"],
        dataset_build_id=payload["dataset_build_id"], admission_manifest_sha=payload["admission_manifest_sha"],
        outer_fold_sha=payload["outer_fold_sha"], operation_registry_sha=payload["operation_registry_sha"],
        command_table_sha=payload["command_table_sha"], operation_metadata_sidecar_sha=payload["operation_metadata_sidecar_sha"],
        system_type=payload["system_type"], train_row_count=payload["train_row_count"],
        train_batch_count=payload["train_batch_count"], train_query_count=payload["train_query_count"],
        development_row_counts=tuple(development_row_counts), seed=payload["seed"],
        batch_size=payload["batch_size"], optimizer_steps=payload["optimizer_steps"],
        baseline_variant=payload["baseline_variant"], candidate_variant=payload["candidate_variant"],
        initialization_contract=payload["initialization_contract"], selection_policy=payload["selection_policy"],
        selected_model=payload["selected_model"], execution_status=payload["execution_status"],
        scientific_gate_issued=payload["scientific_gate_issued"],
        hardware_action_authorized=payload["hardware_action_authorized"],
        heldout_selection_allowed=payload["heldout_selection_allowed"], schema=payload["schema"],
        runtime_device=payload["runtime_device"], runtime_precision=payload["runtime_precision"],
        comparison_metrics=tuple(comparison_metrics),
    )
    protocol.validate()
    if protocol.to_dict() != dict(payload):
        raise ValueError("Track B model protocol changed during strict decoding")
    return protocol


__all__ = [
    "COMPARISON_METRICS", "EFFECTIVE_FAMILY_CONTEXT_V3_VARIANT", "LEGACY_V1_VARIANT",
    "PROTOCOL_SCHEMA", "REGISTERED_MODEL_VARIANTS", "SUPPORTED_VARIANTS",
    "TrackBModelComparisonProtocolV1", "TrackBModelVariantSpecV1",
    "build_recommended_trackb_model_protocol", "load_trackb_model_protocol",
]
