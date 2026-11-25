"""Typed, no-optimizer inspection and held-out replay helpers for Track B pilots.

This module reads an already-completed bounded pilot.  It never constructs an
optimizer, calls a training runtime, resumes a checkpoint, writes a candidate
artifact, or creates RF/AWG/FPGA/hardware actions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional-ML installation only.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from .contracts import canonical_sha256, parse_canonical_json_object, require_git_oid, require_sha256
from .operation_metadata_sidecar import (
    load_verified_v240_operation_metadata_sidecar_bytes,
    require_detached_sidecar_path,
)
from .observation_plate import TrainableMaskedObservationPlateConfigV1, TrainableMaskedObservationPlateV1
from .training import (
    MAX_SMOKE_OPTIMIZER_STEPS,
    TrackBAtomicMonitorSnapshotV1,
    TrackBRealBatchPlanV1,
    TrackBRealRuntimeIdentityV1,
    TrackBRealTrainingRuntimeV1,
    TrackBRuntimeEnvironmentV1,
    _assert_finite_runtime_state,
    _runtime_state_sha,
    _validated_rng_state,
    trackb_model_state_sha,
)


_CHECKPOINT_FIELDS = frozenset({
    "metadata", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    "scaler_state_dict", "rng_state", "row_visits", "last_loss_components",
    "last_gradient_norm",
})
_CHECKPOINT_METADATA_FIELDS = frozenset({
    "schema", "identity", "identity_sha", "model_config", "model_config_sha",
    "runtime_config_sha", "runtime_environment", "runtime_environment_sha",
    "loss_weights_sha", "model_state_sha", "global_step", "completed",
    "research_only", "selection_evidence", "hardware_action_authorized",
    "long_training_started", "long_authorization_sha",
})
_RUNTIME_ENVIRONMENT_FIELDS = frozenset({
    "schema", "device_type", "torch_version", "cuda_available", "cuda_device_name",
    "cuda_runtime_version", "cuda_driver_version", "cuda_device_index",
    "cuda_device_capability", "cuda_total_memory_bytes", "precision", "amp_enabled",
})
_PILOT_RECEIPT_FIELDS = frozenset({
    "schema", "identity_sha", "runtime_config_sha", "committed_optimizer_steps",
    "unique_rows_seen", "row_visit_min", "row_visit_max", "checkpoint", "monitor_sha256",
    "research_only", "selection_evidence", "hardware_action_authorized",
    "long_training_started", "source_commit", "source_tree", "source_acceptance_sha",
    "admission_manifest_sha", "outer_fold_sha", "pilot_budget_max",
    "preflight_replayed_in_process", "preflight_receipt_sha", "preflight_receipt",
})
_PREFLIGHT_RECEIPT_FIELDS = frozenset({
    "schema", "status", "system_type", "source_commit", "source_tree",
    "source_acceptance_sha", "admission_manifest_sha", "outer_fold_sha",
    "row_identity_sha", "sample_count", "group_count", "projector_dim", "raw_rf_dim",
    "prepared_command_dim", "session_count", "operation_metadata_sidecar_file_sha",
    "operation_metadata_sidecar_semantic_sha", "operation_metadata_origin",
    "authoritative_registry_binding", "all_row_forward_loss_finite",
    "forward_loss_batch_count", "optimizer_started", "hardware_action_authorized",
    "output_root_checked", "resolved_device", "cuda_available", "torch_version",
    "initial_model_state_sha", "exact_resume_checked", "outer_role",
})
_MONITOR_FIELDS = frozenset({
    "schema", "identity_sha", "global_step", "status", "loss_components", "learning_rate",
    "gradient_norm", "unique_rows_seen", "row_visit_min", "row_visit_max",
})


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Track B read-only pilot audit requires PyTorch")


def _require_exact_mapping(value: object, *, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"Track B {name} field set is invalid")
    return value


def _expected_row_visits(identity: TrackBRealRuntimeIdentityV1, *, global_step: int) -> dict[str, int]:
    """Reconstruct committed visits, including partial passes and short tails."""
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("Track B expected terminal row visits require a positive committed optimizer step")
    batch_plan = TrackBRealBatchPlanV1.from_runtime_identity(identity)
    samples = tuple(batch_plan.sample_ids)
    batch_size = batch_plan.batch_size
    batches_per_pass = max(1, (len(samples) + batch_size - 1) // batch_size)
    full_passes, partial_batches = divmod(global_step, batches_per_pass)

    visits = Counter({sample_id: full_passes for sample_id in samples})
    for batch_index in range(partial_batches):
        batch_start = batch_index * batch_size
        batch_end = batch_start + batch_size
        for sample_id in samples[batch_start:batch_end]:
            visits[sample_id] += 1
    return {sample_id: count for sample_id, count in visits.items() if count > 0}


def _runtime_environment_from_dict(payload: Mapping[str, Any]) -> TrackBRuntimeEnvironmentV1:
    _require_exact_mapping(payload, fields=_RUNTIME_ENVIRONMENT_FIELDS, name="runtime environment")
    if payload.get("schema") != "TrackBRuntimeEnvironmentV1":
        raise ValueError("Track B runtime environment schema is invalid")
    capability = payload["cuda_device_capability"]
    if capability is not None and type(capability) is not tuple:
        raise TypeError("Track B runtime environment CUDA capability must remain a tuple")
    environment = TrackBRuntimeEnvironmentV1(
        device_type=payload["device_type"],
        torch_version=payload["torch_version"],
        cuda_available=payload["cuda_available"],
        cuda_device_name=payload["cuda_device_name"],
        cuda_runtime_version=payload["cuda_runtime_version"],
        cuda_driver_version=payload["cuda_driver_version"],
        cuda_device_index=payload["cuda_device_index"],
        cuda_device_capability=capability,
        cuda_total_memory_bytes=payload["cuda_total_memory_bytes"],
        precision=payload["precision"],
        amp_enabled=payload["amp_enabled"],
    )
    if environment.to_dict() != dict(payload):
        raise ValueError("Track B runtime environment changes under typed parsing")
    return environment


def _model_from_cpu_seed(
    config: TrainableMaskedObservationPlateConfigV1, *, seed: int,
) -> TrainableMaskedObservationPlateV1:
    """Recreate the CPU initialization stream without invoking CUDA seed APIs."""
    _require_torch()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.random.default_generator.manual_seed(seed)
        with torch.no_grad():
            return TrainableMaskedObservationPlateV1(config)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu_state)


def _validate_model_state(
    model: TrainableMaskedObservationPlateV1, model_state: Mapping[str, Any], *, expected_sha: str,
) -> None:
    _require_torch()
    expected = model.state_dict()
    if set(model_state) != set(expected) or any(
        not isinstance(model_state[name], torch.Tensor)
        or model_state[name].shape != expected[name].shape
        or model_state[name].dtype != expected[name].dtype
        or (
            (model_state[name].is_floating_point() or model_state[name].is_complex())
            and not bool(torch.all(torch.isfinite(model_state[name])).item())
        )
        for name in expected
    ):
        raise ValueError("Track B read-only checkpoint model state does not match its typed model")
    with torch.no_grad():
        try:
            model.load_state_dict(model_state, strict=True)
        except RuntimeError as error:
            raise ValueError("Track B read-only checkpoint strict model load failed") from error
    if trackb_model_state_sha(model) != expected_sha:
        raise ValueError("Track B read-only checkpoint model-state SHA is invalid")
    model.eval()


def _validate_terminal_runtime_state(
    payload: Mapping[str, Any], *, identity: TrackBRealRuntimeIdentityV1,
    global_step: int,
) -> tuple[
    tuple[tuple[str, int], ...], tuple[tuple[str, float], ...], float,
    str, str, str, str,
]:
    """Validate terminal pilot state structurally without constructing an optimizer."""
    optimizer_state = payload["optimizer_state_dict"]
    scheduler_state = payload["scheduler_state_dict"]
    scaler_state = payload["scaler_state_dict"]
    if not isinstance(optimizer_state, Mapping) or not isinstance(scheduler_state, Mapping) \
            or not isinstance(scaler_state, Mapping):
        raise ValueError("Track B read-only checkpoint terminal runtime state is not mapping-shaped")
    _assert_finite_runtime_state(optimizer_state, name="optimizer_state_dict")
    _assert_finite_runtime_state(scheduler_state, name="scheduler_state_dict")
    _assert_finite_runtime_state(scaler_state, name="scaler_state_dict")
    optimizer_slots = optimizer_state.get("state")
    optimizer_groups = optimizer_state.get("param_groups")
    if not isinstance(optimizer_slots, Mapping) or not optimizer_slots \
            or not isinstance(optimizer_groups, list) or not optimizer_groups:
        raise ValueError("Track B completed pilot optimizer state is incomplete")
    for group in optimizer_groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list) \
                or not group["params"]:
            raise ValueError("Track B completed pilot optimizer parameter groups are incomplete")
        for name in ("lr", "weight_decay"):
            value = group.get(name)
            if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("Track B completed pilot optimizer hyperparameters are invalid")
    if scheduler_state.get("T_max") != global_step \
            or scheduler_state.get("last_epoch") != global_step \
            or scheduler_state.get("_step_count") != global_step + 1 \
            or not isinstance(scheduler_state.get("base_lrs"), list) \
            or not isinstance(scheduler_state.get("_last_lr"), list):
        raise ValueError("Track B completed pilot scheduler terminal state is invalid")
    if scaler_state != {}:
        raise ValueError("Track B completed CPU fp32 pilot must retain the disabled empty scaler state")
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping):
        raise ValueError("Track B read-only checkpoint RNG state is not a mapping")
    _validated_rng_state(rng_state)
    row_visits = payload["row_visits"]
    expected_row_visits = _expected_row_visits(identity, global_step=global_step)
    if not isinstance(row_visits, Mapping) or any(
            type(sample_id) is not str or type(count) is not int or count <= 0
            for sample_id, count in row_visits.items()) or row_visits != expected_row_visits:
        raise ValueError("Track B completed pilot row visits do not match the exact committed batch schedule")
    loss_components = payload["last_loss_components"]
    if not isinstance(loss_components, Mapping) or "loss" not in loss_components \
            or any(type(name) is not str or not math.isfinite(float(value))
                   for name, value in loss_components.items()):
        raise ValueError("Track B completed pilot loss components are invalid")
    gradient_norm = payload["last_gradient_norm"]
    if type(gradient_norm) not in {int, float} or not math.isfinite(float(gradient_norm)) \
            or float(gradient_norm) < 0.0:
        raise ValueError("Track B completed pilot gradient norm is invalid")
    return (
        tuple(sorted((str(sample_id), int(count)) for sample_id, count in row_visits.items())),
        tuple(sorted((str(name), float(value)) for name, value in loss_components.items())),
        float(gradient_norm),
        _runtime_state_sha(optimizer_state),
        _runtime_state_sha(scheduler_state),
        _runtime_state_sha(scaler_state),
        _runtime_state_sha(rng_state),
    )


@dataclass(frozen=True)
class TrackBReadOnlyPilotCheckpointAuditV1:
    """Typed inspection result for a completed CPU bounded-pilot checkpoint."""

    checkpoint_file_sha: str
    identity_sha: str
    source_commit: str
    source_tree: str
    system_type: str
    admission_manifest_sha: str
    outer_fold_sha: str
    model_config_sha: str
    initial_model_state_sha: str
    final_model_state_sha: str
    runtime_config_sha: str
    runtime_environment_sha: str
    optimizer_state_sha: str
    scheduler_state_sha: str
    scaler_state_sha: str
    rng_state_sha: str
    runtime_device_type: str
    runtime_precision: str
    batch_plan_batch_size: int
    global_step: int
    row_visit_total: int
    row_visit_min: int
    row_visit_max: int
    terminal_loss_components_sha: str
    terminal_gradient_norm: float
    runtime_config_preimage_available: bool
    completed: bool
    research_only: bool
    selection_evidence: bool
    hardware_action_authorized: bool
    long_training_started: bool
    long_authorization_sha: str | None
    optimizer_constructed: bool = False
    gradients_enabled_for_reload: bool = False
    schema: str = "TrackBReadOnlyPilotCheckpointAuditV1"

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_file_sha", "identity_sha", "admission_manifest_sha", "outer_fold_sha",
            "model_config_sha", "initial_model_state_sha", "final_model_state_sha",
            "runtime_config_sha", "runtime_environment_sha", "optimizer_state_sha",
            "scheduler_state_sha", "scaler_state_sha", "rng_state_sha",
            "terminal_loss_components_sha",
        ):
            require_sha256(getattr(self, name), name)
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        if self.system_type not in {"3qubit", "4qubit"} or self.runtime_device_type != "cpu" \
                or self.runtime_precision != "fp32":
            raise ValueError("Track B read-only audit requires the completed CPU fp32 pilot identity")
        if type(self.global_step) is not int or not 1 <= self.global_step <= MAX_SMOKE_OPTIMIZER_STEPS:
            raise ValueError("Track B read-only audit step is outside the bounded pilot range")
        if type(self.batch_plan_batch_size) is not int or self.batch_plan_batch_size <= 0 \
                or type(self.row_visit_total) is not int or self.row_visit_total <= 0 \
                or self.row_visit_total < self.global_step \
                or self.row_visit_total > self.global_step * self.batch_plan_batch_size \
                or type(self.row_visit_min) is not int or self.row_visit_min <= 0 \
                or type(self.row_visit_max) is not int or self.row_visit_max <= 0 \
                or self.row_visit_min > self.row_visit_max \
                or not math.isfinite(self.terminal_gradient_norm) \
                or self.terminal_gradient_norm < 0.0 \
                or self.row_visit_max > self.row_visit_total \
                or self.row_visit_max > self.global_step \
                or self.runtime_config_preimage_available:
            raise ValueError("Track B read-only audit terminal runtime evidence is invalid")
        if not self.completed or self.research_only is not True \
                or self.selection_evidence is not False \
                or self.hardware_action_authorized is not False \
                or self.long_training_started is not False \
                or self.long_authorization_sha is not None \
                or self.optimizer_constructed or self.gradients_enabled_for_reload:
            raise ValueError("Track B read-only audit safety status is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "checkpoint_file_sha": self.checkpoint_file_sha,
            "identity_sha": self.identity_sha,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "system_type": self.system_type,
            "admission_manifest_sha": self.admission_manifest_sha,
            "outer_fold_sha": self.outer_fold_sha,
            "model_config_sha": self.model_config_sha,
            "initial_model_state_sha": self.initial_model_state_sha,
            "final_model_state_sha": self.final_model_state_sha,
            "runtime_config_sha": self.runtime_config_sha,
            "runtime_environment_sha": self.runtime_environment_sha,
            "optimizer_state_sha": self.optimizer_state_sha,
            "scheduler_state_sha": self.scheduler_state_sha,
            "scaler_state_sha": self.scaler_state_sha,
            "rng_state_sha": self.rng_state_sha,
            "runtime_device_type": self.runtime_device_type,
            "runtime_precision": self.runtime_precision,
            "batch_plan_batch_size": self.batch_plan_batch_size,
            "global_step": self.global_step,
            "row_visit_total": self.row_visit_total,
            "row_visit_min": self.row_visit_min,
            "row_visit_max": self.row_visit_max,
            "terminal_loss_components_sha": self.terminal_loss_components_sha,
            "terminal_gradient_norm": self.terminal_gradient_norm,
            "runtime_config_preimage_available": self.runtime_config_preimage_available,
            "completed": self.completed,
            "research_only": self.research_only,
            "selection_evidence": self.selection_evidence,
            "hardware_action_authorized": self.hardware_action_authorized,
            "long_training_started": self.long_training_started,
            "long_authorization_sha": self.long_authorization_sha,
            "optimizer_constructed": self.optimizer_constructed,
            "gradients_enabled_for_reload": self.gradients_enabled_for_reload,
        }


@dataclass(frozen=True)
class TrackBReadOnlyPilotReplayV1:
    """A typed final/M0 model pair made without an optimizer or gradient path."""

    audit: TrackBReadOnlyPilotCheckpointAuditV1
    identity: TrackBRealRuntimeIdentityV1
    initial_model: TrainableMaskedObservationPlateV1
    final_model: TrainableMaskedObservationPlateV1
    terminal_row_visits: tuple[tuple[str, int], ...]
    terminal_loss_components: tuple[tuple[str, float], ...]
    terminal_gradient_norm: float


def load_trackb_read_only_pilot_checkpoint_bytes(
    checkpoint_bytes: bytes, *, expected_file_sha: str,
) -> TrackBReadOnlyPilotReplayV1:
    """Rehash and strictly load one snapshotted pilot checkpoint without resuming it."""
    _require_torch()
    require_sha256(expected_file_sha, "expected_file_sha")
    if type(checkpoint_bytes) is not bytes:
        raise TypeError("Track B read-only checkpoint must be supplied as immutable bytes")
    observed_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    if observed_sha != expected_file_sha:
        raise ValueError("Track B read-only checkpoint file SHA mismatch")
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    _require_exact_mapping(payload, fields=_CHECKPOINT_FIELDS, name="checkpoint")
    metadata = _require_exact_mapping(payload["metadata"], fields=_CHECKPOINT_METADATA_FIELDS, name="checkpoint metadata")
    if metadata.get("schema") != TrackBRealTrainingRuntimeV1.CHECKPOINT_SCHEMA:
        raise ValueError("Track B read-only checkpoint schema is invalid")
    try:
        identity = TrackBRealRuntimeIdentityV1.from_dict(metadata["identity"])
        model_config = TrainableMaskedObservationPlateConfigV1.from_dict(metadata["model_config"])
    except (TypeError, ValueError) as error:
        raise ValueError("Track B read-only checkpoint typed identity/config is invalid") from error
    if metadata["identity_sha"] != identity.sha256() \
            or metadata["model_config_sha"] != model_config.sha256() \
            or identity.model_config_sha != model_config.sha256() \
            or identity.system_type != model_config.system_type:
        raise ValueError("Track B read-only checkpoint identity/config binding is invalid")
    for name in ("runtime_config_sha", "runtime_environment_sha", "loss_weights_sha", "model_state_sha"):
        require_sha256(metadata[name], name)
    environment = _runtime_environment_from_dict(metadata["runtime_environment"])
    if environment.sha256() != metadata["runtime_environment_sha"]:
        raise ValueError("Track B read-only checkpoint runtime environment SHA is invalid")
    global_step = metadata["global_step"]
    if type(global_step) is not int or not 1 <= global_step <= MAX_SMOKE_OPTIMIZER_STEPS:
        raise ValueError("Track B read-only checkpoint step is outside the bounded pilot range")
    if metadata["completed"] is not True \
            or metadata["research_only"] is not True \
            or metadata["selection_evidence"] is not False \
            or metadata["hardware_action_authorized"] is not False \
            or metadata["long_training_started"] is not False \
            or metadata["long_authorization_sha"] is not None:
        raise ValueError("Track B read-only checkpoint violates bounded-pilot authority")
    terminal_row_visits, terminal_loss_components, terminal_gradient_norm, \
        optimizer_state_sha, scheduler_state_sha, scaler_state_sha, rng_state_sha = \
        _validate_terminal_runtime_state(payload, identity=identity, global_step=global_step)
    model_state = payload["model_state_dict"]
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("Track B read-only checkpoint model state is invalid")
    initial_model = _model_from_cpu_seed(model_config, seed=identity.seed)
    if trackb_model_state_sha(initial_model) != identity.initial_model_state_sha:
        raise ValueError("Track B read-only M0 reconstruction does not match checkpoint identity")
    initial_model.eval()
    with torch.no_grad():
        final_model = TrainableMaskedObservationPlateV1(model_config)
    _validate_model_state(final_model, model_state, expected_sha=metadata["model_state_sha"])
    audit = TrackBReadOnlyPilotCheckpointAuditV1(
        checkpoint_file_sha=observed_sha,
        identity_sha=identity.sha256(),
        source_commit=identity.source_commit,
        source_tree=identity.source_tree,
        system_type=identity.system_type,
        admission_manifest_sha=identity.admission_manifest_sha,
        outer_fold_sha=identity.outer_fold_sha,
        model_config_sha=model_config.sha256(),
        initial_model_state_sha=identity.initial_model_state_sha,
        final_model_state_sha=metadata["model_state_sha"],
        runtime_config_sha=metadata["runtime_config_sha"],
        runtime_environment_sha=environment.sha256(),
        optimizer_state_sha=optimizer_state_sha,
        scheduler_state_sha=scheduler_state_sha,
        scaler_state_sha=scaler_state_sha,
        rng_state_sha=rng_state_sha,
        runtime_device_type=environment.device_type,
        runtime_precision=environment.precision,
        batch_plan_batch_size=identity.batch_plan_batch_size,
        global_step=global_step,
        row_visit_total=sum(count for _sample_id, count in terminal_row_visits),
        row_visit_min=min(count for _sample_id, count in terminal_row_visits),
        row_visit_max=max(count for _sample_id, count in terminal_row_visits),
        terminal_loss_components_sha=canonical_sha256(dict(terminal_loss_components)),
        terminal_gradient_norm=terminal_gradient_norm,
        runtime_config_preimage_available=False,
        completed=True,
        research_only=True,
        selection_evidence=False,
        hardware_action_authorized=False,
        long_training_started=False,
        long_authorization_sha=None,
    )
    return TrackBReadOnlyPilotReplayV1(
        audit=audit, identity=identity, initial_model=initial_model, final_model=final_model,
        terminal_row_visits=terminal_row_visits,
        terminal_loss_components=terminal_loss_components,
        terminal_gradient_norm=terminal_gradient_norm,
    )


def load_trackb_read_only_pilot_checkpoint(
    checkpoint_path: str | Path, *, expected_file_sha: str,
) -> TrackBReadOnlyPilotReplayV1:
    """Snapshot one checkpoint once, then run the typed no-grad reader on those bytes."""
    source = Path(checkpoint_path)
    if not source.is_file():
        raise ValueError("Track B read-only checkpoint is missing")
    return load_trackb_read_only_pilot_checkpoint_bytes(
        source.read_bytes(), expected_file_sha=expected_file_sha,
    )


def _load_pilot_receipt_bytes(payload: bytes) -> Mapping[str, Any]:
    if type(payload) is not bytes:
        raise TypeError("Track B pilot receipt must be supplied as immutable bytes")
    receipt = parse_canonical_json_object(
        payload, schema="TrackBRealTrainingReceiptV1", exact_fields=_PILOT_RECEIPT_FIELDS,
    )
    checkpoint = _require_exact_mapping(receipt["checkpoint"], fields=frozenset({"path", "sha256", "model_state_sha"}), name="pilot receipt checkpoint")
    for name in ("sha256", "model_state_sha"):
        require_sha256(checkpoint[name], f"pilot receipt checkpoint {name}")
    preflight = _require_exact_mapping(receipt["preflight_receipt"], fields=_PREFLIGHT_RECEIPT_FIELDS, name="pilot preflight receipt")
    if preflight.get("schema") != "TrackBRealPilotPreflightReceiptV1":
        raise ValueError("Track B pilot preflight receipt schema is invalid")
    for name in (
        "source_acceptance_sha", "admission_manifest_sha", "outer_fold_sha", "row_identity_sha",
        "operation_metadata_sidecar_file_sha", "operation_metadata_sidecar_semantic_sha",
        "initial_model_state_sha",
    ):
        require_sha256(preflight[name], f"pilot preflight {name}")
    for name in (
        "identity_sha", "runtime_config_sha", "monitor_sha256", "source_acceptance_sha",
        "admission_manifest_sha", "outer_fold_sha", "preflight_receipt_sha",
    ):
        require_sha256(receipt[name], f"pilot receipt {name}")
    for name in ("source_commit", "source_tree"):
        require_git_oid(receipt[name], f"pilot receipt {name}")
    if canonical_sha256(preflight) != receipt["preflight_receipt_sha"]:
        raise ValueError("Track B pilot receipt preflight SHA is invalid")
    committed_optimizer_steps = receipt["committed_optimizer_steps"]
    if type(committed_optimizer_steps) is not int \
            or not 1 <= committed_optimizer_steps <= MAX_SMOKE_OPTIMIZER_STEPS \
            or receipt["pilot_budget_max"] != MAX_SMOKE_OPTIMIZER_STEPS \
            or receipt["preflight_replayed_in_process"] is not True \
            or receipt["research_only"] is not True \
            or receipt["selection_evidence"] is not False \
            or receipt["hardware_action_authorized"] is not False \
            or receipt["long_training_started"] is not False:
        raise ValueError("Track B pilot receipt is not a completed bounded observation-only pilot")
    if preflight["all_row_forward_loss_finite"] is not True \
            or preflight["optimizer_started"] is not False \
            or preflight["hardware_action_authorized"] is not False \
            or preflight["resolved_device"] != "cpu" \
            or preflight["outer_role"] != "train":
        raise ValueError("Track B pilot receipt preflight safety status is invalid")
    return receipt


def _load_monitor_bytes(payload_bytes: bytes) -> TrackBAtomicMonitorSnapshotV1:
    if type(payload_bytes) is not bytes:
        raise TypeError("Track B pilot monitor must be supplied as immutable bytes")
    payload = parse_canonical_json_object(
        payload_bytes, schema="TrackBAtomicMonitorSnapshotV1", exact_fields=_MONITOR_FIELDS,
    )
    components = payload["loss_components"]
    if not isinstance(components, Mapping):
        raise ValueError("Track B pilot monitor loss components are invalid")
    monitor = TrackBAtomicMonitorSnapshotV1(
        identity_sha=payload["identity_sha"],
        global_step=payload["global_step"],
        status=payload["status"],
        loss_components=tuple(sorted((str(name), float(value)) for name, value in components.items())),
        learning_rate=payload["learning_rate"],
        gradient_norm=payload["gradient_norm"],
        unique_rows_seen=payload["unique_rows_seen"],
        row_visit_min=payload["row_visit_min"],
        row_visit_max=payload["row_visit_max"],
    )
    if monitor.to_dict() != payload:
        raise ValueError("Track B pilot monitor changes under typed parsing")
    return monitor


def _captured_file_status(path: Path, *, expected_bytes: bytes | None = None) -> tuple[str, str | None]:
    if not path.is_file():
        return "NOT_PROVIDED", None
    contents = path.read_bytes()
    if expected_bytes is not None:
        lines = tuple(line for line in contents.splitlines() if line)
        if not lines or lines[-1] != expected_bytes:
            raise ValueError("Track B captured stdout does not end in the canonical pilot receipt")
    return "VERIFIED", hashlib.sha256(contents).hexdigest()


@dataclass(frozen=True)
class TrackBReadOnlyPilotArtifactAuditV1:
    """Rehash result for a completed pilot, preserving any native-capture gaps."""

    checkpoint_audit: TrackBReadOnlyPilotCheckpointAuditV1
    receipt_file_sha: str
    monitor_file_sha: str
    sidecar_file_sha: str
    sidecar_semantic_sha: str
    root_identity_sha: str
    root_ready_sha: str
    root_build_manifest_sha: str
    root_contract_file_sha: str
    root_validation_report_sha: str
    root_index_sha: str
    root_split_sha: str
    root_ready_shard_set_sha: str
    root_selected_index_rows_sha: str
    stdout_capture_status: str
    stdout_capture_sha: str | None
    stderr_capture_status: str
    stderr_capture_sha: str | None
    argv_json_status: str
    admission_manifest_status: str
    status: str
    schema: str = "TrackBReadOnlyPilotArtifactAuditV1"

    def __post_init__(self) -> None:
        for name in (
            "receipt_file_sha", "monitor_file_sha", "sidecar_file_sha", "sidecar_semantic_sha",
            "root_identity_sha",
            "root_ready_sha", "root_build_manifest_sha", "root_contract_file_sha",
            "root_validation_report_sha", "root_index_sha", "root_split_sha",
            "root_ready_shard_set_sha", "root_selected_index_rows_sha",
        ):
            require_sha256(getattr(self, name), name)
        for value in (self.stdout_capture_sha, self.stderr_capture_sha):
            if value is not None:
                require_sha256(value, "capture_sha")
        native_capture_statuses = {
            "MISSING_FROM_ORIGINAL_CANDIDATE_OUTPUT",
            "PRESENT_UNVERIFIED_NOT_USED_FOR_AUTHORITY",
        }
        if self.argv_json_status not in native_capture_statuses \
                or self.admission_manifest_status not in native_capture_statuses:
            raise ValueError("Track B read-only artifact native-capture status is invalid")
        expected_status = (
            "VERIFIED_REHASH_AND_TYPED_RELOAD_WITH_NATIVE_CAPTURE_GAPS"
            if "MISSING_FROM_ORIGINAL_CANDIDATE_OUTPUT" in {
                self.argv_json_status, self.admission_manifest_status,
            }
            else "VERIFIED_REHASH_AND_TYPED_RELOAD_WITH_UNBOUND_NATIVE_CAPTURE"
        )
        if self.status != expected_status:
            raise ValueError("Track B read-only artifact audit status is inconsistent with native capture")
        if self.stdout_capture_status not in {"NOT_PROVIDED", "VERIFIED"} \
                or self.stderr_capture_status not in {"NOT_PROVIDED", "VERIFIED_EMPTY"}:
            raise ValueError("Track B read-only artifact capture status is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "checkpoint_audit": self.checkpoint_audit.to_dict(),
            "receipt_file_sha": self.receipt_file_sha,
            "monitor_file_sha": self.monitor_file_sha,
            "sidecar_file_sha": self.sidecar_file_sha,
            "sidecar_semantic_sha": self.sidecar_semantic_sha,
            "root_identity_sha": self.root_identity_sha,
            "immutable_root_artifacts": {
                "ready_sha": self.root_ready_sha,
                "build_manifest_sha": self.root_build_manifest_sha,
                "contract_file_sha": self.root_contract_file_sha,
                "validation_report_sha": self.root_validation_report_sha,
                "index_sha": self.root_index_sha,
                "split_sha": self.root_split_sha,
                "ready_shard_set_sha": self.root_ready_shard_set_sha,
                "selected_index_rows_sha": self.root_selected_index_rows_sha,
            },
            "stdout_capture_status": self.stdout_capture_status,
            "stdout_capture_sha": self.stdout_capture_sha,
            "stderr_capture_status": self.stderr_capture_status,
            "stderr_capture_sha": self.stderr_capture_sha,
            "argv_json_status": self.argv_json_status,
            "admission_manifest_status": self.admission_manifest_status,
            "status": self.status,
        }


def audit_trackb_bounded_pilot_artifacts(
    *, checkpoint_path: str | Path, receipt_path: str | Path, monitor_path: str | Path,
    sidecar_path: str | Path, dataset_root: str | Path, system_type: str,
    expected_checkpoint_sha: str, expected_receipt_sha: str, expected_monitor_sha: str,
    expected_sidecar_sha: str,
    stdout_path: str | Path | None = None, stderr_path: str | Path | None = None,
) -> tuple[TrackBReadOnlyPilotArtifactAuditV1, TrackBReadOnlyPilotReplayV1]:
    """Cross-bind immutable candidate byte snapshots without changing their output root."""
    for value, name in (
        (expected_checkpoint_sha, "expected_checkpoint_sha"),
        (expected_receipt_sha, "expected_receipt_sha"),
        (expected_monitor_sha, "expected_monitor_sha"),
        (expected_sidecar_sha, "expected_sidecar_sha"),
    ):
        require_sha256(value, name)
    receipt_source = Path(receipt_path)
    monitor_source = Path(monitor_path)
    checkpoint_source = Path(checkpoint_path)
    if not receipt_source.is_file() or not monitor_source.is_file() or not checkpoint_source.is_file():
        raise ValueError("Track B bounded-pilot artifact audit requires checkpoint, receipt, and monitor files")
    checkpoint_bytes = checkpoint_source.read_bytes()
    receipt_bytes = receipt_source.read_bytes()
    monitor_bytes = monitor_source.read_bytes()
    if hashlib.sha256(checkpoint_bytes).hexdigest() != expected_checkpoint_sha \
            or hashlib.sha256(receipt_bytes).hexdigest() != expected_receipt_sha \
            or hashlib.sha256(monitor_bytes).hexdigest() != expected_monitor_sha:
        raise ValueError("Track B audit input artifact SHA does not match its external expectation")
    receipt = _load_pilot_receipt_bytes(receipt_bytes)
    receipt_checkpoint = receipt["checkpoint"]
    if Path(receipt_checkpoint["path"]).resolve() != checkpoint_source.resolve():
        raise ValueError("Track B pilot receipt checkpoint path does not match the audited checkpoint")
    if receipt_checkpoint["sha256"] != expected_checkpoint_sha \
            or receipt["monitor_sha256"] != expected_monitor_sha:
        raise ValueError("Track B external artifact expectations disagree with the pilot receipt")
    replay = load_trackb_read_only_pilot_checkpoint_bytes(
        checkpoint_bytes, expected_file_sha=expected_checkpoint_sha,
    )
    audit = replay.audit
    preflight = receipt["preflight_receipt"]
    if audit.identity_sha != receipt["identity_sha"] \
            or audit.final_model_state_sha != receipt_checkpoint["model_state_sha"] \
            or audit.runtime_config_sha != receipt["runtime_config_sha"] \
            or audit.global_step != receipt["committed_optimizer_steps"] \
            or audit.source_commit != receipt["source_commit"] \
            or audit.source_tree != receipt["source_tree"] \
            or audit.admission_manifest_sha != receipt["admission_manifest_sha"] \
            or audit.outer_fold_sha != receipt["outer_fold_sha"] \
            or audit.initial_model_state_sha != preflight["initial_model_state_sha"]:
        raise ValueError("Track B pilot checkpoint does not match its completion receipt")
    if replay.identity.operation_metadata_sidecar_file_sha != preflight["operation_metadata_sidecar_file_sha"] \
            or replay.identity.operation_metadata_sidecar_semantic_sha != preflight["operation_metadata_sidecar_semantic_sha"]:
        raise ValueError("Track B pilot identity does not bind its preflight sidecar")
    monitor = _load_monitor_bytes(monitor_bytes)
    if monitor.identity_sha != audit.identity_sha or monitor.status != "COMPLETED" \
            or monitor.global_step != audit.global_step \
            or monitor.unique_rows_seen != receipt["unique_rows_seen"] \
            or monitor.row_visit_min != receipt["row_visit_min"] \
            or monitor.row_visit_max != receipt["row_visit_max"] \
            or dict(replay.terminal_loss_components) != dict(monitor.loss_components) \
            or replay.terminal_gradient_norm != monitor.gradient_norm:
        raise ValueError("Track B pilot monitor does not match its completed receipt")
    if len(replay.terminal_row_visits) != receipt["unique_rows_seen"] \
            or min(count for _sample_id, count in replay.terminal_row_visits) != receipt["row_visit_min"] \
            or max(count for _sample_id, count in replay.terminal_row_visits) != receipt["row_visit_max"]:
        raise ValueError("Track B terminal checkpoint row visits do not match its receipt")
    sidecar_source = require_detached_sidecar_path(sidecar_path, dataset_root)
    sidecar_bytes = sidecar_source.read_bytes()
    verified_sidecar = load_verified_v240_operation_metadata_sidecar_bytes(
        sidecar_bytes,
        expected_file_sha=expected_sidecar_sha,
        dataset_root=dataset_root,
        system_type=system_type,
    )
    if expected_sidecar_sha != preflight["operation_metadata_sidecar_file_sha"] \
            or verified_sidecar.sidecar_semantic_sha != preflight["operation_metadata_sidecar_semantic_sha"]:
        raise ValueError("Track B audited sidecar semantic SHA does not match its preflight receipt")
    stdout_status, stdout_sha = (
        _captured_file_status(Path(stdout_path), expected_bytes=receipt_bytes)
        if stdout_path is not None else ("NOT_PROVIDED", None)
    )
    if stderr_path is None:
        stderr_status, stderr_sha = "NOT_PROVIDED", None
    else:
        stderr_contents = Path(stderr_path).read_bytes()
        if stderr_contents:
            raise ValueError("Track B bounded pilot stderr must be empty for this candidate audit")
        stderr_status, stderr_sha = "VERIFIED_EMPTY", hashlib.sha256(stderr_contents).hexdigest()
    output_root = checkpoint_source.parent
    argv_status = (
        "PRESENT_UNVERIFIED_NOT_USED_FOR_AUTHORITY" if (output_root / "trackb_real_pilot_argv.json").is_file()
        else "MISSING_FROM_ORIGINAL_CANDIDATE_OUTPUT"
    )
    manifest_status = (
        "PRESENT_UNVERIFIED_NOT_USED_FOR_AUTHORITY" if (output_root / "trackb_real_pilot_admission_manifest.json").is_file()
        else "MISSING_FROM_ORIGINAL_CANDIDATE_OUTPUT"
    )
    audit_status = (
        "VERIFIED_REHASH_AND_TYPED_RELOAD_WITH_NATIVE_CAPTURE_GAPS"
        if "MISSING_FROM_ORIGINAL_CANDIDATE_OUTPUT" in {argv_status, manifest_status}
        else "VERIFIED_REHASH_AND_TYPED_RELOAD_WITH_UNBOUND_NATIVE_CAPTURE"
    )
    return TrackBReadOnlyPilotArtifactAuditV1(
        checkpoint_audit=audit,
        receipt_file_sha=expected_receipt_sha,
        monitor_file_sha=expected_monitor_sha,
        sidecar_file_sha=verified_sidecar.sidecar_file_sha,
        sidecar_semantic_sha=verified_sidecar.sidecar_semantic_sha,
        root_identity_sha=verified_sidecar.root_identity_sha,
        root_ready_sha=dict(verified_sidecar.root_artifact_shas)["ready_sha"],
        root_build_manifest_sha=dict(verified_sidecar.root_artifact_shas)["build_manifest_sha"],
        root_contract_file_sha=dict(verified_sidecar.root_artifact_shas)["contract_file_sha"],
        root_validation_report_sha=dict(verified_sidecar.root_artifact_shas)["validation_report_sha"],
        root_index_sha=dict(verified_sidecar.root_artifact_shas)["index_sha"],
        root_split_sha=dict(verified_sidecar.root_artifact_shas)["split_sha"],
        root_ready_shard_set_sha=dict(verified_sidecar.root_artifact_shas)["ready_shard_set_sha"],
        root_selected_index_rows_sha=dict(verified_sidecar.root_artifact_shas)["selected_index_rows_sha"],
        stdout_capture_status=stdout_status,
        stdout_capture_sha=stdout_sha,
        stderr_capture_status=stderr_status,
        stderr_capture_sha=stderr_sha,
        argv_json_status=argv_status,
        admission_manifest_status=manifest_status,
        status=audit_status,
    ), replay


__all__ = [
    "TORCH_AVAILABLE", "TrackBReadOnlyPilotArtifactAuditV1", "TrackBReadOnlyPilotCheckpointAuditV1",
    "TrackBReadOnlyPilotReplayV1", "audit_trackb_bounded_pilot_artifacts",
    "load_trackb_read_only_pilot_checkpoint", "load_trackb_read_only_pilot_checkpoint_bytes",
]
