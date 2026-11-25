"""Executable, resumable and reproducible V10 PyTorch training runtime."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml_recon.data_v2_4.version import DATASET_CONTRACT_VERSION
from ml_recon.data_v2_4.hashing import read_canonical_json, sha256_file, sha256_json

from ml_recon.physics_hybrid_v10.checkpoint_schema import validate_checkpoint_envelope
from ml_recon.physics_hybrid_v10.control_coordinate_layout import INTERLEAVED_CONTROL_LAYOUT_SHA
from ml_recon.physics_hybrid_v10.contracts import V10_MODEL_IDS
from ml_recon.physics_hybrid_v10.gradient_audit import V10GradientAuditObserver
from ml_recon.physics_hybrid_v10.sample_scaling import SAMPLE_RUNGS, validate_rung_authorization
from ml_recon.physics_hybrid_v10.training_publication import publish_training_completion
from ml_recon.physics_hybrid_v10.training_telemetry import (
    TELEMETRY_INTERVAL_SECONDS,
    TELEMETRY_INTERVAL_STEPS,
    Training_telemetry,
    archive_training_summary,
    atomic_training_json,
    training_step_record,
)


RUN_MODES = ("smoke", "pilot", "full")
PHYSICAL_TRAINING_BRANCH = "codex/v10_tracka_physical_budget_v2"
TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH = "track_a_operation_poisson_research"
RESUME_STATE_SCHEMA = "V10TrainingResumeStateV1"
PHYSICAL_MODE_BINDING_SCHEMA = "V10PhysicalModeBindingV1"
_PHYSICAL_MODE_ALLOWLIST = {
    "legacy_v1": ("legacy", "V10PhysicalAodDatasetRecipeV3"),
    "operation_poisson_v1": (
        "diagnostic",
        "V10PhysicalOperationPoissonDatasetRecipeV4",
    ),
    "operation_poisson_v1:coupled": (
        "coupled",
        "V10PhysicalOperationPoissonDatasetRecipeV4",
    ),
}
TRAINING_OUTPUT_NAMES = (
    "training_summary.json",
    "checkpoint_latest.pt",
    "metrics.jsonl",
    "training_steps.jsonl",
    "gradient_audit.jsonl",
    "gradient_audit_summary.json",
    "training_progress.json",
    "checkpoint_manifest.json",
    "config_snapshot.json",
    "checkpoint_validation.json",
    "training_completion.json",
)


def _physical_mode_contract(operation_input_recipe: str, excitation_mode: str) -> str:
    """Return the sole dataset schema permitted for a physical mode pair."""
    key = (
        operation_input_recipe
        if excitation_mode != "coupled"
        else f"{operation_input_recipe}:coupled"
    )
    allowed = _PHYSICAL_MODE_ALLOWLIST.get(key)
    if allowed is None or allowed[0] != excitation_mode:
        raise ValueError(
            "physical operation recipe and excitation mode are not an admitted pair"
        )
    return allowed[1]


def _physical_output_manifest_sha(excitation_mode: str) -> str:
    """Read the canonical mode-specific output consumer manifest identity."""
    from ml_recon.physics_hybrid_v10.published_output_registry import physical_output_training_manifest_sha

    return physical_output_training_manifest_sha(excitation_mode)


def _physical_mode_binding(identity: "V10TrainingIdentity") -> dict[str, str]:
    """Serialize the non-interchangeable physical recipe/model/output contract."""
    output_manifest_sha = (
        identity.output_training_manifest_sha
        or _physical_output_manifest_sha(identity.excitation_mode)
    )
    return {
        "schema": PHYSICAL_MODE_BINDING_SCHEMA,
        "operation_input_recipe": identity.operation_input_recipe,
        "excitation_mode": identity.excitation_mode,
        "output_training_manifest_sha": output_manifest_sha,
    }


def resolve_range_mode(
    train_legacy_path: str,
    validation_legacy_path: str,
    train_cell_paths: Dict[str, str] | None,
    validation_cell_paths: Dict[str, str] | None,
) -> bool:
    """Return ``True`` when either legacy or per-cell range indexes are present.

    A per-cell ``cell_id=path`` CLI mapping is the canonical multi-cell range
    entry-point; the legacy single shared path is retained for backward
    compatibility.  Both sides (train *and* validation) must agree.
    """

    has_train = bool(train_legacy_path) or bool(train_cell_paths)
    has_validation = bool(validation_legacy_path) or bool(validation_cell_paths)
    if has_train != has_validation:
        raise ValueError("range mode requires both train and validation range indexes")
    return has_train


@dataclass(frozen=True)
class V10TrainingConfig:
    model_id: str
    output_dir: str
    run_mode: str = "smoke"
    global_seed: int = 20261001
    seed_panel: tuple[int, ...] = (20261001,)
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    amp_enabled: bool = True
    amp_initial_scale: float = 65_536.0
    scheduler_kind: str = "cosine"
    early_stopping_patience: int = 10
    metric_name: str = "validation_loss"
    metric_direction: str = "min"
    num_workers: int = 0
    max_train_batches: Optional[int] = 2
    pause_after_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = 1
    resume_checkpoint: Optional[str] = None
    training_authorized: bool = False
    phase_1_to_7_gate_sha: str = ""
    phase_8_gate_sha: str = ""
    sample_count: int = 0
    global_sample_count: int = 0
    rung_plan_sha: str = ""
    completed_sample_rungs: tuple[int, ...] = ()
    track_id: str = "unspecified"
    run_id: str = ""
    telemetry_enabled: bool = True
    telemetry_interval_steps: int = TELEMETRY_INTERVAL_STEPS
    telemetry_interval_seconds: float = TELEMETRY_INTERVAL_SECONDS

    def validate(self) -> None:
        if self.model_id not in V10_MODEL_IDS:
            raise ValueError("training config must use a formal V10 model id")
        if self.run_mode not in RUN_MODES:
            raise ValueError(f"run_mode must be one of {RUN_MODES}")
        if self.batch_size <= 0 or self.epochs <= 0 or self.learning_rate <= 0.0:
            raise ValueError("batch size, epochs, and learning rate must be positive")
        if self.amp_initial_scale <= 0.0:
            raise ValueError("AMP initial scale must be positive")
        if (
            type(self.telemetry_enabled) is not bool
            or type(self.telemetry_interval_steps) is not int
            or self.telemetry_interval_steps < 1
            or isinstance(self.telemetry_interval_seconds, bool)
            or not isinstance(self.telemetry_interval_seconds, (int, float))
            or not math.isfinite(self.telemetry_interval_seconds)
            or self.telemetry_interval_seconds <= 0
        ):
            raise ValueError("telemetry cadence must be finite and positive")
        if self.track_id not in ("track_a", "track_b", "unspecified"):
            raise ValueError("track_id must be track_a, track_b or unspecified")
        if not isinstance(self.run_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.:-]{0,128}", self.run_id
        ):
            raise ValueError("run_id must be a bounded opaque identifier")
        if self.metric_direction not in ("min", "max"):
            raise ValueError("metric_direction must be min or max")
        if self.run_mode == "smoke" and (self.max_train_batches or 0) < 2:
            raise ValueError("V10 smoke requires at least two training batches")
        if self.pause_after_train_batches is not None and (
            type(self.pause_after_train_batches) is not int
            or self.pause_after_train_batches < 1
            or (self.max_train_batches is not None and self.pause_after_train_batches > self.max_train_batches)
        ):
            raise ValueError("pause_after_train_batches must be within the train batch limit")
        if self.pause_after_train_batches is not None and self.run_mode != "pilot":
            raise ValueError("mid-epoch pause is reserved for admitted pilot runs")
        if self.sample_count in SAMPLE_RUNGS:
            validate_rung_authorization(
                self.completed_sample_rungs,
                self.sample_count,
                training_authorized=self.training_authorized,
                phase_1_to_7_gate_sha=self.phase_1_to_7_gate_sha,
                phase_8_gate_sha=self.phase_8_gate_sha,
            )
        if self.global_sample_count:
            if self.global_sample_count not in (1_000, *SAMPLE_RUNGS):
                raise ValueError("global sample count is not registered")
            if len(self.rung_plan_sha) != 64:
                raise ValueError("global sample count requires a rung plan SHA-256")
            if self.global_sample_count in SAMPLE_RUNGS:
                validate_rung_authorization(
                    self.completed_sample_rungs,
                    self.global_sample_count,
                    training_authorized=self.training_authorized,
                    phase_1_to_7_gate_sha=self.phase_1_to_7_gate_sha,
                    phase_8_gate_sha=self.phase_8_gate_sha,
                )
        effective_rung = self.global_sample_count or self.sample_count
        if self.run_mode == "full" and effective_rung != SAMPLE_RUNGS[-1]:
            raise ValueError(
                "full training is reserved for the authorized 2M sample rung"
            )
        if effective_rung == SAMPLE_RUNGS[-1] and self.run_mode != "full":
            raise ValueError("the 2M sample rung requires run_mode=full")


@dataclass(frozen=True)
class V10TrainingIdentity:
    branch: str
    source_commit: str
    source_tree_sha: str
    dirty_patch_sha: str
    v23_manifest_sha: str
    synthetic_manifest_sha: str
    measurement_contract_sha: str
    shot_budget_contract_sha: str
    rho8_sha: str
    config_sha: str
    dataset_hashes: Dict[str, str]
    split_hash: str
    dataset_contract_version: str = DATASET_CONTRACT_VERSION
    dataset_contract_sha: str = ""
    dataset_build_id: str = ""
    dataset_index_sha: str = ""
    dataset_cell_id: str = ""
    input_delivery: str = "materialized"
    dataset_index_kind: str = "materialized"
    stream_recipe_sha: str = ""
    stream_recipe_file_sha: str = ""
    stream_pilot_admission_sha: str = ""
    physical_namespace: str = ""
    physical_admission_sha: str = ""
    strict_stream_only_identity: bool = False
    coordinate_layout_sha: str = ""
    physical_identity: Dict[str, str] = field(default_factory=dict)
    operation_input_recipe: str = "legacy_v1"
    excitation_mode: str = "legacy"
    output_training_manifest_sha: str = ""

    def validate(self) -> None:
        from ml_recon.physics_hybrid_v10.checkpoint_schema import validate_input_delivery

        validate_input_delivery(asdict(self))
        is_physical = self.branch == PHYSICAL_TRAINING_BRANCH
        is_operation_poisson_research = (
            self.branch == TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH
        )
        if not (is_physical or is_operation_poisson_research) and any(
            (
                self.physical_namespace,
                self.physical_admission_sha,
                self.strict_stream_only_identity,
                self.coordinate_layout_sha,
                self.physical_identity,
            )
        ):
            raise ValueError(
                "physical identity fields are reserved for the physical branch"
            )
        if self.branch not in (
            "v10",
            "codex/v10_lowdisk_training_prep",
            "codex/v10_real_data_recovery_gate",
            PHYSICAL_TRAINING_BRANCH,
            TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH,
        ):
            raise ValueError(
                "formal training identity must bind v10 or the authorized V10 low-disk preparation branch"
            )
        digests = (
            self.source_commit,
            self.source_tree_sha,
            self.dirty_patch_sha,
            self.v23_manifest_sha,
            self.synthetic_manifest_sha,
            self.measurement_contract_sha,
            self.shot_budget_contract_sha,
            self.rho8_sha,
            self.config_sha,
            self.split_hash,
            *self.dataset_hashes.values(),
        )
        if any(len(value) != 40 and len(value) != 64 for value in digests):
            raise ValueError(
                "training identity hashes must be Git SHA-1 or SHA-256 digests"
            )
        # The generic runtime remains usable by isolated unit-test models. Formal
        # V10 entrypoints always populate all five V2.4 fields below.
        v24_values = (
            self.dataset_contract_sha,
            self.dataset_build_id,
            self.dataset_index_sha,
            self.dataset_cell_id,
        )
        if (
            not (is_physical or is_operation_poisson_research)
            and any(v24_values)
            and (
                self.dataset_contract_version != DATASET_CONTRACT_VERSION
                or len(self.dataset_contract_sha) != 64
                or len(self.dataset_index_sha) != 64
                or not self.dataset_build_id
                or not self.dataset_cell_id
            )
        ):
            raise ValueError("partial V2.4 training identity is forbidden")
        if is_physical:
            from ml_recon.physics_hybrid_v10.training.physical_admission import PHYSICAL_NAMESPACE
            from ml_recon.v10_recoverability_foundry.training.physical_gates import (
                validate_physical_identity,
            )

            if (
                self.input_delivery != "stream_only"
                or self.dataset_index_kind != "virtual_stream"
            ):
                raise ValueError(
                    "physical training requires strict stream-only identity"
                )
            if (
                self.physical_namespace != PHYSICAL_NAMESPACE
                or self.strict_stream_only_identity is not True
            ):
                raise ValueError(
                    "physical training namespace and strict stream identity are required"
                )
            if len(self.physical_admission_sha) != 64 or any(
                ch not in "0123456789abcdef" for ch in self.physical_admission_sha
            ):
                raise ValueError("physical training requires a canonical admission SHA")
            if self.coordinate_layout_sha != INTERLEAVED_CONTROL_LAYOUT_SHA:
                raise ValueError(
                    "physical training requires the canonical interleaved coordinate layout SHA"
                )
            validate_physical_identity(self.physical_identity)
            expected_contract = _physical_mode_contract(
                self.operation_input_recipe, self.excitation_mode
            )
            if (
                self.dataset_contract_version != expected_contract
                or len(self.dataset_contract_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.dataset_contract_sha
                )
                or self.dataset_index_sha != self.stream_recipe_sha
                or not self.dataset_build_id.startswith("physical_stream_")
                or not self.dataset_cell_id
            ):
                raise ValueError(
                    "physical training requires a complete independent physical data contract"
                )
            expected_manifest_sha = _physical_output_manifest_sha(
                self.excitation_mode
            )
            if self.output_training_manifest_sha not in (
                "",
                expected_manifest_sha,
            ):
                raise ValueError(
                    "physical training output manifest SHA disagrees with its excitation mode"
                )
            if (
                self.excitation_mode != "legacy"
                and self.output_training_manifest_sha != expected_manifest_sha
            ):
                raise ValueError(
                    "nonlegacy physical training requires an explicit output manifest SHA"
                )
        if is_operation_poisson_research:
            if (
                self.input_delivery != "materialized"
                or self.dataset_index_kind != "materialized"
                or self.stream_recipe_sha
                or self.stream_recipe_file_sha
                or self.stream_pilot_admission_sha
                or self.physical_namespace
                or self.physical_admission_sha
                or self.strict_stream_only_identity
                or self.coordinate_layout_sha
                or self.physical_identity
            ):
                raise ValueError(
                    "operation-Poisson research requires materialized membership without physical admission identity"
                )
            expected_contract = _physical_mode_contract(
                self.operation_input_recipe, self.excitation_mode
            )
            if (
                self.operation_input_recipe != "operation_poisson_v1"
                or self.excitation_mode not in ("diagnostic", "coupled")
                or self.dataset_contract_version != expected_contract
                or len(self.dataset_contract_sha) != 64
                or any(character not in "0123456789abcdef" for character in self.dataset_contract_sha)
                or not self.dataset_build_id
                or not self.dataset_cell_id
                or not {"train", "validation"}.issubset(self.dataset_hashes)
            ):
                raise ValueError(
                    "operation-Poisson research requires a complete V4 materialized data identity"
                )
            if (
                self.output_training_manifest_sha
                != _physical_output_manifest_sha(self.excitation_mode)
            ):
                raise ValueError(
                    "operation-Poisson research requires its exact mode output manifest SHA"
                )


class V10DictTensorDataset(Dataset):
    """Minimal map-style tensor dataset used by executable training entrypoints."""

    def __init__(self, tensors: Mapping[str, torch.Tensor], sample_ids: Sequence[str]):
        if not tensors:
            raise ValueError("tensor dataset cannot be empty")
        lengths = {int(tensor.shape[0]) for tensor in tensors.values()}
        if len(lengths) != 1 or next(iter(lengths)) != len(sample_ids):
            raise ValueError(
                "all tensors and sample IDs must share their first dimension"
            )
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample IDs must be unique")
        self.tensors = dict(tensors)
        self.sample_ids = tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            **{key: value[index] for key, value in self.tensors.items()},
            "sample_id": self.sample_ids[index],
        }


def stable_split_for_group(group_id: str, global_seed: int = 20261001) -> str:
    digest = hashlib.sha256(f"{global_seed}|{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return "train" if bucket < 8000 else ("validation" if bucket < 9000 else "test")


def grouped_split(
    sample_ids: Sequence[str], group_ids: Sequence[str], global_seed: int = 20261001
) -> Dict[str, tuple[str, ...]]:
    if len(sample_ids) != len(group_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "sample/group identities must align and sample IDs must be unique"
        )
    result: Dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    group_assignment: Dict[str, str] = {}
    for sample_id, group_id in zip(sample_ids, group_ids):
        split = group_assignment.setdefault(
            group_id, stable_split_for_group(group_id, global_seed)
        )
        result[split].append(sample_id)
    return {key: tuple(value) for key, value in result.items()}


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_move_to_device(item, device) for item in value)
    return value


def _state_dict_sha(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _model_build_spec(model: torch.nn.Module) -> Dict[str, Any]:
    """Describe the architecture needed for exact checkpoint reconstruction."""

    explicit_spec = getattr(model, "checkpoint_build_spec", None)
    if callable(explicit_spec):
        return explicit_spec()

    mode_input = getattr(model, "mode_input", None)
    measurement_input = getattr(model, "measurement_input", None)
    cross = getattr(model, "cross", None)
    encoder = getattr(model, "mode_encoder", None)
    moe = getattr(model, "moe", None)
    if any(
        value is None for value in (mode_input, measurement_input, cross, encoder, moe)
    ):
        return {"architecture_class": type(model).__name__}
    gate = getattr(moe, "gate", None)
    gate_input = gate[0].in_features if gate is not None else 0
    router = getattr(model, "relational_router", None)
    control_block = getattr(router, "control_block", None)
    node_projection = getattr(control_block, "node_projection", None)
    global_projection = getattr(control_block, "global_projection", None)
    return {
        "architecture_class": type(model).__name__,
        "input_dim": int(mode_input.in_features),
        "measurement_input_dim": int(measurement_input.in_features),
        "gate_input_dim": int(gate_input),
        "width": int(mode_input.out_features),
        "heads": int(cross.num_heads),
        "blocks": int(len(encoder.layers)),
        "csm_node_context_dim": int(node_projection.in_features)
        if node_projection
        else 0,
        "csm_global_context_dim": int(global_projection.in_features)
        if global_projection
        else 0,
    }


def _prediction_max_difference(before: Any, after: Any) -> float:
    if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
        return float((before.detach().cpu() - after.detach().cpu()).abs().max().item())
    if isinstance(before, dict) and isinstance(after, dict):
        common = sorted(set(before) & set(after))
        values = [
            _prediction_max_difference(before[key], after[key])
            for key in common
            if isinstance(before[key], (torch.Tensor, dict))
        ]
        return max(values, default=0.0)
    raise TypeError(
        "prediction function must return tensors or nested dictionaries of tensors"
    )


@runtime_checkable
class V10TrainingObserver(Protocol):
    """Lightweight observer for V10 training events.

    Observers may compute auxiliary diagnostics (e.g. gradient audit)
    but must NOT modify parameter.grad, optimizer state, or RNG.
    An optional on_step_end(record=...) receives detached scalar metadata only
    after an optimizer step has committed. An optional
    on_training_complete(final_global_step=..., registered_cell_ids=...) is
    called after the final committed step has been validated.
    It is not a required Protocol member, preserving older observers.
    """

    def on_before_backward(
        self,
        *,
        model: torch.nn.Module,
        batch: Dict[str, Any],
        losses: Dict[str, torch.Tensor],
        global_step: int,
        epoch: int,
    ) -> None:
        """Called BEFORE the main backward(), while the graph is alive.

        Must use torch.autograd.grad(retain_graph=True) on sub-losses.
        Must not call the main loss.backward() or modify parameter.grad.
        """
        ...

    def on_epoch_end(
        self,
        *,
        epoch: int,
        train_metrics: Dict[str, float],
        validation_metrics: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Called after each epoch completes."""
        ...

    def finalize(self) -> Dict[str, Any]:
        """Called after training completes. Returns summary dict."""
        ...

    def write_outputs(self) -> None:
        """Persist any accumulated diagnostics to disk."""
        ...


class V10Trainer:
    """One-model trainer with atomic checkpoints and exact RNG resume state."""

    @staticmethod
    def _validate_physical_datasets(
        train_dataset: Dataset,
        validation_dataset: Dataset,
        evidence: Mapping[str, Any],
        identity: V10TrainingIdentity,
        config: V10TrainingConfig,
    ) -> None:
        """Reject substitute datasets before any loader, output, or optimizer activity."""
        from ml_recon.physics_hybrid_v10.training.physical_dataset import Physical_aod_dataset

        if not isinstance(train_dataset, Physical_aod_dataset) or not isinstance(
            validation_dataset, Physical_aod_dataset
        ):
            raise ValueError(
                "physical admission requires Physical_aod_dataset train and validation inputs"
            )
        if train_dataset.split != "train" or validation_dataset.split != "validation":
            raise ValueError(
                "physical admission requires train and validation dataset roles"
            )
        if (
            train_dataset.recipe_sha != evidence["recipe_sha"]
            or validation_dataset.recipe_sha != evidence["recipe_sha"]
        ):
            raise ValueError("physical datasets do not use the admitted recipe")
        if (
            getattr(train_dataset.recipe, "operation_input_recipe", None)
            != identity.operation_input_recipe
            or getattr(validation_dataset.recipe, "operation_input_recipe", None)
            != identity.operation_input_recipe
        ):
            raise ValueError("physical datasets do not use the runtime operation recipe")
        if (
            train_dataset.manifest.sha256() != identity.synthetic_manifest_sha
            or validation_dataset.manifest.sha256() != identity.synthetic_manifest_sha
        ):
            raise ValueError("physical datasets do not use the admitted manifest")
        if (
            train_dataset.manifest.measurement_contract_sha
            != identity.measurement_contract_sha
        ):
            raise ValueError(
                "physical datasets do not use the admitted measurement contract"
            )
        if identity.dataset_hashes != {
            "train": train_dataset.membership_sha,
            "validation": validation_dataset.membership_sha,
        }:
            raise ValueError(
                "physical dataset membership hashes differ from the runtime identity"
            )
        if len(train_dataset) != config.sample_count or not len(validation_dataset):
            raise ValueError(
                "physical dataset row counts differ from the admitted runtime config"
            )

    @staticmethod
    def _validate_physical_model_binding(
        model: torch.nn.Module, identity: V10TrainingIdentity
    ) -> None:
        """Reject a recipe/model/output contract swap before optimizer creation."""
        actual_mode = getattr(model, "excitation_mode", None)
        if actual_mode != identity.excitation_mode:
            raise ValueError("physical model excitation mode differs from runtime identity")
        manifest_method = getattr(model, "output_training_manifest", None)
        if not callable(manifest_method):
            raise ValueError("physical model lacks a mode-bound output-training manifest")
        from ml_recon.physics_hybrid_v10.published_output_registry import (
            physical_output_training_manifest_sha,
            validate_physical_output_training_manifest,
        )

        manifest = validate_physical_output_training_manifest(manifest_method())
        manifest_sha = physical_output_training_manifest_sha(identity.excitation_mode)
        if (
            sha256_json(manifest) != manifest_sha
            or (
                identity.output_training_manifest_sha
                and identity.output_training_manifest_sha != manifest_sha
            )
        ):
            raise ValueError("physical model output manifest differs from runtime identity")

    def _verify_physical_admission_files(self) -> None:
        """Rehash every bound admission file before each physical batch/step."""
        for path_text, expected_sha in self._physical_admission_files.items():
            path = Path(path_text)
            try:
                path.stat()
            except OSError as error:
                raise ValueError(
                    "admitted physical evidence file disappeared during run"
                ) from error
            if sha256_file(path) != expected_sha:
                raise ValueError("admitted physical evidence bytes changed during run")

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: V10TrainingConfig,
        identity: V10TrainingIdentity,
        step_function: Callable[
            [torch.nn.Module, Dict[str, Any]], Dict[str, torch.Tensor]
        ],
        prediction_function: Callable[[torch.nn.Module, Dict[str, Any]], Any],
        device: str = "cpu",
        batch_sampler_factory: Optional[Callable[[Dataset, int, bool], Any]] = None,
        observers: Optional[Sequence[V10TrainingObserver]] = None,
        stream_pilot_admission_path: str | Path | None = None,
        physical_admission_path: str | Path | None = None,
        collate_function: Optional[Callable[[Any], Any]] = None,
    ):
        config.validate()
        identity.validate()
        self.stream_pilot_recipe = None
        self.stream_pilot_evidence = {}
        self.stream_pilot_admission_path = stream_pilot_admission_path
        self.physical_admission_evidence = {}
        self._physical_admission_files: dict[str, str] = {}
        self._physical_admission_stats: dict[str, tuple[int, int]] = {}
        self.physical_admission_path = physical_admission_path
        if identity.branch == PHYSICAL_TRAINING_BRANCH:
            if config.run_mode == "smoke":
                raise ValueError("physical training cannot use run_mode=smoke")
            from ml_recon.physics_hybrid_v10.training.physical_admission import validate_physical_admission

            self.physical_admission_evidence = validate_physical_admission(
                physical_admission_path,
                identity=asdict(identity),
                config=asdict(config),
                device=device,
            )
            if (
                self.physical_admission_evidence["admission_sha"]
                != identity.physical_admission_sha
            ):
                raise ValueError(
                    "physical admission file SHA differs from runtime identity"
                )
            self._physical_admission_files = dict(
                self.physical_admission_evidence.get("immutable_files", {})
            )
            self._physical_admission_stats = {
                path_text: (
                    Path(path_text).stat().st_mtime_ns,
                    Path(path_text).stat().st_size,
                )
                for path_text in self._physical_admission_files
            }
        if (
            identity.input_delivery == "stream_only"
            and config.run_mode != "smoke"
            and identity.branch != PHYSICAL_TRAINING_BRANCH
        ):
            from ml_recon.physics_hybrid_v10.stream_admission import validate_core_stream_pilot

            self.stream_pilot_recipe = validate_core_stream_pilot(
                admission_path=stream_pilot_admission_path,
                config=asdict(config),
                identity=asdict(identity),
                device=device,
                model_profile=getattr(model, "profile_name", None),
            )
        if not callable(
            getattr(torch.optim.Optimizer, "register_step_post_hook", None)
        ):
            raise RuntimeError(
                "training requires PyTorch optimizer post-step hook support"
            )
        self.model = model
        if identity.branch in (
            PHYSICAL_TRAINING_BRANCH,
            TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH,
        ):
            self._validate_physical_model_binding(model, identity)
        self.config = config
        self.identity = identity
        self.step_function = step_function
        self.prediction_function = prediction_function
        self.batch_sampler_factory = batch_sampler_factory
        self.collate_function = collate_function
        self.observers = list(observers) if observers else []
        if config.run_mode != "smoke":
            required = [
                observer
                for observer in self.observers
                if isinstance(observer, V10GradientAuditObserver)
            ]
            if not required or any(
                not observer.config.enabled
                or any(
                    not callable(getattr(observer, method, None))
                    for method in ("on_before_backward", "finalize", "write_outputs")
                )
                for observer in required
            ):
                raise RuntimeError(
                    "required gradient audit observer is unavailable or disabled"
                )
            model_parameters = {
                id(parameter)
                for parameter in model.parameters()
                if parameter.requires_grad
            }
            if any(
                not observer.shared_params
                or any(
                    id(parameter) not in model_parameters
                    for parameter in observer.shared_params
                )
                for observer in required
            ):
                raise RuntimeError(
                    "required gradient observer is not bound to trainable model parameters"
                )
        self.device = torch.device(device)
        self.output_dir = Path(config.output_dir)
        if not config.resume_checkpoint and any(
            (self.output_dir / name).exists() for name in TRAINING_OUTPUT_NAMES
        ):
            raise ValueError(
                "existing training output requires a new directory or explicit resume"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.stream_pilot_recipe is not None:
            from ml_recon.physics_hybrid_v10.stream_admission import preserve_stream_pilot_evidence

            self.stream_pilot_evidence = preserve_stream_pilot_evidence(
                stream_pilot_admission_path,
                identity.stream_pilot_admission_sha,
                self.output_dir,
            )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(config.epochs, 1)
        )
        amp_active = config.amp_enabled and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_active, init_scale=config.amp_initial_scale
        )
        self.amp_active = amp_active
        self.start_epoch = 0
        self.resume_batch_cursor = 0
        self.resume_loader_identity = ""
        self.resume_committed_memberships: list[tuple[str, ...]] = []
        self.resume_epoch_complete = False
        self._pending_observer_state: list[dict[str, Any]] | None = None
        self.global_step = 0
        self.amp_overflow_count = 0
        self.optimizer_attempt_count = 0
        self.overflow_counter_scope = "SINCE_RUN_START"
        self._optimizer_step_handle = self.optimizer.register_step_post_hook(
            self._record_optimizer_step
        )
        self.best_metric = (
            float("inf") if config.metric_direction == "min" else -float("inf")
        )
        self.bad_epochs = 0
        self._seed_all(config.global_seed)
        if config.resume_checkpoint:
            self._load_checkpoint(Path(config.resume_checkpoint))
        self.telemetry = Training_telemetry(
            self.output_dir,
            identity=asdict(identity),
            model_id=config.model_id,
            track_id=config.track_id,
            run_id=config.run_id,
            enabled=config.telemetry_enabled,
            interval_steps=config.telemetry_interval_steps,
            interval_seconds=config.telemetry_interval_seconds,
        )
        registered_cell_ids = tuple(
            sorted(filter(None, identity.dataset_cell_id.split(",")))
        )
        for observer in self.observers:
            identity_bind = getattr(observer, "bind_training_identity", None)
            if callable(identity_bind):
                identity_bind(
                    identity=asdict(identity),
                    run_id=config.run_id,
                    execution_id=self.telemetry.execution_id,
                )
            bind = getattr(observer, "bind_run_context", None)
            if isinstance(observer, V10GradientAuditObserver) and callable(bind):
                bind(
                    run_id=config.run_id,
                    execution_id=self.telemetry.execution_id,
                    initial_attempt_index=self.optimizer_attempt_count,
                )
            configure_cells = getattr(observer, "configure_registered_cells", None)
            if callable(configure_cells):
                configure_cells(registered_cell_ids)
        if self._pending_observer_state is not None:
            for observer, state in zip(self.observers, self._pending_observer_state, strict=True):
                restore = getattr(observer, "restore_checkpoint_state", None)
                if not callable(restore):
                    raise ValueError("physical checkpoint observer cannot restore its state")
                restore(state, identity=asdict(identity))
            self._pending_observer_state = None

    def _record_optimizer_step(self, optimizer, args, kwargs) -> None:
        # A GradScaler attempt may skip optimizer.step(); count committed calls.
        self.global_step += 1

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _loader(self, dataset: Dataset, *, shuffle: bool, epoch: int = 0) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(
            self.config.global_seed + (17 if shuffle else 29) + 1_000_003 * int(epoch)
        )
        if self.batch_sampler_factory is not None:
            batch_sampler = self.batch_sampler_factory(dataset, epoch, shuffle)
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.num_workers,
                generator=generator,
                collate_fn=self.collate_function,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            generator=generator,
            drop_last=False,
            collate_fn=self.collate_function,
        )

    def _train_loader_identity(self, dataset: Dataset, epoch: int) -> str:
        """Bind an epoch cursor to the exact deterministic loader recipe.

        A mid-epoch cursor without an ordering identity is unsafe: it can replay
        a batch from another sampler or silently skip a new one.  The identity
        intentionally uses source-bound membership when the dataset exposes it
        and otherwise refuses mid-epoch recovery.
        """
        sample_ids = getattr(dataset, "sample_ids", None)
        if not isinstance(sample_ids, tuple) or len(sample_ids) != len(dataset):
            raise ValueError(
                "mid-epoch resume requires source-bound dataset sample_ids"
            )
        sampler = (
            self.batch_sampler_factory(dataset, epoch, True)
            if self.batch_sampler_factory is not None
            else None
        )
        sampler_payload = {
            "class": None if sampler is None else type(sampler).__qualname__,
            "module": None if sampler is None else type(sampler).__module__,
            "batch_size": self.config.batch_size,
            "epoch": int(epoch),
            "shuffle": True,
            "seed": self.config.global_seed + 17 + 1_000_003 * int(epoch),
        }
        for name in ("batch_size", "epoch", "shuffle"):
            if sampler is not None and hasattr(sampler, name):
                sampler_payload[name] = getattr(sampler, name)
        return sha256_json(
            {
                "schema": "V10DeterministicTrainLoaderIdentityV1",
                "dataset_type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
                "dataset_membership_sha": getattr(dataset, "membership_sha", ""),
                "sample_ids": list(sample_ids),
                "sampler": sampler_payload,
                **(
                    {"physical_mode_binding": _physical_mode_binding(self.identity)}
                    if self.identity.branch == PHYSICAL_TRAINING_BRANCH
                    else {}
                ),
            }
        )

    @staticmethod
    def _batch_membership(batch: Mapping[str, Any]) -> tuple[str, ...]:
        values = batch.get("sample_id")
        if isinstance(values, str):
            values = (values,)
        if not isinstance(values, (tuple, list)) or not values:
            raise ValueError("mid-epoch resume requires committed batch sample identities")
        members = tuple(str(value) for value in values)
        if any(not value for value in members) or len(members) != len(set(members)):
            raise ValueError("committed batch sample identities must be unique non-empty strings")
        return members

    def _require_complete_output_audit(
        self,
        summary: Mapping[str, Any],
        train_dataset: Dataset,
    ) -> None:
        """Reject completion unless registered output slices audited real fit rows."""
        from ml_recon.physics_hybrid_v10.classifier_output_audit import output_slices

        output_audit = summary.get("observer_Classifier_output_audit", {})
        expected_cells = {
            value for value in self.identity.dataset_cell_id.split(",") if value
        }
        expected_rows = set(getattr(train_dataset, "sample_ids", ()))
        observed = output_audit.get("observed_fit_membership", {})
        observed_rows = set(observed.get("row_ids", ())) if isinstance(observed, dict) else set()
        observed_row_ids = observed.get("row_ids", ()) if isinstance(observed, dict) else ()
        observed_group_ids = (
            observed.get("acquisition_group_ids", ()) if isinstance(observed, dict) else ()
        )
        observed_membership = (
            dict(zip(observed_row_ids, observed_group_ids, strict=True))
            if isinstance(observed_row_ids, list)
            and isinstance(observed_group_ids, list)
            and len(observed_row_ids) == len(observed_group_ids)
            and len(observed_row_ids) == len(set(observed_row_ids))
            else {}
        )
        memberships = output_audit.get("committed_batch_memberships", ())
        records = output_audit.get("records", ())
        expected_outputs = {spec.output_name for spec in output_slices(self.model)}
        membership_by_step: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        membership_rows: list[str] = []
        membership_valid = isinstance(memberships, list)
        if membership_valid:
            for membership in memberships:
                step = membership.get("global_step") if isinstance(membership, dict) else None
                rows = membership.get("sample_ids") if isinstance(membership, dict) else None
                groups = membership.get("acquisition_group_ids") if isinstance(membership, dict) else None
                if (
                    type(step) is not int
                    or step < 1
                    or step in membership_by_step
                    or not isinstance(rows, list)
                    or not isinstance(groups, list)
                    or not rows
                    or len(rows) != len(groups)
                    or any(not isinstance(value, str) or not value for value in (*rows, *groups))
                    or len(rows) != len(set(rows))
                ):
                    membership_valid = False
                    break
                membership_by_step[step] = (tuple(rows), tuple(groups))
                membership_rows.extend(rows)
        stage = self.physical_admission_evidence.get("stage") if hasattr(self, "physical_admission_evidence") else None
        row_visits = {row_id: membership_rows.count(row_id) for row_id in expected_rows}
        visit_valid = (
            all(count == 1 for count in row_visits.values()) if stage == "E0"
            else all(count == self.config.epochs for count in row_visits.values()) if stage == "E3"
            else True
        )
        membership_valid = membership_valid and (
            set(membership_by_step) == set(range(1, self.global_step + 1))
            and set(membership_rows) == expected_rows
            and visit_valid
            and all(
                observed_membership.get(row_id) == group_id
                for rows, groups in membership_by_step.values()
                for row_id, group_id in zip(rows, groups, strict=True)
            )
        )
        records_cover_outputs = bool(records) and all(
            isinstance(record, dict)
            and set(record.get("outputs", {})) == expected_outputs
            and type(record.get("global_step")) is int
            and record["global_step"] in membership_by_step
            and record.get("cell_id") in expected_cells
            and tuple(record.get("sample_ids", ()))
            == membership_by_step[record["global_step"]][0]
            and tuple(record.get("acquisition_group_ids", ()))
            == membership_by_step[record["global_step"]][1]
            for record in records
        )
        if (
            output_audit.get("audit_complete") is not True
            or set(output_audit.get("audited_cells", ())) != expected_cells
            or set(output_audit.get("registered_cells", ())) != expected_cells
            or output_audit.get("final_committed_step") != self.global_step
            or output_audit.get("fit_membership_capture_status")
            != "CAPTURED_FROM_COMMITTED_BATCH_IDENTITIES"
            or not membership_valid
            or observed_rows != expected_rows
            or not records_cover_outputs
        ):
            raise RuntimeError(
                "physical/stream completion requires complete registered-cell output auditing, "
                "committed source memberships, and every registered output slice"
            )

    def _rng_state(self) -> Dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else (),
        }

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state["torch_cuda"]:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _checkpoint(self, epoch: int, training_complete: bool) -> Dict[str, Any]:
        from ml_recon.physics_hybrid_v10.checkpoint_storage import cpu_checkpoint_snapshot

        self._validate_stream_recipe_for_resume(
            self.output_dir / "checkpoint_latest.pt"
        )
        model_sha = _state_dict_sha(self.model.state_dict())
        model_build_spec = _model_build_spec(self.model)
        metadata = {
            "schema": "V10CheckpointV1",
            "stream_pilot_evidence": self.stream_pilot_evidence,
            "model_id": self.config.model_id,
            "branch": self.identity.branch,
            "source_commit": self.identity.source_commit,
            "source_tree_sha": self.identity.source_tree_sha,
            "dirty_patch_sha": self.identity.dirty_patch_sha,
            "v23_manifest_sha": self.identity.v23_manifest_sha,
            "synthetic_manifest_sha": self.identity.synthetic_manifest_sha,
            "measurement_contract_sha": self.identity.measurement_contract_sha,
            "shot_budget_contract_sha": self.identity.shot_budget_contract_sha,
            "rho8_sha": self.identity.rho8_sha,
            "config_sha": self.identity.config_sha,
            "seed_panel": self.config.seed_panel,
            "exact_parameter_count": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
            "calibration": {},
            "negative_controls": {},
            "selection_evidence": False,
            "research_only": True,
            "source_generation": "V10",
            "run_mode": self.config.run_mode,
            "sample_count": self.config.sample_count,
            "completed_sample_rungs": self.config.completed_sample_rungs,
            "phase_1_to_7_gate_sha": self.config.phase_1_to_7_gate_sha,
            "phase_8_gate_sha": self.config.phase_8_gate_sha,
            "dataset_hashes": self.identity.dataset_hashes,
            "split_hash": self.identity.split_hash,
            "dataset_contract_version": self.identity.dataset_contract_version,
            "dataset_contract_sha": self.identity.dataset_contract_sha,
            "dataset_build_id": self.identity.dataset_build_id,
            "dataset_index_sha": self.identity.dataset_index_sha,
            "dataset_cell_id": self.identity.dataset_cell_id,
            "input_delivery": self.identity.input_delivery,
            "dataset_index_kind": self.identity.dataset_index_kind,
            "stream_recipe_sha": self.identity.stream_recipe_sha,
            "stream_recipe_file_sha": self.identity.stream_recipe_file_sha,
            "stream_pilot_admission_sha": self.identity.stream_pilot_admission_sha,
            "model_state_sha": model_sha,
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "metric_name": self.config.metric_name,
            "metric_direction": self.config.metric_direction,
            "training_complete": bool(training_complete),
        }
        output_manifest = getattr(self.model, "output_training_manifest", None)
        if callable(output_manifest):
            metadata["published_output_training"] = output_manifest()
        if self.identity.branch in (
            PHYSICAL_TRAINING_BRANCH,
            TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH,
        ):
            metadata.update(
                {
                    "operation_input_recipe": self.identity.operation_input_recipe,
                    "excitation_mode": self.identity.excitation_mode,
                    "output_training_manifest_sha": (
                        self.identity.output_training_manifest_sha
                        or _physical_output_manifest_sha(self.identity.excitation_mode)
                    ),
                    "physical_mode_binding": _physical_mode_binding(self.identity),
                }
            )
        if self.identity.branch == PHYSICAL_TRAINING_BRANCH:
            metadata.update(
                {
                    "physical_namespace": self.identity.physical_namespace,
                    "physical_admission_sha": self.identity.physical_admission_sha,
                    "strict_stream_only_identity": self.identity.strict_stream_only_identity,
                    "coordinate_layout_sha": self.identity.coordinate_layout_sha,
                    "physical_identity": self.identity.physical_identity,
                    "physical_admission_evidence": self.physical_admission_evidence,
                }
            )
            metadata["resume_protocol"] = RESUME_STATE_SCHEMA
        validate_checkpoint_envelope(metadata)
        observer_state = []
        for observer in self.observers:
            snapshot = getattr(observer, "checkpoint_state", None)
            if self.identity.branch == PHYSICAL_TRAINING_BRANCH and not callable(snapshot):
                raise RuntimeError("physical checkpoint requires restorable observer state")
            if callable(snapshot):
                observer_state.append({"type": f"{type(observer).__module__}.{type(observer).__qualname__}", "identity": asdict(self.identity), "state": snapshot()})
        return cpu_checkpoint_snapshot(
            {
                "metadata": metadata,
                "model_build_spec": model_build_spec,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "rng_state": self._rng_state(),
                "trainer_state": {
                    "bad_epochs": self.bad_epochs,
                    "amp_overflow_count": self.amp_overflow_count,
                    "optimizer_attempt_count": self.optimizer_attempt_count,
                    "overflow_counter_scope": self.overflow_counter_scope,
                },
                **({"resume_state": {
                    "schema": RESUME_STATE_SCHEMA,
                    "epoch": int(epoch),
                    "next_train_batch_index": self.resume_batch_cursor,
                    "epoch_complete": self.resume_epoch_complete,
                    "train_loader_identity": self.resume_loader_identity,
                    "committed_batch_memberships": [
                        list(values) for values in self.resume_committed_memberships
                    ],
                }, "observer_state": observer_state} if self.identity.branch == PHYSICAL_TRAINING_BRANCH else {}),
            }
        )

    def _save_checkpoint(self, epoch: int, training_complete: bool) -> Path:
        if training_complete:
            raise ValueError(
                "complete checkpoints require validated completion publication"
            )
        checkpoint = self._checkpoint(epoch, training_complete)
        destination = self.output_dir / "checkpoint_latest.pt"
        temporary = self.output_dir / "checkpoint_latest.pt.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
        self.telemetry.record_checkpoint(
            destination,
            epoch=epoch,
            global_step=self.global_step,
            training_complete=training_complete,
            model_state_sha=checkpoint["metadata"]["model_state_sha"],
        )
        return destination

    def _validate_physical_checkpoint_output_manifest(
        self, metadata: Mapping[str, Any]
    ) -> None:
        """Bind a physical resume to the current model's exact consumer contract."""
        from ml_recon.physics_hybrid_v10.published_output_registry import (
            validate_physical_output_training_manifest,
        )

        current_manifest = getattr(self.model, "output_training_manifest", None)
        if not callable(current_manifest):
            raise ValueError("physical resume model lacks an output-training manifest")
        expected_manifest = validate_physical_output_training_manifest(
            current_manifest()
        )
        identity = getattr(self, "identity", None)
        if identity is None:
            if metadata.get("published_output_training") != expected_manifest:
                raise ValueError("physical resume checkpoint output-training manifest mismatch")
            return
        historical_legacy_v3 = (
            identity.branch == PHYSICAL_TRAINING_BRANCH
            and identity.operation_input_recipe == "legacy_v1"
            and identity.excitation_mode == "legacy"
            and not identity.output_training_manifest_sha
            and all(
                metadata.get(key) is None
                for key in (
                    "operation_input_recipe",
                    "excitation_mode",
                    "output_training_manifest_sha",
                    "physical_mode_binding",
                )
            )
        )
        if historical_legacy_v3:
            if metadata.get("published_output_training") != expected_manifest:
                raise ValueError("physical resume checkpoint output-training manifest mismatch")
            return
        expected_manifest_sha = _physical_output_manifest_sha(identity.excitation_mode)
        if (
            metadata.get("published_output_training") != expected_manifest
            or metadata.get("output_training_manifest_sha") != expected_manifest_sha
            or metadata.get("physical_mode_binding") != _physical_mode_binding(identity)
        ):
            raise ValueError("physical resume checkpoint output-training manifest mismatch")

    def _load_checkpoint(self, path: Path) -> None:
        self._validate_stream_recipe_for_resume(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        validate_checkpoint_envelope(checkpoint["metadata"])
        historical_legacy_v3 = (
            self.identity.branch == PHYSICAL_TRAINING_BRANCH
            and self.identity.operation_input_recipe == "legacy_v1"
            and self.identity.excitation_mode == "legacy"
            and not self.identity.output_training_manifest_sha
            and all(
                checkpoint["metadata"].get(key) is None
                for key in (
                    "operation_input_recipe",
                    "excitation_mode",
                    "output_training_manifest_sha",
                    "physical_mode_binding",
                )
            )
        )
        if self.identity.branch == PHYSICAL_TRAINING_BRANCH and not historical_legacy_v3:
            observer_state = checkpoint.get("observer_state")
            if not isinstance(observer_state, list) or len(observer_state) != len(self.observers):
                raise ValueError("physical checkpoint lacks complete observer state")
            if any(row.get("type") != f"{type(observer).__module__}.{type(observer).__qualname__}" or row.get("identity") != asdict(self.identity) or "state" not in row for row, observer in zip(observer_state, self.observers, strict=True)):
                raise ValueError("physical checkpoint observer state identity mismatch")
            self._pending_observer_state = [row["state"] for row in observer_state]
        if checkpoint["metadata"]["model_id"] != self.config.model_id:
            raise ValueError("resume checkpoint model_id mismatch")
        expected_identity = {
            "branch": self.identity.branch,
            "source_commit": self.identity.source_commit,
            "source_tree_sha": self.identity.source_tree_sha,
            "dirty_patch_sha": self.identity.dirty_patch_sha,
            "v23_manifest_sha": self.identity.v23_manifest_sha,
            "synthetic_manifest_sha": self.identity.synthetic_manifest_sha,
            "measurement_contract_sha": self.identity.measurement_contract_sha,
            "shot_budget_contract_sha": self.identity.shot_budget_contract_sha,
            "rho8_sha": self.identity.rho8_sha,
            "config_sha": self.identity.config_sha,
            "dataset_hashes": self.identity.dataset_hashes,
            "split_hash": self.identity.split_hash,
            "dataset_contract_version": self.identity.dataset_contract_version,
            "dataset_contract_sha": self.identity.dataset_contract_sha,
            "dataset_build_id": self.identity.dataset_build_id,
            "dataset_index_sha": self.identity.dataset_index_sha,
            "dataset_cell_id": self.identity.dataset_cell_id,
        }
        if self.identity.branch == PHYSICAL_TRAINING_BRANCH:
            expected_identity.update(
                {
                    "physical_namespace": self.identity.physical_namespace,
                    "physical_admission_sha": self.identity.physical_admission_sha,
                    "strict_stream_only_identity": self.identity.strict_stream_only_identity,
                    "coordinate_layout_sha": self.identity.coordinate_layout_sha,
                    "physical_identity": self.identity.physical_identity,
                }
            )
        if self.identity.branch in (
            PHYSICAL_TRAINING_BRANCH,
            TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH,
        ) and not historical_legacy_v3:
            expected_identity.update(
                {
                    "operation_input_recipe": self.identity.operation_input_recipe,
                    "excitation_mode": self.identity.excitation_mode,
                    "output_training_manifest_sha": (
                        self.identity.output_training_manifest_sha
                        or _physical_output_manifest_sha(self.identity.excitation_mode)
                    ),
                }
            )
        mismatches = [
            key
            for key, value in expected_identity.items()
            if checkpoint["metadata"].get(key) != value
        ]
        for key, default in (
            ("input_delivery", "materialized"),
            ("dataset_index_kind", "materialized"),
            ("stream_recipe_sha", ""),
            ("stream_recipe_file_sha", ""),
            ("stream_pilot_admission_sha", ""),
        ):
            if checkpoint["metadata"].get(key, default) != getattr(self.identity, key):
                mismatches.append(key)
        if self.identity.branch == PHYSICAL_TRAINING_BRANCH and not historical_legacy_v3 and (
            checkpoint["metadata"].get("physical_mode_binding")
            != _physical_mode_binding(self.identity)
        ):
            mismatches.append("physical_mode_binding")
        if self.identity.branch == TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH and (
            checkpoint["metadata"].get("physical_mode_binding")
            != _physical_mode_binding(self.identity)
        ):
            mismatches.append("physical_mode_binding")
        if mismatches:
            raise ValueError(f"resume checkpoint identity mismatch: {mismatches}")
        if self.identity.branch in (
            PHYSICAL_TRAINING_BRANCH,
            TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH,
        ):
            build_spec = checkpoint.get("model_build_spec")
            from ml_recon.physics_hybrid_v10.four_cell_layout import TRACK_A_FOUR_CELL_LAYOUT_SHA
            if (
                not isinstance(build_spec, dict)
                or build_spec.get("four_cell_layout_sha")
                != TRACK_A_FOUR_CELL_LAYOUT_SHA
                or (
                    self.identity.branch == PHYSICAL_TRAINING_BRANCH
                    and build_spec.get("coordinate_layout_sha")
                    != self.identity.coordinate_layout_sha
                )
                or (
                    self.identity.branch
                    == TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH
                    and (
                        build_spec.get("excitation_mode")
                        != self.identity.excitation_mode
                        or build_spec.get("objective_supervision_schema")
                        != "V10PhysicalObjectiveSupervisionV2"
                    )
                )
            ):
                raise ValueError(
                    "physical checkpoint model metadata has a coordinate or four-cell layout mismatch"
                )
            self._validate_physical_checkpoint_output_manifest(checkpoint["metadata"])
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if (
            _state_dict_sha(self.model.state_dict())
            != checkpoint["metadata"]["model_state_sha"]
        ):
            raise ValueError("resume checkpoint model-state hash mismatch")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self._restore_rng_state(checkpoint["rng_state"])
        resume_state = checkpoint.get("resume_state")
        if checkpoint["metadata"].get("resume_protocol") != RESUME_STATE_SCHEMA:
            # Checkpoints predating per-step snapshots were only written after
            # an epoch.  They may resume at the next epoch, never mid-epoch.
            self.start_epoch = int(checkpoint["metadata"]["epoch"]) + 1
            self.resume_batch_cursor = 0
            self.resume_loader_identity = ""
            self.resume_committed_memberships = []
            resume_state = None
        elif not isinstance(resume_state, dict) or resume_state.get("schema") != RESUME_STATE_SCHEMA:
            raise ValueError(
                "checkpoint lacks a fail-closed deterministic mid-epoch resume state"
            )
        checkpoint_epoch = int(checkpoint["metadata"]["epoch"])
        if resume_state is not None and resume_state.get("epoch") != checkpoint_epoch:
            raise ValueError("checkpoint resume cursor epoch disagrees with metadata")
        cursor = None if resume_state is None else resume_state.get("next_train_batch_index")
        loader_identity = None if resume_state is None else resume_state.get("train_loader_identity")
        memberships = None if resume_state is None else resume_state.get("committed_batch_memberships")
        epoch_complete = None if resume_state is None else resume_state.get("epoch_complete")
        if resume_state is not None and (
            type(cursor) is not int
            or cursor < 0
            or not isinstance(loader_identity, str)
            or not isinstance(memberships, list)
            or type(epoch_complete) is not bool
            or len(memberships) != cursor
            or any(
                not isinstance(row, list)
                or not row
                or any(not isinstance(value, str) or not value for value in row)
                or len(row) != len(set(row))
                for row in memberships
            )
        ):
            raise ValueError("checkpoint deterministic resume cursor is invalid")
        if resume_state is not None:
            training_complete = bool(checkpoint["metadata"]["training_complete"])
            if epoch_complete:
                if cursor or loader_identity or memberships:
                    raise ValueError("epoch-complete checkpoint cannot retain a cursor")
            elif not cursor or not loader_identity:
                raise ValueError("mid-epoch checkpoint requires a non-empty cursor and loader identity")
            if training_complete and not epoch_complete:
                raise ValueError("complete checkpoint must identify an epoch boundary")
            self.start_epoch = checkpoint_epoch + int(epoch_complete)
            self.resume_batch_cursor = cursor
            self.resume_loader_identity = loader_identity
            self.resume_committed_memberships = [tuple(row) for row in memberships]
            self.resume_epoch_complete = epoch_complete
        self.global_step = int(checkpoint["metadata"]["global_step"])
        self.best_metric = float(checkpoint["metadata"]["best_metric"])
        self.bad_epochs = int(checkpoint["trainer_state"]["bad_epochs"])
        for key in ("amp_overflow_count", "optimizer_attempt_count"):
            value = checkpoint["trainer_state"].get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError(
                    "checkpoint observation counters must be nonnegative integers"
                )
        self.amp_overflow_count = checkpoint["trainer_state"].get(
            "amp_overflow_count", 0
        )
        self.optimizer_attempt_count = checkpoint["trainer_state"].get(
            "optimizer_attempt_count", 0
        )
        self.overflow_counter_scope = checkpoint["trainer_state"].get(
            "overflow_counter_scope", "SINCE_RESUME_MISSING_HISTORICAL_BASELINE"
        )

    def _validate_stream_recipe_for_resume(self, checkpoint_path: Path) -> None:
        """Make direct V10Trainer resume reject swapped stream recipe bytes too."""
        if self.identity.input_delivery != "stream_only":
            return
        recipe_path = checkpoint_path.parent / "stream_recipe.json"
        if not recipe_path.is_file():
            raise ValueError(
                "stream checkpoint resume requires sibling stream_recipe.json"
            )
        if sha256_file(recipe_path) != self.identity.stream_recipe_file_sha:
            raise ValueError("stream checkpoint recipe file SHA mismatch")
        try:
            recipe = read_canonical_json(recipe_path)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                "stream checkpoint recipe is malformed or noncanonical"
            ) from exc
        if not isinstance(recipe, dict):
            raise ValueError("stream checkpoint recipe must be a JSON object")
        if self.identity.branch == PHYSICAL_TRAINING_BRANCH:
            from ml_recon.physics_hybrid_v10.training.physical_dataset import Physical_dataset_recipe

            try:
                operation_input_recipe = recipe.get("operation_input_recipe", "legacy_v1")
                template = Physical_dataset_recipe(
                    operation_input_recipe=operation_input_recipe
                )
                if set(recipe) != set(template.to_dict()) | {"recipe_sha256"}:
                    raise ValueError("physical recipe requires all declared layout fields")
                recipe_value = Physical_dataset_recipe(
                    **{
                        key: tuple(value)
                        if key in ("n_values", "split_sizes") and value is not None
                        else value
                        for key, value in recipe.items()
                        if key not in ("recipe_sha256", "schema")
                    }
                )
                recipe_value.validate()
                if recipe_value.sha256() != recipe.get("recipe_sha256"):
                    raise ValueError("physical recipe layout payload is not canonical")
                if (
                    recipe_value.operation_input_recipe
                    != self.identity.operation_input_recipe
                ):
                    raise ValueError("physical recipe operation-input mode mismatch")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "physical stream checkpoint recipe four-cell layout is invalid"
                ) from exc
        declared_sha = recipe.get("recipe_sha256")
        payload = {
            key: value for key, value in recipe.items() if key != "recipe_sha256"
        }
        if (
            declared_sha != sha256_json(payload)
            or declared_sha != self.identity.stream_recipe_sha
        ):
            raise ValueError("stream checkpoint canonical recipe payload SHA mismatch")

    def _run_batches(
        self,
        loader: DataLoader,
        *,
        training: bool,
        maximum_batches: Optional[int],
        epoch: int = 0,
        start_batch_index: int = 0,
        expected_memberships: Sequence[tuple[str, ...]] = (),
        pause_after_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        self.model.train(training)
        totals: Dict[str, float] = {}
        count = 0
        gradient_finite = True
        reached_loader_end = True
        last_batch_finished_at = time.perf_counter()
        if not training:
            self.telemetry.record_phase("evaluating")
        for batch_index, raw_batch in enumerate(loader):
            received_at = time.perf_counter()
            generator_wait = received_at - last_batch_finished_at
            if batch_index < start_batch_index:
                if not training or batch_index >= len(expected_memberships):
                    raise ValueError("checkpoint resume cursor cannot be verified")
                if self._batch_membership(raw_batch) != expected_memberships[batch_index]:
                    raise ValueError("checkpoint resume would replay a different committed batch")
                continue
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            if self._physical_admission_files:
                self._verify_physical_admission_files()
            batch = _move_to_device(raw_batch, self.device)
            if training:
                self.optimizer_attempt_count += 1
                self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_active):
                losses = self.step_function(self.model, batch)
                if "loss" not in losses or losses["loss"].ndim != 0:
                    raise ValueError("step_function must return a scalar loss")
                loss = losses["loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training/evaluation loss")
            if training:
                # Observers compute auxiliary gradients BEFORE the main backward,
                # while the computation graph is still alive.
                for observer in self.observers:
                    arguments = dict(
                        model=self.model,
                        batch=batch,
                        losses=losses,
                        global_step=self.global_step + 1,
                        epoch=epoch,
                    )
                    if isinstance(observer, V10GradientAuditObserver):
                        arguments["attempt_index"] = self.optimizer_attempt_count
                    observer.on_before_backward(**arguments)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradients = [
                    parameter.grad
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                ]
                if not gradients:
                    raise RuntimeError("training loss produced no gradients")
                gradient_finite = gradient_finite and all(
                    bool(torch.all(torch.isfinite(gradient))) for gradient in gradients
                )
                if not gradient_finite:
                    invalid_parameters = [
                        name
                        for name, parameter in self.model.named_parameters()
                        if parameter.grad is not None
                        and not bool(torch.all(torch.isfinite(parameter.grad)))
                    ]
                    scalar_losses = {
                        key: float(value.detach().float().cpu())
                        for key, value in losses.items()
                        if isinstance(value, torch.Tensor) and value.ndim == 0
                    }
                    raise FloatingPointError(
                        "non-finite gradient detected; "
                        f"epoch={epoch}; batch_index={batch_index}; "
                        f"parameters={invalid_parameters[:16]}; "
                        f"losses={scalar_losses}"
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                if self._physical_admission_files:
                    self._verify_physical_admission_files()
                previous_step = self.global_step
                previous_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                current_scale = self.scaler.get_scale()
                self.amp_overflow_count += int(current_scale < previous_scale)
                scalar_losses = {
                    key: float(value.detach().cpu())
                    for key, value in losses.items()
                    if isinstance(value, torch.Tensor) and value.ndim == 0
                }
                record = training_step_record(
                    batch=batch,
                    scalar_losses=scalar_losses,
                    epoch=epoch,
                    global_step=self.global_step,
                    optimizer_step_executed=self.global_step > previous_step,
                    gradient_norm=float(gradient_norm),
                    learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    amp_active=self.amp_active,
                    amp_scale=current_scale,
                    overflow_count=self.amp_overflow_count,
                    attempt_index=self.optimizer_attempt_count,
                    execution_id=self.telemetry.execution_id,
                    amp_scale_before=previous_scale,
                    overflow_counter_scope=self.overflow_counter_scope,
                    step_seconds=time.perf_counter() - received_at,
                    generator_wait_seconds=generator_wait,
                    device=str(self.device),
                )
                raw_cell_ids = batch.get("cell_id")
                if isinstance(raw_cell_ids, (list, tuple)):
                    record["cell_ids"] = sorted(
                        {
                            cell_id
                            for cell_id in raw_cell_ids
                            if isinstance(cell_id, str) and cell_id
                        }
                    )
                elif isinstance(raw_cell_ids, str):
                    record["cell_ids"] = [raw_cell_ids]
                else:
                    record["cell_ids"] = []
                if self.device.type == "cuda":
                    record["gpu_memory"] = {
                        "allocated_bytes": torch.cuda.memory_allocated(self.device),
                        "reserved_bytes": torch.cuda.memory_reserved(self.device),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                            self.device
                        ),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                            self.device
                        ),
                    }
                else:
                    record["gpu_memory"] = None
                if self.global_step > previous_step:
                    record["committed_step"] = True
                self.telemetry.record_step(record)
                if self.global_step > previous_step:
                    for observer in self.observers:
                        callback = getattr(observer, "on_step_end", None)
                        if callable(callback):
                            callback(record=copy.deepcopy(record))
                    if training and self.identity.branch == PHYSICAL_TRAINING_BRANCH:
                        self.resume_batch_cursor = batch_index + 1
                        self.resume_epoch_complete = False
                        self.resume_committed_memberships.append(
                            self._batch_membership(raw_batch)
                        )
                        self._save_checkpoint(epoch, training_complete=False)
                elif training:
                    raise RuntimeError(
                        "deterministic checkpointing rejects an uncommitted training batch"
                    )
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            count += 1
            if pause_after_batches is not None and count >= pause_after_batches:
                reached_loader_end = False
                break
            last_batch_finished_at = time.perf_counter()
        if count == 0:
            raise ValueError("training/evaluation loader produced zero batches")
        result = {key: value / count for key, value in totals.items()}
        result.update(
            {
                "batches": float(count),
                "gradient_finite": float(gradient_finite),
                "epoch_complete": float(reached_loader_end),
            }
        )
        return result

    def fit(
        self, train_dataset: Dataset, validation_dataset: Dataset
    ) -> Dict[str, Any]:
        if (
            self.config.pause_after_train_batches is not None
            and self.identity.branch != PHYSICAL_TRAINING_BRANCH
        ):
            raise ValueError("mid-epoch pause requires the physical admission route")
        if self.identity.branch == PHYSICAL_TRAINING_BRANCH:
            from ml_recon.physics_hybrid_v10.training.physical_admission import validate_physical_admission

            current = validate_physical_admission(
                self.physical_admission_path,
                identity=asdict(self.identity),
                config=asdict(self.config),
                device=str(self.device),
            )
            if current["admission_sha"] != self.identity.physical_admission_sha:
                raise ValueError(
                    "physical admission changed before the first training batch"
                )
            self._validate_physical_datasets(
                train_dataset,
                validation_dataset,
                self.physical_admission_evidence,
                self.identity,
                self.config,
            )
        if self.stream_pilot_recipe is not None:
            from ml_recon.physics_hybrid_v10.stream_admission import validate_stream_pilot_datasets

            validate_stream_pilot_datasets(
                train_dataset, validation_dataset, self.stream_pilot_recipe
            )
        self._validate_stream_recipe_for_resume(
            self.output_dir / "checkpoint_latest.pt"
        )
        for name in (
            "training_summary.json",
            "gradient_audit_summary.json",
            "training_completion.json",
            "checkpoint_validation.json",
        ):
            old_summary = self.output_dir / name
            if old_summary.exists():
                if not self.config.resume_checkpoint:
                    raise ValueError("completed output requires explicit resume")
                archive_training_summary(old_summary)
        self.telemetry.begin(self.global_step, self.start_epoch)
        atomic_training_json(
            self.output_dir / "training_completion.json",
            {
                "schema": "V10TrainingCompletionV1",
                "status": "IN_PROGRESS",
                "run_id": self.config.run_id or None,
                "execution_id": self.telemetry.execution_id,
                "source_identity": asdict(self.identity),
                "scientific_gate_evaluated": False,
            },
        )
        try:
            return self._fit_with_observers(train_dataset, validation_dataset)
        except Exception as exc:
            self.telemetry.current["optimizer_attempt_count"] = (
                self.optimizer_attempt_count
            )
            self.telemetry.record_phase("failed", error_kind=type(exc).__name__)
            raise

    def _fit_with_observers(
        self, train_dataset: Dataset, validation_dataset: Dataset
    ) -> Dict[str, Any]:
        initial_global_step = self.global_step
        validation_loader = self._loader(validation_dataset, shuffle=False)
        fixed_validation_batch = _move_to_device(
            next(iter(validation_loader)), self.device
        )
        history = []
        checkpoint_path = None
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loader = self._loader(train_dataset, shuffle=True, epoch=epoch)
            loader_identity = ""
            start_batch_index = 0
            expected_memberships: Sequence[tuple[str, ...]] = ()
            if self.identity.branch == PHYSICAL_TRAINING_BRANCH:
                loader_identity = self._train_loader_identity(train_dataset, epoch)
            if self.resume_batch_cursor:
                if epoch != self.start_epoch:
                    raise ValueError("mid-epoch cursor cannot cross an epoch boundary")
                if self.resume_loader_identity != loader_identity:
                    raise ValueError("checkpoint train-loader ordering identity mismatch")
                start_batch_index = self.resume_batch_cursor
                expected_memberships = tuple(self.resume_committed_memberships)
            else:
                self.resume_loader_identity = loader_identity
            validation_loader = self._loader(
                validation_dataset, shuffle=False, epoch=epoch
            )
            train_metrics = self._run_batches(
                train_loader,
                training=True,
                maximum_batches=self.config.max_train_batches,
                epoch=epoch,
                start_batch_index=start_batch_index,
                expected_memberships=expected_memberships,
                pause_after_batches=self.config.pause_after_train_batches,
            )
            if not bool(train_metrics.pop("epoch_complete")):
                checkpoint_path = self.output_dir / "checkpoint_latest.pt"
                return {
                    "schema": "V10TrainingSummaryV1",
                    "status": "PAUSED_MID_EPOCH",
                    "model_id": self.config.model_id,
                    "execution_id": self.telemetry.execution_id,
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "next_train_batch_index": self.resume_batch_cursor,
                    "checkpoint_path": str(checkpoint_path),
                    "research_only": True,
                    "scientific_gate_evaluated": False,
                }
            self.resume_batch_cursor = 0
            self.resume_loader_identity = ""
            self.resume_committed_memberships = []
            self.resume_epoch_complete = True
            with torch.no_grad():
                validation_metrics = self._run_batches(
                    validation_loader,
                    training=False,
                    maximum_batches=self.config.max_validation_batches,
                    epoch=epoch,
                )
            for observer in self.observers:
                callback = getattr(observer, "on_epoch_end", None)
                if callable(callback):
                    callback(
                        epoch=epoch,
                        train_metrics=dict(train_metrics),
                        validation_metrics=dict(validation_metrics),
                    )
            self.telemetry.record_epoch(epoch, train_metrics, validation_metrics)
            metric = float(validation_metrics["loss"])
            improved = (
                metric < self.best_metric
                if self.config.metric_direction == "min"
                else metric > self.best_metric
            )
            if improved:
                self.best_metric = metric
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            self.scheduler.step()
            entry = {
                "epoch": epoch,
                "global_step": self.global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "improved": improved,
            }
            history.append(entry)
            with (self.output_dir / "metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
            checkpoint_path = self._save_checkpoint(epoch, training_complete=False)
            if self.bad_epochs >= self.config.early_stopping_patience:
                break

        if self.config.run_mode == "smoke" and self.global_step < 2:
            raise RuntimeError("smoke did not execute the required two optimizer steps")
        if self.global_step <= initial_global_step:
            raise RuntimeError("training did not execute an optimizer step")
        self.model.eval()
        with torch.no_grad():
            prediction_before = self.prediction_function(
                self.model, fixed_validation_batch
            )
        # Reload the last incomplete checkpoint before observer validation.
        saved = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model.eval()
        with torch.no_grad():
            prediction_after = self.prediction_function(
                self.model, fixed_validation_batch
            )
        inference_difference = _prediction_max_difference(
            prediction_before, prediction_after
        )
        if inference_difference > 1e-7:
            raise RuntimeError("checkpoint reload changed inference")
        summary = {
            "schema": "V10TrainingSummaryV1",
            "model_id": self.config.model_id,
            "run_id": self.config.run_id or None,
            "execution_id": self.telemetry.execution_id,
            "run_mode": self.config.run_mode,
            "global_step": self.global_step,
            "epochs_completed": len(history),
            "best_metric": self.best_metric,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_model_state_sha": saved["metadata"]["model_state_sha"],
            "reload_inference_max_abs_difference": inference_difference,
            "finite_gradient_audit_passed": all(
                bool(entry["train"]["gradient_finite"]) for entry in history
            ),
            "selection_evidence": False,
            "research_only": True,
            "stream_recipe_sha": self.identity.stream_recipe_sha,
            "stream_recipe_file_sha": self.identity.stream_recipe_file_sha,
            "stream_pilot_admission_sha": self.identity.stream_pilot_admission_sha,
            "history": history,
        }
        # Capture the final committed step before freezing observer summaries.
        for observer in self.observers:
            callback = getattr(observer, "on_training_complete", None)
            if callable(callback):
                callback(final_global_step=self.global_step)
        # Finalize observers and merge their summaries.
        observer_required = self.config.run_mode != "smoke"
        for observer in self.observers:
            try:
                observer_summary = observer.finalize()
                if observer_summary:
                    summary[f"observer_{type(observer).__name__}"] = observer_summary
                observer.write_outputs()
            except Exception as exc:
                if observer_required:
                    raise RuntimeError(
                        f"required observer {type(observer).__name__} "
                        f"failed during finalize: {exc}"
                    ) from exc
        # Non-smoke training requires at least one gradient audit step.
        if observer_required:
            audit_key = "observer_V10GradientAuditObserver"
            audit_summary = summary.get(audit_key, {})
            audited_steps = audit_summary.get("audited_steps", 0)
            if audited_steps < 1:
                raise RuntimeError(
                    f"gradient audit recorded {audited_steps} steps; "
                    f"non-smoke training requires audited_steps >= 1"
                )
            if (
                self.stream_pilot_recipe is not None
                or self.identity.branch == PHYSICAL_TRAINING_BRANCH
            ):
                self._require_complete_output_audit(summary, train_dataset)
        checkpoint_path = self.output_dir / "checkpoint_latest.pt"
        summary["checkpoint_path"] = str(checkpoint_path)
        summary["checkpoint_validation_path"] = str(
            self.output_dir / "checkpoint_validation.json"
        )
        summary["completion_receipt_path"] = str(
            self.output_dir / "training_completion.json"
        )
        summary["telemetry"] = self.telemetry.summary()
        if self._physical_admission_files:
            self._verify_physical_admission_files()
        completion = publish_training_completion(
            output_dir=self.output_dir,
            checkpoint=self._checkpoint(history[-1]["epoch"], training_complete=True),
            config=asdict(self.config),
            source_identity=asdict(self.identity),
            summary=summary,
            execution_id=self.telemetry.execution_id,
            compute_state_sha=_state_dict_sha,
        )
        self.telemetry.record_checkpoint(
            checkpoint_path,
            epoch=history[-1]["epoch"],
            global_step=self.global_step,
            training_complete=True,
            model_state_sha=summary["checkpoint_model_state_sha"],
        )
        self.telemetry.current["completion"] = completion
        self.telemetry.record_phase("completed")
        return summary
