"""Executable, resumable and reproducible V10 PyTorch training runtime."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import re
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml_recon.data_v2_4.version import DATASET_CONTRACT_VERSION
from ml_recon.data_v2_4.hashing import (
    canonical_json_bytes, read_canonical_json, sha256_file, sha256_json,
)

from .checkpoint_schema import validate_checkpoint_envelope
from .checkpoint_storage import (acquire_checkpoint_writer_lease, commit_checkpoint,
                                  publish_checkpoint_commit, read_checkpoint_commit,
                                  resolve_checkpoint_resume, validate_output_root)
from .contracts import V10_MODEL_IDS
from .gradient_audit import V10GradientAuditObserver
from .loss_profiles import resolve_loss_weights
from .sample_scaling import SAMPLE_RUNGS, validate_rung_authorization
from .training_publication import (
    build_training_execution_evidence, publish_training_completion,
    training_execution_evidence_sha256,
)
from .training_identity import config_identity_sha
from .training_telemetry import (
    TELEMETRY_INTERVAL_SECONDS, TELEMETRY_INTERVAL_STEPS, Training_telemetry,
    archive_training_summary, atomic_training_json, training_step_record,
)


RUN_MODES = ("smoke", "pilot", "full")
TRAINING_OUTPUT_NAMES = (
    "training_summary.json", "checkpoint_latest.pt", "metrics.jsonl", "training_steps.jsonl",
    "gradient_audit.jsonl", "gradient_audit_summary.json", "training_progress.json",
    "checkpoint_manifest.json", "config_snapshot.json", "checkpoint_validation.json",
    "training_completion.json",
)


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
        raise ValueError(
            "range mode requires both train and validation range indexes")
    return has_train


@dataclass(frozen=True)
class V10TrainingConfig:
    model_id: str
    output_dir: str
    model_profile: str = ""
    run_mode: str = "smoke"
    loss_weight_profile: str = "default"
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
        if not isinstance(self.model_profile, str) or len(self.model_profile) > 64:
            raise ValueError("model_profile must be a bounded string")
        if self.run_mode not in RUN_MODES:
            raise ValueError(f"run_mode must be one of {RUN_MODES}")
        resolve_loss_weights(self.loss_weight_profile)
        if self.batch_size <= 0 or self.epochs <= 0 or self.learning_rate <= 0.0:
            raise ValueError("batch size, epochs, and learning rate must be positive")
        if self.amp_initial_scale <= 0.0:
            raise ValueError("AMP initial scale must be positive")
        if (type(self.telemetry_enabled) is not bool
                or type(self.telemetry_interval_steps) is not int
                or self.telemetry_interval_steps < 1
                or isinstance(self.telemetry_interval_seconds, bool)
                or not isinstance(self.telemetry_interval_seconds, (int, float))
                or not math.isfinite(self.telemetry_interval_seconds)
                or self.telemetry_interval_seconds <= 0):
            raise ValueError("telemetry cadence must be finite and positive")
        if self.track_id not in ("track_a", "track_b", "unspecified"):
            raise ValueError("track_id must be track_a, track_b or unspecified")
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{0,128}", self.run_id):
            raise ValueError("run_id must be a bounded opaque identifier")
        if self.metric_direction not in ("min", "max"):
            raise ValueError("metric_direction must be min or max")
        if self.run_mode == "smoke" and (self.max_train_batches or 0) < 2:
            raise ValueError("V10 smoke requires at least two training batches")
        if self.sample_count in SAMPLE_RUNGS:
            validate_rung_authorization(
                self.completed_sample_rungs, self.sample_count,
                training_authorized=self.training_authorized,
                phase_1_to_7_gate_sha=self.phase_1_to_7_gate_sha,
                phase_8_gate_sha=self.phase_8_gate_sha)
        if self.global_sample_count:
            if self.global_sample_count not in (1_000, *SAMPLE_RUNGS):
                raise ValueError("global sample count is not registered")
            if len(self.rung_plan_sha) != 64:
                raise ValueError("global sample count requires a rung plan SHA-256")
            if self.global_sample_count in SAMPLE_RUNGS:
                validate_rung_authorization(
                    self.completed_sample_rungs, self.global_sample_count,
                    training_authorized=self.training_authorized,
                    phase_1_to_7_gate_sha=self.phase_1_to_7_gate_sha,
                    phase_8_gate_sha=self.phase_8_gate_sha)
        effective_rung = self.global_sample_count or self.sample_count
        if self.run_mode == "full" and effective_rung != SAMPLE_RUNGS[-1]:
            raise ValueError("full training is reserved for the authorized 2M sample rung")
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

    def validate(self) -> None:
        from .checkpoint_schema import validate_input_delivery
        validate_input_delivery(asdict(self))
        if self.branch != "v10" and self.branch != "codex/v10_lowdisk_training_prep":
            raise ValueError("formal training identity must bind v10 or the authorized V10 low-disk preparation branch")
        digests = (
            self.source_commit, self.source_tree_sha, self.dirty_patch_sha, self.v23_manifest_sha,
            self.synthetic_manifest_sha, self.measurement_contract_sha,
            self.shot_budget_contract_sha, self.rho8_sha, self.config_sha,
            self.split_hash, *self.dataset_hashes.values(),
        )
        if any(len(value) != 40 and len(value) != 64 for value in digests):
            raise ValueError("training identity hashes must be Git SHA-1 or SHA-256 digests")
        # The generic runtime remains usable by isolated unit-test models. Formal
        # V10 entrypoints always populate all five V2.4 fields below.
        v24_values = (self.dataset_contract_sha, self.dataset_build_id,
                      self.dataset_index_sha, self.dataset_cell_id)
        if any(v24_values) and (self.dataset_contract_version != DATASET_CONTRACT_VERSION
                               or len(self.dataset_contract_sha) != 64
                               or len(self.dataset_index_sha) != 64
                               or not self.dataset_build_id or not self.dataset_cell_id):
            raise ValueError("partial V2.4 training identity is forbidden")


class V10DictTensorDataset(Dataset):
    """Minimal map-style tensor dataset used by executable training entrypoints."""

    def __init__(self, tensors: Mapping[str, torch.Tensor], sample_ids: Sequence[str]):
        if not tensors:
            raise ValueError("tensor dataset cannot be empty")
        lengths = {int(tensor.shape[0]) for tensor in tensors.values()}
        if len(lengths) != 1 or next(iter(lengths)) != len(sample_ids):
            raise ValueError("all tensors and sample IDs must share their first dimension")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample IDs must be unique")
        self.tensors = dict(tensors)
        self.sample_ids = tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {**{key: value[index] for key, value in self.tensors.items()},
                "sample_id": self.sample_ids[index]}


def stable_split_for_group(group_id: str, global_seed: int = 20261001) -> str:
    digest = hashlib.sha256(f"{global_seed}|{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return "train" if bucket < 8000 else ("validation" if bucket < 9000 else "test")


def grouped_split(sample_ids: Sequence[str], group_ids: Sequence[str],
                  global_seed: int = 20261001) -> Dict[str, tuple[str, ...]]:
    if len(sample_ids) != len(group_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample/group identities must align and sample IDs must be unique")
    result: Dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    group_assignment: Dict[str, str] = {}
    for sample_id, group_id in zip(sample_ids, group_ids):
        split = group_assignment.setdefault(
            group_id, stable_split_for_group(group_id, global_seed))
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
    if any(value is None for value in (mode_input, measurement_input, cross, encoder, moe)):
        return {
            "architecture_class": type(model).__name__,
            "state_dict_schema": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(model.state_dict().items())
            },
        }
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
        "csm_node_context_dim": int(node_projection.in_features) if node_projection else 0,
        "csm_global_context_dim": int(global_projection.in_features) if global_projection else 0,
    }


def _model_build_spec_identity(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the canonical producer identity for an architecture specification."""
    payload = canonical_json_bytes(dict(spec))
    return {
        "schema": "V10ModelBuildSpecIdentityV1",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _validate_model_profile_binding(model: torch.nn.Module,
                                    config: V10TrainingConfig) -> None:
    """Bind every explicitly selected config profile to the constructed model."""
    if not config.model_profile:
        return
    model_profile = getattr(model, "profile_name", None)
    if model_profile != config.model_profile:
        raise ValueError(
            "training model_profile does not match constructed model profile: "
            f"config={config.model_profile!r}, model={model_profile!r}")

def _prediction_max_difference(before: Any, after: Any) -> float:
    if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
        return float((before.detach().cpu() - after.detach().cpu()).abs().max().item())
    if isinstance(before, dict) and isinstance(after, dict):
        common = sorted(set(before) & set(after))
        values = [_prediction_max_difference(before[key], after[key]) for key in common
                  if isinstance(before[key], (torch.Tensor, dict))]
        return max(values, default=0.0)
    raise TypeError("prediction function must return tensors or nested dictionaries of tensors")


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

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: V10TrainingConfig,
        identity: V10TrainingIdentity,
        step_function: Callable[[torch.nn.Module, Dict[str, Any]], Dict[str, torch.Tensor]],
        prediction_function: Callable[[torch.nn.Module, Dict[str, Any]], Any],
        device: str = "cpu",
        batch_sampler_factory: Optional[
            Callable[[Dataset, int, bool], Any]] = None,
        observers: Optional[Sequence[V10TrainingObserver]] = None,
        stream_pilot_admission_path: str | Path | None = None,
        defer_initialization: bool = False,
        stream_migration_proof: object = None,
    ):
        config.validate()
        identity.validate()
        if (identity.input_delivery == "stream_only" and config.run_mode == "pilot"
                and identity.config_sha != config_identity_sha(config)):
            raise ValueError("training identity config_sha does not match the actual V10 training config")
        _validate_model_profile_binding(model, config)
        self.stream_pilot_recipe = None
        self.stream_pilot_evidence = {}
        self.stream_pilot_admission_path = stream_pilot_admission_path
        if identity.input_delivery == "stream_only" and config.run_mode != "smoke":
            from .stream_admission import validate_core_stream_pilot
            self.stream_pilot_recipe = validate_core_stream_pilot(
                admission_path=stream_pilot_admission_path,
                config=asdict(config), identity=asdict(identity), device=device,
                model=model,
                migration_proof=stream_migration_proof)
        self._stream_migration_proof = (
            stream_migration_proof if self.stream_pilot_recipe is not None else None)
        if not callable(getattr(torch.optim.Optimizer, "register_step_post_hook", None)):
            raise RuntimeError("training requires PyTorch optimizer post-step hook support")
        self.model = model
        self.config = config
        self.identity = identity
        self.step_function = step_function
        self.prediction_function = prediction_function
        self.batch_sampler_factory = batch_sampler_factory
        self.observers = list(observers) if observers else []
        self.defer_initialization = bool(defer_initialization)
        if config.run_mode != "smoke":
            required = [observer for observer in self.observers
                        if isinstance(observer, V10GradientAuditObserver)]
            if not required or any(
                    not observer.config.enabled or any(not callable(getattr(observer, method, None))
                        for method in ("on_before_backward", "finalize", "write_outputs"))
                    for observer in required):
                raise RuntimeError("required gradient audit observer is unavailable or disabled")
            model_parameters = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
            if any(not observer.shared_params or any(id(parameter) not in model_parameters
                                                      for parameter in observer.shared_params)
                   for observer in required):
                raise RuntimeError("required gradient observer is not bound to trainable model parameters")
        self.device = torch.device(device)
        self.output_dir = Path(config.output_dir)
        self.amp_active = bool(config.amp_enabled and self.device.type == "cuda")
        self._initialized = False
        self._resume_checkpoint_payload: Optional[Dict[str, Any]] = None
        self._resume_checkpoint_was_complete = False
        self._writer_lease = None
        if not self.defer_initialization:
            self._initialize_training_state()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        lease = self._writer_lease
        if lease is not None:
            lease.release()
            self._writer_lease = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _initialize_training_state(self) -> None:
        """Fence resume validation and resource construction with a root lease."""
        if self._stream_migration_proof is not None:
            from .stream_admission import validate_core_stream_pilot
            validate_core_stream_pilot(
                admission_path=self.stream_pilot_admission_path,
                config=asdict(self.config), identity=asdict(self.identity),
                device=str(self.device), model=self.model,
                migration_proof=self._stream_migration_proof)
        if self._initialized:
            return
        validate_output_root(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writer_lease = acquire_checkpoint_writer_lease(
            self.output_dir, "initialization_" + str(id(self)))
        try:
            self._writer_lease.assert_active()
            self._initialize_training_state_unlocked()
        except Exception:
            self._writer_lease.release()
            self._writer_lease = None
            raise

    def _initialize_training_state_unlocked(self) -> None:
        """Create training resources when the caller explicitly starts execution."""
        if self._stream_migration_proof is not None:
            from .stream_admission import validate_core_stream_pilot
            validate_core_stream_pilot(
                admission_path=self.stream_pilot_admission_path,
                config=asdict(self.config), identity=asdict(self.identity),
                device=str(self.device), model=self.model,
                migration_proof=self._stream_migration_proof)
        if self._initialized:
            return
        if not self.config.resume_checkpoint and any((self.output_dir / name).exists() for name in TRAINING_OUTPUT_NAMES):
            raise ValueError("existing training output requires a new directory or explicit resume")
        if self.config.resume_checkpoint:
            self._resume_checkpoint_payload = self._read_resume_checkpoint(
                Path(self.config.resume_checkpoint))
            self._resume_checkpoint_was_complete = bool(
                self._resume_checkpoint_payload.get("metadata", {}).get("training_complete"))
        if self.stream_pilot_recipe is not None:
            from .stream_admission import preserve_stream_pilot_evidence
            self._writer_lease.assert_active()
            self._writer_lease.assert_rooted_path("admission_evidence", allow_missing=True)
            self.stream_pilot_evidence = preserve_stream_pilot_evidence(
                self.stream_pilot_admission_path, self.identity.stream_pilot_admission_sha,
                self.output_dir)
            self._writer_lease.assert_active()
            for evidence in self.stream_pilot_evidence.values():
                rooted = self._writer_lease.assert_rooted_path(evidence["path"])
                if sha256_file(rooted) != evidence["sha256"]:
                    raise RuntimeError("stream admission evidence identity changed after preservation")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.config.epochs, 1))
        amp_active = self.config.amp_enabled and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_active, init_scale=self.config.amp_initial_scale)
        self.amp_active = amp_active
        self.start_epoch = 0
        self.global_step = 0
        self.amp_overflow_count = 0
        self.optimizer_attempt_count = 0
        self.overflow_counter_scope = "SINCE_RUN_START"
        self._optimizer_step_handle = self.optimizer.register_step_post_hook(self._record_optimizer_step)
        self.best_metric = float("inf") if self.config.metric_direction == "min" else -float("inf")
        self.bad_epochs = 0
        self._seed_all(self.config.global_seed)
        if self.config.resume_checkpoint:
            self._load_checkpoint(Path(self.config.resume_checkpoint))
        self.telemetry = Training_telemetry(
            self.output_dir, identity=asdict(self.identity), model_id=self.config.model_id,
            track_id=self.config.track_id, run_id=self.config.run_id, enabled=self.config.telemetry_enabled,
            interval_steps=self.config.telemetry_interval_steps,
            interval_seconds=self.config.telemetry_interval_seconds)
        registered_cell_ids = tuple(sorted(filter(None, self.identity.dataset_cell_id.split(","))))
        for observer in self.observers:
            identity_bind = getattr(observer, "bind_training_identity", None)
            if callable(identity_bind):
                identity_bind(identity=asdict(self.identity), run_id=self.config.run_id,
                              execution_id=self.telemetry.execution_id)
            bind = getattr(observer, "bind_run_context", None)
            if isinstance(observer, V10GradientAuditObserver) and callable(bind):
                bind(run_id=self.config.run_id, execution_id=self.telemetry.execution_id,
                     initial_attempt_index=self.optimizer_attempt_count)
            configure_cells = getattr(observer, "configure_registered_cells", None)
            if callable(configure_cells):
                configure_cells(registered_cell_ids)
        self._initialized = True

    def _record_optimizer_step(self, optimizer, args, kwargs) -> None:
        # A GradScaler attempt may skip optimizer.step(); count committed calls.
        self.global_step += 1

    def _read_resume_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Read and bind resume metadata before creating optimizer resources."""
        self._validate_stream_recipe_for_resume(path)
        resolved_path = self._validate_resume_artifact_binding(path)
        if not isinstance(resolved_path, Path):
            resolved_path = path
        checkpoint = self._load_verified_checkpoint(resolved_path)
        validate_checkpoint_envelope(checkpoint["metadata"])
        if checkpoint["metadata"]["model_id"] != self.config.model_id:
            raise ValueError("resume checkpoint model_id mismatch")
        from .classifier_n_full import V10ClassifierNFull
        requires_model_spec = (
            self.identity.input_delivery == "stream_only"
            or bool(self.config.model_profile)
            or type(self.model) is V10ClassifierNFull
        )
        if "model_build_spec" in checkpoint or requires_model_spec:
            stored_model_build_spec = checkpoint.get("model_build_spec")
            if not isinstance(stored_model_build_spec, Mapping) or not stored_model_build_spec:
                raise ValueError("resume checkpoint model architecture metadata invalid")
            current_model_build_spec = _model_build_spec(self.model)
            if stored_model_build_spec != current_model_build_spec:
                raise ValueError("resume checkpoint model architecture mismatch")
        return checkpoint

    def _load_verified_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Hash and unpickle through one retained descriptor before fstat recheck."""
        descriptor = read_checkpoint_commit(self.output_dir)
        expected_sha = None
        if descriptor is not None:
            expected_sha = descriptor["checkpoint"]["sha256"]
        else:
            manifest = json.loads((self.output_dir / "checkpoint_manifest.json").read_text())
            expected_sha = manifest.get("sha256")
        open_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(str(path), open_flags)
        with os.fdopen(file_descriptor, "rb", closefd=True) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("checkpoint descriptor is not a regular single-link file")
            digest = hashlib.sha256()
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if digest.hexdigest() != expected_sha:
                raise ValueError("checkpoint changed during descriptor attestation")
            stream.seek(0)
            loaded = torch.load(stream, map_location=self.device, weights_only=False)
            after = os.fstat(stream.fileno())
            if (before.st_ino != after.st_ino or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns):
                raise ValueError("checkpoint changed during verified load")
        if descriptor is not None:
            after_descriptor = read_checkpoint_commit(self.output_dir)
            if (after_descriptor is None
                    or after_descriptor.get("generation_id") != descriptor.get("generation_id")
                    or after_descriptor.get("checkpoint") != descriptor.get("checkpoint")):
                raise ValueError("checkpoint authority changed during verified load")
        return loaded

    def _validate_resume_artifact_binding(self, path: Path) -> Path:
        """Authenticate the producer-owned resume artifact before unpickling it.

        A resume checkpoint contains optimizer, scheduler, and RNG objects and
        therefore cannot be treated as a weights-only file.  The telemetry
        manifest is the producer-owned receipt for interrupted runs; require it
        to bind the exact output-root artifact and the current execution
        identity before allowing ``torch.load(weights_only=False)``.
        """
        output_root = self.output_dir.resolve()
        checkpoint_path = Path(path)
        expected_source_identity = asdict(self.identity)
        expected_model_build_spec_identity = _model_build_spec_identity(
            _model_build_spec(self.model))
        expected_profile = self.config.model_profile or None

        def validate_manifest_identity(manifest: Mapping[str, Any], *, legacy: bool = False) -> None:
            if manifest.get("config_sha") != self.identity.config_sha:
                raise ValueError("resume checkpoint config identity mismatch")
            if manifest.get("source_identity") != expected_source_identity:
                raise ValueError("resume checkpoint source identity mismatch")
            if manifest.get("run_id") != (self.config.run_id or None):
                raise ValueError("resume checkpoint run identity mismatch")
            if manifest.get("model_id") != self.config.model_id:
                raise ValueError("resume checkpoint model_id mismatch")
            if ("model_profile" in manifest
                    and manifest.get("model_profile") != expected_profile):
                raise ValueError("resume checkpoint model profile mismatch")
            if ("model_profile" not in manifest and (not legacy or expected_profile)):
                raise ValueError("resume checkpoint model profile is unavailable")
            if ("model_build_spec_identity" in manifest
                    and manifest.get("model_build_spec_identity") != expected_model_build_spec_identity):
                raise ValueError("resume checkpoint model architecture mismatch (identity mismatch)")
            if ("model_build_spec_identity" not in manifest
                    and (not legacy or expected_profile)):
                raise ValueError("resume checkpoint model architecture identity is unavailable")

        descriptor = read_checkpoint_commit(output_root)
        if descriptor is not None:
            if checkpoint_path.name != "checkpoint_latest.pt" or checkpoint_path.resolve().parent != output_root:
                raise ValueError("resume checkpoint must be the output root checkpoint_latest.pt")
            resolved = resolve_checkpoint_resume(output_root, checkpoint_path)
            manifest_path = resolved.parent / "checkpoint_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"),
                                      object_pairs_hook=self._reject_duplicate_json_keys,
                                      parse_constant=self._reject_nonfinite_json)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("committed checkpoint manifest is malformed") from exc
            if (manifest.get("generation_id") != descriptor.get("generation_id")
                    or manifest.get("execution_id") != descriptor.get("execution_id")
                    or manifest.get("checkpoint_sha256") != descriptor["checkpoint"].get("sha256")):
                raise ValueError("checkpoint generation and descriptor identity mismatch")
            validate_manifest_identity(manifest)
            for key, expected in (
                ("source_identity", expected_source_identity),
                ("config_sha", self.identity.config_sha),
                ("run_id", self.config.run_id or None),
                ("model_id", self.config.model_id),
                ("model_profile", expected_profile),
                ("model_build_spec_identity", expected_model_build_spec_identity),
            ):
                if descriptor.get(key) != expected:
                    raise ValueError(f"resume checkpoint descriptor {key} mismatch")
            return resolved
        if checkpoint_path.name != "checkpoint_latest.pt":
            raise ValueError("resume checkpoint must be the output root checkpoint_latest.pt")
        if checkpoint_path.is_symlink():
            raise ValueError("resume checkpoint must not be a symlink")
        try:
            resolved_checkpoint = checkpoint_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError("resume checkpoint path is unavailable") from exc
        if resolved_checkpoint.parent != output_root:
            raise ValueError("resume checkpoint path escaped the output root")

        def _regular_single_link(candidate: Path, label: str) -> os.stat_result:
            try:
                info = candidate.lstat()
            except OSError as exc:
                raise ValueError(f"resume {label} is unavailable") from exc
            if (not stat.S_ISREG(info.st_mode) or candidate.is_symlink()
                    or info.st_nlink != 1):
                raise ValueError(f"resume {label} must be a regular single-link file")
            return info

        checkpoint_info = _regular_single_link(resolved_checkpoint, "checkpoint")
        manifest_path = output_root / "checkpoint_manifest.json"
        manifest_info = _regular_single_link(manifest_path, "checkpoint manifest")
        if manifest_info.st_size > 2 * 1024 * 1024:
            raise ValueError("resume checkpoint manifest is too large")
        try:
            raw_manifest = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(raw_manifest, object_pairs_hook=self._reject_duplicate_json_keys,
                                  parse_constant=self._reject_nonfinite_json)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("resume checkpoint manifest is malformed") from exc
        if (not isinstance(manifest, dict)
                or manifest.get("schema") not in {
                    "V10TrainingCheckpointManifestV1", "V10TrainingCheckpointManifestV2"}):
            raise ValueError("resume checkpoint manifest schema is invalid")

        if manifest.get("source_identity") != expected_source_identity:
            source_identity = manifest.get("source_identity")
            if (isinstance(source_identity, dict)
                    and source_identity.get("config_sha") != self.identity.config_sha):
                raise ValueError("resume checkpoint model architecture mismatch (config identity mismatch)")
            raise ValueError("resume checkpoint source identity mismatch")
        if manifest.get("model_id") != self.config.model_id:
            raise ValueError("resume checkpoint model_id mismatch")
        if manifest.get("path") != str(resolved_checkpoint):
            raise ValueError("resume checkpoint manifest path mismatch")
        if manifest.get("sha256") != sha256_file(resolved_checkpoint):
            raise ValueError("resume checkpoint manifest SHA mismatch")
        if manifest.get("bytes") != checkpoint_info.st_size:
            raise ValueError("resume checkpoint manifest byte-size mismatch")
        validate_manifest_identity(manifest, legacy=True)
        # Detect a replacement between the attestation read and the eventual
        # unpickle.  A changed inode/size/hash is fail-closed.
        try:
            after_info = resolved_checkpoint.lstat()
        except OSError as exc:
            raise ValueError("resume checkpoint disappeared after attestation") from exc
        if (after_info.st_ino != checkpoint_info.st_ino
                or after_info.st_size != checkpoint_info.st_size
                or after_info.st_mtime_ns != checkpoint_info.st_mtime_ns):
            raise ValueError("resume checkpoint changed during attestation")
        return resolved_checkpoint

    @staticmethod
    def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    @staticmethod
    def _reject_nonfinite_json(value: str) -> None:
        raise ValueError(f"nonfinite JSON value: {value}")

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _loader(self, dataset: Dataset, *, shuffle: bool, epoch: int = 0) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(
            self.config.global_seed + (17 if shuffle else 29) + 1_000_003 * int(epoch))
        worker_arguments: Dict[str, Any] = {"num_workers": self.config.num_workers, "generator": generator}
        if self.identity.input_delivery == "stream_only" and self.config.num_workers:
            from .stream_admission import STREAM_WORKER_TIMEOUT_SECONDS
            from .stream_admission import StreamSourceIdentityGuard
            # Spawn CPU stream workers without inheriting initialized CUDA state.
            worker_arguments["multiprocessing_context"] = "spawn"
            worker_arguments["timeout"] = STREAM_WORKER_TIMEOUT_SECONDS
            worker_arguments["worker_init_fn"] = StreamSourceIdentityGuard({
                "branch": self.identity.branch,
                "source_commit": self.identity.source_commit,
                "source_tree_sha": self.identity.source_tree_sha,
            })
        if self.batch_sampler_factory is not None:
            batch_sampler = self.batch_sampler_factory(dataset, epoch, shuffle)
            return DataLoader(
                dataset, batch_sampler=batch_sampler,
                **worker_arguments)
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=shuffle,
            **worker_arguments,
            drop_last=False)

    def _rng_state(self) -> Dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
        }

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state["torch_cuda"]:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _checkpoint(self, epoch: int, training_complete: bool) -> Dict[str, Any]:
        from .checkpoint_storage import cpu_checkpoint_snapshot
        self._validate_stream_recipe_for_resume(self.output_dir / "checkpoint_latest.pt")
        model_sha = _state_dict_sha(self.model.state_dict())
        model_build_spec = _model_build_spec(self.model)
        model_build_spec_identity = _model_build_spec_identity(model_build_spec)
        metadata = {
            "schema": "V10CheckpointV1",
            "stream_pilot_evidence": self.stream_pilot_evidence,
            "model_id": self.config.model_id,
            "model_profile": self.config.model_profile or None,
            "model_build_spec_identity": model_build_spec_identity,
            "run_id": self.config.run_id or None,
            "execution_id": self.telemetry.execution_id if hasattr(self, "telemetry") else "pending",
            "source_identity": asdict(self.identity),
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
            "exact_parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
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
        validate_checkpoint_envelope(metadata)
        return cpu_checkpoint_snapshot({
            "metadata": metadata,
            "model_build_spec": model_build_spec,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "rng_state": self._rng_state(),
            "trainer_state": {"bad_epochs": self.bad_epochs,
                              "amp_overflow_count": self.amp_overflow_count,
                              "optimizer_attempt_count": self.optimizer_attempt_count,
                              "overflow_counter_scope": self.overflow_counter_scope},
        })

    def _save_checkpoint(self, epoch: int, training_complete: bool,
                         metric_record: Dict[str, Any] | None = None) -> Path:
        if training_complete:
            raise ValueError("complete checkpoints require validated completion publication")
        checkpoint = self._checkpoint(epoch, training_complete)
        commit = commit_checkpoint(self.output_dir, checkpoint,
                                   execution_id=self.telemetry.execution_id,
                                   epoch=epoch, global_step=self.global_step,
                                   writer_lease=self._writer_lease,
                                   metric_record=metric_record)
        self._last_checkpoint_commit = commit
        destination = self.output_dir / "checkpoint_latest.pt"
        self.telemetry.record_checkpoint(
            destination, epoch=epoch, global_step=self.global_step,
            training_complete=training_complete,
            model_state_sha=checkpoint["metadata"]["model_state_sha"], commit=commit)
        return destination

    def _write_disabled_telemetry_checkpoint_manifest(
            self, path: Path, *, epoch: int, global_step: int,
            training_complete: bool, model_state_sha: str) -> None:
        """Keep resume attestation available for bounded telemetry-off fixtures."""
        model_build_spec = _model_build_spec(self.model)
        manifest = {
            "schema": "V10TrainingCheckpointManifestV1",
            "observed_at": "telemetry_disabled",
            "run_id": self.config.run_id or None,
            "model_id": self.config.model_id,
            "model_profile": self.config.model_profile or None,
            "model_build_spec_identity": _model_build_spec_identity(model_build_spec),
            "execution_id": self.telemetry.execution_id,
            "source_identity": asdict(self.identity),
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "epoch": epoch,
            "global_step": global_step,
            "training_complete": training_complete,
            "model_state_sha": model_state_sha,
            "authority": {},
        }
        atomic_training_json(self.output_dir / "checkpoint_manifest.json", manifest)

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = self._resume_checkpoint_payload
        if checkpoint is None:
            checkpoint = self._read_resume_checkpoint(path)
        self._resume_checkpoint_payload = None
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
        mismatches = [key for key, value in expected_identity.items()
                      if checkpoint["metadata"].get(key) != value]
        for key, default in (("input_delivery", "materialized"),
                             ("dataset_index_kind", "materialized"), ("stream_recipe_sha", ""),
                             ("stream_recipe_file_sha", ""),
                             ("stream_pilot_admission_sha", "")):
            if checkpoint["metadata"].get(key, default) != getattr(self.identity, key):
                mismatches.append(key)
        if mismatches:
            raise ValueError(f"resume checkpoint identity mismatch: {mismatches}")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if _state_dict_sha(self.model.state_dict()) != checkpoint["metadata"]["model_state_sha"]:
            raise ValueError("resume checkpoint model-state hash mismatch")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self._restore_rng_state(checkpoint["rng_state"])
        self.start_epoch = int(checkpoint["metadata"]["epoch"]) + 1
        self.global_step = int(checkpoint["metadata"]["global_step"])
        self.best_metric = float(checkpoint["metadata"]["best_metric"])
        self.bad_epochs = int(checkpoint["trainer_state"]["bad_epochs"])
        for key in ("amp_overflow_count", "optimizer_attempt_count"):
            value = checkpoint["trainer_state"].get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError("checkpoint observation counters must be nonnegative integers")
        self.amp_overflow_count = checkpoint["trainer_state"].get("amp_overflow_count", 0)
        self.optimizer_attempt_count = checkpoint["trainer_state"].get("optimizer_attempt_count", 0)
        self.overflow_counter_scope = checkpoint["trainer_state"].get(
            "overflow_counter_scope", "SINCE_RESUME_MISSING_HISTORICAL_BASELINE")

    def _validate_stream_recipe_for_resume(self, checkpoint_path: Path) -> None:
        """Make direct V10Trainer resume reject swapped stream recipe bytes too."""
        if self.identity.input_delivery != "stream_only":
            return
        recipe_path = checkpoint_path.parent / "stream_recipe.json"
        if not recipe_path.is_file():
            raise ValueError("stream checkpoint resume requires sibling stream_recipe.json")
        if sha256_file(recipe_path) != self.identity.stream_recipe_file_sha:
            raise ValueError("stream checkpoint recipe file SHA mismatch")
        try:
            recipe = read_canonical_json(recipe_path)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("stream checkpoint recipe is malformed or noncanonical") from exc
        if not isinstance(recipe, dict):
            raise ValueError("stream checkpoint recipe must be a JSON object")
        declared_sha = recipe.get("recipe_sha256")
        payload = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
        if (declared_sha != sha256_json(payload)
                or declared_sha != self.identity.stream_recipe_sha):
            raise ValueError("stream checkpoint canonical recipe payload SHA mismatch")

    def _run_batches(self, loader: DataLoader, *, training: bool,
                     maximum_batches: Optional[int],
                     epoch: int = 0) -> Dict[str, float]:
        self.model.train(training)
        totals: Dict[str, float] = {}
        count = 0
        gradient_finite = True
        last_batch_finished_at = time.perf_counter()
        if not training:
            self.telemetry.record_phase("evaluating")
        for batch_index, raw_batch in enumerate(loader):
            received_at = time.perf_counter()
            generator_wait = received_at - last_batch_finished_at
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
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
                    arguments = dict(model=self.model, batch=batch, losses=losses,
                                     global_step=self.global_step + 1, epoch=epoch)
                    if isinstance(observer, V10GradientAuditObserver):
                        arguments["attempt_index"] = self.optimizer_attempt_count
                    observer.on_before_backward(**arguments)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradients = [parameter.grad for parameter in self.model.parameters()
                             if parameter.grad is not None]
                if not gradients:
                    raise RuntimeError("training loss produced no gradients")
                gradient_finite = gradient_finite and all(
                    bool(torch.all(torch.isfinite(gradient))) for gradient in gradients)
                if not gradient_finite:
                    invalid_parameters = [
                        name for name, parameter in self.model.named_parameters()
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
                        f"losses={scalar_losses}")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm)
                previous_step = self.global_step
                previous_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                current_scale = self.scaler.get_scale()
                self.amp_overflow_count += int(current_scale < previous_scale)
                scalar_losses = {key: float(value.detach().cpu()) for key, value in losses.items()
                                 if isinstance(value, torch.Tensor) and value.ndim == 0}
                record = training_step_record(
                    batch=batch, scalar_losses=scalar_losses, epoch=epoch,
                    global_step=self.global_step,
                    optimizer_step_executed=self.global_step > previous_step,
                    gradient_norm=float(gradient_norm),
                    learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    amp_active=self.amp_active, amp_scale=current_scale,
                    overflow_count=self.amp_overflow_count,
                    attempt_index=self.optimizer_attempt_count, execution_id=self.telemetry.execution_id,
                    amp_scale_before=previous_scale, overflow_counter_scope=self.overflow_counter_scope,
                    step_seconds=time.perf_counter() - received_at,
                    generator_wait_seconds=generator_wait, device=str(self.device))
                raw_cell_ids = batch.get("cell_id")
                if isinstance(raw_cell_ids, (list, tuple)):
                    record["cell_ids"] = sorted({cell_id for cell_id in raw_cell_ids
                                                  if isinstance(cell_id, str) and cell_id})
                elif isinstance(raw_cell_ids, str):
                    record["cell_ids"] = [raw_cell_ids]
                else:
                    record["cell_ids"] = []
                if self.device.type == "cuda":
                    record["gpu_memory"] = {
                        "allocated_bytes": torch.cuda.memory_allocated(self.device),
                        "reserved_bytes": torch.cuda.memory_reserved(self.device),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}
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
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            count += 1
            last_batch_finished_at = time.perf_counter()
        if count == 0:
            raise ValueError("training/evaluation loader produced zero batches")
        result = {key: value / count for key, value in totals.items()}
        result.update({"batches": float(count), "gradient_finite": float(gradient_finite)})
        return result

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset) -> Dict[str, Any]:
        self._validate_actual_dataset_delivery(train_dataset, validation_dataset)
        self._initialize_training_state()
        if self._writer_lease is None:
            raise RuntimeError("training writer lease was not retained from initialization")
        try:
            return self._fit_after_lease(train_dataset, validation_dataset)
        except Exception as exc:
            if getattr(self, "telemetry", None) is not None:
                self.telemetry.current["optimizer_attempt_count"] = self.optimizer_attempt_count
                self.telemetry.record_phase("failed", error_kind=type(exc).__name__)
            raise
        finally:
            self.close()

    @staticmethod
    def _stream_dataset_kind(dataset: Dataset) -> str:
        """Classify every V2.4 multi-cell dataset by its member delivery."""
        from ml_recon.data_v2_4.multi_cell import V24MultiCellRangeDataset
        from ml_recon.data_v2_4.streaming import V24_stream_dataset
        if not isinstance(dataset, V24MultiCellRangeDataset):
            return "nonstream"
        members = tuple(dataset.datasets)
        if not members:
            return "nonstream"
        stream_members = [isinstance(member, V24_stream_dataset) for member in members]
        if all(stream_members):
            return "all_stream"
        if any(stream_members):
            return "mixed"
        return "nonstream"

    def _validate_actual_dataset_delivery(self, train_dataset: Dataset,
                                          validation_dataset: Dataset) -> None:
        train_kind = self._stream_dataset_kind(train_dataset)
        validation_kind = self._stream_dataset_kind(validation_dataset)
        has_stream = train_kind != "nonstream" or validation_kind != "nonstream"
        actual_stream = train_kind == validation_kind == "all_stream"
        if has_stream and not actual_stream:
            raise ValueError("mixed or mismatched V2.4 stream delivery is not admissible")
        if actual_stream and self.identity.input_delivery != "stream_only":
            raise ValueError("actual V2.4 stream datasets require stream_only identity")
        if not actual_stream and self.identity.input_delivery == "stream_only":
            raise ValueError("stream_only identity requires actual V2.4 stream datasets")
        if not actual_stream:
            return
        # Every direct stream smoke must bind the current config identity.  This
        # prevents a stale receipt/config snapshot from being paired with a new
        # virtual recipe while retaining the registered smoke profiles.
        if (self.config.run_mode == "smoke" and self.stream_pilot_recipe is None
                and self.identity.config_sha != config_identity_sha(self.config)):
            raise ValueError("stream dataset config identity does not match the actual V10 training config")
        from ml_recon.data_v2_4.rung_plan import plan_registered_rung
        from ml_recon.data_v2_4.streaming import classifier_stream_datasets
        if (self.config.run_mode == "smoke"
                and (self.config.model_profile in self._registered_smoke_profiles()
                     or self.config.sample_count
                     or self.config.global_sample_count)):
            if (self.config.model_profile not in self._registered_smoke_profiles()
                    or self.config.epochs != 1
                    or self.config.batch_size != 1
                    or self.config.max_train_batches != 4
                    or self.config.max_validation_batches != 4
                    or self.config.sample_count != 8):
                raise ValueError("stream smoke must use the registered bounded smoke contract")
            from .classifier_n_full import V10ClassifierNFull
            if type(self.model) is not V10ClassifierNFull:
                raise ValueError("stream smoke requires the canonical V10ClassifierNFull model")
            expected_model_spec = {
                "architecture_class": "V10ClassifierNFull",
                "profile": self.config.model_profile,
                "descriptor_dim": 15, "mode_feature_dim": 12,
                "certificate_dim": 10,
            }
            if _model_build_spec(self.model) != expected_model_spec:
                raise ValueError("stream smoke model build spec does not match registered profile")
            plan = plan_registered_rung("small_n_classifier", 8, 4)
            _, _, recipe = classifier_stream_datasets(
                plan, global_seed=self.config.global_seed)
            self._validate_stream_identity_recipe(recipe)
            from .stream_admission import validate_stream_pilot_datasets
            validate_stream_pilot_datasets(train_dataset, validation_dataset, recipe)
        elif self.stream_pilot_recipe is None:
            raise ValueError("direct stream pilot requires admitted C-S recipe")

    @staticmethod
    def _registered_smoke_profiles() -> frozenset[str]:
        from .classifier_n_full import CLASSIFIER_PROFILES
        return frozenset(CLASSIFIER_PROFILES)

    def _validate_stream_identity_recipe(self, recipe: Mapping[str, Any]) -> None:
        recipe_sha = recipe.get("recipe_sha256")
        if not isinstance(recipe_sha, str) or self.identity.stream_recipe_sha != recipe_sha:
            raise ValueError("stream dataset recipe does not match training identity")
        if self.identity.dataset_index_sha != recipe_sha:
            raise ValueError("stream dataset index identity does not match recipe")
        if self.identity.split_hash != sha256_json(recipe["splits"]):
            raise ValueError("stream dataset split identity does not match recipe")
        recipe_path = self.output_dir / "stream_recipe.json"
        if not recipe_path.is_file() or sha256_file(recipe_path) != self.identity.stream_recipe_file_sha:
            raise ValueError("stream dataset recipe file identity does not match recipe")

    def _fit_after_lease(self, train_dataset: Dataset,
                         validation_dataset: Dataset) -> Dict[str, Any]:
        self._reject_completed_resume_before_mutation()
        if self.stream_pilot_recipe is not None:
            from .stream_admission import validate_stream_pilot_datasets
            validate_stream_pilot_datasets(train_dataset, validation_dataset, self.stream_pilot_recipe)
        self._validate_stream_recipe_for_resume(self.output_dir / "checkpoint_latest.pt")
        for name in ("training_summary.json", "gradient_audit_summary.json", "training_completion.json", "checkpoint_validation.json"):
            old_summary = self.output_dir / name
            if old_summary.exists():
                if not self.config.resume_checkpoint:
                    raise ValueError("completed output requires explicit resume")
                archive_training_summary(old_summary)
        self.telemetry.begin(self.global_step, self.start_epoch)
        atomic_training_json(self.output_dir / "training_completion.json", {
            "schema": "V10TrainingCompletionV1", "status": "IN_PROGRESS",
            "run_id": self.config.run_id or None, "execution_id": self.telemetry.execution_id,
            "source_identity": asdict(self.identity), "scientific_gate_evaluated": False})
        return self._fit_with_observers(train_dataset, validation_dataset)

    def _reject_completed_resume_before_mutation(self) -> None:
        """Never replace a canonical COMPLETED projection during resume."""
        if not self.config.resume_checkpoint:
            return
        descriptor = read_checkpoint_commit(self.output_dir)
        if isinstance(descriptor, Mapping) and descriptor.get("training_complete") is True:
            raise ValueError("completed output cannot be resumed in place")
        if self._resume_checkpoint_was_complete:
            raise ValueError("completed checkpoint cannot be resumed in place")

    def _fit_with_observers(self, train_dataset: Dataset,
                            validation_dataset: Dataset) -> Dict[str, Any]:
        initial_global_step = self.global_step
        validation_loader = self._loader(validation_dataset, shuffle=False)
        fixed_validation_batch = _move_to_device(next(iter(validation_loader)), self.device)
        history = []
        checkpoint_path = None
        deferred_terminal_checkpoint = False
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loader = self._loader(train_dataset, shuffle=True, epoch=epoch)
            validation_loader = self._loader(validation_dataset, shuffle=False, epoch=epoch)
            train_metrics = self._run_batches(
                train_loader, training=True,
                maximum_batches=self.config.max_train_batches, epoch=epoch)
            with torch.no_grad():
                validation_metrics = self._run_batches(
                    validation_loader, training=False,
                    maximum_batches=self.config.max_validation_batches, epoch=epoch)
            for observer in self.observers:
                callback = getattr(observer, "on_epoch_end", None)
                if callable(callback):
                    callback(epoch=epoch, train_metrics=dict(train_metrics),
                             validation_metrics=dict(validation_metrics))
            metric = float(validation_metrics["loss"])
            improved = (metric < self.best_metric if self.config.metric_direction == "min"
                        else metric > self.best_metric)
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
            deferred_terminal_checkpoint = self._should_defer_terminal_checkpoint(epoch)
            if deferred_terminal_checkpoint:
                # Keep the last resumable generation at epoch 10.  If the
                # process dies before completion is sealed, resume replays
                # epoch 11 instead of observing a terminal incomplete epoch.
                checkpoint_path = checkpoint_path or Path(self.config.resume_checkpoint)
                if not hasattr(self, "_last_checkpoint_commit"):
                    self._last_checkpoint_commit = read_checkpoint_commit(self.output_dir)
                commit = self._last_checkpoint_commit
            else:
                checkpoint_path = self._save_checkpoint(epoch, training_complete=False,
                                                        metric_record=entry)
                commit = self._last_checkpoint_commit
            if deferred_terminal_checkpoint:
                entry["checkpoint_deferred"] = True
                entry["resume_generation_id"] = commit["generation_id"]
                entry["resume_checkpoint_sha256"] = commit["checkpoint"]["sha256"]
            else:
                entry.update({"execution_id": self.telemetry.execution_id,
                              "generation_id": commit["generation_id"],
                              "checkpoint_sha256": commit["checkpoint"]["sha256"],
                              "training_complete": False,
                              "generation_manifest": {
                                  "execution_id": commit["execution_id"],
                                  "generation_id": commit["generation_id"],
                                  "checkpoint_sha256": commit["checkpoint"]["sha256"],
                                  "checkpoint_bytes": commit["checkpoint"]["bytes"],
                                  "model_id": commit["model_id"], "config_sha": commit["config_sha"],
                                  "source_identity": commit["source_identity"],
                                  "epoch": commit["epoch"], "global_step": commit["global_step"],
                                  "training_complete": False,
                                  "model_state_sha": commit["model_state_sha"],
                              }})
            history.append(entry)
            self.telemetry.record_epoch(epoch, train_metrics, validation_metrics)
            if self._should_stop_early():
                break

        self._validate_stream_pilot_epoch_completion(history)

        if self.config.run_mode == "smoke" and self.global_step < 2:
            raise RuntimeError("smoke did not execute the required two optimizer steps")
        if self.global_step <= initial_global_step:
            raise RuntimeError("training did not execute an optimizer step")
        self.model.eval()
        with torch.no_grad():
            prediction_before = self.prediction_function(self.model, fixed_validation_batch)
        # Reload the last incomplete checkpoint before observer validation,
        # except when the terminal stream-pilot epoch was deliberately kept
        # out of the resumable generation.
        if deferred_terminal_checkpoint:
            saved = self._checkpoint(history[-1]["epoch"], training_complete=False)
            prediction_after = self._round_trip_prediction(saved, fixed_validation_batch)
            inference_difference = _prediction_max_difference(prediction_before, prediction_after)
        else:
            saved = self._load_verified_checkpoint(resolve_checkpoint_resume(
                self.output_dir, checkpoint_path))
            self.model.load_state_dict(saved["model_state_dict"], strict=True)
            self.model.eval()
            with torch.no_grad():
                prediction_after = self.prediction_function(self.model, fixed_validation_batch)
            inference_difference = _prediction_max_difference(prediction_before, prediction_after)
        if inference_difference > 1e-7:
            raise RuntimeError("checkpoint reload changed inference")
        summary = {
            "schema": "V10TrainingSummaryV1",
            "model_id": self.config.model_id,
            "run_id": self.config.run_id or None,
            "execution_id": self.telemetry.execution_id,
            "run_mode": self.config.run_mode,
            "global_step": self.global_step,
            "epochs_completed": self._cumulative_epochs_completed(history),
            "epochs_this_execution": len(history),
            "best_metric": self.best_metric,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_model_state_sha": saved["metadata"]["model_state_sha"],
            "reload_inference_max_abs_difference": inference_difference,
            "finite_gradient_audit_passed": all(
                bool(entry["train"]["gradient_finite"]) for entry in history),
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
                        f"failed during finalize: {exc}") from exc
        # Non-smoke training requires at least one gradient audit step.
        if observer_required:
            audit_key = "observer_V10GradientAuditObserver"
            audit_summary = summary.get(audit_key, {})
            audited_steps = audit_summary.get("audited_steps", 0)
            if audited_steps < 1:
                raise RuntimeError(
                    f"gradient audit recorded {audited_steps} steps; "
                    f"non-smoke training requires audited_steps >= 1")
            if self.stream_pilot_recipe is not None:
                output_audit = summary.get("observer_Classifier_output_audit", {})
                expected_cells = set(self.identity.dataset_cell_id.split(","))
                if (output_audit.get("audit_complete") is not True
                        or set(output_audit.get("audited_cells", ())) != expected_cells
                        or set(output_audit.get("registered_cells", ())) != expected_cells
                        or output_audit.get("final_committed_step") != self.global_step):
                    raise RuntimeError("stream pilot requires complete registered-cell output auditing")
        execution_evidence = None
        if self._is_admitted_stream_pilot():
            gradient_audit_summary = dict(summary.get("observer_V10GradientAuditObserver", {}))
            gradient_audit_summary["source_identity"] = asdict(self.identity)
            summary["observer_V10GradientAuditObserver"] = gradient_audit_summary
            execution_evidence = build_training_execution_evidence(
                run_id=self.config.run_id,
                execution_id=self.telemetry.execution_id,
                producer_pid=os.getpid(),
                telemetry_started_at=self.telemetry.current["producer_started_at"],
                initial_global_step=self.telemetry.current["initial_global_step"],
                final_global_step=self.global_step,
                optimizer_steps_this_execution=self.telemetry.current["optimizer_steps_this_execution"],
                finite_gradient_audit_passed=summary["finite_gradient_audit_passed"],
                gradient_audit_summary=gradient_audit_summary,
                output_audit_summary=summary.get("observer_Classifier_output_audit", {}),
                source_identity=asdict(self.identity),
                registered_cells=sorted(self.identity.dataset_cell_id.split(",")),
            )
            summary["execution_evidence"] = execution_evidence
            summary["execution_evidence_sha256"] = training_execution_evidence_sha256(execution_evidence)
            summary["execution_evidence_validated"] = True
        checkpoint_path = self.output_dir / "checkpoint_latest.pt"
        summary["checkpoint_path"] = str(checkpoint_path)
        summary["audit_projection_path"] = str(self.output_dir / "checkpoint_audit.pt")
        summary["checkpoint_validation_path"] = str(self.output_dir / "checkpoint_validation.json")
        summary["completion_receipt_path"] = str(self.output_dir / "training_completion.json")
        summary["telemetry"] = self.telemetry.summary()
        final_metric = dict(history[-1])
        final_checkpoint = self._checkpoint(history[-1]["epoch"], training_complete=True)
        final_commit = commit_checkpoint(self.output_dir, final_checkpoint,
                                         execution_id=self.telemetry.execution_id,
                                         epoch=history[-1]["epoch"], global_step=self.global_step,
                                         authoritative=False, writer_lease=self._writer_lease,
                                         metric_record=final_metric)
        summary["checkpoint_commit"] = final_commit
        final_metric.update({"execution_id": self.telemetry.execution_id,
                             "generation_id": final_commit["generation_id"],
                             "checkpoint_sha256": final_commit["checkpoint"]["sha256"],
                             "training_complete": True,
                             "generation_manifest": {
                                 "execution_id": final_commit["execution_id"],
                                 "generation_id": final_commit["generation_id"],
                                 "checkpoint_sha256": final_commit["checkpoint"]["sha256"],
                                 "checkpoint_bytes": final_commit["checkpoint"]["bytes"],
                                 "model_id": final_commit["model_id"], "config_sha": final_commit["config_sha"],
                                 "source_identity": final_commit["source_identity"],
                                 "epoch": final_commit["epoch"], "global_step": final_commit["global_step"],
                                 "training_complete": True,
                                 "model_state_sha": final_commit["model_state_sha"],
                             }})
        summary["metrics_journal"] = final_commit["metrics_journal"]
        if self.identity.input_delivery == "stream_only":
            from .stream_admission import assert_stream_source_identity
            assert_stream_source_identity(asdict(self.identity))
        completion = publish_training_completion(
            output_dir=self.output_dir, checkpoint=final_checkpoint,
            config=asdict(self.config), source_identity=asdict(self.identity), summary=summary,
            execution_id=self.telemetry.execution_id, compute_state_sha=_state_dict_sha,
            checkpoint_commit=final_commit, writer_lease=self._writer_lease)
        self.telemetry.record_checkpoint(checkpoint_path, epoch=history[-1]["epoch"], global_step=self.global_step,
                                         training_complete=True, model_state_sha=summary["checkpoint_model_state_sha"],
                                         commit=final_commit)
        self.telemetry.current["completion"] = completion
        self.telemetry.record_phase("completed")
        return summary

    def _validate_stream_pilot_epoch_completion(self, history: Sequence[Mapping[str, Any]]) -> None:
        """Require the cumulative admitted stream pilot to reach its final epoch."""
        if not self._is_admitted_stream_pilot():
            return
        terminal_epoch = history[-1].get("epoch") if history else None
        if type(terminal_epoch) is not int or terminal_epoch + 1 != self.config.epochs:
            raise RuntimeError(
                "admitted stream pilot must complete exactly 12 epochs before publication")

    def _should_stop_early(self) -> bool:
        """Keep early stopping out of the fixed admitted stream pilot."""
        return (not self._is_admitted_stream_pilot()
                and self.bad_epochs >= self.config.early_stopping_patience)

    def _should_defer_terminal_checkpoint(self, epoch: int) -> bool:
        """Keep the final incomplete stream-pilot epoch replayable after a crash."""
        return self._is_admitted_stream_pilot() and epoch == self.config.epochs - 1

    def _cumulative_epochs_completed(self, history: Sequence[Mapping[str, Any]]) -> int:
        """Report terminal epoch progress, including epochs completed before resume."""
        if self._is_admitted_stream_pilot():
            terminal_epoch = history[-1].get("epoch") if history else self.start_epoch - 1
            return int(terminal_epoch) + 1
        return len(history)

    def _is_admitted_stream_pilot(self) -> bool:
        return self.config.run_mode == "pilot" and self.stream_pilot_recipe is not None

    def _round_trip_prediction(self, checkpoint: Mapping[str, Any], batch: Any) -> Any:
        """Serialize and reload a staged checkpoint before measuring inference parity."""
        buffer = io.BytesIO()
        torch.save(dict(checkpoint), buffer)
        buffer.seek(0)
        reloaded = torch.load(buffer, weights_only=False)
        self.model.load_state_dict(reloaded["model_state_dict"], strict=True)
        self.model.eval()
        with torch.no_grad():
            return self.prediction_function(self.model, batch)
