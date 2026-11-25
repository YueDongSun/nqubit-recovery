"""Explicitly bounded CPU smoke interface; this is not long training evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import math
import json
import os
from pathlib import Path
import random
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional-ML installation only.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    require_exact_canonical_json_round_trip,
    require_git_oid,
    require_sha256,
)
from .losses import (
    TrackBRealStaticLossWeightsV1,
    ratio_of_sums_query_absolute_error,
    trackb_masked_real_static_objective,
    validate_real_static_loss_plan,
)
from .observation_plate import (
    MaskedObservationPlateV1,
    MaskedSetQueryObservationV1,
    TrainableMaskedObservationPlateV1,
    deterministic_query_mask,
)
from .row_identity import ROW_CONTENT_FIELDS, row_identity_manifest_sha, source_row_content_sha


# A bounded contract smoke may exercise the requested 2,000 committed local
# scalar steps.  It remains a synthetic/in-memory plumbing check and is not an
# O4/F2/F4 long-training arm, a checkpoint artifact, or scientific evidence.
MAX_SMOKE_OPTIMIZER_STEPS = 2000
_NO_OPERATION_METADATA_SIDECAR_SHA = canonical_sha256({
    "schema": "TrackBNoOperationMetadataSidecarV1",
})


@dataclass(frozen=True)
class TrackBSmokeStateV1:
    """Minimal deterministic in-memory checkpoint state for resume-parity checks."""

    step: int
    scalar_bias: float
    seed: int
    schema: str = field(default="TrackBSmokeStateV1", init=False)

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0 or type(self.seed) is not int:
            raise ValueError("smoke state step/seed must be native nonnegative integers")
        if not math.isfinite(float(self.scalar_bias)):
            raise ValueError("smoke state scalar bias must be finite")

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {"schema": self.schema, "step": self.step, "scalar_bias": self.scalar_bias, "seed": self.seed},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "TrackBSmokeStateV1":
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("smoke checkpoint is not valid UTF-8 JSON") from error
        if type(value) is not dict or value.get("schema") != "TrackBSmokeStateV1":
            raise ValueError("smoke checkpoint schema is invalid")
        state = cls(step=value.get("step"), scalar_bias=value.get("scalar_bias"), seed=value.get("seed"))
        if state.canonical_bytes() != payload:
            raise ValueError("smoke checkpoint bytes are not canonical")
        return state


def smoke_optimizer_step(state: TrackBSmokeStateV1, residual: np.ndarray) -> TrackBSmokeStateV1:
    """One deterministic scalar update used exclusively for in-memory smoke parity."""
    value = np.asarray(residual, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("smoke residual must be a finite rank-2 query array")
    gradient = float(np.sign(value + state.scalar_bias).mean()) if value.size else 0.0
    return TrackBSmokeStateV1(step=state.step + 1, scalar_bias=state.scalar_bias - 0.01 * gradient, seed=state.seed)


@dataclass(frozen=True)
class TrackBSmokeTrainingReceiptV1:
    requested_steps: int
    committed_optimizer_steps: int
    final_query_ratio_of_sums_error: float
    checkpoint_roundtrip_verified: bool
    long_training_started: bool = False
    status: str = "SMOKE_ONLY_NOT_TRAINING_EVIDENCE"
    schema: str = field(default="TrackBSmokeTrainingReceiptV1", init=False)

    def __post_init__(self) -> None:
        if type(self.requested_steps) is not int or not 0 <= self.requested_steps <= MAX_SMOKE_OPTIMIZER_STEPS:
            raise ValueError("smoke training steps must remain within the small fixed limit")
        if self.committed_optimizer_steps != self.requested_steps:
            raise ValueError("smoke receipt must account for every requested local smoke step")
        if not math.isfinite(self.final_query_ratio_of_sums_error) or self.final_query_ratio_of_sums_error < 0.0:
            raise ValueError("smoke loss must be finite and nonnegative")
        if self.long_training_started or self.checkpoint_roundtrip_verified is not True:
            raise ValueError("Track B smoke interface cannot start long training")


def run_short_smoke_training(
    observation: MaskedSetQueryObservationV1, *, optimizer_steps: int = 2,
) -> TrackBSmokeTrainingReceiptV1:
    """Exercise set/query loss plumbing with bounded local scalar updates only.

    It creates neither a checkpoint nor a data artifact and can never expand into
    O4/F2/F4 long training.  The scalar is intentionally disposable.
    """
    if type(optimizer_steps) is not int or not 0 <= optimizer_steps <= MAX_SMOKE_OPTIMIZER_STEPS:
        raise ValueError(f"optimizer_steps must be in [0,{MAX_SMOKE_OPTIMIZER_STEPS}] for a smoke run")
    validate_real_static_loss_plan(("masked_four_cell",))
    plate = MaskedObservationPlateV1()
    prediction = np.asarray(plate.predict(observation).four_cell_mean, dtype=np.float64)
    target, query_mask = observation.query_targets()
    state = TrackBSmokeStateV1(step=0, scalar_bias=0.0, seed=0)
    residual = prediction[query_mask] - target[query_mask]
    for _ in range(optimizer_steps):
        state = smoke_optimizer_step(state, residual)
    restored = TrackBSmokeStateV1.from_canonical_bytes(state.canonical_bytes())
    if restored != state:
        raise RuntimeError("in-memory smoke checkpoint/resume parity failed")
    error = ratio_of_sums_query_absolute_error(prediction + state.scalar_bias, target, query_mask)
    return TrackBSmokeTrainingReceiptV1(
        requested_steps=optimizer_steps,
        committed_optimizer_steps=optimizer_steps,
        final_query_ratio_of_sums_error=error,
        checkpoint_roundtrip_verified=True,
    )


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Track B real runtime requires PyTorch")


def _state_dict_sha(state_dict: Mapping[str, "torch.Tensor"]) -> str:
    """Hash a state dict without relying on pickle serialization details."""
    _require_torch()
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _runtime_state_value_identity(value: Any) -> object:
    """Serialize mutable Torch state without pickle or object-address identity."""
    _require_torch()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        entries = [
            [
                _runtime_state_value_identity(key),
                _runtime_state_value_identity(item),
            ]
            for key, item in value.items()
        ]
        return {
            "kind": "mapping",
            "entries": sorted(entries, key=lambda item: canonical_sha256(item[0])),
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [_runtime_state_value_identity(item) for item in value],
        }
    if value is None or type(value) in (bool, int, str):
        return {"kind": type(value).__name__, "value": value}
    if type(value) is float:
        return {"kind": "float", "value_hex": value.hex()}
    raise TypeError(f"Track B runtime state contains unsupported {type(value).__name__}")


def _runtime_state_sha(value: Any) -> str:
    return canonical_sha256({
        "schema": "TrackBPortableRuntimeStateV1",
        "value": _runtime_state_value_identity(value),
    })


def _model_optimizer_topology_sha(model: Any, optimizer: Any) -> str:
    """Bind optimizer groups to the exact live model parameter objects."""
    _require_torch()
    named_parameters = tuple(model.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in named_parameters}
    if len(names_by_id) != len(named_parameters):
        raise ValueError("Track B model exposes duplicate parameter identities")
    groups = []
    observed_names = []
    for group in optimizer.param_groups:
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise ValueError("Track B optimizer parameter group is malformed")
        group_names = []
        for parameter in parameters:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError("Track B optimizer owns a parameter outside the runtime model")
            group_names.append(name)
            observed_names.append(name)
        groups.append(tuple(group_names))
    expected_names = tuple(name for name, _parameter in named_parameters)
    if tuple(sorted(observed_names)) != tuple(sorted(expected_names)) \
            or len(observed_names) != len(set(observed_names)):
        raise ValueError("Track B optimizer/model parameter topology is not one-to-one")
    return canonical_sha256({
        "schema": "TrackBModelOptimizerTopologyV1",
        "model_parameter_names": list(expected_names),
        "optimizer_parameter_groups": [list(group) for group in groups],
    })


def _assert_finite_runtime_state(value: Any, *, name: str) -> None:
    """Reject non-finite tensors or scalar state before a transition is sealed."""
    _require_torch()
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.all(torch.isfinite(value)).detach()):
            raise FloatingPointError(f"Track B runtime {name} contains non-finite tensor state")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_runtime_state(key, name=name)
            _assert_finite_runtime_state(item, name=name)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_finite_runtime_state(item, name=name)
        return
    if type(value) is float and not math.isfinite(value):
        raise FloatingPointError(f"Track B runtime {name} contains a non-finite scalar")
    if value is None or type(value) in (bool, int, float, str):
        return
    raise TypeError(f"Track B runtime {name} contains unsupported {type(value).__name__}")


def trackb_model_state_sha(model: Any) -> str:
    """Return the portable initial-state identity used by pilot admission."""
    _require_torch()
    return _state_dict_sha(model.state_dict())


def seed_trackb_runtime(seed: int) -> None:
    """Seed every runtime RNG before constructing a pilot model."""
    _require_torch()
    if type(seed) is not int:
        raise TypeError("Track B runtime seed must be an integer")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_rng_state() -> dict[str, Any]:
    """Return a weights-only-loadable RNG snapshot without NumPy pickle objects."""
    _require_torch()
    numpy_state = np.random.get_state()
    return {
        "python": _tuples_to_lists(random.getstate()),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": [int(value) for value in numpy_state[1].tolist()],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
    }


def _tuples_to_lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, list):
        return [_tuples_to_lists(item) for item in value]
    return value


def _lists_to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


def _validated_rng_state(state: Mapping[str, Any]) -> tuple[object, tuple[object, ...], "torch.Tensor", list["torch.Tensor"]]:
    _require_torch()
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("Track B checkpoint RNG state is incomplete")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator", "state", "position", "has_gauss", "cached_gaussian",
    }:
        raise ValueError("Track B checkpoint NumPy RNG state is invalid")
    if not isinstance(numpy_state["state"], list) \
            or any(type(value) is not int for value in numpy_state["state"]):
        raise ValueError("Track B checkpoint NumPy RNG vector is invalid")
    if not isinstance(state["torch_cpu"], torch.Tensor) \
            or not isinstance(state["torch_cuda"], list) \
            or any(not isinstance(value, torch.Tensor) for value in state["torch_cuda"]):
        raise ValueError("Track B checkpoint Torch RNG state is invalid")
    python_state = _lists_to_tuples(state["python"])
    numpy_state_tuple = (
        str(numpy_state["bit_generator"]),
        np.asarray(numpy_state["state"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    )
    return python_state, numpy_state_tuple, state["torch_cpu"], list(state["torch_cuda"])


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    python_state, numpy_state, torch_cpu_state, torch_cuda_states = _validated_rng_state(state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_cpu_state)
    if torch.cuda.is_available() and torch_cuda_states:
        torch.cuda.set_rng_state_all(torch_cuda_states)


def _move_to_device(value: Any, device: "torch.device") -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _batch_row_identity_values(raw_batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Extract one pre-query source row for content rehashing."""
    batch_level = {
        "operation_ids", "operation_families", "cycle_indices", "system_type",
        "prepared_command_semantics",
    }
    values: dict[str, Any] = {}
    for name in ROW_CONTENT_FIELDS:
        if name == "sample_id":
            values[name] = raw_batch["sample_ids"][index]
        elif name == "group_id":
            values[name] = raw_batch["group_ids"][index]
        elif name == "source_role":
            values[name] = raw_batch["source_roles"][index]
        elif name in batch_level:
            values[name] = raw_batch.get(name)
        elif name == "raw_rf_status":
            values[name] = raw_batch["raw_rf_status"][index]
        elif name == "global_multi_scope":
            values[name] = raw_batch["global_multi_scope"][index]
        else:
            value = raw_batch.get(name)
            if value is None:
                values[name] = None
            elif isinstance(value, torch.Tensor):
                values[name] = value[index]
            else:
                values[name] = value[index]
    return values


@dataclass(frozen=True)
class TrackBRealRuntimeIdentityV1:
    """All immutable identities required for an exact step-level resume."""

    run_id: str
    system_type: str
    admission_manifest_sha: str
    outer_fold_sha: str
    model_config_sha: str
    source_commit: str
    source_tree: str
    initial_model_state_sha: str
    seed: int = 20261001
    allowed_sample_ids: tuple[str, ...] = ()
    allowed_group_ids: tuple[str, ...] = ()
    allowed_roles: tuple[str, ...] = ("train",)
    allowed_row_identities: tuple[tuple[str, str, str, str], ...] = ()
    row_identity_sha: str = ""
    batch_plan_sha: str = ""
    batch_plan_sample_ids: tuple[str, ...] = ()
    batch_plan_batch_size: int = 0
    batch_plan_view_seed: int = 0
    batch_plan_query_fraction: float = 0.25
    session_index_bindings: tuple[tuple[str, int], ...] = ()
    operation_metadata_sidecar_file_sha: str = _NO_OPERATION_METADATA_SIDECAR_SHA
    operation_metadata_sidecar_semantic_sha: str = _NO_OPERATION_METADATA_SIDECAR_SHA
    schema: str = field(default="TrackBRealRuntimeIdentityV1", init=False)

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or not self.run_id:
            raise ValueError("Track B runtime run_id must be nonempty")
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("Track B runtime system type is invalid")
        for name in (
            "admission_manifest_sha", "outer_fold_sha", "model_config_sha",
            "operation_metadata_sidecar_file_sha", "operation_metadata_sidecar_semantic_sha",
        ):
            require_sha256(getattr(self, name), name)
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        require_sha256(self.initial_model_state_sha, "initial_model_state_sha")
        if type(self.seed) is not int:
            raise TypeError("Track B runtime seed must be an integer")
        for values, name in (
            (self.allowed_sample_ids, "allowed_sample_ids"),
            (self.allowed_group_ids, "allowed_group_ids"),
        ):
            if not values or len(set(values)) != len(values) \
                    or any(type(value) is not str or not value for value in values):
                raise ValueError(f"{name} must be a unique nonempty allowlist")
        if self.allowed_roles != ("train",):
            raise ValueError("Track B training runtime may open only the train role")
        require_sha256(self.row_identity_sha, "row_identity_sha")
        require_sha256(self.batch_plan_sha, "batch_plan_sha")
        if not self.batch_plan_sample_ids:
            raise ValueError("Track B optimizer runtime requires an immutable nonempty batch plan")
        if tuple(self.batch_plan_sample_ids) != tuple(self.allowed_sample_ids):
            raise ValueError("Track B batch plan sample order must equal the admitted source order")
        if type(self.batch_plan_batch_size) is not int or self.batch_plan_batch_size <= 0:
            raise ValueError("Track B batch plan requires a positive batch size")
        if type(self.batch_plan_view_seed) is not int:
            raise TypeError("Track B batch plan view seed must be an integer")
        if type(self.batch_plan_query_fraction) not in (int, float) \
                or not math.isfinite(float(self.batch_plan_query_fraction)) \
                or not 0.20 <= float(self.batch_plan_query_fraction) <= 0.30:
            raise ValueError("Track B batch plan query fraction must lie in the registered [0.20,0.30] range")
        object.__setattr__(self, "batch_plan_query_fraction", float(self.batch_plan_query_fraction))
        expected_plan_sha = canonical_sha256({
            "schema": "TrackBBatchPlanV1",
            "sample_ids": list(self.batch_plan_sample_ids),
            "batch_size": self.batch_plan_batch_size,
            "view_seed": self.batch_plan_view_seed,
            "query_fraction": float(self.batch_plan_query_fraction),
            "shuffle": False,
            "drop_last": False,
        })
        if expected_plan_sha != self.batch_plan_sha:
            raise ValueError("Track B batch plan SHA does not match its immutable plan fields")
        if not self.session_index_bindings:
            raise ValueError("Track B optimizer runtime requires admitted session-index bindings")
        if len(self.session_index_bindings) != len(self.allowed_sample_ids) \
                or tuple(str(item[0]) for item in self.session_index_bindings) != self.allowed_sample_ids:
            raise ValueError("Track B session bindings must cover the admitted sample order exactly")
        if any(
            type(sample_id) is not str or not sample_id
            or type(session_index) is not int or session_index < 0
            for sample_id, session_index in self.session_index_bindings
        ):
            raise ValueError("Track B session bindings must contain nonnegative native indices")
        if not self.allowed_row_identities:
            raise ValueError("Track B runtime requires source row identity records")
        records = tuple(tuple(record) for record in self.allowed_row_identities)
        if any(len(record) != 4 for record in records):
            raise ValueError("Track B row identity records must contain sample/group/content/role")
        if len({record[0] for record in records}) != len(records):
            raise ValueError("Track B row identity records require unique sample IDs")
        for record in records:
            sample_id, group_id, content_sha, role = record
            if any(type(value) is not str or not value for value in record):
                raise ValueError("Track B row identity records must contain nonempty strings")
            require_sha256(content_sha, "row_content_sha")
            if role != "train":
                raise ValueError("Track B row identity records may open only train rows")
        if row_identity_manifest_sha(records) != self.row_identity_sha:
            raise ValueError("Track B row identity manifest SHA does not match its records")
        if {record[0] for record in records} != set(self.allowed_sample_ids) \
                or {record[1] for record in records} != set(self.allowed_group_ids):
            raise ValueError("Track B row identity records disagree with the runtime allowlists")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "system_type": self.system_type,
            "admission_manifest_sha": self.admission_manifest_sha,
            "outer_fold_sha": self.outer_fold_sha,
            "model_config_sha": self.model_config_sha,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "initial_model_state_sha": self.initial_model_state_sha,
            "seed": self.seed,
            "allowed_sample_ids": list(self.allowed_sample_ids),
            "allowed_group_ids": list(self.allowed_group_ids),
            "allowed_roles": list(self.allowed_roles),
            "allowed_row_identities": [list(record) for record in self.allowed_row_identities],
            "row_identity_sha": self.row_identity_sha,
            "batch_plan_sha": self.batch_plan_sha,
            "batch_plan_sample_ids": list(self.batch_plan_sample_ids),
            "batch_plan_batch_size": self.batch_plan_batch_size,
            "batch_plan_view_seed": self.batch_plan_view_seed,
            "batch_plan_query_fraction": self.batch_plan_query_fraction,
            "session_index_bindings": [list(item) for item in self.session_index_bindings],
            "operation_metadata_sidecar_file_sha": self.operation_metadata_sidecar_file_sha,
            "operation_metadata_sidecar_semantic_sha": self.operation_metadata_sidecar_semantic_sha,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrackBRealRuntimeIdentityV1":
        expected = {
            "schema", "run_id", "system_type", "admission_manifest_sha", "outer_fold_sha",
            "model_config_sha", "source_commit", "source_tree", "initial_model_state_sha", "seed",
            "allowed_sample_ids", "allowed_group_ids", "allowed_roles", "allowed_row_identities",
            "row_identity_sha", "batch_plan_sha", "batch_plan_sample_ids", "batch_plan_batch_size",
            "batch_plan_view_seed", "batch_plan_query_fraction", "session_index_bindings",
            "operation_metadata_sidecar_file_sha", "operation_metadata_sidecar_semantic_sha",
        }
        if set(payload) != expected or payload.get("schema") != "TrackBRealRuntimeIdentityV1":
            raise ValueError("Track B runtime identity schema is invalid")
        sequence_fields = (
            "allowed_sample_ids", "allowed_group_ids", "allowed_roles", "allowed_row_identities",
            "batch_plan_sample_ids", "session_index_bindings",
        )
        if any(type(payload[name]) is not list for name in sequence_fields):
            raise TypeError("Track B runtime identity sequence fields must be JSON arrays")
        if any(type(record) is not list for record in payload["allowed_row_identities"]) \
                or any(type(binding) is not list for binding in payload["session_index_bindings"]):
            raise TypeError("Track B runtime identity nested records must be JSON arrays")
        return cls(
            run_id=payload["run_id"],
            system_type=payload["system_type"],
            admission_manifest_sha=payload["admission_manifest_sha"],
            outer_fold_sha=payload["outer_fold_sha"],
            model_config_sha=payload["model_config_sha"],
            source_commit=payload["source_commit"],
            source_tree=payload["source_tree"],
            initial_model_state_sha=payload["initial_model_state_sha"],
            seed=payload["seed"],
            allowed_sample_ids=tuple(payload["allowed_sample_ids"]),
            allowed_group_ids=tuple(payload["allowed_group_ids"]),
            allowed_roles=tuple(payload["allowed_roles"]),
            allowed_row_identities=tuple(
                tuple(record) for record in payload["allowed_row_identities"]
            ),
            row_identity_sha=payload["row_identity_sha"],
            batch_plan_sha=payload["batch_plan_sha"],
            batch_plan_sample_ids=tuple(payload["batch_plan_sample_ids"]),
            batch_plan_batch_size=payload["batch_plan_batch_size"],
            batch_plan_view_seed=payload["batch_plan_view_seed"],
            batch_plan_query_fraction=payload["batch_plan_query_fraction"],
            session_index_bindings=tuple(
                tuple(binding) for binding in payload["session_index_bindings"]
            ),
            operation_metadata_sidecar_file_sha=payload["operation_metadata_sidecar_file_sha"],
            operation_metadata_sidecar_semantic_sha=payload["operation_metadata_sidecar_semantic_sha"],
        )

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TrackBRealBatchPlanV1:
    """Typed immutable batch plan replayed against one runtime identity."""

    sample_ids: tuple[str, ...]
    batch_size: int
    view_seed: int
    query_fraction: float
    shuffle: bool = False
    drop_last: bool = False
    schema: str = field(default="TrackBRealBatchPlanV1", init=False)

    def __post_init__(self) -> None:
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids) \
                or any(type(value) is not str or not value for value in self.sample_ids):
            raise ValueError("Track B batch plan requires a unique nonempty sample order")
        if type(self.batch_size) is not int or self.batch_size <= 0 \
                or type(self.view_seed) is not int:
            raise ValueError("Track B batch plan batch size/view seed is invalid")
        if type(self.query_fraction) not in (int, float) \
                or not math.isfinite(float(self.query_fraction)) \
                or not 0.20 <= float(self.query_fraction) <= 0.30:
            raise ValueError("Track B batch plan query fraction is invalid")
        if self.shuffle is not False or self.drop_last is not False:
            raise ValueError("Track B actionful batch plan must be deterministic without drop-last")
        object.__setattr__(self, "query_fraction", float(self.query_fraction))

    def semantic_sha256(self) -> str:
        return canonical_sha256({
            "schema": "TrackBBatchPlanV1",
            "sample_ids": list(self.sample_ids),
            "batch_size": self.batch_size,
            "view_seed": self.view_seed,
            "query_fraction": self.query_fraction,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        })

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sample_ids": list(self.sample_ids),
            "batch_size": self.batch_size,
            "view_seed": self.view_seed,
            "query_fraction": self.query_fraction,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "semantic_sha256": self.semantic_sha256(),
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrackBRealBatchPlanV1":
        expected = {
            "schema", "sample_ids", "batch_size", "view_seed", "query_fraction",
            "shuffle", "drop_last", "semantic_sha256",
        }
        if set(payload) != expected or payload.get("schema") != "TrackBRealBatchPlanV1":
            raise ValueError("Track B batch plan artifact schema is invalid")
        if type(payload["sample_ids"]) is not list:
            raise TypeError("Track B batch plan sample IDs must be a JSON array")
        plan = cls(
            sample_ids=tuple(payload["sample_ids"]),
            batch_size=payload["batch_size"],
            view_seed=payload["view_seed"],
            query_fraction=payload["query_fraction"],
            shuffle=payload["shuffle"],
            drop_last=payload["drop_last"],
        )
        if payload["semantic_sha256"] != plan.semantic_sha256():
            raise ValueError("Track B batch plan artifact semantic SHA is invalid")
        return plan

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBRealBatchPlanV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="TrackBRealBatchPlanV1",
            exact_fields=frozenset({
                "schema", "sample_ids", "batch_size", "view_seed", "query_fraction",
                "shuffle", "drop_last", "semantic_sha256",
            }),
        )
        plan = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, plan)
        return plan

    @classmethod
    def from_runtime_identity(cls, identity: TrackBRealRuntimeIdentityV1) -> "TrackBRealBatchPlanV1":
        plan = cls(
            sample_ids=tuple(identity.batch_plan_sample_ids),
            batch_size=identity.batch_plan_batch_size,
            view_seed=identity.batch_plan_view_seed,
            query_fraction=identity.batch_plan_query_fraction,
        )
        if plan.semantic_sha256() != identity.batch_plan_sha:
            raise ValueError("runtime identity batch plan does not match the typed artifact")
        return plan


@dataclass(frozen=True)
class TrackBLongTrainingAuthorizationV1:
    """External, source-bound authorization for a long optimizer run.

    A runtime configuration describes computation.  It cannot grant itself
    permission to run beyond the pilot budget; this artifact must be supplied by
    an independent caller and must bind the exact output root and identities.
    """

    source_commit: str
    source_tree: str
    admission_manifest_sha: str
    outer_fold_sha: str
    model_config_sha: str
    runtime_config_sha: str
    maximum_optimizer_steps: int
    allowed_systems: tuple[str, ...]
    allowed_roles: tuple[str, ...]
    output_root: str
    expires_at_utc: str
    approval_receipt_sha: str
    hardware_action_authorized: bool = False
    schema: str = field(default="TrackBLongTrainingAuthorizationV1", init=False)

    def __post_init__(self) -> None:
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        for name in (
            "admission_manifest_sha", "outer_fold_sha", "model_config_sha",
            "runtime_config_sha", "approval_receipt_sha",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.maximum_optimizer_steps) is not int or self.maximum_optimizer_steps <= 2000:
            raise ValueError("long-training authorization must exceed the bounded pilot budget")
        if not self.allowed_systems or any(value not in ("3qubit", "4qubit") for value in self.allowed_systems):
            raise ValueError("long-training authorization systems are invalid")
        if len(set(self.allowed_systems)) != len(self.allowed_systems):
            raise ValueError("long-training authorization systems must be unique")
        if self.allowed_roles != ("train",):
            raise ValueError("long-training authorization may open only the train role")
        if type(self.output_root) is not str or not self.output_root.strip():
            raise ValueError("long-training authorization needs a nonempty output root")
        output = Path(self.output_root)
        if not output.is_absolute():
            raise ValueError("long-training authorization needs an absolute output root")
        if type(self.expires_at_utc) is not str:
            raise ValueError("long-training authorization expiry must be ISO-8601 UTC")
        try:
            expires = datetime.fromisoformat(
                self.expires_at_utc[:-1] + "+00:00"
                if self.expires_at_utc.endswith("Z") else self.expires_at_utc,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("long-training authorization expiry must be ISO-8601 UTC") from error
        if expires.tzinfo is None or expires.utcoffset() != timezone.utc.utcoffset(expires):
            raise ValueError("long-training authorization expiry must use UTC")
        if self.hardware_action_authorized is not False:
            raise ValueError("long-training authorization cannot grant hardware action")

    def validate_for(
        self,
        identity: TrackBRealRuntimeIdentityV1,
        config: TrackBRealRuntimeConfigV1,
        *,
        checkpoint_path: str | Path,
        monitor_path: str | Path,
        approval_receipt_path: str | Path | None = None,
        now: datetime | None = None,
    ) -> None:
        identity.__post_init__()
        config.validate()
        if config.execution_mode != "long":
            raise ValueError("long-training authorization requires long runtime mode")
        if identity.source_commit != self.source_commit or identity.source_tree != self.source_tree:
            raise PermissionError("long-training authorization source identity mismatch")
        if identity.admission_manifest_sha != self.admission_manifest_sha \
                or identity.outer_fold_sha != self.outer_fold_sha:
            raise PermissionError("long-training authorization data/fold identity mismatch")
        if identity.model_config_sha != self.model_config_sha \
                or config.sha256() != self.runtime_config_sha:
            raise PermissionError("long-training authorization model/runtime identity mismatch")
        if identity.system_type not in self.allowed_systems \
                or identity.allowed_roles != self.allowed_roles:
            raise PermissionError("long-training authorization system/role mismatch")
        if config.optimizer_steps > self.maximum_optimizer_steps:
            raise PermissionError("runtime optimizer budget exceeds external long-training authorization")
        if approval_receipt_path is None or not Path(approval_receipt_path).is_file():
            raise PermissionError("long-training authorization requires an external approval receipt file")
        observed_receipt_sha = _file_sha256(Path(approval_receipt_path))
        if observed_receipt_sha != self.approval_receipt_sha:
            raise PermissionError("external approval receipt SHA does not match long-training authorization")
        output = Path(self.output_root).resolve()
        if Path(checkpoint_path).resolve().parent != output \
                or Path(monitor_path).resolve().parent != output:
            raise PermissionError("checkpoint/monitor paths are outside the authorized output root")
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        expires = datetime.fromisoformat(
            self.expires_at_utc[:-1] + "+00:00"
            if self.expires_at_utc.endswith("Z") else self.expires_at_utc,
        )
        if expires <= current:
            raise PermissionError("long-training authorization has expired")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "admission_manifest_sha": self.admission_manifest_sha,
            "outer_fold_sha": self.outer_fold_sha,
            "model_config_sha": self.model_config_sha,
            "runtime_config_sha": self.runtime_config_sha,
            "maximum_optimizer_steps": self.maximum_optimizer_steps,
            "allowed_systems": list(self.allowed_systems),
            "allowed_roles": list(self.allowed_roles),
            "output_root": self.output_root,
            "expires_at_utc": self.expires_at_utc,
            "approval_receipt_sha": self.approval_receipt_sha,
            "hardware_action_authorized": False,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TrackBThreeSeedEnsemblePlanV1:
    """Immutable candidate-only three-seed confirmation plan.

    It records the review-required independent seed panel without pretending
    that planning or a completed individual seed establishes selection evidence.
    """

    admission_manifest_sha: str
    outer_fold_sha: str
    model_config_sha: str
    runtime_config_sha: str
    seeds: tuple[int, int, int]
    schema: str = field(default="TrackBThreeSeedEnsemblePlanV1", init=False)

    def __post_init__(self) -> None:
        for name in (
            "admission_manifest_sha", "outer_fold_sha", "model_config_sha", "runtime_config_sha",
        ):
            require_sha256(getattr(self, name), name)
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3 or any(type(value) is not int for value in self.seeds):
            raise ValueError("Track B ensemble confirmation requires exactly three distinct integer seeds")

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "admission_manifest_sha": self.admission_manifest_sha,
            "outer_fold_sha": self.outer_fold_sha,
            "model_config_sha": self.model_config_sha,
            "runtime_config_sha": self.runtime_config_sha,
            "seeds": list(self.seeds),
            "selection_evidence": False,
            "hardware_action_authorized": False,
        })


@dataclass(frozen=True)
class TrackBRealRuntimeConfigV1:
    """Bounded run configuration; long execution remains explicit-only."""

    optimizer_steps: int
    batch_size: int
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 1.0
    scheduler_kind: str = "cosine"
    precision: str = "fp32"
    checkpoint_interval_steps: int = 100
    monitor_interval_steps: int = 10
    execution_mode: str = "pilot"
    long_training_authorized: bool = False
    schema: str = "TrackBRealRuntimeConfigV1"

    def validate(self) -> None:
        if type(self.optimizer_steps) is not int or self.optimizer_steps <= 0:
            raise ValueError("Track B runtime optimizer steps must be positive")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("Track B runtime batch size must be positive")
        for name in ("learning_rate", "weight_decay", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Track B runtime {name} must be finite nonnegative")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("Track B runtime learning rate/gradient clip must be positive")
        if self.scheduler_kind != "cosine" or self.precision not in ("fp32", "bf16"):
            raise ValueError("Track B runtime scheduler/precision is unsupported")
        if min(self.checkpoint_interval_steps, self.monitor_interval_steps) <= 0:
            raise ValueError("Track B runtime checkpoint/monitor intervals must be positive")
        if self.execution_mode not in ("pilot", "long"):
            raise ValueError("Track B runtime execution mode must be pilot or long")
        if self.execution_mode == "pilot":
            if self.optimizer_steps > MAX_SMOKE_OPTIMIZER_STEPS:
                raise ValueError("Track B pilot is bounded to 2,000 committed optimizer steps")
            if self.long_training_authorized:
                raise ValueError("pilot runtime cannot claim long-training authorization")
        elif self.long_training_authorized:
            raise ValueError("runtime config cannot self-authorize long training")

    def sha256(self) -> str:
        self.validate()
        return canonical_sha256({
            "schema": self.schema,
            "optimizer_steps": self.optimizer_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "scheduler_kind": self.scheduler_kind,
            "precision": self.precision,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "monitor_interval_steps": self.monitor_interval_steps,
            "execution_mode": self.execution_mode,
            "long_training_authorized": self.long_training_authorized,
        })


@dataclass(frozen=True)
class TrackBAtomicMonitorSnapshotV1:
    """Append-safe monitor state, separate from optimizer and checkpoint payloads."""

    identity_sha: str
    global_step: int
    status: str
    loss_components: tuple[tuple[str, float], ...]
    learning_rate: float
    gradient_norm: float | None
    unique_rows_seen: int
    row_visit_min: int
    row_visit_max: int
    schema: str = field(default="TrackBAtomicMonitorSnapshotV1", init=False)

    def __post_init__(self) -> None:
        require_sha256(self.identity_sha, "identity_sha")
        if type(self.global_step) is not int or self.global_step < 0:
            raise ValueError("monitor step must be a nonnegative integer")
        if self.status not in ("RUNNING", "COMPLETED", "FAILED"):
            raise ValueError("Track B monitor status is invalid")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0.0:
            raise ValueError("Track B monitor learning rate is invalid")
        if self.gradient_norm is not None and (not math.isfinite(self.gradient_norm) or self.gradient_norm < 0.0):
            raise ValueError("Track B monitor gradient norm is invalid")
        if min(self.unique_rows_seen, self.row_visit_min, self.row_visit_max) < 0:
            raise ValueError("Track B monitor visit counts are invalid")
        if self.row_visit_min > self.row_visit_max:
            raise ValueError("Track B monitor visit bounds are inverted")
        if any(type(name) is not str or not name or not math.isfinite(float(value)) for name, value in self.loss_components):
            raise ValueError("Track B monitor loss components are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "identity_sha": self.identity_sha,
            "global_step": self.global_step,
            "status": self.status,
            "loss_components": {name: value for name, value in self.loss_components},
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "unique_rows_seen": self.unique_rows_seen,
            "row_visit_min": self.row_visit_min,
            "row_visit_max": self.row_visit_max,
        }


def write_trackb_atomic_monitor(path: str | Path, snapshot: TrackBAtomicMonitorSnapshotV1) -> str:
    """Atomically replace the current monitor JSON and return its file SHA-256."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return _file_sha256(destination)


def _cuda_driver_version() -> str:
    """Read a CUDA driver identity or fail closed before an exact CUDA resume."""
    _require_torch()
    getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if callable(getter):
        try:
            value = int(getter())
        except (RuntimeError, TypeError, ValueError):
            value = 0
        if value > 0:
            return f"torch_cuda_api_{value}"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Track B cannot bind the CUDA driver version for exact resume") from error
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not versions or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", versions[0]):
        raise RuntimeError("Track B cannot bind the CUDA driver version for exact resume")
    return f"nvidia_smi_{versions[0]}"


@dataclass(frozen=True)
class TrackBRuntimeEnvironmentV1:
    """Actual runtime/device identity required for exact checkpoint resume."""

    device_type: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: str | None
    cuda_runtime_version: str | None
    cuda_driver_version: str | None
    cuda_device_index: int | None
    cuda_device_capability: tuple[int, int] | None
    cuda_total_memory_bytes: int | None
    precision: str
    amp_enabled: bool
    schema: str = field(default="TrackBRuntimeEnvironmentV1", init=False)

    def __post_init__(self) -> None:
        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError("Track B runtime environment device type is invalid")
        if type(self.torch_version) is not str or not self.torch_version:
            raise ValueError("Track B runtime environment torch version is required")
        if type(self.cuda_available) is not bool or type(self.amp_enabled) is not bool:
            raise TypeError("Track B runtime environment booleans must be exact")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("Track B runtime environment precision is invalid")
        cuda_fields = (
            self.cuda_device_name,
            self.cuda_runtime_version,
            self.cuda_driver_version,
            self.cuda_device_index,
            self.cuda_device_capability,
            self.cuda_total_memory_bytes,
        )
        if self.device_type == "cuda":
            if not self.cuda_available or type(self.cuda_device_name) is not str or not self.cuda_device_name:
                raise ValueError("CUDA runtime environment requires an available named CUDA device")
            if type(self.cuda_runtime_version) is not str or not self.cuda_runtime_version:
                raise ValueError("CUDA runtime environment requires its runtime version")
            if type(self.cuda_driver_version) is not str or not self.cuda_driver_version:
                raise ValueError("CUDA runtime environment requires its driver version")
            if type(self.cuda_device_index) is not int or self.cuda_device_index < 0:
                raise ValueError("CUDA runtime environment requires its resolved device index")
            if type(self.cuda_device_capability) is not tuple or len(self.cuda_device_capability) != 2 \
                    or any(type(value) is not int or value < 0 for value in self.cuda_device_capability):
                raise ValueError("CUDA runtime environment requires its device capability")
            if type(self.cuda_total_memory_bytes) is not int or self.cuda_total_memory_bytes <= 0:
                raise ValueError("CUDA runtime environment requires its total device memory")
        elif any(value is not None for value in cuda_fields):
            raise ValueError("CPU runtime environment cannot claim CUDA runtime details")
        if self.amp_enabled != (self.device_type == "cuda" and self.precision == "bf16"):
            raise ValueError("Track B runtime AMP state disagrees with device and precision")

    @classmethod
    def from_runtime(cls, *, device: "torch.device", config: TrackBRealRuntimeConfigV1) -> "TrackBRuntimeEnvironmentV1":
        _require_torch()
        cuda_available = bool(torch.cuda.is_available())
        if device.type == "cuda":
            device_index = torch.cuda.current_device() if device.index is None else device.index
            properties = torch.cuda.get_device_properties(device_index)
            driver_version = _cuda_driver_version()
            runtime_version = str(getattr(torch.version, "cuda", "") or "")
            return cls(
                device_type=device.type,
                torch_version=str(torch.__version__),
                cuda_available=cuda_available,
                cuda_device_name=str(properties.name),
                cuda_runtime_version=runtime_version,
                cuda_driver_version=driver_version,
                cuda_device_index=int(device_index),
                cuda_device_capability=(int(properties.major), int(properties.minor)),
                cuda_total_memory_bytes=int(properties.total_memory),
                precision=config.precision,
                amp_enabled=config.precision == "bf16",
            )
        return cls(
            device_type=device.type,
            torch_version=str(torch.__version__),
            cuda_available=cuda_available,
            cuda_device_name=None,
            cuda_runtime_version=None,
            cuda_driver_version=None,
            cuda_device_index=None,
            cuda_device_capability=None,
            cuda_total_memory_bytes=None,
            precision=config.precision,
            amp_enabled=False,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "device_type": self.device_type,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "cuda_device_index": self.cuda_device_index,
            "cuda_device_capability": self.cuda_device_capability,
            "cuda_total_memory_bytes": self.cuda_total_memory_bytes,
            "precision": self.precision,
            "amp_enabled": self.amp_enabled,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


class TrackBRealTrainingRuntimeV1:
    """Step-granular AdamW runtime with exact identity-bound resume semantics.

    Constructing this runtime does not read a data root, write an artifact, or
    start optimization.  ``run`` is the only method that updates parameters and
    must receive caller-supplied, already-admitted Track B batches.
    """

    CHECKPOINT_SCHEMA = "TrackBRealTrainingCheckpointV1"
    RECEIPT_SCHEMA = "TrackBRealTrainingReceiptV1"
    MODEL_CONFIG_SCHEMA = "TrainableMaskedObservationPlateConfigV1"
    LOSS_CONFIG_SCHEMA = "TrackBRealStaticLossWeightsV1"

    def __init__(
        self, *, model: TrainableMaskedObservationPlateV1,
        identity: TrackBRealRuntimeIdentityV1, config: TrackBRealRuntimeConfigV1,
        loss_weights: TrackBRealStaticLossWeightsV1 = TrackBRealStaticLossWeightsV1(),
        device: str = "cpu",
    ) -> None:
        _require_torch()
        identity.__post_init__()
        config.validate()
        loss_weights.validate()
        if model.config.schema != self.MODEL_CONFIG_SCHEMA or loss_weights.schema != self.LOSS_CONFIG_SCHEMA:
            raise ValueError("Track B runtime model/loss version binding is invalid")
        if config.execution_mode != "pilot":
            raise PermissionError(
                "generic Track B runtime has no long-training optimizer authority; "
                "a future externally verified gate service must own that path"
            )
        if model.config.system_type != identity.system_type:
            raise ValueError("Track B runtime model system does not match its identity")
        if model.config.sha256() != identity.model_config_sha:
            raise ValueError("Track B runtime model config SHA is not bound to its identity")
        if identity.batch_plan_sample_ids and config.batch_size != identity.batch_plan_batch_size:
            raise ValueError("Track B runtime config batch size differs from its immutable batch plan")
        self._model = model
        self._identity = identity
        self._config = config
        self._loss_weights = loss_weights
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Track B requested CUDA runtime but CUDA is unavailable")
        if trackb_model_state_sha(model) != identity.initial_model_state_sha:
            raise ValueError("Track B model initial state does not match runtime identity")
        caller_rng_state = _capture_rng_state()
        seed_trackb_runtime(identity.seed)
        self._runtime_rng_state = _capture_rng_state()
        _restore_rng_state(caller_rng_state)
        self._model.to(self._device)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=max(config.optimizer_steps, 1),
        )
        self._amp_enabled = config.precision == "bf16" and self._device.type == "cuda"
        self._runtime_environment = TrackBRuntimeEnvironmentV1.from_runtime(
            device=self._device, config=config,
        )
        self._scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.global_step = 0
        self.row_visits: dict[str, int] = {}
        self.last_loss_components: dict[str, float] = {}
        self.last_gradient_norm: float | None = None
        self.long_authorization_sha: str | None = None
        self._assert_runtime_transition_semantics()
        self._controlled_execution_state_sha = self._execution_state_sha()

    def _assert_runtime_transition_semantics(self) -> None:
        """Validate state shape, finiteness and cursor semantics before sealing."""
        _model_optimizer_topology_sha(self._model, self._optimizer)
        _assert_finite_runtime_state(self._model.state_dict(), name="model")
        _assert_finite_runtime_state(self._optimizer.state_dict(), name="optimizer")
        _assert_finite_runtime_state(self._scheduler.state_dict(), name="scheduler")
        _assert_finite_runtime_state(self._scaler.state_dict(), name="scaler")
        _assert_finite_runtime_state(self._runtime_rng_state, name="RNG")
        if self._scheduler.T_max != self._config.optimizer_steps:
            raise ValueError("Track B scheduler horizon differs from the sealed optimizer budget")
        if self._scheduler.last_epoch != self.global_step \
                or getattr(self._scheduler, "_step_count", None) != self.global_step + 1:
            raise ValueError("Track B scheduler cursor does not match the committed optimizer step")
        learning_rates = tuple(float(group["lr"]) for group in self._optimizer.param_groups)
        last_learning_rates = tuple(float(value) for value in self._scheduler.get_last_lr())
        if len(learning_rates) != len(last_learning_rates) or any(
            not math.isfinite(value) or value < 0.0
            for value in (*learning_rates, *last_learning_rates)
        ):
            raise ValueError("Track B optimizer/scheduler learning rates are invalid")
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15)
            for left, right in zip(learning_rates, last_learning_rates, strict=True)
        ):
            raise ValueError("Track B scheduler learning rates differ from optimizer parameter groups")

    def _execution_state_sha(self) -> str:
        forward_function = getattr(self._model.forward, "__func__", self._model.forward)
        forward_code = getattr(forward_function, "__code__", None)
        return canonical_sha256({
            "schema": "TrackBControlledExecutionStateV1",
            "identity_sha": self._identity.sha256(),
            "runtime_config_sha": self._config.sha256(),
            "loss_weights_sha": canonical_sha256(self._loss_weights.__dict__),
            "runtime_environment_sha": self._runtime_environment.sha256(),
            "model_state_sha": _state_dict_sha(self._model.state_dict()),
            "model_config_sha": self._model.config.sha256(),
            "model_forward_code_sha": (
                None if forward_code is None else hashlib.sha256(forward_code.co_code).hexdigest()
            ),
            "amp_enabled": self._amp_enabled,
            "global_step": self.global_step,
            "row_visits": dict(sorted(self.row_visits.items())),
            "last_loss_components": dict(sorted(self.last_loss_components.items())),
            "last_gradient_norm": self.last_gradient_norm,
            "long_authorization_sha": self.long_authorization_sha,
            "model_optimizer_topology_sha": _model_optimizer_topology_sha(
                self._model, self._optimizer,
            ),
            "optimizer_state_sha": _runtime_state_sha(self._optimizer.state_dict()),
            "scheduler_state_sha": _runtime_state_sha(self._scheduler.state_dict()),
            "scaler_state_sha": _runtime_state_sha(self._scaler.state_dict()),
            "rng_state_sha": _runtime_state_sha(self._runtime_rng_state),
        })

    def _assert_controlled_execution_state(self) -> None:
        if self._execution_state_sha() != self._controlled_execution_state_sha:
            raise PermissionError(
                "Track B model/optimizer state changed outside the controlled pilot transition"
            )

    def _seal_controlled_execution_state(self) -> None:
        self._controlled_execution_state_sha = self._execution_state_sha()

    @contextmanager
    def _runtime_rng_scope(self) -> Iterable[None]:
        """Execute one transition with runtime-owned RNG state only."""
        caller_rng_state = _capture_rng_state()
        _restore_rng_state(self._runtime_rng_state)
        try:
            yield
        finally:
            self._runtime_rng_state = _capture_rng_state()
            _restore_rng_state(caller_rng_state)

    def scheduler_state_for_diagnostics(self) -> dict[str, object]:
        """Return a detached scheduler snapshot without exposing a mutable scheduler."""
        self._assert_controlled_execution_state()
        return copy.deepcopy(self._scheduler.state_dict())

    def _compute_training_loss(self, prediction: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
        """Version-owned objective; subclasses must retain the admitted loop."""
        return trackb_masked_real_static_objective(prediction, batch, weights=self._loss_weights)

    def _validate_checkpoint_extension(self, metadata: Mapping[str, Any]) -> None:
        """Validate version-specific metadata before any restore side effect.

        Historical V1 has no configuration preimages beyond its model configuration.
        New checkpoint versions may require them without reinterpreting old bytes.
        """

    def _checkpoint_payload(self, *, completed: bool) -> dict[str, Any]:
        self._assert_controlled_execution_state()
        model_state = self._model.state_dict()
        model_sha = _state_dict_sha(model_state)
        metadata = {
            "schema": self.CHECKPOINT_SCHEMA,
            "identity": self._identity.to_dict(),
            "identity_sha": self._identity.sha256(),
            "model_config": self._model.config.to_dict(),
            "model_config_sha": self._model.config.sha256(),
            "runtime_config_sha": self._config.sha256(),
            "runtime_environment": self._runtime_environment.to_dict(),
            "runtime_environment_sha": self._runtime_environment.sha256(),
            "loss_weights_sha": canonical_sha256(self._loss_weights.__dict__),
            "model_state_sha": model_sha,
            "global_step": self.global_step,
            "completed": bool(completed),
            "research_only": True,
            "selection_evidence": False,
            "hardware_action_authorized": False,
            "long_training_started": self._config.execution_mode == "long",
            "long_authorization_sha": self.long_authorization_sha,
        }
        return {
            "metadata": metadata,
            "model_state_dict": model_state,
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler.state_dict(),
            "scaler_state_dict": self._scaler.state_dict(),
            "rng_state": copy.deepcopy(self._runtime_rng_state),
            "row_visits": dict(self.row_visits),
            "last_loss_components": dict(self.last_loss_components),
            "last_gradient_norm": self.last_gradient_norm,
        }

    def save_checkpoint(self, path: str | Path, *, completed: bool = False) -> dict[str, str]:
        """Write an atomic step checkpoint after all state is internally consistent."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint_payload(completed=completed), temporary)
        os.replace(temporary, destination)
        return {
            "path": str(destination), "sha256": _file_sha256(destination),
            "model_state_sha": _state_dict_sha(self._model.state_dict()),
        }

    def resume_checkpoint(self, path: str | Path, *, expected_sha256: str) -> None:
        """Restore optimizer/scheduler/scaler/RNG at the exact next optimizer step."""
        source = Path(path)
        self._assert_controlled_execution_state()
        if not source.is_file():
            raise ValueError("Track B resume checkpoint is missing")
        require_sha256(expected_sha256, "expected_sha256")
        payload_bytes = source.read_bytes()
        observed_sha = hashlib.sha256(payload_bytes).hexdigest()
        if observed_sha != expected_sha256:
            raise ValueError("Track B resume checkpoint file SHA mismatch")
        payload = torch.load(io.BytesIO(payload_bytes), map_location=self._device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("Track B safe checkpoint payload is not a mapping")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("schema") != self.CHECKPOINT_SCHEMA:
            raise ValueError("Track B resume checkpoint schema is invalid")
        self._validate_checkpoint_extension(metadata)
        expected = {
            "identity_sha": self._identity.sha256(),
            "model_config_sha": self._model.config.sha256(),
            "runtime_config_sha": self._config.sha256(),
            "loss_weights_sha": canonical_sha256(self._loss_weights.__dict__),
            "runtime_environment_sha": self._runtime_environment.sha256(),
        }
        mismatches = [name for name, value in expected.items() if metadata.get(name) != value]
        if mismatches:
            raise ValueError(f"Track B resume identity mismatch: {mismatches}")
        if metadata.get("runtime_environment") != self._runtime_environment.to_dict():
            raise ValueError("Track B resume device/runtime environment changed")
        try:
            restored_model_config = self._model.config.from_dict(metadata.get("model_config", {}))
        except (TypeError, ValueError) as error:
            raise ValueError("Track B resume checkpoint model config is invalid") from error
        if restored_model_config.sha256() != metadata.get("model_config_sha"):
            raise ValueError("Track B resume checkpoint model config SHA is invalid")
        if restored_model_config != self._model.config:
            raise ValueError("Track B resume checkpoint model config differs from the runtime model")
        if metadata.get("hardware_action_authorized") is not False or metadata.get("research_only") is not True:
            raise ValueError("Track B resume checkpoint safety authority changed")
        authorization_sha = metadata.get("long_authorization_sha")
        if self._config.execution_mode == "long":
            require_sha256(authorization_sha, "long_authorization_sha")
            self.long_authorization_sha = authorization_sha
        elif authorization_sha is not None:
            raise ValueError("pilot checkpoint cannot carry a long-training authorization")
        model_state = payload.get("model_state_dict")
        optimizer_state = payload.get("optimizer_state_dict")
        scheduler_state = payload.get("scheduler_state_dict")
        scaler_state = payload.get("scaler_state_dict")
        if not isinstance(model_state, Mapping) \
                or not isinstance(optimizer_state, Mapping) \
                or not isinstance(scheduler_state, Mapping) \
                or not isinstance(scaler_state, Mapping):
            raise ValueError("Track B resume checkpoint state payload is incomplete")
        expected_model_state = self._model.state_dict()
        if set(model_state) != set(expected_model_state) or any(
            not isinstance(model_state[name], torch.Tensor)
            or model_state[name].shape != expected_model_state[name].shape
            or model_state[name].dtype != expected_model_state[name].dtype
            for name in expected_model_state
        ):
            raise ValueError("Track B resume model-state schema does not match the runtime model")
        if _state_dict_sha(model_state) != metadata.get("model_state_sha"):
            raise ValueError("Track B resume model-state SHA mismatch")
        step = metadata.get("global_step")
        if type(step) is not int or step < 0 or step > self._config.optimizer_steps:
            raise ValueError("Track B resume global step is invalid")
        visits = payload.get("row_visits")
        if not isinstance(visits, dict) or any(type(key) is not str or type(value) is not int or value < 0 for key, value in visits.items()):
            raise ValueError("Track B resume row visits are invalid")
        last_loss_components = payload.get("last_loss_components", {})
        if not isinstance(last_loss_components, Mapping) or any(
            type(key) is not str or not math.isfinite(float(value))
            for key, value in last_loss_components.items()
        ):
            raise ValueError("Track B resume loss components are invalid")
        last_gradient_norm = payload.get("last_gradient_norm")
        if last_gradient_norm is not None and (
            not isinstance(last_gradient_norm, (int, float))
            or not math.isfinite(float(last_gradient_norm))
            or float(last_gradient_norm) < 0.0
        ):
            raise ValueError("Track B resume gradient norm is invalid")
        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("Track B resume RNG state is invalid")
        _validated_rng_state(rng_state)
        previous_model_state = copy.deepcopy(self._model.state_dict())
        previous_optimizer_state = copy.deepcopy(self._optimizer.state_dict())
        previous_scheduler_state = copy.deepcopy(self._scheduler.state_dict())
        previous_scaler_state = copy.deepcopy(self._scaler.state_dict())
        previous_rng_state = copy.deepcopy(self._runtime_rng_state)
        previous_global_step = self.global_step
        previous_row_visits = dict(self.row_visits)
        previous_loss_components = dict(self.last_loss_components)
        previous_gradient_norm = self.last_gradient_norm
        previous_authorization_sha = self.long_authorization_sha
        try:
            self._model.load_state_dict(model_state, strict=True)
            self._optimizer.load_state_dict(optimizer_state)
            self._scheduler.load_state_dict(scheduler_state)
            self._scaler.load_state_dict(scaler_state)
            self._runtime_rng_state = copy.deepcopy(dict(rng_state))
            self.global_step = step
            self.row_visits = dict(visits)
            self.last_loss_components = {str(key): float(value) for key, value in last_loss_components.items()}
            self.last_gradient_norm = None if last_gradient_norm is None else float(last_gradient_norm)
            self._assert_runtime_transition_semantics()
            self._seal_controlled_execution_state()
        except BaseException:
            self._model.load_state_dict(previous_model_state, strict=True)
            self._optimizer.load_state_dict(previous_optimizer_state)
            self._scheduler.load_state_dict(previous_scheduler_state)
            self._scaler.load_state_dict(previous_scaler_state)
            self._runtime_rng_state = previous_rng_state
            self.global_step = previous_global_step
            self.row_visits = previous_row_visits
            self.last_loss_components = previous_loss_components
            self.last_gradient_norm = previous_gradient_norm
            self.long_authorization_sha = previous_authorization_sha
            self._seal_controlled_execution_state()
            raise

    def monitor_snapshot(self, *, status: str = "RUNNING") -> TrackBAtomicMonitorSnapshotV1:
        self._assert_controlled_execution_state()
        visits = tuple(self.row_visits.values())
        return TrackBAtomicMonitorSnapshotV1(
            identity_sha=self._identity.sha256(), global_step=self.global_step, status=status,
            loss_components=tuple(sorted(self.last_loss_components.items())),
            learning_rate=float(self._optimizer.param_groups[0]["lr"]),
            gradient_norm=self.last_gradient_norm, unique_rows_seen=len(self.row_visits),
            row_visit_min=min(visits) if visits else 0,
            row_visit_max=max(visits) if visits else 0,
        )

    def _step(self, raw_batch: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_controlled_execution_state()
        if self._config.execution_mode != "pilot":
            raise PermissionError("generic Track B optimizer accepts pilot execution only")
        if self.global_step >= self._config.optimizer_steps \
                or self.global_step >= MAX_SMOKE_OPTIMIZER_STEPS:
            raise RuntimeError("Track B optimizer step budget is exhausted before update")
        self._validate_batch_identity(raw_batch)
        batch = _move_to_device(raw_batch, self._device)
        previous_model_state = copy.deepcopy(self._model.state_dict())
        previous_optimizer_state = copy.deepcopy(self._optimizer.state_dict())
        previous_scheduler_state = copy.deepcopy(self._scheduler.state_dict())
        previous_rng_state = copy.deepcopy(self._runtime_rng_state)
        previous_global_step = self.global_step
        previous_row_visits = dict(self.row_visits)
        previous_loss_components = dict(self.last_loss_components)
        previous_gradient_norm = self.last_gradient_norm
        try:
            with self._runtime_rng_scope():
                self._model.train(True)
                self._optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self._device.type, dtype=torch.bfloat16, enabled=self._amp_enabled,
                ):
                    prediction = self._model(batch)
                    losses = self._compute_training_loss(prediction, batch)
                    loss = losses["loss"]
                if loss.ndim != 0 or not bool(torch.isfinite(loss).detach()):
                    raise FloatingPointError("Track B real-static runtime produced a non-finite loss")
                loss.backward()
                gradients = [parameter.grad for parameter in self._model.parameters() if parameter.grad is not None]
                if not gradients:
                    raise RuntimeError("Track B native objective produced no gradients")
                if not all(bool(torch.all(torch.isfinite(value)).detach()) for value in gradients):
                    raise FloatingPointError("Track B real-static runtime produced a non-finite gradient")
                norm = torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self._config.gradient_clip_norm,
                )
                if not bool(torch.isfinite(norm).detach()):
                    raise FloatingPointError("Track B gradient norm became non-finite")
                self._optimizer.step()
                self._scheduler.step()
                self.global_step += 1
                self.last_gradient_norm = float(norm.detach().cpu())
                self.last_loss_components = {
                    name: float(value.detach().cpu())
                    for name, value in losses.items() if isinstance(value, torch.Tensor) and value.ndim == 0
                }
                sample_ids = raw_batch["sample_ids"]
                for sample_id in sample_ids:
                    self.row_visits[sample_id] = self.row_visits.get(sample_id, 0) + 1
            self._assert_runtime_transition_semantics()
            self._seal_controlled_execution_state()
        except BaseException:
            self._model.load_state_dict(previous_model_state, strict=True)
            self._optimizer.load_state_dict(previous_optimizer_state)
            self._scheduler.load_state_dict(previous_scheduler_state)
            self._runtime_rng_state = previous_rng_state
            self.global_step = previous_global_step
            self.row_visits = previous_row_visits
            self.last_loss_components = previous_loss_components
            self.last_gradient_norm = previous_gradient_norm
            self._seal_controlled_execution_state()
            raise
        return losses

    def _validate_batch_identity(self, raw_batch: Mapping[str, Any]) -> None:
        """Validate source row identities before any optimizer side effect."""
        sample_ids = raw_batch.get("sample_ids")
        if (
            not isinstance(sample_ids, tuple)
            or not sample_ids
            or any(type(value) is not str or not value for value in sample_ids)
            or len(set(sample_ids)) != len(sample_ids)
        ):
            raise ValueError("Track B batch must preserve unique source sample IDs for monitoring")
        if raw_batch.get("system_type") != self._identity.system_type:
            raise ValueError("Track B batch system_type does not match runtime identity")
        counts = raw_batch.get("four_cell_counts")
        if not isinstance(counts, torch.Tensor) or counts.ndim < 1:
            raise ValueError("Track B batch must expose a leading row dimension")
        if counts.shape[0] != len(sample_ids):
            raise ValueError("Track B sample IDs do not align with the batch row dimension")
        if not set(sample_ids).issubset(set(self._identity.allowed_sample_ids)):
            raise ValueError("Track B batch contains a sample outside the admitted train fold")
        for name in ("query_mask", "operation_mask"):
            value = raw_batch.get(name)
            if isinstance(value, torch.Tensor) and (
                value.ndim < 1 or value.shape[0] != len(sample_ids)
            ):
                raise ValueError(f"Track B {name} does not align with source sample IDs")
        query_mask = raw_batch.get("query_mask")
        operation_mask = raw_batch.get("operation_mask")
        visible_operation_mask = raw_batch.get("visible_operation_mask")
        if not isinstance(query_mask, torch.Tensor) or not isinstance(operation_mask, torch.Tensor) \
                or not isinstance(visible_operation_mask, torch.Tensor):
            raise ValueError("Track B batch requires query/operation/visible masks")
        if query_mask.shape != operation_mask.shape \
                or visible_operation_mask.shape != operation_mask.shape:
            raise ValueError("Track B query/operation/visible masks must share one shape")
        operation_ids = raw_batch.get("operation_ids")
        operation_families = raw_batch.get("operation_families")
        cycle_indices = raw_batch.get("cycle_indices")
        epoch = raw_batch.get("epoch")
        view_seed = raw_batch.get("view_seed")
        query_fraction = raw_batch.get("query_fraction")
        if not isinstance(operation_ids, tuple) or not isinstance(operation_families, tuple) \
                or not isinstance(cycle_indices, tuple) \
                or len(operation_ids) != operation_mask.shape[1] \
                or len(operation_families) != operation_mask.shape[1] \
                or len(cycle_indices) != operation_mask.shape[1] \
                or type(epoch) is not int or type(view_seed) is not int \
                or not isinstance(query_fraction, (float, int)) \
                or not math.isfinite(float(query_fraction)) \
                or not 0.0 < float(query_fraction) < 1.0:
            raise ValueError("Track B batch deterministic query-plan metadata is invalid")
        expected_query_mask = torch.stack([
            torch.from_numpy(deterministic_query_mask(
                sample_id=sample_id, epoch=epoch, view_seed=view_seed,
                operation_ids=operation_ids, operation_families=operation_families,
                cycle_indices=cycle_indices, query_fraction=float(query_fraction),
            ))
            for sample_id in sample_ids
        ]).to(dtype=torch.bool)
        if not torch.equal(query_mask.to(dtype=torch.bool).cpu(), expected_query_mask):
            raise ValueError("Track B batch query mask differs from the deterministic source plan")
        expected_visible = operation_mask.to(dtype=torch.bool) & ~query_mask.to(dtype=torch.bool)
        if not torch.equal(visible_operation_mask.to(dtype=torch.bool), expected_visible):
            raise ValueError("Track B visible operation mask differs from the deterministic query plan")
        expected_query_mask_sha = canonical_sha256({
            "schema": "TrackBDeterministicQueryMaskV1",
            "sample_ids": list(sample_ids),
            "epoch": epoch,
            "view_seed": view_seed,
            "query_fraction": float(query_fraction),
            "masks": expected_query_mask.to(dtype=torch.int8).tolist(),
        })
        if raw_batch.get("query_mask_sha") != expected_query_mask_sha:
            raise ValueError("Track B batch query-mask identity does not match the deterministic plan")
        group_ids = raw_batch.get("group_ids")
        if (
            not isinstance(group_ids, tuple)
            or len(group_ids) != len(sample_ids)
            or any(type(value) is not str or not value for value in group_ids)
        ):
            raise ValueError("Track B group IDs must align with source sample IDs")
        if not set(group_ids).issubset(set(self._identity.allowed_group_ids)):
            raise ValueError("Track B batch contains a group outside the admitted train fold")
        source_roles = raw_batch.get("source_roles")
        if (
            not isinstance(source_roles, tuple)
            or len(source_roles) != len(sample_ids)
            or any(value not in self._identity.allowed_roles for value in source_roles)
            or any(value != "train" for value in source_roles)
        ):
            raise ValueError("Track B batch source roles are not admitted for training")
        if raw_batch.get("admission_manifest_sha") != self._identity.admission_manifest_sha:
            raise ValueError("Track B batch admission manifest identity does not match runtime")
        if raw_batch.get("outer_fold_sha") != self._identity.outer_fold_sha:
            raise ValueError("Track B batch outer fold identity does not match runtime")
        if raw_batch.get("row_identity_sha") != self._identity.row_identity_sha:
            raise ValueError("Track B batch row identity manifest does not match runtime")
        if raw_batch.get("batch_plan_sha") != self._identity.batch_plan_sha:
            raise ValueError("Track B batch plan identity does not match runtime")
        batches_per_pass = max(
            1, (len(self._identity.batch_plan_sample_ids) + self._identity.batch_plan_batch_size - 1)
            // self._identity.batch_plan_batch_size,
        )
        expected_pass_index = self.global_step // batches_per_pass
        expected_batch_index = self.global_step % batches_per_pass
        expected_start = expected_batch_index * self._identity.batch_plan_batch_size
        expected_ids = self._identity.batch_plan_sample_ids[
            expected_start:expected_start + self._identity.batch_plan_batch_size
        ]
        if sample_ids != expected_ids:
            raise ValueError("Track B batch sample order/cursor differs from the immutable batch plan")
        if epoch != expected_pass_index or raw_batch.get("batch_index") != expected_batch_index \
                or raw_batch.get("cursor_start") != expected_start \
                or raw_batch.get("cursor_end") != expected_start + len(expected_ids):
            raise ValueError("Track B batch pass/index/cursor differs from the immutable batch plan")
        if view_seed != self._identity.batch_plan_view_seed \
                or float(query_fraction) != float(self._identity.batch_plan_query_fraction):
            raise ValueError("Track B batch query settings differ from the immutable batch plan")
        row_content_shas = raw_batch.get("row_content_shas")
        raw_status = raw_batch.get("raw_rf_status")
        global_scopes = raw_batch.get("global_multi_scope")
        if (
            not isinstance(row_content_shas, tuple)
            or len(row_content_shas) != len(sample_ids)
            or any(type(value) is not str or not value for value in row_content_shas)
            or not isinstance(raw_status, tuple)
            or len(raw_status) != len(sample_ids)
            or not isinstance(global_scopes, tuple)
            or len(global_scopes) != len(sample_ids)
        ):
            raise ValueError("Track B batch row content identities are incomplete")
        expected_rows = {
            record[0]: record for record in self._identity.allowed_row_identities
        }
        session_index = raw_batch.get("session_index")
        session_known_mask = raw_batch.get("session_known_mask")
        if not isinstance(session_index, torch.Tensor) or session_index.shape != (len(sample_ids),) \
                or not isinstance(session_known_mask, torch.Tensor) \
                or session_known_mask.shape != (len(sample_ids),):
            raise ValueError("Track B batch session identity does not align with source rows")
        expected_sessions = dict(self._identity.session_index_bindings)
        actual_sessions = tuple(int(value) for value in session_index.detach().cpu().tolist())
        if actual_sessions != tuple(expected_sessions[sample_id] for sample_id in sample_ids) \
                or not bool(torch.all(session_known_mask.to(dtype=torch.bool)).item()):
            raise ValueError("Track B batch session mapping differs from the admitted source mapping")
        for index, sample_id in enumerate(sample_ids):
            expected = expected_rows.get(sample_id)
            if expected is None or expected[1] != group_ids[index] or expected[3] != source_roles[index]:
                raise ValueError("Track B batch sample/group/role mapping is not admitted")
            if row_content_shas[index] != expected[2]:
                raise ValueError("Track B batch row content identity does not match admission")
            actual_content_sha = source_row_content_sha(
                _batch_row_identity_values(raw_batch, index),
            )
            if actual_content_sha != expected[2]:
                raise ValueError("Track B batch row content changed after admission")

    def run(
        self, batch_iterator_factory: Callable[[int], Iterable[Mapping[str, Any]]], *,
        checkpoint_path: str | Path, monitor_path: str | Path,
        long_authorization: TrackBLongTrainingAuthorizationV1 | None = None,
        approval_receipt_path: str | Path | None = None,
        step_observer: Callable[[int, Mapping[str, Any]], None] | None = None,
    ) -> dict[str, object]:
        """Execute the explicitly configured real-static runtime.

        This method is intentionally not called by module import, CLI parsing,
        or any smoke helper.  A caller must first pass the V2.4 admission/fold
        gates and explicitly invoke it.
        """
        self._config.validate()
        if self._config.execution_mode == "long":
            raise PermissionError(
                "generic Track B runtime has no long-training execution authority; "
                "a future externally verified gate service must own that path"
            )
        elif long_authorization is not None:
            raise ValueError("pilot runtime cannot consume a long-training authorization artifact")
        if self.global_step >= self._config.optimizer_steps:
            raise ValueError("Track B runtime is already at or beyond its configured step budget")
        iterator: Iterable[Mapping[str, Any]] | None = None
        iterator_object = None
        empty_iterator_count = 0
        while self.global_step < self._config.optimizer_steps:
            if iterator_object is None:
                iterator = batch_iterator_factory(self.global_step)
                iterator_object = iter(iterator)
            try:
                batch = next(iterator_object)
            except StopIteration:
                iterator_object = None
                empty_iterator_count += 1
                if empty_iterator_count > 1:
                    raise ValueError("Track B batch iterator factory produced no training batches")
                continue
            empty_iterator_count = 0
            try:
                losses = self._step(batch)
                if step_observer is not None:
                    step_observer(self.global_step, losses)
                    self._assert_controlled_execution_state()
            except Exception:
                write_trackb_atomic_monitor(monitor_path, self.monitor_snapshot(status="FAILED"))
                raise
            if self.global_step % self._config.monitor_interval_steps == 0:
                write_trackb_atomic_monitor(monitor_path, self.monitor_snapshot(status="RUNNING"))
            if self.global_step % self._config.checkpoint_interval_steps == 0:
                self.save_checkpoint(checkpoint_path, completed=False)
        checkpoint = self.save_checkpoint(checkpoint_path, completed=True)
        monitor_sha = write_trackb_atomic_monitor(monitor_path, self.monitor_snapshot(status="COMPLETED"))
        return {
            "schema": self.RECEIPT_SCHEMA,
            "identity_sha": self._identity.sha256(),
            "runtime_config_sha": self._config.sha256(),
            "committed_optimizer_steps": self.global_step,
            "unique_rows_seen": len(self.row_visits),
            "row_visit_min": min(self.row_visits.values()) if self.row_visits else 0,
            "row_visit_max": max(self.row_visits.values()) if self.row_visits else 0,
            "checkpoint": checkpoint,
            "monitor_sha256": monitor_sha,
            "research_only": True,
            "selection_evidence": False,
            "hardware_action_authorized": False,
            "long_training_started": self._config.execution_mode == "long",
        }


__all__ = [
    "MAX_SMOKE_OPTIMIZER_STEPS", "TORCH_AVAILABLE", "TrackBAtomicMonitorSnapshotV1",
    "TrackBRealBatchPlanV1", "TrackBRealRuntimeConfigV1", "TrackBRealRuntimeIdentityV1", "TrackBLongTrainingAuthorizationV1",
    "TrackBRealTrainingRuntimeV1",
    "TrackBThreeSeedEnsemblePlanV1",
    "TrackBSmokeStateV1", "TrackBSmokeTrainingReceiptV1", "run_short_smoke_training",
    "smoke_optimizer_step", "write_trackb_atomic_monitor",
    "seed_trackb_runtime", "trackb_model_state_sha",
]
