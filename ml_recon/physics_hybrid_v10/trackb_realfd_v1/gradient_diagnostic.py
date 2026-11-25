"""Frozen endpoint gradient geometry for a bound Track B pilot.

This diagnostic uses ``torch.autograd.grad`` on deterministic TRAIN batches to
measure local action/gain gradient geometry.  It never constructs an optimizer,
writes parameter gradients, updates model state, or represents historical
training dynamics.  The result is local endpoint evidence only.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Sequence

import math

import torch
from torch.utils.data import DataLoader

from ml_recon.data_v2_4.loaders import V24TrainingDataset

from .contracts import canonical_sha256
from .diagnostic_runtime import preserve_python_numpy_torch_rng
from .inverse_transformer import LEGACY_LINEAR_MEAN_ACTION_VALUE
from .inverse_transformer_pilot import (
    STANDARDIZED_HUBER_DELTA,
    TEN_ROUND_COUNT,
    TrackBTargetNormalizationV1,
    _logical_action_loss_components,
    _round_inputs,
)
from .policy_ablation import (
    _bound_v24_source,
    _dataset_source_identity,
    _load_bound_model,
    _matching_trackb_cell,
    _require_matching_source_identity,
    _require_oracle_history_reader,
)
from .recomputable_diagnostic import (
    _normalization_from_bound_records,
    _publish_ad1_report,
    _require_recomputed_train_normalization,
)
from .recomputable_metrics import _require_strict_normalization_object
from .train_fit_diagnostic import (
    _bound_batch_plan,
    _checkpoint_path_from_receipt,
    _diagnostic_repository_identity,
    _prepare_new_output_path,
    _read_verified_pilot_receipt,
    _require_isolated_report_output_path,
)
from .training import trackb_model_state_sha


FROZEN_GRADIENT_DIAGNOSTIC_SCHEMA = "TrackBFrozenGradientDiagnosticV1"
LOCAL_ENDPOINT_GRADIENT_SCOPE = "FINAL_CHECKPOINT_LOCAL_TRAIN_GRADIENT_NOT_TRAINING_HISTORY"
NOT_RECORDED = "NOT_RECORDED"


def _snapshot_model_runtime(model: torch.nn.Module) -> dict[str, object]:
    mappings: list[tuple[torch.nn.Module, str, dict[str, object]]] = []
    entries: list[tuple[torch.nn.Module, str, str, object, object]] = []
    for module in model.modules():
        parameter_mapping = dict(module._parameters)
        buffer_mapping = dict(module._buffers)
        mappings.extend((
            (module, "parameter", parameter_mapping),
            (module, "buffer", buffer_mapping),
        ))
        for name, parameter in parameter_mapping.items():
            if parameter is not None:
                entries.append((module, name, "parameter", parameter, parameter.detach().clone()))
        for name, buffer in buffer_mapping.items():
            if buffer is not None:
                value = buffer.detach().clone() if torch.is_tensor(buffer) else copy.deepcopy(buffer)
                entries.append((module, name, "buffer", buffer, value))
    gradients = {
        id(parameter): (
            parameter,
            parameter.grad,
            parameter.grad.detach().clone() if torch.is_tensor(parameter.grad) else copy.deepcopy(parameter.grad),
            bool(parameter.requires_grad),
        )
        for module in model.modules()
        for parameter in module._parameters.values()
        if parameter is not None
    }
    return {
        "mappings": mappings,
        "entries": entries,
        "gradients": gradients,
        "module_modes": {module: bool(module.training) for module in model.modules()},
        "non_persistent_buffers": {
            module: set(module._non_persistent_buffers_set)
            for module in model.modules()
        },
    }


def _restore_model_runtime(snapshot: Mapping[str, object]) -> None:
    with torch.no_grad():
        for module, kind, original_mapping in snapshot["mappings"]:
            mapping = module._parameters if kind == "parameter" else module._buffers
            mapping.clear()
            mapping.update(original_mapping)
        for module, names in snapshot["non_persistent_buffers"].items():
            module._non_persistent_buffers_set.clear()
            module._non_persistent_buffers_set.update(names)
        for module, name, kind, original, value in snapshot["entries"]:
            if torch.is_tensor(original):
                original.copy_(value)
        for parameter, original_grad, grad_value, requires_grad in snapshot["gradients"].values():
            parameter.requires_grad_(requires_grad)
            parameter.grad = original_grad
            if torch.is_tensor(original_grad):
                original_grad.copy_(grad_value)
    for module, training in snapshot["module_modes"].items():
        module.training = training


@dataclass(frozen=True)
class TrackBFrozenGradientDiagnosticConfigV1:
    """Immutable inputs for one no-update local gradient diagnostic."""

    pilot_receipt_path: str
    expected_pilot_receipt_sha: str
    output_path: str
    batch_size: int = 8
    batch_count: int = 2
    round_indices: tuple[int, ...] = tuple(range(TEN_ROUND_COUNT))
    clip_max_norm: float = 1.0
    device: str = "cpu"
    schema: str = FROZEN_GRADIENT_DIAGNOSTIC_SCHEMA

    def validate(self) -> None:
        for name in ("pilot_receipt_path", "output_path"):
            if not Path(str(getattr(self, name))).is_absolute():
                raise ValueError(f"{name} must be absolute")
        if type(self.expected_pilot_receipt_sha) is not str or len(self.expected_pilot_receipt_sha) != 64:
            raise ValueError("expected_pilot_receipt_sha must be a SHA-256 hex value")
        try:
            int(self.expected_pilot_receipt_sha, 16)
        except ValueError as error:
            raise ValueError("expected_pilot_receipt_sha must be a SHA-256 hex value") from error
        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError("gradient diagnostic device is invalid") from error
        _validate_gradient_evaluator_arguments(
            batch_size=self.batch_size,
            batch_count=self.batch_count,
            round_indices=self.round_indices,
            clip_max_norm=self.clip_max_norm,
            device=device,
        )
        if self.schema != FROZEN_GRADIENT_DIAGNOSTIC_SCHEMA:
            raise ValueError("unsupported Track B gradient diagnostic schema")


def _validate_gradient_evaluator_arguments(
    *,
    batch_size: object,
    batch_count: object,
    round_indices: object,
    clip_max_norm: object,
    device: object,
) -> tuple[int, ...]:
    """Validate direct evaluator controls without coercing user values."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("gradient diagnostic batch_size is invalid")
    if type(batch_count) is not int or batch_count <= 0:
        raise ValueError("gradient diagnostic batch_count is invalid")
    if type(round_indices) not in (tuple, list) or not round_indices:
        raise ValueError("gradient diagnostic round_indices are invalid")
    if any(type(value) is not int or not 0 <= value < TEN_ROUND_COUNT for value in round_indices):
        raise ValueError("gradient diagnostic round_indices are invalid")
    if len(set(round_indices)) != len(round_indices):
        raise ValueError("gradient diagnostic round_indices are invalid")
    if (type(clip_max_norm) not in (int, float) or isinstance(clip_max_norm, bool)
            or not math.isfinite(float(clip_max_norm)) or float(clip_max_norm) <= 0.0):
        raise ValueError("gradient diagnostic clip_max_norm is invalid")
    if not isinstance(device, torch.device) or device.type != "cpu":
        raise ValueError("gradient diagnostic supports CPU only")
    return tuple(round_indices)


def _trainable_parameter_partition(
    model: torch.nn.Module,
) -> tuple[
    tuple[str, ...],
    tuple[torch.nn.Parameter, ...],
    dict[str, tuple[int, ...]],
    tuple[int, ...],
]:
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        raise ValueError("Track B gradient diagnostic found no trainable parameters")
    names = tuple(name for name, _parameter in named_parameters)
    parameters = tuple(parameter for _name, parameter in named_parameters)
    prefixes = {
        "history_encoder": "history_encoder.",
        "last_history_encoder": "last_history_encoder.",
        "prefix_length_encoder": "prefix_length_encoder.",
        "mode_encoder": "mode_encoder.",
        "inverse_transformer": "inverse_transformer.",
        "action_head": "action_head.",
        "gain_action_encoder": "gain_action_encoder.",
        "action_value_encoder": "action_value_encoder.",
        "gain_head": "gain_head.",
        "risk_head": "risk_head.",
    }
    groups = {
        group_name: tuple(
            index for index, name in enumerate(names) if name.startswith(prefix)
        )
        for group_name, prefix in prefixes.items()
    }
    groups = {name: indices for name, indices in groups.items() if indices}
    shared_names = (
        "history_encoder", "last_history_encoder", "prefix_length_encoder",
        "mode_encoder", "inverse_transformer",
    )
    shared_indices = tuple(index for name in shared_names for index in groups.get(name, ()))
    if not groups.get("action_head") or not groups.get("gain_head") or not shared_indices:
        raise ValueError("Track B gradient diagnostic parameter partition is incomplete")
    membership = [index for values in groups.values() for index in values]
    if sorted(membership) != list(range(len(parameters))) or len(set(membership)) != len(parameters):
        raise ValueError("Track B gradient diagnostic parameter partition overlaps or omits parameters")
    return names, parameters, groups, shared_indices


def _gradient_norm(
    gradients: Sequence[torch.Tensor | None],
    indices: Sequence[int],
) -> float:
    square_sum = math.fsum(
        float(gradient.detach().to(dtype=torch.float64).square().sum())
        for index in indices
        for gradient in (gradients[index],)
        if gradient is not None
    )
    return math.sqrt(max(square_sum, 0.0))


def _gradient_dot(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    indices: Sequence[int],
) -> float:
    return math.fsum(
        float((left_gradient.detach().to(dtype=torch.float64)
               * right_gradient.detach().to(dtype=torch.float64)).sum())
        for index in indices
        for left_gradient, right_gradient in ((left[index], right[index]),)
        if left_gradient is not None and right_gradient is not None
    )


def _combined_gradients(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor, ...]:
    result = []
    for left_gradient, right_gradient, parameter in zip(left, right, parameters, strict=True):
        combined = torch.zeros_like(parameter)
        if left_gradient is not None:
            combined = combined + left_gradient
        if right_gradient is not None:
            combined = combined + right_gradient
        result.append(combined)
    return tuple(result)


def _scaled_gradients(
    gradients: Sequence[torch.Tensor | None],
    scale: float,
) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if gradient is None else gradient * float(scale) for gradient in gradients)


def _require_finite_gradients(
    gradients: Sequence[torch.Tensor | None],
    *,
    label: str,
) -> None:
    if any(gradient is not None and not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError(f"Track B {label} gradients are non-finite")


def _joint_gradient_matches_sum(
    action_gradients: Sequence[torch.Tensor | None],
    gain_gradients: Sequence[torch.Tensor | None],
    joint_gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> bool:
    combined = _combined_gradients(action_gradients, gain_gradients, parameters)
    for expected, actual in zip(combined, joint_gradients, strict=True):
        actual_value = torch.zeros_like(expected) if actual is None else actual
        absolute_tolerance = max(1e-12, 4.0 * torch.finfo(expected.dtype).eps)
        if not torch.allclose(
            expected,
            actual_value,
            rtol=1e-5,
            atol=absolute_tolerance,
        ):
            return False
    return True


def _gradient_record(
    model: torch.nn.Module,
    *,
    batch: Mapping[str, Any],
    batch_index: int,
    round_index: int,
    device: torch.device,
    normalization: TrackBTargetNormalizationV1,
    gain_loss_weight: float,
    clip_max_norm: float,
    parameter_names: Sequence[str],
    parameters: Sequence[torch.nn.Parameter],
    groups: Mapping[str, Sequence[int]],
    shared_indices: Sequence[int],
) -> dict[str, object]:
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("Track B gradient diagnostic refuses pre-existing parameter gradients")
    inputs, target_action, target_gain = _round_inputs(
        batch,
        round_index=round_index,
        device=device,
    )
    output = model(inputs)
    action_loss, gain_loss, loss_metrics = _logical_action_loss_components(
        output,
        target_action,
        target_gain,
        normalization,
        round_index=round_index,
    )
    weighted_gain_loss = float(gain_loss_weight) * gain_loss
    joint_loss = action_loss + weighted_gain_loss
    action_gradients = torch.autograd.grad(
        action_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    gain_gradients_unweighted = torch.autograd.grad(
        gain_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    gain_gradients_weighted = _scaled_gradients(gain_gradients_unweighted, gain_loss_weight)
    joint_gradients = torch.autograd.grad(
        joint_loss,
        parameters,
        allow_unused=True,
    )
    if not _joint_gradient_matches_sum(
        action_gradients,
        gain_gradients_weighted,
        joint_gradients,
        parameters,
    ):
        raise ValueError("Track B joint gradient differs from its task-gradient sum")
    _require_finite_gradients(action_gradients, label="action")
    _require_finite_gradients(gain_gradients_unweighted, label="gain")
    _require_finite_gradients(joint_gradients, label="joint")
    action_shared_norm = _gradient_norm(action_gradients, shared_indices)
    gain_shared_norm_unweighted = _gradient_norm(gain_gradients_unweighted, shared_indices)
    gain_shared_norm_weighted = _gradient_norm(gain_gradients_weighted, shared_indices)
    shared_dot = _gradient_dot(action_gradients, gain_gradients_weighted, shared_indices)
    shared_denominator = action_shared_norm * gain_shared_norm_weighted
    combined = _combined_gradients(action_gradients, gain_gradients_weighted, parameters)
    joint_global_norm = _gradient_norm(combined, tuple(range(len(parameters))))
    clip_coefficient = min(1.0, float(clip_max_norm) / (joint_global_norm + 1e-6))
    sample_ids = [str(value) for value in batch.get("sample_id", ())]
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Track B gradient diagnostic batch sample identities are missing or duplicated")
    parameter_layout = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "element_count": parameter.numel(),
            "group": next(group for group, indices in groups.items() if index in indices),
        }
        for index, (name, parameter) in enumerate(zip(parameter_names, parameters, strict=True))
    ]
    return {
        "batch_index": batch_index,
        "round_index": round_index + 1,
        "sample_count": len(sample_ids),
        "sample_ids_sha": canonical_sha256(sample_ids),
        "losses": {
            **loss_metrics,
            "weighted_gain_standardized_huber": float(weighted_gain_loss.detach().cpu()),
            "joint_loss": float(joint_loss.detach().cpu()),
        },
        "shared": {
            "action_gradient_norm": action_shared_norm,
            "gain_gradient_norm_unweighted": gain_shared_norm_unweighted,
            "gain_gradient_norm_weighted": gain_shared_norm_weighted,
            "inner_product": shared_dot,
            "cosine": shared_dot / shared_denominator if shared_denominator > 0.0 else None,
            "cosine_status": "DEFINED" if shared_denominator > 0.0 else "ZERO_GRADIENT",
            "negative_inner_product": bool(shared_dot < 0.0 and shared_denominator > 0.0),
        },
        "parameter_groups": {
            group_name: {
                "parameter_count": len(indices),
                "action_gradient_norm": _gradient_norm(action_gradients, indices),
                "gain_gradient_norm_unweighted": _gradient_norm(gain_gradients_unweighted, indices),
                "gain_gradient_norm_weighted": _gradient_norm(gain_gradients_weighted, indices),
                "joint_gradient_norm": _gradient_norm(joint_gradients, indices),
            }
            for group_name, indices in groups.items()
        },
        "global": {
            "joint_gradient_norm": joint_global_norm,
            "clip_max_norm": float(clip_max_norm),
            "clip_coefficient": clip_coefficient,
            "joint_gradient_matches_task_sum": True,
            "unused_action_parameter_names": [
                name for name, gradient in zip(parameter_names, action_gradients, strict=True)
                if gradient is None
            ],
            "unused_gain_parameter_names": [
                name for name, gradient in zip(parameter_names, gain_gradients_unweighted, strict=True)
                if gradient is None
            ],
        },
    }


def evaluate_frozen_gradient_geometry(
    model: torch.nn.Module,
    dataset: V24TrainingDataset,
    *,
    batch_size: int,
    batch_count: int,
    round_indices: Sequence[int],
    device: torch.device,
    normalization: TrackBTargetNormalizationV1,
    gain_loss_weight: float,
    clip_max_norm: float,
) -> dict[str, object]:
    """Measure local task-gradient geometry while restoring model and RNG state."""
    round_indices = _validate_gradient_evaluator_arguments(
        batch_size=batch_size,
        batch_count=batch_count,
        round_indices=round_indices,
        clip_max_norm=clip_max_norm,
        device=device,
    )
    if getattr(dataset, "split", None) != "train":
        raise ValueError("Track B frozen gradient geometry accepts the TRAIN split only")
    _require_strict_normalization_object(normalization)
    normalization.validate()
    if (type(gain_loss_weight) not in (int, float) or isinstance(gain_loss_weight, bool)
            or not math.isfinite(float(gain_loss_weight)) or float(gain_loss_weight) < 0.0):
        raise ValueError("Track B gain_loss_weight is invalid")
    parameter_names, parameters, groups, shared_indices = _trainable_parameter_partition(model)
    runtime_snapshot = _snapshot_model_runtime(model)
    model_state_before = trackb_model_state_sha(model)
    module_modes = {name: bool(module.training) for name, module in model.named_modules()}
    requires_grad_flags = tuple(bool(parameter.requires_grad) for parameter in parameters)
    cpu_rng_state = torch.random.get_rng_state()
    records = []
    with preserve_python_numpy_torch_rng():
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                generator=torch.Generator().manual_seed(0),
            )
            model.eval()
            for batch_index, batch in enumerate(islice(loader, batch_count)):
                history_mask = batch["history_mask"].to(device=device, dtype=torch.bool)
                if (history_mask.ndim != 2 or history_mask.shape[1] != TEN_ROUND_COUNT
                        or not bool(history_mask.all())):
                    raise ValueError("Track B gradient diagnostic requires all ten history records")
                for round_index in round_indices:
                    records.append(_gradient_record(
                        model,
                        batch=batch,
                        batch_index=batch_index,
                        round_index=int(round_index),
                        device=device,
                        normalization=normalization,
                        gain_loss_weight=float(gain_loss_weight),
                        clip_max_norm=float(clip_max_norm),
                        parameter_names=parameter_names,
                        parameters=parameters,
                        groups=groups,
                        shared_indices=shared_indices,
                    ))
        finally:
            _restore_model_runtime(runtime_snapshot)
    if len(records) != batch_count * len(round_indices):
        raise ValueError("Track B gradient diagnostic could not read the requested TRAIN batches")
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("Track B gradient diagnostic wrote parameter gradients")
    if tuple(bool(parameter.requires_grad) for parameter in parameters) != requires_grad_flags:
        raise ValueError("Track B gradient diagnostic changed requires_grad flags")
    model_state_after = trackb_model_state_sha(model)
    if model_state_after != model_state_before:
        raise ValueError("Track B gradient diagnostic changed model state")
    nonzero_pairs = [
        record
        for record in records
        if record["shared"]["action_gradient_norm"] > 0.0
        and record["shared"]["gain_gradient_norm_weighted"] > 0.0
    ]
    negative_pairs = [record for record in nonzero_pairs if record["shared"]["negative_inner_product"]]
    parameter_layout = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "element_count": parameter.numel(),
            "group": next(group for group, indices in groups.items() if index in indices),
        }
        for index, (name, parameter) in enumerate(zip(parameter_names, parameters, strict=True))
    ]
    return {
        "schema": "TrackBFrozenGradientGeometryV1",
        "scope": LOCAL_ENDPOINT_GRADIENT_SCOPE,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "round_indices": [int(value) + 1 for value in round_indices],
        "gain_loss_weight": float(gain_loss_weight),
        "parameter_layout_sha": canonical_sha256(parameter_layout),
        "parameter_layout": parameter_layout,
        "records": records,
        "summary": {
            "record_count": len(records),
            "nonzero_shared_gradient_pair_count": len(nonzero_pairs),
            "negative_shared_inner_product_count": len(negative_pairs),
            "negative_shared_inner_product_fraction": (
                len(negative_pairs) / len(nonzero_pairs) if nonzero_pairs else None
            ),
        },
        "model_state_sha_before": model_state_before,
        "model_state_sha_after": model_state_after,
        "model_state_unchanged": True,
        "module_modes_restored": all(
            bool(module.training) == module_modes[name]
            for name, module in model.named_modules()
        ),
        "cpu_rng_state_restored": bool(torch.equal(torch.random.get_rng_state(), cpu_rng_state)),
        "parameter_gradients_remain_none": True,
        "requires_grad_flags_unchanged": True,
        "optimizer_history": NOT_RECORDED,
        "historical_clip_frequency": NOT_RECORDED,
        "diagnostic_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def _run_trackb_frozen_gradient_diagnostic(
    config: TrackBFrozenGradientDiagnosticConfigV1,
) -> tuple[Path, dict[str, object]]:
    """Internal source-bound execution under caller RNG preservation."""
    config.validate()
    output_path = _prepare_new_output_path(Path(config.output_path))
    diagnostic_source_commit, diagnostic_source_tree = _diagnostic_repository_identity()
    receipt_path = Path(config.pilot_receipt_path).resolve()
    receipt = _read_verified_pilot_receipt(receipt_path, config.expected_pilot_receipt_sha)
    _require_oracle_history_reader(receipt, label="Track B gradient diagnostic")
    _bound_batch_plan(receipt, requested_batch_size=config.batch_size)
    root, build_id, _identity, manifest = _bound_v24_source(receipt)
    checkpoint_path = _checkpoint_path_from_receipt(receipt)
    _require_isolated_report_output_path(
        output_path,
        dataset_root=root,
        receipt_path=receipt_path,
        checkpoint_path=checkpoint_path,
    )
    model, model_identity, checkpoint = _load_bound_model(receipt)
    if model.config.action_value_feature_kind != LEGACY_LINEAR_MEAN_ACTION_VALUE:
        raise ValueError("Track B gradient diagnostic supports the legacy teacher-value path only")
    normalization = _normalization_from_bound_records(receipt, checkpoint)
    receipt_config = receipt.get("config")
    if not isinstance(receipt_config, Mapping):
        raise ValueError("Track B pilot receipt lacks configuration")
    gain_loss_weight = receipt_config.get("gain_loss_weight")
    if (type(gain_loss_weight) not in (int, float) or isinstance(gain_loss_weight, bool)
            or not math.isfinite(float(gain_loss_weight)) or float(gain_loss_weight) < 0.0):
        raise ValueError("Track B pilot receipt gain_loss_weight is invalid")
    cell = _matching_trackb_cell(manifest, system_type=model.config.system_type)
    train_dataset = V24TrainingDataset(
        root,
        build_id=build_id,
        cell_id=str(cell["cell_id"]),
        split="train",
        task="feedback",
        arm="A0_RAW",
    )
    validation_dataset = V24TrainingDataset(
        root,
        build_id=build_id,
        cell_id=str(cell["cell_id"]),
        split="validation",
        task="feedback",
        arm="A0_RAW",
    )
    try:
        receipt_dataset = receipt.get("dataset")
        if not isinstance(receipt_dataset, Mapping):
            raise ValueError("Track B pilot receipt lacks dataset identities")
        source_records = (
            ("train", checkpoint.get("train_source"), receipt_dataset.get("train"), train_dataset),
            (
                "validation",
                checkpoint.get("validation_source"),
                receipt_dataset.get("validation"),
                validation_dataset,
            ),
        )
        source_identities: dict[str, dict[str, Any]] = {}
        for label, checkpoint_source, receipt_source, dataset in source_records:
            if not isinstance(checkpoint_source, Mapping) or not isinstance(receipt_source, Mapping):
                raise ValueError(f"Track B pilot receipt lacks bound {label} identity")
            opened_source = _dataset_source_identity(dataset)
            _require_matching_source_identity(
                checkpoint_source,
                receipt_source,
                label=f"checkpoint/receipt {label}",
            )
            _require_matching_source_identity(
                checkpoint_source,
                opened_source,
                label=f"checkpoint/opened {label}",
            )
            source_identities[label] = opened_source
        normalization_recomputation = _require_recomputed_train_normalization(
            normalization,
            train_dataset,
            pilot_batch_size=config.batch_size,
        )
        geometry = evaluate_frozen_gradient_geometry(
            model,
            train_dataset,
            batch_size=config.batch_size,
            batch_count=config.batch_count,
            round_indices=config.round_indices,
            device=torch.device(config.device),
            normalization=normalization,
            gain_loss_weight=float(gain_loss_weight),
            clip_max_norm=float(config.clip_max_norm),
        )
        output = {
            "schema": FROZEN_GRADIENT_DIAGNOSTIC_SCHEMA,
            "config": asdict(config),
            "diagnostic_source_commit": diagnostic_source_commit,
            "diagnostic_source_tree": diagnostic_source_tree,
            "pilot": {
                "receipt_path": str(receipt_path),
                "receipt_file_sha": config.expected_pilot_receipt_sha,
                "pilot_source_commit": receipt.get("source_commit"),
                "pilot_source_tree": receipt.get("source_tree"),
                "checkpoint_sha": model_identity["checkpoint_sha"],
                "checkpoint_model_state_sha": model_identity["model_state_sha"],
            },
            "model": model_identity,
            "train_source": source_identities["train"],
            "validation_source": source_identities["validation"],
            "normalization_recomputation": normalization_recomputation,
            "loss_definition": {
                "action_normalization_kind": normalization.action_loss_normalization_kind,
                "gain_loss_weight": float(gain_loss_weight),
                "gain_standardized_huber_beta": STANDARDIZED_HUBER_DELTA,
            },
            "gradient_geometry": geometry,
            "caller_rng_restoration_contract": "PYTHON_NUMPY_TORCH_CPU_RESTORED_ON_SUCCESS_OR_ERROR",
            "diagnostic_only": True,
            "research_only": True,
            "selection_evidence": False,
            "promotion_ready": False,
            "hardware_action_authorized": False,
            "external_waveform_write": "NOT_IMPLEMENTED",
        }
        return output_path, output
    finally:
        train_dataset.close()
        validation_dataset.close()


def run_trackb_frozen_gradient_diagnostic(
    config: TrackBFrozenGradientDiagnosticConfigV1,
) -> dict[str, object]:
    """Publish a source-bound local gradient report without updating the model."""
    config.validate()
    with preserve_python_numpy_torch_rng():
        output_path, output = _run_trackb_frozen_gradient_diagnostic(config)
        output_sha = _publish_ad1_report(output_path, output)
    output["output_path"] = str(output_path)
    output["output_sha"] = output_sha
    output["caller_rng_restored"] = True
    return output


__all__ = [
    "FROZEN_GRADIENT_DIAGNOSTIC_SCHEMA",
    "LOCAL_ENDPOINT_GRADIENT_SCOPE",
    "TrackBFrozenGradientDiagnosticConfigV1",
    "evaluate_frozen_gradient_geometry",
    "run_trackb_frozen_gradient_diagnostic",
]
