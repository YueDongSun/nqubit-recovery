"""Pure ledger construction and reduction for Track B AD1 metrics."""

from __future__ import annotations

import copy
from statistics import median
from typing import Any, Mapping, Sequence

import math

import torch
from torch.utils.data import DataLoader

from ml_recon.data_v2_4.loaders import V24TrainingDataset

from .contracts import canonical_sha256
from .diagnostic_runtime import preserve_python_numpy_torch_rng
from .inverse_transformer_pilot import (
    GLOBAL_ACTION_LOSS_NORMALIZATION,
    PER_ROUND_ACTION_LOSS_NORMALIZATION,
    STANDARDIZED_HUBER_DELTA,
    TEN_ROUND_COUNT,
    TrackBTargetNormalizationV1,
    _round_inputs,
)
from .train_fit_diagnostic import EQUAL_BATCH_METRIC_AGGREGATION


RECOMPUTABLE_METRICS_SCHEMA = "TrackBRecomputableTeacherForcedMetricsV1"
ROW_LEDGER_SCHEMA = "TrackBTeacherForcedRowLedgerV1"
ROW_EQUAL_METRIC_AGGREGATION = "ROW_EQUAL"
MEDIAN_CONVENTION = "STATISTICS_MEDIAN_AVERAGE_MIDDLE_TWO"
LEDGER_ORDER = "ROUND_MAJOR_SOURCE_ORDINAL_V1"


def _snapshot_model_runtime(model: torch.nn.Module) -> dict[str, object]:
    """Capture every owned parameter/buffer object, gradients, and module modes."""
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
    """Restore object identity and contents, including non-persistent buffers."""
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


def _strict_real(value: object, *, name: str, positive: bool = False) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a literal finite real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _require_strict_normalization_object(
    normalization: TrackBTargetNormalizationV1,
) -> None:
    _strict_real(normalization.amplitude_scale, name="amplitude_scale", positive=True)
    _strict_real(normalization.phase_scale, name="phase_scale", positive=True)
    _strict_real(normalization.gain_center, name="gain_center")
    _strict_real(normalization.gain_scale, name="gain_scale", positive=True)
    if type(normalization.action_loss_normalization_kind) is not str or (
        normalization.action_loss_normalization_kind not in (
            GLOBAL_ACTION_LOSS_NORMALIZATION,
            PER_ROUND_ACTION_LOSS_NORMALIZATION,
        )
    ):
        raise ValueError("Track B action-loss normalization kind is invalid")
    for name, values in (
        ("amplitude_scale_by_round", normalization.amplitude_scale_by_round),
        ("phase_scale_by_round", normalization.phase_scale_by_round),
    ):
        if type(values) not in (list, tuple):
            raise ValueError(f"{name} must be an ordered sequence")
        for index, value in enumerate(values):
            _strict_real(value, name=f"{name}[{index}]", positive=True)
    if normalization.schema != "TrackBTargetNormalizationV1":
        raise ValueError("Track B target normalization schema is invalid")


def _finite_vector(value: object, *, name: str, expected_rows: int | None = None) -> list[list[float]]:
    if type(value) not in (list, tuple) or not value:
        raise ValueError(f"{name} must be a nonempty ordered matrix")
    result = []
    for row_index, row in enumerate(value):
        if type(row) not in (list, tuple) or len(row) != 2:
            raise ValueError(f"{name}[{row_index}] must contain amplitude and phase")
        result.append([
            _strict_real(row[0], name=f"{name}[{row_index}][0]"),
            _strict_real(row[1], name=f"{name}[{row_index}][1]"),
        ])
    if expected_rows is not None and len(result) != expected_rows:
        raise ValueError(f"{name} mode count is inconsistent")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Track B diagnostic metric has no values")
    return math.fsum(float(value) for value in values) / len(values)


def _smooth_l1(value: float, *, beta: float) -> float:
    magnitude = abs(value)
    return 0.5 * magnitude * magnitude / beta if magnitude < beta else magnitude - 0.5 * beta


def _weighted_metric(
    rows: Sequence[Mapping[str, Any]],
    values: Sequence[float],
    *,
    weight_key: str,
) -> float:
    if len(rows) != len(values):
        raise ValueError("Track B diagnostic values and weights differ in length")
    weights = [_strict_real(row[weight_key], name=weight_key, positive=True) for row in rows]
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Track B {weight_key} values do not sum to one")
    return math.fsum(weight * float(value) for weight, value in zip(weights, values, strict=True))


def _metric_record(
    row: Mapping[str, Any],
    *,
    constant_gain: float,
    normalization: TrackBTargetNormalizationV1,
) -> dict[str, float]:
    prediction = _finite_vector(row.get("predicted_action"), name="predicted_action")
    target = _finite_vector(
        row.get("target_teacher_action"),
        name="target_teacher_action",
        expected_rows=len(prediction),
    )
    previous = _finite_vector(
        row.get("previous_teacher_action"),
        name="previous_teacher_action",
        expected_rows=len(prediction),
    )
    round_index = int(row["round_index"]) - 1
    amplitude_scale, phase_scale = normalization.action_scales_for_round(round_index)
    amplitude_errors = [
        (predicted[0] - teacher[0]) ** 2
        for predicted, teacher in zip(prediction, target, strict=True)
    ]
    phase_errors = [
        (predicted[1] - teacher[1]) ** 2
        for predicted, teacher in zip(prediction, target, strict=True)
    ]
    action_errors = [
        (predicted[coordinate] - teacher[coordinate]) ** 2
        for predicted, teacher in zip(prediction, target, strict=True)
        for coordinate in (0, 1)
    ]
    zero_errors = [teacher[coordinate] ** 2 for teacher in target for coordinate in (0, 1)]
    previous_errors = [
        (baseline[coordinate] - teacher[coordinate]) ** 2
        for baseline, teacher in zip(previous, target, strict=True)
        for coordinate in (0, 1)
    ]
    prediction_square = [value * value for mode in prediction for value in mode]
    target_square = [value * value for mode in target for value in mode]
    cross = [
        predicted[coordinate] * teacher[coordinate]
        for predicted, teacher in zip(prediction, target, strict=True)
        for coordinate in (0, 1)
    ]
    predicted_gain = _strict_real(
        row.get("predicted_teacher_transition_gain"),
        name="predicted_teacher_transition_gain",
    )
    target_gain = _strict_real(
        row.get("target_teacher_transition_gain"),
        name="target_teacher_transition_gain",
    )
    previous_gain = _strict_real(
        row.get("previous_teacher_transition_gain"),
        name="previous_teacher_transition_gain",
    )
    standardized_gain_error = (predicted_gain - target_gain) / normalization.gain_scale
    amplitude_mse = _mean(amplitude_errors)
    phase_mse = _mean(phase_errors)
    return {
        "action_mse": _mean(action_errors),
        "amplitude_mse": amplitude_mse,
        "phase_mse": phase_mse,
        "action_normalized_mse": 0.5 * (
            amplitude_mse / amplitude_scale**2 + phase_mse / phase_scale**2
        ),
        "zero_action_mse": _mean(zero_errors),
        "previous_action_mse": _mean(previous_errors),
        "prediction_square_mean": _mean(prediction_square),
        "target_square_mean": _mean(target_square),
        "prediction_target_inner_mean": _mean(cross),
        "gain_mae": abs(predicted_gain - target_gain),
        "zero_gain_mae": abs(target_gain),
        "previous_gain_mae": abs(previous_gain - target_gain),
        "constant_gain_mae": abs(constant_gain - target_gain),
        "gain_standardized_huber": _smooth_l1(
            standardized_gain_error,
            beta=STANDARDIZED_HUBER_DELTA,
        ),
        "gain_prediction": predicted_gain,
        "gain_target": target_gain,
        "gain_signed_error": predicted_gain - target_gain,
    }


def _round_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    normalization: TrackBTargetNormalizationV1,
) -> dict[str, object]:
    target_gains = [
        _strict_real(row.get("target_teacher_transition_gain"), name="target_teacher_transition_gain")
        for row in rows
    ]
    constant_gain = float(median(target_gains))
    metrics = [
        _metric_record(row, constant_gain=constant_gain, normalization=normalization)
        for row in rows
    ]
    result: dict[str, object] = {
        "row_count": len(rows),
        "train_constant_gain": constant_gain,
        "train_constant_gain_fit": {
            "loss_target": "MAE",
            "median_convention": MEDIAN_CONVENTION,
            "fit_uses_unique_train_rows": True,
        },
    }
    metric_names = tuple(metrics[0])
    for aggregation, weight_key in (
        (EQUAL_BATCH_METRIC_AGGREGATION, "equal_batch_weight"),
        (ROW_EQUAL_METRIC_AGGREGATION, "row_equal_weight"),
    ):
        aggregated = {
            name: _weighted_metric(
                rows,
                [record[name] for record in metrics],
                weight_key=weight_key,
            )
            for name in metric_names
        }
        prediction_square = aggregated.pop("prediction_square_mean")
        target_square = aggregated.pop("target_square_mean")
        cross = aggregated.pop("prediction_target_inner_mean")
        prediction_rms = math.sqrt(max(prediction_square, 0.0))
        target_rms = math.sqrt(max(target_square, 0.0))
        denominator = prediction_rms * target_rms
        action_mse_identity = prediction_square + target_square - 2.0 * cross
        if not math.isclose(
            aggregated["action_mse"],
            action_mse_identity,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError("Track B action MSE does not match its energy/alignment identity")
        aggregated.update({
            "prediction_square_mean": prediction_square,
            "target_square_mean": target_square,
            "prediction_target_inner_mean": cross,
            "prediction_rms": prediction_rms,
            "target_rms": target_rms,
            "cosine_alignment": cross / denominator if denominator > 0.0 else None,
            "cosine_alignment_status": (
                "DEFINED" if denominator > 0.0 else "ZERO_PREDICTION_OR_TARGET_NORM"
            ),
        })
        result[aggregation] = aggregated
    return result


def reduce_recomputable_row_records(
    row_records: Sequence[Mapping[str, Any]],
    *,
    ordered_sample_ids: Sequence[str],
    pilot_batch_size: int,
    normalization: TrackBTargetNormalizationV1,
) -> dict[str, object]:
    """Validate a complete ledger and recompute every published summary."""
    _require_strict_normalization_object(normalization)
    normalization.validate()
    sample_ids = tuple(str(value) for value in ordered_sample_ids)
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Track B recomputable ledger requires unique ordered sample identities")
    if type(pilot_batch_size) is not int or pilot_batch_size <= 0:
        raise ValueError("Track B pilot batch size is invalid")
    if len(row_records) != len(sample_ids) * TEN_ROUND_COUNT:
        raise ValueError("Track B recomputable ledger is incomplete")
    expected_order = tuple(
        (sample_id, round_index)
        for round_index in range(1, TEN_ROUND_COUNT + 1)
        for sample_id in sample_ids
    )
    expected_pairs = set(expected_order)
    actual_pairs: set[tuple[str, int]] = set()
    group_by_ordinal: dict[int, str | None] = {}
    rows_by_round = [[] for _ in range(TEN_ROUND_COUNT)]
    pilot_batch_count = math.ceil(len(sample_ids) / pilot_batch_size)
    for record_index, record in enumerate(row_records):
        if not isinstance(record, Mapping):
            raise ValueError("Track B recomputable ledger row is invalid")
        sample_id = record.get("sample_id")
        source_ordinal = record.get("source_ordinal")
        round_index = record.get("round_index")
        if type(sample_id) is not str or not sample_id:
            raise ValueError("Track B recomputable ledger sample ID is invalid")
        if type(source_ordinal) is not int or not 0 <= source_ordinal < len(sample_ids):
            raise ValueError("Track B recomputable ledger source ordinal is invalid")
        if sample_ids[source_ordinal] != sample_id:
            raise ValueError("Track B recomputable ledger sample order differs from its source ordinal")
        source_group_id = record.get("source_group_id")
        if source_group_id is not None and (type(source_group_id) is not str or not source_group_id):
            raise ValueError("Track B recomputable ledger source group identity is invalid")
        if source_ordinal in group_by_ordinal and group_by_ordinal[source_ordinal] != source_group_id:
            raise ValueError("Track B recomputable ledger source group identity changes across rounds")
        group_by_ordinal[source_ordinal] = source_group_id
        if type(round_index) is not int or not 1 <= round_index <= TEN_ROUND_COUNT:
            raise ValueError("Track B recomputable ledger round index is invalid")
        pair = (sample_id, round_index)
        if pair != expected_order[record_index]:
            raise ValueError(
                "Track B recomputable ledger order differs from the canonical round-major source order"
            )
        if pair in actual_pairs:
            raise ValueError("Track B recomputable ledger has a duplicate sample-round pair")
        actual_pairs.add(pair)
        expected_batch_index = source_ordinal // pilot_batch_size
        expected_batch_rows = min(
            pilot_batch_size,
            len(sample_ids) - expected_batch_index * pilot_batch_size,
        )
        if record.get("pilot_batch_index") != expected_batch_index:
            raise ValueError("Track B recomputable ledger pilot batch membership is invalid")
        expected_equal_weight = 1.0 / (pilot_batch_count * expected_batch_rows)
        expected_row_weight = 1.0 / len(sample_ids)
        if not math.isclose(
            _strict_real(record.get("equal_batch_weight"), name="equal_batch_weight", positive=True),
            expected_equal_weight,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Track B recomputable ledger equal-batch weight is invalid")
        if not math.isclose(
            _strict_real(record.get("row_equal_weight"), name="row_equal_weight", positive=True),
            expected_row_weight,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Track B recomputable ledger row-equal weight is invalid")
        rows_by_round[round_index - 1].append(record)
    if actual_pairs != expected_pairs:
        raise ValueError("Track B recomputable ledger sample-round membership is incomplete")
    target_map = {
        (str(record["sample_id"]), int(record["round_index"])): (
            _finite_vector(record.get("target_teacher_action"), name="target_teacher_action"),
            _strict_real(
                record.get("target_teacher_transition_gain"),
                name="target_teacher_transition_gain",
            ),
        )
        for record in row_records
    }
    for record in row_records:
        sample_id = str(record["sample_id"])
        round_index = int(record["round_index"])
        current_action, _current_gain = target_map[(sample_id, round_index)]
        previous_action = _finite_vector(
            record.get("previous_teacher_action"),
            name="previous_teacher_action",
            expected_rows=len(current_action),
        )
        previous_gain = _strict_real(
            record.get("previous_teacher_transition_gain"),
            name="previous_teacher_transition_gain",
        )
        if round_index == 1:
            expected_action = [[0.0, 0.0] for _ in current_action]
            expected_gain = 0.0
        else:
            expected_action, expected_gain = target_map[(sample_id, round_index - 1)]
        if previous_action != expected_action or previous_gain != expected_gain:
            raise ValueError("Track B recomputable ledger previous-teacher baseline is inconsistent")
    round_metrics = []
    for round_index, rows in enumerate(rows_by_round, start=1):
        round_metrics.append({
            "round_index": round_index,
            **_round_summary(rows, normalization=normalization),
        })
    ordered_group_ids = [group_by_ordinal[index] for index in range(len(sample_ids))]
    return {
        "schema": RECOMPUTABLE_METRICS_SCHEMA,
        "sample_count": len(sample_ids),
        "round_count": TEN_ROUND_COUNT,
        "ledger_order": LEDGER_ORDER,
        "row_record_count": len(row_records),
        "ordered_sample_ids_sha": canonical_sha256(list(sample_ids)),
        "ordered_source_group_ids_sha": (
            canonical_sha256(ordered_group_ids)
            if all(value is not None for value in ordered_group_ids)
            else None
        ),
        "pilot_batch_size": pilot_batch_size,
        "pilot_batch_row_counts": [
            min(pilot_batch_size, len(sample_ids) - index * pilot_batch_size)
            for index in range(pilot_batch_count)
        ],
        "aggregations": [EQUAL_BATCH_METRIC_AGGREGATION, ROW_EQUAL_METRIC_AGGREGATION],
        "round_metrics": round_metrics,
    }


def evaluate_recomputable_teacher_forced(
    model: torch.nn.Module,
    dataset: V24TrainingDataset,
    *,
    pilot_batch_size: int,
    inference_batch_size: int,
    device: torch.device,
    normalization: TrackBTargetNormalizationV1,
) -> dict[str, object]:
    """Create an independent ordered ledger and reduce it without gradients."""
    if device.type != "cpu":
        raise ValueError("Track B recomputable evaluation supports CPU only")
    if type(inference_batch_size) is not int or inference_batch_size <= 0:
        raise ValueError("Track B inference batch size is invalid")
    if getattr(dataset, "split", None) != "train":
        raise ValueError("Track B recomputable evaluation accepts the TRAIN split only")
    _require_strict_normalization_object(normalization)
    normalization.validate()
    source_sample_ids = tuple(str(value) for value in getattr(dataset, "sample_ids", ()))
    if len(source_sample_ids) != len(dataset) or len(set(source_sample_ids)) != len(source_sample_ids):
        raise ValueError("Track B bound split lacks unique ordered sample identities")
    pilot_batch_count = math.ceil(len(source_sample_ids) / pilot_batch_size)
    source_rows = getattr(dataset, "rows", ())
    if source_rows and len(source_rows) != len(source_sample_ids):
        raise ValueError("Track B source rows differ from ordered sample identities")
    source_group_ids: tuple[str | None, ...] = tuple(
        (
            f"{row['acquisition_run_id']}|{row['repeat_group_id']}|{row['trajectory_id']}"
            if isinstance(row, Mapping)
            and all(type(row.get(key)) is str for key in (
                "acquisition_run_id", "repeat_group_id", "trajectory_id",
            ))
            else None
        )
        for row in source_rows
    ) if source_rows else tuple(None for _ in source_sample_ids)
    row_records_by_round: list[list[dict[str, object]]] = [
        [] for _ in range(TEN_ROUND_COUNT)
    ]
    source_offset = 0
    runtime_snapshot = _snapshot_model_runtime(model)
    model_state_before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    cpu_rng_state = torch.random.get_rng_state().clone()
    with preserve_python_numpy_torch_rng():
        try:
            loader = DataLoader(
                dataset,
                batch_size=inference_batch_size,
                shuffle=False,
                num_workers=0,
                generator=torch.Generator().manual_seed(0),
            )
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    batch_sample_ids = [str(value) for value in batch.get("sample_id", ())]
                    if not batch_sample_ids:
                        raise ValueError("Track B recomputable evaluation lacks sample identities")
                    expected_batch_ids = source_sample_ids[
                        source_offset:source_offset + len(batch_sample_ids)
                    ]
                    if tuple(batch_sample_ids) != expected_batch_ids:
                        raise ValueError("Track B recomputable inference order differs from the bound split")
                    history_mask = batch["history_mask"].to(device=device, dtype=torch.bool)
                    if (history_mask.ndim != 2 or history_mask.shape[1] != TEN_ROUND_COUNT
                            or not bool(history_mask.all())):
                        raise ValueError("Track B recomputable diagnostic requires all ten history records")
                    mode_mask = batch["mode_mask"].to(device=device, dtype=torch.bool)
                    if not bool(mode_mask.all()):
                        raise ValueError("Track B recomputable diagnostic requires canonical fully active modes")
                    history = batch["history_tokens"].to(device=device, dtype=torch.float32)
                    for round_index in range(TEN_ROUND_COUNT):
                        inputs, target_action, target_gain = _round_inputs(
                            batch,
                            round_index=round_index,
                            device=device,
                        )
                        output = model(inputs)
                        prediction = torch.stack(
                            (output["amplitude_delta"], output["phase_delta"]),
                            dim=-1,
                        )
                        if round_index:
                            previous_action = history[:, round_index - 1, :, 2:4]
                            previous_gain = history[:, round_index - 1, 0, 6]
                        else:
                            previous_action = torch.zeros_like(target_action)
                            previous_gain = torch.zeros_like(target_gain)
                        for local_index, sample_id in enumerate(batch_sample_ids):
                            source_ordinal = source_offset + local_index
                            pilot_batch_index = source_ordinal // pilot_batch_size
                            pilot_batch_rows = min(
                                pilot_batch_size,
                                len(source_sample_ids) - pilot_batch_index * pilot_batch_size,
                            )
                            row_records_by_round[round_index].append({
                                "sample_id": sample_id,
                                "source_ordinal": source_ordinal,
                                "source_group_id": source_group_ids[source_ordinal],
                                "round_index": round_index + 1,
                                "pilot_batch_index": pilot_batch_index,
                                "equal_batch_weight": 1.0 / (pilot_batch_count * pilot_batch_rows),
                                "row_equal_weight": 1.0 / len(source_sample_ids),
                                "predicted_action": prediction[local_index].detach().cpu().tolist(),
                                "target_teacher_action": target_action[local_index].detach().cpu().tolist(),
                                "previous_teacher_action": previous_action[local_index].detach().cpu().tolist(),
                                "predicted_teacher_transition_gain": float(
                                    output["predicted_teacher_transition_gain"][local_index].detach().cpu()
                                ),
                                "target_teacher_transition_gain": float(target_gain[local_index].detach().cpu()),
                                "previous_teacher_transition_gain": float(previous_gain[local_index].detach().cpu()),
                            })
                    source_offset += len(batch_sample_ids)
        finally:
            _restore_model_runtime(runtime_snapshot)
    if source_offset != len(source_sample_ids):
        raise ValueError("Track B recomputable evaluation did not consume the full bound split")
    row_records = [
        record
        for round_records in row_records_by_round
        for record in round_records
    ]
    for name, before in model_state_before.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), before):
            raise ValueError("Track B recomputable evaluation changed model state")
    reduced = reduce_recomputable_row_records(
        row_records,
        ordered_sample_ids=source_sample_ids,
        pilot_batch_size=pilot_batch_size,
        normalization=normalization,
    )
    return {
        **reduced,
        "split": "train",
        "inference_batch_size": inference_batch_size,
        "target_normalization": normalization.to_dict(),
        "row_ledger": {
            "schema": ROW_LEDGER_SCHEMA,
            "order": LEDGER_ORDER,
            "row_records": row_records,
            "row_records_sha": canonical_sha256(row_records),
        },
        "model_state_unchanged": True,
        "model_mode_restored": all(
            bool(module.training) == training
            for module, training in runtime_snapshot["module_modes"].items()
        ),
        "cpu_rng_state_restored": bool(torch.equal(torch.random.get_rng_state(), cpu_rng_state)),
        "caller_rng_restoration_contract": "PYTHON_NUMPY_TORCH_CPU_RESTORED_ON_SUCCESS_OR_ERROR",
        "diagnostic_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def legacy_arrays_from_recomputable_metrics(metrics: Mapping[str, Any]) -> dict[str, list[float]]:
    rounds = metrics.get("round_metrics")
    if not isinstance(rounds, list) or len(rounds) != TEN_ROUND_COUNT:
        raise ValueError("Track B recomputable diagnostic lacks all ten rounds")
    mapping = {
        "action_mse_by_round": "action_mse",
        "gain_mae_by_round": "gain_mae",
        "zero_action_mse_by_round": "zero_action_mse",
        "previous_action_mse_by_round": "previous_action_mse",
        "zero_gain_mae_by_round": "zero_gain_mae",
        "previous_gain_mae_by_round": "previous_gain_mae",
    }
    return {
        output_key: [
            float(round_record[EQUAL_BATCH_METRIC_AGGREGATION][metric_name])
            for round_record in rounds
        ]
        for output_key, metric_name in mapping.items()
    }


def require_legacy_metric_reconstruction(
    legacy: Mapping[str, Any],
    recomputable: Mapping[str, Any],
) -> dict[str, list[float]]:
    reconstructed = legacy_arrays_from_recomputable_metrics(recomputable)
    for key, values in reconstructed.items():
        legacy_values = legacy.get(key)
        if not isinstance(legacy_values, list) or len(legacy_values) != TEN_ROUND_COUNT:
            raise ValueError(f"legacy Track B evaluation lacks {key}")
        for left, right in zip(values, legacy_values, strict=True):
            if not math.isclose(left, float(right), rel_tol=1e-6, abs_tol=1e-10):
                raise ValueError(f"recomputable Track B EQUAL_BATCH metric differs for {key}")
    return reconstructed


__all__ = [
    "MEDIAN_CONVENTION",
    "LEDGER_ORDER",
    "RECOMPUTABLE_METRICS_SCHEMA",
    "ROW_EQUAL_METRIC_AGGREGATION",
    "ROW_LEDGER_SCHEMA",
    "evaluate_recomputable_teacher_forced",
    "legacy_arrays_from_recomputable_metrics",
    "reduce_recomputable_row_records",
    "require_legacy_metric_reconstruction",
]
