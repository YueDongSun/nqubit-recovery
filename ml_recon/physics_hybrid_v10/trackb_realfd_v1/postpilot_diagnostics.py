"""Read-only full-fold failure diagnostics for completed Track B pilots.

The helpers in this module consume already-admitted validation or calibration
batches.  They do not construct an optimizer, update a model, write candidate
artifacts, or create any action/RF/AWG/FPGA/hardware output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional-ML installation only.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from .evaluation import (
    build_trackb_fixed_visible_marginal_prediction,
    derange_trackb_conditioning,
)
from .losses import (
    calibration_log_score,
    native_masked_four_cell_nll,
    trackb_masked_real_static_objective,
)
from .observation_plate import TrainableMaskedObservationPlateV1


_DIAGNOSTIC_CONTROL_FIELDS = ("projector", "prepared_command", "session")
_QUANTILE_BUCKET_COUNT = 4


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Track B post-pilot diagnostics require PyTorch")


def _one_row_batch(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    counts = batch.get("four_cell_counts")
    if not isinstance(counts, torch.Tensor) or counts.ndim != 3:
        raise ValueError("Track B post-pilot diagnostics require [batch,operations,4] counts")
    if index < 0 or index >= counts.shape[0]:
        raise IndexError("Track B post-pilot diagnostic row index is out of range")
    result: dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == counts.shape[0]:
            result[name] = value[index:index + 1]
        elif name in {"sample_ids", "group_ids", "global_multi_scope", "raw_rf_status", "source_roles", "outer_roles"} \
                and isinstance(value, tuple) and len(value) == counts.shape[0]:
            result[name] = (value[index],)
        else:
            result[name] = value
    return result


def _move_batch_to_cpu(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.to("cpu") if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _finite_scalar(value: "torch.Tensor", name: str) -> float:
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not bool(torch.isfinite(value).item()):
        raise ValueError(f"Track B post-pilot {name} must be a finite scalar tensor")
    return float(value.detach().cpu())


def _query_absolute_error(prediction: Mapping[str, "torch.Tensor"], batch: Mapping[str, Any]) -> float:
    mean = prediction.get("four_cell_mean")
    target = batch.get("four_cell_counts")
    query = batch.get("query_mask")
    available = batch.get("operation_mask")
    if not isinstance(mean, torch.Tensor) or not isinstance(target, torch.Tensor) \
            or not isinstance(query, torch.Tensor) or not isinstance(available, torch.Tensor):
        raise ValueError("Track B post-pilot query error requires count predictions and masks")
    active = query & available
    if not bool(active.any()):
        raise ValueError("Track B post-pilot query error requires a query operation")
    value = (mean[active] - target.to(dtype=mean.dtype)[active]).abs().mean()
    return _finite_scalar(value, "query absolute error")


def _row_change_fraction(
    original: Mapping[str, Any], deranged: Mapping[str, Any], *, field: str,
) -> tuple[float, ...]:
    if field == "projector":
        before = original["projector_descriptor"]
        after = deranged["projector_descriptor"]
        changed = torch.any(before != after, dim=-1).to(dtype=torch.float32).mean(dim=1)
    elif field == "prepared_command":
        before = original["prepared_command"]
        after = deranged["prepared_command"]
        changed = torch.any(before != after, dim=1).to(dtype=torch.float32)
    elif field == "session":
        before = original["session_index"]
        after = deranged["session_index"]
        changed = (before != after).to(dtype=torch.float32)
    else:
        raise ValueError("Track B post-pilot diagnostic control field is unknown")
    return tuple(float(value.detach().cpu()) for value in changed)


@dataclass(frozen=True)
class TrackBPostPilotDiagnosticRecordV1:
    """One group-bound held-out comparison record; never a promotion artifact."""

    sample_id: str
    group_id: str
    outer_role: str
    session_index: int
    exposure_s: float
    query_operation_count: int
    visible_operation_count: int
    visible_four_cell_total: float
    m0_four_cell_nll: float
    m1_four_cell_nll: float
    b0_four_cell_nll: float
    m0_loss: float
    m1_loss: float
    b0_calibration_log_score: float
    m1_query_absolute_error: float
    m1_mean_signal_probability: float
    query_coin_zero_fraction: float
    control_deltas: tuple[tuple[str, float], ...]
    control_change_fractions: tuple[tuple[str, float], ...]
    schema: str = "TrackBPostPilotDiagnosticRecordV1"

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id or self.outer_role not in {"validation", "calibration"}:
            raise ValueError("Track B post-pilot diagnostic record identity is invalid")
        if self.session_index < 0 or self.query_operation_count <= 0 or self.visible_operation_count <= 0:
            raise ValueError("Track B post-pilot diagnostic record cardinality is invalid")
        numeric = (
            self.exposure_s, self.visible_four_cell_total, self.m0_four_cell_nll,
            self.m1_four_cell_nll, self.b0_four_cell_nll, self.m0_loss, self.m1_loss,
            self.b0_calibration_log_score, self.m1_query_absolute_error,
            self.m1_mean_signal_probability, self.query_coin_zero_fraction,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Track B post-pilot diagnostic record has a non-finite metric")
        if any(name not in _DIAGNOSTIC_CONTROL_FIELDS or not math.isfinite(value)
               for name, value in self.control_deltas):
            raise ValueError("Track B post-pilot diagnostic control deltas are invalid")
        if any(name not in _DIAGNOSTIC_CONTROL_FIELDS or not 0.0 <= value <= 1.0
               for name, value in self.control_change_fractions):
            raise ValueError("Track B post-pilot diagnostic control change fractions are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "outer_role": self.outer_role,
            "session_index": self.session_index,
            "exposure_s": self.exposure_s,
            "query_operation_count": self.query_operation_count,
            "visible_operation_count": self.visible_operation_count,
            "visible_four_cell_total": self.visible_four_cell_total,
            "m0_four_cell_nll": self.m0_four_cell_nll,
            "m1_four_cell_nll": self.m1_four_cell_nll,
            "b0_four_cell_nll": self.b0_four_cell_nll,
            "m0_loss": self.m0_loss,
            "m1_loss": self.m1_loss,
            "b0_calibration_log_score": self.b0_calibration_log_score,
            "m1_query_absolute_error": self.m1_query_absolute_error,
            "m1_mean_signal_probability": self.m1_mean_signal_probability,
            "query_coin_zero_fraction": self.query_coin_zero_fraction,
            "control_deltas": {name: value for name, value in self.control_deltas},
            "control_change_fractions": {name: value for name, value in self.control_change_fractions},
        }


def _paired_group_values(
    records: Sequence[TrackBPostPilotDiagnosticRecordV1], *, left: str, right: str,
) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for record in records:
        value = float(getattr(record, left)) - float(getattr(record, right))
        grouped.setdefault(record.group_id, []).append(value)
    if not grouped:
        raise ValueError("Track B post-pilot paired summary received no groups")
    return np.asarray([sum(values) / len(values) for _group, values in sorted(grouped.items())], dtype=np.float64)


def _control_group_values(
    records: Sequence[TrackBPostPilotDiagnosticRecordV1], *, field: str,
) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record.group_id, []).append(dict(record.control_deltas)[field])
    if not grouped:
        raise ValueError("Track B post-pilot control summary received no groups")
    return np.asarray([sum(values) / len(values) for _group, values in sorted(grouped.items())], dtype=np.float64)


def _bootstrap_summary(
    values: np.ndarray, *, metric: str, seed: int, resamples: int,
) -> dict[str, object]:
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Track B post-pilot bootstrap values are invalid")
    if type(seed) is not int or type(resamples) is not int or resamples < 100:
        raise ValueError("Track B post-pilot bootstrap configuration is invalid")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(resamples, len(values)), endpoint=False)
    bootstrap_means = values[indices].mean(axis=1)
    return {
        "metric": metric,
        "group_count": int(len(values)),
        "mean": float(values.mean()),
        "lcb95": float(np.quantile(bootstrap_means, 0.05)),
        "ucb95": float(np.quantile(bootstrap_means, 0.95)),
        "seed": seed,
        "resamples": resamples,
    }


def _stratum_label(values: Sequence[float], *, prefix: str) -> tuple[str, ...]:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or len(numeric) == 0 or not np.all(np.isfinite(numeric)):
        raise ValueError("Track B post-pilot stratum values are invalid")
    if len(np.unique(numeric)) == 1:
        return tuple(f"{prefix}=constant" for _value in numeric)
    cuts = np.quantile(numeric, [index / _QUANTILE_BUCKET_COUNT for index in range(1, _QUANTILE_BUCKET_COUNT)])
    return tuple(f"{prefix}=q{int(np.searchsorted(cuts, value, side='right')) + 1}" for value in numeric)


def _stratum_summary(
    records: Sequence[TrackBPostPilotDiagnosticRecordV1], labels: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    if len(records) != len(labels):
        raise ValueError("Track B post-pilot stratum labels do not align to records")
    buckets: dict[str, list[TrackBPostPilotDiagnosticRecordV1]] = {}
    for record, label in zip(records, labels, strict=True):
        buckets.setdefault(label, []).append(record)
    result: dict[str, dict[str, float | int]] = {}
    for label, values in sorted(buckets.items()):
        result[label] = {
            "records": len(values),
            "m0_four_cell_nll": float(np.mean([value.m0_four_cell_nll for value in values])),
            "m1_four_cell_nll": float(np.mean([value.m1_four_cell_nll for value in values])),
            "b0_four_cell_nll": float(np.mean([value.b0_four_cell_nll for value in values])),
            "m0_minus_m1": float(np.mean([value.m0_four_cell_nll - value.m1_four_cell_nll for value in values])),
            "b0_minus_m1": float(np.mean([value.b0_four_cell_nll - value.m1_four_cell_nll for value in values])),
        }
    return result


def evaluate_trackb_full_fold_postpilot_diagnostics(
    m0_model: TrainableMaskedObservationPlateV1,
    m1_model: TrainableMaskedObservationPlateV1,
    batches: Iterable[Mapping[str, Any]], *, outer_role: str, device: str = "cpu",
    bootstrap_seed: int = 20261001, bootstrap_resamples: int = 2048,
) -> dict[str, object]:
    """Compute group-paired, full-fold diagnostics without an optimizer or gradients."""
    _require_torch()
    if device != "cpu" or outer_role not in {"validation", "calibration"}:
        raise ValueError("Track B post-pilot diagnostics are CPU-only validation/calibration evaluation")
    m0_model.to("cpu")
    m1_model.to("cpu")
    m0_model.eval()
    m1_model.eval()
    records: list[TrackBPostPilotDiagnosticRecordV1] = []
    family_cycle_errors: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    raw_rf_available = False
    system_type: str | None = None
    with torch.no_grad():
        for raw_batch in batches:
            if raw_batch.get("read_only_evaluation") is not True \
                    or raw_batch.get("evaluation_outer_role") != outer_role:
                raise ValueError("Track B post-pilot diagnostics require an explicit matching read-only batch")
            batch = _move_batch_to_cpu(raw_batch)
            current_system = str(batch.get("system_type", ""))
            if current_system not in {"3qubit", "4qubit"}:
                raise ValueError("Track B post-pilot diagnostics require a registered system")
            if system_type is None:
                system_type = current_system
            elif current_system != system_type:
                raise ValueError("Track B post-pilot diagnostics cannot mix systems")
            deranged_batches = {
                field: derange_trackb_conditioning(batch, field=field)
                for field in _DIAGNOSTIC_CONTROL_FIELDS
            }
            change_fractions = {
                field: _row_change_fraction(batch, deranged, field=field)
                for field, deranged in deranged_batches.items()
            }
            raw_rf_mask = batch.get("raw_rf_feature_mask")
            raw_rf_available = raw_rf_available or bool(
                isinstance(raw_rf_mask, torch.Tensor) and raw_rf_mask.any().item()
            )
            sample_ids = batch.get("sample_ids")
            group_ids = batch.get("group_ids")
            if not isinstance(sample_ids, tuple) or not isinstance(group_ids, tuple) \
                    or len(sample_ids) != batch["four_cell_counts"].shape[0] \
                    or len(group_ids) != batch["four_cell_counts"].shape[0]:
                raise ValueError("Track B post-pilot diagnostics require row sample/group identities")
            for index, (sample_id, group_id) in enumerate(zip(sample_ids, group_ids, strict=True)):
                one_row = _one_row_batch(batch, index)
                m0_prediction = m0_model(one_row)
                m1_prediction = m1_model(one_row)
                m0_losses = trackb_masked_real_static_objective(m0_prediction, one_row)
                m1_losses = trackb_masked_real_static_objective(m1_prediction, one_row)
                b0_prediction = build_trackb_fixed_visible_marginal_prediction(one_row)
                b0_four_cell_nll = _finite_scalar(
                    native_masked_four_cell_nll(b0_prediction, one_row), "B0 four-cell NLL",
                )
                active_query = one_row["query_mask"] & one_row["operation_mask"]
                b0_calibration = _finite_scalar(
                    calibration_log_score(
                        b0_prediction["four_cell_mean"], one_row["four_cell_counts"], active_query,
                    ),
                    "B0 calibration log score",
                )
                control_deltas: list[tuple[str, float]] = []
                for field, deranged_batch in deranged_batches.items():
                    deranged_row = _one_row_batch(deranged_batch, index)
                    deranged_prediction = m1_model(deranged_row)
                    deranged_loss = _finite_scalar(
                        trackb_masked_real_static_objective(deranged_prediction, deranged_row)["loss"],
                        f"{field} derangement loss",
                    )
                    control_deltas.append((field, deranged_loss - _finite_scalar(m1_losses["loss"], "M1 loss")))
                query_counts = one_row["four_cell_counts"].to(dtype=torch.float32)
                visible_mask = one_row["operation_mask"] & ~one_row["query_mask"]
                query_count = int(active_query.sum().item())
                visible_count = int(visible_mask.sum().item())
                visible_total = float((query_counts * visible_mask.unsqueeze(-1)).sum().item())
                signal_mean = float(m1_prediction["signal_probability"][active_query].mean().item())
                coin_zero_fraction = float((query_counts[..., 3][active_query] == 0).to(dtype=torch.float32).mean().item())
                record = TrackBPostPilotDiagnosticRecordV1(
                    sample_id=str(sample_id),
                    group_id=str(group_id),
                    outer_role=outer_role,
                    session_index=int(one_row["session_index"].item()),
                    exposure_s=float(one_row["exposure_s"].item()),
                    query_operation_count=query_count,
                    visible_operation_count=visible_count,
                    visible_four_cell_total=visible_total,
                    m0_four_cell_nll=_finite_scalar(m0_losses["four_cell_nll"], "M0 four-cell NLL"),
                    m1_four_cell_nll=_finite_scalar(m1_losses["four_cell_nll"], "M1 four-cell NLL"),
                    b0_four_cell_nll=b0_four_cell_nll,
                    m0_loss=_finite_scalar(m0_losses["loss"], "M0 loss"),
                    m1_loss=_finite_scalar(m1_losses["loss"], "M1 loss"),
                    b0_calibration_log_score=b0_calibration,
                    m1_query_absolute_error=_query_absolute_error(m1_prediction, one_row),
                    m1_mean_signal_probability=signal_mean,
                    query_coin_zero_fraction=coin_zero_fraction,
                    control_deltas=tuple(control_deltas),
                    control_change_fractions=tuple(
                        (field, change_fractions[field][index]) for field in _DIAGNOSTIC_CONTROL_FIELDS
                    ),
                )
                records.append(record)
                for operation_index in torch.nonzero(active_query[0], as_tuple=False).reshape(-1).tolist():
                    family = str(one_row["operation_families"][operation_index])
                    cycle = int(one_row["cycle_indices"][operation_index])
                    target = query_counts[0, operation_index]
                    family_cycle_errors.setdefault((family, cycle), []).append((
                        float((m0_prediction["four_cell_mean"][0, operation_index] - target).abs().mean().item()),
                        float((m1_prediction["four_cell_mean"][0, operation_index] - target).abs().mean().item()),
                        float((b0_prediction["four_cell_mean"][0, operation_index] - target).abs().mean().item()),
                    ))
    if not records or system_type is None:
        raise ValueError("Track B post-pilot diagnostics received no rows")
    control_summary: dict[str, dict[str, object]] = {}
    for field in _DIAGNOSTIC_CONTROL_FIELDS:
        deltas = [dict(record.control_deltas)[field] for record in records]
        changes = [dict(record.control_change_fractions)[field] for record in records]
        control_summary[field] = {
            "status": "FULL_FOLD_DIAGNOSTIC_ONLY",
            "records": len(deltas),
            "mean_delta": float(np.mean(deltas)),
            "changed_fraction": float(np.mean(changes)),
            "identity_fraction": float(np.mean([value == 0.0 for value in changes])),
            "paired_bootstrap": _bootstrap_summary(
                _control_group_values(records, field=field),
                metric=f"{field}_derangement_minus_registered_loss",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
        }
    family_cycle_summary = [
        {
            "operation_family": family,
            "cycle_index": cycle,
            "query_operations": len(values),
            "m0_query_cell_absolute_error": float(np.mean([value[0] for value in values])),
            "m1_query_cell_absolute_error": float(np.mean([value[1] for value in values])),
            "b0_query_cell_absolute_error": float(np.mean([value[2] for value in values])),
        }
        for (family, cycle), values in sorted(family_cycle_errors.items())
    ]
    strata = {
        "session_index": _stratum_summary(records, tuple(f"session={record.session_index}" for record in records)),
        "exposure_s": _stratum_summary(records, _stratum_label([record.exposure_s for record in records], prefix="exposure")),
        "visible_four_cell_total": _stratum_summary(records, _stratum_label([record.visible_four_cell_total for record in records], prefix="visible_total")),
        "m1_signal_probability": _stratum_summary(records, _stratum_label([record.m1_mean_signal_probability for record in records], prefix="signal")),
        "query_coin_zero_fraction": _stratum_summary(records, _stratum_label([record.query_coin_zero_fraction for record in records], prefix="coin_zero")),
        "query_operation_count": _stratum_summary(records, tuple(f"query_operations={record.query_operation_count}" for record in records)),
    }
    return {
        "schema": "TrackBFullFoldPostPilotDiagnosticV1",
        "system_type": system_type,
        "outer_role": outer_role,
        "records": [record.to_dict() for record in records],
        "paired_group_bootstrap": {
            "m0_minus_m1": _bootstrap_summary(
                _paired_group_values(records, left="m0_four_cell_nll", right="m1_four_cell_nll"),
                metric="m0_four_cell_nll_minus_m1_four_cell_nll",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
            "b0_minus_m1": _bootstrap_summary(
                _paired_group_values(records, left="b0_four_cell_nll", right="m1_four_cell_nll"),
                metric="b0_four_cell_nll_minus_m1_four_cell_nll",
                seed=bootstrap_seed + 1,
                resamples=bootstrap_resamples,
            ),
        },
        "full_fold_controls": control_summary,
        "raw_rf_control_status": "FULL_FOLD_AVAILABLE" if raw_rf_available else "NOT_APPLICABLE_RAW_RF_UNAVAILABLE",
        "operation_family_cycle_query_error": family_cycle_summary,
        "strata": strata,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "optimizer_constructed": False,
        "gradients_enabled": False,
        "further_training_authorized": False,
        "checkpoint_promotion_authorized": False,
        "actionful_fd_authorized": False,
        "rf_awg_fpga_hardware_authorized": False,
    }


__all__ = [
    "TORCH_AVAILABLE", "TrackBPostPilotDiagnosticRecordV1",
    "evaluate_trackb_full_fold_postpilot_diagnostics",
]
