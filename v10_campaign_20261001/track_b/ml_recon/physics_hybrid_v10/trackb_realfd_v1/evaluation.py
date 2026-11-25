"""Research-only static and grouped evaluation helpers for Track B."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional-ML installation only.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from .candidate_library import RegisteredActionCandidateLibraryV1
from .losses import (
    TrackBRealStaticLossWeightsV1,
    calibration_log_score,
    native_masked_four_cell_nll,
    trackb_masked_real_static_objective,
)
from .observation_plate import TrainableMaskedObservationPlateV1
from .recoverable_envelope import TrackBRegisteredTryableRecoveryEnvelopeV1


@dataclass(frozen=True)
class TrackBStaticEvaluationV1:
    """Static-only scorecard; it intentionally has no trial outcome or causal gain."""

    selected_candidate_sha: str
    candidate_count: int
    registered_tryable_utility_lcb95: float
    causal_gain_available: bool = False
    physical_global_maximum_proven: bool = False
    schema: str = "TrackBStaticEvaluationV1"

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError("static evaluation requires at least one registered candidate")
        if self.causal_gain_available or self.physical_global_maximum_proven:
            raise ValueError("static evaluation cannot claim causal gain or a physical global maximum")


def evaluate_registered_library(
    library: RegisteredActionCandidateLibraryV1,
    envelope: TrackBRegisteredTryableRecoveryEnvelopeV1,
) -> TrackBStaticEvaluationV1:
    if envelope.candidate_library_sha != library.sha256():
        raise ValueError("static evaluation requires one exact registered candidate library")
    library.select_by_sha(envelope.registered_tryable_best_candidate_sha)
    return TrackBStaticEvaluationV1(
        selected_candidate_sha=envelope.registered_tryable_best_candidate_sha,
        candidate_count=len(library.candidates),
        registered_tryable_utility_lcb95=envelope.counterfactual_static_utility_lcb95,
    )


@dataclass(frozen=True)
class TrackBGroupedObservationEvaluationV1:
    """Static replay metrics grouped by admitted acquisition/repeat components."""

    system_type: str
    rows: int
    group_count: int
    component_means: tuple[tuple[str, float], ...]
    overall_mean_loss: float
    group_mean_loss: float
    native_global_multi_observed_rows: int
    causal_gain_available: bool = False
    selection_evidence: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBGroupedObservationEvaluationV1"

    def __post_init__(self) -> None:
        if self.system_type not in ("3qubit", "4qubit") or self.rows <= 0 or self.group_count <= 0:
            raise ValueError("Track B grouped evaluation system/row/group count is invalid")
        if not all(type(name) is str and name and math.isfinite(value) for name, value in self.component_means):
            raise ValueError("Track B grouped evaluation components are invalid")
        if not math.isfinite(self.overall_mean_loss) or not math.isfinite(self.group_mean_loss):
            raise ValueError("Track B grouped evaluation losses must be finite")
        if self.native_global_multi_observed_rows < 0:
            raise ValueError("Track B grouped evaluation global-multi count is invalid")
        if self.causal_gain_available or self.selection_evidence or self.hardware_action_authorized:
            raise ValueError("Track B static evaluation cannot claim causal or hardware authority")


@dataclass(frozen=True)
class TrackBFixedVisibleMarginalBaselineEvaluationV1:
    """A parameter-free held-out count baseline, not a learned Track B model."""

    system_type: str
    rows: int
    group_count: int
    overall_mean_four_cell_nll: float
    group_mean_four_cell_nll: float
    mean_calibration_log_score: float
    mean_query_cell_absolute_error: float
    uses_train_data: bool = False
    uses_query_counts_as_input: bool = False
    uses_global_multi_target: bool = False
    parameters_learned: bool = False
    causal_gain_available: bool = False
    selection_evidence: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBFixedVisibleMarginalBaselineEvaluationV1"

    def __post_init__(self) -> None:
        if self.system_type not in ("3qubit", "4qubit") or self.rows <= 0 or self.group_count <= 0:
            raise ValueError("Track B fixed baseline system/row/group count is invalid")
        if not all(math.isfinite(float(value)) for value in (
            self.overall_mean_four_cell_nll,
            self.group_mean_four_cell_nll,
            self.mean_calibration_log_score,
            self.mean_query_cell_absolute_error,
        )):
            raise ValueError("Track B fixed baseline metrics must be finite")
        if any((
            self.uses_train_data,
            self.uses_query_counts_as_input,
            self.uses_global_multi_target,
            self.parameters_learned,
            self.causal_gain_available,
            self.selection_evidence,
            self.hardware_action_authorized,
        )):
            raise ValueError("Track B fixed baseline cannot claim learned, causal, or hardware authority")


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Track B grouped observation evaluation requires PyTorch")


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[name] = value.clone()
        elif isinstance(value, tuple):
            result[name] = tuple(value)
        elif isinstance(value, list):
            result[name] = list(value)
        else:
            result[name] = value
    return result


def _one_row_batch(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Slice every row-aligned field without changing shared operation metadata."""
    counts = batch.get("four_cell_counts")
    if not isinstance(counts, torch.Tensor) or counts.ndim != 3:
        raise ValueError("Track B grouped evaluation requires four-cell [batch,operations,4] counts")
    batch_size = counts.shape[0]
    if index < 0 or index >= batch_size:
        raise IndexError("Track B grouped evaluation row index is out of range")
    result: dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            result[name] = value[index:index + 1]
        elif name in {"sample_ids", "group_ids", "global_multi_scope", "raw_rf_status"} \
                and isinstance(value, tuple) and len(value) == batch_size:
            result[name] = (value[index],)
        elif name in {"sample_ids", "group_ids", "global_multi_scope", "raw_rf_status"} \
                and isinstance(value, list) and len(value) == batch_size:
            result[name] = [value[index]]
        else:
            result[name] = value
    return result


def _nonidentity_permutation(count: int, *, seed: int, label: str, device: "torch.device") -> "torch.Tensor":
    if count < 2:
        raise ValueError("Track B descriptor derangement requires at least two items")
    raw = hashlib.sha256(f"TrackBNegativeControlV1|{seed}|{label}".encode("utf-8")).digest()
    shift = 1 + int.from_bytes(raw[:8], "big") % (count - 1)
    return torch.roll(torch.arange(count, device=device), shifts=shift)


def derange_trackb_conditioning(
    batch: Mapping[str, Any], *, field: str, seed: int = 20261001,
) -> dict[str, Any]:
    """Create an explicit no-training negative control without modifying targets.

    ``projector`` and ``raw_rf`` permute operation descriptors within each row;
    ``prepared_command`` and ``session`` permute only across batch rows.  Count
    targets, query masks, and global-multi targets are left intact.
    """
    _require_torch()
    result = _clone_batch(batch)
    if field == "projector":
        value = result.get("projector_descriptor")
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError("projector derangement requires [batch,operations,features]")
        order = _nonidentity_permutation(value.shape[1], seed=seed, label=field, device=value.device)
        result["projector_descriptor"] = value[:, order]
    elif field == "raw_rf":
        value = result.get("raw_rf_descriptor")
        mask = result.get("raw_rf_feature_mask")
        if not isinstance(value, torch.Tensor) or not isinstance(mask, torch.Tensor) or value.ndim != 3 or mask.shape != value.shape:
            raise ValueError("raw-RF derangement requires descriptor and availability mask")
        order = _nonidentity_permutation(value.shape[1], seed=seed, label=field, device=value.device)
        result["raw_rf_descriptor"] = value[:, order]
        result["raw_rf_feature_mask"] = mask[:, order]
    elif field == "prepared_command":
        value = result.get("prepared_command")
        mask = result.get("prepared_command_mask")
        if not isinstance(value, torch.Tensor) or not isinstance(mask, torch.Tensor) or value.ndim != 2 or mask.shape != value.shape:
            raise ValueError("prepared-command derangement requires values and mask")
        order = _nonidentity_permutation(value.shape[0], seed=seed, label=field, device=value.device)
        result["prepared_command"] = value[order]
        result["prepared_command_mask"] = mask[order]
    elif field == "session":
        value = result.get("session_index")
        mask = result.get("session_known_mask")
        if not isinstance(value, torch.Tensor) or not isinstance(mask, torch.Tensor) or value.ndim != 1 or mask.shape != value.shape:
            raise ValueError("session derangement requires session index and mask")
        order = _nonidentity_permutation(value.shape[0], seed=seed, label=field, device=value.device)
        result["session_index"] = value[order]
        result["session_known_mask"] = mask[order]
    else:
        raise ValueError("Track B conditioning derangement field is unknown")
    return result


def evaluate_trackb_grouped_observation(
    model: TrainableMaskedObservationPlateV1,
    batches: Iterable[Mapping[str, Any]], *,
    loss_weights: TrackBRealStaticLossWeightsV1 = TrackBRealStaticLossWeightsV1(),
    device: str = "cpu",
) -> TrackBGroupedObservationEvaluationV1:
    """Evaluate native count losses without opening locked data or updating a model."""
    _require_torch()
    loss_weights.validate()
    model_device = torch.device(device)
    model.to(model_device)
    model.eval()
    row_losses: dict[str, list[float]] = {}
    component_values: dict[str, list[float]] = {}
    global_observed = 0
    system_type: str | None = None
    rows = 0
    with torch.no_grad():
        for raw_batch in batches:
            batch = {
                key: value.to(model_device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            current_system = str(batch.get("system_type", ""))
            if current_system not in ("3qubit", "4qubit"):
                raise ValueError("grouped Track B evaluation requires a registered system type")
            if system_type is None:
                system_type = current_system
            elif system_type != current_system:
                raise ValueError("grouped Track B evaluation cannot mix systems")
            groups = batch.get("group_ids")
            if not isinstance(groups, tuple) or not groups:
                raise ValueError("grouped Track B evaluation requires source group IDs")
            for index, group in enumerate(groups):
                one_row = _one_row_batch(batch, index)
                prediction = model(one_row)
                losses = trackb_masked_real_static_objective(prediction, one_row, weights=loss_weights)
                row_losses.setdefault(str(group), []).append(float(losses["loss"].detach().cpu()))
                for name, value in losses.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        component_values.setdefault(name, []).append(float(value.detach().cpu()))
                global_observed += int(losses["global_multi_available_rows"].detach().cpu())
                rows += 1
    if not row_losses or system_type is None:
        raise ValueError("grouped Track B evaluation received no rows")
    group_means = {name: float(sum(values) / len(values)) for name, values in row_losses.items()}
    return TrackBGroupedObservationEvaluationV1(
        system_type=system_type, rows=rows, group_count=len(group_means),
        component_means=tuple(sorted((name, float(sum(values) / len(values))) for name, values in component_values.items())),
        overall_mean_loss=float(sum(sum(values) for values in row_losses.values()) / rows),
        group_mean_loss=float(sum(group_means.values()) / len(group_means)),
        native_global_multi_observed_rows=global_observed,
    )


def evaluate_trackb_descriptor_negative_controls(
    model: TrainableMaskedObservationPlateV1, batch: Mapping[str, Any], *,
    loss_weights: TrackBRealStaticLossWeightsV1 = TrackBRealStaticLossWeightsV1(),
) -> dict[str, float]:
    """Return static loss deltas for registered descriptor/session derangements."""
    _require_torch()
    model.eval()
    with torch.no_grad():
        baseline = float(trackb_masked_real_static_objective(
            model(batch), batch, weights=loss_weights,
        )["loss"].detach().cpu())
        result = {"baseline_loss": baseline}
        for field in ("projector", "raw_rf", "prepared_command", "session"):
            deranged = derange_trackb_conditioning(batch, field=field)
            value = float(trackb_masked_real_static_objective(
                model(deranged), deranged, weights=loss_weights,
            )["loss"].detach().cpu())
            result[f"{field}_derangement_loss"] = value
            result[f"{field}_derangement_delta"] = value - baseline
        return result


def build_trackb_fixed_visible_marginal_prediction(batch: Mapping[str, Any]) -> dict[str, "torch.Tensor"]:
    """Predict query cells from each row's visible cells only, with no fit state."""
    counts = batch.get("four_cell_counts")
    available = batch.get("operation_mask")
    query = batch.get("query_mask")
    if not isinstance(counts, torch.Tensor) or not isinstance(available, torch.Tensor) \
            or not isinstance(query, torch.Tensor):
        raise ValueError("Track B fixed baseline requires native counts and masks")
    if counts.ndim != 3 or counts.shape[-1] != 4 \
            or available.dtype != torch.bool or query.dtype != torch.bool \
            or available.shape != counts.shape[:2] or query.shape != counts.shape[:2]:
        raise ValueError("Track B fixed baseline count/mask shapes are invalid")
    visible = available & ~query
    if not bool(torch.all(visible.any(dim=1))) or not bool(torch.all((query & available).any(dim=1))):
        raise ValueError("Track B fixed baseline requires visible and query operations in every row")
    visible_counts = counts.to(dtype=torch.float32) * visible.unsqueeze(-1).to(dtype=torch.float32)
    aggregate = visible_counts.sum(dim=1)
    total = aggregate.sum(dim=-1).clamp_min(1.0)
    signal_total = aggregate[:, 1] + aggregate[:, 3]
    no_signal_total = aggregate[:, 0] + aggregate[:, 2]
    signal = (signal_total / total).clamp(1.0e-5, 1.0 - 1.0e-5)
    coin_given_signal = (aggregate[:, 3] / signal_total.clamp_min(1.0)).clamp(1.0e-5, 1.0 - 1.0e-5)
    idler_given_no_signal = (aggregate[:, 2] / no_signal_total.clamp_min(1.0)).clamp(1.0e-5, 1.0 - 1.0e-5)
    operation_total = (total / visible.sum(dim=1).to(dtype=torch.float32).clamp_min(1.0)).clamp_min(1.0e-6)
    signal = signal.unsqueeze(-1).expand_as(available)
    coin_given_signal = coin_given_signal.unsqueeze(-1).expand_as(available)
    idler_given_no_signal = idler_given_no_signal.unsqueeze(-1).expand_as(available)
    four_cell_mean = torch.stack((
        (1.0 - signal) * (1.0 - idler_given_no_signal),
        signal * (1.0 - coin_given_signal),
        (1.0 - signal) * idler_given_no_signal,
        signal * coin_given_signal,
    ), dim=-1) * operation_total[:, None, None]
    return {
        "signal_probability": signal,
        "coin_given_signal_probability": coin_given_signal,
        "idler_given_no_signal_probability": idler_given_no_signal,
        "four_cell_concentration": torch.full_like(total, 5.0),
        "four_cell_mean": four_cell_mean,
    }


def evaluate_trackb_fixed_visible_marginal_baseline(
    batches: Iterable[Mapping[str, Any]], *, device: str = "cpu",
) -> TrackBFixedVisibleMarginalBaselineEvaluationV1:
    """Score a no-fit visible-marginal baseline on held-out Track B folds only."""
    _require_torch()
    target_device = torch.device(device)
    row_nlls: dict[str, list[float]] = {}
    calibration_values: list[float] = []
    absolute_errors: list[float] = []
    system_type: str | None = None
    rows = 0
    with torch.no_grad():
        for raw_batch in batches:
            if raw_batch.get("read_only_evaluation") is not True:
                raise ValueError("Track B fixed baseline requires an explicit read-only evaluation batch")
            source_roles = raw_batch.get("source_roles")
            outer_roles = raw_batch.get("outer_roles")
            if not isinstance(source_roles, tuple) or not source_roles \
                    or any(role != "train" for role in source_roles) \
                    or not isinstance(outer_roles, tuple) or not outer_roles \
                    or any(role not in {"validation", "calibration"} for role in outer_roles):
                raise ValueError("Track B fixed baseline may score only held-out development rows")
            batch = {
                key: value.to(target_device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            current_system = str(batch.get("system_type", ""))
            if current_system not in {"3qubit", "4qubit"}:
                raise ValueError("Track B fixed baseline requires a registered system type")
            if system_type is None:
                system_type = current_system
            elif system_type != current_system:
                raise ValueError("Track B fixed baseline cannot mix systems")
            groups = batch.get("group_ids")
            if not isinstance(groups, tuple) or not groups:
                raise ValueError("Track B fixed baseline requires source group IDs")
            for index, group in enumerate(groups):
                one_row = _one_row_batch(batch, index)
                prediction = build_trackb_fixed_visible_marginal_prediction(one_row)
                nll = native_masked_four_cell_nll(prediction, one_row)
                query = one_row["query_mask"] & one_row["operation_mask"]
                calibration = calibration_log_score(
                    prediction["four_cell_mean"], one_row["four_cell_counts"], query,
                )
                absolute_error = (
                    prediction["four_cell_mean"][query]
                    - one_row["four_cell_counts"].to(dtype=torch.float32)[query]
                ).abs().mean()
                row_nlls.setdefault(str(group), []).append(float(nll.detach().cpu()))
                calibration_values.append(float(calibration.detach().cpu()))
                absolute_errors.append(float(absolute_error.detach().cpu()))
                rows += 1
    if not row_nlls or system_type is None:
        raise ValueError("Track B fixed baseline received no held-out rows")
    group_means = {group: sum(values) / len(values) for group, values in row_nlls.items()}
    return TrackBFixedVisibleMarginalBaselineEvaluationV1(
        system_type=system_type,
        rows=rows,
        group_count=len(group_means),
        overall_mean_four_cell_nll=sum(sum(values) for values in row_nlls.values()) / rows,
        group_mean_four_cell_nll=sum(group_means.values()) / len(group_means),
        mean_calibration_log_score=sum(calibration_values) / len(calibration_values),
        mean_query_cell_absolute_error=sum(absolute_errors) / len(absolute_errors),
    )


__all__ = [
    "TORCH_AVAILABLE", "TrackBFixedVisibleMarginalBaselineEvaluationV1",
    "TrackBGroupedObservationEvaluationV1", "TrackBStaticEvaluationV1",
    "derange_trackb_conditioning", "evaluate_registered_library",
    "build_trackb_fixed_visible_marginal_prediction",
    "evaluate_trackb_descriptor_negative_controls", "evaluate_trackb_fixed_visible_marginal_baseline",
    "evaluate_trackb_grouped_observation",
]
