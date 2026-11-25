"""Native masked real-static likelihoods and non-causal regularizers for Track B."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional-ML installation only.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from .contracts import validate_real_static_loss_terms
from .measurement_rf_map import measurement_rf_regularization


REAL_STATIC_ALLOWED_LOSSES = frozenset({
    "masked_four_cell",
    "global_multi",
    "cross_repeat_or_thinning",
    "calibration",
    "uncertainty",
    "nominal_born_residual",
    "measurement_rf_map",
    "session_hierarchical_prior",
})


def ratio_of_sums_query_absolute_error(
    prediction: np.ndarray, target: np.ndarray, query_mask: np.ndarray,
) -> float:
    """Evaluate only query operations using a public-cardinality denominator."""
    predicted = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(target, dtype=np.float64)
    mask = np.asarray(query_mask)
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("prediction and target require matching [operations,4] arrays")
    if mask.shape != (predicted.shape[0],) or mask.dtype != np.bool_:
        raise ValueError("query mask must be strict bool [operations]")
    if not bool(mask.any()):
        raise ValueError("query-only evaluation requires at least one query operation")
    active_prediction = predicted[mask]
    active_target = actual[mask]
    if not np.all(np.isfinite(active_prediction)) or not np.all(np.isfinite(active_target)):
        raise ValueError("active query values must be finite")
    return float(np.abs(active_prediction - active_target).sum() / float(active_target.size))


def validate_real_static_loss_plan(loss_names: Sequence[str]) -> tuple[str, ...]:
    """Validate an observation-only objective before an optimizer can receive it."""
    values = tuple(loss_names)
    if not values:
        raise ValueError("real-static loss plan cannot be empty")
    if any(type(value) is not str for value in values):
        raise TypeError("real-static loss names must be strings")
    unknown = set(values) - REAL_STATIC_ALLOWED_LOSSES
    if unknown:
        raise ValueError(
            "real-static loss plan contains unregistered or causal terms: "
            + ", ".join(sorted(unknown))
        )
    validate_real_static_loss_terms(values)
    return values


def validate_global_multi_scope(scope: str, values: Sequence[int]) -> None:
    if scope not in ("global", "state"):
        raise ValueError("global multi counts cannot be spread over operations")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("global multi must retain native nonnegative count values")


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Track B native likelihoods require PyTorch")


def _query_mask(batch: Mapping[str, Any]) -> "torch.Tensor":
    if "query_mask" not in batch or "operation_mask" not in batch:
        raise ValueError("Track B likelihood requires query and operation masks")
    query = batch["query_mask"]
    available = batch["operation_mask"]
    if query.dtype != torch.bool or available.dtype != torch.bool or query.shape != available.shape:
        raise ValueError("query/operation masks must be matching bool [batch,operations]")
    result = query & available
    if not bool(torch.all(result.any(dim=1))):
        raise ValueError("Track B loss requires at least one query operation per row")
    return result


def native_masked_four_cell_nll(
    prediction: Mapping[str, "torch.Tensor"], batch: Mapping[str, Any],
) -> "torch.Tensor":
    """Use the canonical nested four-cell likelihood on query operations only.

    The likelihood is the native factorization
    ``P(S|R) * P(C|S) * P(I-C|R-S)`` implemented by the existing stable V10
    count kernel.  It receives unmodified native integer cells; no clipping,
    reliability floor, or operation-local global-multi expansion is applied.
    """
    _require_torch()
    from ml_recon.physics_hybrid_v10.real_count_likelihood import four_cell_nll

    required = {
        "signal_probability", "coin_given_signal_probability",
        "idler_given_no_signal_probability", "four_cell_concentration",
    }
    missing = required - set(prediction)
    if missing:
        raise ValueError(f"Track B four-cell prediction is incomplete: {sorted(missing)}")
    if "four_cell_counts" not in batch:
        raise ValueError("Track B four-cell target is missing")
    return four_cell_nll(
        batch["four_cell_counts"], prediction["signal_probability"],
        prediction["coin_given_signal_probability"],
        prediction["idler_given_no_signal_probability"],
        prediction["four_cell_concentration"], _query_mask(batch),
    )


def native_global_multi_nll(
    prediction: Mapping[str, "torch.Tensor"], batch: Mapping[str, Any],
) -> tuple["torch.Tensor", int]:
    """Use native global NB2 + pair multinomial likelihood when observed.

    Global multi-click counts remain loss-only state/global targets.  A batch
    containing no recorded global observation returns a differentiable zero and
    an availability count of zero rather than inventing a zero observation.
    """
    _require_torch()
    from ml_recon.physics_hybrid_v10.real_count_likelihood import global_multi_nll

    required = {"multi_means", "pair_logits", "multi_dispersion"}
    missing = required - set(prediction)
    if missing:
        raise ValueError(f"Track B global-multi prediction is incomplete: {sorted(missing)}")
    for name in ("global_multi", "global_multi_mask", "global_multi_scope"):
        if name not in batch:
            raise ValueError(f"Track B global-multi batch is missing {name}")
    mask = batch["global_multi_mask"]
    if mask.dtype != torch.bool or mask.ndim != 1:
        raise ValueError("global-multi mask must be bool [batch]")
    available = int(mask.sum().detach().cpu())
    if available == 0:
        return prediction["multi_means"].sum() * 0.0, 0
    value = global_multi_nll(
        batch["global_multi"], mask, batch["global_multi_scope"],
        prediction["multi_means"], prediction["pair_logits"], prediction["multi_dispersion"],
    )
    return value, available


def calibration_log_score(
    four_cell_mean: "torch.Tensor", target_cells: "torch.Tensor", query_mask: "torch.Tensor",
) -> "torch.Tensor":
    """Query-only categorical log score, a proper calibration diagnostic."""
    _require_torch()
    if four_cell_mean.shape != target_cells.shape or four_cell_mean.ndim != 3 or four_cell_mean.shape[-1] != 4:
        raise ValueError("calibration score requires matching [batch,operations,4] tensors")
    if query_mask.dtype != torch.bool or query_mask.shape != four_cell_mean.shape[:2]:
        raise ValueError("calibration score query mask is invalid")
    totals = target_cells.to(dtype=four_cell_mean.dtype).sum(dim=-1)
    active = query_mask & (totals > 0.0)
    if not bool(active.any()):
        return four_cell_mean.sum() * 0.0
    probability = four_cell_mean / four_cell_mean.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    empirical = target_cells.to(dtype=four_cell_mean.dtype) / totals.unsqueeze(-1).clamp_min(1.0)
    return -(empirical[active] * probability[active].clamp_min(1.0e-12).log()).sum(dim=-1).mean()


def uncertainty_log_score(
    four_cell_mean: "torch.Tensor", four_cell_log_variance: "torch.Tensor",
    target_cells: "torch.Tensor", query_mask: "torch.Tensor",
) -> "torch.Tensor":
    """A finite query-only heteroscedastic proper score for count means."""
    _require_torch()
    if four_cell_mean.shape != target_cells.shape or four_cell_log_variance.shape != four_cell_mean.shape:
        raise ValueError("uncertainty score requires matching four-cell tensors")
    if query_mask.dtype != torch.bool or query_mask.shape != four_cell_mean.shape[:2]:
        raise ValueError("uncertainty score query mask is invalid")
    if not bool(query_mask.any()):
        raise ValueError("uncertainty score requires query operations")
    log_variance = four_cell_log_variance.clamp(-12.0, 20.0)
    variance = torch.exp(log_variance)
    error = target_cells.to(dtype=four_cell_mean.dtype) - four_cell_mean
    per_cell = 0.5 * (log_variance + error.square() / variance)
    return per_cell[query_mask].mean()


def cross_repeat_or_thinning_consistency(
    first_mean: "torch.Tensor", second_mean: "torch.Tensor", mask: "torch.Tensor",
) -> "torch.Tensor":
    """Non-causal consistency penalty for paired repeats or explicit thinnings."""
    _require_torch()
    if first_mean.shape != second_mean.shape or first_mean.ndim != 3 or first_mean.shape[-1] != 4:
        raise ValueError("repeat/thinning consistency requires matching [batch,operations,4] means")
    if mask.dtype != torch.bool or mask.shape != first_mean.shape[:2]:
        raise ValueError("repeat/thinning consistency mask is invalid")
    if not bool(mask.any()):
        return first_mean.sum() * 0.0
    return (first_mean[mask] - second_mean[mask]).square().mean()


@dataclass(frozen=True)
class TrackBRealStaticLossWeightsV1:
    """Registered observation-only objective weights for a Track B runtime."""

    four_cell: float = 1.0
    global_multi: float = 1.0
    calibration: float = 0.05
    uncertainty: float = 0.01
    cross_repeat_or_thinning: float = 0.0
    nominal_born_residual: float = 0.01
    measurement_rf_map: float = 0.01
    session_hierarchical_prior: float = 0.001
    schema: str = "TrackBRealStaticLossWeightsV1"

    def validate(self) -> None:
        values = (
            self.four_cell, self.global_multi, self.calibration, self.uncertainty,
            self.cross_repeat_or_thinning, self.nominal_born_residual,
            self.measurement_rf_map, self.session_hierarchical_prior,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("Track B real-static loss weights must be finite nonnegative")
        if self.four_cell <= 0.0:
            raise ValueError("Track B native four-cell likelihood must remain active")


def trackb_masked_real_static_objective(
    prediction: Mapping[str, Any], batch: Mapping[str, Any], *,
    weights: TrackBRealStaticLossWeightsV1 = TrackBRealStaticLossWeightsV1(),
    paired_prediction: Mapping[str, "torch.Tensor"] | None = None,
) -> dict[str, "torch.Tensor"]:
    """Assemble native likelihoods and registered non-causal regularizers.

    The returned components are separate for monitoring.  It never accepts an
    action/gain/Jacobian label and refuses a fake global-multi observation.
    """
    _require_torch()
    weights.validate()
    validate_real_static_loss_plan((
        "masked_four_cell", "global_multi", "calibration", "uncertainty",
        "nominal_born_residual", "measurement_rf_map", "session_hierarchical_prior",
    ))
    query = _query_mask(batch)
    four_cell = native_masked_four_cell_nll(prediction, batch)
    global_multi, global_available = native_global_multi_nll(prediction, batch)
    calibration = calibration_log_score(prediction["four_cell_mean"], batch["four_cell_counts"], query)
    uncertainty = uncertainty_log_score(
        prediction["four_cell_mean"], prediction["four_cell_log_variance"], batch["four_cell_counts"], query,
    )
    if "nominal_four_cell_mean" not in prediction:
        raise ValueError("Track B observation prediction lacks nominal Born measurement path")
    born_residual = (prediction["four_cell_mean"][query] - prediction["nominal_four_cell_mean"][query]).square().mean()
    if "measurement_rf" not in prediction:
        raise ValueError("Track B observation prediction lacks the measurement RF map")
    rf_regularizer = measurement_rf_regularization(prediction["measurement_rf"])
    session_effect = prediction["measurement_rf"].get("session_effect")
    if session_effect is None:
        raise ValueError("Track B measurement RF output lacks session effect")
    session_prior = session_effect.square().mean()
    if paired_prediction is None:
        repeat_consistency = prediction["four_cell_mean"].sum() * 0.0
    else:
        if "four_cell_mean" not in paired_prediction:
            raise ValueError("paired Track B prediction lacks four-cell mean")
        repeat_consistency = cross_repeat_or_thinning_consistency(
            prediction["four_cell_mean"], paired_prediction["four_cell_mean"], query,
        )
    total = (
        weights.four_cell * four_cell
        + weights.global_multi * global_multi
        + weights.calibration * calibration
        + weights.uncertainty * uncertainty
        + weights.cross_repeat_or_thinning * repeat_consistency
        + weights.nominal_born_residual * born_residual
        + weights.measurement_rf_map * rf_regularizer
        + weights.session_hierarchical_prior * session_prior
    )
    if not bool(torch.isfinite(total).detach()):
        raise RuntimeError("Track B native real-static objective became non-finite")
    return {
        "loss": total,
        "four_cell_nll": four_cell,
        "global_multi_nll": global_multi,
        "global_multi_available_rows": total.new_tensor(float(global_available)),
        "calibration_log_score": calibration,
        "uncertainty_log_score": uncertainty,
        "cross_repeat_or_thinning": repeat_consistency,
        "nominal_born_residual": born_residual,
        "measurement_rf_regularization": rf_regularizer,
        "session_hierarchical_prior": session_prior,
    }


__all__ = [
    "REAL_STATIC_ALLOWED_LOSSES", "TORCH_AVAILABLE", "TrackBRealStaticLossWeightsV1",
    "calibration_log_score", "cross_repeat_or_thinning_consistency", "native_global_multi_nll",
    "native_masked_four_cell_nll", "ratio_of_sums_query_absolute_error",
    "trackb_masked_real_static_objective", "uncertainty_log_score",
    "validate_global_multi_scope", "validate_real_static_loss_plan",
]
