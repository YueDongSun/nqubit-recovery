"""Mask-safe Track A multi-task, calibration, and decision losses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from ml_recon.physics_hybrid_v10.gain_quantiles import GAIN_QUANTILE_LEVELS


@dataclass(frozen=True)
class TrackALossWeights:
    measurement: float = 1.0
    structure: float = 1.0
    uncertainty: float = 0.5
    certificate: float = 0.5
    csm: float = 0.5
    recoverability: float = 1.0
    recoverability_class: float = 0.5
    action: float = 1.0
    gain_uncertainty: float = 0.5
    harm: float = 1.0
    stop: float = 0.5
    abstention: float = 0.5
    next_measurement: float = 0.5
    risk: float = 1.0
    decision: float = 0.5
    coverage: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return _validated_weight_map(self.__dict__)


def _validated_weight_map(weights: Mapping[str, object]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for name, raw_value in weights.items():
        if not isinstance(name, str) or not name:
            raise TypeError("loss weight names must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise TypeError(f"loss weight {name} must be a real scalar")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("loss weights must be finite and non-negative")
        result[name] = value
    return result


def _expanded_boolean_mask(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("masked values must be tensors")
    if not value.dtype.is_floating_point:
        raise TypeError("masked values must use a floating-point dtype")
    if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
        raise TypeError("loss masks must be strict boolean tensors")
    if mask.ndim > value.ndim:
        raise ValueError("loss mask cannot broadcast to the value shape")
    for axis, mask_size in enumerate(mask.shape):
        if mask_size not in (1, value.shape[axis]):
            raise ValueError("loss mask cannot prefix-broadcast to the value shape")
    view_shape = tuple(mask.shape) + (1,) * (value.ndim - mask.ndim)
    try:
        return mask.to(device=value.device).reshape(view_shape).expand_as(value)
    except RuntimeError as exc:
        raise ValueError("loss mask cannot broadcast to the value shape") from exc


def _require_same_shape(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    description: str,
) -> None:
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError(f"{description} values must be tensors")
    if first.shape != second.shape:
        raise ValueError(f"{description} tensors must have identical shape")


def _safe_masked_value(
    value: torch.Tensor,
    expanded_mask: torch.Tensor,
    *,
    fill_value: float,
) -> torch.Tensor:
    fill = torch.as_tensor(fill_value, dtype=value.dtype, device=value.device)
    return torch.where(expanded_mask, value, fill)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reduce a floating loss with strict prefix-broadcast mask semantics.

    ``torch.where`` prevents masked NaN/Inf values from reaching the numerator.
    A fully masked reduction remains connected to ``value`` and therefore
    backpropagates an exact zero instead of returning a detached constant.
    """

    expanded_mask = _expanded_boolean_mask(value, mask)
    active_values = value[expanded_mask]
    if active_values.numel() and not bool(torch.all(torch.isfinite(active_values))):
        raise ValueError("active masked loss values must be finite")
    selected = torch.where(expanded_mask, value, torch.zeros_like(value))
    denominator = expanded_mask.sum().to(dtype=value.dtype)
    return selected.sum() / denominator.clamp_min(1.0)


def masked_binary_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Probability BCE that sanitizes masked predictions and targets first."""

    _require_same_shape(probability, target, description="binary loss")
    expanded_mask = _expanded_boolean_mask(probability, mask)
    safe_probability = _safe_masked_value(
        probability, expanded_mask, fill_value=0.5)
    safe_target = _safe_masked_value(target, expanded_mask, fill_value=0.0)
    with torch.autocast(device_type=probability.device.type, enabled=False):
        raw_loss = F.binary_cross_entropy(
            safe_probability.float(), safe_target.float(), reduction="none")
    return masked_mean(raw_loss, expanded_mask)


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Class-index cross entropy over authoritative, mask-valid rows."""

    if not isinstance(logits, torch.Tensor) or not logits.dtype.is_floating_point:
        raise TypeError("cross-entropy logits must be a floating-point tensor")
    if not isinstance(target, torch.Tensor) or target.dtype is torch.bool \
            or target.dtype.is_floating_point:
        raise TypeError("cross-entropy targets must use an integer dtype")
    if logits.ndim < 2 or logits.shape[:-1] != target.shape:
        raise ValueError("cross-entropy logits and target shapes do not align")
    target_mask = _expanded_boolean_mask(target.to(dtype=logits.dtype), mask)
    logit_mask = _expanded_boolean_mask(logits, mask)
    safe_logits = _safe_masked_value(logits, logit_mask, fill_value=0.0)
    safe_target = torch.where(
        target_mask,
        target.to(device=logits.device, dtype=torch.long),
        torch.zeros_like(target, device=logits.device, dtype=torch.long),
    )
    raw_loss = F.cross_entropy(
        safe_logits.float().reshape(-1, logits.shape[-1]),
        safe_target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    return masked_mean(raw_loss, target_mask)


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 loss with invalid target rows isolated before evaluation."""

    _require_same_shape(prediction, target, description="smooth-L1")
    expanded_mask = _expanded_boolean_mask(prediction, mask)
    safe_prediction = _safe_masked_value(
        prediction, expanded_mask, fill_value=0.0)
    safe_target = _safe_masked_value(target, expanded_mask, fill_value=0.0)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        raw_loss = F.smooth_l1_loss(
            safe_prediction.float(), safe_target.float(), reduction="none")
    return masked_mean(raw_loss, expanded_mask)


def masked_heteroscedastic_nll(
    prediction: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    minimum_variance: float = 1e-6,
) -> torch.Tensor:
    """Gaussian proper score for a published mean/variance uncertainty pair."""

    _require_same_shape(prediction, target, description="uncertainty")
    _require_same_shape(prediction, variance, description="uncertainty")
    if not math.isfinite(minimum_variance) or minimum_variance <= 0.0:
        raise ValueError("minimum variance must be finite and positive")
    expanded_mask = _expanded_boolean_mask(prediction, mask)
    safe_prediction = _safe_masked_value(
        prediction, expanded_mask, fill_value=0.0)
    safe_target = _safe_masked_value(target, expanded_mask, fill_value=0.0)
    safe_variance = _safe_masked_value(
        variance, expanded_mask, fill_value=1.0).clamp_min(minimum_variance)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        residual = safe_prediction.float() - safe_target.float()
        variance_value = safe_variance.float()
        raw_loss = 0.5 * (
            torch.log(variance_value) + residual.square() / variance_value)
    return masked_mean(raw_loss, expanded_mask)


def masked_pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    quantile_levels: Sequence[float] = GAIN_QUANTILE_LEVELS,
) -> torch.Tensor:
    """Registered gain-quantile proper score with authoritative row masking."""

    if not isinstance(prediction, torch.Tensor) or not prediction.dtype.is_floating_point:
        raise TypeError("quantile predictions must be a floating-point tensor")
    if not isinstance(target, torch.Tensor) or not target.dtype.is_floating_point:
        raise TypeError("quantile targets must be a floating-point tensor")
    levels = tuple(float(level) for level in quantile_levels)
    if prediction.ndim < 1 or prediction.shape[:-1] != target.shape \
            or prediction.shape[-1] != len(levels):
        raise ValueError("quantile prediction and target shapes do not align")
    if not levels or any(not math.isfinite(level) or not 0.0 < level < 1.0
                         for level in levels) or tuple(sorted(levels)) != levels:
        raise ValueError("quantile levels must be finite, increasing, and inside (0,1)")
    target_mask = _expanded_boolean_mask(target, mask)
    prediction_mask = _expanded_boolean_mask(prediction, mask)
    safe_prediction = _safe_masked_value(
        prediction, prediction_mask, fill_value=0.0)
    safe_target = _safe_masked_value(target, target_mask, fill_value=0.0)
    level_tensor = prediction.new_tensor(levels)
    error = safe_target.unsqueeze(-1) - safe_prediction
    raw_loss = torch.maximum(
        level_tensor * error,
        (level_tensor - 1.0) * error,
    ).mean(-1)
    return masked_mean(raw_loss, target_mask)


def published_head_losses(
    classifier_output: Mapping[str, torch.Tensor],
    feedback_output: Mapping[str, torch.Tensor],
    classifier_targets: Mapping[str, torch.Tensor],
    feedback_targets: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Bind published Track A heads only to authoritative V10 targets.

    ``model_uncertainty`` is trained as the variance in a heteroscedastic
    proper score around the OOD probability. Certificate status uses the
    registered nine-class target. Feedback gain, recoverability, harm, stop,
    and derived abstention use their V2.4 truth fields and availability masks.
    No surrogate target is invented for outputs without an authoritative field.
    """

    classifier_required = {
        "OOD_probability", "model_uncertainty", "certificate_status_logits"}
    classifier_target_required = {
        "target_ood", "row_valid_mask", "target_certificate_status",
        "target_certificate_mask"}
    feedback_required = {
        "gain_quantiles", "recoverability", "recoverability_class_logits",
        "harmful_probability", "stop_probability", "abstain_probability"}
    feedback_target_required = {
        "target_gain", "target_recoverability", "target_recoverable_class",
        "recoverability_mask", "target_harm", "target_stop",
        "oracle_action_mask"}
    missing = (
        classifier_required - set(classifier_output)
        | classifier_target_required - set(classifier_targets)
        | feedback_required - set(feedback_output)
        | feedback_target_required - set(feedback_targets)
    )
    if missing:
        raise ValueError(f"published head losses are missing fields: {sorted(missing)}")

    oracle_mask = feedback_targets["oracle_action_mask"]
    abstention_target = torch.maximum(
        feedback_targets["target_harm"], feedback_targets["target_stop"])
    return {
        "uncertainty": masked_heteroscedastic_nll(
            classifier_output["OOD_probability"],
            classifier_output["model_uncertainty"],
            classifier_targets["target_ood"],
            classifier_targets["row_valid_mask"],
        ),
        "certificate": masked_cross_entropy(
            classifier_output["certificate_status_logits"],
            classifier_targets["target_certificate_status"],
            classifier_targets["target_certificate_mask"],
        ),
        "gain_uncertainty": masked_pinball_loss(
            feedback_output["gain_quantiles"],
            feedback_targets["target_gain"],
            oracle_mask,
        ),
        "recoverability": masked_smooth_l1_loss(
            feedback_output["recoverability"],
            feedback_targets["target_recoverability"],
            feedback_targets["recoverability_mask"],
        ),
        "recoverability_class": masked_cross_entropy(
            feedback_output["recoverability_class_logits"],
            feedback_targets["target_recoverable_class"],
            feedback_targets["recoverability_mask"],
        ),
        "harm": masked_binary_cross_entropy(
            feedback_output["harmful_probability"],
            feedback_targets["target_harm"],
            oracle_mask,
        ),
        "stop": masked_binary_cross_entropy(
            feedback_output["stop_probability"],
            feedback_targets["target_stop"],
            oracle_mask,
        ),
        "abstention": masked_binary_cross_entropy(
            feedback_output["abstain_probability"],
            abstention_target,
            oracle_mask,
        ),
    }


def balanced_multitask_loss(
    losses: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    weights: TrackALossWeights | Mapping[str, float],
) -> Dict[str, torch.Tensor]:
    weight_map = (
        weights.to_dict()
        if isinstance(weights, TrackALossWeights)
        else _validated_weight_map(weights)
    )
    unknown = set(losses) - set(weight_map)
    if unknown:
        raise ValueError(f"unregistered Track A losses: {sorted(unknown)}")
    missing_masks = set(losses) - set(masks)
    if missing_masks:
        raise ValueError(f"loss masks are missing: {sorted(missing_masks)}")
    reduced: Dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for name, value in losses.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("Track A losses must be tensors")
        component = masked_mean(value, masks[name])
        reduced[name] = component
        weighted = weight_map[name] * component
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("at least one Track A loss is required")
    reduced["loss"] = total
    return reduced


def posterior_coverage_penalty(
    empirical_coverage: torch.Tensor,
    nominal_coverage: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    _require_same_shape(
        empirical_coverage, nominal_coverage, description="coverage")
    expanded_mask = _expanded_boolean_mask(empirical_coverage, mask)
    safe_empirical = _safe_masked_value(
        empirical_coverage, expanded_mask, fill_value=0.0)
    safe_nominal = _safe_masked_value(
        nominal_coverage, expanded_mask, fill_value=0.0)
    return masked_mean((safe_empirical - safe_nominal).square(), expanded_mask)


def decision_aware_expected_cost_loss(
    predicted_cost: torch.Tensor,
    realized_cost: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    _require_same_shape(
        predicted_cost, realized_cost, description="decision cost")
    expanded_mask = _expanded_boolean_mask(predicted_cost, action_mask)
    safe_predicted = _safe_masked_value(
        predicted_cost, expanded_mask, fill_value=0.0)
    safe_realized = _safe_masked_value(
        realized_cost, expanded_mask, fill_value=0.0)
    return masked_mean((safe_predicted - safe_realized).square(), expanded_mask)
