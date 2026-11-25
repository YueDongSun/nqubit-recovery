"""V2.4-history inverse Transformer for Track B logical amp/phase guidance.

This model is trained directly from V2.4 feedback-history inputs and feeds the
model-only ten-round guidance protocol.  Its action heads consume only the
inverse Transformer representation: no analytic-action or direct linear bypass
can emit amplitude/phase guidance outside that representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math

import torch
import torch.nn as nn


V24_HISTORY_TOKEN_DIMENSION = 7
V24_MODE_TOKEN_DIMENSION = 2
TRACKB_MAX_HISTORY_ROUNDS = 10
V24_HISTORY_GAIN_FEATURE_INDEX = V24_HISTORY_TOKEN_DIMENSION - 1
INVERSE_TRANSFORMER_SCHEMA = "TrackBInverseTransformerV1"
LEGACY_LINEAR_MEAN_ACTION_VALUE = "legacy_linear_mean_v0"
NONLINEAR_ACTION_SQUARE_VALUE = "nonlinear_action_square_v1"
ACTION_VALUE_FEATURE_KINDS = (
    LEGACY_LINEAR_MEAN_ACTION_VALUE,
    NONLINEAR_ACTION_SQUARE_VALUE,
)
LEGACY_CENTER_AFTER_BOUND_PROJECTION = "legacy_center_after_bound_v0"
BOUNDED_GAUGE_PROJECTION = "bounded_gauge_projection_v1"
CENTERED_LOGIT_BOUNDED_GAUGE_PROJECTION = "centered_logit_bounded_gauge_v2"
TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION = "train_rms_centered_bounded_gauge_v3"
ACTION_PROJECTION_KINDS = (
    LEGACY_CENTER_AFTER_BOUND_PROJECTION,
    BOUNDED_GAUGE_PROJECTION,
    CENTERED_LOGIT_BOUNDED_GAUGE_PROJECTION,
    TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION,
)
RISK_NOT_TRAINED = "not_trained_v0"
RISK_TRAINED_UNCALIBRATED = "trained_uncalibrated_v1"
RISK_CALIBRATED = "calibrated_v1"
RISK_QUALIFICATION_STATES = (
    RISK_NOT_TRAINED,
    RISK_TRAINED_UNCALIBRATED,
    RISK_CALIBRATED,
)
RAW_GAIN_PARAMETERIZATION_V1 = "raw_gain_v1"
STANDARDIZED_LATENT_GAIN_PARAMETERIZATION_V1 = "standardized_latent_gain_v1"
GAIN_PARAMETERIZATION_KINDS = (
    RAW_GAIN_PARAMETERIZATION_V1,
    STANDARDIZED_LATENT_GAIN_PARAMETERIZATION_V1,
)
LEGACY_ORACLE_HISTORY_GAIN_INPUT = "oracle_history_gain_v0"
UNOBSERVED_ZERO_HISTORY_GAIN_INPUT = "unobserved_zero_v1"
HISTORY_GAIN_INPUT_KINDS = (
    LEGACY_ORACLE_HISTORY_GAIN_INPUT,
    UNOBSERVED_ZERO_HISTORY_GAIN_INPUT,
)


@dataclass(frozen=True)
class TrackBInverseTransformerConfigV1:
    """Serializable architecture identity for V2.4 ten-round control learning."""

    system_type: str
    mode_count: int
    width: int = 64
    heads: int = 4
    blocks: int = 2
    amplitude_bound: float = 0.35
    phase_bound: float = 0.70
    action_value_feature_kind: str = LEGACY_LINEAR_MEAN_ACTION_VALUE
    action_projection_kind: str = LEGACY_CENTER_AFTER_BOUND_PROJECTION
    risk_qualification_state: str = RISK_NOT_TRAINED
    schema: str = INVERSE_TRANSFORMER_SCHEMA
    gain_parameterization: str = RAW_GAIN_PARAMETERIZATION_V1
    gain_center: float = 0.0
    gain_scale: float = 1.0
    amplitude_decode_scale_by_round: tuple[float, ...] = ()
    phase_decode_scale_by_round: tuple[float, ...] = ()
    history_gain_input_kind: str = LEGACY_ORACLE_HISTORY_GAIN_INPUT

    def validate(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("inverse Transformer supports only 3qubit or 4qubit")
        if self.mode_count != int(self.system_type[0]):
            raise ValueError("inverse Transformer system type and mode count differ")
        if any(type(value) is not int or value <= 0 for value in (
            self.width, self.heads, self.blocks,
        )) or self.width % self.heads:
            raise ValueError("inverse Transformer width/heads/blocks are invalid")
        for value in (self.amplitude_bound, self.phase_bound):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("inverse Transformer logical action bounds must be positive")
        if self.action_value_feature_kind not in ACTION_VALUE_FEATURE_KINDS:
            raise ValueError("inverse Transformer action-value feature kind is unsupported")
        if self.action_projection_kind not in ACTION_PROJECTION_KINDS:
            raise ValueError("inverse Transformer action projection kind is unsupported")
        if self.history_gain_input_kind not in HISTORY_GAIN_INPUT_KINDS:
            raise ValueError("inverse Transformer history gain input kind is unsupported")
        for name in ("amplitude_decode_scale_by_round", "phase_decode_scale_by_round"):
            scales = getattr(self, name)
            if not isinstance(scales, (tuple, list)):
                raise ValueError(f"{name} must be a sequence")
            if self.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION:
                if (len(scales) != TRACKB_MAX_HISTORY_ROUNDS
                        or any(type(value) not in (int, float)
                               or not math.isfinite(float(value)) or float(value) <= 0.0
                               for value in scales)):
                    raise ValueError(f"{name} requires ten finite positive TRAIN scales")
            elif scales:
                raise ValueError(f"{name} is only supported by the TRAIN-RMS decoder")
        if self.risk_qualification_state not in RISK_QUALIFICATION_STATES:
            raise ValueError("inverse Transformer risk qualification state is unsupported")
        if self.risk_qualification_state == RISK_CALIBRATED:
            raise ValueError("inverse Transformer cannot self-declare calibrated confidence")
        if self.gain_parameterization not in GAIN_PARAMETERIZATION_KINDS:
            raise ValueError("inverse Transformer gain parameterization is unsupported")
        if type(self.gain_center) not in (int, float) or not math.isfinite(float(self.gain_center)):
            raise ValueError("inverse Transformer gain center must be a finite real scalar")
        if (type(self.gain_scale) not in (int, float)
                or not math.isfinite(float(self.gain_scale))
                or float(self.gain_scale) <= 0.0):
            raise ValueError("inverse Transformer gain scale must be a finite positive real scalar")
        if (self.gain_parameterization == RAW_GAIN_PARAMETERIZATION_V1
                and (float(self.gain_center) != 0.0 or float(self.gain_scale) != 1.0)):
            raise ValueError("raw gain parameterization must use zero center and unit scale")
        if self.schema != INVERSE_TRANSFORMER_SCHEMA:
            raise ValueError("unsupported inverse Transformer schema")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = {
            "schema": self.schema,
            "system_type": self.system_type,
            "mode_count": self.mode_count,
            "width": self.width,
            "heads": self.heads,
            "blocks": self.blocks,
            "amplitude_bound": self.amplitude_bound,
            "phase_bound": self.phase_bound,
            "action_value_feature_kind": self.action_value_feature_kind,
            "action_projection_kind": self.action_projection_kind,
            "risk_qualification_state": self.risk_qualification_state,
            "gain_parameterization": self.gain_parameterization,
            "gain_center": self.gain_center,
            "gain_scale": self.gain_scale,
        }
        if self.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION:
            payload["amplitude_decode_scale_by_round"] = list(self.amplitude_decode_scale_by_round)
            payload["phase_decode_scale_by_round"] = list(self.phase_decode_scale_by_round)
        if self.history_gain_input_kind != LEGACY_ORACLE_HISTORY_GAIN_INPUT:
            payload["history_gain_input_kind"] = self.history_gain_input_kind
        return payload


def require_gain_parameterization_binding(
    config: TrackBInverseTransformerConfigV1,
    target_normalization: Mapping[str, Any],
) -> None:
    """Require a standardized gain head to bind its decoding statistics exactly."""
    config.validate()
    if config.gain_parameterization == RAW_GAIN_PARAMETERIZATION_V1:
        return
    gain_center = target_normalization.get("gain_center")
    gain_scale = target_normalization.get("gain_scale")
    if type(gain_center) not in (int, float) or type(gain_scale) not in (int, float):
        raise ValueError("Track B gain parameterization lacks target normalization statistics")
    if (not math.isfinite(float(gain_center)) or not math.isfinite(float(gain_scale))
            or float(gain_scale) <= 0.0):
        raise ValueError("Track B gain parameterization target normalization is invalid")
    if (float(config.gain_center) != float(gain_center)
            or float(config.gain_scale) != float(gain_scale)):
        raise ValueError("Track B gain parameterization differs from target normalization")


def require_action_decode_scale_binding(
    config: TrackBInverseTransformerConfigV1,
    target_normalization: Mapping[str, Any],
) -> None:
    """Bind the opt-in action decoder to TRAIN-only per-round RMS targets."""
    config.validate()
    if config.action_projection_kind != TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION:
        return
    if target_normalization.get("action_loss_normalization_kind") != "per_round_v1":
        raise ValueError("Track B TRAIN-RMS decoder requires per-round TRAIN normalization")
    for decode_name, target_name in (
        ("amplitude_decode_scale_by_round", "amplitude_scale_by_round"),
        ("phase_decode_scale_by_round", "phase_scale_by_round"),
    ):
        target_scales = target_normalization.get(target_name)
        if (not isinstance(target_scales, (tuple, list))
                or len(target_scales) != TRACKB_MAX_HISTORY_ROUNDS
                or any(type(value) not in (int, float)
                       or not math.isfinite(float(value)) or float(value) <= 0.0
                       for value in target_scales)
                or tuple(getattr(config, decode_name)) != tuple(target_scales)):
            raise ValueError(f"Track B TRAIN-RMS decoder differs from {target_name}")


def _require_tensor(value: Any, name: str, *, dtype: torch.dtype | None = None,
                    dimensions: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} has the wrong dtype")
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError(f"{name} has the wrong rank")
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _masked_zero_mean(value: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
    weight = mode_mask.to(value.dtype)
    mean = (value * weight).sum(-1, keepdim=True) / weight.sum(-1, keepdim=True).clamp_min(1.0)
    return (value - mean) * weight


def project_bounded_gauge_action(
    value: torch.Tensor,
    mode_mask: torch.Tensor,
    *,
    bound: float,
) -> torch.Tensor:
    """Project one action coordinate into the registered gauge/bound intersection."""
    if not math.isfinite(float(bound)) or float(bound) <= 0.0:
        raise ValueError("Track B gauge projection bound must be finite and positive")
    centered = _masked_zero_mean(value, mode_mask)
    weight = mode_mask.to(centered.dtype)
    maximum = (centered.abs() * weight).amax(dim=-1, keepdim=True)
    scale = torch.clamp(float(bound) / maximum.clamp_min(float(bound)), max=1.0)
    return centered * scale * weight


class TrackBInverseTransformerV1(nn.Module):
    """Inverse logical controller with no action-producing residual bypass."""

    def __init__(self, config: TrackBInverseTransformerConfigV1) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.history_encoder = nn.Linear(V24_HISTORY_TOKEN_DIMENSION, config.width)
        self.last_history_encoder = nn.Linear(V24_HISTORY_TOKEN_DIMENSION, config.width)
        self.prefix_length_encoder = nn.Linear(1, config.width)
        self.mode_encoder = nn.Linear(2 * V24_MODE_TOKEN_DIMENSION, config.width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=4 * config.width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.inverse_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.blocks,
            enable_nested_tensor=False,
        )
        self.action_head = nn.Linear(config.width, 2)
        if config.action_value_feature_kind == LEGACY_LINEAR_MEAN_ACTION_VALUE:
            self.gain_action_encoder: nn.Module = nn.Linear(2, config.width)
            self.action_value_encoder: nn.Module | None = None
        else:
            self.gain_action_encoder = nn.Sequential(
                nn.Linear(4, config.width),
                nn.GELU(),
                nn.Linear(config.width, config.width),
                nn.GELU(),
            )
            self.action_value_encoder = nn.Sequential(
                nn.Linear(2 * config.width, config.width),
                nn.GELU(),
                nn.Linear(config.width, config.width),
                nn.GELU(),
            )
        self.gain_head = nn.Linear(config.width, 1)
        self.risk_head = nn.Linear(config.width, 2)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)

    def _validate_inputs(self, inputs: Mapping[str, Any]) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        required = {
            "history_tokens",
            "history_mask",
            "noisy_measurement_tokens",
            "target_mode_tokens",
            "mode_mask",
        }
        missing = required - set(inputs)
        if missing:
            raise ValueError(f"inverse Transformer inputs are missing: {sorted(missing)}")
        unknown = set(inputs) - required
        if unknown:
            raise ValueError(f"inverse Transformer received unknown inputs: {sorted(unknown)}")
        history = _require_tensor(inputs["history_tokens"], "history_tokens", dimensions=4)
        history_mask = _require_tensor(inputs["history_mask"], "history_mask", dtype=torch.bool, dimensions=2)
        noisy = _require_tensor(inputs["noisy_measurement_tokens"], "noisy_measurement_tokens", dimensions=3)
        target = _require_tensor(inputs["target_mode_tokens"], "target_mode_tokens", dimensions=3)
        mode_mask = _require_tensor(inputs["mode_mask"], "mode_mask", dtype=torch.bool, dimensions=2)
        batch_size = noisy.shape[0]
        modes = self.config.mode_count
        if (history.shape[0] != batch_size or history.shape[2:] != (modes, V24_HISTORY_TOKEN_DIMENSION)
                or history_mask.shape != history.shape[:2]
                or noisy.shape != (batch_size, modes, V24_MODE_TOKEN_DIMENSION)
                or target.shape != noisy.shape
                or mode_mask.shape != (batch_size, modes)):
            raise ValueError("inverse Transformer input shapes do not match the V2.4/Track B contract")
        if not bool(mode_mask.any(-1).all()):
            raise ValueError("each inverse Transformer row requires one active mode")
        if self.config.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION:
            if (history.shape[1] > TRACKB_MAX_HISTORY_ROUNDS
                    or bool((history_mask[:, 1:] & ~history_mask[:, :-1]).any())
                    or bool((history_mask.sum(1) >= TRACKB_MAX_HISTORY_ROUNDS).any())):
                raise ValueError("TRAIN-RMS decoder requires a left-aligned 0..9-round history prefix")
        return history, history_mask, noisy, target, mode_mask

    def forward(self, inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        history, history_mask, noisy, target, mode_mask = self._validate_inputs(inputs)
        if self.config.history_gain_input_kind == UNOBSERVED_ZERO_HISTORY_GAIN_INPUT:
            history = history.clone()
            history[..., V24_HISTORY_GAIN_FEATURE_INDEX] = 0.0
        encoded_history = self.history_encoder(history)
        history_weight = history_mask.to(encoded_history.dtype)[:, :, None, None]
        history_context = (encoded_history * history_weight).sum(1) / history_weight.sum(1).clamp_min(1.0)
        mode_context = self.mode_encoder(torch.cat((noisy, target), dim=-1))
        history_length = history_mask.sum(1)
        if history.shape[1]:
            last_index = (history_length - 1).clamp_min(0)
            last_history = history[torch.arange(history.shape[0], device=history.device), last_index]
            last_context = self.last_history_encoder(last_history) * (history_length > 0)[:, None, None]
        else:
            last_context = torch.zeros_like(mode_context)
        prefix_length = (history_length.to(mode_context.dtype) / TRACKB_MAX_HISTORY_ROUNDS).unsqueeze(-1)
        prefix_context = self.prefix_length_encoder(prefix_length)[:, None, :]
        inverse_input = mode_context + history_context + last_context + prefix_context
        encoded = self.inverse_transformer(
            inverse_input,
            src_key_padding_mask=~mode_mask,
        )
        raw_action = self.action_head(encoded)
        if self.config.action_projection_kind in (
            CENTERED_LOGIT_BOUNDED_GAUGE_PROJECTION,
            TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION,
        ):
            # Remove common-mode logits before tanh can saturate their relative action.
            raw_action = torch.stack(tuple(
                _masked_zero_mean(raw_action[..., coordinate], mode_mask)
                for coordinate in range(2)
            ), dim=-1)
        if self.config.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION:
            amplitude_scales = torch.as_tensor(
                self.config.amplitude_decode_scale_by_round, dtype=raw_action.dtype,
                device=raw_action.device,
            )[history_length, None]
            phase_scales = torch.as_tensor(
                self.config.phase_decode_scale_by_round, dtype=raw_action.dtype,
                device=raw_action.device,
            )[history_length, None]
            amplitude_raw = self.config.amplitude_bound * torch.tanh(
                raw_action[..., 0] * amplitude_scales / self.config.amplitude_bound)
            phase_raw = self.config.phase_bound * torch.tanh(
                raw_action[..., 1] * phase_scales / self.config.phase_bound)
        else:
            amplitude_raw = torch.tanh(raw_action[..., 0]) * self.config.amplitude_bound
            phase_raw = torch.tanh(raw_action[..., 1]) * self.config.phase_bound
        if self.config.action_projection_kind == LEGACY_CENTER_AFTER_BOUND_PROJECTION:
            amplitude_delta = _masked_zero_mean(amplitude_raw, mode_mask)
            phase_delta = _masked_zero_mean(phase_raw, mode_mask)
        else:
            amplitude_delta = project_bounded_gauge_action(
                amplitude_raw,
                mode_mask,
                bound=self.config.amplitude_bound,
            )
            phase_delta = project_bounded_gauge_action(
                phase_raw,
                mode_mask,
                bound=self.config.phase_bound,
            )
        pooled = (encoded * mode_mask[..., None].to(encoded.dtype)).sum(1) / mode_mask.sum(1, keepdim=True)
        proposed_action = torch.stack((amplitude_delta, phase_delta), dim=-1)
        if self.config.action_value_feature_kind == LEGACY_LINEAR_MEAN_ACTION_VALUE:
            action_context = self.gain_action_encoder(proposed_action)
            pooled_action = (
                action_context * mode_mask[..., None].to(action_context.dtype)
            ).sum(1) / mode_mask.sum(1, keepdim=True)
            value_context = pooled + pooled_action
        else:
            action_features = torch.cat((proposed_action, proposed_action.square()), dim=-1)
            action_context = self.gain_action_encoder(action_features)
            if self.action_value_encoder is None:
                raise RuntimeError("nonlinear Track B action-value encoder is unavailable")
            action_value_tokens = self.action_value_encoder(torch.cat((encoded, action_context), dim=-1))
            value_context = (
                action_value_tokens * mode_mask[..., None].to(action_value_tokens.dtype)
            ).sum(1) / mode_mask.sum(1, keepdim=True)
        risk_context = (
            pooled if self.config.action_value_feature_kind == LEGACY_LINEAR_MEAN_ACTION_VALUE
            else value_context
        )
        risk = torch.sigmoid(self.risk_head(risk_context))
        gain_latent = self.gain_head(value_context).squeeze(-1)
        teacher_transition_gain = (
            gain_latent
            if self.config.gain_parameterization == RAW_GAIN_PARAMETERIZATION_V1
            else self.config.gain_center + self.config.gain_scale * gain_latent
        )
        confidence_qualified = torch.full(
            (amplitude_delta.shape[0],),
            False,
            dtype=torch.bool,
            device=amplitude_delta.device,
        )
        return {
            "amplitude_delta": amplitude_delta,
            "phase_delta": phase_delta,
            "predicted_gain": teacher_transition_gain,
            "predicted_teacher_transition_gain": teacher_transition_gain,
            "predicted_harm_probability": risk[:, 0],
            "predicted_stop_probability": risk[:, 1],
            "confidence_qualification": self.config.risk_qualification_state,
            "confidence_qualified": confidence_qualified,
            "inverse_representation": encoded,
            "hardware_action_authorized": torch.zeros(
                amplitude_delta.shape[0], dtype=torch.bool, device=amplitude_delta.device),
        }


@dataclass(frozen=True)
class TrackBInverseTransformerDependencyV1:
    """Ablation measurement for V2.4 history and target-token reliance."""

    action_norm: float
    history_ablation_delta_norm: float
    target_ablation_delta_norm: float
    history_dependency_ratio: float | None
    target_dependency_ratio: float | None
    status: str
    minimum_dependency_ratio: float
    hardware_action_authorized: bool = False
    schema: str = "TrackBInverseTransformerDependencyV1"

    def validate(self) -> None:
        if self.status not in ("EVALUABLE", "NOT_EVALUABLE_NEAR_ZERO_ACTION"):
            raise ValueError("inverse Transformer dependency status is invalid")
        for name in (
            "action_norm",
            "history_ablation_delta_norm",
            "target_ablation_delta_norm",
            "minimum_dependency_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.status == "EVALUABLE":
            for name in ("history_dependency_ratio", "target_dependency_ratio"):
                value = getattr(self, name)
                if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
                    raise ValueError(f"{name} must be finite when dependency is evaluable")
        elif self.history_dependency_ratio is not None or self.target_dependency_ratio is not None:
            raise ValueError("near-zero actions cannot claim a dependency ratio")
        if self.hardware_action_authorized:
            raise ValueError("dependency analysis cannot grant hardware authority")
        if self.schema != "TrackBInverseTransformerDependencyV1":
            raise ValueError("unsupported inverse Transformer dependency schema")

    @property
    def dependency_sufficient(self) -> bool | None:
        if self.status != "EVALUABLE":
            return None
        return bool(
            self.history_dependency_ratio is not None
            and self.target_dependency_ratio is not None
            and self.history_dependency_ratio >= self.minimum_dependency_ratio
            and self.target_dependency_ratio >= self.minimum_dependency_ratio
        )


def measure_inverse_transformer_dependency(
    model: TrackBInverseTransformerV1,
    inputs: Mapping[str, Any],
    *,
    minimum_dependency_ratio: float = 0.05,
    near_zero_action_norm: float = 1e-8,
) -> TrackBInverseTransformerDependencyV1:
    """Measure action sensitivity to V2.4 history and target inputs."""
    if not isinstance(model, TrackBInverseTransformerV1):
        raise TypeError("dependency measurement requires the exact Track B inverse Transformer")
    if not math.isfinite(minimum_dependency_ratio) or minimum_dependency_ratio < 0.0:
        raise ValueError("minimum dependency ratio must be finite and nonnegative")
    if not math.isfinite(near_zero_action_norm) or near_zero_action_norm <= 0.0:
        raise ValueError("near-zero action norm must be positive")
    model.eval()
    with torch.no_grad():
        full = model(inputs)
        history_ablated = dict(inputs)
        history_ablated["history_tokens"] = torch.zeros_like(inputs["history_tokens"])
        if model.config.action_projection_kind != TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION:
            history_ablated["history_mask"] = torch.zeros_like(inputs["history_mask"])
        # V3 keeps the prefix mask so TRAIN-RMS decoder scale does not change
        # during the history-content ablation.
        without_history = model(history_ablated)
        target_ablated = dict(inputs)
        target_ablated["target_mode_tokens"] = torch.zeros_like(inputs["target_mode_tokens"])
        without_target = model(target_ablated)
    full_action = torch.stack((full["amplitude_delta"], full["phase_delta"]), dim=-1)
    history_action = torch.stack((without_history["amplitude_delta"], without_history["phase_delta"]), dim=-1)
    target_action = torch.stack((without_target["amplitude_delta"], without_target["phase_delta"]), dim=-1)
    action_norm = float(torch.linalg.vector_norm(full_action).cpu())
    history_delta = float(torch.linalg.vector_norm(full_action - history_action).cpu())
    target_delta = float(torch.linalg.vector_norm(full_action - target_action).cpu())
    if action_norm <= near_zero_action_norm:
        result = TrackBInverseTransformerDependencyV1(
            action_norm=action_norm,
            history_ablation_delta_norm=history_delta,
            target_ablation_delta_norm=target_delta,
            history_dependency_ratio=None,
            target_dependency_ratio=None,
            status="NOT_EVALUABLE_NEAR_ZERO_ACTION",
            minimum_dependency_ratio=minimum_dependency_ratio,
        )
    else:
        result = TrackBInverseTransformerDependencyV1(
            action_norm=action_norm,
            history_ablation_delta_norm=history_delta,
            target_ablation_delta_norm=target_delta,
            history_dependency_ratio=history_delta / action_norm,
            target_dependency_ratio=target_delta / action_norm,
            status="EVALUABLE",
            minimum_dependency_ratio=minimum_dependency_ratio,
        )
    result.validate()
    return result


__all__ = [
    "INVERSE_TRANSFORMER_SCHEMA",
    "V24_HISTORY_TOKEN_DIMENSION",
    "V24_MODE_TOKEN_DIMENSION",
    "V24_HISTORY_GAIN_FEATURE_INDEX",
    "TRACKB_MAX_HISTORY_ROUNDS",
    "LEGACY_LINEAR_MEAN_ACTION_VALUE",
    "NONLINEAR_ACTION_SQUARE_VALUE",
    "LEGACY_CENTER_AFTER_BOUND_PROJECTION",
    "BOUNDED_GAUGE_PROJECTION",
    "CENTERED_LOGIT_BOUNDED_GAUGE_PROJECTION",
    "TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION",
    "RAW_GAIN_PARAMETERIZATION_V1",
    "STANDARDIZED_LATENT_GAIN_PARAMETERIZATION_V1",
    "GAIN_PARAMETERIZATION_KINDS",
    "LEGACY_ORACLE_HISTORY_GAIN_INPUT",
    "UNOBSERVED_ZERO_HISTORY_GAIN_INPUT",
    "HISTORY_GAIN_INPUT_KINDS",
    "RISK_NOT_TRAINED",
    "RISK_TRAINED_UNCALIBRATED",
    "RISK_CALIBRATED",
    "TrackBInverseTransformerConfigV1",
    "TrackBInverseTransformerDependencyV1",
    "TrackBInverseTransformerV1",
    "project_bounded_gauge_action",
    "require_gain_parameterization_binding",
    "require_action_decode_scale_binding",
    "measure_inverse_transformer_dependency",
]
