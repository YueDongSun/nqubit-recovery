"""Research-only shared logical RF representation for Track B 3Q/4Q models.

The module operates only on the registered V2.4 phase-sensitive history domain.
It has no RF command, waveform, AWG, FPGA, or hardware execution dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math

import torch
import torch.nn as nn

from .inverse_transformer import project_bounded_gauge_action


MAX_TRACKB_MODES = 4
TRACKB_HISTORY_TOKEN_DIMENSION = 7
TRACKB_MODE_TOKEN_DIMENSION = 2
TRACKB_SHARED_OBSERVATION_DOMAIN = "phase_sensitive_history_v24"


def _require_finite_tensor(value: Any, name: str, *, dimensions: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != dimensions:
        raise ValueError(f"{name} must be a rank-{dimensions} tensor")
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite before shared RF encoding")
    return value


def _padded_mode_tensor(value: torch.Tensor, *, mode_count: int) -> torch.Tensor:
    if value.shape[-2] != mode_count:
        raise ValueError("Track B shared RF input mode count is inconsistent")
    if mode_count == MAX_TRACKB_MODES:
        return value
    pad_shape = (*value.shape[:-2], MAX_TRACKB_MODES - mode_count, value.shape[-1])
    return torch.cat((value, torch.zeros(pad_shape, dtype=value.dtype, device=value.device)), dim=-2)


@dataclass(frozen=True)
class TrackBSharedRFInputsV1:
    """V2.4 history input padded to four modes without inventing observations."""

    history_tokens: torch.Tensor
    history_mask: torch.Tensor
    noisy_measurement_tokens: torch.Tensor
    target_mode_tokens: torch.Tensor
    mode_mask: torch.Tensor
    system_type: str
    source_identity_sha: str
    observation_domain: str = TRACKB_SHARED_OBSERVATION_DOMAIN
    schema: str = "TrackBSharedRFInputsV1"

    @classmethod
    def from_trackb_inputs(
        cls,
        inputs: Mapping[str, Any],
        *,
        system_type: str,
        source_identity_sha: str,
    ) -> "TrackBSharedRFInputsV1":
        """Adapt one homogeneous 3Q or 4Q Track B batch to the shared shape."""
        if system_type not in ("3qubit", "4qubit"):
            raise ValueError("shared RF input supports only 3qubit or 4qubit")
        required = {
            "history_tokens",
            "history_mask",
            "noisy_measurement_tokens",
            "target_mode_tokens",
            "mode_mask",
        }
        if set(inputs) != required:
            raise ValueError("shared RF input requires exactly the registered Track B fields")
        mode_count = int(system_type[0])
        history = _require_finite_tensor(inputs["history_tokens"], "history_tokens", dimensions=4)
        history_mask = _require_finite_tensor(inputs["history_mask"], "history_mask", dimensions=2)
        noisy = _require_finite_tensor(inputs["noisy_measurement_tokens"], "noisy_measurement_tokens", dimensions=3)
        target = _require_finite_tensor(inputs["target_mode_tokens"], "target_mode_tokens", dimensions=3)
        mode_mask = _require_finite_tensor(inputs["mode_mask"], "mode_mask", dimensions=2)
        if history_mask.dtype != torch.bool or mode_mask.dtype != torch.bool:
            raise ValueError("shared RF masks must be boolean")
        if (history.shape[2:] != (mode_count, TRACKB_HISTORY_TOKEN_DIMENSION)
                or noisy.shape[1:] != (mode_count, TRACKB_MODE_TOKEN_DIMENSION)
                or target.shape != noisy.shape
                or history_mask.shape != history.shape[:2]
                or mode_mask.shape != noisy.shape[:2]):
            raise ValueError("shared RF input does not meet the V2.4 Track B shape contract")
        if (history.shape[0] != noisy.shape[0]
                or history_mask.shape[0] != noisy.shape[0]
                or mode_mask.shape[0] != noisy.shape[0]
                or history.device != noisy.device
                or target.device != noisy.device
                or history_mask.device != noisy.device
                or mode_mask.device != noisy.device
                or history.dtype != noisy.dtype
                or target.dtype != noisy.dtype):
            raise ValueError("shared RF inputs must have one consistent batch, device, and dtype")
        padded_mask = _padded_mode_tensor(mode_mask.unsqueeze(-1), mode_count=mode_count).squeeze(-1)
        result = cls(
            history_tokens=_padded_mode_tensor(history, mode_count=mode_count),
            history_mask=history_mask,
            noisy_measurement_tokens=_padded_mode_tensor(noisy, mode_count=mode_count),
            target_mode_tokens=_padded_mode_tensor(target, mode_count=mode_count),
            mode_mask=padded_mask,
            system_type=system_type,
            source_identity_sha=source_identity_sha,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.system_type not in ("3qubit", "4qubit") or len(self.source_identity_sha) != 64:
            raise ValueError("shared RF system/source identity is invalid")
        if self.observation_domain != TRACKB_SHARED_OBSERVATION_DOMAIN:
            raise ValueError("shared RF encoder does not adapt a different observation domain")
        history = _require_finite_tensor(self.history_tokens, "history_tokens", dimensions=4)
        history_mask = _require_finite_tensor(self.history_mask, "history_mask", dimensions=2)
        noisy = _require_finite_tensor(self.noisy_measurement_tokens, "noisy_measurement_tokens", dimensions=3)
        target = _require_finite_tensor(self.target_mode_tokens, "target_mode_tokens", dimensions=3)
        mode_mask = _require_finite_tensor(self.mode_mask, "mode_mask", dimensions=2)
        if history_mask.dtype != torch.bool or mode_mask.dtype != torch.bool:
            raise ValueError("shared RF masks must be boolean")
        if (history.shape[2:] != (MAX_TRACKB_MODES, TRACKB_HISTORY_TOKEN_DIMENSION)
                or noisy.shape[1:] != (MAX_TRACKB_MODES, TRACKB_MODE_TOKEN_DIMENSION)
                or target.shape != noisy.shape
                or history_mask.shape != history.shape[:2]
                or mode_mask.shape != noisy.shape[:2]):
            raise ValueError("shared RF padded input shapes are invalid")
        if (history.shape[0] != noisy.shape[0]
                or history_mask.shape[0] != noisy.shape[0]
                or mode_mask.shape[0] != noisy.shape[0]
                or history.device != noisy.device
                or target.device != noisy.device
                or history_mask.device != noisy.device
                or mode_mask.device != noisy.device
                or history.dtype != noisy.dtype
                or target.dtype != noisy.dtype):
            raise ValueError("shared RF inputs must have one consistent batch, device, and dtype")
        if history.shape[1] > 10:
            raise ValueError("shared RF history cannot exceed ten Track B rounds")
        if bool((history_mask[:, 1:] & ~history_mask[:, :-1]).any()):
            raise ValueError("shared RF history mask must describe a contiguous prefix")
        if not bool(mode_mask.any(-1).all()):
            raise ValueError("shared RF input requires at least one active mode per row")
        if self.system_type == "3qubit" and bool(mode_mask[:, 3].any()):
            raise ValueError("3qubit shared RF input must reserve mode four for padding")
        if self.schema != "TrackBSharedRFInputsV1":
            raise ValueError("unsupported shared RF input schema")


@dataclass(frozen=True)
class TrackBSharedRFEncoderConfigV1:
    """Shared encoder capacity; system heads remain outside this configuration."""

    width: int = 64
    heads: int = 4
    blocks: int = 2
    schema: str = "TrackBSharedRFEncoderConfigV1"

    def validate(self) -> None:
        if any(type(value) is not int or value <= 0 for value in (self.width, self.heads, self.blocks)):
            raise ValueError("shared RF encoder width/heads/blocks are invalid")
        if self.width % self.heads:
            raise ValueError("shared RF encoder width must divide evenly across heads")
        if self.schema != "TrackBSharedRFEncoderConfigV1":
            raise ValueError("unsupported shared RF encoder config schema")


class TrackBSharedRFEncoderV1(nn.Module):
    """The sole parameter-shared 3Q/4Q logical RF representation layer."""

    def __init__(self, config: TrackBSharedRFEncoderConfigV1) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.history_encoder = nn.Linear(TRACKB_HISTORY_TOKEN_DIMENSION, config.width)
        self.last_history_encoder = nn.Linear(TRACKB_HISTORY_TOKEN_DIMENSION, config.width)
        self.prefix_length_encoder = nn.Linear(1, config.width)
        self.mode_encoder = nn.Linear(2 * TRACKB_MODE_TOKEN_DIMENSION, config.width)
        self.system_embedding = nn.Embedding(2, config.width)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=4 * config.width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.blocks, enable_nested_tensor=False)

    def forward(self, inputs: TrackBSharedRFInputsV1) -> dict[str, torch.Tensor | str]:
        inputs.validate()
        history = inputs.history_tokens
        history_mask = inputs.history_mask
        mode_mask = inputs.mode_mask
        encoded_history = self.history_encoder(history)
        history_weight = history_mask.to(encoded_history.dtype)[:, :, None, None]
        history_context = encoded_history.mul(history_weight).sum(1) / history_weight.sum(1).clamp_min(1.0)
        history_length = history_mask.sum(1)
        if history.shape[1]:
            last_index = (history_length - 1).clamp_min(0)
            last_history = history[torch.arange(history.shape[0], device=history.device), last_index]
            last_context = self.last_history_encoder(last_history) * (history_length > 0)[:, None, None]
        else:
            last_context = torch.zeros_like(history_context)
        prefix_context = self.prefix_length_encoder(
            (history_length.to(history.dtype) / 10.0).unsqueeze(-1))[:, None, :]
        system_index = 0 if inputs.system_type == "3qubit" else 1
        system_context = self.system_embedding(
            torch.full((history.shape[0],), system_index, dtype=torch.long, device=history.device))[:, None, :]
        mode_context = self.mode_encoder(torch.cat((inputs.noisy_measurement_tokens, inputs.target_mode_tokens), dim=-1))
        mode_representation = self.encoder(
            mode_context + history_context + last_context + prefix_context + system_context,
            src_key_padding_mask=~mode_mask,
        )
        weight = mode_mask.to(mode_representation.dtype)
        pooled_representation = mode_representation.mul(weight[..., None]).sum(1) / weight.sum(1, keepdim=True)
        return {
            "mode_representation": mode_representation,
            "pooled_representation": pooled_representation,
            "mode_mask": mode_mask,
            "system_type": inputs.system_type,
            "source_identity_sha": inputs.source_identity_sha,
            "hardware_action_authorized": torch.zeros(
                history.shape[0], dtype=torch.bool, device=history.device),
        }


@dataclass(frozen=True)
class TrackBSystemTaskHeadConfigV1:
    """System-specific heads and bounds layered above the one shared encoder."""

    width: int
    recognition_classes_3q: int
    recognition_classes_4q: int
    amplitude_bound: float = 0.35
    phase_bound: float = 0.70
    schema: str = "TrackBSystemTaskHeadConfigV1"

    def validate(self) -> None:
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("system task head width is invalid")
        if any(type(value) is not int or value <= 1 for value in (
            self.recognition_classes_3q,
            self.recognition_classes_4q,
        )):
            raise ValueError("system task head class counts are invalid")
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in (
            self.amplitude_bound,
            self.phase_bound,
        )):
            raise ValueError("system task head action bounds are invalid")
        if self.schema != "TrackBSystemTaskHeadConfigV1":
            raise ValueError("unsupported system task head config schema")


class TrackBSystemTaskHeadsV1(nn.Module):
    """Independent recognition and actor heads; no cross-system decision sharing."""

    def __init__(self, config: TrackBSystemTaskHeadConfigV1) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.recognition_heads = nn.ModuleDict({
            "3qubit": nn.Linear(config.width, config.recognition_classes_3q),
            "4qubit": nn.Linear(config.width, config.recognition_classes_4q),
        })
        self.actor_heads = nn.ModuleDict({
            "3qubit": nn.Linear(config.width, 2),
            "4qubit": nn.Linear(config.width, 2),
        })

    def forward(
        self,
        shared_output: Mapping[str, torch.Tensor | str],
        *,
        system_type: str,
    ) -> dict[str, torch.Tensor | str]:
        if system_type not in ("3qubit", "4qubit") or shared_output.get("system_type") != system_type:
            raise ValueError("system task head received a mismatched shared RF representation")
        representation = shared_output.get("mode_representation")
        pooled = shared_output.get("pooled_representation")
        mode_mask = shared_output.get("mode_mask")
        if not isinstance(representation, torch.Tensor) or not isinstance(pooled, torch.Tensor) or not isinstance(mode_mask, torch.Tensor):
            raise ValueError("system task head shared RF output is incomplete")
        mode_count = int(system_type[0])
        raw_action = self.actor_heads[system_type](representation[:, :mode_count])
        active_mask = mode_mask[:, :mode_count]
        amplitude_delta = project_bounded_gauge_action(
            torch.tanh(raw_action[..., 0]) * self.config.amplitude_bound,
            active_mask,
            bound=self.config.amplitude_bound,
        )
        phase_delta = project_bounded_gauge_action(
            torch.tanh(raw_action[..., 1]) * self.config.phase_bound,
            active_mask,
            bound=self.config.phase_bound,
        )
        return {
            "recognition_logits": self.recognition_heads[system_type](pooled),
            "amplitude_delta": amplitude_delta,
            "phase_delta": phase_delta,
            "system_type": system_type,
            "hardware_action_authorized": torch.zeros(
                pooled.shape[0], dtype=torch.bool, device=pooled.device),
        }


__all__ = [
    "MAX_TRACKB_MODES",
    "TRACKB_SHARED_OBSERVATION_DOMAIN",
    "TrackBSharedRFInputsV1",
    "TrackBSharedRFEncoderConfigV1",
    "TrackBSharedRFEncoderV1",
    "TrackBSystemTaskHeadConfigV1",
    "TrackBSystemTaskHeadsV1",
]
