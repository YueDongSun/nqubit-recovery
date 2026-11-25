"""Research-only candidate-action critic with frozen actor/encoder inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ml_recon.data_v2_4.hashing import sha256_array

from .candidate_transition_contract import TrackBCandidateTransitionLabelV1
from .contracts import canonical_sha256


def _tensor_sha256(tensor: Any) -> str:
    """Hash an immutable tensor snapshot without retaining its graph."""
    if not isinstance(tensor, torch.Tensor):
        return ""
    return sha256_array(tensor.detach().cpu().numpy())


@dataclass(frozen=True)
class TrackBCandidateCriticConfigV1:
    """One system-specific critic; representation sharing is supplied upstream."""

    system_type: str
    mode_count: int
    width: int
    schema: str = "TrackBCandidateCriticConfigV1"

    def validate(self) -> None:
        if self.system_type not in ("3qubit", "4qubit") or self.mode_count != int(self.system_type[0]):
            raise ValueError("candidate critic system identity is invalid")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("candidate critic width is invalid")
        if self.schema != "TrackBCandidateCriticConfigV1":
            raise ValueError("unsupported candidate critic config schema")

    def sha256(self) -> str:
        self.validate()
        return canonical_sha256({
            "schema": self.schema,
            "system_type": self.system_type,
            "mode_count": self.mode_count,
            "width": self.width,
        })


@dataclass(frozen=True)
class TrackBCandidateCriticBatchV1:
    """One source-bound critic batch; labels are paired to exact candidate actions."""

    system_type: str
    v24_contract_sha: str
    v24_schema_sha: str
    v24_generator_sha: str
    actor_checkpoint_sha: str
    shared_encoder_sha: str
    critic_config_sha: str
    mode_representation: torch.Tensor
    candidate_action: torch.Tensor
    mode_mask: torch.Tensor
    observation_shas: tuple[str, ...]
    labels: tuple[TrackBCandidateTransitionLabelV1, ...]
    schema: str = "TrackBCandidateCriticBatchV1"
    mode_representation_sha: str = field(init=False)
    representation_binding_sha: str = field(init=False)

    def __post_init__(self) -> None:
        representation_sha = _tensor_sha256(self.mode_representation)
        object.__setattr__(self, "mode_representation_sha", representation_sha)
        object.__setattr__(self, "representation_binding_sha", self._expected_representation_binding_sha())

    def _expected_representation_binding_sha(self) -> str:
        """Bind encoder identity, observed rows, mask, and representation bytes."""
        return canonical_sha256({
            "schema": "TrackBCandidateRepresentationBindingV1",
            "system_type": self.system_type,
            "v24_contract_sha": self.v24_contract_sha,
            "v24_schema_sha": self.v24_schema_sha,
            "v24_generator_sha": self.v24_generator_sha,
            "shared_encoder_sha": self.shared_encoder_sha,
            "observation_shas": self.observation_shas,
            "mode_representation_sha": _tensor_sha256(self.mode_representation),
            "mode_mask_sha": _tensor_sha256(self.mode_mask),
        })

    def validate(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("candidate critic batch system identity is invalid")
        for value in (
            self.v24_contract_sha,
            self.v24_schema_sha,
            self.v24_generator_sha,
            self.actor_checkpoint_sha,
            self.shared_encoder_sha,
            self.critic_config_sha,
        ):
            if type(value) is not str or len(value) != 64:
                raise ValueError("candidate critic batch source identity is invalid")
        if (not isinstance(self.mode_representation, torch.Tensor)
                or not isinstance(self.candidate_action, torch.Tensor)
                or not isinstance(self.mode_mask, torch.Tensor)
                or self.mode_mask.dtype != torch.bool):
            raise ValueError("candidate critic batch requires tensor inputs and a boolean mask")
        mode_count = int(self.system_type[0])
        if (self.mode_representation.ndim != 3
                or self.candidate_action.ndim != 3
                or self.candidate_action.shape[-1] != 2
                or self.mode_representation.shape[:2] != self.candidate_action.shape[:2]
                or self.mode_representation.shape[1] != mode_count
                or self.mode_mask.shape != self.mode_representation.shape[:2]
                or self.mode_representation.shape[0] != len(self.labels)
                or self.mode_representation.shape[0] != len(self.observation_shas)):
            raise ValueError("candidate critic batch shapes and labels are inconsistent")
        if (not bool(torch.isfinite(self.mode_representation).all())
                or not bool(torch.isfinite(self.candidate_action).all())
                or not bool(self.mode_mask.any(-1).all())):
            raise ValueError("candidate critic batch inputs are invalid")
        if self.mode_representation_sha != _tensor_sha256(self.mode_representation):
            raise ValueError("candidate critic representation changed after source binding")
        if self.representation_binding_sha != self._expected_representation_binding_sha():
            raise ValueError("candidate critic representation source binding changed")
        for index, label in enumerate(self.labels):
            label.validate()
            action_sha = sha256_array(
                self.candidate_action[index].detach().cpu().numpy().astype("float64"))
            if (label.system_type != self.system_type
                    or label.split != "train"
                    or label.v24_contract_sha != self.v24_contract_sha
                    or label.v24_schema_sha != self.v24_schema_sha
                    or label.v24_generator_sha != self.v24_generator_sha
                    or label.actor_checkpoint_sha != self.actor_checkpoint_sha
                    or label.observation_sha != self.observation_shas[index]
                    or label.candidate_action_sha != action_sha):
                raise ValueError("candidate critic label is not paired to its source/action input")
        if self.schema != "TrackBCandidateCriticBatchV1":
            raise ValueError("unsupported candidate critic batch schema")

    def sha256(self) -> str:
        """Identify the ordered inputs and labels consumed by one critic forward."""
        self.validate()
        return canonical_sha256({
            "schema": "TrackBCandidateCriticBatchBindingV1",
            "system_type": self.system_type,
            "v24_contract_sha": self.v24_contract_sha,
            "v24_schema_sha": self.v24_schema_sha,
            "v24_generator_sha": self.v24_generator_sha,
            "actor_checkpoint_sha": self.actor_checkpoint_sha,
            "shared_encoder_sha": self.shared_encoder_sha,
            "critic_config_sha": self.critic_config_sha,
            "representation_binding_sha": self.representation_binding_sha,
            "candidate_action_sha": _tensor_sha256(self.candidate_action),
            "mode_mask_sha": _tensor_sha256(self.mode_mask),
            "observation_shas": self.observation_shas,
            "label_shas": tuple(label.sha256() for label in self.labels),
        })

    def gain_target(self) -> torch.Tensor:
        self.validate()
        return torch.tensor(
            [label.gain for label in self.labels],
            dtype=self.mode_representation.dtype,
            device=self.mode_representation.device,
        )

    def harm_target(self) -> torch.Tensor:
        self.validate()
        return torch.tensor(
            [float(label.harm) for label in self.labels],
            dtype=self.mode_representation.dtype,
            device=self.mode_representation.device,
        )

    def valid_mask(self) -> torch.Tensor:
        self.validate()
        return torch.tensor(
            [label.valid_mask for label in self.labels],
            dtype=torch.bool,
            device=self.mode_representation.device,
        )


@dataclass(frozen=True)
class TrackBCandidateCriticOutputV1:
    """Predictions cryptographically bound to their ordered critic batch."""

    critic_config_sha: str
    critic_batch_sha: str
    representation_binding_sha: str
    predicted_candidate_gain: torch.Tensor
    predicted_harm_probability: torch.Tensor
    critic_inputs_detached: torch.Tensor
    hardware_action_authorized: torch.Tensor
    schema: str = "TrackBCandidateCriticOutputV1"

    def validate_for(self, batch: TrackBCandidateCriticBatchV1) -> None:
        batch.validate()
        for value in (
            self.critic_config_sha,
            self.critic_batch_sha,
            self.representation_binding_sha,
        ):
            if type(value) is not str or len(value) != 64:
                raise ValueError("candidate critic output identity is invalid")
        if self.schema != "TrackBCandidateCriticOutputV1":
            raise ValueError("unsupported candidate critic output schema")
        expected_shape = (batch.mode_representation.shape[0],)
        for tensor in (
            self.predicted_candidate_gain,
            self.predicted_harm_probability,
            self.critic_inputs_detached,
            self.hardware_action_authorized,
        ):
            if not isinstance(tensor, torch.Tensor) or tensor.shape != expected_shape:
                raise ValueError("candidate critic output shape is inconsistent with its batch")
        if (not bool(torch.isfinite(self.predicted_candidate_gain).all())
                or not bool(torch.isfinite(self.predicted_harm_probability).all())
                or bool(((self.predicted_harm_probability < 0.0)
                         | (self.predicted_harm_probability > 1.0)).any())
                or self.critic_inputs_detached.dtype != torch.bool
                or self.hardware_action_authorized.dtype != torch.bool):
            raise ValueError("candidate critic output is invalid")
        if self.critic_batch_sha != batch.sha256():
            raise ValueError("candidate critic output belongs to a different source-bound batch")
        if self.representation_binding_sha != batch.representation_binding_sha:
            raise ValueError("candidate critic output representation binding differs from its batch")
        if self.critic_config_sha != batch.critic_config_sha:
            raise ValueError("candidate critic output configuration differs from its batch")


class TrackBCandidateCriticV1(nn.Module):
    """Predict action-specific synthetic gain/harm without updating actor inputs."""

    def __init__(self, config: TrackBCandidateCriticConfigV1) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.action_encoder = nn.Sequential(
            nn.Linear(4, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
            nn.GELU(),
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(2 * config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
            nn.GELU(),
        )
        self.gain_head = nn.Linear(config.width, 1)
        self.harm_head = nn.Linear(config.width, 1)

    def forward(
        self,
        mode_representation: Any,
        candidate_action: Any,
        mode_mask: Any,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(mode_representation, torch.Tensor) or not isinstance(candidate_action, torch.Tensor):
            raise TypeError("candidate critic requires tensor representation and action")
        if not isinstance(mode_mask, torch.Tensor) or mode_mask.dtype != torch.bool:
            raise ValueError("candidate critic requires a boolean mode mask")
        if (mode_representation.ndim != 3
                or candidate_action.ndim != 3
                or candidate_action.shape[-1] != 2
                or mode_representation.shape[:2] != candidate_action.shape[:2]
                or mode_representation.shape[-1] != self.config.width
                or mode_representation.shape[1] != self.config.mode_count
                or mode_mask.shape != mode_representation.shape[:2]):
            raise ValueError("candidate critic inputs do not meet the system-specific shape contract")
        if not bool(torch.isfinite(mode_representation).all()) or not bool(torch.isfinite(candidate_action).all()):
            raise ValueError("candidate critic inputs must be finite")
        if not bool(mode_mask.any(-1).all()):
            raise ValueError("candidate critic requires one active mode per row")
        frozen_representation = mode_representation.detach()
        frozen_action = candidate_action.detach()
        action_features = torch.cat((frozen_action, frozen_action.square()), dim=-1)
        action_context = self.action_encoder(action_features)
        value_tokens = self.value_encoder(torch.cat((frozen_representation, action_context), dim=-1))
        weight = mode_mask.to(value_tokens.dtype)
        pooled = value_tokens.mul(weight[..., None]).sum(1) / weight.sum(1, keepdim=True)
        return {
            "predicted_candidate_gain": self.gain_head(pooled).squeeze(-1),
            "predicted_harm_probability": torch.sigmoid(self.harm_head(pooled).squeeze(-1)),
            "critic_inputs_detached": torch.ones(
                pooled.shape[0], dtype=torch.bool, device=pooled.device),
            "hardware_action_authorized": torch.zeros(
                pooled.shape[0], dtype=torch.bool, device=pooled.device),
        }

    def forward_batch(self, batch: TrackBCandidateCriticBatchV1) -> TrackBCandidateCriticOutputV1:
        """Consume a validated source/action-paired batch rather than bare tensors."""
        batch.validate()
        if batch.system_type != self.config.system_type:
            raise ValueError("candidate critic batch system differs from its configured critic")
        if batch.critic_config_sha != self.config.sha256():
            raise ValueError("candidate critic batch configuration differs from its configured critic")
        output = self(batch.mode_representation, batch.candidate_action, batch.mode_mask)
        return TrackBCandidateCriticOutputV1(
            critic_config_sha=self.config.sha256(),
            critic_batch_sha=batch.sha256(),
            representation_binding_sha=batch.representation_binding_sha,
            predicted_candidate_gain=output["predicted_candidate_gain"],
            predicted_harm_probability=output["predicted_harm_probability"],
            critic_inputs_detached=output["critic_inputs_detached"],
            hardware_action_authorized=output["hardware_action_authorized"],
        )


def candidate_critic_loss(
    output: TrackBCandidateCriticOutputV1,
    *,
    batch: TrackBCandidateCriticBatchV1,
    harm_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Fit only action-paired labels; actor and shared encoder remain detached."""
    if not math.isfinite(float(harm_loss_weight)) or float(harm_loss_weight) < 0.0:
        raise ValueError("candidate critic harm loss weight is invalid")
    if not isinstance(output, TrackBCandidateCriticOutputV1):
        raise TypeError("candidate critic loss requires a source-bound critic output")
    output.validate_for(batch)
    gain = output.predicted_candidate_gain
    harm = output.predicted_harm_probability
    gain_target = batch.gain_target()
    harm_target = batch.harm_target()
    valid_mask = batch.valid_mask()
    if (gain_target.shape != gain.shape or harm_target.shape != harm.shape
            or not bool(torch.isfinite(gain_target).all()) or not bool(torch.isfinite(harm_target).all())):
        raise ValueError("candidate critic targets are invalid")
    if bool(((harm_target < 0.0) | (harm_target > 1.0)).any()):
        raise ValueError("candidate critic harm target must be a probability label")
    if not bool(valid_mask.any()):
        raise ValueError("candidate critic batch has no valid action-paired labels")
    return functional.smooth_l1_loss(gain[valid_mask], gain_target[valid_mask]) + float(
        harm_loss_weight) * functional.binary_cross_entropy(harm[valid_mask], harm_target[valid_mask])


__all__ = [
    "TrackBCandidateCriticConfigV1",
    "TrackBCandidateCriticBatchV1",
    "TrackBCandidateCriticOutputV1",
    "TrackBCandidateCriticV1",
    "candidate_critic_loss",
]
