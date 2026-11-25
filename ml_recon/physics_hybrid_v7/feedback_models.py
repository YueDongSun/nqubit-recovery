"""Shared-backbone V7 feedback Model A and Model B.

Model A is an action-conditioned probabilistic world model.  Model B is a
history-conditioned policy/ranker that consumes at most nine completed
transitions.  The two systems share the operation encoder and physical expert
structure, while 3Q and 4Q keep independent density, nuisance, count-total and
policy heads.

This module is research-only and has no hardware-write authority.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    functional = None
    nn = None

from .feedback_data import (
    MAX_ACTION_DIM,
    OPERATION_FEATURE_NAMES,
    TARGET_FEATURE_NAMES,
)
from .feedback_synthetic import (
    ACTION_MODE_NAMES,
    HISTORY_FEATURE_NAMES,
    MECHANISM_NAMES,
    PHASE_ACTION_MIN_RAD,
)


FEEDBACK_MODEL_SCHEMA = "v7_feedback_model_bundle_v1"
ACTION_STEP_LIMITS = (0.005, 0.005, 0.005, 0.10, 0.10, 0.10)
HISTORY_IDENTIFICATION_RESPONSE_SCALES = (0.02, 0.05, 0.05)
HISTORY_IDENTIFICATION_DIM = 26
NUISANCE_TARGET_NAMES = (
    "retrieval_scale",
    "log_gsi_diagnostic",
    "multi_excitation_probability",
    "dephasing_fraction",
    "crosstalk_norm",
)


@dataclass(frozen=True)
class V7FeedbackModelConfig:
    width: int = 128
    operation_layers: int = 4
    history_layers: int = 3
    attention_heads: int = 8
    feedforward_multiplier: float = 3.0
    dropout: float = 0.05
    mechanism_top_k: int = 2
    classifier_latent_dim: int = 64
    history_identification_enabled: bool = False
    maximum_history: int = 9
    operation_feature_dim: int = len(OPERATION_FEATURE_NAMES)
    target_feature_dim: int = len(TARGET_FEATURE_NAMES)
    history_feature_dim: int = len(HISTORY_FEATURE_NAMES)
    action_dim: int = MAX_ACTION_DIM
    research_only: bool = True
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def validate(self) -> None:
        if self.width <= 0 or self.operation_layers <= 0 or self.history_layers <= 0:
            raise ValueError("model width and layer counts must be positive")
        if self.width % self.attention_heads != 0:
            raise ValueError("width must be divisible by attention_heads")
        if self.maximum_history != 9 or self.action_dim != MAX_ACTION_DIM:
            raise ValueError("history/action contract changed")
        if self.operation_feature_dim != len(OPERATION_FEATURE_NAMES):
            raise ValueError("operation feature contract changed")
        if self.target_feature_dim != len(TARGET_FEATURE_NAMES):
            raise ValueError("target feature contract changed")
        if self.history_feature_dim != len(HISTORY_FEATURE_NAMES):
            raise ValueError("history feature contract changed")
        if not 1 <= self.mechanism_top_k <= len(MECHANISM_NAMES):
            raise ValueError("mechanism_top_k is invalid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

    def to_json_dict(self) -> Dict[str, Any]:
        return {"schema": FEEDBACK_MODEL_SCHEMA, **asdict(self)}


if nn is not None:

    class _SwiGlu(nn.Module):
        def __init__(self, width: int, hidden: int, dropout: float) -> None:
            super().__init__()
            self.gate = nn.Linear(width, hidden)
            self.value = nn.Linear(width, hidden)
            self.output = nn.Linear(hidden, width)
            self.dropout = nn.Dropout(dropout)

        def forward(self, value: "torch.Tensor") -> "torch.Tensor":
            return self.output(
                self.dropout(functional.silu(self.gate(value)) * self.value(value))
            )


    class _TransformerBlock(nn.Module):
        def __init__(self, width: int, heads: int, hidden: int, dropout: float) -> None:
            super().__init__()
            self.attention_norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(
                width,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            self.feedforward_norm = nn.LayerNorm(width)
            self.feedforward = _SwiGlu(width, hidden, dropout)

        def forward(
            self,
            value: "torch.Tensor",
            *,
            attention_mask: Optional["torch.Tensor"] = None,
            padding_mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            normalized = self.attention_norm(value)
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=attention_mask,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            value = value + attended
            return value + self.feedforward(self.feedforward_norm(value))


    class _OperationMechanismExperts(nn.Module):
        """Operation-level physics experts with sparse top-k routing."""

        def __init__(self, width: int, hidden: int, dropout: float, top_k: int) -> None:
            super().__init__()
            self.top_k = int(top_k)
            self.router = nn.Linear(width, len(MECHANISM_NAMES))
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, width),
                )
                for _ in MECHANISM_NAMES
            ])

        def forward(self, value: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            logits = self.router(value)
            probability = torch.softmax(logits, dim=-1)
            if self.top_k < len(MECHANISM_NAMES):
                top_value, top_index = torch.topk(probability, self.top_k, dim=-1)
                sparse = torch.zeros_like(probability).scatter(-1, top_index, top_value)
                probability = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            expert_value = torch.stack(
                [expert(value) for expert in self.experts],
                dim=-2,
            )
            mixture = (probability.unsqueeze(-1) * expert_value).sum(dim=-2)
            mean_load = probability.mean(dim=(0, 1))
            uniform = torch.full_like(mean_load, 1.0 / float(len(MECHANISM_NAMES)))
            load_balance = torch.square(mean_load - uniform).mean()
            return value + mixture, probability, load_balance


    class SharedOperationEncoder(nn.Module):
        """Variable-operation graph Transformer shared by 3Q and 4Q."""

        def __init__(self, config: V7FeedbackModelConfig) -> None:
            super().__init__()
            config.validate()
            self.config = config
            width = int(config.width)
            hidden = int(round(width * config.feedforward_multiplier))
            self.operation_projection = nn.Sequential(
                nn.LayerNorm(config.operation_feature_dim),
                nn.Linear(config.operation_feature_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.target_projection = nn.Sequential(
                nn.LayerNorm(config.target_feature_dim),
                nn.Linear(config.target_feature_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            # The physical coordinate encoding uses actual x/y operation
            # coordinates.  Unlike the legacy RoPE path, token position is
            # explicitly multiplied into every harmonic.
            self.coordinate_projection = nn.Linear(18, width, bias=False)
            self.system_embedding = nn.Embedding(2, width)
            self.graph_projection = nn.Linear(width, width, bias=False)
            self.blocks = nn.ModuleList([
                _TransformerBlock(width, config.attention_heads, hidden, config.dropout)
                for _ in range(config.operation_layers)
            ])
            self.experts = _OperationMechanismExperts(
                width,
                hidden,
                config.dropout,
                config.mechanism_top_k,
            )
            self.token_norm = nn.LayerNorm(width)
            self.pool_norm = nn.LayerNorm(width)
            self.classifier_projection = nn.Sequential(
                nn.LayerNorm(config.classifier_latent_dim),
                nn.Linear(config.classifier_latent_dim, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            self.classifier_gate = nn.Linear(width + 1, width)
            nn.init.constant_(self.classifier_gate.bias, -2.0)

        def forward(
            self,
            operation_features: "torch.Tensor",
            target_features: "torch.Tensor",
            graph_kernel: "torch.Tensor",
            *,
            system_type: str,
            classifier_latent: Optional["torch.Tensor"] = None,
            classifier_uncertainty: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            if operation_features.ndim != 3:
                raise ValueError("operation_features must be [batch, operation, feature]")
            if operation_features.shape[-1] != self.config.operation_feature_dim:
                raise ValueError("operation feature dimension changed")
            batch, operation_count, _ = operation_features.shape
            if target_features.shape != (batch, self.config.target_feature_dim):
                raise ValueError("target_features shape changed")
            kernel = graph_kernel.to(dtype=operation_features.dtype, device=operation_features.device)
            if kernel.shape != (operation_count, operation_count):
                raise ValueError("graph kernel and operation count disagree")
            token = self.operation_projection(operation_features)
            x_coordinate = operation_features[:, :, OPERATION_FEATURE_NAMES.index("x_frequency")]
            y_coordinate = operation_features[:, :, OPERATION_FEATURE_NAMES.index("y_frequency")]
            coordinate = _coordinate_fourier_features(x_coordinate, y_coordinate)
            token = token + self.coordinate_projection(coordinate)
            system_index = 0 if system_type == "3qubit" else 1 if system_type == "4qubit" else -1
            if system_index < 0:
                raise ValueError("system_type must be 3qubit or 4qubit")
            token = token + self.system_embedding.weight[system_index][None, None, :]
            graph_message = torch.einsum("ij,bjd->bid", kernel, token)
            token = token + self.graph_projection(graph_message)
            target = self.target_projection(target_features)
            token = token + target[:, None, :]
            for block in self.blocks:
                token = block(token)
            token, mechanism_gates, load_balance = self.experts(token)
            token = self.token_norm(token)
            observed_mask = operation_features[:, :, OPERATION_FEATURE_NAMES.index("observed_coin_mask")]
            observed_pool = (token * observed_mask[:, :, None]).sum(dim=1) / observed_mask.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
            pooled = 0.5 * (token.mean(dim=1) + observed_pool) + target
            classifier_gate = torch.zeros_like(pooled)
            if classifier_latent is not None:
                if classifier_latent.shape != (batch, self.config.classifier_latent_dim):
                    raise ValueError("classifier_latent shape changed")
                uncertainty = torch.zeros(batch, 1, device=pooled.device, dtype=pooled.dtype)
                if classifier_uncertainty is not None:
                    uncertainty = classifier_uncertainty.to(dtype=pooled.dtype).reshape(batch, 1)
                classifier_value = self.classifier_projection(classifier_latent.to(dtype=pooled.dtype))
                classifier_gate = torch.sigmoid(
                    self.classifier_gate(torch.cat((classifier_value, uncertainty), dim=-1))
                ) * torch.exp(-uncertainty.clamp_min(0.0))
                pooled = pooled + classifier_gate * classifier_value
            return {
                "operation_tokens": token,
                "state_embedding": self.pool_norm(pooled),
                "mechanism_gates": mechanism_gates,
                "mechanism_load_balance": load_balance,
                "classifier_gate": classifier_gate,
            }


    class _WCoreDensityHead(nn.Module):
        def __init__(self, width: int, dimension: int) -> None:
            super().__init__()
            self.dimension = int(dimension)
            lower_count = self.dimension * (self.dimension + 1) // 2
            self.projection = nn.Linear(width, 2 * lower_count)
            indices = torch.tril_indices(self.dimension, self.dimension)
            self.register_buffer("lower_rows", indices[0], persistent=False)
            self.register_buffer("lower_columns", indices[1], persistent=False)

        def forward(self, latent: "torch.Tensor") -> "torch.Tensor":
            with torch.autocast(device_type=latent.device.type, enabled=False):
                real_value, imag_value = self.projection(latent.float()).chunk(2, dim=-1)
                batch = latent.shape[0]
                real = torch.zeros(
                    batch,
                    self.dimension,
                    self.dimension,
                    dtype=torch.float32,
                    device=latent.device,
                )
                imag = torch.zeros_like(real)
                real[:, self.lower_rows, self.lower_columns] = real_value
                imag[:, self.lower_rows, self.lower_columns] = imag_value
                diagonal = torch.arange(self.dimension, device=latent.device)
                real[:, diagonal, diagonal] = functional.softplus(real[:, diagonal, diagonal]) + 1e-5
                imag[:, diagonal, diagonal] = 0.0
                lower = torch.complex(real, imag)
                density = lower @ lower.mH
                trace = torch.diagonal(density, dim1=-2, dim2=-1).real.sum(dim=-1)
                return density / trace.clamp_min(1e-8)[:, None, None]


    class V7ActionConditionedWorldModel(nn.Module):
        """Model A: action-conditioned next-count/state/noise distribution."""

        independent_count_jacobian_head = False
        count_jacobian_source = "autograd_same_forward_map"

        def __init__(self, config: V7FeedbackModelConfig, encoder: SharedOperationEncoder) -> None:
            super().__init__()
            self.config = config
            self.encoder = encoder
            width = config.width
            hidden = int(round(width * config.feedforward_multiplier))
            self.action_projection = nn.Sequential(
                nn.LayerNorm(config.action_dim),
                nn.Linear(config.action_dim, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            self.action_film = nn.Linear(width, 2 * width)
            self.transition_block = _TransformerBlock(
                width,
                config.attention_heads,
                hidden,
                config.dropout,
            )
            self.count_logit_head = nn.Linear(width, 1)
            self.current_density_heads = nn.ModuleDict({
                "3qubit": _WCoreDensityHead(width, 3),
                "4qubit": _WCoreDensityHead(width, 4),
            })
            self.next_density_heads = nn.ModuleDict({
                "3qubit": _WCoreDensityHead(width, 3),
                "4qubit": _WCoreDensityHead(width, 4),
            })
            self.total_heads = nn.ModuleDict({
                system: nn.Linear(width, 1) for system in ("3qubit", "4qubit")
            })
            self.dispersion_heads = nn.ModuleDict({
                system: nn.Linear(width, 1) for system in ("3qubit", "4qubit")
            })
            self.nuisance_heads = nn.ModuleDict({
                system: nn.Linear(width, 2 * len(NUISANCE_TARGET_NAMES))
                for system in ("3qubit", "4qubit")
            })
            self.utility_heads = nn.ModuleDict({
                system: nn.Linear(width, 2) for system in ("3qubit", "4qubit")
            })
            self.harmful_heads = nn.ModuleDict({
                system: nn.Linear(width, 1) for system in ("3qubit", "4qubit")
            })
            self.multi_interval_heads = nn.ModuleDict({
                system: nn.Linear(width, 3) for system in ("3qubit", "4qubit")
            })
            self.ood_heads = nn.ModuleDict({
                system: nn.Linear(width, 1) for system in ("3qubit", "4qubit")
            })

        def forward(
            self,
            operation_features: "torch.Tensor",
            target_features: "torch.Tensor",
            graph_kernel: "torch.Tensor",
            action: "torch.Tensor",
            action_mask: "torch.Tensor",
            *,
            system_type: str,
            classifier_latent: Optional["torch.Tensor"] = None,
            classifier_uncertainty: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            encoded = self.encoder(
                operation_features,
                target_features,
                graph_kernel,
                system_type=system_type,
                classifier_latent=classifier_latent,
                classifier_uncertainty=classifier_uncertainty,
            )
            masked_action = action * action_mask
            action_embedding = self.action_projection(masked_action)
            scale, shift = self.action_film(action_embedding).chunk(2, dim=-1)
            token = encoded["operation_tokens"]
            conditioned = token * (1.0 + 0.10 * torch.tanh(scale[:, None, :])) + shift[:, None, :]
            conditioned = self.transition_block(conditioned)
            next_embedding = conditioned.mean(dim=1) + encoded["state_embedding"] + action_embedding
            # Count rates span several orders of magnitude.  Keeping this
            # branch in FP16 makes exp(log_total) finite in the forward pass
            # but can overflow the Addmm/NB2 backward pass.  The expensive
            # Transformer remains autocast; only statistical count parameters
            # and their exponentiation are fixed to FP32.
            with torch.autocast(device_type=conditioned.device.type, enabled=False):
                count_logits = self.count_logit_head(conditioned.float()).squeeze(-1)
                probability = torch.softmax(count_logits, dim=-1)
                log_total = self.total_heads[system_type](
                    next_embedding.float()
                ).squeeze(-1).clamp(0.0, 16.0)
                expected_total = torch.exp(log_total)
                expected_counts = expected_total[:, None] * probability
                dispersion = functional.softplus(
                    self.dispersion_heads[system_type](next_embedding.float()).squeeze(-1)
                ).clamp(1e-5, 0.75)
            nuisance_mean, nuisance_log_scale = self.nuisance_heads[system_type](
                next_embedding
            ).chunk(2, dim=-1)
            nuisance_log_scale = nuisance_log_scale.clamp(-5.0, 2.0)
            utility_mean, utility_log_scale = self.utility_heads[system_type](
                next_embedding
            ).chunk(2, dim=-1)
            return {
                **encoded,
                "next_state_embedding": next_embedding,
                "count_probability": probability,
                "expected_total": expected_total,
                "expected_counts": expected_counts,
                "nb2_dispersion": dispersion,
                "rho_current": self.current_density_heads[system_type](encoded["state_embedding"]),
                "rho_next": self.next_density_heads[system_type](next_embedding),
                "nuisance_mean": nuisance_mean,
                "nuisance_log_scale": nuisance_log_scale,
                "utility_mean": utility_mean.squeeze(-1),
                "utility_log_scale": utility_log_scale.squeeze(-1).clamp(-5.0, 2.0),
                "harmful_logit": self.harmful_heads[system_type](next_embedding).squeeze(-1),
                "multi_interval_prediction": functional.softplus(
                    self.multi_interval_heads[system_type](encoded["state_embedding"])
                ),
                "ood_energy": self.ood_heads[system_type](next_embedding).squeeze(-1),
                "actual_action": masked_action,
            }

        def count_jacobian(
            self,
            operation_features: "torch.Tensor",
            target_features: "torch.Tensor",
            graph_kernel: "torch.Tensor",
            action: "torch.Tensor",
            action_mask: "torch.Tensor",
            *,
            system_type: str,
        ) -> "torch.Tensor":
            """Autograd Jacobian from the same integrable count forward map."""

            if action.ndim != 2:
                raise ValueError("action must be [batch, action]")
            action_value = action.detach().clone().requires_grad_(True)
            output = self.forward(
                operation_features,
                target_features,
                graph_kernel,
                action_value,
                action_mask,
                system_type=system_type,
            )["expected_counts"]
            rows = []
            for operation_index in range(output.shape[1]):
                gradient = torch.autograd.grad(
                    output[:, operation_index].sum(),
                    action_value,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                rows.append(gradient)
            return torch.stack(rows, dim=1) * action_mask[:, None, :]


    class V7HistoryConditionedPolicy(nn.Module):
        """Model B: causal history policy and candidate-action ranker."""

        def __init__(self, config: V7FeedbackModelConfig) -> None:
            super().__init__()
            self.config = config
            width = config.width
            hidden = int(round(width * config.feedforward_multiplier))
            self.history_projection = nn.Sequential(
                nn.LayerNorm(config.history_feature_dim),
                nn.Linear(config.history_feature_dim, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            self.position_embedding = nn.Parameter(
                torch.randn(config.maximum_history + 1, width) / math.sqrt(width)
            )
            self.query_projection = nn.Linear(width, width)
            self.history_identification_projection = (
                nn.Sequential(
                    nn.LayerNorm(HISTORY_IDENTIFICATION_DIM),
                    nn.Linear(HISTORY_IDENTIFICATION_DIM, width),
                    nn.SiLU(),
                    nn.Linear(width, width),
                )
                if config.history_identification_enabled else None
            )
            self.blocks = nn.ModuleList([
                _TransformerBlock(width, config.attention_heads, hidden, config.dropout)
                for _ in range(config.history_layers)
            ])
            self.output_norm = nn.LayerNorm(width)
            self.action_heads = nn.ModuleDict({
                system: nn.Linear(width, config.action_dim) for system in ("3qubit", "4qubit")
            })
            self.mode_heads = nn.ModuleDict({
                system: nn.Linear(width, len(ACTION_MODE_NAMES)) for system in ("3qubit", "4qubit")
            })
            self.gain_heads = nn.ModuleDict({
                system: nn.Linear(width, 2) for system in ("3qubit", "4qubit")
            })
            self.harmful_heads = nn.ModuleDict({
                system: nn.Linear(width, 1) for system in ("3qubit", "4qubit")
            })
            self.stop_heads = nn.ModuleDict({
                system: nn.Linear(width, 1) for system in ("3qubit", "4qubit")
            })

        def forward(
            self,
            state_embedding: "torch.Tensor",
            history: "torch.Tensor",
            history_mask: "torch.Tensor",
            action_mask: "torch.Tensor",
            *,
            system_type: str,
        ) -> Dict[str, "torch.Tensor"]:
            batch = state_embedding.shape[0]
            if history.shape != (
                batch,
                self.config.maximum_history,
                self.config.history_feature_dim,
            ):
                raise ValueError("history shape changed")
            if history_mask.shape != history.shape[:2]:
                raise ValueError("history_mask shape changed")
            history_token = self.history_projection(history)
            identification_summary = torch.zeros(
                batch,
                HISTORY_IDENTIFICATION_DIM,
                dtype=history.dtype,
                device=history.device,
            )
            query_value = self.query_projection(state_embedding)
            if self.history_identification_projection is not None:
                action_limits = torch.as_tensor(
                    ACTION_STEP_LIMITS,
                    dtype=history.dtype,
                    device=history.device,
                )
                valid_history = history_mask.unsqueeze(-1)
                normalized_action = (
                    history[:, :, :self.config.action_dim]
                    / action_limits[None, None, :]
                ) * valid_history
                information = torch.sum(
                    torch.square(normalized_action),
                    dim=1,
                )
                response_gradients = []
                for response_name, response_scale in zip(
                    ("utility_delta", "retrieval_delta", "multi_risk_delta"),
                    HISTORY_IDENTIFICATION_RESPONSE_SCALES,
                ):
                    response = torch.tanh(
                        history[:, :, HISTORY_FEATURE_NAMES.index(response_name)]
                        / float(response_scale)
                    ) * history_mask
                    numerator = torch.sum(
                        normalized_action * response.unsqueeze(-1),
                        dim=1,
                    )
                    response_gradients.append(
                        numerator / (information + 0.25)
                    )
                valid_count = history_mask.sum(dim=1).clamp_min(1.0)
                accepted_rate = torch.sum(
                    history[:, :, HISTORY_FEATURE_NAMES.index("accepted")]
                    * history_mask,
                    dim=1,
                ) / valid_count
                rollback_rate = torch.sum(
                    history[:, :, HISTORY_FEATURE_NAMES.index("rollback")]
                    * history_mask,
                    dim=1,
                ) / valid_count
                identification_summary = torch.cat((
                    *response_gradients,
                    torch.log1p(information),
                    accepted_rate[:, None],
                    rollback_rate[:, None],
                ), dim=1)
                query_value = query_value + self.history_identification_projection(
                    identification_summary
                )
            query = query_value.unsqueeze(1)
            sequence = torch.cat((history_token, query), dim=1)
            sequence = sequence + self.position_embedding[None, :, :]
            query_valid = torch.ones(batch, 1, dtype=history_mask.dtype, device=history_mask.device)
            valid = torch.cat((history_mask, query_valid), dim=1) > 0.5
            padding_mask = ~valid
            length = sequence.shape[1]
            causal_mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=sequence.device),
                diagonal=1,
            )
            for block in self.blocks:
                sequence = block(
                    sequence,
                    attention_mask=causal_mask,
                    padding_mask=padding_mask,
                )
                # For a zero-history sample, causal attention leaves early
                # padded query rows with no valid keys.  Some optimized SDPA
                # kernels emit NaNs for those unused rows; a later layer can
                # then leak them into the valid policy query despite the key
                # padding mask.  Padded tokens carry no information, so reset
                # them after every block while preserving the final query.
                sequence = sequence.masked_fill(padding_mask[:, :, None], 0.0)
            policy = self.output_norm(sequence[:, -1])
            raw_action = self.action_heads[system_type](policy)
            limits = torch.as_tensor(
                ACTION_STEP_LIMITS,
                dtype=raw_action.dtype,
                device=raw_action.device,
            )
            action = limits[None, :] * torch.tanh(raw_action) * action_mask
            gain_mean, gain_log_scale = self.gain_heads[system_type](policy).chunk(2, dim=-1)
            candidates = _candidate_action_set(action, action_mask)
            return {
                "policy_embedding": policy,
                "action": action,
                "candidate_actions": candidates,
                "mode_logits": self.mode_heads[system_type](policy),
                "gain_mean": gain_mean.squeeze(-1),
                "gain_log_scale": gain_log_scale.squeeze(-1).clamp(-5.0, 2.0),
                "harmful_logit": self.harmful_heads[system_type](policy).squeeze(-1),
                "stop_logit": self.stop_heads[system_type](policy).squeeze(-1),
                "history_length": history_mask.sum(dim=1),
                "history_identification": identification_summary,
            }


    class V7FeedbackModelBundle(nn.Module):
        def __init__(self, config: V7FeedbackModelConfig) -> None:
            super().__init__()
            config.validate()
            self.config = config
            self.encoder = SharedOperationEncoder(config)
            self.model_a = V7ActionConditionedWorldModel(config, self.encoder)
            self.model_b = V7HistoryConditionedPolicy(config)

        def audit_dict(self) -> Dict[str, Any]:
            return {
                "schema": FEEDBACK_MODEL_SCHEMA,
                "config": self.config.to_json_dict(),
                "shared_backbone": True,
                "independent_system_heads": True,
                "model_a": "action_conditioned_probabilistic_world_model",
                "model_b": "history_conditioned_causal_policy_ranker",
                "maximum_history": 9,
                "production_training_allowed": False,
                "online_parameter_updates_allowed": False,
                "hardware_action_authorized": False,
                "legacy_global_multi_scope": "interval_censored_auxiliary_only",
                "classifier_contract": "soft_latent_plus_uncertainty_gate_no_hard_argmax",
                "count_jacobian_source": "autograd_same_forward_map",
                "history_system_identification": (
                    "frozen_secant_summary"
                    if self.config.history_identification_enabled else "disabled"
                ),
            }


    def _coordinate_fourier_features(
        x_coordinate: "torch.Tensor",
        y_coordinate: "torch.Tensor",
    ) -> "torch.Tensor":
        values = [x_coordinate, y_coordinate]
        for harmonic in (1.0, 2.0, 4.0, 8.0):
            values.extend((
                torch.sin(math.pi * harmonic * x_coordinate),
                torch.cos(math.pi * harmonic * x_coordinate),
                torch.sin(math.pi * harmonic * y_coordinate),
                torch.cos(math.pi * harmonic * y_coordinate),
            ))
        return torch.stack(values, dim=-1)


    def _candidate_action_set(
        action: "torch.Tensor",
        action_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        no_op = torch.zeros_like(action)
        phase_only = action.clone()
        phase_only[:, :3] = 0.0
        amplitude_only = action.clone()
        amplitude_only[:, 3:] = 0.0
        candidates = torch.stack((
            no_op,
            0.25 * phase_only,
            0.50 * phase_only,
            phase_only,
            amplitude_only,
            action,
        ), dim=1)
        phase_size = candidates[:, :, 3:].abs().amax(dim=-1)
        phase_keep = (
            (phase_size == 0.0) | (phase_size >= PHASE_ACTION_MIN_RAD)
        ).to(dtype=candidates.dtype)
        candidates = torch.cat((
            candidates[:, :, :3],
            candidates[:, :, 3:] * phase_keep[:, :, None],
        ), dim=-1)
        return candidates * action_mask[:, None, :]


else:  # pragma: no cover

    class SharedOperationEncoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("V7 feedback models require torch")

    class V7ActionConditionedWorldModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("V7 feedback models require torch")

    class V7HistoryConditionedPolicy:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("V7 feedback models require torch")

    class V7FeedbackModelBundle:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("V7 feedback models require torch")


def parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


__all__ = [
    "FEEDBACK_MODEL_SCHEMA",
    "NUISANCE_TARGET_NAMES",
    "SharedOperationEncoder",
    "V7ActionConditionedWorldModel",
    "V7FeedbackModelBundle",
    "V7FeedbackModelConfig",
    "V7HistoryConditionedPolicy",
    "parameter_count",
]
