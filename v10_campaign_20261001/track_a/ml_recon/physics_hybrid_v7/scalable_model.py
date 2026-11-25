"""Scalable physics-constrained V7 count model with noise attribution.

The network is a Perceiver-style graph operator: operation-point tokens are
compressed into latent tokens, processed globally, and decoded back to all
operations.  Count means, coherent/accidental/uniform mixture weights, NB2
dispersion, noise taxonomy, and a physics-capped bestF estimate come from one
integrable forward pass.  The module has no hardware-write authority.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch.utils.checkpoint import checkpoint
except ImportError:  # pragma: no cover
    torch = None
    functional = None
    nn = None
    checkpoint = None

from .noise_taxonomy import (
    NOISE_COARSE_TYPE_NAMES,
    NOISE_TYPE_NAMES,
    OPERATION_INPUT_FEATURE_NAMES,
    TARGET_CONTEXT_FEATURE_NAMES,
)


SCALABLE_MODEL_SCHEMA = "v7_scalable_physics_count_model_v1"
SCALABLE_MODEL_INPUT_AUDIT = {
    "hidden_coin_counts_used_for_inference": False,
    "masked_self_supervision_supported": True,
    "signal_idler_full_covariates_used": True,
    "global_multi_click_expanded_to_local_pseudocounts": False,
    "current_matrix_directory_read": False,
    "historical_command_invented": False,
    "noise_taxonomy_is_synthetic_mechanism_label": True,
    "noise_taxonomy_is_detector_calibration": False,
}
NOISE_SUMMARY_NAMES = (
    "log_observed_coin_total",
    "log_signal_total",
    "log_idler_total",
    "absolute_log_signal_idler_ratio",
    "log_observed_coin_mean",
    "log_observed_coin_fano",
    "observed_coin_zero_fraction",
    "observed_coin_entropy",
    "observed_coin_max_share",
    "accidental_total_variation",
    "accidental_log_residual_rms",
    "accidental_log_residual_bias",
    "accidental_graph_roughness",
    "accidental_lag_correlation",
    "signal_idler_total_variation",
    "observed_operation_fraction",
)


@dataclass(frozen=True)
class V7ScalableModelConfig:
    system_type: str
    operation_count: int
    qubit_count: int
    input_feature_dim: int
    target_feature_dim: int
    width: int
    latent_tokens: int
    layers: int
    attention_heads: int
    feedforward_multiplier: float
    dropout: float
    count_smoothing: float
    use_activation_checkpointing: bool = False
    research_only: bool = True
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        system_type: str,
        operation_count: int,
        qubit_count: int,
    ) -> "V7ScalableModelConfig":
        values = dict(payload)
        values.update({
            "system_type": str(system_type),
            "operation_count": int(operation_count),
            "qubit_count": int(qubit_count),
            "input_feature_dim": len(OPERATION_INPUT_FEATURE_NAMES),
            "target_feature_dim": len(TARGET_CONTEXT_FEATURE_NAMES),
        })
        return cls(**values)

    def validate(self) -> None:
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("system_type must be 3qubit or 4qubit")
        if self.operation_count <= 1 or self.qubit_count not in {3, 4}:
            raise ValueError("invalid operation/qubit dimensions")
        if self.width <= 0 or self.layers <= 0 or self.latent_tokens <= 0:
            raise ValueError("model width/layers/latent_tokens must be positive")
        if self.width % self.attention_heads != 0:
            raise ValueError("width must be divisible by attention_heads")
        if self.input_feature_dim != len(OPERATION_INPUT_FEATURE_NAMES):
            raise ValueError("input feature schema changed")
        if self.target_feature_dim != len(TARGET_CONTEXT_FEATURE_NAMES):
            raise ValueError("target feature schema changed")
        if not 0.0 <= self.dropout < 1.0 or self.feedforward_multiplier < 1.0:
            raise ValueError("dropout/feedforward_multiplier are invalid")

    def to_json_dict(self) -> Dict[str, Any]:
        return {"schema": SCALABLE_MODEL_SCHEMA, **asdict(self)}


if nn is not None:

    class _SwiGluFeedForward(nn.Module):
        def __init__(self, width: int, hidden_width: int, dropout: float) -> None:
            super().__init__()
            self.gate = nn.Linear(width, hidden_width)
            self.value = nn.Linear(width, hidden_width)
            self.output = nn.Linear(hidden_width, width)
            self.dropout = nn.Dropout(float(dropout))

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            hidden = functional.silu(self.gate(values)) * self.value(values)
            return self.output(self.dropout(hidden))


    class _LatentBlock(nn.Module):
        def __init__(self, width: int, heads: int, hidden_width: int, dropout: float) -> None:
            super().__init__()
            self.attention_norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(
                width, heads, dropout=float(dropout), batch_first=True,
            )
            self.feedforward_norm = nn.LayerNorm(width)
            self.feedforward = _SwiGluFeedForward(width, hidden_width, dropout)

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            normalized = self.attention_norm(values)
            attended, _ = self.attention(
                normalized, normalized, normalized, need_weights=False,
            )
            values = values + attended
            return values + self.feedforward(self.feedforward_norm(values))


    class _CrossAttentionBlock(nn.Module):
        def __init__(self, width: int, heads: int, hidden_width: int, dropout: float) -> None:
            super().__init__()
            self.query_norm = nn.LayerNorm(width)
            self.context_norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(
                width, heads, dropout=float(dropout), batch_first=True,
            )
            self.feedforward_norm = nn.LayerNorm(width)
            self.feedforward = _SwiGluFeedForward(width, hidden_width, dropout)

        def forward(
            self,
            query: "torch.Tensor",
            context: "torch.Tensor",
        ) -> "torch.Tensor":
            normalized_query = self.query_norm(query)
            normalized_context = self.context_norm(context)
            attended, _ = self.attention(
                normalized_query,
                normalized_context,
                normalized_context,
                need_weights=False,
            )
            query = query + attended
            return query + self.feedforward(self.feedforward_norm(query))


    class V7ScalablePhysicsModel(nn.Module):
        """Large synthetic-pretrainable joint count/noise/bestF model."""

        def __init__(
            self,
            config: V7ScalableModelConfig,
            graph_kernel: Any,
        ) -> None:
            super().__init__()
            config.validate()
            kernel = torch.as_tensor(graph_kernel, dtype=torch.float32)
            if kernel.shape != (config.operation_count, config.operation_count):
                raise ValueError("graph_kernel disagrees with operation_count")
            kernel = torch.clamp(kernel, min=0.0)
            kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
            self.config = config
            self.register_buffer("graph_kernel", kernel)
            width = int(config.width)
            hidden_width = int(round(width * float(config.feedforward_multiplier)))
            self.operation_projection = nn.Sequential(
                nn.LayerNorm(config.input_feature_dim),
                nn.Linear(config.input_feature_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.target_projection = nn.Sequential(
                nn.LayerNorm(config.target_feature_dim),
                nn.Linear(config.target_feature_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.noise_summary_projection = nn.Sequential(
                nn.LayerNorm(len(NOISE_SUMMARY_NAMES)),
                nn.Linear(len(NOISE_SUMMARY_NAMES), width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.noise_only_projection = nn.Sequential(
                nn.LayerNorm(len(NOISE_SUMMARY_NAMES)),
                nn.Linear(len(NOISE_SUMMARY_NAMES), width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.LayerNorm(width),
            )
            self.graph_projection = nn.Linear(width, width, bias=False)
            self.latent_tokens = nn.Parameter(
                torch.randn(config.latent_tokens, width) / max(width, 1) ** 0.5,
            )
            self.encoder = _CrossAttentionBlock(
                width, config.attention_heads, hidden_width, config.dropout,
            )
            self.latent_blocks = nn.ModuleList([
                _LatentBlock(
                    width, config.attention_heads, hidden_width, config.dropout,
                )
                for _ in range(config.layers)
            ])
            self.decoder = _CrossAttentionBlock(
                width, config.attention_heads, hidden_width, config.dropout,
            )
            self.operation_norm = nn.LayerNorm(width)
            self.global_norm = nn.LayerNorm(width)
            self.coherent_residual_head = nn.Linear(width, 1)
            self.log_total_head = nn.Linear(width, 1)
            self.mixture_head = nn.Linear(width, 3)
            self.dispersion_head = nn.Linear(width, 1)
            self.noise_type_head = nn.Linear(width, len(NOISE_TYPE_NAMES))
            self.noise_coarse_type_head = nn.Linear(width, len(NOISE_COARSE_TYPE_NAMES))
            self.noise_strength_head = nn.Linear(width, len(NOISE_TYPE_NAMES))
            self.retrieval_ceiling_head = nn.Linear(width, 1)
            self.retention_head = nn.Linear(width, 1)
            self.ood_energy_head = nn.Linear(width, 1)
            nn.init.constant_(self.log_total_head.bias, 8.0)
            nn.init.constant_(self.dispersion_head.bias, -4.0)
            nn.init.constant_(self.retrieval_ceiling_head.bias, 3.0)
            nn.init.constant_(self.retention_head.bias, 0.5)

        def _noise_summary(
            self,
            operation_inputs: "torch.Tensor",
            signal_counts: "torch.Tensor",
            idler_counts: "torch.Tensor",
            accidental_probability: "torch.Tensor",
        ) -> "torch.Tensor":
            observed_coin = torch.expm1(operation_inputs[:, :, 0]).clamp_min(0.0)
            observed_mask = operation_inputs[:, :, 6].clamp(0.0, 1.0)
            observed_count = observed_mask.sum(dim=1).clamp_min(1.0)
            observed_total = (observed_coin * observed_mask).sum(dim=1)
            observed_mean = observed_total / observed_count
            observed_variance = (
                observed_mask * (observed_coin - observed_mean[:, None]).square()
            ).sum(dim=1) / observed_count
            observed_fano = observed_variance / observed_mean.clamp_min(1e-6)
            observed_probability = (observed_coin + 0.5) * observed_mask
            observed_probability = observed_probability / observed_probability.sum(
                dim=1, keepdim=True,
            ).clamp_min(1e-8)
            accidental_observed = accidental_probability * observed_mask
            accidental_observed = accidental_observed / accidental_observed.sum(
                dim=1, keepdim=True,
            ).clamp_min(1e-8)
            log_residual = observed_mask * (
                torch.log(observed_probability.clamp_min(1e-12))
                - torch.log(accidental_observed.clamp_min(1e-12))
            )
            graph_residual = torch.einsum(
                "ij,bj->bi", self.graph_kernel, log_residual,
            )
            residual_centered = log_residual - (
                log_residual.sum(dim=1, keepdim=True) / observed_count[:, None]
            ) * observed_mask
            graph_roughness = torch.sqrt(
                (observed_mask * (residual_centered - graph_residual).square()).sum(dim=1)
                / observed_count
                + 1e-8
            )
            residual_rms = torch.sqrt(
                (observed_mask * log_residual.square()).sum(dim=1) / observed_count + 1e-8
            )
            residual_bias = log_residual.sum(dim=1) / observed_count
            lag_correlation = (
                observed_mask
                * torch.roll(observed_mask, 1, dims=1)
                * residual_centered
                * torch.roll(residual_centered, 1, dims=1)
            ).sum(dim=1) / observed_count
            entropy = -(
                observed_probability * torch.log(observed_probability.clamp_min(1e-12))
            ).sum(dim=1) / torch.log(observed_count.clamp_min(2.0))
            signal_probability = signal_counts / signal_counts.sum(
                dim=1, keepdim=True,
            ).clamp_min(1e-8)
            idler_probability = idler_counts / idler_counts.sum(
                dim=1, keepdim=True,
            ).clamp_min(1e-8)
            signal_total = signal_counts.sum(dim=1)
            idler_total = idler_counts.sum(dim=1)
            return torch.stack((
                torch.log1p(observed_total),
                torch.log1p(signal_total),
                torch.log1p(idler_total),
                torch.abs(torch.log((signal_total + 1.0) / (idler_total + 1.0))),
                torch.log1p(observed_mean),
                torch.log1p(observed_fano),
                (observed_mask * (observed_coin < 0.5).to(observed_mask.dtype)).sum(dim=1)
                / observed_count,
                entropy,
                observed_probability.max(dim=1).values,
                0.5 * torch.abs(observed_probability - accidental_observed).sum(dim=1),
                residual_rms,
                residual_bias,
                graph_roughness,
                lag_correlation,
                0.5 * torch.abs(signal_probability - idler_probability).sum(dim=1),
                observed_count / float(self.config.operation_count),
            ), dim=1)

        def _latent_forward(self, values: "torch.Tensor") -> "torch.Tensor":
            for block in self.latent_blocks:
                if self.config.use_activation_checkpointing and self.training:
                    values = checkpoint(block, values, use_reentrant=False)
                else:
                    values = block(values)
            return values

        def forward(
            self,
            operation_inputs: "torch.Tensor",
            target_context: "torch.Tensor",
            signal_counts: "torch.Tensor",
            idler_counts: "torch.Tensor",
        ) -> Dict[str, "torch.Tensor"]:
            if operation_inputs.ndim != 3:
                raise ValueError("operation_inputs must have shape [batch, operation, feature]")
            batch_size, operation_count, feature_count = operation_inputs.shape
            if operation_count != self.config.operation_count or feature_count != self.config.input_feature_dim:
                raise ValueError("operation_inputs disagree with model config")
            if target_context.shape != (batch_size, self.config.target_feature_dim):
                raise ValueError("target_context disagrees with model config")
            if signal_counts.shape != (batch_size, operation_count) or idler_counts.shape != signal_counts.shape:
                raise ValueError("signal/idler counts have the wrong shape")
            operation = self.operation_projection(operation_inputs)
            graph_message = torch.einsum("ij,bjd->bid", self.graph_kernel, operation)
            operation = operation + self.graph_projection(graph_message)
            target = self.target_projection(target_context)
            latent = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            latent = latent + target[:, None, :]
            latent = self.encoder(latent, operation)
            latent = self._latent_forward(latent)
            decoded = self.decoder(operation, latent)
            decoded = self.operation_norm(decoded)

            smoothing = float(self.config.count_smoothing)
            accidental_seed = (signal_counts + smoothing) * (idler_counts + smoothing)
            accidental_probability = accidental_seed / accidental_seed.sum(
                dim=1, keepdim=True,
            ).clamp_min(1e-8)
            noise_summary = self._noise_summary(
                operation_inputs, signal_counts, idler_counts, accidental_probability,
            )
            noise_embedding = self.noise_only_projection(noise_summary)
            pooled = self.global_norm(
                latent.mean(dim=1) + target + self.noise_summary_projection(noise_summary)
            )
            accidental_logit = torch.log(accidental_probability.clamp_min(1e-12))
            coherent_residual = self.coherent_residual_head(decoded).squeeze(-1)
            coherent_residual = coherent_residual - coherent_residual.mean(dim=1, keepdim=True)
            coherent_probability = torch.softmax(accidental_logit + coherent_residual, dim=1)
            mixture_weights = torch.softmax(self.mixture_head(pooled), dim=1)
            uniform_probability = torch.full_like(
                accidental_probability, 1.0 / float(operation_count),
            )
            probability = (
                mixture_weights[:, 0:1] * coherent_probability
                + mixture_weights[:, 1:2] * accidental_probability
                + mixture_weights[:, 2:3] * uniform_probability
            )
            probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
            expected_total = torch.exp(torch.clamp(self.log_total_head(pooled).squeeze(-1), 2.0, 15.0))
            expected_counts = expected_total[:, None] * probability
            dispersion = torch.clamp(
                functional.softplus(self.dispersion_head(pooled).squeeze(-1)),
                1e-5,
                0.75,
            )
            active_mixture = mixture_weights[:, :2].sum(dim=1).clamp_min(1e-8)
            coherent_fraction = mixture_weights[:, 0] / active_mixture
            retrieval_ceiling = torch.sigmoid(self.retrieval_ceiling_head(pooled).squeeze(-1))
            retention = torch.sigmoid(self.retention_head(pooled).squeeze(-1))
            physics_ceiling = (
                coherent_fraction * retrieval_ceiling
                + (1.0 - coherent_fraction) / float(self.config.qubit_count)
            )
            noise_type_logits = self.noise_type_head(noise_embedding)
            noise_coarse_type_logits = self.noise_coarse_type_head(noise_embedding)
            noise_coarse_probability = torch.softmax(noise_coarse_type_logits, dim=1)
            noise_coarse_entropy = -(
                noise_coarse_probability
                * torch.log(noise_coarse_probability.clamp_min(1e-12))
            ).sum(dim=1) / float(math.log(len(NOISE_COARSE_TYPE_NAMES)))
            return {
                "probability": probability,
                "expected_counts": expected_counts,
                "expected_total": expected_total,
                "coherent_probability": coherent_probability,
                "accidental_probability": accidental_probability,
                "mixture_weights": mixture_weights,
                "coherent_fraction": coherent_fraction,
                "nb2_dispersion": dispersion,
                "noise_type_logits": noise_type_logits,
                "noise_coarse_type_logits": noise_coarse_type_logits,
                "noise_coarse_probability": noise_coarse_probability,
                "noise_coarse_entropy": noise_coarse_entropy,
                "noise_coarse_confidence": noise_coarse_probability.max(dim=1).values,
                "noise_strength": torch.sigmoid(self.noise_strength_head(noise_embedding)),
                "retrieval_ceiling": retrieval_ceiling,
                "retention": retention,
                "physics_ceiling": physics_ceiling,
                "best_fidelity": physics_ceiling * retention,
                "ood_energy": self.ood_energy_head(pooled).squeeze(-1),
                "global_embedding": pooled,
                "noise_summary": noise_summary,
                "noise_embedding": noise_embedding,
            }

        def audit_dict(self) -> Dict[str, Any]:
            return {
                "schema": SCALABLE_MODEL_SCHEMA,
                "config": self.config.to_json_dict(),
                "input_audit": dict(SCALABLE_MODEL_INPUT_AUDIT),
                "noise_type_names": list(NOISE_TYPE_NAMES),
                "noise_coarse_type_names": list(NOISE_COARSE_TYPE_NAMES),
                "count_mean_formula": (
                    "expected_total * (w_coherent*p_coherent + "
                    "w_accidental*p_accidental + w_uniform*p_uniform)"
                ),
                "production_subtraction_allowed": False,
            }


else:  # pragma: no cover
    class V7ScalablePhysicsModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("V7ScalablePhysicsModel requires torch")


def _compute_w_witness_loss(
    output: Mapping[str, Any],
    qubit_count: int,
    device: Any,
) -> Any:
    """W-state fidelity witness loss on the physical density matrix.

    The W-state witness margins should be < 0 for genuine multipartite
    entanglement.  Positive margins indicate the state is outside the
    W-state polytope — penalise positive margins.
    """
    rho = output.get("rho_physical")
    if rho is None:
        return torch.zeros((), device=device)
    from ml_recon.physics_hybrid_v4.differentiable_physics import (
        w_witness_margins_torch,
    )
    try:
        margins = w_witness_margins_torch(rho, qubit_count)
        return sum(torch.clamp(m, min=0.0).float().mean() for m in margins) / max(qubit_count, 1)
    except Exception:
        return torch.zeros((), device=device)


def _compute_ppt_negativity_loss(
    output: Mapping[str, Any],
    qubit_count: int,
    device: Any,
) -> Any:
    """PPT negativity entanglement certification loss.

    PPT negativity > 0 across all 1-vs-rest bipartitions certifies
    genuine entanglement.  Penalise negative negativity (entanglement = 0).
    """
    rho = output.get("rho_physical")
    if rho is None:
        return torch.zeros((), device=device)
    from ml_recon.physics_hybrid_v4.differentiable_physics import (
        cross_cut_negativity_1vsrest_torch,
    )
    try:
        negativity = cross_cut_negativity_1vsrest_torch(rho, qubit_count)
        return torch.clamp(-negativity, min=0.0).float().mean()
    except Exception:
        return torch.zeros((), device=device)


def v7_scalable_multitask_loss(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    weights: Mapping[str, float],
) -> Tuple[Any, Dict[str, Any]]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("V7 scalable loss requires torch")
    observed = batch["coin_counts"].to(dtype=output["expected_counts"].dtype)
    observed_mask = batch["observed_mask"].to(dtype=observed.dtype)
    hidden_mask = 1.0 - observed_mask
    mean = output["expected_counts"].clamp_min(1e-8)
    alpha = output["nb2_dispersion"].clamp_min(1e-8)[:, None]
    shape = 1.0 / alpha
    nb2_log_probability = (
        torch.lgamma(observed + shape)
        - torch.lgamma(shape)
        - torch.lgamma(observed + 1.0)
        + shape * (torch.log(shape) - torch.log(shape + mean))
        + observed * (torch.log(mean) - torch.log(shape + mean))
    )
    nb2_nll = -(nb2_log_probability * hidden_mask).sum() / hidden_mask.sum().clamp_min(1.0)
    # Physics-math v3 opt-in: NB2 NLL on the observed mask (real counts).
    # By default this path contributes zero; turning it on lets the model
    # learn the absolute count scale on observed data, which is the
    # bottleneck when training on real measured snapshots where most
    # operations are observed.
    qfi_weight = output.get("qfi_fisher_weight")
    if qfi_weight is not None:
        # qfi_weight is per-sample [batch]; reshape to [batch, 1] so it
        # broadcasts across operations.
        qfi_weight = qfi_weight.to(dtype=observed.dtype)[:, None]
    observed_weight = observed_mask * qfi_weight if qfi_weight is not None else observed_mask
    # Physical Fisher weight (v4 opt-in): multiplied on top of the
    # observed weight when present.  Disabled path leaves the value at
    # 1.0 and is byte-identical to v3.
    physical_fisher_weight = output.get("physical_fisher_weight")
    if physical_fisher_weight is not None:
        physical_fisher_weight = physical_fisher_weight.to(dtype=observed.dtype)[:, None]
        observed_weight = observed_weight * physical_fisher_weight
    # V9 adaptive loss weight (opt-in): per-sample inverse-count weight
    # that up-weights low-count samples (Poisson noise is relatively
    # larger there, so they carry more Fisher information about the
    # rate).  Disabled path leaves the value at 1.0 and the math is
    # byte-identical to v8.
    adaptive_loss_weight = output.get("adaptive_loss_weight")
    if adaptive_loss_weight is not None:
        adaptive_loss_weight = adaptive_loss_weight.to(dtype=observed.dtype)[:, None]
        observed_weight = observed_weight * adaptive_loss_weight
    nb2_nll_observed = -(nb2_log_probability * observed_weight).sum() / observed_weight.sum().clamp_min(1.0)
    hidden_probability = (output["probability"] * hidden_mask).clamp_min(1e-12)
    hidden_probability = hidden_probability / hidden_probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
    hidden_total = (observed * hidden_mask).sum(dim=1).clamp_min(1.0)
    conditional_nll = -(
        observed * hidden_mask * torch.log(hidden_probability)
    ).sum(dim=1).div(hidden_total).mean()
    teacher_probability = batch.get("true_probability")
    if teacher_probability is not None:
        teacher_probability = teacher_probability.to(dtype=mean.dtype).clamp_min(1e-12)
        teacher_cross_entropy = -(
            teacher_probability * hidden_mask * torch.log(hidden_probability)
        ).sum(dim=1).div(hidden_mask.sum(dim=1).clamp_min(1.0)).mean()
    else:
        teacher_cross_entropy = torch.zeros((), dtype=observed.dtype, device=observed.device)
    noise_type_index = batch.get("noise_type_index")
    if noise_type_index is not None:
        noise_type_loss = functional.cross_entropy(
            output["noise_type_logits"], noise_type_index.long(),
        )
    else:
        noise_type_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    noise_coarse_type_index = batch.get("noise_coarse_type_index")
    if noise_coarse_type_index is not None:
        noise_coarse_type_loss = functional.cross_entropy(
            output["noise_coarse_type_logits"], noise_coarse_type_index.long(),
        )
    else:
        noise_coarse_type_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    noise_strength_target = batch.get("noise_strength")
    if noise_strength_target is not None:
        noise_strength_loss = functional.smooth_l1_loss(
            output["noise_strength"], noise_strength_target.to(dtype=mean.dtype),
        )
    else:
        noise_strength_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    mixture_target = batch.get("mixture_weights")
    if mixture_target is not None:
        mixture_loss = functional.smooth_l1_loss(
            output["mixture_weights"], mixture_target.to(dtype=mean.dtype),
        )
    else:
        mixture_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    dispersion_target = batch.get("nb2_dispersion")
    if dispersion_target is not None:
        dispersion_loss = functional.smooth_l1_loss(
            torch.log(output["nb2_dispersion"].clamp_min(1e-6)),
            torch.log(dispersion_target.to(dtype=mean.dtype).clamp_min(1e-6)),
        )
    else:
        dispersion_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    retrieval_truth = batch.get("retrieval_ceiling")
    if retrieval_truth is not None:
        retrieval_loss = functional.smooth_l1_loss(
            output["retrieval_ceiling"], retrieval_truth.to(dtype=mean.dtype),
        )
    else:
        retrieval_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    bestf_target = batch.get("best_fidelity")
    if bestf_target is not None:
        bestf_loss = functional.smooth_l1_loss(
            output["best_fidelity"], bestf_target.to(dtype=mean.dtype),
        )
    else:
        bestf_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v3 opt-in: count calibration.  Compares the predicted
    # total against the observed total on the observed mask; controls the
    # total-count drift that the conditional NLL ignores by construction.
    # Disabled (default) path contributes zero and the loss is
    # byte-identical to v2 when ``count_calibration_weight == 0``.
    expected_total = output["expected_total"]
    observed_total = (observed * observed_mask).sum(dim=1).clamp_min(1.0)
    count_calibration_loss = functional.smooth_l1_loss(
        torch.log(expected_total.clamp_min(1e-6)),
        torch.log(observed_total),
    )
    # Physics-math v4 opt-in: adaptive count calibration.  Matches the
    # second moment (variance) of the predicted counts against the observed
    # variance on the observed mask, in addition to the first moment (the
    # v3 ``count_calibration`` already covers the log-mean).  When
    # ``adaptive_count_calibration_weight == 0`` the loss contributes
    # zero and the math is byte-identical to v3.
    observed_per_op = observed * observed_mask
    observed_var = (
        observed_per_op.var(dim=1, unbiased=False)
    ).clamp_min(1e-6)
    output_probability = output["probability"].clamp_min(1e-6)
    expected_var = (
        expected_total[:, None] * output_probability * (1.0 - output_probability)
    ).sum(dim=1).clamp_min(1e-6)
    adaptive_count_calibration_loss = functional.smooth_l1_loss(
        torch.log(expected_var),
        torch.log(observed_var),
    )
    # Physics-math v4 opt-in: phase residual consistency.  Penalises
    # correlation between the log-amplitude and cyclic phase components
    # of ``phase_residual``.  When ``phase_residual_consistency_weight
    # == 0`` the loss contributes zero and the math is byte-identical to
    # v3.
    phase_residual = output.get("phase_residual")
    if phase_residual is not None:
        log_amp = phase_residual[..., 0]
        cyclic = phase_residual[..., 1]
        log_amp_centered = log_amp - log_amp.mean(dim=1, keepdim=True)
        cyclic_centered = cyclic - cyclic.mean(dim=1, keepdim=True)
        covariance = (log_amp_centered * cyclic_centered).mean(dim=1)
        log_amp_std = log_amp_centered.std(dim=1).clamp_min(1e-6)
        cyclic_std = cyclic_centered.std(dim=1).clamp_min(1e-6)
        consistency = (covariance / (log_amp_std * cyclic_std)).abs().mean()
    else:
        consistency = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v5 opt-in: log-likelihood self-consistency.  The v5
    # forward emits a per-batch ``log_rate_residual``; the loss penalises
    # its magnitude so the count-scale head does not decouple from the
    # dispersion head.  Disabled path contributes zero and the math is
    # byte-identical to v4.
    log_rate_residual = output.get("log_rate_residual")
    if log_rate_residual is not None:
        log_rate_residual = log_rate_residual.to(dtype=observed.dtype)
        log_rate_consistency = log_rate_residual.abs().mean()
    else:
        log_rate_consistency = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v5 opt-in: frequency feedback base.  The v5 forward
    # emits a per-batch ``frequency_feedback_base_norm``; the loss
    # penalises its magnitude so the global frequency feedback does not
    # accumulate a base shift.  Disabled path contributes zero and the
    # math is byte-identical to v4.
    frequency_feedback_base_norm = output.get("frequency_feedback_base_norm")
    if frequency_feedback_base_norm is not None:
        frequency_feedback_base_norm = frequency_feedback_base_norm.to(dtype=observed.dtype)
        frequency_feedback_base_loss = frequency_feedback_base_norm.mean()
    else:
        frequency_feedback_base_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v6 opt-in: Poisson NLL on the observed mask.  Real
    # measured counts at ~30 counts/point with 18% Poisson noise are
    # essentially Poisson; this is the physically correct low-count
    # choice and lets the model learn the absolute count scale without
    # the NB2 dispersion head dragging the rate.  Disabled path
    # contributes zero and the math is byte-identical to v5.
    poisson_log_probability = observed * torch.log(mean.clamp_min(1e-8)) - mean - (
        torch.lgamma(observed + 1.0)
    )
    poisson_nll_observed = -(
        poisson_log_probability * observed_mask
    ).sum() / observed_mask.sum().clamp_min(1.0)
    # Physics-math v7 opt-in: spectral smoothness.  Penalises the
    # graph-Laplacian of the coherent_residual so the per-operation
    # response stays smooth on the AOD frequency grid.  Real measured
    # counts have a smooth frequency response; this prior is equivalent
    # to extra effective samples on noisy measured data.  Disabled path
    # contributes zero and is byte-identical to v6.
    coherent_residual = output.get("coherent_residual")
    graph_kernel = output.get("graph_kernel")
    if coherent_residual is not None and graph_kernel is not None:
        # graph_kernel is [op, op]; neighbour-average operator.
        neighbour_mean = torch.einsum(
            "ij,bj->bi", graph_kernel.to(dtype=coherent_residual.dtype), coherent_residual,
        )
        laplacian = coherent_residual - neighbour_mean
        spectral_smoothness_loss = laplacian.square().mean()
    else:
        spectral_smoothness_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v7 opt-in: operation-point equivariance.  The
    # coherent_residual should be equivariant to a one-step cycle shift
    # of the operation index (physical symmetry of the AOD grid).  The
    # v2 forward precomputes the real double-pass equivariance loss and
    # emits it under ``operation_equivariance``; when the head is off
    # (or the model did not emit it) the component is zero.
    precomputed_equivariance = output.get("operation_equivariance")
    if precomputed_equivariance is not None:
        operation_equivariance_loss = precomputed_equivariance.to(dtype=observed.dtype).mean()
    elif coherent_residual is not None:
        shifted_residual = torch.roll(coherent_residual, shifts=1, dims=1)
        equivariance_target = torch.roll(coherent_residual, shifts=1, dims=1)
        operation_equivariance_loss = (shifted_residual - equivariance_target).square().mean()
    else:
        operation_equivariance_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v8 opt-in: count-scale equivariance.  The v2 forward
    # precomputes the real double-pass loss and emits it under
    # ``count_scale_equivariance``; when the head is off the component
    # is zero and the math is byte-identical to v7.
    precomputed_scale = output.get("count_scale_equivariance")
    if precomputed_scale is not None:
        count_scale_equivariance_loss = precomputed_scale.to(dtype=observed.dtype).mean()
    else:
        count_scale_equivariance_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v12 opt-in: expected calibration error (ECE) on the
    # observed mask.  For each operation, compute the predicted
    # probability that observed_count >= 1 (binary event) and compare
    # against the empirical rate using 5 reliability bins.  ECE is the
    # weighted absolute difference between predicted probability and
    # empirical frequency; minimising ECE directly targets calibration.
    # Disabled path contributes zero and the math is byte-identical to
    # v8 / v9 / v10 / v11.
    teacher_probability_for_ece = batch.get("true_probability")
    if teacher_probability_for_ece is not None:
        # output['probability'] is (batch, operation) per-operation
        # count probability.  Calibration metric: bin operations by
        # predicted probability and compare against the empirical mean
        # observed count within each bin.  This is a count-aware
        # generalisation of the standard classification ECE.
        predicted_probability = torch.clamp(output["probability"], 1e-4, 1.0 - 1e-4)
        flat_confidence = predicted_probability.reshape(-1)
        flat_observed_count = observed.reshape(-1)
        # Build the empirical expected-count under the true probability.
        # teacher_probability_for_ece is (batch, op); expand to (batch*op).
        flat_true_probability = teacher_probability_for_ece.reshape(-1).to(dtype=mean.dtype)
        # Per-operation "accuracy" = whether observed count is above the
        # true probability (a noisy proxy for the model's calibration).
        flat_accuracy = (flat_observed_count > flat_true_probability).to(dtype=mean.dtype)
        bin_count = 5
        bin_edges = torch.linspace(0.0, 1.0, bin_count + 1, dtype=mean.dtype, device=mean.device)
        ece_loss = torch.zeros((), dtype=mean.dtype, device=mean.device)
        for bin_index in range(bin_count):
            lower = bin_edges[bin_index]
            upper = bin_edges[bin_index + 1]
            if bin_index == bin_count - 1:
                in_bin = (flat_confidence >= lower) & (flat_confidence <= upper)
            else:
                in_bin = (flat_confidence >= lower) & (flat_confidence < upper)
            bin_size = in_bin.float().sum()
            if bin_size > 0:
                bin_accuracy = flat_accuracy[in_bin].mean()
                bin_confidence = flat_confidence[in_bin].mean()
                ece_loss = ece_loss + (bin_size / float(flat_confidence.numel())) * (
                    bin_accuracy - bin_confidence
                ).abs()
        expected_calibration_error_loss = ece_loss
    else:
        expected_calibration_error_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v36 opt-in: low-count-tail calibration.  V35's audit
    # showed the <=8-count region (DLCZ single-photon detection) has the
    # worst ECE (0.42 vs global 0.31).  This component re-computes the
    # 5-bin ECE weighted by the inverse observed count, so low-count
    # operations dominate the calibration target.  Disabled path
    # contributes zero.
    low_count_weights = torch.zeros_like(observed, dtype=mean.dtype)
    positive_mask = observed.clamp_min(0.0) > 0.5
    inv_count = 1.0 / observed.clamp_min(1.0).clamp_max(50.0)
    low_count_weights = torch.where(
        positive_mask, inv_count,
        torch.zeros_like(inv_count),
    )
    weight_sum = low_count_weights.sum().clamp_min(1e-8)
    low_count_weights = low_count_weights / weight_sum * float(low_count_weights.numel())
    if teacher_probability_for_ece is not None:
        flat_weights = low_count_weights.reshape(-1)
        weighted_ece_loss = torch.zeros((), dtype=mean.dtype, device=mean.device)
        for bin_index in range(bin_count):
            lower = bin_edges[bin_index]
            upper = bin_edges[bin_index + 1]
            if bin_index == bin_count - 1:
                in_bin = (flat_confidence >= lower) & (flat_confidence <= upper)
            else:
                in_bin = (flat_confidence >= lower) & (flat_confidence < upper)
            bin_weight = flat_weights[in_bin].sum()
            if bin_weight > 0:
                bin_accuracy = (
                    flat_weights[in_bin] * flat_accuracy[in_bin]
                ).sum() / bin_weight
                bin_confidence = (
                    flat_weights[in_bin] * flat_confidence[in_bin]
                ).sum() / bin_weight
                weighted_ece_loss = weighted_ece_loss + (
                    bin_weight / flat_weights.sum()
                ) * (bin_accuracy - bin_confidence).abs()
        low_count_calibration_loss = weighted_ece_loss
    else:
        low_count_calibration_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v27 opt-in: DLCZ Born-rule consistency.  The v22
    # forward emits a per-batch born_consistency = mean |coherent_prob
    # - born_target| where born_target is the Born-rule measurement
    # probability (retrieval-weighted target overlap).  Minimising it
    # makes the coherent component follow the quantum measurement
    # probability.  Disabled path contributes zero.
    born_consistency = output.get("born_consistency")
    if born_consistency is not None:
        born_consistency_loss = born_consistency.to(dtype=observed.dtype).mean()
    else:
        born_consistency_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v41 opt-in: confidence (overconfidence) regularisation.
    # V37/V38 found the raw v22+Bose model is ~5x overconfident (global
    # temperature T=5 needed).  This component penalises the per-operation
    # negative entropy of the predicted probability, so the distribution
    # stays flatter and the raw confidence is not inflated.  Minimising
    # it should pull the fitted temperature toward 1.0.  Disabled path
    # contributes zero.
    probability_for_conf = output.get("probability")
    if probability_for_conf is not None:
        pred_conf = probability_for_conf.clamp(1e-8, 1.0 - 1e-8)
        per_op_entropy = -(pred_conf * torch.log(pred_conf)).sum(dim=1)
        max_entropy = math.log(float(pred_conf.shape[1]))
        confidence_regularization_loss = (max_entropy - per_op_entropy).mean() / max_entropy
    else:
        confidence_regularization_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v55: circular phase consistency ported from V4's
    # differentiable_physics.circular_phase_error.  Uses 1-cos(delta)
    # on the coherent_residual itself (a phase-like quantity in the
    # log-probability space).  The wrap-invariant metric is smooth at
    # +/-pi unlike squared residuals.  Disabled path contributes zero.
    coherent_residual_for_circ = output.get("coherent_residual")
    if coherent_residual_for_circ is not None:
        circular_phase_consistency_loss = (
            1.0 - torch.cos(coherent_residual_for_circ.float())
        ).mean()
    else:
        circular_phase_consistency_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # V69: V4 Von Mises NLL on the coherent_residual with a fixed
    # concentration (log_std=-3, high kappa=20) so the loss becomes
    # equivalent to a sharper wrap-invariant phase regulariser.
    # Output: per-operation log-probability under Von Mises density.
    from ml_recon.physics_hybrid_v4.differentiable_physics import (
        von_mises_nll,
    )
    coherent_residual_vm = output.get("coherent_residual")
    if coherent_residual_vm is not None:
        # High concentration (kappa~20) matches the resolved-phase regime.
        vm_log_std = torch.full_like(
            coherent_residual_vm, -3.0,
            dtype=observed.dtype, device=coherent_residual_vm.device,
        )
        von_mises_phase_loss = von_mises_nll(
            coherent_residual_vm.float(), vm_log_std,
        ).mean()
    else:
        von_mises_phase_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v30 opt-in: DLCZ Bose-statistics consistency.  The
    # v22 forward emits bose_consistency = mean |implied_nbar -
    # physical_nbar| where implied_nbar = dispersion*expected_total
    # (normalised) and physical_nbar is the sqrt(signal*idler) excitation
    # estimate.  Minimising it couples the NB2 dispersion to the Bose
    # thermal excitation statistics.  Disabled path contributes zero.
    bose_consistency = output.get("bose_consistency")
    if bose_consistency is not None:
        bose_consistency_loss = bose_consistency.to(dtype=observed.dtype).mean()
    else:
        bose_consistency_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    # Physics-math v33 opt-in: DLCZ Bose-Einstein counting.  The v22
    # forward emits be_consistency = mean |expected_total -
    # bose_einstein_total| / bose_einstein_total, where the BE total is
    # the exact Bose-Einstein thermal-state mean detected counts
    # (eta*n_bar/(1-eta*n_bar)).  Disabled path contributes zero.
    be_consistency = output.get("be_consistency")
    if be_consistency is not None:
        be_consistency_loss = be_consistency.to(dtype=observed.dtype).mean()
    else:
        be_consistency_loss = torch.zeros((), dtype=observed.dtype, device=observed.device)
    components = {
        "nb2_nll": nb2_nll,
        "nb2_nll_observed": nb2_nll_observed,
        "poisson_nll_observed": poisson_nll_observed,
        "conditional_nll": conditional_nll,
        "teacher_cross_entropy": teacher_cross_entropy,
        "noise_type": noise_type_loss,
        "noise_coarse_type": noise_coarse_type_loss,
        "noise_strength": noise_strength_loss,
        "mixture": mixture_loss,
        "dispersion": dispersion_loss,
        "retrieval": retrieval_loss,
        "best_fidelity": bestf_loss,
        "count_calibration": count_calibration_loss,
        "adaptive_count_calibration": adaptive_count_calibration_loss,
        "phase_residual_consistency": consistency,
        "log_rate_consistency": log_rate_consistency,
        "frequency_feedback_base": frequency_feedback_base_loss,
        "spectral_smoothness": spectral_smoothness_loss,
        "operation_equivariance": operation_equivariance_loss,
        "count_scale_equivariance": count_scale_equivariance_loss,
        "expected_calibration_error": expected_calibration_error_loss,
        "born_consistency": born_consistency_loss,
        "bose_consistency": bose_consistency_loss,
        "be_consistency": be_consistency_loss,
        "low_count_calibration": low_count_calibration_loss,
        "confidence_regularization": confidence_regularization_loss,
        "circular_phase_consistency": circular_phase_consistency_loss,
        "von_mises_phase": von_mises_phase_loss,
        # Physics-math V57: W-state fidelity witness (Gühne & Tóth 2009)
        # and PPT negativity entanglement certification ported from V4's
        # differentiable_physics.  Requires rho_physical in the output
        # (from v22's _count_density_proxy + HermitianDensityGraphEncoder).
        "w_witness_loss": (
            _compute_w_witness_loss(output, qubit_count=observed.shape[1], device=observed.device)
            if "rho_physical" in output
            else torch.zeros((), dtype=observed.dtype, device=observed.device)
        ),
        "ppt_negativity_loss": (
            _compute_ppt_negativity_loss(output, qubit_count=observed.shape[1], device=observed.device)
            if "rho_physical" in output
            else torch.zeros((), dtype=observed.dtype, device=observed.device)
        ),
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in components.items())
    diagnostics = {
        name: value.detach() for name, value in components.items()
    }
    noise_index = batch.get("noise_type_index")
    noise_coarse_index = batch.get("noise_coarse_type_index")
    bestf_target = batch.get("best_fidelity")
    if noise_index is not None:
        noise_accuracy = (
            output["noise_type_logits"].argmax(dim=1) == noise_index.long()
        ).float().mean().detach()
    else:
        noise_accuracy = torch.zeros((), dtype=torch.float32)
    if noise_coarse_index is not None:
        noise_coarse_accuracy = (
            output["noise_coarse_type_logits"].argmax(dim=1) == noise_coarse_index.long()
        ).float().mean().detach()
    else:
        noise_coarse_accuracy = torch.zeros((), dtype=torch.float32)
    if bestf_target is not None:
        bestf_mae = torch.mean(torch.abs(
            output["best_fidelity"] - bestf_target.to(dtype=mean.dtype)
        )).detach()
    else:
        bestf_mae = torch.zeros((), dtype=torch.float32)
    diagnostics.update({
        "total": total.detach(),
        "noise_accuracy": noise_accuracy,
        "noise_coarse_accuracy": noise_coarse_accuracy,
        "bestf_mae": bestf_mae,
    })
    return total, diagnostics


def parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def estimate_fsdp_memory_bytes(
    parameter_count_value: int,
    config: V7ScalableModelConfig,
    *,
    world_size: int,
    batch_size_per_gpu: int,
    bytes_per_activation: int = 2,
) -> Dict[str, int]:
    """Conservative planning estimate; not execution evidence."""

    parameters = int(parameter_count_value)
    workers = max(1, int(world_size))
    model_state = int(np_ceil_div(parameters * 18, workers))
    activation_factor = 6 if config.use_activation_checkpointing else 18
    activations = int(
        batch_size_per_gpu
        * (config.operation_count + config.latent_tokens)
        * config.width
        * max(config.layers, 1)
        * int(bytes_per_activation)
        * activation_factor
    )
    reserve = int(2.0 * 1024 ** 3)
    return {
        "sharded_model_optimizer_gradient_bytes": model_state,
        "estimated_activation_bytes": activations,
        "runtime_reserve_bytes": reserve,
        "estimated_peak_bytes_per_gpu": model_state + activations + reserve,
    }


def np_ceil_div(numerator: int, denominator: int) -> int:
    return -(-int(numerator) // int(denominator))


__all__ = [
    "NOISE_SUMMARY_NAMES",
    "SCALABLE_MODEL_INPUT_AUDIT",
    "SCALABLE_MODEL_SCHEMA",
    "V7ScalableModelConfig",
    "V7ScalablePhysicsModel",
    "estimate_fsdp_memory_bytes",
    "parameter_count",
    "v7_scalable_multitask_loss",
]
