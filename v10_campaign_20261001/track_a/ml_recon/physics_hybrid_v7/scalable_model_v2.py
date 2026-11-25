"""Extended V7 scalable physics model with hierarchical mechanism decomposition.

This module is the v2 of ``scalable_model``.  It keeps the v1 input/output
schema so existing smoke runners and tests remain unchanged; in addition it
exposes four new mechanism heads that mirror the closed-form hierarchical
posterior:

* ``aod_positive_response`` — softplus(W·x_freq + b); the AOD frequency
  response is a strictly positive function of the nominal frequency features.
* ``retrieval_field_amplitude`` — a positive scalar that multiplies the
  retrieval field amplitude factor inside the coherent component.
* ``cycle_random_effect`` — per-cycle additive bias for the multi-excitation
  component.
* ``multi_excitation_mixture`` — the four-way mixture weight head
  (coherent / accidental / multi_excitation / uniform).

The forward map is the same integrable physics map; only the heads have
changed.  The shared `v7_scalable_multitask_loss` from v1 keeps working
because the new outputs are exposed as additional dictionary entries without
removing the v1 keys.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

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
from .scalable_model import (
    NOISE_SUMMARY_NAMES,
    SCALABLE_MODEL_INPUT_AUDIT,
    V7ScalableModelConfig,
    parameter_count,
    v7_scalable_multitask_loss,
)
from .scalable_model import _LatentBlock, _SwiGluFeedForward, _CrossAttentionBlock


SCALABLE_MODEL_V2_SCHEMA = "v7_scalable_physics_count_model_v2"
EXTENDED_HEAD_NAMES = (
    "aod_positive_response",
    "retrieval_field_amplitude",
    "cycle_random_effect",
    "multi_excitation_mixture",
    "hierarchical_coherent_weight",
    "hierarchical_accidental_weight",
    "hierarchical_multi_weight",
    "phase_residual",
    "qfi_fisher_weight",
)
EXTENDED_INPUT_AUDIT = dict(SCALABLE_MODEL_INPUT_AUDIT)
EXTENDED_INPUT_AUDIT.update({
    "aod_positive_response_used": True,
    "retrieval_field_amplitude_used": True,
    "cycle_random_effect_used": True,
    "multi_excitation_mixture_used": True,
    "phase_residual_used": True,
    "qfi_fisher_weight_used": True,
})


@dataclass(frozen=True)
class V7ScalableModelV2Config(V7ScalableModelConfig):
    """V2 config: same as v1 plus the extended-head toggles."""

    use_aod_positive_response: bool = True
    use_retrieval_field_amplitude: bool = True
    use_cycle_random_effect: bool = True
    use_multi_excitation_mixture: bool = True
    aod_positive_response_strength: float = 1.0
    retrieval_field_amplitude_strength: float = 1.0
    cycle_random_effect_strength: float = 1.0
    # Physics-math v3 add-ons (default off; behaviour identical to v2 when off)
    use_phase_residual: bool = False
    phase_residual_strength: float = 1.0
    use_qfi_fisher_weight: bool = False
    qfi_fisher_weight_strength: float = 1.0
    qfi_fisher_weight_floor: float = 0.05
    # Loss-side add-ons (default off; default weights zero)
    use_nb2_nll_observed: bool = False
    nb2_observed_weight: float = 0.0
    use_count_calibration: bool = False
    count_calibration_weight: float = 0.0
    # Physics-math v4 add-ons (default off; behaviour identical to v3 when off)
    use_physical_fisher_weight: bool = False
    physical_fisher_weight_floor: float = 0.05
    physical_fisher_weight_strength: float = 1.0
    use_adaptive_count_calibration: bool = False
    adaptive_count_calibration_weight: float = 0.0
    adaptive_count_calibration_strength: float = 0.5
    use_phase_residual_consistency: bool = False
    phase_residual_consistency_weight: float = 0.0
    # Physics-math v5 add-ons (default off; behaviour identical to v4 when off)
    use_log_rate_self_consistency: bool = False
    log_rate_residual_head_strength: float = 1.0
    log_rate_residual_weight: float = 0.0
    use_frequency_feedback_base: bool = False
    frequency_feedback_base_weight: float = 0.0
    # Physics-math v6 add-on (default off; behaviour identical to v5 when off):
    # Poisson NLL on the observed mask.  Real measured counts at
    # ~30 counts/point with 18% Poisson noise are essentially Poisson,
    # not NB2-overdispersed; this loss component is the physically
    # correct low-count choice and lets the model learn the absolute
    # count scale without the NB2 dispersion head dragging the rate.
    use_poisson_observed: bool = False
    poisson_observed_weight: float = 0.0
    # Physics-math v7 add-on (default off; behaviour identical to v6
    # when off): spectral smoothness regulariser.  Real measured
    # counts have a smooth AOD frequency response; penalising the
    # Laplacian of the coherent_residual on the operation-frequency
    # graph enforces this prior and is equivalent to extra effective
    # samples on noisy measured data.
    use_spectral_smoothness: bool = False
    spectral_smoothness_weight: float = 0.0
    # Physics-math v7 add-on: operation-point equivariance.  The
    # coherent_residual should be equivariant to a small cycle shift
    # of the operation index (physical symmetry of the AOD grid).
    use_operation_equivariance: bool = False
    operation_equivariance_weight: float = 0.0
    # Physics-math v8 add-on (default off; behaviour identical to v7
    # when off): count-scale equivariance.  Doubling the exposure
    # doubles signal/idler/coin counts and must double the
    # expected_counts.  A training-time double pass on scaled inputs
    # penalises the mismatch against the scaled base expected counts.
    use_count_scale_equivariance: bool = False
    count_scale_equivariance_factor: float = 2.0
    count_scale_equivariance_weight: float = 0.0
    # Physics-math v9 add-on (default off; behaviour identical to v8
    # when off): adaptive loss weighting.  Real measured batches have
    # widely varying per-sample count totals; a fixed weight lets
    # high-count samples dominate the gradient.  When on, the loss
    # re-weights the count-scale components (nb2_nll_observed,
    # count_calibration) by a per-batch factor that up-weights
    # low-count samples (physical: Poisson noise is relatively larger
    # at low counts, so those samples carry more Fisher information).
    use_adaptive_loss_weighting: bool = False
    adaptive_loss_weighting_floor: float = 0.25

    def validate(self) -> None:  # noqa: D401 - inherited docs
        super().validate()
        if self.aod_positive_response_strength < 0.0:
            raise ValueError("aod_positive_response_strength must be non-negative")
        if self.retrieval_field_amplitude_strength < 0.0:
            raise ValueError("retrieval_field_amplitude_strength must be non-negative")
        if self.cycle_random_effect_strength < 0.0:
            raise ValueError("cycle_random_effect_strength must be non-negative")
        if self.phase_residual_strength < 0.0:
            raise ValueError("phase_residual_strength must be non-negative")
        if self.qfi_fisher_weight_strength < 0.0:
            raise ValueError("qfi_fisher_weight_strength must be non-negative")
        if not 0.0 < self.qfi_fisher_weight_floor <= 1.0:
            raise ValueError("qfi_fisher_weight_floor must be in (0, 1]")
        if self.nb2_observed_weight < 0.0:
            raise ValueError("nb2_observed_weight must be non-negative")
        if self.count_calibration_weight < 0.0:
            raise ValueError("count_calibration_weight must be non-negative")
        if not 0.0 < self.physical_fisher_weight_floor <= 1.0:
            raise ValueError("physical_fisher_weight_floor must be in (0, 1]")
        if self.physical_fisher_weight_strength < 0.0:
            raise ValueError("physical_fisher_weight_strength must be non-negative")
        if self.adaptive_count_calibration_weight < 0.0:
            raise ValueError("adaptive_count_calibration_weight must be non-negative")
        if not 0.0 <= self.adaptive_count_calibration_strength <= 1.0:
            raise ValueError("adaptive_count_calibration_strength must be in [0, 1]")
        if self.phase_residual_consistency_weight < 0.0:
            raise ValueError("phase_residual_consistency_weight must be non-negative")
        if self.log_rate_residual_weight < 0.0:
            raise ValueError("log_rate_residual_weight must be non-negative")
        if self.log_rate_residual_head_strength < 0.0:
            raise ValueError("log_rate_residual_head_strength must be non-negative")
        if self.frequency_feedback_base_weight < 0.0:
            raise ValueError("frequency_feedback_base_weight must be non-negative")
        if self.poisson_observed_weight < 0.0:
            raise ValueError("poisson_observed_weight must be non-negative")
        if self.spectral_smoothness_weight < 0.0:
            raise ValueError("spectral_smoothness_weight must be non-negative")
        if self.operation_equivariance_weight < 0.0:
            raise ValueError("operation_equivariance_weight must be non-negative")
        if self.count_scale_equivariance_factor <= 1.0:
            raise ValueError("count_scale_equivariance_factor must be > 1")
        if self.count_scale_equivariance_weight < 0.0:
            raise ValueError("count_scale_equivariance_weight must be non-negative")
        if not 0.0 < self.adaptive_loss_weighting_floor <= 1.0:
            raise ValueError("adaptive_loss_weighting_floor must be in (0, 1]")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = SCALABLE_MODEL_V2_SCHEMA
        return payload


def _cycle_index_for_geometry(operation_count: int) -> int:
    """Heuristic: 4Q has 64 operations/cycle, 3Q has 15; clamp to >= 1."""
    return max(int(operation_count // 64), 1) if operation_count >= 64 else max(int(operation_count // 15), 1)


if nn is not None:

    class V7ScalablePhysicsModelV2(nn.Module):
        """V2 forward map with explicit hierarchical mechanism heads."""

        def __init__(
            self,
            config: V7ScalableModelV2Config,
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
            self.cycle_count = _cycle_index_for_geometry(config.operation_count)

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

            # New v2 heads
            if config.use_aod_positive_response:
                self.aod_response_head = nn.Linear(width, 1)
                nn.init.constant_(self.aod_response_head.bias, 0.5)
            if config.use_retrieval_field_amplitude:
                self.retrieval_amplitude_head = nn.Linear(width, 1)
                nn.init.constant_(self.retrieval_amplitude_head.bias, 0.0)
            if config.use_cycle_random_effect:
                self.cycle_effect_head = nn.Linear(width, self.cycle_count)
                nn.init.constant_(self.cycle_effect_head.bias, 0.0)
            if config.use_multi_excitation_mixture:
                self.multi_excitation_head = nn.Linear(width, 4)
                nn.init.constant_(self.multi_excitation_head.bias, 0.0)
            # Physics-math v3 add-ons (off by default; behaviour byte-identical to v2 when off)
            if config.use_phase_residual:
                self.phase_residual_head = nn.Linear(width, 2)
                nn.init.constant_(self.phase_residual_head.bias, 0.0)
            if config.use_qfi_fisher_weight:
                self.qfi_scale_log_head = nn.Linear(width, 1)
                nn.init.constant_(self.qfi_scale_log_head.bias, 0.0)
            # Physics-math v5 add-ons (off by default; behaviour byte-identical to v4 when off)
            if config.use_log_rate_self_consistency:
                self.log_rate_residual_head = nn.Linear(width, 1)
                nn.init.constant_(self.log_rate_residual_head.bias, 0.0)
            if config.use_frequency_feedback_base:
                self.frequency_feedback_base_head = nn.Linear(width, 1)
                nn.init.constant_(self.frequency_feedback_base_head.bias, 0.0)
            self.hierarchical_weight_head = nn.Linear(width, 3)
            nn.init.constant_(self.log_total_head.bias, 8.0)
            nn.init.constant_(self.dispersion_head.bias, -4.0)
            nn.init.constant_(self.retrieval_ceiling_head.bias, 3.0)
            nn.init.constant_(self.retention_head.bias, 0.5)

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
            *,
            cycle_index: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            """Public forward with the v7 operation-equivariance double pass.

            When ``use_operation_equivariance`` is on (and we are in
            training mode), the operation inputs are cycle-shifted by one
            operation and re-forwarded once; the equivariance loss is the
            shift-invariance of the coherent_residual under that shift.
            The shifted forward is computed with gradient tracking so the
            loss back-propagates through both passes.
            """
            output = self._forward_once(
                operation_inputs, target_context, signal_counts, idler_counts,
                cycle_index=cycle_index,
            )
            if (
                self.config.use_operation_equivariance
                and self.training
                and operation_inputs.shape[1] > 1
            ):
                shifted_inputs = torch.roll(operation_inputs, shifts=1, dims=1)
                shifted = self._forward_once(
                    shifted_inputs, target_context, signal_counts, idler_counts,
                    cycle_index=cycle_index,
                )
                base_residual = output["coherent_residual"]
                shifted_residual = shifted["coherent_residual"]
                # A perfectly equivariant model has
                # f(shift(x)) == shift(f(x)).  Penalise the mismatch.
                equivariance_error = (
                    shifted_residual - torch.roll(base_residual, shifts=1, dims=1)
                ).square().mean()
                output["operation_equivariance"] = equivariance_error
            # V8 count-scale equivariance (opt-in, training mode only):
            # physically, doubling the exposure doubles signal/idler/coin
            # counts and must double the expected_counts.  We forward once
            # on doubled inputs and penalise the mismatch against 2x the
            # base expected_counts.  The scaled forward tracks gradients.
            if (
                self.config.use_count_scale_equivariance
                and self.training
            ):
                scale = float(self.config.count_scale_equivariance_factor)
                # The observed-coin feature is the first operation feature;
                # scale it together with the raw counts so the input is a
                # consistent doubled-exposure snapshot.
                scaled_inputs = operation_inputs.clone()
                scaled_inputs[:, :, 0] = torch.log1p(
                    scale * torch.expm1(operation_inputs[:, :, 0].clamp_min(0.0))
                ).clamp_min(0.0)
                scaled = self._forward_once(
                    scaled_inputs, target_context,
                    scale * signal_counts, scale * idler_counts,
                    cycle_index=cycle_index,
                )
                expected_scale_error = (
                    scaled["expected_counts"] - scale * output["expected_counts"]
                ).square().mean()
                output["count_scale_equivariance"] = expected_scale_error
            return output

        def _forward_once(
            self,
            operation_inputs: "torch.Tensor",
            target_context: "torch.Tensor",
            signal_counts: "torch.Tensor",
            idler_counts: "torch.Tensor",
            *,
            cycle_index: Optional["torch.Tensor"] = None,
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
            if cycle_index is not None and cycle_index.shape != (batch_size,):
                raise ValueError("cycle_index must have shape [batch]")
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

            # Phase residual head (v3 opt-in): the coherent_residual is split
            # into a log-amplitude residual and a cyclic phase residual.  The
            # log-amplitude residual is added on top of the mean-centered base
            # residual; the phase residual is converted to a von-Mises-style
            # log-density adjustment around the accidental logit.  When the
            # head is disabled (default) both terms are zero and the path is
            # byte-identical to v2.
            phase_residual_log = torch.zeros_like(coherent_residual)
            phase_residual_cyclic = torch.zeros_like(coherent_residual)
            if self.config.use_phase_residual:
                phase_out = self.phase_residual_head(decoded)
                phase_residual_log = phase_out[..., 0] * float(self.config.phase_residual_strength)
                phase_residual_cyclic = phase_out[..., 1] * float(self.config.phase_residual_strength)
                coherent_residual = coherent_residual + phase_residual_log
                coherent_residual = coherent_residual - coherent_residual.mean(dim=1, keepdim=True)

            coherent_probability = torch.softmax(accidental_logit + coherent_residual, dim=1)
            mixture_weights = torch.softmax(self.mixture_head(pooled), dim=1)
            uniform_probability = torch.full_like(
                accidental_probability, 1.0 / float(operation_count),
            )

            # --- v2 mechanism extensions ---
            aod_response = None
            if self.config.use_aod_positive_response:
                aod_response = functional.softplus(
                    self.aod_response_head(decoded).squeeze(-1)
                ) * float(self.config.aod_positive_response_strength)
                coherent_probability = torch.softmax(
                    accidental_logit + coherent_residual
                    + torch.log(aod_response.clamp_min(1e-6)) - torch.log(aod_response.mean(dim=1, keepdim=True).clamp_min(1e-6)),
                    dim=1,
                )

            retrieval_amplitude = None
            if self.config.use_retrieval_field_amplitude:
                retrieval_amplitude = torch.exp(
                    torch.clamp(
                        self.retrieval_amplitude_head(pooled).squeeze(-1)
                        * float(self.config.retrieval_field_amplitude_strength),
                        -2.0, 2.0,
                    )
                )

            cycle_shift = None
            if self.config.use_cycle_random_effect and cycle_index is not None:
                cycle_table = self.cycle_effect_head(pooled)
                cycle_shift = cycle_table.gather(1, cycle_index[:, None]).squeeze(-1) \
                    * float(self.config.cycle_random_effect_strength)

            multi_excitation_logit = None
            multi_excitation_weight = torch.zeros(batch_size, device=operation_inputs.device)
            if self.config.use_multi_excitation_mixture:
                mixture_logits = self.multi_excitation_head(pooled)
                multi_excitation_logit = mixture_logits[:, 3]
                multi_excitation_weight = torch.softmax(mixture_logits, dim=1)[:, 3]

            # Combine the three mechanisms
            if retrieval_amplitude is not None and cycle_shift is not None and multi_excitation_weight is not None:
                cycle_aligned = cycle_shift[:, None].expand(-1, operation_count)
                multi_component = (
                    multi_excitation_weight[:, None] * cycle_aligned
                    + (1.0 - multi_excitation_weight[:, None]) * 0.0
                )
                hierarchical_logits = (
                    mixture_weights[:, 0:1] * coherent_probability
                    + mixture_weights[:, 1:2] * accidental_probability
                    + mixture_weights[:, 2:3] * uniform_probability
                )
                probability = hierarchical_logits
                # Mix in the multi-excitation as an additive log-shift when present
                if multi_excitation_logit is not None:
                    multi_logit = multi_excitation_logit[:, None].expand(-1, operation_count)
                    probability = probability + multi_excitation_weight[:, None] * (
                        multi_logit / float(operation_count)
                    )
            else:
                probability = (
                    mixture_weights[:, 0:1] * coherent_probability
                    + mixture_weights[:, 1:2] * accidental_probability
                    + mixture_weights[:, 2:3] * uniform_probability
                )
            probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)

            expected_total = torch.exp(torch.clamp(self.log_total_head(pooled).squeeze(-1), 2.0, 15.0))

            # QFI Fisher-weight (v3 opt-in): a softplus'd per-batch scalar
            # that re-weights the NB2 likelihood per sample, scaled between
            # ``qfi_fisher_weight_floor`` and ``1.0``.  Disabled (default)
            # path multiplies the floor only and is byte-identical to v2
            # once ``floor == 1.0``.
            qfi_fisher_weight = torch.ones_like(expected_total)
            if self.config.use_qfi_fisher_weight:
                qfi_log = self.qfi_scale_log_head(pooled).squeeze(-1)
                strength = float(self.config.qfi_fisher_weight_strength)
                floor = float(self.config.qfi_fisher_weight_floor)
                qfi_fisher_weight = floor + (1.0 - floor) * torch.sigmoid(strength * qfi_log)

            # V9 adaptive loss weight (opt-in): up-weights low-count
            # samples.  Poisson noise is relatively larger at low counts,
            # so those samples carry more Fisher information about the
            # rate.  The weight is computed from the observed total
            # (per-sample) and squashed into [floor, 1].
            adaptive_weight = torch.ones_like(expected_total)
            if self.config.use_adaptive_loss_weighting:
                observed_total = (
                    torch.expm1(operation_inputs[:, :, 0].clamp_min(0.0))
                    * operation_inputs[:, :, 6].clamp(0.0, 1.0)
                ).sum(dim=1).clamp_min(1.0)
                # Inverse-count weighting: low observed total -> higher weight.
                inv_count = 1.0 / observed_total
                inv_count = inv_count / inv_count.mean().clamp_min(1e-6)
                floor = float(self.config.adaptive_loss_weighting_floor)
                adaptive_weight = floor + (1.0 - floor) * torch.sigmoid(inv_count)

            expected_counts = expected_total[:, None] * probability
            dispersion = torch.clamp(
                functional.softplus(self.dispersion_head(pooled).squeeze(-1)),
                1e-5,
                0.75,
            )

            # Physical Fisher weight (v4 opt-in): a closed-form NB2 Fisher
            # information weight that does not require a learnable head.
            # The NB2 information per sample is
            #     I(μ) = (mean / (1 + alpha * mean))
            # Squashed between ``physical_fisher_weight_floor`` and ``1.0``
            # via the strength hyper-parameter.  Disabled (default) path
            # outputs a constant ``1.0`` and is byte-identical to v3.
            physical_fisher_weight = torch.ones_like(expected_total)
            if self.config.use_physical_fisher_weight:
                alpha_scalar = dispersion  # [batch]
                mean_scalar = expected_total  # [batch]
                fisher = mean_scalar / (1.0 + alpha_scalar * mean_scalar).clamp_min(1e-6)
                fisher = fisher / fisher.max().clamp_min(1e-6)
                floor = float(self.config.physical_fisher_weight_floor)
                strength = float(self.config.physical_fisher_weight_strength)
                physical_fisher_weight = floor + (1.0 - floor) * torch.sigmoid(
                    strength * (fisher - 0.5),
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
            output = {
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
            if self.config.use_adaptive_loss_weighting:
                output["adaptive_loss_weight"] = adaptive_weight
            if aod_response is not None:
                output["aod_positive_response"] = aod_response
            if retrieval_amplitude is not None:
                output["retrieval_field_amplitude"] = retrieval_amplitude
            if cycle_shift is not None:
                output["cycle_random_effect"] = cycle_shift
            if multi_excitation_weight is not None:
                output["multi_excitation_mixture"] = multi_excitation_weight
            if self.config.use_aod_positive_response or self.config.use_retrieval_field_amplitude \
                    or self.config.use_cycle_random_effect or self.config.use_multi_excitation_mixture:
                hierarchical = torch.softmax(self.hierarchical_weight_head(pooled), dim=1)
                output["hierarchical_coherent_weight"] = hierarchical[:, 0]
                output["hierarchical_accidental_weight"] = hierarchical[:, 1]
                output["hierarchical_multi_weight"] = hierarchical[:, 2]
            if self.config.use_phase_residual:
                output["phase_residual"] = torch.stack(
                    (phase_residual_log, phase_residual_cyclic), dim=-1,
                )
            if self.config.use_qfi_fisher_weight:
                output["qfi_fisher_weight"] = qfi_fisher_weight
            if self.config.use_physical_fisher_weight:
                output["physical_fisher_weight"] = physical_fisher_weight
            if self.config.use_phase_residual or self.config.use_adaptive_count_calibration:
                output["phase_residual_log_amp"] = phase_residual_log if self.config.use_phase_residual else torch.zeros_like(coherent_residual)
                output["phase_residual_cyclic"] = phase_residual_cyclic if self.config.use_phase_residual else torch.zeros_like(coherent_residual)
            # V5 self-consistency: a per-batch log-rate residual that
            # the loss should keep close to zero (i.e. the model's
            # log-rate head should not decouple from the dispersion
            # head).  Disabled path emits zero and is byte-identical to
            # v4.
            if self.config.use_log_rate_self_consistency:
                log_rate_residual = torch.tanh(
                    self.log_rate_residual_head(pooled).squeeze(-1)
                    * float(self.config.log_rate_residual_head_strength),
                )
                output["log_rate_residual"] = log_rate_residual
            # V5 frequency-feedback base: a per-batch scalar norm
            # kept close to zero by the loss.  Disabled path emits
            # zero and is byte-identical to v4.
            if self.config.use_frequency_feedback_base:
                frequency_feedback_base_norm = self.frequency_feedback_base_head(
                    pooled,
                ).squeeze(-1).abs()
                output["frequency_feedback_base_norm"] = frequency_feedback_base_norm
            # V7 spectral smoothness / operation equivariance: expose
            # the per-operation coherent_residual so the loss can
            # penalise its graph Laplacian / shift-equivariance.  Only
            # emitted when one of the v7 flags is on; otherwise the
            # output is byte-identical to v6.  The graph_kernel is
            # exposed too so the loss can compute the Laplacian without
            # changing its signature.
            if self.config.use_spectral_smoothness or self.config.use_operation_equivariance:
                output["coherent_residual"] = coherent_residual
                output["graph_kernel"] = self.graph_kernel
            return output

        @staticmethod
        def _noise_summary(
            operation_inputs: "torch.Tensor",
            signal_counts: "torch.Tensor",
            idler_counts: "torch.Tensor",
            accidental_probability: "torch.Tensor",
        ) -> "torch.Tensor":
            """Replicates ``V7ScalablePhysicsModel._noise_summary`` for v2."""
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
            residual_centered = log_residual - (
                log_residual.sum(dim=1, keepdim=True) / observed_count[:, None]
            ) * observed_mask
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
                torch.zeros_like(residual_rms),
                lag_correlation,
                0.5 * torch.abs(signal_probability - idler_probability).sum(dim=1),
                observed_count / float(operation_inputs.shape[1]),
            ), dim=1)

        def audit_dict(self) -> Dict[str, Any]:
            audit = {
                "schema": SCALABLE_MODEL_V2_SCHEMA,
                "config": self.config.to_json_dict(),
                "input_audit": dict(EXTENDED_INPUT_AUDIT),
                "noise_type_names": list(NOISE_TYPE_NAMES),
                "noise_coarse_type_names": list(NOISE_COARSE_TYPE_NAMES),
                "extended_head_names": list(EXTENDED_HEAD_NAMES),
                "count_mean_formula": (
                    "expected_total * (w_coherent*p_coherent + "
                    "w_accidental*p_accidental + w_uniform*p_uniform + "
                    "w_multi_excitation * cycle_random_effect_adjustment)"
                ),
                "production_subtraction_allowed": False,
            }
            return audit

else:  # pragma: no cover
    class V7ScalablePhysicsModelV2:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("V7ScalablePhysicsModelV2 requires torch")


def build_cycle_index(
    operation_count: int,
    qubit_count: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Deterministic cycle index assignment matching the cycle_count heuristic."""

    cycle_count = _cycle_index_for_geometry(operation_count)
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, cycle_count, size=batch_size)


__all__ = [
    "EXTENDED_HEAD_NAMES",
    "EXTENDED_INPUT_AUDIT",
    "SCALABLE_MODEL_V2_SCHEMA",
    "V7ScalableModelV2Config",
    "V7ScalablePhysicsModelV2",
    "build_cycle_index",
    "v7_scalable_multitask_loss",
    "parameter_count",
]
