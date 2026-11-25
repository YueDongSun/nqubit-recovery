"""Campaign-scoped source multiplicity for the V6 research count model.

The source model produces exclusive vacuum/single/pair/triple/quadruple
photon-pattern probabilities before detector loss and detector dark counts are
applied.  Multiplicity nuisance parameters are campaign/time-block quantities;
they are never predicted independently for each target sample.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - NumPy contracts remain available.
    torch = None
    nn = None


SOURCE_MULTIPLICITY_SCHEMA = "source_multiplicity"
SOURCE_MULTIPLICITY_VERSION = 1
DEFAULT_MULTIPLICITY_PRIOR: Tuple[float, ...] = (0.05, 0.94, 0.009, 0.0009, 0.0001)
SINGLE_PATTERN_INDICES: Tuple[int, ...] = (1, 2, 4, 8)
PAIR_PATTERN_INDICES: Tuple[int, ...] = (3, 5, 9, 6, 10, 12)
TRIPLE_PATTERN_INDICES: Tuple[int, ...] = (7, 11, 13, 14)


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _normalize_nonnegative(value: Any, *, epsilon: float = 1e-12) -> Any:
    if _is_torch(value):
        nonnegative = value.clamp_min(0.0)
        return nonnegative / nonnegative.sum(dim=-1, keepdim=True).clamp_min(float(epsilon))
    nonnegative = np.maximum(np.asarray(value, dtype=float), 0.0)
    return nonnegative / np.maximum(nonnegative.sum(axis=-1, keepdims=True), float(epsilon))


def source_pattern_probabilities(
    mode_population: Any,
    mixture_weights: Any,
    *,
    pair_residual: Optional[Any] = None,
) -> Any:
    """Return normalized exclusive 16-pattern source probabilities.

    ``mode_population`` has a final four-detector axis.  ``mixture_weights`` has
    final axis ``[pi0, pi1, pi2, pi3, pi4]`` and is broadcast across operation
    axes.  Pair residuals are bounded nuisance corrections to the six pair
    categories and cannot create or remove total ``pi2`` mass.
    """

    population = _normalize_nonnegative(mode_population)
    weights = _normalize_nonnegative(mixture_weights)
    if population.shape[-1] != 4 or weights.shape[-1] != 5:
        raise ValueError("mode_population must end in 4 and mixture_weights must end in 5")

    if _is_torch(population):
        output = torch.zeros(population.shape[:-1] + (16,), dtype=population.dtype, device=population.device)
        output[..., 0] = weights[..., 0]
        for detector, pattern_index in enumerate(SINGLE_PATTERN_INDICES):
            output[..., pattern_index] = weights[..., 1] * population[..., detector]
        pair_terms = torch.stack(
            [population[..., left] * population[..., right] for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))],
            dim=-1,
        )
        if pair_residual is not None:
            residual = 0.1 * torch.tanh(torch.as_tensor(pair_residual, dtype=population.dtype, device=population.device))
            pair_terms = pair_terms * torch.exp(residual)
        pair_terms = _normalize_nonnegative(pair_terms)
        for pair_index, pattern_index in enumerate(PAIR_PATTERN_INDICES):
            output[..., pattern_index] = weights[..., 2] * pair_terms[..., pair_index]
        triple_terms = torch.stack(
            [population[..., a] * population[..., b] * population[..., c] for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))],
            dim=-1,
        )
        triple_terms = _normalize_nonnegative(triple_terms)
        for triple_index, pattern_index in enumerate(TRIPLE_PATTERN_INDICES):
            output[..., pattern_index] = weights[..., 3] * triple_terms[..., triple_index]
        output[..., 15] = weights[..., 4]
        return output / output.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(output.dtype).eps)

    output = np.zeros(population.shape[:-1] + (16,), dtype=float)
    output[..., 0] = weights[..., 0]
    for detector, pattern_index in enumerate(SINGLE_PATTERN_INDICES):
        output[..., pattern_index] = weights[..., 1] * population[..., detector]
    pair_terms = np.stack(
        [population[..., left] * population[..., right] for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))],
        axis=-1,
    )
    if pair_residual is not None:
        pair_terms *= np.exp(0.1 * np.tanh(np.asarray(pair_residual, dtype=float)))
    pair_terms = _normalize_nonnegative(pair_terms)
    for pair_index, pattern_index in enumerate(PAIR_PATTERN_INDICES):
        output[..., pattern_index] = weights[..., 2] * pair_terms[..., pair_index]
    triple_terms = np.stack(
        [population[..., a] * population[..., b] * population[..., c] for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))],
        axis=-1,
    )
    triple_terms = _normalize_nonnegative(triple_terms)
    for triple_index, pattern_index in enumerate(TRIPLE_PATTERN_INDICES):
        output[..., pattern_index] = weights[..., 3] * triple_terms[..., triple_index]
    output[..., 15] = weights[..., 4]
    return output / np.maximum(output.sum(axis=-1, keepdims=True), np.finfo(float).eps)


if nn is not None:

    class SourceMultiplicityModel(nn.Module):
        """Campaign prior plus strongly shrunk time-block nuisance residuals."""

        def __init__(
            self,
            *,
            mixture_prior: Sequence[float] = DEFAULT_MULTIPLICITY_PRIOR,
            time_blocks: int = 1,
            time_block_residual_scale: float = 0.05,
            prior_std_logit: float = 0.25,
        ) -> None:
            super().__init__()
            prior = torch.as_tensor(mixture_prior, dtype=torch.float32)
            if prior.shape != (5,) or bool(torch.any(prior < 0.0)) or float(prior.sum()) <= 0.0:
                raise ValueError("mixture_prior must contain five non-negative values with positive sum")
            prior = prior / prior.sum()
            prior_logit = torch.log(prior.clamp_min(1e-12))
            self.register_buffer("mixture_prior_logit", prior_logit)
            self.campaign_mixture_logit = nn.Parameter(prior_logit.clone())
            self.time_block_residual = nn.Parameter(torch.zeros(max(1, int(time_blocks)), 5))
            self.time_block_residual_scale = float(time_block_residual_scale)
            self.prior_std_logit = max(float(prior_std_logit), 1e-6)

        def mixture_weights(self, time_block_index: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            logits = self.campaign_mixture_logit
            if time_block_index is not None:
                block = self.time_block_residual[time_block_index.long()]
                logits = logits + self.time_block_residual_scale * torch.tanh(block)
            return torch.softmax(logits, dim=-1)

        def forward(
            self,
            mode_population: "torch.Tensor",
            time_block_index: Optional["torch.Tensor"] = None,
            *,
            pair_residual: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            weights = self.mixture_weights(time_block_index)
            while weights.ndim < mode_population.ndim:
                weights = weights.unsqueeze(-2)
            return source_pattern_probabilities(mode_population, weights, pair_residual=pair_residual)

        def shrinkage_loss(self) -> "torch.Tensor":
            campaign = ((self.campaign_mixture_logit - self.mixture_prior_logit) / self.prior_std_logit).square().mean()
            return campaign + self.time_block_residual.square().mean()

        def diagnostics(self, time_block_index: Optional["torch.Tensor"] = None) -> Dict[str, "torch.Tensor"]:
            return {
                "source_mixture_weights": self.mixture_weights(time_block_index),
                "source_prior_deviation": (
                    self.campaign_mixture_logit - self.mixture_prior_logit
                ).square().mean(),
                "source_time_block_residual_norm": torch.linalg.vector_norm(self.time_block_residual),
                "source_regularization": self.shrinkage_loss(),
            }


else:  # pragma: no cover

    class SourceMultiplicityModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("SourceMultiplicityModel requires torch")


__all__ = [
    "SourceMultiplicityModel",
    "source_pattern_probabilities",
    "SOURCE_MULTIPLICITY_SCHEMA",
    "SOURCE_MULTIPLICITY_VERSION",
    "DEFAULT_MULTIPLICITY_PRIOR",
    "SINGLE_PATTERN_INDICES",
    "PAIR_PATTERN_INDICES",
    "TRIPLE_PATTERN_INDICES",
]
