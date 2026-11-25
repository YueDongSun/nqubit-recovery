"""Count-native likelihoods for V10 real observation recovery.

The authoritative FPGA four-cell event is consumed exactly once.  Global
multi-click counts remain a separate global-scope likelihood and are never
expanded into operation-local targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


WEAK_POSTERIOR_MAX_WEIGHT = 0.05
_MAX_EXACT_NB2_RISING_TERMS = 1_000_000
_MIN_POSITIVE_MEAN_NB2_DISPERSION = 1.0e-6
_NB2_ALPHA_SERIES_MAX_R = 0.125
_NB2_ALPHA_SERIES_TERMS = 20


def _nb2_log_mean_value(mean: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Evaluate log1p(alpha * mean) / alpha without losing subnormal products."""
    product = alpha * mean
    small_product = product < 1.0e-4
    small_value = mean * (
        1.0
        + product * (-0.5 + product * (1.0 / 3.0 + product * (-0.25 + product / 5.0)))
    )
    # The omitted alternating term on the small branch is bounded by
    # mean * product**5 / 6.  Keep the nonselected denominator nonzero.
    large_product = torch.where(small_product, torch.ones_like(product), product)
    large_value = (mean / large_product) * torch.log1p(large_product)
    return torch.where(small_product, small_value, large_value)


def _nb2_alpha_first_second(
    mean: torch.Tensor, alpha: torch.Tensor, count: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact alpha derivatives, with a convergent r-series near zero product."""
    product = alpha * mean
    inverse_one_plus_product = torch.reciprocal(1.0 + product)
    r_value = product * inverse_one_plus_product
    small_product = r_value < _NB2_ALPHA_SERIES_MAX_R
    r_over_alpha = mean * inverse_one_plus_product
    d_series = torch.full_like(r_value, 1.0 / (_NB2_ALPHA_SERIES_TERMS + 1))
    e_series = torch.full_like(r_value, 1.0 / (_NB2_ALPHA_SERIES_TERMS + 2))
    for index in range(_NB2_ALPHA_SERIES_TERMS - 2, -1, -1):
        d_series = 1.0 / (index + 2) + r_value * d_series
        e_series = 1.0 / (index + 3) + r_value * e_series
    # The d/e series tails are bounded by r**20 / (22 * (1-r)) and
    # r**20 / (23 * (1-r)), respectively, before their prefactors.
    # The wider interval avoids cancellation of O(r**3) Hessian terms.
    d_value = r_over_alpha.square() * d_series
    e_value = 2.0 * r_over_alpha.pow(3) * e_series
    # Divide the finite numerator successively: alpha**(-2/-3) alone may
    # underflow although the complete derivative remains representable.
    log_difference = torch.log1p(product) - r_value
    large_d_value = (log_difference / alpha) / alpha
    large_e_value = (
        ((2.0 * log_difference - r_value.square()) / alpha) / alpha
    ) / alpha
    f_alpha = -torch.where(small_product, d_value, large_d_value)
    f_alpha_alpha = torch.where(small_product, e_value, large_e_value)
    count_value = int(count.detach().cpu())
    terms = torch.arange(count_value, dtype=torch.float64, device=alpha.device)
    term_inverse = torch.reciprocal(1.0 + alpha * terms)
    term_ratio = terms * term_inverse
    # Combine rising-factorial and count-scale derivatives before summing;
    # (mean-j)*u/(1+alpha*j) is exactly mean*u-j/(1+alpha*j).
    combined_alpha = ((mean - terms) * inverse_one_plus_product * term_inverse).sum()
    rising_alpha_alpha = term_ratio.square().sum()
    count_alpha_alpha = -count * (mean * inverse_one_plus_product).square()
    return (
        combined_alpha + f_alpha,
        rising_alpha_alpha + f_alpha_alpha + count_alpha_alpha,
    )


class _Stable_nb2_mean_gradient(torch.autograd.Function):
    """First mean derivative with analytic, stable Hessian-vector support."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, mean: torch.Tensor,
                alpha: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mean, alpha, count)
        inverse_one_plus_product = torch.reciprocal(1.0 + alpha * mean)
        return ((mean - count) / mean) * inverse_one_plus_product

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        mean, alpha, count = ctx.saved_tensors
        inverse_one_plus_product = torch.reciprocal(1.0 + alpha * mean)
        count_over_mean = count / mean
        mean_hessian = (
            (count_over_mean * inverse_one_plus_product) / mean
            - (1.0 - count_over_mean)
            * inverse_one_plus_product * (alpha * inverse_one_plus_product)
        )
        cross_hessian = (
            (count - mean)
            * inverse_one_plus_product * inverse_one_plus_product
        )
        return grad_output * mean_hessian, grad_output * cross_hessian, None


class _Stable_nb2_alpha_gradient(torch.autograd.Function):
    """First alpha derivative with the matching analytic Hessian row."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, mean: torch.Tensor,
                alpha: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mean, alpha, count)
        first, _second = _nb2_alpha_first_second(mean, alpha, count)
        return first

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        mean, alpha, count = ctx.saved_tensors
        inverse_one_plus_product = torch.reciprocal(1.0 + alpha * mean)
        cross_hessian = (
            (count - mean)
            * inverse_one_plus_product * inverse_one_plus_product
        )
        _first, alpha_hessian = _nb2_alpha_first_second(mean, alpha, count)
        return grad_output * cross_hessian, grad_output * alpha_hessian, None


class _Stable_nb2_scalar_nll(torch.autograd.Function):
    """Exact NB2 scalar NLL with analytic, finite-product derivative plumbing."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, count: torch.Tensor,
                mean: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(count, mean, alpha)
        count_value = int(count.detach().cpu())
        terms = torch.arange(count_value, dtype=torch.float64, device=alpha.device)
        rising_correction = torch.log1p(alpha * terms).sum()
        return (
            torch.lgamma(count + 1.0)
            - rising_correction
            + _nb2_log_mean_value(mean, alpha)
            + count * torch.logaddexp(torch.log(alpha), -torch.log(mean))
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, torch.Tensor]:
        count, mean, alpha = ctx.saved_tensors
        mean_gradient = _Stable_nb2_mean_gradient.apply(mean, alpha, count)
        alpha_gradient = _Stable_nb2_alpha_gradient.apply(mean, alpha, count)
        return None, grad_output * mean_gradient, grad_output * alpha_gradient


def _require_finite(name: str, value: torch.Tensor) -> torch.Tensor:
    result = value.to(torch.float64)
    if not bool(torch.all(torch.isfinite(result)).detach()):
        raise ValueError(f"{name} must be finite")
    return result


def _require_strict_probability(name: str, value: torch.Tensor) -> torch.Tensor:
    """Require the finite likelihood interior; exact 0/1 is physical but not finite-NLL input.

    The physical forward may represent deterministic 0/1 limits, but an observed
    beta-binomial likelihood must declare a separate limiting model.  Silently
    clipping those limits changes its likelihood, so OBS_V5 rejects them here.
    """
    result = _require_finite(name, value)
    if bool(torch.any((result <= 0.0) | (result >= 1.0)).detach()):
        raise ValueError(f"{name} must be strictly between zero and one for finite NLL")
    return result


def _require_positive(name: str, value: torch.Tensor) -> torch.Tensor:
    result = _require_finite(name, value)
    if bool(torch.any(result <= 0.0).detach()):
        raise ValueError(f"{name} must be strictly positive")
    return result


@dataclass(frozen=True)
class RealCountLikelihoodWeights:
    four_cell: float = 1.0
    global_multi: float = 1.0
    target_pattern: float = 1.0
    weak_posterior: float = 0.0
    action: float = 0.0
    gain: float = 0.0
    jacobian: float = 0.0

    def validate(self) -> None:
        if min(self.four_cell, self.global_multi, self.target_pattern) <= 0.0:
            raise ValueError("real primary count likelihood weights must be positive")
        if not 0.0 <= self.weak_posterior <= WEAK_POSTERIOR_MAX_WEIGHT:
            raise ValueError("real weak posterior weight exceeds 0.05")
        if any(value != 0.0 for value in (self.action, self.gain, self.jacobian)):
            raise ValueError("real_static cannot carry action, gain, or Jacobian loss")


def four_cell_nll(
    four_cell_counts: torch.Tensor,
    signal_probability: torch.Tensor,
    coin_given_signal_probability: torch.Tensor,
    idler_given_no_signal_probability: torch.Tensor,
    concentration: torch.Tensor,
    operation_mask: torch.Tensor,
) -> torch.Tensor:
    """Nested beta-binomial NLL per visible operation, used exactly once."""
    from ml_recon.physics_hybrid_v8.fpga_joint_observation import (
        FPGAJointFourCellCounts,
        nested_beta_binomial_log_likelihood,
    )

    if four_cell_counts.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise ValueError("four-cell counts must be native integer tensors")
    if (
        four_cell_counts.ndim != 3
        or four_cell_counts.shape[-1] != 4
        or operation_mask.dtype != torch.bool
        or operation_mask.shape != four_cell_counts.shape[:2]
    ):
        raise ValueError("four_cell_counts/mask must be [batch,operations,4]/[batch,operations]")
    mask = operation_mask
    expected_operation_shape = four_cell_counts.shape[:2]
    for name, value in (
        ("signal probability", signal_probability),
        ("coin-given-signal probability", coin_given_signal_probability),
        ("idler-given-no-signal probability", idler_given_no_signal_probability),
    ):
        if value.shape != expected_operation_shape:
            raise ValueError(f"{name} requires exact [batch,operations] shape")
    if concentration.shape != (four_cell_counts.shape[0],):
        raise ValueError("beta-binomial concentration requires exact [batch] shape")
    if not bool(torch.any(mask).detach()):
        raise ValueError("real four-cell likelihood selected zero operations")
    total_log_likelihood = torch.zeros(
        (), dtype=torch.float64, device=four_cell_counts.device
    )
    visible_operations = 0
    for batch_index in range(four_cell_counts.shape[0]):
        visible = mask[batch_index]
        count = int(visible.sum().item())
        if count == 0:
            continue
        cells = _require_finite(
            "four-cell counts",
            four_cell_counts[batch_index:batch_index + 1, visible],
        )
        if bool(torch.any(cells < 0.0).detach()):
            raise ValueError("four-cell counts cannot be negative")
        signal = _require_strict_probability(
            "signal probability",
            signal_probability[batch_index:batch_index + 1, visible],
        )
        coin_given_signal = _require_strict_probability(
            "coin-given-signal probability",
            coin_given_signal_probability[batch_index:batch_index + 1, visible],
        )
        idler_given_no_signal = _require_strict_probability(
            "idler-given-no-signal probability",
            idler_given_no_signal_probability[batch_index:batch_index + 1, visible],
        )
        batch_concentration = _require_positive(
            "beta-binomial concentration", concentration[batch_index:batch_index + 1]
        )
        observation = FPGAJointFourCellCounts(
            counts=cells,
            valid_mask=torch.ones((1, count), dtype=torch.bool, device=cells.device))
        likelihood = nested_beta_binomial_log_likelihood(
            observation,
            signal,
            coin_given_signal,
            idler_given_no_signal,
            batch_concentration,
        )
        total_log_likelihood = total_log_likelihood + likelihood.sum()
        visible_operations += count
    if visible_operations == 0:
        raise ValueError("real four-cell likelihood selected zero operations")
    return -total_log_likelihood / visible_operations


def _nb2_nll(count: torch.Tensor, mean: torch.Tensor,
             dispersion: torch.Tensor) -> torch.Tensor:
    value = _require_finite("NB2 count", count)
    if bool(torch.any(value < 0.0).detach()):
        raise ValueError("NB2 count cannot be negative")
    mu = _require_finite("NB2 mean", mean)
    if bool(torch.any(mu < 0.0).detach()):
        raise ValueError("NB2 mean cannot be negative")
    alpha = _require_positive("NB2 dispersion", dispersion)
    degenerate = mu == 0.0
    if bool(torch.any(degenerate & (value != 0.0)).detach()):
        raise ValueError("zero-exposure global multi law requires zero observed counts")
    # At zero exposure the exact count law is degenerate at zero.  Do not use
    # a numerical floor, because that would invent a physical exposure.
    positive = ~degenerate
    result = mu * 0.0 + alpha * 0.0
    for index in zip(*torch.where(positive)):
        count_value = int(value[index].detach().cpu())
        alpha_value = alpha[index]
        mean_value = mu[index]
        if float(alpha_value.detach()) < _MIN_POSITIVE_MEAN_NB2_DISPERSION:
            raise ValueError(
                "NB2 positive-mean dispersion is outside the registered stable numeric domain"
            )
        if count_value > _MAX_EXACT_NB2_RISING_TERMS:
            raise ValueError(
                "NB2 positive-mean count is outside the registered stable numeric domain"
            )
        alpha_mean_product = alpha_value * mean_value
        if (
            not bool(
                torch.isfinite(float(max(count_value, 1)) / mean_value).detach()
            )
            or not bool(torch.isfinite(alpha_mean_product).detach())
            or not bool((alpha_mean_product > 0.0).detach())
            or not bool(
                torch.isfinite(
                    alpha_value * float(max(count_value - 1, 0))
                ).detach()
            )
        ):
            raise ValueError(
                "NB2 positive-mean value or gradient is outside the registered stable numeric domain"
            )
        # This is the exact NB2 log-PMF, not a Poisson limit.  The custom
        # derivative path preserves finite first and second derivatives without
        # relying on an overflow-prone elementary backward graph.
        result[index] = _Stable_nb2_scalar_nll.apply(
            value[index], mean_value, alpha_value
        )
    return result


def global_multi_nll(
    global_multi: torch.Tensor,
    global_multi_mask: torch.Tensor,
    global_multi_scope: tuple[str, ...] | list[str],
    multi_means: torch.Tensor,
    pair_logits: torch.Tensor,
    dispersion: torch.Tensor,
) -> torch.Tensor:
    """Native global NB2 + exact-pair multinomial likelihood."""
    if global_multi.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("global multi must be native integer tensors")
    if global_multi.ndim != 2 or global_multi.shape[1] != 9:
        raise ValueError("global_multi must be [batch,9]")
    if (
        multi_means.shape != (global_multi.shape[0], 3)
        or pair_logits.shape != (global_multi.shape[0], 6)
    ):
        raise ValueError("global multi predictions have the wrong shape")
    if (
        global_multi_mask.shape != (global_multi.shape[0],)
        or global_multi_mask.dtype != torch.bool
    ):
        raise ValueError("global multi mask requires exact bool [batch] shape")
    if len(global_multi_scope) != global_multi.shape[0]:
        raise ValueError("global multi scope must align to the batch")
    if dispersion.shape != (global_multi.shape[0], 3):
        raise ValueError("global multi dispersion requires exact [batch,3] shape")
    mask = global_multi_mask
    if not bool(torch.any(mask).detach()):
        raise ValueError("global multi likelihood selected zero observations")
    values = _require_finite("global multi", global_multi[mask])
    if bool(torch.any(values < 0.0).detach()):
        raise ValueError("global multi counts cannot be negative")
    active_means = _require_finite("global multi means", multi_means[mask])
    if bool(torch.any(active_means < 0.0).detach()):
        raise ValueError("global multi means cannot be negative")
    active_pair_logits = _require_finite("pair logits", pair_logits[mask])
    active_dispersion = _require_positive(
        "global multi dispersion", dispersion[mask]
    )
    active_scope = tuple(
        scope
        for scope, visible in zip(global_multi_scope, mask.detach().cpu().tolist())
        if visible
    )
    if any(scope not in ("global", "state") for scope in active_scope):
        raise ValueError("global multi cannot have operation-local scope")
    if bool(torch.any(values[:, 1:7].sum(dim=1) != values[:, 0]).detach()):
        raise ValueError("exact pair composition must sum to multi2")
    totals = torch.stack((values[:, 0], values[:, 7], values[:, 8]), dim=-1)
    total_nll = _nb2_nll(totals, active_means, active_dispersion).sum(dim=-1)
    pair_nll = -(
        values[:, 1:7] * F.log_softmax(active_pair_logits.to(torch.float64), dim=-1)
    ).sum(dim=-1)
    return (total_nll + pair_nll).mean()


def target_pattern_hellinger(
    predicted_counts: torch.Tensor,
    observed_counts: torch.Tensor,
    operation_mask: torch.Tensor,
) -> torch.Tensor:
    mask = operation_mask.to(predicted_counts.dtype)
    predicted = predicted_counts.clamp_min(0.0) * mask
    observed = observed_counts.clamp_min(0.0) * mask
    predicted = predicted / predicted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    observed = observed / observed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (1.0 - torch.sqrt(predicted * observed + 1e-12).sum(dim=-1)).mean()


def combine_real_losses(losses: Dict[str, torch.Tensor],
                        weights: RealCountLikelihoodWeights) -> torch.Tensor:
    weights.validate()
    required = {"four_cell", "global_multi", "target_pattern"}
    if missing := required - set(losses):
        raise ValueError(f"real likelihood is missing components: {sorted(missing)}")
    total = (weights.four_cell * losses["four_cell"]
             + weights.global_multi * losses["global_multi"]
             + weights.target_pattern * losses["target_pattern"])
    if weights.weak_posterior:
        if "weak_posterior" not in losses:
            raise ValueError("weak posterior weight requires a masked weak loss")
        total = total + weights.weak_posterior * losses["weak_posterior"]
    return total


def combine_real_sector_losses(
    losses: Dict[str, torch.Tensor],
    weights: RealCountLikelihoodWeights,
) -> torch.Tensor:
    """V4 native sector objective without a duplicate pattern loss."""
    if weights.four_cell <= 0.0 or weights.global_multi <= 0.0:
        raise ValueError("V4 sector primary likelihood weights must be positive")
    if not 0.0 <= weights.weak_posterior <= WEAK_POSTERIOR_MAX_WEIGHT:
        raise ValueError("V4 weak posterior weight exceeds 0.05")
    if any(value != 0.0 for value in (weights.action, weights.gain, weights.jacobian)):
        raise ValueError("real_static cannot carry action, gain, or Jacobian loss")
    if set(losses) - {"four_cell", "global_multi", "weak_posterior"}:
        raise ValueError("V4 sector objective received a duplicate/unregistered loss")
    if not {"four_cell", "global_multi"} <= set(losses):
        raise ValueError("V4 sector objective requires four-cell and global-multi losses")
    total = weights.four_cell * losses["four_cell"] \
        + weights.global_multi * losses["global_multi"]
    if weights.weak_posterior:
        if "weak_posterior" not in losses:
            raise ValueError("weak posterior weight requires a weak posterior loss")
        total = total + weights.weak_posterior * losses["weak_posterior"]
    return total


__all__ = [
    "RealCountLikelihoodWeights", "WEAK_POSTERIOR_MAX_WEIGHT",
    "combine_real_losses", "combine_real_sector_losses", "four_cell_nll", "global_multi_nll",
    "target_pattern_hellinger",
]
