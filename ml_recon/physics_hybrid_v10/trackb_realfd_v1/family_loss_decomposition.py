"""Exact native-likelihood family/branch contributions, without reweighting."""
from __future__ import annotations

from typing import Any, Mapping

import torch

from ml_recon.physics_hybrid_v8.fpga_joint_observation import _beta_binomial_log_probability


branch_names = ('signal_given_read', 'coin_given_signal', 'idler_given_no_signal')
probability_names = ('signal_probability', 'coin_given_signal_probability', 'idler_given_no_signal_probability')


def family_native_loss_contributions(
    prediction: Mapping[str, Any], batch: Mapping[str, Any], theta: torch.Tensor,
    *, registered_families: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Attribute the native objective and its theta gradient to every family.

Each contribution uses ALL selected query operations as its denominator, so
their sum reconstructs the unchanged four-cell objective. Family means and
zero-parent counts are descriptive fields, never replacement training losses.
"""
    selected = batch['query_mask'] & batch['operation_mask']
    denominator = int(selected.sum())
    if denominator == 0:
        raise ValueError('Native loss decomposition requires selected queries')
    # Select before evaluating log-gamma: ignored counts must not create NaN
    # derivatives that later leak through a zero mask into shared theta.
    cells = batch['four_cell_counts'][selected].double()
    n00, n10, n01, n11 = cells.unbind(-1)
    signal = n10 + n11
    no_signal = n00 + n01
    parents = (signal + no_signal, signal, no_signal)
    successes = (signal, n11, n01)
    concentration = prediction['four_cell_concentration'].double()[:, None].expand_as(selected)[selected]
    operation_indices = torch.where(selected)[1]
    families = batch['operation_families']
    if set(families) != set(registered_families):
        raise ValueError('Loss decomposition family metadata is incomplete')
    output = {family: {} for family in registered_families}
    for branch, probability_name, success, parent in zip(
        branch_names, probability_names, successes, parents, strict=True,
    ):
        terms = -_beta_binomial_log_probability(success, parent, prediction[probability_name][selected].double(), concentration)
        if not bool(torch.isfinite(terms).all()):
            raise FloatingPointError('Native branch terms are nonfinite')
        for family in registered_families:
            family_mask = torch.tensor([value == family for value in families], device=selected.device)
            mask = family_mask[operation_indices]
            selected_count = int(mask.sum())
            informative_count = int((mask & (parent > 0)).sum())
            if selected_count:
                native_sum = terms[mask].sum()
                contribution = native_sum / denominator
                gradient = torch.autograd.grad(contribution, theta, retain_graph=True)[0].detach().double()
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError('Family/branch contribution gradient is nonfinite')
            else:
                native_sum = terms.new_zeros(())
                contribution = terms.new_zeros(())
                gradient = theta.detach().double().new_zeros(theta.shape)
            output[family][branch] = {
                'selected_queries': selected_count, 'informative_queries': informative_count,
                'zero_parent_queries': selected_count - informative_count,
                'native_objective_query_denominator': denominator,
                'native_objective_contribution': float(contribution.detach()),
                'native_nll_sum': float(native_sum.detach()),
                'family_query_mean_nll': None if not selected_count else float(native_sum.detach()) / selected_count,
                'gradient_contribution': gradient,
                'status': 'NO_SELECTED_QUERIES' if not selected_count else 'ZERO_PARENT_ONLY' if not informative_count else 'INFORMATIVE',
            }
    return output
