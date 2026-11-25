"""Exact moments of the native nested beta-binomial four-cell distribution.

This is the same generative factorization as the native count likelihood:
S|R, C|S, T|(R-S), with independent beta draws for the three branches and
one shared concentration. It is not a Dirichlet-multinomial and does not
introduce a separate free Gaussian count variance.
"""

from __future__ import annotations

import torch

from .detected_count_forward import require_probability_tensor


def nested_four_cell_moments(
    *, read_trials: torch.Tensor, signal_probability: torch.Tensor,
    coin_given_signal_probability: torch.Tensor,
    idler_given_no_signal_probability: torch.Tensor,
    concentration: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return exact means and full singular covariance conditional on known R.

The caller must supply native/public integer opportunities. Inferring R from
hidden query counts is forbidden at the model-input boundary; estimating an
average R from visible operations does not make that estimate a known trial
count. This mathematical helper consumes neither batches nor query targets.
All tensors use ordinary broadcasting (use [batch,1] for row parameters).
"""
    signal = require_probability_tensor("signal_probability", signal_probability)
    coin = require_probability_tensor("coin_given_signal_probability", coin_given_signal_probability)
    idler = require_probability_tensor("idler_given_no_signal_probability", idler_given_no_signal_probability)
    if read_trials.dtype == torch.bool or read_trials.is_complex():
        raise TypeError("read trials must be nonnegative integers")
    trials = read_trials.to(torch.float64)
    dispersion = concentration.to(torch.float64)
    if not bool(torch.isfinite(trials).all()) or bool(((trials < 0) | (trials != trials.round())).any()):
        raise ValueError("read trials must be finite nonnegative integers")
    if not bool(torch.isfinite(dispersion).all()) or bool((dispersion <= 0).any()):
        raise ValueError("concentration must be finite and positive")
    trials, signal, coin, idler, dispersion = torch.broadcast_tensors(trials, signal, coin, idler, dispersion)
    signal_mean = trials * signal
    no_signal_mean = trials * (1 - signal)
    signal_variance = trials * signal * (1 - signal) * (trials + dispersion) / (1 + dispersion)
    coin_variance = coin.square() * signal_variance + coin * (1 - coin) * (
        signal_variance + signal_mean.square() + dispersion * signal_mean
    ) / (1 + dispersion)
    idler_variance = idler.square() * signal_variance + idler * (1 - idler) * (
        signal_variance + no_signal_mean.square() + dispersion * no_signal_mean
    ) / (1 + dispersion)
    signal_idler_covariance = -idler * signal_variance
    signal_coin_covariance = coin * signal_variance
    idler_coin_covariance = -idler * coin * signal_variance
    branch_covariance = torch.stack((
        torch.stack((signal_variance, signal_idler_covariance, signal_coin_covariance), -1),
        torch.stack((signal_idler_covariance, idler_variance, idler_coin_covariance), -1),
        torch.stack((signal_coin_covariance, idler_coin_covariance, coin_variance), -1),
    ), -2)
    # [n00,n10,n01,n11] = [R-S-T,S-C,T,C].
    transform = trials.new_tensor(((-1, -1, 0), (1, 0, -1), (0, 1, 0), (0, 0, 1)))
    covariance = transform @ branch_covariance @ transform.T
    mean = torch.stack((
        no_signal_mean * (1 - idler), signal_mean * (1 - coin),
        no_signal_mean * idler, signal_mean * coin,
    ), -1)
    return {"four_cell_mean": mean, "four_cell_covariance": covariance}
