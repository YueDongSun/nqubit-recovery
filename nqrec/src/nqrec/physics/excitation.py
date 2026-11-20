"""Excitation rate, cross-correlation g_si and background (DECISIONS J0, J4).

The PI's 1-4% is the measured signal rate per trial, s = chi * eta_s, with chi
the thermal mean of the heralded collective mode. Background b is a click
probability per trial on one arm.

    g_si = 1 + 1 / [chi (1 + r_s)(1 + r_i)],  r_k = b_k / (eta_k chi)

(DLCZ 2001; Sangouard et al., RMP 83, 33, 2011). Double excitation is thermal,
p2 ~ chi^2. The closed form sets b; the full sector forward in
nqrec.physics.dlcz_forward reports its own g_si as a diagnostic.
"""

from __future__ import annotations

import math

import torch


def chi_from_signal_rate(signal_rate: float, eta_signal: float) -> float:
    if not 0.0 < signal_rate < eta_signal:
        raise ValueError(f"signal rate {signal_rate} must lie in (0, eta_s = {eta_signal})")
    return signal_rate / eta_signal


def g_si_closed_form(chi: float, eta_signal: float, eta_idler: float, b_signal: float, b_idler: float) -> float:
    r_s = b_signal / (eta_signal * chi)
    r_i = b_idler / (eta_idler * chi)
    return 1.0 + 1.0 / (chi * (1.0 + r_s) * (1.0 + r_i))


def background_for_g_si(g_si: float, chi: float, eta_signal: float, eta_idler: float) -> float:
    """Equal background on both arms that gives g_si (exact root of the closed form).

    (1 + A b)(1 + B b) = 1 / (chi (g_si - 1)) with A = 1/(eta_s chi), B = 1/(eta_i chi).
    """
    ceiling = 1.0 + 1.0 / chi
    if not 1.0 < g_si < ceiling:
        raise ValueError(f"g_si = {g_si} is outside (1, {ceiling:.3f}) at chi = {chi}")
    a = 1.0 / (eta_signal * chi)
    b = 1.0 / (eta_idler * chi)
    rhs = 1.0 / (chi * (g_si - 1.0))
    # a b x^2 + (a + b) x + 1 - rhs = 0, positive root.
    quad_a = a * b
    quad_b = a + b
    quad_c = 1.0 - rhs
    return (-quad_b + math.sqrt(quad_b * quad_b - 4.0 * quad_a * quad_c)) / (2.0 * quad_a)


def thermal_sectors(chi: torch.Tensor, n_max: int) -> torch.Tensor:
    """P(n) = chi^n / (1 + chi)^(n + 1) for n = 0..n_max; the tail above n_max goes to n_max.

    chi has any shape; the result appends a sector axis of length n_max + 1.
    """
    if torch.any(chi < 0):
        raise ValueError("thermal mean must be non-negative")
    ratio = chi / (1.0 + chi)
    n = torch.arange(n_max + 1, dtype=chi.dtype, device=chi.device)
    probs = (1.0 - ratio).unsqueeze(-1) * ratio.unsqueeze(-1).pow(n)
    tail = ratio.pow(n_max + 1)
    probs = torch.cat([probs[..., :-1], (probs[..., -1] + tail).unsqueeze(-1)], dim=-1)
    return probs
