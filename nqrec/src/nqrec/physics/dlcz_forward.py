"""Round-6 forward map: thermal sectors, finite signal efficiency (DECISIONS J3).

The heralded collective mode holds n spin-wave quanta with thermal weights
P(n) = chi^n / (1 + chi)^(n + 1). Each quantum emitted a Stokes photon that the
signal arm detects with probability eta_s. An optional orthogonal thermal mode
(occupation n_o) adds Stokes clicks and no idler (round-5 observation rule).

Given n quanta in the W mode:
    no signal click   A_n = (1 - b_s) (1 - eta_s)^n / (1 + n_o eta_s)
    no idler click    D_n = (1 - b_i) (1 - q_n)
    multi-cell read   q_n = 1 - (1 - q_ret)^n,  q_ret = u^dagger (Gamma o c c^dagger) u
    single-cell read  q_1 = e p, q_2 = 2 e (1 - e) p^2 + 2 e p (1 - p)  (Pu 2018, B3),
                      q_n = 1 - (1 - e p)^n for n >= 3,  p = |c_j|^2, e = beta_j^2 eta_j
with u_j = beta_j sqrt(eta_j) exp(i phase_sign theta_j). Cells, in Track B order
[00, 10, 01, 11]: sum_n P(n) [A D, (1 - A) D, A (1 - D), (1 - A)(1 - D)].

Each setting is evaluated on its support only: c, Gamma and eta are gathered
onto the at most m cells the setting reads, with c the amplitudes of the full
normalized state. Padding positions carry beta = 0.
"""

from __future__ import annotations

import torch

from nqrec.conventions import require_phase_sign
from nqrec.physics.excitation import thermal_sectors


def analyzer_on_support(beta: torch.Tensor, theta: torch.Tensor, eta: torch.Tensor, phase_sign: int) -> torch.Tensor:
    """u (B, S, m) from beta, theta (S, m) and eta (B, S, m)."""
    sign = require_phase_sign(phase_sign)
    phase = torch.exp(1j * sign * theta.to(torch.complex128))
    return beta.to(torch.complex128) * phase * torch.sqrt(eta).to(torch.complex128)


def retrieval_overlap(c: torch.Tensor, gamma: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """q_ret = sum_jk conj(u_j) c_j Gamma_jk conj(c_k) u_k on each setting's support."""
    z = u.conj() * c
    return torch.einsum("bsj,bsjk,bsk->bs", z, gamma, z.conj()).real


def sector_readout(q_ret, population_cell, eff_cell, is_population, n_max: int) -> torch.Tensor:
    """q_n for n = 0..n_max, shape (B, S, n_max + 1)."""
    n = torch.arange(n_max + 1, dtype=q_ret.dtype, device=q_ret.device)
    q_multi = 1.0 - (1.0 - q_ret).unsqueeze(-1).pow(n)
    single = eff_cell * population_cell
    q_pop = 1.0 - (1.0 - single).unsqueeze(-1).pow(n)
    q_two = 2.0 * eff_cell * (1.0 - eff_cell) * population_cell.square() + 2.0 * single * (1.0 - population_cell)
    q_pop = torch.cat([q_pop[..., :2], q_two.unsqueeze(-1), q_pop[..., 3:]], dim=-1)
    return torch.where(is_population.unsqueeze(-1), q_pop, q_multi)


def four_cell_probabilities(
    c: torch.Tensor,
    gamma: torch.Tensor,
    eta: torch.Tensor,
    beta: torch.Tensor,
    theta: torch.Tensor,
    chi: torch.Tensor,
    eta_signal: torch.Tensor,
    b_signal: torch.Tensor,
    b_idler: torch.Tensor,
    n_orthogonal: torch.Tensor,
    phase_sign: int,
    n_max: int,
) -> dict:
    """Four-cell probabilities per row and setting.

    c (B, S, m) complex; gamma (B, S, m, m) complex, unit diagonal; eta (B, S, m);
    beta, theta (S, m); chi (B, S); eta_signal, b_signal, b_idler, n_orthogonal (B,).
    """
    u = analyzer_on_support(beta, theta, eta, phase_sign)
    q_ret = retrieval_overlap(c, gamma, u)
    active = beta > 0
    is_population = (active.sum(dim=-1) == 1).unsqueeze(0).expand_as(q_ret)
    cell = beta.argmax(dim=-1)
    index = cell.view(1, -1, 1).expand(c.shape[0], -1, 1)
    population_cell = c.abs().square().gather(-1, index).squeeze(-1)
    beta_cell = beta.gather(-1, cell.unsqueeze(-1)).squeeze(-1)
    eff_cell = beta_cell.square().unsqueeze(0) * eta.gather(-1, index).squeeze(-1)
    q_n = sector_readout(q_ret, population_cell, eff_cell, is_population, n_max)

    weights = thermal_sectors(chi, n_max)
    n = torch.arange(n_max + 1, dtype=chi.dtype, device=chi.device)
    orthogonal = 1.0 / (1.0 + n_orthogonal * eta_signal)
    no_signal = ((1.0 - b_signal) * orthogonal).view(-1, 1, 1) * (1.0 - eta_signal).view(-1, 1, 1).pow(n)
    no_idler = (1.0 - b_idler).view(-1, 1, 1) * (1.0 - q_n)
    p00 = (weights * no_signal * no_idler).sum(dim=-1)
    p10 = (weights * (1.0 - no_signal) * no_idler).sum(dim=-1)
    p01 = (weights * no_signal * (1.0 - no_idler)).sum(dim=-1)
    p11 = (weights * (1.0 - no_signal) * (1.0 - no_idler)).sum(dim=-1)
    pi = torch.stack([p00, p10, p01, p11], dim=-1)
    return {"pi": pi, "q_ret": q_ret, "is_population": is_population}


def matched_read(
    chi: float, eta_signal: float, eta_idler: float, b_signal: float, b_idler: float, n_max: int, n_orthogonal: float = 0.0
) -> dict:
    """Four cells of a read whose analyzer matches the heralded mode (q_ret = eta_idler).

    Returns the cells, the coincidence probability per trial and the full-model g_si.
    """
    one = lambda value: torch.tensor([value], dtype=torch.float64)
    # Equal two-cell state read by the matching two-cell analyzer: q_ret = eta_idler,
    # and the read stays on the multi-cell (collective-mode) branch.
    c2 = torch.full((1, 1, 2), 2 ** -0.5, dtype=torch.complex128)
    gamma2 = torch.ones(1, 1, 2, 2, dtype=torch.complex128)
    eta2 = torch.full((1, 1, 2), eta_idler, dtype=torch.float64)
    beta2 = torch.full((1, 2), 2 ** -0.5, dtype=torch.float64)
    theta2 = torch.zeros(1, 2, dtype=torch.float64)
    out = four_cell_probabilities(
        c2, gamma2, eta2, beta2, theta2, torch.tensor([[chi]], dtype=torch.float64),
        one(eta_signal), one(b_signal), one(b_idler), one(n_orthogonal), phase_sign=-1, n_max=n_max,
    )
    pi = out["pi"][0, 0]
    signal = float(pi[1] + pi[3])
    idler = float(pi[2] + pi[3])
    coincidence = float(pi[3])
    return {
        "pi": pi,
        "q_ret": float(out["q_ret"][0, 0]),
        "signal_rate": signal,
        "idler_rate": idler,
        "coincidence": coincidence,
        "g_si": coincidence / (signal * idler),
    }
