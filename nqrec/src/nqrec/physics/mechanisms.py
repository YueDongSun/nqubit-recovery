"""Noise mechanisms of the round-6 Monte Carlo (DECISIONS J5-J15).

Every mechanism is a departure from the nominal device (J5). One dataset (row)
carries one mechanism, chosen by its label; the nominal device itself is the
class `four_cell`. Per-cell quantities live on the full 16x16 grid (256 cells)
so the hardware and fixed-cell designs share them.

Coherence on a set of cells is the Schur product of
    E1  exp(-sigma1^2 (kappa_j - kappa_k)^2 / 2)
    E2  exp(-sigma2^2)                  for j != k
    E4  d_j d_k                         for j != k
which is positive semidefinite with unit diagonal. Static (E2) and drifting
(E3) phases are diagonal unitaries applied to the state, not to Gamma.
"""

from __future__ import annotations

import numpy as np
import torch

from nqrec.physics.efficiency import GRID_CELLS, GRID_SIDE


MECHANISMS = (
    "four_cell",
    "nb2",
    "thermal_dlcz",
    "thermal_multimode",
    "e1_laser_phase",
    "e2_aod_phase",
    "e3_slow_drift",
    "e4_cell_dephasing",
    "e5_amplitude",
    "e6_background",
)
# Matched-strength diagnostics outside the PI priors (J15).
CONTROLS = ("e1_strong", "e2_small")
ALL_CLASSES = MECHANISMS + CONTROLS





def _uniform(rng, bounds, size=None):
    low, high = float(bounds[0]), float(bounds[1])
    return rng.uniform(low, high, size=size)


def _unit(value, bounds) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    return (value - low) / (high - low)


def correlated_steps(rng: np.random.Generator, corr_length: float, n_steps: int) -> np.ndarray:
    """(n_steps, 256) Gaussian steps with covariance exp(-d^2 / (2 l^2)) on the grid.

    The Gaussian kernel separates over grid rows and columns, K = K_1 (x) K_1, so the
    step field is L_1 Z L_1^T with L_1 the 16x16 Cholesky factor. This is exact.
    """
    axis = np.arange(GRID_SIDE, dtype=np.float64)
    kernel_1d = np.exp(-((axis[:, None] - axis[None, :]) ** 2) / (2.0 * corr_length**2))
    # A smooth kernel is numerically rank deficient; 1e-10 keeps Cholesky defined and
    # changes the step variance by 2e-10.
    chol = np.linalg.cholesky(kernel_1d + 1e-10 * np.eye(GRID_SIDE))
    noise = rng.standard_normal((n_steps, GRID_SIDE, GRID_SIDE))
    return (chol @ noise @ chol.T).reshape(n_steps, GRID_CELLS)


def draw_mechanisms(labels: np.ndarray, n_rounds: int, nominal: dict, priors: dict, rng: np.random.Generator) -> dict:
    """Per-row mechanism parameters. `labels` index ALL_CLASSES.

    nominal: signal_rate, background (signal arm), background_idler. Returns numpy arrays, grid vectors
    of length 256.
    `strength` is the drawn magnitude scaled to [0, 1] on its prior range (0 for the null).
    """
    n_rows = labels.shape[0]
    out = {
        "label": labels.astype(np.int64),
        "strength": np.zeros(n_rows),
        "signal_rate": np.full(n_rows, float(nominal["signal_rate"])),
        "background": np.full(n_rows, float(nominal["background"])),
        "background_idler": np.full(n_rows, float(nominal["background_idler"])),
        "nb2_alpha": np.zeros(n_rows),
        "n_orthogonal": np.zeros(n_rows),
        "sigma1": np.zeros(n_rows),
        "kappa": np.zeros((n_rows, GRID_CELLS)),
        "sigma2": np.zeros(n_rows),
        "theta_static": np.zeros((n_rows, GRID_CELLS)),
        "drift": np.zeros((n_rows, n_rounds, GRID_CELLS)),
        "coherence_factor": np.ones((n_rows, GRID_CELLS)),
        "amplitude_lambda": np.zeros(n_rows),
        "tau_bar_us": np.full(n_rows, np.inf),
        "corr_length": np.zeros(n_rows),
        "step_sigma": np.zeros(n_rows),
    }
    storage = float(nominal["storage_time_us"])
    for row, label in enumerate(labels):
        name = ALL_CLASSES[int(label)]
        if name == "four_cell":
            continue
        if name == "nb2":
            alpha = _uniform(rng, priors["nb2_alpha"])
            out["nb2_alpha"][row] = alpha
            out["strength"][row] = _unit(alpha, priors["nb2_alpha"])
        elif name == "thermal_dlcz":
            rate = _uniform(rng, priors["thermal_dlcz_signal_rate"])
            out["signal_rate"][row] = rate
            out["strength"][row] = _unit(rate, priors["thermal_dlcz_signal_rate"])
        elif name == "thermal_multimode":
            n_o = _uniform(rng, priors["multimode_nbar"])
            out["n_orthogonal"][row] = n_o
            out["strength"][row] = _unit(n_o, priors["multimode_nbar"])
        elif name in ("e1_laser_phase", "e1_strong"):
            bounds = priors["e1_sigma"] if name == "e1_laser_phase" else priors["e1_strong_sigma"]
            sigma = _uniform(rng, bounds)
            out["sigma1"][row] = sigma
            out["kappa"][row] = rng.uniform(0.0, 1.0, size=GRID_CELLS)
            out["strength"][row] = _unit(sigma, bounds)
        elif name == "e2_aod_phase":
            sigma = _uniform(rng, priors["e2_jitter"])
            out["sigma2"][row] = sigma
            out["theta_static"][row] = rng.uniform(-np.pi, np.pi, size=GRID_CELLS)
            out["strength"][row] = _unit(sigma, priors["e2_jitter"])
        elif name in ("e3_slow_drift", "e2_small"):
            step = _uniform(rng, priors["e3_step"])
            length = _uniform(rng, priors["e3_corr_length"])
            out["step_sigma"][row] = step
            out["corr_length"][row] = length
            out["strength"][row] = _unit(step, priors["e3_step"])
            if name == "e3_slow_drift":
                steps = correlated_steps(rng, length, n_rounds)
                out["drift"][row] = step * np.cumsum(steps, axis=0)
            else:
                # Static offset with the round-averaged variance of the E3 walk,
                # step^2 (k + 1) / 2, so only time dependence separates it from E3 (J15).
                matched = step * np.sqrt((n_rounds + 1) / 2.0)
                out["theta_static"][row] = matched * correlated_steps(rng, length, 1)[0]
        elif name == "e4_cell_dephasing":
            tau_bar = _uniform(rng, priors["e4_tau_us"])
            tau = tau_bar * np.exp(float(priors["e4_tau_spread"]) * rng.standard_normal(GRID_CELLS))
            out["tau_bar_us"][row] = tau_bar
            out["coherence_factor"][row] = np.exp(-((storage / tau) ** 2))
            # Larger tau is weaker dephasing; strength 1 is the shortest tau.
            out["strength"][row] = 1.0 - _unit(tau_bar, priors["e4_tau_us"])
        elif name == "e5_amplitude":
            lam = _uniform(rng, priors["e5_lambda"])
            out["amplitude_lambda"][row] = lam
            out["strength"][row] = _unit(lam, priors["e5_lambda"])
        elif name == "e6_background":
            low, high = np.log(float(priors["e6_background"][0])), np.log(float(priors["e6_background"][1]))
            log_b = rng.uniform(low, high)
            out["background"][row] = np.exp(log_b)
            out["background_idler"][row] = np.exp(log_b)
            out["strength"][row] = (log_b - low) / (high - low)
        else:
            raise ValueError(f"unknown mechanism {name!r}")
    return out


def coherence_on_support(kappa: torch.Tensor, sigma1: torch.Tensor, sigma2: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Gamma (B, S, m, m) from per-cell vectors gathered on each setting's support.

    kappa, factor: (B, S, m); sigma1, sigma2: (B,).
    """
    s1 = sigma1.view(-1, 1, 1, 1)
    s2 = sigma2.view(-1, 1, 1, 1)
    diff = kappa.unsqueeze(-1) - kappa.unsqueeze(-2)
    common = torch.exp(-0.5 * s1.square() * diff.square())
    local = torch.exp(-s2.square()) * factor.unsqueeze(-1) * factor.unsqueeze(-2)
    m = kappa.shape[-1]
    eye = torch.eye(m, dtype=torch.bool, device=kappa.device)
    gamma = common * torch.where(eye, torch.ones_like(local), local)
    return gamma.to(torch.complex128)
