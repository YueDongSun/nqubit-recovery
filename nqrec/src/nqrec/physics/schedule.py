"""Readout schedules and the three budget laws.

A setting is a read-beam splitter: beta_j >= 0, sum_j beta_j^2 <= 1, plus stored
analyzer phases theta (radians, before phase_sign). Population settings address
one cell. Interference settings address a pair at a trine phase.
"""

from __future__ import annotations

import numpy as np

TRINE = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)


def _population(n_modes: int):
    beta = np.zeros((n_modes, n_modes), dtype=np.float64)
    theta = np.zeros((n_modes, n_modes), dtype=np.float64)
    for j in range(n_modes):
        beta[j, j] = 1.0
    return beta, theta


def _pairs(n_modes: int, edges):
    betas = []
    thetas = []
    for j, k in edges:
        for alpha in TRINE:
            beta = np.zeros(n_modes, dtype=np.float64)
            theta = np.zeros(n_modes, dtype=np.float64)
            beta[j] = 1.0 / np.sqrt(2.0)
            beta[k] = 1.0 / np.sqrt(2.0)
            theta[k] = alpha
            betas.append(beta)
            thetas.append(theta)
    if not betas:
        raise ValueError("pair schedule has no edges")
    return np.stack(betas), np.stack(thetas)


def _pack(name: str, n_modes: int, beta: np.ndarray, theta: np.ndarray) -> dict:
    power = np.sum(beta**2, axis=-1)
    if power.max() > 1.0 + 1e-12:
        raise AssertionError("schedule read power exceeds 1")
    return {
        "name": name,
        "n_modes": n_modes,
        "n_settings": int(beta.shape[0]),
        "beta": beta,
        "theta": theta,
    }


def population(n_modes: int) -> dict:
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    beta, theta = _population(n_modes)
    return _pack("population", n_modes, beta, theta)


def spanning_tree_edges(n_modes: int):
    if n_modes < 2:
        raise ValueError("a spanning tree needs n_modes >= 2")
    return [(i, i + 1) for i in range(n_modes - 1)]


def complete_edges(n_modes: int):
    if n_modes < 2:
        raise ValueError("a complete schedule needs n_modes >= 2")
    return [(j, k) for j in range(n_modes) for k in range(j + 1, n_modes)]


def spanning_tree_trine(n_modes: int) -> dict:
    """N population settings plus 3 trine settings on each tree edge. Count = 4N - 3."""
    pop_b, pop_t = _population(n_modes)
    pair_b, pair_t = _pairs(n_modes, spanning_tree_edges(n_modes))
    return _pack(
        "spanning_tree_trine",
        n_modes,
        np.concatenate([pop_b, pair_b], axis=0),
        np.concatenate([pop_t, pair_t], axis=0),
    )


def complete_kn(n_modes: int) -> dict:
    """Population plus trine on every pair. Count = N + 3 N (N-1) / 2."""
    pop_b, pop_t = _population(n_modes)
    pair_b, pair_t = _pairs(n_modes, complete_edges(n_modes))
    return _pack(
        "complete_kn",
        n_modes,
        np.concatenate([pop_b, pair_b], axis=0),
        np.concatenate([pop_t, pair_t], axis=0),
    )


def build_schedule(name: str, n_modes: int) -> dict:
    if name == "population":
        return population(n_modes)
    if name == "spanning_tree_trine":
        return spanning_tree_trine(n_modes)
    if name == "complete_kn":
        return complete_kn(n_modes)
    raise ValueError(f"unknown schedule {name!r}")


def n_parameters_gauge_reduced(n_modes: int) -> int:
    """N-1 log-amplitudes and N-1 phases."""
    if n_modes < 2:
        raise ValueError("gauge-reduced W parameters need n_modes >= 2")
    return 2 * n_modes - 2


def information_shots(n_settings: int, n_pairs: int, n_parameters: int, law: str, budget: float) -> float:
    """Expected trials per setting. Real-valued: F_s = s F_1 allows a real s.

    fixed_total: `budget` trials split evenly across settings.
    fixed_per_pair: each trine setting of a pair gets budget/3 trials, and each
    population setting gets the same number. `budget` is trials per pair.
    The Fisher ceiling does not use that trial count. H3 counts full rounds
    and splits k * 256 points evenly; per-pair there is k = scale * (N - 1)
    (see nqrec.physics.fisher.rounds_for_law).
    fixed_per_parameter: total trials = budget * n_parameters, split evenly.
    """
    if n_settings < 1:
        raise ValueError("n_settings must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if law == "fixed_total":
        return float(budget) / float(n_settings)
    if law == "fixed_per_pair":
        if n_pairs < 1:
            raise ValueError("fixed_per_pair needs at least one pair")
        if budget % 3 != 0:
            raise ValueError("fixed_per_pair budget must be divisible by 3 (one share per trine phase)")
        return float(budget) / 3.0
    if law == "fixed_per_parameter":
        if n_parameters < 1:
            raise ValueError("n_parameters must be positive")
        return float(budget) * float(n_parameters) / float(n_settings)
    raise ValueError(f"unknown budget law {law!r}")


def integer_shots(n_settings: int, n_pairs: int, n_parameters: int, law: str, budget: int) -> int:
    """Same laws, requiring an exact integer number of trials. No truncated remainder."""
    if int(budget) != budget:
        raise ValueError("integer_shots budget must be an integer")
    shots = information_shots(n_settings, n_pairs, n_parameters, law, int(budget))
    if abs(shots - round(shots)) > 1e-12:
        raise ValueError(
            f"{law} budget {budget} does not divide evenly into an integer shot count "
            f"({shots} trials per setting). Refusing to drop the remainder."
        )
    return round(shots)
