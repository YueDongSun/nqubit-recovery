"""Paired simultaneous decisions for the new 3Q/4Q preparation-control study."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta

# Twelve independently registered cells: two N, three environments, and two upstreams.
# In each cell: four boundary quantities for each fit and three state-by-horizon contrasts.
FAMILY_SIZE = 2 * 3 * 2 * (4 * 2 + 3)
HARM_TOL = 1e-10


def select_horizon(rows):
    """DEV-only choice from equal-cost horizons; keep the best point with a 0.005 tolerance."""
    one = rows[1]
    passing = [
        k
        for k, row in rows.items()
        if k > 1
        and row["mean"] - one["mean"] > 0.01
        and row["gain_over_noop"] > 0
        and row["harm_vs_noop"] < 0.10
    ]
    if not passing:
        return None
    best = max(rows[k]["mean"] for k in passing)
    return min(k for k in passing if rows[k]["mean"] >= best - 0.005)


def control_differences(values, chosen):
    """Construct per-fit-seed, per-episode contrasts without pooling independent states as seeds."""
    terms, harms = {}, {}
    for fit in ("ordered", "learner"):

        def end(source, k, fit=fit):
            return values[f"{fit}_{source}_k{k}"][..., -1]

        k = chosen[fit] if chosen[fit] is not None else 10
        terminal = end("inferred", k)
        terms[f"{fit}_over_one"] = terminal - end("inferred", 1)
        terms[f"{fit}_over_noop"] = terminal - values["noop"][..., -1]
        terms[f"{fit}_over_analytic"] = terminal - values["analytic"][..., -1]
        harms[fit] = terminal < values["noop"][..., -1] - HARM_TOL
        terms[f"{fit}_state_by_horizon"] = (
            end("inferred", 10) - end("inferred", 1) - end("command", 10) + end("command", 1)
        )
    terms["learner_minus_teacher_state_by_horizon"] = (
        terms["learner_state_by_horizon"] - terms["ordered_state_by_horizon"]
    )
    return terms, harms


def simultaneous_results(values, chosen, *, bootstrap_draws=10000):
    """Keep each episode paired across all arms and fixed fitted seeds; bound absolute harm."""
    terms, harms = control_differences(values, chosen)
    keys = list(terms)
    data = np.stack([terms[k].mean(axis=0) for k in keys], axis=1)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("nonfinite paired confirmation scores")
    tail = 0.025 / FAMILY_SIZE
    rng = np.random.default_rng(581307)
    boot = []
    for offset in range(0, bootstrap_draws, 128):
        size = min(128, bootstrap_draws - offset)
        draws = rng.integers(len(data), size=(size, len(data)))
        boot.append(data[draws].mean(axis=1))
    limits = np.quantile(np.concatenate(boot), [tail, 1 - tail], axis=0)
    contrasts = {
        k: {
            "mean": float(data[:, j].mean()),
            "simultaneous_ci95": limits[:, j].tolist(),
            "per_seed_mean": terms[k].mean(axis=1).tolist(),
        }
        for j, k in enumerate(keys)
    }
    decisions = {}
    for fit in ("ordered", "learner"):
        k = chosen[fit]
        nr, ne = harms[fit].shape
        counts = harms[fit].sum(axis=1).astype(int)
        # Familywise exact per-seed binomial intervals remain valid with shared episodes;
        # averaging bounds covers the mean harm risk over these fixed seeds.
        lo = [float(beta.ppf(tail / nr, x, ne - x + 1)) if x else 0.0 for x in counts]
        hi = [float(beta.ppf(1 - tail / nr, x + 1, ne - x)) if x < ne else 1.0 for x in counts]
        low, high = float(np.mean(lo)), float(np.mean(hi))
        over_noop = contrasts[f"{fit}_over_noop"]["simultaneous_ci95"]
        over_one = contrasts[f"{fit}_over_one"]["simultaneous_ci95"]
        if over_one[0] > 0.01 and over_noop[0] > 0 and high < 0.10:
            status = "usable_increment_over_one"
        elif over_noop[1] < 0 or low > 0.10:
            status = "harmful"
        else:
            status = "unresolved"
        decisions[fit] = {
            "selected_k": k,
            "tested_k": 10 if k is None else k,
            "status": status,
            "benefit_over_noop": over_noop[0] > 0,
            "harm_vs_noop": float(harms[fit].mean()),
            "harm_counts_per_seed": counts.tolist(),
            "episodes_per_seed": ne,
            "harm_simultaneous_ci95": [low, high],
        }
    return {
        "family_size": FAMILY_SIZE,
        "bootstrap_draws": bootstrap_draws,
        "resampling_unit": "independent episode, averaged across fixed fitted seeds",
        "gain_intervals": "two-sided Bonferroni percentile (both decision directions protected)",
        "harm_intervals": "mean of simultaneous exact per-seed Clopper-Pearson intervals",
        "contrasts": contrasts,
        "decisions": decisions,
    }
