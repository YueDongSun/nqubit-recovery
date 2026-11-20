"""Binary-herald W4 + W4 fusion under a finite, charged two-memory clock."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
from math import comb, factorial, sqrt
from pathlib import Path
import json
import argparse

import numpy as np
import torch

from .optical_fusion import fusion_output
from nqrec.physics.single_event_mc import (
    EVENT_LAW,
    calibrate_source_weights,
    diagonal_fock2_event_probability,
)

Q = 0.02
ETA_SIGNAL = 0.5
BACKGROUND = 0.0005
H2 = BACKGROUND + (1 - BACKGROUND) * (1 - (1 - ETA_SIGNAL) ** 2)
ETA_FUSION = 0.10
QUALITY = 0.70
HORIZONS = (32, 128)
DOUBLES = (0.0, 0.0004, 0.002)
LIFETIMES = (None, 50)
TOPOLOGIES = ("local", "cross")
N = 4
SEED = 2026112023


def module_components():
    vacuum = (0,) * N
    units = [tuple(int(k == i) for k in range(N)) for i in range(N)]
    local = [tuple(2 * v for v in row) for row in units]
    cross = [tuple(x + y for x, y in zip(a, b)) for a, b in combinations(units, 2)]

    def diagonal(states):
        return [(1 / len(states), {s: 1.0}) for s in states]

    return [
        diagonal([vacuum]),
        [(1.0, {s: 1 / sqrt(N) for s in units})],
        diagonal(units),
        diagonal(local),
        diagonal(cross),
    ]


def splitter(a, b):
    """Amplitude from input |a,b> to |k,a+b-k> at a balanced splitter."""
    total = a + b
    for k in range(total + 1):
        coeff = sum(
            comb(a, u) * comb(b, k - u) * (-1) ** (b - k + u)
            for u in range(max(0, k - b), min(a, k) + 1)
        )
        yield (
            k,
            coeff
            * sqrt(
                factorial(k)
                * factorial(total - k)
                / (2**total * factorial(a) * factorial(b))
            ),
        )


def optical_moments(state_a, state_b):
    """Unnormalized probabilities of click, click-and-single, click-and-W6."""
    waves = defaultdict(lambda: defaultdict(float))
    for (a, amp_a), (b, amp_b) in product(state_a.items(), state_b.items()):
        for k, bs in splitter(a[0], b[0]):
            waves[k, a[0] + b[0] - k][a[1:] + b[1:]] += amp_a * amp_b * bs
    moments = np.zeros(3)
    norm = 0.0
    for (k, _), wave in waves.items():
        mass = sum(x * x for x in wave.values())
        single = [x for fock, x in wave.items() if sum(fock) == 1]
        norm += mass
        click = 1 - (1 - ETA_FUSION) ** k
        moments += click * np.array(
            [mass, sum(x * x for x in single), sum(single) ** 2 / (2 * N - 2)]
        )
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-13):
        raise ValueError("Beam-splitter norm failed")
    return moments


def optical_kernel():
    components = module_components()
    kernel = np.empty((5, 5, 3))
    for i, left in enumerate(components):
        for j, right in enumerate(components):
            kernel[i, j] = sum(
                wa * wb * optical_moments(a, b)
                for (wa, a), (wb, b) in product(left, right)
            )
    rho = np.ones((N, N)) / N
    for survival in (0.0, 0.3, 1.0):
        ref = fusion_output(rho, rho, ETA_FUSION, survival_a=survival)
        target = ref.single_block.sum().real / (2 * N - 2)
        np.testing.assert_allclose(
            (1 - survival) * kernel[0, 1] + survival * kernel[1, 1],
            [ref.click_probability, ref.single_probability, target],
            atol=1e-14,
        )
    # With the other port in vacuum, the entire multi-mode calculation reduces
    # to the public owner's K2 single-output marginal. Detector loss is included
    # in the transfer amplitude, so any two transmitted photons produce one TTL.
    transfer = torch.zeros(N, dtype=torch.float64)
    transfer[0] = sqrt(ETA_FUSION / 2.0)
    local = torch.ones_like(transfer) / N
    cross = torch.triu(torch.ones(N, N, dtype=torch.float64), diagonal=1) / comb(N, 2)
    for index, lw, cw in ((3, local, 0 * cross), (4, 0 * local, cross)):
        reduced = diagonal_fock2_event_probability(
            transfer,
            lw,
            cw,
            two_photon_single_event_acceptance=transfer.new_tensor(1.0),
            background=transfer.new_tensor(0.0),
        )
        np.testing.assert_allclose(kernel[index, 0, 0], float(reduced), atol=1e-14)
    return kernel


def herald_weights(p2):
    s, eta, bg, p, h2 = [
        torch.tensor(v, dtype=torch.float64)
        for v in (Q, ETA_SIGNAL, BACKGROUND, p2, H2)
    ]
    source = calibrate_source_weights(
        s, eta, bg, double_probability=p, double_signal_event_probability=h2
    )
    heralded = source["joint"][:, 1].numpy() / Q
    np.testing.assert_allclose(heralded.sum(), 1.0)
    return heralded


def stored_components(heralded, survival, topology):
    """Uniform independent pure loss; a surviving partner of diagonal K2 is I/N."""
    w0, w1, w2 = heralded
    s = np.asarray(survival)
    result = np.zeros(s.shape + (5,))
    result[..., 0] = w0 + w1 * (1 - s) + w2 * (1 - s) ** 2
    result[..., 1] = w1 * s
    result[..., 2] = 2 * w2 * s * (1 - s)
    result[..., 3 if topology == "local" else 4] = w2 * s * s
    return result


def arrival_distribution(horizon, strategy):
    age = np.arange(horizon)
    pair_probability = np.zeros(horizon)
    if strategy == "synchronous":
        pair_probability[0] = 1 - (1 - Q * Q) ** horizon
        paid = 2 * pair_probability[0] / (Q * Q)
    else:
        r = 1 - Q
        pair_probability = (
            Q**2 * r**age * (1 - r ** (2 * (horizon - age))) / (1 - r * r)
        )
        pair_probability[1:] *= 2
        direct = np.zeros(horizon)
        geom = Q * r**age
        for a, b in product(range(horizon), repeat=2):
            direct[abs(a - b)] += geom[a] * geom[b]
        np.testing.assert_allclose(direct, pair_probability, rtol=1.0e-12)
        paid = 2 * (1 - r**horizon) / Q
    return pair_probability, paid


def measure(kernel, p2, topology, lifetime, horizon, strategy):
    posterior = herald_weights(p2)
    pair, paid = arrival_distribution(horizon, strategy)
    age = np.arange(horizon)
    s = np.ones(horizon) if lifetime is None else np.exp(-age / lifetime)
    old = stored_components(posterior, s, topology)
    fresh = stored_components(posterior, 1.0, topology)
    moments = np.einsum("ai,ijm,j->am", old, kernel, fresh)
    clicks, single, overlap = moments.T
    quality_pass = overlap >= QUALITY * clicks
    per_block = pair @ moments
    qualified = pair @ (quality_pass * clicks)
    return (
        dict(
            horizon=horizon,
            policy=strategy,
            lifetime=lifetime,
            p2=p2,
            topology=topology,
            posterior_k2=float(posterior[2]),
            pair_probability=pair.sum(),
            expected_paid_writes=paid,
            expected_paid_reads=paid,
            mean_storage_age_given_pair=pair @ age / pair.sum(),
            mean_older_survival_given_pair=pair @ s / pair.sum(),
            click_probability_per_episode=per_block[0],
            single_weight_given_click=per_block[1] / per_block[0],
            w_fidelity_given_single_click=per_block[2] / per_block[1],
            full_w_overlap_given_click=per_block[2] / per_block[0],
            click_per_million_writes=1e6 * per_block[0] / paid,
            w_weight_per_million_writes=1e6 * per_block[2] / paid,
            qualified_click_per_million_writes=1e6 * qualified / paid,
        ),
        clicks,
        quality_pass,
    )


def sampled_check(row, clicks, quality_pass):
    """Independent geometric arrivals and actual binary outputs, with variable cost."""
    rng = np.random.default_rng(SEED)
    h = row["horizon"]
    totals = []
    for _ in range(32):
        shape = 16384
        if row["policy"] == "synchronous":
            t = rng.geometric(Q**2, shape)
            mask = t <= h
            age = np.zeros(shape, dtype=int)
            paid = 2 * np.minimum(t, h).sum()
        else:
            t = rng.geometric(Q, (2, shape))
            mask = np.max(t, axis=0) <= h
            age = np.where(mask, np.abs(t[0] - t[1]), 0)
            paid = np.minimum(t, h).sum()
        clicked = mask & (rng.random(shape) < clicks[age])
        totals.append(
            [int(paid), int(clicked.sum()), int((clicked & quality_pass[age]).sum())]
        )
    totals = np.array(totals)
    rates = 1e6 * totals[:, 1:] / totals[:, :1]
    reference = np.array(
        [row["click_per_million_writes"], row["qualified_click_per_million_writes"]]
    )
    se = rates.std(axis=0, ddof=1) / sqrt(len(rates))
    if np.any(abs(rates.mean(axis=0) - reference) > 5 * se + 1e-10):
        raise ValueError("Sampler disagrees with exact clock or optical law")
    return dict(
        policy=row["policy"],
        episodes_per_batch=16384,
        independent_batches=32,
        totals_paid_click_qualified=totals.sum(axis=0).tolist(),
        mean_batch_rates=rates.mean(axis=0).tolist(),
        batch_mean_se=se.tolist(),
        expected_rates=reference.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    kernel = optical_kernel()
    rows, checks = [], []
    for horizon, lifetime, p2, topology, strategy in product(
        HORIZONS, LIFETIMES, DOUBLES, TOPOLOGIES, ("synchronous", "buffered")
    ):
        if p2 == 0 and topology == "cross":
            continue
        row, clicks, passed = measure(kernel, p2, topology, lifetime, horizon, strategy)
        rows.append(row)
        if (horizon, lifetime, p2, topology) == (128, 50, 0.0004, "cross"):
            checks.append(sampled_check(row, clicks, passed))
    report = dict(
        schema="two_w4_threshold_fusion_v1",
        source_law=EVENT_LAW,
        signal_including_background=Q,
        binary_click_given_two_incident=1 - (1 - ETA_FUSION) ** 2,
        exact_conditions=len(rows),
        statistical_unit="independent two-module episode",
        qualification="model-conditioned age-class mean full W6 overlap >= 0.70",
        capacity=dict(
            available_modes=256,
            reference_groups=1,
            addressing_cycles=1,
            group_sizes={
                str(k): dict(
                    group_width=k,
                    disjoint_groups=256 // k,
                    unused_modes=256 % k,
                    continuous_mode_ratio=256 / k,
                )
                for k in (3, 4)
            },
            interpretation="available parallel groups; expected output ratio if all groups can run concurrently at equal per-group success and exposure",
        ),
        sampling_checks=checks,
        rows=rows,
    )
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"{len(rows)} conditions; {len(checks)} independent sampling checks.")


if __name__ == "__main__":
    main()
