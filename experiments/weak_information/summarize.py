"""Compute paired exploratory READ-budget effects from sealed, fixed-policy trajectory files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .score import SLOTS, digest, read


N_DRAWS = 10_000
FAMILY = 38
HARM_TOL = 1e-10


def bounds(values, tail=0.025):
    return np.quantile(values, [tail, 1 - tail], axis=0).tolist()


def exact(count, total, tail):
    if total == 0:
        return None
    return [
        float(beta.ppf(tail, count, total - count + 1)) if count else 0.0,
        float(beta.ppf(1 - tail, count + 1, total - count)) if count < total else 1.0,
    ]


def bootstrap_weights(size, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.multinomial(size, np.full(size, 1 / size), size=N_DRAWS).astype(float)
        / size
    )


def point_record(x, w, tail=0.025):
    if x.ndim != 1:
        raise ValueError("expected paired episode scores")
    return dict(mean=float(x.mean()), interval=bounds(w @ x, tail))


def action_description(f, actions, noop, w):
    active = np.abs(actions).max(axis=(-2, -1)) > 1e-8
    acted = active.any(axis=2)
    diff = f[:, :, -1] - noop[None, :, -1]
    harm = diff < -HARM_TOL
    per_seed_acted = acted.sum(axis=1).tolist()
    per_seed_harmed = (acted & harm).sum(axis=1).tolist()
    p = w @ acted.mean(axis=0)
    if np.any(acted):
        point = float(np.sum(acted & harm) / np.sum(acted))
        g = float((acted * diff).sum() / acted.sum())
        good = p > 0
        interval = (
            bounds((w @ (acted & harm).mean(axis=0))[good] / p[good])
            if np.any(good)
            else None
        )
        gi = (
            bounds((w @ (acted * diff).mean(axis=0))[good] / p[good])
            if np.any(good)
            else None
        )
    else:
        point = g = interval = gi = None
    return dict(
        trajectory_count=noop.shape[0],
        fixed_seed_count=actions.shape[0],
        terminal_conditional_fidelity=point_record(f[:, :, -1].mean(axis=0), w),
        terminal_gain_vs_noop=point_record(diff.mean(axis=0), w),
        harm_vs_noop=point_record(harm.mean(axis=0), w),
        harm_counts_per_policy=harm.sum(axis=1).tolist(),
        exact_harm_intervals=[
            exact(int(n), noop.shape[0], 0.025) for n in harm.sum(axis=1)
        ],
        episode_execution=point_record(acted.mean(axis=0), w),
        active_slots=active.sum(axis=(1, 2)).tolist(),
        slots_per_policy=int(active.shape[1] * active.shape[2]),
        slot_execution=point_record(active.mean(axis=(0, 2)), w),
        executed_episodes_per_policy=per_seed_acted,
        harmed_executed_episodes=per_seed_harmed,
        harm_given_execution=dict(mean=point, interval=interval),
        gain_given_execution=dict(mean=g, interval=gi),
        conditional_exact_harm_intervals=[
            exact(k, n, 0.025) for k, n in zip(per_seed_harmed, per_seed_acted)
        ],
        round_mean_fidelity=f.mean(axis=(0, 1)).tolist(),
    )


def geometry_summary(n):
    g = read(BASE / f"geometry_n{n}.json")
    strong = (
        np.asarray(g["per_read"]["strong"]["fisher_per_read"]) * g["reads"]["strong_1x"]
    )
    weak = (
        np.asarray(g["per_read"]["weak"]["fisher_per_read"])
        * g["reads"]["weak_trace_matched"]
    )
    eig, basis = np.linalg.eigh(strong)
    mask = eig > eig.max() * 1e-10
    whitening = basis[:, mask] / np.sqrt(eig[mask])[None, :]
    ratio = whitening.T @ weak @ whitening
    g["matched_weak_over_strong_directional_information_on_identifiable_subspace"] = (
        np.linalg.eigvalsh((ratio + ratio.T) / 2).tolist()
    )
    g["matched_weak_in_strong_gauge_subspace_norm"] = float(
        np.linalg.norm(weak @ basis[:, ~mask])
    )
    return g


def size_result(n):
    reg = read(BASE / "registration.json")
    seeds = reg["seeds"]
    heldout = reg["roles"]["heldout"]
    w = bootstrap_weights(heldout, 721401 + n)
    tail = 0.025 / FAMILY
    terms = {}
    cell_rows = {}
    horizons = {}
    shared_ids = shared_initial = None
    for slot in SLOTS:
        root = BASE / f"n{n}" / slot
        receipt = read(root / "result.json")
        dev = read(root / "dev.json")
        if receipt["dev_lock"] != digest(BASE / "frozen_dev.json") or receipt[
            "panel_sha256"
        ] != digest(root / "episode_panels.npz"):
            raise ValueError("selection or returned episode payload differs")
        with np.load(root / "episode_panels.npz", allow_pickle=False) as p:
            ids = p["ids"]
            noop = p["noop_fidelity"]
            total = receipt["cell"]["reads"]
            operation_count = 3 * n - 2
            if (
                len(np.unique(ids)) != heldout
                or list(p["seeds"]) != seeds
                or p["reads"].item() != total
                or noop.shape != (heldout, 11)
            ):
                raise ValueError("incorrect READ or trajectory denominator")
            if shared_ids is None:
                shared_ids, shared_initial = ids.copy(), noop[:, 0].copy()
            if not np.array_equal(ids, shared_ids) or not np.array_equal(
                noop[:, 0], shared_initial
            ):
                raise ValueError("latent states differ across budgets")
            summary = {}
            ends = {}
            for arm in ("noop", "analytic1", "analytic10", "learned1", "learned10"):
                keys = (
                    [f"s{s}_{arm}" for s in seeds]
                    if arm.startswith("learned")
                    else [arm]
                )
                f = np.stack([p[k + "_fidelity"] for k in keys])
                a = np.stack([p[k + "_executed"] for k in keys])
                for k in keys:
                    if (
                        int(p[k + "_reads_per_setting"].sum()) * operation_count
                        != total
                    ):
                        raise ValueError("nominal and paid read totals differ")
                if not np.all(f[:, :, 0] == shared_initial[None, :]):
                    raise ValueError("policy initial fidelity differs")
                if (
                    a.shape != (len(keys), heldout, 10, n, 2)
                    or not np.isfinite(f).all()
                ):
                    raise ValueError("invalid action shape or fidelity")
                summary[arm] = action_description(f, a, noop, w)
                ends[arm] = f[:, :, -1]
            ten = ends["learned10"]
            horizon = ten - ends["learned1"]
            horizons[slot] = horizon
            terms[(slot, "ten_minus_one")] = horizon.mean(axis=0)
            terms[(slot, "ten_minus_noop")] = (ten - ends["noop"]).mean(axis=0)
            terms[(slot, "ten_minus_analytic")] = (ten - ends["analytic10"]).mean(
                axis=0
            )
            harm = (ten < ends["noop"] - HARM_TOL).sum(axis=1)
            cp = np.asarray([exact(int(k), heldout, tail / len(seeds)) for k in harm])
            cell_rows[slot] = dict(
                reads=total,
                efficiency=receipt["cell"]["efficiency"],
                dev_seconds=dev["elapsed_s"],
                heldout_seconds=receipt["elapsed_s"],
                optimized_one=dev["single_fractions"],
                analytic_damping=dev["analytic_damping"],
                analytic_one_fraction=dev["analytic_fraction"],
                policies=summary,
                ten_harm_vs_noop=dict(
                    mean=float(harm.mean() / heldout),
                    simultaneous_ci95=cp.mean(axis=0).tolist(),
                    harm_counts=harm.tolist(),
                    episodes_per_seed=heldout,
                ),
            )
    for hi, lo in [
        ("weak_4x", "weak_1x"),
        ("weak_trace_matched", "weak_1x"),
        ("weak_trace_matched", "strong_1x"),
    ]:
        terms[("cross", hi + "_minus_" + lo)] = (horizons[hi] - horizons[lo]).mean(
            axis=0
        )
    cross = {}
    for (slot, term), row in terms.items():
        rec = point_record(row, w, tail)
        rec["simultaneous_ci95"] = rec.pop("interval")
        if slot == "cross":
            cross[term] = rec
        else:
            cell_rows[slot][term] = rec
    for slot, record in cell_rows.items():
        usable = (
            record["ten_minus_one"]["simultaneous_ci95"][0] > 0.01
            and record["ten_minus_noop"]["simultaneous_ci95"][0] > 0
            and record["ten_harm_vs_noop"]["simultaneous_ci95"][1] < 0.10
        )
        record["exceeds_registered_benefit_and_safety"] = bool(usable)
    primary = (
        cell_rows["weak_trace_matched"]["exceeds_registered_benefit_and_safety"]
        and cross["weak_trace_matched_minus_weak_1x"]["simultaneous_ci95"][0] > 0
    )
    return dict(
        cells=cell_rows,
        paired_cross_budget_horizon_effect=cross,
        weak_information_budget_recovery_supported=bool(primary),
        independent_heldout_episodes=heldout,
        independent_dev_episodes=reg["roles"]["dev"],
        fixed_training_seeds=seeds,
    )


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    BASE = cli.parse_args().root
    reg = read(BASE / "registration.json")
    assert reg["statistical_family"] == FAMILY and reg["bootstrap_draws"] == N_DRAWS
    out = dict(
        study=reg["study"],
        evidence=reg["status"],
        source=reg["source"],
        registration=digest(BASE / "registration.json"),
        sealed_dev=digest(BASE / "frozen_dev.json"),
        geometry={str(n): geometry_summary(n) for n in reg["ns"]},
        systems={str(n): size_result(n) for n in reg["ns"]},
        uncertainty="simultaneous Bonferroni 95% for 38 registered quantities; descriptive policy intervals pointwise 95% resampling paired episodes",
        distinction="equal trace at the public uniform target need not match eigenvalue shape, actual trajectory information, or economic cost",
    )
    (BASE / "exploratory_results.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n"
    )
