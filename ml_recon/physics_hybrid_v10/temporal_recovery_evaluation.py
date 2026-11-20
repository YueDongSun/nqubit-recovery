"""Evaluate registered eleven-window preparation-control arms and export paired evidence."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .temporal_recovery_environment import (
    CALIBRATION,
    WINDOW_READS,
    PreparationEnvironment,
    allocation,
    classical_policy,
    rollout,
)

TRAIN_SEEDS = (11031, 11047, 11071)
DEV_SEED = 22039
CONFIRM_SEED = 33049
BOOT_SEED = 44059
BOOT_DRAWS = 10000


def publish_atomic(source, dest):
    """Give short-lived Windows file readers time to release their sharing handles."""
    for attempt in range(12):
        try:
            os.replace(source, dest)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 11:
                raise
            time.sleep(0.2 * (attempt + 1))


def atomic_json(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    publish_atomic(temporary, path)


def one_step(policy):
    def decide(prefix):
        if len(prefix.observations) == 1:
            return policy(prefix)
        return torch.zeros_like(prefix.actions[0])

    return decide


def evidence_arrays(result):
    return {
        "fidelity": result["scores"].numpy(),
        "executed_actions": result["actions"].numpy(),
        "native_counts": torch.stack(
            [o.counts for o in result["observations"]], 1
        ).numpy(),
        "reads_per_setting": np.array([o.shots for o in result["observations"]]),
        "analyzers": result["observations"][0].analyzers.numpy(),
    }


def observations_summary(arrays):
    f, a = arrays["fidelity"], arrays["executed_actions"]
    gain = f[:, -1] - f[:, 0]
    active = np.max(np.abs(a), axis=(2, 3)) > 1e-8
    boot = np.random.default_rng(BOOT_SEED).integers(
        0, len(f), size=(BOOT_DRAWS, len(f))
    )

    def interval(x):
        means = np.concatenate(
            [x[boot[i : i + 200]].mean(axis=1) for i in range(0, BOOT_DRAWS, 200)]
        )
        return np.quantile(means, [0.025, 0.975], axis=0).T.tolist()

    delta, steps = f - f[:, :1], np.diff(f, axis=1)
    return {
        "episodes": len(f),
        "paid_reads": int(arrays["native_counts"][0].sum()),
        "acquisitions": f.shape[1],
        "decisions": a.shape[1],
        "terminal_fidelity": float(f[:, -1].mean()),
        "terminal_gain": float(gain.mean()),
        "terminal_gain_ci95": interval(gain),
        "terminal_harm": float((gain < -1e-10).mean()),
        "terminal_harm_ci95": interval(gain < -1e-10),
        "any_step_harm": float(np.any(steps < -1e-10, axis=1).mean()),
        "late_gain_after_round_five": float((f[:, -1] - f[:, 5]).mean()),
        "action_coverage": float(active.mean()),
        "round_active_fraction": active.mean(0).tolist(),
        "round_mean_gain": delta.mean(0).tolist(),
        "round_gain_ci95": interval(delta),
        "round_harm_fraction": (steps < -1e-10).mean(0).tolist(),
        "cumulative_read_cost": (
            arrays["reads_per_setting"].cumsum() * arrays["native_counts"].shape[2]
        ).tolist(),
    }


def paired_interval(values, family_size=1):
    values = np.asarray(values)
    boot = np.random.default_rng(BOOT_SEED).integers(
        0, len(values), size=(BOOT_DRAWS, len(values))
    )
    diff = values[boot].mean(1)
    q = 0.025 / family_size
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
        "family_ci95": np.quantile(diff, [q, 1 - q]).tolist(),
        "family_size": family_size,
    }


def run_classical(n, dest, storage):
    env = PreparationEnvironment(n)
    schedule = allocation(n)
    tuning = {}
    dev_terminal = {}

    def panel(policy, exposure=schedule, seed=DEV_SEED, count=64):
        return rollout(env, policy, episodes=count, seed=seed + n, schedule=exposure)

    for kind in ("analytic", "pid"):
        candidates = {}
        for gain in (0.25, 0.45, 0.70):
            scores = panel(classical_policy(env, kind, gain))["scores"]
            candidates[gain] = float((scores[:, -1] - scores[:, 0]).mean())
        tuning[kind] = max(candidates, key=candidates.get)
        dev_terminal[kind] = candidates[tuning[kind]]
    analytic = classical_policy(env, "analytic", tuning["analytic"])
    single = one_step(analytic)
    candidates = {}
    for fraction in (0.30, 0.55, 0.80):
        scores = panel(single, allocation(n, fraction))["scores"]
        candidates[fraction] = float((scores[:, -1] - scores[:, 0]).mean())
    fraction = max(candidates, key=candidates.get)
    arms = {
        "noop": (classical_policy(env, "noop"), schedule),
        "analytic": (analytic, schedule),
        "pid": (classical_policy(env, "pid", tuning["pid"]), schedule),
        "one_common": (single, schedule),
        "one_optimized": (single, allocation(n, fraction)),
    }
    summaries, terminals = {}, {}
    for name, (policy, exposure) in arms.items():
        arr = evidence_arrays(panel(policy, exposure, CONFIRM_SEED, 128))
        np.savez_compressed(storage / f"classical_n{n}_{name}.npz", **arr)
        summaries[name] = observations_summary(arr)
        terminals[name] = arr["fidelity"][:, -1]
    # Secondary fixed-window stop: own initial measurement and own final
    # diagnostic, generated by the same law at the intervention-dependent state.
    secondary_result = rollout(
        env,
        single,
        episodes=128,
        seed=CONFIRM_SEED + n,
        schedule=schedule[:2],
        secondary_one_step=True,
    )
    secondary = evidence_arrays(secondary_result)
    np.savez_compressed(storage / f"classical_n{n}_one_fixed_window.npz", **secondary)
    contrasts = {
        f"analytic_minus_{b}": paired_interval(terminals["analytic"] - terminals[b], 6)
        for b in ("pid", "one_common", "one_optimized")
    }
    return {
        "n": n,
        "trained_models": 0,
        "dev_gain": tuning,
        "secondary_fixed_window_one_step": {
            "paid_reads": int(secondary["native_counts"][0].sum()),
            "windows": 2,
            "mean_terminal_gain": float(
                np.mean(secondary["fidelity"][:, 1] - secondary["fidelity"][:, 0])
            ),
        },
        "dev_selected_classical": max(dev_terminal, key=dev_terminal.get),
        "dev_first_fraction": fraction,
        "arms": summaries,
        "paired_contrasts": contrasts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    args.results.mkdir(exist_ok=True, parents=True)
    args.artifacts.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    report = {
        "schema": "ten_step_classical_effective_v1",
        "law": "effective_single_event_v1",
        "calibration": CALIBRATION,
        "window_reads": WINDOW_READS,
        "k2_source_weight": 0,
        "action": "next_preparation",
        "endpoint": "conditional_K1_target_fidelity",
        "development_seed_plus_n": DEV_SEED,
        "confirmation_seed_plus_n": CONFIRM_SEED,
        "bootstrap_draws": BOOT_DRAWS,
        "bootstrap_seed": BOOT_SEED,
        "systems": {},
    }
    for n in (3, 4):
        report["systems"][str(n)] = run_classical(n, args.results, args.artifacts)
    panels = {}
    for source in sorted(args.artifacts.glob("classical_n*.npz")):
        with np.load(source) as data:
            panels[source.stem + "_fidelity"] = data["fidelity"]
            panels[source.stem + "_coverage"] = (
                np.abs(data["executed_actions"]).max(axis=(2, 3)) > 1e-8
            )
    np.savez_compressed(args.results / "classical_episode_panels.npz", **panels)
    report["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.results / "classical_results.json", report)


if __name__ == "__main__":
    main()
