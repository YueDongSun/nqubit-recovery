"""Screen, freeze, and independently confirm the 3Q/4Q Fig. 5 synthetic control design.

Run screen for each N without model weights, freeze for each B0/B2 and N, and
then confirm each of the three predeclared conditions. All stages use disjoint episode
namespaces. Screening uses analytic feedback; learned inference is limited to
the three predeclared conditions. Nonselected grid cells have only development evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .temporal_operating_metrics import HARM_TOL, select_horizon, simultaneous_results
from .temporal_recovery_environment import (
    AMPLITUDE_BOUND,
    CALIBRATION,
    PHASE_BOUND,
    PreparationEnvironment,
    classical_policy,
    rollout,
)
from .temporal_recovery_evaluation import TRAIN_SEEDS, atomic_json, publish_atomic
from .temporal_recovery_long_training import checkpoint_digest
from .temporal_recovery_shared_evaluation import PublicCommandState
from .temporal_recovery_shared_export import load_shared_route
from .temporal_recovery_shared_qualification import campaign_seed_base
from .temporal_recovery_training import callback
from .temporal_state_attribution import FRACTIONS, acquisition_schedule, limited_policy

READS = (38500, 77000, 154000)
EFFICIENCIES = (0.04, 0.12, 0.35)
SPREADS = (0.5, 1.0, 2.0)
K_GRID = (0, 1, 2, 4, 6, 8, 10)
FITS = ("ordered", "learner")
SLOTS = ("nominal", "weak_static", "strong_drift")
EXPLORATORY_SLOTS = ("nominal", "boundary_static", "stress_drift")
DEV_SIZE = 128
CONFIRM_SIZE = 512


def grid(n):
    for i, (reads, eta, drift, spread) in enumerate(
        itertools.product(READS, EFFICIENCIES, (False, True), SPREADS)
    ):
        yield {
            "n": n,
            "cell_id": i,
            "reads": reads,
            "efficiency": eta,
            "drift": drift,
            "spread": spread,
        }


def environment(cell):
    params = list(CALIBRATION)
    params[3] = cell["efficiency"]
    return PreparationEnvironment(cell["n"], calibration=params)


def seed_prefix(namespace, cell, role):
    roles = {"screen": 4, "freeze": 5, "confirm": 6}
    return (
        campaign_seed_base(namespace)
        + roles[role] * 10**12
        + cell["n"] * 10**9
        + cell["cell_id"] * 10**5
    )


def trajectories(env, cell, policy, k, count, prefix, fraction=None, multiplier=1.0):
    """All eleven actual preparations and READ costs, including natural drift at skipped slots."""
    shots = acquisition_schedule(env.n, cell["reads"], fraction if k == 1 else None)
    decide = limited_policy(policy, k)
    parts = {key: [] for key in ("fidelity", "executed", "requested")}
    for i in range(0, count, 64):
        data = rollout(
            env,
            decide,
            episodes=min(64, count - i),
            seed=prefix + i,
            schedule=shots,
            initial_scale=cell["spread"],
            constant_velocity=cell["drift"],
            action_scale=multiplier,
            record_proposals=True,
        )
        costs = torch.stack([o.counts.sum((1, 2)) for o in data["observations"]], dim=1)
        expected = torch.tensor(shots * len(env.analyzers))
        if not torch.equal(costs, expected.expand_as(costs)):
            raise ValueError("realized READs do not match the registered eleven-window allocation")
        for dest, key in (
            ("fidelity", "scores"),
            ("executed", "actions"),
            ("requested", "proposals"),
        ):
            parts[dest].append(data[key].detach().cpu().numpy())
    return {**{k: np.concatenate(v) for k, v in parts.items()}, "reads_per_setting": shots.copy()}


def arm_metrics(panel, noop):
    f = panel["fidelity"]
    actions = panel["executed"]
    active = np.max(np.abs(actions), axis=(-1, -2)) > 1e-8
    if not np.array_equal(f[:, 0], noop[:, 0]):
        raise ValueError("arms do not share the initial episode panel")
    d = f[:, -1] - noop[:, -1]
    initial_gain = f[:, -1] - f[:, 0]
    # Equal-weight observed lower tail, including fractional occupancy at the percentile boundary.
    tail_n = 0.10 * len(d)
    sorted_gain = np.sort(d)
    i = int(tail_n)
    tail = (sorted_gain[:i].sum() + (tail_n - i) * sorted_gain[i]) / tail_n
    scale = np.array([AMPLITUDE_BOUND, PHASE_BOUND])
    return {
        "mean": float(f[:, -1].mean()),
        "gain_over_noop": float(d.mean()),
        "harm_vs_noop": float(np.mean(d < -HARM_TOL)),
        "harm_vs_initial": float(np.mean(initial_gain < -HARM_TOL)),
        "harm_any_step": float(np.any(np.diff(f, axis=1) < -HARM_TOL, axis=1).mean()),
        "worst_decile_gain_over_noop": float(tail),
        "round_mean_fidelity": f.mean(0).tolist(),
        "round_mean_increment": np.diff(f, axis=1).mean(0).tolist(),
        "late_gain": float((f[:, -1] - f[:, 5]).mean()),
        "n_episodes": len(f),
        "n_active_slots": int(active.sum()),
        "n_slots": active.size,
        "action_coverage": float(active.mean()),
        "cumulative_action_norm": float(
            np.linalg.norm(actions.reshape(len(f), 10, -1), axis=2).sum(1).mean()
        ),
        "radial_boundary_fraction": float(
            (np.abs(actions / scale).max(axis=(2, 3)) >= 1.0 - 1e-7).mean()
        ),
    }


def optimal_single(env, cell, fn, count, prefix):
    means = [
        trajectories(env, cell, fn, 1, count, prefix, r)["fidelity"][:, -1].mean()
        for r in FRACTIONS
    ]
    return FRACTIONS[int(np.argmax(means))]


def screen_cell(cell, namespace, count):
    env = environment(cell)
    prefix = seed_prefix(namespace, cell, "screen")
    zero = classical_policy(env, "noop")
    noop = trajectories(env, cell, zero, 0, count, prefix)["fidelity"]
    gains = (0.25, 0.45, 0.70)
    trials = [
        trajectories(env, cell, classical_policy(env, "analytic", g), 10, count, prefix)
        for g in gains
    ]
    index = int(np.argmax([r["fidelity"][:, -1].mean() for r in trials]))
    analytic = classical_policy(env, "analytic", gains[index])
    fraction = optimal_single(env, cell, analytic, count, prefix)
    estimates = {
        k: arm_metrics(
            trials[index]
            if k == 10
            else trajectories(env, cell, analytic, k, count, prefix, fraction),
            noop,
        )
        for k in K_GRID
    }
    k = select_horizon(estimates)
    return {
        "cell": cell,
        "analytic_damping": gains[index],
        "analytic_one_fraction": fraction,
        "chosen_k": k,
        "reference_information_trace": float(env.linear_fisher[0].trace())
        * int(acquisition_schedule(env.n, cell["reads"])[0]),
        "analytic": estimates,
    }


def choose_cells(rows):
    """Exploratory analytic screen extremes, excluded from the registered confirmation selection."""
    nominal = next(
        row
        for row in rows
        if (
            row["cell"]["reads"],
            row["cell"]["efficiency"],
            row["cell"]["drift"],
            row["cell"]["spread"],
        )
        == (77000, 0.35, False, 1.0)
    )
    static = [r for r in rows if not r["cell"]["drift"] and r is not nominal]

    def tie(row):
        c = row["cell"]
        return c["reads"], c["efficiency"], c["spread"]

    passing = [r for r in static if r["chosen_k"] is not None]
    if passing:
        boundary = min(passing, key=lambda r: (r["reference_information_trace"], tie(r)))
    else:

        def gap(row):
            a = row["analytic"]
            return min(
                max(
                    0.01 - (a[k]["mean"] - a[1]["mean"]),
                    -a[k]["gain_over_noop"],
                    a[k]["harm_vs_noop"] - 0.10,
                )
                for k in K_GRID
                if k > 1
            )

        boundary = min(static, key=lambda r: (gap(r), tie(r)))
    stress = min(
        (r for r in rows if r["cell"]["drift"]),
        key=lambda r: (-r["analytic"][10]["harm_vs_noop"], tie(r)),
    )
    return dict(zip(EXPLORATORY_SLOTS, (nominal, boundary, stress), strict=True))


def registered_cells(rows):
    """Select the three approved fixed cells by design factors alone, regardless of DEV outcomes."""
    targets = {"nominal": (0.35, False), "weak_static": (0.04, False), "strong_drift": (0.35, True)}
    fixed = {}
    for name, (efficiency, drift) in targets.items():
        matches = [
            row
            for row in rows
            if row["cell"]["reads"] == 77000
            and row["cell"]["spread"] == 1.0
            and row["cell"]["efficiency"] == efficiency
            and row["cell"]["drift"] == drift
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one prespecified {name} cell")
        fixed[name] = matches[0]
    return fixed


def publish_registered(base, screen, namespace):
    fixed = registered_cells(screen["cells"])
    value = {
        "namespace": namespace,
        "roles": "fixed by the approved plan: 77000 READ and unit initial spread",
        "registered": {k: v["cell"] for k, v in fixed.items()},
        "exploratory_only": {k: v["cell"] for k, v in screen["selected"].items()},
        "screen_identity": checkpoint_digest(base / "screen.json"),
    }
    path = base / "registered_cells.json"
    if not path.exists():
        atomic_json(path, value)
    elif json.loads(path.read_text()) != value:
        raise ValueError("registered cell mapping does not match the analytic screen")
    return fixed


def load_routes(root, lane, namespace):
    routes = {}
    signatures = {}
    for seed in TRAIN_SEEDS:
        seed_root = root / f"seed_{seed}"
        core, models, spec = load_shared_route(seed_root)
        b = spec["binding"]
        arm = {"b0": "state", "b2": "joint"}[lane]
        if (
            b.get("campaign_namespace") != namespace
            or b.get("upstream_seed") != 7301
            or b.get("upstream_run_id") != f"{arm}_cs_32768_s7301"
            or b["seed"] != seed
            or b.get("shared_lane") != lane
            or b.get("upstream_updates") != 32768
        ):
            raise ValueError(
                "portable control weights do not match the registered matrix anchor and fitted seed"
            )
        anchor_pair = b["classifier_sha256"]
        if anchor_pair["3"] != anchor_pair["4"]:
            raise ValueError("upstream weights differ across N")
        routes[seed] = core, models
        signatures[str(seed)] = {
            "portable_adapters": checkpoint_digest(seed_root / "shared_control_adapters.pt"),
            "upstream_classifier": anchor_pair["3"],
        }
    if len({s["upstream_classifier"] for s in signatures.values()}) != 1:
        raise ValueError("downstream seeds use different upstream states")
    return routes, signatures


def learned_policy(env, route, fit, source, history="ordered"):
    core, models = route
    if source == "command":
        core = PublicCommandState(core)
    return callback(models[fit], history, env, core, "cpu")


def freeze_cell(reference, namespace, routes, count, *, multipliers=(0.25, 0.5, 1.0)):
    cell = reference["cell"]
    env = environment(cell)
    prefix = seed_prefix(namespace, cell, "freeze")
    noop = trajectories(env, cell, classical_policy(env, "noop"), 0, count, prefix)["fidelity"]
    decisions = {}
    for fit in FITS:
        all_stats, one_fractions = {}, {}
        for seed, route in routes.items():
            fn = learned_policy(env, route, fit, "inferred")
            fraction = optimal_single(env, cell, fn, count, prefix)
            one_fractions[str(seed)] = fraction
            all_stats[seed] = {
                k: arm_metrics(trajectories(env, cell, fn, k, count, prefix, fraction), noop)
                for k in K_GRID
            }
        avg = {
            k: {
                name: float(np.mean([s[k][name] for s in all_stats.values()]))
                for name in ("mean", "gain_over_noop", "harm_vs_noop")
            }
            for k in K_GRID
        }
        selected = select_horizon(avg)
        # Multiplier sensitivity at K=10 in development does not select confirmatory policies.
        sensitivity = {"1.0": avg[10]}
        for m in multipliers:
            if m == 1:
                continue
            records = [
                arm_metrics(
                    trajectories(
                        env,
                        cell,
                        learned_policy(env, r, fit, "inferred"),
                        10,
                        count,
                        prefix,
                        multiplier=m,
                    ),
                    noop,
                )
                for r in routes.values()
            ]
            sensitivity[str(m)] = {
                key: float(np.mean([r[key] for r in records])) for key in avg[10]
            }
        decisions[fit] = {
            "selected_k": selected,
            "frozen_one_fractions": one_fractions,
            "dev_by_k": avg,
            "dev_action_multiplier_sensitivity": sensitivity,
            "right_edge_still_improving": bool(
                avg[10]["mean"] > avg[8]["mean"]
                and avg[10]["mean"] == max(row["mean"] for row in avg.values())
            ),
        }
    return {
        "cell": cell,
        "decisions": decisions,
        "analytic_damping": reference["analytic_damping"],
        "analytic_one_fraction": reference["analytic_one_fraction"],
    }


def confirm_cell(plan, namespace, routes, count, *, bootstrap_draws=10000):
    cell = plan["cell"]
    env = environment(cell)
    prefix = seed_prefix(namespace, cell, "confirm")
    analytic = classical_policy(env, "analytic", plan["analytic_damping"])
    panels = {
        "noop": trajectories(env, cell, classical_policy(env, "noop"), 0, count, prefix),
        "analytic": trajectories(env, cell, analytic, 10, count, prefix),
        "analytic_one": trajectories(
            env, cell, analytic, 1, count, prefix, plan["analytic_one_fraction"]
        ),
    }
    noop = panels["noop"]["fidelity"]
    curves = {
        k: np.broadcast_to(v["fidelity"], (len(routes), *noop.shape)) for k, v in panels.items()
    }
    storage = {
        "episode_ids": np.asarray(
            [f"n{env.n}_{prefix + i - i % 64}_{i % 64}" for i in range(count)]
        ),
        "seeds": np.asarray(list(routes)),
        "paid_reads": np.array(cell["reads"]),
    }
    scores = {}

    def retain(name, p):
        scores[name] = arm_metrics(p, noop)
        for field, array in p.items():
            storage[f"{name}_{field}"] = array

    for key, p in panels.items():
        retain(key, p)
    chosen = {f: plan["decisions"][f]["selected_k"] for f in FITS}
    for seed, route in routes.items():
        for fit in FITS:
            choice = plan["decisions"][fit]
            one = choice["frozen_one_fractions"][str(seed)]
            for source in ("inferred", "command"):
                fn = learned_policy(env, route, fit, source)
                horizons = K_GRID if source == "inferred" else sorted({1, 10, chosen[fit] or 10})
                for k in horizons:
                    key = f"{fit}_{source}_k{k}"
                    p = (
                        panels["noop"]
                        if k == 0
                        else trajectories(env, cell, fn, k, count, prefix, one)
                    )
                    retain(f"seed{seed}_{key}", p)
                    curves.setdefault(key, []).append(p["fidelity"])
        for history in ("shuffled", "current"):
            # Shuffling acts on the teacher's ordered adapter. Current-only is its separately fitted adapter.
            fit = "ordered" if history == "shuffled" else "current"
            p = trajectories(
                env, cell, learned_policy(env, route, fit, "inferred", history), 10, count, prefix
            )
            retain(f"seed{seed}_history_{history}", p)
    curves = {k: np.asarray(v) for k, v in curves.items()}
    summary = simultaneous_results(curves, chosen, bootstrap_draws=bootstrap_draws)
    descriptive_history = {}
    for ablation in ("shuffled", "current"):
        differences = np.stack(
            [
                storage[f"seed{seed}_ordered_inferred_k10_fidelity"][:, -1]
                - storage[f"seed{seed}_history_{ablation}_fidelity"][:, -1]
                for seed in routes
            ]
        )
        descriptive_history[f"ordered_minus_{ablation}"] = {
            "mean": float(differences.mean()),
            "per_seed_mean": differences.mean(axis=1).tolist(),
        }
    summary.update(
        descriptive_history=descriptive_history,
        arms=scores,
        cell=cell,
        seeds=list(routes),
        paid_reads=cell["reads"],
        episodes=count,
        problem="static or constant-velocity coherent error; conditional K1 next preparations",
    )
    return summary, storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["screen", "freeze", "confirm"])
    parser.add_argument("--n", type=int, required=True, choices=(3, 4))
    parser.add_argument("--campaign-namespace", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=("b0", "b2"))
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument("--cell", choices=SLOTS)
    args = parser.parse_args()
    if args.stage != "screen" and (not args.lane or not args.shared_root or not args.cell):
        parser.error("freeze/confirm require --lane, --shared-root and --cell")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = args.output / f"n{args.n}"
    base.mkdir(parents=True, exist_ok=True)
    screen_path = base / "screen.json"
    ns = args.campaign_namespace
    if args.stage == "screen":
        if screen_path.exists():
            screen = json.loads(screen_path.read_text())
            if screen["namespace"] != ns:
                raise ValueError("existing screen belongs to another campaign")
            publish_registered(base, screen, ns)
            return
        rows = [screen_cell(c, ns, DEV_SIZE) for c in grid(args.n)]
        atomic_json(
            screen_path,
            {
                "namespace": ns,
                "dev_episodes": DEV_SIZE,
                "screened_by": "analytic controls only",
                "cells": rows,
                "selected": choose_cells(rows),
            },
        )
        publish_registered(base, json.loads(screen_path.read_text()), ns)
        return
    screen = json.loads(screen_path.read_text())
    if (
        screen["namespace"] != ns
        or len(screen["cells"]) != 54
        or screen["dev_episodes"] != DEV_SIZE
    ):
        raise ValueError("wrong screening design")
    registered = publish_registered(base, screen, ns)
    dest = base / args.lane / args.cell
    dest.mkdir(parents=True, exist_ok=True)
    freeze_path = dest / "freeze.json"
    routes, signatures = load_routes(args.shared_root, args.lane, ns)
    identity = {
        "namespace": ns,
        "lane": args.lane,
        "slot": args.cell,
        "screen_identity": checkpoint_digest(screen_path),
        "weights": signatures,
        "n": args.n,
        "dev_episodes": DEV_SIZE,
        "confirm_episodes": CONFIRM_SIZE,
        "k_grid": list(K_GRID),
        "shared_seed_panel": list(TRAIN_SEEDS),
    }
    if args.stage == "freeze":
        if freeze_path.exists():
            if json.loads(freeze_path.read_text())["identity"] != identity:
                raise ValueError("development choices already frozen for a different design")
            return
        plan = freeze_cell(registered[args.cell], ns, routes, DEV_SIZE)
        atomic_json(freeze_path, {"identity": identity, "plan": plan})
        return
    frozen = json.loads(freeze_path.read_text())
    if frozen["identity"] != identity:
        raise ValueError("confirmation differs from frozen development choices")
    opened = dest / "confirmation_identity.json"
    signature = {"frozen": checkpoint_digest(freeze_path), "identity": identity}
    if opened.exists() and json.loads(opened.read_text()) != signature:
        raise ValueError("different confirmation previously opened at this location")
    if (dest / "result.json").exists():
        if not opened.exists() or not (dest / "episode_panels.npz").is_file():
            raise ValueError("incomplete confirmation evidence")
        return
    atomic_json(opened, signature)
    result, rows = confirm_cell(frozen["plan"], ns, routes, CONFIRM_SIZE)
    tmp = dest / "episode_panels.tmp.npz"
    np.savez_compressed(tmp, **rows)
    publish_atomic(tmp, dest / "episode_panels.npz")
    atomic_json(dest / "result.json", {"identity": signature, **result})


if __name__ == "__main__":
    main()
