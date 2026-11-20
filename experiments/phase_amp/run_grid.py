"""Measure frozen preparation control on independently set amplitude and phase radii, in independent simulated preparations."""

import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from .coordinates import grid_coords, primary_initial

from experiments.protocol import file_digest, register


def control_design(weights, lane, namespace):
    seeds = [11031, 11047, 11071]
    study = "phase_amp_control_range_v1_20261120"
    fractions = {"3_strong": 0.8, "3_weak": 0.55, "4_strong": 0.8, "4_weak": 0.8}
    damping = {"3_strong": 0.45, "3_weak": 0.70, "4_strong": 0.45, "4_weak": 0.45}
    files = ("shared_backbone.pt", "shared_control_adapters.pt")
    return dict(
        study=study if lane == "b2" else "phase_amp_range_b0_secondary_v1",
        namespace=study,
        training_namespace=namespace,
        source="effective_single_event_v1",
        lane=lane,
        seeds=seeds,
        n=[3, 4],
        calibration={"strong": 0.35, "weak": 0.04},
        reads=77000,
        windows=11,
        k_by_n={"3": 8, "4": 10},
        amplitudes=[0, 0.12, 0.35, 0.70, 1.40],
        phases_rad=[0, 0.20, 0.45, 0.70, 1.10],
        episodes_per_cell=256,
        frozen_control={
            key: dict(
                analytic_damping=damping[key],
                single_fractions={str(seed): v for seed in seeds},
            )
            for key, v in fractions.items()
        },
        fixed_points={
            "strong": [[0, 0], [2, 0], [0, 3], [2, 3], [4, 4]],
            "weak": [[2, 3]],
        },
        weights={
            str(seed): {
                name: file_digest(weights / f"seed_{seed}" / name) for name in files
            }
            for seed in seeds
        },
    )


def atomic_json(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def setup(root, weights):
    global torch, sim, envmod, routes, envs, REG, HERE
    HERE = root
    REG = json.loads((root / "registration.json").read_text())
    import torch
    from ml_recon.physics_hybrid_v10 import (
        temporal_operating_range as sim,
        temporal_recovery_environment as envmod,
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    routes, sig = sim.load_routes(weights, REG["lane"], REG["training_namespace"])
    if list(routes) != REG["seeds"] or any(
        sig[k]["portable_adapters"] != v["shared_control_adapters.pt"]
        for k, v in REG["weights"].items()
    ):
        raise ValueError("wrong frozen policy seeds or exports")
    envs = {}


def run_fixed_coords(env, bias, policy, k, seed, fraction=None):
    """Original eleven-window simulator with an explicit static latent error vector instead of a Gaussian draw."""
    schedule = sim.acquisition_schedule(
        env.n, REG["reads"], fraction if k == 1 else None
    )
    choose = sim.limited_policy(policy, k)
    command = torch.zeros_like(bias)
    observations = []
    actions = []
    scores = []
    with torch.no_grad():
        for t, exposure in enumerate(schedule):
            x = bias + command
            scores.append(envmod.fidelity(x))
            obs = env.acquire(x, command, int(exposure), seed, t)
            observations.append(obs)
            if not torch.all(
                obs.counts.sum((1, 2)) == int(exposure) * len(env.analyzers)
            ):
                raise ValueError("actual native counts and exposure disagree")
            if t == 10:
                break
            move = envmod.feasible(
                choose(envmod.Prefix(tuple(observations), tuple(actions))).double(),
                env.projector,
            )
            actions.append(move)
            command = command + move
    if int(schedule.sum()) * len(env.analyzers) != REG["reads"]:
        raise ValueError("unequal paid acquisition budget")
    return (
        torch.stack(scores, 1).cpu().numpy(),
        torch.stack(actions, 1).cpu().numpy(),
        schedule,
    )


def run_cell(job, engineering=False):
    n, condition, ai, pi = job
    dest = HERE / f"n{n}" / condition / f"a{ai}_p{pi}"
    if not engineering and (dest / "result.json").is_file():
        return json.loads((dest / "result.json").read_text())
    start = time.monotonic()
    ns = REG["namespace"] + ("_engineering" if engineering else "")
    from ml_recon.physics_hybrid_v10.temporal_recovery_shared_qualification import (
        campaign_seed_base,
    )

    base = campaign_seed_base(ns) + n * 10**9
    rows = 64 if engineering else REG["episodes_per_cell"]
    amp = REG["amplitudes"][ai]
    phase = REG["phases_rad"][pi]
    initial = grid_coords(n, rows, base, amp, phase)
    p_read = REG["calibration"][condition]
    key = (n, condition)
    if key not in envs:
        envs[key] = sim.environment({"n": n, "efficiency": p_read})
    env = envs[key]
    frozen = REG["frozen_control"][f"{n}_{condition}"]
    k = REG["k_by_n"][str(n)]

    def panel(fn, opportunities, fraction=None):
        ff = []
        aa = []
        for offset in range(0, rows, 64):
            f, a, shots = run_fixed_coords(
                env,
                torch.from_numpy(initial[offset : offset + 64]),
                fn,
                opportunities,
                base + offset,
                fraction,
            )
            ff.append(f)
            aa.append(a)
        return np.concatenate(ff), np.concatenate(aa), shots

    data = dict(
        episode_ids=np.array(
            [f"n{n}_{base + i - i % 64}_{i % 64}" for i in range(rows)]
        ),
        seeds=np.asarray(REG["seeds"]),
        initial_coords=initial,
        paid_reads=np.int64(REG["reads"]),
        frozen_k=np.int64(k),
    )
    data["noop_fidelity"], _, data["noop_read_schedule"] = panel(
        envmod.classical_policy(env, "noop"), 0
    )
    (
        data["analytic_fidelity"],
        data["analytic_actions"],
        data["analytic_read_schedule"],
    ) = panel(envmod.classical_policy(env, "analytic", frozen["analytic_damping"]), 10)
    for key, opportunities in [("learned_k", k), ("learned_one", 1)]:
        ff = []
        aa = []
        readouts = []
        for seed, route in routes.items():
            fn = sim.learned_policy(env, route, "learner", "inferred")
            fraction = (
                frozen["single_fractions"][str(seed)] if opportunities == 1 else None
            )
            f, a, shots = panel(fn, opportunities, fraction)
            ff.append(f)
            aa.append(a)
            readouts.append(shots)
        data[key + "_fidelity"] = np.stack(ff)
        data[key + "_actions"] = np.stack(aa)
        data[key + "_read_schedule"] = np.stack(readouts)
    info = dict(
        study=REG["study"],
        source=REG["source"],
        n=n,
        condition=condition,
        amplitude_index=ai,
        phase_index=pi,
        amplitude_rms=amp,
        phase_rms_rad=phase,
        phase_rms_deg=float(np.degrees(phase)),
        efficiency=p_read,
        k=k,
        reads=REG["reads"],
        episodes=rows,
        seeds=REG["seeds"],
        actual_per_policy_reads_per_episode=77000,
        analytic_damping=frozen["analytic_damping"],
        single_fractions=frozen["single_fractions"],
        role="engineering" if engineering else "independent_range",
        runtime_seconds=time.monotonic() - start,
    )
    if not engineering:
        dest.mkdir(exist_ok=True, parents=True)
        tmp = dest / "episode_panels.partial.npz"
        np.savez_compressed(tmp, **data)
        tmp.replace(dest / "episode_panels.npz")
        atomic_json(dest / "result.json", info)
    return info


def engineering(n):
    seed = 711042731 + n
    ns = REG["namespace"] + "_engineering"
    env = sim.environment({"n": n, "efficiency": 0.35})
    route = routes[REG["seeds"][0]]
    rows = 8
    ids = [f"n{n}_{seed}_{i}" for i in range(rows)]
    b, _ = primary_initial(ids, n, 1.0, False)
    fn = sim.learned_policy(env, route, "learner", "inferred")
    k = REG["k_by_n"][str(n)]
    f, a, s = run_fixed_coords(env, torch.from_numpy(b), fn, k, seed)
    orig = envmod.rollout(
        env, sim.limited_policy(fn, k), episodes=rows, seed=seed, schedule=s
    )
    fd = float(np.max(np.abs(f - orig["scores"].numpy())))
    ad = float(np.max(np.abs(a - orig["actions"].numpy())))
    if fd > 2e-12 or ad > 2e-11:
        raise ValueError("new coordinate entry and original simulator disagree")
    row = run_cell((n, "strong", 2, 3), True)
    row.update(parity_max_fidelity=fd, parity_max_action=ad, fixture_namespace=ns)
    atomic_json(HERE / f"engineering_n{n}.json", row)
    print(json.dumps(row), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("engineering", "run"))
    cli.add_argument("--root", required=True, type=Path, help="output for one lane")
    cli.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="directory containing the three seed exports",
    )
    cli.add_argument(
        "--namespace", required=True, help="campaign_namespace of the shared exports"
    )
    cli.add_argument("--lane", choices=("b0", "b2"), default="b2")
    cli.add_argument("--n", required=True, type=int, choices=(3, 4))
    cli.add_argument("--workers", type=int, default=1)
    args = cli.parse_args()
    reg = control_design(args.weights, args.lane, args.namespace)
    register(args.root / "registration.json", reg)
    if args.stage == "engineering":
        setup(args.root, args.weights)
        engineering(args.n)
        return
    jobs = [
        (args.n, condition, a, p)
        for condition in reg["calibration"]
        for a in range(5)
        for p in range(5)
        if args.lane == "b2" or [a, p] in reg["fixed_points"][condition]
    ]
    if args.workers == 1:
        setup(args.root, args.weights)
        for job in jobs:
            run_cell(job)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=setup,
            initargs=(args.root, args.weights),
        ) as pool:
            list(pool.map(run_cell, jobs))
    atomic_json(args.root / f"finished_n{args.n}.json", dict(n=args.n, cells=len(jobs)))


if __name__ == "__main__":
    main()
