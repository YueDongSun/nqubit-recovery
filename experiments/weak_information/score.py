"""Evaluate a frozen synthetic controller at preregistered strong and weak measurement budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from experiments.protocol import register

SLOTS = ("strong_1x", "weak_1x", "weak_4x", "weak_trace_matched")


def read(p):
    return json.loads(p.read_text())


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def publish(p, record):
    p.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(dir=p.parent, mode="w", delete=False) as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write("\n")
        tmp = Path(f.name)
    try:
        try:
            os.link(tmp, p)
        except FileExistsError:
            if read(p) != record:
                raise ValueError(f"inconsistent existing selection: {p}")
    finally:
        tmp.unlink()


def simulation():
    import torch
    from ml_recon.physics_hybrid_v10 import temporal_operating_range as sim

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return sim


def geometry(n, multiplier=None):
    sim = simulation()
    full = {}
    for name, eta in [("strong", 0.35), ("weak", 0.04)]:
        env = sim.environment({"n": n, "efficiency": eta})
        a = env.linear_fisher.squeeze(0).numpy() / (3 * n - 2)
        sym = (a + a.T) / 2
        spectrum = np.linalg.eigvalsh(sym)
        full[name] = dict(
            fisher_per_read=sym.tolist(),
            spectrum_per_read=spectrum.tolist(),
            relative_support=int((spectrum > spectrum.max() * 1e-10).sum()),
            trace_per_read=float(np.trace(sym)),
        )
    ratio = full["strong"]["trace_per_read"] / full["weak"]["trace_per_read"]
    unit = 11 * (3 * n - 2)
    matched = unit * math.floor(77000 * ratio / unit + 0.5)
    totals = {
        "strong_1x": 77000,
        "weak_1x": 77000,
        "weak_4x": 308000,
        "weak_trace_matched": matched,
    }
    for name, total in totals.items():
        g = full["strong" if name == "strong_1x" else "weak"]
        full[name] = dict(
            reads=total,
            trace=total * g["trace_per_read"],
            spectrum=(total * np.array(g["spectrum_per_read"])).tolist(),
        )
    value = dict(
        n=n,
        per_read=full,
        required_weak_to_strong_trace_ratio=ratio,
        reads=totals,
        matched_trace_over_strong_trace=full["weak_trace_matched"]["trace"]
        / full["strong_1x"]["trace"],
        public_operating_point="uniform conditional target, fixed calibration, across all eleven windows",
    )
    publish(BASE / f"geometry_n{n}.json", value)
    return value


def runner(n, slot, stage):
    start = time.monotonic()
    reg = read(BASE / "registration.json")
    out = BASE / f"n{n}" / slot
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / ("dev.json" if stage == "dev" else "result.json")
    if result_path.exists():
        return
    geom = read(BASE / f"geometry_n{n}.json")
    sim = simulation()
    from ml_recon.physics_hybrid_v10.temporal_recovery_shared_qualification import (
        campaign_seed_base,
    )

    train_ns = reg["training_namespace"]
    routes, sig = sim.load_routes(MODELS, "b2", train_ns)
    for seed, files in reg["weights"].items():
        if any(digest(MODELS / f"seed_{seed}" / k) != v for k, v in files.items()):
            raise ValueError("frozen weights differ from preregistration")
    if list(routes) != reg["seeds"]:
        raise ValueError("fixed training-seed panel changed")
    prefix = (
        campaign_seed_base(reg["namespace"])
        + n * 10**9
        + ({"dev": 1, "heldout": 2}[stage]) * 10**12
    )
    if any(
        abs(prefix - sim.seed_prefix(train_ns, c, role)) < 1024
        for m in (3, 4)
        for c in sim.grid(m)
        for role in ("screen", "freeze", "confirm")
    ):
        raise ValueError("exploratory cohort overlaps primary study")
    eta = reg["conditions"][slot][0]
    cell = dict(n=n, reads=geom["reads"][slot], efficiency=eta, spread=1.0, drift=False)
    env = sim.environment(cell)
    count = reg["roles"][stage]

    def panel(fn, k, fraction=None):
        return sim.trajectories(env, cell, fn, k, count, prefix, fraction)

    if stage == "dev":
        fractions = {}
        for seed, route in routes.items():
            fn = sim.learned_policy(env, route, "learner", "inferred")
            fractions[str(seed)] = sim.optimal_single(env, cell, fn, count, prefix)
        dampings = (0.25, 0.45, 0.70)
        scores = [
            panel(sim.classical_policy(env, "analytic", g), 10)["fidelity"][
                :, -1
            ].mean()
            for g in dampings
        ]
        damping = dampings[int(np.argmax(scores))]
        analytic_fraction = sim.optimal_single(
            env, cell, sim.classical_policy(env, "analytic", damping), count, prefix
        )
        publish(
            result_path,
            dict(
                n=n,
                condition=slot,
                cell=cell,
                role="independent_dev",
                episodes=count,
                prefix=prefix,
                single_fractions=fractions,
                analytic_damping=damping,
                analytic_fraction=analytic_fraction,
                weights=sig,
                registration=digest(BASE / "registration.json"),
                elapsed_s=time.monotonic() - start,
            ),
        )
        return
    seal = read(BASE / "frozen_dev.json")
    if seal != {
        f"n{k}/{s}": digest(BASE / f"n{k}" / s / "dev.json")
        for k in (3, 4)
        for s in SLOTS
    }:
        raise ValueError("global DEV lock differs from its eight inputs")
    dev = read(out / "dev.json")
    if (
        dev["cell"] != cell
        or dev["weights"] != sig
        or dev["registration"] != digest(BASE / "registration.json")
    ):
        raise ValueError("frozen DEV differs from held-out policy")
    data = {
        "ids": np.array([f"n{n}_{prefix + i - i % 64}_{i % 64}" for i in range(count)]),
        "seeds": np.array(list(routes)),
        "reads": np.array(cell["reads"]),
    }

    def retain(key, p):
        for name in ("fidelity", "executed", "reads_per_setting"):
            data[f"{key}_{name}"] = p[name]

    retain("noop", panel(sim.classical_policy(env, "noop"), 0))
    analytic = sim.classical_policy(env, "analytic", dev["analytic_damping"])
    retain("analytic10", panel(analytic, 10))
    retain("analytic1", panel(analytic, 1, dev["analytic_fraction"]))
    for seed, route in routes.items():
        fn = sim.learned_policy(env, route, "learner", "inferred")
        retain(f"s{seed}_learned1", panel(fn, 1, dev["single_fractions"][str(seed)]))
        retain(f"s{seed}_learned10", panel(fn, 10))
    tmp = out / "timed_panels.tmp.npz"
    np.savez_compressed(tmp, **data)
    tmp.replace(out / "episode_panels.npz")
    publish(
        result_path,
        dict(
            n=n,
            condition=slot,
            cell=cell,
            role="exploratory_heldout",
            episodes=count,
            prefix=prefix,
            dev_lock=digest(BASE / "frozen_dev.json"),
            panel_sha256=digest(out / "episode_panels.npz"),
            elapsed_s=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("design", "geometry", "dev", "seal", "heldout"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument(
        "--weights",
        type=Path,
        help="B2 export directory containing seed subdirectories",
    )
    p.add_argument("--namespace", help="training namespace of supplied exports")
    p.add_argument("--n", type=int, choices=(3, 4))
    p.add_argument("--slot", choices=SLOTS)
    a = p.parse_args()
    BASE, MODELS = a.root, a.weights
    if a.stage in ("design", "dev", "heldout") and MODELS is None:
        p.error("supply --weights for this stage")
    if a.stage == "design":
        if not a.namespace:
            p.error("supply --namespace for design")
        seeds = [11031, 11047, 11071]
        names = ("shared_backbone.pt", "shared_control_adapters.pt")
        reg = dict(
            study="weak_read_information_budget_v1",
            source="effective_single_event_v1",
            status="exploratory synthetic",
            training_namespace=a.namespace,
            namespace="weak_read_information_budget_v1_20261120",
            ns=[3, 4],
            seeds=seeds,
            conditions={s: [0.35 if s == "strong_1x" else 0.04] for s in SLOTS},
            roles=dict(dev=96, heldout=512),
            statistical_family=38,
            bootstrap_draws=10000,
            weights={
                str(seed): {f: digest(MODELS / f"seed_{seed}" / f) for f in names}
                for seed in seeds
            },
        )
        register(BASE / "registration.json", reg)
    elif a.stage == "geometry":
        for n in (3, 4):
            geometry(n)
    elif a.stage == "seal":
        publish(
            BASE / "frozen_dev.json",
            {
                f"n{n}/{s}": digest(BASE / f"n{n}" / s / "dev.json")
                for n in (3, 4)
                for s in SLOTS
            },
        )
    elif a.n is None or a.slot is None:
        p.error("select N and condition")
    else:
        runner(a.n, a.slot, a.stage)
