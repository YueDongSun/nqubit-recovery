"""Explore fixed-setting exposure and equal per-relative-coordinate information at larger true N."""

from pathlib import Path
import argparse
import csv
import gzip
import json

import numpy as np
import torch

from . import run_grid as grid, shared_scale
from experiments.protocol import file_digest, register, pending_panel


def geometry(n):
    s = 4 * n - 3
    dz = np.zeros((s, 2 * n))
    dz[:n, ::2] = 2 / n * (np.eye(n) - 1 / n)
    for j in range(1, n):
        for m, theta in enumerate(grid.ANGLES):
            t = n + 3 * (j - 1) + m
            dz[t, ::2] = -2 * (1 + np.cos(theta)) / n**2
            dz[t, [0, 2 * j]] += (1 + np.cos(theta)) / n
            dz[t, [1, 2 * j + 1]] += np.sin(theta) / n * np.array([-1.0, 1.0])
    pi = grid.pure_prob(np.full((1, n), n**-0.5))[0]
    weight = (grid.event_coefficients()[1] ** 2 / pi).sum(-1)
    # Check two independent derivative directions with finite differences of the actual four-cell law.
    eps = 1e-5
    perturb = np.zeros((2, n, 2))
    perturb[0, 1, 0] = eps
    perturb[1, 1, 1] = eps
    numeric = (
        grid.pure_prob(grid.vector(perturb)) - grid.pure_prob(grid.vector(-perturb))
    ) / (2 * eps)
    np.testing.assert_allclose(
        numeric,
        np.stack([dz[:, 2, None], dz[:, 3, None]]) * grid.event_coefficients()[1],
        rtol=1e-5,
        atol=1e-10,
    )
    return dz, weight


def fisher(n, paid, dz, weight):
    shots = grid.allocate_integer_budget(paid, 4 * n - 3)
    mat = dz.T @ ((shots * weight)[:, None] * dz)

    def report(sub):
        eig = np.linalg.eigvalsh(sub)
        # The two exactly known common gauges have analytic eigenvalue zero.
        return [float(x) for x in eig]

    return dict(
        total_read_decision=int(paid),
        joint_mean_per_relative_coordinate=float(np.trace(mat) / (2 * n - 2)),
        amplitude_mean_per_relative_coordinate=float(np.trace(mat[::2, ::2]) / (n - 1)),
        phase_mean_per_relative_coordinate=float(np.trace(mat[1::2, 1::2]) / (n - 1)),
        amplitude_spectrum=report(mat[::2, ::2]),
        phase_spectrum=report(mat[1::2, 1::2]),
        joint_spectrum=report(mat),
    )


def create_design(root, core):
    reference = None
    actual = {}
    rows = []
    for n in shared_scale.SIZES:
        dz, w = geometry(n)
        unit = np.sum(dz**2, axis=1) * w / (2 * n - 2)

        def information(paid):
            return float(grid.allocate_integer_budget(paid, len(unit)) @ unit)

        if n == 3:
            reference = information(70000)
        lo, hi = len(unit), 70000
        while information(hi) < reference:
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if information(mid) >= reference:
                hi = mid
            else:
                lo = mid + 1
        actual[n] = {}
        for mode, paid in (
            ("fixed", 70000),
            ("per_setting", round(70000 * (4 * n - 3) / 9)),
            ("mean_fisher", lo),
        ):
            record = fisher(n, paid, dz, w)
            record["terminal"] = round(paid / 10)
            record["total"] = int(paid + record["terminal"])
            actual[n][mode] = record
            if mode != "fixed":
                for a, p, tag in (
                    (0.35, 0.0, "a2_p0"),
                    (0.0, 0.70, "a0_p2"),
                    (0.35, 0.70, "a2_p2"),
                ):
                    rows.append(
                        dict(N=n, K=1, A=a, P=p, tag=f"n{n}_{tag}_k1_{mode}", mode=mode)
                    )
    out = dict(
        schema="highN_count_budget_mechanism_v1",
        question="Can larger READ budgets rescue the fixed-weight response when the original 77k curve loses gain?",
        cases=rows,
        budgets=actual,
        parents=grid.ROWS,
        parent_seed=grid.PARENT_SEED,
        weights=file_digest(core),
        calibration=grid.CAL,
        training_sizes=[3, 4, 8],
        shared_true_sizes=shared_scale.SIZES,
        comparators="Same held parents, RNG namespaces, frozen network and bounded actions as the fixed-budget primary. Fixed reference is drawn from the earlier raw rows.",
        budget_convention="Decision READ is either N=3 exposure per projector setting (integer rounded) or the smallest integer whose known-apparatus, uniform-target four-cell Fisher trace per 2(N-1) relative amplitude/phase coordinates equals/exceeds its N=3 value. Terminal diagnostic costs another round(decision/10) READ. These are prospective costs, not delivered rates; trace equality need not equalize the spectra.",
        interpretation="Exploratory, one optimization seed, shared latent parent directions. Distinguish oracle action feasibility, learned inference, classical Fisher correction, and FD library choice. No new primary certification or retraining.",
    )
    register(root / "protocol.json", out)


def sample(v, n, paid, tag, ids):
    shots = grid.allocate_integer_budget(paid, 4 * n - 3)
    if int(shots.max()) > np.iinfo(np.int32).max:
        raise OverflowError("per-setting count requires a larger integer")
    return grid.observe(v, n, paid, tag, ids)


def cell(root, reg, case, model, device):
    n, a, p, mode = case["N"], case["A"], case["P"], case["mode"]
    design = reg["budgets"][str(n)][mode]
    paid, final = design["total_read_decision"], design["terminal"]
    initial = grid.coordinates(n, a, p)
    rows = []
    arrays = {}
    batch = {3: 32, 4: 32, 8: 32, 16: 32, 32: 16, 64: 8, 128: 4, 256: 2}[n]
    for i in range(0, grid.ROWS, batch):
        ids = np.arange(i, min(i + batch, grid.ROWS))
        x = initial[ids]
        f0 = grid.fidelity(x)
        ca, cp = grid.residual(x)
        counts, _ = sample(grid.vector(x), n, paid, 120, ids)
        learned, fd = grid.neural_actions(
            model, counts, grid.vector(np.zeros_like(x)), n, paid, device, True
        )
        arrays.setdefault("initial_counts", []).append(counts)
        for name, act in (
            ("noop", np.zeros_like(x)),
            ("learned_plugin", learned),
            ("classical_fisher", grid.classical_action(counts, n, paid)),
            ("oracle_bounded", grid.bounded(-x)),
            ("learned_fd", fd),
        ):
            end = x + act
            counts_t, _ = sample(grid.vector(end), n, final, 701, ids)
            arrays.setdefault(name + "_terminal_counts", []).append(counts_t)
            arrays.setdefault(name + "_action", []).append(act.astype(np.float32))
            ft = grid.fidelity(end)
            ea, ep = grid.residual(end)
            tot = counts_t.sum(1)
            for j, parent in enumerate(ids):
                mag = float(np.max(np.abs(act[j]) / np.array([0.35, 0.70])))
                gain = ft[j] - f0[j]
                rows.append(
                    dict(
                        N=n,
                        K=1,
                        A=a,
                        P=p,
                        budget_mode=mode,
                        policy=name,
                        parent_id=int(parent),
                        seed=7301,
                        reads_decision=paid,
                        reads_terminal=final,
                        F0=f0[j],
                        FT=ft[j],
                        delta_F=gain,
                        amp_rms0=ca[j],
                        amp_rmsT=ea[j],
                        phase_rms0=cp[j],
                        phase_rmsT=ep[j],
                        harm_vs_noop=int(gain < -1e-12),
                        executed=int(mag > 1e-10),
                        actions_taken=int(mag > 1e-10),
                        saturated=int(mag >= 1 - 1e-6),
                        terminal_00=tot[j, 0],
                        terminal_10=tot[j, 1],
                        terminal_01=tot[j, 2],
                        terminal_11=tot[j, 3],
                    )
                )
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    arrays.update(
        initial_coords=initial,
        parent_ids=np.arange(grid.ROWS),
        decision_exposure=grid.allocate_integer_budget(paid, 4 * n - 3),
        terminal_exposure=grid.allocate_integer_budget(final, 4 * n - 3),
    )
    tag = case["tag"]
    for folder in ("arrays", "rows"):
        (root / folder).mkdir(exist_ok=True)
    tmp = root / "arrays" / (tag + ".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(root / "arrays" / (tag + ".npz"))
    tmp = root / "rows" / (tag + ".partial")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(root / "rows" / (tag + ".csv.gz"))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("design", "run"))
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--core", required=True, type=Path)
    cli.add_argument("--n", nargs="+", type=int, choices=shared_scale.SIZES)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    torch.set_num_threads(1)
    if args.action == "design":
        create_design(args.root, args.core)
    else:
        reg = json.loads((args.root / "protocol.json").read_text())
        if file_digest(args.core) != reg["weights"]:
            raise ValueError("Supplied weights differ from the registered experiment")
        device = torch.device(args.device)
        model = shared_scale.load_shared(args.core, device)
        for case in reg["cases"]:
            if args.n and case["N"] not in args.n:
                continue
            with pending_panel(args.root, case["tag"]) as pending:
                if pending:
                    cell(args.root, reg, case, model, device)
