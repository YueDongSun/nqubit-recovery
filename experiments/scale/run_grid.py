"""Prospective fixed-READ recovery map for the actual N=3/4/8-trained shared projector model."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from . import shared_scale as shared
from experiments.protocol import file_digest, register, pending_panel
from ml_recon.physics_hybrid_v10.training_ideas_data import (
    allocate_integer_budget,
    projector_schedule,
)
from ml_recon.physics_hybrid_v10.effective_projector_observation import (
    make_public_observation,
)
from ml_recon.physics_hybrid_v10.training_ideas_data import public_projector_inputs
from ml_recon.physics_hybrid_v10.training_ideas_objective import normalized_state
from nqrec.physics.single_event_mc import (
    calibrate_source_weights,
    four_cell_probabilities,
)

PARENT_SEED = 202611201937713
ROWS = 128
CAL = (0.025, 0.65, 0.002, 0.35, 0.002)
ANGLES = 2 * np.pi * np.arange(3) / 3


def coordinates(n, a, p):
    """Rank-balanced signs keep wrapped RMS equal to requested RMS on this grid."""
    rng = np.random.Generator(np.random.Philox(PARENT_SEED))
    if n not in shared.SIZES:
        raise ValueError(f"Unsupported size for the paired N study: {n}")
    score = rng.normal(size=(ROWS, max(shared.SIZES), 2))[:, :n]
    magnitude = rng.uniform(0.8, 1.0, size=(ROWS, max(shared.SIZES), 2))[:, :n]
    signs = 2 * (score.argsort(axis=1).argsort(axis=1) >= n // 2) - 1
    x = signs * magnitude
    x -= x.mean(axis=1, keepdims=True)
    x /= np.sqrt(np.mean(x * x, axis=1, keepdims=True))
    x *= np.array([a, p])
    actual_a, actual_p = residual(x)
    if not np.allclose(actual_a, a) or not np.allclose(actual_p, p):
        raise ArithmeticError(
            "initial gauge-invariant physical coordinates differ from design"
        )
    return x


def residual(x):
    n = x.shape[1]
    amp = x[..., 0] - x[..., 0].mean(1, keepdims=True)
    left, right = np.triu_indices(n, 1)
    pair = x[:, left, 1] - x[:, right, 1]
    phase = np.angle(np.exp(1j * pair))
    return np.sqrt(np.mean(amp**2, 1)), np.sqrt(
        np.mean(phase**2, 1) * (n - 1) / (2 * n)
    )


def vector(x):
    v = np.exp(x[..., 0] + 1j * x[..., 1])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def fidelity(x):
    v = vector(x)
    return np.abs(v.sum(1)) ** 2 / v.shape[1]


def bounded(u):
    u = u - u.mean(axis=1, keepdims=True)
    ratio = np.max(np.abs(u) / np.asarray([0.35, 0.70]), axis=(1, 2))
    return u / np.maximum(ratio, 1)[:, None, None]


@lru_cache(maxsize=1)
def event_coefficients():
    q, es, bs, ei, bi = (torch.tensor(x, dtype=torch.float64) for x in CAL)
    joint = calibrate_source_weights(q, es, bs, double_probability=q.new_zeros(()))[
        "joint"
    ]

    def cells(z):
        r = torch.stack((bi, bi + (1 - bi) * ei * z, bi.new_zeros(())))
        return four_cell_probabilities(joint, r[:, None].expand(-1, 2)).numpy()

    intercept = cells(0.0)
    return intercept, cells(1.0) - intercept


def pure_prob(v):
    z = np.abs(v) ** 2
    pair = np.abs(v[:, :1, None] + v[:, 1:, None] * np.exp(-1j * ANGLES)) ** 2 / 2
    proj = np.concatenate((z, pair.reshape(len(v), -1)), axis=1)
    base, slope = event_coefficients()
    return base + proj[..., None] * slope


def observe(v, n, paid, tag, parent_ids):
    pi = pure_prob(v)
    shots = allocate_integer_budget(paid, 4 * n - 3).astype(np.int64)
    counts = np.empty(pi.shape, dtype=np.int32)
    for row, parent in enumerate(parent_ids):
        rng = np.random.Generator(
            np.random.Philox(np.random.SeedSequence([PARENT_SEED, n, tag, int(parent)]))
        )
        counts[row] = rng.multinomial(shots, pi[row])
    if not np.all(counts.sum(-1) == shots):
        raise ArithmeticError("READ accounting")
    return counts, shots


@lru_cache(maxsize=40)
def template(n, paid):
    analyzer = torch.from_numpy(projector_schedule(n)[0])[None]
    shots = torch.from_numpy(allocate_integer_budget(paid, 4 * n - 3).astype(np.int64))[
        None
    ]
    init = torch.zeros((*shots.shape, 4), dtype=torch.int64)
    init[..., 0] = shots
    cmd = torch.full((1, n), n**-0.5, dtype=torch.complex128)
    return make_public_observation(
        counts=init,
        shots=shots,
        analyzers=analyzer,
        prepared=cmd,
        calibration=torch.tensor(CAL, dtype=torch.float64)[None],
        visible_mask=torch.ones_like(shots, dtype=torch.bool),
    )


def public_inputs(n, paid, counts, command, device):
    # Fixed laboratory projectors do not rotate with the applied preparation command.
    # Only these four public fields change; the remaining tensors come directly from the canonical ABI.
    rep = {
        k: v.expand((len(counts), *v.shape[1:])) for k, v in template(n, paid).items()
    }
    cmd = torch.from_numpy(command)
    rep.update(
        counts=torch.from_numpy(counts).long(),
        prepared_re=cmd.real.float(),
        prepared_im=cmd.imag.float(),
        public_control_target=torch.stack((cmd.abs(), cmd.angle()), -1).float(),
    )
    return {k: v.to(device) for k, v in public_projector_inputs(rep).items()}


@lru_cache(maxsize=40)
def classical_operator(n, paid):
    # Ridge one-step multinomial Fisher scoring about the public ideal command.
    settings = 4 * n - 3
    dz = np.zeros((settings, 2 * n))
    dz[:n, ::2] = 2 / n * (np.eye(n) - 1 / n)
    for j in range(1, n):
        for t, angle in enumerate(ANGLES):
            k = n + 3 * (j - 1) + t
            dz[k, ::2] = -(1 + np.cos(angle)) * 2 / n**2
            dz[k, [0, 2 * j]] += (1 + np.cos(angle)) / n
            dz[k, [1, 2 * j + 1]] += np.sin(angle) / n * np.array([-1, 1])
    probs = pure_prob(np.full((1, n), 1 / np.sqrt(n)))[0]
    slope = event_coefficients()[1]
    coeff = slope / probs
    shots = allocate_integer_budget(paid, settings)
    info = dz.T @ ((shots * (coeff * slope).sum(-1))[:, None] * dz) + np.diag(
        np.tile([0.30**-2, 0.40**-2], n)
    )
    return coeff, np.linalg.solve(info, dz.T).T


def classical_action(counts, n, paid):
    coeff, op = classical_operator(n, paid)
    return bounded(-((counts * coeff).sum(-1) @ op).reshape(len(counts), n, 2))


@torch.inference_mode()
def neural_actions(model, counts, cmd, n, paid, device, fd):
    public = public_inputs(n, paid, counts, cmd, device)
    raw = model(public, N=n, fine_identifiable=False)
    rho = normalized_state(raw)
    target = torch.full((len(counts), n), n**-0.5, dtype=torch.complex128)
    plug = shared.bounded_action(rho, target).numpy()
    if not fd:
        return plug, None
    # The registered ungated FD rule ranks seven public library members including NOOP.
    ids = (
        raw["idea_fd_logits"]
        .masked_fill(~public["public_candidate_valid_mask"], -torch.inf)
        .argmax(-1)
    )
    a = (
        public["public_candidate_actions"][torch.arange(len(ids), device=device), ids]
        .cpu()
        .double()
        .numpy()
    )
    return plug, bounded(a)


def validate(n, device):
    from ml_recon.physics_hybrid_v10.effective_projector_law import (
        effective_probabilities,
    )

    x = coordinates(n, 0.35, 0.70)[:2]
    v = vector(x)
    rho = torch.from_numpy(v[:, :, None] * v[:, None, :].conj())
    av = torch.from_numpy(projector_schedule(n)[0])[None].expand(2, -1, -1)
    cal = torch.tensor(CAL, dtype=torch.float64).expand(2, -1)
    np.testing.assert_allclose(
        pure_prob(v),
        effective_probabilities(rho, av, cal).numpy(),
        atol=5e-14,
        rtol=1e-12,
    )
    counts, shots = observe(v, n, 70000, 889, [0, 1])
    cmd = vector(x * 0.2)
    ref = make_public_observation(
        counts=torch.from_numpy(counts).long(),
        shots=torch.from_numpy(shots)[None].expand(2, -1),
        analyzers=av,
        prepared=torch.from_numpy(cmd),
        calibration=cal,
        visible_mask=torch.ones((2, len(shots)), dtype=torch.bool),
    )
    fast = public_inputs(n, 70000, counts, cmd, torch.device("cpu"))
    for k, v in public_projector_inputs(ref).items():
        torch.testing.assert_close(v, fast[k], rtol=0, atol=0)
    # Analytic derivative for both public amplitude and phase controls agrees with a central difference at W.
    e = np.zeros((2, n, 2))
    e[0, 1, 0] = 1e-5
    e[1, 1, 1] = 1e-5
    fd = (pure_prob(vector(e)) - pure_prob(vector(-e))) / (2e-5)
    assert np.isfinite(fd).all() and not np.allclose(fd[0], fd[1], atol=1e-12)
    return dict(
        n=n, canonical_public_fields=len(fast), exact_shape=[n, n], settings=4 * n - 3
    )


def run_cell(task, model, device, root):
    n, k, a, p = task["N"], task["K"], task["A"], task["P"]
    initial = coordinates(n, a, p)
    policies = ["noop", "learned_plugin", "classical_fisher", "oracle_bounded"] + (
        ["learned_fd"] if k == 1 else []
    )
    rows = []
    arrays = {}
    batch = {3: 32, 4: 32, 8: 32, 16: 32, 32: 16, 64: 8, 128: 4, 256: 2}[n]
    paid = 70000 // k
    for start in range(0, ROWS, batch):
        ids = np.arange(start, min(start + batch, ROWS))
        x = initial[ids]
        start_f = fidelity(x)
        a0, p0 = residual(x)
        issued = {key: np.zeros_like(x) for key in policies}
        acts = {key: [] for key in policies}
        traces = {key: [start_f] for key in policies}
        for t in range(k):
            common = None
            fd_action = None
            for name in policies:
                u = issued[name]
                if t == 0 and common is not None:
                    counts = common
                else:
                    counts, _ = observe(vector(x + u), n, paid, 100 + k * 20 + t, ids)
                    if t == 0:
                        common = counts
                arr_key = name + "_decision_" + str(t)
                if t != 0 or name == "noop":
                    arrays.setdefault(
                        "initial_counts" if t == 0 else arr_key, []
                    ).append(counts)
                if name == "noop":
                    act = np.zeros_like(x)
                elif name == "oracle_bounded":
                    act = bounded(-(x + u))
                elif name == "classical_fisher":
                    act = classical_action(counts, n, paid)
                elif name == "learned_plugin":
                    act, fd_action = neural_actions(
                        model, counts, vector(u), n, paid, device, k == 1
                    )
                else:
                    act = fd_action
                if np.any(np.abs(act) > np.array([0.35, 0.7]) + 2e-7):
                    raise ArithmeticError("action range exceeded")
                acts[name].append(act.copy())
                issued[name] = u + act
                traces[name].append(fidelity(x + issued[name]))
        for name in policies:
            terminal, _ = observe(vector(x + issued[name]), n, 7000, 701, ids)
            arrays.setdefault(name + "_terminal_counts", []).append(terminal)
            actions = np.stack(acts[name], axis=1)
            hist = np.stack(traces[name], axis=1)
            arrays.setdefault(name + "_actions", []).append(actions.astype(np.float32))
            arrays.setdefault(name + "_fidelity", []).append(hist)
            err_a, err_p = residual(x + issued[name])
            tot = terminal.sum(1)
            for j, parent in enumerate(ids):
                magn = np.max(np.abs(actions[j]) / np.array([0.35, 0.70]), axis=(1, 2))
                change = float(hist[j, -1] - start_f[j])
                rows.append(
                    dict(
                        N=n,
                        K=k,
                        A=a,
                        P=p,
                        policy=name,
                        parent_id=int(parent),
                        seed=7301,
                        reads_decision=70000,
                        reads_terminal=7000,
                        F0=float(start_f[j]),
                        FT=float(hist[j, -1]),
                        delta_F=change,
                        amp_rms0=float(a0[j]),
                        amp_rmsT=float(err_a[j]),
                        phase_rms0=float(p0[j]),
                        phase_rmsT=float(err_p[j]),
                        harm_vs_noop=int(change < -1e-12),
                        executed=int((magn > 1e-10).any()),
                        actions_taken=int((magn > 1e-10).sum()),
                        saturated=int((magn >= 1 - 1e-6).any()),
                        terminal_00=tot[j, 0],
                        terminal_10=tot[j, 1],
                        terminal_01=tot[j, 2],
                        terminal_11=tot[j, 3],
                    )
                )
    arrays = {key: np.concatenate(parts) for key, parts in arrays.items()}
    arrays.update(
        initial_coords=initial,
        parent_ids=np.arange(ROWS),
        decision_exposure=allocate_integer_budget(paid, 4 * n - 3),
        terminal_exposure=allocate_integer_budget(7000, 4 * n - 3),
    )
    tag = task["tag"]
    (root / "rows").mkdir(exist_ok=True)
    (root / "arrays").mkdir(exist_ok=True)
    with (root / "arrays" / (tag + ".partial")).open("wb") as f:
        np.savez_compressed(f, **arrays)
    (root / "arrays" / (tag + ".partial")).replace(root / "arrays" / (tag + ".npz"))
    tmp = root / "rows" / (tag + ".partial")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(root / "rows" / (tag + ".csv.gz"))


def registered_cells():
    amplitude = (0.0, 0.12, 0.35, 0.70)
    phase = (0.0, 0.20, 0.70, 1.10)
    points = (
        (2, 2, 1),
        (2, 0, 1),
        (0, 2, 1),
        (0, 0, 1),
        (2, 2, 4),
        (1, 0, 1),
        (3, 0, 1),
        (0, 1, 1),
        (0, 3, 1),
        (1, 1, 1),
        (3, 3, 1),
        (2, 0, 4),
        (0, 2, 4),
    )
    return [
        dict(N=n, K=k, A=amplitude[a], P=phase[p], tag=f"n{n}_a{a}_p{p}_k{k}")
        for a, p, k in points
        for n in shared.SIZES
    ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("validate", "run"))
    cli.add_argument("--n", type=int, nargs="+", choices=shared.SIZES)
    cli.add_argument("--root", type=Path)
    cli.add_argument("--core", type=Path)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    sizes = args.n or shared.SIZES
    torch.set_num_threads(1)
    for n in sizes:
        print(json.dumps(validate(n, "cpu")), flush=True)
    if args.stage == "validate":
        return
    if args.root is None or args.core is None:
        cli.error("run requires --root and --core")
    model = shared.load_shared(args.core, args.device)
    cells = registered_cells()
    design = dict(
        schema="shared_true_n_recovery_v1",
        checkpoint_sha256=file_digest(args.core),
        training_panel=shared.training_panel(
            torch.load(args.core, weights_only=True, mmap=True, map_location="cpu")
        ),
        optimizer_seed=7301,
        training_sizes=[3, 4, 8],
        extrapolation_sizes=list(shared.SIZES[3:]),
        sizes=list(shared.SIZES),
        parents=ROWS,
        parent_seed=PARENT_SEED,
        cells=cells,
        reads_decision=70000,
        reads_terminal=7000,
        policies=[
            "noop",
            "learned_plugin",
            "classical_fisher",
            "oracle_bounded",
            "learned_fd",
        ],
    )
    register(args.root / "protocol.json", design)
    for task in cells:
        if task["N"] not in sizes:
            continue
        with pending_panel(args.root, task["tag"]) as pending:
            if pending:
                run_cell(task, model, torch.device(args.device), args.root)
                print(task["tag"], flush=True)


if __name__ == "__main__":
    main()
