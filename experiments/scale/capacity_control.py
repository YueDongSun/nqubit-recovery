"""Compare frozen large checkpoints on the already registered raw 77k-READ parent trajectories."""

from pathlib import Path
import argparse
import csv
import gzip
import json

import numpy as np
import torch

from . import run_grid as grid, shared_scale
from experiments.protocol import file_digest, register, pending_panel

TIERS = {
    "C-S": (19709583, "joint_cs_32768_s7301"),
    "C-L": (85992847, "joint_cl_32768_s7301"),
    "C-XXL": (766045839, "xxl_joint_32768_s7301"),
}


def load_model(case, device):
    if case["tier"] == "C-XXL":
        from training.large_scale_train import register_capacity

        register_capacity("C-XXL")
    ck = torch.load(case["core"], map_location="cpu", weights_only=True, mmap=True)
    ident = ck["identity"]
    if shared_scale.training_panel(ck) != case["training_panel"]:
        raise ValueError(
            "The training split, rank, seed, or optimization setup differs from C-S"
        )
    if (
        ck["law"] != shared_scale.EFFECTIVE_LAW
        or ck["observation_schema"] != shared_scale.PUBLIC_SCHEMA
        or ident["lane"] != case["run_id"]
        or int(ck["updates"]) != 32768
        or ident["sizes"] != [3, 4, 8]
        or ident["train_states_per_n"] != 32768
        or ident["seed"] != 7301
        or ck["classifier_config"]["profile"] != case["tier"]
        or ck["classifier_config"]["factor_rank"] != 4
        or ident["learning_rate"] != 0.0002
    ):
        raise ValueError("unmatched training identity")
    with torch.device("meta"):
        model = shared_scale.V10ClassifierNFull(**ck["classifier_config"])
    model.load_state_dict(ck["classifier_state_dict"], strict=True, assign=True)
    if sum(p.numel() for p in model.parameters()) != case["parameters"]:
        raise ValueError("different parameter count")
    return model.eval().requires_grad_(False).to(device)


def evaluate(root, input_dir, cell, model, tier, device):
    n, tag = cell["N"], cell["tag"]
    with np.load(input_dir / "arrays" / (tag + ".npz")) as data:
        x = data["initial_coords"]
        counts = data["initial_counts"]
        ids = data["parent_ids"]
        shots = data["decision_exposure"]
    if (
        counts.shape != (128, 4 * n - 3, 4)
        or not np.array_equal(ids, np.arange(128))
        or not np.all(counts.sum(-1) == shots)
        or shots.sum() != 70000
    ):
        raise ValueError("paired source counts/exposure are inconsistent")
    # Relative LAPACK/libm rounding may differ when verifying a panel on another machine.
    # The original stored coordinates and integer counts drive every capacity comparison.
    if not np.allclose(
        x, grid.coordinates(n, cell["A"], cell["P"]), rtol=0, atol=3e-14
    ):
        raise ValueError("paired source parents differ")
    a0, p0 = grid.residual(x)
    f0 = grid.fidelity(x)
    actions = []
    # The high-N panel uses exactly the original frozen public fields; only checkpoint weights vary.
    batch = (
        {3: 32, 4: 32, 8: 32, 16: 32, 32: 16, 64: 8, 128: 4, 256: 2}
        if tier == "C-L"
        else {3: 16, 4: 16, 8: 16, 16: 16, 32: 8, 64: 4, 128: 2, 256: 1}
    )[n]
    for i in range(0, len(x), batch):
        cmd = grid.vector(np.zeros_like(x[i : i + batch]))
        act, _ = grid.neural_actions(
            model, counts[i : i + batch], cmd, n, 70000, device, False
        )
        actions.append(act)
    act = np.concatenate(actions)
    if np.any(np.abs(act) > np.array([0.35, 0.70]) + 3e-7):
        raise ArithmeticError("invalid bounded action")
    end = x + act
    gain = grid.fidelity(end) - f0
    ea, ep = grid.residual(end)
    terminal, terminal_exposure = grid.observe(grid.vector(end), n, 7000, 701, ids)
    total = terminal.sum(axis=1)
    rows = []
    for j, parent in enumerate(ids):
        mag = np.max(np.abs(act[j]) / np.array([0.35, 0.7]))
        rows.append(
            dict(
                N=n,
                K=1,
                A=cell["A"],
                P=cell["P"],
                capacity=tier,
                policy="learned_plugin",
                parent_id=int(parent),
                seed=7301,
                reads_decision=70000,
                reads_terminal=7000,
                F0=f0[j],
                FT=f0[j] + gain[j],
                delta_F=gain[j],
                amp_rms0=a0[j],
                amp_rmsT=ea[j],
                phase_rms0=p0[j],
                phase_rmsT=ep[j],
                harm_vs_noop=int(gain[j] < -1e-12),
                executed=int(mag > 1e-10),
                actions_taken=int(mag > 1e-10),
                saturated=int(mag >= 1 - 1e-6),
                terminal_00=total[j, 0],
                terminal_10=total[j, 1],
                terminal_01=total[j, 2],
                terminal_11=total[j, 3],
            )
        )
    name = tag + "_" + tier.lower()
    tmp = root / "arrays" / (name + ".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            actions=act,
            terminal_counts=terminal,
            parent_ids=ids,
            terminal_exposure=terminal_exposure,
            source_panel=tag,
        )
    tmp.replace(root / "arrays" / (name + ".npz"))
    tmp = root / "rows" / (name + ".partial")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(root / "rows" / (name + ".csv.gz"))


def copy_reference(primary, root, cell):
    tag = cell["tag"]
    with gzip.open(primary / "rows" / (tag + ".csv.gz"), "rt") as src:
        rows = [
            dict(row, capacity="C-S")
            for row in csv.DictReader(src)
            if row["policy"] == "learned_plugin"
        ]
    if len(rows) != grid.ROWS:
        raise ValueError(f"Missing C-S rows at {tag}")
    target = root / "rows" / (tag + "_c-s.csv.gz")
    temporary = target.with_suffix(".partial")
    with gzip.open(temporary, "wt", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--primary", required=True, type=Path, help="original C-S run_grid output"
    )
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--tier", required=True, choices=("C-L", "C-XXL"))
    cli.add_argument("--core", required=True, type=Path)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--n", nargs="+", type=int, choices=shared_scale.SIZES)
    cli.add_argument(
        "--reference-core",
        type=Path,
        help="original C-S export if the supplied protocol predates training-panel metadata",
    )
    args = cli.parse_args()
    torch.set_num_threads(1)
    cases = [
        c
        for c in grid.registered_cells()
        if c["K"] == 1 and (c["A"], c["P"]) in ((0.35, 0.70), (0.35, 0), (0, 0.70))
    ]
    primary = json.loads((args.primary / "protocol.json").read_text())
    training_panel = primary.get("training_panel")
    if training_panel is None:
        if (
            args.reference_core is None
            or file_digest(args.reference_core) != primary["checkpoint_sha256"]
        ):
            cli.error(
                "This historical protocol requires its registered --reference-core"
            )
        training_panel = shared_scale.training_panel(
            torch.load(
                args.reference_core, weights_only=True, mmap=True, map_location="cpu"
            )
        )
    models = {k: dict(tier=k, parameters=v[0], run_id=v[1]) for k, v in TIERS.items()}
    register(
        args.root / "protocol.json",
        dict(
            models=models,
            parents=grid.ROWS,
            seed=7301,
            updates=32768,
            sizes=shared_scale.SIZES,
            cells=cases,
            primary_weights=primary["checkpoint_sha256"],
        ),
    )
    register(
        args.root / (args.tier + "_weights.json"),
        dict(core_digest=file_digest(args.core)),
    )
    device = torch.device(args.device)
    model = load_model(
        dict(models[args.tier], core=args.core, training_panel=training_panel), device
    )
    for output in ("arrays", "rows"):
        (args.root / output).mkdir(exist_ok=True)
    for cell in cases:
        if args.n and cell["N"] not in args.n:
            continue
        with pending_panel(args.root, cell["tag"] + "_c-s") as pending:
            if pending:
                copy_reference(args.primary, args.root, cell)
        with pending_panel(args.root, cell["tag"] + "_" + args.tier.lower()) as pending:
            if pending:
                evaluate(args.root, args.primary, cell, model, args.tier, device)
