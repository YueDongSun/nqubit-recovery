"""Paired N=8 warm/cold adaptation of the retained shared STAR checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import paired_event_impact as impact
import star_data

ROOT = Path(__file__).resolve().parents[2]
from star_train import atom_json, atom_torch, features, owner
from unified_model import Unified_star_density_net
from nqrec.physics.single_event_mc import sample_counts

N = 8
EXPOSURE = 65536
SPLITS = {"train": (4096, 711041), "development": (512, 711059), "confirmation": (1024, 711061)}
SEEDS = (73, 79)
UPDATES = 1024
BATCH = 64
LR = 0.0003


def bank(rows, seed, device, evaluate):
    mode, visibility, rho, analyzers = impact.paired_states(N, seed=seed, rows=rows)
    new = impact.new_cells(rho, analyzers, 0.025, 0.002, "k1")
    if not torch.allclose(
        new[..., 1] + new[..., 3], new.new_full(new.shape[:2], 0.025), atol=1e-13, rtol=0
    ):
        raise ValueError("corrected total signal differs from preregistration")
    corrected = sample_counts(new, EXPOSURE, np.random.default_rng([seed, 313]))
    if not np.all(corrected.sum(-1) == EXPOSURE):
        raise ValueError("uncounted READs")
    data = {
        "truth": rho.to(device),
        "visibility": visibility,
        "corrected": torch.from_numpy(features(corrected[..., [0, 2, 1, 3]], EXPOSURE)).to(device),
    }
    if evaluate:
        archive = star_data.observation_probabilities(mode, visibility)
        old = sample_counts(torch.from_numpy(archive), EXPOSURE, np.random.default_rng([seed, 719]))
        data["archived"] = torch.from_numpy(features(old, EXPOSURE)).to(device)
    return data


def risks(model, data):
    with torch.no_grad():
        return {
            law: (model(data[law]) - data["truth"]).abs().square().sum((-2, -1)).cpu().numpy()
            for law in ("corrected", "archived")
        }


def estimate(rows, visibility):
    rows = np.asarray(rows)
    rng = np.random.default_rng(69661)
    bd = np.zeros(3000)
    for v in (0.7, 1.0):
        ix = rows[visibility == v]
        if ix.size:
            bd += rng.choice(ix, size=(len(bd), len(ix))).sum(1)
    return {
        "mean": float(rows.mean()),
        "interval95": np.quantile(bd / len(rows), [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--source", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    a = parser.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    hidden = state["leaf_encoder.0.weight"].shape[0]
    rank = (state["leaf_decoder.2.bias"].numel() - 1) // 2

    def new_model():
        return Unified_star_density_net(hidden=hidden, rank_columns=rank, device=device)

    frozen = new_model()
    frozen.load_state_dict(state, strict=True)
    # Separate fixture states verify tensor order, strict load, and finite gradients.
    small = bank(8, 199173, device, True)
    fixture_loss = (frozen(small["corrected"]) - small["truth"]).abs().square().sum((-2, -1)).mean()
    fixture_loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in frozen.parameters()):
        raise FloatingPointError("invalid engineering gradient")
    frozen.zero_grad()
    frozen.eval()
    if a.smoke_only:
        print(
            json.dumps(
                dict(
                    status="smoke_passed",
                    actual_updates=0,
                    device=str(device),
                    hidden=hidden,
                    rank=rank,
                    fixture_loss=float(fixture_loss.detach()),
                    torch_version=str(torch.__version__),
                )
            ),
            flush=True,
        )
        return
    a.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "source": a.source,
        "parent_checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "n": N,
        "shots_per_setting": EXPOSURE,
        "settings": 2 * (N - 1),
        "seeds": list(SEEDS),
        "updates_per_fit": UPDATES,
        "batch": BATCH,
        "learning_rate": LR,
        "splits": {k: list(v) for k, v in SPLITS.items()},
    }
    with owner(a.output):
        card = a.output / "registration.json"
        if card.is_file():
            if json.loads(card.read_text()) != identity:
                raise ValueError("recovery identity mismatch")
        else:
            atom_json(card, identity)
        if (a.output / "result.json").is_file():
            return
        train = bank(*SPLITS["train"], device, False)
        results = {}
        before = {}
        for seed in SEEDS:
            for arm in ("warm", "cold"):
                key = f"{arm}_s{seed}"
                torch.manual_seed(seed)
                model = new_model()
                if arm == "warm":
                    model.load_state_dict(state, strict=True)
                before[key] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
                index = torch.Generator().manual_seed(seed + 437)
                folder = a.output / key
                folder.mkdir(exist_ok=True)
                ckpt = folder / "latest.pt"
                core = folder / "core.pt"
                updates, seconds = 0, 0.0
                if ckpt.exists():
                    last = torch.load(ckpt, map_location="cpu", weights_only=True)
                    if last["identity"] != identity or last["key"] != key:
                        raise ValueError("incompatible optimizer recovery")
                    model.load_state_dict(last["model"], strict=True)
                    optimizer.load_state_dict(last["optimizer"])
                    index.set_state(last["index"])
                    updates, seconds = last["updates"], last["seconds"]
                start = time.perf_counter()
                model.train()
                for step in range(updates, UPDATES):
                    take = torch.randint(len(train["truth"]), (BATCH,), generator=index).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    difference = model(train["corrected"][take]) - train["truth"][take]
                    loss = difference.abs().square().sum((-2, -1)).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite objective at {key}:{step+1}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 10.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    if (step + 1) % 128 == 0:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        atom_torch(
                            ckpt,
                            dict(
                                identity=identity,
                                key=key,
                                model=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                index=index.get_state(),
                                updates=step + 1,
                                seconds=seconds + time.perf_counter() - start,
                            ),
                        )
                seconds += time.perf_counter() - start
                atom_torch(core, model.state_dict())
                results[key] = dict(actual_updates=UPDATES, optimizer_seconds=seconds)
        # Open fresh confirmation only after every predeclared fit has been finalized.
        for role in ("development", "confirmation"):
            data = bank(*SPLITS[role], device, True)
            vis = data["visibility"]
            truth = data["truth"]
            identity_risk = (
                (truth - torch.eye(N, dtype=truth.dtype, device=device) / N)
                .abs()
                .square()
                .sum((-2, -1))
                .cpu()
                .numpy()
            )
            command_risk = (
                (truth - torch.ones(N, N, dtype=truth.dtype, device=device) / N)
                .abs()
                .square()
                .sum((-2, -1))
                .cpu()
                .numpy()
            )
            base = risks(frozen, data)
            rows = {
                "visibility": vis.tolist(),
                "phase_symmetric_prior": identity_risk.tolist(),
                "uniform_W_command": command_risk.tolist(),
                "frozen": {k: v.tolist() for k, v in base.items()},
            }
            seed_scores = {}
            for key in results:
                model = new_model().eval()
                model.load_state_dict(before[key], strict=True)
                initial = risks(model, data)
                model.load_state_dict(
                    torch.load(a.output / key / "core.pt", map_location=device, weights_only=True),
                    strict=True,
                )
                final = risks(model, data)
                seed_scores[key] = final
                rows[key] = {
                    "initial": {k: v.tolist() for k, v in initial.items()},
                    "final": {k: v.tolist() for k, v in final.items()},
                }
                results[key][role] = {
                    law: {
                        "initial": estimate(initial[law], vis),
                        "terminal": estimate(values, vis),
                        "terminal_minus_initial": estimate(values - initial[law], vis),
                        "terminal_minus_frozen": estimate(values - base[law], vis),
                        "terminal_minus_phase_symmetric_prior": estimate(
                            values - identity_risk, vis
                        ),
                        "terminal_minus_uniform_W_command": estimate(values - command_risk, vis),
                    }
                    for law, values in final.items()
                }
            pooled = {}
            for law in base:
                warm = np.mean([seed_scores[f"warm_s{s}"][law] for s in SEEDS], axis=0)
                cold = np.mean([seed_scores[f"cold_s{s}"][law] for s in SEEDS], axis=0)
                pooled[law] = {
                    k: estimate(v, vis)
                    for k, v in dict(
                        frozen=base[law],
                        warm=warm,
                        cold=cold,
                        uniform_W_command=command_risk,
                        phase_symmetric_prior=identity_risk,
                        warm_minus_cold=warm - cold,
                        warm_minus_frozen=warm - base[law],
                        warm_minus_phase_symmetric_prior=warm - identity_risk,
                        cold_minus_phase_symmetric_prior=cold - identity_risk,
                    ).items()
                }
            atom_json(a.output / f"{role}_states.json", rows)
            atom_json(a.output / f"{role}_paired.json", pooled)
        record = dict(
            identity=identity,
            arms=results,
            actual_updates=4 * UPDATES,
            torch_version=str(torch.__version__),
            device=str(device),
            gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            complete_per_state_reads=EXPOSURE * 2 * (N - 1),
            train_unique_states=SPLITS["train"][0],
            shared_data_across_arms=True,
        )
        atom_json(a.output / "result.json", record)


if __name__ == "__main__":
    main()
