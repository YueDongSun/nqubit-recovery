"""Train one shared high-N amplitude estimator or qualify its exact computational path."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F

if __package__:
    from .amp_data import population_probabilities
    from .amp_fit import encode_observations
    from .long_model import AmplitudeSetModel, torch_centered_action, torch_conditional_fidelity
else:
    from amp_data import population_probabilities
    from amp_fit import encode_observations
    from long_model import AmplitudeSetModel, torch_centered_action, torch_conditional_fidelity


SIZES = (32, 64, 128, 256)
BUDGETS = ((30, 1), (300, 16))
CELLS = tuple((n, *budget) for n in SIZES for budget in BUDGETS)
BATCH = 64
WIDTH = 48
BOUND = 0.25
LR = 3e-4
WALL_CAP = 24 * 3600
FORMAL_STEPS = 180000
VALID_INTERVAL = 3000
SAVE_INTERVAL = 600
QUAL_STEPS = 64
OBJECTIVE = "equal cell mean of KL(true || estimated population) minus true bounded-action conditional W gain"


def atomic_json(path, record):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_state(path, record):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(record, temporary)
    os.replace(temporary, path)


@contextmanager
def run_owner(directory):
    """Hold a process lock across setup, training, checkpoint publication and scoring."""
    with (directory / "owner.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare_banks(banks, qualify):
    prepared = {}
    for role in ("train", "validation"):
        for cell in CELLS:
            raw = banks[(role, *cell)]
            row_count = (64 if role == "train" else 32) if qualify else (8192 if role == "train" else 512)
            if raw["population"].shape != (row_count, cell[0]) or raw["metadata"]["family"] != "structured":
                raise ValueError(f"Unexpected rows or family for {role} {cell}")
            public = {k: raw[k] for k in ("counts", "efficiency", "log_weight", "metadata", "cell_index")}
            features = torch.from_numpy(encode_observations(public))
            if features.shape != (row_count, cell[0], 10) or features.dtype != torch.float64:
                raise ValueError("Unexpected public calibration/count feature layout")
            prepared[(role, *cell)] = (features, torch.from_numpy(raw["population"]), public if role == "validation" else None)
    return prepared


def make_recipe(args, bank_id):
    device_name = torch.cuda.get_device_name(0) if args.device == "cuda" else platform.machine()
    sources = {}
    for name in ("long_fit.py", "long_model.py", "long_campaign.py", "amp_fit.py", "amp_data.py", "amp_control.py"):
        source = Path(__file__).with_name(name)
        sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return {"schema": "high_n_amplitude_long_v1", "bank_id": bank_id, "source_sha256": sources,
            "mode": args.mode, "arm": args.arm, "seed": args.seed, "microbatch": args.microbatch,
            "logical_batch": BATCH, "width": WIDTH, "precision": "float64", "learning_rate": LR,
            "bound": BOUND, "loss": OBJECTIVE, "device": args.device, "device_name": device_name,
            "torch": str(torch.__version__), "cuda": str(torch.version.cuda), "numpy": np.__version__,
            "python": platform.python_version(), "wall_cap_seconds": WALL_CAP, "cell_order": list(map(list, CELLS)),
            "max_updates": QUAL_STEPS if args.mode == "qualify" else FORMAL_STEPS,
            "validate_every": QUAL_STEPS // 2 if args.mode == "qualify" else VALID_INTERVAL,
            "checkpoint_every": QUAL_STEPS // 2 if args.mode == "qualify" else SAVE_INTERVAL,
            "sampler": "fixed cyclic (N,shots,rounds), PCG64 uniform with replacement within each cell"}


def batch_scores(model, x, p):
    logp = model(x)
    if logp.shape != p.shape:
        raise ValueError("Estimator did not return one log population per addressed mode")
    action = torch_centered_action(logp, BOUND)
    after = torch_conditional_fidelity(p, action)
    before = torch_conditional_fidelity(p, torch.zeros_like(action))
    kl = F.kl_div(logp, p, reduction="none").sum(-1)
    return kl, after - before, logp


def optimizer_step(model, optimizer, rng, bank, device, microbatch):
    x, p, _ = bank
    indices = torch.from_numpy(rng.integers(0, len(x), BATCH))
    mini_x, mini_p = x[indices].to(device), p[indices].to(device)
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, BATCH, microbatch):
        section = slice(start, start + microbatch)
        kl, gain, _ = batch_scores(model, mini_x[section], mini_p[section])
        ((kl - gain).sum() / BATCH).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
    optimizer.step()
    return norm


@torch.no_grad()
def validation(model, prepared, device, microbatch, check_time, with_likelihood):
    model.eval()
    cells = {}
    for cell in CELLS:
        check_time()
        x, truth, public = prepared[("validation", *cell)]
        values = []
        for start in range(0, len(x), microbatch):
            region = slice(start, start + microbatch)
            p = truth[region].to(device)
            kl, gain, logp = batch_scores(model, x[region].to(device), p)
            err = (logp.exp() - p).square().sum(-1)
            metrics = torch.stack((kl, gain, err, (gain < -1.e-12).double()), -1).cpu().numpy()
            if with_likelihood:
                estimates = logp.exp().cpu().numpy()
                uniform = np.full_like(estimates, 1 / cell[0])
                eff = public["efficiency"][region]
                meta = public["metadata"]
                pi = population_probabilities(estimates, eff, meta)
                ref = population_probabilities(uniform, eff, meta)
                if np.min(pi) <= 0 or np.min(ref) <= 0:
                    raise ArithmeticError("Positive four-cell predictions required")
                # This is a fitted-observation diagnostic: counts were inputs.
                ll_gain = (public["counts"][region] * np.log(pi / ref)).sum(axis=(1, 2))
                metrics = np.column_stack((metrics, ll_gain / (cell[0] * meta["trials"])))
            values.append(metrics)
        rows = np.concatenate(values)
        if not np.isfinite(rows).all():
            raise FloatingPointError("Nonfinite validation score")
        mean = rows.mean(0)
        key = f"n{cell[0]}_c{cell[1]}_k{cell[2]}"
        cells[key] = {"rows": len(rows), "kl": mean[0], "gain": mean[1], "population_sse": mean[2],
                      "harmed_rows": int(rows[:, 3].sum()), "harm_rate": mean[3]}
        if with_likelihood:
            cells[key]["fitted_log_likelihood_gain_per_read_vs_uniform"] = mean[4]
    model.train()
    objective = np.mean([v["kl"] - v["gain"] for v in cells.values()])
    return {"objective": float(objective), "cells": cells}


def weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def capture(model, optimizer, rng, device, recipe, step, best, history, timing):
    return {"recipe": recipe, "step": step, "next_cell": step % len(CELLS), "weights": weights(model),
            "optimizer": optimizer.state_dict(), "sampler": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "best": best, "validation_history": history, "timing": timing}


def restore(path, model, optimizer, rng, device, recipe):
    state = torch.load(path, weights_only=True, map_location="cpu")
    step = state["step"]
    if (state["recipe"] != recipe or type(step) is not int or not 0 <= step <= recipe["max_updates"]
            or state["next_cell"] != step % len(CELLS)):
        raise ValueError("Resume bank, recipe, step or sampler cycle is incompatible")
    model.load_state_dict(state["weights"])
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["sampler"]
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    return state


def run(args):
    if __package__:
        from .long_campaign import load_banks
    else:
        from long_campaign import load_banks
    output = Path(args.output)
    if args.resume and not output.is_dir():
        raise ValueError("No previous run directory to resume")
    output.mkdir(exist_ok=args.resume, parents=True)
    with run_owner(output):
        if (output / "result.json").exists():
            raise ValueError("A completed run already owns this directory")
        origin_path = output / "origin.json"
        if args.resume:
            origin = json.loads(origin_path.read_text(encoding="utf-8"))
        else:
            started_at = time.time()
            origin = {"started_unix": started_at, "deadline_unix": started_at + WALL_CAP}
            atomic_json(origin_path, origin)
        monotonic = time.monotonic()
        remaining = origin["deadline_unix"] - time.time()
        def check_time():
            if time.monotonic() - monotonic >= remaining or time.time() >= origin["deadline_unix"]:
                raise TimeoutError("Initial cumulative wall-clock budget exhausted")
        check_time()
        qualify = args.mode == "qualify"
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device unavailable")
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        bank_id, banks = load_banks(Path(args.bank_dir), qualification=qualify)
        prepared = prepare_banks(banks, qualify)
        del banks
        recipe = make_recipe(args, bank_id)
        model = AmplitudeSetModel(args.arm, feature_dim=10, width=WIDTH).to(device=device, dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        latest = output / "latest.pt"
        if args.resume:
            state = restore(latest, model, optimizer, rng, device, recipe)
            step, best, history, timing = (state[k] for k in ("step", "best", "validation_history", "timing"))
            del state
        else:
            step, best, history = 0, None, []
            timing = {"by_cell_s": [0.0]*len(CELLS), "validation_s": 0., "rss_bytes": 0}
            save_state(latest, capture(model, optimizer, rng, device, recipe, step, best, history, timing))
        check_time()
        process = psutil.Process()
        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        replayed = False
        try:
            while step < recipe["max_updates"]:
                check_time()
                ci = step % len(CELLS)
                t0 = time.monotonic()
                optimizer_step(model, optimizer, rng, prepared[("train", *CELLS[ci])], device, args.microbatch)
                if qualify:
                    sync()
                timing["by_cell_s"][ci] += time.monotonic() - t0
                step += 1
                if qualify or step % SAVE_INTERVAL == 0:
                    timing["rss_bytes"] = max(timing["rss_bytes"], process.memory_info().rss)
                validate_now = step % recipe["validate_every"] == 0
                if validate_now:
                    t0 = time.monotonic()
                    report = validation(model, prepared, device, args.microbatch, check_time, qualify)
                    timing["validation_s"] += time.monotonic() - t0
                    history.append({"step": step, "objective": report["objective"]})
                    if best is None or report["objective"] < best["objective"]:
                        best = {"step": step, "objective": report["objective"], "weights": weights(model)}
                if validate_now or step % recipe["checkpoint_every"] == 0:
                    save_state(latest, capture(model, optimizer, rng, device, recipe, step, best, history, timing))
                    atomic_json(output / "status.json", {"state": "running", "step": step, "latest_checkpoint": latest.name,
                                "cumulative_wall_s": time.time() - origin["started_unix"], "deadline_unix": origin["deadline_unix"]})
                    if qualify and step == recipe["validate_every"]:
                        next_bank = prepared[("train", *CELLS[step % len(CELLS)])]
                        optimizer_step(model, optimizer, rng, next_bank, device, args.microbatch)
                        reference = weights(model)
                        restore(latest, model, optimizer, rng, device, recipe)
                        optimizer_step(model, optimizer, rng, next_bank, device, args.microbatch)
                        if not all(torch.equal(reference[k], v) for k, v in weights(model).items()):
                            raise ArithmeticError("Checkpoint replay changed the first subsequent optimizer update")
                        restore(latest, model, optimizer, rng, device, recipe)
                        replayed = True
            check_time()
            terminal_report = validation(model, prepared, device, args.microbatch, check_time, True)
            model.load_state_dict(best["weights"])
            selected_report = validation(model, prepared, device, args.microbatch, check_time, True)
            save_state(output / "selected.pt", {"weights": best["weights"], "step": best["step"], "recipe": recipe})
            check_time()
            result = {"state": "completed", "recipe": recipe, "origin": origin, "step": step,
                      "updates_per_cell": step // len(CELLS), "rows_per_update": BATCH,
                      "cumulative_wall_s": time.time() - origin["started_unix"], "timing": timing,
                      "parameter_count": sum(p.numel() for p in model.parameters()),
                      "selected_step": best["step"], "selected_validation": selected_report,
                      "terminal_validation": terminal_report, "validation_history": history,
                      "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
            if qualify:
                result["replayed_identical_optimizer_update_in_this_invocation"] = replayed
                result["projected_training_s"] = sum(timing["by_cell_s"]) / step * FORMAL_STEPS
                result["projected_validation_s"] = timing["validation_s"] / len(history) * 16 * (FORMAL_STEPS // VALID_INTERVAL)
                result["projection_scope"] = "Same model, microbatch and all shapes. Linear extrapolation, excludes remote contention, checkpoint I/O and bank setup."
            atomic_json(output / "result.json", result)
            atomic_json(output / "status.json", {"state": "completed", "step": step, "deadline_unix": origin["deadline_unix"]})
        except BaseException as exc:
            atomic_json(output / "status.json", {"state": "interrupted", "attempt_step": step, "error": repr(exc),
                                                "deadline_unix": origin["deadline_unix"]})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qualify", "train"), required=True)
    parser.add_argument("--arm", choices=("first_order", "second_order"), required=True)
    parser.add_argument("--seed", type=int, choices=(61, 62, 63), required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--microbatch", type=int, choices=(16, 32), default=16)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
