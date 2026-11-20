"""Time-bounded, resumable sustained training of actual C-S shared-N classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .classifier_n_full import V10ClassifierNFull
from .effective_inference_experiment import atomic_save, write_json
from .effective_inference_metrics import command_and_truth, prior_density
from .effective_projector_observation import EFFECTIVE_LAW
from .long_projector_data import (
    BATCH,
    CHUNK,
    LARGE,
    SIZES,
    SMALL,
    TrainingBanks,
    build_bank,
    train_bank_namespace,
)
from .long_projector_prior import prior_for_seen
from .projector_core_export import CLASSIFIER_SCHEMA
from .training_ideas import TRAINING_IDEAS_PROFILES, TrainingIdeasConfig
from .training_ideas_experiment import _take
from .training_ideas_objective import (
    normalized_state,
    projector_classifier_step,
    projector_predictions,
    projector_primary_losses,
)

LANES = {
    "b0": ("state", LARGE),
    "b1": ("staged", LARGE),
    "b2": ("joint", LARGE),
    "b3": ("joint", SMALL),
}
SEED = 73
LR = 0.0002
EVAL_ROWS = 192
RESERVE = 45 * 60
QUALITY_RUNGS = (256, 1024, 4096, 8192, 16384, 32768, 65536, 131072)
QUALITY_MARGIN = 0.05


def utc():
    return datetime.now(timezone.utc).isoformat()


def to_device(data, device):
    return {k: v.to(device) for k, v in data.items()}


def resolved_profile(arm, requested=None):
    profile = requested or ("baseline" if arm == "state" else "all")
    if arm == "state" and profile != "baseline":
        raise ValueError("state-only runs require the baseline architecture")
    if arm in {"joint", "staged"} and profile == "baseline":
        raise ValueError(
            "use noise_ce_control for a joint objective with no optional module"
        )
    if profile not in (*TRAINING_IDEAS_PROFILES, "noise_ce_control"):
        raise ValueError(f"unsupported idea profile: {profile}")
    return profile


def constructor(
    arm, conditional_state_form="sector_trace_v1", capacity="C-S", idea_profile=None
):
    profile = resolved_profile(arm, idea_profile)
    if profile == "noise_ce_control":
        profile = "baseline"
    form = (
        {}
        if conditional_state_form == "sector_trace_v1"
        else {"conditional_state_form": conditional_state_form}
    )
    return dict(
        **form,
        profile=capacity,
        training_ideas=TrainingIdeasConfig.from_profile(profile).to_dict(),
        factor_rank=4,
        readout_domain="single_excitation_projector",
        observation_law=EFFECTIVE_LAW,
    )


def objective(model, n, batch, state_only):
    if state_only:
        out = model(batch, N=n)
        terms = projector_primary_losses(out, batch)
        loss = (
            terms["state_risk"]
            + terms["query_count_nll"]
            + 0.02 * terms["uncertainty_nll"]
        )
    else:
        terms = projector_classifier_step(model, batch, N=n)
        loss = terms["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite corrected C-S objective")
    return loss, terms


def active_loss_weights(config, state_only):
    weights = {"state_risk": 1.0, "query_count_nll": 1.0, "uncertainty_nll": 0.02}
    if state_only:
        return weights
    weights["noise_classification"] = 0.1
    if config.noise_moe:
        weights.update(
            ideas_noise_balance=config.noise_balance_weight,
            ideas_noise_severity=config.noise_severity_weight,
        )
    if config.esm:
        weights["ideas_esm"] = config.esm_weight
    if config.csm:
        weights["ideas_csm_gram"] = config.csm_weight
    if config.laplacian:
        weights["ideas_phase_laplacian"] = config.laplacian_weight
    if config.rho8:
        weights["ideas_rho8"] = config.rho8_weight
    if config.fd:
        weights.update(
            ideas_rsm=config.rsm_weight,
            ideas_fd_utility=config.fd_weight,
            ideas_fd_harm=config.fd_harm_weight,
        )
    return weights


def needs_teacher_bank(config):
    return any(config[key] for key in ("rho8", "csm", "laplacian", "fd"))


def update(model, opt, n, batch, state_only, *, record_metrics=False):
    opt.zero_grad(set_to_none=True)
    loss, terms = objective(model, n, batch, state_only)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    opt.step()
    if not record_metrics:
        return None
    weights = active_loss_weights(model.training_ideas_config, state_only)
    raw = {k: terms[k].detach() for k in weights}
    weighted = {"weighted_" + k: raw[k] * w for k, w in weights.items()}
    return {
        **raw,
        **weighted,
        "loss": loss.detach(),
        "preclip_gradient_norm": norm.detach(),
        "clip_fraction": (norm > 1.0).float(),
    }


def measure(
    arm,
    device,
    conditional_state_form="sector_trace_v1",
    capacity="C-S",
    learning_rate=LR,
    idea_profile=None,
):
    """Independent full optimizer fixture; no campaign TRAIN/VALIDATION/CONFIRMATION states."""
    cfg = constructor(arm, conditional_state_form, capacity, idea_profile)
    model = V10ClassifierNFull(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    batches = {
        n: to_device(
            build_bank(
                n, BATCH, "fixture", 1, needs_teacher_bank(cfg["training_ideas"])
            ),
            device,
        )
        for n in SIZES
    }
    for n in SIZES:
        update(model, opt, n, batches[n], arm == "state")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    for n in (*SIZES, *SIZES):
        update(model, opt, n, batches[n], arm == "state")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.monotonic() - started) / 6


def validation(model, banks, device):
    previous = model.training
    model.eval()
    result = {}
    with torch.no_grad():
        for n, data in banks.items():
            all_risks = []
            for i in range(0, len(data["counts"]), BATCH):
                idx = torch.arange(i, min(i + BATCH, len(data["counts"])))
                batch = to_device(_take(data, idx), device)
                truth = torch.complex(batch["target_R1_re"], batch["target_R1_im"])
                all_risks.extend(
                    (normalized_state(model(batch, N=n)) - truth)
                    .abs()
                    .square()
                    .sum((-2, -1))
                    .cpu()
                    .tolist()
                )
            result[str(n)] = sum(all_risks) / len(all_risks)
    model.train(previous)
    return result


def dev_means(rows, indices):
    """Aggregate the same scored state indices for the network and its TRAIN comparator."""
    count = len(indices)
    means = {"states": count}
    for name in ("state_risk", "query_cross_entropy_per_read", "train_prior_risk"):
        values = rows[name]
        means[name] = (
            None if values is None else sum(values[i] for i in indices) / count
        )
    return means


@torch.no_grad()
def development_scores(model, banks, seen, namespace, device):
    """Unselected per-state scores, grouped by simulated family on the fixed DEV panels."""
    previous = model.training
    model.eval()
    rows, summary = {}, {}
    try:
        for n, data in banks.items():
            key = str(n)
            observed = int(seen[n].sum())
            retention = prior_for_seen(n, seen[n], namespace)[0] if observed else None
            state_scores, prior_scores, count_scores = [], [], []
            for start in range(0, len(data["counts"]), BATCH):
                indices = torch.arange(start, min(start + BATCH, len(data["counts"])))
                batch = to_device(_take(data, indices), device)
                output = model(batch, N=n)
                cmd, truth = command_and_truth(batch)
                risk = (normalized_state(output) - truth).abs().square().sum((-2, -1))
                query = batch["query_counts"]
                pi = projector_predictions(output, batch)
                cross_entropy = -(query * pi.log()).sum((-2, -1)) / query.sum((-2, -1))
                state_scores.extend(risk.cpu().tolist())
                count_scores.extend(cross_entropy.cpu().tolist())
                if observed:
                    prior = prior_density(cmd, retention)
                    prior_scores.extend(
                        (prior - truth).abs().square().sum((-2, -1)).cpu().tolist()
                    )
            family = torch.where(
                data["target_is_nominal"], -1, data["target_operational_noise"]
            )
            family = family.cpu().tolist()
            rows[key] = {
                "state_index": list(range(len(family))),
                "family": family,
                "state_risk": state_scores,
                "query_cross_entropy_per_read": count_scores,
                "train_prior_risk": prior_scores if observed else None,
            }
            summary[key] = dict(
                **dev_means(rows[key], range(len(family))),
                unique_train_states=observed,
                train_prior_retention=retention,
                by_family={
                    str(f): dev_means(
                        rows[key], [i for i, v in enumerate(family) if v == f]
                    )
                    for f in sorted(set(family))
                },
            )
    finally:
        model.train(previous)
    return summary, rows


def quality_status(history, sizes=SIZES, *, finished=False):
    """First of two consecutive scheduled rungs at least 5% below the visited-TRAIN prior."""
    result = {}
    for n in sizes:
        key = str(n)
        last, preceding = None, None
        status = {"attained": False, "last_observed_step": None}
        for point in history:
            numbers = point["scores"][key]
            good = numbers["train_prior_risk"] > 0 and numbers["state_risk"] <= (
                (1 - QUALITY_MARGIN) * numbers["train_prior_risk"]
            )
            this = {"step": point["step"], "seconds": point["active_training_seconds"]}
            status.update(
                last_observed_step=this["step"], last_observed_seconds=this["seconds"]
            )
            if good and last is not None and last[1]:
                status.update(
                    attained=True,
                    bracket_start=preceding,
                    first_passing=last[0],
                    confirmed_at=this,
                )
                break
            preceding = last[0] if last is not None else {"step": 0, "seconds": 0.0}
            last = (this, good)
        status["right_censored"] = bool(finished and not status["attained"])
        result[key] = status
    return result


def restore_window_device(window, device):
    """The optimizer checkpoint is deserialized on CPU, including partial log-window sums."""
    for group in window["by_size_stage"].values():
        group["sums"] = {key: value.to(device) for key, value in group["sums"].items()}
    return window


def core_payload(cfg, state, identity, step):
    return {
        "schema": CLASSIFIER_SCHEMA,
        "classifier_config": cfg,
        "classifier_state_dict": state,
        "observation_schema": "projector_observation_v2",
        "law": EFFECTIVE_LAW,
        "sector_family": "K0_K1_reference",
        "identity": identity,
        "updates": step,
        "uncertainty": "heteroscedastic_centered_log_amplitude_wrapped_phase",
    }


def matrix_fit(path, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("invalid run_id")
    with path.open(encoding="utf-8", newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected one matrix row for {run_id}, found {len(matches)}")
    row = matches[0]
    if (
        int(row["effective_batch"]),
        int(row["train_reads_per_state"]),
        row["sizes"],
    ) != (BATCH, 4096, "3|4|8"):
        raise ValueError("unsupported batch, read budget or sizes in matrix")
    cap, arm = row["capacity"], row["objective"]
    rows, steps = int(row["train_states_per_n"]), int(row["terminal_updates"])
    stage = int(row["switch_after_update"]) if row["switch_after_update"] else None
    seed, lr = int(row["optimizer_seed"]), float(row["learning_rate"])
    if cap not in {"C-S", "C-M", "C-L"} or arm not in {"state", "joint", "staged"}:
        raise ValueError("unsupported matrix capacity or objective")
    if (
        rows < CHUNK
        or rows % CHUNK
        or steps < 1
        or seed < 0
        or not (math.isfinite(lr) and lr > 0)
    ):
        raise ValueError(
            "invalid bank size, update cap, optimizer seed or learning rate"
        )
    if (stage is not None) != (arm == "staged") or (
        stage is not None and not 0 < stage < steps
    ):
        raise ValueError("staged runs require an interior switch_after_update")
    profile = resolved_profile(arm, (row.get("idea_profile") or "").strip())
    spec = {
        "arm": arm,
        "capacity": cap,
        "rows": rows,
        "steps": steps,
        "stage": stage,
        "seed": seed,
        "lr": lr,
        "idea_profile": profile,
    }
    train_namespace = (row.get("train_namespace") or "").strip()
    if train_namespace:
        spec["train_namespace"] = train_namespace
    return spec


def restore_metric_log(path, step):
    """Drop uncheckpointed metrics after an interruption before replaying those updates."""
    if not path.exists():
        if step >= 64:
            raise ValueError(
                "resuming this matrix run also requires its training_metrics.jsonl"
            )
        return
    with path.open("rb+") as stream:
        end = previous = 0
        for line in iter(stream.readline, b""):
            if not line.endswith(b"\n"):
                break
            record_step = json.loads(line)["step"]
            if record_step > step:
                break
            if record_step != previous + 64:
                raise ValueError(
                    "training metrics are not contiguous 64-update windows"
                )
            previous, end = record_step, stream.tell()
        if previous != (step // 64) * 64:
            raise ValueError("training metrics missing before the resume checkpoint")
        stream.seek(end)
        stream.truncate()


def resource_metrics(device):
    result = {}
    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["host_process_peak_bytes"] = (
            peak if sys.platform == "darwin" else peak * 1024
        )
    if device.type == "cuda":
        result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        try:
            result["gpu_utilization_sample_percent"] = torch.cuda.utilization(device)
        except (ImportError, RuntimeError, AttributeError):
            result["gpu_utilization_sample_percent"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument("--lane", choices=tuple(LANES))
    route.add_argument("--matrix", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--split-namespace")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--hard-stop-utc",
        required=False,
        help="Shared hard stop, 10h after first cloud power-on, ISO UTC",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Network initialization and dropout seed"
    )
    parser.add_argument(
        "--conditional-state-form",
        choices=("sector_trace_v1", "factor_gram_v2"),
        default=None,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate matrix/constructor configuration without fitting or writing outputs",
    )
    args = parser.parse_args()
    fixed = args.matrix is not None
    if fixed:
        if (
            not args.run_id
            or not args.split_namespace
            or not args.split_namespace.strip()
        ):
            parser.error("--matrix requires --run-id and a fresh --split-namespace")
        spec = matrix_fit(args.matrix, args.run_id)
        arm, rows, capacity = spec["arm"], spec["rows"], spec["capacity"]
        lr, seed, run_name = spec["lr"], spec["seed"], args.run_id
        form = args.conditional_state_form or "factor_gram_v2"
        idea_profile = spec["idea_profile"]
    else:
        if not args.hard_stop_utc or args.run_id or args.split_namespace:
            parser.error(
                "--lane requires --hard-stop-utc and uses its original data namespace"
            )
        arm, rows = LANES[args.lane]
        lr, seed, capacity, run_name = LR, args.seed, "C-S", args.lane
        form = args.conditional_state_form or "sector_trace_v1"
        idea_profile = None
    cfg = constructor(arm, form, capacity, idea_profile)
    teachers = needs_teacher_bank(cfg["training_ideas"])
    if args.validate_only:
        print(
            json.dumps(
                {
                    "run": run_name,
                    "constructor": cfg,
                    "teacher_bank": teachers,
                    "schedule": spec if fixed else None,
                },
                sort_keys=True,
            )
        )
        return
    train_stop = math.inf
    if args.hard_stop_utc:
        stop = datetime.fromisoformat(args.hard_stop_utc.replace("Z", "+00:00"))
        if stop.tzinfo is None or stop.utcoffset().total_seconds() != 0:
            raise ValueError("hard stop requires UTC offset")
        train_stop = stop.timestamp() - (0 if fixed else RESERVE)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable")
    out = args.output_root / run_name
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".owner").open("a+b")
    if os.name == "nt":
        import msvcrt

        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {
        "lane": run_name,
        "arm": arm,
        "train_states_per_n": rows,
        "batch": BATCH,
        "learning_rate": lr,
        "seed": seed,
        "sizes": list(SIZES),
        "source": args.source,
        "constructor": cfg,
    }
    if fixed:
        identity.update(
            fixed_updates=spec["steps"],
            staged_primary_updates=spec["stage"],
            split_namespace=args.split_namespace,
        )
        if "train_namespace" in spec:
            identity["train_namespace"] = spec["train_namespace"]
    else:
        identity["hard_stop_utc"] = stop.isoformat()
    reg = out / "registration.json"
    latest = out / "latest.pt"
    if (out / "training_done.json").is_file():
        if json.loads(reg.read_text())["identity"] != identity:
            raise ValueError("terminal identity mismatch")
        return
    if time.time() >= train_stop:
        raise RuntimeError("shared training deadline already reached")
    if reg.is_file():
        registration = json.loads(reg.read_text())
        if registration["identity"] != identity:
            raise ValueError("incompatible registered experiment")
    else:
        torch.manual_seed(98301)
        cost = measure(arm, device, form, capacity, lr, idea_profile)
        remaining = max(0, train_stop - time.time())
        # A finite profiled ceiling with 30% headroom prevents efficient kernels
        # from idling. The shared clock has priority over the numerical ceiling.
        cap = (
            spec["steps"]
            if fixed
            else max(1, min(1048576, math.ceil(1.30 * remaining / cost)))
        )
        stage = (spec["stage"] or 0) if fixed else max(1, int(0.33 * remaining / cost))
        registration = {
            "identity": identity,
            "profile_seconds_per_update": cost,
            "step_ceiling": cap,
            "staged_primary_updates": stage,
            "train_stop_utc": (
                datetime.fromtimestamp(train_stop, timezone.utc).isoformat()
                if math.isfinite(train_stop)
                else None
            ),
            "recorded_utc": utc(),
            "torch": str(torch.__version__),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "peak_fixture_bytes": (
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None
            ),
        }
        if args.profile_only:
            print(json.dumps(registration), flush=True)
            return
        write_json(reg, registration)
    if args.profile_only:
        print(json.dumps(registration), flush=True)
        return
    torch.manual_seed(seed)
    model = V10ClassifierNFull(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    torch.manual_seed(seed + 701)
    banks = TrainingBanks(
        out / "banks", rows, teachers, split_namespace=train_bank_namespace(identity)
    )
    dev = {
        n: build_bank(
            n,
            EVAL_ROWS,
            "validation",
            0,
            teachers,
            split_namespace=args.split_namespace,
        )
        for n in SIZES
    }
    step = 0
    best = float("inf")
    best_step = 0
    active_snapshot = None
    best_snapshot = None
    log_path = out / "training_metrics.jsonl"
    window = {"by_size_stage": {}, "data_wait_seconds": 0.0, "checkpoint_seconds": 0.0}
    net_seconds = net_at_window_start = 0.0
    active_since = None
    clock_complete = True
    history = []
    gradient_names = set()
    if latest.is_file():
        ck = torch.load(latest, map_location="cpu", weights_only=True)
        if ck["identity"] != identity:
            raise ValueError("incompatible resume")
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(ck["device_rng"], device)
        step, best, best_step = ck["step"], ck["best"], ck["best_step"]
        active_snapshot = ck["active_snapshot"]
        best_snapshot = ck["best_snapshot"]
        banks.seen = ck["seen"]
        if fixed:
            window = restore_window_device(ck["metric_window"], device)
            clock_complete = ck.get("net_clock_complete", "net_training_seconds" in ck)
            net_seconds = ck.get("net_training_seconds", 0.0)
            net_at_window_start = ck.get("net_at_window_start", -ck["window_seconds"])
            history = ck.get("development_quality_history", [])
            gradient_names = set(ck["gradient_names"])
            if best_step == 0 and best_snapshot is not None:
                # Pre-upgrade matrix checkpoints could select the untrained initial score.
                positive = [
                    json.loads(path.read_text())
                    for path in sorted((out / "snapshots").glob("step_*/metrics.json"))
                ]
                candidates = [
                    record for record in positive if 0 < record["step"] <= step
                ]
                chosen = (
                    min(candidates, key=lambda x: (x["macro_risk"], x["step"]))
                    if candidates
                    else None
                )
                best, best_step, best_snapshot = (
                    (chosen["macro_risk"], chosen["step"], chosen["core"])
                    if chosen
                    else (float("inf"), 0, None)
                )
        del ck
    if fixed:
        restore_metric_log(log_path, step)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    cap = registration["step_ceiling"]
    stage = registration["staged_primary_updates"]
    last_save = last_export = last_progress = time.monotonic()

    def pause_clock():
        nonlocal net_seconds, active_since
        if active_since is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            net_seconds += time.monotonic() - active_since
            active_since = None

    def recovery():
        pause_clock()
        checkpoint_start = time.monotonic()
        atomic_save(
            dict(
                identity=identity,
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                step=step,
                updates=step,
                best=best,
                best_step=best_step,
                active_snapshot=active_snapshot,
                best_snapshot=best_snapshot,
                seen=banks.seen,
                **(
                    {
                        "metric_window": window,
                        "window_seconds": net_seconds - net_at_window_start,
                        "net_training_seconds": net_seconds,
                        "net_at_window_start": net_at_window_start,
                        "net_clock_complete": clock_complete,
                        "development_quality_history": history,
                        "gradient_names": sorted(gradient_names),
                    }
                    if fixed
                    else {}
                ),
                torch_rng=torch.get_rng_state(),
                device_rng=torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
            ),
            latest,
        )
        if fixed:
            window["checkpoint_seconds"] += time.monotonic() - checkpoint_start

    def publish(done=False):
        nonlocal best, best_step, active_snapshot, best_snapshot
        pause_clock()
        rel = Path("snapshots") / f"step_{step:09d}"
        folder = out / rel
        tmp = folder.with_name("pending_" + folder.name)
        if folder.is_dir():
            record = json.loads((folder / "metrics.json").read_text())
        else:
            tmp.mkdir(parents=True, exist_ok=True)
            if fixed:
                details, per_state = development_scores(
                    model, dev, banks.seen, train_bank_namespace(identity), device
                )
                metrics = {key: item["state_risk"] for key, item in details.items()}
                write_json(tmp / "development_rows.json", per_state)
            else:
                metrics = validation(model, dev, device)
            score = sum(metrics.values()) / len(metrics)
            if not math.isfinite(score):
                raise FloatingPointError("nonfinite development state risk")
            atomic_save(
                core_payload(cfg, model.state_dict(), identity, step), tmp / "core.pt"
            )
            record = {
                "step": step,
                "updates": step,
                "utc": utc(),
                "development_state_risk": metrics,
                "macro_risk": score,
                "core": (rel / "core.pt").as_posix(),
                "core_bytes": (tmp / "core.pt").stat().st_size,
            }
            if fixed:
                record.update(
                    development=details,
                    development_rows="development_rows.json",
                    active_training_seconds=net_seconds if clock_complete else None,
                    clock_scope="data loading and optimizer work; excludes DEV, checkpointing and downtime",
                )
            write_json(tmp / "metrics.json", record)
        active_snapshot = record["core"]
        if fixed and step in QUALITY_RUNGS and "development" in record:
            point = {
                "step": step,
                "active_training_seconds": record["active_training_seconds"],
                "scores": {
                    key: {k: item[k] for k in ("state_risk", "train_prior_risk")}
                    for key, item in record["development"].items()
                },
            }
            if history and history[-1]["step"] == step:
                history[-1] = point
            else:
                history.append(point)
        if (not fixed or step > 0) and record["macro_risk"] < best:
            best, best_step, best_snapshot = record["macro_risk"], step, active_snapshot
        # Save sufficient recovery state first. On interruption a complete pending
        # directory can be atomically committed on restore without refitting.
        recovery()
        if not folder.is_dir():
            tmp.rename(folder)
        return record

    def status(done=False):
        return dict(
            lane=run_name,
            utc=utc(),
            status="training_complete" if done else "training",
            step=step,
            step_ceiling=cap,
            stage_boundary=stage,
            training_stop_utc=registration["train_stop_utc"],
            actual_distinct_train_states={
                str(n): int(mask.sum()) for n, mask in banks.seen.items()
            },
            latest_core=active_snapshot,
            best_core=best_snapshot,
            best_validation_step=best_step,
            best_validation_risk=best if math.isfinite(best) else None,
            latest_checkpoint="latest.pt" if latest.exists() else None,
            latest_checkpoint_bytes=latest.stat().st_size if latest.exists() else None,
            **(
                {
                    "time_to_quality": quality_status(history, SIZES, finished=done),
                    "net_training_seconds": net_seconds if clock_complete else None,
                }
                if fixed
                else {}
            ),
        )

    model.train()
    if active_snapshot and not (out / active_snapshot).exists():
        if f"step_{step:09d}" not in active_snapshot:
            raise ValueError(
                "latest optimizer points at a missing older immutable snapshot"
            )
        publish()
    if fixed and not active_snapshot:
        publish()
    parameters = dict(model.named_parameters())
    total_parameters = sum(p.numel() for p in parameters.values())
    trainable_parameters = sum(
        p.numel() for p in parameters.values() if p.requires_grad
    )
    while step < cap and time.time() < train_stop:
        wait_start = time.monotonic()
        if fixed and active_since is None:
            active_since = wait_start
        seen_before = None
        if fixed and math.isfinite(train_stop):
            seen_before = banks.seen[SIZES[step % len(SIZES)]].clone()
        n, batch = banks.take(step, device)
        if time.time() >= train_stop:
            if seen_before is not None:
                banks.seen[n] = seen_before
            break
        state_only = arm == "state" or (arm == "staged" and step < stage)
        if fixed:
            window["data_wait_seconds"] += time.monotonic() - wait_start
        try:
            metrics = update(model, opt, n, batch, state_only, record_metrics=fixed)
        except (FloatingPointError, ValueError, RuntimeError) as error:
            if fixed:
                write_json(
                    out / "failure.json",
                    {"failed_update": step + 1, "n": n, "error": str(error)},
                )
            raise
        if fixed:
            label = f"N{n}_{'state' if state_only else 'joint'}"
            group = window["by_size_stage"].setdefault(
                label, {"updates": 0, "sums": {}}
            )
            group["updates"] += 1
            # Sum detached scalars on-device and synchronize once per logged window.
            for key, value in metrics.items():
                if key in group["sums"]:
                    group["sums"][key].add_(value)
                else:
                    group["sums"][key] = value.clone()
            if step < 3 or (arm == "staged" and stage <= step < stage + 3):
                gradient_names.update(
                    k for k, p in parameters.items() if p.grad is not None
                )
        step += 1
        now = time.monotonic()
        if fixed and step % 64 == 0:
            pause_clock()
            seconds = net_seconds - net_at_window_start
            grouped = {}
            for label, group in window["by_size_stage"].items():
                count = group["updates"]
                names = list(group["sums"])
                sums = torch.stack([group["sums"][k] for k in names]).cpu().tolist()
                grouped[label] = {
                    "updates": count,
                    "samples": count * BATCH,
                    "averages": {k: v / count for k, v in zip(names, sums)},
                }
            record = dict(
                step=step,
                by_size_stage=grouped,
                learning_rate=lr,
                finite=True,
                elapsed_seconds=seconds,
                updates_per_second=64 / seconds,
                cumulative_active_seconds=net_seconds if clock_complete else None,
                data_wait_seconds=window["data_wait_seconds"],
                checkpoint_seconds=window["checkpoint_seconds"],
                row_visits=step * BATCH,
                training_read_uses=step * BATCH * 4096,
                distinct_states_per_n={
                    str(n): int(m.sum()) for n, m in banks.seen.items()
                },
                parameters_total=total_parameters,
                parameters_trainable=trainable_parameters,
                parameters_receiving_gradient=sum(
                    parameters[k].numel() for k in gradient_names
                ),
                **resource_metrics(device),
            )
            record["distinct_training_reads"] = (
                sum(record["distinct_states_per_n"].values()) * 4096
            )
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, allow_nan=False) + "\n")
            window = {
                "by_size_stage": {},
                "data_wait_seconds": 0.0,
                "checkpoint_seconds": 0.0,
            }
            net_at_window_start = net_seconds
        rung = (
            step == 32
            or (step >= 256 and step & (step - 1) == 0)
            or (fixed and step in QUALITY_RUNGS)
        )
        if fixed and arm == "staged" and step in (stage, stage + 1):
            rung = True
        if rung or (not fixed and now - last_export >= 900):
            publish()
            write_json(out / "progress.json", status())
            last_save = last_export = last_progress = time.monotonic()
        elif now - last_save >= 300:
            recovery()
            write_json(out / "progress.json", status())
            last_save = last_progress = time.monotonic()
        elif now - last_progress >= 60 or step == 1:
            pause_clock()
            write_json(out / "progress.json", status())
            last_progress = now
    if fixed and step < cap:
        recovery()
        write_json(out / "progress.json", status())
        return
    if step == 0:
        write_json(out / "progress.json", status())
        raise RuntimeError("deadline before first campaign update")
    if (
        active_snapshot
        != (Path("snapshots") / f"step_{step:09d}" / "core.pt").as_posix()
    ):
        publish()
    recovery()
    write_json(out / "progress.json", status(True))
    write_json(out / "training_done.json", status(True))


if __name__ == "__main__":
    main()
