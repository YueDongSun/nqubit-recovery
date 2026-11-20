"""FP32 CUDA scale extension of the corrected-projector trainer."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

SOURCE = "public-ff2a-projector-v1"
SPLIT = "compshare_scale_20261120"
RUNGS = (0, 1024, 4096, 8192, 16384, 32768, 65536)
CAPACITIES = {
    "C-XL": (512, 12, 6, 16),
    "C-XXL": (768, 16, 8, 16),
    "C-XL-D": (384, 22, 11, 16),
    "C-XL-W": (704, 6, 3, 16),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("qualify", "train"))
    p.add_argument("--capacity", choices=tuple(CAPACITIES), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--arm", choices=("state", "joint"), default="joint")
    p.add_argument("--factor-rank", type=int, choices=(4, 8), default=4)
    p.add_argument("--seed", type=int, default=7301)
    p.add_argument("--learning-rate", type=float, default=0.0002)
    p.add_argument("--steps", type=int, default=32768)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument(
        "--checkpoint-root", type=Path, help="Restart and final model directory (default: OUTPUT_ROOT/checkpoints)"
    )
    p.add_argument(
        "--bank-root", type=Path, help="Per-run deterministic TRAIN cache (default: OUTPUT_ROOT/banks)"
    )
    p.add_argument("--split-namespace", choices=(SPLIT,), default=SPLIT)
    p.add_argument("--fit-stop-utc", help="Optional stop time in ISO 8601 UTC")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--checkpoint-seconds", type=int, default=300)
    return p


def register_capacity(capacity):
    from ml_recon.physics_hybrid_v10.classifier_n_full import (
        CLASSIFIER_PROFILES,
        ClassifierProfile,
    )

    declaration = ClassifierProfile(*CAPACITIES[capacity])
    if CLASSIFIER_PROFILES.setdefault(capacity, declaration) != declaration:
        raise ValueError("classifier profile differs from its registered dimensions")
    return declaration


def learning_rate_at(base, completed, warmup):
    """The first update uses base/warmup and the warmup-th update uses base."""
    return base if not warmup else base * min(completed + 1, warmup) / warmup


def training_source(args):
    import torch

    from ml_recon.physics_hybrid_v10 import long_projector_training as train

    register_capacity(args.capacity)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("large scale runs require a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("scale extension requires FP32")
    return torch, train, device


def configuration(train, args):
    config = train.constructor(args.arm, "factor_gram_v2", args.capacity)
    config["factor_rank"] = args.factor_rank
    return config


def new_model(torch, train, args, config, device):
    torch.manual_seed(args.seed)
    model = train.V10ClassifierNFull(**config).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    torch.manual_seed(args.seed + 701)
    return model, opt


def qualify(args, torch, train, device, out):
    cfg = configuration(train, args)
    model, opt = new_model(torch, train, args, cfg, device)
    model.train()
    teachers = train.needs_teacher_bank(cfg["training_ideas"])
    fixture = {
        n: train.to_device(
            train.build_bank(
                n,
                train.BATCH,
                "fixture",
                1,
                teachers,
                split_namespace="scale_engineering_only_20261120",
            ),
            device,
        )
        for n in train.SIZES
    }
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    for repeat in range(3):
        for n in train.SIZES:
            if time.time() >= deadline(args.fit_stop_utc):
                raise TimeoutError("qualification cutoff reached")
            torch.cuda.synchronize(device)
            t = time.perf_counter()
            metrics = train.update(
                model, opt, n, fixture[n], args.arm == "state", record_metrics=True
            )
            torch.cuda.synchronize(device)
            loss = float(metrics["loss"])
            grad = float(metrics["preclip_gradient_norm"])
            if not math.isfinite(loss) or not math.isfinite(grad) or grad <= 0:
                raise FloatingPointError("invalid real CUDA qualification update")
            records.append(
                dict(
                    n=n,
                    seconds=time.perf_counter() - t,
                    loss=loss,
                    preclip_gradient_norm=grad,
                    warmup=repeat == 0,
                )
            )
    times = [row["seconds"] for row in records if not row["warmup"]]
    result = dict(
        role="engineering_fixture_only",
        source=SOURCE,
        capacity=args.capacity,
        architecture=asdict(model.profile),
        constructor=cfg,
        parameters=sum(p.numel() for p in model.parameters()),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device),
        device_uuid=str(torch.cuda.get_device_properties(device).uuid),
        batch=train.BATCH,
        device_total_bytes=torch.cuda.get_device_properties(device).total_memory,
        dtype=str(next(model.parameters()).dtype),
        learning_rate=args.learning_rate,
        seed=args.seed,
        step_records=records,
        timed_updates_per_second=len(times) / sum(times),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )
    train.write_json(out / "qualification.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "capacity",
                    "parameters",
                    "timed_updates_per_second",
                    "peak_reserved_bytes",
                )
            }
        ),
        flush=True,
    )


def deadline(value):
    if value is None:
        return math.inf
    when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if when.tzinfo is None or when.utcoffset().total_seconds() != 0:
        raise ValueError("use an absolute UTC cutoff")
    return when.timestamp()


def main():
    import fcntl

    args = parser().parse_args()
    args.bank_root = args.bank_root or args.output_root / "banks"
    args.checkpoint_root = args.checkpoint_root or args.output_root / "checkpoints"
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id)
        or args.seed < 0
        or args.steps < 1
        or not 0 <= args.warmup_steps <= args.steps
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or args.threads < 1
        or args.checkpoint_seconds < 1
    ):
        raise ValueError("invalid registered run settings")
    end = deadline(args.fit_stop_utc)
    if time.time() >= end:
        raise RuntimeError("requested stop time has passed")
    out = args.output_root.resolve() / args.run_id
    out.mkdir(exist_ok=True, parents=True)
    with (out / ".owner").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch, train, device = training_source(args)
        if args.mode == "qualify":
            qualify(args, torch, train, device, out)
        else:
            run_training(args, torch, train, device, out, end)


def evaluation_steps(ceiling):
    return tuple(s for s in RUNGS if s <= ceiling)


def shared_banks(args, train, teachers):
    banks = train.TrainingBanks(
        args.bank_root.resolve() / args.run_id, 32768, teachers, split_namespace=args.split_namespace
    )
    return banks, banks.take


def add_window(window, n, label, values):
    group = window["by_size_stage"].setdefault(
        f"N{n}_{label}", {"updates": 0, "sums": {}}
    )
    group["updates"] += 1
    for key, value in values.items():
        if key in group["sums"]:
            group["sums"][key].add_(value)
        else:
            group["sums"][key] = value.detach().clone()


def new_window():
    return {"by_size_stage": {}, "data_wait_seconds": 0.0}


def reduced_window(torch, window):
    result = {}
    for label, item in window["by_size_stage"].items():
        count = item["updates"]
        sums = item["sums"]
        values = torch.stack(list(sums.values())).detach().cpu().tolist()
        if not all(math.isfinite(x) for x in values):
            raise FloatingPointError("nonfinite logged component")
        means = {k: v / count for k, v in zip(sums, values)}
        result[label] = dict(updates=count, samples=32 * count, averages=means)
    return result


def run_training(args, torch, train, device, out, end):
    import signal

    from training.large_scale_evaluate import score_model_at_step
    from ml_recon.physics_hybrid_v10.matrix_campaign import register_once

    cfg = configuration(train, args)
    teachers = train.needs_teacher_bank(cfg["training_ideas"])
    identity = dict(
        lane=args.run_id,
        run_id=args.run_id,
        source=SOURCE,
        arm=args.arm,
        capacity=args.capacity,
        seed=args.seed,
        learning_rate=args.learning_rate,
        constructor=cfg,
        architecture=list(CAPACITIES[args.capacity]),
        fixed_updates=args.steps,
        sizes=list(train.SIZES),
        batch=train.BATCH,
        train_states_per_n=32768,
        reads_per_train_state=4096,
        split_namespace=args.split_namespace,
        factor_rank=args.factor_rank,
        dtype="torch.float32",
        schedule="linear_warmup_then_constant_adamw"
        if args.warmup_steps
        else "constant_adamw",
        warmup_steps=args.warmup_steps,
        weight_decay=1e-5,
        clip_norm=1.0,
        evaluation_steps=list(evaluation_steps(args.steps)),
    )
    # Code identities are portable across node and device changes.
    import hashlib
    from inspect import getfile, unwrap

    with Path(__file__).open("rb") as f:
        identity["trainer_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    with Path(getfile(unwrap(score_model_at_step))).open("rb") as f:
        identity["evaluation_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    register_once(out / "registration.json", {"identity": identity})
    if (out / "training_done.json").is_file():
        return
    store = args.checkpoint_root.resolve() / args.run_id
    store.mkdir(parents=True, exist_ok=True)
    latest = store / "latest.pt"
    core = store / "final_core.pt"
    log = out / "training_metrics.jsonl"
    model, opt = new_model(torch, train, args, cfg, device)
    params = sum(p.numel() for p in model.parameters())
    banks, take = shared_banks(args, train, teachers)
    dev = {
        n: train.build_bank(
            n,
            train.EVAL_ROWS,
            "validation",
            0,
            teachers,
            split_namespace=args.split_namespace,
        )
        for n in train.SIZES
    }
    window = new_window()
    step = 0
    net = 0.0
    window_start = 0.0
    if latest.is_file():
        ck = torch.load(latest, map_location="cpu", weights_only=True)
        if ck["identity"] != identity:
            raise ValueError("resume source/recipe differs")
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["device_rng"], device)
        banks.seen = ck["seen"]
        step = int(ck["step"])
        net = ck["active_seconds"]
        window_start = ck["window_active_start"]
        window = train.restore_window_device(ck["metric_window"], device)
        del ck
    elif (out / "checkpoint.json").is_file():
        raise FileNotFoundError(
            "registered optimizer state must be restored before resuming"
        )
    train.restore_metric_log(log, step)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    interruption = []

    def stop(sig, _):
        interruption.append(sig)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    active = None
    last_save = time.monotonic()
    last_progress = last_save

    def pause():
        nonlocal net, active
        if active is not None:
            torch.cuda.synchronize(device)
            net += time.monotonic() - active
            active = None

    def progress(status):
        return dict(
            run_id=args.run_id,
            step=step,
            updates=step,
            step_ceiling=args.steps,
            status=status,
            utc=train.utc(),
            active_seconds=net,
            parameters=params,
            device=str(device),
            gpu=torch.cuda.get_device_name(device),
            distinct_states_per_n={
                str(n): int(mask.sum()) for n, mask in banks.seen.items()
            },
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        )

    def save():
        nonlocal last_save
        pause()
        train.atomic_save(
            dict(
                identity=identity,
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                step=step,
                seen=banks.seen,
                torch_rng=torch.get_rng_state(),
                device_rng=torch.cuda.get_rng_state(device),
                metric_window=window,
                active_seconds=net,
                window_active_start=window_start,
            ),
            latest,
        )
        train.write_json(
            out / "checkpoint.json",
            dict(
                step=step,
                bytes=latest.stat().st_size,
                path=str(latest),
                utc=train.utc(),
            ),
        )
        last_save = time.monotonic()

    def evaluate():
        pause()
        target = out / "scores" / f"step_{step:09d}"
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "evaluation_done.json"
        if marker.exists():
            rec = json.loads(marker.read_text())
            if rec["step"] != step or rec["identity"] != identity:
                raise ValueError("stored assessment differs from the registered run")
            return
        was_training = model.training
        tick = time.monotonic()
        try:
            with torch.random.fork_rng(devices=[device.index]):
                model.eval()
                outcome = score_model_at_step(
                    model,
                    step=step,
                    identity=identity,
                    dev_banks=dev,
                    seen=banks.seen,
                    device=device,
                    output_dir=target,
                )
        finally:
            model.train(was_training)
        train.write_json(
            marker,
            dict(
                step=step,
                identity=identity,
                active_seconds=net,
                evaluation_seconds=time.monotonic() - tick,
                summary=outcome,
            ),
        )

    rungs = set(identity["evaluation_steps"])
    if step in rungs:
        evaluate()
    while step < args.steps and time.time() < end and not interruption:
        if active is None:
            active = time.monotonic()
        n = train.SIZES[step % len(train.SIZES)]
        before = banks.seen[n].clone()
        tick = time.monotonic()
        n, batch = take(step, device)
        window["data_wait_seconds"] += time.monotonic() - tick
        if time.time() >= end or interruption:
            banks.seen[n] = before
            break
        actual_lr = learning_rate_at(args.learning_rate, step, args.warmup_steps)
        if args.warmup_steps:
            for group in opt.param_groups:
                group["lr"] = actual_lr
        try:
            values = train.update(
                model, opt, n, batch, args.arm == "state", record_metrics=True
            )
            add_window(window, n, args.arm, values)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            train.write_json(
                out / "failure.json",
                dict(
                    failed_update=step + 1,
                    last_successful_step=step,
                    n=n,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    finite=False
                    if isinstance(exc, FloatingPointError) or "non-finite" in str(exc)
                    else None,
                    last_recovery=str(latest) if latest.exists() else None,
                ),
            )
            raise
        step += 1
        if step % 64 == 0:
            pause()
            elapsed = net - window_start
            record = progress("training")
            record.update(
                by_size_stage=reduced_window(torch, window),
                finite=True,
                learning_rate=actual_lr,
                target_learning_rate=args.learning_rate,
                elapsed_seconds=elapsed,
                updates_per_second=64 / elapsed,
                data_wait_seconds=window["data_wait_seconds"],
                row_visits=step * train.BATCH,
                training_read_uses=step * train.BATCH * 4096,
                active_loss_weights=train.active_loss_weights(
                    model.training_ideas_config, args.arm == "state"
                ),
            )
            with log.open("a") as fp:
                fp.write(json.dumps(record, allow_nan=False) + "\n")
            window = new_window()
            window_start = net
        if step in rungs:
            save()
            evaluate()
        elif time.monotonic() - last_save >= args.checkpoint_seconds:
            save()
        if step == 1 or step % 64 == 0 or time.monotonic() - last_progress >= 60:
            train.write_json(out / "progress.json", progress("training"))
            last_progress = time.monotonic()
    pause()
    receipt = out / "checkpoint.json"
    if (
        not latest.is_file()
        or not receipt.is_file()
        or json.loads(receipt.read_text())["step"] != step
    ):
        save()
    status = (
        "training_complete"
        if step == args.steps
        else "interrupted"
        if interruption
        else "time_censored"
    )
    if status != "interrupted":
        payload = train.core_payload(cfg, model.state_dict(), identity, step)
        payload["architecture"] = dict(profile=args.capacity, **asdict(model.profile))
        train.atomic_save(payload, core)
    result = progress(status)
    result.update(
        final_core=str(core) if status != "interrupted" else None,
        latest_checkpoint=str(latest),
        last_partial_window=reduced_window(torch, window),
    )
    train.write_json(out / "progress.json", result)
    if status == "training_complete":
        train.write_json(out / "training_done.json", result)
    elif status == "time_censored":
        train.write_json(out / "time_censored.json", result)


if __name__ == "__main__":
    main()
