"""Train a new frozen STAR reference on deterministic, fresh archived-law minibatches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

import star_long as study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--engineering", action="store_true", help="Label the frozen weight as a test fixture")
    args = parser.parse_args(argv)
    sizes = (8, 16, 32, 64, 128, 256)
    if args.updates <= 0 or args.updates % len(sizes) or min(args.split_seed, args.init_seed) < 0:
        parser.error("choose nonnegative seeds and a positive whole number of six-size rotations")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    identity = dict(schema="fresh_archived_star_reference_v1", source=args.source, sizes=list(sizes),
                    updates=args.updates, init_seed=args.init_seed, split_seed=args.split_seed,
                    hidden=128, rank=4, dtype="float64", batch=64,
                    instrument="archived_star", learning_rate=.0003, engineering=args.engineering)
    args.output.mkdir(exist_ok=True, parents=True)
    with study.owner(args.output):
        receipt = args.output / "reference.json"
        weight_file = args.output / "reference.pt"
        registration = args.output / "reference_registration.json"
        if registration.exists() and json.loads(registration.read_text())["identity"] != identity:
            raise ValueError("Output directory is registered to another reference")
        study.atom_json(registration, dict(identity=identity, selection="fixed terminal update",
                       provenance="new reference trained afresh; historical seed-61 identity is distinct",
                       torch_version=str(torch.__version__), device=str(device), cpu_threads=torch.get_num_threads()))
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["identity"] != identity or study.file_sha(weight_file) != record["sha256"]:
                raise ValueError("Reference result conflicts with the requested source or specimen")
            return
        torch.manual_seed(args.init_seed)
        model = study.Unified_star_density_net(hidden=128, rank_columns=4, device="cpu").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0)
        checkpoint = args.output / "reference_latest.pt"
        step = 0
        elapsed = 0.0
        history = []
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["identity"] != identity:
                raise ValueError("Incompatible reference checkpoint")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            step = saved["step"]
            elapsed = saved.get("elapsed_seconds", 0.0)
            history = saved.get("history", [])
        start = time.monotonic()
        while step < args.updates:
            n = sizes[step % len(sizes)]
            batch = study.bank("reference_train", n, step, device, args.split_seed,
                               rows=identity["batch"], laws=("archived",))
            optimizer.zero_grad(set_to_none=True)
            loss = study.state_risk(model, batch["archived"], batch["mode"], batch["vis"]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            elapsed_now = elapsed + time.monotonic() - start
            history.append(dict(step=step, n=n, batch_loss=float(loss.detach()), elapsed_seconds=elapsed_now))
            if step == 12 or step % 128 == 0 or step == args.updates:
                study.atom_torch(checkpoint, dict(identity=identity, step=step, model=model.state_dict(),
                                                  optimizer=optimizer.state_dict(), history=history,
                                                  elapsed_seconds=elapsed_now))
                study.atom_json(args.output / "training_loss.json", history)
                study.atom_json(args.output / "progress.json", dict(pid=os.getpid(), step=step,
                               target_updates=args.updates, elapsed_seconds=elapsed_now,
                               last_batch_loss=history[-1]["batch_loss"], last_n=n))
        study.atom_torch(weight_file, model.cpu().state_dict())
        study.atom_json(receipt, dict(identity=identity, sha256=study.file_sha(weight_file),
                                      rows_per_size=(step // len(sizes)) * identity["batch"],
                                      elapsed_seconds=elapsed + time.monotonic() - start,
                                      exposures_per_operation="8192*N", selected="terminal_fixed_update"))


if __name__ == "__main__":
    main()
