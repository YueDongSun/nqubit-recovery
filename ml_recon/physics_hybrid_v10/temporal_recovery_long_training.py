"""Bounded long temporal fits on independent fresh episodes, interleaved across ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .temporal_recovery_environment import PreparationEnvironment
from .temporal_recovery_evaluation import TRAIN_SEEDS, atomic_json, publish_atomic
from .temporal_recovery_policy import RecoveryController
from .temporal_recovery_training import (
    FEEDBACK_CONFIG,
    HIERARCHY_CONFIG,
    labeled_panel,
    load_classifier,
    save_pt,
)
from .projector_core_export import make_backbone_envelope

VERSION = "corrected_streamed_feedback_hierarchy_v1"
ARMS = ("current", "summary", "ordered", "learner")
TRAIN_EPISODES = 64
BATCH = 32
UPDATES = 32
PAID_READS = 77000
DEV_EPISODES = 128
EVAL_EPISODES = 512
MODEL_SEEDS = TRAIN_SEEDS
TRAIN_PREFIX = 1_000_000_000_000
DEV_PREFIX = 6_000_000_000_000
CONFIRM_PREFIX = 7_000_000_000_000


def training_seed(seed, stage, block):
    return TRAIN_PREFIX + 100_000_000 * seed + 1_000_000 * stage + 10 * block


def checkpoint_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_model(arm, device, seed, feedback=None):
    torch.manual_seed(seed)
    model = RecoveryController().to(device)
    if feedback is not None:
        model.feedback.load_state_dict(feedback, strict=True)
    is_core = arm == "zero"
    model.feedback.requires_grad_(is_core)
    model.hierarchy.requires_grad_(not is_core)
    model.raw_encoder.requires_grad_(not is_core)
    model.raw_query.requires_grad_(not is_core)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=0.0008 if is_core else 0.0004, weight_decay=0.0001
    )
    return model, opt


def train_block(model, opt, panel, arm, seed, device):
    model.train()
    if arm != "zero":
        model.feedback.eval()
    # Each arm has identical episode/round sampling and starting dropout RNG.
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    total = 0.0
    for update in range(UPDATES):
        n = (3, 4)[update % 2]
        t = int(rng.integers(10))
        inputs, target = panel[n][t]
        idx = torch.from_numpy(rng.integers(len(target), size=BATCH))
        x = {k: v[idx].to(device) for k, v in inputs.items()}
        y = target[idx].to(device)
        pred = model(x, "ordered" if arm == "learner" else arm)
        phase = pred[..., 1] - y[..., 1]
        loss = (
            ((pred[..., 0] - y[..., 0]) / 0.35).square()
            + (torch.atan2(phase.sin(), phase.cos()) / 0.70).square()
        ).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite long temporal loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            1.0,
            error_if_nonfinite=True,
        )
        opt.step()
        total += loss.detach().item()
    return total / UPDATES


def storage_state(binding, core_blocks, adapter_blocks, core, controllers, stats):
    return dict(
        binding=binding,
        core_blocks=core_blocks,
        adapter_blocks=adapter_blocks,
        updates=UPDATES * (core_blocks + len(ARMS) * adapter_blocks),
        core_state=core[0].state_dict(),
        core_optimizer=core[1].state_dict(),
        states={k: m.state_dict() for k, (m, _) in controllers.items()},
        optimizers={k: opt.state_dict() for k, (_, opt) in controllers.items()},
        stats=stats,
    )


def publish_snapshot(
    root,
    checkpoint,
    binding,
    core_done,
    adapt_done,
    core,
    controllers,
    stats,
    envs,
    classifiers,
    device,
    *,
    final=False,
):
    """Publish a self-consistent immutable step after the atomic optimizer state."""
    index = adapt_done or core_done
    if not index:
        return
    stage = "adapter" if adapt_done else "core"
    if not final and index not in (1, 4, 16, 32, 64, 128) and index % 64:
        return
    label = f"step_{stage}_{index:05d}" if adapt_done else f"step_core_{index:04d}"
    snapshots = root / "snapshots"
    snapshots.mkdir(exist_ok=True)
    dest = snapshots / label
    if dest.exists():
        return
    # The publisher is the only writer. A uniquely named staging directory does
    # not match the collector's completed-snapshot discovery prefix.
    tmp = Path(tempfile.mkdtemp(prefix="staging_", dir=snapshots))
    try:
        os.link(checkpoint, tmp / "state.pt")
    except OSError:
        shutil.copy2(checkpoint, tmp / "state.pt")
    updates = UPDATES * (core_done + len(ARMS) * adapt_done)
    save_pt(
        dict(
            schema=VERSION,
            binding=binding,
            updates=updates,
            core_updates=core_done * UPDATES,
            updates_per_adapter=adapt_done * UPDATES,
            feedback_config=FEEDBACK_CONFIG,
            hierarchy_config=HIERARCHY_CONFIG,
            feedback_state_dict=core[0].feedback.state_dict(),
            adapter_model_states={
                k: m.state_dict() for k, (m, _) in controllers.items()
            },
        ),
        tmp / "weights.pt",
    )
    from .temporal_recovery_training import callback
    from .temporal_recovery_environment import allocation, rollout

    dev = {}
    model = controllers["ordered"][0] if adapt_done else core[0]
    for n, env in envs.items():
        fn = callback(
            model.eval(),
            "ordered" if adapt_done else "zero",
            env,
            classifiers[n],
            device,
        )
        panel = rollout(
            env,
            fn,
            episodes=32,
            seed=binding.get("monitor_dev_prefix", DEV_PREFIX) + 50000 + n,
            schedule=allocation(n),
        )
        gain = panel["scores"][:, -1] - panel["scores"][:, 0]
        dev[str(n)] = {
            "episodes": 32,
            "mean_terminal_gain": float(gain.mean()),
            "terminal_harm_fraction": float((gain < -1e-10).double().mean()),
            "acquisitions": 11,
            "paid_per_episode": PAID_READS,
            "role": "fixed_monitoring_development_only",
        }
    atomic_json(
        tmp / "metrics.json",
        dict(
            binding=binding,
            stage=stage,
            updates=updates,
            core_blocks=core_done,
            adapter_blocks=adapt_done,
            stats=stats,
            development=dev,
            files={"full_optimizer_resume": "state.pt", "lean_weights": "weights.pt"},
        ),
    )
    publish_atomic(tmp, dest)
    atomic_json(
        root / "newest_snapshot.json",
        dict(
            relative_path=f"snapshots/{label}",
            updates=updates,
            core_actual_updates=core_done * UPDATES,
            updates_per_adapter=adapt_done * UPDATES,
        ),
    )


def train(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.artifacts / f"seed_{args.seed}"
    root.mkdir(exist_ok=True, parents=True)
    args.results.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA runtime is unavailable")
    paths = {3: args.classifier3, 4: args.classifier4}
    shared_lane = getattr(args, "shared_lane", None)
    qual = None
    namespace = getattr(args, "campaign_namespace", None)
    if namespace is not None and (not shared_lane or args.stage not in ("train", "fit-export")):
        raise ValueError("new shared training requires a lane and train/fit-export; use the separate Fig. 5 evaluator")
    if args.stage == "fit-export" and not shared_lane:
        raise ValueError("fit-export requires a shared source")
    if shared_lane:
        from .temporal_recovery_shared_qualification import (
            campaign_seed_base,
            check_source,
            qualified_route,
        )

        if (
            paths[3].resolve() != paths[4].resolve()
            or args.core_blocks != 128
            or args.adapter_blocks != 256
            or not args.structure_qualification
        ):
            raise ValueError(
                "shared B鈫扖 requires one checkpoint path, fixed matched budget and qualification receipt"
            )
        single = load_classifier(paths[3], device)
        check_source(single[1], shared_lane, namespace)
        qual = json.loads(args.structure_qualification.read_text())
        if namespace and qual.get("source_identity") != single[1]["identity"]:
            raise ValueError("qualification does not bind the new anchor identity")
        single[0].partition_control_qualified = qualified_route(
            qual, shared_lane, checkpoint_digest(paths[3])
        )
        loaded = {3: single, 4: single}
    else:
        loaded = {n: load_classifier(path, device) for n, path in paths.items()}
    cores = {n: pair[0] for n, pair in loaded.items()}
    weak_source = getattr(args, "weak_calibration", None)
    weak = None
    if weak_source:
        if shared_lane or args.core_blocks != 128 or args.adapter_blocks != 128:
            raise ValueError(
                "weak fits require the independent fixed 4096/4096 control design"
            )
        weak = json.loads(weak_source.read_text())
        if (
            weak.get("schema") != "temporal_train_grounded_weak_v1"
            or weak.get("law") != "effective_single_event_v1"
            or weak.get("k2") != 0
            or set(weak.get("systems", {})) != {"3", "4"}
        ):
            raise ValueError("invalid weak training operating point")
        for row in weak["systems"].values():
            cal = row["calibration"]
            if len(cal) != 5 or cal[1] != 0.2 or cal[3] != 0.04:
                raise ValueError("unregistered weak training efficiencies")
        envs = {
            n: PreparationEnvironment(
                n, calibration=weak["systems"][str(n)]["calibration"]
            )
            for n in paths
        }
    else:
        envs = {n: PreparationEnvironment(n) for n in paths}
    cutoff = args.deadline_unix - 60 * args.reserve_minutes
    binding = {
        "version": VERSION,
        "seed": args.seed,
        "device": str(device),
        "classifier_sha256": {
            str(n): checkpoint_digest(path) for n, path in paths.items()
        },
        "core_target_blocks": args.core_blocks,
        "adapter_cap_blocks": args.adapter_blocks,
        "episodes_per_system_per_block": TRAIN_EPISODES,
        "updates_per_block": UPDATES,
        "batch_size": BATCH,
        "registered_deadline_unix": args.deadline_unix,
        "reserve_minutes": args.reserve_minutes,
    }
    if shared_lane:
        binding.update(
            shared_lane=shared_lane,
            train_namespace_offset=9_000_000_000_000,
            monitor_dev_prefix=9_200_000_000_000,
            partition_control_qualified=qual["allow_partition"],
            qualification_identity=checkpoint_digest(args.structure_qualification),
            upstream_updates=32768,
            upstream_training_sizes=[3, 4, 8],
        )
    if namespace:
        base = campaign_seed_base(namespace)
        binding.update(campaign_namespace=namespace, upstream_seed=7301,
                       upstream_run_id=loaded[3][1]["identity"]["lane"],
                       train_namespace_offset=base, monitor_dev_prefix=base + 3_000_000_000_000)
    if weak:
        binding.update(
            weak_calibration_sha256=checkpoint_digest(weak_source),
            weak_calibration=weak["systems"],
            monitor_dev_prefix=11_200_000_000_000,
        )
    checkpoint = root / "long_state.pt"
    core = configure_model("zero", device, args.seed)
    controllers = {}
    stats = {"on_policy_blocks": 0, "wall_seconds": 0.0}
    core_done = adapt_done = 0
    if checkpoint.exists():
        record = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if record["binding"] != binding:
            raise ValueError(
                "resume does not match original cores, device, seed, deadline and run design"
            )
        core_done, adapt_done, stats = (
            record["core_blocks"],
            record["adapter_blocks"],
            record["stats"],
        )
        core[0].load_state_dict(record["core_state"], strict=True)
        core[1].load_state_dict(record["core_optimizer"])
        for arm, state in record["states"].items():
            pair = configure_model(arm, device, args.seed + 1)
            pair[0].load_state_dict(state, strict=True)
            pair[1].load_state_dict(record["optimizers"][arm])
            controllers[arm] = pair
    if checkpoint.exists():
        publish_snapshot(
            root,
            checkpoint,
            binding,
            core_done,
            adapt_done,
            core,
            controllers,
            stats,
            envs,
            cores,
            device,
        )
    if args.stop_after_blocks and core_done + adapt_done >= args.stop_after_blocks:
        return
    starting = time.monotonic()
    carried = stats["wall_seconds"]
    log = root / "blocks.jsonl"
    # Iterate only through complete blocks, so every temporal arm is evaluated at the same rung.
    while (
        (core_done < args.core_blocks or adapt_done < args.adapter_blocks)
        and time.time() < cutoff
        and args.stage != "evaluate"
    ):
        tick = time.monotonic()
        core_phase = core_done < args.core_blocks
        index = core_done if core_phase else adapt_done
        seed = training_seed(args.seed, 0 if core_phase else 1, index) + binding.get(
            "train_namespace_offset", 0
        )
        teacher = labeled_panel(envs, cores, seed, device, episodes=TRAIN_EPISODES)
        losses = {}
        if core_phase:
            losses["feedback"] = train_block(*core, teacher, "zero", seed + 5, device)
            core_done += 1
        else:
            if not controllers:
                for arm in ARMS:
                    controllers[arm] = configure_model(
                        arm, device, args.seed + 1, core[0].feedback.state_dict()
                    )
            on_policy = index >= 8 and index % 2 == 1
            # Snapshot visitation before this block's optimizer updates.
            if on_policy:
                learner = labeled_panel(
                    envs,
                    cores,
                    seed,
                    device,
                    controllers["learner"][0].eval(),
                    episodes=TRAIN_EPISODES,
                )
            else:
                learner = teacher
            for arm, pair in controllers.items():
                losses[arm] = train_block(
                    *pair,
                    learner if arm == "learner" else teacher,
                    arm,
                    seed + 5,
                    device,
                )
            stats["on_policy_blocks"] += int(on_policy)
            adapt_done += 1
        stats["wall_seconds"] = carried + time.monotonic() - starting
        stats["teacher_stream_episodes"] = (core_done + adapt_done) * 2 * TRAIN_EPISODES
        stats["additional_on_policy_episodes"] = (
            stats["on_policy_blocks"] * 2 * TRAIN_EPISODES
        )
        stats["unique_training_reads"] = (
            stats["teacher_stream_episodes"] + stats["additional_on_policy_episodes"]
        ) * PAID_READS
        stats["per_adapter_training_reads"] = (
            adapt_done * 2 * TRAIN_EPISODES * PAID_READS
        )
        stats["core_training_reads"] = core_done * 2 * TRAIN_EPISODES * PAID_READS
        stats["core_actual_updates"] = core_done * UPDATES
        stats["updates_per_adapter"] = adapt_done * UPDATES
        save_pt(
            storage_state(binding, core_done, adapt_done, core, controllers, stats),
            checkpoint,
        )
        publish_snapshot(
            root,
            checkpoint,
            binding,
            core_done,
            adapt_done,
            core,
            controllers,
            stats,
            envs,
            cores,
            device,
        )
        row = dict(
            core_blocks=core_done,
            adapter_blocks=adapt_done,
            loss=losses,
            seconds=time.monotonic() - tick,
            checkpoint=str(checkpoint),
            utc_unix=time.time(),
            **stats,
        )
        # One durable status describes the checkpoint. The append-only timing log may retain
        # an interrupted prior attempt, distinguishable by block number.
        atomic_json(root / "status.json", row)
        with log.open("a", encoding="utf-8") as h:
            h.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "core_blocks",
                        "adapter_blocks",
                        "seconds",
                        "updates_per_adapter",
                    )
                }
            ),
            flush=True,
        )
        if args.stop_after_blocks and core_done + adapt_done >= args.stop_after_blocks:
            return
    if args.stage == "train":
        if weak and (core_done != 128 or adapt_done != 128):
            raise ValueError(
                "weak retraining stage unfinished, keep corrective confirmation closed"
            )
        return
    if weak:
        raise ValueError(
            "weak training uses --stage train; use the separate matched corrective evaluator"
        )
    if shared_lane and (core_done != 128 or adapt_done != 256):
        raise ValueError(
            "registered shared B-C update budget unfinished; preserve checkpoint, keep confirmation closed"
        )
    if not controllers or adapt_done < 1:
        raise ValueError(
            "no completed matched adapter block before the evaluation cutoff"
        )
    feedback_export = {
        "feedback_config": FEEDBACK_CONFIG,
        "feedback_state_dict": {
            k: v.cpu() for k, v in core[0].feedback.state_dict().items()
        },
    }
    save_pt(feedback_export, root / "long_feedback_export.pt")
    if shared_lane:
        envelope = make_backbone_envelope(
            loaded[3][1], feedback_export, hierarchy_config=HIERARCHY_CONFIG
        )
        save_pt(envelope, root / "shared_backbone.pt")
        atomic_json(
            root / "shared_route.json",
            {
                "schema": "trained_shared_classifier_feedback_temporal_v1",
                "shared_core_file": "shared_backbone.pt",
                "source_classifier_sha256": binding["classifier_sha256"]["3"],
                "core_constructor": envelope["classifier_config"],
                "feedback_constructor": FEEDBACK_CONFIG,
                "hierarchy_constructor": HIERARCHY_CONFIG,
                "raw_encoder": "RecoveryController.raw_encoder",
                "fully_trained_adapters": "long_state.pt#states",
                "upstream_updates": 32768,
                "upstream_training_sizes": [3, 4, 8],
                "control_evaluated_sizes": [3, 4],
                "shared_classifier_object": cores[3] is cores[4],
                "partition_control_qualified": qual["allow_partition"],
                "binding": binding,
                "training_cost": stats,
            },
        )
    else:
        for n in (3, 4):
            make_backbone_envelope(
                loaded[n][1], feedback_export, hierarchy_config=HIERARCHY_CONFIG
            )
    publish_snapshot(
        root,
        checkpoint,
        binding,
        core_done,
        adapt_done,
        core,
        controllers,
        stats,
        envs,
        cores,
        device,
        final=True,
    )
    if args.stage == "fit-export":
        from .temporal_recovery_shared_export import export_and_load
        export_and_load(root)
        return
    opened = root / "confirmation_opened.json"
    evaluation_binding = {
        "binding": binding,
        "core_blocks": core_done,
        "adapter_blocks": adapt_done,
    }
    if opened.exists() and json.loads(opened.read_text()) != evaluation_binding:
        raise ValueError(
            "fresh confirmation already opened for a different checkpoint rung"
        )
    atomic_json(opened, evaluation_binding)
    from .temporal_recovery_long_evaluation import evaluate_seed

    evaluate_seed(
        args,
        root,
        cores,
        envs,
        core[0],
        {k: v[0] for k, v in controllers.items()},
        binding,
        stats,
        dev_prefix=9_200_000_000_000 if shared_lane else DEV_PREFIX,
        confirm_prefix=9_300_000_000_000 if shared_lane else CONFIRM_PREFIX,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classifier3", type=Path, required=True)
    p.add_argument("--shared-lane", choices=("b0", "b2"))
    p.add_argument("--campaign-namespace", help="upstream split namespace of the new 7301 matrix anchors")
    p.add_argument("--weak-calibration", type=Path)
    p.add_argument("--structure-qualification", type=Path)
    p.add_argument("--classifier4", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=TRAIN_SEEDS, required=True)
    p.add_argument(
        "--deadline-unix",
        type=int,
        required=True,
        help="absolute shared campaign deadline, UTC epoch seconds",
    )
    p.add_argument("--reserve-minutes", type=int, default=45)
    p.add_argument("--core-blocks", type=int, default=128)
    p.add_argument("--adapter-blocks", type=int, default=8192)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stage", choices=("train", "fit-export", "evaluate", "all"), default="all")
    p.add_argument(
        "--stop-after-blocks",
        type=int,
        default=0,
        help="leave a resumable engineering checkpoint at this complete total block count",
    )
    args = p.parse_args()
    if (
        min(args.core_blocks, args.adapter_blocks, args.reserve_minutes) <= 0
        or args.adapter_blocks >= 100000
    ):
        raise ValueError("invalid finite positive time or update envelope")
    train(args)


if __name__ == "__main__":
    main()
