"""Fit shared-N CSM feedback/hierarchy adapters and evaluate learner-owned closed loops."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .classifier_n_full import V10ClassifierNFull
from .projector_core_export import classifier_configuration, make_backbone_envelope
from .temporal_recovery_environment import PreparationEnvironment, allocation, rollout
from .temporal_recovery_evaluation import (
    CONFIRM_SEED,
    DEV_SEED,
    TRAIN_SEEDS,
    atomic_json,
    publish_atomic,
    evidence_arrays,
    observations_summary,
    one_step,
)
from .temporal_recovery_policy import (
    FEEDBACK_CONFIG,
    HIERARCHY_CONFIG,
    RecoveryController,
    current_context,
    prepare_prefix,
)

CORE_STEPS = 96
ADAPTER_STEPS = 64
BATCH = 12
SEED_ARMS = ("current", "summary", "ordered", "learner")
VERSION = "temporal_recovery_native_v2"


def save_pt(value, dest):
    temporary = dest.with_suffix(".tmp")
    torch.save(value, temporary)
    publish_atomic(temporary, dest)


def load_classifier(path, device):
    exported = torch.load(path, map_location="cpu", weights_only=True)
    cfg, state = classifier_configuration(exported)
    if cfg.get("observation_law") != "effective_single_event_v1":
        raise ValueError(
            "classifier must explicitly match the simulated observation law"
        )
    model = V10ClassifierNFull(**cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device).requires_grad_(False).eval(), exported


def callback(model, arm, env, classifier, device):
    @torch.no_grad()
    def decide(prefix):
        current = current_context(env, prefix.observations[-1], classifier, device)
        x = {k: v.to(device) for k, v in prepare_prefix(env, prefix, current).items()}
        return model(x, arm).detach().cpu()

    return decide


def labeled_panel(envs, cores, seed, device, model=None, *, episodes=48):
    data = {}
    for n, env in envs.items():
        policy = (
            None
            if model is None
            else callback(model.eval(), "ordered", env, cores[n], device)
        )
        batch = rollout(
            env,
            policy,
            episodes=episodes,
            seed=seed + n,
            schedule=allocation(n),
            collect_training=True,
            privileged_teacher=policy is None,
        )
        data[n] = []
        for prefix, target in batch["training"]:
            current = current_context(env, prefix.observations[-1], cores[n], device)
            data[n].append((prepare_prefix(env, prefix, current), target.float()))
    return data


def fit(model, arm, data, seed, device, checkpoint, binding, learner_data=None):
    """Atomic optimizer, model and sampler milestones; resume only identical inputs."""
    is_core = arm == "zero"
    model.to(device)
    model.feedback.requires_grad_(is_core)
    model.hierarchy.requires_grad_(not is_core)
    for block in (model.raw_encoder, model.raw_query):
        block.requires_grad_(not is_core)
    model.train()
    if not is_core:
        model.feedback.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=0.002 if is_core else 0.001, weight_decay=0.0001
    )
    rng = np.random.default_rng(seed)
    start = 0
    limit = CORE_STEPS if is_core else ADAPTER_STEPS
    losses = []
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ckpt["binding"] != binding or ckpt["arm"] != arm or ckpt["limit"] != limit:
            raise ValueError("incompatible temporal checkpoint")
        model.load_state_dict(ckpt["state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        rng.bit_generator.state = ckpt["rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(ckpt["cuda_rng"], device)
        start, losses = ckpt["updates"], ckpt["losses"]
    panel = data
    for index in range(start, limit):
        if arm == "learner" and index >= limit // 2 and panel is data:
            # Inference/restoration constructs modules. Keep the optimizer dropout stream
            # independent of whether on-policy collection was materialized before resume.
            with torch.random.fork_rng(
                devices=[torch.device(device).index or 0]
                if str(device).startswith("cuda")
                else []
            ):
                panel = learner_data(model, index)
            model.train()
            model.feedback.eval()
        n = (3, 4)[index % 2]
        t = int(rng.integers(10))
        x, y = panel[n][t]
        ix = torch.from_numpy(rng.integers(len(y), size=BATCH))
        inputs = {k: v[ix].to(device) for k, v in x.items()}
        target = y[ix].to(device)
        output = model(inputs, "ordered" if arm == "learner" else arm)
        # Fixed coordinate units; shortest angular distance in radians.
        amp_error = (output[..., 0] - target[..., 0]) / 0.35
        phase = output[..., 1] - target[..., 1]
        loss = (
            amp_error.square() + (torch.atan2(phase.sin(), phase.cos()) / 0.70).square()
        ).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite action loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if (index + 1) % 16 == 0 or index + 1 == limit:
            losses.append([index + 1, float(loss.detach())])
            save_pt(
                dict(
                    binding=binding,
                    arm=arm,
                    limit=limit,
                    updates=index + 1,
                    state=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device)
                    if str(device).startswith("cuda")
                    else None,
                    losses=losses,
                ),
                checkpoint,
            )
    return {
        "optimizer_updates": limit,
        "batch_size": BATCH,
        "loss_trace": losses,
        "trainable_parameters": sum(p.numel() for p in trainable),
    }


def run(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.artifacts / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    envs = {n: PreparationEnvironment(n) for n in (3, 4)}
    paths = {3: args.classifier3, 4: args.classifier4}
    loaded = {n: load_classifier(path, args.device) for n, path in paths.items()}
    cores = {n: x[0] for n, x in loaded.items()}
    binding = {
        "version": VERSION,
        "seed": args.seed,
        "device": args.device,
        "classifier": {
            str(n): hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()
        },
    }
    base_path = root / "feedback.pt"
    summaries = root / "training.json"
    info = json.loads(summaries.read_text()) if summaries.exists() else {}
    if args.stage in ("train", "all"):
        started = time.monotonic()
        teacher = labeled_panel(envs, cores, args.seed, args.device)
        torch.manual_seed(args.seed)
        base = RecoveryController()
        info["feedback"] = fit(
            base, "zero", teacher, args.seed, args.device, base_path, binding
        )
        feedback_export = {
            "feedback_config": FEEDBACK_CONFIG,
            "feedback_state_dict": {
                k: v.cpu() for k, v in base.feedback.state_dict().items()
            },
        }
        save_pt(feedback_export, root / "feedback_export.pt")
        for n in (3, 4):
            envelope = make_backbone_envelope(
                loaded[n][1], feedback_export, hierarchy_config=HIERARCHY_CONFIG
            )
            # Keep classifier and feedback in their existing retrievable exports. The
            # strict reconstruction is exercised again by the evaluation-only entry point.
            info[f"envelope_n{n}"] = {
                "schema": envelope["schema"],
                "classifier": str(paths[n]),
                "feedback": "feedback_export.pt",
            }
        for arm in SEED_ARMS:
            torch.manual_seed(args.seed + 1)
            model = RecoveryController()
            model.feedback.load_state_dict(
                feedback_export["feedback_state_dict"], strict=True
            )

            def fresh_learner_panel(m, completed_steps):
                # Immutable halfway state identifies the learner-visited records on resume.
                half = root / "learner_half.pt"
                if not half.exists() and completed_steps != ADAPTER_STEPS // 2:
                    raise ValueError(
                        "missing halfway learner-state collection checkpoint for exact resume"
                    )
                if not half.exists():
                    save_pt({"binding": binding, "state": m.state_dict()}, half)
                previous = torch.load(half, map_location="cpu", weights_only=True)
                if previous["binding"] != binding:
                    raise ValueError(
                        "learner visitation snapshot belongs to a different fit"
                    )
                frozen = RecoveryController().to(args.device)
                frozen.load_state_dict(previous["state"], strict=True)
                return labeled_panel(
                    envs, cores, args.seed + 50000, args.device, frozen
                )

            info[arm] = fit(
                model,
                arm,
                teacher,
                args.seed + 2,
                args.device,
                root / f"{arm}.pt",
                binding,
                fresh_learner_panel,
            )
            atomic_json(summaries, info)
        info["elapsed_seconds"] = time.monotonic() - started
        atomic_json(summaries, info)
    if args.stage == "train":
        return
    result = {
        "schema": VERSION,
        "training_seed": args.seed,
        "binding": binding,
        "arms": {},
    }
    feedback_export = torch.load(
        root / "feedback_export.pt", map_location="cpu", weights_only=True
    )
    for n, env in envs.items():
        envelope = make_backbone_envelope(
            loaded[n][1], feedback_export, hierarchy_config=HIERARCHY_CONFIG
        )
        result["arms"][str(n)] = {}
        frozen_scenarios = []
        for name in (
            "zero",
            "current",
            "summary",
            "ordered",
            "pooled",
            "shuffled",
            "learner",
        ):
            fit_name = "ordered" if name in ("pooled", "shuffled") else name
            path = base_path if name == "zero" else root / f"{fit_name}.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state["binding"] != binding or state["updates"] != state["limit"]:
                raise ValueError(
                    "evaluation requires fit completion and the same frozen cores"
                )
            model = RecoveryController().to(args.device)
            model.load_state_dict(state["state"], strict=True)
            if any(
                not torch.equal(v.cpu(), envelope["feedback_state_dict"][k])
                for k, v in model.feedback.state_dict().items()
            ):
                raise ValueError("hierarchy changed a frozen core")
            model.eval()
            arm = "ordered" if name == "learner" else name
            policy = callback(model, arm, env, cores[n], args.device)
            scenarios = [(name, policy, allocation(n))]
            if name in ("ordered", "learner"):
                single = one_step(policy)
                firsts = (0.30, 0.55, 0.80)
                dev = []
                for frac in firsts:
                    score = rollout(
                        env,
                        single,
                        episodes=32,
                        seed=DEV_SEED + n,
                        schedule=allocation(n, frac),
                    )["scores"]
                    dev.append(float((score[:, -1] - score[:, 0]).mean()))
                chosen = firsts[int(np.argmax(dev))]
                scenarios.extend(
                    [
                        (f"{name}_one_common", single, allocation(n)),
                        (f"{name}_one_optimized", single, allocation(n, chosen)),
                    ]
                )
                result.setdefault("dev_single_fractions", {}).setdefault(str(n), {})[
                    name
                ] = chosen
            frozen_scenarios.extend(scenarios)
        # Every DEV choice for this system is frozen before accessing confirmation.
        for label, actual, budget in frozen_scenarios:
            rr = rollout(
                env, actual, episodes=128, seed=CONFIRM_SEED + n, schedule=budget
            )
            arr = evidence_arrays(rr)
            np.savez_compressed(root / f"n{n}_{label}.npz", **arr)
            result["arms"][str(n)][label] = observations_summary(arr)
        atomic_json(args.results / f"learned_seed_{args.seed}.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classifier3", type=Path, required=True)
    p.add_argument("--classifier4", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=TRAIN_SEEDS, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    run(p.parse_args())


if __name__ == "__main__":
    main()
