"""Paired synthetic attribution of the inferred W-state channel in temporal feedback.

The command substitution replaces the classifier population/reference coherence.
Counts, calibration, Fisher geometry, past actions, and an admitted partition head
remain available. Both policies freely generate their own next preparations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .temporal_recovery_environment import CALIBRATION, DECISIONS, PreparationEnvironment
from .temporal_recovery_evaluation import TRAIN_SEEDS, atomic_json
from .temporal_recovery_long_evaluation import CHUNK_SIZE, sample_panel
from .temporal_recovery_long_training import checkpoint_digest
from .temporal_recovery_pairing import paired_seed_panel
from .temporal_recovery_policy import RecoveryController
from .temporal_recovery_shared_evaluation import PublicCommandState
from .temporal_recovery_shared_export import load_shared_route
from .temporal_recovery_training import callback, load_classifier

SOURCES = ("inferred", "command")
FITS = ("ordered", "learner")
FRACTIONS = (0.30, 0.55, 0.80)
DEV_PREFIX = 21_120_000_000_000
CONFIRM_PREFIX = 21_130_000_000_000


def acquisition_schedule(
    n: int, total_reads: int, first_fraction: float | None = None
) -> np.ndarray:
    """Keep all eleven acquisitions and one eleventh of the total for the diagnostic."""
    operations = 3 * n - 2
    base, remainder = divmod(total_reads, (DECISIONS + 1) * operations)
    if n not in (3, 4) or base < 1 or remainder:
        raise ValueError(
            "total READs must support eleven equal positive windows for these settings"
        )
    shots = np.full(DECISIONS + 1, base, dtype=np.int64)
    if first_fraction is not None:
        if not 0 < first_fraction < 1:
            raise ValueError("invalid front-loaded allocation")
        first = round(DECISIONS * base * first_fraction)
        other, extra = divmod(DECISIONS * base - first, DECISIONS - 1)
        shots[0], shots[1:-1] = first, other
        shots[1 : 1 + extra] += 1
    if shots.min() < 1:
        raise ValueError("the schedule must buy at least one READ per setting per window")
    return shots


def limited_policy(policy, horizon: int):
    """Choose up to K actions at predeclared slots, retaining the full common clock."""
    if not 0 <= horizon <= DECISIONS:
        raise ValueError("horizon must be between zero and ten")
    slots = {i * DECISIONS // horizon for i in range(horizon)}

    def decide(prefix):
        if len(prefix.actions) in slots:
            return policy(prefix)
        return torch.zeros_like(prefix.observations[-1].executed_command)

    return decide


def evaluate_fit(
    env, policies, *, horizons, total_reads, dev_prefix, confirm_prefix, dev_size, confirm_size
):
    """Use DEV to fix one allocation, then pair every source/horizon on untouched episodes."""
    if (
        dev_size < 1
        or confirm_size < 1
        or min(dev_prefix, confirm_prefix) < 0
        or not (
            dev_prefix + dev_size <= confirm_prefix or confirm_prefix + confirm_size <= dev_prefix
        )
    ):
        raise ValueError("development and confirmation must be positive and disjoint")
    if set(policies) != set(SOURCES) or 1 not in horizons or len(set(horizons)) != len(horizons):
        raise ValueError("both state sources and a unique one-action reference are required")
    single = limited_policy(policies["inferred"], 1)
    dev_gains = {}
    for fraction in FRACTIONS:
        schedule = acquisition_schedule(env.n, total_reads, fraction)
        f = sample_panel(env, single, schedule, dev_size, dev_prefix)["fidelity"]
        dev_gains[fraction] = float((f[:, -1] - f[:, 0]).mean())
    one_fraction = max(dev_gains, key=dev_gains.get)
    fields, profiles = {}, {}
    initial = None
    for source, fn in policies.items():
        for horizon in horizons:
            key = f"{source}_k{horizon}"
            schedule = acquisition_schedule(
                env.n, total_reads, one_fraction if horizon == 1 else None
            )
            panel = sample_panel(
                env, limited_policy(fn, horizon), schedule, confirm_size, confirm_prefix
            )
            f, actions = panel["fidelity"], panel["executed_actions"]
            if initial is None:
                initial = f[:, 0]
            if not np.array_equal(initial, f[:, 0]):
                raise ValueError("paired initial states differ")
            actual_reads = panel["native_counts"].sum((1, 2, 3))
            if not (
                np.all(actual_reads == total_reads)
                and np.array_equal(panel["reads_per_setting"], schedule)
            ):
                raise ValueError("realized acquisition budget differs from the registered schedule")
            active = np.abs(actions).max(axis=(2, 3)) > 1e-8
            profiles[key] = {
                "round_mean_gain": (f - f[:, :1]).mean(0).tolist(),
                "terminal_gain": float((f[:, -1] - f[:, 0]).mean()),
                "terminal_harm": float(np.mean(f[:, -1] < f[:, 0] - 1e-10)),
                "terminal_harm_count": int(np.sum(f[:, -1] < f[:, 0] - 1e-10)),
                "active_slots": int(active.sum()),
                "total_slots": int(active.size),
                "active_episodes": int(active.any(axis=1).sum()),
                "any_step_harm": float(np.mean(np.any(np.diff(f, axis=1) < -1e-10, axis=1))),
                "late_gain_after_round_five": float((f[:, -1] - f[:, 5]).mean()),
                "action_fraction_of_all_ten_slots": float(active.mean()),
                "paid_reads": int(total_reads),
            }
            for field in ("fidelity", "executed_actions", "reads_per_setting"):
                fields[f"{key}_{field}"] = panel[field]
    return fields, {"one_fraction": one_fraction, "inferred_dev_gains": dev_gains, "arms": profiles}


def paired_contrasts(arrays: dict, horizons: list[int]) -> dict:
    """Return within-episode differences in terminal conditional fidelity."""
    result = {}
    for k in horizons:
        state = f"inferred_k{k}_fidelity"
        command = f"command_k{k}_fidelity"
        result[f"inferred_minus_command_k{k}"] = arrays[state][:, -1] - arrays[command][:, -1]
        if k != 1:
            for source in SOURCES:
                result[f"{source}_k{k}_minus_k1"] = (
                    arrays[f"{source}_k{k}_fidelity"][:, -1]
                    - arrays[f"{source}_k1_fidelity"][:, -1]
                )
            result[f"state_by_horizon_interaction_k{k}"] = (
                result[f"inferred_k{k}_minus_k1"] - result[f"command_k{k}_minus_k1"]
            )
    return result


def load_fit(args, seed: int):
    if args.shared_root:
        core, models, spec = load_shared_route(args.shared_root / f"seed_{seed}", "cpu")
        if spec["binding"]["seed"] != seed:
            raise ValueError("shared controller seed differs from the requested fit")
        return (
            core,
            models,
            {
                "layout": "shared",
                "binding": spec["binding"],
                "backbone_sha256": spec["backbone_sha256"],
            },
        )
    paths = {3: args.classifier3, 4: args.classifier4}
    if None in paths.values():
        raise ValueError("short fits require both original classifier exports")
    digests = {str(n): checkpoint_digest(p) for n, p in paths.items()}
    core = load_classifier(paths[args.n], "cpu")[0]
    root = args.short_root / f"seed_{seed}"
    models = {}
    for arm in FITS:
        ck = torch.load(root / f"{arm}.pt", weights_only=True, map_location="cpu")
        if (
            ck["arm"] != arm
            or ck["binding"]["seed"] != seed
            or ck["binding"]["classifier"] != digests
            or ck["updates"] != ck["limit"]
            or ck["updates"] <= 0
        ):
            raise ValueError(f"incomplete or mismatched {arm} control fit")
        ctrl = RecoveryController()
        ctrl.load_state_dict(ck["state"], strict=True)
        models[arm] = ctrl.eval().requires_grad_(False)
    if any(
        not torch.equal(v, models["learner"].feedback.state_dict()[k])
        for k, v in models["ordered"].feedback.state_dict().items()
    ):
        raise ValueError(
            "teacher-trajectory and learner-visited adapters have different feedback cores"
        )
    return (
        core,
        models,
        {
            "layout": "short",
            "seed": seed,
            "classifier_sha256": digests,
            "controller_sha256": {arm: checkpoint_digest(root / f"{arm}.pt") for arm in FITS},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--shared-root", type=Path)
    roots.add_argument("--short-root", type=Path)
    parser.add_argument("--classifier3", type=Path)
    parser.add_argument("--classifier4", type=Path)
    parser.add_argument("--n", type=int, required=True, choices=(3, 4))
    parser.add_argument("--seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 10], choices=range(11))
    parser.add_argument("--total-reads", type=int, default=77000)
    parser.add_argument(
        "--calibration",
        type=float,
        nargs=5,
        default=list(CALIBRATION),
        metavar=("S_PER_READ", "SIGNAL_ETA", "SIGNAL_BG", "IDLER_ETA", "IDLER_BG"),
    )
    parser.add_argument("--dev-episodes", type=int, default=128)
    parser.add_argument("--confirm-episodes", type=int, default=512)
    parser.add_argument("--dev-prefix", type=int, default=DEV_PREFIX)
    parser.add_argument("--confirm-prefix", type=int, default=CONFIRM_PREFIX)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("each fitted seed must appear exactly once")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(exist_ok=True, parents=True)
    env = PreparationEnvironment(args.n, calibration=tuple(args.calibration))
    contrast_rows = {}
    by_seed = {}
    ids = np.array(
        [
            f"n{args.n}_{args.confirm_prefix + 1000 * args.n + i - i % CHUNK_SIZE}_{i % CHUNK_SIZE}"
            for i in range(args.confirm_episodes)
        ]
    )
    for seed in args.seeds:
        core, controllers, binding = load_fit(args, seed)
        by_seed[seed] = {"weights": binding, "fits": {}}
        saved = {"episode_ids": ids}
        fit_rows = {}
        for arm in FITS:
            policies = {
                s: callback(controllers[arm], "ordered", env, c, "cpu")
                for s, c in (("inferred", core), ("command", PublicCommandState(core)))
            }
            arrays, info = evaluate_fit(
                env,
                policies,
                horizons=args.horizons,
                total_reads=args.total_reads,
                dev_prefix=args.dev_prefix,
                confirm_prefix=args.confirm_prefix,
                dev_size=args.dev_episodes,
                confirm_size=args.confirm_episodes,
            )
            by_seed[seed]["fits"][arm] = info
            saved.update({f"{arm}_{k}": v for k, v in arrays.items()})
            rows = paired_contrasts(arrays, args.horizons)
            fit_rows[arm] = rows
            for key, values in rows.items():
                contrast_rows.setdefault(f"{arm}_{key}", {})[seed] = values
        for k in args.horizons:
            key = f"inferred_minus_command_k{k}"
            contrast_rows.setdefault(f"learner_minus_teacher_state_effect_k{k}", {})[seed] = (
                fit_rows["learner"][key] - fit_rows["ordered"][key]
            )
        np.savez_compressed(args.output / f"episodes_n{args.n}_seed{seed}.npz", **saved)
    family_size = len(contrast_rows)
    results = {}
    for key, differences in contrast_rows.items():
        pooled, per_seed = paired_seed_panel(differences, ids, family_size)
        results[key] = {"fixed_seed_panel": pooled, "by_fit_seed": per_seed}
    atomic_json(
        args.output / "state_attribution.json",
        {
            "schema": "temporal_state_channel_attribution_v1",
            "N": args.n,
            "law": "effective_single_event_v1",
            "initial_bias": "static coherent Gaussian; centered log-amplitude SD 0.30, phase SD 0.40 rad",
            "execution": "bounded continuous CSM student and temporal adapter; static FD candidate head unused",
            "k2": 0,
            "calibration": args.calibration,
            "paid_reads": args.total_reads,
            "horizons": args.horizons,
            "seeds": args.seeds,
            "dev_episodes": args.dev_episodes,
            "confirm_episodes": args.confirm_episodes,
            "dev_prefix": args.dev_prefix,
            "confirm_prefix": args.confirm_prefix,
            "acquisitions": DECISIONS + 1,
            "single_schedule_selection": "one DEV-optimal inferred schedule per fit seed and training arm, "
            "fixed for both state sources",
            "estimand": "terminal conditional K1 W fidelity of the next preparation, free simulated rollouts",
            "interval_scope": "paired independent episodes, conditional on supplied fitted seeds; "
            "family correction within this N/calibration/budget invocation only",
            "family_size": family_size,
            "per_seed": by_seed,
            "paired_contrasts": results,
        },
    )


if __name__ == "__main__":
    main()
