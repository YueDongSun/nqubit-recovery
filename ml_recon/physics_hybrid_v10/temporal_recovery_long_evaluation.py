"""Frozen development selection and fresh-episode comparisons for the long temporal fits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .temporal_recovery_environment import allocation, classical_policy, rollout
from .temporal_recovery_evaluation import (
    TRAIN_SEEDS,
    atomic_json,
    evidence_arrays,
    observations_summary,
    one_step,
    paired_interval,
)
from .temporal_recovery_training import callback
from .temporal_recovery_pairing import (
    confirmation_axis,
    check_endpoint_panel,
    paired_seed_panel,
    LONG_ARMS,
    registered_reports,
    checked_archive,
)
from .temporal_recovery_long_training import (
    DEV_PREFIX,
    CONFIRM_PREFIX,
    DEV_EPISODES,
    EVAL_EPISODES,
)

CHUNK_SIZE = 64
FAMILY_SIZE = 28


def sample_panel(env, policy, schedule, size, base):
    chunks = []
    for i in range(0, size, CHUNK_SIZE):
        # Namespace and chunk ID define disjoint independent initial states and noise.
        data = rollout(
            env,
            policy,
            episodes=min(CHUNK_SIZE, size - i),
            seed=base + 1000 * env.n + i,
            schedule=schedule,
        )
        chunks.append(evidence_arrays(data))
    return {
        k: (
            chunks[0][k]
            if k in ("analyzers", "reads_per_setting")
            else np.concatenate([v[k] for v in chunks], axis=0)
        )
        for k in chunks[0]
    }


def dev_gain(env, policy, budget, base=DEV_PREFIX):
    arr = sample_panel(env, policy, budget, DEV_EPISODES, base)["fidelity"]
    return float((arr[:, -1] - arr[:, 0]).mean())


def evaluate_seed(
    args,
    root,
    cores,
    envs,
    base,
    trained,
    binding,
    stats,
    *,
    dev_prefix=DEV_PREFIX,
    confirm_prefix=CONFIRM_PREFIX,
):
    report = {
        "binding": binding,
        "training_seed": args.seed,
        "confirmation_prefix": confirm_prefix,
        "development_prefix": dev_prefix,
        "stats": stats,
        "confirmation": "fresh predeclared 512 paired episodes",
        "independent_unit": "initial physical episode, same state and noise seed across policies",
        "separate_noninterchangeable_sources": (
            "one frozen shared C-S classifier and shared feedback/temporal parameters for 3Q/4Q"
            if binding.get("shared_lane")
            else "shared feedback and hierarchy for 3/4, distinct frozen classifiers"
        ),
        "systems": {},
    }
    for n, env in envs.items():
        schedule = allocation(n)
        choices = {}
        policies = {"noop": classical_policy(env, "noop")}
        for name in ("analytic", "pid"):
            values = {
                str(g): dev_gain(
                    env, classical_policy(env, name, g), schedule, dev_prefix
                )
                for g in (0.25, 0.45, 0.70)
            }
            best = max(values, key=values.get)
            choices[name] = {"gain": float(best), "dev_gain": values[best]}
            policies[name] = classical_policy(env, name, float(best))
        best_simple = max(
            ("analytic", "pid"), key=lambda item: choices[item]["dev_gain"]
        )
        for name in (
            "zero",
            "current",
            "summary",
            "ordered",
            "pooled",
            "shuffled",
            "learner",
        ):
            controller = (
                base
                if name == "zero"
                else trained["ordered" if name in ("pooled", "shuffled") else name]
            )
            policies[name] = callback(
                controller.eval(),
                "ordered" if name == "learner" else name,
                env,
                cores[n],
                args.device,
            )
        scenarios = {label: (fn, schedule) for label, fn in policies.items()}
        for label in ("analytic", "ordered", "learner"):
            single = one_step(policies[label])
            values = {
                str(frac): dev_gain(env, single, allocation(n, frac), dev_prefix)
                for frac in (0.30, 0.55, 0.80)
            }
            frac = float(max(values, key=values.get))
            choices[label + "_single"] = {
                "first_fraction": frac,
                "dev_terminal_gain": values[str(frac)],
            }
            scenarios[label + "_one_common"] = (single, schedule)
            scenarios[label + "_one_optimized"] = (single, allocation(n, frac))
        # Record all frozen choice criteria before any confirmatory observation draw.
        atomic_json(root / f"development_n{n}.json", choices)
        per_arm = {}
        for label, (policy, budget) in scenarios.items():
            arrays = sample_panel(env, policy, budget, EVAL_EPISODES, confirm_prefix)
            arr_path = root / f"long_n{n}_{label}.npz"
            np.savez_compressed(arr_path, **arrays)
            per_arm[label] = observations_summary(arrays)
        report["systems"][str(n)] = {
            "best_development_classical": best_simple,
            "choices": choices,
            "arms": per_arm,
        }
    atomic_json(args.results / f"long_seed_{args.seed}.json", report)


def aggregate(args):
    reports = registered_reports(
        args.results, args.artifacts, seeds=args.seeds, long_run=True
    )
    if getattr(args, "cohort", "terminal") == "common":
        rungs = {report["binding"].get("common_rung") for report in reports.values()}
        if len(rungs) != 1 or not rungs.issubset({512, 1024, 2048, 3072}):
            raise ValueError(
                "common-rung confirmation mixes or omits its registered stages"
            )
        matched = rungs.pop()
        for report in reports.values():
            stats = report.get("stats", {})
            if (
                stats.get("core_actual_updates") != 4096
                or stats.get("updates_per_adapter") != 32 * matched
            ):
                raise ValueError(
                    "common-rung report has unequal or missing actual optimizer counts"
                )
    summary = {
        "seeds": list(args.seeds),
        "multiplicity": "24 primary gain contrasts and ordered/learner terminal-harm bounds in each N; joint Bonferroni family size 28. Other harm estimates are descriptive.",
        "N_specific": {},
        "estimand": "initial-state-cluster paired mean across trained seeds at their recorded update counts",
        "evaluation": "fixed paid total 77000 READs including 11th diagnostic, 512 fresh episodes/N",
        "matched_offline_arms": "every long adapter consumed the same number of TRAIN episodes per block; teacher data shared across arms",
    }
    panels = {}
    for n in (3, 4):
        rows = {}
        expected_prefix = (
            8_000_000_000_000
            if getattr(args, "cohort", "terminal") == "common"
            else CONFIRM_PREFIX
        )
        if any(
            r.get("confirmation_prefix", CONFIRM_PREFIX) != expected_prefix
            for r in reports.values()
        ):
            raise ValueError("cannot pool terminal and common-rung cohorts")
        ids, initial = confirmation_axis(n, long_run=True, prefix=expected_prefix)
        panels[f"n{n}_episode_ids"] = ids
        storage = {}
        for s in args.seeds:
            for name in LONG_ARMS:
                with np.load(
                    args.artifacts / f"seed_{s}" / f"long_n{n}_{name}.npz"
                ) as f:
                    arr, active = checked_archive(
                        f,
                        reports[s]["systems"][str(n)]["arms"][name],
                        n=n,
                        ids=ids,
                        initial=initial,
                    )
                    storage[s, name] = arr
                    panels[f"n{n}_{s}_{name}_fidelity"] = arr
                    panels[f"n{n}_{s}_{name}_active"] = active
        pairs = [
            ("ordered", "ordered_one_common"),
            ("ordered", "ordered_one_optimized"),
            ("learner", "learner_one_common"),
            ("learner", "learner_one_optimized"),
            ("ordered", "current"),
            ("ordered", "summary"),
            ("ordered", "pooled"),
            ("ordered", "shuffled"),
            ("learner", "ordered"),
            ("ordered", "zero"),
            ("ordered", "best_simple"),
            ("ordered", "noop"),
        ]
        for a, b in pairs:
            vals = {}
            for s in args.seeds:
                ref = (
                    reports[s]["systems"][str(n)]["best_development_classical"]
                    if b == "best_simple"
                    else b
                )
                vals[s] = storage[s, a][:, -1] - storage[s, ref][:, -1]
            pooled, per_seed = paired_seed_panel(vals, ids, FAMILY_SIZE)
            rows[f"{a}_minus_{b}"] = {"across_seed_panel": pooled, "per_seed": per_seed}
        arm_summary = {}
        for name in LONG_ARMS:
            curves = np.stack([storage[s, name] for s in args.seeds])
            gains = curves - curves[:, :, :1]
            harms = gains[:, :, -1] < -1e-10
            act = np.stack([panels[f"n{n}_{s}_{name}_active"] for s in args.seeds])
            arm_summary[name] = {
                "terminal_gain": paired_interval(gains[:, :, -1].mean(0)),
                "harm": paired_interval(harms.mean(0), FAMILY_SIZE),
                "any_step_harm": paired_interval(
                    (np.diff(curves, axis=2) < -1e-10).any(2).mean(0)
                ),
                "coverage": paired_interval(act.mean((0, 2))),
                "late_gain": paired_interval(
                    (curves[:, :, -1] - curves[:, :, 5]).mean(0)
                ),
                "round_mean_gain": gains.mean((0, 1)).tolist(),
            }
        summary["N_specific"][str(n)] = {"paired_contrasts": rows, "arms": arm_summary}
        legacy_name = (
            "long_common" if getattr(args, "cohort", "terminal") == "common" else "long"
        )
        legacy_path = (
            args.artifacts / f"historical_4q_{legacy_name}_corrected_native.npz"
        )
        if n == 4 and legacy_path.is_file():
            with np.load(legacy_path) as archive:
                historical = archive["fidelity"]
                check_endpoint_panel(historical, ids, initial)
            delta = {
                seed: storage[seed, "ordered"][:, -1] - historical[:, -1]
                for seed in args.seeds
            }
            pooled, conditioned = paired_seed_panel(delta, ids, 1)
            summary["N_specific"]["4"]["archived_frozen_vs_current_secondary"] = {
                "orientation": "ordered minus archived frozen four-mode checkpoint with corrected frontend",
                "evidence": "secondary paired 95% intervals, outside primary multiplicity family",
                "across_seed_panel": pooled,
                "per_seed": conditioned,
            }
            panels["n4_historical_fidelity"] = historical
    np.savez_compressed(args.results / "long_episode_panels.npz", **panels)
    atomic_json(args.results / "long_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--cohort", choices=("terminal", "common"), default="terminal")
    parser.add_argument("--seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate seed")
    aggregate(args)


if __name__ == "__main__":
    main()
