"""Rebuild portable shared control models and compare matched B0/B2 training and trajectories."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from .temporal_recovery_environment import PreparationEnvironment, allocation, rollout
from .temporal_recovery_long_evaluation import sample_panel
from .temporal_recovery_evaluation import TRAIN_SEEDS, atomic_json, paired_interval
from .temporal_recovery_pairing import (
    LONG_ARMS,
    checked_archive,
    check_endpoint_panel,
    confirmation_axis,
    paired_seed_panel,
    registered_reports,
)
from .temporal_recovery_shared_export import export_and_load
from .temporal_recovery_training import callback

DEV = 9_200_000_000_000
CONFIRM = 9_300_000_000_000
FAMILY = 14


class PublicCommandState(torch.nn.Module):
    """Replace only the inferred W population/reference-coherence semantic input."""

    def __init__(self, trained):
        super().__init__()
        self.trained = trained
        self.partition_control_qualified = getattr(
            trained, "partition_control_qualified", False
        )

    def forward(self, batch, *, N, fine_identifiable=False):
        command = torch.complex(batch["prepared_re"], batch["prepared_im"])
        mag = command.abs()
        if (mag == 0).any():
            raise ValueError("registered coherent-control command has a missing mode")
        unit = command / mag
        coherence = unit[:, :, None] * unit[:, None, :].conj()
        fields = {
            "W_populations": mag.square(),
            "W_coherence_re": coherence.real,
            "W_coherence_im": coherence.imag,
        }
        if self.partition_control_qualified:
            fields["pair_entanglement_logits"] = self.trained(
                batch, N=N, fine_identifiable=fine_identifiable
            )["pair_entanglement_logits"]
        return fields


def check_shared_seed(seed_root, report, lane, seed):
    binding = report["binding"]
    stats = report["stats"]
    if (
        binding.get("shared_lane") != lane
        or binding.get("upstream_updates") != 32768
        or set(binding["classifier_sha256"]) != {"3", "4"}
        or len(set(binding["classifier_sha256"].values())) != 1
        or report.get("confirmation_prefix") != CONFIRM
        or report.get("development_prefix") != DEV
        or stats["core_actual_updates"] != 4096
        or stats["updates_per_adapter"] != 8192
    ):
        raise ValueError(
            "shared variant, upstream source, downstream stage or fresh cohort mismatch"
        )
    shared_classifier, controllers, spec = export_and_load(seed_root)
    if spec["binding"] != binding:
        raise ValueError("terminal report differs from exported policy")
    metrics = json.loads(
        (seed_root / "snapshots" / "step_adapter_00256" / "metrics.json").read_text()
    )
    if metrics["binding"] != binding:
        raise ValueError("different development snapshot")
    replay = {}
    for n in (3, 4):
        env = PreparationEnvironment(n)
        policy = callback(
            controllers["ordered"], "ordered", env, shared_classifier, "cpu"
        )
        rr = rollout(
            env, policy, episodes=32, seed=DEV + 50000 + n, schedule=allocation(n)
        )
        d = rr["scores"][:, -1] - rr["scores"][:, 0]
        old = metrics["development"][str(n)]
        diff = abs(float(d.mean()) - old["mean_terminal_gain"])
        if (
            diff > 1e-8
            or abs(float((d < -1e-10).float().mean()) - old["terminal_harm_fraction"])
            > 1e-10
        ):
            raise ValueError(
                "portable complete shared policy did not reproduce its local CPU development rollout"
            )
        baseline = PublicCommandState(shared_classifier)
        policy = callback(controllers["ordered"], "ordered", env, baseline, "cpu")
        paid = sample_panel(env, policy, allocation(n), 512, CONFIRM)
        np.savez_compressed(seed_root / f"public_command_channel_n{n}.npz", **paid)
        replay[str(n)] = {
            "episodes": 32,
            "mean_gain": float(d.mean()),
            "abs_difference": diff,
        }
    return {
        "seed": seed,
        "training_cost": stats,
        "classifier_sha256": binding["classifier_sha256"]["3"],
        "partition_control_qualified": binding["partition_control_qualified"],
        "development_replay": replay,
        "backbone": str(seed_root / "shared_backbone.pt"),
        "control_export": str(seed_root / "shared_control_adapters.pt"),
        "stored_classifier_parameters": sum(
            p.numel() for p in shared_classifier.parameters()
        ),
        "stored_feedback_parameters": sum(
            p.numel() for p in controllers["ordered"].feedback.parameters()
        ),
        "trained_ordered_hierarchy_and_raw_parameters": sum(
            p.numel()
            for name, p in controllers["ordered"].named_parameters()
            if not name.startswith("feedback.")
        ),
        "same_frozen_classifier_for_both_sizes": True,
    }


def paired_history(panels, n):
    """Descriptive registered history controls on the preserved common episode axis."""
    ids = panels[f"n{n}_episode_ids"]
    registered, initial = confirmation_axis(n, long_run=True, prefix=CONFIRM)
    if not np.array_equal(ids, registered):
        raise ValueError("wrong fully shared confirmation identity")
    ans = {}
    pairs = (
        ("learner", "ordered"),
        ("learner", "learner_one_optimized"),
        ("learner", "learner_one_common"),
        ("learner", "analytic"),
        ("ordered", "summary"),
        ("ordered", "pooled"),
        ("ordered", "shuffled"),
    )
    for lane in ("b0", "b2"):
        rows = {}
        for left, right in pairs:
            differences = {}
            for seed in TRAIN_SEEDS:
                a = panels[f"n{n}_{lane}_{seed}_{left}_fidelity"]
                b = panels[f"n{n}_{lane}_{seed}_{right}_fidelity"]
                check_endpoint_panel(a, ids, initial)
                check_endpoint_panel(b, ids, initial)
                differences[seed] = a[:, -1] - b[:, -1]
            pooled, by_seed = paired_seed_panel(differences, ids, 1)
            rows[f"{left}_minus_{right}"] = {
                "pointwise_95": pooled,
                "per_seed": by_seed,
            }
        ans[lane] = rows
    return ans


def refresh_descriptions(results):
    file = results / "shared_control_results.json"
    report = json.loads(file.read_text())
    with np.load(results / "shared_control_episode_panels.npz") as panels:
        for n in (3, 4):
            report["systems"][str(n)]["descriptive_paired_history_contrasts"] = (
                paired_history(panels, n)
            )
    atomic_json(file, report)


def aggregate(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reports = {
        lane: registered_reports(
            args.results / lane, args.artifacts / lane, seeds=TRAIN_SEEDS, long_run=True
        )
        for lane in ("b0", "b2")
    }
    out = {
        "schema": "shared_b0_b2_native_control_v1",
        "upstream_update_rung": 32768,
        "seeds": list(TRAIN_SEEDS),
        "control_sizes": [3, 4],
        "upstream_training_sizes": [3, 4, 8],
        "fitted_separate_models": 6,
        "gain_harm_family": 14,
        "confidence_unit": "initial physical episode, fixed three fitted replicas",
        "frozen_export_readback": {},
        "systems": {},
    }
    for lane, records in reports.items():
        proofs = {
            str(s): check_shared_seed(
                args.artifacts / lane / f"seed_{s}", records[s], lane, s
            )
            for s in TRAIN_SEEDS
        }
        if len({proof["classifier_sha256"] for proof in proofs.values()}) != 1:
            raise ValueError("seed-specific upstream classifier in shared comparison")
        out["frozen_export_readback"][lane] = proofs
    compact = {}
    for n in (3, 4):
        ids, initial = confirmation_axis(n, long_run=True, prefix=CONFIRM)
        compact[f"n{n}_episode_ids"] = ids
        data = {}
        arm_metrics = {}
        rows = {}
        for lane, records in reports.items():
            arm_metrics[lane] = {}
            for arm in LONG_ARMS:
                curves = {}
                acts = {}
                for seed, report in records.items():
                    root = args.artifacts / lane / f"seed_{seed}"
                    with np.load(root / f"long_n{n}_{arm}.npz") as f:
                        curve, active = checked_archive(
                            f,
                            report["systems"][str(n)]["arms"][arm],
                            n=n,
                            ids=ids,
                            initial=initial,
                        )
                    curves[seed] = curve
                    acts[seed] = active
                    compact[f"n{n}_{lane}_{seed}_{arm}_fidelity"] = curve
                    compact[f"n{n}_{lane}_{seed}_{arm}_active"] = active
                data[lane, arm] = curves
                cube = np.stack([curves[s] for s in TRAIN_SEEDS])
                active = np.stack([acts[s] for s in TRAIN_SEEDS])
                gains = cube - cube[:, :, :1]
                harms = gains[:, :, -1] < -1e-10
                k = int(harms.any(0).sum())
                arm_metrics[lane][arm] = {
                    "terminal_gain": paired_interval(gains[:, :, -1].mean(0)),
                    "terminal_harm": paired_interval(
                        harms.mean(0), FAMILY if arm == "ordered" else 1
                    ),
                    "exact_any_seed_harm_upper": 1.0
                    if k == len(ids)
                    else float(beta.ppf(1 - 0.025 / FAMILY, k + 1, len(ids) - k)),
                    "coverage": float(active.mean()),
                    "round_mean_gain": gains.mean((0, 1)).tolist(),
                    "late_gain": float((cube[:, :, -1] - cube[:, :, 5]).mean()),
                    "any_step_harm": float(
                        (np.diff(cube, axis=2) < -1e-10).any(2).mean()
                    ),
                }
        for lane, records in reports.items():
            for comparator in ("ordered_one_optimized", "best_development_classical"):
                deltas = {}
                for seed in TRAIN_SEEDS:
                    reference = (
                        records[seed]["systems"][str(n)]["best_development_classical"]
                        if comparator == "best_development_classical"
                        else comparator
                    )
                    deltas[seed] = (
                        data[lane, "ordered"][seed][:, -1]
                        - data[lane, reference][seed][:, -1]
                    )
                pooled, by_seed = paired_seed_panel(deltas, ids, FAMILY)
                rows[f"{lane}_ordered_minus_{comparator}"] = {
                    "across_seed_panel": pooled,
                    "per_seed": by_seed,
                }
        diff = {
            s: data["b2", "ordered"][s][:, -1] - data["b0", "ordered"][s][:, -1]
            for s in TRAIN_SEEDS
        }
        pooled, per = paired_seed_panel(diff, ids, FAMILY)
        rows["b2_minus_b0_ordered"] = {"across_seed_panel": pooled, "per_seed": per}
        channel = {}
        for lane in ("b0", "b2"):
            diffs = {}
            for seed in TRAIN_SEEDS:
                with np.load(
                    args.artifacts
                    / lane
                    / f"seed_{seed}"
                    / f"public_command_channel_n{n}.npz"
                ) as z:
                    f = z["fidelity"]

                    check_endpoint_panel(f, ids, initial)
                    if not (z["native_counts"].sum((1, 2, 3)) == 77000).all():
                        raise ValueError("unmatched ablation cost")
                    diffs[seed] = data[lane, "ordered"][seed][:, -1] - f[:, -1]
                    compact[f"n{n}_{lane}_{seed}_public_command_channel_fidelity"] = f
            score, by_seed = paired_seed_panel(diffs, ids, 1)
            channel[lane] = {
                "orientation": "full ordered minus public-command state semantic channel",
                "paired_95": score,
                "per_seed": by_seed,
            }
        out["systems"][str(n)] = {
            "registered_primary": rows,
            "descriptive_state_channel_ablation": channel,
            "descriptive_paired_history_contrasts": paired_history(compact, n),
            "descriptive_arms": arm_metrics,
            "episodes": len(ids),
        }
    np.savez_compressed(args.results / "shared_control_episode_panels.npz", **compact)
    atomic_json(args.results / "shared_control_results.json", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument(
        "--from-verified-panels",
        action="store_true",
        help="refresh only descriptive intervals from saved complete paired outcomes",
    )
    args = p.parse_args()
    if args.from_verified_panels:
        refresh_descriptions(args.results)
    else:
        aggregate(args)
