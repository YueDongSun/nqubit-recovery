"""Track B research evaluation: validation-fold comparison of pilot checkpoints.

Campaign v10_campaign_20261001 (user-authorized research route). Read-only:
opens the admitted real V2.4 4Q train and validation outer folds, rebuilds the
registered effective_family_context_v3 model, and scores:

1. the untrained initialization (prior baseline, seeded like the pilot),
2. the fixed-visible-marginal non-learned baseline,
3. each trained pilot checkpoint (191-step repro, 2000-step cap run),

with the registered objective, native four-cell per-row NLL, and evaluation
metrics on the held-out validation fold (and the train fold for reference).

Run with PYTHONPATH pointed at the clean f8e4a813 Track B worktree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def _score(model, batch, objective, metrics_fn, per_row_fn, label):
    prediction = model(batch)
    losses = objective(prediction, batch)
    metrics = metrics_fn(prediction, batch)
    per_row = per_row_fn(prediction, batch)
    rows = per_row.detach().float()
    return {
        "label": label,
        "losses": {key: float(value.detach().cpu())
                   for key, value in losses.items()
                   if isinstance(value, torch.Tensor) and value.ndim == 0},
        "metrics": {key: float(value.detach().cpu())
                    for key, value in metrics.items()
                    if isinstance(value, torch.Tensor) and value.ndim == 0},
        "per_row_four_cell_nll": {
            "mean": float(rows.mean()), "median": float(rows.median()),
            "p25": float(rows.quantile(0.25)), "p75": float(rows.quantile(0.75)),
            "rows": int(rows.numel()),
        },
        "_per_row_vector": rows.tolist(),
    }


def _paired_stats(baseline_vec, model_vec, group_ids=None):
    """Paired per-row NLL improvement (baseline - model), group-aware bootstrap."""
    import numpy as np
    base = np.asarray(baseline_vec, dtype=np.float64)
    model = np.asarray(model_vec, dtype=np.float64)
    diff = base - model  # positive => model better (lower NLL)
    out = {
        "n_rows": int(diff.size),
        "mean_improvement_nats": float(diff.mean()),
        "median_improvement_nats": float(np.median(diff)),
        "frac_rows_improved": float((diff > 0).mean()),
        "baseline_mean": float(base.mean()),
        "model_mean": float(model.mean()),
        "relative_reduction_pct": float(100.0 * diff.mean() / base.mean())
        if base.mean() else None,
    }
    rng = np.random.default_rng(20261001)
    if group_ids is not None and len(group_ids) == diff.size:
        groups = np.asarray(group_ids)
        unique = np.unique(groups)
        boot = []
        for _ in range(2000):
            pick = rng.choice(unique, size=unique.size, replace=True)
            idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
            boot.append(diff[idx].mean())
        out["ci_method"] = "group_bootstrap_2000"
        out["n_groups"] = int(unique.size)
    else:
        boot = [diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(2000)]
        out["ci_method"] = "row_bootstrap_2000"
    boot = np.asarray(boot)
    out["improvement_ci95_low"] = float(np.quantile(boot, 0.025))
    out["improvement_ci95_high"] = float(np.quantile(boot, 0.975))
    out["improvement_lower_bound_exceeds_zero"] = bool(out["improvement_ci95_low"] > 0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--system-type", default="4qubit")
    parser.add_argument("--basis-config-root", required=True)
    parser.add_argument("--global-initialization", required=True)
    parser.add_argument("--expected-global-initialization-sha", required=True)
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="label=path, repeatable")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.cli import (
        _build_real_pilot_model, seed_trackb_runtime,
    )
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.datasets import (
        open_trackb_v24_real_static_fold, trackb_real_static_evaluation_collate,
    )
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.effective_observation_loss import (
        effective_observation_objective,
    )
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.effective_observation_evaluation import (
        effective_observation_metrics, per_row_native_four_cell_nll,
    )
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.evaluation import (
        build_trackb_fixed_visible_marginal_prediction,
    )
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.global_count_initialization import (
        load_global_initialization_bytes,
    )

    initialization = load_global_initialization_bytes(
        Path(args.global_initialization).read_bytes(),
        expected_sha=args.expected_global_initialization_sha,
    )

    folds = {}
    for role in ("validation", "calibration"):
        folds[role] = open_trackb_v24_real_static_fold(
            args.dataset_root, build_id=args.build_id,
            system_type=args.system_type,
            basis_config_root=args.basis_config_root, outer_role=role,
        )
    train_fold = open_trackb_v24_real_static_fold(
        args.dataset_root, build_id=args.build_id,
        system_type=args.system_type,
        basis_config_root=args.basis_config_root, outer_role="train",
    )
    first = train_fold[0]

    def build_model():
        seed_trackb_runtime(20261001)
        namespace = SimpleNamespace(
            system_type=args.system_type,
            observation_model="effective_family_context_v3",
            global_initialization=args.global_initialization,
        )
        return _build_real_pilot_model(
            namespace, first, len(train_fold.session_index),
            global_initialization=initialization,
            operation_metadata_sha=getattr(train_fold, "operation_metadata_sha", None),
        )

    checkpoints = {}
    for entry in args.checkpoint:
        label, _, path = entry.partition("=")
        checkpoints[label] = path

    report = {
        "schema": "TrackBResearchValidationEvaluationV1",
        "route_id": "user_authorized_research_route_v1",
        "research_only": True,
        "scientific_gate_evaluated": False,
        "promotion_ready": False,
        "selection_evidence": False,
        "hardware_action_authorized": False,
        "dataset_root": args.dataset_root,
        "build_id": args.build_id,
        "system_type": args.system_type,
        "folds": {role: {"rows": len(dataset),
                         "outer_fold_sha": dataset.fold_plan.sha256()}
                  for role, dataset in folds.items()},
        "evaluations": {},
    }

    for role, dataset in folds.items():
        items = [dataset[index] for index in range(len(dataset))]
        batch = trackb_real_static_evaluation_collate(items, outer_role=role)
        group_ids = batch.get("group_ids", batch.get("group_id"))
        if isinstance(group_ids, torch.Tensor):
            group_codes = group_ids.detach().cpu().tolist()
        elif isinstance(group_ids, (list, tuple)):
            label_to_code = {}
            group_codes = []
            for value in group_ids:
                key = str(value)
                if key not in label_to_code:
                    label_to_code[key] = len(label_to_code)
                group_codes.append(label_to_code[key])
        else:
            group_codes = None
        role_report = []
        with torch.no_grad():
            baseline_prediction = build_trackb_fixed_visible_marginal_prediction(batch)
            baseline_rows = per_row_native_four_cell_nll(baseline_prediction, batch).detach().float()
            baseline_vec = baseline_rows.tolist()
            role_report.append({
                "label": "fixed_visible_marginal_baseline",
                "per_row_four_cell_nll": {
                    "mean": float(baseline_rows.mean()), "median": float(baseline_rows.median()),
                    "p25": float(baseline_rows.quantile(0.25)),
                    "p75": float(baseline_rows.quantile(0.75)),
                    "rows": int(baseline_rows.numel())},
                "_per_row_vector": baseline_vec,
            })
            role_report.append(_score(
                build_model(), batch, effective_observation_objective,
                effective_observation_metrics, per_row_native_four_cell_nll,
                "untrained_initialization"))
            for label, path in checkpoints.items():
                model = build_model()
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                state = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state)
                try:
                    role_report.append(_score(
                        model, batch, effective_observation_objective,
                        effective_observation_metrics, per_row_native_four_cell_nll,
                        f"checkpoint:{label}"))
                except (IndexError, RuntimeError) as error:
                    role_report.append({"label": f"checkpoint:{label}",
                                        "error": f"{type(error).__name__}: {error}"})
        paired = {}
        for entry in role_report:
            vector = entry.get("_per_row_vector")
            if vector is not None and entry["label"] != "fixed_visible_marginal_baseline":
                paired[entry["label"]] = _paired_stats(baseline_vec, vector, group_codes)
        report["evaluations"][role] = role_report
        report.setdefault("paired_vs_baseline", {})[role] = paired

    for dataset in (*folds.values(), train_fold):
        dataset.close()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(output_path),
                      "validation": report["evaluations"].get("validation")},
                     indent=2)[:4000])


if __name__ == "__main__":
    main()
