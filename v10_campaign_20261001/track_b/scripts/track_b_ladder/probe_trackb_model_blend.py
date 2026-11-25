"""Track B model-blend probe: post-hoc calibration and saturated-anchor blending
of the trained effective_family_context_v3 checkpoint.

Campaign v10_campaign_20261001 (user-authorized research route), research-only,
read-only with respect to folds and checkpoints. The headroom probe showed a
no-training estimator (per-op empirical + row-visible mixture, fitted
concentration) beats the trained checkpoint. This probe measures how much of
that gap is recoverable WITHOUT retraining:

  G_refit_concentration: trained model probabilities, global concentration
     re-fitted on the other held-out fold (calibration-layer fix only).
  H_blend_per_op: w * trained-model probabilities + (1-w) * per-op empirical
     probabilities, (w, c) fitted on the other held-out fold.

Everything is scored with the registered ``per_row_native_four_cell_nll`` and
compared with paired group-aware bootstrap against the trained model itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from probe_trackb_headroom import (
    CONCENTRATION_GRID, MIXTURE_WEIGHT_GRID, _paired_group_bootstrap,
    _per_op_probabilities, _score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--basis-config-root", required=True)
    parser.add_argument("--system-type", required=True)
    parser.add_argument("--global-initialization", required=True)
    parser.add_argument("--expected-global-initialization-sha", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trained-label", required=True)
    parser.add_argument("--trained-eval-json", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.cli import (
        _build_real_pilot_model, seed_trackb_runtime,
    )
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.datasets import (
        open_trackb_v24_real_static_fold, trackb_real_static_collate,
        trackb_real_static_evaluation_collate,
    )
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.effective_observation_evaluation import (
        per_row_native_four_cell_nll,
    )
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.global_count_initialization import (
        load_global_initialization_bytes,
    )

    initialization = load_global_initialization_bytes(
        Path(args.global_initialization).read_bytes(),
        expected_sha=args.expected_global_initialization_sha,
    )
    folds = {}
    for role in ("train", "validation", "calibration"):
        folds[role] = open_trackb_v24_real_static_fold(
            args.dataset_root, build_id=args.build_id, system_type=args.system_type,
            basis_config_root=args.basis_config_root, outer_role=role,
        )
    train_dataset = folds["train"]
    train_items = [train_dataset[index] for index in range(len(train_dataset))]
    first = train_items[0]
    train_batch = trackb_real_static_collate(train_items)

    seed_trackb_runtime(20261001)
    namespace = SimpleNamespace(
        system_type=args.system_type,
        observation_model="effective_family_context_v3",
        global_initialization=args.global_initialization,
    )
    model = _build_real_pilot_model(
        namespace, first, len(train_dataset.session_index),
        global_initialization=initialization,
        operation_metadata_sha=getattr(train_dataset, "operation_metadata_sha", None),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    per_op = _per_op_probabilities(train_batch)
    recorded = json.loads(Path(args.trained_eval_json).read_text(encoding="utf-8"))
    trained_vectors = {}
    for role, entries in recorded["evaluations"].items():
        for entry in entries:
            if entry.get("label") == args.trained_label and "_per_row_vector" in entry:
                trained_vectors[role] = entry["_per_row_vector"]

    batches = {}
    model_probabilities = {}
    with torch.no_grad():
        for role in ("validation", "calibration"):
            dataset = folds[role]
            items = [dataset[index] for index in range(len(dataset))]
            batch = trackb_real_static_evaluation_collate(items, outer_role=role)
            prediction = model(batch)
            batches[role] = batch
            model_probabilities[role] = {
                "signal": prediction["signal_probability"].detach(),
                "coin": prediction["coin_given_signal_probability"].detach(),
                "idler": prediction["idler_given_no_signal_probability"].detach(),
            }

    report = {
        "schema": "TrackBModelBlendProbeV1",
        "route_id": "user_authorized_research_route_v1",
        "research_only": True,
        "scientific_gate_evaluated": False,
        "promotion_ready": False,
        "selection_evidence": False,
        "hardware_action_authorized": False,
        "system_type": args.system_type,
        "checkpoint": args.checkpoint,
        "estimators": {},
        "paired_vs_trained": {},
    }

    vectors = {role: {} for role in ("validation", "calibration")}

    # Score the trained model with its OWN concentrations (per-row vector).
    with torch.no_grad():
        for role in ("validation", "calibration"):
            batch = batches[role]
            prediction = model(batch)
            own_prediction = {
                "signal_probability": model_probabilities[role]["signal"],
                "coin_given_signal_probability": model_probabilities[role]["coin"],
                "idler_given_no_signal_probability": model_probabilities[role]["idler"],
                "four_cell_concentration": prediction["four_cell_concentration"].detach(),
            }
            vectors[role]["T_trained_model"] = per_row_native_four_cell_nll(
                own_prediction, batch).detach().float().tolist()

    fit_directions = {
        "fit_calibration_score_validation": ("calibration", "validation"),
        "fit_validation_score_calibration": ("validation", "calibration"),
    }
    for direction, (fit_role, score_role) in fit_directions.items():
        fit_batch = batches[fit_role]
        best_g = {"c": 200.0, "loss": float("inf")}
        for candidate in CONCENTRATION_GRID:
            loss = float(np.sum(_score(
                model_probabilities[fit_role], candidate, fit_batch,
                per_row_native_four_cell_nll)))
            if loss < best_g["loss"]:
                best_g = {"c": float(candidate), "loss": loss}
        rows, ops = fit_batch["four_cell_counts"].shape[:2]
        best_h = {"w": 0.5, "c": 200.0, "loss": float("inf")}
        for weight in MIXTURE_WEIGHT_GRID:
            blended = {
                name: (weight * model_probabilities[fit_role][name]
                       + (1.0 - weight) * per_op[name].unsqueeze(0).expand(rows, ops))
                .clamp(1.0e-5, 1.0 - 1.0e-5)
                for name in ("signal", "coin", "idler")
            }
            for candidate in CONCENTRATION_GRID:
                loss = float(np.sum(_score(
                    blended, candidate, fit_batch, per_row_native_four_cell_nll)))
                if loss < best_h["loss"]:
                    best_h = {"w": float(weight), "c": float(candidate), "loss": loss}

        score_batch = batches[score_role]
        score_rows, score_ops = score_batch["four_cell_counts"].shape[:2]
        vectors[score_role][f"G_refit_concentration_{direction}"] = _score(
            model_probabilities[score_role], best_g["c"], score_batch,
            per_row_native_four_cell_nll)
        blended_score = {
            name: (best_h["w"] * model_probabilities[score_role][name]
                   + (1.0 - best_h["w"]) * per_op[name].unsqueeze(0).expand(score_rows, score_ops))
            .clamp(1.0e-5, 1.0 - 1.0e-5)
            for name in ("signal", "coin", "idler")
        }
        vectors[score_role][f"H_blend_per_op_{direction}"] = _score(
            blended_score, best_h["c"], score_batch, per_row_native_four_cell_nll)
        report["estimators"][direction] = {
            "G_refit_concentration": best_g["c"],
            "H_blend_weight": best_h["w"],
            "H_blend_concentration": best_h["c"],
        }

    for role in ("validation", "calibration"):
        batch = batches[role]
        group_ids = batch["group_ids"]
        label_to_code = {}
        group_codes = []
        for value in group_ids:
            key = str(value)
            label_to_code.setdefault(key, len(label_to_code))
            group_codes.append(label_to_code[key])
        report["estimators"][role] = {
            name: {"mean": float(np.mean(values)), "median": float(np.median(values))}
            for name, values in sorted(vectors[role].items())
        }
        report["paired_vs_trained"][role] = {
            name: _paired_group_bootstrap(
                trained_vectors.get(role, vectors[role]["T_trained_model"]),
                values, group_codes)
            for name, values in sorted(vectors[role].items())
            if name != "T_trained_model"
        }

    for dataset in folds.values():
        dataset.close()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "system": args.system_type,
        "fitted": report["estimators"].get("fit_calibration_score_validation"),
        "validation": report["estimators"]["validation"],
        "validation_vs_trained": {
            key: {k2: round(v2, 4) for k2, v2 in value.items()
                  if k2 in ("mean_improvement_nats", "ci95_low", "ci95_high")}
            for key, value in report["paired_vs_trained"]["validation"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
