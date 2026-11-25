"""Track B headroom probe: how much extractable signal is left in the four-cell channel?

Campaign v10_campaign_20261001 (user-authorized research route), research-only,
read-only with respect to all folds and checkpoints. Motivation: the trained
effective_family_context_v3 model reached held-out four-cell NLL 20.13 (4Q) /
26.01 (3Q). This probe measures whether simple saturated estimators built from
the SAME train fold reach lower NLL, which would bound the remaining
architecture/regularization headroom; if not, further gains need NEW physics
information rather than better optimization.

Estimators (all scored with the registered ``per_row_native_four_cell_nll``):
  A. fixed visible marginal (registered baseline, recomputed)
  F. visible marginal + fitted global concentration
  B. per-operation empirical probabilities from train-fold pooled counts (c=5)
  C. per-operation empirical + fitted global concentration
  E. mixture w*per-op + (1-w)*row-visible, (w, c) fitted on the other held-out
     fold only (calibration->validation and validation->calibration)

Each estimator is compared to the recorded per-row vectors of the trained
one-pass checkpoints (repro_191 4Q / repro_110 3Q) with paired group-aware
bootstrap. No model training occurs; no gated code is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CONCENTRATION_GRID = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)
MIXTURE_WEIGHT_GRID = tuple(round(w, 2) for w in np.arange(0.0, 1.0001, 0.1))


def _paired_group_bootstrap(baseline_vec, model_vec, group_codes, seed=20261001):
    base = np.asarray(baseline_vec, dtype=np.float64)
    model = np.asarray(model_vec, dtype=np.float64)
    diff = base - model
    out = {
        "n_rows": int(diff.size),
        "mean_improvement_nats": float(diff.mean()),
        "frac_rows_improved": float((diff > 0).mean()),
        "baseline_mean": float(base.mean()),
        "model_mean": float(model.mean()),
    }
    rng = np.random.default_rng(seed)
    groups = np.asarray(group_codes)
    unique = np.unique(groups)
    boot = []
    for _ in range(2000):
        pick = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        boot.append(diff[idx].mean())
    boot = np.asarray(boot)
    out["ci95_low"] = float(np.quantile(boot, 0.025))
    out["ci95_high"] = float(np.quantile(boot, 0.975))
    out["lower_bound_exceeds_zero"] = bool(out["ci95_low"] > 0)
    return out


def _per_op_probabilities(train_batch):
    """Pool train-fold counts per operation -> (signal, coin, idler) [ops]."""
    counts = train_batch["four_cell_counts"].to(dtype=torch.float32)
    available = train_batch["operation_mask"].to(dtype=torch.float32)
    pooled = (counts * available.unsqueeze(-1)).sum(dim=0)  # [ops, 4]
    total = pooled.sum(dim=-1).clamp_min(1.0)
    signal_total = (pooled[:, 1] + pooled[:, 3]).clamp_min(1.0)
    no_signal_total = (pooled[:, 0] + pooled[:, 2]).clamp_min(1.0)
    signal = (signal_total / total).clamp(1.0e-5, 1.0 - 1.0e-5)
    coin = (pooled[:, 3] / signal_total).clamp(1.0e-5, 1.0 - 1.0e-5)
    idler = (pooled[:, 2] / no_signal_total).clamp(1.0e-5, 1.0 - 1.0e-5)
    return {
        "signal": signal,
        "coin": coin,
        "idler": idler,
        "train_rows_per_op": available.sum(dim=0),
        "pooled_counts_per_op": pooled.sum(dim=-1),
    }


def _row_visible_probabilities(batch):
    """Row-local visible-marginal probabilities, broadcast over operations."""
    counts = batch["four_cell_counts"].to(dtype=torch.float32)
    visible = batch["visible_operation_mask"].to(dtype=torch.float32)
    aggregate = (counts * visible.unsqueeze(-1)).sum(dim=1)  # [rows, 4]
    total = aggregate.sum(dim=-1).clamp_min(1.0)
    signal_total = (aggregate[:, 1] + aggregate[:, 3]).clamp_min(1.0)
    no_signal_total = (aggregate[:, 0] + aggregate[:, 2]).clamp_min(1.0)
    signal = (signal_total / total).clamp(1.0e-5, 1.0 - 1.0e-5)
    coin = (aggregate[:, 3] / signal_total).clamp(1.0e-5, 1.0 - 1.0e-5)
    idler = (aggregate[:, 2] / no_signal_total).clamp(1.0e-5, 1.0 - 1.0e-5)
    return {"signal": signal, "coin": coin, "idler": idler}


def _prediction_dict(probabilities, concentration_value, row_count, op_count):
    signal = probabilities["signal"]
    coin = probabilities["coin"]
    idler = probabilities["idler"]
    if signal.ndim == 1 and signal.shape[0] == op_count:
        expand = lambda tensor: tensor.unsqueeze(0).expand(row_count, op_count)
    elif signal.ndim == 1 and signal.shape[0] == row_count:
        expand = lambda tensor: tensor.unsqueeze(-1).expand(row_count, op_count).clone()
    else:
        expand = lambda tensor: tensor
    return {
        "signal_probability": expand(signal),
        "coin_given_signal_probability": expand(coin),
        "idler_given_no_signal_probability": expand(idler),
        "four_cell_concentration": torch.full((row_count,), float(concentration_value)),
    }


def _score(probabilities, concentration_value, batch, nll_fn):
    rows, ops = batch["four_cell_counts"].shape[:2]
    prediction = _prediction_dict(probabilities, concentration_value, rows, ops)
    return nll_fn(prediction, batch).detach().float().tolist()


def _fit_concentration(probabilities_by_op, batch, nll_fn):
    """Grid-fit ONE global concentration minimizing summed per-row NLL."""
    best_c, best_loss = 5.0, float("inf")
    for candidate in CONCENTRATION_GRID:
        vec = _score(probabilities_by_op, candidate, batch, nll_fn)
        loss = float(np.sum(vec))
        if loss < best_loss:
            best_loss, best_c = loss, candidate
    return best_c


def _mix_probabilities(per_op, row_visible, weight, row_count, op_count):
    """Broadcast per-op [ops] and row-visible [rows] to [rows, ops] and blend."""
    per_op_b = {
        name: tensor.unsqueeze(0).expand(row_count, op_count)
        for name, tensor in per_op.items()
    }
    row_visible_b = {
        name: tensor.unsqueeze(-1).expand(row_count, op_count).clone()
        for name, tensor in row_visible.items()
    }
    return {
        name: (weight * per_op_b[name] + (1.0 - weight) * row_visible_b[name])
        .clamp(1.0e-5, 1.0 - 1.0e-5)
        for name in ("signal", "coin", "idler")
    }


def _fit_mixture(per_op, row_visible, batch, nll_fn):
    """Jointly fit (w, c) minimizing summed NLL; returns best parameters."""
    rows, ops = batch["four_cell_counts"].shape[:2]
    best = {"w": 0.5, "c": 5.0, "loss": float("inf")}
    for weight in MIXTURE_WEIGHT_GRID:
        mixed = _mix_probabilities(per_op, row_visible, weight, rows, ops)
        for concentration in CONCENTRATION_GRID:
            vec = _score(mixed, concentration, batch, nll_fn)
            loss = float(np.sum(vec))
            if loss < best["loss"]:
                best = {"w": float(weight), "c": float(concentration), "loss": loss}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--basis-config-root", required=True)
    parser.add_argument("--system-type", default="4qubit")
    parser.add_argument("--trained-eval-json", required=True,
                        help="recorded validation_evaluation.json with per-row vectors")
    parser.add_argument("--trained-label", default="checkpoint:repro_191")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.datasets import (
        open_trackb_v24_real_static_fold, trackb_real_static_collate,
        trackb_real_static_evaluation_collate,
    )
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.effective_observation_evaluation import (
        per_row_native_four_cell_nll,
    )

    folds = {}
    for role in ("train", "validation", "calibration"):
        folds[role] = open_trackb_v24_real_static_fold(
            args.dataset_root, build_id=args.build_id, system_type=args.system_type,
            basis_config_root=args.basis_config_root, outer_role=role,
        )

    def collate_fold(role):
        dataset = folds[role]
        items = [dataset[index] for index in range(len(dataset))]
        if role == "train":
            return trackb_real_static_collate(items)
        return trackb_real_static_evaluation_collate(items, outer_role=role)

    train_batch = collate_fold("train")
    held_out = {role: collate_fold(role) for role in ("validation", "calibration")}

    per_op = _per_op_probabilities(train_batch)
    coverage = {
        "train_rows_per_op_min": int(per_op["train_rows_per_op"].min()),
        "train_rows_per_op_median": float(per_op["train_rows_per_op"].median()),
        "pooled_counts_per_op_min": float(per_op["pooled_counts_per_op"].min()),
        "pooled_counts_per_op_median": float(per_op["pooled_counts_per_op"].median()),
    }

    recorded = json.loads(Path(args.trained_eval_json).read_text(encoding="utf-8"))
    trained_vectors = {}
    for role, entries in recorded["evaluations"].items():
        for entry in entries:
            if entry.get("label") == args.trained_label and "_per_row_vector" in entry:
                trained_vectors[role] = entry["_per_row_vector"]

    report = {
        "schema": "TrackBHeadroomProbeV1",
        "route_id": "user_authorized_research_route_v1",
        "research_only": True,
        "scientific_gate_evaluated": False,
        "promotion_ready": False,
        "selection_evidence": False,
        "hardware_action_authorized": False,
        "system_type": args.system_type,
        "dataset_root": args.dataset_root,
        "build_id": args.build_id,
        "train_coverage": coverage,
        "estimators": {},
        "paired_vs_trained": {},
    }

    vectors = {}
    for role, batch in held_out.items():
        rows, ops = batch["four_cell_counts"].shape[:2]
        group_ids = batch["group_ids"]
        label_to_code = {}
        group_codes = []
        for value in group_ids:
            key = str(value)
            label_to_code.setdefault(key, len(label_to_code))
            group_codes.append(label_to_code[key])

        row_visible = _row_visible_probabilities(batch)
        role_vectors = {
            "A_visible_marginal_c5": _score(row_visible, 5.0, batch, per_row_native_four_cell_nll),
            "B_per_op_empirical_c5": _score(per_op, 5.0, batch, per_row_native_four_cell_nll),
        }
        vectors[role] = {"group_codes": group_codes, **role_vectors}

    # Fit shared tuning parameters on calibration only, apply to validation
    # (and the symmetric direction), never fitting on the scored fold itself.
    fit_directions = {
        "fit_calibration_score_validation": ("calibration", "validation"),
        "fit_validation_score_calibration": ("validation", "calibration"),
    }
    for direction, (fit_role, score_role) in fit_directions.items():
        fit_batch = held_out[fit_role]
        fit_visible = _row_visible_probabilities(fit_batch)
        fitted_concentration = _fit_concentration(per_op, fit_batch, per_row_native_four_cell_nll)
        mixture_fit = _fit_mixture(per_op, fit_visible, fit_batch, per_row_native_four_cell_nll)
        score_batch = held_out[score_role]
        score_visible = _row_visible_probabilities(score_batch)
        score_rows, score_ops = score_batch["four_cell_counts"].shape[:2]
        mixed_probs = _mix_probabilities(
            per_op, score_visible, mixture_fit["w"], score_rows, score_ops)
        vectors[score_role][f"F_visible_marginal_cfitted_{direction}"] = _score(
            score_visible, fitted_concentration, score_batch, per_row_native_four_cell_nll)
        vectors[score_role][f"C_per_op_cfitted_{direction}"] = _score(
            per_op, fitted_concentration, score_batch, per_row_native_four_cell_nll)
        vectors[score_role][f"E_mixture_{direction}"] = _score(
            mixed_probs, mixture_fit["c"], score_batch, per_row_native_four_cell_nll)
        report["estimators"].setdefault(direction, {
            "fitted_concentration": fitted_concentration,
            "mixture_weight": mixture_fit["w"],
            "mixture_concentration": mixture_fit["c"],
        })

    for role, role_vectors in vectors.items():
        group_codes = role_vectors.pop("group_codes")
        report["estimators"][role] = {
            name: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            }
            for name, values in sorted(role_vectors.items())
        }
        trained = trained_vectors.get(role)
        if trained is not None:
            report["paired_vs_trained"][role] = {
                name: _paired_group_bootstrap(trained, values, group_codes)
                for name, values in sorted(role_vectors.items())
            }

    for dataset in folds.values():
        dataset.close()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "system": args.system_type,
        "train_coverage": coverage,
        "validation_estimators": report["estimators"].get("validation"),
        "validation_paired_vs_trained": {
            key: {
                "mean_improvement_nats": value["mean_improvement_nats"],
                "ci95_low": value["ci95_low"],
                "ci95_high": value["ci95_high"],
            }
            for key, value in report["paired_vs_trained"].get("validation", {}).items()
        },
    }, indent=2)[:6000])


if __name__ == "__main__":
    main()
