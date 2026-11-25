"""Track B research v4 observation candidate: saturated anchor + learned tilts.

Campaign v10_campaign_20261001 (user-authorized research route). Motivated by
the headroom probe: a no-training saturated estimator (per-operation empirical
anchor blended with row-visible marginals, fitted concentration) beats the
trained effective_family_context_v3 checkpoint by +1.36 (4Q) / +0.73 (3Q)
nats/row. This module tests whether a minimal learned model can beat that
saturated reference by adding structured per-operation and per-row tilts on
top of the SAME anchor, trained one-pass on the train fold only.

Architecture (all tilts zero-initialized, so training starts exactly at the
anchor and can only move where gradients support it):
  logit(p_branch) = logit(anchor_p) + Linear(per_op_features) + Linear(row_features)
for the three nested branches (signal, coin-given-signal, idler-given-no-signal);
one learnable global log-concentration initialized at log(200).

Feature arms:
  anchor_only : no per-op features (anchor + trainable concentration + row head)
  arch        : projector_descriptor physics columns [0:n^2] (per operation)
  phys        : log-accidental proxy from train-pooled singles + retrieval
                efficiency vector statistics (mean / log-mean / spread)
  full        : arch + phys
Every arm carries the row head (visible-marginal logits + log visible count),
which replaces the fixed global blend weight of probe estimator E with a
learned, row-adaptive blend.

Evaluation: registered ``per_row_native_four_cell_nll`` on the held-out
validation/calibration folds; paired group-aware bootstrap against (a) the
recorded v3 checkpoint per-row vectors, (b) the no-training E-mixture refit on
the opposite fold, (c) the H-blend (v3 probabilities blended with the anchor,
(w, c) fitted on the opposite fold). Research-only; read-only with respect to
folds and checkpoints; no gated code is modified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from probe_trackb_headroom import (
    CONCENTRATION_GRID, MIXTURE_WEIGHT_GRID, _fit_mixture, _mix_probabilities,
    _paired_group_bootstrap, _per_op_probabilities, _row_visible_probabilities,
    _score,
)

BRANCHES = ("signal", "coin", "idler")
ARMS = ("anchor_only", "arch", "phys", "full")
ARM_GROUPS = {
    "anchor_only": (),
    "arch": ("projector",),
    "phys": ("accidental", "retrieval"),
    "full": ("projector", "accidental", "retrieval"),
}


def _logit(probability: torch.Tensor) -> torch.Tensor:
    clipped = probability.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(clipped / (1.0 - clipped))


def _standardize_with(matrix: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return (matrix - mean) / std.clamp_min(1.0e-6)


def _standardize(matrix: torch.Tensor) -> torch.Tensor:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return centered / scale


def _standardize_fit(matrix: torch.Tensor):
    mean = matrix.mean(dim=0, keepdim=True)
    std = matrix.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return mean, std


def _row_features(batch) -> torch.Tensor:
    visible = _row_visible_probabilities(batch)
    counts = batch["four_cell_counts"].to(dtype=torch.float32)
    visible_mask = batch["visible_operation_mask"].to(dtype=torch.float32)
    visible_total = (counts * visible_mask.unsqueeze(-1)).sum(dim=(1, 2)).clamp_min(1.0)
    columns = [
        _logit(visible["signal"]),
        _logit(visible["coin"]),
        _logit(visible["idler"]),
        torch.log(visible_total),
    ]
    return torch.stack(columns, dim=-1)


class ResearchV4Tilt(torch.nn.Module):
    """Zero-init additive logit tilts over the saturated per-operation anchor."""

    def __init__(self, anchor_probabilities, op_feature_dim: int,
                 row_feature_dim: int, init_concentration: float = 200.0):
        super().__init__()
        anchor = torch.stack(
            [_logit(anchor_probabilities[name]) for name in BRANCHES], dim=-1)
        self.register_buffer("anchor_logits", anchor)
        self.op_head = torch.nn.Linear(op_feature_dim, 3)
        self.row_head = torch.nn.Linear(row_feature_dim, 3)
        torch.nn.init.zeros_(self.op_head.weight)
        torch.nn.init.zeros_(self.op_head.bias)
        torch.nn.init.zeros_(self.row_head.weight)
        torch.nn.init.zeros_(self.row_head.bias)
        self.log_concentration = torch.nn.Parameter(
            torch.tensor(float(math.log(init_concentration))))

    def forward(self, op_features: torch.Tensor, row_features: torch.Tensor):
        op_tilt = self.op_head(op_features)
        row_tilt = self.row_head(row_features)
        logits = (
            self.anchor_logits.unsqueeze(0)
            + op_tilt.unsqueeze(0)
            + row_tilt.unsqueeze(1)
        )
        probabilities = torch.sigmoid(logits).clamp(1.0e-5, 1.0 - 1.0e-5)
        concentration = torch.exp(self.log_concentration).expand(row_features.shape[0])
        return {
            "signal_probability": probabilities[..., 0],
            "coin_given_signal_probability": probabilities[..., 1],
            "idler_given_no_signal_probability": probabilities[..., 2],
            "four_cell_concentration": concentration,
        }


def _pooled_cells(train_batch):
    counts = train_batch["four_cell_counts"].to(dtype=torch.float32)
    available = train_batch["operation_mask"].to(dtype=torch.float32)
    return (counts * available.unsqueeze(-1)).sum(dim=0)


def _build_op_feature_groups(groups, train_batch, per_op, system_type: str,
                             efficiency_map_dir: str, n_qubits: int):
    """Assemble standardized per-operation feature groups; returns (matrix, diag)."""
    pooled = _pooled_cells(train_batch)
    signal_singles = pooled[:, 1] + pooled[:, 3]
    idler_singles = pooled[:, 2]

    columns = []
    diagnostics = {}
    if "projector" in groups:
        physics = train_batch["projector_descriptor"][0, :, : n_qubits * n_qubits]
        columns.append(_standardize(physics.to(dtype=torch.float32)))
        diagnostics["projector_dim"] = int(physics.shape[-1])
    if "accidental" in groups:
        accidental = torch.log((signal_singles + 0.5) * (idler_singles + 0.5))
        columns.append(_standardize(accidental.unsqueeze(-1)))
        diagnostics["accidental_proxy_log_mean"] = float(accidental.mean())
    if "retrieval" in groups:
        from ml_recon.physics_hybrid_v7.physical_ceiling import (
            load_retrieval_rate_vectors,
        )
        vectors, _info = load_retrieval_rate_vectors(
            system_type, int(pooled.shape[0]),
            efficiency_map_path=str(Path(efficiency_map_dir)
                                    / f"retrieval_efficiency_"
                                      f"{3 if system_type == '3qubit' else 4}qubit.txt"))
        vectors = torch.from_numpy(np.asarray(vectors, dtype=np.float32))
        retrieval_columns = torch.stack((
            vectors.mean(dim=-1),
            torch.log(vectors.mean(dim=-1).clamp_min(1.0e-12)),
            vectors.std(dim=-1),
        ), dim=-1)
        columns.append(_standardize(retrieval_columns))
        correlation = float(np.corrcoef(
            vectors.mean(dim=-1).numpy(), per_op["coin"].numpy())[0, 1])
        diagnostics["retrieval_mean_vs_coin_rate_correlation"] = correlation
    if not columns:
        columns.append(torch.zeros(pooled.shape[0], 1))
    return torch.cat(columns, dim=-1), diagnostics


def _train_arm(anchor_probabilities, op_features, train_row_features, train_batch,
               learning_rate: float, batch_rows: int, l2_weight: float,
               nll_fn):
    torch.manual_seed(20261001)
    model = ResearchV4Tilt(
        anchor_probabilities, op_features.shape[-1], train_row_features.shape[-1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0)
    row_total = train_batch["four_cell_counts"].shape[0]
    trajectory = []
    model.train()
    step = 0
    for start in range(0, row_total, batch_rows):
        stop = min(start + batch_rows, row_total)
        optimizer.zero_grad()
        prediction = model(op_features, train_row_features[start:stop])
        slice_batch = {
            "four_cell_counts": train_batch["four_cell_counts"][start:stop],
            "operation_mask": train_batch["operation_mask"][start:stop],
            "query_mask": train_batch["query_mask"][start:stop],
        }
        nll = nll_fn(prediction, slice_batch).mean()
        l2 = l2_weight * (
            model.op_head.weight.square().sum()
            + model.row_head.weight.square().sum())
        loss = nll + l2
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        trajectory.append(float(loss.detach()))
        step += 1
    model.eval()
    return model, {
        "steps": step,
        "final_train_loss": trajectory[-1] if trajectory else None,
        "initial_train_loss": trajectory[0] if trajectory else None,
        "learned_concentration": float(math.exp(model.log_concentration.detach())),
        "op_head_weight_norm": float(model.op_head.weight.detach().norm()),
        "row_head_weight_norm": float(model.row_head.weight.detach().norm()),
    }


def _load_v3_model(args, train_dataset, first, initialization):
    from ml_recon.physics_hybrid_v10.trackb_realfd_v1.cli import (
        _build_real_pilot_model, seed_trackb_runtime,
    )
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
    return model


def _fit_h_blend(model_probabilities, per_op, fit_batch, nll_fn):
    """Fit (w, c) for v3-probabilities blended with the per-op anchor."""
    rows, ops = fit_batch["four_cell_counts"].shape[:2]
    best = {"w": 0.5, "c": 200.0, "loss": float("inf")}
    for weight in MIXTURE_WEIGHT_GRID:
        blended = {
            name: (weight * model_probabilities[name]
                   + (1.0 - weight) * per_op[name].unsqueeze(0).expand(rows, ops))
            .clamp(1.0e-5, 1.0 - 1.0e-5)
            for name in BRANCHES
        }
        for concentration in CONCENTRATION_GRID:
            loss = float(np.sum(_score(blended, concentration, fit_batch, nll_fn)))
            if loss < best["loss"]:
                best = {"w": float(weight), "c": float(concentration), "loss": loss}
    return best


def _blend_model_anchor(model_probabilities, per_op, weight, rows, ops):
    return {
        name: (weight * model_probabilities[name]
               + (1.0 - weight) * per_op[name].unsqueeze(0).expand(rows, ops))
        .clamp(1.0e-5, 1.0 - 1.0e-5)
        for name in BRANCHES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--basis-config-root", required=True,
                        help="directory holding retrieval_efficiency_*.txt and basis configs")
    parser.add_argument("--system-type", required=True)
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--global-initialization", required=True)
    parser.add_argument("--expected-global-initialization-sha", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="trained v3 checkpoint for the H-blend reference")
    parser.add_argument("--trained-label", required=True)
    parser.add_argument("--trained-eval-json", required=True)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--batch-rows", type=int, default=4)
    parser.add_argument("--l2-weight", type=float, default=1.0e-3)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

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

    folds = {}
    for role in ("train", "validation", "calibration"):
        folds[role] = open_trackb_v24_real_static_fold(
            args.dataset_root, build_id=args.build_id, system_type=args.system_type,
            basis_config_root=args.basis_config_root, outer_role=role,
        )
    train_dataset = folds["train"]
    train_items = [train_dataset[index] for index in range(len(train_dataset))]
    train_batch = trackb_real_static_collate(train_items)
    per_op = _per_op_probabilities(train_batch)
    anchor = {name: per_op[name] for name in BRANCHES}
    raw_row_features = _row_features(train_batch)
    row_feature_mean, row_feature_std = _standardize_fit(raw_row_features)
    train_row_features = _standardize_with(raw_row_features, row_feature_mean, row_feature_std)

    feature_cache = {}
    feature_diagnostics = {}
    for arm, groups in ARM_GROUPS.items():
        matrix, diagnostics = _build_op_feature_groups(
            groups, train_batch, per_op, args.system_type,
            args.basis_config_root, args.n_qubits)
        feature_cache[arm] = matrix
        feature_diagnostics[arm] = diagnostics

    initialization = load_global_initialization_bytes(
        Path(args.global_initialization).read_bytes(),
        expected_sha=args.expected_global_initialization_sha,
    )
    v3_model = _load_v3_model(args, train_dataset, train_items[0], initialization)

    eval_batches = {}
    eval_row_features = {}
    v3_probabilities = {}
    with torch.no_grad():
        for role in ("validation", "calibration"):
            dataset = folds[role]
            items = [dataset[index] for index in range(len(dataset))]
            batch = trackb_real_static_evaluation_collate(items, outer_role=role)
            eval_batches[role] = batch
            eval_row_features[role] = _standardize_with(
                _row_features(batch), row_feature_mean, row_feature_std)
            prediction = v3_model(batch)
            v3_probabilities[role] = {
                "signal": prediction["signal_probability"].detach(),
                "coin": prediction["coin_given_signal_probability"].detach(),
                "idler": prediction["idler_given_no_signal_probability"].detach(),
            }

    recorded = json.loads(Path(args.trained_eval_json).read_text(encoding="utf-8"))
    trained_vectors = {}
    for role, entries in recorded["evaluations"].items():
        for entry in entries:
            if entry.get("label") == args.trained_label and "_per_row_vector" in entry:
                trained_vectors[role] = entry["_per_row_vector"]

    # References refit strictly on the opposite held-out fold.
    references = {}
    with torch.no_grad():
        for fit_role, score_role in (
            ("calibration", "validation"),
            ("validation", "calibration"),
        ):
            fit_batch = eval_batches[fit_role]
            fit_visible = _row_visible_probabilities(fit_batch)
            e_fit = _fit_mixture(per_op, fit_visible, fit_batch, per_row_native_four_cell_nll)
            h_fit = _fit_h_blend(
                v3_probabilities[fit_role], per_op, fit_batch, per_row_native_four_cell_nll)
            score_batch = eval_batches[score_role]
            rows, ops = score_batch["four_cell_counts"].shape[:2]
            score_visible = _row_visible_probabilities(score_batch)
            e_probs = _mix_probabilities(per_op, score_visible, e_fit["w"], rows, ops)
            references[(score_role, "E_mixture")] = (
                _score(e_probs, e_fit["c"], score_batch, per_row_native_four_cell_nll),
                {"w": e_fit["w"], "c": e_fit["c"]})
            h_probs = _blend_model_anchor(
                v3_probabilities[score_role], per_op, h_fit["w"], rows, ops)
            references[(score_role, "H_blend")] = (
                _score(h_probs, h_fit["c"], score_batch, per_row_native_four_cell_nll),
                {"w": h_fit["w"], "c": h_fit["c"]})

    report = {
        "schema": "TrackBResearchV4RunV1",
        "route_id": "user_authorized_research_route_v1",
        "research_only": True,
        "scientific_gate_evaluated": False,
        "promotion_ready": False,
        "selection_evidence": False,
        "hardware_action_authorized": False,
        "system_type": args.system_type,
        "config": {
            "learning_rate": args.learning_rate,
            "batch_rows": args.batch_rows,
            "l2_weight": args.l2_weight,
        },
        "feature_diagnostics": feature_diagnostics,
        "arms": {},
    }

    for arm in ARMS:
        op_features = feature_cache[arm]
        model, train_summary = _train_arm(
            anchor, op_features, train_row_features, train_batch,
            args.learning_rate, args.batch_rows, args.l2_weight,
            per_row_native_four_cell_nll)
        arm_report = {"train": train_summary, "folds": {}}
        with torch.no_grad():
            for role in ("validation", "calibration"):
                batch = eval_batches[role]
                prediction = model(op_features, eval_row_features[role])
                vector = per_row_native_four_cell_nll(
                    prediction, batch).detach().float().tolist()
                group_ids = batch["group_ids"]
                label_to_code = {}
                group_codes = []
                for value in group_ids:
                    key = str(value)
                    label_to_code.setdefault(key, len(label_to_code))
                    group_codes.append(label_to_code[key])
                comparisons = {
                    "v3_trained": _paired_group_bootstrap(
                        trained_vectors[role], vector, group_codes),
                    "E_mixture": _paired_group_bootstrap(
                        references[(role, "E_mixture")][0], vector, group_codes),
                    "H_blend": _paired_group_bootstrap(
                        references[(role, "H_blend")][0], vector, group_codes),
                }
                arm_report["folds"][role] = {
                    "mean_nll": float(np.mean(vector)),
                    "median_nll": float(np.median(vector)),
                    "paired_gain_vs": {
                        name: {
                            "mean_improvement_nats": value["mean_improvement_nats"],
                            "ci95_low": value["ci95_low"],
                            "ci95_high": value["ci95_high"],
                        }
                        for name, value in comparisons.items()
                    },
                }
        report["arms"][arm] = arm_report
        checkpoint_path = Path(args.output_path).parent / f"research_v4_{args.system_type}_{arm}.pt"
        torch.save({
            "schema": "TrackBResearchV4CheckpointV1",
            "route_id": "user_authorized_research_route_v1",
            "research_only": True,
            "arm": arm,
            "feature_groups": list(ARM_GROUPS[arm]),
            "system_type": args.system_type,
            "model_state_dict": model.state_dict(),
            "anchor_probabilities": {name: anchor[name].detach().cpu() for name in BRANCHES},
            "op_features": op_features.detach().cpu(),
            "row_feature_mean": row_feature_mean.detach().cpu(),
            "row_feature_std": row_feature_std.detach().cpu(),
            "train_summary": train_summary,
            "config": {
                "learning_rate": args.learning_rate,
                "batch_rows": args.batch_rows,
                "l2_weight": args.l2_weight,
                "init_concentration": 200.0,
            },
        }, checkpoint_path)
        arm_report["checkpoint"] = str(checkpoint_path)
        print(json.dumps({
            "arm": arm,
            "train": train_summary,
            "folds": {
                role: {
                    "mean": round(values["mean_nll"], 4),
                    "gain_vs_H": round(
                        values["paired_gain_vs"]["H_blend"]["mean_improvement_nats"], 4),
                    "ci_vs_H": [
                        round(values["paired_gain_vs"]["H_blend"]["ci95_low"], 4),
                        round(values["paired_gain_vs"]["H_blend"]["ci95_high"], 4)],
                }
                for role, values in arm_report["folds"].items()
            },
        }))

    for dataset in folds.values():
        dataset.close()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
