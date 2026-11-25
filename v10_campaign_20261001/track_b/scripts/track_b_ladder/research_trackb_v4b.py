"""Track B research v4b: gated per-(row, operation, branch) blend + row-level concentration.

Campaign v10_campaign_20261001 (user-authorized research route). Successor to
research_trackb_v4: the v4 result showed a LINEAR row-adaptive logit tilt over
the saturated per-operation anchor is the winning ingredient (3Q +0.42 nats vs
the best no-training reference, all CIs positive; 4Q ties). v4b generalizes the
fixed global blend weight into a learned per-(row, op, branch) gate and adds a
row-level concentration head:

  logit(p) = w(r,o,b) * [logit(anchor_o,b) + op_tilt(o,b)]
             + (1 - w(r,o,b)) * logit(row_visible_r,b)
  w = sigmoid(gate_bias_b + Linear_r(row_feats) + Linear_o(op_feats))
  log kappa_r = clamp(log 200 + kappa_row_head(row_feats), log 2, log 20000)

All learned maps are zero-initialized except the gate bias, which starts at
logit(0.65) so the model begins at the E-like anchor-dominated blend. Logits
are clamped to +-8. One pass over the train fold; same folds, likelihood,
references, and paired group-aware bootstrap as v4.

Additionally prints a retrieval-efficiency join diagnostic battery (identity
plus a small family of grid permutations). The v4 retrieval feature join is
unvalidated (correlation ~0), and the ``gated_phys`` arm deliberately includes
the retrieval covariate as a GENERIC per-op feature that rides along for fit
quality: the battery decides only whether that covariate may be INTERPRETED
as retrieval physics — if no permutation correlates with the per-op
signal/coin rates, the join is declared unrecoverable without the
operation-metadata sidecar and the retrieval covariate must be read as an
unvalidated generic feature, never as validated retrieval physics. The
accidental-coincidence proxy is the only validated physics feature in any
arm, so gated_phys's gains cannot be attributed to a validated mechanism
alone.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from probe_trackb_headroom import (
    _fit_mixture, _mix_probabilities, _paired_group_bootstrap,
    _per_op_probabilities, _row_visible_probabilities, _score,
)
from research_trackb_v4 import (
    BRANCHES, _blend_model_anchor, _build_op_feature_groups, _fit_h_blend,
    _load_v3_model, _logit, _row_features, _standardize_fit,
    _standardize_with,
)

V4B_ARMS = {
    "gated": ("projector",),
    "gated_phys": ("projector", "accidental", "retrieval"),
    "gated_rowonly": (),
}
LOGIT_CLAMP = 8.0
GATE_INITIAL_LOGIT = 0.619  # sigmoid -> 0.65 toward the anchor side


class ResearchV4BGated(torch.nn.Module):
    """Per-(row, op, branch) gated blend of anchor and row-visible logits."""

    def __init__(self, anchor_probabilities, op_feature_dim: int,
                 row_feature_dim: int, row_kappa_enabled: bool = True):
        super().__init__()
        anchor = torch.stack(
            [_logit(anchor_probabilities[name]) for name in BRANCHES], dim=-1)
        self.register_buffer("anchor_logits", anchor.clamp(-LOGIT_CLAMP, LOGIT_CLAMP))
        self.row_kappa_enabled = bool(row_kappa_enabled)
        self.op_tilt = torch.nn.Linear(op_feature_dim, 3)
        self.gate_op = torch.nn.Linear(op_feature_dim, 3)
        self.gate_row = torch.nn.Linear(row_feature_dim, 3)
        self.gate_bias = torch.nn.Parameter(
            torch.full((3,), float(GATE_INITIAL_LOGIT)))
        self.kappa_row_head = torch.nn.Linear(row_feature_dim, 1)
        self.log_kappa_global = torch.nn.Parameter(torch.tensor(math.log(200.0)))
        for layer in (self.op_tilt, self.gate_op, self.gate_row, self.kappa_row_head):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, op_features, row_features, row_visible_logits):
        anchor_side = (
            self.anchor_logits.unsqueeze(0)
            + self.op_tilt(op_features).unsqueeze(0)
        ).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
        row_side = row_visible_logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP).unsqueeze(1)
        gate = torch.sigmoid(
            self.gate_bias
            + self.gate_row(row_features).unsqueeze(1)
            + self.gate_op(op_features).unsqueeze(0)
        )
        logits = gate * anchor_side + (1.0 - gate) * row_side
        probabilities = torch.sigmoid(logits).clamp(1.0e-5, 1.0 - 1.0e-5)
        log_kappa = self.log_kappa_global
        if self.row_kappa_enabled:
            log_kappa = log_kappa + self.kappa_row_head(row_features).squeeze(-1)
        else:
            log_kappa = log_kappa.expand(row_features.shape[0])
        log_kappa = log_kappa.clamp(math.log(2.0), math.log(20000.0))
        return {
            "signal_probability": probabilities[..., 0],
            "coin_given_signal_probability": probabilities[..., 1],
            "idler_given_no_signal_probability": probabilities[..., 2],
            "four_cell_concentration": torch.exp(log_kappa),
        }


def _retrieval_permutation_battery(system_type: str, operation_count: int,
                                   efficiency_map_dir: str, per_op):
    """Try identity + simple grid permutations; report correlations."""
    from ml_recon.physics_hybrid_v7.physical_ceiling import (
        load_retrieval_rate_vectors,
    )
    file_name = f"retrieval_efficiency_{3 if system_type == '3qubit' else 4}qubit.txt"
    vectors, _info = load_retrieval_rate_vectors(
        system_type, operation_count,
        efficiency_map_path=str(Path(efficiency_map_dir) / file_name))
    vectors = np.asarray(vectors, dtype=np.float64)
    mean_rate = vectors.mean(axis=-1)
    n = operation_count
    signal = per_op["signal"].numpy()
    coin = per_op["coin"].numpy()

    def correlation(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    permutations = {"identity": np.arange(n)}
    if system_type == "4qubit":
        cycles, per_cycle = 4, n // 4
        grid_ops = per_cycle // 16 * 16  # 64 ops per cycle as 16 rows x 4 blocks
        block = per_cycle
        permutations["cycle_reverse"] = (
            (cycles - 1 - np.arange(n) // block) * block
            + np.arange(n) % block)
        within = np.arange(block)
        perm_within = (within % 4) * 16 + (within // 4)
        permutations["within_cycle_transpose_blocks"] = (
            (np.arange(n) // block) * block + np.tile(perm_within, cycles))
        permutations["global_reverse"] = n - 1 - np.arange(n)
    else:
        permutations["global_reverse"] = n - 1 - np.arange(n)
        permutations["half_swap"] = np.roll(np.arange(n), n // 2)

    report = {}
    for name, order in permutations.items():
        report[name] = {
            "corr_mean_vs_signal": correlation(mean_rate[order], signal),
            "corr_mean_vs_coin": correlation(mean_rate[order], coin),
        }
    return report


def _train_v4b_arm(anchor, op_features, train_row_features, train_row_visible_logits,
                   train_batch, learning_rate, batch_rows, l2_weight, nll_fn,
                   row_kappa_enabled: bool = True):
    torch.manual_seed(20261001)
    model = ResearchV4BGated(
        anchor, op_features.shape[-1], train_row_features.shape[-1],
        row_kappa_enabled=row_kappa_enabled)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.0)
    row_total = train_batch["four_cell_counts"].shape[0]
    trajectory = []
    model.train()
    step = 0
    for start in range(0, row_total, batch_rows):
        stop = min(start + batch_rows, row_total)
        optimizer.zero_grad()
        prediction = model(op_features, train_row_features[start:stop],
                           train_row_visible_logits[start:stop])
        slice_batch = {
            "four_cell_counts": train_batch["four_cell_counts"][start:stop],
            "operation_mask": train_batch["operation_mask"][start:stop],
            "query_mask": train_batch["query_mask"][start:stop],
        }
        nll = nll_fn(prediction, slice_batch).mean()
        l2 = l2_weight * sum(
            layer.weight.square().sum()
            for layer in (model.op_tilt, model.gate_op, model.gate_row,
                          model.kappa_row_head))
        loss = nll + l2
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        trajectory.append(float(loss.detach()))
        step += 1
    model.eval()
    gate_stats = {}
    with torch.no_grad():
        sample_gate = torch.sigmoid(
            model.gate_bias + model.gate_row(train_row_features[:64]).unsqueeze(1)
            + model.gate_op(op_features).unsqueeze(0))
        if model.row_kappa_enabled:
            log_kappa_rows = (
                model.log_kappa_global
                + model.kappa_row_head(train_row_features).squeeze(-1))
        else:
            log_kappa_rows = model.log_kappa_global.expand(
                train_row_features.shape[0])
        gate_stats = {
            "gate_mean": float(sample_gate.mean()),
            "gate_std": float(sample_gate.std()),
            "kappa_min": float(torch.exp(log_kappa_rows).min()),
            "kappa_max": float(torch.exp(log_kappa_rows).max()),
        }
    return model, {
        "steps": step,
        "initial_train_loss": trajectory[0] if trajectory else None,
        "final_train_loss": trajectory[-1] if trajectory else None,
        "gate_stats": gate_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--basis-config-root", required=True)
    parser.add_argument("--system-type", required=True)
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--global-initialization", required=True)
    parser.add_argument("--expected-global-initialization-sha", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trained-label", required=True)
    parser.add_argument("--trained-eval-json", required=True)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--batch-rows", type=int, default=4)
    parser.add_argument("--l2-weight", type=float, default=1.0e-3)
    parser.add_argument("--no-row-kappa", dest="row_kappa_enabled",
                        action="store_false",
                        help="disable the row-level concentration head (global kappa only)")
    parser.add_argument("--arm-suffix", default="",
                        help="suffix for output arm/checkpoint names (e.g. _v4c)")
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
    train_row_features = _standardize_with(raw_row_features, row_feature_mean,
                                           row_feature_std)
    train_visible = _row_visible_probabilities(train_batch)
    train_row_visible_logits = torch.stack(
        [_logit(train_visible[name]) for name in BRANCHES], dim=-1)

    retrieval_battery = _retrieval_permutation_battery(
        args.system_type, int(per_op["signal"].shape[0]),
        args.basis_config_root, per_op)

    feature_cache = {}
    feature_diagnostics = {}
    for arm, groups in V4B_ARMS.items():
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
    eval_row_visible_logits = {}
    v3_probabilities = {}
    with torch.no_grad():
        for role in ("validation", "calibration"):
            dataset = folds[role]
            items = [dataset[index] for index in range(len(dataset))]
            batch = trackb_real_static_evaluation_collate(items, outer_role=role)
            eval_batches[role] = batch
            eval_row_features[role] = _standardize_with(
                _row_features(batch), row_feature_mean, row_feature_std)
            visible = _row_visible_probabilities(batch)
            eval_row_visible_logits[role] = torch.stack(
                [_logit(visible[name]) for name in BRANCHES], dim=-1)
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

    references = {}
    with torch.no_grad():
        for fit_role, score_role in (
            ("calibration", "validation"),
            ("validation", "calibration"),
        ):
            fit_batch = eval_batches[fit_role]
            fit_visible = _row_visible_probabilities(fit_batch)
            e_fit = _fit_mixture(per_op, fit_visible, fit_batch,
                                 per_row_native_four_cell_nll)
            h_fit = _fit_h_blend(v3_probabilities[fit_role], per_op, fit_batch,
                                 per_row_native_four_cell_nll)
            score_batch = eval_batches[score_role]
            rows, ops = score_batch["four_cell_counts"].shape[:2]
            score_visible = _row_visible_probabilities(score_batch)
            e_probs = _mix_probabilities(per_op, score_visible, e_fit["w"], rows, ops)
            references[(score_role, "E_mixture")] = _score(
                e_probs, e_fit["c"], score_batch, per_row_native_four_cell_nll)
            h_probs = _blend_model_anchor(
                v3_probabilities[score_role], per_op, h_fit["w"], rows, ops)
            references[(score_role, "H_blend")] = _score(
                h_probs, h_fit["c"], score_batch, per_row_native_four_cell_nll)

    report = {
        "schema": "TrackBResearchV4BRunV1",
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
            "row_kappa_enabled": args.row_kappa_enabled,
            "gate_initial_logit": GATE_INITIAL_LOGIT,
            "logit_clamp": LOGIT_CLAMP,
        },
        "retrieval_join_battery": retrieval_battery,
        "feature_diagnostics": feature_diagnostics,
        "arms": {},
    }

    # Reload the v4 canonical checkpoints for paired v4b-vs-v4 comparison.
    from research_trackb_v4 import ResearchV4Tilt
    v4_checkpoints = {}
    v4_arm_for = {"gated": "arch", "gated_phys": "full", "gated_rowonly": "anchor_only"}
    v4_vectors = {}
    with torch.no_grad():
        for arm, v4_arm in v4_arm_for.items():
            path = (Path(args.output_path).parent.parent
                    / f"research_v4_{'3' if args.system_type == '3qubit' else '4'}q"
                    / f"research_v4_{args.system_type}_{v4_arm}.pt")
            if not path.is_file():
                continue
            blob = torch.load(path, map_location="cpu", weights_only=False)
            v4_model = ResearchV4Tilt(
                {name: blob["anchor_probabilities"][name] for name in BRANCHES},
                blob["op_features"].shape[-1],
                int(blob["row_feature_mean"].numel()))
            v4_model.load_state_dict(blob["model_state_dict"])
            v4_model.eval()
            v4_checkpoints[arm] = str(path)
            for role in ("validation", "calibration"):
                batch = eval_batches[role]
                prediction = v4_model(blob["op_features"], eval_row_features[role])
                v4_vectors[(arm, role)] = per_row_native_four_cell_nll(
                    prediction, batch).detach().float().tolist()
    report["v4_checkpoints"] = v4_checkpoints

    for arm, op_features in feature_cache.items():
        model, train_summary = _train_v4b_arm(
            anchor, op_features, train_row_features, train_row_visible_logits,
            train_batch, args.learning_rate, args.batch_rows, args.l2_weight,
            per_row_native_four_cell_nll,
            row_kappa_enabled=args.row_kappa_enabled)
        arm_report = {"train": train_summary, "folds": {}}
        with torch.no_grad():
            for role in ("validation", "calibration"):
                batch = eval_batches[role]
                prediction = model(op_features, eval_row_features[role],
                                   eval_row_visible_logits[role])
                vector = per_row_native_four_cell_nll(
                    prediction, batch).detach().float().tolist()
                group_ids = batch["group_ids"]
                label_to_code = {}
                group_codes = []
                for value in group_ids:
                    key = str(value)
                    label_to_code.setdefault(key, len(label_to_code))
                    group_codes.append(label_to_code[key])
                v4_gain = None
                if (arm, role) in v4_vectors:
                    v4_gain = {
                        key: value for key, value in _paired_group_bootstrap(
                            v4_vectors[(arm, role)], vector, group_codes).items()
                        if key in ("mean_improvement_nats", "ci95_low",
                                   "ci95_high")}
                arm_report["folds"][role] = {
                    "mean_nll": float(np.mean(vector)),
                    "median_nll": float(np.median(vector)),
                    "paired_gain_vs": {
                        "v3_trained": {
                            key: value for key, value in _paired_group_bootstrap(
                                trained_vectors[role], vector, group_codes).items()
                            if key in ("mean_improvement_nats", "ci95_low",
                                       "ci95_high")},
                        "E_mixture": {
                            key: value for key, value in _paired_group_bootstrap(
                                references[(role, "E_mixture")], vector,
                                group_codes).items()
                            if key in ("mean_improvement_nats", "ci95_low",
                                       "ci95_high")},
                        "H_blend": {
                            key: value for key, value in _paired_group_bootstrap(
                                references[(role, "H_blend")], vector,
                                group_codes).items()
                            if key in ("mean_improvement_nats", "ci95_low",
                                       "ci95_high")},
                        "v4_canonical": v4_gain,
                    },
                }
        report["arms"][arm + args.arm_suffix] = arm_report
        checkpoint_path = (Path(args.output_path).parent
                           / f"research_v4b_{args.system_type}_{arm}{args.arm_suffix}.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "TrackBResearchV4BCheckpointV1",
            "route_id": "user_authorized_research_route_v1",
            "research_only": True,
            "arm": arm,
            "feature_groups": list(V4B_ARMS[arm]),
            "system_type": args.system_type,
            "model_state_dict": model.state_dict(),
            "anchor_probabilities": {name: anchor[name].detach().cpu()
                                     for name in BRANCHES},
            "op_features": op_features.detach().cpu(),
            "row_feature_mean": row_feature_mean.detach().cpu(),
            "row_feature_std": row_feature_std.detach().cpu(),
            "train_summary": train_summary,
        }, checkpoint_path)
        arm_report["checkpoint"] = str(checkpoint_path)
        print(json.dumps({
            "arm": arm,
            "train": {"final": round(train_summary["final_train_loss"], 4),
                      "initial": round(train_summary["initial_train_loss"], 4),
                      "gate": train_summary["gate_stats"]},
            "folds": {
                role: {
                    "mean": round(values["mean_nll"], 4),
                    "vsE": round(values["paired_gain_vs"]["E_mixture"]["mean_improvement_nats"], 4),
                    "ciE": [round(values["paired_gain_vs"]["E_mixture"]["ci95_low"], 4),
                            round(values["paired_gain_vs"]["E_mixture"]["ci95_high"], 4)],
                }
                for role, values in arm_report["folds"].items()
            },
        }))
    print("retrieval_join_battery:", json.dumps(retrieval_battery, indent=1))

    for dataset in folds.values():
        dataset.close()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
