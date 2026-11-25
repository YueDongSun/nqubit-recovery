"""Target-free OOD composite scores on a validation-stream proxy (CPU).

Campaign v10_campaign_20261001 exploration phase, adversarially-verified
"CPU-first OOD rescue" proposal. The registered LockedOODIntervention is a
state/apparatus/detector channel applied at RENDER time (not a latent edit):
corrupted rows are sample-level renders of the same key with
``locked_ood=True``.

Dev discipline: the proxy uses ONLY validation-stream keys in index block
[proxy_start, proxy_start + proxy_count) per cell, rendered locally. The
persistent-eval store's locked_ood rows pair with validation [0, 10) and are
NEVER read here. Every number produced by this module is PROXY-ONLY decision
evidence; real locked_ood recovery numbers require a registered new
confirmation design (the first confirmation pass is spent) and are labeled
second-inspection evidence wherever the composite is applied.

Research-only: research_only=true, scientific_gate_evaluated=false,
promotion_ready=false, selection_evidence=false,
hardware_action_authorized=false.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .research_evaluate_classifier import AUTHORITY_FLAGS, load_evaluable_model

PROXY_SPLIT = "validation"
SCORE_FEATURE_NAMES = (
    "surprise",
    "energy_deficit",
    "ood_probability",
    "model_uncertainty",
)


@dataclass(frozen=True)
class OODProxyRowSpec:
    """One render job: a stream key plus the locked-OOD render flag."""

    cell_id: str
    cell_n: int
    generator_cell: Dict[str, Any]
    split: str
    index: int
    locked_ood: bool
    global_seed: int


def build_proxy_row_specs(*, train_rows: int = 1000, validation_rows: int = 200,
                          proxy_start: int = 10, proxy_count: int = 40,
                          split: str = PROXY_SPLIT,
                          global_seed: int = 20261001) -> List[OODProxyRowSpec]:
    """Validation-stream proxy jobs: [proxy_start, proxy_start+proxy_count) per cell.

    The default block [10, 50) is disjoint from the persistent-eval store's
    locked_ood pairing, which uses validation [0, 10) only.
    """
    from ml_recon.data_v2_4.rung_plan import plan_registered_rung

    plan = plan_registered_rung("small_n_classifier", train_rows, validation_rows)
    specs: List[OODProxyRowSpec] = []
    for cell in plan.cell_allocations:
        available = (cell.validation_count if split == "validation"
                     else cell.train_count)
        if proxy_start + proxy_count > available:
            raise ValueError(
                f"proxy block [{proxy_start}, {proxy_start + proxy_count}) exceeds "
                f"{split} count {available} for cell {cell.cell_id}")
        for index in range(proxy_start, proxy_start + proxy_count):
            for locked_ood in (False, True):
                specs.append(OODProxyRowSpec(
                    cell_id=cell.cell_id, cell_n=cell.N,
                    generator_cell=cell.to_generator_cell(), split=split,
                    index=index, locked_ood=locked_ood,
                    global_seed=global_seed))
    return specs


class OODProxyRenderDataset(Dataset):
    """Renders each row spec clean or with the registered OOD intervention."""

    def __init__(self, specs: Sequence[OODProxyRowSpec]):
        self.specs = tuple(specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        from ml_recon.data_v2_4.sample_keys import make_sample_key
        from ml_recon.data_v2_4.version import GENERATOR_SHA
        from ml_recon.data_v2_4.views import classifier_input_for_family
        from ml_recon.data_v2_4.synthetic import generator

        spec = self.specs[int(index)]
        key = make_sample_key(
            global_seed=spec.global_seed, split=spec.split,
            dataset_family="small_n_family", N=spec.cell_n, index=spec.index,
            generator_sha=GENERATOR_SHA, cell_id=spec.cell_id)
        rendered = generator._render_classifier(
            key, cell=spec.generator_cell, locked_ood=spec.locked_ood)
        observation = generator._stack([rendered["observation"]], [key.sample_id])
        fields = classifier_input_for_family("small_n_family")
        missing = fields - observation.keys()
        if missing:
            raise ValueError(f"renderer omitted fields: {sorted(missing)}")
        from ml_recon.data_v2_4.streaming import _row_tensor

        row = {name: _row_tensor(observation[name][0]) for name in fields}
        row["proxy_cell_id"] = spec.cell_id
        row["proxy_cell_n"] = spec.cell_n
        row["proxy_target_ood"] = torch.tensor(float(spec.locked_ood))
        return row


def _outcome_bit_weights(n_qubits: int) -> torch.Tensor:
    weights = torch.zeros(2 ** n_qubits)
    for outcome in range(2 ** n_qubits):
        weights[outcome] = bin(outcome).count("1")
    return weights


def observed_sector_probabilities(batch: Dict[str, torch.Tensor],
                                  n_qubits: int) -> torch.Tensor:
    """Observed (p0, p1, p_ge2) from the global outcome histogram (input view).

    The histogram is per measurement setting (batch, settings, 2**N); masked
    setting counts pool into one observed population before the sector split.
    """
    histogram = batch["global_outcome_histogram"].float()
    if histogram.ndim == 3:
        setting_mask = batch.get("full_outcome_mask")
        if setting_mask is not None:
            histogram = (histogram
                         * setting_mask.float()).sum(dim=1)
        else:
            histogram = histogram.sum(dim=1)
    totals = histogram.sum(dim=-1, keepdim=True).clamp_min(1.0)
    bit_weights = _outcome_bit_weights(n_qubits).to(histogram.device)
    p0 = histogram[:, 0] / totals[:, 0]
    p1 = ((histogram * (bit_weights == 1).float()).sum(-1)) / totals[:, 0]
    p_ge2 = ((histogram * (bit_weights >= 2).float()).sum(-1)) / totals[:, 0]
    return torch.stack((p0, p1, p_ge2), dim=-1)


def _classifier_input_field_names() -> set:
    from ml_recon.data_v2_4.views import classifier_input_for_family

    return set(classifier_input_for_family("small_n_family"))


def _identity_collate(row_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep per-row dicts; stacking happens per-cell in collect_ood_features."""
    return row_list


def collect_ood_features(model, dataset: Dataset, device: str,
                         batch_size: int = 16, num_workers: int = 1,
                         include_latent: bool = False) -> Dict[str, np.ndarray]:
    """One frozen forward per row; returns per-row target-free OOD scores.

    Scores: observed-vs-predicted sector surprise, 1 - max-softmax energy of
    the fine-family head, the OOD head probability, and the uncertainty head.
    Temperature is deliberately excluded: monotone transforms cannot change
    AUROC. With ``include_latent`` the pooled ``semantic_latent`` rows are
    returned too (for frozen-trunk OOD-head refits).
    """
    # Identity collate: rows from different cells (N=3 vs N=4) have different
    # panel sizes; stacking happens per-cell inside the loop instead.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=min(max(int(num_workers), 0), 2),
                        collate_fn=_identity_collate)
    dev = torch.device(device)
    rows: Dict[str, List[Any]] = {name: [] for name in
                                  ("cell_id", "target",) + SCORE_FEATURE_NAMES}
    if include_latent:
        rows["semantic_latent"] = []
    model.eval()
    with torch.no_grad():
        for row_batch in loader:
            cell_ids = [row["proxy_cell_id"] for row in row_batch]
            for group in _group_indices_by_cell(cell_ids):
                cell_id = cell_ids[group[0]]
                n_qubits = int(row_batch[group[0]]["proxy_cell_n"])
                input_fields = _classifier_input_field_names()
                sub_batch = {
                    name: torch.stack(
                        [row_batch[i][name] for i in group]).to(dev)
                    for name in input_fields if name in row_batch[group[0]]}
                output = model(sub_batch, N=n_qubits, fine_identifiable=True)
                # The W head already mode-pools internally; p0/p1/p_ge2 are
                # per-row sector probabilities (no extra masking needed).
                predicted_sector = torch.stack(
                    (output["p0"], output["p1"], output["p_ge2"]), dim=-1)
                observed = observed_sector_probabilities(sub_batch, n_qubits)
                surprise = -(observed
                             * torch.log(predicted_sector.clamp_min(1e-6))).sum(-1)
                fine_probs = torch.softmax(
                    output["fine_family_logits"].float(), dim=-1)
                energy_deficit = 1.0 - fine_probs.max(dim=-1).values
                rows["cell_id"].extend([cell_id] * len(group))
                rows["target"].extend(
                    [float(row_batch[i]["proxy_target_ood"]) for i in group])
                rows["surprise"].extend(surprise.cpu().tolist())
                rows["energy_deficit"].extend(energy_deficit.cpu().tolist())
                rows["ood_probability"].extend(
                    output["OOD_probability"].float().cpu().tolist())
                rows["model_uncertainty"].extend(
                    output["model_uncertainty"].float().cpu().tolist())
                if include_latent:
                    rows["semantic_latent"].append(
                        output["semantic_latent"].float().cpu().numpy())
    result = {name: np.asarray(values) for name, values in rows.items()
              if name != "semantic_latent"}
    if include_latent:
        result["semantic_latent"] = np.concatenate(rows["semantic_latent"], axis=0)
    return result


def _group_indices_by_cell(cell_ids: Sequence[str]) -> List[List[int]]:
    groups: Dict[str, List[int]] = {}
    for index, cell_id in enumerate(cell_ids):
        groups.setdefault(cell_id, []).append(index)
    return list(groups.values())


def rank_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Deterministic rank-based (Mann-Whitney) AUROC; higher score = positive."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum = ranks[labels > 0.5].sum()
    n_pos, n_neg = positives.size, negatives.size
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def zscore_per_cell(features: Dict[str, np.ndarray],
                    cell_ids: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Within-cell z-scores; per-cell mean/std recorded for later application."""
    zscored: Dict[str, np.ndarray] = {}
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in SCORE_FEATURE_NAMES:
        values = features[name].astype(np.float64)
        scaled = np.empty_like(values)
        stats[name] = {}
        for cell_id in np.unique(cell_ids):
            mask = cell_ids == cell_id
            mean, std = float(values[mask].mean()), float(values[mask].std())
            std_safe = std if std > 1e-12 else 1.0
            scaled[mask] = (values[mask] - mean) / std_safe
            stats[name][str(cell_id)] = {"mean": mean, "std": std_safe}
        zscored[name] = scaled
    return zscored, stats


def fit_logistic_composite(zscored: Dict[str, np.ndarray], labels: np.ndarray,
                           *, steps: int = 600, learning_rate: float = 0.05,
                           folds: int = 5, seed: int = 20261001) -> Dict[str, Any]:
    """Class-balanced logistic combination of the four z-scored features.

    Full-batch deterministic Adam on a fixed seed; k-fold stability of the
    combined AUROC and each single feature is reported alongside the fit.
    """
    design = np.stack([zscored[name] for name in SCORE_FEATURE_NAMES], axis=1)
    labels = labels.astype(np.float64)
    positive_weight = 0.5 / max(labels.mean(), 1e-9)
    negative_weight = 0.5 / max(1.0 - labels.mean(), 1e-9)
    sample_weights = np.where(labels > 0.5, positive_weight, negative_weight)

    def fit(train_x: np.ndarray, train_y: np.ndarray,
            train_w: np.ndarray) -> np.ndarray:
        generator = torch.Generator().manual_seed(seed)
        weights = torch.zeros(train_x.shape[1] + 1, dtype=torch.float64,
                              requires_grad=True)
        x = torch.from_numpy(train_x)
        y = torch.from_numpy(train_y)
        w = torch.from_numpy(train_w)
        optimizer = torch.optim.Adam([weights], lr=learning_rate)
        for _ in range(int(steps)):
            optimizer.zero_grad()
            logits = x @ weights[:-1] + weights[-1]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, weight=w)
            loss.backward()
            optimizer.step()
        return weights.detach().numpy().copy()

    def apply(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
        return x @ weights[:-1] + weights[-1]

    generator = np.random.default_rng(seed)
    fold_ids = np.arange(labels.size)
    generator.shuffle(fold_ids)
    fold_assignments = np.array_split(fold_ids, max(int(folds), 2))
    fold_aurocs, fold_weights = [], []
    for held_out in fold_assignments:
        train_mask = np.ones(labels.size, dtype=bool)
        train_mask[held_out] = False
        weights = fit(design[train_mask], labels[train_mask],
                      sample_weights[train_mask])
        fold_aurocs.append(rank_auroc(apply(weights, design[held_out]),
                                      labels[held_out]))
        fold_weights.append(weights)
    full_weights = fit(design, labels, sample_weights)
    combined = apply(full_weights, design)
    report = {
        "feature_names": list(SCORE_FEATURE_NAMES),
        "logistic_weights": {
            name: float(weight) for name, weight
            in zip(SCORE_FEATURE_NAMES, full_weights[:-1])},
        "logistic_bias": float(full_weights[-1]),
        "combined_proxy_auroc": rank_auroc(combined, labels),
        "single_feature_proxy_auroc": {
            name: rank_auroc(zscored[name], labels)
            for name in SCORE_FEATURE_NAMES},
        "fold_combined_auroc": [float(value) for value in fold_aurocs],
        "fold_combined_auroc_std": float(np.std(fold_aurocs)),
        "fold_weight_std": {
            name: float(np.std([weights[i] for weights in fold_weights]))
            for i, name in enumerate(SCORE_FEATURE_NAMES)},
        "rows": int(labels.size),
        "positive_rows": int((labels > 0.5).sum()),
    }
    return report


def evaluate_checkpoint_on_proxy(*, checkpoint_path: str, device: str = "cpu",
                                 train_rows: int = 1000, validation_rows: int = 200,
                                 proxy_start: int = 10, proxy_count: int = 40,
                                 batch_size: int = 16, num_workers: int = 1,
                                 global_seed: int = 20261001) -> Dict[str, Any]:
    model, checkpoint_info = load_evaluable_model(checkpoint_path, device)
    specs = build_proxy_row_specs(
        train_rows=train_rows, validation_rows=validation_rows,
        proxy_start=proxy_start, proxy_count=proxy_count,
        global_seed=global_seed)
    dataset = OODProxyRenderDataset(specs)
    features = collect_ood_features(model, dataset, device, batch_size,
                                    num_workers)
    zscored, z_stats = zscore_per_cell(features, features["cell_id"])
    composite = fit_logistic_composite(zscored, features["target"])
    per_cell = {}
    for cell_id in np.unique(features["cell_id"]):
        mask = features["cell_id"] == cell_id
        per_cell[str(cell_id)] = {
            "combined_proxy_auroc": rank_auroc(
                np.stack([zscored[name][mask]
                          for name in SCORE_FEATURE_NAMES], axis=1)
                @ np.array([composite["logistic_weights"][name]
                            for name in SCORE_FEATURE_NAMES])
                + composite["logistic_bias"],
                features["target"][mask]),
            "single_feature_proxy_auroc": {
                name: rank_auroc(zscored[name][mask], features["target"][mask])
                for name in SCORE_FEATURE_NAMES},
            "rows": int(mask.sum()),
        }
    return {
        "schema": "V10ResearchOODScoreProxyReportV1",
        "route_id": "user_authorized_research_route_v1",
        **AUTHORITY_FLAGS,
        "evidence_class": "proxy_only_validation_stream",
        "checkpoint": checkpoint_info,
        "proxy": {"split": PROXY_SPLIT, "start_index": proxy_start,
                  "count_per_cell": proxy_count,
                  "rows_total": len(specs),
                  "train_rows_rung": train_rows,
                  "validation_rows_rung": validation_rows,
                  "disjoint_from_eval_locked_ood_block": "[0, 10)"},
        "composite": composite,
        "per_cell": per_cell,
        "per_cell_zscore_stats": z_stats,
    }


def apply_composite_to_scores(features: Dict[str, np.ndarray],
                              zscore_stats: Dict[str, Any],
                              composite: Dict[str, Any]) -> np.ndarray:
    """Apply a fitted proxy composite to freshly collected eval features.

    The z-score statistics and logistic weights come from the dev proxy; this
    function never refits on evaluation rows.
    """
    combined = np.zeros(features["surprise"].shape[0], dtype=np.float64)
    for name in SCORE_FEATURE_NAMES:
        values = features[name].astype(np.float64)
        scaled = np.empty_like(values)
        for cell_id in np.unique(features["cell_id"]):
            mask = features["cell_id"] == cell_id
            cell_stats = zscore_stats[name].get(str(cell_id))
            if cell_stats is None:
                raise ValueError(
                    f"composite statistics missing for cell {cell_id}")
            scaled[mask] = (values[mask] - cell_stats["mean"]) / cell_stats["std"]
        combined += scaled * composite["logistic_weights"][name]
    return combined + composite["logistic_bias"]


def _concat_features(parts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    merged: Dict[str, np.ndarray] = {}
    for name in parts[0]:
        if name == "semantic_latent":
            merged[name] = np.concatenate([part[name] for part in parts], axis=0)
        else:
            merged[name] = np.concatenate(
                [np.asarray(part[name]) for part in parts], axis=0)
    return merged


class _EvalRowAdaptor(Dataset):
    """Adds the proxy bookkeeping fields to persistent-eval store rows."""

    def __init__(self, inner: Any, cell_id: str, cell_n: int):
        self.inner = inner
        self.cell_id = cell_id
        self.cell_n = cell_n

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = dict(self.inner[int(index)])
        row["proxy_cell_id"] = self.cell_id
        row["proxy_cell_n"] = self.cell_n
        target_ood = row.get("target_ood")
        row["proxy_target_ood"] = (
            target_ood.float() if isinstance(target_ood, torch.Tensor)
            else torch.tensor(float(target_ood or 0.0)))
        return row


def composite_score_report_on_eval_store(
        *, model, dataset_root: str, index_paths: Dict[str, str],
        weights_payload: Dict[str, Any], device: str = "cpu",
        batch_size: int = 16) -> Dict[str, Any]:
    """Apply proxy-fitted composite weights to eval-store validation/locked_ood rows.

    SECOND-INSPECTION EVIDENCE: the underlying rows are the reserved
    persistent-eval validation/locked_ood cells. The composite is fitted on
    the dev proxy only and never refit here; this report is the labeled
    second inspection of the reserved cells and requires a registered
    confirmation design to carry decision weight.
    """
    from .evaluate_classifier_1k import _build_dataset
    from .research_evaluate_classifier import CELL_N, _resolve_paths

    composite = weights_payload["composite"]
    z_stats = weights_payload["per_cell_zscore_stats"]
    per_cell: Dict[str, Any] = {}
    for cell_id, cell_n in CELL_N.items():
        validation_path, _, ood_path, _ = _resolve_paths(
            index_paths, dataset_root, cell_id)
        if not ood_path:
            continue
        validation_rows = _EvalRowAdaptor(
            _build_dataset(dataset_root, validation_path, cell_id, "validation"),
            cell_id, cell_n)
        locked_rows = _EvalRowAdaptor(
            _build_dataset(dataset_root, ood_path, cell_id, "locked_ood"),
            cell_id, cell_n)
        features = _concat_features([
            collect_ood_features(model, validation_rows, device, batch_size),
            collect_ood_features(model, locked_rows, device, batch_size)])
        combined = apply_composite_to_scores(features, z_stats, composite)
        per_cell[cell_id] = {
            "rows": int(features["target"].size),
            "positive_rows": int((features["target"] > 0.5).sum()),
            "composite_auroc": rank_auroc(combined, features["target"]),
            "single_feature_auroc": {
                name: rank_auroc(features[name], features["target"])
                for name in SCORE_FEATURE_NAMES},
        }
    return {
        "schema": "V10ResearchOODCompositeEvalReportV1",
        "inspection_class": "second_inspection",
        "warning": ("reserved persistent-eval cells inspected under a "
                    "proxy-fitted composite; carries decision weight only "
                    "inside a registered new confirmation design"),
        "weights_provenance": {
            "schema": weights_payload.get("schema"),
            "checkpoint": weights_payload.get("checkpoint"),
            "evidence_class": weights_payload.get("evidence_class")},
        "per_cell": per_cell,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="checkpoint path; repeat for multiple members")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-rows", type=int, default=1000)
    parser.add_argument("--validation-rows", type=int, default=200)
    parser.add_argument("--proxy-start", type=int, default=10)
    parser.add_argument("--proxy-count", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=20261001)
    parser.add_argument("--export-weights", default=None,
                        help="write the fitted composite + z-score stats JSON")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)

    reports = []
    for checkpoint_path in arguments.checkpoint:
        report = evaluate_checkpoint_on_proxy(
            checkpoint_path=checkpoint_path, device=arguments.device,
            train_rows=arguments.train_rows,
            validation_rows=arguments.validation_rows,
            proxy_start=arguments.proxy_start,
            proxy_count=arguments.proxy_count,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            global_seed=arguments.global_seed)
        reports.append(report)
        print(json.dumps({
            "checkpoint": Path(checkpoint_path).name,
            "combined_proxy_auroc": report["composite"]["combined_proxy_auroc"],
            "fold_std": report["composite"]["fold_combined_auroc_std"],
            "single_features": report["composite"]["single_feature_proxy_auroc"],
        }, indent=2))
    if arguments.export_weights:
        export = reports[0]
        payload = {
            "schema": "V10ResearchOODCompositeWeightsV1",
            **AUTHORITY_FLAGS,
            "evidence_class": "proxy_only_validation_stream",
            "feature_names": list(SCORE_FEATURE_NAMES),
            "composite": export["composite"],
            "per_cell_zscore_stats": export["per_cell_zscore_stats"],
            "checkpoint": export["checkpoint"],
        }
        export_path = Path(arguments.export_weights)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reports, indent=2, default=str),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
