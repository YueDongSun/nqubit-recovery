"""V10 small-N classifier scientific evaluator.

Loads a frozen checkpoint and computes all required metrics on persistent
validation / test / OOD / negative-control ranges.  Never uses train rows.
All output is JSON-safe.  Missing metrics become ``"NOT_RUN"``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Optional sklearn metrics (numpy fallbacks provided)
# ---------------------------------------------------------------------------

try:
    from sklearn.metrics import (
        average_precision_score as _sk_pr_auc,
        f1_score as _sk_f1_score,
        roc_auc_score as _sk_roc_auc,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metric utility functions
# ---------------------------------------------------------------------------


def compute_macro_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> float:
    """Standard macro-F1 with zero-division handling.

    Parameters
    ----------
    predictions : int array of shape ``(N,)``
    targets : int array of shape ``(N,)``
    num_classes : total number of classes.
    """
    predictions = np.asarray(predictions, dtype=np.int64).ravel()
    targets = np.asarray(targets, dtype=np.int64).ravel()
    if predictions.size == 0:
        return float("nan")
    if _SKLEARN_AVAILABLE:
        return float(_sk_f1_score(
            targets, predictions, average="macro", labels=np.arange(num_classes),
            zero_division=0,
        ))
    # numpy fallback
    f1_per_class = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        tp = float(np.sum((predictions == c) & (targets == c)))
        fp = float(np.sum((predictions == c) & (targets != c)))
        fn = float(np.sum((predictions != c) & (targets == c)))
        if tp + fp == 0.0 or tp + fn == 0.0:
            f1_per_class[c] = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_per_class[c] = (2 * precision * recall / (precision + recall)
                               if precision + recall > 0 else 0.0)
    return float(np.nanmean(f1_per_class))


def compute_pr_auc(binary_targets: np.ndarray, scores: np.ndarray) -> float:
    """Non-interpolated average precision, matching the sklearn backend.

    The public metric name is retained. Empty or single-class observations
    remain undefined; both backends require finite scores and binary labels.
    """
    original_targets = np.asarray(binary_targets)
    scores = np.asarray(scores)
    for values in (original_targets, scores):
        if np.iscomplexobj(values) or (
                values.dtype.kind == "O" and any(np.iscomplexobj(value) for value in values.flat)):
            raise ValueError("PR-AUC labels and scores must be real-valued")
    binary_targets = np.asarray(original_targets, dtype=np.float64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if binary_targets.shape != scores.shape:
        raise ValueError("PR-AUC requires equal target and score counts")
    if not np.isfinite(binary_targets).all() or not np.isfinite(scores).all():
        raise ValueError("PR-AUC requires finite targets and scores")
    if not np.isin(original_targets, (0, 1)).all():
        raise ValueError("PR-AUC requires binary targets encoded as zero or one")
    if binary_targets.size == 0 or len(np.unique(binary_targets)) < 2:
        return float("nan")
    if _SKLEARN_AVAILABLE:
        return float(_sk_pr_auc(binary_targets, scores))
    # A threshold includes its entire tied-score group, irrespective of row order.
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    threshold_ends = np.r_[
        np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), scores.size - 1]
    true_positives = np.cumsum(binary_targets[order])[threshold_ends]
    precision = true_positives / (threshold_ends + 1)
    recall = true_positives / true_positives[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def compute_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Expected Calibration Error (top-label, uniform-width bins)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).ravel()
    if probabilities.ndim == 2:
        # Multi-class: use top-label confidence
        top_class = np.argmax(probabilities, axis=-1)
        confidence = probabilities[np.arange(len(targets)), top_class]
        is_correct = (top_class == targets).astype(np.float64)
    else:
        # Binary
        confidence = np.where(targets == 1, probabilities, 1 - probabilities)
        is_correct = (np.round(probabilities) == targets).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count > 0:
            bin_acc = float(is_correct[mask].mean())
            bin_conf = float(confidence[mask].mean())
            ece += count / len(targets) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Brier score for multi-class (sum of squared errors on one-hot)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).ravel()
    if probabilities.ndim == 2:
        one_hot = np.zeros_like(probabilities)
        valid = targets >= 0
        one_hot[valid, targets[valid]] = 1.0
        return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=-1)))
    # Binary
    one_hot = (targets == 1).astype(np.float64)
    return float(np.mean((probabilities - one_hot) ** 2))


def compute_hellinger(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hellinger distance per sample.

    Parameters
    ----------
    p, q : arrays of shape ``(N, C)`` or ``(N,)``.
    Returns
    -------
    Array of shape ``(N,)`` with per-sample Hellinger distances.
    """
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    q = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    if p.ndim == 1:
        p = p[:, None]
        q = q[:, None]
    # normalize each row
    p_sum = p.sum(axis=-1, keepdims=True)
    q_sum = q.sum(axis=-1, keepdims=True)
    p_norm = p / np.maximum(p_sum, 1e-15)
    q_norm = q / np.maximum(q_sum, 1e-15)
    bc = np.sum(np.sqrt(p_norm * q_norm), axis=-1)
    return np.clip(np.sqrt(np.maximum(1.0 - bc, 0.0)), 0.0, 1.0)


def compute_ood_auroc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """AUROC for OOD detection.  *Higher* in_scores = more in-distribution."""
    in_scores = np.asarray(in_scores, dtype=np.float64).ravel()
    out_scores = np.asarray(out_scores, dtype=np.float64).ravel()
    labels = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    scores = np.concatenate([in_scores, out_scores])
    if len(np.unique(labels)) < 2:
        return float("nan")
    if _SKLEARN_AVAILABLE:
        return float(_sk_roc_auc(labels, scores))
    # numpy fallback (Mann-Whitney U)
    n_in, n_out = len(in_scores), len(out_scores)
    concordant = 0.0
    for s in in_scores:
        concordant += np.sum(s > out_scores) + 0.5 * np.sum(s == out_scores)
    return float(concordant / (n_in * n_out))


def compute_reliability_diagram(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, Any]:
    """Per-bin accuracy and confidence for reliability diagrams."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).ravel()
    if probabilities.ndim == 2:
        top_class = np.argmax(probabilities, axis=-1)
        confidence = probabilities[np.arange(len(targets)), top_class]
        is_correct = (top_class == targets).astype(np.float64)
    else:
        confidence = np.where(targets == 1, probabilities, 1 - probabilities)
        is_correct = (np.round(probabilities) == targets).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies: List[float] = []
    bin_confidences: List[float] = []
    bin_counts: List[int] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count > 0:
            bin_accuracies.append(float(is_correct[mask].mean()))
            bin_confidences.append(float(confidence[mask].mean()))
        else:
            bin_accuracies.append(float("nan"))
            bin_confidences.append(float((lo + hi) / 2.0))
    return {
        "bin_edges": [float(v) for v in bin_edges],
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def compute_abstention_risk_curve(
    abstention_probs: np.ndarray,
    is_correct: np.ndarray,
    n_thresholds: int = 20,
) -> Dict[str, Any]:
    """Risk vs abstention-rate curve: sweep threshold on abstention probability."""
    abstention_probs = np.asarray(abstention_probs, dtype=np.float64).ravel()
    is_correct = np.asarray(is_correct, dtype=np.float64).ravel()
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    abstention_rates: List[float] = []
    risks: List[float] = []
    for thr in thresholds:
        accept = abstention_probs < thr
        n_accept = int(accept.sum())
        abstention_rates.append(float(1.0 - n_accept / len(abstention_probs)))
        if n_accept > 0:
            risks.append(float(1.0 - is_correct[accept].mean()))
        else:
            risks.append(float("nan"))
    return {
        "thresholds": [float(v) for v in thresholds],
        "abstention_rates": abstention_rates,
        "risks": risks,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    samples: int = 2000,
    seed: int = 20261001,
) -> Dict[str, float]:
    """Parametric bootstrap CI for an arbitrary statistic.

    Returns ``{"mean": ..., "ci_low": ..., "ci_high": ...}``.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    point = float(statistic(values))
    rng = np.random.Generator(np.random.Philox(seed))
    indices = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap_values = np.array([statistic(values[idx]) for idx in indices])
    return {
        "mean": point,
        "ci_low": float(np.quantile(bootstrap_values, 0.025)),
        "ci_high": float(np.quantile(bootstrap_values, 0.975)),
    }


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClassificationMetrics:
    """Classification head metrics for one cell."""
    macro_f1: float = float("nan")
    identifiable_equivalence_f1: float = float("nan")
    equivalence_f1_status: str = "NOT_RUN"  # Requires target_equivalence_class
    coarse_family_macro_f1: float = float("nan")


@dataclass
class StructureMetrics:
    """ESM / structure head metrics for one cell."""
    pair_entanglement_pr_auc: float = float("nan")
    partition_link_pr_auc: float = float("nan")
    hyperedge_pr_auc: float = float("nan")
    depth_accuracy: float = float("nan")
    intactness_accuracy: float = float("nan")
    stretchability_mae: float = float("nan")


@dataclass
class WPhysicsMetrics:
    """W-subspace physics metrics for one cell."""
    r1_frobenius_error_normalized: float = float("nan")
    population_mae: float = float("nan")
    complex_coherence_error: float = float("nan")
    circular_phase_error: float = float("nan")
    p0_error: float = float("nan")
    p1_error: float = float("nan")
    p_ge2_error: float = float("nan")


@dataclass
class ReliabilityDiagramData:
    bin_edges: List[float] = field(default_factory=list)
    bin_accuracies: List[float] = field(default_factory=list)
    bin_confidences: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)


@dataclass
class MeasurementCalibrationMetrics:
    """Measurement and calibration metrics for one cell."""
    measurement_nll: float = float("nan")
    hellinger_error: float = float("nan")
    denoising_error: float = float("nan")
    ece: float = float("nan")
    brier_score: float = float("nan")
    reliability_diagram: ReliabilityDiagramData = field(
        default_factory=ReliabilityDiagramData)
    abstention_risk: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OODMetrics:
    """OOD detection metrics (requires OOD or negative-control split)."""
    auroc: float = float("nan")
    auprc: float = float("nan")
    fpr_at_95_tpr: float = float("nan")


@dataclass
class CellMetrics:
    """Complete metric set for one cell (e.g. 3Q standard)."""
    cell_id: str = ""
    n_samples: int = 0
    n_validation: int = 0
    n_test: int = 0
    n_ood: int = 0
    n_negative_control: int = 0
    classification: ClassificationMetrics = field(
        default_factory=ClassificationMetrics)
    structure: StructureMetrics = field(
        default_factory=StructureMetrics)
    w_physics: WPhysicsMetrics = field(
        default_factory=WPhysicsMetrics)
    measurement_calibration: MeasurementCalibrationMetrics = field(
        default_factory=MeasurementCalibrationMetrics)
    ood: OODMetrics = field(
        default_factory=OODMetrics)
    negative_controls: Dict[str, Any] = field(default_factory=dict)
    # Test-set metrics (unbiased, never used for checkpoint selection)
    test_classification: Optional[ClassificationMetrics] = None
    test_structure: Optional[StructureMetrics] = None
    test_w_physics: Optional[WPhysicsMetrics] = None


@dataclass
class AggregatedMetrics:
    """Aggregate across cells."""
    macro_average: Dict[str, Any] = field(default_factory=dict)
    weighted_average: Dict[str, Any] = field(default_factory=dict)
    worst_cell: Dict[str, Any] = field(default_factory=dict)
    bootstrap_ci: Dict[str, Any] = field(default_factory=dict)


@dataclass
class V10Classifier1kReport:
    """Comprehensive report for the small-N classifier evaluator."""
    schema: str = "V10Classifier1kReportV2"
    checkpoint_path: str = ""
    checkpoint_model_id: str = ""
    checkpoint_sha: str = ""
    dataset_root: str = ""
    plan_path: str = ""
    storage_profile_path: str = ""
    persistent_eval_manifest_path: str = ""
    formal_binding: Dict[str, Any] = field(default_factory=dict)
    seed: int = 20261001
    bootstrap_samples: int = 2000
    device: str = "cpu"
    cell_metrics: Dict[str, CellMetrics] = field(default_factory=dict)
    aggregated: AggregatedMetrics = field(default_factory=AggregatedMetrics)
    selection_evidence: bool = False
    research_only: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRIMARY_FLOAT_METRICS = [
    "classification/macro_f1",
    "classification/coarse_family_macro_f1",
    "structure/pair_entanglement_pr_auc",
    "structure/partition_link_pr_auc",
    "structure/hyperedge_pr_auc",
    "structure/depth_accuracy",
    "structure/intactness_accuracy",
    "structure/stretchability_mae",
    "w_physics/r1_frobenius_error_normalized",
    "w_physics/population_mae",
    "w_physics/complex_coherence_error",
    "w_physics/circular_phase_error",
    "w_physics/p0_error",
    "w_physics/p1_error",
    "w_physics/p_ge2_error",
    "measurement_calibration/measurement_nll",
    "measurement_calibration/hellinger_error",
    "measurement_calibration/denoising_error",
    "measurement_calibration/ece",
    "measurement_calibration/brier_score",
]


def _metric_direction(key: str) -> str:
    """``'min'`` for error/loss metrics, ``'max'`` for accuracy/AUC/F1."""
    min_keys = {"error", "mae", "nll", "hellinger", "denoising", "brier", "ece",
                "frobenius", "circular_phase", "fpr"}
    parts = key.lower().split("/")
    combined = "/".join(parts)
    for fragment in min_keys:
        if fragment in combined:
            return "min"
    return "max"


def _extract_float_metric(cell: CellMetrics, key: str) -> float:
    """Extract a float metric by dotted path like ``'classification/macro_f1'``."""
    group, _, name = key.partition("/")
    group_obj = getattr(cell, group, None)
    if group_obj is None:
        return float("nan")
    return float(getattr(group_obj, name, float("nan")))


def _dataclass_to_json(obj: Any) -> Any:
    """Recursively convert dataclass / dict / list to JSON-safe values."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _dataclass_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj):
            return "NOT_RUN"
        if np.isinf(obj):
            return "INF" if obj > 0 else "-INF"
        return round(obj, 8)
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val):
            return "NOT_RUN"
        return round(val, 8)
    return obj


def _checkpoint_sha(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


FORMAL_CLASSIFIER_CELLS = (
    "classifier_n3_standard",
    "classifier_n3_noisy",
    "classifier_n4_standard",
    "classifier_n4_noisy",
)


def validate_classifier_evaluation_binding(
    *,
    checkpoint_metadata: Dict[str, Any],
    dataset_root: str,
    plan_path: str,
    storage_profile_path: str,
    persistent_eval_manifest_path: str,
    train_range_index_paths: Dict[str, str],
    global_seed: int,
) -> Dict[str, Any]:
    """Bind a formal classifier evaluation to current plan, data, and checkpoint."""

    from ml_recon.data_v2_4.ephemeral_loader import range_training_identity
    from ml_recon.data_v2_4.hashing import (
        read_canonical_json,
        sha256_file,
        sha256_json,
    )
    from ml_recon.data_v2_4.materialization_ops import (
        load_empirical_storage_profile,
        validate_range_training_plan_binding,
    )
    from ml_recon.data_v2_4.rung_plan import read_rung_plan
    from ml_recon.data_v2_4.range_index import read_range_index

    required_paths = {
        "plan_path": plan_path,
        "storage_profile_path": storage_profile_path,
        "persistent_eval_manifest_path": persistent_eval_manifest_path,
    }
    missing_paths = [name for name, value in required_paths.items() if not value]
    if missing_paths:
        raise ValueError(f"formal classifier evaluation is missing paths: {missing_paths}")

    plan = read_rung_plan(plan_path)
    if (plan.task_pool != "small_n_classifier"
            or {cell.cell_id for cell in plan.cell_allocations}
            != set(FORMAL_CLASSIFIER_CELLS)):
        raise ValueError("formal classifier evaluation requires the registered four-cell plan")
    storage_report, materialization_profile = load_empirical_storage_profile(
        storage_profile_path)
    expected_dependencies = {
        "rung_plan_sha": plan.sha256(),
        "generator_sha": plan.generator_sha,
        "contract_sha": plan.contract_sha,
        "schema_sha": plan.schema_sha,
    }
    if any(storage_report.get(name) != expected
           for name, expected in expected_dependencies.items()):
        raise ValueError("formal storage profile dependencies do not match the rung plan")
    if int(storage_report.get("samples_per_cell", 0)) < 1_024:
        raise ValueError("formal storage profile requires at least 1024 samples per cell")
    profile_cells = storage_report.get("cells")
    if not isinstance(profile_cells, list) or {
            str(cell.get("cell_id")) for cell in profile_cells if isinstance(cell, dict)
            } != set(FORMAL_CLASSIFIER_CELLS):
        raise ValueError("formal storage profile does not contain the registered cells")
    for cell in profile_cells:
        static_sha = str(cell.get("cell_static_sha", ""))
        trials = cell.get("compression_trials")
        if (len(static_sha) != 64
                or any(character not in "0123456789abcdef" for character in static_sha)
                or not isinstance(trials, dict)
                or set(trials) != {"gzip", "lzf", "none"}):
            raise ValueError("formal storage profile is not empirical for every cell")

    manifest = dict(read_canonical_json(persistent_eval_manifest_path))
    if (manifest.get("manifest_kind") != "v24_frozen_persistent_eval_v2"
            or manifest.get("rung_plan_sha") != plan.sha256()
            or manifest.get("storage_profile_sha") != storage_report["profile_sha"]
            or manifest.get("materialization_profile_sha")
            != materialization_profile.sha256()):
        raise ValueError("persistent evaluation manifest identity does not match plan/profile")
    if set(train_range_index_paths) != set(FORMAL_CLASSIFIER_CELLS):
        raise ValueError("formal evaluation requires one train range for every classifier cell")
    manifest_ranges = manifest.get("range_index_paths")
    if not isinstance(manifest_ranges, dict):
        raise ValueError("persistent evaluation manifest range paths are missing")
    required_controls = {
        "target_shuffle",
        "operation_subset_shuffle",
        "mechanism_label_permutation",
        "global_multi_pair_shuffle",
        "environment_permutation",
    }
    if set(manifest.get("negative_control_kinds", ())) != required_controls:
        raise ValueError("persistent evaluation manifest negative controls are incomplete")

    for cell_id in FORMAL_CLASSIFIER_CELLS:
        for key in (f"{cell_id}:validation", f"{cell_id}:test"):
            if key not in manifest_ranges:
                raise ValueError(f"persistent evaluation manifest is missing {key}")
        if (f"{cell_id}:locked_ood" not in manifest_ranges
                and f"{cell_id}_locked_ood:locked_ood" not in manifest_ranges):
            raise ValueError(f"persistent evaluation manifest is missing {cell_id} locked_ood")
        for control in required_controls:
            key = f"{cell_id}_negative_control_{control}:negative_control"
            if key not in manifest_ranges:
                raise ValueError(f"persistent evaluation manifest is missing {key}")
    split_identities = manifest.get("split_identities")
    if not isinstance(split_identities, dict):
        raise ValueError("persistent evaluation manifest split identities are missing")
    manifest_global_seed = manifest.get("global_seed")
    if (isinstance(manifest_global_seed, bool)
            or not isinstance(manifest_global_seed, int)
            or manifest_global_seed != int(global_seed)):
        raise ValueError("persistent evaluation manifest global seed is stale or mismatched")

    def _validate_persistent_range(
        *, manifest_key: str, expected_cell_id: str, split: str,
        expected_rows: int, expected_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate one complete canonical range, including its row identity."""
        relative_path = manifest_ranges.get(manifest_key)
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"persistent evaluation manifest is missing {manifest_key}")
        expected_identity = split_identities.get(manifest_key)
        if not isinstance(expected_identity, str) or len(expected_identity) != 64:
            raise ValueError(f"persistent evaluation manifest identity is missing {manifest_key}")
        range_path = root / relative_path
        try:
            entries = tuple(read_range_index(range_path))
        except Exception as exc:
            raise ValueError(f"{split} range index is unreadable: {range_path}") from exc
        matching = tuple(
            entry for entry in entries
            if entry.cell_id == expected_cell_id and entry.split == split
        )
        if not matching or len(matching) != len(entries):
            raise ValueError(f"{manifest_key} range coverage is incomplete or mismatched")
        if sum(entry.row_count for entry in matching) != int(expected_rows):
            raise ValueError(f"{manifest_key} range coverage count is incomplete")
        expected_start = 0
        for entry in sorted(matching, key=lambda value: value.start_index):
            if entry.start_index != expected_start:
                raise ValueError(f"{manifest_key} range coverage has a gap")
            expected_start = entry.stop_index
        if expected_start != int(expected_rows):
            raise ValueError(f"{manifest_key} range coverage is truncated")
        first = matching[0]
        for entry in matching:
            for field_name, expected_value in expected_fields.items():
                if getattr(entry, field_name, None) != expected_value:
                    raise ValueError(
                        f"{manifest_key} range entry has stale or wrong-source "
                        f"{field_name}")
        actual_identity = sha256_json([
            f"v2_4:{entry.dataset_family}:{entry.cell_id}:{entry.split}:"
            f"N{entry.N}:{index:012d}"
            for entry in sorted(matching, key=lambda value: value.start_index)
            for index in range(entry.start_index, entry.stop_index)
        ])
        if actual_identity != expected_identity:
            raise ValueError(f"{manifest_key} range identity is stale or mismatched")
        return {
            "range_index_sha": sha256_file(range_path),
            "split_identity": actual_identity,
            "row_count": int(expected_rows),
            "cell_id": expected_cell_id,
            "split": split,
            "path": relative_path,
            "dataset_family": first.dataset_family,
            "N": int(first.N),
        }

    root = Path(dataset_root)
    root_resolved = root.resolve()
    for manifest_key, relative_path in manifest_ranges.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"persistent evaluation manifest path is invalid: {manifest_key}")
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"persistent evaluation manifest range path is outside dataset_root: "
                f"{manifest_key}") from exc
    expected_dataset_hashes: Dict[str, str] = {}
    range_index_shas: Dict[str, str] = {
        str(key): sha256_file(root / str(relative_path))
        for key, relative_path in sorted(manifest_ranges.items())
    }
    persistent_eval_ranges: Dict[str, Dict[str, Any]] = {}
    cell_bindings: Dict[str, Any] = {}
    for cell_id in FORMAL_CLASSIFIER_CELLS:
        allocation = next(
            cell for cell in plan.cell_allocations if cell.cell_id == cell_id)
        validation_rows = int(allocation.validation_count)
        locked_ood_rows = max(1, validation_rows // 5)
        common_range_fields = {
            "global_seed": manifest_global_seed,
            "build_id": manifest.get("build_id"),
            "generator_sha": plan.generator_sha,
            "contract_sha": plan.contract_sha,
            "schema_sha": plan.schema_sha,
            "N": int(allocation.N),
            "noise_family": allocation.noise_family,
            "eta": allocation.eta,
            "kappa": allocation.kappa,
            "shot_mode": allocation.shot_mode,
            "shot_budget": allocation.shot_budget,
            "measurement_schedule": allocation.measurement_schedule,
        }
        if not isinstance(common_range_fields["build_id"], str) or not common_range_fields["build_id"]:
            raise ValueError("persistent evaluation manifest build identity is missing")
        validation_key = f"{cell_id}:validation"
        test_key = f"{cell_id}:test"
        locked_ood_key = f"{cell_id}:locked_ood"
        locked_ood_key_alt = f"{cell_id}_locked_ood:locked_ood"
        if validation_key not in manifest_ranges or test_key not in manifest_ranges:
            raise ValueError(f"persistent evaluation manifest is missing {cell_id} validation/test")
        if (locked_ood_key not in manifest_ranges
                and locked_ood_key_alt not in manifest_ranges):
            raise ValueError(f"persistent evaluation manifest is missing {cell_id} locked_ood")
        for control_name in required_controls:
            control_key = f"{cell_id}_negative_control_{control_name}:negative_control"
            if control_key not in manifest_ranges:
                raise ValueError(
                    f"persistent evaluation manifest is missing {cell_id} {control_name}"
                )
        locked_ood_manifest_key = locked_ood_key
        if locked_ood_key_alt in manifest_ranges:
            locked_ood_manifest_key = locked_ood_key_alt
        persistent_eval_ranges[validation_key] = _validate_persistent_range(
            manifest_key=validation_key, expected_cell_id=cell_id,
            split="validation", expected_rows=validation_rows,
            expected_fields={**common_range_fields,
                             "dataset_family": allocation.dataset_family})
        persistent_eval_ranges[test_key] = _validate_persistent_range(
            manifest_key=test_key, expected_cell_id=cell_id,
            split="test", expected_rows=validation_rows,
            expected_fields={**common_range_fields,
                             "dataset_family": allocation.dataset_family})
        persistent_eval_ranges[locked_ood_manifest_key] = _validate_persistent_range(
            manifest_key=locked_ood_manifest_key,
            expected_cell_id=f"{cell_id}_locked_ood", split="locked_ood",
            expected_rows=locked_ood_rows,
            expected_fields={**common_range_fields,
                             "dataset_family": f"{allocation.dataset_family}_locked_ood"})
        train_path = Path(train_range_index_paths[cell_id])
        validation_path = root / str(manifest_ranges[validation_key])
        identity = range_training_identity(
            dataset_root,
            train_range_index_path=train_path,
            validation_range_index_path=validation_path,
            cell_id=cell_id,
        )
        binding = validate_range_training_plan_binding(
            identity=identity,
            plan=plan,
            storage_profile_path=storage_profile_path,
            global_seed=global_seed,
        )
        expected_dataset_hashes[f"train:{cell_id}"] = identity[
            "cell_split_hashes"]["train"]
        expected_dataset_hashes[f"validation:{cell_id}"] = identity[
            "cell_split_hashes"]["validation"]
        range_index_shas[f"train:{cell_id}"] = sha256_file(train_path)
        cell_bindings[cell_id] = binding
        for control_name in required_controls:
            control_key = f"{cell_id}_negative_control_{control_name}:negative_control"
            persistent_eval_ranges[control_key] = _validate_persistent_range(
                manifest_key=control_key,
                expected_cell_id=f"{cell_id}_negative_control_{control_name}",
                split="negative_control", expected_rows=locked_ood_rows,
                expected_fields={**common_range_fields,
                                 "dataset_family": (
                                     f"{allocation.dataset_family}_negative_control_"
                                     f"{control_name}")})

    if checkpoint_metadata.get("dataset_hashes") != expected_dataset_hashes:
        raise ValueError("checkpoint dataset hashes do not match formal evaluation ranges")
    if (checkpoint_metadata.get("dataset_contract_sha") != plan.contract_sha
            or checkpoint_metadata.get("sample_count") != plan.total_train_samples
            or set(str(checkpoint_metadata.get("dataset_cell_id", "")).split(","))
            != set(FORMAL_CLASSIFIER_CELLS)):
        raise ValueError("checkpoint dataset contract, count, or cell identity is stale")
    payload = {
        "status": "PASS_FORMAL_BINDING",
        "formal_binding_passed": True,
        "plan_sha": plan.sha256(),
        "storage_profile_sha": storage_report["profile_sha"],
        "storage_profile_file_sha": sha256_file(storage_profile_path),
        "persistent_eval_manifest_sha": sha256_file(
            persistent_eval_manifest_path),
        "materialization_profile_sha": materialization_profile.sha256(),
        "range_index_shas": range_index_shas,
        "persistent_eval_ranges": persistent_eval_ranges,
        "checkpoint_dataset_hashes_sha": hashlib.sha256(
            json.dumps(
                expected_dataset_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "cell_bindings": cell_bindings,
    }
    return payload


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _restore_model(
    checkpoint: Dict[str, Any],
    device: str,
) -> Tuple[torch.nn.Module, int, str]:
    """Rebuild the model from checkpoint metadata and load frozen weights.

    Returns ``(model, N, profile)``.
    """
    from .classifier_n_full import CLASSIFIER_PROFILES, V10ClassifierNFull
    from .classifier_checkpoint import classifier_from_build_spec

    metadata = checkpoint["metadata"]
    model_id = metadata["model_id"]
    if model_id != "C_N_FULL_V10":
        raise ValueError(f"checkpoint model_id is {model_id}, expected C_N_FULL_V10")
    state_dict = checkpoint["model_state_dict"]
    if "model_build_spec" in checkpoint:
        model = classifier_from_build_spec(checkpoint["model_build_spec"])
        profile = model.profile_name
        expected_count = metadata.get("exact_parameter_count")
        if type(expected_count) is not int or expected_count != model.parameter_count():
            raise ValueError("classifier constructor and checkpoint parameter count disagree")
    else:
        # Historical checkpoints without a constructor retain their legacy route.
        param_count = sum(p.numel() for p in state_dict.values())
        profile = None
        for name in CLASSIFIER_PROFILES:
            tmp = V10ClassifierNFull(profile=name)
            if sum(p.numel() for p in tmp.parameters()) == param_count:
                profile = name
                break
        if profile is None:
            profile = "C-S"
        model = V10ClassifierNFull(profile=profile)
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()
    # N is determined from the fine-family head output dim
    # (14 for 3Q, 24 for 4Q); infer from checkpoint if stored, else default 3
    N = metadata.get("N", 3)
    return model, N, profile


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_cell_splits(
    dataset_root: str,
    range_index_path: str,
    cell_id: str,
) -> Dict[str, Any]:
    """Discover non-train splits available for *cell_id* from a range index.

    Returns ``{split_name: row_count, ...}``.
    """
    from ml_recon.data_v2_4.range_index import read_range_index

    entries = read_range_index(range_index_path)
    cell_entries = [e for e in entries if e.cell_id == cell_id]
    if not cell_entries:
        raise ValueError(
            f"cell {cell_id!r} not found in range index {range_index_path}")
    non_train = {}
    for entry in cell_entries:
        if entry.split == "train":
            continue
        non_train[entry.split] = non_train.get(entry.split, 0) + entry.row_count
    if not non_train:
        raise ValueError(
            f"cell {cell_id!r} has no non-train splits in {range_index_path}")
    return non_train


def _range_split_row_count(
    *,
    range_index_path: str,
    cell_id: str,
    split: str,
) -> int:
    """Return row count for a specific cell/split from a V24 range index."""
    from ml_recon.data_v2_4.range_index import read_range_index

    try:
        entries = list(read_range_index(range_index_path))
    except Exception as exc:
        raise ValueError(
            f"{split} range index path for {cell_id!r} is unreadable: "
            f"{range_index_path}") from exc

    actual_cell_id = cell_id
    if split == "locked_ood":
        actual_cell_id = f"{cell_id}_locked_ood"
    matching = [entry for entry in entries
                if entry.cell_id == actual_cell_id and entry.split == split]
    if not matching:
        raise ValueError(
            f"{cell_id!r} has no {split} split entries in {range_index_path}")
    row_count = int(sum(entry.row_count for entry in matching))
    if row_count <= 0:
        raise ValueError(
            f"{cell_id!r} has non-positive {split} row_count={row_count} in "
            f"{range_index_path}")
    return row_count


def _build_dataset(
    dataset_root: str,
    range_index_path: str,
    cell_id: str,
    split: str,
) -> Any:
    """Build a ``V24EphemeralRangeDataset`` for one cell/split pair."""
    from ml_recon.data_v2_4.ephemeral_loader import V24EphemeralRangeDataset

    # Try to find the exact cell_id for this split (may have suffix like _locked_ood)
    actual_cell_id = cell_id
    if split == "locked_ood":
        actual_cell_id = f"{cell_id}_locked_ood"
    elif split == "negative_control":
        # Find the matching negative control cell_id from the range index
        from ml_recon.data_v2_4.range_index import read_range_index
        entries = list(read_range_index(range_index_path))
        for entry in entries:
            if entry.cell_id.startswith(cell_id) and entry.split == split:
                actual_cell_id = entry.cell_id
                break

    return V24EphemeralRangeDataset(
        dataset_root, range_index_path,
        cell_id=actual_cell_id, split=split,
        include_truth=True, task="classifier",
    )


# ---------------------------------------------------------------------------
# Inference collector
# ---------------------------------------------------------------------------


class _InferenceCollector:
    """Accumulate model predictions and ground-truth targets over a dataset."""

    def __init__(self) -> None:
        self._predictions: Dict[str, List[np.ndarray]] = {}
        self._targets: Dict[str, List[np.ndarray]] = {}
        self._raw_logits: List[np.ndarray] = []
        self._ood_scores: List[np.ndarray] = []

    def accumulate(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> None:
        """Collect one batch's predictions and targets (CPU numpy)."""
        # -- predictions from model output --
        pred_keys_float = [
            "p0", "p1", "p_ge2",
            "R1_re", "R1_im",
            "p_clean",
        ]
        pred_keys_pair = [
            "pair_entanglement_probs",
            "hyperedge_entanglement_probs",
            "partition_link_probs",
        ]
        for key in pred_keys_float + pred_keys_pair:
            if key in output:
                self._predictions.setdefault(key, []).append(
                    output[key].detach().cpu().numpy())
        # depth/intactness from posteriors (argmax)
        for post_key, argmax_key in [
            ("depth_posterior", "depth_pred"),
            ("intactness_posterior", "intactness_pred"),
        ]:
            if post_key in output:
                self._predictions.setdefault(argmax_key, []).append(
                    np.argmax(output[post_key].detach().cpu().numpy(), axis=-1))
        # stretchability: expected value from posterior + support values
        if ("stretchability_posterior" in output
                and "stretchability_values" in output):
            probs = output["stretchability_posterior"].detach().cpu().numpy()
            values = output["stretchability_values"].detach().cpu().numpy()
            expected = (probs * values[None, :]).sum(axis=-1)
            self._predictions.setdefault("stretchability_pred", []).append(expected)
        # fine family
        if "fine_family_logits" in output:
            logits = output["fine_family_logits"].detach().cpu().numpy()
            self._predictions.setdefault("fine_family_pred", []).append(
                np.argmax(logits, axis=-1))
            self._raw_logits.append(logits)
        # OOD
        if "OOD_probability" in output:
            self._ood_scores.append(
                output["OOD_probability"].detach().cpu().numpy())
        # coarse template
        if "coarse_template_posterior" in output:
            post = output["coarse_template_posterior"].detach().cpu().numpy()
            self._predictions.setdefault("coarse_pred", []).append(
                np.argmax(post, axis=-1))
        # abstention
        if "abstention_probability" in output:
            self._predictions.setdefault("abstention_prob", []).append(
                output["abstention_probability"].detach().cpu().numpy())

        # -- targets from batch --
        target_map = {
            "target_p_clean": "target_p_clean",
            "target_R1_re": "target_R1_re",
            "target_R1_im": "target_R1_im",
            "target_pair": "target_pair",
            "target_hyperedge": "target_hyperedge",
            "target_partition": "target_partition",
            "target_depth": "target_depth",
            "target_intactness": "target_intactness",
            "target_stretchability": "target_stretchability",
            "target_coarse": "target_coarse",
            "target_ood": "target_ood",
            "target_fine": "target_fine",
            "target_sector": "target_sector",
            "target_fine_mask": "target_fine_mask",
            "target_fine_identifiable_mask": "target_fine_identifiable_mask",
            "target_w_mask": "target_w_mask",
            "target_structure_mask": "target_structure_mask",
            "target_pair_mask": "target_pair_mask",
            "target_hyperedge_mask": "target_hyperedge_mask",
            "target_partition_mask": "target_partition_mask",
            "target_coarse_mask": "target_coarse_mask",
            # For the denoising metric: the model's p_clean output is a
            # [batch, settings, 4] probability; target_p_clean has the same
            # shape.  We also need setting_mask for masking.
            "setting_mask": "setting_mask",
        }
        for src_key, dst_key in target_map.items():
            if src_key in batch:
                val = batch[src_key]
                if isinstance(val, torch.Tensor):
                    self._targets.setdefault(dst_key, []).append(
                        val.detach().cpu().numpy())
                else:
                    self._targets.setdefault(dst_key, []).append(np.asarray(val))

    @property
    def n_samples(self) -> int:
        for key in ("target_ood", "target_fine", "target_depth"):
            if key in self._targets and self._targets[key]:
                return int(sum(arr.shape[0] for arr in self._targets[key]))
        return 0

    def accumulate_batch(self, other: "_InferenceCollector") -> None:
        """Merge all accumulated data from *other* into this collector."""
        for key, arrays in other._predictions.items():
            self._predictions.setdefault(key, []).extend(arrays)
        for key, arrays in other._targets.items():
            self._targets.setdefault(key, []).extend(arrays)
        self._raw_logits.extend(other._raw_logits)
        self._ood_scores.extend(other._ood_scores)

    def _concat(self, d: Dict[str, List[np.ndarray]], key: str) -> np.ndarray:
        if key not in d or not d[key]:
            return np.array([], dtype=np.float64)
        return np.concatenate(d[key], axis=0)

    def compute_classification(self, num_classes: int) -> ClassificationMetrics:
        pred = self._concat(self._predictions, "fine_family_pred")
        tgt = self._concat(self._targets, "target_fine")
        mask = self._concat(self._targets, "target_fine_mask")
        id_mask = self._concat(self._targets, "target_fine_identifiable_mask")
        coarse_pred = self._concat(self._predictions, "coarse_pred")
        coarse_tgt = self._concat(self._targets, "target_coarse")
        coarse_mask = self._concat(self._targets, "target_coarse_mask")
        result = ClassificationMetrics()
        if pred.size > 0 and tgt.size > 0:
            valid = mask.astype(bool)
            if valid.any():
                result.macro_f1 = compute_macro_f1(
                    pred[valid], tgt[valid], num_classes)
            id_valid = (mask.astype(bool) & id_mask.astype(bool))
            if id_valid.any():
                result.identifiable_equivalence_f1 = compute_macro_f1(
                    pred[id_valid], tgt[id_valid], num_classes)
        if coarse_pred.size > 0 and coarse_tgt.size > 0:
            c_valid = coarse_mask.astype(bool)
            if c_valid.any():
                result.coarse_family_macro_f1 = compute_macro_f1(
                    coarse_pred[c_valid], coarse_tgt[c_valid], 10)
        return result

    def compute_structure(self) -> StructureMetrics:
        result = StructureMetrics()
        # Pair entanglement PR-AUC
        pred = self._concat(self._predictions, "pair_entanglement_probs")
        tgt = self._concat(self._targets, "target_pair")
        mask = self._concat(self._targets, "target_pair_mask")
        if pred.ndim >= 2 and tgt.ndim >= 2 and mask.ndim >= 2:
            valid = mask.astype(bool)
            if valid.any():
                p_flat = pred[valid].ravel()
                t_flat = tgt[valid].ravel()
                result.pair_entanglement_pr_auc = compute_pr_auc(t_flat, p_flat)
        # Partition link PR-AUC
        pred = self._concat(self._predictions, "partition_link_probs")
        tgt = self._concat(self._targets, "target_partition")
        mask = self._concat(self._targets, "target_partition_mask")
        if pred.ndim >= 2 and tgt.ndim >= 2 and mask.ndim >= 2:
            valid = mask.astype(bool)
            if valid.any():
                result.partition_link_pr_auc = compute_pr_auc(
                    tgt[valid].ravel(), pred[valid].ravel())
        # Hyperedge PR-AUC
        pred = self._concat(self._predictions, "hyperedge_entanglement_probs")
        tgt = self._concat(self._targets, "target_hyperedge")
        mask = self._concat(self._targets, "target_hyperedge_mask")
        if pred.ndim >= 1 and tgt.ndim >= 1:
            valid = mask.astype(bool) if mask.size > 0 else np.ones(
                tgt.shape[0], dtype=bool)
            if valid.any():
                result.hyperedge_pr_auc = compute_pr_auc(
                    tgt[valid].ravel(), pred[valid].ravel())
        # Depth accuracy
        pred = self._concat(self._predictions, "depth_pred")
        tgt = self._concat(self._targets, "target_depth")
        mask = self._concat(self._targets, "target_structure_mask")
        if pred.size > 0 and tgt.size > 0:
            valid = mask.astype(bool) if mask.size == pred.size else np.ones(
                pred.shape[0], dtype=bool)
            if valid.any():
                result.depth_accuracy = float(
                    np.mean(pred[valid] == tgt[valid]))
        # Intactness accuracy
        pred = self._concat(self._predictions, "intactness_pred")
        tgt = self._concat(self._targets, "target_intactness")
        if pred.size > 0 and tgt.size > 0:
            valid = mask.astype(bool) if mask.size == pred.size else np.ones(
                pred.shape[0], dtype=bool)
            if valid.any():
                result.intactness_accuracy = float(
                    np.mean(pred[valid] == tgt[valid]))
        # Stretchability MAE
        pred = self._concat(self._predictions, "stretchability_pred")
        tgt = self._concat(self._targets, "target_stretchability")
        if pred.size > 0 and tgt.size > 0:
            valid = mask.astype(bool) if mask.size == pred.size else np.ones(
                pred.shape[0], dtype=bool)
            if valid.any():
                result.stretchability_mae = float(
                    np.mean(np.abs(pred[valid] - tgt[valid].astype(np.float64))))
        return result

    def compute_w_physics(self) -> WPhysicsMetrics:
        result = WPhysicsMetrics()
        r1_re_pred = self._concat(self._predictions, "R1_re")
        r1_im_pred = self._concat(self._predictions, "R1_im")
        r1_re_tgt = self._concat(self._targets, "target_R1_re")
        r1_im_tgt = self._concat(self._targets, "target_R1_im")
        w_mask = self._concat(self._targets, "target_w_mask")
        if r1_re_pred.size > 0 and r1_re_tgt.size > 0:
            valid = w_mask.astype(bool) if w_mask.size == r1_re_pred.shape[0] \
                else np.ones(r1_re_pred.shape[0], dtype=bool)
            if valid.any():
                pred_complex = r1_re_pred[valid] + 1j * r1_im_pred[valid]
                tgt_complex = r1_re_tgt[valid] + 1j * r1_im_tgt[valid]
                diff = pred_complex - tgt_complex
                # Normalized Frobenius error
                frob_diff = np.linalg.norm(
                    diff.reshape(diff.shape[0], -1), axis=-1)
                frob_tgt = np.linalg.norm(
                    tgt_complex.reshape(tgt_complex.shape[0], -1), axis=-1)
                result.r1_frobenius_error_normalized = float(
                    np.mean(frob_diff / np.maximum(frob_tgt, 1e-12)))
                # Population MAE
                diag_pred = np.diagonal(r1_re_pred[valid], axis1=-2, axis2=-1)
                diag_tgt = np.diagonal(r1_re_tgt[valid], axis1=-2, axis2=-1)
                result.population_mae = float(
                    np.mean(np.abs(diag_pred - diag_tgt)))
                # Complex coherence error
                coh_pred = r1_re_pred[valid] + 1j * r1_im_pred[valid]
                coh_tgt = r1_re_tgt[valid] + 1j * r1_im_tgt[valid]
                pop_pred = np.maximum(
                    np.diagonal(r1_re_pred[valid], axis1=-2, axis2=-1).real,
                    1e-12)
                pop_tgt = np.maximum(
                    np.diagonal(r1_re_tgt[valid], axis1=-2, axis2=-1).real,
                    1e-12)
                norm_pred = np.abs(coh_pred) / np.sqrt(
                    np.maximum(
                        _outer_diag(pop_pred, pop_pred),
                        1e-12,
                    ))
                norm_tgt = np.abs(coh_tgt) / np.sqrt(
                    np.maximum(
                        _outer_diag(pop_tgt, pop_tgt),
                        1e-12,
                    ))
                norm_pred = np.clip(norm_pred, 0.0, 1.0)
                norm_tgt = np.clip(norm_tgt, 0.0, 1.0)
                result.complex_coherence_error = float(
                    np.mean(np.abs(norm_pred - norm_tgt)))
                # Circular phase error (min angular distance)
                phase_pred = np.angle(coh_pred)
                phase_tgt = np.angle(coh_tgt)
                angular_diff = np.abs(np.angle(
                    np.exp(1j * (phase_pred - phase_tgt))))
                result.circular_phase_error = float(np.mean(angular_diff))
        # Sector errors (p0, p1, p_ge2)
        sector_tgt = self._concat(self._targets, "target_sector")
        w_mask_s = self._concat(self._targets, "target_w_mask")
        p0_pred = self._concat(self._predictions, "p0")
        p1_pred = self._concat(self._predictions, "p1")
        p_ge2_pred = self._concat(self._predictions, "p_ge2")
        if sector_tgt.ndim == 2 and p0_pred.size > 0:
            valid = w_mask_s.astype(bool) if w_mask_s.size == p0_pred.size \
                else np.ones(p0_pred.size, dtype=bool)
            if valid.any():
                result.p0_error = float(np.mean(
                    np.abs(p0_pred[valid] - sector_tgt[valid, 0])))
                result.p1_error = float(np.mean(
                    np.abs(p1_pred[valid] - sector_tgt[valid, 1])))
                result.p_ge2_error = float(np.mean(
                    np.abs(p_ge2_pred[valid] - sector_tgt[valid, 2])))
        return result

    def compute_measurement_calibration(
        self,
        num_classes: int,
    ) -> MeasurementCalibrationMetrics:
        result = MeasurementCalibrationMetrics()
        # Denoising: p_clean prediction vs target (per-setting)
        p_pred = self._concat(self._predictions, "p_clean")
        p_tgt = self._concat(self._targets, "target_p_clean")
        setting_mask = self._concat(self._targets, "setting_mask")
        if p_pred.ndim == 3 and p_tgt.ndim == 3:
            # p_pred: [batch, settings, 4], p_tgt: [batch, settings, 4]
            if setting_mask.ndim == 2:
                valid_s = setting_mask.astype(bool)
                pred_flat = p_pred[valid_s]
                tgt_flat = p_tgt[valid_s]
            else:
                pred_flat = p_pred.reshape(-1, p_pred.shape[-1])
                tgt_flat = p_tgt.reshape(-1, p_tgt.shape[-1])
            if pred_flat.size > 0:
                result.denoising_error = float(
                    np.mean((pred_flat - tgt_flat) ** 2))
                # Measurement NLL: cross-entropy on clean probability
                tgt_clamped = np.maximum(tgt_flat, 1e-12)
                result.measurement_nll = float(
                    -np.mean(np.sum(tgt_clamped * np.log(
                        np.maximum(pred_flat, 1e-12)), axis=-1)))
                # Hellinger error
                hell = compute_hellinger(pred_flat, tgt_flat)
                result.hellinger_error = float(np.mean(hell))
        # ECE and Brier on fine-family classification
        logits_list = self._raw_logits
        fine_tgt = self._concat(self._targets, "target_fine")
        fine_mask = self._concat(self._targets, "target_fine_mask")
        if logits_list and fine_tgt.size > 0:
            all_logits = np.concatenate(logits_list, axis=0)
            # softmax
            exp_l = np.exp(all_logits - all_logits.max(axis=-1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
            valid = fine_mask.astype(bool) if fine_mask.size == fine_tgt.size \
                else np.ones(fine_tgt.size, dtype=bool)
            if valid.any():
                result.ece = compute_ece(
                    probs[valid], fine_tgt[valid], n_bins=5)
                result.brier_score = compute_brier(
                    probs[valid], fine_tgt[valid])
                result.reliability_diagram = ReliabilityDiagramData(
                    **compute_reliability_diagram(
                        probs[valid], fine_tgt[valid], n_bins=5))
        # Abstention-risk curve
        ab_prob = self._concat(self._predictions, "abstention_prob")
        if ab_prob.size > 0 and fine_tgt.size > 0:
            valid = fine_mask.astype(bool) if fine_mask.size == fine_tgt.size \
                else np.ones(fine_tgt.size, dtype=bool)
            if valid.any():
                # Use fine-family predictions for correctness
                fine_pred = self._concat(
                    self._predictions, "fine_family_pred")
                if fine_pred.size > 0:
                    is_correct = (fine_pred[valid] == fine_tgt[valid]).astype(
                        np.float64)
                    result.abstention_risk = compute_abstention_risk_curve(
                        ab_prob[valid], is_correct)
        return result

    def compute_ood(self) -> OODMetrics:
        """OOD metrics (requires OOD split scores to have been collected).

        This collector instance only holds in-distribution scores.
        OOD metrics are computed externally by comparing two collectors.
        """
        return OODMetrics()


# ---------------------------------------------------------------------------
# Inference collector
# ---------------------------------------------------------------------------


def _outer_diag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product of row-diagonals: ``sqrt(a_i * b_j)`` for each batch.

    Parameters
    ----------
    a, b : arrays of shape ``(B, N)`` (per-qubit populations).
    Returns
    -------
    Array of shape ``(B, N, N)``.
    """
    return np.sqrt(np.maximum(a[:, :, None] * b[:, None, :], 1e-12))


def _evaluate_single_cell(
    model: torch.nn.Module,
    dataset_root: str,
    range_index_path: str,
    cell_id: str,
    N: int,
    device: str,
    seed: int,
    batch_size: int = 64,
    test_range_index_path: str = "",
    ood_range_index_path: str = "",
    negative_control_range_index_paths: Optional[Dict[str, str]] = None,
) -> CellMetrics:
    """Run inference on all non-train splits of one cell and compute metrics."""
    # Load splits from the primary validation range index
    splits = _load_cell_splits(dataset_root, range_index_path, cell_id)
    num_classes = 14 if N == 3 else 24

    # Also load from additional range indexes (test, OOD, negative_control)
    extra_range_paths = {
        "test": test_range_index_path,
        "locked_ood": ood_range_index_path,
    }
    for split_name, rpath in extra_range_paths.items():
        if rpath:
            splits[split_name] = _range_split_row_count(
                range_index_path=rpath,
                cell_id=cell_id,
                split=split_name,
            )

    # Collectors per split — validation and test are kept separate for
    # unbiased reporting; OOD and negative_control are also separate.
    val_collector = _InferenceCollector()
    test_collector = _InferenceCollector()
    ood_collector = _InferenceCollector()
    n_validation = 0
    n_test = 0
    n_ood = 0
    n_negative_control = 0

    dev = torch.device(device)
    # Map each split to its correct range index path
    split_range_paths = {"validation": range_index_path}
    if test_range_index_path:
        split_range_paths["test"] = test_range_index_path
    if ood_range_index_path:
        split_range_paths["locked_ood"] = ood_range_index_path

    for split_name, split_count in sorted(splits.items()):
        rpath = split_range_paths.get(split_name, range_index_path)
        dataset = _build_dataset(
            dataset_root, rpath, cell_id, split_name)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        collector = _InferenceCollector()
        with torch.no_grad():
            for raw_batch in loader:
                batch = {
                    k: v.to(dev) if isinstance(v, torch.Tensor) else v
                    for k, v in raw_batch.items()
                }
                output = model(batch, N=N, fine_identifiable=(N in (3, 4)))
                collector.accumulate(output, batch)
        if split_name == "validation":
            val_collector.accumulate_batch(collector)
            n_validation = collector.n_samples
        elif split_name == "test":
            test_collector.accumulate_batch(collector)
            n_test = collector.n_samples
        elif split_name == "locked_ood":
            ood_collector.accumulate_batch(collector)
            n_ood = collector.n_samples
        else:
            # Unknown non-train split: treat as validation
            val_collector.accumulate_batch(collector)

    negative_control_metrics: Dict[str, Any] = {}
    for control_name, control_path in sorted(
            (negative_control_range_index_paths or {}).items()):
        dataset = _build_dataset(
            dataset_root, control_path, cell_id, "negative_control")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        collector = _InferenceCollector()
        with torch.no_grad():
            for raw_batch in loader:
                batch = {
                    key: value.to(dev) if isinstance(value, torch.Tensor) else value
                    for key, value in raw_batch.items()
                }
                output = model(batch, N=N, fine_identifiable=(N in (3, 4)))
                collector.accumulate(output, batch)
        count = collector.n_samples
        n_negative_control += count
        negative_control_metrics[control_name] = {
            "n_samples": count,
            "classification": _dataclass_to_json(
                collector.compute_classification(num_classes)),
            "structure": _dataclass_to_json(collector.compute_structure()),
            "w_physics": _dataclass_to_json(collector.compute_w_physics()),
            "measurement_calibration": _dataclass_to_json(
                collector.compute_measurement_calibration(num_classes)),
            "gate_status": "DIAGNOSTIC_ONLY",
        }

    total_samples = n_validation + n_test + n_ood + n_negative_control

    # Primary metrics on validation only (checkpoint selection / calibration)
    classification = val_collector.compute_classification(num_classes)
    structure = val_collector.compute_structure()
    w_physics = val_collector.compute_w_physics()
    measurement_cal = val_collector.compute_measurement_calibration(num_classes)

    # Test metrics (unbiased final report) — computed separately
    test_classification = test_collector.compute_classification(num_classes) \
        if n_test > 0 else None
    test_structure = test_collector.compute_structure() \
        if n_test > 0 else None
    test_w_physics = test_collector.compute_w_physics() \
        if n_test > 0 else None

    # OOD: compare validation (in-distribution) against locked_ood only.
    # Negative controls are NOT OOD — they are causal/semantic tests.
    ood_metrics = OODMetrics()
    val_ood_scores = np.concatenate(val_collector._ood_scores) \
        if val_collector._ood_scores else np.array([])
    ood_ood_scores = np.concatenate(ood_collector._ood_scores) \
        if ood_collector._ood_scores else np.array([])
    if val_ood_scores.size > 0 and ood_ood_scores.size > 0:
        labels = np.concatenate([
            np.zeros(len(val_ood_scores)),
            np.ones(len(ood_ood_scores))])
        scores = np.concatenate([val_ood_scores, ood_ood_scores])
        ood_metrics.auroc = compute_ood_auroc(
            1 - val_ood_scores, 1 - ood_ood_scores)
        ood_metrics.auprc = compute_pr_auc(labels, scores)
        sorted_ood = np.sort(ood_ood_scores)
        thr_idx = max(0, int(np.floor(0.05 * len(sorted_ood))))
        thr = sorted_ood[thr_idx]
        ood_metrics.fpr_at_95_tpr = float(np.mean(val_ood_scores >= thr))

    return CellMetrics(
        cell_id=cell_id,
        n_samples=total_samples,
        n_validation=n_validation,
        n_test=n_test,
        n_ood=n_ood,
        n_negative_control=n_negative_control,
        classification=classification,
        structure=structure,
        w_physics=w_physics,
        measurement_calibration=measurement_cal,
        ood=ood_metrics,
        negative_controls=negative_control_metrics,
        test_classification=test_classification,
        test_structure=test_structure,
        test_w_physics=test_w_physics,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_cell_metrics(
    cells: Dict[str, CellMetrics],
    bootstrap_samples: int,
    seed: int,
) -> AggregatedMetrics:
    """Compute macro / weighted / worst-cell / bootstrap CI across cells."""
    if not cells:
        return AggregatedMetrics()
    cell_list = list(cells.values())
    cell_ids = list(cells.keys())
    sample_counts = [c.n_samples for c in cell_list]
    # Helper: collect a metric across cells
    def _collect(path: str) -> Tuple[List[float], List[float]]:
        """Return ``(values, weights)`` skipping NaN."""
        vals = []
        wts = []
        for c, w in zip(cell_list, sample_counts):
            v = _extract_float_metric(c, path)
            if np.isfinite(v):
                vals.append(v)
                wts.append(float(w))
        return vals, wts

    macro_avg: Dict[str, Any] = {}
    weighted_avg: Dict[str, Any] = {}
    worst_cell: Dict[str, Any] = {}
    bootstrap: Dict[str, Any] = {}

    for metric_path in _PRIMARY_FLOAT_METRICS:
        vals, wts = _collect(metric_path)
        if not vals:
            macro_avg[metric_path] = "NOT_RUN"
            weighted_avg[metric_path] = "NOT_RUN"
            worst_cell[metric_path] = "NOT_RUN"
            bootstrap[metric_path] = "NOT_RUN"
            continue
        arr = np.array(vals, dtype=np.float64)
        direction = _metric_direction(metric_path)
        macro_avg[metric_path] = float(np.mean(arr))
        if sum(wts) > 0:
            warr = np.array(wts, dtype=np.float64)
            weighted_avg[metric_path] = float(
                np.average(arr, weights=warr))
        else:
            weighted_avg[metric_path] = float(np.mean(arr))
        if direction == "min":
            worst_idx = int(np.argmax(arr))
        else:
            worst_idx = int(np.argmin(arr))
        worst_cell[metric_path] = {
            "cell_id": cell_ids[worst_idx] if worst_idx < len(cell_ids) else "",
            "value": float(arr[worst_idx]),
        }
        if len(arr) >= 2:
            bootstrap[metric_path] = bootstrap_ci(
                arr, samples=bootstrap_samples, seed=seed)
        else:
            bootstrap[metric_path] = {
                "mean": float(arr[0]),
                "ci_low": float(arr[0]),
                "ci_high": float(arr[0]),
            }

    return AggregatedMetrics(
        macro_average=macro_avg,
        weighted_average=weighted_avg,
        worst_cell=worst_cell,
        bootstrap_ci=bootstrap,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def evaluate_classifier_1k(
    *,
    checkpoint_path: str,
    dataset_root: str,
    plan_path: str = "",
    storage_profile_path: str = "",
    persistent_eval_manifest_path: str = "",
    train_range_index_paths: Optional[Dict[str, str]] = None,
    validation_range_index_paths: Optional[Dict[str, str]] = None,
    device: str = "cpu",
    bootstrap_samples: int = 2000,
    seed: int = 20261001,
    output_dir: str = "",
    require_formal_binding: bool = False,
) -> Dict[str, Any]:
    """Evaluate a frozen small-N classifier on all persistent ranges.

    Parameters
    ----------
    checkpoint_path : Path to a ``checkpoint_latest.pt`` produced by
        :func:`ml_recon.physics_hybrid_v10.train_classifier.run_classifier_training`.
    dataset_root : Root of the V2.4 dataset.
    plan_path : Rung plan path (for provenance binding).
    storage_profile_path : Storage profile path (for provenance binding).
    persistent_eval_manifest_path : Path to the ``v24_frozen_persistent_eval_v2``
        manifest JSON.  When provided, all range indexes (validation, test,
        OOD, negative-control) are resolved from this manifest instead of
        from ``validation_range_index_paths``.  This is the preferred entry
        point for scientific evaluation.
    train_range_index_paths : Per-cell mapping ``{cell_id: path}`` for
        **train** range indexes.  Used only to identify which cells exist;
        train rows are never evaluated.
    validation_range_index_paths : Per-cell mapping ``{cell_id: path}`` for
        **validation** range indexes.  Ignored when ``persistent_eval_manifest_path``
        is provided.
    device : Torch device string.
    bootstrap_samples : Number of bootstrap resamples for CI.
    seed : Master RNG seed.
    output_dir : If set, write ``evaluate_classifier_1k_report.json`` and
        ``evaluate_classifier_1k_summary.json`` here.

    Returns
    -------
    Dict suitable for ``json.dumps``.
    """
    train_range_index_paths = train_range_index_paths or {}
    validation_range_index_paths = validation_range_index_paths or {}
    from .checkpoint_schema import validate_checkpoint_envelope

    # -- Validate inputs --
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if not dataset_root:
        raise ValueError("dataset_root is required")

    # -- Load persistent eval manifest if provided --
    # This resolves all range indexes (validation, test, locked_ood,
    # negative_control) from the manifest instead of requiring them
    # to be passed individually.
    if persistent_eval_manifest_path:
        from ml_recon.data_v2_4.hashing import read_canonical_json
        manifest = dict(read_canonical_json(persistent_eval_manifest_path))
        range_paths = manifest.get("range_index_paths", {})
        for key, path_str in range_paths.items():
            # key format: "full_cell_id:split" where full_cell_id may
            # include suffixes like _locked_ood, _negative_control_*
            if ":" not in key:
                continue
            full_cell_id, split = key.rsplit(":", 1)
            path = Path(dataset_root) / path_str
            if split == "validation":
                validation_range_index_paths[full_cell_id] = str(path)
            elif split == "test":
                # Extract base cell_id (e.g. classifier_n3_standard from classifier_n3_standard)
                validation_range_index_paths[f"{full_cell_id}:test"] = str(path)
            elif split == "locked_ood":
                # Base cell_id strips _locked_ood suffix
                base = full_cell_id.replace("_locked_ood", "")
                validation_range_index_paths[f"{base}:locked_ood"] = str(path)
            elif split == "negative_control":
                # Base cell_id strips _negative_control_* suffix
                for ctrl in ["target_shuffle", "operation_subset_shuffle",
                             "mechanism_label_permutation",
                             "global_multi_pair_shuffle",
                             "environment_permutation"]:
                    if full_cell_id.endswith(f"_negative_control_{ctrl}"):
                        base = full_cell_id[:-(len(f"_negative_control_{ctrl}"))]
                        validation_range_index_paths[
                            f"{base}:negative_control:{ctrl}"] = str(path)
                        break

    if not validation_range_index_paths:
        raise ValueError("validation_range_index_paths is required (at least one cell)")

    # -- Validate requested split paths before checkpoint load --
    # In non-formal mode, optional split paths are allowed to be omitted.
    # When provided explicitly, every path must be readable and provide matching
    # cell/split entries with a positive row count.
    base_val_paths = {
        key: path for key, path in validation_range_index_paths.items()
        if ":" not in key
    }
    if not base_val_paths:
        raise ValueError("validation_range_index_paths is empty")
    for cell_id, path in base_val_paths.items():
        _range_split_row_count(
            range_index_path=path,
            cell_id=cell_id,
            split="validation",
        )
        test_range_index_path = validation_range_index_paths.get(
            f"{cell_id}:test", "")
        if test_range_index_path:
            _range_split_row_count(
                range_index_path=test_range_index_path,
                cell_id=cell_id,
                split="test",
            )
        locked_ood_range_index_path = validation_range_index_paths.get(
            f"{cell_id}:locked_ood", "")
        if locked_ood_range_index_path:
            _range_split_row_count(
                range_index_path=locked_ood_range_index_path,
                cell_id=cell_id,
                split="locked_ood",
            )

    # -- Load checkpoint --
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_envelope(checkpoint["metadata"])
    metadata = checkpoint["metadata"]

    formal_binding = {
        "status": "NOT_REQUESTED",
        "formal_binding_passed": False,
        "reason": "diagnostic evaluation did not request plan/profile/range binding",
    }
    if require_formal_binding:
        formal_binding = validate_classifier_evaluation_binding(
            checkpoint_metadata=metadata,
            dataset_root=dataset_root,
            plan_path=plan_path,
            storage_profile_path=storage_profile_path,
            persistent_eval_manifest_path=persistent_eval_manifest_path,
            train_range_index_paths=train_range_index_paths,
            global_seed=seed,
        )

    # -- Restore model --
    model, default_N, profile = _restore_model(checkpoint, device)

    # -- Discover cells from validation range indexes --
    # Filter out split-qualified keys (e.g. "cell:test") for cell discovery
    base_val_paths = {k: v for k, v in validation_range_index_paths.items()
                      if ":" not in k}
    cell_ids = sorted(base_val_paths.keys())
    if not cell_ids:
        raise ValueError("validation_range_index_paths is empty")
    # Validate train side matches (fail closed) — only compare base cell IDs
    train_cells = set(train_range_index_paths.keys())
    val_cells = set(base_val_paths.keys())
    missing_train = val_cells - train_cells
    if missing_train:
        raise ValueError(
            f"validation cells {sorted(missing_train)} have no matching "
            "train range index")

    # -- Evaluate each cell --
    # Read N from range entries rather than parsing cell_id strings.
    from ml_recon.data_v2_4.range_index import read_range_index
    cell_N: Dict[str, int] = {}
    for cell_id, path in base_val_paths.items():
        try:
            entries = list(read_range_index(path))
            if entries:
                cell_N[cell_id] = int(entries[0].N)
        except Exception:
            pass
    for cell_id in cell_ids:
        if cell_id not in cell_N:
            raise ValueError(
                f"cannot determine N for cell {cell_id}; "
                "range index has no entries or is unreadable")

    cell_metrics: Dict[str, CellMetrics] = {}
    for cell_id in cell_ids:
        N = cell_N[cell_id]
        cell_metrics[cell_id] = _evaluate_single_cell(
            model=model,
            dataset_root=dataset_root,
            range_index_path=validation_range_index_paths[cell_id],
            cell_id=cell_id,
            N=N,
            device=device,
            seed=seed,
            test_range_index_path=validation_range_index_paths.get(
                f"{cell_id}:test", ""),
            ood_range_index_path=validation_range_index_paths.get(
                f"{cell_id}:locked_ood", ""),
            negative_control_range_index_paths={
                key.rsplit(":", 1)[-1]: path
                for key, path in validation_range_index_paths.items()
                if key.startswith(f"{cell_id}:negative_control:")
            },
        )

    # -- Aggregate --
    aggregated = _aggregate_cell_metrics(
        cell_metrics, bootstrap_samples=bootstrap_samples, seed=seed)

    # -- Build report --
    report = V10Classifier1kReport(
        checkpoint_path=checkpoint_path,
        checkpoint_model_id=metadata["model_id"],
        checkpoint_sha=_checkpoint_sha(checkpoint_path),
        dataset_root=dataset_root,
        plan_path=plan_path,
        storage_profile_path=storage_profile_path,
        persistent_eval_manifest_path=persistent_eval_manifest_path,
        formal_binding=formal_binding,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        device=device,
        cell_metrics=cell_metrics,
        aggregated=aggregated,
    )

    # -- Serialize --
    report_dict = _dataclass_to_json(report)

    # -- Write output files --
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with (out_path / "evaluate_classifier_1k_report.json").open(
                "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2, sort_keys=True)
        # Compact summary: omit per-cell details, keep aggregates + metadata
        summary = {
            "schema": report_dict["schema"],
            "checkpoint_path": report_dict["checkpoint_path"],
            "checkpoint_model_id": report_dict["checkpoint_model_id"],
            "seed": report_dict["seed"],
            "cell_ids": cell_ids,
            "cell_sample_counts": {
                cid: report_dict["cell_metrics"][cid]["n_samples"]
                for cid in cell_ids
            },
            "aggregated": report_dict["aggregated"],
            "formal_binding": report_dict["formal_binding"],
            "selection_evidence": False,
            "research_only": True,
        }
        with (out_path / "evaluate_classifier_1k_summary.json").open(
                "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)

    return report_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="V10 small-N classifier scientific evaluator")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint_latest.pt")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--plan", default="")
    parser.add_argument("--storage-profile", default="")
    parser.add_argument("--persistent-eval-manifest", default="",
                        help="Path to v24_frozen_persistent_eval manifest JSON")
    parser.add_argument("--train-range-index", action="append", default=[],
                        help="cell_id=path (repeatable)")
    parser.add_argument("--validation-range-index", action="append", default=[],
                        help="cell_id=path (repeatable)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--require-formal-binding", action="store_true")
    args = parser.parse_args(argv)

    from ml_recon.data_v2_4.multi_cell import parse_cell_keyed_paths

    train_map = parse_cell_keyed_paths(args.train_range_index)
    val_map = parse_cell_keyed_paths(args.validation_range_index)
    # Bare paths (no cell_id= prefix) are not valid here; require cell_id keys.
    if "" in train_map:
        raise ValueError(
            "train-range-index must use cell_id=path format "
            "(e.g. 3q_standard=/path/to/index.json)")
    if "" in val_map:
        raise ValueError(
            "validation-range-index must use cell_id=path format "
            "(e.g. 3q_standard=/path/to/index.json)")

    report = evaluate_classifier_1k(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        plan_path=args.plan,
        storage_profile_path=args.storage_profile,
        persistent_eval_manifest_path=args.persistent_eval_manifest,
        train_range_index_paths=train_map,
        validation_range_index_paths=val_map,
        device=args.device,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        require_formal_binding=args.require_formal_binding,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
