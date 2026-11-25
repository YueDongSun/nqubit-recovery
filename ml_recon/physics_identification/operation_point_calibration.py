"""Learn interpretable target-conditioned operation-point calibration laws."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml_recon.measured_data.contracts import MeasuredSampleRecordV1
from ml_recon.measured_data.datasets import MeasuredManifestDataset, load_measured_records


CALIBRATION_SCHEMA = "operation_point_calibration_v1"
LATENT_CALIBRATION_SCHEMA = "latent_operation_calibration_v1"
DEFAULT_CONFIG_PATH = Path("ml_recon/configs/v6_operation_point_calibration.json")
MODEL_INPUT_AUDIT = {
    "snapshot_path_numeric_input": False,
    "state_name_numeric_input": False,
    "target_hash_numeric_input": False,
    "table_hash_numeric_input": False,
    "current_matrix_directory_read": False,
    "hidden_operation_counts_used_for_inference": False,
}


@dataclass(frozen=True)
class OperationPointDatasetV1:
    system_type: str
    records: Tuple[MeasuredSampleRecordV1, ...]
    coin_counts: np.ndarray
    signal_counts: np.ndarray
    idler_counts: np.ndarray
    target_features: np.ndarray
    target_feature_names: Tuple[str, ...]
    split_names: np.ndarray


@dataclass(frozen=True)
class OperationPointCalibrationModelV1:
    system_type: str
    feature_names: Tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    baseline_logit: np.ndarray
    target_coefficients: np.ndarray
    signal_idler_coefficients: np.ndarray
    ridge_alpha: float
    smoothing: float
    training_sample_count: int
    split_hash: str
    research_only: bool = True
    promotion_ready: bool = False

    def predict_probability(
        self,
        target_features: Any,
        *,
        signal_counts: Optional[Any] = None,
        idler_counts: Optional[Any] = None,
        include_target: bool = True,
        include_nuisance: bool = True,
    ) -> np.ndarray:
        features = np.asarray(target_features, dtype=float)
        normalized = (features - self.feature_mean) / self.feature_scale
        logits = np.broadcast_to(self.baseline_logit, (features.shape[0], self.baseline_logit.size)).copy()
        if include_target:
            logits += normalized @ self.target_coefficients
        if include_nuisance:
            if signal_counts is None or idler_counts is None:
                raise ValueError("signal/idler counts are required for the nuisance-conditioned model")
            signal_share = _smoothed_share(np.asarray(signal_counts, dtype=float), self.smoothing)
            idler_share = _smoothed_share(np.asarray(idler_counts, dtype=float), self.smoothing)
            logits += self.signal_idler_coefficients[0] * np.log(signal_share)
            logits += self.signal_idler_coefficients[1] * np.log(idler_share)
        return _softmax(logits)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in (
            "feature_mean", "feature_scale", "baseline_logit",
            "target_coefficients", "signal_idler_coefficients",
        ):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["schema"] = CALIBRATION_SCHEMA
        payload["model_input_audit"] = dict(MODEL_INPUT_AUDIT)
        payload["operation_point_formula"] = (
            "softmax(baseline_logit + standardized_target_density @ target_coefficients "
            "+ beta_signal*log(signal_share) + beta_idler*log(idler_share))"
        )
        return payload


@dataclass(frozen=True)
class LatentOperationCalibrationV1:
    """Global count-pattern calibrator inferred from a subset of operation points."""

    system_type: str
    baseline_logit: np.ndarray
    loading_matrix: np.ndarray
    selected_rank: int
    smoothing: float
    training_sample_count: int
    split_hash: str
    research_only: bool = True
    promotion_ready: bool = False

    def predict_hidden_probability(
        self,
        observed_counts: Any,
        observed_mask: Any,
    ) -> np.ndarray:
        observed = np.asarray(observed_counts, dtype=float)
        mask = np.asarray(observed_mask, dtype=bool).reshape(-1)
        if observed.ndim != 2 or observed.shape[1] != int(np.sum(mask)):
            raise ValueError("observed_counts must contain exactly the selected operation points")
        if mask.size != self.baseline_logit.size or np.all(mask) or not np.any(mask):
            raise ValueError("observed_mask must select a strict non-empty operation subset")
        observed_probability = _smoothed_share(observed, self.smoothing)
        baseline_observed = _softmax(self.baseline_logit[mask][None, :])
        residual = np.log(observed_probability) - np.log(baseline_observed)
        residual -= residual.mean(axis=1, keepdims=True)
        observed_loading = self.loading_matrix[:, mask]
        hidden_loading = self.loading_matrix[:, ~mask]
        score = residual @ np.linalg.pinv(observed_loading)
        return _softmax(self.baseline_logit[~mask] + score @ hidden_loading)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": LATENT_CALIBRATION_SCHEMA,
            "system_type": self.system_type,
            "baseline_logit": self.baseline_logit.tolist(),
            "loading_matrix": self.loading_matrix.tolist(),
            "selected_rank": int(self.selected_rank),
            "smoothing": float(self.smoothing),
            "training_sample_count": int(self.training_sample_count),
            "split_hash": self.split_hash,
            "research_only": self.research_only,
            "promotion_ready": self.promotion_ready,
            "model_input_audit": dict(MODEL_INPUT_AUDIT),
            "inference_contract": (
                "infer latent operation-pattern scores from observed operation counts only; "
                "predict the disjoint hidden operation subset"
            ),
        }


def load_operation_point_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "v6_operation_point_calibration" or int(payload.get("version", 0)) != 1:
        raise ValueError("unsupported operation-point calibration config")
    if payload.get("research_only") is not True or payload.get("promotion_ready") is not False:
        raise ValueError("operation-point calibration must remain research-only and non-promotable")
    if not payload.get("ridge_alphas"):
        raise ValueError("ridge_alphas must be non-empty")
    if not payload.get("latent_ranks") or not payload.get("operation_mask_schemes"):
        raise ValueError("latent_ranks and operation_mask_schemes must be non-empty")
    return payload


def target_density_features(target: Mapping[str, Any]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Return gauge-invariant independent single-excitation density coordinates."""
    amplitude = np.asarray(target.get("amplitudes", []), dtype=float).reshape(-1)
    phase = np.asarray(target.get("phases_rad", []), dtype=float).reshape(-1)
    if amplitude.size < 2 or phase.shape != amplitude.shape:
        raise ValueError("target amplitudes/phases are incomplete")
    amplitude = amplitude / max(float(np.linalg.norm(amplitude)), 1e-12)
    state = amplitude * np.exp(1j * phase)
    density = state[:, None] * state.conj()[None, :]
    values: List[float] = []
    names: List[str] = []
    qubit_count = amplitude.size
    for index in range(qubit_count - 1):
        values.append(float(density[index, index].real))
        names.append(f"population_q{index + 1}_vs_q{qubit_count}")
    for first in range(qubit_count):
        for second in range(first + 1, qubit_count):
            values.extend((float(density[first, second].real), float(density[first, second].imag)))
            names.extend((
                f"coherence_real_q{first + 1}_q{second + 1}",
                f"coherence_imag_q{first + 1}_q{second + 1}",
            ))
    return np.asarray(values, dtype=np.float64), tuple(names)


def load_operation_point_dataset(
    manifest_dir: str | Path,
    *,
    data_root: str | Path,
    system_type: str,
) -> OperationPointDatasetV1:
    source_records = tuple(load_measured_records(manifest_dir, system_type=system_type))
    split_payload = json.loads(
        (Path(manifest_dir) / "measured_splits_v1.json").read_text(encoding="utf-8")
    )
    if not bool((split_payload.get("audit") or {}).get("passed")):
        raise ValueError("operation-point learning requires a passing split audit")
    assignments = dict(split_payload.get("assignments") or {})
    measured = MeasuredManifestDataset(source_records, data_root=data_root)
    # MeasuredManifestDataset owns the deterministic row order.  Targets,
    # split assignments, and downstream snapshot-bound commands must follow
    # that same order; pairing its sorted rows with the source JSONL order
    # silently cross-links almost every measured count array to another state.
    records = tuple(measured.records)
    coin_counts: List[np.ndarray] = []
    signal_counts: List[np.ndarray] = []
    idler_counts: List[np.ndarray] = []
    target_rows: List[np.ndarray] = []
    feature_names: Optional[Tuple[str, ...]] = None
    for index, record in enumerate(records):
        row = measured[index]
        coin_counts.append(np.asarray(row["counts"]["coin_bins"], dtype=np.float64).reshape(-1))
        signal_counts.append(np.asarray(row["counts"]["signal_bins"], dtype=np.float64).reshape(-1))
        idler_counts.append(np.asarray(row["counts"]["idler_bins"], dtype=np.float64).reshape(-1))
        features, names = target_density_features(record.exact_target_state_params)
        if feature_names is not None and names != feature_names:
            raise ValueError("target feature schema changed within one system")
        feature_names = names
        target_rows.append(features)
    operation_counts = {values.size for values in coin_counts + signal_counts + idler_counts}
    if len(operation_counts) != 1:
        raise ValueError("measured operation arrays have inconsistent lengths")
    return OperationPointDatasetV1(
        system_type=system_type,
        records=records,
        coin_counts=np.stack(coin_counts),
        signal_counts=np.stack(signal_counts),
        idler_counts=np.stack(idler_counts),
        target_features=np.stack(target_rows),
        target_feature_names=feature_names or (),
        split_names=np.asarray([assignments[record.sample_id] for record in records]),
    )


def _smoothed_share(counts: np.ndarray, smoothing: float) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    return (values + float(smoothing)) / (
        values.sum(axis=1, keepdims=True) + float(smoothing) * values.shape[1]
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(centered)
    return probability / probability.sum(axis=1, keepdims=True)


def conditional_nll_per_click(counts: Any, probability: Any) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    predicted = np.maximum(np.asarray(probability, dtype=float), 1e-12)
    predicted = predicted / predicted.sum(axis=1, keepdims=True)
    return -np.sum(values * np.log(predicted), axis=1) / np.maximum(values.sum(axis=1), 1.0)


def _split_masks(dataset: OperationPointDatasetV1) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    names = dataset.split_names.astype(str)
    if dataset.system_type == "3qubit":
        return names == "train", names == "validation", {
            "target_holdout": names == "target_holdout",
            "probe_holdout": names == "probe_holdout",
        }
    return names == "train_target_family", names == "validation_target_family", {
        "target_ood": names == "target_ood",
        "time_run_ood": names == "time_run_ood",
        "q4_plus90_intervention_test": names == "q4_plus90_intervention_test",
        "probe_holdout": names == "probe_holdout",
    }


def _fit_model(
    dataset: OperationPointDatasetV1,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    ridge_alphas: Sequence[float],
    smoothing: float,
    split_hash: str,
) -> Tuple[OperationPointCalibrationModelV1, Dict[str, Any]]:
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError("training and validation splits must be non-empty")
    counts = dataset.coin_counts
    target = dataset.target_features
    target_mean = target[train_mask].mean(axis=0)
    target_scale = target[train_mask].std(axis=0)
    target_scale[target_scale < 1e-8] = 1.0
    normalized = (target - target_mean) / target_scale
    operation_probability = dataset.coin_counts[train_mask].sum(axis=0) + smoothing
    operation_probability /= operation_probability.sum()
    baseline_logit = np.log(operation_probability)
    coin_log_share = np.log(_smoothed_share(counts, smoothing))
    signal_log_share = np.log(_smoothed_share(dataset.signal_counts, smoothing))
    idler_log_share = np.log(_smoothed_share(dataset.idler_counts, smoothing))
    nuisance_design = np.stack((
        signal_log_share[train_mask].reshape(-1),
        idler_log_share[train_mask].reshape(-1),
    ), axis=1)
    nuisance_target = (coin_log_share[train_mask] - baseline_logit).reshape(-1)
    nuisance_coefficients = np.linalg.solve(
        nuisance_design.T @ nuisance_design + 1e-3 * np.eye(2),
        nuisance_design.T @ nuisance_target,
    )
    # Keep the target-density law independently identifiable from the detector
    # marginals.  This makes the target-only rung a genuine nested baseline;
    # the full rung adds the signal/idler nuisance correction afterwards.
    target_residual = coin_log_share - baseline_logit
    gram = normalized[train_mask].T @ normalized[train_mask]
    cross = normalized[train_mask].T @ target_residual[train_mask]
    validation_scores: Dict[str, float] = {}
    coefficient_by_alpha: Dict[float, np.ndarray] = {}
    for alpha_value in ridge_alphas:
        alpha = float(alpha_value)
        coefficients = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), cross)
        coefficients -= coefficients.mean(axis=1, keepdims=True)
        # Select the target-density regularizer on the target-only rung.  The
        # contemporaneous detector marginals are a separate diagnostic rung.
        logits = baseline_logit + normalized[validation_mask] @ coefficients
        validation_scores[f"{alpha:g}"] = float(np.mean(
            conditional_nll_per_click(counts[validation_mask], _softmax(logits))
        ))
        coefficient_by_alpha[alpha] = coefficients
    selected_alpha = min(
        coefficient_by_alpha,
        key=lambda value: (validation_scores[f"{value:g}"], value),
    )
    model = OperationPointCalibrationModelV1(
        system_type=dataset.system_type,
        feature_names=dataset.target_feature_names,
        feature_mean=target_mean,
        feature_scale=target_scale,
        baseline_logit=baseline_logit,
        target_coefficients=coefficient_by_alpha[selected_alpha],
        signal_idler_coefficients=nuisance_coefficients,
        ridge_alpha=selected_alpha,
        smoothing=float(smoothing),
        training_sample_count=int(np.sum(train_mask)),
        split_hash=str(split_hash),
    )
    return model, {
        "selected_ridge_alpha": selected_alpha,
        "validation_nll_by_alpha": validation_scores,
    }


def _bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.asarray([
        values[rng.integers(0, values.size, values.size)].mean()
        for _ in range(int(repetitions))
    ])
    return float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def _model_probabilities(
    model: OperationPointCalibrationModelV1,
    dataset: OperationPointDatasetV1,
    mask: np.ndarray,
    *,
    kind: str,
) -> np.ndarray:
    if kind == "uniform":
        return np.full((int(np.sum(mask)), dataset.coin_counts.shape[1]), 1.0 / dataset.coin_counts.shape[1])
    if kind == "operation_only":
        probability = _softmax(model.baseline_logit[None, :])
        return np.broadcast_to(probability, (int(np.sum(mask)), probability.shape[1]))
    return model.predict_probability(
        dataset.target_features[mask],
        signal_counts=dataset.signal_counts[mask],
        idler_counts=dataset.idler_counts[mask],
        include_target=kind in {"target_density", "full"},
        include_nuisance=kind in {"signal_idler", "full"},
    )


def _evaluate_split(
    model: OperationPointCalibrationModelV1,
    dataset: OperationPointDatasetV1,
    mask: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> Dict[str, Any]:
    if not np.any(mask):
        return {"available": False, "sample_count": 0}
    metrics: Dict[str, Any] = {"available": True, "sample_count": int(np.sum(mask))}
    nll: Dict[str, np.ndarray] = {}
    for kind in ("uniform", "operation_only", "target_density", "signal_idler", "full"):
        nll[kind] = conditional_nll_per_click(
            dataset.coin_counts[mask], _model_probabilities(model, dataset, mask, kind=kind),
        )
        metrics[f"{kind}_conditional_nll_per_click"] = float(np.mean(nll[kind]))
    for label, candidate in (("target_gain", "target_density"), ("nuisance_gain", "signal_idler"), ("full_gain", "full")):
        improvement = nll["operation_only"] - nll[candidate]
        interval = _bootstrap_interval(improvement, repetitions, seed + len(metrics))
        metrics[label] = float(np.mean(improvement))
        metrics[f"{label}_bootstrap_95"] = list(interval)
        metrics[f"{label}_significant"] = bool(interval[0] > 0.0)
        metrics[f"{label}_win_fraction"] = float(np.mean(improvement > 0.0))
    return metrics


def _group_fold_indices(dataset: OperationPointDatasetV1, folds: int, seed: int) -> np.ndarray:
    group_ids = sorted(set(record.base_state_family_id for record in dataset.records))
    fold_by_group = {
        group_id: int.from_bytes(
            hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()[:8], "big"
        ) % int(folds)
        for group_id in group_ids
    }
    return np.asarray([fold_by_group[record.base_state_family_id] for record in dataset.records])


def _cross_validation(
    dataset: OperationPointDatasetV1,
    model: OperationPointCalibrationModelV1,
    *,
    folds: int,
    smoothing: float,
    repetitions: int,
    seed: int,
) -> Dict[str, Any]:
    fold_index = _group_fold_indices(dataset, folds, seed)
    eligible = np.asarray([
        record.intervention_id is None and record.probe_anchor_id is None
        for record in dataset.records
    ])
    fold_metrics = []
    all_improvements: List[float] = []
    for fold in range(int(folds)):
        test_mask = eligible & (fold_index == fold)
        train_mask = eligible & (fold_index != fold)
        if not np.any(test_mask) or not np.any(train_mask):
            continue
        fitted, _ = _fit_model(
            dataset, train_mask, test_mask,
            ridge_alphas=(model.ridge_alpha,), smoothing=smoothing, split_hash=model.split_hash,
        )
        operation_nll = conditional_nll_per_click(
            dataset.coin_counts[test_mask],
            _model_probabilities(fitted, dataset, test_mask, kind="operation_only"),
        )
        target_nll = conditional_nll_per_click(
            dataset.coin_counts[test_mask],
            _model_probabilities(fitted, dataset, test_mask, kind="target_density"),
        )
        improvement = operation_nll - target_nll
        all_improvements.extend(improvement.tolist())
        fold_metrics.append({
            "fold": fold,
            "sample_count": int(np.sum(test_mask)),
            "target_gain": float(np.mean(improvement)),
        })
    interval = _bootstrap_interval(np.asarray(all_improvements), repetitions, seed + 77)
    return {
        "folds": fold_metrics,
        "sample_count": len(all_improvements),
        "mean_target_gain": float(np.mean(all_improvements)),
        "bootstrap_95": list(interval),
        "significant": bool(interval[0] > 0.0),
        "all_fold_directions_positive": all(item["target_gain"] > 0.0 for item in fold_metrics),
    }


def _permutation_control(
    dataset: OperationPointDatasetV1,
    model: OperationPointCalibrationModelV1,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> Dict[str, Any]:
    normalized = (dataset.target_features - model.feature_mean) / model.feature_scale
    share = np.log(_smoothed_share(dataset.coin_counts, model.smoothing))
    residual = share - model.baseline_logit
    train_index = np.flatnonzero(train_mask)
    baseline_nll = conditional_nll_per_click(
        dataset.coin_counts[test_mask], _model_probabilities(model, dataset, test_mask, kind="operation_only"),
    )
    observed_nll = conditional_nll_per_click(
        dataset.coin_counts[test_mask], _model_probabilities(model, dataset, test_mask, kind="target_density"),
    )
    observed_gain = float(np.mean(baseline_nll - observed_nll))
    rng = np.random.default_rng(int(seed))
    null_gains = []
    gram = normalized[train_mask].T @ normalized[train_mask]
    regularized = gram + model.ridge_alpha * np.eye(gram.shape[0])
    for _ in range(int(repetitions)):
        shuffled = train_index.copy()
        rng.shuffle(shuffled)
        cross = normalized[shuffled].T @ residual[train_mask]
        coefficients = np.linalg.solve(regularized, cross)
        coefficients -= coefficients.mean(axis=1, keepdims=True)
        probability = _softmax(
            model.baseline_logit + normalized[test_mask] @ coefficients
        )
        null_nll = conditional_nll_per_click(dataset.coin_counts[test_mask], probability)
        null_gains.append(float(np.mean(baseline_nll - null_nll)))
    null = np.asarray(null_gains)
    return {
        "repetitions": int(repetitions),
        "observed_target_gain": observed_gain,
        "null_gain_mean": float(np.mean(null)),
        "null_gain_95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "permutation_p_value": float((1 + np.sum(null >= observed_gain)) / (1 + null.size)),
        "passed": bool(observed_gain > 0.0 and np.sum(null >= observed_gain) == 0),
    }


def _learning_curve(
    dataset: OperationPointDatasetV1,
    model: OperationPointCalibrationModelV1,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    fractions: Sequence[float],
    smoothing: float,
    seed: int,
) -> List[Dict[str, Any]]:
    groups = sorted({dataset.records[index].base_state_family_id for index in np.flatnonzero(train_mask)})
    groups = sorted(groups, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    result = []
    for fraction in fractions:
        group_count = max(2, min(len(groups), int(round(len(groups) * float(fraction)))))
        selected = set(groups[:group_count])
        subset = train_mask & np.asarray([record.base_state_family_id in selected for record in dataset.records])
        fitted, _ = _fit_model(
            dataset, subset, validation_mask,
            ridge_alphas=(model.ridge_alpha,), smoothing=smoothing, split_hash=model.split_hash,
        )
        baseline = conditional_nll_per_click(
            dataset.coin_counts[test_mask], _model_probabilities(fitted, dataset, test_mask, kind="operation_only"),
        )
        prediction = conditional_nll_per_click(
            dataset.coin_counts[test_mask], _model_probabilities(fitted, dataset, test_mask, kind="target_density"),
        )
        result.append({
            "fraction": float(fraction),
            "training_group_count": group_count,
            "training_sample_count": int(np.sum(subset)),
            "target_gain": float(np.mean(baseline - prediction)),
        })
    return result


def _operation_masks(operation_count: int, names: Sequence[str]) -> Dict[str, np.ndarray]:
    operation_index = np.arange(int(operation_count))
    available = {
        "parity_even": operation_index % 2 == 0,
        "parity_odd": operation_index % 2 == 1,
        "block_outer": np.isin(operation_index % 4, (0, 3)),
        "block_inner": np.isin(operation_index % 4, (1, 2)),
    }
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown operation mask schemes: {unknown}")
    selected = {name: available[name] for name in names}
    if any(not np.any(mask) or np.all(mask) for mask in selected.values()):
        raise ValueError("operation masks must define disjoint observed and hidden subsets")
    return selected


def _latent_nll_arrays(
    model: LatentOperationCalibrationV1,
    dataset: OperationPointDatasetV1,
    sample_mask: np.ndarray,
    operation_mask_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    baseline_rows: List[np.ndarray] = []
    predicted_rows: List[np.ndarray] = []
    for observed_mask in _operation_masks(dataset.coin_counts.shape[1], operation_mask_names).values():
        hidden_counts = dataset.coin_counts[sample_mask][:, ~observed_mask]
        baseline_probability = _softmax(model.baseline_logit[~observed_mask][None, :])
        baseline_probability = np.broadcast_to(
            baseline_probability, (hidden_counts.shape[0], hidden_counts.shape[1])
        )
        predicted_probability = model.predict_hidden_probability(
            dataset.coin_counts[sample_mask][:, observed_mask], observed_mask,
        )
        baseline_rows.append(conditional_nll_per_click(hidden_counts, baseline_probability))
        predicted_rows.append(conditional_nll_per_click(hidden_counts, predicted_probability))
    return np.stack(baseline_rows), np.stack(predicted_rows)


def _fit_latent_model(
    dataset: OperationPointDatasetV1,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    ranks: Sequence[int],
    operation_mask_names: Sequence[str],
    smoothing: float,
    split_hash: str,
) -> Tuple[LatentOperationCalibrationV1, Dict[str, Any]]:
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError("latent calibration requires non-empty training and validation splits")
    operation_probability = dataset.coin_counts[train_mask].sum(axis=0) + float(smoothing)
    operation_probability /= operation_probability.sum()
    baseline_logit = np.log(operation_probability)
    residual = np.log(_smoothed_share(dataset.coin_counts[train_mask], smoothing)) - baseline_logit
    residual -= residual.mean(axis=1, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(residual, full_matrices=False)
    maximum_rank = min(right_vectors.shape[0], min(
        int(np.sum(mask)) for mask in _operation_masks(
            dataset.coin_counts.shape[1], operation_mask_names,
        ).values()
    ))
    valid_ranks = sorted({int(rank) for rank in ranks if 0 < int(rank) <= maximum_rank})
    if not valid_ranks:
        raise ValueError("no latent rank is compatible with the operation masks")
    validation_nll: Dict[str, float] = {}
    validation_gain: Dict[str, float] = {}
    model_by_rank: Dict[int, LatentOperationCalibrationV1] = {}
    for rank in valid_ranks:
        candidate = LatentOperationCalibrationV1(
            system_type=dataset.system_type,
            baseline_logit=baseline_logit,
            loading_matrix=right_vectors[:rank],
            selected_rank=rank,
            smoothing=float(smoothing),
            training_sample_count=int(np.sum(train_mask)),
            split_hash=str(split_hash),
        )
        baseline, predicted = _latent_nll_arrays(
            candidate, dataset, validation_mask, operation_mask_names,
        )
        validation_nll[str(rank)] = float(np.mean(predicted))
        validation_gain[str(rank)] = float(np.mean(baseline - predicted))
        model_by_rank[rank] = candidate
    selected_rank = min(valid_ranks, key=lambda rank: (validation_nll[str(rank)], rank))
    total_energy = float(np.sum(np.square(singular_values)))
    explained = float(
        np.sum(np.square(singular_values[:selected_rank])) / max(total_energy, 1e-12)
    )
    return model_by_rank[selected_rank], {
        "selected_rank": selected_rank,
        "validation_nll_by_rank": validation_nll,
        "validation_gain_by_rank": validation_gain,
        "selected_rank_explained_residual_variance": explained,
        "leading_singular_values": singular_values[:20].tolist(),
    }


def _evaluate_latent_split(
    model: LatentOperationCalibrationV1,
    dataset: OperationPointDatasetV1,
    sample_mask: np.ndarray,
    *,
    operation_mask_names: Sequence[str],
    repetitions: int,
    seed: int,
) -> Dict[str, Any]:
    if not np.any(sample_mask):
        return {"available": False, "sample_count": 0}
    baseline, predicted = _latent_nll_arrays(
        model, dataset, sample_mask, operation_mask_names,
    )
    uniform_rows = []
    for observed_mask in _operation_masks(
        dataset.coin_counts.shape[1], operation_mask_names,
    ).values():
        hidden_counts = dataset.coin_counts[sample_mask][:, ~observed_mask]
        uniform_rows.append(conditional_nll_per_click(
            hidden_counts,
            np.full_like(hidden_counts, 1.0 / hidden_counts.shape[1], dtype=float),
        ))
    uniform = np.stack(uniform_rows)
    gain_by_sample = np.mean(baseline - predicted, axis=0)
    static_gain_by_sample = np.mean(uniform - baseline, axis=0)
    total_gain_by_sample = np.mean(uniform - predicted, axis=0)
    interval = _bootstrap_interval(gain_by_sample, repetitions, seed)
    static_interval = _bootstrap_interval(static_gain_by_sample, repetitions, seed + 1)
    mask_metrics = []
    for mask_index, name in enumerate(operation_mask_names):
        mask_gain = baseline[mask_index] - predicted[mask_index]
        mask_metrics.append({
            "mask": str(name),
            "baseline_nll_per_click": float(np.mean(baseline[mask_index])),
            "calibrated_nll_per_click": float(np.mean(predicted[mask_index])),
            "gain": float(np.mean(mask_gain)),
            "win_fraction": float(np.mean(mask_gain > 0.0)),
        })
    return {
        "available": True,
        "sample_count": int(np.sum(sample_mask)),
        "observed_operation_fraction": float(np.mean([
            np.mean(mask) for mask in _operation_masks(
                dataset.coin_counts.shape[1], operation_mask_names,
            ).values()
        ])),
        "baseline_nll_per_click": float(np.mean(baseline)),
        "calibrated_nll_per_click": float(np.mean(predicted)),
        "uniform_nll_per_click": float(np.mean(uniform)),
        "static_operation_gain": float(np.mean(static_gain_by_sample)),
        "static_operation_gain_bootstrap_95": list(static_interval),
        "total_gain_over_uniform": float(np.mean(total_gain_by_sample)),
        "latent_operation_gain": float(np.mean(gain_by_sample)),
        "bootstrap_95": list(interval),
        "win_fraction": float(np.mean(gain_by_sample > 0.0)),
        "significant": bool(interval[0] > 0.0),
        "mask_metrics": mask_metrics,
    }


def _latent_permutation_control(
    model: LatentOperationCalibrationV1,
    dataset: OperationPointDatasetV1,
    test_mask: np.ndarray,
    *,
    operation_mask_names: Sequence[str],
    repetitions: int,
    seed: int,
) -> Dict[str, Any]:
    baseline, predicted = _latent_nll_arrays(model, dataset, test_mask, operation_mask_names)
    observed_gain = float(np.mean(baseline - predicted))
    rng = np.random.default_rng(int(seed))
    masks = _operation_masks(dataset.coin_counts.shape[1], operation_mask_names)
    null_gains = []
    for _ in range(int(repetitions)):
        repetition_gains = []
        for mask_index, observed_mask in enumerate(masks.values()):
            observed_counts = dataset.coin_counts[test_mask][:, observed_mask]
            observed_probability = _smoothed_share(observed_counts, model.smoothing)
            baseline_observed = _softmax(model.baseline_logit[observed_mask][None, :])
            residual = np.log(observed_probability) - np.log(baseline_observed)
            residual -= residual.mean(axis=1, keepdims=True)
            score = residual @ np.linalg.pinv(model.loading_matrix[:, observed_mask])
            hidden_loading = model.loading_matrix[:, ~observed_mask].copy()
            hidden_loading = hidden_loading[:, rng.permutation(hidden_loading.shape[1])]
            probability = _softmax(
                model.baseline_logit[~observed_mask] + score @ hidden_loading
            )
            null_nll = conditional_nll_per_click(
                dataset.coin_counts[test_mask][:, ~observed_mask], probability,
            )
            repetition_gains.append(baseline[mask_index] - null_nll)
        null_gains.append(float(np.mean(np.stack(repetition_gains))))
    null = np.asarray(null_gains)
    exceedance = int(np.sum(null >= observed_gain))
    return {
        "control": "permute hidden operation labels while preserving latent scores",
        "repetitions": int(repetitions),
        "observed_gain": observed_gain,
        "null_gain_mean": float(np.mean(null)),
        "null_gain_95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "permutation_p_value": float((1 + exceedance) / (1 + null.size)),
        "passed": bool(observed_gain > 0.0 and exceedance == 0),
    }


def _latent_cross_validation(
    dataset: OperationPointDatasetV1,
    *,
    folds: int,
    ranks: Sequence[int],
    operation_mask_names: Sequence[str],
    smoothing: float,
    split_hash: str,
    repetitions: int,
    seed: int,
) -> Dict[str, Any]:
    fold_index = _group_fold_indices(dataset, folds, seed)
    eligible = np.asarray([
        record.intervention_id is None and record.probe_anchor_id is None
        for record in dataset.records
    ])
    fold_metrics = []
    all_gain: List[float] = []
    for fold in range(int(folds)):
        test_mask = eligible & (fold_index == fold)
        validation_mask = eligible & (fold_index == ((fold + 1) % int(folds)))
        train_mask = eligible & ~(test_mask | validation_mask)
        if not np.any(test_mask) or not np.any(validation_mask) or not np.any(train_mask):
            continue
        fitted, selection = _fit_latent_model(
            dataset, train_mask, validation_mask,
            ranks=ranks, operation_mask_names=operation_mask_names,
            smoothing=smoothing, split_hash=split_hash,
        )
        baseline, predicted = _latent_nll_arrays(
            fitted, dataset, test_mask, operation_mask_names,
        )
        gain_by_sample = np.mean(baseline - predicted, axis=0)
        all_gain.extend(gain_by_sample.tolist())
        fold_metrics.append({
            "fold": fold,
            "sample_count": int(np.sum(test_mask)),
            "selected_rank": int(selection["selected_rank"]),
            "latent_operation_gain": float(np.mean(gain_by_sample)),
        })
    interval = _bootstrap_interval(np.asarray(all_gain), repetitions, seed + 71)
    return {
        "nested_rank_selection": True,
        "folds": fold_metrics,
        "sample_count": len(all_gain),
        "mean_latent_operation_gain": float(np.mean(all_gain)),
        "bootstrap_95": list(interval),
        "all_fold_directions_positive": all(
            item["latent_operation_gain"] > 0.0 for item in fold_metrics
        ),
        "significant": bool(interval[0] > 0.0),
    }


def _latent_learning_curve(
    dataset: OperationPointDatasetV1,
    model: LatentOperationCalibrationV1,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    fractions: Sequence[float],
    operation_mask_names: Sequence[str],
    smoothing: float,
    seed: int,
) -> List[Dict[str, Any]]:
    groups = sorted({dataset.records[index].base_state_family_id for index in np.flatnonzero(train_mask)})
    groups = sorted(groups, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    curve = []
    for fraction in fractions:
        group_count = max(2, min(len(groups), int(round(len(groups) * float(fraction)))))
        selected_groups = set(groups[:group_count])
        subset = train_mask & np.asarray([
            record.base_state_family_id in selected_groups for record in dataset.records
        ])
        fitted, _ = _fit_latent_model(
            dataset, subset, validation_mask,
            ranks=(model.selected_rank,), operation_mask_names=operation_mask_names,
            smoothing=smoothing, split_hash=model.split_hash,
        )
        baseline, predicted = _latent_nll_arrays(
            fitted, dataset, test_mask, operation_mask_names,
        )
        curve.append({
            "fraction": float(fraction),
            "training_group_count": group_count,
            "training_sample_count": int(np.sum(subset)),
            "latent_operation_gain": float(np.mean(baseline - predicted)),
        })
    return curve


def _latent_component_features(
    model: LatentOperationCalibrationV1,
    system_type: str,
) -> Dict[str, Any]:
    components = []
    for component_index, loading in enumerate(model.loading_matrix):
        positive = np.argsort(loading)[-5:][::-1]
        negative = np.argsort(loading)[:5]
        metadata = [_operation_metadata(system_type, index) for index in range(loading.size)]
        correlations = {}
        for name in ("cycle_index", "local_index", "row", "group"):
            coordinate = np.asarray([item[name] for item in metadata], dtype=float)
            correlations[name] = float(np.corrcoef(loading, coordinate)[0, 1]) \
                if np.std(coordinate) > 0.0 else 0.0
        components.append({
            "component": component_index + 1,
            "loading_rms": float(np.sqrt(np.mean(np.square(loading)))),
            "coordinate_correlations": correlations,
            "positive_operation_points": [
                {**metadata[index], "loading": float(loading[index])} for index in positive
            ],
            "negative_operation_points": [
                {**metadata[index], "loading": float(loading[index])} for index in negative
            ],
        })
    return {
        "schema": "latent_operation_features_v1",
        "selected_rank": int(model.selected_rank),
        "feature_interpretation": (
            "joint operation-point variation modes learned from coin-count log shares; "
            "component sign is conventional"
        ),
        "components": components,
    }


def _operation_metadata(system_type: str, operation_index: int) -> Dict[str, int]:
    if system_type == "3qubit":
        return {
            "operation_index": int(operation_index), "cycle_index": 0,
            "local_index": int(operation_index), "row": int(operation_index) // 5,
            "group": int(operation_index) % 5,
        }
    local = int(operation_index) % 64
    return {
        "operation_index": int(operation_index), "cycle_index": int(operation_index) // 64,
        "local_index": local, "row": local // 4, "group": local % 4,
    }


def _coefficient_bootstrap(
    dataset: OperationPointDatasetV1,
    model: OperationPointCalibrationModelV1,
    train_mask: np.ndarray,
    *,
    repetitions: int,
    sign_stability_threshold: float,
    seed: int,
) -> Dict[str, Any]:
    normalized = (dataset.target_features - model.feature_mean) / model.feature_scale
    residual = np.log(_smoothed_share(dataset.coin_counts, model.smoothing)) - model.baseline_logit
    grouped: Dict[str, List[int]] = {}
    for index in np.flatnonzero(train_mask):
        grouped.setdefault(dataset.records[index].base_state_family_id, []).append(int(index))
    group_ids = sorted(grouped)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty((int(repetitions),) + model.target_coefficients.shape, dtype=np.float32)
    for repetition in range(int(repetitions)):
        selected_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
        selected_index = np.asarray([
            index for group_id in selected_groups for index in grouped[str(group_id)]
        ], dtype=np.int64)
        x = normalized[selected_index]
        coefficient = np.linalg.solve(
            x.T @ x + model.ridge_alpha * np.eye(x.shape[1]),
            x.T @ residual[selected_index],
        )
        coefficient -= coefficient.mean(axis=1, keepdims=True)
        bootstrap[repetition] = coefficient
    laws = []
    for feature_index, feature_name in enumerate(model.feature_names):
        operation_index = int(np.argmax(np.abs(model.target_coefficients[feature_index])))
        values = bootstrap[:, feature_index, operation_index]
        lower, upper = np.quantile(values, (0.025, 0.975))
        point = float(model.target_coefficients[feature_index, operation_index])
        sign_stability = float(max(np.mean(values > 0.0), np.mean(values < 0.0)))
        laws.append({
            "feature_name": feature_name,
            "unit": "log_operation_share_per_standardized_density_coordinate",
            **_operation_metadata(dataset.system_type, operation_index),
            "coefficient": point,
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_std": float(np.std(values, ddof=1)),
            "bootstrap_95": [float(lower), float(upper)],
            "sign_stability": sign_stability,
            "significant": bool(lower > 0.0 or upper < 0.0) and sign_stability >= float(sign_stability_threshold),
            "global_coefficient_rms": float(np.sqrt(np.mean(np.square(model.target_coefficients[feature_index])))),
        })
    laws.sort(key=lambda item: (-item["global_coefficient_rms"], item["feature_name"]))
    return {
        "bootstrap_repetitions": int(repetitions),
        "significant_law_count": sum(bool(item["significant"]) for item in laws),
        "laws": laws,
    }


def _intervention_benchmark(
    model: OperationPointCalibrationModelV1,
    dataset: OperationPointDatasetV1,
) -> Dict[str, Any]:
    if dataset.system_type != "4qubit":
        return {"available": False, "reason": "plus90 intervention exists only for 4Q"}
    grouped: Dict[str, Dict[str, List[int]]] = {}
    for index, record in enumerate(dataset.records):
        if dataset.split_names[index] != "q4_plus90_intervention_test":
            continue
        role = "intervention" if record.intervention_id == "q4_recovery_plus90" else "base"
        grouped.setdefault(record.base_state_family_id, {"base": [], "intervention": []})[role].append(index)
    observed_delta = []
    predicted_delta = []
    valid_families = 0
    for values in grouped.values():
        if not values["base"] or not values["intervention"]:
            continue
        valid_families += 1
        base_index = np.asarray(values["base"])
        intervention_index = np.asarray(values["intervention"])
        observed_base = _smoothed_share(dataset.coin_counts[base_index], model.smoothing).mean(axis=0)
        observed_intervention = _smoothed_share(dataset.coin_counts[intervention_index], model.smoothing).mean(axis=0)
        predicted_base = model.predict_probability(
            dataset.target_features[base_index], signal_counts=dataset.signal_counts[base_index],
            idler_counts=dataset.idler_counts[base_index], include_nuisance=False,
        ).mean(axis=0)
        predicted_intervention = model.predict_probability(
            dataset.target_features[intervention_index], signal_counts=dataset.signal_counts[intervention_index],
            idler_counts=dataset.idler_counts[intervention_index], include_nuisance=False,
        ).mean(axis=0)
        observed_delta.extend((np.log(observed_intervention) - np.log(observed_base)).tolist())
        predicted_delta.extend((np.log(predicted_intervention) - np.log(predicted_base)).tolist())
    if not observed_delta:
        return {"available": False, "reason": "no complete base/intervention families"}
    observed = np.asarray(observed_delta)
    predicted = np.asarray(predicted_delta)
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if np.std(predicted) > 0.0 else 0.0
    return {
        "available": True,
        "family_count": valid_families,
        "delta_log_rate_rmse": float(np.sqrt(np.mean(np.square(predicted - observed)))),
        "delta_log_rate_correlation": correlation,
        "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
        "accepted_as_significant_evidence": False,
    }


def run_operation_point_validation(
    manifest_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    data_root: str | Path = "data",
    systems: Sequence[str] = ("3qubit", "4qubit"),
) -> Dict[str, Any]:
    config = load_operation_point_config(config_path)
    split_payload = json.loads(
        (Path(manifest_dir) / "measured_splits_v1.json").read_text(encoding="utf-8")
    )
    split_hash = str(split_payload["split_hash"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "schema": "operation_point_validation_summary_v1",
        "systems": {},
        "model_input_audit": dict(MODEL_INPUT_AUDIT),
        "research_only": True,
        "promotion_ready": False,
        "hardware_operations": False,
        "absolute_rate_calibration_available": False,
        "absolute_rate_calibration_reason": (
            "legacy 3Q duration/gate exposure is missing and target-only provenance cannot identify historical commands"
        ),
    }
    all_laws = {}
    all_curves = {}
    for system_offset, system_type in enumerate(systems):
        dataset = load_operation_point_dataset(
            manifest_dir, data_root=data_root, system_type=system_type,
        )
        train_mask, validation_mask, test_masks = _split_masks(dataset)
        target_model, target_selection = _fit_model(
            dataset, train_mask, validation_mask,
            ridge_alphas=config["ridge_alphas"], smoothing=float(config["count_smoothing"]),
            split_hash=split_hash,
        )
        operation_mask_names = tuple(str(value) for value in config["operation_mask_schemes"])
        latent_model, latent_selection = _fit_latent_model(
            dataset, train_mask, validation_mask,
            ranks=config["latent_ranks"], operation_mask_names=operation_mask_names,
            smoothing=float(config["count_smoothing"]), split_hash=split_hash,
        )
        primary_test_name = "target_holdout" if system_type == "3qubit" else "target_ood"
        primary_test_mask = test_masks[primary_test_name]
        target_split_metrics = {
            name: _evaluate_split(
                target_model, dataset, mask,
                repetitions=int(config["bootstrap_repetitions"]),
                seed=int(config["seed"]) + system_offset * 100 + index,
            )
            for index, (name, mask) in enumerate(test_masks.items())
        }
        target_cross_validation = _cross_validation(
            dataset, target_model, folds=int(config["cross_validation_folds"]),
            smoothing=float(config["count_smoothing"]),
            repetitions=int(config["bootstrap_repetitions"]),
            seed=int(config["seed"]) + system_offset * 1000,
        )
        target_permutation = _permutation_control(
            dataset, target_model, train_mask, primary_test_mask,
            repetitions=int(config["permutation_repetitions"]),
            seed=int(config["seed"]) + system_offset * 1000 + 1,
        )
        target_curve = _learning_curve(
            dataset, target_model, train_mask, validation_mask, primary_test_mask,
            fractions=config["learning_curve_fractions"],
            smoothing=float(config["count_smoothing"]),
            seed=int(config["seed"]) + system_offset * 1000 + 2,
        )
        target_coefficient_evidence = _coefficient_bootstrap(
            dataset, target_model, train_mask,
            repetitions=int(config["bootstrap_repetitions"]),
            sign_stability_threshold=float(config["sign_stability_threshold"]),
            seed=int(config["seed"]) + system_offset * 1000 + 3,
        )
        latent_split_metrics = {
            name: _evaluate_latent_split(
                latent_model, dataset, mask,
                operation_mask_names=operation_mask_names,
                repetitions=int(config["bootstrap_repetitions"]),
                seed=int(config["seed"]) + system_offset * 100 + 20 + index,
            )
            for index, (name, mask) in enumerate(test_masks.items())
        }
        latent_cross_validation = _latent_cross_validation(
            dataset, folds=int(config["cross_validation_folds"]),
            ranks=config["latent_ranks"], operation_mask_names=operation_mask_names,
            smoothing=float(config["count_smoothing"]), split_hash=split_hash,
            repetitions=int(config["bootstrap_repetitions"]),
            seed=int(config["seed"]) + system_offset * 1000 + 20,
        )
        latent_permutation = _latent_permutation_control(
            latent_model, dataset, primary_test_mask,
            operation_mask_names=operation_mask_names,
            repetitions=int(config["permutation_repetitions"]),
            seed=int(config["seed"]) + system_offset * 1000 + 21,
        )
        latent_curve = _latent_learning_curve(
            dataset, latent_model, train_mask, validation_mask, primary_test_mask,
            fractions=config["learning_curve_fractions"],
            operation_mask_names=operation_mask_names,
            smoothing=float(config["count_smoothing"]),
            seed=int(config["seed"]) + system_offset * 1000 + 22,
        )
        latent_features = _latent_component_features(latent_model, system_type)
        system_key = "3q" if system_type == "3qubit" else "4q"
        target_model_path = destination / f"target_conditioned_calibration_{system_key}.json"
        target_model_path.write_text(
            json.dumps(target_model.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        latent_model_path = destination / f"global_operation_calibration_{system_key}.json"
        latent_model_path.write_text(
            json.dumps(latent_model.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        target_verified = bool(
            target_split_metrics[primary_test_name].get("target_gain_significant")
            and target_permutation["passed"]
            and target_cross_validation["significant"]
        )
        latent_verified = bool(
            latent_split_metrics[primary_test_name].get("significant")
            and latent_permutation["passed"]
            and latent_cross_validation["significant"]
            and latent_cross_validation["all_fold_directions_positive"]
        )
        summary["systems"][system_type] = {
            "sample_count": len(dataset.records),
            "operation_count": int(dataset.coin_counts.shape[1]),
            "feature_count": int(dataset.target_features.shape[1]),
            "training_sample_count": int(np.sum(train_mask)),
            "validation_sample_count": int(np.sum(validation_mask)),
            "target_conditioning": {
                "selection": target_selection,
                "heldout": target_split_metrics,
                "group_cross_validation": target_cross_validation,
                "permutation_control": target_permutation,
                "intervention_benchmark": _intervention_benchmark(target_model, dataset),
                "significant_candidate_law_count": target_coefficient_evidence["significant_law_count"],
                "model_path": target_model_path.name,
                "verified": target_verified,
                "interpretation": (
                    "target-only historical command approximation; not accepted when grouped evidence fails"
                ),
            },
            "global_operation_calibration": {
                "selection": latent_selection,
                "heldout": latent_split_metrics,
                "nested_group_cross_validation": latent_cross_validation,
                "operation_label_permutation_control": latent_permutation,
                "model_path": latent_model_path.name,
                "selected_operation_feature_count": int(latent_model.selected_rank),
                "verified": latent_verified,
                "usage": (
                    "infer global operation-pattern state from a measured operation subset and "
                    "calibrate the disjoint remaining operation points"
                ),
            },
            "significant_learning_verified": latent_verified,
        }
        all_laws[system_type] = {
            "validated_latent_operation_features": latent_features,
            "unvalidated_target_density_candidate_laws": target_coefficient_evidence,
        }
        all_curves[system_type] = {
            "latent_operation_calibration": latent_curve,
            "target_conditioning": target_curve,
        }
    summary["significant_learning_verified"] = all(
        bool(item["significant_learning_verified"]) for item in summary["systems"].values()
    )
    summary["target_conditioned_learning_verified"] = all(
        bool(item["target_conditioning"]["verified"]) for item in summary["systems"].values()
    )
    summary["calibration_scope"] = (
        "conditional spatial operation-pattern calibration only; no absolute-rate or historical-command claim"
    )
    (destination / "significant_operation_point_features.json").write_text(
        json.dumps(all_laws, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "operation_point_learning_curves.json").write_text(
        json.dumps(all_curves, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "operation_point_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--systems", default="3qubit,4qubit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_operation_point_validation(
        args.measured_manifest_dir, args.output_dir, args.config,
        data_root=args.data_root,
        systems=tuple(value.strip() for value in args.systems.split(",") if value.strip()),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["significant_learning_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_SCHEMA",
    "LATENT_CALIBRATION_SCHEMA",
    "LatentOperationCalibrationV1",
    "OperationPointCalibrationModelV1",
    "OperationPointDatasetV1",
    "conditional_nll_per_click",
    "load_operation_point_dataset",
    "main",
    "run_operation_point_validation",
    "target_density_features",
]
