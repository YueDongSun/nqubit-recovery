"""Sample-efficient covariance-feature classifier for rho diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .features import density_fidelity_pure_target
from .synthetic import ERROR_SOURCES, SyntheticRhoSample
from .tomography_features import (
    PAULI_AXES,
    pauli_covariance_matrix,
    pauli_expectation_vector,
    rho_pair_distance_features,
    single_excitation_pair_features,
    single_excitation_subspace_features,
    validate_density_pair,
)
from .torch_model import default_mode_labels, default_w_target

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None
    Ridge = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


DEFAULT_ERROR_LABELS = list(ERROR_SOURCES)


@dataclass
class CovarianceTrainingConfig:
    num_qubits: int = 4
    seed: int = 1234
    validation_fraction: float = 0.2
    regularization_c: float = 1.0
    ridge_alpha: float = 1.0
    max_iter: int = 2000
    physics_feature_schema: str = "physics_v1"


def rho_pair_covariance_features(rho_target: np.ndarray, rho_measured: np.ndarray, num_qubits: int) -> np.ndarray:
    """Build low-sample features from tomographic covariance and coherence."""

    target, measured = validate_density_pair(rho_target, rho_measured)
    cov_target = pauli_covariance_matrix(target, num_qubits)
    cov_measured = pauli_covariance_matrix(measured, num_qubits)
    cov_residual = cov_measured - cov_target
    exp_target = pauli_expectation_vector(target, num_qubits)
    exp_measured = pauli_expectation_vector(measured, num_qubits)
    exp_residual = exp_measured - exp_target
    pieces = [
        _upper_triangle(cov_target),
        _upper_triangle(cov_measured),
        _upper_triangle(cov_residual),
        exp_target,
        exp_measured,
        exp_residual,
        single_excitation_subspace_features(target, num_qubits),
        single_excitation_subspace_features(measured, num_qubits),
        single_excitation_pair_features(target, measured, num_qubits),
        rho_pair_distance_features(target, measured),
    ]
    return np.concatenate(pieces).astype(np.float32)


def rho_pair_legacy_covariance_features(rho_target: np.ndarray, rho_measured: np.ndarray, num_qubits: int) -> np.ndarray:
    """Build the historical covariance feature vector for old JSON checkpoints."""

    target, measured = validate_density_pair(rho_target, rho_measured)
    cov_target = pauli_covariance_matrix(target, num_qubits)
    cov_measured = pauli_covariance_matrix(measured, num_qubits)
    cov_delta = cov_measured - cov_target
    pieces = [
        _upper_triangle(cov_target),
        _upper_triangle(cov_measured),
        _upper_triangle(cov_delta),
        _single_excitation_legacy_features(target, num_qubits),
        _single_excitation_legacy_features(measured, num_qubits),
        _single_excitation_legacy_features(target - measured, num_qubits),
    ]
    return np.concatenate(pieces).astype(np.float32)


def train_covariance_diagnosis_model(
    samples: Sequence[SyntheticRhoSample],
    output: str,
    config: Optional[CovarianceTrainingConfig] = None,
    mode_labels: Optional[Sequence[str]] = None,
    error_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train a compact classifier on explicit covariance features."""

    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for covariance backend training")
    if not samples:
        raise ValueError("At least one sample is required")

    cfg = config or CovarianceTrainingConfig()
    rng = np.random.default_rng(int(cfg.seed))
    mode_labels = list(mode_labels or default_mode_labels(cfg.num_qubits))
    error_labels = list(error_labels or DEFAULT_ERROR_LABELS)

    x = np.vstack([
        rho_pair_covariance_features(sample.rho_target, sample.rho_measured, cfg.num_qubits)
        for sample in samples
    ])
    y_mode = np.asarray([int(sample.excitation_mode) - 1 for sample in samples], dtype=int)
    y_error = np.asarray([error_labels.index(sample.error_source) for sample in samples], dtype=int)
    y_metrics = np.asarray([[sample.fidelity, sample.purity] for sample in samples], dtype=np.float64)

    train_idx, val_idx = _train_val_indices(len(samples), cfg.validation_fraction, rng)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_val = scaler.transform(x[val_idx])

    mode_classifier = _fit_classifier(x_train, y_mode[train_idx], cfg, len(mode_labels))
    error_classifier = _fit_classifier(x_train, y_error[train_idx], cfg, len(error_labels))
    regressor = Ridge(alpha=float(cfg.ridge_alpha))
    regressor.fit(x_train, y_metrics[train_idx])

    metrics_out = _evaluate_arrays(
        x_val,
        y_mode[val_idx],
        y_error[val_idx],
        y_metrics[val_idx],
        mode_classifier,
        error_classifier,
        regressor,
        len(mode_labels),
        len(error_labels),
    )
    metrics_out["epoch"] = 1
    metrics_out["train_loss"] = 0.0

    training_metrics = {
        "num_samples": len(samples),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "epochs": 1,
        "final": metrics_out,
        "best": metrics_out,
        "checkpoint_selection": "single_fit_validation",
        "history": [metrics_out],
    }
    payload = {
        "model_type": "covariance_logistic",
        "backend": "covariance",
        "num_qubits": int(cfg.num_qubits),
        "mode_labels": mode_labels,
        "error_labels": error_labels,
        "feature_config": {
            "input": "pauli_expectation_covariance_and_single_excitation_pair_residuals",
            "feature_dim": int(x.shape[1]),
            "pauli_axes": list(PAULI_AXES),
            "covariance_variables": int(3 * cfg.num_qubits),
            "physics_feature_schema": "physics_v1",
            "requested_physics_feature_schema": str(cfg.physics_feature_schema or "physics_v1"),
            "schema_note": (
                "covariance backend keeps its explicit low-sample feature vector; "
                "physics_feature_schema is recorded for orchestration compatibility"
            ),
            "uses_raw_rho_diff": False,
        },
        "scaler": {
            "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        },
        "mode_classifier": _classifier_to_payload(mode_classifier),
        "error_classifier": _classifier_to_payload(error_classifier),
        "metric_regressor": {
            "coef": np.asarray(regressor.coef_, dtype=float).tolist(),
            "intercept": np.asarray(regressor.intercept_, dtype=float).tolist(),
        },
        "training_metrics": training_metrics,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return {
        "output": str(output_path),
        "backend": "covariance",
        "num_qubits": int(cfg.num_qubits),
        **training_metrics,
    }


class RhoPairCovarianceDiagnosisModel:
    """Loaded covariance-feature diagnosis model with the standard predict API."""

    backend = "covariance"

    def __init__(self, payload: Dict[str, Any]):
        if payload.get("model_type") != "covariance_logistic":
            raise ValueError("Not a covariance_logistic checkpoint")
        self.payload = payload
        self.num_qubits = int(payload["num_qubits"])
        self.mode_labels = list(payload.get("mode_labels") or default_mode_labels(self.num_qubits))
        self.error_labels = list(payload.get("error_labels") or DEFAULT_ERROR_LABELS)
        self.feature_config = dict(payload.get("feature_config") or {
            "input": "pauli_covariance_matrix_and_single_excitation_coherence",
            "uses_raw_rho_diff": True,
        })
        self.training_metrics = dict(payload.get("training_metrics") or {})
        scaler = payload.get("scaler") or {}
        self.scaler_mean = np.asarray(scaler.get("mean", []), dtype=float)
        self.scaler_scale = np.asarray(scaler.get("scale", []), dtype=float)
        self.scaler_scale[self.scaler_scale == 0.0] = 1.0

    @classmethod
    def load(cls, path: str) -> "RhoPairCovarianceDiagnosisModel":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def predict(self, rho_measured: np.ndarray, rho_target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if rho_target is None:
            rho_target = default_w_target(self.num_qubits)
        feature_builder = (
            rho_pair_legacy_covariance_features
            if _uses_legacy_covariance_features(self.feature_config)
            else rho_pair_covariance_features
        )
        features = feature_builder(rho_target, rho_measured, self.num_qubits)
        scaled = self._scale(features)
        mode_probs = _predict_classifier_payload(
            self.payload["mode_classifier"],
            scaled,
            len(self.mode_labels),
        )
        error_probs = _predict_classifier_payload(
            self.payload["error_classifier"],
            scaled,
            len(self.error_labels),
        )
        metric = self.payload.get("metric_regressor") or {}
        coef = np.asarray(metric.get("coef", [[0.0], [0.0]]), dtype=float)
        intercept = np.asarray(metric.get("intercept", [0.0, 0.0]), dtype=float)
        estimates = np.clip(coef @ scaled + intercept, 0.0, 1.0)
        return {
            "available": True,
            "backend": "covariance",
            "fidelity_estimate": float(estimates[0]) if estimates.size else 0.0,
            "purity_estimate": float(estimates[1]) if estimates.size > 1 else 0.0,
            "mode_probabilities": {
                label: float(prob)
                for label, prob in zip(self.mode_labels, mode_probs)
            },
            "error_source_probabilities": {
                label: float(prob)
                for label, prob in zip(self.error_labels, error_probs)
            },
            "unknown_probability": _unknown_probability(mode_probs, error_probs),
            "covariance_feature_vector": [float(value) for value in scaled],
            "target_fidelity_proxy": density_fidelity_pure_target(rho_target, rho_measured),
            "tomographic_distance_features": [
                float(value)
                for value in rho_pair_distance_features(rho_target, rho_measured)
            ],
        }

    def _scale(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if self.scaler_mean.size != values.size:
            return values
        return (values - self.scaler_mean) / self.scaler_scale


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0])
    return np.asarray(matrix[idx], dtype=float)


def _single_excitation_legacy_features(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    single = [2 ** (int(num_qubits) - 1 - i) for i in range(int(num_qubits))]
    sub = np.asarray(rho, dtype=complex)[np.ix_(single, single)]
    pieces = [
        np.real(sub).reshape(-1),
        np.imag(sub).reshape(-1),
        np.abs(sub).reshape(-1),
        np.angle(sub + 1e-12).reshape(-1),
        np.linalg.eigvalsh(0.5 * (np.real(sub) + np.real(sub).T)),
    ]
    return np.concatenate(pieces).astype(np.float64)


def _uses_legacy_covariance_features(feature_config: Dict[str, Any]) -> bool:
    if bool(feature_config.get("uses_raw_rho_diff")):
        return True
    value = str(feature_config.get("input", "")).lower()
    return value == "pauli_covariance_matrix_and_single_excitation_coherence" or "rho_diff" in value


def _train_val_indices(count: int, fraction: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count)
    rng.shuffle(indices)
    val_count = max(1, int(round(count * float(fraction)))) if count > 4 else 1
    train_count = max(1, count - val_count)
    if train_count + val_count > count:
        val_count = count - train_count
    return indices[:train_count], indices[train_count:train_count + val_count]


def _fit_classifier(x: np.ndarray, y: np.ndarray, cfg: CovarianceTrainingConfig, class_count: int):
    unique = np.unique(y)
    if unique.size < 2:
        return {
            "kind": "constant",
            "class": int(unique[0]) if unique.size else 0,
            "class_count": int(class_count),
        }
    classifier = LogisticRegression(
        C=float(cfg.regularization_c),
        max_iter=int(cfg.max_iter),
        class_weight="balanced",
        solver="lbfgs",
    )
    classifier.fit(x, y)
    return classifier


def _classifier_to_payload(classifier) -> Dict[str, Any]:
    if isinstance(classifier, dict):
        return dict(classifier)
    return {
        "kind": "logistic_regression",
        "classes": np.asarray(classifier.classes_, dtype=int).tolist(),
        "coef": np.asarray(classifier.coef_, dtype=float).tolist(),
        "intercept": np.asarray(classifier.intercept_, dtype=float).tolist(),
    }


def _predict_classifier_payload(payload: Dict[str, Any], x_scaled: np.ndarray, class_count: int) -> np.ndarray:
    if payload.get("kind") == "constant":
        probs = np.zeros(class_count, dtype=float)
        probs[int(payload.get("class", 0)) % class_count] = 1.0
        return probs
    classes = np.asarray(payload.get("classes", []), dtype=int)
    coef = np.asarray(payload.get("coef", []), dtype=float)
    intercept = np.asarray(payload.get("intercept", []), dtype=float)
    if classes.size == 0 or coef.size == 0:
        return np.ones(class_count, dtype=float) / max(class_count, 1)
    logits = coef @ x_scaled + intercept
    if coef.shape[0] == 1 and classes.size == 2:
        p1 = 1.0 / (1.0 + np.exp(-float(logits[0])))
        class_probs = np.asarray([1.0 - p1, p1], dtype=float)
    else:
        class_probs = _softmax(logits)
    probs = np.zeros(class_count, dtype=float)
    for cls, prob in zip(classes, class_probs):
        if 0 <= int(cls) < class_count:
            probs[int(cls)] = float(prob)
    total = float(np.sum(probs))
    if total <= 0.0:
        return np.ones(class_count, dtype=float) / max(class_count, 1)
    return probs / total


def _evaluate_arrays(
    x_scaled: np.ndarray,
    y_mode: np.ndarray,
    y_error: np.ndarray,
    y_metrics: np.ndarray,
    mode_classifier,
    error_classifier,
    regressor,
    mode_count: int,
    error_count: int,
) -> Dict[str, float]:
    mode_payload = _classifier_to_payload(mode_classifier)
    error_payload = _classifier_to_payload(error_classifier)
    mode_correct = 0
    error_correct = 0
    metric_abs = np.zeros(2, dtype=float)
    for row, mode, error, metrics in zip(x_scaled, y_mode, y_error, y_metrics):
        mode_probs = _predict_classifier_payload(mode_payload, row, mode_count)
        error_probs = _predict_classifier_payload(error_payload, row, error_count)
        mode_correct += int(np.argmax(mode_probs) == int(mode))
        error_correct += int(np.argmax(error_probs) == int(error))
        pred_metrics = np.clip(np.asarray(regressor.predict(row.reshape(1, -1))[0], dtype=float), 0.0, 1.0)
        metric_abs += np.abs(pred_metrics - metrics[:2])
    total = max(len(x_scaled), 1)
    return {
        "mode_accuracy": mode_correct / total,
        "error_source_accuracy": error_correct / total,
        "mean_abs_fidelity_error": float(metric_abs[0] / total),
        "mean_abs_purity_error": float(metric_abs[1] / total),
    }


def _softmax(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return exp / max(float(np.sum(exp)), 1e-12)


def _unknown_probability(mode_probs: Sequence[float], error_probs: Sequence[float]) -> float:
    max_conf = float(min(np.max(mode_probs), np.max(error_probs)))
    return float(np.clip(1.0 - max_conf, 0.0, 1.0))
