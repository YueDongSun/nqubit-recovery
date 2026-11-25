"""NumPy rho representation model and excitation-mode diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .features import rho_pair_features, rho_pair_legacy_diff_features, rho_single_features
from .synthetic import ERROR_SOURCES, SyntheticRhoSample
from .tomography_features import RHO_PAIR_DISTANCE_FEATURE_NAMES, rho_pair_distance_features


class QuantumStateDiagnosisModel:
    """Lightweight trainable model for rho-pair representation diagnostics."""

    def __init__(self, num_qubits: int = 4):
        if num_qubits not in {3, 4}:
            raise ValueError("num_qubits must be 3 or 4")
        self.num_qubits = int(num_qubits)
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_scale: Optional[np.ndarray] = None
        self.metric_weights: Optional[np.ndarray] = None
        self.mode_weights: Optional[np.ndarray] = None
        self.error_weights: Optional[np.ndarray] = None
        self.mode_labels = [f"mode_{i}" for i in range(1, self.num_qubits + 1)]
        self.error_labels = list(ERROR_SOURCES)
        self.feature_config: Dict[str, Any] = {
            "input": "rho_pair_tomographic_features",
            "distance_features": list(RHO_PAIR_DISTANCE_FEATURE_NAMES),
            "uses_raw_rho_diff": False,
        }

    def fit(self, samples: Iterable[SyntheticRhoSample], ridge: float = 1e-3) -> "QuantumStateDiagnosisModel":
        samples = list(samples)
        if not samples:
            raise ValueError("At least one sample is required")

        x = np.vstack([rho_pair_features(s.rho_target, s.rho_measured) for s in samples])
        self.feature_config = {
            "input": "rho_pair_tomographic_features",
            "distance_features": list(RHO_PAIR_DISTANCE_FEATURE_NAMES),
            "uses_raw_rho_diff": False,
        }
        self.feature_mean = np.mean(x, axis=0)
        self.feature_scale = np.std(x, axis=0) + 1e-9
        x_norm = self._normalize(x)
        x_aug = np.column_stack([x_norm, np.ones(x_norm.shape[0])])

        metrics = np.asarray([[s.fidelity, s.purity] for s in samples], dtype=float)
        mode_targets = np.zeros((len(samples), len(self.mode_labels)), dtype=float)
        error_targets = np.zeros((len(samples), len(self.error_labels)), dtype=float)
        for row, sample in enumerate(samples):
            mode_targets[row, int(sample.excitation_mode) - 1] = 1.0
            error_targets[row, self.error_labels.index(sample.error_source)] = 1.0

        self.metric_weights = _ridge_fit(x_aug, metrics, ridge)
        self.mode_weights = _ridge_fit(x_aug, mode_targets, ridge)
        self.error_weights = _ridge_fit(x_aug, error_targets, ridge)
        return self

    def predict(self, rho_measured: np.ndarray, rho_target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if rho_target is None:
            rho_target = _default_w_target(self.num_qubits)
        feature_builder = (
            rho_pair_legacy_diff_features
            if _uses_legacy_rho_diff_features(self.feature_config)
            else rho_pair_features
        )
        features = feature_builder(rho_target, rho_measured).reshape(1, -1)
        x_aug = np.column_stack([self._normalize(features), np.ones(1)])

        metrics = np.clip(x_aug @ self._require(self.metric_weights, "metric_weights"), 0.0, 1.0)[0]
        mode_probs = _softmax((x_aug @ self._require(self.mode_weights, "mode_weights"))[0])
        error_probs = _softmax((x_aug @ self._require(self.error_weights, "error_weights"))[0])
        single = rho_single_features(rho_measured)

        return {
            "fidelity_estimate": float(metrics[0]),
            "purity_estimate": float(metrics[1]),
            "mode_probabilities": {
                label: float(prob)
                for label, prob in zip(self.mode_labels, mode_probs)
            },
            "error_source_probabilities": {
                label: float(prob)
                for label, prob in zip(self.error_labels, error_probs)
            },
            "unknown_probability": float(max(0.0, 1.0 - np.max(mode_probs))),
            "diagnostic_features": single.tolist(),
            "tomographic_distance_features": [
                float(value)
                for value in rho_pair_distance_features(rho_target, rho_measured)
            ],
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_qubits": self.num_qubits,
            "feature_mean": self._array_or_none(self.feature_mean),
            "feature_scale": self._array_or_none(self.feature_scale),
            "metric_weights": self._array_or_none(self.metric_weights),
            "mode_weights": self._array_or_none(self.mode_weights),
            "error_weights": self._array_or_none(self.error_weights),
            "mode_labels": self.mode_labels,
            "error_labels": self.error_labels,
            "feature_config": self.feature_config,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "QuantumStateDiagnosisModel":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        model = cls(int(payload["num_qubits"]))
        model.feature_mean = np.asarray(payload["feature_mean"], dtype=float)
        model.feature_scale = np.asarray(payload["feature_scale"], dtype=float)
        model.metric_weights = np.asarray(payload["metric_weights"], dtype=float)
        model.mode_weights = np.asarray(payload["mode_weights"], dtype=float)
        model.error_weights = np.asarray(payload["error_weights"], dtype=float)
        model.mode_labels = list(payload.get("mode_labels", model.mode_labels))
        model.error_labels = list(payload.get("error_labels", model.error_labels))
        model.feature_config = dict(payload.get("feature_config") or {
            "input": "rho_pair_legacy_diff_features",
            "uses_raw_rho_diff": True,
        })
        return model

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Model is not fitted")
        return (np.asarray(x, dtype=float) - self.feature_mean) / self.feature_scale

    def _require(self, weights: Optional[np.ndarray], name: str) -> np.ndarray:
        if weights is None:
            raise RuntimeError(f"Model is missing {name}; train or load a model first")
        return weights

    def _array_or_none(self, values: Optional[np.ndarray]):
        return None if values is None else values.tolist()


def _ridge_fit(x_aug: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    reg = ridge * np.eye(x_aug.shape[1], dtype=float)
    reg[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def _softmax(logits: np.ndarray) -> np.ndarray:
    vals = np.asarray(logits, dtype=float)
    vals = vals - np.max(vals)
    exp = np.exp(vals)
    total = np.sum(exp)
    if total <= 0:
        return np.ones_like(exp) / max(exp.size, 1)
    return exp / total


def _uses_legacy_rho_diff_features(feature_config: Dict[str, Any]) -> bool:
    if bool(feature_config.get("uses_raw_rho_diff")):
        return True
    value = str(feature_config.get("input", "")).lower()
    return "legacy" in value or "rho_diff" in value


def _default_w_target(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    psi = np.zeros(dim, dtype=complex)
    for i in range(num_qubits):
        psi[2 ** (num_qubits - 1 - i)] = 1.0 / np.sqrt(num_qubits)
    return np.outer(psi, psi.conj())
