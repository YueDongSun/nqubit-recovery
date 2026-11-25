"""Dependency-light sparse equation fitting with train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


DEFAULT_RIDGE = 1e-6
DEFAULT_STLSQ_THRESHOLD = 0.05
DEFAULT_MAXIMUM_ITERATIONS = 50


@dataclass(frozen=True)
class SparseFitResult:
    coefficients: np.ndarray
    intercept: float
    selected_mask: np.ndarray
    condition_number: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    normalized_coefficients: np.ndarray
    method: str
    iterations: int
    support_stability: Optional[np.ndarray] = None

    def predict(self, features: Any) -> np.ndarray:
        return np.asarray(features, dtype=float) @ self.coefficients + self.intercept


class SparseEquationFitter:
    """Fit ridge, STLSQ, group-lasso, or elastic-net linear dictionaries."""

    def __init__(
        self,
        *,
        method: str = "stlsq",
        ridge: float = DEFAULT_RIDGE,
        threshold: float = DEFAULT_STLSQ_THRESHOLD,
        l1_weight: float = 0.0,
        group_weight: float = 0.0,
        maximum_iterations: int = DEFAULT_MAXIMUM_ITERATIONS,
    ) -> None:
        if method not in {"ridge", "stlsq", "group_lasso", "elastic_net"}:
            raise ValueError(f"unsupported sparse-fit method: {method}")
        self.method = method
        self.ridge = float(max(ridge, 0.0))
        self.threshold = float(max(threshold, 0.0))
        self.l1_weight = float(max(l1_weight, 0.0))
        self.group_weight = float(max(group_weight, 0.0))
        self.maximum_iterations = int(max(maximum_iterations, 1))

    def fit(
        self,
        features: Any,
        target: Any,
        *,
        sample_weight: Optional[Any] = None,
        groups: Optional[Sequence[str | int]] = None,
    ) -> SparseFitResult:
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.size:
            raise ValueError("features must be [samples, terms] and match target")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("fit inputs must be finite")
        weight = np.ones(y.size) if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
        if weight.size != y.size or np.any(weight < 0.0):
            raise ValueError("sample weights must be non-negative and match target")
        weight_sum = float(np.sum(weight))
        if weight_sum <= 0.0:
            raise ValueError("at least one sample must have positive weight")
        mean = np.sum(x * weight[:, None], axis=0) / weight_sum
        centered = x - mean
        variance = np.sum(np.square(centered) * weight[:, None], axis=0) / weight_sum
        scale = np.sqrt(np.maximum(variance, 1e-24))
        constant = scale <= 1e-10
        scale[constant] = 1.0
        z = centered / scale
        y_mean = float(np.sum(y * weight) / weight_sum)
        centered_y = y - y_mean
        root_weight = np.sqrt(weight)
        zw = z * root_weight[:, None]
        yw = centered_y * root_weight
        condition = float(np.linalg.cond(zw)) if zw.size else float("inf")
        group_values = tuple(groups or tuple(range(x.shape[1])))
        if len(group_values) != x.shape[1]:
            raise ValueError("groups must have one entry per feature")
        coefficient, iterations = self._solve(zw, yw, group_values)
        coefficient[constant] = 0.0
        unscaled = coefficient / scale
        intercept = y_mean - float(mean @ unscaled)
        selected = np.abs(unscaled) > 0.0
        return SparseFitResult(
            coefficients=unscaled,
            intercept=intercept,
            selected_mask=selected,
            condition_number=condition,
            feature_mean=mean,
            feature_scale=scale,
            normalized_coefficients=coefficient,
            method=self.method,
            iterations=iterations,
        )

    def _ridge_solution(self, x: np.ndarray, y: np.ndarray, active: np.ndarray) -> np.ndarray:
        coefficient = np.zeros(x.shape[1], dtype=float)
        if not np.any(active):
            return coefficient
        selected = x[:, active]
        gram = selected.T @ selected + self.ridge * np.eye(selected.shape[1])
        coefficient[active] = np.linalg.solve(gram, selected.T @ y)
        return coefficient

    def _solve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: Sequence[str | int],
    ) -> tuple[np.ndarray, int]:
        coefficient = self._ridge_solution(x, y, np.ones(x.shape[1], dtype=bool))
        if self.method == "ridge":
            return coefficient, 1
        if self.method == "stlsq":
            active = np.ones(x.shape[1], dtype=bool)
            for iteration in range(1, self.maximum_iterations + 1):
                next_active = np.abs(coefficient) >= self.threshold
                if np.array_equal(active, next_active):
                    return coefficient, iteration
                active = next_active
                coefficient = self._ridge_solution(x, y, active)
            return coefficient, self.maximum_iterations

        lipschitz = float(np.linalg.norm(x, ord=2) ** 2 + self.ridge)
        step = 1.0 / max(lipschitz, 1e-12)
        group_index: Dict[str | int, np.ndarray] = {
            group: np.asarray([value == group for value in groups], dtype=bool)
            for group in dict.fromkeys(groups)
        }
        for iteration in range(1, self.maximum_iterations + 1):
            previous = coefficient.copy()
            gradient = x.T @ (x @ coefficient - y) + self.ridge * coefficient
            coefficient -= step * gradient
            l1 = self.l1_weight if self.method == "elastic_net" else 0.0
            if l1 > 0.0:
                coefficient = np.sign(coefficient) * np.maximum(np.abs(coefficient) - step * l1, 0.0)
            group_penalty = self.group_weight if self.group_weight > 0.0 else self.threshold
            for mask in group_index.values():
                norm = float(np.linalg.norm(coefficient[mask]))
                shrink = max(0.0, 1.0 - step * group_penalty / max(norm, 1e-12))
                coefficient[mask] *= shrink
            if np.linalg.norm(coefficient - previous) <= 1e-9 * (1.0 + np.linalg.norm(previous)):
                return coefficient, iteration
        return coefficient, self.maximum_iterations


__all__ = ["SparseEquationFitter", "SparseFitResult"]
