"""Gauge-fixed hierarchical circular phase-drift identification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


def wrap_phase(value: np.ndarray | float) -> np.ndarray | float:
    wrapped = (np.asarray(value, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def circular_difference(first: np.ndarray | float, second: np.ndarray | float) -> np.ndarray | float:
    return wrap_phase(np.asarray(first) - np.asarray(second))


@dataclass(frozen=True)
class PhaseDriftObservation:
    timestamp: float
    operation_id: str
    cell_id: str
    measured_phase_rad: float
    commanded_phase_rad: float = 0.0
    variance_rad2: float = 0.01
    channel: str = "W_CH1"
    frequency_hz: Optional[float] = None


@dataclass
class HierarchicalCircularDriftModel:
    kernel: str
    reference_cell: str
    time_origin: float
    time_scale: float
    coefficients: np.ndarray
    cell_columns: Dict[str, int]
    operation_columns: Dict[str, int]
    ridge: float
    cv_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    residual_rmse_rad: float = float("nan")
    training_span_s: float = 0.0

    def predict(
        self,
        timestamps: Sequence[float],
        cell_ids: Sequence[str],
        operation_ids: Sequence[str],
    ) -> np.ndarray:
        if not (len(timestamps) == len(cell_ids) == len(operation_ids)):
            raise ValueError("timestamps, cell_ids and operation_ids must have equal length")
        design = _design_matrix(
            np.asarray(timestamps, dtype=float),
            list(cell_ids),
            list(operation_ids),
            kernel=self.kernel,
            time_origin=self.time_origin,
            time_scale=self.time_scale,
            cell_columns=self.cell_columns,
            operation_columns=self.operation_columns,
        )
        return wrap_phase(design @ self.coefficients)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kernel": self.kernel,
            "reference_cell": self.reference_cell,
            "time_origin": self.time_origin,
            "time_scale": self.time_scale,
            "coefficients": self.coefficients.tolist(),
            "cell_columns": self.cell_columns,
            "operation_columns": self.operation_columns,
            "ridge": self.ridge,
            "cv_scores": self.cv_scores,
            "residual_rmse_rad": self.residual_rmse_rad,
            "training_span_s": self.training_span_s,
        }


def fit_hierarchical_phase_drift(
    observations: Sequence[PhaseDriftObservation],
    *,
    reference_cell: str,
    kernels: Sequence[str] = ("integrated_wiener", "matern32", "quasi_periodic_matern"),
    ridge: float = 1.0,
    cv_blocks: int = 5,
) -> HierarchicalCircularDriftModel:
    if len(observations) < 8:
        raise ValueError("at least eight phase observations are required")
    ordered = sorted(observations, key=lambda item: item.timestamp)
    timestamps = np.asarray([item.timestamp for item in ordered], dtype=float)
    cells = [item.cell_id for item in ordered]
    operations = [item.operation_id for item in ordered]
    raw_residual = np.asarray(
        [circular_difference(item.measured_phase_rad, item.commanded_phase_rad) for item in ordered],
        dtype=float,
    )
    targets = _unwrap_by_cell(raw_residual, cells)
    variances = np.asarray([max(item.variance_rad2, 1e-8) for item in ordered], dtype=float)
    origin = float(np.min(timestamps))
    scale = max(float(np.max(timestamps) - origin), 1.0)
    unique_cells = sorted(set(cells) - {reference_cell})
    unique_operations = sorted(set(operations))
    # Time features occupy the first columns.  Cell and operation effects are
    # partial-pooled by the ridge term; the reference cell is omitted to fix
    # the unobservable global-phase gauge.
    cell_columns = {cell: index for index, cell in enumerate(unique_cells)}
    operation_columns = {op: index for index, op in enumerate(unique_operations)}

    allowed = {"integrated_wiener", "matern32", "quasi_periodic_matern"}
    kernels = tuple(kernel for kernel in kernels if kernel in allowed)
    if not kernels:
        raise ValueError("no supported drift kernels were requested")
    cv_scores: Dict[str, Dict[str, float]] = {}
    block_ids = _contiguous_blocks(len(ordered), min(max(2, cv_blocks), len(ordered) // 2))
    for kernel in kernels:
        fold_scores = []
        for block_id in sorted(set(block_ids)):
            validation = block_ids == block_id
            training = ~validation
            design = _design_matrix(
                timestamps,
                cells,
                operations,
                kernel=kernel,
                time_origin=origin,
                time_scale=scale,
                cell_columns=cell_columns,
                operation_columns=operation_columns,
            )
            coefficients = _weighted_ridge(design[training], targets[training], variances[training], ridge)
            error = circular_difference(design[validation] @ coefficients, targets[validation])
            fold_scores.append(float(np.sqrt(np.mean(np.square(error)))))
        cv_scores[kernel] = {
            "mean_rmse_rad": float(np.mean(fold_scores)),
            "se_rmse_rad": float(np.std(fold_scores, ddof=1) / math.sqrt(len(fold_scores)))
            if len(fold_scores) > 1
            else 0.0,
        }

    best = min(kernels, key=lambda name: cv_scores[name]["mean_rmse_rad"])
    threshold = cv_scores[best]["mean_rmse_rad"] + cv_scores[best]["se_rmse_rad"]
    complexity_order = ("integrated_wiener", "matern32", "quasi_periodic_matern")
    eligible = [name for name in complexity_order if name in kernels and cv_scores[name]["mean_rmse_rad"] <= threshold]
    selected = eligible[0] if eligible else best
    design = _design_matrix(
        timestamps,
        cells,
        operations,
        kernel=selected,
        time_origin=origin,
        time_scale=scale,
        cell_columns=cell_columns,
        operation_columns=operation_columns,
    )
    coefficients = _weighted_ridge(design, targets, variances, ridge)
    residual = circular_difference(design @ coefficients, targets)
    return HierarchicalCircularDriftModel(
        kernel=selected,
        reference_cell=reference_cell,
        time_origin=origin,
        time_scale=scale,
        coefficients=coefficients,
        cell_columns=cell_columns,
        operation_columns=operation_columns,
        ridge=float(ridge),
        cv_scores=cv_scores,
        residual_rmse_rad=float(np.sqrt(np.mean(np.square(residual)))),
        training_span_s=scale,
    )


def build_sparse_crosstalk_edges(
    cells: Sequence[Mapping[str, Any]],
    *,
    max_frequency_distance_hz: float,
) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for index, first in enumerate(cells):
        for second in cells[index + 1 :]:
            if first.get("channel") != second.get("channel"):
                continue
            first_frequency = first.get("frequency_hz")
            second_frequency = second.get("frequency_hz")
            if first_frequency is None or second_frequency is None:
                continue
            if abs(float(first_frequency) - float(second_frequency)) <= max_frequency_distance_hz:
                edges.append((str(first["cell_id"]), str(second["cell_id"])))
    return edges


def phase_identifiability_report(
    observations: Sequence[PhaseDriftObservation],
    model: HierarchicalCircularDriftModel,
    *,
    required_cell_observations: int = 3,
    excitation_rank_fraction: float = 0.8,
) -> Dict[str, Any]:
    cell_counts: Dict[str, int] = {}
    operations = set()
    for item in observations:
        cell_counts[item.cell_id] = cell_counts.get(item.cell_id, 0) + 1
        operations.add(item.operation_id)
    minimum = min(cell_counts.values()) if cell_counts else 0
    cells = max(len(cell_counts), 1)
    rank_proxy = min(1.0, len(operations) / cells)
    phase_ready = (
        minimum >= required_cell_observations
        and model.training_span_s > 0
        and model.residual_rmse_rad <= 0.35
    )
    amplitude_ready = phase_ready and model.residual_rmse_rad <= 0.20 and rank_proxy >= excitation_rank_fraction
    crosstalk_ready = amplitude_ready and minimum >= 5 and model.residual_rmse_rad <= 0.15
    return {
        "phase_ready": phase_ready,
        "amplitude_ready": amplitude_ready,
        "crosstalk_ready": crosstalk_ready,
        "minimum_observations_per_cell": minimum,
        "operation_to_cell_rank_proxy": rank_proxy,
        "residual_rmse_rad": model.residual_rmse_rad,
        "training_span_s": model.training_span_s,
        "gauge_reference_cell": model.reference_cell,
    }


def _time_features(timestamps: np.ndarray, kernel: str, origin: float, scale: float) -> np.ndarray:
    x = (np.asarray(timestamps, dtype=float) - origin) / max(scale, 1.0)
    if kernel == "integrated_wiener":
        columns = [np.ones_like(x), x, x * x, x * x * x]
        for knot in (0.25, 0.5, 0.75):
            columns.append(np.maximum(x - knot, 0.0) ** 3)
        return np.column_stack(columns)
    centers = np.linspace(0.0, 1.0, 7)
    distance = np.abs(x[:, None] - centers[None, :]) / 0.25
    matern = (1.0 + np.sqrt(3.0) * distance) * np.exp(-np.sqrt(3.0) * distance)
    columns = [np.ones_like(x), x, *[matern[:, index] for index in range(matern.shape[1])]]
    if kernel == "quasi_periodic_matern":
        columns.extend([np.sin(2.0 * np.pi * x), np.cos(2.0 * np.pi * x)])
    return np.column_stack(columns)


def _design_matrix(
    timestamps: np.ndarray,
    cells: Sequence[str],
    operations: Sequence[str],
    *,
    kernel: str,
    time_origin: float,
    time_scale: float,
    cell_columns: Mapping[str, int],
    operation_columns: Mapping[str, int],
) -> np.ndarray:
    time_design = _time_features(timestamps, kernel, time_origin, time_scale)
    design = np.zeros(
        (len(timestamps), time_design.shape[1] + len(cell_columns) + len(operation_columns)),
        dtype=float,
    )
    design[:, : time_design.shape[1]] = time_design
    cell_offset = time_design.shape[1]
    operation_offset = cell_offset + len(cell_columns)
    for row, (cell, operation) in enumerate(zip(cells, operations)):
        if cell in cell_columns:
            design[row, cell_offset + cell_columns[cell]] = 1.0
        if operation in operation_columns:
            design[row, operation_offset + operation_columns[operation]] = 1.0
    return design


def _weighted_ridge(design: np.ndarray, target: np.ndarray, variances: np.ndarray, ridge: float) -> np.ndarray:
    weights = 1.0 / np.sqrt(np.clip(variances, 1e-8, None))
    weighted_design = design * weights[:, None]
    weighted_target = target * weights
    penalty = np.eye(design.shape[1], dtype=float) * float(max(ridge, 0.0))
    penalty[0, 0] = 0.0
    return np.linalg.solve(weighted_design.T @ weighted_design + penalty + np.eye(design.shape[1]) * 1e-10,
                           weighted_design.T @ weighted_target)


def _unwrap_by_cell(values: np.ndarray, cells: Sequence[str]) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    indices_by_cell: Dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        indices_by_cell.setdefault(cell, []).append(index)
    for indices in indices_by_cell.values():
        output[indices] = np.unwrap(output[indices])
    return output


def _contiguous_blocks(length: int, blocks: int) -> np.ndarray:
    return np.minimum(np.arange(length) * blocks // max(length, 1), blocks - 1)
