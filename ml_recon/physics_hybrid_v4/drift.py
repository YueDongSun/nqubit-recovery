"""Blocked-validation circular drift models with temperature controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


DRIFT_CANDIDATES = ("integrated_wiener", "matern_3_2", "quasi_periodic_matern")
TEMPERATURE_MODES = ("time_only", "temperature_only", "time_plus_temperature", "time_plus_shuffled_temperature")


@dataclass
class DriftFitResult:
    candidate: str
    temperature_mode: str
    length_scale: float
    regularization: float
    period: float
    train_time: np.ndarray
    train_temperature: np.ndarray
    alpha: np.ndarray
    temperature_coefficients: np.ndarray
    reference_cell: int
    validation_mean_loss: float
    validation_standard_error: float
    candidate_scores: Dict[str, Dict[str, float]]

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("train_time", "train_temperature", "alpha", "temperature_coefficients"):
            payload[key] = np.asarray(payload[key]).tolist()
        return payload

    def predict(
        self,
        time_s: Sequence[float],
        temperature_c: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        query_time = np.asarray(time_s, dtype=float).reshape(-1)
        query_temperature = _temperature_array(temperature_c, query_time.size)
        kernel = drift_kernel(
            self.candidate,
            query_time,
            self.train_time,
            length_scale=self.length_scale,
            period=self.period,
        )
        prediction = kernel @ self.alpha
        design = _temperature_design(query_time, query_temperature, self.temperature_mode)
        prediction += design @ self.temperature_coefficients
        prediction -= prediction[:, [self.reference_cell]]
        return np.angle(np.exp(1j * prediction))


def fit_drift_candidates(
    time_s: Sequence[float],
    phase_rad: np.ndarray,
    block_id: Sequence[int],
    *,
    temperature_c: Optional[Sequence[float]] = None,
    reference_cell: int = 0,
    candidates: Iterable[str] = DRIFT_CANDIDATES,
    temperature_modes: Iterable[str] = TEMPERATURE_MODES,
    length_scales: Sequence[float] = (300.0, 1800.0, 7200.0),
    regularizations: Sequence[float] = (1e-4, 1e-3, 1e-2),
    period_s: float = 86400.0,
    seed: int = 0,
) -> DriftFitResult:
    """Select the simplest candidate within one standard error of the best."""

    time = np.asarray(time_s, dtype=float).reshape(-1)
    values = np.asarray(phase_rad, dtype=float)
    blocks = np.asarray(block_id, dtype=int).reshape(-1)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != time.size or blocks.size != time.size:
        raise ValueError("time, phase, and block_id lengths must match")
    if not 0 <= int(reference_cell) < values.shape[1]:
        raise ValueError("reference_cell is out of range")
    temperature = _temperature_array(temperature_c, time.size)
    gauge_free = np.angle(np.exp(1j * (values - values[:, [reference_cell]])))
    candidate_values = tuple(str(value) for value in candidates)
    mode_values = tuple(str(value) for value in temperature_modes)
    if not set(candidate_values).issubset(DRIFT_CANDIDATES):
        raise ValueError("unknown drift candidate")
    if not set(mode_values).issubset(TEMPERATURE_MODES):
        raise ValueError("unknown temperature mode")
    shuffled = np.random.default_rng(seed).permutation(temperature)
    rows = []
    unique_blocks = np.unique(blocks)
    if unique_blocks.size < 2:
        raise ValueError("blocked validation requires at least two blocks")
    for candidate in candidate_values:
        for mode in mode_values:
            used_temperature = shuffled if mode == "time_plus_shuffled_temperature" else temperature
            for length_scale in length_scales:
                for regularization in regularizations:
                    losses = []
                    for held_out in unique_blocks:
                        train = blocks != held_out
                        validation = ~train
                        alpha, coefficients = _fit_one(
                            candidate,
                            time[train],
                            gauge_free[train],
                            used_temperature[train],
                            mode,
                            float(length_scale),
                            float(regularization),
                            float(period_s),
                        )
                        kernel = drift_kernel(
                            candidate,
                            time[validation],
                            time[train],
                            length_scale=float(length_scale),
                            period=float(period_s),
                        )
                        prediction = kernel @ alpha
                        prediction += _temperature_design(
                            time[validation], used_temperature[validation], mode
                        ) @ coefficients
                        circular_error = np.angle(np.exp(1j * (prediction - gauge_free[validation])))
                        losses.append(float(np.mean(np.square(circular_error))))
                    rows.append({
                        "candidate": candidate,
                        "temperature_mode": mode,
                        "length_scale": float(length_scale),
                        "regularization": float(regularization),
                        "mean": float(np.mean(losses)),
                        "standard_error": float(np.std(losses, ddof=1) / np.sqrt(len(losses))) if len(losses) > 1 else 0.0,
                    })
    best = min(rows, key=lambda row: row["mean"])
    threshold = best["mean"] + best["standard_error"]
    complexity = {name: index for index, name in enumerate(DRIFT_CANDIDATES)}
    mode_complexity = {"time_only": 0, "temperature_only": 1, "time_plus_temperature": 2, "time_plus_shuffled_temperature": 3}
    eligible = [row for row in rows if row["mean"] <= threshold]
    selected = min(
        eligible,
        key=lambda row: (
            complexity[row["candidate"]],
            mode_complexity[row["temperature_mode"]],
            row["length_scale"],
            row["regularization"],
        ),
    )
    selected_temperature = shuffled if selected["temperature_mode"] == "time_plus_shuffled_temperature" else temperature
    alpha, coefficients = _fit_one(
        selected["candidate"], time, gauge_free, selected_temperature,
        selected["temperature_mode"], selected["length_scale"], selected["regularization"], float(period_s),
    )
    score_map = {
        f"{row['candidate']}:{row['temperature_mode']}:{row['length_scale']}:{row['regularization']}": {
            "mean_loss": row["mean"], "standard_error": row["standard_error"],
        }
        for row in rows
    }
    return DriftFitResult(
        candidate=selected["candidate"],
        temperature_mode=selected["temperature_mode"],
        length_scale=selected["length_scale"],
        regularization=selected["regularization"],
        period=float(period_s),
        train_time=time,
        train_temperature=selected_temperature,
        alpha=alpha,
        temperature_coefficients=coefficients,
        reference_cell=int(reference_cell),
        validation_mean_loss=selected["mean"],
        validation_standard_error=selected["standard_error"],
        candidate_scores=score_map,
    )


def drift_kernel(
    candidate: str,
    left_time: Sequence[float],
    right_time: Sequence[float],
    *,
    length_scale: float,
    period: float,
) -> np.ndarray:
    left = np.asarray(left_time, dtype=float).reshape(-1)
    right = np.asarray(right_time, dtype=float).reshape(-1)
    scale = max(float(length_scale), 1e-9)
    if candidate == "integrated_wiener":
        origin = min(float(np.min(left)), float(np.min(right)))
        return np.minimum(left[:, None] - origin + 1.0, right[None, :] - origin + 1.0) / scale
    distance = np.abs(left[:, None] - right[None, :])
    root_three = np.sqrt(3.0) * distance / scale
    matern = (1.0 + root_three) * np.exp(-root_three)
    if candidate == "matern_3_2":
        return matern
    if candidate == "quasi_periodic_matern":
        periodic_distance = np.sin(np.pi * distance / max(float(period), 1e-9))
        periodic = np.exp(-2.0 * np.square(periodic_distance))
        return matern * periodic
    raise ValueError(f"unknown drift candidate {candidate!r}")


def _fit_one(candidate, time, values, temperature, mode, length_scale, regularization, period):
    design = _temperature_design(time, temperature, mode)
    if design.shape[1]:
        coefficients = np.linalg.lstsq(design, values, rcond=1e-6)[0]
        residual = values - design @ coefficients
    else:
        coefficients = np.zeros((0, values.shape[1]), dtype=float)
        residual = values
    kernel = drift_kernel(candidate, time, time, length_scale=length_scale, period=period)
    alpha = np.linalg.solve(kernel + regularization * np.eye(kernel.shape[0]), residual)
    return alpha, coefficients


def _temperature_design(time, temperature, mode):
    time_value = np.asarray(time, dtype=float).reshape(-1)
    temperature_value = np.asarray(temperature, dtype=float).reshape(-1)
    if mode == "time_only":
        return np.zeros((time_value.size, 0), dtype=float)
    derivative = np.gradient(temperature_value, time_value, edge_order=1) if time_value.size > 1 else np.zeros_like(temperature_value)
    centered = temperature_value - float(np.mean(temperature_value))
    if mode == "temperature_only":
        return np.stack((np.ones_like(centered), centered, derivative), axis=1)
    normalized_time = (time_value - float(np.mean(time_value))) / max(float(np.std(time_value)), 1.0)
    return np.stack((np.ones_like(centered), normalized_time, centered, derivative), axis=1)


def _temperature_array(value, length):
    if value is None:
        return np.zeros(int(length), dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != int(length):
        raise ValueError("temperature length must match time")
    return array
