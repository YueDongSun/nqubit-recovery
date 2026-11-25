"""Epistemic ensemble and real-group conformal gates for V8 feedback.

Aleatoric NB2/utility scale and model-to-model disagreement are deliberately
kept separate.  Conformal calibration accepts only authoritative paired real
episodes and aggregates at episode level, never at operation-row level.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


V8_ENSEMBLE_SCHEMA = "v8_feedback_ensemble_prediction_v1"
V8_CONFORMAL_SCHEMA = "v8_feedback_grouped_conformal_v1"
REAL_PAIRED_EVIDENCE_SOURCE = "real_paired_episode"


@dataclass(frozen=True)
class V8ConfidenceConfig:
    epistemic_penalty: float = 1.0
    harm_penalty: float = 0.25
    harm_upper_z: float = 1.645
    conformal_alpha: float = 0.10
    minimum_calibration_groups: int = 20
    require_real_paired_episode: bool = True

    def validate(self) -> None:
        if self.epistemic_penalty < 0.0 or self.harm_penalty < 0.0:
            raise ValueError("confidence penalties must be non-negative")
        if self.harm_upper_z < 0.0:
            raise ValueError("harm upper-bound multiplier must be non-negative")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError("conformal_alpha must lie in (0, 1)")
        if self.minimum_calibration_groups < 2:
            raise ValueError("minimum_calibration_groups must be at least two")


@dataclass(frozen=True)
class V8EnsemblePrediction:
    utility_mean: np.ndarray
    aleatoric_variance: np.ndarray
    epistemic_variance: np.ndarray
    total_variance: np.ndarray
    harm_mean: np.ndarray
    harm_upper: np.ndarray
    pessimistic_score: np.ndarray
    selected_index: np.ndarray
    member_count: int

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": V8_ENSEMBLE_SCHEMA,
            "utility_mean": self.utility_mean.tolist(),
            "aleatoric_variance": self.aleatoric_variance.tolist(),
            "epistemic_variance": self.epistemic_variance.tolist(),
            "total_variance": self.total_variance.tolist(),
            "harm_mean": self.harm_mean.tolist(),
            "harm_upper": self.harm_upper.tolist(),
            "pessimistic_score": self.pessimistic_score.tolist(),
            "selected_index": self.selected_index.tolist(),
            "member_count": int(self.member_count),
        }


def aggregate_ensemble_predictions(
    utility_mean: np.ndarray,
    utility_log_scale: np.ndarray,
    harmful_probability: np.ndarray,
    *,
    config: V8ConfidenceConfig | None = None,
    candidate_mask: Optional[np.ndarray] = None,
) -> V8EnsemblePrediction:
    """Aggregate `[member, batch, candidate]` predictions pessimistically."""

    config = config or V8ConfidenceConfig()
    config.validate()
    mean = np.asarray(utility_mean, dtype=np.float64)
    log_scale = np.asarray(utility_log_scale, dtype=np.float64)
    harm = np.asarray(harmful_probability, dtype=np.float64)
    if mean.ndim != 3 or log_scale.shape != mean.shape or harm.shape != mean.shape:
        raise ValueError("ensemble inputs must share [member, batch, candidate] shape")
    if mean.shape[0] < 2:
        raise ValueError("epistemic ensemble requires at least two independent members")
    if not (
        np.all(np.isfinite(mean))
        and np.all(np.isfinite(log_scale))
        and np.all(np.isfinite(harm))
    ):
        raise ValueError("ensemble predictions contain non-finite values")
    if np.any((harm < 0.0) | (harm > 1.0)):
        raise ValueError("harmful probabilities must lie in [0, 1]")
    ensemble_mean = np.mean(mean, axis=0)
    aleatoric = np.mean(np.exp(2.0 * np.clip(log_scale, -10.0, 5.0)), axis=0)
    epistemic = np.var(mean, axis=0, ddof=1)
    total = aleatoric + epistemic
    harm_mean = np.mean(harm, axis=0)
    harm_epistemic_std = np.std(harm, axis=0, ddof=1)
    harm_upper = np.clip(
        harm_mean + float(config.harm_upper_z) * harm_epistemic_std,
        0.0,
        1.0,
    )
    score = (
        ensemble_mean
        - float(config.epistemic_penalty) * np.sqrt(np.maximum(epistemic, 0.0))
        - float(config.harm_penalty) * harm_upper
    )
    if candidate_mask is not None:
        mask = np.asarray(candidate_mask, dtype=bool)
        if mask.shape != score.shape:
            raise ValueError("candidate_mask shape changed")
        score = np.where(mask, score, -np.inf)
        if np.any(~np.any(mask, axis=1)):
            raise ValueError("each batch row needs at least one permitted candidate")
    selected = np.argmax(score, axis=1)
    return V8EnsemblePrediction(
        utility_mean=ensemble_mean.astype(np.float32),
        aleatoric_variance=aleatoric.astype(np.float32),
        epistemic_variance=epistemic.astype(np.float32),
        total_variance=total.astype(np.float32),
        harm_mean=harm_mean.astype(np.float32),
        harm_upper=harm_upper.astype(np.float32),
        pessimistic_score=score.astype(np.float32),
        selected_index=selected.astype(np.int64),
        member_count=int(mean.shape[0]),
    )


@dataclass(frozen=True)
class V8ConformalCell:
    cell_id: str
    group_count: int
    nonconformity_quantile: float
    empirical_coverage: float


@dataclass(frozen=True)
class V8GroupedConformalCalibration:
    alpha: float
    minimum_groups: int
    cells: Mapping[str, V8ConformalCell]
    calibration_group_hash: str
    evidence_source: str
    fitted: bool
    abstain_without_cell: bool = True
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def to_json_dict(self) -> Dict[str, Any]:
        payload = {
            "schema": V8_CONFORMAL_SCHEMA,
            "alpha": float(self.alpha),
            "minimum_groups": int(self.minimum_groups),
            "cells": {
                key: asdict(value) for key, value in sorted(self.cells.items())
            },
            "calibration_group_hash": self.calibration_group_hash,
            "evidence_source": self.evidence_source,
            "fitted": bool(self.fitted),
            "abstain_without_cell": bool(self.abstain_without_cell),
            "promotion_ready": False,
            "hardware_action_authorized": False,
        }
        unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        payload["calibration_hash"] = hashlib.sha256(unsigned).hexdigest()
        return payload

    def threshold(self, cell_id: str) -> Optional[float]:
        cell = self.cells.get(str(cell_id))
        if cell is None or cell.group_count < self.minimum_groups:
            return None
        return float(cell.nonconformity_quantile)

    def action_allowed(self, cell_id: str, nonconformity: float) -> bool:
        threshold = self.threshold(cell_id)
        if threshold is None or not np.isfinite(float(nonconformity)):
            return False
        return float(nonconformity) <= threshold


def fit_grouped_conformal(
    *,
    cell_ids: Sequence[str],
    group_ids: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
    target: Sequence[float],
    evidence_source: str,
    config: V8ConfidenceConfig | None = None,
) -> V8GroupedConformalCalibration:
    """Fit CQR expansion by independent episode group.

    Multiple rows from one episode are collapsed to the maximum
    nonconformity, preventing operation/subwindow pseudo-replication.
    """

    config = config or V8ConfidenceConfig()
    config.validate()
    source = str(evidence_source)
    if config.require_real_paired_episode and source != REAL_PAIRED_EVIDENCE_SOURCE:
        raise ValueError("V8 conformal calibration requires paired real episodes")
    cell = np.asarray(cell_ids, dtype=str)
    group = np.asarray(group_ids, dtype=str)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    value = np.asarray(target, dtype=np.float64)
    if not (cell.shape == group.shape == lo.shape == hi.shape == value.shape):
        raise ValueError("conformal calibration arrays must share one-dimensional shape")
    if cell.ndim != 1 or cell.size == 0:
        raise ValueError("conformal calibration requires at least one row")
    if not np.all(np.isfinite(lo) & np.isfinite(hi) & np.isfinite(value)):
        raise ValueError("conformal calibration contains non-finite values")
    if np.any(lo > hi):
        raise ValueError("conformal lower bound exceeds upper bound")
    nonconformity = np.maximum(lo - value, value - hi)
    grouped: Dict[Tuple[str, str], float] = {}
    for cell_id, group_id, score in zip(cell, group, nonconformity):
        key = (str(cell_id), str(group_id))
        grouped[key] = max(grouped.get(key, -math.inf), float(score))
    cells: Dict[str, V8ConformalCell] = {}
    for cell_id in sorted(set(cell.tolist())):
        scores = np.asarray(
            [score for (candidate_cell, _), score in grouped.items() if candidate_cell == cell_id],
            dtype=np.float64,
        )
        group_count = int(scores.size)
        if group_count < int(config.minimum_calibration_groups):
            continue
        rank = int(math.ceil((group_count + 1) * (1.0 - config.conformal_alpha)))
        rank = min(max(rank, 1), group_count)
        quantile = float(np.partition(scores, rank - 1)[rank - 1])
        empirical_coverage = float(np.mean(scores <= quantile))
        cells[cell_id] = V8ConformalCell(
            cell_id=cell_id,
            group_count=group_count,
            nonconformity_quantile=quantile,
            empirical_coverage=empirical_coverage,
        )
    group_payload = sorted((key[0], key[1], value) for key, value in grouped.items())
    group_hash = hashlib.sha256(
        json.dumps(group_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return V8GroupedConformalCalibration(
        alpha=float(config.conformal_alpha),
        minimum_groups=int(config.minimum_calibration_groups),
        cells=cells,
        calibration_group_hash=group_hash,
        evidence_source=source,
        fitted=bool(cells),
    )


def empty_real_conformal_gate(
    *,
    config: V8ConfidenceConfig | None = None,
) -> V8GroupedConformalCalibration:
    """Return the required fail-closed state before paired episodes exist."""

    config = config or V8ConfidenceConfig()
    config.validate()
    return V8GroupedConformalCalibration(
        alpha=float(config.conformal_alpha),
        minimum_groups=int(config.minimum_calibration_groups),
        cells={},
        calibration_group_hash=hashlib.sha256(b"no-real-paired-calibration").hexdigest(),
        evidence_source=REAL_PAIRED_EVIDENCE_SOURCE,
        fitted=False,
    )


__all__ = [
    "REAL_PAIRED_EVIDENCE_SOURCE",
    "V8_CONFORMAL_SCHEMA",
    "V8_ENSEMBLE_SCHEMA",
    "V8ConfidenceConfig",
    "V8ConformalCell",
    "V8EnsemblePrediction",
    "V8GroupedConformalCalibration",
    "aggregate_ensemble_predictions",
    "empty_real_conformal_gate",
    "fit_grouped_conformal",
]
