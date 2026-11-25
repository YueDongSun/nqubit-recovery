"""Leakage-controlled GSI mixture estimate and research-only coin-noise subtraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


NOISE_MODEL_SCHEMA = "v7_gsi_noise_subtraction_v1"


@dataclass(frozen=True)
class V7GsiNoiseModelV1:
    system_type: str
    operation_accidental_floor: np.ndarray
    coin_total_per_singles_geomean: float
    gsi_estimate_raw: float
    gsi_estimate: float
    gsi_lower_bound: float
    gsi_upper_bound: float
    ideal_gsi: float
    noise_fraction_prior: float
    confidence: float
    training_contrast_median: float
    training_contrast_mad: float
    multi_log_center: float
    multi_log_scale: float
    count_smoothing: float
    maximum_subtraction_fraction: float
    minimum_noise_fraction: float
    maximum_noise_fraction: float
    training_sample_count: int
    research_only: bool = True
    heuristic_only: bool = True
    production_subtraction_allowed: bool = False
    hardware_action_authorized: bool = False

    def predict_background_counts(
        self,
        signal_counts: Any,
        idler_counts: Any,
        *,
        multi_click_counts: Optional[Any] = None,
        subtraction_strength: float = 1.0,
        multi_click_weight: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        signal = np.asarray(signal_counts, dtype=float)
        idler = np.asarray(idler_counts, dtype=float)
        if signal.ndim != 2 or signal.shape != idler.shape:
            raise ValueError("signal/idler counts must be same-shaped two-dimensional arrays")
        if signal.shape[1] != self.operation_accidental_floor.size:
            raise ValueError("signal/idler operation count disagrees with the noise model")
        if np.any(signal < 0.0) or np.any(idler < 0.0):
            raise ValueError("signal/idler counts must be non-negative")
        strength = float(subtraction_strength)
        if not 0.0 <= strength <= 1.5:
            raise ValueError("subtraction_strength must lie in [0, 1.5]")
        accidental = (signal + self.count_smoothing) * (idler + self.count_smoothing)
        accidental /= np.maximum(accidental.sum(axis=1, keepdims=True), 1e-12)
        background_shape = accidental * self.operation_accidental_floor
        background_shape /= np.maximum(background_shape.sum(axis=1, keepdims=True), 1e-12)
        singles_geomean = np.sqrt(np.maximum(
            signal.sum(axis=1) * idler.sum(axis=1), 1.0,
        ))
        predicted_coin_total = self.coin_total_per_singles_geomean * singles_geomean
        multi_z_score = np.zeros(signal.shape[0], dtype=float)
        if multi_click_counts is not None:
            multi = np.asarray(multi_click_counts, dtype=float)
            if multi.ndim != 2 or multi.shape[0] != signal.shape[0]:
                raise ValueError("multi_click_counts must have one row per sample")
            multi_total = np.maximum(multi, 0.0).sum(axis=1)
            multi_log_rate = np.log((multi_total + 0.5) / (singles_geomean + 1.0))
            multi_z_score = np.clip(
                (multi_log_rate - self.multi_log_center) / self.multi_log_scale,
                -3.0,
                3.0,
            )
        sample_noise_fraction = np.clip(
            self.noise_fraction_prior * np.exp(float(multi_click_weight) * multi_z_score),
            self.minimum_noise_fraction,
            self.maximum_noise_fraction,
        )
        background_total = strength * sample_noise_fraction * predicted_coin_total
        background = background_total[:, None] * background_shape
        return background, {
            "predicted_coin_total": predicted_coin_total,
            "sample_noise_fraction": sample_noise_fraction,
            "multi_z_score": multi_z_score,
            "background_total": background_total,
        }

    def subtract_noise(
        self,
        coin_counts: Any,
        signal_counts: Any,
        idler_counts: Any,
        *,
        multi_click_counts: Optional[Any] = None,
        subtraction_strength: float = 1.0,
        multi_click_weight: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        coin = np.asarray(coin_counts, dtype=float)
        background, diagnostics = self.predict_background_counts(
            signal_counts,
            idler_counts,
            multi_click_counts=multi_click_counts,
            subtraction_strength=subtraction_strength,
            multi_click_weight=multi_click_weight,
        )
        if coin.shape != background.shape or np.any(coin < 0.0):
            raise ValueError("coin counts must be non-negative and match the background shape")
        maximum_removal = float(self.maximum_subtraction_fraction) * coin
        removed = np.minimum(background, maximum_removal)
        corrected = np.maximum(coin - removed, 0.0)
        raw_total = coin.sum(axis=1)
        removed_total = removed.sum(axis=1)
        report = {
            "schema": "v7_gsi_noise_subtraction_batch_v1",
            "subtraction_strength": float(subtraction_strength),
            "multi_click_weight": float(multi_click_weight),
            "mean_removed_fraction": float(np.mean(
                removed_total / np.maximum(raw_total, 1.0)
            )),
            "median_removed_fraction": float(np.median(
                removed_total / np.maximum(raw_total, 1.0)
            )),
            "cap_active_fraction": float(np.mean(background >= maximum_removal)),
            "minimum_remaining_fraction": float(1.0 - self.maximum_subtraction_fraction),
            "background_uses_hidden_coin_counts": False,
            "global_multi_click_scope_preserved": True,
            "predicted_coin_total_ratio_median": float(np.median(
                diagnostics["predicted_coin_total"] / np.maximum(raw_total, 1.0)
            )),
        }
        return corrected, report

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["operation_accidental_floor"] = self.operation_accidental_floor.tolist()
        payload["schema"] = NOISE_MODEL_SCHEMA
        payload["noise_fraction_formula"] = (
            "clip((ideal_gsi - gsi_estimate) / (ideal_gsi - 1), 0, 1)"
        )
        payload["is_direct_per_run_gsi_measurement"] = False
        payload["gsi_interpretation"] = (
            "training-only robust interference-contrast proxy constrained by the configured GSI range"
        )
        payload["background_shape"] = (
            "training lower-envelope operation response times normalized signal*idler"
        )
        payload["inference_contract"] = (
            "test background uses signal/idler, optional global multi-click totals, and "
            "training-only parameters; no hidden coin count enters background prediction"
        )
        return payload


def fit_v7_gsi_noise_model(
    system_type: str,
    coin_counts: Any,
    signal_counts: Any,
    idler_counts: Any,
    train_mask: Any,
    *,
    multi_click_counts: Optional[Any] = None,
    gsi_lower_bound: float = 1.2,
    gsi_upper_bound: float = 1.5,
    ideal_gsi: float = 2.0,
    lower_contrast_quantile: float = 0.10,
    central_contrast_quantile: float = 0.25,
    accidental_floor_quantile: float = 0.20,
    relaxed_confidence_floor: float = 0.35,
    count_smoothing: float = 0.5,
    maximum_subtraction_fraction: float = 0.85,
    minimum_noise_fraction: float = 0.20,
    maximum_noise_fraction: float = 0.85,
) -> Tuple[V7GsiNoiseModelV1, Dict[str, Any]]:
    coin = np.asarray(coin_counts, dtype=float)
    signal = np.asarray(signal_counts, dtype=float)
    idler = np.asarray(idler_counts, dtype=float)
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    if coin.ndim != 2 or coin.shape != signal.shape or coin.shape != idler.shape:
        raise ValueError("coin/signal/idler counts must have the same two-dimensional shape")
    if train.shape != (coin.shape[0],) or not np.any(train):
        raise ValueError("train_mask must select at least one sample")
    if not 1.0 <= float(gsi_lower_bound) <= float(gsi_upper_bound) <= float(ideal_gsi):
        raise ValueError("GSI bounds must satisfy 1 <= lower <= upper <= ideal")
    if not 0.0 < float(lower_contrast_quantile) < float(central_contrast_quantile) < 0.5:
        raise ValueError("contrast quantiles must satisfy 0 < lower < central < 0.5")
    smoothing = float(count_smoothing)
    coin_share = (coin + smoothing) / (
        coin.sum(axis=1, keepdims=True) + smoothing * coin.shape[1]
    )
    accidental = (signal + smoothing) * (idler + smoothing)
    accidental_share = accidental / np.maximum(accidental.sum(axis=1, keepdims=True), 1e-12)
    lift = coin_share / np.maximum(accidental_share, 1e-12)
    operation_center = np.median(lift[train], axis=0)
    normalized_lift = lift / np.maximum(operation_center, 1e-12)
    tail_low = np.quantile(normalized_lift, lower_contrast_quantile, axis=1)
    tail_high = np.quantile(normalized_lift, 1.0 - lower_contrast_quantile, axis=1)
    central_low = np.quantile(normalized_lift, central_contrast_quantile, axis=1)
    central_high = np.quantile(normalized_lift, 1.0 - central_contrast_quantile, axis=1)
    tail_contrast = tail_high / np.maximum(tail_low, 1e-12)
    central_contrast = central_high / np.maximum(central_low, 1e-12)
    combined_contrast = np.sqrt(tail_contrast * central_contrast)
    finite_contrast = combined_contrast[train & np.isfinite(combined_contrast)]
    if finite_contrast.size == 0:
        raise ValueError("training data produced no finite GSI contrasts")
    raw_gsi = float(np.median(finite_contrast))
    effective_gsi = float(np.clip(raw_gsi, gsi_lower_bound, gsi_upper_bound))
    contrast_mad = float(np.median(np.abs(
        np.log(np.maximum(finite_contrast, 1e-12))
        - np.median(np.log(np.maximum(finite_contrast, 1e-12)))
    )))
    clipping_distance = abs(np.log(max(raw_gsi, 1e-12) / max(effective_gsi, 1e-12)))
    agreement = float(np.exp(-contrast_mad / 0.5 - clipping_distance / 0.35))
    confidence = float(np.clip(
        relaxed_confidence_floor + (1.0 - relaxed_confidence_floor) * agreement,
        relaxed_confidence_floor,
        1.0,
    ))
    noise_fraction = float(np.clip(
        (float(ideal_gsi) - effective_gsi) / max(float(ideal_gsi) - 1.0, 1e-12),
        minimum_noise_fraction,
        maximum_noise_fraction,
    ))
    operation_floor = np.quantile(lift[train], float(accidental_floor_quantile), axis=0)
    operation_floor = np.clip(
        operation_floor,
        np.quantile(operation_floor, 0.02),
        np.quantile(operation_floor, 0.98),
    )
    singles_geomean = np.sqrt(np.maximum(
        signal.sum(axis=1) * idler.sum(axis=1), 1.0,
    ))
    coin_scale = float(np.median(
        coin.sum(axis=1)[train] / singles_geomean[train]
    ))
    multi_log_rate = np.zeros(coin.shape[0], dtype=float)
    if multi_click_counts is not None:
        multi = np.asarray(multi_click_counts, dtype=float)
        if multi.ndim != 2 or multi.shape[0] != coin.shape[0]:
            raise ValueError("multi_click_counts must have one row per sample")
        multi_log_rate = np.log(
            (np.maximum(multi, 0.0).sum(axis=1) + 0.5) / (singles_geomean + 1.0)
        )
    multi_center = float(np.median(multi_log_rate[train]))
    multi_scale = float(1.4826 * np.median(np.abs(
        multi_log_rate[train] - multi_center
    )))
    multi_scale = max(multi_scale, 1e-6)
    model = V7GsiNoiseModelV1(
        system_type=str(system_type),
        operation_accidental_floor=operation_floor.astype(np.float64),
        coin_total_per_singles_geomean=coin_scale,
        gsi_estimate_raw=raw_gsi,
        gsi_estimate=effective_gsi,
        gsi_lower_bound=float(gsi_lower_bound),
        gsi_upper_bound=float(gsi_upper_bound),
        ideal_gsi=float(ideal_gsi),
        noise_fraction_prior=noise_fraction,
        confidence=confidence,
        training_contrast_median=float(np.median(finite_contrast)),
        training_contrast_mad=contrast_mad,
        multi_log_center=multi_center,
        multi_log_scale=multi_scale,
        count_smoothing=smoothing,
        maximum_subtraction_fraction=float(maximum_subtraction_fraction),
        minimum_noise_fraction=float(minimum_noise_fraction),
        maximum_noise_fraction=float(maximum_noise_fraction),
        training_sample_count=int(np.sum(train)),
    )
    return model, {
        "schema": "v7_gsi_noise_fit_report_v1",
        "raw_gsi_estimate": raw_gsi,
        "bounded_gsi_estimate": effective_gsi,
        "noise_fraction_prior": noise_fraction,
        "confidence": confidence,
        "training_contrast_quantiles": np.quantile(
            finite_contrast, (0.10, 0.50, 0.90),
        ).tolist(),
        "coin_total_prediction_ratio_quantiles": np.quantile(
            coin_scale * singles_geomean / np.maximum(coin.sum(axis=1), 1.0),
            (0.10, 0.50, 0.90),
        ).tolist(),
        "production_subtraction_allowed": False,
        "confidence_policy": "relaxed_research_only",
        "is_direct_per_run_gsi_measurement": False,
        "gsi_interpretation": (
            "training-only robust interference-contrast proxy constrained by the configured GSI range"
        ),
    }


__all__ = ["NOISE_MODEL_SCHEMA", "V7GsiNoiseModelV1", "fit_v7_gsi_noise_model"]
