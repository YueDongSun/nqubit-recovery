"""Noise-identification and calibration metrics without false real labels."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ml_recon.physics_identification.hierarchical_nb2 import nb2_nll


def binary_mechanism_metrics(target: Any, score: Any, *, threshold: float = 0.5) -> Dict[str, Any]:
    truth = np.asarray(target, dtype=bool)
    probability = np.asarray(score, dtype=float)
    if truth.shape != probability.shape or truth.ndim != 2:
        raise ValueError("mechanism target/score must be matching [samples, mechanisms]")
    prediction = probability >= float(threshold)
    per_mechanism = []
    for index in range(truth.shape[1]):
        tp = int(np.sum(prediction[:, index] & truth[:, index]))
        fp = int(np.sum(prediction[:, index] & ~truth[:, index]))
        fn = int(np.sum(~prediction[:, index] & truth[:, index]))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        # Dependency-light average precision from the ranked precision curve.
        order = np.argsort(-probability[:, index])
        ranked_truth = truth[order, index]
        cumulative = np.cumsum(ranked_truth)
        precision_curve = cumulative / np.arange(1, ranked_truth.size + 1)
        average_precision = float(np.sum(precision_curve * ranked_truth) / max(np.sum(ranked_truth), 1))
        per_mechanism.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": average_precision,
        })
    return {
        "macro_AUPRC": float(np.mean([item["average_precision"] for item in per_mechanism])),
        "macro_F1": float(np.mean([item["f1"] for item in per_mechanism])),
        "minimum_recall": float(np.min([item["recall"] for item in per_mechanism])),
        "per_mechanism_precision_recall": per_mechanism,
    }


def real_noise_evidence(
    count: Any,
    predicted_mean: Any,
    *,
    dispersion: float,
    intercept_mean: Any,
) -> Dict[str, Any]:
    heldout_nll = float(nb2_nll(count, predicted_mean, dispersion))
    intercept_nll = float(nb2_nll(count, intercept_mean, dispersion))
    return {
        "heldout_NB2_NLL": heldout_nll,
        "intercept_NB2_NLL": intercept_nll,
        "delta_NLL_vs_intercept": intercept_nll - heldout_nll,
        "deviance_explained": 1.0 - heldout_nll / max(intercept_nll, 1e-12),
        "real_mechanism_accuracy": None,
        "unavailable_metrics": ["real_mechanism_accuracy"],
    }


def calibration_envelope(
    action_errors: Any,
    post_action_infidelity: Any,
    action_norms: Any,
    safe_mask: Any,
    *,
    amplitude_empirical_evidence: bool = False,
) -> Dict[str, Any]:
    errors = np.asarray(action_errors, dtype=float)
    infidelity = np.asarray(post_action_infidelity, dtype=float).reshape(-1)
    norms = np.asarray(action_norms, dtype=float).reshape(-1)
    safe = np.asarray(safe_mask, dtype=bool).reshape(-1)
    if errors.shape[0] != infidelity.size or norms.size != infidelity.size or safe.size != infidelity.size:
        raise ValueError("calibration arrays must share a sample dimension")
    safe_norms = norms[safe]
    return {
        "action_coordinate_MAE": float(np.mean(np.abs(errors))),
        "post_action_infidelity": float(np.mean(infidelity)),
        "one_step_success": float(np.mean(infidelity <= 0.15)),
        "three_step_success": float(np.mean(infidelity <= 0.10)),
        "five_step_success": float(np.mean(infidelity <= 0.05)),
        "action_saturation": float(np.mean(norms >= np.max(norms) if norms.size else False)),
        "safe_action_coverage": float(np.mean(safe)),
        "phase_correctable_volume": float(np.mean(safe) * np.max(safe_norms, initial=0.0)),
        "amplitude_correctable_volume": (
            float(np.mean(safe) * np.max(safe_norms, initial=0.0))
            if amplitude_empirical_evidence else None
        ),
        "full_correctable_volume": float(np.mean(safe) * np.max(safe_norms, initial=0.0)),
        "maximum_safe_radius": float(np.max(safe_norms, initial=0.0)),
        "real_amplitude_metrics": {"available": bool(amplitude_empirical_evidence)},
    }


__all__ = [
    "binary_mechanism_metrics",
    "calibration_envelope",
    "real_noise_evidence",
]
