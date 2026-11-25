"""Hard-gated noise/calibrator ranking and Pareto selection."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


HARD_GATE_NAMES = (
    "no_split_leakage",
    "real_nll_beats_baseline",
    "no_zero_recall_isolated_mechanism",
    "uq_report_present",
    "no_command_hash_path_numeric_input",
    "coefficient_identifiability_passed",
    "no_hardware_write",
    "amplitude_claim_honest",
)


def _harmonic_mean(values: Sequence[float]) -> float:
    positive = [max(float(value), 1e-12) for value in values]
    return len(positive) / sum(1.0 / value for value in positive)


def score_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    gates = dict(candidate.get("hard_gates") or {})
    gate_passed = all(bool(gates.get(name, False)) for name in HARD_GATE_NAMES)
    noise_score = float(candidate.get("NoiseScore", candidate.get("macro_F1", 0.0)))
    calibration_score = float(candidate.get(
        "CalibrationScore", candidate.get("safe_action_coverage", 0.0)
    ))
    return {
        **dict(candidate),
        "NoiseScore": noise_score,
        "CalibrationScore": calibration_score,
        "OverallCalibratorScore": _harmonic_mean((noise_score, calibration_score)),
        "hard_gate_passed": gate_passed,
        "failed_hard_gates": [name for name in HARD_GATE_NAMES if not bool(gates.get(name, False))],
    }


def pareto_front(candidates: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    eligible = [score_candidate(item) for item in candidates]
    eligible = [item for item in eligible if item["hard_gate_passed"]]
    front = []
    for candidate in eligible:
        dominated = any(
            other is not candidate
            and other["NoiseScore"] >= candidate["NoiseScore"]
            and other["CalibrationScore"] >= candidate["CalibrationScore"]
            and (
                other["NoiseScore"] > candidate["NoiseScore"]
                or other["CalibrationScore"] > candidate["CalibrationScore"]
            )
            for other in eligible
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: (-item["OverallCalibratorScore"], str(item.get("model_id"))))


__all__ = ["HARD_GATE_NAMES", "pareto_front", "score_candidate"]
