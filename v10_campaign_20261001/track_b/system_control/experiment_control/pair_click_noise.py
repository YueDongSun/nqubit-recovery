"""Conservative two-click pair mixture model.

The model retains raw counters regardless of confidence.  Automatic topology
selection is diagnostic only until persistent, statistically separated low
pairs have been demonstrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


PAIR_NAMES = ("12", "13", "14", "23", "24", "34")
PAIR_MODEL_MODES = {"disabled", "manual_topology", "auto_suppressed_pair"}


@dataclass
class PairClickNoiseResult:
    mode: str
    raw_pair_counts: Dict[str, int]
    raw_pair_distribution: Dict[str, float]
    candidate_suppressed_pairs: list[str] = field(default_factory=list)
    true_multi2_estimate: float | None = None
    dark_background_multi2_estimate: float | None = None
    pair_dark_fraction: float | None = None
    confidence: float = 0.0
    true_multi2_std: float | None = None
    heuristic_only: bool = True
    true_multi2_is_supervision_target: bool = False
    dark_not_identifiable: bool = True
    posterior_uncertainty_required: bool = True
    external_calibration_id: str | None = None
    production_subtraction_allowed: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, object]:
        return asdict(self)


def fit_pair_click_noise(
    pair_counts: Mapping[str, int | float] | None,
    *,
    mode: str = "disabled",
    suppressed_pairs: Optional[Sequence[str]] = None,
    detector_efficiencies: Optional[Mapping[str, float]] = None,
    dark_rates: Optional[Mapping[str, float]] = None,
    bootstrap_samples: int = 512,
    off_window_calibrated: bool = False,
    external_calibration_id: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> PairClickNoiseResult:
    if mode not in PAIR_MODEL_MODES:
        raise ValueError(f"mode must be one of {sorted(PAIR_MODEL_MODES)}")
    raw = {name: max(0, int((pair_counts or {}).get(name, 0))) for name in PAIR_NAMES}
    total = int(sum(raw.values()))
    distribution = {name: (raw[name] / total if total else 0.0) for name in PAIR_NAMES}
    result = PairClickNoiseResult(mode=mode, raw_pair_counts=raw, raw_pair_distribution=distribution)
    result.dark_not_identifiable = not bool(off_window_calibrated)
    result.external_calibration_id = external_calibration_id
    result.warnings.append("suppressed-pair estimate is initialization/diagnostic only, never a direct supervision target")
    if mode == "disabled":
        result.warnings.append("pair mixture subtraction disabled")
        return result
    if total == 0:
        result.warnings.append("no pair counters available")
        return result

    if mode == "manual_topology":
        chosen = _validate_pairs(suppressed_pairs or [])
        if len(chosen) != 2:
            result.warnings.append("manual topology requires exactly two suppressed pair labels")
            return result
        result.candidate_suppressed_pairs = chosen
        confidence = 1.0
    else:
        chosen, confidence, warning = _auto_suppressed_pairs(raw, bootstrap_samples, rng)
        result.candidate_suppressed_pairs = chosen
        if warning:
            result.warnings.append(warning)
        if len(chosen) != 2:
            return result

    background_pair_mean = float(np.mean([raw[name] for name in chosen]))
    # Three physical pair pathways remain after the two topology-suppressed
    # channels.  The estimate is deliberately bounded by the observed total.
    background = min(float(total), background_pair_mean * len(PAIR_NAMES))
    result.dark_background_multi2_estimate = background
    result.true_multi2_estimate = max(0.0, float(total) - background)
    result.pair_dark_fraction = background / total if total else None
    detector_factor = _detector_confidence(detector_efficiencies, dark_rates)
    result.confidence = float(np.clip(confidence * detector_factor, 0.0, 1.0))
    result.true_multi2_std = float(np.sqrt(max(float(total) + len(PAIR_NAMES) ** 2 * background_pair_mean, 1.0)))
    calibrated_nuisance = detector_efficiencies is not None and dark_rates is not None
    result.production_subtraction_allowed = bool(
        mode == "manual_topology" and off_window_calibrated
        and calibrated_nuisance and result.confidence >= 0.85
    )
    if result.dark_not_identifiable:
        result.warnings.append("dark_not_identifiable: no off-window calibration; retain posterior uncertainty")
    if not result.production_subtraction_allowed:
        result.warnings.append("insufficient bootstrap confidence; no production subtraction")
    return result


def _validate_pairs(values: Sequence[str]) -> list[str]:
    output = []
    for value in values:
        label = str(value)
        if label not in PAIR_NAMES:
            raise ValueError(f"unknown detector pair {label!r}")
        if label not in output:
            output.append(label)
    return output


def _auto_suppressed_pairs(raw: Mapping[str, int], samples: int, rng: Optional[np.random.Generator]) -> tuple[list[str], float, str]:
    values = np.asarray([raw[name] for name in PAIR_NAMES], dtype=float)
    order = np.argsort(values)
    chosen = [PAIR_NAMES[int(order[0])], PAIR_NAMES[int(order[1])]]
    if values[order[2]] <= values[order[1]]:
        return [], 0.0, "pair counts are not separated"
    generator = rng or np.random.default_rng(20260710)
    stable = 0
    separated = 0
    for _ in range(max(32, int(samples))):
        draw = generator.poisson(values)
        draw_order = np.argsort(draw)
        picked = {PAIR_NAMES[int(draw_order[0])], PAIR_NAMES[int(draw_order[1])]}
        if picked == set(chosen):
            stable += 1
        # A two-standard-deviation gap avoids treating random ties as topology.
        if draw[draw_order[2]] - draw[draw_order[1]] > 2.0 * np.sqrt(max(draw[draw_order[2]], 1.0)):
            separated += 1
    stability = stable / max(32, int(samples))
    separation = separated / max(32, int(samples))
    confidence = min(stability, separation)
    if confidence < 0.85:
        return chosen, confidence, "candidate suppressed pairs fail bootstrap stability/separation criterion"
    return chosen, confidence, ""


def _detector_confidence(efficiencies: Optional[Mapping[str, float]], dark_rates: Optional[Mapping[str, float]]) -> float:
    if efficiencies is None and dark_rates is None:
        return 1.0
    factors = []
    if efficiencies:
        values = np.asarray(list(efficiencies.values()), dtype=float)
        factors.append(float(np.clip(np.nanmean(values), 0.0, 1.0)))
    if dark_rates:
        values = np.asarray(list(dark_rates.values()), dtype=float)
        factors.append(float(np.clip(1.0 - np.nanmean(values), 0.0, 1.0)))
    return float(np.prod(factors)) if factors else 1.0
