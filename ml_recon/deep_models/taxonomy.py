"""Hierarchical physical error taxonomy for deep synthetic training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


STATE_FAMILY_LABELS: List[str] = [
    "w_state",
    "ghz_state",
    "single_excitation",
    "pair_entangled",
    "dicke_like",
    "mixed",
]

ENTANGLEMENT_TIER_LABELS: List[str] = [
    "mixed_or_inconclusive",
    "separable_like",
    "biseparable_pair_like",
    "w_genuine_candidate",
    "ghz_genuine_candidate",
]

PHYSICS_AUX_TARGET_NAMES: List[str] = [
    "fidelity_proxy",
    "purity",
    "w_overlap",
    "ghz_overlap",
    "w_witness_margin",
    "ghz_witness_margin",
    "single_excitation_population",
    "double_excitation_population",
    "off_w_subspace_population",
    "single_population_imbalance",
    "offdiag_coherence_mean",
    "coherence_residual_mean",
    "single_subspace_phase_spread",
    "w_subspace_residual_fraction",
    "coherent_phase_fraction",
    "phase_map_rms_norm",
    "phase_map_tv_norm",
    "freq_delta_rms_norm",
    "crosstalk_rms",
    "severity_hint",
    "single_population_slope",
    "power_compression_signature",
    "random_amp_signature",
    "phase_residual_vs_scan_control",
    "scan_control_vs_phase_residual",
    "amplitude_slope_vs_random_imbalance",
]

EXCITATION_MODE_LABELS: List[str] = [
    "ideal_single_excitation",
    "under_excited",
    "over_excited",
    "mode_contaminated",
    "incoherent_mixed",
]

COARSE_ERROR_LABELS: List[str] = [
    "coherent_control",
    "amplitude_power",
    "incoherent_noise",
    "readout_retrieval",
    "leakage_population",
    "hardware_scan",
]


@dataclass(frozen=True)
class ErrorSpec:
    fine: str
    coarse: str
    correctable: bool
    reversible: bool
    description: str


ERROR_SPECS: List[ErrorSpec] = [
    ErrorSpec("phase_drift", "coherent_control", True, True, "Local W-subspace phase drift."),
    ErrorSpec("wch1_phase_map_error", "coherent_control", True, True, "Pointwise W_CH1 phase-map error."),
    ErrorSpec("phase_crosstalk", "coherent_control", True, True, "Correlated phase coupling between addressed modes."),
    ErrorSpec("amp_imbalance", "amplitude_power", True, False, "Unequal mode amplitudes or RF power split."),
    ErrorSpec("rf_power_compression", "amplitude_power", True, False, "High-power nonlinear amplitude compression."),
    ErrorSpec("dephasing", "incoherent_noise", False, False, "Loss of off-diagonal coherence."),
    ErrorSpec("depolarizing", "incoherent_noise", False, False, "Mixing with maximally mixed density."),
    ErrorSpec("retrieval_imbalance", "readout_retrieval", True, False, "Readout/retrieval efficiency imbalance."),
    ErrorSpec("shot_dark_noise", "readout_retrieval", False, False, "Finite-count and dark-count measurement noise."),
    ErrorSpec("double_excitation", "leakage_population", False, False, "Population leaks into double-excitation basis states."),
    ErrorSpec("mode_contamination", "leakage_population", False, False, "State contaminated by non-target excitation modes."),
    ErrorSpec("frequency_nonlinearity", "hardware_scan", True, True, "AOD set frequency differs from actual frequency."),
    ErrorSpec("thermal_drift", "hardware_scan", True, False, "Slow thermal drift in scan response."),
]

FINE_ERROR_LABELS: List[str] = [spec.fine for spec in ERROR_SPECS]
FINE_TO_COARSE: Dict[str, str] = {spec.fine: spec.coarse for spec in ERROR_SPECS}
FINE_TO_SPEC: Dict[str, ErrorSpec] = {spec.fine: spec for spec in ERROR_SPECS}
COARSE_TO_FINE: Dict[str, List[str]] = {
    coarse: [spec.fine for spec in ERROR_SPECS if spec.coarse == coarse]
    for coarse in COARSE_ERROR_LABELS
}


def fine_index(label: str) -> int:
    return FINE_ERROR_LABELS.index(label)


def coarse_index(label: str) -> int:
    return COARSE_ERROR_LABELS.index(label)


def family_index(label: str) -> int:
    return STATE_FAMILY_LABELS.index(label)


def entanglement_index(label: str) -> int:
    return ENTANGLEMENT_TIER_LABELS.index(label)


def excitation_index(label: str) -> int:
    return EXCITATION_MODE_LABELS.index(label)


def correctable_fraction_for_errors(labels: List[str], severity: float) -> float:
    if not labels:
        return 0.0
    weights = []
    for label in labels:
        spec = FINE_TO_SPEC[label]
        if spec.correctable and spec.reversible:
            weights.append(0.90)
        elif spec.correctable:
            weights.append(0.65)
        else:
            weights.append(0.18)
    value = sum(weights) / len(weights)
    return float(max(0.0, min(1.0, value * (0.7 + 0.3 * float(severity)))))
