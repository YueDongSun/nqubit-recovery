"""Auditable DLCZ noise simulator for the V8 high-risk research track.

The simulator is deliberately independent from the V8 model and trainer.  It
models a weak-write DLCZ source, exact one/two-excitation bosonic occupancy,
four-detector efficiency and dark events, accidental coincidences, and a
Gamma--Poisson brightness process.  Every unobserved clean/noise quantity is
named ``synthetic_truth_*`` and every measured-like output remains an
observation.  In particular, global multi-click totals are never expanded into
operation-local pseudo-labels.

The registered 75/256 read bases map one complex clean density operator to Born
probabilities, then an explicit Q-mode-to-four-detector calibration maps each
read basis into detector patterns.  The recorded click model is a rare-event
marked Poisson approximation.  Single-, two-, higher-excitation, accidental,
and dark point processes are sampled independently and their realized pattern
counts are summed exactly; the returned collision bound makes ignored
same-gate overlaps auditable.
This is appropriate for the weak-excitation regime and fails closed when the
configured per-gate event probability is no longer small.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


DLCZ_NOISE_SYNTHETIC_SCHEMA = "v8_dlcz_noise_synthetic_batch_v2"
DLCZ_NOISE_FAMILY_SCHEMA = "v8_dlcz_noise_family_v1"
DLCZ_GLOBAL_MULTI_SCHEMA = "v8_dlcz_global_multi_click_v1"
DLCZ_NEGATIVE_CONTROL_SCHEMA = "v8_dlcz_noise_negative_controls_v2"
DLCZ_SEMISYNTHETIC_SCHEMA = "v8_dlcz_semisynthetic_injection_v1"

SYSTEM_OPERATION_COUNTS: Mapping[str, int] = MappingProxyType(
    {"3qubit": 75, "4qubit": 256}
)
EXPOSURE_LEVELS_S: Tuple[float, ...] = (150.0, 300.0, 600.0, 1200.0)
VALID_EXPOSURE_MODES = frozenset(("known", "unknown", "mixed"))
PAIR_LABELS: Tuple[str, ...] = ("12", "13", "14", "23", "24", "34")
PAIR_DETECTOR_INDICES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)
PATTERNS = np.asarray(
    [[float((index >> bit) & 1) for bit in range(4)] for index in range(16)],
    dtype=np.float64,
)
PATTERN_WEIGHTS = PATTERNS.sum(axis=1).astype(np.int64)
PAIR_PATTERN_INDICES = np.asarray(
    [(1 << first) | (1 << second) for first, second in PAIR_DETECTOR_INDICES],
    dtype=np.int64,
)
ONE_CLICK_PATTERN_INDICES = np.asarray((1, 2, 4, 8), dtype=np.int64)
READ_MULTI_SIMULTANEITY_CLOCK_S = 10.0e-9
READ_WINDOW_ACCEPTED_CLOCK_COUNT = 19
READ_WINDOW_ACCEPTED_SUPPORT_S = 190.0e-9
BASIS_PHASE_SIGN = -1.0
BASIS_PHASE_CONVENTION = "read_aod_negative_order"
HERALD_SEMANTIC = (
    "aggregate_binary_external_herald_spad_clicks_over_write_gates;"
    "each_200ns_gate_records_presence_not_photon_number;"
    "four_on_off_detectors_are_read_arm_only"
)
MODE_TO_DETECTOR_BASELINES: Mapping[str, Tuple[Tuple[float, ...], ...]] = (
    MappingProxyType(
        {
            "3qubit": (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
            ),
            "4qubit": (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
)


@dataclass(frozen=True)
class DLCZNoiseFamilyConfig:
    """One immutable, falsifiable DLCZ mechanism family."""

    family_id: str
    role: str
    chi_lower: float
    chi_upper: float
    brightness_dispersion: float
    detector_efficiency: Tuple[float, float, float, float]
    detector_efficiency_logit_std: float
    global_optical_path_efficiency: float
    dark_probability: Tuple[float, float, float, float]
    dark_log_std: float
    gate_rate_hz_by_system: Tuple[float, float]
    background_singles_rate_hz: Tuple[float, float, float, float]
    accidental_window_s: float
    off_window_width_s: float
    off_window_gate_fraction: float
    dephasing_lower: float
    dephasing_upper: float
    crosstalk_lower: float
    crosstalk_upper: float
    n2_nuisance_fraction_range: Tuple[float, float]
    n2_split_fraction_of_nuisance_range: Tuple[float, float]
    n2_hom_indistinguishability_range: Tuple[float, float]
    locked_holdout: bool = False

    def validate(self) -> None:
        if self.role not in ("train", "mechanism_shift_holdout"):
            raise ValueError("DLCZ family role is invalid")
        if (self.role == "mechanism_shift_holdout") != bool(self.locked_holdout):
            raise ValueError("only mechanism-shift holdout families may be locked")
        if not 0.0 < self.chi_lower < self.chi_upper < 0.25:
            raise ValueError("weak-write chi range must lie in (0, 0.25)")
        if not 0.0 < self.brightness_dispersion <= 1.0:
            raise ValueError("brightness dispersion must lie in (0, 1]")
        efficiency = np.asarray(self.detector_efficiency, dtype=np.float64)
        dark = np.asarray(self.dark_probability, dtype=np.float64)
        background = np.asarray(self.background_singles_rate_hz, dtype=np.float64)
        if efficiency.shape != (4,) or np.any(efficiency <= 0.0) or np.any(efficiency > 1.0):
            raise ValueError("detector_efficiency must contain four values in (0, 1]")
        if dark.shape != (4,) or np.any(dark < 0.0) or np.any(dark >= 1.0):
            raise ValueError("dark_probability must contain four values in [0, 1)")
        if background.shape != (4,) or np.any(background < 0.0):
            raise ValueError("background singles rates must be four non-negative values")
        if min(
            self.detector_efficiency_logit_std,
            self.dark_log_std,
            *self.gate_rate_hz_by_system,
            self.accidental_window_s,
            self.off_window_width_s,
            self.off_window_gate_fraction,
        ) <= 0.0:
            raise ValueError("detector, gate, and window parameters must be positive")
        if not 0.0 < float(self.global_optical_path_efficiency) <= 1.0:
            raise ValueError("global optical-path efficiency must lie in (0, 1]")
        if not 0.0 <= self.dephasing_lower < self.dephasing_upper < 1.0:
            raise ValueError("dephasing range is invalid")
        if not 0.0 <= self.crosstalk_lower < self.crosstalk_upper < 0.5:
            raise ValueError("crosstalk range is invalid")
        for interval in (
            self.n2_nuisance_fraction_range,
            self.n2_split_fraction_of_nuisance_range,
            self.n2_hom_indistinguishability_range,
        ):
            if len(interval) != 2 or not 0.0 <= interval[0] <= interval[1] <= 1.0:
                raise ValueError("n=2 partial-identification intervals must lie in [0, 1]")

    @property
    def family_hash(self) -> str:
        self.validate()
        return _json_hash(
            {"schema": DLCZ_NOISE_FAMILY_SCHEMA, "config": asdict(self)}
        )


@dataclass(frozen=True)
class DLCZIntervention:
    """Registered synthetic intervention with common-random-number support."""

    write_power_scale: float = 1.0
    source_brightness_scale: float = 1.0
    hbt_beamsplitter_transmission: Optional[float] = None
    accidental_window_scale: float = 1.0
    off_window_width_scale: float = 1.0
    off_window_gate_scale: float = 1.0
    detector_efficiency_scale: float = 1.0
    dark_probability_scale: float = 1.0

    def validate(self) -> None:
        brightness_scale = float(self.source_brightness_scale)
        if not np.isfinite(brightness_scale) or brightness_scale < 0.0:
            raise ValueError("source_brightness_scale must be finite and non-negative")
        for name in (
            "write_power_scale",
            "accidental_window_scale",
            "off_window_width_scale",
            "off_window_gate_scale",
            "detector_efficiency_scale",
            "dark_probability_scale",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        transmission = self.hbt_beamsplitter_transmission
        if transmission is not None and not 0.0 < float(transmission) < 1.0:
            raise ValueError("HBT beamsplitter transmission must lie in (0, 1)")

    @property
    def intervention_hash(self) -> str:
        self.validate()
        return _json_hash(asdict(self))


_DLCZ_NOISE_FAMILIES = {
    "measured_domain_train": DLCZNoiseFamilyConfig(
        family_id="measured_domain_train",
        role="train",
        chi_lower=0.0100,
        chi_upper=0.2000,
        brightness_dispersion=0.16,
        detector_efficiency=(0.30, 0.31, 0.32, 0.33),
        detector_efficiency_logit_std=0.10,
        global_optical_path_efficiency=0.40,
        dark_probability=(8.0e-6, 1.1e-5, 9.0e-6, 1.0e-5),
        dark_log_std=0.25,
        gate_rate_hz_by_system=(419.6, 137.35),
        background_singles_rate_hz=(1_800.0, 2_100.0, 1_900.0, 2_000.0),
        accidental_window_s=10.0e-9,
        off_window_width_s=4.0e-9,
        off_window_gate_fraction=0.25,
        dephasing_lower=0.02,
        dephasing_upper=0.20,
        crosstalk_lower=0.00,
        crosstalk_upper=0.10,
        n2_nuisance_fraction_range=(0.05, 0.45),
        n2_split_fraction_of_nuisance_range=(0.25, 0.75),
        n2_hom_indistinguishability_range=(0.0, 1.0),
    ),
    "mechanism_shift_holdout": DLCZNoiseFamilyConfig(
        family_id="mechanism_shift_holdout",
        role="mechanism_shift_holdout",
        chi_lower=0.030,
        chi_upper=0.180,
        brightness_dispersion=0.38,
        detector_efficiency=(0.25, 0.40, 0.28, 0.38),
        detector_efficiency_logit_std=0.18,
        global_optical_path_efficiency=0.40,
        dark_probability=(2.2e-5, 0.9e-5, 2.8e-5, 1.4e-5),
        dark_log_std=0.45,
        gate_rate_hz_by_system=(419.6, 137.35),
        background_singles_rate_hz=(2_800.0, 1_500.0, 3_200.0, 2_100.0),
        accidental_window_s=10.0e-9,
        off_window_width_s=7.0e-9,
        off_window_gate_fraction=0.40,
        dephasing_lower=0.24,
        dephasing_upper=0.52,
        crosstalk_lower=0.14,
        crosstalk_upper=0.30,
        n2_nuisance_fraction_range=(0.35, 0.80),
        n2_split_fraction_of_nuisance_range=(0.10, 0.90),
        n2_hom_indistinguishability_range=(0.0, 1.0),
        locked_holdout=True,
    ),
}
DLCZ_NOISE_FAMILIES: Mapping[str, DLCZNoiseFamilyConfig] = MappingProxyType(
    _DLCZ_NOISE_FAMILIES
)


def resolve_dlcz_noise_family(
    family_id: str,
    *,
    expected_role: Optional[str] = None,
    allow_holdout_evaluation: bool = False,
) -> DLCZNoiseFamilyConfig:
    """Resolve a family and prevent mechanism-shift leakage into training."""

    try:
        family = DLCZ_NOISE_FAMILIES[str(family_id)]
    except KeyError as exc:
        raise ValueError(f"unknown DLCZ noise family: {family_id}") from exc
    family.validate()
    if expected_role is not None and family.role != str(expected_role):
        raise ValueError(
            f"DLCZ family {family.family_id} has role {family.role}, "
            f"expected {expected_role}"
        )
    if family.locked_holdout and not bool(allow_holdout_evaluation):
        raise PermissionError(
            "mechanism-shift holdout is evaluation-only and requires explicit unlock"
        )
    return family


def geometric_excitation_probabilities(chi: Any) -> np.ndarray:
    """Return exact ``P(n=0), P(n=1), P(n=2), P(n>=3)`` for DLCZ write.

    ``P(n)=(1-chi) chi**n`` is the thermal/geometric weak-write law.  Hence
    ``P(2)=(1-chi) chi**2`` and the audited unresolved tail is exactly
    ``P(n>=3)=chi**3``.
    """

    value = np.asarray(chi, dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0) or np.any(value >= 1.0):
        raise ValueError("chi must be finite and lie in [0, 1)")
    p0 = 1.0 - value
    p1 = p0 * value
    p2 = p0 * np.square(value)
    tail = np.power(value, 3)
    result = np.stack((p0, p1, p2, tail), axis=-1)
    if not np.allclose(result.sum(axis=-1), 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError("geometric excitation audit lost normalization")
    return result


def single_excitation_occupancy(mode_probability: Any) -> np.ndarray:
    """Return the four one-boson Fock-state probabilities."""

    return _validate_mode_probability(mode_probability).copy()


def two_excitation_bosonic_occupancy(mode_probability: Any) -> Dict[str, np.ndarray]:
    """Return exact same-mode and distinct-mode occupancy for two bosons.

    For two identical bosons in the same single-particle mode, the normalized
    Fock probabilities are ``p_i**2`` for ``|2_i>`` and ``2 p_i p_j`` for
    ``|1_i 1_j>``.  Keeping the factor of two explicit is important for HBT
    and detector-efficiency interventions.
    """

    probability = _validate_mode_probability(mode_probability)
    same_mode = np.square(probability)
    distinct_mode = np.stack(
        (
            2.0 * probability[..., 0] * probability[..., 1],
            2.0 * probability[..., 0] * probability[..., 2],
            2.0 * probability[..., 0] * probability[..., 3],
            2.0 * probability[..., 1] * probability[..., 2],
            2.0 * probability[..., 1] * probability[..., 3],
            2.0 * probability[..., 2] * probability[..., 3],
        ),
        axis=-1,
    )
    if not np.allclose(
        same_mode.sum(axis=-1) + distinct_mode.sum(axis=-1),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("two-boson occupancy lost normalization")
    return {"same_mode": same_mode, "distinct_mode": distinct_mode}


def detected_pattern_probabilities_for_n(
    mode_probability: Any,
    detector_efficiency: Sequence[float] | np.ndarray,
    excitation_number: int,
) -> np.ndarray:
    """Exact detected click-pattern probabilities for an ``n``-boson Fock state.

    Each photon occupies detector mode ``i`` with probability ``p_i`` and is
    registered with efficiency ``eta_i``.  Inclusion--exclusion converts the
    multinomial photon occupancy to the 16 binary click patterns.  Therefore
    two photons in one detector respond as ``1-(1-eta_i)**2`` rather than as a
    single Bernoulli trial.
    """

    if int(excitation_number) != excitation_number or int(excitation_number) < 0:
        raise ValueError("excitation_number must be a non-negative integer")
    probability = _validate_mode_probability(mode_probability)
    efficiency = _broadcast_detector_parameter(
        detector_efficiency, probability.shape[:-1], "detector_efficiency"
    )
    if np.any(efficiency < 0.0) or np.any(efficiency > 1.0):
        raise ValueError("detector efficiency must lie in [0, 1]")
    registered = probability * efficiency
    missed = 1.0 - registered.sum(axis=-1)
    result = _exact_presence_patterns_from_power(
        registered,
        lambda included_probability: np.power(
            np.clip(missed + included_probability, 0.0, 1.0),
            int(excitation_number),
        ),
    )
    return _normalize_pattern_probability(result)


def geometric_detected_pattern_components(
    mode_probability: Any,
    detector_efficiency: Sequence[float] | np.ndarray,
    chi: Any,
) -> Dict[str, np.ndarray]:
    """Return exact geometric photon-pattern mixture and its n=0/1/2/tail audit."""

    probability = _validate_mode_probability(mode_probability)
    efficiency = _broadcast_detector_parameter(
        detector_efficiency, probability.shape[:-1], "detector_efficiency"
    )
    chi_value = np.asarray(chi, dtype=np.float64)
    try:
        chi_value = np.broadcast_to(chi_value, probability.shape[:-1])
    except ValueError as exc:
        raise ValueError("chi cannot broadcast to the mode-probability batch") from exc
    excitation = geometric_excitation_probabilities(chi_value)
    registered = probability * efficiency
    missed = 1.0 - registered.sum(axis=-1)

    total = _exact_presence_patterns_from_power(
        registered,
        lambda included_probability: (1.0 - chi_value)
        / np.maximum(
            1.0
            - chi_value
            * np.clip(missed + included_probability, 0.0, 1.0),
            np.finfo(np.float64).eps,
        ),
    )
    total = _normalize_pattern_probability(total)
    n0 = np.zeros_like(total)
    n0[..., 0] = excitation[..., 0]
    n1 = excitation[..., 1, None] * detected_pattern_probabilities_for_n(
        probability, efficiency, 1
    )
    n2 = excitation[..., 2, None] * detected_pattern_probabilities_for_n(
        probability, efficiency, 2
    )
    tail = np.maximum(total - n0 - n1 - n2, 0.0)
    tail_sum = tail.sum(axis=-1, keepdims=True)
    tail_target = excitation[..., 3, None]
    tail = np.where(
        tail_target > 0.0,
        tail * tail_target / np.maximum(tail_sum, np.finfo(np.float64).eps),
        0.0,
    )
    total = n0 + n1 + n2 + tail
    return {"total": total, "n0": n0, "n1": n1, "n2": n2, "tail": tail}


def sample_gamma_poisson_nb2(
    mean: Any,
    dispersion: float,
    *,
    seed: int,
    size: Optional[int | Tuple[int, ...]] = None,
) -> np.ndarray:
    """Sample NB2 counts through its auditable Gamma--Poisson construction."""

    alpha = float(dispersion)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("NB2 dispersion must be finite and positive")
    mean_value = np.asarray(mean, dtype=np.float64)
    if np.any(~np.isfinite(mean_value)) or np.any(mean_value < 0.0):
        raise ValueError("NB2 mean must be finite and non-negative")
    if size is not None:
        target_shape = (int(size),) if isinstance(size, int) else tuple(size)
        mean_value = np.broadcast_to(mean_value, target_shape)
    rng = np.random.default_rng(int(seed))
    brightness = rng.gamma(
        shape=1.0 / alpha,
        scale=alpha,
        size=mean_value.shape,
    )
    return rng.poisson(mean_value * brightness).astype(np.int64)


def registered_povm_contract(system_type: str) -> Dict[str, Any]:
    """Build the committed 75/256 read-basis POVM and immutable hashes."""

    from system_control.qubit_recon_files.basis_configs import (
        build_3qubit_basis_config,
        build_4qubit_basis_config,
    )

    if system_type == "3qubit":
        basis_config = build_3qubit_basis_config()
        entries = list(basis_config["entries"])
        qubit_count = 3
    elif system_type == "4qubit":
        basis_config = build_4qubit_basis_config()
        entries = [
            entry
            for cycle in basis_config["cycles"]
            for entry in cycle["entries"]
        ]
        qubit_count = 4
    else:
        raise ValueError("system_type must be 3qubit or 4qubit")
    expected_count = SYSTEM_OPERATION_COUNTS[system_type]
    if len(entries) != expected_count:
        raise AssertionError("registered basis operation count changed")

    basis_states = []
    for entry in entries:
        amplitude = np.asarray(entry["amp"], dtype=np.float64)
        phase = BASIS_PHASE_SIGN * np.asarray(entry["phase"], dtype=np.float64)
        if amplitude.shape != (qubit_count,) or phase.shape != (qubit_count,):
            raise AssertionError("registered basis dimension changed")
        state = amplitude * np.exp(1j * phase)
        norm = float(np.linalg.norm(state))
        if norm <= 0.0:
            raise AssertionError("registered basis contains a zero state")
        basis_states.append(state / norm)
    state_array = np.asarray(basis_states, dtype=np.complex128)
    operator = np.einsum("oi,oj->oij", state_array, state_array.conj())
    basis_config_hash = _json_hash(basis_config)
    povm_hash = _array_contract_hash(
        operator,
        {
            "basis_config_hash": basis_config_hash,
            "basis_phase_sign": BASIS_PHASE_SIGN,
            "basis_phase_convention": BASIS_PHASE_CONVENTION,
        },
    )
    return {
        "system_type": system_type,
        "qubit_count": qubit_count,
        "operation_count": expected_count,
        "basis_config_hash": basis_config_hash,
        "povm_hash": povm_hash,
        "basis_phase_sign": BASIS_PHASE_SIGN,
        "basis_phase_convention": BASIS_PHASE_CONVENTION,
        "basis_labels": tuple(str(entry["label"]) for entry in entries),
        "basis_states": state_array,
        "operators": operator,
        "basis_mode_probability": np.square(np.abs(state_array)),
    }


def born_probabilities(clean_rho: Any, povm_operators: Any) -> np.ndarray:
    """Evaluate the registered single-excitation Born map."""

    rho = np.asarray(clean_rho, dtype=np.complex128)
    operators = np.asarray(povm_operators, dtype=np.complex128)
    if rho.ndim != 3 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("clean_rho must be [batch, qubit, qubit]")
    if operators.ndim != 3 or operators.shape[1:] != rho.shape[1:]:
        raise ValueError("POVM operators must be [operation, qubit, qubit]")
    probability = np.real(np.einsum("oij,bji->bo", operators, rho))
    if np.any(probability < -1.0e-10) or np.any(probability > 1.0 + 1.0e-10):
        raise AssertionError("Born probability left [0, 1]")
    return np.clip(probability, 0.0, 1.0)


def _structured_double_excitation_selected_kernel(
    target_state: np.ndarray,
    povm_operators: np.ndarray,
    *,
    hom_indistinguishability: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized n=2 selected-mode kernel matching the sidecar physics."""

    target = np.asarray(target_state, dtype=np.complex128)
    projectors = np.asarray(povm_operators, dtype=np.complex128)
    xi = np.asarray(hom_indistinguishability, dtype=np.float64)
    if target.ndim != 2 or projectors.shape[1:] != (
        target.shape[1],
        target.shape[1],
    ):
        raise ValueError("target/projector dimensions differ for structured n=2")
    if xi.shape != (target.shape[0],) or np.any((xi < 0.0) | (xi > 1.0)):
        raise ValueError("HOM sensitivity must contain one [0,1] value per row")
    target = target / np.maximum(
        np.linalg.norm(target, axis=-1, keepdims=True),
        np.finfo(np.float64).eps,
    )
    qubit_count = target.shape[1]
    selected = np.real(
        np.einsum("bi,oij,bj->bo", target.conj(), projectors, target)
    ).clip(0.0, 1.0)
    population = np.square(np.abs(target))
    projector_diagonal = np.real(np.diagonal(projectors, axis1=-2, axis2=-1))
    dephased_q = np.einsum("bi,oi->bo", population, projector_diagonal).clip(
        0.0,
        1.0,
    )
    same_q = selected
    same_p2 = np.square(selected)
    orthogonal_selected = (1.0 - selected) / float(qubit_count - 1)
    split_q = 0.5 * (selected + orthogonal_selected)
    split_p2 = (1.0 + xi[:, None]) * selected * orthogonal_selected
    diagonal_overlap = np.einsum(
        "bi,oi->bo",
        np.square(population),
        np.square(projector_diagonal),
    )
    dephased_p2 = 2.0 * np.square(dephased_q) - diagonal_overlap
    one_body = np.stack((same_q, split_q, dephased_q), axis=2)
    selected_two = np.stack((same_p2, split_p2, dephased_p2), axis=2)
    selected_one = 2.0 * (one_body - selected_two)
    selected_zero = 1.0 - selected_one - selected_two
    kernel = np.stack((selected_zero, selected_one, selected_two), axis=-1)
    if np.any(kernel < -5.0e-8) or np.any(kernel > 1.0 + 5.0e-8):
        raise AssertionError("synthetic structured n=2 kernel left its simplex")
    kernel = np.clip(kernel, 0.0, 1.0)
    kernel /= np.maximum(kernel.sum(axis=-1, keepdims=True), 1.0e-15)
    component_fidelity = np.stack(
        (
            np.ones(target.shape[0], dtype=np.float64),
            np.full(target.shape[0], 0.5, dtype=np.float64),
            np.square(population).sum(axis=-1),
        ),
        axis=-1,
    )
    return kernel, component_fidelity


def generate_dlcz_noise_batch(
    *,
    system_type: str,
    batch_size: int,
    seed: int,
    family_id: str = "measured_domain_train",
    exposure_mode: str = "mixed",
    intervention: Optional[DLCZIntervention] = None,
    allow_holdout_evaluation: bool = False,
) -> Dict[str, Any]:
    """Generate one deterministic, provenance-bound DLCZ synthetic batch."""

    if system_type not in SYSTEM_OPERATION_COUNTS:
        raise ValueError("system_type must be 3qubit or 4qubit")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if exposure_mode not in VALID_EXPOSURE_MODES:
        raise ValueError(f"exposure_mode must be one of {sorted(VALID_EXPOSURE_MODES)}")
    family = resolve_dlcz_noise_family(
        family_id,
        allow_holdout_evaluation=allow_holdout_evaluation,
    )
    intervention = intervention or DLCZIntervention()
    intervention.validate()
    maximum_chi = family.chi_upper * float(intervention.write_power_scale)
    if maximum_chi >= 0.25:
        raise ValueError("write-power intervention leaves the registered weak-write regime")

    operation_count = int(SYSTEM_OPERATION_COUNTS[system_type])
    batch_size = int(batch_size)
    gate_rate_hz = float(
        family.gate_rate_hz_by_system[0 if system_type == "3qubit" else 1]
    )
    povm_contract = registered_povm_contract(system_type)
    qubit_count = int(povm_contract["qubit_count"])
    family_seed = _family_seed(seed, family.family_hash)
    seed_sequence = np.random.SeedSequence(family_seed)
    (
        latent_seed,
        exposure_seed,
        single_seed,
        p2_seed,
        tail_seed,
        accidental_seed,
        dark_seed,
        off_window_seed,
        double_excitation_seed,
        signal_seed,
        fpga_clock_seed,
    ) = (np.random.default_rng(child) for child in seed_sequence.spawn(11))

    base_chi = np.exp(
        latent_seed.uniform(
            np.log(family.chi_lower),
            np.log(family.chi_upper),
            size=batch_size,
        )
    )
    chi = base_chi * float(intervention.write_power_scale)
    excitation = geometric_excitation_probabilities(chi)
    external_herald_probability = 1.0 - excitation[:, 0]
    excitation_given_external_herald = excitation[:, 1:] / np.maximum(
        external_herald_probability[:, None], np.finfo(np.float64).eps
    )
    base_brightness = latent_seed.gamma(
        shape=1.0 / family.brightness_dispersion,
        scale=family.brightness_dispersion,
        size=batch_size,
    )
    brightness = base_brightness * float(intervention.source_brightness_scale)
    detector_efficiency = _sample_detector_efficiency(family, batch_size, latent_seed)
    detector_efficiency = np.clip(
        detector_efficiency * float(intervention.detector_efficiency_scale),
        1.0e-5,
        1.0,
    )
    global_optical_path_efficiency = np.full(
        batch_size,
        float(family.global_optical_path_efficiency),
        dtype=np.float64,
    )
    effective_read_efficiency = (
        detector_efficiency * global_optical_path_efficiency[:, None]
    )
    dark_probability = _sample_dark_probability(family, batch_size, latent_seed)
    dark_probability = np.clip(
        dark_probability * float(intervention.dark_probability_scale),
        0.0,
        0.05,
    )
    dephasing = latent_seed.uniform(
        family.dephasing_lower, family.dephasing_upper, size=batch_size
    )
    crosstalk = latent_seed.uniform(
        family.crosstalk_lower, family.crosstalk_upper, size=batch_size
    )
    clean_state_truth = _sample_clean_rho(
        batch_size=batch_size,
        qubit_count=qubit_count,
        dephasing=dephasing,
        crosstalk=crosstalk,
        rng=latent_seed,
    )
    clean_rho = clean_state_truth["clean_rho"]
    pre_channel_state = clean_state_truth["pre_channel_state"]
    pre_channel_rho = clean_state_truth["pre_channel_rho"]
    target_amplitudes = np.zeros((batch_size, 4), dtype=np.float64)
    target_phases_rad = np.zeros((batch_size, 4), dtype=np.float64)
    target_mask = np.zeros((batch_size, 4), dtype=np.float32)
    target_amplitudes[:, :qubit_count] = clean_state_truth["target_amplitudes"]
    target_phases_rad[:, :qubit_count] = clean_state_truth["target_phases_rad"]
    target_mask[:, :qubit_count] = 1.0
    target_truth_hash = _batch_content_hash(
        {
            "semantic": "exogenous_pre_channel_w_target_v1",
            "target_amplitudes": target_amplitudes,
            "target_phases_rad": target_phases_rad,
            "target_mask": target_mask,
            "pre_channel_state": pre_channel_state,
            "pre_channel_rho": pre_channel_rho,
        }
    )
    born_probability = born_probabilities(clean_rho, povm_contract["operators"])
    operation_mode_probability = np.broadcast_to(
        np.asarray(povm_contract["basis_mode_probability"], dtype=np.float64)[None, :, :],
        (batch_size, operation_count, qubit_count),
    ).copy()
    # The AOD analyzer first combines the selected idler into one optical
    # output.  A fixed two-layer beamsplitter then fans that one output to four
    # threshold SPADs.  Every analyzer mode therefore shares one splitter
    # simplex; there is no learnable Q-mode-to-detector routing truth.
    splitter_probability = np.full((batch_size, 4), 0.25, dtype=np.float64)
    mode_to_detector_routing = np.broadcast_to(
        splitter_probability[:, None, :],
        (batch_size, qubit_count, 4),
    ).copy()
    routing_truth_hash = _array_contract_hash(
        mode_to_detector_routing,
        {
            "system_type": system_type,
            "qubit_count": qubit_count,
            "detector_count": 4,
            "semantic": (
                "single_combined_aod_idler_output_then_fixed_two_layer_"
                "beamsplitter_to_four_read_detectors"
            ),
        },
    )
    operation_detector_probability = np.broadcast_to(
        splitter_probability[:, None, :],
        (batch_size, operation_count, 4),
    ).copy()
    if intervention.hbt_beamsplitter_transmission is not None:
        operation_detector_probability = _apply_hbt_beamsplitter(
            operation_detector_probability,
            float(intervention.hbt_beamsplitter_transmission),
        )
    n2_nuisance_fraction = double_excitation_seed.uniform(
        *family.n2_nuisance_fraction_range,
        size=batch_size,
    )
    n2_split_fraction = double_excitation_seed.uniform(
        *family.n2_split_fraction_of_nuisance_range,
        size=batch_size,
    )
    n2_hom_indistinguishability = double_excitation_seed.uniform(
        *family.n2_hom_indistinguishability_range,
        size=batch_size,
    )
    n2_component_probability = np.stack(
        (
            1.0 - n2_nuisance_fraction,
            n2_nuisance_fraction * n2_split_fraction,
            n2_nuisance_fraction * (1.0 - n2_split_fraction),
        ),
        axis=-1,
    )
    (
        n2_component_selected_probability,
        n2_component_one_body_fidelity,
    ) = _structured_double_excitation_selected_kernel(
        pre_channel_state,
        povm_contract["operators"],
        hom_indistinguishability=n2_hom_indistinguishability,
    )
    n2_selected_probability = np.einsum(
        "bc,bock->bok",
        n2_component_probability,
        n2_component_selected_probability,
    )
    n2_one_body_target_fidelity = np.einsum(
        "bc,bc->b",
        n2_component_probability,
        n2_component_one_body_fidelity,
    )

    exposure_s = exposure_seed.choice(
        np.asarray(EXPOSURE_LEVELS_S, dtype=np.float64),
        size=batch_size,
        replace=True,
    )
    exposure_known_mask = _exposure_known_mask(exposure_mode, batch_size, exposure_seed)
    observed_exposure_s = np.where(exposure_known_mask > 0.5, exposure_s, np.nan)
    gate_exposure = np.rint(
        exposure_s[:, None] * gate_rate_hz
    ).astype(np.int64)
    gate_exposure = np.broadcast_to(
        gate_exposure, (batch_size, operation_count)
    ).copy()
    observed_signal_counts = signal_seed.binomial(
        gate_exposure,
        np.broadcast_to(chi[:, None], gate_exposure.shape),
    ).astype(np.int64)
    gate_column = gate_exposure[..., None].astype(np.float64)
    brightness_column = brightness[:, None, None]

    photon_components = geometric_detected_pattern_components(
        operation_detector_probability,
        effective_read_efficiency[:, None, :],
        chi[:, None],
    )
    single_gate_pattern = born_probability[..., None] * photon_components["n1"]
    selected_one_pattern = detected_pattern_probabilities_for_n(
        operation_detector_probability,
        effective_read_efficiency[:, None, :],
        1,
    )
    selected_two_pattern = detected_pattern_probabilities_for_n(
        operation_detector_probability,
        effective_read_efficiency[:, None, :],
        2,
    )
    p2_gate_pattern = excitation[:, 2, None, None] * (
        n2_selected_probability[:, :, 1, None] * selected_one_pattern
        + n2_selected_probability[:, :, 2, None] * selected_two_pattern
    )
    tail_gate_pattern = born_probability[..., None] * photon_components["tail"]
    for pattern in (single_gate_pattern, p2_gate_pattern, tail_gate_pattern):
        pattern[..., 0] = 0.0
    source_gate_pattern = single_gate_pattern + p2_gate_pattern + tail_gate_pattern
    source_singles_probability = _pattern_detector_marginals(source_gate_pattern)
    source_singles_rate_per_brightness_hz = (
        gate_rate_hz * source_singles_probability
    )
    background_rate = np.broadcast_to(
        np.asarray(family.background_singles_rate_hz, dtype=np.float64)[None, None, :],
        (batch_size, operation_count, 4),
    )
    accidental_components = _accidental_pair_probability_components(
        source_singles_rate_per_brightness_hz,
        background_rate,
        brightness[:, None, None],
        window_s=family.accidental_window_s
        * float(intervention.accidental_window_scale),
        gate_rate_hz=gate_rate_hz,
    )
    accidental_pair_probability = accidental_components["total"]
    accidental_pattern = np.zeros_like(source_gate_pattern)
    accidental_pattern[..., PAIR_PATTERN_INDICES] = accidental_pair_probability

    dark_pattern = _independent_binary_pattern_probability(dark_probability)
    dark_pattern[:, 0] = 0.0
    dark_pattern = np.broadcast_to(
        dark_pattern[:, None, :], (batch_size, operation_count, 16)
    )
    single_expected_patterns = gate_column * brightness_column * single_gate_pattern
    p2_expected_patterns = gate_column * brightness_column * p2_gate_pattern
    tail_expected_patterns = gate_column * brightness_column * tail_gate_pattern
    accidental_expected_patterns = gate_column * accidental_pattern
    dark_expected_patterns = gate_column * dark_pattern
    raw_expected_patterns = (
        single_expected_patterns
        + p2_expected_patterns
        + tail_expected_patterns
        + accidental_expected_patterns
        + dark_expected_patterns
    )

    single_pattern_counts = single_seed.poisson(single_expected_patterns).astype(np.int64)
    p2_pattern_counts = p2_seed.poisson(p2_expected_patterns).astype(np.int64)
    tail_pattern_counts = tail_seed.poisson(tail_expected_patterns).astype(np.int64)
    accidental_pattern_counts = accidental_seed.poisson(
        accidental_expected_patterns
    ).astype(np.int64)
    dark_pattern_counts = dark_seed.poisson(dark_expected_patterns).astype(np.int64)
    raw_pattern_counts_operation = (
        single_pattern_counts
        + p2_pattern_counts
        + tail_pattern_counts
        + accidental_pattern_counts
        + dark_pattern_counts
    )
    noise_expected_patterns = (
        p2_expected_patterns
        + tail_expected_patterns
        + accidental_expected_patterns
        + dark_expected_patterns
    )
    noise_pattern_counts = (
        p2_pattern_counts
        + tail_pattern_counts
        + accidental_pattern_counts
        + dark_pattern_counts
    )

    off_window_gate_exposure = np.rint(
        gate_exposure.astype(np.float64)
        * family.off_window_gate_fraction
        * float(intervention.off_window_gate_scale)
    ).astype(np.int64)
    off_dark_probability = 1.0 - np.power(
        1.0 - dark_probability,
        float(intervention.off_window_width_scale),
    )
    off_dark_pattern = _independent_binary_pattern_probability(off_dark_probability)
    off_dark_pattern[:, 0] = 0.0
    off_dark_pattern = np.broadcast_to(
        off_dark_pattern[:, None, :], (batch_size, operation_count, 16)
    )
    off_accidental_pair_probability = _accidental_pair_probability(
        background_rate,
        window_s=family.off_window_width_s
        * float(intervention.off_window_width_scale),
        gate_rate_hz=gate_rate_hz,
    )
    off_accidental_pattern = np.zeros_like(source_gate_pattern)
    off_accidental_pattern[..., PAIR_PATTERN_INDICES] = (
        off_accidental_pair_probability
    )
    off_window_expected_patterns = off_window_gate_exposure[..., None] * (
        off_accidental_pattern + off_dark_pattern
    )
    off_window_pattern_counts_operation = off_window_seed.poisson(
        off_window_expected_patterns
    ).astype(np.int64)

    per_gate_record_probability = (
        brightness_column * source_gate_pattern + accidental_pattern + dark_pattern
    ).sum(axis=-1)
    collision_upper_bound = 0.5 * np.square(per_gate_record_probability)
    if float(np.max(per_gate_record_probability)) >= 0.15:
        raise ValueError(
            "DLCZ marked-Poisson rare-event approximation exceeded 0.15 per gate"
        )

    raw_pattern_counts_global = raw_pattern_counts_operation.sum(axis=1)
    off_window_pattern_counts_global = off_window_pattern_counts_operation.sum(axis=1)
    raw_operation_any_click_counts = raw_pattern_counts_operation[..., 1:].sum(axis=-1)
    raw_operation_one_click_counts = raw_pattern_counts_operation[
        ..., ONE_CLICK_PATTERN_INDICES
    ].sum(axis=-1)
    # Reconstruct the actual FPGA read semantics.  Exactly-one detector sets a
    # sticky operation bit over 19 accepted 100 MHz clocks; exact 2/3/4 click
    # events increment global counters independently at every clock.
    background_clock_click_probability = 1.0 - (
        (1.0 - dark_probability[:, None, :])
        * np.exp(-background_rate * READ_MULTI_SIMULTANEITY_CLOCK_S)
    )
    background_clock_pattern = _independent_binary_pattern_probability(
        background_clock_click_probability
    )
    single_source_pattern = (
        born_probability[..., None]
        * detected_pattern_probabilities_for_n(
            operation_detector_probability,
            effective_read_efficiency[:, None, :],
            1,
        )
    )
    single_source_pattern[..., 0] += 1.0 - single_source_pattern.sum(axis=-1)
    p2_source_pattern = (
        n2_selected_probability[:, :, 1, None] * selected_one_pattern
        + n2_selected_probability[:, :, 2, None] * selected_two_pattern
    )
    p2_source_pattern[..., 0] += 1.0 - p2_source_pattern.sum(axis=-1)
    conditional_tail_pattern = photon_components["tail"] / np.maximum(
        excitation[:, 3, None, None],
        np.finfo(np.float64).eps,
    )
    tail_source_pattern = born_probability[..., None] * conditional_tail_pattern
    tail_source_pattern[..., 0] += 1.0 - tail_source_pattern.sum(axis=-1)
    single_clock_pattern = _combine_independent_patterns(
        single_source_pattern,
        background_clock_pattern,
    )
    p2_clock_pattern = _combine_independent_patterns(
        p2_source_pattern,
        background_clock_pattern,
    )
    tail_clock_pattern = _combine_independent_patterns(
        tail_source_pattern,
        background_clock_pattern,
    )
    clock_count = READ_WINDOW_ACCEPTED_CLOCK_COUNT
    conditional_sector = excitation_given_external_herald[:, None, :]
    signal_clock_pattern_expectation = (
        conditional_sector[..., 0, None]
        * (single_clock_pattern + (clock_count - 1) * background_clock_pattern)
        + conditional_sector[..., 1, None]
        * (p2_clock_pattern + (clock_count - 1) * background_clock_pattern)
        + conditional_sector[..., 2, None]
        * (tail_clock_pattern + (clock_count - 1) * background_clock_pattern)
    )
    expected_fpga_clock_patterns_per_trial = (
        (1.0 - chi[:, None, None]) * clock_count * background_clock_pattern
        + chi[:, None, None] * signal_clock_pattern_expectation
    )
    if not np.allclose(
        expected_fpga_clock_patterns_per_trial.sum(axis=-1),
        clock_count,
        atol=1.0e-10,
        rtol=1.0e-10,
    ):
        raise AssertionError("synthetic FPGA clock patterns do not conserve 19 clocks")

    q_background = background_clock_pattern[..., ONE_CLICK_PATTERN_INDICES].sum(axis=-1)
    q_single = single_clock_pattern[..., ONE_CLICK_PATTERN_INDICES].sum(axis=-1)
    q_p2 = p2_clock_pattern[..., ONE_CLICK_PATTERN_INDICES].sum(axis=-1)
    q_tail = tail_clock_pattern[..., ONE_CLICK_PATTERN_INDICES].sum(axis=-1)
    background_singleton_probability = 1.0 - np.power(1.0 - q_background, clock_count)
    singleton_by_sector = np.stack(
        (
            1.0 - (1.0 - q_single) * np.power(1.0 - q_background, clock_count - 1),
            1.0 - (1.0 - q_p2) * np.power(1.0 - q_background, clock_count - 1),
            1.0 - (1.0 - q_tail) * np.power(1.0 - q_background, clock_count - 1),
        ),
        axis=-1,
    )
    singleton_given_signal_probability = (
        conditional_sector * singleton_by_sector
    ).sum(axis=-1)
    fpga_coin_probability_by_sector = (
        chi[:, None, None] * conditional_sector * singleton_by_sector
    )
    fpga_coin_expected_counts_by_sector = (
        gate_exposure[..., None] * fpga_coin_probability_by_sector
    )
    fpga_single_expected_counts = fpga_coin_expected_counts_by_sector[..., 0]
    fpga_p2_expected_counts = fpga_coin_expected_counts_by_sector[..., 1]
    fpga_tail_expected_counts = fpga_coin_expected_counts_by_sector[..., 2]
    fpga_raw_expected_counts = fpga_coin_expected_counts_by_sector.sum(axis=-1)
    fpga_zero_local_component_counts = np.zeros_like(fpga_raw_expected_counts)
    observed_coin_counts = signal_seed.binomial(
        observed_signal_counts,
        singleton_given_signal_probability,
    ).astype(np.int64)
    attribution_mass = (
        conditional_sector * singleton_by_sector
    ) / singleton_given_signal_probability[..., None].clip(min=1.0e-12)
    fpga_single_operation_counts = signal_seed.binomial(
        observed_coin_counts,
        attribution_mass[..., 0].clip(0.0, 1.0),
    ).astype(np.int64)
    fpga_remaining_operation_counts = observed_coin_counts - fpga_single_operation_counts
    fpga_p2_conditional_probability = (
        attribution_mass[..., 1]
        / (attribution_mass[..., 1] + attribution_mass[..., 2]).clip(min=1.0e-12)
    )
    fpga_p2_operation_counts = signal_seed.binomial(
        fpga_remaining_operation_counts,
        fpga_p2_conditional_probability.clip(0.0, 1.0),
    ).astype(np.int64)
    fpga_tail_operation_counts = (
        fpga_remaining_operation_counts - fpga_p2_operation_counts
    )
    observed_no_signal_singleton_counts = signal_seed.binomial(
        gate_exposure - observed_signal_counts,
        background_singleton_probability,
    ).astype(np.int64)
    observed_singleton_idler_counts = (
        observed_coin_counts + observed_no_signal_singleton_counts
    )
    observed_operation_click_counts = observed_coin_counts
    observed_operation_one_click_counts = observed_singleton_idler_counts
    fpga_clock_pattern_counts_operation = fpga_clock_seed.poisson(
        gate_exposure[..., None] * expected_fpga_clock_patterns_per_trial
    ).astype(np.int64)
    fpga_clock_pattern_counts_global = fpga_clock_pattern_counts_operation.sum(axis=1)
    global_multi = _global_multi_click_observation(fpga_clock_pattern_counts_global)
    global_multi_scope_hash = _json_hash(
        {
            "schema": DLCZ_GLOBAL_MULTI_SCHEMA,
            "scope": "global",
            "field_scopes": global_multi["field_scopes"],
            "pair_labels": PAIR_LABELS,
        }
    )
    n2_physics_truth_hash = _batch_content_hash(
        {
            "component_probability": n2_component_probability,
            "component_selected_probability": n2_component_selected_probability,
            "selected_probability": n2_selected_probability,
            "component_one_body_fidelity": n2_component_one_body_fidelity,
            "one_body_target_fidelity": n2_one_body_target_fidelity,
            "hom_indistinguishability": n2_hom_indistinguishability,
        }
    )

    contract_payload = {
        "schema": DLCZ_NOISE_SYNTHETIC_SCHEMA,
        "family_hash": family.family_hash,
        "intervention_hash": intervention.intervention_hash,
        "system_type": system_type,
        "qubit_count": qubit_count,
        "operation_count": operation_count,
        "basis_config_hash": povm_contract["basis_config_hash"],
        "povm_hash": povm_contract["povm_hash"],
        "routing_truth_hash": routing_truth_hash,
        "target_truth_hash": target_truth_hash,
        "herald_semantic": HERALD_SEMANTIC,
        "exposure_mode": exposure_mode,
        "global_multi_scope_hash": global_multi_scope_hash,
        "n2_physics_truth_hash": n2_physics_truth_hash,
    }
    batch: Dict[str, Any] = {
        "schema": DLCZ_NOISE_SYNTHETIC_SCHEMA,
        "system_type": system_type,
        "qubit_count": qubit_count,
        "detector_count": 4,
        "operation_count": operation_count,
        "family_id": family.family_id,
        "family_role": family.role,
        "family_hash": family.family_hash,
        "intervention": asdict(intervention),
        "intervention_hash": intervention.intervention_hash,
        "contract_hash": _json_hash(contract_payload),
        "seed": int(seed),
        "seed_namespace_hash": _json_hash(
            {"seed": int(seed), "family_hash": family.family_hash}
        ),
        "label_provenance": "synthetic_truth_only",
        "real_label_claim_allowed": False,
        "production_promotion_allowed": False,
        "real_supervision_mask": np.zeros(batch_size, dtype=np.float32),
        "synthetic_truth_mask": np.ones(batch_size, dtype=np.float32),
        "herald_semantic": HERALD_SEMANTIC,
        "four_read_detectors_include_herald": False,
        "external_herald_efficiency_assumption": 1.0,
        "external_herald_dark_probability_assumption": 0.0,
        "external_herald_observation_semantic": (
            "binary_spad_click_in_200ns_window_no_photon_number_resolution"
        ),
        "fpga_read_window_nominal_s": 200.0e-9,
        "fpga_read_window_accepted_clock_count": READ_WINDOW_ACCEPTED_CLOCK_COUNT,
        "fpga_read_window_accepted_support_s": READ_WINDOW_ACCEPTED_SUPPORT_S,
        "fpga_read_singleton_semantic": (
            "sticky_or_of_exactly_one_read_spad_over_19_accepted_clocks"
        ),
        "fpga_global_multi_semantic": (
            "per_clock_exact_2_3_4_spad_counts_summed_over_operations_and_trials"
        ),
        "external_herald_number_resolving": False,
        "external_herald_eventwise_excitation_label_available": False,
        "external_herald_signal_bin_semantic": (
            "aggregate_binary_spad_click_count_across_write_gates"
        ),
        "external_herald_direct_real_n_sector_information": False,
        "external_herald_efficiency_sensitivity_role": (
            "fixed_acquisition_bias_sensitivity_not_eventwise_number_resolution"
        ),
        "basis_config_hash": povm_contract["basis_config_hash"],
        "povm_hash": povm_contract["povm_hash"],
        "routing_truth_hash": routing_truth_hash,
        "target_truth_hash": target_truth_hash,
        "basis_phase_sign": BASIS_PHASE_SIGN,
        "basis_phase_convention": BASIS_PHASE_CONVENTION,
        "registered_basis_labels": povm_contract["basis_labels"],
        "registered_povm_operators": povm_contract["operators"].copy(),
        "registered_measurement_operator_semantic": (
            "rank_one_projectors_evaluated_individually_not_one_collectively_normalized_povm"
        ),
        "single_excitation_born_semantic": "probability=trace(projector_times_clean_rho)",
        "p2_tail_born_semantic": (
            "structured_n2_selected_mode_kernel_tail_remains_geometric_forward"
        ),
        "two_particle_povm_calibrated": False,
        "sector_specific_fock_measurement_unitaries_available": True,
        "fock_v2_selection_evidence_allowed": True,
        "n2_physics_truth_hash": n2_physics_truth_hash,
        "exposure_mode": exposure_mode,
        "observed_exposure_s": observed_exposure_s.astype(np.float64),
        "exposure_known_mask": exposure_known_mask.astype(np.float32),
        "synthetic_truth_exposure_s": exposure_s.astype(np.float64),
        "gate_exposure": gate_exposure,
        "off_window_gate_exposure": off_window_gate_exposure,
        "pattern_bits": PATTERNS.astype(np.int8),
        "raw_pattern_scope": "global",
        "raw_pattern_counts_global": raw_pattern_counts_global,
        "synthetic_fpga_clock_pattern_counts_global": (
            fpga_clock_pattern_counts_global
        ),
        "synthetic_fpga_clock_pattern_counts_operation": (
            fpga_clock_pattern_counts_operation
        ),
        "synthetic_truth_fpga_expected_clock_patterns_per_trial": (
            expected_fpga_clock_patterns_per_trial
        ),
        "off_window_pattern_scope": "global",
        "off_window_pattern_counts_global": off_window_pattern_counts_global,
        "observed_operation_click_counts": observed_operation_click_counts,
        "observed_operation_one_click_counts": observed_operation_one_click_counts,
        "raw_operation_any_click_counts_diagnostic": raw_operation_any_click_counts,
        "raw_operation_one_click_counts_diagnostic": raw_operation_one_click_counts,
        "synthetic_observed_singleton_idler_counts": (
            observed_singleton_idler_counts
        ),
        "synthetic_observed_coin_counts": observed_coin_counts,
        "synthetic_fpga_joint_observation_semantic": (
            "R_read_trials_S_herald_I_singleton_read_clock_C_signal_and_singleton"
        ),
        "synthetic_truth_singleton_given_signal_probability": (
            singleton_given_signal_probability
        ),
        "synthetic_truth_singleton_given_no_signal_probability": (
            background_singleton_probability
        ),
        "synthetic_observed_signal_counts": observed_signal_counts,
        "synthetic_observed_signal_count_semantic": (
            "binary_threshold_spad_herald_clicks_among_read_trials"
        ),
        "global_multi_click_observation": global_multi,
        "global_multi_scope_hash": global_multi_scope_hash,
        "synthetic_truth_base_chi": base_chi,
        "synthetic_truth_chi": chi,
        "synthetic_truth_operation_excitation_rate_chi": np.broadcast_to(
            chi[:, None],
            (batch_size, operation_count),
        ).copy(),
        "synthetic_truth_operation_conditional_p2_probability": np.broadcast_to(
            ((1.0 - chi) * chi)[:, None],
            (batch_size, operation_count),
        ).copy(),
        "synthetic_truth_operation_conditional_tail_probability": np.broadcast_to(
            np.square(chi)[:, None],
            (batch_size, operation_count),
        ).copy(),
        "synthetic_truth_operation_excitation_rate_semantic": (
            "chi_estimated_by_signal_clicks_over_read_trials_and_bounded_to_family_support"
        ),
        "synthetic_truth_excitation_probabilities": excitation,
        "synthetic_truth_external_herald_probability": external_herald_probability,
        "synthetic_truth_excitation_given_external_herald": (
            excitation_given_external_herald
        ),
        "synthetic_truth_base_brightness": base_brightness,
        "synthetic_truth_brightness": brightness,
        "synthetic_truth_nb2_dispersion": np.full(
            batch_size, family.brightness_dispersion, dtype=np.float64
        ),
        "synthetic_truth_nb2_dispersion_semantic": (
            "gamma_source_brightness_only_not_marginal_raw_count_nb2"
        ),
        "synthetic_truth_detector_efficiency": detector_efficiency,
        "synthetic_truth_global_optical_path_efficiency": (
            global_optical_path_efficiency
        ),
        "synthetic_truth_effective_read_efficiency": effective_read_efficiency,
        "synthetic_truth_gate_rate_hz": np.full(
            batch_size,
            gate_rate_hz,
            dtype=np.float64,
        ),
        "synthetic_truth_fpga_coincidence_window_s": np.full(
            batch_size,
            family.accidental_window_s,
            dtype=np.float64,
        ),
        "synthetic_truth_dark_probability": dark_probability,
        "synthetic_truth_dephasing": dephasing,
        "synthetic_truth_crosstalk": crosstalk,
        "synthetic_truth_target_amplitudes": target_amplitudes,
        "synthetic_truth_target_phases_rad": target_phases_rad,
        "synthetic_truth_target_mask": target_mask,
        "synthetic_truth_pre_channel_state": pre_channel_state,
        "synthetic_truth_pre_channel_rho": pre_channel_rho,
        "synthetic_truth_clean_rho": clean_rho,
        "synthetic_truth_born_probability": born_probability,
        "synthetic_truth_n2_component_names": (
            "same_collective_mode_fock2",
            "target_plus_orthogonal_schmidt_mode",
            "occupation_dephased_fock2",
        ),
        "synthetic_truth_n2_component_probability": n2_component_probability,
        "synthetic_truth_n2_component_selected_photon_probability": (
            n2_component_selected_probability
        ),
        "synthetic_truth_n2_selected_photon_probability": n2_selected_probability,
        "synthetic_truth_n2_component_one_body_target_fidelity": (
            n2_component_one_body_fidelity
        ),
        "synthetic_truth_n2_target_one_body_fidelity": (
            n2_one_body_target_fidelity
        ),
        "synthetic_truth_n2_phase_violation_probability": (
            1.0 - n2_one_body_target_fidelity
        ),
        "synthetic_truth_n2_hom_indistinguishability": (
            n2_hom_indistinguishability
        ),
        "synthetic_truth_n2_hom_point_identified_in_real_data": False,
        "synthetic_truth_mode_to_detector_routing": mode_to_detector_routing,
        "synthetic_truth_operation_mode_probability": operation_mode_probability,
        "synthetic_truth_operation_mode_probability_semantic": (
            "registered_read_basis_population_in_q_quantum_modes"
        ),
        "synthetic_truth_operation_detector_probability": (
            operation_detector_probability
        ),
        "synthetic_truth_accidental_pair_probability": accidental_pair_probability,
        "synthetic_truth_accidental_source_source_probability": accidental_components[
            "source_source"
        ],
        "synthetic_truth_accidental_source_background_probability": accidental_components[
            "source_background"
        ],
        "synthetic_truth_accidental_background_background_probability": accidental_components[
            "background_background"
        ],
        "synthetic_truth_off_window_accidental_pair_probability": (
            off_accidental_pair_probability
        ),
        "synthetic_truth_component_pattern_scope": "operation_synthetic_truth_only",
        "synthetic_truth_operation_count_semantic": (
            "fpga_coin_signal_and_sticky_exactly_one_idler_attributed_by_"
            "heralded_source_sector"
        ),
        "synthetic_truth_single_expected_pattern_counts": single_expected_patterns,
        "synthetic_truth_p2_expected_pattern_counts": p2_expected_patterns,
        "synthetic_truth_tail_expected_pattern_counts": tail_expected_patterns,
        "synthetic_truth_dark_expected_pattern_counts": dark_expected_patterns,
        "synthetic_truth_accidental_expected_pattern_counts": (
            accidental_expected_patterns
        ),
        "synthetic_truth_single_pattern_counts": single_pattern_counts,
        "synthetic_truth_p2_pattern_counts": p2_pattern_counts,
        "synthetic_truth_tail_pattern_counts": tail_pattern_counts,
        "synthetic_truth_dark_pattern_counts": dark_pattern_counts,
        "synthetic_truth_accidental_pattern_counts": accidental_pattern_counts,
        "synthetic_truth_clean_expected_counts": fpga_single_expected_counts,
        "synthetic_truth_p2_expected_counts": fpga_p2_expected_counts,
        "synthetic_truth_tail_expected_counts": fpga_tail_expected_counts,
        "synthetic_truth_dark_expected_counts": fpga_zero_local_component_counts,
        "synthetic_truth_accidental_expected_counts": fpga_zero_local_component_counts,
        "synthetic_truth_noise_expected_counts": (
            fpga_p2_expected_counts + fpga_tail_expected_counts
        ),
        "synthetic_truth_raw_expected_counts": fpga_raw_expected_counts,
        "synthetic_truth_clean_operation_counts": fpga_single_operation_counts,
        "synthetic_truth_p2_operation_counts": fpga_p2_operation_counts,
        "synthetic_truth_tail_operation_counts": fpga_tail_operation_counts,
        "synthetic_truth_operation_noise_counts": (
            fpga_p2_operation_counts + fpga_tail_operation_counts
        ),
        "synthetic_truth_marked_poisson_raw_expected_counts_diagnostic": (
            raw_expected_patterns[..., 1:].sum(axis=-1)
        ),
        "synthetic_truth_off_window_expected_pattern_counts_global": (
            off_window_expected_patterns.sum(axis=1)
        ),
        "rare_event_collision_upper_bound": collision_upper_bound,
        "rare_event_approximation": "mutually_exclusive_marked_poisson_components_with_collision_bound",
        "count_likelihood": "conditional_marked_poisson_with_gamma_source_brightness",
        "global_multi_was_distributed_to_operations": False,
    }
    batch["batch_hash"] = _batch_content_hash(batch)
    return batch


def generate_dlcz_intervention_pair(
    *,
    system_type: str,
    batch_size: int,
    seed: int,
    intervention: DLCZIntervention,
    family_id: str = "measured_domain_train",
    exposure_mode: str = "known",
    allow_holdout_evaluation: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Generate baseline/intervention batches with common latent random streams."""

    return {
        "baseline": generate_dlcz_noise_batch(
            system_type=system_type,
            batch_size=batch_size,
            seed=seed,
            family_id=family_id,
            exposure_mode=exposure_mode,
            intervention=DLCZIntervention(),
            allow_holdout_evaluation=allow_holdout_evaluation,
        ),
        "intervened": generate_dlcz_noise_batch(
            system_type=system_type,
            batch_size=batch_size,
            seed=seed,
            family_id=family_id,
            exposure_mode=exposure_mode,
            intervention=intervention,
            allow_holdout_evaluation=allow_holdout_evaluation,
        ),
    }


def inject_semisynthetic_dlcz_noise(
    real_observation_counts: Any,
    synthetic_truth_operation_noise_counts: Any,
    *,
    seed: int,
    noise_scale: float = 1.0,
    source_scope: str = "operation_synthetic_truth",
) -> Dict[str, Any]:
    """Purely inject operation-native synthetic noise into real observations.

    The function refuses global inputs: a global multi-click total cannot be
    expanded into operation bins.  The output is a semi-synthetic observation,
    never a fabricated real clean label.
    """

    if source_scope != "operation_synthetic_truth":
        raise ValueError(
            "semi-synthetic injection requires native operation synthetic truth; "
            "global multi-click totals cannot be distributed"
        )
    real_counts = np.asarray(real_observation_counts)
    noise_counts = np.asarray(synthetic_truth_operation_noise_counts)
    if real_counts.shape != noise_counts.shape or real_counts.ndim not in (1, 2):
        raise ValueError("real and synthetic operation counts must have identical 1D/2D shapes")
    if (
        np.any(~np.isfinite(real_counts))
        or np.any(~np.isfinite(noise_counts))
        or np.any(real_counts < 0)
        or np.any(noise_counts < 0)
    ):
        raise ValueError("semi-synthetic counts must be finite and non-negative")
    if not np.allclose(real_counts, np.rint(real_counts)) or not np.allclose(
        noise_counts, np.rint(noise_counts)
    ):
        raise ValueError("semi-synthetic injection requires integer count arrays")
    scale = float(noise_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("noise_scale must be finite and non-negative")
    noise_integer = np.rint(noise_counts).astype(np.int64)
    whole = int(np.floor(scale))
    fractional = scale - whole
    rng = np.random.default_rng(int(seed))
    added = whole * noise_integer
    if fractional > 0.0:
        added = added + rng.binomial(noise_integer, fractional)
    observed = np.rint(real_counts).astype(np.int64) + added
    result = {
        "schema": DLCZ_SEMISYNTHETIC_SCHEMA,
        "source_scope": source_scope,
        "source_global_multi_used": False,
        "label_provenance": "semi_synthetic_observation_not_real_clean_truth",
        "real_label_claim_allowed": False,
        "real_observation_counts": np.rint(real_counts).astype(np.int64).copy(),
        "synthetic_truth_added_noise_counts": added,
        "semi_synthetic_observation_counts": observed,
        "real_supervision_mask": np.zeros(real_counts.shape[0], dtype=np.float32),
        "seed": int(seed),
        "noise_scale": scale,
    }
    result["injection_hash"] = _batch_content_hash(result)
    return result


def build_dlcz_negative_controls(
    batch: Mapping[str, Any],
    *,
    seed: int,
) -> Dict[str, Any]:
    """Permute global multi, p2 truth, and operation mapping at fixed marginals."""

    if batch.get("schema") != DLCZ_NOISE_SYNTHETIC_SCHEMA:
        raise ValueError("negative controls require a V8 DLCZ synthetic batch")
    operation_counts = np.asarray(batch["observed_operation_click_counts"])
    p2_probability = np.asarray(
        batch["synthetic_truth_excitation_probabilities"], dtype=np.float64
    )[:, 2]
    global_multi = batch["global_multi_click_observation"]
    if not isinstance(global_multi, Mapping) or global_multi.get("scope") != "global":
        raise ValueError("global multi-click observation lost its native scope")
    batch_size, operation_count = operation_counts.shape
    rng = np.random.default_rng(int(seed))
    global_permutation = _non_identity_permutation(batch_size, rng)
    p2_permutation = _non_identity_permutation(batch_size, rng)
    latent_block_permutation = _non_identity_permutation(batch_size, rng)
    operation_permutation = _non_identity_permutation(operation_count, rng)
    permuted_global = {
        key: (
            np.asarray(value)[global_permutation].copy()
            if key in ("multi2_pairs", "multi2_count", "multi3_count", "multi4_count")
            else value
        )
        for key, value in global_multi.items()
    }
    permuted_global["scope"] = "global"
    permuted_operation_counts = operation_counts[:, operation_permutation].copy()
    target_amplitudes = np.asarray(
        batch["synthetic_truth_target_amplitudes"], dtype=np.float64
    )
    target_phases_rad = np.asarray(
        batch["synthetic_truth_target_phases_rad"], dtype=np.float64
    )
    target_mask = np.asarray(batch["synthetic_truth_target_mask"], dtype=np.float32)
    pre_channel_state = np.asarray(
        batch["synthetic_truth_pre_channel_state"], dtype=np.complex128
    )
    pre_channel_rho = np.asarray(
        batch["synthetic_truth_pre_channel_rho"], dtype=np.complex128
    )
    clean_rho = np.asarray(batch["synthetic_truth_clean_rho"], dtype=np.complex128)
    povm_operators = np.asarray(batch["registered_povm_operators"], dtype=np.complex128)
    shuffled_born_probability = born_probabilities(
        clean_rho[latent_block_permutation],
        povm_operators,
    )
    result: Dict[str, Any] = {
        "schema": DLCZ_NEGATIVE_CONTROL_SCHEMA,
        "source_batch_hash": str(batch["batch_hash"]),
        "global_multi_permutation": global_permutation,
        "global_multi_click_observation": permuted_global,
        "p2_label_permutation": p2_permutation,
        "synthetic_truth_p2_probability": p2_probability[p2_permutation].copy(),
        "latent_block_permutation": latent_block_permutation,
        "born_rho_permutation": latent_block_permutation.copy(),
        "born_pairing_control": (
            "target_pre_channel_state_clean_rho_block_permuted_at_fixed_registered_povm"
        ),
        "source_target_truth_hash": str(batch["target_truth_hash"]),
        "synthetic_truth_target_amplitudes": target_amplitudes[
            latent_block_permutation
        ].copy(),
        "synthetic_truth_target_phases_rad": target_phases_rad[
            latent_block_permutation
        ].copy(),
        "synthetic_truth_target_mask": target_mask[latent_block_permutation].copy(),
        "synthetic_truth_pre_channel_state": pre_channel_state[
            latent_block_permutation
        ].copy(),
        "synthetic_truth_pre_channel_rho": pre_channel_rho[
            latent_block_permutation
        ].copy(),
        "synthetic_truth_clean_rho": clean_rho[latent_block_permutation].copy(),
        "synthetic_truth_born_probability": shuffled_born_probability,
        "registered_povm_hash": str(batch["povm_hash"]),
        "operation_mapping_permutation": operation_permutation,
        "observed_operation_click_counts": permuted_operation_counts,
        "global_multi_was_distributed_to_operations": False,
        "marginal_preservation_required": True,
        "seed": int(seed),
    }
    result["control_hash"] = _batch_content_hash(result)
    return result


def _validate_mode_probability(value: Any) -> np.ndarray:
    probability = np.asarray(value, dtype=np.float64)
    if probability.ndim < 1 or probability.shape[-1] != 4:
        raise ValueError("mode_probability must end in four detector modes")
    if np.any(~np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("mode_probability must be finite and non-negative")
    total = probability.sum(axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("mode_probability rows must have positive mass")
    return probability / total


def _broadcast_detector_parameter(
    value: Sequence[float] | np.ndarray,
    target_shape: Tuple[int, ...],
    name: str,
) -> np.ndarray:
    parameter = np.asarray(value, dtype=np.float64)
    if parameter.shape == (4,):
        parameter = np.broadcast_to(parameter, target_shape + (4,))
    else:
        try:
            parameter = np.broadcast_to(parameter, target_shape + (4,))
        except ValueError as exc:
            raise ValueError(f"{name} cannot broadcast to detector modes") from exc
    if np.any(~np.isfinite(parameter)):
        raise ValueError(f"{name} must be finite")
    return parameter


def _exact_presence_patterns_from_power(
    registered_probability: np.ndarray,
    power_function: Any,
) -> np.ndarray:
    result = np.zeros(registered_probability.shape[:-1] + (16,), dtype=np.float64)
    for pattern_index in range(16):
        detector_set = [bit for bit in range(4) if pattern_index & (1 << bit)]
        value = np.zeros(registered_probability.shape[:-1], dtype=np.float64)
        for subset_bits in range(1 << len(detector_set)):
            included = [
                detector_set[position]
                for position in range(len(detector_set))
                if subset_bits & (1 << position)
            ]
            included_probability = (
                registered_probability[..., included].sum(axis=-1)
                if included
                else np.zeros(registered_probability.shape[:-1], dtype=np.float64)
            )
            sign = -1.0 if (len(detector_set) - len(included)) % 2 else 1.0
            value += sign * power_function(included_probability)
        result[..., pattern_index] = value
    return np.maximum(result, 0.0)


def _normalize_pattern_probability(value: np.ndarray) -> np.ndarray:
    probability = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = probability.sum(axis=-1, keepdims=True)
    return probability / np.maximum(total, np.finfo(np.float64).eps)


def _sample_detector_efficiency(
    family: DLCZNoiseFamilyConfig,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.clip(
        np.asarray(family.detector_efficiency, dtype=np.float64),
        1.0e-6,
        1.0 - 1.0e-6,
    )
    logit = np.log(mean) - np.log1p(-mean)
    sampled = logit[None, :] + rng.normal(
        0.0, family.detector_efficiency_logit_std, size=(batch_size, 4)
    )
    return 1.0 / (1.0 + np.exp(-sampled))


def _sample_dark_probability(
    family: DLCZNoiseFamilyConfig,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.asarray(family.dark_probability, dtype=np.float64)
    sampled = mean[None, :] * np.exp(
        rng.normal(0.0, family.dark_log_std, size=(batch_size, 4))
        - 0.5 * np.square(family.dark_log_std)
    )
    return np.clip(sampled, 0.0, 0.05)


def _sample_clean_rho(
    *,
    batch_size: int,
    qubit_count: int,
    dephasing: np.ndarray,
    crosstalk: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Sample an exogenous pre-channel W target, then apply physical channels."""

    amplitude = np.exp(rng.normal(0.0, 0.45, size=(batch_size, qubit_count)))
    amplitude /= np.linalg.norm(amplitude, axis=-1, keepdims=True)
    phase = rng.uniform(-np.pi, np.pi, size=(batch_size, qubit_count))
    phase[:, 0] = 0.0
    pre_channel_state = amplitude * np.exp(1j * phase)
    pre_channel_rho = np.einsum(
        "bi,bj->bij",
        pre_channel_state,
        pre_channel_state.conj(),
    )
    rho = pre_channel_rho.copy()
    diagonal = np.zeros_like(rho)
    diagonal_index = np.arange(qubit_count)
    diagonal[:, diagonal_index, diagonal_index] = np.real(
        rho[:, diagonal_index, diagonal_index]
    )
    rho = (
        (1.0 - dephasing[:, None, None]) * rho
        + dephasing[:, None, None] * diagonal
    )
    forward = np.roll(np.roll(rho, 1, axis=-2), 1, axis=-1)
    backward = np.roll(np.roll(rho, -1, axis=-2), -1, axis=-1)
    rho = (
        (1.0 - crosstalk[:, None, None]) * rho
        + 0.5 * crosstalk[:, None, None] * (forward + backward)
    )
    rho = 0.5 * (rho + rho.conj().transpose(0, 2, 1))
    trace = np.real(np.trace(rho, axis1=-2, axis2=-1))
    rho = rho / trace[:, None, None]
    if float(np.min(np.linalg.eigvalsh(rho))) < -1.0e-10:
        raise AssertionError("synthetic clean rho lost positive semidefiniteness")
    return {
        "target_amplitudes": amplitude.astype(np.float64),
        "target_phases_rad": phase.astype(np.float64),
        "pre_channel_state": pre_channel_state.astype(np.complex128),
        "pre_channel_rho": pre_channel_rho.astype(np.complex128),
        "clean_rho": rho.astype(np.complex128),
    }


def _mode_to_detector_routing(
    system_type: str,
    batch_size: int,
    crosstalk: np.ndarray,
) -> np.ndarray:
    """Return the explicit Q-mode to four-read-detector synthetic calibration."""

    baseline = np.asarray(MODE_TO_DETECTOR_BASELINES[system_type], dtype=np.float64)
    uniform = np.full_like(baseline, 0.25)
    routing = (
        (1.0 - crosstalk[:, None, None]) * baseline[None, :, :]
        + crosstalk[:, None, None] * uniform[None, :, :]
    )
    if routing.shape != (batch_size, baseline.shape[0], 4):
        raise AssertionError("mode-to-detector routing shape changed")
    return routing / routing.sum(axis=-1, keepdims=True)


def _apply_hbt_beamsplitter(
    probability: np.ndarray,
    transmission: float,
) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64).copy()
    first_group = value[..., 0] + value[..., 1]
    second_group = value[..., 2] + value[..., 3]
    value[..., 0] = transmission * first_group
    value[..., 1] = (1.0 - transmission) * first_group
    value[..., 2] = transmission * second_group
    value[..., 3] = (1.0 - transmission) * second_group
    return _validate_mode_probability(value)


def _pattern_detector_marginals(pattern_probability: np.ndarray) -> np.ndarray:
    return np.einsum(
        "...p,pd->...d",
        np.asarray(pattern_probability, dtype=np.float64),
        PATTERNS,
    )


def _combine_independent_patterns(
    first_probability: np.ndarray,
    second_probability: np.ndarray,
) -> np.ndarray:
    """Detector-wise OR convolution for two independent 16-pattern laws."""

    first = np.asarray(first_probability, dtype=np.float64)
    second = np.asarray(second_probability, dtype=np.float64)
    if first.shape != second.shape or first.shape[-1] != 16:
        raise ValueError("independent pattern laws must have equal [...,16] shape")
    if not np.allclose(first.sum(axis=-1), 1.0) or not np.allclose(
        second.sum(axis=-1),
        1.0,
    ):
        raise ValueError("independent pattern laws must sum to one")
    output = np.zeros_like(first)
    for first_index in range(16):
        for second_index in range(16):
            output[..., first_index | second_index] += (
                first[..., first_index] * second[..., second_index]
            )
    return _normalize_pattern_probability(output)


def _accidental_pair_probability(
    singles_rate_hz: np.ndarray,
    *,
    window_s: float,
    gate_rate_hz: float,
) -> np.ndarray:
    rates = np.asarray(singles_rate_hz, dtype=np.float64)
    pair_rate = np.stack(
        [
            float(window_s) * rates[..., first] * rates[..., second]
            for first, second in PAIR_DETECTOR_INDICES
        ],
        axis=-1,
    )
    probability = pair_rate / float(gate_rate_hz)
    if np.any(probability < 0.0) or float(np.max(probability.sum(axis=-1))) >= 0.10:
        raise ValueError("accidental pair probability left the registered rare-event regime")
    return probability


def _accidental_pair_probability_components(
    source_singles_rate_per_brightness_hz: np.ndarray,
    background_singles_rate_hz: np.ndarray,
    brightness: np.ndarray,
    *,
    window_s: float,
    gate_rate_hz: float,
) -> Dict[str, np.ndarray]:
    """Expand accidentals into brightness orders zero, one, and two."""

    source_unit = np.asarray(source_singles_rate_per_brightness_hz, dtype=np.float64)
    background = np.asarray(background_singles_rate_hz, dtype=np.float64)
    brightness_value = np.asarray(brightness, dtype=np.float64)
    if source_unit.shape != background.shape or source_unit.shape[-1] != 4:
        raise ValueError("source/background singles must share [..., four detector] shape")
    try:
        brightness_value = np.broadcast_to(brightness_value, source_unit.shape[:-1] + (1,))
    except ValueError as exc:
        raise ValueError("brightness cannot broadcast to singles rates") from exc
    if (
        np.any(source_unit < 0.0)
        or np.any(background < 0.0)
        or np.any(brightness_value < 0.0)
    ):
        raise ValueError("singles rates and brightness must be non-negative")
    factor = float(window_s) / float(gate_rate_hz)
    source_scaled = brightness_value * source_unit
    source_source = np.stack(
        [
            factor * source_scaled[..., first] * source_scaled[..., second]
            for first, second in PAIR_DETECTOR_INDICES
        ],
        axis=-1,
    )
    source_background = np.stack(
        [
            factor
            * (
                source_scaled[..., first] * background[..., second]
                + background[..., first] * source_scaled[..., second]
            )
            for first, second in PAIR_DETECTOR_INDICES
        ],
        axis=-1,
    )
    background_background = np.stack(
        [
            factor * background[..., first] * background[..., second]
            for first, second in PAIR_DETECTOR_INDICES
        ],
        axis=-1,
    )
    total = source_source + source_background + background_background
    total_singles = source_scaled + background
    direct = _accidental_pair_probability(
        total_singles,
        window_s=window_s,
        gate_rate_hz=gate_rate_hz,
    )
    if not np.allclose(total, direct, rtol=1.0e-12, atol=1.0e-15):
        raise AssertionError("accidental brightness-order expansion lost conservation")
    return {
        "source_source": source_source,
        "source_background": source_background,
        "background_background": background_background,
        "total": total,
    }


def _independent_binary_pattern_probability(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    pattern = PATTERNS.reshape((1,) * (value.ndim - 1) + PATTERNS.shape)
    result = np.prod(
        np.where(
            pattern > 0.5,
            value[..., None, :],
            1.0 - value[..., None, :],
        ),
        axis=-1,
    )
    return _normalize_pattern_probability(result)


def _global_multi_click_observation(
    raw_pattern_counts_global: np.ndarray,
) -> Dict[str, Any]:
    counts = np.asarray(raw_pattern_counts_global, dtype=np.int64)
    pair_counts = counts[:, PAIR_PATTERN_INDICES]
    return {
        "schema": DLCZ_GLOBAL_MULTI_SCHEMA,
        "scope": "global",
        "field_scopes": {
            "multi2_pairs": "global",
            "multi2_count": "global",
            "multi3_count": "global",
            "multi4_count": "global",
        },
        "pair_labels": PAIR_LABELS,
        "multi2_pairs": pair_counts,
        "multi2_count": pair_counts.sum(axis=1),
        "multi3_count": counts[:, PATTERN_WEIGHTS == 3].sum(axis=1),
        "multi4_count": counts[:, 15],
        "operation_local_bins_available": False,
    }


def _exposure_known_mask(
    exposure_mode: str,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if exposure_mode == "known":
        return np.ones(batch_size, dtype=np.float64)
    if exposure_mode == "unknown":
        return np.zeros(batch_size, dtype=np.float64)
    mask = (rng.random(batch_size) >= 0.5).astype(np.float64)
    if batch_size >= 2:
        mask[0] = 1.0
        mask[1] = 0.0
    return mask


def _non_identity_permutation(
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(size) <= 1:
        return np.arange(int(size), dtype=np.int64)
    permutation = rng.permutation(int(size))
    if np.array_equal(permutation, np.arange(int(size))):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64)


def _family_seed(seed: int, family_hash: str) -> int:
    namespace = int(family_hash[:16], 16)
    return int((int(seed) ^ namespace) % (2**63 - 1))


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_contract_hash(value: np.ndarray, metadata: Mapping[str, Any]) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    digest.update(_json_hash(dict(metadata)).encode("utf-8"))
    return digest.hexdigest()


def _batch_content_hash(batch: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        if key in ("batch_hash", "control_hash", "injection_hash"):
            continue
        value = batch[key]
        digest.update(key.encode("utf-8"))
        _update_digest(digest, value)
    return digest.hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _update_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _update_digest(digest, item)
        return
    digest.update(repr(value).encode("utf-8"))


__all__ = [
    "BASIS_PHASE_CONVENTION",
    "BASIS_PHASE_SIGN",
    "DLCZ_GLOBAL_MULTI_SCHEMA",
    "DLCZ_NEGATIVE_CONTROL_SCHEMA",
    "DLCZ_NOISE_FAMILIES",
    "DLCZ_NOISE_SYNTHETIC_SCHEMA",
    "DLCZ_SEMISYNTHETIC_SCHEMA",
    "DLCZIntervention",
    "DLCZNoiseFamilyConfig",
    "EXPOSURE_LEVELS_S",
    "PAIR_LABELS",
    "PAIR_PATTERN_INDICES",
    "PATTERNS",
    "PATTERN_WEIGHTS",
    "SYSTEM_OPERATION_COUNTS",
    "build_dlcz_negative_controls",
    "born_probabilities",
    "detected_pattern_probabilities_for_n",
    "generate_dlcz_intervention_pair",
    "generate_dlcz_noise_batch",
    "geometric_detected_pattern_components",
    "geometric_excitation_probabilities",
    "inject_semisynthetic_dlcz_noise",
    "resolve_dlcz_noise_family",
    "registered_povm_contract",
    "sample_gamma_poisson_nb2",
    "single_excitation_occupancy",
    "two_excitation_bosonic_occupancy",
]
