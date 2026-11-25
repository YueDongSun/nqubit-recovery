"""Bosonic Fock-leakage physics and synthetic pretraining hooks for V8.

The feedback target is an N-dimensional single-excitation W-core, but the
generative observation model spans vacuum/background, one excitation, two
excitations, and an aggregated n>=3 tail.  Sector measurement expectations are
computed from bosonic transition amplitudes; n=2 and tail probabilities are
never allocated with the W-core Born distribution.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional ML dependency
    torch = None
    nn = None


FOCK_LEAKAGE_CONTRACT_SCHEMA = "physics_hybrid_v8_fock_leakage_contract_v2"
FOCK_PRETRAIN_SCHEMA = "physics_hybrid_v8_fock_leakage_pretrain_v2"
DLCZ_DOUBLE_EXCITATION_SCHEMA = "physics_hybrid_v8_dlcz_double_excitation_state_v2"
SECTOR_NAMES = ("n0_background", "n1_w_core", "n2", "n_ge_3_tail")
DOUBLE_EXCITATION_COMPONENT_NAMES = (
    "same_collective_mode_fock2",
    "target_plus_orthogonal_schmidt_mode",
    "occupation_dephased_fock2",
)
MAX_EXACT_BOSON_NUMBER = 4
OPERATION_EXCITATION_RATE_CHI_MIN = 0.01
OPERATION_EXCITATION_RATE_CHI_MAX = 0.20


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FockLeakageContractV2:
    qubit_count: int
    maximum_exact_boson_number: int = MAX_EXACT_BOSON_NUMBER
    sectors: Tuple[str, ...] = SECTOR_NAMES
    feedback_target_space: str = "conditional_single_excitation_w_core"
    generator_space: str = "bosonic_fock_occupation_sectors"
    full_two_power_n_density_claim_allowed: bool = False
    hom_relative_phase_authoritatively_observable: bool = False
    unobservable_phase_policy: str = "monte_carlo_marginalize_and_sensitivity_bounds"
    sector_specific_measurement_expectation_required: bool = True
    w_core_allocation_allowed_for_n2_or_tail: bool = False
    real_corrected_label_export_allowed: bool = False
    real_action_evidence_created_by_pretraining: bool = False
    schema: str = FOCK_LEAKAGE_CONTRACT_SCHEMA

    def validate(self) -> None:
        if self.schema != FOCK_LEAKAGE_CONTRACT_SCHEMA:
            raise ValueError("unsupported Fock leakage contract schema")
        if self.qubit_count not in {3, 4}:
            raise ValueError("Fock leakage contract supports exact 3Q/4Q systems")
        if self.sectors != SECTOR_NAMES:
            raise ValueError("Fock leakage sector order changed")
        if not 2 <= int(self.maximum_exact_boson_number) <= MAX_EXACT_BOSON_NUMBER:
            raise ValueError("exact boson truncation must remain in [2, 4]")
        if self.feedback_target_space != "conditional_single_excitation_w_core":
            raise ValueError("feedback target must remain the W-core")
        if self.generator_space != "bosonic_fock_occupation_sectors":
            raise ValueError("leakage generator must use bosonic occupation sectors")
        for field_name in (
            "full_two_power_n_density_claim_allowed",
            "hom_relative_phase_authoritatively_observable",
            "w_core_allocation_allowed_for_n2_or_tail",
            "real_corrected_label_export_allowed",
            "real_action_evidence_created_by_pretraining",
        ):
            if bool(getattr(self, field_name)):
                raise ValueError(f"Fock leakage contract exceeded authority: {field_name}")
        if not self.sector_specific_measurement_expectation_required:
            raise ValueError("sector-specific measurement expectation is mandatory")

    @property
    def n1_dimension(self) -> int:
        return self.qubit_count

    @property
    def n2_dimension(self) -> int:
        return self.qubit_count * (self.qubit_count + 1) // 2

    @property
    def hard_core_hilbert_dimension(self) -> int:
        return 2 ** self.qubit_count

    def to_json_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "n1_dimension": self.n1_dimension,
            "n2_dimension": self.n2_dimension,
            "hard_core_hilbert_dimension": self.hard_core_hilbert_dimension,
        }


@dataclass(frozen=True)
class FockAnalyzerKernelV2:
    """Precomputed exact sector transform for one registered analyzer set."""

    analyzer_states: np.ndarray
    boson_number: int
    transition: np.ndarray
    output_basis: np.ndarray
    input_basis: np.ndarray
    kernel_sha256: str

    def evaluate(self, sector_density: np.ndarray) -> np.ndarray:
        density = np.asarray(sector_density, dtype=np.complex128)
        squeeze = density.ndim == 2
        if squeeze:
            density = density[None, :, :]
        expected = self.transition.shape[-1]
        if density.ndim != 3 or density.shape[1:] != (expected, expected):
            raise ValueError("sector_density must be [batch, sector, sector]")
        if np.any(~np.isfinite(density)) or not np.allclose(
            density,
            density.conj().transpose(0, 2, 1),
            atol=1.0e-8,
            rtol=1.0e-8,
        ):
            raise ValueError("sector density batch must be finite and Hermitian")
        trace = np.real(np.trace(density, axis1=-2, axis2=-1))
        if np.any(np.abs(trace - 1.0) > 1.0e-7):
            raise ValueError("sector density batch must have unit trace")
        probability = np.real(
            np.einsum(
                "oki,bij,okj->bok",
                self.transition,
                density,
                self.transition.conj(),
                optimize=True,
            )
        )
        probability = np.clip(probability, 0.0, None)
        probability /= np.maximum(
            probability.sum(axis=-1, keepdims=True),
            np.finfo(np.float64).eps,
        )
        selected = np.zeros(
            (density.shape[0], self.transition.shape[0], self.boson_number + 1),
            dtype=np.float64,
        )
        for photon_number in range(self.boson_number + 1):
            selected[..., photon_number] = probability[
                ..., self.output_basis[:, 0] == photon_number
            ].sum(axis=-1)
        return selected[0] if squeeze else selected


@dataclass(frozen=True)
class DLCZDoubleExcitationStateV2:
    """Structured n=2 spin-wave hypothesis used only as latent/synthetic truth.

    The three components distinguish an ideal phase-consistent Fock-2 state
    from a second Schmidt mode and from which-cell/occupation dephasing.  The
    density matrix lives in the symmetric two-boson sector.  Its normalized
    one-body reduction is the object that can be compared with the target
    W-core after one excitation is lost or remains undetected.
    """

    sector_density: np.ndarray
    component_probability: np.ndarray
    same_mode_density: np.ndarray
    split_mode_density: np.ndarray
    occupation_dephased_density: np.ndarray
    nuisance_single_particle_mode: np.ndarray
    one_body_reduced_density: np.ndarray
    target_one_body_fidelity: float
    phase_violation_probability: float
    schema: str = DLCZ_DOUBLE_EXCITATION_SCHEMA

    def validate(self, target_state: Sequence[complex]) -> None:
        target = np.asarray(target_state, dtype=np.complex128)
        target = target / max(float(np.linalg.norm(target)), 1.0e-15)
        sector_dimension = target.size * (target.size + 1) // 2
        if self.schema != DLCZ_DOUBLE_EXCITATION_SCHEMA:
            raise ValueError("unsupported DLCZ double-excitation schema")
        if self.component_probability.shape != (len(DOUBLE_EXCITATION_COMPONENT_NAMES),):
            raise ValueError("double-excitation component registry changed")
        if np.any(self.component_probability < 0.0) or not np.isclose(
            float(np.sum(self.component_probability)), 1.0
        ):
            raise ValueError("double-excitation component probability is not a simplex")
        for density in (
            self.sector_density,
            self.same_mode_density,
            self.split_mode_density,
            self.occupation_dephased_density,
        ):
            if density.shape != (sector_dimension, sector_dimension):
                raise ValueError("double-excitation sector density has wrong shape")
            if not np.allclose(density, density.conj().T, atol=1.0e-9, rtol=1.0e-9):
                raise ValueError("double-excitation sector density is not Hermitian")
            if float(np.min(np.linalg.eigvalsh(density))) < -1.0e-9:
                raise ValueError("double-excitation sector density is not PSD")
            if not np.isclose(float(np.real(np.trace(density))), 1.0, atol=1.0e-9):
                raise ValueError("double-excitation sector density is not trace one")
        if self.one_body_reduced_density.shape != (target.size, target.size):
            raise ValueError("one-body reduced density has wrong shape")
        if not np.isclose(
            float(np.real(np.trace(self.one_body_reduced_density))), 1.0, atol=1.0e-9
        ):
            raise ValueError("one-body reduced density is not trace one")
        if not 0.0 <= float(self.target_one_body_fidelity) <= 1.0:
            raise ValueError("target one-body fidelity is outside [0, 1]")
        if not np.isclose(
            float(self.phase_violation_probability),
            1.0 - float(self.target_one_body_fidelity),
            atol=1.0e-10,
        ):
            raise ValueError("phase-violation semantic changed")


def build_fock_analyzer_kernel(
    analyzer_states: np.ndarray,
    *,
    boson_number: int,
    detector_mode_count: int = 4,
) -> FockAnalyzerKernelV2:
    analyzers = np.asarray(analyzer_states, dtype=np.complex128)
    if analyzers.ndim != 2 or analyzers.shape[1] not in {3, 4}:
        raise ValueError("analyzer_states must be [operation, 3|4]")
    if int(detector_mode_count) < analyzers.shape[1]:
        raise ValueError("detector_mode_count cannot be smaller than W dimension")
    transitions = []
    input_basis = None
    output_basis = None
    for analyzer in analyzers:
        embedded = np.zeros(int(detector_mode_count), dtype=np.complex128)
        embedded[: analyzers.shape[1]] = analyzer
        unitary = complete_analyzer_unitary(embedded)
        current_transition, current_input, current_output = (
            bosonic_sector_transition_matrix(
                unitary,
                boson_number=int(boson_number),
                populated_input_mode_count=analyzers.shape[1],
            )
        )
        transitions.append(current_transition)
        if input_basis is None:
            input_basis = current_input
            output_basis = current_output
        elif not np.array_equal(input_basis, current_input) or not np.array_equal(
            output_basis,
            current_output,
        ):
            raise AssertionError("Fock analyzer basis changed across operations")
    transition = np.stack(transitions)
    payload = {
        "schema": "physics_hybrid_v8_fock_analyzer_kernel_v2",
        "boson_number": int(boson_number),
        "analyzer_sha256": hashlib.sha256(
            np.ascontiguousarray(analyzers).tobytes()
        ).hexdigest(),
        "transition_sha256": hashlib.sha256(
            np.ascontiguousarray(transition).tobytes()
        ).hexdigest(),
    }
    return FockAnalyzerKernelV2(
        analyzer_states=analyzers,
        boson_number=int(boson_number),
        transition=transition,
        output_basis=np.asarray(output_basis, dtype=np.int64),
        input_basis=np.asarray(input_basis, dtype=np.int64),
        kernel_sha256=_stable_hash(payload),
    )


def bosonic_occupation_basis(mode_count: int, boson_number: int) -> np.ndarray:
    """Enumerate non-negative occupations summing to ``boson_number``."""

    if int(mode_count) <= 0 or int(boson_number) < 0:
        raise ValueError("mode_count must be positive and boson_number non-negative")

    def recurse(remaining_modes: int, remaining_bosons: int) -> List[Tuple[int, ...]]:
        if remaining_modes == 1:
            return [(remaining_bosons,)]
        result: List[Tuple[int, ...]] = []
        # Descending order makes the n=1 basis exactly the laboratory mode
        # order e_0, e_1, ...; higher sectors then retain lexicographic mode
        # priority and accept density matrices built from that same registry.
        for first in range(remaining_bosons, -1, -1):
            for tail in recurse(remaining_modes - 1, remaining_bosons - first):
                result.append((first, *tail))
        return result

    return np.asarray(recurse(int(mode_count), int(boson_number)), dtype=np.int64)


def _matrix_permanent(matrix: np.ndarray) -> complex:
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise ValueError("permanent requires a square matrix")
    if size == 0:
        return 1.0 + 0.0j
    value = 0.0 + 0.0j
    for permutation in itertools.permutations(range(size)):
        term = 1.0 + 0.0j
        for row_index, column_index in enumerate(permutation):
            term *= matrix[row_index, column_index]
        value += term
    return value


def bosonic_transition_probability(
    unitary: np.ndarray,
    input_occupation: Sequence[int],
    output_occupation: Sequence[int],
) -> float:
    """Exact number-resolving transition probability for a small Fock sector."""

    transform = np.asarray(unitary, dtype=np.complex128)
    input_values = np.asarray(input_occupation, dtype=np.int64)
    output_values = np.asarray(output_occupation, dtype=np.int64)
    mode_count = input_values.size
    if transform.shape != (mode_count, mode_count) or output_values.shape != (mode_count,):
        raise ValueError("unitary and occupation dimensions disagree")
    if np.any(input_values < 0) or np.any(output_values < 0):
        raise ValueError("occupations must be non-negative")
    boson_number = int(np.sum(input_values))
    if int(np.sum(output_values)) != boson_number:
        return 0.0
    if boson_number > MAX_EXACT_BOSON_NUMBER:
        raise ValueError("exact permanent kernel supports at most four bosons")
    identity = np.eye(mode_count, dtype=np.complex128)
    if not np.allclose(
        transform.conj().T @ transform,
        identity,
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ValueError("AOD/beam-splitter transform must be unitary")
    input_columns = np.repeat(np.arange(mode_count), input_values)
    output_rows = np.repeat(np.arange(mode_count), output_values)
    submatrix = transform[np.ix_(output_rows, input_columns)]
    normalization = math.sqrt(
        math.prod(math.factorial(int(value)) for value in input_values)
        * math.prod(math.factorial(int(value)) for value in output_values)
    )
    amplitude = _matrix_permanent(submatrix) / normalization
    probability = float(abs(amplitude) ** 2)
    return min(max(probability, 0.0), 1.0)


def sector_measurement_expectation(
    unitary: np.ndarray,
    input_distribution: np.ndarray,
    *,
    boson_number: int,
) -> np.ndarray:
    """Propagate one complete sector distribution through an AOD unitary."""

    transform = np.asarray(unitary, dtype=np.complex128)
    if transform.ndim != 2 or transform.shape[0] != transform.shape[1]:
        raise ValueError("unitary must be square")
    basis = bosonic_occupation_basis(transform.shape[0], int(boson_number))
    distribution = np.asarray(input_distribution, dtype=np.float64)
    if distribution.shape != (basis.shape[0],):
        raise ValueError("input sector distribution has wrong dimension")
    if np.any(distribution < 0.0) or not np.isclose(np.sum(distribution), 1.0):
        raise ValueError("input sector distribution must be a probability simplex")
    transition = np.empty((basis.shape[0], basis.shape[0]), dtype=np.float64)
    for output_index, output_occupation in enumerate(basis):
        for input_index, input_occupation in enumerate(basis):
            transition[output_index, input_index] = bosonic_transition_probability(
                transform,
                input_occupation,
                output_occupation,
            )
    output = transition @ distribution
    if not np.isclose(np.sum(output), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("bosonic measurement expectation is not normalized")
    return output / np.sum(output)


def complete_analyzer_unitary(analyzer_state: Sequence[complex]) -> np.ndarray:
    """Complete one measured analyzer row to a deterministic unitary.

    The first output mode is the registered tomography analyzer.  Remaining
    rows are only a mathematical completion and are never treated as measured
    HOM-phase calibration.  Selected-port occupation probabilities are
    invariant to this completion; full output-pattern claims are prohibited.
    """

    analyzer = np.asarray(analyzer_state, dtype=np.complex128)
    if analyzer.ndim != 1 or analyzer.size < 2 or np.any(~np.isfinite(analyzer)):
        raise ValueError("analyzer_state must be one finite complex vector")
    norm = float(np.linalg.norm(analyzer))
    if norm <= 0.0:
        raise ValueError("analyzer_state must be non-zero")
    rows = [analyzer.conj() / norm]
    for basis_index in range(analyzer.size):
        candidate = np.zeros(analyzer.size, dtype=np.complex128)
        candidate[basis_index] = 1.0
        for row in rows:
            candidate -= np.vdot(row, candidate) * row
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm > 1.0e-10:
            rows.append(candidate / candidate_norm)
        if len(rows) == analyzer.size:
            break
    unitary = np.stack(rows)
    if unitary.shape != (analyzer.size, analyzer.size) or not np.allclose(
        unitary @ unitary.conj().T,
        np.eye(analyzer.size),
        atol=1.0e-9,
        rtol=1.0e-9,
    ):
        raise AssertionError("deterministic analyzer completion is not unitary")
    return unitary


def bosonic_sector_transition_matrix(
    unitary: np.ndarray,
    *,
    boson_number: int,
    populated_input_mode_count: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact small-sector Fock transition amplitude matrix.

    ``populated_input_mode_count`` supports the 3Q experiment embedded into
    four optical outputs: the fourth input starts in vacuum, while all four
    read/idler detector outputs remain explicit.
    """

    transform = np.asarray(unitary, dtype=np.complex128)
    if transform.ndim != 2 or transform.shape[0] != transform.shape[1]:
        raise ValueError("unitary must be square")
    mode_count = int(transform.shape[0])
    populated = int(populated_input_mode_count or mode_count)
    if not 1 <= populated <= mode_count:
        raise ValueError("populated_input_mode_count is outside the unitary")
    if not 0 <= int(boson_number) <= MAX_EXACT_BOSON_NUMBER:
        raise ValueError("boson_number exceeds the exact V2 contract")
    input_basis_small = bosonic_occupation_basis(populated, int(boson_number))
    input_basis = np.pad(
        input_basis_small,
        ((0, 0), (0, mode_count - populated)),
        mode="constant",
    )
    output_basis = bosonic_occupation_basis(mode_count, int(boson_number))
    transition = np.empty(
        (output_basis.shape[0], input_basis.shape[0]),
        dtype=np.complex128,
    )
    for output_index, output_occupation in enumerate(output_basis):
        for input_index, input_occupation in enumerate(input_basis):
            input_columns = np.repeat(np.arange(mode_count), input_occupation)
            output_rows = np.repeat(np.arange(mode_count), output_occupation)
            submatrix = transform[np.ix_(output_rows, input_columns)]
            normalization = math.sqrt(
                math.prod(math.factorial(int(value)) for value in input_occupation)
                * math.prod(math.factorial(int(value)) for value in output_occupation)
            )
            transition[output_index, input_index] = (
                _matrix_permanent(submatrix) / normalization
            )
    gram = transition.conj().T @ transition
    if not np.allclose(
        gram,
        np.eye(input_basis.shape[0]),
        atol=2.0e-8,
        rtol=2.0e-8,
    ):
        raise AssertionError("restricted Fock transition does not conserve probability")
    return transition, input_basis_small, output_basis


def bosonic_product_state_amplitude(
    single_particle_state: Sequence[complex],
    *,
    boson_number: int,
) -> np.ndarray:
    """Lift one normalized spin-wave state into its symmetric n-boson sector."""

    state = np.asarray(single_particle_state, dtype=np.complex128)
    if state.ndim != 1 or np.any(~np.isfinite(state)):
        raise ValueError("single_particle_state must be one finite vector")
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError("single_particle_state must be non-zero")
    state = state / norm
    basis = bosonic_occupation_basis(state.size, int(boson_number))
    amplitude = np.empty(basis.shape[0], dtype=np.complex128)
    numerator = math.factorial(int(boson_number))
    for index, occupation in enumerate(basis):
        coefficient = math.sqrt(
            numerator
            / math.prod(math.factorial(int(value)) for value in occupation)
        )
        amplitude[index] = coefficient * np.prod(np.power(state, occupation))
    amplitude /= max(float(np.linalg.norm(amplitude)), 1.0e-15)
    return amplitude


def bosonic_two_mode_state_amplitude(
    first_state: Sequence[complex],
    second_state: Sequence[complex],
) -> np.ndarray:
    """Create the normalized symmetric state ``a(first)^dag a(second)^dag|0>``.

    For equal modes this is exactly the phase-consistent Fock-2 state.  For
    orthogonal modes it represents one excitation in the target spin wave and
    one in a distinct Schmidt/spatial spin-wave mode.
    """

    first = np.asarray(first_state, dtype=np.complex128)
    second = np.asarray(second_state, dtype=np.complex128)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("two single-particle modes must have equal one-dimensional shape")
    if np.any(~np.isfinite(first)) or np.any(~np.isfinite(second)):
        raise ValueError("single-particle modes must be finite")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if min(first_norm, second_norm) <= 0.0:
        raise ValueError("single-particle modes must be non-zero")
    first = first / first_norm
    second = second / second_norm
    basis = bosonic_occupation_basis(first.size, 2)
    amplitude = np.zeros(basis.shape[0], dtype=np.complex128)
    for index, occupation in enumerate(basis):
        occupied = np.flatnonzero(occupation)
        if occupied.size == 1:
            mode = int(occupied[0])
            amplitude[index] = math.sqrt(2.0) * first[mode] * second[mode]
        elif occupied.size == 2:
            first_mode, second_mode = (int(value) for value in occupied)
            amplitude[index] = (
                first[first_mode] * second[second_mode]
                + first[second_mode] * second[first_mode]
            )
        else:
            raise AssertionError("two-boson occupation registry is invalid")
    expected_norm = math.sqrt(1.0 + abs(np.vdot(first, second)) ** 2)
    amplitude /= max(expected_norm, 1.0e-15)
    amplitude /= max(float(np.linalg.norm(amplitude)), 1.0e-15)
    return amplitude


def one_body_reduced_density(
    sector_density: np.ndarray,
    *,
    mode_count: int,
    boson_number: int,
) -> np.ndarray:
    """Return ``gamma_ij = Tr(rho a_j^dag a_i) / n`` for one Fock sector."""

    if int(mode_count) <= 0 or int(boson_number) <= 0:
        raise ValueError("mode_count and boson_number must be positive")
    basis = bosonic_occupation_basis(int(mode_count), int(boson_number))
    reduced_basis = bosonic_occupation_basis(int(mode_count), int(boson_number) - 1)
    density = np.asarray(sector_density, dtype=np.complex128)
    if density.shape != (basis.shape[0], basis.shape[0]):
        raise ValueError("sector_density shape does not match the requested Fock sector")
    if not np.allclose(density, density.conj().T, atol=1.0e-9, rtol=1.0e-9):
        raise ValueError("sector_density must be Hermitian")
    reduced_index = {tuple(row.tolist()): index for index, row in enumerate(reduced_basis)}
    annihilation = []
    for mode_index in range(int(mode_count)):
        operator = np.zeros((reduced_basis.shape[0], basis.shape[0]), dtype=np.complex128)
        for column, occupation in enumerate(basis):
            if occupation[mode_index] <= 0:
                continue
            reduced = occupation.copy()
            reduced[mode_index] -= 1
            operator[reduced_index[tuple(reduced.tolist())], column] = math.sqrt(
                float(occupation[mode_index])
            )
        annihilation.append(operator)
    gamma = np.empty((int(mode_count), int(mode_count)), dtype=np.complex128)
    for first_mode, first_operator in enumerate(annihilation):
        for second_mode, second_operator in enumerate(annihilation):
            gamma[first_mode, second_mode] = np.trace(
                first_operator @ density @ second_operator.conj().T
            ) / float(boson_number)
    gamma = 0.5 * (gamma + gamma.conj().T)
    gamma /= max(float(np.real(np.trace(gamma))), 1.0e-15)
    return gamma


def _sample_orthogonal_single_particle_mode(
    target_state: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    target = np.asarray(target_state, dtype=np.complex128)
    target = target / max(float(np.linalg.norm(target)), 1.0e-15)
    for _ in range(16):
        candidate = rng.normal(size=target.size) + 1j * rng.normal(size=target.size)
        candidate -= target * np.vdot(target, candidate)
        norm = float(np.linalg.norm(candidate))
        if norm > 1.0e-10:
            return candidate / norm
    raise RuntimeError("failed to sample an orthogonal spin-wave mode")


def build_dlcz_double_excitation_state(
    target_state: Sequence[complex],
    *,
    component_probability: Sequence[float],
    nuisance_single_particle_mode: Sequence[complex],
) -> DLCZDoubleExcitationStateV2:
    """Build the structured DLCZ n=2 density and its one-body phase diagnostic."""

    target = np.asarray(target_state, dtype=np.complex128)
    nuisance = np.asarray(nuisance_single_particle_mode, dtype=np.complex128)
    if target.ndim != 1 or nuisance.shape != target.shape:
        raise ValueError("target and nuisance modes must have equal one-dimensional shape")
    target /= max(float(np.linalg.norm(target)), 1.0e-15)
    nuisance -= target * np.vdot(target, nuisance)
    nuisance_norm = float(np.linalg.norm(nuisance))
    if nuisance_norm <= 1.0e-10:
        raise ValueError("nuisance mode must contain a component orthogonal to target")
    nuisance /= nuisance_norm
    weights = np.asarray(component_probability, dtype=np.float64)
    if weights.shape != (len(DOUBLE_EXCITATION_COMPONENT_NAMES),):
        raise ValueError("component_probability must follow the double-excitation registry")
    if np.any(weights < 0.0) or not np.isclose(float(np.sum(weights)), 1.0):
        raise ValueError("component_probability must be a simplex")
    same_amplitude = bosonic_product_state_amplitude(target, boson_number=2)
    split_amplitude = bosonic_two_mode_state_amplitude(target, nuisance)
    same_density = np.outer(same_amplitude, same_amplitude.conj())
    split_density = np.outer(split_amplitude, split_amplitude.conj())
    dephased_density = np.diag(np.real(np.diag(same_density))).astype(np.complex128)
    dephased_density /= max(float(np.real(np.trace(dephased_density))), 1.0e-15)
    density = (
        weights[0] * same_density
        + weights[1] * split_density
        + weights[2] * dephased_density
    )
    gamma = one_body_reduced_density(
        density,
        mode_count=target.size,
        boson_number=2,
    )
    target_fidelity = float(np.real(np.vdot(target, gamma @ target)))
    target_fidelity = min(max(target_fidelity, 0.0), 1.0)
    state = DLCZDoubleExcitationStateV2(
        sector_density=density,
        component_probability=weights,
        same_mode_density=same_density,
        split_mode_density=split_density,
        occupation_dephased_density=dephased_density,
        nuisance_single_particle_mode=nuisance,
        one_body_reduced_density=gamma,
        target_one_body_fidelity=target_fidelity,
        phase_violation_probability=1.0 - target_fidelity,
    )
    state.validate(target)
    return state


def sample_dlcz_double_excitation_state(
    target_state: Sequence[complex],
    *,
    nuisance_fraction: float,
    split_fraction_of_nuisance: float,
    rng: np.random.Generator,
) -> DLCZDoubleExcitationStateV2:
    """Sample a physically named partial-identification hypothesis for n=2."""

    nuisance_fraction = float(nuisance_fraction)
    split_fraction = float(split_fraction_of_nuisance)
    if not 0.0 <= nuisance_fraction <= 1.0:
        raise ValueError("nuisance_fraction must lie in [0, 1]")
    if not 0.0 <= split_fraction <= 1.0:
        raise ValueError("split_fraction_of_nuisance must lie in [0, 1]")
    target = np.asarray(target_state, dtype=np.complex128)
    nuisance = _sample_orthogonal_single_particle_mode(target, rng)
    weights = np.asarray(
        (
            1.0 - nuisance_fraction,
            nuisance_fraction * split_fraction,
            nuisance_fraction * (1.0 - split_fraction),
        ),
        dtype=np.float64,
    )
    return build_dlcz_double_excitation_state(
        target,
        component_probability=weights,
        nuisance_single_particle_mode=nuisance,
    )


def threshold_herald_click_probability(
    photon_number: Sequence[int] | np.ndarray | int,
    *,
    detector_efficiency: float,
    dark_probability: float = 0.0,
) -> np.ndarray:
    """Binary 200 ns SPAD POVM, never an eventwise photon-number readout.

    ``P(click|n)=1-(1-p_dark)(1-eta)^n`` only models acquisition bias.  The
    returned click cannot distinguish ``n=1`` from ``n>=2`` on one gate.
    """

    efficiency = float(detector_efficiency)
    dark = float(dark_probability)
    if not 0.0 <= efficiency <= 1.0 or not 0.0 <= dark <= 1.0:
        raise ValueError("herald efficiency and dark probability must lie in [0, 1]")
    number = np.asarray(photon_number, dtype=np.int64)
    if np.any(number < 0):
        raise ValueError("herald photon number must be non-negative")
    return 1.0 - (1.0 - dark) * np.power(1.0 - efficiency, number)


def condition_excitation_prior_on_threshold_herald(
    excitation_prior: np.ndarray,
    *,
    detector_efficiency: float,
    dark_probability: float = 0.0,
    sector_photon_number: Sequence[int] = (0, 1, 2, 3),
) -> np.ndarray:
    """Condition n-sector priors on a binary SPAD click for sensitivity only.

    The last ``n=3`` entry is a conservative representative for an aggregated
    n>=3 tail.  A calibrated tail ratio is required for a sharper real-data
    likelihood, so this helper is a sensitivity endpoint rather than a claim
    that the tail photon number is point identified.  At the authoritative
    default ``eta=1, p_dark=0`` every non-vacuum sector has equal acceptance.
    """

    prior = np.asarray(excitation_prior, dtype=np.float64)
    number = np.asarray(sector_photon_number, dtype=np.int64)
    if prior.ndim < 1 or prior.shape[-1] != number.size:
        raise ValueError("excitation prior and sector photon registry disagree")
    if np.any(prior < 0.0) or np.any(~np.isfinite(prior)):
        raise ValueError("excitation prior must be finite and non-negative")
    prior = prior / np.maximum(prior.sum(axis=-1, keepdims=True), 1.0e-15)
    click = threshold_herald_click_probability(
        number,
        detector_efficiency=detector_efficiency,
        dark_probability=dark_probability,
    )
    weighted = prior * click
    normalization = weighted.sum(axis=-1, keepdims=True)
    if np.any(normalization <= 0.0):
        raise ValueError("threshold-herald conditioning has zero probability")
    return weighted / normalization


def two_photon_selected_mode_probability_with_indistinguishability(
    analyzer_state: Sequence[complex],
    first_state: Sequence[complex],
    second_state: Sequence[complex],
    *,
    hom_indistinguishability: float,
) -> np.ndarray:
    """Selected-port n=2 probabilities between HOM endpoints.

    ``xi=1`` uses the exact permanent of the symmetric two-boson state;
    ``xi=0`` treats the two internal wave packets as distinguishable.  This is
    an explicit sensitivity parameter because the present AOD setup has no
    independent HOM-relative-phase/overlap calibration.
    """

    xi = float(hom_indistinguishability)
    if not 0.0 <= xi <= 1.0:
        raise ValueError("hom_indistinguishability must lie in [0, 1]")
    analyzer = np.asarray(analyzer_state, dtype=np.complex128)
    first = np.asarray(first_state, dtype=np.complex128)
    second = np.asarray(second_state, dtype=np.complex128)
    if analyzer.ndim != 1 or first.shape != analyzer.shape or second.shape != analyzer.shape:
        raise ValueError("analyzer and two modes must have equal one-dimensional shape")
    analyzer /= max(float(np.linalg.norm(analyzer)), 1.0e-15)
    first /= max(float(np.linalg.norm(first)), 1.0e-15)
    second /= max(float(np.linalg.norm(second)), 1.0e-15)
    symmetric = bosonic_two_mode_state_amplitude(first, second)
    indistinguishable = selected_mode_occupation_probability(
        analyzer,
        np.outer(symmetric, symmetric.conj()),
        boson_number=2,
    )
    first_selected = float(abs(np.vdot(analyzer, first)) ** 2)
    second_selected = float(abs(np.vdot(analyzer, second)) ** 2)
    distinguishable = np.asarray(
        (
            (1.0 - first_selected) * (1.0 - second_selected),
            first_selected * (1.0 - second_selected)
            + (1.0 - first_selected) * second_selected,
            first_selected * second_selected,
        ),
        dtype=np.float64,
    )
    probability = xi * indistinguishable + (1.0 - xi) * distinguishable
    return probability / max(float(np.sum(probability)), 1.0e-15)


def selected_mode_occupation_probability(
    analyzer_state: Sequence[complex],
    sector_density: np.ndarray,
    *,
    boson_number: int,
    detector_mode_count: int = 4,
) -> np.ndarray:
    """Compute P(k photons in the selected AOD read mode), k=0..n.

    This is the identifiable measurement primitive needed by the experiment:
    it does not assert a full output density matrix or an observed relative
    phase for the unmeasured orthogonal AOD/HOM outputs.
    """

    analyzer = np.asarray(analyzer_state, dtype=np.complex128)
    if analyzer.ndim != 1 or analyzer.size not in {3, 4}:
        raise ValueError("analyzer_state must be a registered 3Q/4Q vector")
    if int(detector_mode_count) < analyzer.size:
        raise ValueError("detector_mode_count cannot be smaller than the W modes")
    embedded = np.zeros(int(detector_mode_count), dtype=np.complex128)
    embedded[: analyzer.size] = analyzer
    unitary = complete_analyzer_unitary(embedded)
    transition, input_basis, output_basis = bosonic_sector_transition_matrix(
        unitary,
        boson_number=int(boson_number),
        populated_input_mode_count=analyzer.size,
    )
    density = np.asarray(sector_density, dtype=np.complex128)
    if density.shape != (input_basis.shape[0], input_basis.shape[0]):
        raise ValueError("sector_density dimension does not match the Fock sector")
    if not np.allclose(density, density.conj().T, atol=1.0e-9, rtol=1.0e-9):
        raise ValueError("sector_density must be Hermitian")
    eigenvalues = np.linalg.eigvalsh(density)
    if float(np.min(eigenvalues)) < -1.0e-9 or not np.isclose(
        float(np.real(np.trace(density))), 1.0, atol=1.0e-8
    ):
        raise ValueError("sector_density must be PSD and trace one")
    output_probability = np.real(
        np.einsum("oi,ij,oj->o", transition, density, transition.conj())
    )
    output_probability = np.clip(output_probability, 0.0, None)
    output_probability /= max(float(np.sum(output_probability)), 1.0e-15)
    selected = np.zeros(int(boson_number) + 1, dtype=np.float64)
    for occupation, probability in zip(output_basis, output_probability):
        selected[int(occupation[0])] += float(probability)
    return selected / np.sum(selected)


def sample_partially_identified_sector_density(
    single_particle_state: Sequence[complex],
    *,
    boson_number: int,
    nuisance_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a synthetic sector density without pretending it is measured.

    For n=2, the old arbitrary random-sector shortcut is replaced by the
    structured DLCZ mixture (same collective mode, target plus orthogonal
    Schmidt mode, occupation-dephased Fock-2).  Higher exact tail sectors retain
    a broad coherent sensitivity prior.  None is exported as a corrected real
    label.
    """

    if not 0.0 <= float(nuisance_fraction) <= 1.0:
        raise ValueError("nuisance_fraction must lie in [0, 1]")
    if int(boson_number) == 2:
        return sample_dlcz_double_excitation_state(
            single_particle_state,
            nuisance_fraction=float(nuisance_fraction),
            split_fraction_of_nuisance=float(rng.uniform(0.25, 0.75)),
            rng=rng,
        ).sector_density
    target = bosonic_product_state_amplitude(
        single_particle_state,
        boson_number=int(boson_number),
    )
    random_state = rng.normal(size=target.size) + 1j * rng.normal(size=target.size)
    random_state = random_state / max(float(np.linalg.norm(random_state)), 1.0e-15)
    target_density = np.outer(target, target.conj())
    nuisance_density = np.outer(random_state, random_state.conj())
    fraction = float(nuisance_fraction)
    return (1.0 - fraction) * target_density + fraction * nuisance_density


def monte_carlo_selected_mode_expectation(
    analyzer_states: np.ndarray,
    single_particle_state: Sequence[complex],
    *,
    boson_number: int,
    nuisance_fraction: float,
    draw_count: int,
    seed: int,
) -> Dict[str, Any]:
    """Marginalize unobserved multi-excitation coherences for every operation."""

    analyzers = np.asarray(analyzer_states, dtype=np.complex128)
    state = np.asarray(single_particle_state, dtype=np.complex128)
    if analyzers.ndim != 2 or analyzers.shape[1] != state.size:
        raise ValueError("analyzer_states and single-particle state disagree")
    if int(draw_count) < 2:
        raise ValueError("draw_count must be at least two")
    rng = np.random.default_rng(int(seed))
    kernel = build_fock_analyzer_kernel(
        analyzers,
        boson_number=int(boson_number),
    )
    densities = []
    for draw_index in range(int(draw_count)):
        densities.append(sample_partially_identified_sector_density(
            state,
            boson_number=int(boson_number),
            nuisance_fraction=float(nuisance_fraction),
            rng=rng,
        ))
    draws = kernel.evaluate(np.stack(densities))
    payload = {
        "schema": "physics_hybrid_v8_selected_mode_fock_monte_carlo_v2",
        "boson_number": int(boson_number),
        "draw_count": int(draw_count),
        "nuisance_fraction": float(nuisance_fraction),
        "mean_probability": np.mean(draws, axis=0),
        "lower_probability": np.quantile(draws, 0.05, axis=0),
        "upper_probability": np.quantile(draws, 0.95, axis=0),
        "draw_probability": draws,
        "analyzer_kernel_sha256": kernel.kernel_sha256,
        "hom_relative_phase_point_identified": False,
        "full_output_density_claim_allowed": False,
        "real_corrected_label_export_allowed": False,
    }
    payload["audit_sha256"] = _stable_hash({
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in payload.items()
    })
    return payload


def selected_occupation_to_click_patterns(
    selected_occupation_probability: np.ndarray,
    splitter_probability: Sequence[float],
    detector_efficiency: Sequence[float],
) -> np.ndarray:
    """Map selected read-mode photon number to four on/off detector patterns."""

    occupation = np.asarray(selected_occupation_probability, dtype=np.float64)
    splitter = np.asarray(splitter_probability, dtype=np.float64)
    efficiency = np.asarray(detector_efficiency, dtype=np.float64)
    if occupation.ndim != 1 or occupation.size > MAX_EXACT_BOSON_NUMBER + 1:
        raise ValueError("selected occupation must cover k=0..n with n<=4")
    if splitter.shape != (4,) or efficiency.shape != (4,):
        raise ValueError("splitter and detector efficiency must each have four values")
    if np.any(splitter < 0.0) or not np.isclose(np.sum(splitter), 1.0):
        raise ValueError("splitter_probability must be a simplex")
    if np.any((efficiency < 0.0) | (efficiency > 1.0)):
        raise ValueError("detector_efficiency must lie in [0, 1]")
    if np.any(occupation < 0.0) or not np.isclose(np.sum(occupation), 1.0):
        raise ValueError("selected occupation probability must be a simplex")
    detected_category = splitter * efficiency
    undetected = 1.0 - float(np.sum(detected_category))
    if undetected < -1.0e-12:
        raise ValueError("combined splitter/detector probabilities exceed one")
    category_probability = np.concatenate(([max(undetected, 0.0)], detected_category))
    pattern = np.zeros(16, dtype=np.float64)
    for photon_number, photon_mass in enumerate(occupation):
        if photon_mass <= 0.0:
            continue
        for categories in itertools.product(range(5), repeat=int(photon_number)):
            event_probability = float(photon_mass)
            pattern_index = 0
            for category in categories:
                event_probability *= float(category_probability[category])
                if category > 0:
                    pattern_index |= 1 << (category - 1)
            pattern[pattern_index] += event_probability
    return pattern / max(float(np.sum(pattern)), 1.0e-15)


def monte_carlo_sector_measurement_expectation(
    unitary_draws: np.ndarray,
    input_distribution: np.ndarray,
    *,
    boson_number: int,
) -> Dict[str, Any]:
    """Marginalize unobservable phase/nuisance draws without point-identifying them."""

    draws = np.asarray(unitary_draws, dtype=np.complex128)
    if draws.ndim != 3 or draws.shape[1] != draws.shape[2] or draws.shape[0] < 2:
        raise ValueError("unitary_draws must have shape [draw, mode, mode] with draw>=2")
    probabilities = np.stack([
        sector_measurement_expectation(
            draw,
            input_distribution,
            boson_number=int(boson_number),
        )
        for draw in draws
    ])
    result = {
        "schema": "physics_hybrid_v8_fock_measurement_monte_carlo_v2",
        "boson_number": int(boson_number),
        "draw_count": int(draws.shape[0]),
        "mean_probability": np.mean(probabilities, axis=0),
        "lower_probability": np.quantile(probabilities, 0.05, axis=0),
        "upper_probability": np.quantile(probabilities, 0.95, axis=0),
        "draw_probability": probabilities,
        "hom_relative_phase_point_identified": False,
        "w_core_allocation_used": False,
    }
    result["audit_sha256"] = _stable_hash({
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    })
    return result


@dataclass(frozen=True)
class FockLeakageClassifierConfig:
    input_dim: int
    width: int = 256
    embedding_dim: int = 128
    hidden_layers: int = 3
    dropout: float = 0.0
    sector_names: Tuple[str, ...] = SECTOR_NAMES
    synthetic_truth_only: bool = True
    transferred_encoder_creates_action_evidence: bool = False
    schema: str = FOCK_PRETRAIN_SCHEMA

    def validate(self) -> None:
        if self.schema != FOCK_PRETRAIN_SCHEMA or self.sector_names != SECTOR_NAMES:
            raise ValueError("unsupported Fock classifier config")
        if min(self.input_dim, self.width, self.embedding_dim, self.hidden_layers) <= 0:
            raise ValueError("Fock classifier dimensions must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("Fock classifier dropout must be in [0, 1)")
        if not self.synthetic_truth_only or self.transferred_encoder_creates_action_evidence:
            raise ValueError("Fock pretraining exceeded synthetic/action authority")


@dataclass(frozen=True)
class FockLeakageClassifierOutput:
    static_embedding: "torch.Tensor"
    sector_logits: "torch.Tensor"
    sector_probability: "torch.Tensor"
    w_core_mass: "torch.Tensor"
    leakage_mass: "torch.Tensor"
    phase_violation_probability: "torch.Tensor"
    double_excitation_component_logits: "torch.Tensor"
    double_excitation_component_probability: "torch.Tensor"
    double_excitation_one_body_target_fidelity: "torch.Tensor"
    real_action_supervision_mask: "torch.Tensor"


@dataclass(frozen=True)
class FockLeakageSetClassifierConfig:
    input_dim: int
    observation_dim: int = 5
    observation_conditioning_enabled: bool = False
    width: int = 64
    attention_heads: int = 4
    set_layers: int = 2
    pool_seed_count: int = 2
    feedforward_multiplier: float = 2.0
    dropout: float = 0.0
    sector_names: Tuple[str, ...] = SECTOR_NAMES
    synthetic_truth_only: bool = True
    transferred_encoder_creates_action_evidence: bool = False
    schema: str = "physics_hybrid_v8_fock_set_classifier_v2"

    def validate(self) -> None:
        if self.schema != "physics_hybrid_v8_fock_set_classifier_v2":
            raise ValueError("unsupported Fock set-classifier schema")
        if self.sector_names != SECTOR_NAMES:
            raise ValueError("Fock set-classifier sector order changed")
        if min(
            int(self.input_dim),
            int(self.width),
            int(self.attention_heads),
            int(self.set_layers),
            int(self.pool_seed_count),
            int(self.observation_dim),
        ) <= 0:
            raise ValueError("Fock set-classifier dimensions must be positive")
        if int(self.width) % int(self.attention_heads) != 0:
            raise ValueError("width must be divisible by attention_heads")
        if float(self.feedforward_multiplier) < 1.0:
            raise ValueError("feedforward_multiplier must be at least one")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not self.synthetic_truth_only or self.transferred_encoder_creates_action_evidence:
            raise ValueError("Fock set pretraining exceeded synthetic/action authority")


@dataclass(frozen=True)
class FockLeakageSetClassifierOutput:
    operation_embedding: "torch.Tensor"
    static_embedding: "torch.Tensor"
    sector_logits: "torch.Tensor"
    sector_probability: "torch.Tensor"
    operation_sector_log_response: "torch.Tensor"
    operation_excitation_rate_chi: "torch.Tensor"
    operation_conditional_p2_probability: "torch.Tensor"
    operation_conditional_tail_probability: "torch.Tensor"
    w_core_mass: "torch.Tensor"
    leakage_mass: "torch.Tensor"
    phase_violation_probability: "torch.Tensor"
    double_excitation_component_logits: "torch.Tensor"
    double_excitation_component_probability: "torch.Tensor"
    double_excitation_one_body_target_fidelity: "torch.Tensor"
    real_action_supervision_mask: "torch.Tensor"


if nn is not None:

    class _FockMaskedSetTransformer(nn.Module):
        """State-dict-compatible pretrainer for the sidecar static encoder."""

        def __init__(self, config: FockLeakageSetClassifierConfig) -> None:
            super().__init__()
            self.width = int(config.width)
            self.input_projection = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, config.width),
                nn.SiLU(),
                nn.Linear(config.width, config.width),
            )
            hidden = int(round(config.width * config.feedforward_multiplier))
            layer = nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.attention_heads,
                dim_feedforward=hidden,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.set_layers)
            self.seed = nn.Parameter(
                torch.randn(config.pool_seed_count, config.width)
                / math.sqrt(float(config.width))
            )
            self.pool_attention = nn.MultiheadAttention(
                config.width,
                config.attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.output_norm = nn.LayerNorm(config.width)

        def forward(
            self,
            value: "torch.Tensor",
            mask: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            if value.ndim != 3 or tuple(mask.shape) != tuple(value.shape[:2]):
                raise ValueError("Fock set input must be [B,O,F] with mask [B,O]")
            legal = mask.to(dtype=torch.bool)
            if torch.any(legal.sum(dim=-1) <= 0):
                raise ValueError("every Fock set requires a legal operation")
            if torch.any(~torch.isfinite(value) & legal[:, :, None]):
                raise ValueError("unmasked Fock set features must be finite")
            sanitized = torch.where(
                legal[:, :, None],
                torch.nan_to_num(value),
                torch.zeros_like(value),
            )
            token = self.input_projection(sanitized)
            token = self.encoder(token, src_key_padding_mask=~legal)
            token = token * legal[:, :, None].to(dtype=token.dtype)
            query = self.seed[None, :, :].expand(value.shape[0], -1, -1)
            pooled, _ = self.pool_attention(
                query,
                token,
                token,
                key_padding_mask=~legal,
                need_weights=False,
            )
            return self.output_norm(token), self.output_norm(pooled.mean(dim=1))


    class FockLeakageSetClassifier(nn.Module):
        """Synthetic Fock classifier whose set encoder initializes feedback."""

        def __init__(self, config: FockLeakageSetClassifierConfig) -> None:
            super().__init__()
            config.validate()
            self.config = config
            self.set_encoder = _FockMaskedSetTransformer(config)
            self.sector_head = nn.Linear(config.width, len(SECTOR_NAMES))
            self.operation_sector_head = nn.Linear(config.width, len(SECTOR_NAMES))
            self.operation_excitation_rate_head = nn.Linear(config.width, 1)
            self.phase_violation_head = nn.Linear(config.width, 1)
            self.double_excitation_component_head = nn.Linear(
                config.width,
                len(DOUBLE_EXCITATION_COMPONENT_NAMES),
            )
            if config.observation_conditioning_enabled:
                self.observation_projection = nn.Sequential(
                    nn.LayerNorm(config.observation_dim),
                    nn.Linear(config.observation_dim, config.width),
                    nn.SiLU(),
                    nn.Linear(config.width, config.width),
                )
                posterior_layer = nn.TransformerEncoderLayer(
                    d_model=config.width,
                    nhead=config.attention_heads,
                    dim_feedforward=int(round(
                        config.width * config.feedforward_multiplier
                    )),
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.observation_context_encoder = nn.TransformerEncoder(
                    posterior_layer,
                    num_layers=config.set_layers,
                )
                self.observation_cross_attention = nn.MultiheadAttention(
                    config.width,
                    config.attention_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                self.observation_output_norm = nn.LayerNorm(config.width)

        def forward(
            self,
            static_context: "torch.Tensor",
            static_context_mask: "torch.Tensor",
            *,
            observation_context: Optional["torch.Tensor"] = None,
            observation_mask: Optional["torch.Tensor"] = None,
        ) -> FockLeakageSetClassifierOutput:
            operation, pooled = self.set_encoder(
                static_context,
                static_context_mask.to(dtype=torch.bool),
            )
            observation_requested = any(
                value is not None for value in (observation_context, observation_mask)
            )
            if observation_requested:
                if not self.config.observation_conditioning_enabled:
                    raise ValueError("Fock observation conditioning is disabled")
                if observation_context is None or observation_mask is None:
                    raise ValueError(
                        "Fock observation context and mask must be provided together"
                    )
                if (
                    observation_context.ndim != 3
                    or observation_context.shape[:2] != static_context.shape[:2]
                    or observation_context.shape[-1] != self.config.observation_dim
                    or observation_mask.shape != static_context.shape[:2]
                ):
                    raise ValueError("Fock observation input shape changed")
                legal = observation_mask.to(dtype=torch.bool)
                if torch.any(legal.sum(dim=-1) <= 0):
                    raise ValueError("Fock posterior needs at least one observed operation")
                if torch.any(~torch.isfinite(observation_context) & legal[:, :, None]):
                    raise ValueError("unmasked Fock observations must be finite")
                sanitized = torch.where(
                    legal[:, :, None],
                    torch.nan_to_num(observation_context),
                    torch.zeros_like(observation_context),
                )
                posterior_context = (
                    operation + self.observation_projection(sanitized)
                )
                posterior_context = self.observation_context_encoder(
                    posterior_context,
                    src_key_padding_mask=~legal,
                )
                updated, _ = self.observation_cross_attention(
                    operation,
                    posterior_context,
                    posterior_context,
                    key_padding_mask=~legal,
                    need_weights=False,
                )
                operation = self.observation_output_norm(operation + updated)
                operation = operation * legal[:, :, None].to(operation.dtype)
                denominator = legal.sum(dim=-1, keepdim=True).clamp_min(1)
                pooled = self.observation_output_norm(
                    operation.sum(dim=1) / denominator.to(operation.dtype)
                )
            sector_logits = self.sector_head(pooled)
            sector_probability = torch.softmax(sector_logits, dim=-1)
            double_component_logits = self.double_excitation_component_head(pooled)
            phase_violation_probability = torch.sigmoid(
                self.phase_violation_head(pooled).squeeze(-1)
            )
            operation_excitation_rate_chi = (
                OPERATION_EXCITATION_RATE_CHI_MIN
                + (
                    OPERATION_EXCITATION_RATE_CHI_MAX
                    - OPERATION_EXCITATION_RATE_CHI_MIN
                )
                * torch.sigmoid(
                    self.operation_excitation_rate_head(operation).squeeze(-1)
                )
            )
            return FockLeakageSetClassifierOutput(
                operation_embedding=operation,
                static_embedding=pooled,
                sector_logits=sector_logits,
                sector_probability=sector_probability,
                operation_sector_log_response=self.operation_sector_head(operation),
                operation_excitation_rate_chi=operation_excitation_rate_chi,
                operation_conditional_p2_probability=(
                    (1.0 - operation_excitation_rate_chi)
                    * operation_excitation_rate_chi
                ),
                operation_conditional_tail_probability=torch.square(
                    operation_excitation_rate_chi
                ),
                w_core_mass=sector_probability[:, 1],
                leakage_mass=1.0 - sector_probability[:, 1],
                phase_violation_probability=phase_violation_probability,
                double_excitation_component_logits=double_component_logits,
                double_excitation_component_probability=torch.softmax(
                    double_component_logits,
                    dim=-1,
                ),
                double_excitation_one_body_target_fidelity=(
                    1.0 - phase_violation_probability
                ),
                real_action_supervision_mask=torch.zeros(
                    static_context.shape[0],
                    dtype=static_context.dtype,
                    device=static_context.device,
                ),
            )


    def transfer_fock_set_encoder_to_sidecar(
        classifier: "FockLeakageSetClassifier",
        sidecar: "nn.Module",
        *,
        transfer_component_head: bool = False,
    ) -> Dict[str, Any]:
        """Copy the static encoder and, only when authorized, the n=2 head.

        The static target-operation representation and the retrospective
        component posterior have separate validation contracts.  A failed or
        unavailable component-pairing gate must therefore leave both sidecar
        component heads bitwise unchanged.
        """

        if not hasattr(sidecar, "static_encoder"):
            raise ValueError("sidecar does not expose a static_encoder")
        source_state = classifier.set_encoder.state_dict()
        target_state = sidecar.static_encoder.state_dict()
        if tuple(source_state) != tuple(target_state):
            raise ValueError("Fock and sidecar static encoder structures differ")
        for name in source_state:
            if tuple(source_state[name].shape) != tuple(target_state[name].shape):
                raise ValueError(f"Fock transfer shape mismatch: {name}")
        sidecar.static_encoder.load_state_dict(source_state, strict=True)
        source_component_state = (
            classifier.double_excitation_component_head.state_dict()
        )
        if transfer_component_head:
            destination_heads = (
                sidecar.prior_decoder.double_excitation_component_head,
                sidecar.retrospective_decoder.double_excitation_component_head,
            )
            for destination in destination_heads:
                target_component_state = destination.state_dict()
                if tuple(source_component_state) != tuple(target_component_state):
                    raise ValueError("Fock and sidecar n=2 component heads differ")
                for name in source_component_state:
                    if tuple(source_component_state[name].shape) != tuple(
                        target_component_state[name].shape
                    ):
                        raise ValueError(
                            f"Fock n=2 transfer shape mismatch: {name}"
                        )
                destination.load_state_dict(source_component_state, strict=True)
        digest = hashlib.sha256()
        for name, value in sorted(source_state.items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        component_digest = hashlib.sha256()
        for name, value in sorted(source_component_state.items()):
            component_digest.update(name.encode("utf-8"))
            component_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return {
            "schema": "physics_hybrid_v8_fock_encoder_transfer_v2",
            "encoder_sha256": digest.hexdigest(),
            "double_excitation_component_head_sha256": (
                component_digest.hexdigest() if transfer_component_head else None
            ),
            "double_excitation_component_head_transferred": bool(
                transfer_component_head
            ),
            "transferred_parameter_count": int(
                sum(value.numel() for value in source_state.values())
                + (
                    sum(
                        value.numel()
                        for value in source_component_state.values()
                    )
                    if transfer_component_head
                    else 0
                )
            ),
            "source_supervision": "synthetic_fock_truth_only",
            "destination": (
                "sidecar_static_encoder_plus_predictive_and_retrospective_"
                "double_excitation_component_prior_heads"
                if transfer_component_head
                else "sidecar_static_encoder_only"
            ),
            "eventwise_real_double_excitation_label_created": False,
            "real_action_evidence_created": False,
            "policy_or_jacobian_head_transferred": False,
        }

    class FockLeakageClassifier(nn.Module):
        """Synthetic classifier whose encoder may seed static feedback features."""

        def __init__(self, config: FockLeakageClassifierConfig) -> None:
            super().__init__()
            config.validate()
            self.config = config
            layers: List[nn.Module] = []
            input_dim = config.input_dim
            for _ in range(config.hidden_layers):
                layers.extend((
                    nn.Linear(input_dim, config.width),
                    nn.GELU(),
                    nn.LayerNorm(config.width),
                    nn.Dropout(config.dropout),
                ))
                input_dim = config.width
            self.encoder = nn.Sequential(*layers)
            self.embedding_head = nn.Linear(config.width, config.embedding_dim)
            self.sector_head = nn.Linear(config.embedding_dim, len(SECTOR_NAMES))
            self.phase_violation_head = nn.Linear(config.embedding_dim, 1)
            self.double_excitation_component_head = nn.Linear(
                config.embedding_dim,
                len(DOUBLE_EXCITATION_COMPONENT_NAMES),
            )

        def forward(self, features: "torch.Tensor") -> FockLeakageClassifierOutput:
            if features.ndim != 2 or features.shape[-1] != self.config.input_dim:
                raise ValueError("Fock classifier input must have shape [B, input_dim]")
            hidden = self.encoder(features)
            embedding = self.embedding_head(hidden)
            logits = self.sector_head(embedding)
            probability = torch.softmax(logits, dim=-1)
            w_core_mass = probability[:, 1]
            leakage_mass = 1.0 - w_core_mass
            phase_violation = torch.sigmoid(
                self.phase_violation_head(embedding).squeeze(-1)
            )
            double_component_logits = self.double_excitation_component_head(embedding)
            return FockLeakageClassifierOutput(
                static_embedding=embedding,
                sector_logits=logits,
                sector_probability=probability,
                w_core_mass=w_core_mass,
                leakage_mass=leakage_mass,
                phase_violation_probability=phase_violation,
                double_excitation_component_logits=double_component_logits,
                double_excitation_component_probability=torch.softmax(
                    double_component_logits,
                    dim=-1,
                ),
                double_excitation_one_body_target_fidelity=1.0 - phase_violation,
                real_action_supervision_mask=torch.zeros(
                    features.shape[0],
                    dtype=features.dtype,
                    device=features.device,
                ),
            )

else:  # pragma: no cover

    class FockLeakageClassifier:  # type: ignore[no-redef]
        def __init__(self, config: FockLeakageClassifierConfig) -> None:
            raise RuntimeError("PyTorch is required for FockLeakageClassifier")

    class FockLeakageSetClassifier:  # type: ignore[no-redef]
        def __init__(self, config: FockLeakageSetClassifierConfig) -> None:
            raise RuntimeError("PyTorch is required for FockLeakageSetClassifier")

    def transfer_fock_set_encoder_to_sidecar(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("PyTorch is required for Fock encoder transfer")


__all__ = [
    "DLCZ_DOUBLE_EXCITATION_SCHEMA",
    "DOUBLE_EXCITATION_COMPONENT_NAMES",
    "FOCK_LEAKAGE_CONTRACT_SCHEMA",
    "FOCK_PRETRAIN_SCHEMA",
    "MAX_EXACT_BOSON_NUMBER",
    "OPERATION_EXCITATION_RATE_CHI_MAX",
    "OPERATION_EXCITATION_RATE_CHI_MIN",
    "SECTOR_NAMES",
    "FockLeakageClassifier",
    "FockLeakageClassifierConfig",
    "FockLeakageClassifierOutput",
    "FockLeakageSetClassifier",
    "FockLeakageSetClassifierConfig",
    "FockLeakageSetClassifierOutput",
    "FockLeakageContractV2",
    "FockAnalyzerKernelV2",
    "DLCZDoubleExcitationStateV2",
    "build_fock_analyzer_kernel",
    "bosonic_product_state_amplitude",
    "bosonic_two_mode_state_amplitude",
    "bosonic_occupation_basis",
    "bosonic_sector_transition_matrix",
    "bosonic_transition_probability",
    "complete_analyzer_unitary",
    "condition_excitation_prior_on_threshold_herald",
    "build_dlcz_double_excitation_state",
    "monte_carlo_selected_mode_expectation",
    "monte_carlo_sector_measurement_expectation",
    "sample_partially_identified_sector_density",
    "sample_dlcz_double_excitation_state",
    "selected_mode_occupation_probability",
    "selected_occupation_to_click_patterns",
    "sector_measurement_expectation",
    "threshold_herald_click_probability",
    "two_photon_selected_mode_probability_with_indistinguishability",
    "one_body_reduced_density",
    "transfer_fock_set_encoder_to_sidecar",
]
