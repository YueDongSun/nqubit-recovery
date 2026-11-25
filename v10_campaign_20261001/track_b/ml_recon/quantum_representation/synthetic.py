"""Synthetic density-matrix pairs for quantum representation pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .features import density_fidelity_pure_target


ERROR_SOURCES = [
    "phase_drift",
    "amp_imbalance",
    "dephasing",
    "retrieval_imbalance",
    "double_excitation",
]


@dataclass
class SyntheticRhoSample:
    rho_target: np.ndarray
    rho_measured: np.ndarray
    excitation_mode: int
    error_source: str
    fidelity: float
    purity: float


def generate_synthetic_rho_dataset(
    num_samples: int,
    num_qubits: int = 4,
    seed: int = 1234,
) -> List[SyntheticRhoSample]:
    """Generate W-like rho pairs with labelled perturbations."""

    if num_qubits not in {3, 4}:
        raise ValueError("num_qubits must be 3 or 4")
    rng = np.random.default_rng(seed)
    samples: List[SyntheticRhoSample] = []
    for _ in range(int(num_samples)):
        mode = int(rng.integers(1, num_qubits + 1))
        amps = rng.uniform(0.7, 1.3, size=num_qubits)
        phases = rng.uniform(-np.pi, np.pi, size=num_qubits)
        rho_target = _w_density(num_qubits, amps, phases)
        error_source = ERROR_SOURCES[int(rng.integers(0, len(ERROR_SOURCES)))]
        rho_measured = _apply_error(rho_target, num_qubits, mode, error_source, rng)
        purity = float(np.real(np.trace(rho_measured @ rho_measured)))
        fidelity = density_fidelity_pure_target(rho_target, rho_measured)
        samples.append(SyntheticRhoSample(
            rho_target=rho_target,
            rho_measured=rho_measured,
            excitation_mode=mode,
            error_source=error_source,
            fidelity=fidelity,
            purity=purity,
        ))
    return samples


def _w_density(num_qubits: int, amps: np.ndarray, phases: np.ndarray) -> np.ndarray:
    dim = 2 ** num_qubits
    psi = np.zeros(dim, dtype=complex)
    amps = np.asarray(amps, dtype=float)
    phases = np.asarray(phases, dtype=float)
    norm = np.linalg.norm(amps)
    if norm <= 0:
        amps = np.ones(num_qubits, dtype=float)
        norm = np.linalg.norm(amps)
    for i in range(num_qubits):
        idx = 2 ** (num_qubits - 1 - i)
        psi[idx] = amps[i] * np.exp(1j * phases[i]) / norm
    return np.outer(psi, psi.conj())


def _apply_error(
    rho: np.ndarray,
    num_qubits: int,
    excitation_mode: int,
    error_source: str,
    rng: np.random.Generator,
) -> np.ndarray:
    dim = rho.shape[0]
    single = [2 ** (num_qubits - 1 - i) for i in range(num_qubits)]
    measured = np.array(rho, dtype=complex, copy=True)

    if error_source == "phase_drift":
        drift = rng.normal(0.0, 0.35, size=num_qubits)
        phase = np.ones(dim, dtype=complex)
        for i, idx in enumerate(single):
            phase[idx] = np.exp(1j * drift[i])
        op = np.diag(phase)
        measured = op @ measured @ op.conj().T

    elif error_source == "amp_imbalance":
        scale = np.ones(dim, dtype=float)
        imbalance = rng.uniform(0.65, 1.35, size=num_qubits)
        for i, idx in enumerate(single):
            scale[idx] = imbalance[i]
        op = np.diag(scale)
        measured = op @ measured @ op.conj().T

    elif error_source == "dephasing":
        strength = rng.uniform(0.15, 0.55)
        diag = np.diag(np.diag(measured))
        measured = (1.0 - strength) * measured + strength * diag

    elif error_source == "retrieval_imbalance":
        retrieval = rng.uniform(0.45, 1.0, size=num_qubits)
        scale = np.ones(dim, dtype=float)
        for i, idx in enumerate(single):
            scale[idx] = retrieval[i]
        op = np.diag(scale)
        measured = op @ measured @ op.conj().T

    elif error_source == "double_excitation":
        leak = rng.uniform(0.01, 0.08)
        leak_state = _double_excitation_state(num_qubits, rng)
        measured = (1.0 - leak) * measured + leak * np.outer(leak_state, leak_state.conj())

    measured = _mode_mix(measured, num_qubits, excitation_mode, rng)
    measured = _physicalize(measured)
    return measured


def _mode_mix(rho: np.ndarray, num_qubits: int, excitation_mode: int, rng: np.random.Generator) -> np.ndarray:
    """Inject coarse excitation-mode signatures into rho."""

    signature = _excitation_mode_signature(num_qubits, excitation_mode)
    strength = rng.uniform(0.38, 0.58)
    return (1.0 - strength) * rho + strength * signature


def _excitation_mode_signature(num_qubits: int, excitation_mode: int) -> np.ndarray:
    """Return a density matrix with the coherence pattern of a W_CH1 pulse mode.

    The UI excitation modes are pulse-pattern classes, not individual qubit
    populations. Synthetic pretraining therefore needs mode-specific coherence
    structure in the single-excitation subspace:
    simultaneous pulses keep broad coherence, grouped modes keep block
    coherence, and sequential modes are mostly diagonal with weak phase-ordered
    coherence.
    """

    dim = 2 ** num_qubits
    single = [2 ** (num_qubits - 1 - i) for i in range(num_qubits)]
    sub = np.eye(num_qubits, dtype=complex) / num_qubits

    if excitation_mode == 1:
        sub = np.ones((num_qubits, num_qubits), dtype=complex) / num_qubits
    elif num_qubits == 3 and excitation_mode == 2:
        sub = _block_coherence(num_qubits, [(0, 1), (2,)], cross_scale=0.18, phase_step=np.pi / 5.0)
    elif num_qubits == 3 and excitation_mode == 3:
        sub = _sequential_coherence(num_qubits, coherence=0.12, phase_step=np.pi / 2.0)
    elif num_qubits == 4 and excitation_mode == 2:
        sub = _block_coherence(num_qubits, [(0, 1, 2), (3,)], cross_scale=0.12, phase_step=np.pi / 6.0)
    elif num_qubits == 4 and excitation_mode == 3:
        sub = _block_coherence(num_qubits, [(0, 1), (2, 3)], cross_scale=0.08, phase_step=np.pi / 3.0)
    elif num_qubits == 4 and excitation_mode == 4:
        sub = _sequential_coherence(num_qubits, coherence=0.08, phase_step=np.pi / 2.0)

    signature = np.zeros((dim, dim), dtype=complex)
    for row, idx in enumerate(single):
        for col, jdx in enumerate(single):
            signature[idx, jdx] = sub[row, col]
    return _physicalize(signature)


def _block_coherence(
    num_qubits: int,
    groups: list[tuple[int, ...]],
    cross_scale: float,
    phase_step: float,
) -> np.ndarray:
    sub = np.eye(num_qubits, dtype=complex) / num_qubits
    phase = np.exp(1j * phase_step * np.arange(num_qubits))
    for group in groups:
        for i in group:
            for j in group:
                sub[i, j] = phase[i] * np.conj(phase[j]) / num_qubits
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j and abs(sub[i, j]) < 1e-12:
                sub[i, j] = cross_scale * phase[i] * np.conj(phase[j]) / num_qubits
    return sub


def _sequential_coherence(num_qubits: int, coherence: float, phase_step: float) -> np.ndarray:
    sub = np.eye(num_qubits, dtype=complex) / num_qubits
    phase = np.exp(1j * phase_step * np.arange(num_qubits))
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                continue
            distance = abs(i - j)
            sub[i, j] = (coherence ** distance) * phase[i] * np.conj(phase[j]) / num_qubits
    return sub


def _double_excitation_state(num_qubits: int, rng: np.random.Generator) -> np.ndarray:
    dim = 2 ** num_qubits
    pairs = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            idx = 2 ** (num_qubits - 1 - i) + 2 ** (num_qubits - 1 - j)
            pairs.append(idx)
    psi = np.zeros(dim, dtype=complex)
    idx = pairs[int(rng.integers(0, len(pairs)))]
    psi[idx] = 1.0
    return psi


def _physicalize(rho: np.ndarray) -> np.ndarray:
    herm = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = np.linalg.eigh(herm)
    eigvals = np.clip(np.real(eigvals), 0.0, None)
    if np.sum(eigvals) <= 0:
        return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
    out = eigvecs @ np.diag(eigvals / np.sum(eigvals)) @ eigvecs.conj().T
    return 0.5 * (out + out.conj().T)
