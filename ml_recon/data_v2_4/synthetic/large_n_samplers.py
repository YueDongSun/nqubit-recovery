"""Sequential compact-backend Pauli sampler with exact small-N parity support."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ml_recon.physics_hybrid_v10.measurement_model import born_probabilities
from ml_recon.physics_hybrid_v10.state_backends import (
    MPSStateBackend,
    ProductBlockStateBackend,
    V10StateBackend,
    outcome_projector,
)


def _partial_probability_mps(backend: MPSStateBackend, axes: Sequence[str],
                             prefix: Sequence[int]) -> float:
    environment = np.ones((1, 1), dtype=np.complex128)
    for site, tensor in enumerate(backend.tensors):
        operator = (outcome_projector(axes[site], int(prefix[site]))
                    if site < len(prefix) else np.eye(2, dtype=np.complex128))
        environment = np.einsum(
            "ab,asr,st,btu->ru", environment, tensor.conj(), operator, tensor,
            optimize=True,
        )
    return float(np.clip(environment[0, 0].real, 0.0, 1.0))


def _sample_mps(backend: MPSStateBackend, axes: Sequence[str], shots: int,
                rng: np.random.Generator, uniforms: np.ndarray | None = None) -> np.ndarray:
    panel = rng.random((shots, backend.N)) if uniforms is None else np.asarray(uniforms)
    if panel.shape != (shots, backend.N):
        raise ValueError("sequential uniform panel must have shape [shots,N]")
    outcomes = np.empty((shots, backend.N), dtype=np.int8)
    for shot in range(shots):
        prefix: list[int] = []
        prefix_probability = 1.0
        for site in range(backend.N):
            plus_probability = _partial_probability_mps(backend, axes, (*prefix, 1))
            conditional_plus = plus_probability / max(prefix_probability, 1e-300)
            value = 1 if panel[shot, site] < np.clip(conditional_plus, 0.0, 1.0) else -1
            prefix.append(value)
            prefix_probability = _partial_probability_mps(backend, axes, prefix)
        outcomes[shot] = prefix
    return outcomes


def _sample_product_blocks(backend: ProductBlockStateBackend, axes: Sequence[str], shots: int,
                           rng: np.random.Generator, uniforms: np.ndarray | None) -> np.ndarray:
    panel = rng.random((shots, backend.N)) if uniforms is None else np.asarray(uniforms)
    if panel.shape != (shots, backend.N):
        raise ValueError("sequential uniform panel must have shape [shots,N]")
    result = np.empty((shots, backend.N), dtype=np.int8)
    for block_index, (qubits, block) in enumerate(backend.blocks):
        block_axes = tuple(axes[index] for index in qubits)
        block_uniforms = panel[:, list(qubits)]
        block_rng = np.random.Generator(np.random.Philox(
            int(rng.integers(0, 2 ** 63 - 1)) + block_index))
        if isinstance(block, MPSStateBackend):
            sampled = _sample_mps(block, block_axes, shots, block_rng, block_uniforms)
        elif isinstance(block, ProductBlockStateBackend):
            sampled = _sample_product_blocks(
                block, block_axes, shots, block_rng, block_uniforms)
        elif block.N <= 16:
            outcomes, probabilities = born_probabilities(block, block_axes)
            scalar_uniforms = block_uniforms[:, 0]
            indices = np.searchsorted(np.cumsum(probabilities), scalar_uniforms, side="right")
            sampled = outcomes[np.minimum(indices, len(outcomes) - 1)]
        else:
            raise ValueError("large product blocks require an MPS sequential backend")
        result[:, list(qubits)] = sampled
    return result


def sample_global_pauli_outcomes(backend: V10StateBackend, axes: Sequence[str], shots: int,
                                 rng: np.random.Generator, *,
                                 uniforms: np.ndarray | None = None) -> np.ndarray:
    """Sample without constructing a ``2^N`` outcome table for compact large-N states."""
    if shots < 0 or len(axes) != backend.N:
        raise ValueError("invalid sequential Pauli sampling request")
    if isinstance(backend, MPSStateBackend):
        return _sample_mps(backend, axes, shots, rng, uniforms)
    if isinstance(backend, ProductBlockStateBackend):
        return _sample_product_blocks(backend, axes, shots, rng, uniforms)
    from .state_renderers import ExplicitFockMixtureBackend, w_mps_from_coefficients
    if isinstance(backend, ExplicitFockMixtureBackend):
        panel = rng.random((shots, backend.N + 2)) if uniforms is None else np.asarray(uniforms)
        if panel.shape[0] != shots or panel.shape[1] < backend.N + 2:
            raise ValueError("explicit Fock sampler requires [shots,N+2] uniforms")
        result = np.empty((shots, backend.N), dtype=np.int8)
        p1 = float(np.trace(backend.R1).real)
        eigenvalues, eigenvectors = np.linalg.eigh(backend.R1)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        eigenvalues /= max(eigenvalues.sum(), 1e-15)
        pair_indices = tuple((i, j) for i in range(backend.N)
                             for j in range(i + 1, backend.N))
        pair_probability = np.asarray([backend.pair_probabilities[i, j]
                                       for i, j in pair_indices])
        for shot in range(shots):
            sector_uniform = panel[shot, 0]
            bits = np.zeros(backend.N, dtype=np.int8)
            if sector_uniform >= backend.p0 and sector_uniform < backend.p0 + p1:
                component = int(np.searchsorted(np.cumsum(eigenvalues), panel[shot, 1], side="right"))
                component = min(component, backend.N - 1)
                pure_backend = w_mps_from_coefficients(eigenvectors[:, component])
                result[shot] = _sample_mps(
                    pure_backend, axes, 1, rng, panel[shot:shot + 1, 2:])[0]
                continue
            elif sector_uniform >= backend.p0 + p1:
                pair_index = int(np.searchsorted(np.cumsum(pair_probability), panel[shot, 1], side="right"))
                pair = pair_indices[min(pair_index, len(pair_indices) - 1)]
                bits[list(pair)] = 1
            for qubit, bit in enumerate(bits):
                plus = float(outcome_projector(axes[qubit], 1)[bit, bit].real)
                result[shot, qubit] = 1 if panel[shot, qubit + 2] < plus else -1
        return result
    if backend.N > 16:
        raise ValueError("N>16 sampling requires an explicit compact sequential backend")
    outcomes, probabilities = born_probabilities(backend, axes)
    panel = rng.random(shots) if uniforms is None else np.asarray(uniforms)[:, 0]
    indices = np.searchsorted(np.cumsum(probabilities), panel, side="right")
    return outcomes[np.minimum(indices, len(outcomes) - 1)]


def exact_sequential_parity(backend: V10StateBackend, axes: Sequence[str], shots: int,
                            seed: int = 0) -> float:
    if not isinstance(backend, MPSStateBackend) or backend.N > 12:
        raise ValueError("parity audit requires a small MPS backend")
    outcomes, exact = born_probabilities(backend, axes)
    rng = np.random.Generator(np.random.Philox(seed))
    sampled = sample_global_pauli_outcomes(backend, axes, shots, rng)
    lookup = {tuple(row): index for index, row in enumerate(outcomes)}
    counts = np.zeros(len(outcomes), dtype=np.float64)
    for row in sampled:
        counts[lookup[tuple(row)]] += 1
    observed = counts / max(shots, 1)
    return float(0.5 * np.abs(observed - exact).sum())
