"""Physically explicit state backends for V10 Pauli measurements.

Backends expose one narrow contract: the Born probability for a *global* local-
Pauli setting and a concrete +/-1 outcome.  This keeps measurement semantics
identical across dense, product-block, stabilizer, MPS, and W-subspace states.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple

import numpy as np


PAULI_MATRICES = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _validate_axes_outcome(N: int, axes: Sequence[str], outcome: Sequence[int]) -> None:
    if len(axes) != N or len(outcome) != N:
        raise ValueError("global Pauli axes and outcome must both have length N")
    if any(axis not in ("X", "Y", "Z") for axis in axes):
        raise ValueError("physical Pauli axes must be X, Y, or Z")
    if any(int(value) not in (-1, 1) for value in outcome):
        raise ValueError("Pauli outcomes must be +/-1")


def outcome_projector(axis: str, outcome: int) -> np.ndarray:
    if axis not in ("X", "Y", "Z") or int(outcome) not in (-1, 1):
        raise ValueError("invalid local Pauli measurement")
    return 0.5 * (PAULI_MATRICES["I"] + int(outcome) * PAULI_MATRICES[axis])


def all_pauli_outcomes(N: int) -> np.ndarray:
    return np.asarray(tuple(itertools.product((-1, 1), repeat=N)), dtype=np.int8)


def _array_sha(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    payload = (str(contiguous.dtype) + "|" + repr(contiguous.shape)).encode("ascii")
    return hashlib.sha256(payload + contiguous.view(np.uint8).tobytes()).hexdigest()


class V10StateBackend(Protocol):
    N: int
    backend_name: str

    def probability(self, axes: Sequence[str], outcome: Sequence[int]) -> float:
        ...

    def state_sha(self) -> str:
        ...


@dataclass(frozen=True)
class DenseDensityStateBackend:
    rho: np.ndarray
    backend_name: str = "dense_density"

    def __post_init__(self) -> None:
        rho = np.asarray(self.rho, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
            raise ValueError("dense rho must be a non-trivial square matrix")
        N = int(round(np.log2(rho.shape[0])))
        if 2 ** N != rho.shape[0]:
            raise ValueError("dense rho dimension must be a power of two")
        if not np.allclose(rho, rho.conj().T, atol=1e-10):
            raise ValueError("rho must be Hermitian")
        if not np.isclose(np.trace(rho).real, 1.0, atol=1e-9):
            raise ValueError("rho must have unit trace")
        if np.linalg.eigvalsh(rho).min(initial=0.0) < -1e-9:
            raise ValueError("rho must be positive semidefinite")
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "N", N)

    @classmethod
    def from_state_vector(cls, state: np.ndarray) -> "DenseDensityStateBackend":
        vector = np.asarray(state, dtype=np.complex128).reshape(-1)
        norm = np.linalg.norm(vector)
        if norm <= 0.0:
            raise ValueError("state vector cannot be zero")
        vector = vector / norm
        return cls(np.outer(vector, vector.conj()))

    def probability(self, axes: Sequence[str], outcome: Sequence[int]) -> float:
        _validate_axes_outcome(self.N, axes, outcome)
        projector = np.array([[1.0 + 0.0j]])
        for axis, value in zip(axes, outcome):
            projector = np.kron(projector, outcome_projector(axis, int(value)))
        probability = float(np.trace(self.rho @ projector).real)
        return float(np.clip(probability, 0.0, 1.0))

    def state_sha(self) -> str:
        return _array_sha(self.rho)


@dataclass(frozen=True)
class ProductBlockStateBackend:
    """Tensor product of disjoint state backends on explicit physical qubits."""

    N: int
    blocks: Tuple[Tuple[Tuple[int, ...], V10StateBackend], ...]
    backend_name: str = "product_block"

    def __post_init__(self) -> None:
        covered = []
        for qubits, backend in self.blocks:
            if len(qubits) != backend.N:
                raise ValueError("block qubit count does not match backend N")
            covered.extend(qubits)
        if sorted(covered) != list(range(self.N)) or len(covered) != len(set(covered)):
            raise ValueError("product blocks must partition every physical qubit")

    def probability(self, axes: Sequence[str], outcome: Sequence[int]) -> float:
        _validate_axes_outcome(self.N, axes, outcome)
        probability = 1.0
        for qubits, backend in self.blocks:
            probability *= backend.probability(
                tuple(axes[q] for q in qubits), tuple(int(outcome[q]) for q in qubits))
        return float(np.clip(probability, 0.0, 1.0))

    def state_sha(self) -> str:
        payload = "|".join(
            f"{','.join(map(str, qubits))}:{backend.state_sha()}"
            for qubits, backend in self.blocks)
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


class StabilizerStateBackend(DenseDensityStateBackend):
    """Small-N stabilizer state constructed from commuting Pauli generators."""

    backend_name = "stabilizer"

    def __init__(self, generators: Sequence[str]):
        generators = tuple(str(generator).upper() for generator in generators)
        if not generators:
            raise ValueError("at least one stabilizer generator is required")
        N = len(generators[0])
        if N < 3 or N > 12 or len(generators) != N:
            raise ValueError("the dense stabilizer backend requires N independent generators, 3<=N<=12")
        if any(len(generator) != N or any(axis not in PAULI_MATRICES for axis in generator)
               for generator in generators):
            raise ValueError("invalid Pauli stabilizer generator")
        dim = 2 ** N
        rho = np.eye(dim, dtype=np.complex128)
        for generator in generators:
            operator = np.array([[1.0 + 0.0j]])
            for axis in generator:
                operator = np.kron(operator, PAULI_MATRICES[axis])
            if not np.allclose(operator @ rho, rho @ operator, atol=1e-10):
                raise ValueError("stabilizer generators do not commute")
            rho = rho @ (np.eye(dim, dtype=np.complex128) + operator) / 2.0
        trace = float(np.trace(rho).real)
        if trace <= 0.0:
            raise ValueError("stabilizer projector has zero rank")
        rho /= trace
        super().__init__(rho=rho, backend_name="stabilizer")


@dataclass(frozen=True)
class MPSStateBackend:
    """Pure matrix-product state with tensors shaped ``[left, 2, right]``."""

    tensors: Tuple[np.ndarray, ...]
    backend_name: str = "mps"

    def __post_init__(self) -> None:
        tensors = tuple(np.asarray(tensor, dtype=np.complex128) for tensor in self.tensors)
        if len(tensors) < 3:
            raise ValueError("MPS backend requires at least three sites")
        if tensors[0].shape[0] != 1 or tensors[-1].shape[2] != 1:
            raise ValueError("MPS boundary bond dimensions must be one")
        for index, tensor in enumerate(tensors):
            if tensor.ndim != 3 or tensor.shape[1] != 2:
                raise ValueError("MPS tensors must have shape [left, 2, right]")
            if index and tensors[index - 1].shape[2] != tensor.shape[0]:
                raise ValueError("adjacent MPS bond dimensions disagree")
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "N", len(tensors))
        norm = self._expectation(tuple("I" for _ in tensors), tuple(1 for _ in tensors))
        if not np.isclose(norm, 1.0, atol=1e-8):
            raise ValueError("MPS must be normalized")

    @classmethod
    def from_state_vector(cls, state: np.ndarray, N: int, max_bond: int | None = None
                          ) -> "MPSStateBackend":
        vector = np.asarray(state, dtype=np.complex128).reshape(-1)
        if vector.size != 2 ** N:
            raise ValueError("state vector dimension does not match N")
        vector = vector / np.linalg.norm(vector)
        tensors = []
        remainder = vector.reshape(1, -1)
        left_dim = 1
        for site in range(N - 1):
            matrix = remainder.reshape(left_dim * 2, -1)
            u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
            rank = len(singular) if max_bond is None else min(len(singular), max_bond)
            tensors.append(u[:, :rank].reshape(left_dim, 2, rank))
            remainder = singular[:rank, None] * vh[:rank]
            left_dim = rank
        tensors.append(remainder.reshape(left_dim, 2, 1))
        environment = np.ones((1, 1), dtype=np.complex128)
        identity = PAULI_MATRICES["I"]
        for tensor in tensors:
            environment = np.einsum(
                "ab,asr,st,btu->ru", environment, tensor.conj(), identity,
                tensor, optimize=True)
        truncated_norm = float(environment[0, 0].real)
        if truncated_norm <= 0.0:
            raise ValueError("truncated MPS has zero norm")
        tensors[-1] = tensors[-1] / np.sqrt(truncated_norm)
        return cls(tuple(tensors))

    def _expectation(self, axes: Sequence[str], outcome: Sequence[int]) -> float:
        environment = np.ones((1, 1), dtype=np.complex128)
        for tensor, axis, value in zip(self.tensors, axes, outcome):
            operator = (PAULI_MATRICES["I"] if axis == "I"
                        else outcome_projector(axis, int(value)))
            environment = np.einsum(
                "ab,asr,st,btu->ru", environment, tensor.conj(), operator, tensor,
                optimize=True)
        return float(environment[0, 0].real)

    def two_qubit_reduced_density(self, site_a: int, site_b: int) -> np.ndarray:
        """Exact two-site reduced density matrix in the ``|q_a q_b>`` basis.

        Ket bonds run along the first index of each contraction, bra bonds
        along the second, so every stage is the matching transfer contraction.
        The result is ordered rows = ket ``|q_a q_b>``, columns = bra.
        """
        if site_a == site_b or not 0 <= site_a < self.N or not 0 <= site_b < self.N:
            raise ValueError("two distinct in-range site indices are required")
        left, right = sorted((site_a, site_b))
        environment = np.ones((1, 1), dtype=np.complex128)
        for tensor in self.tensors[:left]:
            environment = np.einsum(
                "lp,lsr,psq->rq", environment, tensor.conj(), tensor, optimize=True)
        carry = np.einsum(
            "lp,lsr,ptq->strq", environment,
            self.tensors[left].conj(), self.tensors[left], optimize=True)
        for tensor in self.tensors[left + 1:right]:
            carry = np.einsum(
                "stuv,uxr,vxq->strq", carry, tensor.conj(), tensor, optimize=True)
        carry = np.einsum(
            "stuv,uyr,vzq->styzrq", carry,
            self.tensors[right].conj(), self.tensors[right], optimize=True)
        environment = np.ones((1, 1), dtype=np.complex128)
        # The right environment grows leftward from the boundary bond, so it
        # is built from the LAST tensor inward and enters each tensor on its
        # RIGHT bond (stage 1 above enters on the LEFT bond — the mirror
        # pattern).
        for tensor in reversed(self.tensors[right + 1:]):
            environment = np.einsum(
                "rq,lsr,psq->lp", environment, tensor.conj(), tensor, optimize=True)
        # carry axes: s = bra phys at left, t = ket phys at left,
        # y = bra phys at right, z = ket phys at right.
        reduced = np.einsum("styzrq,rq->tzsy", carry, environment, optimize=True)
        return reduced.reshape(4, 4)

    def probability(self, axes: Sequence[str], outcome: Sequence[int]) -> float:
        _validate_axes_outcome(self.N, axes, outcome)
        return float(np.clip(self._expectation(axes, outcome), 0.0, 1.0))

    def state_sha(self) -> str:
        payload = "|".join(_array_sha(tensor) for tensor in self.tensors)
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class WSubspaceStateBackend:
    """Vacuum plus single-excitation density block ``p0 |0><0| + R1``."""

    p0: float
    R1: np.ndarray
    p_ge2: float = 0.0
    backend_name: str = "w_subspace"

    def __post_init__(self) -> None:
        R1 = np.asarray(self.R1, dtype=np.complex128)
        if R1.ndim != 2 or R1.shape[0] != R1.shape[1] or R1.shape[0] < 3:
            raise ValueError("R1 must be a square single-excitation block")
        if not np.allclose(R1, R1.conj().T, atol=1e-10):
            raise ValueError("R1 must be Hermitian")
        if np.linalg.eigvalsh(R1).min(initial=0.0) < -1e-9:
            raise ValueError("R1 must be positive semidefinite")
        if self.p0 < -1e-12 or self.p_ge2 < -1e-12:
            raise ValueError("sector masses must be non-negative")
        if not np.isclose(self.p0 + np.trace(R1).real + self.p_ge2, 1.0, atol=1e-9):
            raise ValueError("p0 + Tr(R1) + p_ge2 must equal one")
        if self.p_ge2 > 1e-12:
            raise ValueError("Born sampling requires an explicit >=2 excitation backend")
        object.__setattr__(self, "R1", R1)
        object.__setattr__(self, "N", R1.shape[0])

    @classmethod
    def pure_w(cls, amplitudes: np.ndarray, p0: float = 0.0) -> "WSubspaceStateBackend":
        amplitudes = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        R1 = (1.0 - p0) * np.outer(amplitudes, amplitudes.conj())
        return cls(float(p0), R1)

    def probability(self, axes: Sequence[str], outcome: Sequence[int]) -> float:
        _validate_axes_outcome(self.N, axes, outcome)
        projectors = [outcome_projector(axis, int(value))
                      for axis, value in zip(axes, outcome)]
        vacuum_diagonal = np.asarray([projector[0, 0] for projector in projectors])
        vacuum_probability = np.prod(vacuum_diagonal)
        contribution = self.p0 * vacuum_probability
        for i in range(self.N):
            for j in range(self.N):
                local = 1.0 + 0.0j
                for qubit, projector in enumerate(projectors):
                    ket = 1 if qubit == i else 0
                    bra = 1 if qubit == j else 0
                    local *= projector[bra, ket]
                contribution += self.R1[i, j] * local
        return float(np.clip(contribution.real, 0.0, 1.0))

    def state_sha(self) -> str:
        payload = np.asarray((self.p0, self.p_ge2), dtype=np.float64).tobytes()
        return hashlib.sha256(payload + self.R1.view(np.float64).tobytes()).hexdigest()


def named_state_vector(name: str, N: int) -> np.ndarray:
    """Analytic reference states used by tests and deterministic synthetic fixtures."""
    name = name.lower()
    vector = np.zeros(2 ** N, dtype=np.complex128)
    if name == "zero":
        vector[0] = 1.0
    elif name == "ghz":
        vector[0] = vector[-1] = 1.0 / np.sqrt(2.0)
    elif name in ("w", "w_state"):
        for qubit in range(N):
            vector[1 << (N - 1 - qubit)] = 1.0 / np.sqrt(N)
    elif name == "bell_blocks":
        if N % 2:
            raise ValueError("bell_blocks requires even N")
        vector = np.array([1.0 + 0.0j])
        bell = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        for _ in range(N // 2):
            vector = np.kron(vector, bell)
    else:
        raise ValueError(f"unknown named state: {name}")
    return vector
