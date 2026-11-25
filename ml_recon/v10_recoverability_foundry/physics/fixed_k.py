"""Sparse hard-core fixed-K state representation without a large dense path."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..config import DENSE_N_HARD_LIMIT


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + array.view(np.uint8).tobytes()).hexdigest()


@dataclass(frozen=True)
class FixedKState:
    """Normalized sparse state in the hard-core basis ``|S|=K``."""

    n: int
    k: int
    support: tuple[tuple[int, ...], ...]
    amplitudes: np.ndarray

    def __post_init__(self) -> None:
        if not 3 <= self.n <= 256 or not 1 <= self.k <= min(5, self.n):
            raise ValueError("FixedKState requires 3<=N<=256 and 1<=K<=min(5,N)")
        if not self.support:
            raise ValueError("fixed-K support cannot be empty")
        canonical = tuple(tuple(sorted(int(mode) for mode in item)) for item in self.support)
        if canonical != self.support or len(set(canonical)) != len(canonical):
            raise ValueError("support entries must be sorted and unique")
        if any(len(item) != self.k or len(set(item)) != self.k
               or min(item) < 0 or max(item) >= self.n for item in canonical):
            raise ValueError("support entries violate the hard-core fixed-K contract")
        amplitudes = np.asarray(self.amplitudes, dtype=np.complex128).reshape(-1).copy()
        if amplitudes.shape != (len(canonical),) or not np.all(np.isfinite(amplitudes)):
            raise ValueError("one finite complex amplitude is required per support entry")
        norm = float(np.vdot(amplitudes, amplitudes).real)
        if norm <= 0.0:
            raise ValueError("fixed-K amplitudes cannot be zero")
        amplitudes /= np.sqrt(norm)
        amplitudes.setflags(write=False)
        object.__setattr__(self, "amplitudes", amplitudes)

    @property
    def full_basis_size(self) -> int:
        return math.comb(self.n, self.k)

    @property
    def sparse_term_count(self) -> int:
        return len(self.support)

    def state_sha(self) -> str:
        support = np.asarray(self.support, dtype=np.int16)
        payload = f"n={self.n}|k={self.k}|".encode("ascii")
        return hashlib.sha256(
            payload
            + bytes.fromhex(_array_sha(support))
            + bytes.fromhex(_array_sha(self.amplitudes))
        ).hexdigest()

    def dense_state_vector(self, *, dense_n_limit: int = 16) -> np.ndarray:
        if not 1 <= int(dense_n_limit) <= DENSE_N_HARD_LIMIT:
            raise ValueError(
                f"dense_n_limit cannot exceed the hard safety boundary {DENSE_N_HARD_LIMIT}"
            )
        if self.n > dense_n_limit:
            raise ValueError("dense 2^N materialization is forbidden above dense_n_limit")
        vector = np.zeros(2 ** self.n, dtype=np.complex128)
        for modes, amplitude in zip(self.support, self.amplitudes):
            index = sum(1 << (self.n - 1 - mode) for mode in modes)
            vector[index] = amplitude
        return vector

    def coefficient_matrix_k2(self) -> np.ndarray:
        if self.k != 2:
            raise ValueError("the symmetric coefficient matrix is defined only for K=2")
        matrix = np.zeros((self.n, self.n), dtype=np.complex128)
        for (left, right), amplitude in zip(self.support, self.amplitudes):
            matrix[left, right] = amplitude
            matrix[right, left] = amplitude
        if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0.0):
            raise RuntimeError("internal K=2 symmetry construction failed")
        return matrix

    def one_rdm(self) -> np.ndarray:
        """Return ``gamma[i,j]=<sigma_i^+ sigma_j^->`` with trace K."""

        index = {modes: amplitude for modes, amplitude in zip(self.support, self.amplitudes)}
        gamma = np.zeros((self.n, self.n), dtype=np.complex128)
        for modes, amplitude in index.items():
            occupied = set(modes)
            probability = float(abs(amplitude) ** 2)
            for mode in occupied:
                gamma[mode, mode] += probability
            for source in occupied:
                for destination in range(self.n):
                    if destination in occupied:
                        continue
                    target_modes = tuple(sorted((occupied - {source}) | {destination}))
                    target_amplitude = index.get(target_modes)
                    if target_amplitude is not None:
                        gamma[destination, source] += np.conj(target_amplitude) * amplitude
        gamma = 0.5 * (gamma + gamma.conj().T)
        if not np.isclose(np.trace(gamma).real, self.k, atol=1e-9):
            raise RuntimeError("one-RDM trace does not equal K")
        return gamma

    def pair_occupancy(self) -> np.ndarray:
        result = np.zeros((self.n, self.n), dtype=np.float64)
        for modes, amplitude in zip(self.support, self.amplitudes):
            probability = float(abs(amplitude) ** 2)
            for left, right in itertools.combinations(modes, 2):
                result[left, right] += probability
                result[right, left] += probability
        return result

    def selected_functionals(self) -> Dict[str, np.ndarray | float | int]:
        one_rdm = self.one_rdm()
        populations = np.real(np.diag(one_rdm))
        return {
            "n": self.n,
            "k": self.k,
            "support_fraction": self.sparse_term_count / self.full_basis_size,
            "populations": populations,
            "one_rdm_real": one_rdm.real,
            "one_rdm_imag": one_rdm.imag,
            "pair_occupancy": self.pair_occupancy(),
            "purity_one_rdm_normalized": float(np.trace(one_rdm @ one_rdm).real / (self.k ** 2)),
        }


def _sample_support(
    *,
    n: int,
    k: int,
    term_count: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], ...]:
    full_size = math.comb(n, k)
    if term_count >= full_size:
        return tuple(itertools.combinations(range(n), k))
    if full_size <= 4 * term_count:
        candidates = tuple(itertools.combinations(range(n), k))
        selected = rng.choice(len(candidates), size=term_count, replace=False)
        return tuple(sorted(candidates[int(index)] for index in selected))
    support: set[tuple[int, ...]] = set()
    while len(support) < term_count:
        support.add(tuple(sorted(int(value) for value in rng.choice(n, size=k, replace=False))))
    return tuple(sorted(support))


def random_fixed_k_state(
    n: int,
    k: int,
    rng: np.random.Generator,
    *,
    max_sparse_terms: int = 4096,
) -> FixedKState:
    if max_sparse_terms <= 0:
        raise ValueError("max_sparse_terms must be positive")
    term_count = min(math.comb(n, k), int(max_sparse_terms))
    support = _sample_support(n=n, k=k, term_count=term_count, rng=rng)
    amplitudes = rng.normal(size=term_count) + 1j * rng.normal(size=term_count)
    return FixedKState(n=n, k=k, support=support, amplitudes=amplitudes)
