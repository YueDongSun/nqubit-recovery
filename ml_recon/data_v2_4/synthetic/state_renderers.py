"""State renderer adapters and explicit multi-excitation compact backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ml_recon.physics_hybrid_v10.state_backends import (
    MPSStateBackend,
    WSubspaceStateBackend,
    outcome_projector,
)


def w_mps_from_coefficients(coefficients: np.ndarray) -> MPSStateBackend:
    coefficients = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    coefficients /= np.linalg.norm(coefficients)
    tensors = []
    for site, coefficient in enumerate(coefficients):
        if site == 0:
            tensor = np.zeros((1, 2, 2), dtype=np.complex128)
            tensor[0, 0, 0] = 1.0
            tensor[0, 1, 1] = coefficient
        elif site == len(coefficients) - 1:
            tensor = np.zeros((2, 2, 1), dtype=np.complex128)
            tensor[1, 0, 0] = 1.0
            tensor[0, 1, 0] = coefficient
        else:
            tensor = np.zeros((2, 2, 2), dtype=np.complex128)
            tensor[0, 0, 0] = tensor[1, 0, 1] = 1.0
            tensor[0, 1, 1] = coefficient
        tensors.append(tensor)
    return MPSStateBackend(tuple(tensors))


@dataclass(frozen=True)
class ExplicitFockMixtureBackend:
    """Vacuum/single/two-excitation mixture with an explicit two-sector law."""

    p0: float
    R1: np.ndarray
    pair_probabilities: np.ndarray
    p2: float
    p3_plus: float = 0.0
    backend_name: str = "explicit_fock_mixture"

    def __post_init__(self) -> None:
        r1 = np.asarray(self.R1, dtype=np.complex128)
        pairs = np.asarray(self.pair_probabilities, dtype=np.float64)
        if r1.ndim != 2 or r1.shape[0] != r1.shape[1] or pairs.shape != r1.shape:
            raise ValueError("R1 and explicit pair law must have matching [N,N] shape")
        if np.any(pairs < 0.0) or not np.allclose(pairs, pairs.T):
            raise ValueError("two-excitation pair law must be symmetric and non-negative")
        if not np.isclose(np.triu(pairs, 1).sum(), 1.0, atol=1e-9):
            raise ValueError("upper-triangle two-excitation pair law must sum to one")
        masses = self.p0 + float(np.trace(r1).real) + self.p2 + self.p3_plus
        if not np.isclose(masses, 1.0, atol=1e-9) or self.p3_plus != 0.0:
            raise ValueError("explicit V2.4 backend currently requires normalized p0/p1/p2 and p3+=0")
        object.__setattr__(self, "R1", r1)
        object.__setattr__(self, "pair_probabilities", pairs)
        object.__setattr__(self, "N", r1.shape[0])

    def probability(self, axes: Sequence[str], outcome: Sequence[int]) -> float:
        single_mass = self.p0 + float(np.trace(self.R1).real)
        single = 0.0
        if single_mass > 0.0:
            normalized = WSubspaceStateBackend(
                self.p0 / single_mass, self.R1 / single_mass, 0.0)
            single = single_mass * normalized.probability(axes, outcome)
        projectors = [outcome_projector(axis, int(value)) for axis, value in zip(axes, outcome)]
        double = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                local = 1.0
                for qubit, projector in enumerate(projectors):
                    bit = int(qubit in (i, j))
                    local *= float(projector[bit, bit].real)
                double += self.pair_probabilities[i, j] * local
        return float(np.clip(single + self.p2 * double, 0.0, 1.0))

    def state_sha(self) -> str:
        from ..hashing import sha256_json, sha256_array
        return sha256_json({"p0": self.p0, "p2": self.p2,
                            "R1": sha256_array(self.R1.view(np.float64)),
                            "pairs": sha256_array(self.pair_probabilities)})
