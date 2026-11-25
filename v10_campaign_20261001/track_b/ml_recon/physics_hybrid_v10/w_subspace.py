"""Vacuum + single-excitation W-subspace representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class WSubspaceState:
    p0: float
    R1: np.ndarray
    p_ge2: float

    def validate(self, tolerance: float = 1e-8) -> None:
        r = np.asarray(self.R1, dtype=np.complex128)
        if r.ndim != 2 or r.shape[0] != r.shape[1]:
            raise ValueError("R1 must be square")
        if np.linalg.norm(r - r.conj().T) > tolerance:
            raise ValueError("R1 must be Hermitian")
        if np.linalg.eigvalsh(r).min(initial=0.0) < -tolerance:
            raise ValueError("R1 must be PSD")
        if min(self.p0, self.p_ge2) < -tolerance:
            raise ValueError("sector masses must be non-negative")
        if abs(self.p0 + np.trace(r).real + self.p_ge2 - 1.0) > tolerance:
            raise ValueError("vacuum/single/multi masses must sum to one")

    @property
    def N(self) -> int:
        return int(np.asarray(self.R1).shape[0])

    @property
    def p1(self) -> float:
        return float(np.trace(self.R1).real)


def pure_w_state(coefficients: np.ndarray, p0: float = 0.0,
                 p_ge2: float = 0.0) -> WSubspaceState:
    c = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    norm = float(np.vdot(c, c).real)
    p1 = 1.0 - p0 - p_ge2
    if norm <= 0.0 or p1 < 0.0:
        raise ValueError("invalid W coefficients or sector masses")
    c = c / np.sqrt(norm)
    state = WSubspaceState(float(p0), p1 * np.outer(c, c.conj()), float(p_ge2))
    state.validate()
    return state


def factorize_r1(R1: np.ndarray, rank: int | None = None) -> np.ndarray:
    r = np.asarray(R1, dtype=np.complex128)
    values, vectors = np.linalg.eigh(0.5 * (r + r.conj().T))
    order = np.argsort(values)[::-1]
    if rank is not None:
        order = order[:rank]
    values = np.maximum(values[order], 0.0)
    return vectors[:, order] * np.sqrt(values)
