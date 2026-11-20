"""Finite-budget W-resource preparation and single-output optical fusion.

This isolated research module uses row opportunities as its clock.  One row can
herald one W resource; its qubit count enters the resource budget separately.
The fusion instrument follows the single collected output in Shi et al.,
Phys. Rev. Lett. 135, 150802 (2025), with an explicit finite-efficiency threshold
detector and vacuum sector.  Parameter studies are conditional simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


SCHEMA = "w_state_fusion_mc_v1"
NORMAL_95 = 1.959963984540054
PHYSICAL_TOLERANCE = 1e-12


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return value


def _density(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=complex)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or len(value) < 2:
        raise ValueError(
            f"{name} must be a square density matrix with at least two modes"
        )
    if not np.all(np.isfinite(value)) or not np.allclose(
        value, value.conj().T, atol=PHYSICAL_TOLERANCE, rtol=0
    ):
        raise ValueError(f"{name} must be finite and Hermitian")
    if not np.isclose(np.trace(value), 1, atol=PHYSICAL_TOLERANCE, rtol=0):
        raise ValueError(f"{name} must have unit trace")
    if np.min(np.linalg.eigvalsh(value)) < -PHYSICAL_TOLERANCE:
        raise ValueError(f"{name} must be positive semidefinite")
    return value


@dataclass
class Fusion_result:
    """Unnormalized click-conditioned sectors, retaining their probabilities."""

    single_block: np.ndarray
    vacuum_probability: float

    @property
    def single_probability(self) -> float:
        return float(np.trace(self.single_block).real)

    @property
    def click_probability(self) -> float:
        return self.single_probability + self.vacuum_probability

    @property
    def single_weight(self) -> Optional[float]:
        return (
            self.single_probability / self.click_probability
            if self.click_probability > 0
            else None
        )

    @property
    def conditional_fidelity(self) -> Optional[float]:
        if self.single_probability == 0:
            return None
        return float(
            np.sum(self.single_block).real
            / len(self.single_block)
            / self.single_probability
        )

    @property
    def full_state_overlap(self) -> Optional[float]:
        if self.click_probability == 0:
            return None
        return float(
            np.sum(self.single_block).real
            / len(self.single_block)
            / self.click_probability
        )

    @property
    def heralded_density(self) -> Optional[np.ndarray]:
        if self.click_probability == 0:
            return None
        result = np.zeros(
            (len(self.single_block) + 1, len(self.single_block) + 1), dtype=complex
        )
        result[0, 0] = self.vacuum_probability
        result[1:, 1:] = self.single_block
        return result / self.click_probability


def fusion_output(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    efficiency: float,
    *,
    survival_a: float = 1.0,
    survival_b: float = 1.0,
    mode_overlap: complex = 1.0,
    interferometer_phase: float = 0.0,
) -> Fusion_result:
    """Fuse selected mode zero of each resource at one 50:50 output.

    Each input is a normalized single-excitation density matrix, with survival
    specifying its independent vacuum admixture.  ``mode_overlap`` is the
    complex overlap of pure photon wavepackets; HOM contrast is its modulus
    squared.  ``interferometer_phase`` is the calibrated relative optical phase.
    Dark counts and input sectors with more than one excitation are outside this
    declared instrument.  The remaining register has n + m - 2 modes.
    """
    rho_a, rho_b = _density(rho_a, "rho_a"), _density(rho_b, "rho_b")
    eta = _probability(efficiency, "efficiency")
    s_a, s_b = (
        _probability(survival_a, "survival_a"),
        _probability(survival_b, "survival_b"),
    )
    overlap = complex(mode_overlap)
    if (
        not np.isfinite(overlap)
        or abs(overlap) > 1
        or not math.isfinite(interferometer_phase)
    ):
        raise ValueError(
            "mode overlap must have modulus at most one and phase must be finite"
        )
    p_a, p_b = float(rho_a[0, 0].real), float(rho_b[0, 0].real)
    cross = (
        overlap
        * np.exp(1j * interferometer_phase)
        * np.outer(rho_a[1:, 0], rho_b[0, 1:])
    )
    single = (
        0.5
        * eta
        * s_a
        * s_b
        * np.block(
            [
                [p_b * rho_a[1:, 1:], cross],
                [cross.conj().T, p_a * rho_b[1:, 1:]],
            ]
        )
    )
    two_photon_click = eta - (1 + abs(overlap) ** 2) * eta**2 / 4
    vacuum = s_a * s_b * p_a * p_b * two_photon_click
    vacuum += 0.5 * eta * (s_a * (1 - s_b) * p_a + (1 - s_a) * s_b * p_b)
    return Fusion_result(single, float(vacuum))
