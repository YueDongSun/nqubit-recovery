"""V9-R2 full-Hilbert family renderer (charter section 7.1).

Generates the exact 3Q 14-class / 4Q 24-class state-family taxonomy from the
repository's authoritative definition (``quantum_representation.state_family``),
producing legal full-Hilbert (2^N) density matrices per family plus the exact
family / coarse-group / partition truth.

Closes V9-R2 failure #3 (the prototype sampled only 6 families) and #5 (the
prototype's "non-W" family was a random W-core N x N density, not a full-Hilbert
non-W state). This module is numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ml_recon.quantum_representation.state_family import (
    FINE_STATE_FAMILIES_BY_QUBITS,
    ideal_state_family_density,
    state_family_group,
)

SYSTEM_QUBITS = {"3qubit": 3, "4qubit": 4}


@dataclass(frozen=True)
class V9R2FamilyStateV1:
    """One family's full-Hilbert state plus its exact classification truth."""

    family: str
    coarse_group: str
    rho: np.ndarray  # full-Hilbert 2^N x 2^N, PSD, unit trace
    dim: int
    num_qubits: int
    purity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "coarse_group": self.coarse_group,
            "dim": int(self.dim),
            "num_qubits": int(self.num_qubits),
            "purity": float(self.purity),
        }


def family_labels(num_qubits: int) -> List[str]:
    """Exact fine-family labels for a system (3 -> 14, 4 -> 24)."""
    if num_qubits not in FINE_STATE_FAMILIES_BY_QUBITS:
        raise ValueError(f"no fine taxonomy for {num_qubits} qubits")
    return list(FINE_STATE_FAMILIES_BY_QUBITS[num_qubits])


def family_labels_for_system(system_type: str) -> List[str]:
    if system_type not in SYSTEM_QUBITS:
        raise ValueError(f"system_type must be one of {list(SYSTEM_QUBITS)}")
    return family_labels(SYSTEM_QUBITS[system_type])


def n_fine_families_for_system(system_type: str) -> int:
    """Number of exact fine-family logits for a system (3Q=14, 4Q=24)."""
    return len(family_labels_for_system(system_type))


def coarse_groups_for_system(system_type: str) -> List[str]:
    """Exact coarse entanglement-structure groups (3Q=7, 4Q=10)."""
    return sorted({state_family_group(f) for f in family_labels_for_system(system_type)})


def n_coarse_groups_for_system(system_type: str) -> int:
    """Number of coarse-group logits (3Q=7, 4Q=10)."""
    return len(coarse_groups_for_system(system_type))


def render_family_state(num_qubits: int, family: str, seed: int = 0) -> V9R2FamilyStateV1:
    """Render one family to a legal full-Hilbert state (dim = 2^N)."""
    labels = family_labels(num_qubits)
    if family not in labels:
        raise ValueError(f"family {family!r} not in {num_qubits}-qubit fine taxonomy")
    rng = np.random.default_rng(seed)
    rho = np.asarray(ideal_state_family_density(num_qubits, family, rng), dtype=np.complex128)
    expected_dim = 2 ** num_qubits
    if rho.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"{family} produced shape {rho.shape}, expected full-Hilbert ({expected_dim}, {expected_dim})"
        )
    eigenvalues = np.real(np.linalg.eigvalsh(rho))
    if np.min(eigenvalues) < -1e-8:
        raise ValueError(f"{family} is not PSD (min eigenvalue {np.min(eigenvalues)})")
    if not np.isclose(np.trace(rho).real, 1.0, atol=1e-6):
        raise ValueError(f"{family} is not unit trace")
    purity = float(np.real(np.trace(rho @ rho)))
    return V9R2FamilyStateV1(
        family=family,
        coarse_group=state_family_group(family),
        rho=rho,
        dim=expected_dim,
        num_qubits=num_qubits,
        purity=purity,
    )


def full_hilbert_populations(rho: np.ndarray) -> np.ndarray:
    """Full-Hilbert computational-basis populations (deterministic observable).

    This is the interim measurement observable: a full-Hilbert Born/diagonal pattern.
    Wiring the authoritative registered measurement operators (the 75/256-entry
    tomography basis) is a separate count-pipeline axis, not this renderer.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    return np.real(np.diag(rho)).astype(np.float64)
