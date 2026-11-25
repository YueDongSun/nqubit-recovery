"""Read-only adapters to the existing V8 DLCZ probability kernels.

The V8 batch generator is intentionally not reused: it samples expected
pattern counts directly.  Only its pure geometric, occupancy and detector
probability helpers are called by the event-level simulator, and the helper
source file is included in provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ml_recon.physics_hybrid_v8.dlcz_noise_synthetic import (
    detected_pattern_probabilities_for_n,
    geometric_excitation_probabilities,
    single_excitation_occupancy,
    two_excitation_bosonic_occupancy,
)


KERNEL_ADAPTER_SCHEMA = "V10DlczKernelAdapterV1"


def sample_geometric_count(rng: np.random.Generator, chi: float) -> int:
    """Sample the thermal DLCZ write count ``P(K=k)=(1-chi)chi**k``."""
    value = float(chi)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("chi must be finite and lie in [0, 1)")
    return int(rng.geometric(1.0 - value) - 1)


def sample_single_cell(
    rng: np.random.Generator, mode_probability: Sequence[float]
) -> int:
    probability = np.asarray(mode_probability, dtype=np.float64).reshape(-1)
    if probability.size == 4:
        probability = np.asarray(
            single_excitation_occupancy(probability), dtype=np.float64
        )
    if (
        probability.size < 1
        or not np.isfinite(probability).all()
        or np.any(probability < 0)
        or not np.isclose(probability.sum(), 1.0)
    ):
        raise ValueError("mode_probability must be finite and normalized")
    return int(rng.choice(probability.size, p=probability))


def sample_two_boson_occupancy(
    rng: np.random.Generator, mode_probability: Sequence[float]
) -> tuple[int, ...]:
    """Sample a local or cross-cell two-boson occupancy vector."""
    probability = np.asarray(mode_probability, dtype=np.float64)
    if probability.size == 4:
        components = two_excitation_bosonic_occupancy(probability)
        same_weights = np.asarray(components["same_mode"], dtype=np.float64)
        distinct_weights = np.asarray(components["distinct_mode"], dtype=np.float64)
    else:
        same_weights = np.square(probability)
        distinct_weights = np.asarray(
            [
                2.0 * probability[left] * probability[right]
                for left in range(probability.size)
                for right in range(left + 1, probability.size)
            ],
            dtype=np.float64,
        )
    choices = [(index, index) for index in range(probability.size)] + [
        (left, right)
        for left in range(probability.size)
        for right in range(left + 1, probability.size)
    ]
    weights = np.concatenate((same_weights, distinct_weights))
    selected = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
    occupancy = np.zeros(probability.size, dtype=np.int64)
    occupancy[selected[0]] += 1
    occupancy[selected[1]] += 1
    return tuple(int(value) for value in occupancy)


def kernel_provenance() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[2]
        / "physics_hybrid_v8"
        / "dlcz_noise_synthetic.py"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": KERNEL_ADAPTER_SCHEMA,
        "module": "ml_recon.physics_hybrid_v8.dlcz_noise_synthetic",
        "source_path": str(path),
        "source_sha256": digest,
        "batch_generator_reused": False,
        "pure_kernels": [
            "geometric_excitation_probabilities",
            "single_excitation_occupancy",
            "two_excitation_bosonic_occupancy",
            "detected_pattern_probabilities_for_n",
        ],
    }


__all__ = [
    "detected_pattern_probabilities_for_n",
    "geometric_excitation_probabilities",
    "kernel_provenance",
    "sample_geometric_count",
    "sample_single_cell",
    "sample_two_boson_occupancy",
    "single_excitation_occupancy",
    "two_excitation_bosonic_occupancy",
]
