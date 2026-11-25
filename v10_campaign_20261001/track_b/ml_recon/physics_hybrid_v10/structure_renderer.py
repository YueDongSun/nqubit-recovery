"""General-N state-connected structure renderer with exact partition truth."""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from .partition_poset import canonical_partition
from .state_backends import (
    DenseDensityStateBackend,
    MPSStateBackend,
    ProductBlockStateBackend,
    V10StateBackend,
)
from .synthetic_manifest import V10CounterBasedRng, V10SyntheticSampleKey


BLOCK_TEMPLATES = (
    "product",
    "Bell-like",
    "W",
    "GHZ",
    "Dicke",
    "graph_cluster",
    "random_low_bond_MPS",
    "mixed",
    "unknown_template",
)


def _basis_index(bits: Tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def render_block_state(template: str, size: int,
                       rng: np.random.Generator) -> V10StateBackend:
    """Render an actual block state instead of a label-only template."""
    if template not in BLOCK_TEMPLATES or size < 1 or size > 256:
        raise ValueError("unsupported block template or size")
    if size > 8 and template not in ("product", "W"):
        raise ValueError("large blocks require a compact product or W representation")
    if size > 8 and template == "product":
        blocks = []
        for qubit in range(size):
            theta = rng.uniform(0.0, np.pi)
            phase = rng.uniform(-np.pi, np.pi)
            local = np.asarray((np.cos(theta / 2.0),
                                np.exp(1j * phase) * np.sin(theta / 2.0)))
            blocks.append(((qubit,), DenseDensityStateBackend.from_state_vector(local)))
        return ProductBlockStateBackend(size, tuple(blocks))
    if size > 8 and template == "W":
        coefficients = rng.normal(size=size) + 1j * rng.normal(size=size)
        coefficients /= np.linalg.norm(coefficients)
        from ml_recon.data_v2_4.synthetic.state_renderers import w_mps_from_coefficients
        return w_mps_from_coefficients(coefficients)
    dimension = 2 ** size
    vector = np.zeros(dimension, dtype=np.complex128)

    if template == "product":
        vector = np.asarray([1.0 + 0.0j])
        for _ in range(size):
            theta = rng.uniform(0.0, np.pi)
            phase = rng.uniform(-np.pi, np.pi)
            local = np.asarray((np.cos(theta / 2.0),
                                np.exp(1j * phase) * np.sin(theta / 2.0)))
            vector = np.kron(vector, local)
    elif template == "Bell-like":
        theta = rng.uniform(0.2, np.pi / 2.0 - 0.2)
        phase = rng.uniform(-np.pi, np.pi)
        vector[0] = np.cos(theta)
        vector[-1] = np.exp(1j * phase) * np.sin(theta)
    elif template == "W":
        coefficients = rng.normal(size=size) + 1j * rng.normal(size=size)
        coefficients /= np.linalg.norm(coefficients)
        for qubit, coefficient in enumerate(coefficients):
            vector[1 << (size - 1 - qubit)] = coefficient
    elif template == "GHZ":
        phase = rng.uniform(-np.pi, np.pi)
        vector[0] = 1.0 / np.sqrt(2.0)
        vector[-1] = np.exp(1j * phase) / np.sqrt(2.0)
    elif template == "Dicke":
        excitation_count = min(max(1, size // 2), size - 1) if size > 1 else 0
        states = tuple(itertools.combinations(range(size), excitation_count))
        for occupied in states:
            bits = tuple(int(qubit in occupied) for qubit in range(size))
            vector[_basis_index(bits)] = 1.0 / np.sqrt(len(states))
    elif template == "graph_cluster":
        vector[:] = 1.0 / np.sqrt(dimension)
        for index in range(dimension):
            bits = tuple((index >> (size - 1 - q)) & 1 for q in range(size))
            phase = sum(bits[q] * bits[(q + 1) % size] for q in range(size - 1))
            vector[index] *= (-1.0) ** phase
    elif template == "random_low_bond_MPS":
        vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        vector /= np.linalg.norm(vector)
        if size >= 3:
            return MPSStateBackend.from_state_vector(vector, size, max_bond=min(4, dimension))
    elif template == "mixed":
        rank = min(4, dimension)
        factor = rng.normal(size=(dimension, rank)) + 1j * rng.normal(size=(dimension, rank))
        rho = factor @ factor.conj().T
        rho /= np.trace(rho).real
        return DenseDensityStateBackend(rho)
    else:  # unknown_template: physically valid, taxonomy intentionally withheld.
        vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        vector /= np.linalg.norm(vector)

    return DenseDensityStateBackend.from_state_vector(vector)


class V10StructureRenderer:
    def render(self, key: V10SyntheticSampleKey,
               rng: V10CounterBasedRng) -> Dict[str, object]:
        stream = rng.stream("partition")
        state_stream = rng.stream("state")
        permutation = list(map(int, stream.permutation(key.N)))
        blocks: List[Tuple[int, ...]] = []
        cursor = 0
        while cursor < key.N:
            remaining = key.N - cursor
            maximum_block = min(remaining, max(8, min(key.N, 64)))
            block_size = int(stream.integers(1, maximum_block + 1))
            blocks.append(tuple(permutation[cursor:cursor + block_size]))
            cursor += block_size
        partition = canonical_partition(blocks)

        templates = []
        state_blocks = []
        for block in partition:
            compatible = BLOCK_TEMPLATES if len(block) <= 8 else ("product", "W")
            if len(block) == 1:
                compatible = ("product", "mixed", "unknown_template")
            template = compatible[int(stream.integers(len(compatible)))]
            templates.append(template)
            state_blocks.append((block, render_block_state(template, len(block), state_stream)))
        backend = ProductBlockStateBackend(key.N, tuple(state_blocks))
        effective_blocks: List[Tuple[int, ...]] = []
        exact = True
        for block, template in zip(partition, templates):
            if template == "product":
                effective_blocks.extend((qubit,) for qubit in block)
            elif template in ("mixed", "unknown_template"):
                effective_blocks.append(block)
                exact = False
            else:
                effective_blocks.append(block)
        observed_partition = canonical_partition(effective_blocks)
        sizes = [len(block) for block in observed_partition]
        return {
            "sample_id": key.sample_id,
            "N": key.N,
            "state_backend": backend,
            "state_sha": backend.state_sha(),
            "generating_partition": partition,
            "observed_partition": observed_partition,
            "partition_exact": exact,
            "partition_complete": exact,
            "unassigned_qubits": (),
            "block_templates": tuple(templates),
            "entanglement_depth": max(sizes),
            "entanglement_intactness": len(sizes),
            "stretchability": max(sizes) - len(sizes),
            "full_family_claim": False,
        }
