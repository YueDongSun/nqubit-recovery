"""Multipartite certificate hypergraph, distinct from correlation graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .contracts import FourStatus


@dataclass(frozen=True)
class CertificateHyperedge:
    subset: Tuple[int, ...]
    certificate_type: str
    witness_value: float
    uncertainty: float
    p_value: float
    fdr_passed: bool
    measurement_coverage: float
    status: FourStatus


class EntanglementHypergraph:
    def __init__(self, N: int, edges: Iterable[CertificateHyperedge] = ()):
        self.N = int(N)
        self.edges = list(edges)
        for edge in self.edges:
            self._validate_edge(edge)

    def _validate_edge(self, edge: CertificateHyperedge) -> None:
        if len(edge.subset) < 2 or len(set(edge.subset)) != len(edge.subset):
            raise ValueError("hyperedge subset must contain at least two unique nodes")
        if min(edge.subset) < 0 or max(edge.subset) >= self.N:
            raise ValueError("hyperedge node outside graph")

    def add(self, edge: CertificateHyperedge) -> None:
        self._validate_edge(edge)
        self.edges.append(edge)

    def incidence_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.N, len(self.edges)), dtype=np.int8)
        for e, edge in enumerate(self.edges):
            matrix[list(edge.subset), e] = 1
        return matrix

    def to_dict(self) -> Dict[str, object]:
        return {"N": self.N, "edges": [asdict(e) for e in self.edges],
                "incidence_matrix": self.incidence_matrix().tolist()}
