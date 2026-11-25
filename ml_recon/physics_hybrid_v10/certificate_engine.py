"""Deterministic V10 certificate engine; neural predictions cannot override it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import CertificateStatus
from .hypergraph import CertificateHyperedge, EntanglementHypergraph
from .ppt_mixture import negativity


@dataclass(frozen=True)
class CertificateResult:
    certificate_type: str
    subset: Tuple[int, ...]
    tested_partition: Tuple[Tuple[int, ...], ...]
    witness_value: float
    uncertainty: float
    significance: float
    coverage: float
    status: CertificateStatus
    p_value: float = 1.0
    q_value: float = 1.0


@dataclass(frozen=True)
class BenjaminiHochbergResult:
    q_values: np.ndarray
    rejected: np.ndarray


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05
                       ) -> BenjaminiHochbergResult:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must be a one-dimensional vector in [0,1]")
    order = np.argsort(p)
    ranked = p[order]
    m = max(len(p), 1)
    adjusted_ranked = ranked * m / np.arange(1, len(p) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    q_values = np.empty_like(adjusted_ranked)
    q_values[order] = adjusted_ranked
    return BenjaminiHochbergResult(q_values=q_values, rejected=q_values <= alpha)


class V10CertificateEngine:
    def __init__(self, N: int, alpha: float = 0.05):
        self.N = N
        self.alpha = alpha

    def pair_certificates(self, pair_rdms: Dict[Tuple[int, int], np.ndarray],
                          uncertainties: Optional[Dict[Tuple[int, int], float]] = None
                          ) -> List[CertificateResult]:
        raw = []
        for pair, rho in sorted(pair_rdms.items()):
            neg = negativity(rho)
            unc = float((uncertainties or {}).get(pair, 0.0))
            z = neg / max(unc, 1e-12) if unc > 0 else (np.inf if neg > 1e-9 else 0.0)
            p_value = float(np.exp(-0.5 * z * z)) if np.isfinite(z) else 0.0
            raw.append((pair, neg, unc, p_value))
        fdr = benjamini_hochberg([x[3] for x in raw], self.alpha)
        results = []
        for rejected, q_value, (pair, neg, unc, p_value) in zip(
                fdr.rejected, fdr.q_values, raw):
            status = (CertificateStatus.NPT_CERTIFIED
                      if neg > 1e-9 and rejected else CertificateStatus.INCONCLUSIVE)
            results.append(CertificateResult(
                "pair_PPT_negativity", tuple(pair), (tuple(pair),), neg, unc,
                1.0 - p_value, 1.0, status, p_value, float(q_value)))
        return results

    def ghz_coherence_certificate(self, coherence: Optional[complex], population_sum: float,
                                  uncertainty: float = 0.0) -> CertificateResult:
        if coherence is None:
            status = CertificateStatus.UNAVAILABLE
            witness = 0.0
            coverage = 0.0
        else:
            witness = float(abs(coherence) - 0.5 * max(1.0 - population_sum, 0.0))
            status = (CertificateStatus.CERTIFIED_GME
                      if witness > max(uncertainty, 0.0) else CertificateStatus.INCONCLUSIVE)
            coverage = 1.0
        return CertificateResult("global_GHZ_witness", tuple(range(self.N)),
                                 (tuple(range(self.N)),), witness, uncertainty,
                                 float(witness > uncertainty), coverage, status,
                                 0.0 if witness > uncertainty else 1.0,
                                 0.0 if witness > uncertainty else 1.0)

    def enforce_neural_consistency(self, certificates: Iterable[CertificateResult],
                                   neural_separable_probability: Dict[Tuple[int, ...], float],
                                   high_confidence: float = 0.9) -> None:
        for certificate in certificates:
            if (certificate.status in {
                    CertificateStatus.CERTIFIED_ENTANGLED,
                    CertificateStatus.CERTIFIED_GME,
                    CertificateStatus.NPT_CERTIFIED,
                } and
                    neural_separable_probability.get(certificate.subset, 0.0) >= high_confidence):
                raise ValueError("neural posterior contradicts deterministic entanglement certificate")

    def hypergraph(self, certificates: Iterable[CertificateResult]) -> EntanglementHypergraph:
        edges = []
        for c in certificates:
            if len(c.subset) >= 2:
                edges.append(CertificateHyperedge(
                    c.subset, c.certificate_type, c.witness_value, c.uncertainty,
                    c.p_value, c.q_value <= self.alpha,
                    c.coverage, c.status))
        return EntanglementHypergraph(self.N, edges)
