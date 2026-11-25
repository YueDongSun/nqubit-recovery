"""Certificate-safe decoding from pair scores to legal partition posteriors."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .certificate_engine import CertificateResult
from .contracts import CertificateStatus, sha256_of
from .partition_poset import Partition, canonical_partition
from .partition_structure import PartitionCandidate, PartitionStructureV1


@dataclass(frozen=True)
class PartitionDecoderConfig:
    candidate_count: int = 8
    posterior_temperature: float = 1.0
    edge_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75)
    engineering_default: bool = True
    calibration_method: str = "uncalibrated_softmax_engineering_default"

    def validate(self) -> None:
        if self.candidate_count <= 0 or self.candidate_count > 1024:
            raise ValueError("candidate_count must be in [1,1024]")
        if self.posterior_temperature <= 0.0:
            raise ValueError("posterior temperature must be positive")
        if not self.edge_thresholds or any(
                value <= 0.0 or value >= 1.0 for value in self.edge_thresholds):
            raise ValueError("edge thresholds must lie in (0,1)")

    def sha256(self) -> str:
        self.validate()
        return sha256_of({
            "candidate_count": self.candidate_count,
            "posterior_temperature": self.posterior_temperature,
            "edge_thresholds": self.edge_thresholds,
            "engineering_default": self.engineering_default,
            "calibration_method": self.calibration_method,
        })


def _union_find_partition(N: int, edges: Iterable[tuple[int, int]]) -> Partition:
    parent = list(range(N))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(int(left), int(right))
    blocks: dict[int, list[int]] = {}
    for mode in range(N):
        blocks.setdefault(find(mode), []).append(mode)
    return canonical_partition(blocks.values())


def _pair_score_matrix(
    partition_link_scores: np.ndarray,
    *,
    input_kind: str | None = None,
) -> np.ndarray:
    scores = np.asarray(partition_link_scores, dtype=np.float64)
    if input_kind == "logits":
        logits = scores
    elif input_kind == "probabilities":
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("partition link probabilities must lie in [0,1]")
        clipped = np.clip(scores, 1e-12, 1.0 - 1e-12)
        logits = np.log(clipped) - np.log1p(-clipped)
    else:
        raise ValueError("partition decoder input_kind must be logits or probabilities")
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("partition link logits must be square [N,N]")
    if logits.shape[0] < 3 or logits.shape[0] > 256:
        raise ValueError("partition decoder requires 3 <= N <= 256")
    if not np.all(np.isfinite(logits)):
        raise ValueError("partition link logits must be finite")
    return 0.5 * (logits + logits.T)


def _score_partition(logits: np.ndarray, partition: Partition) -> float:
    N = logits.shape[0]
    labels = np.full(N, -1, dtype=np.int64)
    for block_id, block in enumerate(partition):
        labels[list(block)] = block_id
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    score = 0.0
    defined_pairs = 0
    for i, j in itertools.combinations(range(N), 2):
        if labels[i] < 0 or labels[j] < 0:
            # Missing candidate coverage is UNKNOWN, never evidence for a
            # same-block relation and never evidence for a split relation.
            continue
        p = float(np.clip(probabilities[i, j], 1e-12, 1.0 - 1e-12))
        score += float(np.log(p if labels[i] == labels[j] else 1.0 - p))
        defined_pairs += 1
    return score / max(defined_pairs, 1)


def _certificate_constraints(
    certificates: Iterable[CertificateResult] | None,
) -> tuple[list[tuple[int, ...]], list[tuple[tuple[int, ...], tuple[int, ...]]], list[tuple[int, ...]], float]:
    entangled: list[tuple[int, ...]] = []
    cuts: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    gme: list[tuple[int, ...]] = []
    coverage: list[float] = []
    for certificate in certificates or ():
        status = certificate.status
        if status not in {
                CertificateStatus.CERTIFIED_ENTANGLED,
                CertificateStatus.CERTIFIED_GME,
                CertificateStatus.NPT_CERTIFIED,
        }:
            # Inconclusive/unavailable/PPT-compatible results never exclude a
            # candidate.  This is the certificate-safe fail-closed rule.
            continue
        subset = tuple(sorted(set(int(value) for value in certificate.subset)))
        if status in {CertificateStatus.CERTIFIED_ENTANGLED,
                      CertificateStatus.NPT_CERTIFIED} and len(subset) >= 2:
            entangled.append(subset)
        if status == CertificateStatus.CERTIFIED_GME and len(subset) >= 2:
            gme.append(subset)
        tested = tuple(tuple(int(value) for value in side)
                       for side in certificate.tested_partition)
        if status == CertificateStatus.NPT_CERTIFIED and len(tested) == 2:
            cuts.append((tuple(sorted(tested[0])), tuple(sorted(tested[1]))))
        coverage.append(float(np.clip(certificate.coverage, 0.0, 1.0)))
    return entangled, cuts, gme, float(np.mean(coverage)) if coverage else 0.0


def _is_compatible(
    partition: Partition,
    entangled: Sequence[Sequence[int]],
    cuts: Sequence[tuple[Sequence[int], Sequence[int]]],
    gme: Sequence[Sequence[int]],
) -> bool:
    blocks = [set(block) for block in partition]
    if not all(any(len(block & set(subset)) >= 2 for block in blocks)
               for subset in entangled):
        return False
    if not all(any(block & set(left) and block & set(right) for block in blocks)
               for left, right in cuts):
        return False
    if not all(any(set(subset).issubset(block) for block in blocks) for subset in gme):
        return False
    return True


def _generated_candidates(logits: np.ndarray, config: PartitionDecoderConfig) -> tuple[Partition, ...]:
    N = logits.shape[0]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    candidates: set[Partition] = {
        canonical_partition(((mode,) for mode in range(N))),
        canonical_partition((tuple(range(N)),)),
    }
    thresholds = tuple(config.edge_thresholds)
    for threshold in thresholds:
        edges = ((i, j) for i, j in itertools.combinations(range(N), 2)
                 if probabilities[i, j] >= threshold)
        candidates.add(_union_find_partition(N, edges))

    # Add a deterministic agglomerative path.  This explores the candidate
    # space without enumerating the Bell lattice.
    blocks = [tuple((mode,)) for mode in range(N)]
    while len(blocks) > 1:
        best: tuple[float, int, int] | None = None
        for left, right in itertools.combinations(range(len(blocks)), 2):
            values = [probabilities[i, j]
                      for i in blocks[left] for j in blocks[right]]
            score = float(np.mean(values)) if values else 0.0
            proposal = (score, -left, -right)
            if best is None or proposal > best:
                best = proposal
                best_indices = (left, right)
        left, right = best_indices
        merged = tuple(sorted(blocks[left] + blocks[right]))
        blocks = [block for index, block in enumerate(blocks)
                  if index not in (left, right)] + [merged]
        candidates.add(canonical_partition(blocks))
    return tuple(candidates)


def decode_partition_candidates(
    partition_link_logits: np.ndarray,
    *,
    config: PartitionDecoderConfig | None = None,
    candidate_partitions: Iterable[Partition] | None = None,
    certificates: Iterable[CertificateResult] | None = None,
    certificate_coverage: float | None = None,
    abstention_probability: float = 0.0,
    pair_evidence_ref: str = "",
    hyperedge_evidence_ref: str = "",
    source_checkpoint_sha: str = "",
    measurement_contract_sha: str = "",
    candidate_complete: bool | None = None,
    candidate_set_complete: bool | None = None,
    input_kind: str | None = None,
) -> PartitionStructureV1:
    """Decode a normalized posterior over legal candidates.

    ``input_kind`` is explicit so probabilities cannot accidentally be passed
    through a second sigmoid.  ``candidate_complete`` remains a compatibility
    alias for the v1.1 ``candidate_set_complete`` name.
    """
    active = config or PartitionDecoderConfig()
    active.validate()
    logits = _pair_score_matrix(partition_link_logits, input_kind=input_kind)
    N = logits.shape[0]
    entangled, cuts, gme, inferred_coverage = _certificate_constraints(certificates)
    candidates = tuple(
        canonical_partition(partition) for partition in
        (candidate_partitions if candidate_partitions is not None
         else _generated_candidates(logits, active)))
    unique = tuple(sorted(set(candidates), key=lambda value: (len(value), value)))
    compatible = tuple(partition for partition in unique
                       if _is_compatible(partition, entangled, cuts, gme))
    if not compatible:
        raise ValueError("certificate constraints exclude every partition candidate")
    scored = [PartitionCandidate(partition, _score_partition(logits, partition))
              for partition in compatible]
    scored.sort(key=lambda candidate: (-candidate.score, candidate.partition))
    top = scored[:active.candidate_count]
    values = np.asarray([candidate.score for candidate in top], dtype=np.float64)
    values = (values - values.max()) / active.posterior_temperature
    weights = np.exp(values - values.max())
    weights /= weights.sum()
    block_ids = np.stack([candidate.block_ids(N) for candidate in top]).astype(np.int16)
    coverage = inferred_coverage if certificate_coverage is None else float(certificate_coverage)
    if candidate_complete is not None and candidate_set_complete is not None \
            and bool(candidate_complete) != bool(candidate_set_complete):
        raise ValueError("candidate_complete and candidate_set_complete disagree")
    complete = bool(candidate_set_complete if candidate_set_complete is not None
                    else (candidate_complete if candidate_complete is not None else False))
    result = PartitionStructureV1(
        N=N,
        candidate_block_ids=block_ids,
        candidate_weights=weights,
        candidate_set_complete=complete,
        certificate_coverage=coverage,
        abstention_probability=float(abstention_probability),
        pair_evidence_ref=pair_evidence_ref,
        hyperedge_evidence_ref=hyperedge_evidence_ref,
        source_checkpoint_sha=source_checkpoint_sha,
        measurement_contract_sha=measurement_contract_sha,
        decoder_contract_sha=active.sha256(),
    )
    result.validate()
    return result


def decode_partition_posterior(*args, **kwargs) -> PartitionStructureV1:
    return decode_partition_candidates(*args, **kwargs)


__all__ = [
    "PartitionDecoderConfig", "decode_partition_candidates",
    "decode_partition_posterior",
]
