"""Legal partition posterior representation for V10 ESM routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Tuple

import numpy as np

from .contracts import sha256_of
from .partition_poset import Partition, canonical_partition


def _canonical_block_ids(blocks: Partition, N: int) -> np.ndarray:
    result = np.full(N, -1, dtype=np.int16)
    for block_id, block in enumerate(canonical_partition(blocks)):
        for mode in block:
            if mode < 0 or mode >= N:
                raise ValueError("partition mode is outside [0,N)")
            if result[mode] != -1:
                raise ValueError("partition mode is assigned more than once")
            result[mode] = block_id
    return result


def _blocks_from_ids(block_ids: Sequence[int], N: int, *, allow_incomplete: bool = True) -> Partition:
    ids = np.asarray(block_ids, dtype=np.int64).reshape(-1)
    if ids.shape != (N,):
        raise ValueError("partition block ids must have shape [N]")
    if np.any(ids < -1):
        raise ValueError("partition block ids may use only -1 for unavailable modes")
    if not allow_incomplete and np.any(ids < 0):
        raise ValueError("complete partition candidates cannot contain unavailable modes")
    blocks = []
    for block_id in sorted(set(int(value) for value in ids if value >= 0)):
        members = tuple(int(index) for index, value in enumerate(ids) if value == block_id)
        if not members:
            raise ValueError("partition block id has no members")
        blocks.append(members)
    return canonical_partition(blocks) if blocks else ()


@dataclass(frozen=True)
class PartitionCandidate:
    """One canonical candidate before posterior normalization."""

    partition: Partition
    score: float
    provenance: str = "decoder"

    def block_ids(self, N: int) -> np.ndarray:
        return _canonical_block_ids(self.partition, N)


@dataclass
class PartitionStructureV1:
    """Top-K posterior over compatible entangled-block partitions.

    The co-membership matrix is a runtime/evaluation view.  It is never part of
    the low-disk persistent representation returned by :meth:`to_dict`.
    """

    N: int
    candidate_block_ids: np.ndarray
    candidate_weights: np.ndarray
    candidate_set_complete: bool | None = None
    candidate_complete: bool | None = None
    candidate_mode_coverage: np.ndarray | None = None
    partition_entropy: float | None = None
    certificate_coverage: float = 0.0
    abstention_probability: float = 0.0
    pair_evidence_ref: str = ""
    hyperedge_evidence_ref: str = ""
    same_block_probability_materialized: bool = False
    source_checkpoint_sha: str = ""
    measurement_contract_sha: str = ""
    decoder_contract_sha: str = ""
    _runtime_co_membership: np.ndarray | None = None
    _runtime_co_membership_defined: np.ndarray | None = None

    def __post_init__(self) -> None:
        N = int(self.N)
        if N < 3 or N > 256:
            raise ValueError("partition structure requires 3 <= N <= 256")
        ids = np.asarray(self.candidate_block_ids, dtype=np.int16)
        weights = np.asarray(self.candidate_weights, dtype=np.float64).reshape(-1)
        if ids.ndim != 2 or ids.shape[1] != N or ids.shape[0] != weights.size or not ids.shape[0]:
            raise ValueError("candidate ids and weights must align as [K,N] and [K]")
        if np.any(ids < -1) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("partition candidates must be finite and use -1 only for unavailable modes")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("partition posterior weights must have positive mass")
        weights = weights / total
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "candidate_block_ids", ids)
        object.__setattr__(self, "candidate_weights", weights)
        if self.candidate_set_complete is not None and self.candidate_complete is not None \
                and bool(self.candidate_set_complete) != bool(self.candidate_complete):
            raise ValueError("candidate_set_complete and candidate_complete disagree")
        complete = bool(self.candidate_set_complete if self.candidate_set_complete is not None
                        else (self.candidate_complete if self.candidate_complete is not None else False))
        object.__setattr__(self, "candidate_set_complete", complete)
        object.__setattr__(self, "candidate_complete", complete)
        coverage = (ids >= 0 if self.candidate_mode_coverage is None else
                    np.asarray(self.candidate_mode_coverage, dtype=bool))
        object.__setattr__(self, "candidate_mode_coverage", coverage)
        self.validate()
        if self.partition_entropy is None:
            entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-15))))
            object.__setattr__(self, "partition_entropy", entropy)

    def validate(self) -> None:
        ids = np.asarray(self.candidate_block_ids)
        weights = np.asarray(self.candidate_weights)
        coverage = np.asarray(self.candidate_mode_coverage, dtype=bool)
        if ids.dtype.kind not in "iu" or ids.shape[1:] != (self.N,):
            raise ValueError("candidate_block_ids must be integer [K,N]")
        if coverage.shape != ids.shape:
            raise ValueError("candidate_mode_coverage must have shape [K,N]")
        if np.any(ids < -1) or np.any(ids >= self.N):
            raise ValueError("candidate block ids are out of range")
        if np.any(coverage != (ids >= 0)):
            raise ValueError("candidate mode coverage must agree with assigned block ids")
        if not np.isclose(float(weights.sum()), 1.0, atol=1e-8):
            raise ValueError("candidate weights must sum to one")
        if not 0.0 <= float(self.certificate_coverage) <= 1.0:
            raise ValueError("certificate coverage must be in [0,1]")
        if not 0.0 <= float(self.abstention_probability) <= 1.0:
            raise ValueError("abstention probability must be in [0,1]")
        if self.candidate_set_complete and np.any(ids < 0):
            raise ValueError("complete candidate sets cannot contain unavailable modes")
        for row in ids:
            _blocks_from_ids(row, self.N, allow_incomplete=not self.candidate_set_complete)
        for name in ("source_checkpoint_sha", "measurement_contract_sha", "decoder_contract_sha"):
            value = getattr(self, name)
            if value and len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest when supplied")

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_weights.size)

    def partitions(self) -> Tuple[Partition, ...]:
        return tuple(_blocks_from_ids(row, self.N,
                                      allow_incomplete=not self.candidate_set_complete)
                      for row in self.candidate_block_ids)

    def co_membership(self, *, materialize: bool = False) -> np.ndarray:
        """Compute runtime ``P_same``; consult :meth:`co_membership_defined` too."""
        same, _ = self.co_membership_views(materialize=materialize)
        return same

    def co_membership_defined(self, *, materialize: bool = False) -> np.ndarray:
        """Return the pairwise posterior mass for which co-membership is defined.

        An incomplete candidate is not evidence that an uncovered pair is in a
        different block.  The separate matrix therefore carries the missingness
        semantics required by the v1.1 router.
        """
        _, defined = self.co_membership_views(materialize=materialize)
        return defined

    def co_membership_views(
        self, *, materialize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(P_same, P_defined)`` with pairwise missingness preserved."""
        if self._runtime_co_membership is not None:
            return (self._runtime_co_membership.copy(),
                    self._runtime_co_membership_defined.copy())
        numerator = np.zeros((self.N, self.N), dtype=np.float64)
        defined = np.zeros((self.N, self.N), dtype=np.float64)
        for row, weight, coverage in zip(
                self.candidate_block_ids, self.candidate_weights,
                self.candidate_mode_coverage):
            assigned = np.asarray(row, dtype=np.int64)
            mode_mask = np.asarray(coverage, dtype=bool) & (assigned >= 0)
            defined += float(weight) * np.outer(mode_mask, mode_mask)
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    if mode_mask[i] and mode_mask[j] and assigned[i] == assigned[j]:
                        numerator[i, j] += float(weight)
                        numerator[j, i] += float(weight)
        result = np.divide(numerator, np.maximum(defined, 1e-12),
                           out=np.zeros_like(numerator), where=defined > 1e-12)
        result = 0.5 * (result + result.T)
        np.fill_diagonal(result, 1.0)
        defined = np.clip(0.5 * (defined + defined.T), 0.0, 1.0)
        if materialize:
            object.__setattr__(self, "_runtime_co_membership", result.copy())
            object.__setattr__(self, "_runtime_co_membership_defined", defined.copy())
            object.__setattr__(self, "same_block_probability_materialized", True)
        return result, defined

    def co_membership_psd_min_eigenvalue(self) -> float:
        return float(np.linalg.eigvalsh(self.co_membership()).min(initial=0.0))

    def to_dict(self, *, persistent: bool = True) -> dict[str, Any]:
        self.validate()
        value: dict[str, Any] = {
            "schema": "V10PartitionStructureV1",
            "N": self.N,
            "candidate_topk_block_ids": self.candidate_block_ids.astype(np.int16).tolist(),
            "candidate_weights": self.candidate_weights.astype(np.float32).tolist(),
            "candidate_set_complete": bool(self.candidate_set_complete),
            # Kept as a read-only compatibility alias; consumers must use the
            # separate mode coverage field for incomplete candidates.
            "candidate_complete": bool(self.candidate_set_complete),
            "candidate_mode_coverage": self.candidate_mode_coverage.astype(bool).tolist(),
            "partition_entropy": float(self.partition_entropy or 0.0),
            "certificate_coverage": float(self.certificate_coverage),
            "abstention_probability": float(self.abstention_probability),
            "pair_evidence_ref": self.pair_evidence_ref,
            "hyperedge_evidence_ref": self.hyperedge_evidence_ref,
            "same_block_probability_materialized": False,
            "source_checkpoint_sha": self.source_checkpoint_sha,
            "measurement_contract_sha": self.measurement_contract_sha,
            "decoder_contract_sha": self.decoder_contract_sha,
        }
        if not persistent:
            value["same_block_probability"] = self.co_membership().astype(np.float32).tolist()
            value["same_block_probability_defined"] = self.co_membership_defined().astype(np.float32).tolist()
            value["same_block_probability_materialized"] = True
        return value

    def sha256(self) -> str:
        return sha256_of(self.to_dict())

    @classmethod
    def from_partitions(
        cls,
        N: int,
        partitions: Iterable[Partition],
        weights: Sequence[float],
        *,
        candidate_complete: bool = False,
        candidate_set_complete: bool | None = None,
        **metadata: Any,
    ) -> "PartitionStructureV1":
        canonical = tuple(canonical_partition(partition) for partition in partitions)
        weight_values = tuple(weights)
        if len(canonical) != len(weight_values):
            raise ValueError("partitions and weights must have equal length")
        ids = np.stack([_canonical_block_ids(partition, N) for partition in canonical])
        return cls(N, ids, np.asarray(weight_values, dtype=np.float64),
                   candidate_set_complete=(candidate_complete if candidate_set_complete is None
                                           else candidate_set_complete), **metadata)


def validate_partition_structure(structure: PartitionStructureV1) -> dict[str, object]:
    structure.validate()
    co_membership, defined = structure.co_membership_views()
    complete_support = bool(np.all(defined >= 1.0 - 1e-10))
    psd_min = structure.co_membership_psd_min_eigenvalue() if complete_support else None
    return {
        "passed": bool(
            np.allclose(co_membership, co_membership.T, atol=1e-10)
            and np.allclose(np.diag(co_membership), 1.0, atol=1e-10)
            and np.all((co_membership >= -1e-10) & (co_membership <= 1.0 + 1e-10))
            and (not complete_support or (psd_min is not None and psd_min >= -1e-8))
        ),
        "candidate_count": structure.candidate_count,
        "co_membership_symmetry_residual": float(
            np.linalg.norm(co_membership - co_membership.T)),
        "co_membership_defined_min": float(defined.min(initial=0.0)),
        "co_membership_psd_checked": complete_support,
        "co_membership_min_eigenvalue": psd_min,
        "persistent_dense_same_block_present": "same_block_probability" in structure.to_dict(),
    }


__all__ = [
    "PartitionCandidate", "PartitionStructureV1", "validate_partition_structure",
]
