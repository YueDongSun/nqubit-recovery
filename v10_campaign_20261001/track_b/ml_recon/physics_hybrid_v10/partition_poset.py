"""Partition refinement poset and minimal compatible antichain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Partition = Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class PartitionPruningResult:
    minimal_compatible_partitions: Tuple[Partition, ...]
    candidates_complete: bool
    result_exact: bool
    input_count: int
    compatible_count: int
    pruned_by_pair_or_subset: int
    pruned_by_cut: int
    pruned_by_gme: int


def canonical_partition(blocks: Iterable[Iterable[int]]) -> Partition:
    normalized = [tuple(sorted(set(int(i) for i in block))) for block in blocks]
    if any(not block for block in normalized):
        raise ValueError("partition blocks cannot be empty")
    flat = [i for block in normalized for i in block]
    if len(flat) != len(set(flat)):
        raise ValueError("partition blocks must be disjoint")
    return tuple(sorted(normalized, key=lambda b: (b[0], len(b), b)))


def block_size_canonical(partition: Partition) -> Tuple[int, ...]:
    return tuple(sorted((len(b) for b in partition), reverse=True))


def refines(fine: Partition, coarse: Partition) -> bool:
    """True when every fine block is contained in one coarse block."""
    return all(any(set(f).issubset(c) for c in coarse) for f in fine)


def minimal_compatible_antichain(partitions: Iterable[Partition]) -> Tuple[Partition, ...]:
    unique = sorted({canonical_partition(p) for p in partitions},
                    key=lambda p: (-len(p), block_size_canonical(p), p))
    result: List[Partition] = []
    for candidate in unique:
        if any(refines(existing, candidate) and existing != candidate for existing in unique):
            continue
        if not any(refines(candidate, kept) or refines(kept, candidate) for kept in result):
            result.append(candidate)
    return tuple(result)


def prune_partitions_with_audit(
    partitions: Iterable[Partition],
    certified_entangled_subsets: Iterable[Sequence[int]],
    *,
    certified_cut_separations: Iterable[Tuple[Sequence[int], Sequence[int]]] = (),
    certified_gme_subsets: Iterable[Sequence[int]] = (),
    candidates_complete: bool = False,
) -> PartitionPruningResult:
    """Prune only partitions contradicted by the supplied certificate semantics."""
    partitions = tuple(canonical_partition(partition) for partition in partitions)
    entangled = [set(s) for s in certified_entangled_subsets]
    cut_constraints = [(set(left), set(right))
                       for left, right in certified_cut_separations]
    gme_constraints = [set(subset) for subset in certified_gme_subsets]
    compatible = []
    pruned_subset = 0
    pruned_cut = 0
    pruned_gme = 0
    for partition in partitions:
        blocks = [set(block) for block in partition]
        # A generic entangled reduced subset requires at least one global block
        # to contain two of its members; it does not imply GME of that subset.
        if not all(any(len(block & subset) >= 2 for block in blocks) for subset in entangled):
            pruned_subset += 1
            continue
        # An NPT/witness certificate across A|B excludes product partitions with
        # no global block crossing that exact cut.
        if not all(any(block & left and block & right for block in blocks)
                   for left, right in cut_constraints):
            pruned_cut += 1
            continue
        # A GME reduced subset cannot be assembled from multiple product blocks.
        if not all(any(subset.issubset(block) for block in blocks)
                   for subset in gme_constraints):
            pruned_gme += 1
            continue
        compatible.append(partition)
    antichain = minimal_compatible_antichain(compatible)
    return PartitionPruningResult(
        minimal_compatible_partitions=antichain,
        candidates_complete=bool(candidates_complete),
        result_exact=bool(candidates_complete),
        input_count=len(partitions),
        compatible_count=len(compatible),
        pruned_by_pair_or_subset=pruned_subset,
        pruned_by_cut=pruned_cut,
        pruned_by_gme=pruned_gme,
    )


def prune_partitions(partitions: Iterable[Partition], certified_entangled_subsets: Iterable[Sequence[int]]
                     ) -> Tuple[Partition, ...]:
    return prune_partitions_with_audit(
        partitions, certified_entangled_subsets).minimal_compatible_partitions
