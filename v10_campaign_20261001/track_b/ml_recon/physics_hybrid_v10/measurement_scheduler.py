"""Deterministic global-Pauli schedules and auditable subset coverage."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .contracts import V10MeasurementContract, V10MeasurementSchedule, sha256_of
from .pauli_registry import LOCAL_PAULI_AXES, V10BasisDescriptor, full_local_pauli_registry


@dataclass(frozen=True)
class ScheduledSetting:
    descriptor: V10BasisDescriptor
    priority: float
    scheduler: str
    covered_subsets: Tuple[Tuple[int, ...], ...] = ()


def _all_subsets(N: int, size: int) -> Iterable[Tuple[int, ...]]:
    return itertools.combinations(range(N), size)


def _projective_gf3_columns(N: int) -> Tuple[Tuple[int, ...], ...]:
    """Return N pairwise non-collinear vectors over GF(3)."""
    dimension = 1
    while (3 ** dimension - 1) // 2 < N:
        dimension += 1
    columns = []
    for vector in itertools.product(range(3), repeat=dimension):
        if not any(vector):
            continue
        first = next(value for value in vector if value)
        inverse = 1 if first == 1 else 2
        normalized = tuple((inverse * value) % 3 for value in vector)
        if normalized not in columns:
            columns.append(normalized)
        if len(columns) == N:
            return tuple(columns)
    raise RuntimeError("failed to construct enough GF(3) projective columns")


def _strength_two_global_axes(N: int) -> Tuple[Tuple[str, ...], ...]:
    columns = _projective_gf3_columns(N)
    dimension = len(columns[0])
    rows = []
    for coefficient in itertools.product(range(3), repeat=dimension):
        rows.append(tuple(
            LOCAL_PAULI_AXES[sum(a * b for a, b in zip(coefficient, column)) % 3]
            for column in columns))
    return tuple(rows)


def _selected_hyperedges(
    N: int,
    strategy: str,
    max_hyperedges: int,
    topology_edges: Sequence[Tuple[int, int]],
    uncertainty: Optional[Dict[Tuple[int, ...], float]],
) -> Tuple[Tuple[int, ...], ...]:
    if max_hyperedges <= 0 or N > 32:
        return ()
    score = uncertainty or {}
    candidates = [
        subset
        for order in range(3, min(4, N) + 1)
        for subset in _all_subsets(N, order)
    ]
    if strategy == "physical_topology" and topology_edges:
        physical_nodes = set(q for edge in topology_edges for q in edge)
        candidates = [subset for subset in candidates
                      if len(set(subset) & physical_nodes) >= 2]
    candidates.sort(key=lambda subset: (
        -float(score.get(subset, 0.0)) if strategy == "adaptive_uncertainty" else 0.0,
        len(subset), subset))
    candidates_by_order = {
        order: [subset for subset in candidates if len(subset) == order]
        for order in range(3, min(4, N) + 1)
    }
    if candidates_by_order.get(3) and candidates_by_order.get(4) \
            and max_hyperedges >= 2:
        triples = sorted(candidates_by_order[3], key=lambda subset: (
            -float(score.get(subset, 0.0)), subset))
        quads = sorted(candidates_by_order[4], key=lambda subset: (
            -float(score.get(subset, 0.0)), subset))
        selected = []
        while len(selected) < max_hyperedges and (triples or quads):
            if triples and len(selected) < max_hyperedges:
                selected.append(triples.pop(0))
            if quads and len(selected) < max_hyperedges:
                selected.append(quads.pop(0))
        return tuple(selected)
    return tuple(candidates[:max_hyperedges])


class V10MeasurementScheduler:
    """Create physical global settings; marginal settings are derived coverage."""

    def __init__(self, N: int, *, topology_edges: Sequence[Tuple[int, int]] = ()):
        if not 3 <= N <= 256:
            raise ValueError("N must be in [3, 256]")
        self.N = N
        self.topology_edges = tuple(sorted({tuple(sorted(edge)) for edge in topology_edges}))

    def schedule(
        self,
        strategy: str = "covering_array",
        max_hyperedges: int = 128,
        uncertainty: Optional[Dict[Tuple[int, ...], float]] = None,
    ) -> Tuple[ScheduledSetting, ...]:
        allowed = {"covering_array", "graph_coloring", "perfect_hash_overlapping",
                   "physical_topology", "adaptive_uncertainty"}
        if strategy not in allowed:
            raise ValueError(f"unknown measurement scheduling strategy: {strategy}")

        if self.N <= 4:
            full_subset = (tuple(range(self.N)),)
            return tuple(ScheduledSetting(descriptor, 1.0, "full_pauli", full_subset)
                         for descriptor in full_local_pauli_registry(self.N))

        axes_rows: Dict[Tuple[str, ...], Tuple[float, str, Set[Tuple[int, ...]]]] = {}
        pair_subsets = tuple(_all_subsets(self.N, 2))
        singleton_subsets = tuple((q,) for q in range(self.N))
        base_coverage = set(singleton_subsets + pair_subsets)
        for axes in _strength_two_global_axes(self.N):
            axes_rows[axes] = (1.0, strategy, set(base_coverage))

        selected = _selected_hyperedges(
            self.N, strategy, max_hyperedges, self.topology_edges, uncertainty)
        for subset in selected:
            for local_axes in itertools.product(LOCAL_PAULI_AXES, repeat=len(subset)):
                axes = ["Z"] * self.N
                for qubit, axis in zip(subset, local_axes):
                    axes[qubit] = axis
                axes_tuple = tuple(axes)
                priority = float((uncertainty or {}).get(subset, 1.0))
                if axes_tuple in axes_rows:
                    old_priority, old_scheduler, coverage = axes_rows[axes_tuple]
                    updated_coverage = set(coverage)
                    updated_coverage.add(subset)
                    axes_rows[axes_tuple] = (
                        max(old_priority, priority), old_scheduler, updated_coverage)
                else:
                    axes_rows[axes_tuple] = (priority, f"{strategy}_hyperedge", {subset})

        result = []
        for index, axes in enumerate(sorted(axes_rows)):
            digest = sha256_of({"N": self.N, "axes": axes})[:16]
            setting_id = f"global_pauli_{index:05d}_{digest}"
            descriptor = V10BasisDescriptor(
                setting_id=setting_id,
                N=self.N,
                subset=tuple(range(self.N)),
                pauli_axes=axes,
                physical_group="global_local_pauli",
            )
            priority, scheduler, covered = axes_rows[axes]
            result.append(ScheduledSetting(
                descriptor=descriptor,
                priority=priority,
                scheduler=scheduler,
                covered_subsets=tuple(sorted(covered, key=lambda value: (len(value), value))),
            ))
        return tuple(result)

    def typed_schedule(
        self,
        strategy: str = "covering_array",
        max_hyperedges: int = 128,
        uncertainty: Optional[Dict[Tuple[int, ...], float]] = None,
    ) -> V10MeasurementSchedule:
        settings = self.schedule(strategy, max_hyperedges, uncertainty)
        descriptors = tuple(setting.descriptor.to_dict() for setting in settings)
        registry_sha = sha256_of(descriptors)
        covered = tuple(sorted(
            {subset for setting in settings for subset in setting.covered_subsets},
            key=lambda value: (len(value), value)))
        coverage_sha = sha256_of({
            "registry_sha": registry_sha,
            "covered_subsets": covered,
            "coverage": tuple((setting.descriptor.setting_id, setting.covered_subsets)
                              for setting in settings),
        })
        result = V10MeasurementSchedule(
            N=self.N,
            setting_ids=tuple(setting.descriptor.setting_id for setting in settings),
            pauli_axes=tuple(setting.descriptor.pauli_axes for setting in settings),
            setting_registry_sha=registry_sha,
            schedule_kind=("full_local_pauli" if self.N <= 4
                           else f"global_{strategy}_b1_prime"),
            covered_subsets=covered,
            coverage_sha=coverage_sha,
        )
        result.validate()
        return result

    def contract(self, strategy: str = "covering_array") -> V10MeasurementContract:
        schedule = self.typed_schedule(strategy)
        result = V10MeasurementContract(
            N=self.N,
            contract_kind="B1-full-Pauli" if self.N <= 4 else "B1-prime-overlapping",
            setting_ids=schedule.setting_ids,
            full_family_identifiable=self.N <= 4,
            stress_only=self.N == 256,
        )
        result.validate()
        return result
