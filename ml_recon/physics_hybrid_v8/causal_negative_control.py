"""Causal pairing negative controls for Physics Hybrid V8.

The controls in this module preserve split/system/calibration strata and exact
item marginals while forcing donors to come from a different leakage group.
They test whether a model uses a pairing relation rather than target-only
correlation.  Synthetic and weak-transport results remain explicitly separate
from real paired-action evidence.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .contracts import stable_json_sha256


DERANGEMENT_SCHEMA = "physics_hybrid_v8_stratified_derangement_v1"
CAUSAL_NEGATIVE_CONTROL_SCHEMA = "physics_hybrid_v8_causal_negative_control_v1"
SHUFFLE_KINDS = frozenset(
    {"target_action", "target_context", "state_endpoint"}
)
EVIDENCE_SCOPE_BY_SOURCE = {
    "synthetic_transition": "synthetic",
    "state_pair_transport_weak": "weak",
    "real_paired_episode": "real",
}


@dataclass(frozen=True)
class CausalNegativeControlConfig:
    """Pre-registered thresholds for group-bootstrap pairing controls."""

    bootstrap_replicates: int = 2000
    confidence_level: float = 0.95
    minimum_groups: int = 8
    minimum_relative_degradation: float = 0.05
    numerical_epsilon: float = 1e-12
    seed: int = 20260809

    def validate(self) -> None:
        if int(self.bootstrap_replicates) < 100:
            raise ValueError("bootstrap_replicates must be at least 100")
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if int(self.minimum_groups) < 2:
            raise ValueError("minimum_groups must be at least two")
        if not math.isfinite(float(self.minimum_relative_degradation)) or float(
            self.minimum_relative_degradation
        ) < 0.0:
            raise ValueError("minimum_relative_degradation must be finite and non-negative")
        if not math.isfinite(float(self.numerical_epsilon)) or float(self.numerical_epsilon) <= 0.0:
            raise ValueError("numerical_epsilon must be finite and positive")


@dataclass(frozen=True)
class StratifiedDerangementPlanV1:
    """Exact within-stratum item permutation excluding same-group donors."""

    item_ids: Tuple[str, ...]
    stratum_ids: Tuple[str, ...]
    group_ids: Tuple[str, ...]
    donor_indices: Tuple[int, ...]
    shuffle_kind: str
    seed: int
    schema: str = DERANGEMENT_SCHEMA

    def validate(self) -> None:
        if self.schema != DERANGEMENT_SCHEMA:
            raise ValueError("unsupported derangement schema")
        if self.shuffle_kind not in SHUFFLE_KINDS:
            raise ValueError("unsupported derangement shuffle kind")
        size = len(self.item_ids)
        if size == 0 or len(set(self.item_ids)) != size:
            raise ValueError("derangement item IDs must be non-empty and unique")
        if len(self.stratum_ids) != size or len(self.group_ids) != size:
            raise ValueError("derangement metadata widths disagree")
        if any(not value for value in (*self.item_ids, *self.stratum_ids, *self.group_ids)):
            raise ValueError("derangement identities must be non-empty")
        if len(self.donor_indices) != size or sorted(self.donor_indices) != list(range(size)):
            raise ValueError("donor indices must be an exact item permutation")
        for recipient_index, donor_index in enumerate(self.donor_indices):
            if recipient_index == donor_index:
                raise ValueError("derangement cannot retain a self-pair")
            if self.stratum_ids[recipient_index] != self.stratum_ids[donor_index]:
                raise ValueError("derangement cannot cross a comparability stratum")
            if self.group_ids[recipient_index] == self.group_ids[donor_index]:
                raise ValueError("derangement donor must come from a different leakage group")

    def apply(self, values: Sequence[Any]) -> Tuple[Any, ...]:
        """Apply the exact donor permutation to a target/action/endpoint field."""

        self.validate()
        if len(values) != len(self.item_ids):
            raise ValueError("values width disagrees with derangement plan")
        return tuple(values[index] for index in self.donor_indices)

    @property
    def plan_hash(self) -> str:
        self.validate()
        return stable_json_sha256(asdict(self))


@dataclass(frozen=True)
class CausalNegativeControlResultV1:
    """Group-bootstrap comparison between correct and shuffled pairing."""

    evidence_source: str
    evidence_scope: str
    shuffle_kind: str
    group_count: int
    minimum_groups: int
    minimum_groups_met: bool
    paired_loss_mean: float
    shuffled_loss_mean: float
    loss_degradation: float
    degradation_lcb: float
    degradation_ucb: float
    relative_degradation: float
    relative_degradation_lcb: float
    minimum_relative_degradation: float
    confidence_level: float
    bootstrap_replicates: int
    passed_within_evidence_scope: bool
    real_action_causality_supported: bool
    claim: str
    grouped_losses_hash: str
    seed: int
    schema: str = CAUSAL_NEGATIVE_CONTROL_SCHEMA

    def validate(self) -> None:
        if self.schema != CAUSAL_NEGATIVE_CONTROL_SCHEMA:
            raise ValueError("unsupported causal-negative-control schema")
        expected_scope = EVIDENCE_SCOPE_BY_SOURCE.get(self.evidence_source)
        if expected_scope is None or self.evidence_scope != expected_scope:
            raise ValueError("negative-control evidence source/scope is invalid")
        if self.shuffle_kind not in SHUFFLE_KINDS:
            raise ValueError("unsupported negative-control shuffle kind")
        if int(self.group_count) < 1 or int(self.minimum_groups) < 2:
            raise ValueError("negative control requires positive group counts")
        if bool(self.minimum_groups_met) != (int(self.group_count) >= int(self.minimum_groups)):
            raise ValueError("minimum_groups_met is inconsistent")
        numeric_values = (
            self.paired_loss_mean,
            self.shuffled_loss_mean,
            self.loss_degradation,
            self.degradation_lcb,
            self.degradation_ucb,
            self.relative_degradation,
            self.relative_degradation_lcb,
            self.minimum_relative_degradation,
            self.confidence_level,
        )
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("negative-control metrics must be finite")
        if self.degradation_lcb > self.degradation_ucb:
            raise ValueError("negative-control interval is reversed")
        if int(self.bootstrap_replicates) < 100:
            raise ValueError("negative-control bootstrap count is too small")
        if len(self.grouped_losses_hash) != 64:
            raise ValueError("grouped_losses_hash must be a SHA-256")
        if self.real_action_causality_supported and (
            self.evidence_scope != "real" or not self.passed_within_evidence_scope
        ):
            raise ValueError("weak or synthetic evidence cannot support real action causality")
        if self.evidence_scope != "real" and self.real_action_causality_supported:
            raise ValueError("non-real evidence cannot be promoted by a negative control")

    @property
    def result_hash(self) -> str:
        self.validate()
        return stable_json_sha256(asdict(self))


def build_stratified_group_derangement(
    item_ids: Sequence[str],
    stratum_ids: Sequence[str],
    group_ids: Sequence[str],
    *,
    shuffle_kind: str,
    seed: int = 20260809,
) -> StratifiedDerangementPlanV1:
    """Construct an exact, deterministic, group-excluding derangement.

    Each stratum is permuted independently.  The item marginal is therefore
    exactly preserved, while no target/action or state endpoint receives a
    donor from its own repeat/episode group.  A stratum whose largest group is
    more than half of its rows has no such exact permutation and fails closed.
    """

    parsed_items = tuple(str(value) for value in item_ids)
    parsed_strata = tuple(str(value) for value in stratum_ids)
    parsed_groups = tuple(str(value) for value in group_ids)
    if shuffle_kind not in SHUFFLE_KINDS:
        raise ValueError(f"unsupported shuffle_kind: {shuffle_kind}")
    size = len(parsed_items)
    if size == 0 or len(parsed_strata) != size or len(parsed_groups) != size:
        raise ValueError("derangement inputs must have equal non-zero lengths")
    if len(set(parsed_items)) != size:
        raise ValueError("derangement item IDs must be unique")
    strata: Dict[str, list[int]] = defaultdict(list)
    for index, stratum_id in enumerate(parsed_strata):
        if not parsed_items[index] or not stratum_id or not parsed_groups[index]:
            raise ValueError("derangement identities must be non-empty")
        strata[stratum_id].append(index)

    donor_indices = [-1] * size
    for stratum_id in sorted(strata):
        indices = strata[stratum_id]
        groups: Dict[str, list[int]] = defaultdict(list)
        for index in indices:
            groups[parsed_groups[index]].append(index)
        if len(groups) < 2:
            raise ValueError(f"stratum {stratum_id!r} has fewer than two leakage groups")
        maximum_group_size = max(len(values) for values in groups.values())
        if maximum_group_size * 2 > len(indices):
            raise ValueError(
                f"stratum {stratum_id!r} cannot be exactly deranged without same-group donors"
            )
        ordered_group_ids = sorted(
            groups,
            key=lambda group_id: _seeded_key(seed, stratum_id, group_id),
        )
        flattened: list[int] = []
        for group_id in ordered_group_ids:
            flattened.extend(
                sorted(
                    groups[group_id],
                    key=lambda index: _seeded_key(
                        seed,
                        stratum_id,
                        group_id,
                        parsed_items[index],
                    ),
                )
            )
        valid_shift: Optional[int] = None
        candidate_shifts = list(range(maximum_group_size, len(flattened))) + list(
            range(1, maximum_group_size)
        )
        for shift in candidate_shifts:
            if all(
                parsed_groups[recipient_index]
                != parsed_groups[flattened[(position + shift) % len(flattened)]]
                for position, recipient_index in enumerate(flattened)
            ):
                valid_shift = shift
                break
        if valid_shift is None:
            raise ValueError(f"stratum {stratum_id!r} has no exact group-excluding derangement")
        for position, recipient_index in enumerate(flattened):
            donor_indices[recipient_index] = flattened[
                (position + valid_shift) % len(flattened)
            ]

    plan = StratifiedDerangementPlanV1(
        item_ids=parsed_items,
        stratum_ids=parsed_strata,
        group_ids=parsed_groups,
        donor_indices=tuple(donor_indices),
        shuffle_kind=shuffle_kind,
        seed=int(seed),
    )
    plan.validate()
    return plan


def build_target_action_derangement(
    item_ids: Sequence[str],
    stratum_ids: Sequence[str],
    episode_group_ids: Sequence[str],
    *,
    seed: int = 20260809,
) -> StratifiedDerangementPlanV1:
    """Shuffle target-to-action assignments within causal-evaluation strata."""

    return build_stratified_group_derangement(
        item_ids,
        stratum_ids,
        episode_group_ids,
        shuffle_kind="target_action",
        seed=seed,
    )


def build_target_context_derangement(
    item_ids: Sequence[str],
    stratum_ids: Sequence[str],
    context_group_ids: Sequence[str],
    *,
    seed: int = 20260809,
) -> StratifiedDerangementPlanV1:
    """Shuffle whole target-context assignments without splitting groups."""

    return build_stratified_group_derangement(
        item_ids,
        stratum_ids,
        context_group_ids,
        shuffle_kind="target_context",
        seed=seed,
    )


def build_state_endpoint_derangement(
    endpoint_ids: Sequence[str],
    stratum_ids: Sequence[str],
    measurement_group_ids: Sequence[str],
    *,
    seed: int = 20260809,
) -> StratifiedDerangementPlanV1:
    """Shuffle measured endpoint assignments within comparable pair strata."""

    return build_stratified_group_derangement(
        endpoint_ids,
        stratum_ids,
        measurement_group_ids,
        shuffle_kind="state_endpoint",
        seed=seed,
    )


def evaluate_paired_vs_shuffled_loss(
    paired_losses: Sequence[float],
    shuffled_losses: Sequence[float],
    group_ids: Sequence[str],
    *,
    evidence_source: str,
    shuffle_kind: str,
    sample_weights: Optional[Sequence[float]] = None,
    config: CausalNegativeControlConfig = CausalNegativeControlConfig(),
) -> CausalNegativeControlResultV1:
    """Evaluate pairing information with an equal-group bootstrap.

    Rows are first collapsed to weighted means inside an episode/transport
    group.  Bootstrap resampling then treats groups equally.  Passing weak or
    synthetic evidence only validates that signal within that scope; only a
    separately provenance-bound ``real_paired_episode`` result can support a
    real action-causality claim.
    """

    config.validate()
    evidence_scope = EVIDENCE_SCOPE_BY_SOURCE.get(str(evidence_source))
    if evidence_scope is None:
        raise ValueError(f"unsupported evidence_source: {evidence_source}")
    if shuffle_kind not in SHUFFLE_KINDS:
        raise ValueError(f"unsupported shuffle_kind: {shuffle_kind}")
    paired = np.asarray(paired_losses, dtype=np.float64)
    shuffled = np.asarray(shuffled_losses, dtype=np.float64)
    parsed_groups = tuple(str(value) for value in group_ids)
    if paired.ndim != 1 or shuffled.shape != paired.shape or len(parsed_groups) != paired.size:
        raise ValueError("paired, shuffled, and group inputs must have equal one-dimensional widths")
    if paired.size == 0 or not np.all(np.isfinite(paired)) or not np.all(np.isfinite(shuffled)):
        raise ValueError("negative-control losses must be non-empty and finite")
    if any(not group_id for group_id in parsed_groups):
        raise ValueError("negative-control group IDs must be non-empty")
    if sample_weights is None:
        weights = np.ones_like(paired)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != paired.shape:
            raise ValueError("sample_weights shape disagrees with losses")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("sample_weights must be finite and positive")

    grouped_indices: Dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(parsed_groups):
        grouped_indices[group_id].append(index)
    ordered_groups = sorted(grouped_indices)
    grouped_paired = np.asarray(
        [
            np.average(paired[grouped_indices[group_id]], weights=weights[grouped_indices[group_id]])
            for group_id in ordered_groups
        ],
        dtype=np.float64,
    )
    grouped_shuffled = np.asarray(
        [
            np.average(shuffled[grouped_indices[group_id]], weights=weights[grouped_indices[group_id]])
            for group_id in ordered_groups
        ],
        dtype=np.float64,
    )
    group_count = len(ordered_groups)
    paired_mean = float(np.mean(grouped_paired))
    shuffled_mean = float(np.mean(grouped_shuffled))
    degradation = shuffled_mean - paired_mean
    relative_degradation = degradation / max(abs(paired_mean), config.numerical_epsilon)

    rng = np.random.default_rng(int(config.seed))
    bootstrap_indices = rng.integers(
        0,
        group_count,
        size=(int(config.bootstrap_replicates), group_count),
    )
    paired_bootstrap = np.mean(grouped_paired[bootstrap_indices], axis=1)
    shuffled_bootstrap = np.mean(grouped_shuffled[bootstrap_indices], axis=1)
    degradation_bootstrap = shuffled_bootstrap - paired_bootstrap
    relative_bootstrap = degradation_bootstrap / np.maximum(
        np.abs(paired_bootstrap),
        config.numerical_epsilon,
    )
    lower_quantile = 1.0 - float(config.confidence_level)
    upper_quantile = float(config.confidence_level)
    degradation_lcb = float(np.quantile(degradation_bootstrap, lower_quantile))
    degradation_ucb = float(np.quantile(degradation_bootstrap, upper_quantile))
    relative_lcb = float(np.quantile(relative_bootstrap, lower_quantile))
    minimum_groups_met = group_count >= int(config.minimum_groups)
    passed = bool(
        minimum_groups_met
        and degradation_lcb > 0.0
        and relative_lcb >= float(config.minimum_relative_degradation)
    )
    real_supported = bool(passed and evidence_scope == "real")
    if evidence_scope == "synthetic":
        claim = "synthetic_pairing_signal_only" if passed else "synthetic_pairing_signal_not_established"
    elif evidence_scope == "weak":
        claim = "weak_state_transport_signal_only" if passed else "weak_state_transport_signal_not_established"
    else:
        claim = "real_action_causality_supported" if passed else "real_action_causality_not_established"
    grouped_hash = stable_json_sha256(
        {
            "group_ids": ordered_groups,
            "paired_losses": grouped_paired.tolist(),
            "shuffled_losses": grouped_shuffled.tolist(),
            "evidence_source": evidence_source,
            "shuffle_kind": shuffle_kind,
        }
    )
    result = CausalNegativeControlResultV1(
        evidence_source=str(evidence_source),
        evidence_scope=evidence_scope,
        shuffle_kind=shuffle_kind,
        group_count=group_count,
        minimum_groups=int(config.minimum_groups),
        minimum_groups_met=minimum_groups_met,
        paired_loss_mean=paired_mean,
        shuffled_loss_mean=shuffled_mean,
        loss_degradation=degradation,
        degradation_lcb=degradation_lcb,
        degradation_ucb=degradation_ucb,
        relative_degradation=relative_degradation,
        relative_degradation_lcb=relative_lcb,
        minimum_relative_degradation=float(config.minimum_relative_degradation),
        confidence_level=float(config.confidence_level),
        bootstrap_replicates=int(config.bootstrap_replicates),
        passed_within_evidence_scope=passed,
        real_action_causality_supported=real_supported,
        claim=claim,
        grouped_losses_hash=grouped_hash,
        seed=int(config.seed),
    )
    result.validate()
    return result


def _seeded_key(seed: int, *values: object) -> bytes:
    encoded = ":".join(str(value) for value in (seed, *values)).encode("utf-8")
    return hashlib.sha256(encoded).digest()


__all__ = [
    "CAUSAL_NEGATIVE_CONTROL_SCHEMA",
    "DERANGEMENT_SCHEMA",
    "EVIDENCE_SCOPE_BY_SOURCE",
    "SHUFFLE_KINDS",
    "CausalNegativeControlConfig",
    "CausalNegativeControlResultV1",
    "StratifiedDerangementPlanV1",
    "build_state_endpoint_derangement",
    "build_stratified_group_derangement",
    "build_target_action_derangement",
    "build_target_context_derangement",
    "evaluate_paired_vs_shuffled_loss",
]
