"""Weak measured-state transport pairs for Physics Hybrid V8.

The measured corpus contains useful relations between observed states even
when it does not contain an executed action between those states.  This module
builds a bounded number of comparable ordered endpoint pairs for representation
learning.  A target-state displacement is always a *transport proxy*, never an
executed action, a real action label, or a Jacobian label.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml_recon.measured_data.contracts import COMMAND_SUPERVISION_PROVENANCE

from .contracts import stable_json_sha256
from .feedback_data import V8MeasuredRowV1


STATE_PAIR_TRANSPORT_SCHEMA = "physics_hybrid_v8_state_pair_transport_v1"
STATE_PAIR_TRANSPORT_SET_SCHEMA = "physics_hybrid_v8_state_pair_transport_set_v1"
STATE_PAIR_TRANSPORT_WEAK_EVIDENCE = "state_pair_transport_weak"
TARGET_TRANSPORT_PROXY_KIND = "target_state_transport_not_executed_action"
TARGET_TRANSPORT_PROXY_DIM = 6
TARGET_TRANSPORT_PROXY_FEATURE_NAMES = (
    "amplitude_helmert_0",
    "amplitude_helmert_1",
    "amplitude_helmert_2",
    "phase_helmert_0",
    "phase_helmert_1",
    "phase_helmert_2",
)
CALIBRATION_TRANSPORT_TIERS = frozenset(
    {"strict_calibration", "cross_calibration"}
)
CALIBRATION_FEATURE_POLICIES = {
    "strict_calibration": "masked_count_rho_nuisance_transport",
    "cross_calibration": "scale_free_count_shares_and_target_transport_only",
}
PAIR_PURPOSES = frozenset(
    {
        "train_gradient",
        "validation_evaluation",
        "group_cross_fit_train",
        "frozen_architecture_refit",
    }
)
LOCKED_SPLIT_TOKENS = ("holdout", "ood", "test")
ABSOLUTE_COUNT_FEATURE_COUNT = 4
COUNT_FEATURE_NAMES = (
    "log_coin_rate",
    "log_signal_rate",
    "log_idler_rate",
    "log_read_rate",
    "log_coin_per_read",
    "log_signal_per_read",
    "log_idler_per_read",
    "log_write_per_read",
)
NUISANCE_FEATURE_NAMES = (
    "signal_robust_z_score",
    "idler_robust_z_score",
    "log_signal_to_idler",
    "saturation_flag",
    "low_low_collapse_flag",
    "signal_high_idler_low_flag",
)


@dataclass(frozen=True)
class StatePairTransportConfig:
    """Centralized controls for sparse, deterministic endpoint pairing."""

    time_bin_width_s: float = 86400.0
    target_entropy_bin_width: float = 0.25
    target_phase_dispersion_bin_width: float = 0.25
    exposure_log_bin_width: float = 0.10
    maximum_exposure_ratio: float = 1.10
    maximum_target_distance: float = 4.0
    minimum_target_distance: float = 1e-8
    target_similarity_scale: float = 1.0
    near_neighbors_per_anchor: int = 2
    far_neighbors_per_anchor: int = 2
    far_distance_bins: int = 4
    maximum_candidate_scan: int = 64
    unknown_exposure_weight: float = 0.55
    unknown_calibration_weight: float = 0.75
    cross_calibration_known_weight: float = 0.25
    cross_calibration_unknown_weight: float = 0.15
    enable_cross_calibration_fallback: bool = True
    minimum_pair_weight: float = 1e-6
    cross_fit_fold_count: int = 5
    seed: int = 20260809

    def validate(self) -> None:
        positive_values = (
            self.time_bin_width_s,
            self.target_entropy_bin_width,
            self.target_phase_dispersion_bin_width,
            self.exposure_log_bin_width,
            self.maximum_target_distance,
            self.target_similarity_scale,
            self.minimum_pair_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive_values):
            raise ValueError("state-pair scale parameters must be finite and positive")
        if not 1.0 <= float(self.maximum_exposure_ratio):
            raise ValueError("maximum_exposure_ratio must be at least one")
        if not 0.0 <= float(self.minimum_target_distance) < float(self.maximum_target_distance):
            raise ValueError("target-distance limits are inconsistent")
        if min(
            int(self.near_neighbors_per_anchor),
            int(self.far_neighbors_per_anchor),
        ) < 0:
            raise ValueError("neighbor counts must be non-negative")
        if int(self.near_neighbors_per_anchor) + int(self.far_neighbors_per_anchor) <= 0:
            raise ValueError("at least one neighbor per anchor is required")
        if min(
            int(self.far_distance_bins),
            int(self.maximum_candidate_scan),
            int(self.cross_fit_fold_count),
        ) < 1:
            raise ValueError("distance bins, scan bound, and fold count must be positive")
        for value in (
            self.unknown_exposure_weight,
            self.unknown_calibration_weight,
            self.cross_calibration_known_weight,
            self.cross_calibration_unknown_weight,
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError("missing-provenance weights must be in (0, 1]")
        if not (
            float(self.cross_calibration_unknown_weight)
            <= float(self.cross_calibration_known_weight)
            < float(self.unknown_calibration_weight)
        ):
            raise ValueError(
                "cross-calibration weights must stay below strict unknown-calibration weight"
            )
        if not isinstance(self.enable_cross_calibration_fallback, bool):
            raise ValueError("enable_cross_calibration_fallback must be boolean")


@dataclass(frozen=True)
class StateEndpointTransportFeaturesV1:
    """Masked endpoint outputs used to form a weak transport difference."""

    sample_id: str
    count_features: Tuple[float, ...]
    count_feature_mask: Tuple[int, ...]
    nuisance_features: Tuple[float, ...]
    nuisance_feature_mask: Tuple[int, ...]
    rho_features: Optional[Tuple[float, ...]] = None
    rho_feature_mask: int = 0

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("endpoint sample_id is required")
        if len(self.count_features) != len(COUNT_FEATURE_NAMES):
            raise ValueError("endpoint count feature width changed")
        if len(self.count_feature_mask) != len(self.count_features):
            raise ValueError("count feature mask width disagrees")
        if len(self.nuisance_features) != len(NUISANCE_FEATURE_NAMES):
            raise ValueError("endpoint nuisance feature width changed")
        if len(self.nuisance_feature_mask) != len(self.nuisance_features):
            raise ValueError("nuisance feature mask width disagrees")
        if any(int(value) not in {0, 1} for value in (*self.count_feature_mask, *self.nuisance_feature_mask)):
            raise ValueError("endpoint feature masks must be binary")
        if not np.all(np.isfinite(np.asarray(self.count_features, dtype=np.float64))):
            raise ValueError("endpoint count features must be finite")
        if not np.all(np.isfinite(np.asarray(self.nuisance_features, dtype=np.float64))):
            raise ValueError("endpoint nuisance features must be finite")
        rho_mask = int(self.rho_feature_mask)
        if rho_mask not in {0, 1}:
            raise ValueError("rho_feature_mask must be binary")
        if rho_mask == 1:
            if not self.rho_features:
                raise ValueError("available rho transport requires non-empty features")
            if not np.all(np.isfinite(np.asarray(self.rho_features, dtype=np.float64))):
                raise ValueError("rho transport features must be finite")
        elif self.rho_features is not None:
            raise ValueError("missing rho transport features must remain None")


@dataclass(frozen=True)
class StatePairTransportV1:
    """One auditable ordered weak-transport relation between measured states."""

    pair_id: str
    transport_group_id: str
    system_type: str
    split_name: str
    purpose: str
    source_sample_id: str
    target_sample_id: str
    source_measurement_group_id: str
    target_measurement_group_id: str
    comparability_stratum_id: str
    calibration_transport_tier: str
    calibration_feature_policy: str
    calibration_stratum: str
    source_calibration_hash: Optional[str]
    target_calibration_hash: Optional[str]
    exposure_stratum: str
    time_stratum: str
    target_geometry_stratum: str
    sampling_band: str
    distance_rank_bucket: int
    cross_fit_fold: int
    target_transport_proxy: Tuple[float, ...]
    target_transport_proxy_mask: Tuple[int, ...]
    target_distance: float
    w_state_infidelity: float
    elapsed_time_s: float
    exposure_ratio: Optional[float]
    count_transport_delta: Tuple[float, ...]
    count_transport_mask: Tuple[int, ...]
    rho_transport_delta: Optional[Tuple[float, ...]]
    rho_transport_mask: int
    source_rho_available_mask: int
    target_rho_available_mask: int
    nuisance_transport_delta: Tuple[float, ...]
    nuisance_transport_mask: Tuple[int, ...]
    source_global_multi_click_upper_bounds: Tuple[int, int, int]
    target_global_multi_click_upper_bounds: Tuple[int, int, int]
    global_multi_click_upper_bound_delta: Tuple[int, int, int]
    pair_weight: float
    repeat_weight: float
    target_similarity_weight: float
    time_proximity_weight: float
    exposure_comparability_weight: float
    calibration_confidence_weight: float
    source_command_identity_hash: Optional[str]
    target_command_identity_hash: Optional[str]
    strict_paired_validation_candidate: bool
    gradient_eligible: bool
    evidence_kind: str = STATE_PAIR_TRANSPORT_WEAK_EVIDENCE
    action_proxy_kind: str = TARGET_TRANSPORT_PROXY_KIND
    real_action_supervision_mask: int = 0
    jacobian_label: None = None
    jacobian_supervision_mask: int = 0
    operation_local_multi_click_counts: None = None
    schema: str = STATE_PAIR_TRANSPORT_SCHEMA

    def validate(self) -> None:
        if self.schema != STATE_PAIR_TRANSPORT_SCHEMA:
            raise ValueError("unsupported state-pair transport schema")
        if self.purpose not in PAIR_PURPOSES:
            raise ValueError("unsupported state-pair purpose")
        if not self.pair_id or not self.transport_group_id:
            raise ValueError("pair and transport-group identities are required")
        if self.source_sample_id == self.target_sample_id:
            raise ValueError("state-pair endpoints must be distinct")
        if self.source_measurement_group_id == self.target_measurement_group_id:
            raise ValueError("repeat-group endpoints cannot form a transport action proxy")
        if self.evidence_kind != STATE_PAIR_TRANSPORT_WEAK_EVIDENCE:
            raise ValueError("state recombination remains weak transport evidence")
        if self.action_proxy_kind != TARGET_TRANSPORT_PROXY_KIND:
            raise ValueError("target displacement cannot be described as an executed action")
        if int(self.real_action_supervision_mask) != 0:
            raise ValueError("state-pair transport can never supervise a real action")
        if self.jacobian_label is not None or int(self.jacobian_supervision_mask) != 0:
            raise ValueError("state-pair transport cannot fabricate a Jacobian label")
        if self.operation_local_multi_click_counts is not None:
            raise ValueError("global multi-click bounds cannot be expanded per operation")
        if self.calibration_transport_tier not in CALIBRATION_TRANSPORT_TIERS:
            raise ValueError("unsupported calibration transport tier")
        if self.calibration_feature_policy != CALIBRATION_FEATURE_POLICIES[
            self.calibration_transport_tier
        ]:
            raise ValueError("calibration feature policy disagrees with tier")
        exposure_is_authoritative = self.exposure_stratum.startswith(
            "exposure_known_"
        )
        if exposure_is_authoritative != (self.exposure_ratio is not None):
            raise ValueError("exposure ratio and exposure stratum disagree")
        if self.calibration_transport_tier == "strict_calibration":
            if (
                self.source_calibration_hash is None
                or self.target_calibration_hash is None
                or self.source_calibration_hash != self.target_calibration_hash
            ):
                raise ValueError(
                    "strict calibration transport requires two authoritative equal hashes"
                )
            if not exposure_is_authoritative:
                raise ValueError(
                    "strict calibration transport requires authoritative compatible exposures"
                )
        else:
            if (
                self.source_calibration_hash is not None
                and self.target_calibration_hash is not None
                and self.source_calibration_hash == self.target_calibration_hash
                and exposure_is_authoritative
            ):
                raise ValueError(
                    "known equal calibration with compatible known exposure must be strict"
                )
        if self.sampling_band not in {"near", "far"}:
            raise ValueError("sampling_band must be near or far")
        if int(self.distance_rank_bucket) < 0 or int(self.cross_fit_fold) < 0:
            raise ValueError("distance bucket and cross-fit fold must be non-negative")
        if len(self.target_transport_proxy) != TARGET_TRANSPORT_PROXY_DIM:
            raise ValueError("target transport proxy must use the fixed six-coordinate contract")
        if len(self.target_transport_proxy_mask) != TARGET_TRANSPORT_PROXY_DIM:
            raise ValueError("target transport proxy mask width changed")
        if any(int(value) not in {0, 1} for value in self.target_transport_proxy_mask):
            raise ValueError("target transport proxy mask must be binary")
        if any(
            abs(float(value)) > 1e-12
            for value, mask in zip(
                self.target_transport_proxy,
                self.target_transport_proxy_mask,
            )
            if not mask
        ):
            raise ValueError("masked target transport coordinates must remain zero")
        if not math.isfinite(float(self.target_distance)) or float(self.target_distance) <= 0.0:
            raise ValueError("target distance must be finite and positive")
        if not 0.0 <= float(self.w_state_infidelity) <= 1.0 + 1e-9:
            raise ValueError("W-state infidelity must be in [0, 1]")
        if float(self.elapsed_time_s) < 0.0:
            raise ValueError("elapsed time must be non-negative")
        if self.exposure_ratio is not None and (
            not math.isfinite(float(self.exposure_ratio))
            or float(self.exposure_ratio) < 1.0
        ):
            raise ValueError("exposure ratio must be finite and defined as max/min")
        if len(self.count_transport_delta) != len(COUNT_FEATURE_NAMES):
            raise ValueError("count transport feature width changed")
        if len(self.count_transport_mask) != len(self.count_transport_delta):
            raise ValueError("count transport mask width disagrees")
        if len(self.nuisance_transport_delta) != len(NUISANCE_FEATURE_NAMES):
            raise ValueError("nuisance transport feature width changed")
        if len(self.nuisance_transport_mask) != len(self.nuisance_transport_delta):
            raise ValueError("nuisance transport mask width disagrees")
        if any(int(value) not in {0, 1} for value in (*self.count_transport_mask, *self.nuisance_transport_mask)):
            raise ValueError("transport masks must be binary")
        rho_mask = int(self.rho_transport_mask)
        if rho_mask not in {0, 1}:
            raise ValueError("rho_transport_mask must be binary")
        if rho_mask == 1 and not self.rho_transport_delta:
            raise ValueError("available rho transport requires a delta")
        if rho_mask == 0 and self.rho_transport_delta is not None:
            raise ValueError("missing rho transport must remain None")
        if rho_mask > min(int(self.source_rho_available_mask), int(self.target_rho_available_mask)):
            raise ValueError("rho transport cannot exceed endpoint availability")
        if self.calibration_transport_tier == "cross_calibration":
            if any(self.count_transport_mask[:ABSOLUTE_COUNT_FEATURE_COUNT]):
                raise ValueError("cross-calibration transport must mask absolute count rates")
            if rho_mask != 0 or self.rho_transport_delta is not None:
                raise ValueError("cross-calibration transport must mask rho deltas")
            if any(self.nuisance_transport_mask) or any(self.nuisance_transport_delta):
                raise ValueError("cross-calibration transport must mask nuisance deltas")
        if len(self.source_global_multi_click_upper_bounds) != 3 or len(self.target_global_multi_click_upper_bounds) != 3:
            raise ValueError("global multi-click bounds must keep three global totals")
        expected_delta = tuple(
            int(target) - int(source)
            for source, target in zip(
                self.source_global_multi_click_upper_bounds,
                self.target_global_multi_click_upper_bounds,
            )
        )
        if tuple(self.global_multi_click_upper_bound_delta) != expected_delta:
            raise ValueError("global multi-click upper-bound delta is inconsistent")
        components = (
            self.pair_weight,
            self.repeat_weight,
            self.target_similarity_weight,
            self.time_proximity_weight,
            self.exposure_comparability_weight,
            self.calibration_confidence_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in components):
            raise ValueError("pair weight and components must be finite and positive")
        expected_candidate = bool(
            self.calibration_transport_tier == "strict_calibration"
            and self.source_command_identity_hash
            and self.target_command_identity_hash
        )
        if bool(self.strict_paired_validation_candidate) != expected_candidate:
            raise ValueError("strict candidate flag disagrees with endpoint command identities")


@dataclass(frozen=True)
class StatePairTransportAuditV1:
    """Coverage ledger separating strict and cross-calibration weak pairs."""

    source_rows_by_system: Dict[str, int]
    strict_calibration_pair_count_by_system: Dict[str, int]
    cross_calibration_pair_count_by_system: Dict[str, int]
    strict_calibration_endpoint_count_by_system: Dict[str, int]
    cross_calibration_endpoint_count_by_system: Dict[str, int]
    cross_calibration_new_endpoint_count_by_system: Dict[str, int]
    union_endpoint_count_by_system: Dict[str, int]
    union_endpoint_coverage_by_system: Dict[str, float]

    def validate(self) -> None:
        keys = set(self.source_rows_by_system)
        if not keys:
            raise ValueError("state-pair audit requires at least one system")
        mappings = (
            self.strict_calibration_pair_count_by_system,
            self.cross_calibration_pair_count_by_system,
            self.strict_calibration_endpoint_count_by_system,
            self.cross_calibration_endpoint_count_by_system,
            self.cross_calibration_new_endpoint_count_by_system,
            self.union_endpoint_count_by_system,
            self.union_endpoint_coverage_by_system,
        )
        if any(set(mapping) != keys for mapping in mappings):
            raise ValueError("state-pair audit system keys disagree")
        for system_type in keys:
            source_count = int(self.source_rows_by_system[system_type])
            strict_count = int(
                self.strict_calibration_endpoint_count_by_system[system_type]
            )
            cross_count = int(
                self.cross_calibration_endpoint_count_by_system[system_type]
            )
            new_count = int(
                self.cross_calibration_new_endpoint_count_by_system[system_type]
            )
            union_count = int(self.union_endpoint_count_by_system[system_type])
            if min(source_count, strict_count, cross_count, new_count, union_count) < 0:
                raise ValueError("state-pair audit counts must be non-negative")
            if max(strict_count, cross_count, union_count) > source_count:
                raise ValueError("state-pair endpoint coverage cannot exceed source rows")
            if new_count > cross_count or union_count != strict_count + new_count:
                raise ValueError("cross-calibration new endpoint count is inconsistent")
            expected_fraction = union_count / float(source_count)
            if abs(
                float(self.union_endpoint_coverage_by_system[system_type])
                - expected_fraction
            ) > 1e-12:
                raise ValueError("state-pair endpoint coverage fraction is inconsistent")


@dataclass(frozen=True)
class StatePairTransportSetV1:
    """Purpose-bound pair set that records split and metric restrictions."""

    pairs: Tuple[StatePairTransportV1, ...]
    purpose: str
    included_split_names: Tuple[str, ...]
    source_sample_count: int
    audit: StatePairTransportAuditV1
    gradient_eligible: bool
    development_metric_eligible: bool
    architecture_must_be_frozen: bool
    locked_test_accessed: bool = False
    locked_test_metric_inheritance_allowed: bool = False
    evidence_kind: str = STATE_PAIR_TRANSPORT_WEAK_EVIDENCE
    schema: str = STATE_PAIR_TRANSPORT_SET_SCHEMA

    def validate(self) -> None:
        if self.schema != STATE_PAIR_TRANSPORT_SET_SCHEMA:
            raise ValueError("unsupported state-pair set schema")
        if self.purpose not in PAIR_PURPOSES:
            raise ValueError("unsupported state-pair set purpose")
        if not self.included_split_names or len(set(self.included_split_names)) != len(self.included_split_names):
            raise ValueError("pair set split names must be non-empty and unique")
        if int(self.source_sample_count) <= 0:
            raise ValueError("source_sample_count must be positive")
        self.audit.validate()
        if sum(self.audit.source_rows_by_system.values()) != int(self.source_sample_count):
            raise ValueError("pair-set source count disagrees with coverage audit")
        if self.locked_test_accessed or self.locked_test_metric_inheritance_allowed:
            raise ValueError("state-pair construction never opens or inherits a locked test")
        if self.evidence_kind != STATE_PAIR_TRANSPORT_WEAK_EVIDENCE:
            raise ValueError("pair set evidence kind must remain weak")
        expected_gradient = self.purpose != "validation_evaluation"
        if bool(self.gradient_eligible) != expected_gradient:
            raise ValueError("gradient eligibility disagrees with pair purpose")
        if bool(self.development_metric_eligible) != (self.purpose == "validation_evaluation"):
            raise ValueError("development metric eligibility disagrees with pair purpose")
        if bool(self.architecture_must_be_frozen) != (self.purpose == "frozen_architecture_refit"):
            raise ValueError("architecture freeze flag disagrees with refit purpose")
        for pair in self.pairs:
            pair.validate()
            if pair.purpose != self.purpose or pair.split_name not in self.included_split_names:
                raise ValueError("pair metadata disagrees with its purpose-bound set")
            if pair.gradient_eligible != self.gradient_eligible:
                raise ValueError("pair gradient eligibility disagrees with its set")
        for system_type in self.audit.source_rows_by_system:
            strict_pairs = [
                pair
                for pair in self.pairs
                if pair.system_type == system_type
                and pair.calibration_transport_tier == "strict_calibration"
            ]
            cross_pairs = [
                pair
                for pair in self.pairs
                if pair.system_type == system_type
                and pair.calibration_transport_tier == "cross_calibration"
            ]
            if len(strict_pairs) != self.audit.strict_calibration_pair_count_by_system[
                system_type
            ]:
                raise ValueError("strict-calibration pair count disagrees with audit")
            if len(cross_pairs) != self.audit.cross_calibration_pair_count_by_system[
                system_type
            ]:
                raise ValueError("cross-calibration pair count disagrees with audit")
            strict_endpoints = {
                sample_id
                for pair in strict_pairs
                for sample_id in (pair.source_sample_id, pair.target_sample_id)
            }
            cross_endpoints = {
                sample_id
                for pair in cross_pairs
                for sample_id in (pair.source_sample_id, pair.target_sample_id)
            }
            if len(strict_endpoints) != self.audit.strict_calibration_endpoint_count_by_system[
                system_type
            ]:
                raise ValueError("strict-calibration endpoint count disagrees with audit")
            if len(cross_endpoints) != self.audit.cross_calibration_endpoint_count_by_system[
                system_type
            ]:
                raise ValueError("cross-calibration endpoint count disagrees with audit")

    @property
    def strict_candidate_count(self) -> int:
        return sum(int(pair.strict_paired_validation_candidate) for pair in self.pairs)


def build_state_endpoint_transport_features(
    row: V8MeasuredRowV1,
    *,
    rho_features: Optional[Sequence[float]] = None,
) -> StateEndpointTransportFeaturesV1:
    """Build aggregate endpoint features without inventing exposure or rho.

    Unknown exposure masks the four absolute-rate features.  Scale-free count
    shares remain usable.  Global multi-click totals are deliberately absent
    from the feature vector and stay interval-style endpoint audit fields.
    """

    record = row.record
    exposure_known = int(row.exposure_known_mask) == 1
    exposure = float(row.exposure_s) if exposure_known else 1.0
    totals = (
        float(record.coin_total),
        float(record.signal_total),
        float(record.idler_total),
        float(record.read_count),
    )
    absolute_rates = tuple(math.log1p(max(value, 0.0) / exposure) for value in totals)
    read_denominator = max(float(record.read_count), 0.0) + 1.0
    relative_features = (
        math.log((max(float(record.coin_total), 0.0) + 0.5) / read_denominator),
        math.log((max(float(record.signal_total), 0.0) + 0.5) / read_denominator),
        math.log((max(float(record.idler_total), 0.0) + 0.5) / read_denominator),
        math.log((max(float(record.write_count), 0.0) + 0.5) / read_denominator),
    )
    relative_mask = int(record.read_count > 0)
    count_mask = (int(exposure_known),) * ABSOLUTE_COUNT_FEATURE_COUNT + (relative_mask,) * 4
    nuisance = (
        float(record.signal_robust_z_score),
        float(record.idler_robust_z_score),
        math.log((max(float(record.signal_total), 0.0) + 0.5) / (max(float(record.idler_total), 0.0) + 0.5)),
        float(bool(record.saturation_flag)),
        float(bool(record.low_low_collapse_flag)),
        float(bool(record.signal_high_idler_low_flag)),
    )
    parsed_rho: Optional[Tuple[float, ...]] = None
    rho_mask = 0
    if rho_features is not None:
        if int(row.rho_available_mask) != 1:
            raise ValueError("rho features were supplied for an endpoint with missing measured rho")
        parsed_rho = tuple(float(value) for value in rho_features)
        rho_mask = 1
    features = StateEndpointTransportFeaturesV1(
        sample_id=row.sample_id,
        count_features=tuple(float(value) for value in (*absolute_rates, *relative_features)),
        count_feature_mask=tuple(int(value) for value in count_mask),
        nuisance_features=tuple(float(value) for value in nuisance),
        nuisance_feature_mask=(1,) * len(NUISANCE_FEATURE_NAMES),
        rho_features=parsed_rho,
        rho_feature_mask=rho_mask,
    )
    features.validate()
    return features


def pack_target_transport_proxy(
    amplitude_coordinates: Sequence[float],
    phase_coordinates: Sequence[float],
    system_type: str,
) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    """Pack target displacement in the canonical six-coordinate order.

    The order matches the V7/V8 projective coordinate layout solely for batch
    compatibility: ``amp[0:3], phase[0:3]``.  For 3Q, coordinates 2 and 5 are
    padded zero and masked.  Matching an action tensor's shape does not turn
    this target-state displacement into an executed action.
    """

    modes = 2 if system_type == "3qubit" else 3 if system_type == "4qubit" else 0
    if modes == 0:
        raise ValueError("system_type must be 3qubit or 4qubit")
    amplitude = tuple(float(value) for value in amplitude_coordinates)
    phase = tuple(float(value) for value in phase_coordinates)
    if len(amplitude) != modes or len(phase) != modes:
        raise ValueError("target transport coordinate widths disagree with system type")
    proxy = np.zeros(TARGET_TRANSPORT_PROXY_DIM, dtype=np.float64)
    mask = np.zeros(TARGET_TRANSPORT_PROXY_DIM, dtype=np.int64)
    proxy[:modes] = np.asarray(amplitude, dtype=np.float64)
    proxy[3 : 3 + modes] = np.asarray(phase, dtype=np.float64)
    mask[:modes] = 1
    mask[3 : 3 + modes] = 1
    return (
        tuple(float(value) for value in proxy),
        tuple(int(value) for value in mask),
    )


def build_state_pair_transport(
    rows: Sequence[V8MeasuredRowV1],
    *,
    split_names: Sequence[str],
    purpose: str,
    config: StatePairTransportConfig = StatePairTransportConfig(),
    locked_split_names: Sequence[str] = (),
    rho_features_by_sample_id: Optional[Mapping[str, Sequence[float]]] = None,
) -> StatePairTransportSetV1:
    """Build sparse ordered pairs inside strict comparability strata.

    Pairing never crosses a split.  Training/cross-fit accepts train splits,
    validation is evaluation-only, and an all-eligible refit accepts train and
    validation only after architecture freeze.  Names containing holdout, OOD,
    or test are locked by default even if omitted from ``locked_split_names``.
    """

    config.validate()
    normalized_purpose = str(purpose)
    if normalized_purpose not in PAIR_PURPOSES:
        raise ValueError(f"unsupported state-pair purpose: {purpose}")
    normalized_splits = tuple(dict.fromkeys(str(name).strip() for name in split_names))
    if not normalized_splits or any(not name for name in normalized_splits):
        raise ValueError("split_names must be non-empty")
    explicit_locked = {str(name) for name in locked_split_names}
    inferred_locked = {
        name
        for name in normalized_splits
        if any(token in name.lower() for token in LOCKED_SPLIT_TOKENS)
    }
    locked_requested = set(normalized_splits) & (explicit_locked | inferred_locked)
    if locked_requested:
        raise PermissionError(
            "state-pair construction cannot open locked splits: "
            + ", ".join(sorted(locked_requested))
        )
    _validate_purpose_splits(normalized_purpose, normalized_splits)
    selected_rows = sorted(
        (row for row in rows if row.split_name in normalized_splits),
        key=lambda row: row.sample_id,
    )
    if not selected_rows:
        raise ValueError("no measured rows matched the requested non-locked splits")
    if {row.split_name for row in selected_rows} != set(normalized_splits):
        missing = set(normalized_splits) - {row.split_name for row in selected_rows}
        raise ValueError(f"requested splits have no rows: {sorted(missing)}")

    rho_mapping = dict(rho_features_by_sample_id or {})
    unknown_rho_ids = set(rho_mapping) - {row.sample_id for row in selected_rows}
    if unknown_rho_ids:
        raise ValueError("rho feature mapping contains samples outside the selected split set")
    endpoint_features = {
        row.sample_id: build_state_endpoint_transport_features(
            row,
            rho_features=rho_mapping.get(row.sample_id),
        )
        for row in selected_rows
    }
    state_cache = {row.sample_id: _state_geometry(row) for row in selected_rows}
    gradient_eligible = normalized_purpose != "validation_evaluation"
    strict_buckets = _bucket_state_rows(
        selected_rows,
        state_cache,
        normalized_purpose,
        config,
        calibration_transport_tier="strict_calibration",
    )
    strict_pairs = _build_pairs_from_buckets(
        strict_buckets,
        endpoint_features=endpoint_features,
        state_cache=state_cache,
        purpose=normalized_purpose,
        gradient_eligible=gradient_eligible,
        calibration_transport_tier="strict_calibration",
        config=config,
    )
    strict_endpoint_ids = {
        sample_id
        for pair in strict_pairs
        for sample_id in (pair.source_sample_id, pair.target_sample_id)
    }
    cross_pairs: list[StatePairTransportV1] = []
    if config.enable_cross_calibration_fallback:
        uncovered_anchor_ids = {
            row.sample_id for row in selected_rows if row.sample_id not in strict_endpoint_ids
        }
        cross_buckets = _bucket_state_rows(
            selected_rows,
            state_cache,
            normalized_purpose,
            config,
            calibration_transport_tier="cross_calibration",
        )
        cross_pairs = _build_pairs_from_buckets(
            cross_buckets,
            endpoint_features=endpoint_features,
            state_cache=state_cache,
            purpose=normalized_purpose,
            gradient_eligible=gradient_eligible,
            calibration_transport_tier="cross_calibration",
            config=config,
            anchor_sample_ids=uncovered_anchor_ids,
        )
    pairs = [*strict_pairs, *cross_pairs]
    audit = _build_transport_audit(selected_rows, pairs)

    result = StatePairTransportSetV1(
        pairs=tuple(pairs),
        purpose=normalized_purpose,
        included_split_names=normalized_splits,
        source_sample_count=len(selected_rows),
        audit=audit,
        gradient_eligible=gradient_eligible,
        development_metric_eligible=normalized_purpose == "validation_evaluation",
        architecture_must_be_frozen=normalized_purpose == "frozen_architecture_refit",
    )
    result.validate()
    return result


def _bucket_state_rows(
    rows: Sequence[V8MeasuredRowV1],
    state_cache: Mapping[str, Mapping[str, object]],
    purpose: str,
    config: StatePairTransportConfig,
    *,
    calibration_transport_tier: str,
) -> Dict[Tuple[str, ...], list[V8MeasuredRowV1]]:
    buckets: Dict[Tuple[str, ...], list[V8MeasuredRowV1]] = defaultdict(list)
    for row in rows:
        if calibration_transport_tier == "strict_calibration" and (
            row.calibration_hash is None or int(row.exposure_known_mask) != 1
        ):
            continue
        cross_fit_fold = _measurement_group_fold(
            row.measurement_group_key,
            config.cross_fit_fold_count,
        )
        bucket_key = _comparability_key(
            row,
            state_cache[row.sample_id],
            config,
            calibration_transport_tier=calibration_transport_tier,
            cross_fit_fold=(
                cross_fit_fold if purpose == "group_cross_fit_train" else None
            ),
        )
        buckets[bucket_key].append(row)
    return buckets


def _build_pairs_from_buckets(
    buckets: Mapping[Tuple[str, ...], Sequence[V8MeasuredRowV1]],
    *,
    endpoint_features: Mapping[str, StateEndpointTransportFeaturesV1],
    state_cache: Mapping[str, Mapping[str, object]],
    purpose: str,
    gradient_eligible: bool,
    calibration_transport_tier: str,
    config: StatePairTransportConfig,
    anchor_sample_ids: Optional[set[str]] = None,
) -> list[StatePairTransportV1]:
    pairs: list[StatePairTransportV1] = []
    for bucket_key in sorted(buckets):
        bucket = buckets[bucket_key]
        if len(bucket) < 2:
            continue
        ordered = sorted(
            bucket,
            key=lambda row: (
                state_cache[row.sample_id]["sort_projection"],
                row.sample_id,
            ),
        )
        for anchor_position, source in enumerate(ordered):
            if anchor_sample_ids is not None and source.sample_id not in anchor_sample_ids:
                continue
            candidates = _sparse_candidate_positions(
                ordered,
                anchor_position,
                config,
                required_transport_tier=calibration_transport_tier,
            )
            accepted_count = 0
            for target_position, sampling_band, distance_bucket in candidates:
                target = ordered[target_position]
                pair = _make_state_pair(
                    source,
                    target,
                    source_features=endpoint_features[source.sample_id],
                    target_features=endpoint_features[target.sample_id],
                    source_geometry=state_cache[source.sample_id],
                    target_geometry=state_cache[target.sample_id],
                    bucket_key=bucket_key,
                    purpose=purpose,
                    sampling_band=sampling_band,
                    distance_bucket=distance_bucket,
                    gradient_eligible=gradient_eligible,
                    calibration_transport_tier=calibration_transport_tier,
                    config=config,
                )
                if pair is None:
                    continue
                pairs.append(pair)
                accepted_count += 1
                maximum = int(config.near_neighbors_per_anchor) + int(
                    config.far_neighbors_per_anchor
                )
                if accepted_count >= maximum:
                    break
    return pairs


def _build_transport_audit(
    rows: Sequence[V8MeasuredRowV1],
    pairs: Sequence[StatePairTransportV1],
) -> StatePairTransportAuditV1:
    systems = sorted({row.system_type for row in rows})
    source_counts = {
        system_type: sum(int(row.system_type == system_type) for row in rows)
        for system_type in systems
    }
    strict_pairs = {
        system_type: [
            pair
            for pair in pairs
            if pair.system_type == system_type
            and pair.calibration_transport_tier == "strict_calibration"
        ]
        for system_type in systems
    }
    cross_pairs = {
        system_type: [
            pair
            for pair in pairs
            if pair.system_type == system_type
            and pair.calibration_transport_tier == "cross_calibration"
        ]
        for system_type in systems
    }

    def endpoints(values: Sequence[StatePairTransportV1]) -> set[str]:
        return {
            sample_id
            for pair in values
            for sample_id in (pair.source_sample_id, pair.target_sample_id)
        }

    strict_endpoints = {
        system_type: endpoints(strict_pairs[system_type]) for system_type in systems
    }
    cross_endpoints = {
        system_type: endpoints(cross_pairs[system_type]) for system_type in systems
    }
    union_endpoints = {
        system_type: strict_endpoints[system_type] | cross_endpoints[system_type]
        for system_type in systems
    }
    audit = StatePairTransportAuditV1(
        source_rows_by_system=source_counts,
        strict_calibration_pair_count_by_system={
            system_type: len(strict_pairs[system_type]) for system_type in systems
        },
        cross_calibration_pair_count_by_system={
            system_type: len(cross_pairs[system_type]) for system_type in systems
        },
        strict_calibration_endpoint_count_by_system={
            system_type: len(strict_endpoints[system_type]) for system_type in systems
        },
        cross_calibration_endpoint_count_by_system={
            system_type: len(cross_endpoints[system_type]) for system_type in systems
        },
        cross_calibration_new_endpoint_count_by_system={
            system_type: len(
                cross_endpoints[system_type] - strict_endpoints[system_type]
            )
            for system_type in systems
        },
        union_endpoint_count_by_system={
            system_type: len(union_endpoints[system_type]) for system_type in systems
        },
        union_endpoint_coverage_by_system={
            system_type: len(union_endpoints[system_type])
            / float(source_counts[system_type])
            for system_type in systems
        },
    )
    audit.validate()
    return audit


def _validate_purpose_splits(purpose: str, split_names: Sequence[str]) -> None:
    lowered = tuple(name.lower() for name in split_names)
    if purpose in {"train_gradient", "group_cross_fit_train"}:
        if any("train" not in name or "validation" in name for name in lowered):
            raise PermissionError("gradient/cross-fit pair construction accepts train splits only")
    elif purpose == "validation_evaluation":
        if any("validation" not in name for name in lowered):
            raise PermissionError("validation pair construction accepts validation splits only")
    elif purpose == "frozen_architecture_refit":
        if any("train" not in name and "validation" not in name for name in lowered):
            raise PermissionError("frozen refit accepts only train and validation splits")


def _state_geometry(row: V8MeasuredRowV1) -> Dict[str, object]:
    params = row.record.exact_target_state_params
    qubit_count = 3 if row.system_type == "3qubit" else 4
    amplitudes = np.asarray(params.get("amplitudes"), dtype=np.float64)
    phases = np.asarray(params.get("phases_rad"), dtype=np.float64)
    if amplitudes.shape != (qubit_count,) or phases.shape != (qubit_count,):
        raise ValueError(f"target state shape is invalid for {row.sample_id}")
    if not np.all(np.isfinite(amplitudes)) or not np.all(np.isfinite(phases)):
        raise ValueError(f"target state is non-finite for {row.sample_id}")
    amplitudes = np.abs(amplitudes)
    norm = float(np.linalg.norm(amplitudes))
    if norm <= 0.0:
        raise ValueError(f"target amplitudes have zero norm for {row.sample_id}")
    amplitudes = amplitudes / norm
    relative_phase = np.angle(np.exp(1j * (phases - phases[0])))
    probability = np.square(amplitudes)
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-12)))) / math.log(qubit_count)
    phase_dispersion = 1.0 - float(np.abs(np.mean(np.exp(1j * relative_phase))))
    helmert = _helmert_matrix(qubit_count)
    amplitude_coordinate = (np.log(np.maximum(amplitudes, 1e-8)) @ helmert).astype(np.float64)
    phase_coordinate = (relative_phase @ helmert).astype(np.float64)
    sort_vector = np.concatenate(
        (amplitude_coordinate, np.sin(phase_coordinate), np.cos(phase_coordinate))
    )
    coefficients = np.arange(1, sort_vector.size + 1, dtype=np.float64)
    sort_projection = float(np.dot(sort_vector, coefficients) / max(float(sort_vector.size), 1.0))
    return {
        "amplitudes": amplitudes,
        "phases": phases,
        "amplitude_coordinate": amplitude_coordinate,
        "phase_coordinate": phase_coordinate,
        "entropy": entropy,
        "phase_dispersion": phase_dispersion,
        "sort_projection": sort_projection,
    }


def _helmert_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size - 1), dtype=np.float64)
    for column in range(size - 1):
        denominator = math.sqrt(float((column + 1) * (column + 2)))
        matrix[: column + 1, column] = 1.0 / denominator
        matrix[column + 1, column] = -float(column + 1) / denominator
    return matrix


def _comparability_key(
    row: V8MeasuredRowV1,
    geometry: Mapping[str, object],
    config: StatePairTransportConfig,
    *,
    calibration_transport_tier: str,
    cross_fit_fold: Optional[int],
) -> Tuple[str, ...]:
    if calibration_transport_tier not in CALIBRATION_TRANSPORT_TIERS:
        raise ValueError("unsupported calibration transport tier")
    if calibration_transport_tier == "strict_calibration":
        if row.calibration_hash is None or int(row.exposure_known_mask) != 1:
            raise ValueError(
                "strict calibration buckets require authoritative calibration and exposure"
            )
        calibration = row.calibration_hash
    else:
        calibration = "cross_calibration_scale_free"
    if int(row.exposure_known_mask) == 1:
        exposure_bin = int(round(math.log(float(row.exposure_s)) / config.exposure_log_bin_width))
        exposure = f"exposure_known_{exposure_bin}"
    else:
        exposure = "exposure_unknown"
    time_bin = int(math.floor(float(row.record.timestamp) / config.time_bin_width_s))
    entropy_bin = int(math.floor(float(geometry["entropy"]) / config.target_entropy_bin_width))
    phase_bin = int(
        math.floor(float(geometry["phase_dispersion"]) / config.target_phase_dispersion_bin_width)
    )
    key = (
        row.system_type,
        row.split_name,
        calibration,
        exposure,
        f"time_{time_bin}",
        f"target_{entropy_bin}_{phase_bin}",
    )
    if cross_fit_fold is not None:
        key = (*key, f"cross_fit_{cross_fit_fold}")
    return key


def _measurement_group_fold(group_id: str, fold_count: int) -> int:
    digest = hashlib.sha256(str(group_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(fold_count)


def _sparse_candidate_positions(
    ordered: Sequence[V8MeasuredRowV1],
    anchor_position: int,
    config: StatePairTransportConfig,
    *,
    required_transport_tier: Optional[str] = None,
) -> Tuple[Tuple[int, str, int], ...]:
    """Return bounded projection-near and rank-stratified far candidates."""

    if (
        required_transport_tier is not None
        and required_transport_tier not in CALIBRATION_TRANSPORT_TIERS
    ):
        raise ValueError("unsupported required transport tier")
    size = len(ordered)
    anchor = ordered[anchor_position]
    anchor_group = anchor.measurement_group_key

    def candidate_allowed(position: int) -> bool:
        candidate = ordered[position]
        return bool(
            position != anchor_position
            and candidate.measurement_group_key != anchor_group
            and (
                required_transport_tier is None
                or _transport_tier_for_pair(anchor, candidate, config)
                == required_transport_tier
            )
        )

    selected: list[Tuple[int, str, int]] = []
    selected_positions: set[int] = {anchor_position}
    near_needed = int(config.near_neighbors_per_anchor)
    for offset in range(1, min(int(config.maximum_candidate_scan), size - 1) + 1):
        for position in (anchor_position - offset, anchor_position + offset):
            if (
                0 <= position < size
                and position not in selected_positions
                and candidate_allowed(position)
            ):
                selected.append((position, "near", 0))
                selected_positions.add(position)
                if sum(int(item[1] == "near") for item in selected) >= near_needed:
                    break
        if sum(int(item[1] == "near") for item in selected) >= near_needed:
            break

    far_needed = int(config.far_neighbors_per_anchor)
    bin_count = min(int(config.far_distance_bins), size)
    bin_width = int(math.ceil(size / float(bin_count)))
    anchor_id = ordered[anchor_position].sample_id
    candidate_bins = sorted(
        range(bin_count),
        key=lambda bin_index: (
            -abs((bin_index * bin_width + min(bin_width, size - bin_index * bin_width) / 2.0) - anchor_position),
            bin_index,
        ),
    )
    for bin_index in candidate_bins:
        if sum(int(item[1] == "far") for item in selected) >= far_needed:
            break
        start = bin_index * bin_width
        stop = min(size, start + bin_width)
        if start >= stop:
            continue
        digest = hashlib.sha256(
            f"{config.seed}:{anchor_id}:{bin_index}".encode("utf-8")
        ).digest()
        initial = start + int.from_bytes(digest[:8], "big") % (stop - start)
        for scan in range(min(stop - start, int(config.maximum_candidate_scan))):
            position = start + ((initial - start + scan) % (stop - start))
            if (
                position not in selected_positions
                and candidate_allowed(position)
            ):
                selected.append((position, "far", bin_index + 1))
                selected_positions.add(position)
                break
    return tuple(selected)


def _make_state_pair(
    source: V8MeasuredRowV1,
    target: V8MeasuredRowV1,
    *,
    source_features: StateEndpointTransportFeaturesV1,
    target_features: StateEndpointTransportFeaturesV1,
    source_geometry: Mapping[str, object],
    target_geometry: Mapping[str, object],
    bucket_key: Tuple[str, ...],
    purpose: str,
    sampling_band: str,
    distance_bucket: int,
    gradient_eligible: bool,
    calibration_transport_tier: str,
    config: StatePairTransportConfig,
) -> Optional[StatePairTransportV1]:
    inferred_transport_tier = _transport_tier_for_pair(source, target, config)
    if inferred_transport_tier != calibration_transport_tier:
        return None
    source_amplitude_coordinate = np.asarray(source_geometry["amplitude_coordinate"], dtype=np.float64)
    target_amplitude_coordinate = np.asarray(target_geometry["amplitude_coordinate"], dtype=np.float64)
    amplitude_delta = target_amplitude_coordinate - source_amplitude_coordinate
    source_phase = np.asarray(source_geometry["phases"], dtype=np.float64)
    target_phase = np.asarray(target_geometry["phases"], dtype=np.float64)
    phase_delta_raw = np.angle(np.exp(1j * (target_phase - source_phase)))
    global_phase = float(np.angle(np.mean(np.exp(1j * phase_delta_raw))))
    phase_delta_gauge_fixed = np.angle(np.exp(1j * (phase_delta_raw - global_phase)))
    phase_delta = phase_delta_gauge_fixed @ _helmert_matrix(len(source_phase))
    transport_proxy, transport_proxy_mask = pack_target_transport_proxy(
        amplitude_delta,
        phase_delta,
        source.system_type,
    )
    target_distance = float(
        math.sqrt(
            float(np.mean(np.square(amplitude_delta)))
            + float(np.mean(np.square(phase_delta)))
        )
    )
    if not config.minimum_target_distance < target_distance <= config.maximum_target_distance:
        return None
    source_amplitudes = np.asarray(source_geometry["amplitudes"], dtype=np.float64)
    target_amplitudes = np.asarray(target_geometry["amplitudes"], dtype=np.float64)
    overlap = np.sum(
        source_amplitudes
        * target_amplitudes
        * np.exp(1j * (target_phase - source_phase))
    )
    w_state_infidelity = float(np.clip(1.0 - np.square(np.abs(overlap)), 0.0, 1.0))
    elapsed_time = abs(float(target.record.timestamp) - float(source.record.timestamp))
    exposure_ratio: Optional[float]
    if int(source.exposure_known_mask) == 1 and int(target.exposure_known_mask) == 1:
        source_exposure = float(source.exposure_s)
        target_exposure = float(target.exposure_s)
        exposure_ratio = max(source_exposure, target_exposure) / min(source_exposure, target_exposure)
        if exposure_ratio > float(config.maximum_exposure_ratio):
            return None
        exposure_weight = math.exp(-abs(math.log(exposure_ratio)))
    elif int(source.exposure_known_mask) == 0 and int(target.exposure_known_mask) == 0:
        exposure_ratio = None
        exposure_weight = float(config.unknown_exposure_weight)
    else:
        return None
    if calibration_transport_tier == "strict_calibration":
        if (
            source.calibration_hash is None
            or target.calibration_hash is None
            or source.calibration_hash != target.calibration_hash
            or exposure_ratio is None
        ):
            return None
        calibration_weight = 1.0
    elif calibration_transport_tier == "cross_calibration":
        if (
            source.calibration_hash is not None
            and target.calibration_hash is not None
            and source.calibration_hash == target.calibration_hash
            and exposure_ratio is not None
        ):
            return None
        calibration_weight = (
            float(config.cross_calibration_known_weight)
            if source.calibration_hash is not None
            and target.calibration_hash is not None
            else float(config.cross_calibration_unknown_weight)
        )
    else:
        raise ValueError("unsupported calibration transport tier")
    repeat_weight = math.sqrt(
        max(float(source.record.sample_weight), 0.0)
        * max(float(target.record.sample_weight), 0.0)
    )
    target_similarity_weight = math.exp(-target_distance / config.target_similarity_scale)
    time_proximity_weight = math.exp(-elapsed_time / config.time_bin_width_s)
    pair_weight = max(
        float(config.minimum_pair_weight),
        repeat_weight
        * target_similarity_weight
        * time_proximity_weight
        * exposure_weight
        * calibration_weight,
    )

    count_mask = tuple(
        int(source_mask) * int(target_mask)
        for source_mask, target_mask in zip(
            source_features.count_feature_mask,
            target_features.count_feature_mask,
        )
    )
    if calibration_transport_tier == "cross_calibration":
        count_mask = (0,) * ABSOLUTE_COUNT_FEATURE_COUNT + count_mask[
            ABSOLUTE_COUNT_FEATURE_COUNT:
        ]
    count_delta = tuple(
        (float(target_value) - float(source_value)) if mask else 0.0
        for source_value, target_value, mask in zip(
            source_features.count_features,
            target_features.count_features,
            count_mask,
        )
    )
    nuisance_mask = tuple(
        int(source_mask) * int(target_mask)
        for source_mask, target_mask in zip(
            source_features.nuisance_feature_mask,
            target_features.nuisance_feature_mask,
        )
    )
    if calibration_transport_tier == "cross_calibration":
        nuisance_mask = (0,) * len(NUISANCE_FEATURE_NAMES)
    nuisance_delta = tuple(
        (float(target_value) - float(source_value)) if mask else 0.0
        for source_value, target_value, mask in zip(
            source_features.nuisance_features,
            target_features.nuisance_features,
            nuisance_mask,
        )
    )
    rho_mask = int(source_features.rho_feature_mask) * int(target_features.rho_feature_mask)
    if calibration_transport_tier == "cross_calibration":
        rho_mask = 0
    rho_delta: Optional[Tuple[float, ...]] = None
    if rho_mask:
        if len(source_features.rho_features or ()) != len(target_features.rho_features or ()):
            raise ValueError("rho endpoint feature widths disagree")
        rho_delta = tuple(
            float(target_value) - float(source_value)
            for source_value, target_value in zip(
                source_features.rho_features or (),
                target_features.rho_features or (),
            )
        )

    source_command_hash = _authoritative_command_identity(source)
    target_command_hash = _authoritative_command_identity(target)
    strict_candidate = bool(
        calibration_transport_tier == "strict_calibration"
        and source_command_hash
        and target_command_hash
    )
    transport_group_id = stable_json_sha256(
        sorted((source.measurement_group_key, target.measurement_group_key))
    )
    cross_fit_fold = (
        int(bucket_key[6].rsplit("_", 1)[-1])
        if len(bucket_key) > 6
        else _measurement_group_fold(transport_group_id, config.cross_fit_fold_count)
    )
    stratum_payload = {
        "system_type": bucket_key[0],
        "split_name": bucket_key[1],
        "calibration": bucket_key[2],
        "calibration_transport_tier": calibration_transport_tier,
        "exposure": bucket_key[3],
        "time": bucket_key[4],
        "target": bucket_key[5],
        "cross_fit": bucket_key[6] if len(bucket_key) > 6 else None,
    }
    stratum_id = stable_json_sha256(stratum_payload)
    pair_id = stable_json_sha256(
        {
            "source_sample_id": source.sample_id,
            "target_sample_id": target.sample_id,
            "purpose": purpose,
            "stratum_id": stratum_id,
            "evidence_kind": STATE_PAIR_TRANSPORT_WEAK_EVIDENCE,
        }
    )
    source_multi = tuple(int(value) for value in source.global_multi_click_upper_bounds)
    target_multi = tuple(int(value) for value in target.global_multi_click_upper_bounds)
    result = StatePairTransportV1(
        pair_id=pair_id,
        transport_group_id=transport_group_id,
        system_type=source.system_type,
        split_name=source.split_name,
        purpose=purpose,
        source_sample_id=source.sample_id,
        target_sample_id=target.sample_id,
        source_measurement_group_id=source.measurement_group_key,
        target_measurement_group_id=target.measurement_group_key,
        comparability_stratum_id=stratum_id,
        calibration_transport_tier=calibration_transport_tier,
        calibration_feature_policy=CALIBRATION_FEATURE_POLICIES[
            calibration_transport_tier
        ],
        calibration_stratum=bucket_key[2],
        source_calibration_hash=source.calibration_hash,
        target_calibration_hash=target.calibration_hash,
        exposure_stratum=bucket_key[3],
        time_stratum=bucket_key[4],
        target_geometry_stratum=bucket_key[5],
        sampling_band=sampling_band,
        distance_rank_bucket=int(distance_bucket),
        cross_fit_fold=cross_fit_fold,
        target_transport_proxy=transport_proxy,
        target_transport_proxy_mask=transport_proxy_mask,
        target_distance=target_distance,
        w_state_infidelity=w_state_infidelity,
        elapsed_time_s=elapsed_time,
        exposure_ratio=exposure_ratio,
        count_transport_delta=count_delta,
        count_transport_mask=count_mask,
        rho_transport_delta=rho_delta,
        rho_transport_mask=rho_mask,
        source_rho_available_mask=int(source.rho_available_mask),
        target_rho_available_mask=int(target.rho_available_mask),
        nuisance_transport_delta=nuisance_delta,
        nuisance_transport_mask=nuisance_mask,
        source_global_multi_click_upper_bounds=source_multi,
        target_global_multi_click_upper_bounds=target_multi,
        global_multi_click_upper_bound_delta=tuple(
            target_value - source_value
            for source_value, target_value in zip(source_multi, target_multi)
        ),
        pair_weight=pair_weight,
        repeat_weight=repeat_weight,
        target_similarity_weight=target_similarity_weight,
        time_proximity_weight=time_proximity_weight,
        exposure_comparability_weight=exposure_weight,
        calibration_confidence_weight=calibration_weight,
        source_command_identity_hash=source_command_hash,
        target_command_identity_hash=target_command_hash,
        strict_paired_validation_candidate=strict_candidate,
        gradient_eligible=gradient_eligible,
    )
    result.validate()
    return result


def _transport_tier_for_pair(
    source: V8MeasuredRowV1,
    target: V8MeasuredRowV1,
    config: StatePairTransportConfig,
) -> Optional[str]:
    """Classify one physically comparable pair before any feature is exposed."""

    source_exposure_known = int(source.exposure_known_mask) == 1
    target_exposure_known = int(target.exposure_known_mask) == 1
    if source_exposure_known != target_exposure_known:
        return None
    if source_exposure_known:
        source_exposure = float(source.exposure_s)
        target_exposure = float(target.exposure_s)
        exposure_ratio = max(source_exposure, target_exposure) / min(
            source_exposure,
            target_exposure,
        )
        if exposure_ratio > float(config.maximum_exposure_ratio):
            return None
    calibration_authoritative_equal = bool(
        source.calibration_hash is not None
        and target.calibration_hash is not None
        and source.calibration_hash == target.calibration_hash
    )
    if calibration_authoritative_equal and source_exposure_known:
        return "strict_calibration"
    return "cross_calibration"


def _authoritative_command_identity(row: V8MeasuredRowV1) -> Optional[str]:
    record = row.record
    if (
        record.command_provenance not in COMMAND_SUPERVISION_PROVENANCE
        or int(record.command_supervision_mask) != 1
        or not _is_sha256(record.command_table_hash)
    ):
        return None
    physical_hash = record.waveform_hash or record.matrix_snapshot_hash
    if not _is_sha256(physical_hash):
        return None
    return stable_json_sha256(
        {
            "command_table_hash": record.command_table_hash,
            "physical_hash": physical_hash,
            "operation_registry_hash": record.operation_registry_hash,
            "provenance": record.command_provenance,
        }
    )


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


__all__ = [
    "ABSOLUTE_COUNT_FEATURE_COUNT",
    "CALIBRATION_FEATURE_POLICIES",
    "CALIBRATION_TRANSPORT_TIERS",
    "COUNT_FEATURE_NAMES",
    "NUISANCE_FEATURE_NAMES",
    "PAIR_PURPOSES",
    "STATE_PAIR_TRANSPORT_SCHEMA",
    "STATE_PAIR_TRANSPORT_SET_SCHEMA",
    "STATE_PAIR_TRANSPORT_WEAK_EVIDENCE",
    "StateEndpointTransportFeaturesV1",
    "StatePairTransportConfig",
    "StatePairTransportAuditV1",
    "StatePairTransportSetV1",
    "StatePairTransportV1",
    "TARGET_TRANSPORT_PROXY_KIND",
    "TARGET_TRANSPORT_PROXY_DIM",
    "TARGET_TRANSPORT_PROXY_FEATURE_NAMES",
    "build_state_endpoint_transport_features",
    "build_state_pair_transport",
    "pack_target_transport_proxy",
]
