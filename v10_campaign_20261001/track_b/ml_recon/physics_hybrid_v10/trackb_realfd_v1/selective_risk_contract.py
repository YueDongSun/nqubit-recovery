"""Fail-closed calibration and risk-certification contracts for Track B critics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import math

from scipy.stats import beta as beta_distribution

from .contracts import canonical_sha256


RISK_NOT_TRAINED = "NOT_TRAINED"
RISK_TRAINED_UNCALIBRATED = "TRAINED_UNCALIBRATED"
RISK_CALIBRATED_CERTIFIED = "CALIBRATED_CERTIFIED"
RISK_QUALIFICATION_STATES = (
    RISK_NOT_TRAINED,
    RISK_TRAINED_UNCALIBRATED,
    RISK_CALIBRATED_CERTIFIED,
)
CERTIFICATE_NUMERIC_TOLERANCE = 1e-12


def clopper_pearson_one_sided_upper_bound(
    harmful_event_count: int,
    certification_sample_count: int,
    confidence_level: float,
) -> float:
    """Return the exact one-sided binomial upper confidence bound."""
    if (type(harmful_event_count) is not int
            or type(certification_sample_count) is not int
            or certification_sample_count <= 0
            or not 0 <= harmful_event_count <= certification_sample_count
            or not math.isfinite(float(confidence_level))
            or not 0.0 < float(confidence_level) <= 1.0):
        raise ValueError("Clopper-Pearson inputs are invalid")
    if harmful_event_count == certification_sample_count:
        return 1.0
    if harmful_event_count == 0:
        return float(
            1.0 - (1.0 - float(confidence_level)) ** (1.0 / certification_sample_count),
        )
    upper_bound = float(beta_distribution.ppf(
        float(confidence_level), harmful_event_count + 1,
        certification_sample_count - harmful_event_count,
    ))
    if not math.isfinite(upper_bound) or not 0.0 <= upper_bound <= 1.0:
        raise ValueError("Clopper-Pearson upper bound is invalid")
    return upper_bound


def selective_risk_policy_sha(
    *,
    system_type: str,
    harm_event_kind: str,
    statistical_unit: str,
    confidence_level: float,
    upper_bound_method: str,
    policy_harm_probability_threshold: float,
    maximum_harm_upper_bound: float,
    minimum_coverage: float,
) -> str:
    """Content-bind every policy value consumed by a certified decision."""
    if system_type not in ("3qubit", "4qubit"):
        raise ValueError("selective risk policy system is invalid")
    if (harm_event_kind != "AT_LEAST_ONE_MATERIAL_HARMFUL_ALLOW_V1"
            or statistical_unit != "ANCESTOR_TRAJECTORY"
            or upper_bound_method != "CLOPPER_PEARSON_ONE_SIDED_V1"):
        raise ValueError("selective risk policy event definition is invalid")
    for value in (
        confidence_level,
        policy_harm_probability_threshold,
        maximum_harm_upper_bound,
        minimum_coverage,
    ):
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
            raise ValueError("selective risk policy threshold is invalid")
    return canonical_sha256({
        "schema": "TrackBFrozenSelectiveRiskPolicyV1",
        "system_type": system_type,
        "harm_event_kind": harm_event_kind,
        "statistical_unit": statistical_unit,
        "confidence_level": float(confidence_level),
        "upper_bound_method": upper_bound_method,
        "policy_harm_probability_threshold": float(policy_harm_probability_threshold),
        "maximum_harm_upper_bound": float(maximum_harm_upper_bound),
        "minimum_coverage": float(minimum_coverage),
    })


@dataclass(frozen=True)
class TrackBSelectiveRiskQualificationV1:
    """Evidence needed before a critic score can inform a research-only gate."""

    system_type: str
    shared_encoder_sha: str
    critic_checkpoint_sha: str
    calibration_sample_set_sha: str
    certification_sample_set_sha: str
    calibration_split: str
    certification_split: str
    calibrator_sha: str
    frozen_policy_sha: str
    calibration_group_ids: tuple[str, ...]
    certification_group_ids: tuple[str, ...]
    harm_event_kind: str
    statistical_unit: str
    certification_sample_count: int
    harmful_event_count: int
    confidence_level: float
    upper_bound_method: str
    policy_harm_probability_threshold: float
    maximum_harm_upper_bound: float
    minimum_coverage: float
    qualification_state: str
    harm_upper_bound: float | None
    coverage: float | None
    coverage_allowed_count: int
    coverage_total_count: int
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBSelectiveRiskQualificationV1"

    def validate(self) -> None:
        if self.system_type not in ("3qubit", "4qubit") or self.qualification_state not in RISK_QUALIFICATION_STATES:
            raise ValueError("selective risk system/state is invalid")
        for name in (
            "shared_encoder_sha",
            "critic_checkpoint_sha",
            "calibration_sample_set_sha",
            "certification_sample_set_sha",
            "calibrator_sha",
            "frozen_policy_sha",
        ):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ValueError(f"selective risk {name} is invalid")
        if self.calibration_split != "calibration" or self.certification_split != "risk_certification":
            raise ValueError("selective risk requires independent calibration and certification roles")
        if (not self.calibration_group_ids or not self.certification_group_ids
                or len(set(self.calibration_group_ids)) != len(self.calibration_group_ids)
                or len(set(self.certification_group_ids)) != len(self.certification_group_ids)
                or set(self.calibration_group_ids) & set(self.certification_group_ids)):
            raise ValueError("selective risk calibration and certification groups must be nonempty and disjoint")
        if (self.calibration_sample_set_sha != canonical_sha256(sorted(self.calibration_group_ids))
                or self.certification_sample_set_sha != canonical_sha256(sorted(self.certification_group_ids))):
            raise ValueError("selective risk group identities differ from their sample-set digests")
        if (self.harm_event_kind != "AT_LEAST_ONE_MATERIAL_HARMFUL_ALLOW_V1"
                or self.statistical_unit != "ANCESTOR_TRAJECTORY"
                or self.upper_bound_method != "CLOPPER_PEARSON_ONE_SIDED_V1"
                or type(self.certification_sample_count) is not int
                or self.certification_sample_count != len(self.certification_group_ids)
                or type(self.harmful_event_count) is not int
                or not 0 <= self.harmful_event_count <= self.certification_sample_count):
            raise ValueError("selective risk event/certification definition is invalid")
        for value in (
            self.confidence_level,
            self.policy_harm_probability_threshold,
            self.maximum_harm_upper_bound,
            self.minimum_coverage,
        ):
            if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
                raise ValueError("selective risk threshold is invalid")
        expected_policy_sha = selective_risk_policy_sha(
            system_type=self.system_type,
            harm_event_kind=self.harm_event_kind,
            statistical_unit=self.statistical_unit,
            confidence_level=self.confidence_level,
            upper_bound_method=self.upper_bound_method,
            policy_harm_probability_threshold=self.policy_harm_probability_threshold,
            maximum_harm_upper_bound=self.maximum_harm_upper_bound,
            minimum_coverage=self.minimum_coverage,
        )
        if self.frozen_policy_sha != expected_policy_sha:
            raise ValueError("selective risk frozen policy identity does not bind its thresholds")
        if self.qualification_state == RISK_CALIBRATED_CERTIFIED:
            if self.harm_upper_bound is None or self.coverage is None:
                raise ValueError("certified selective risk requires bound and coverage")
            for value in (self.harm_upper_bound, self.coverage):
                if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                    raise ValueError("certified selective risk metric is outside [0, 1]")
            if (type(self.coverage_allowed_count) is not int
                    or type(self.coverage_total_count) is not int
                    or self.coverage_total_count != self.certification_sample_count
                    or not 0 <= self.harmful_event_count <= self.coverage_allowed_count
                    or not self.coverage_allowed_count <= self.coverage_total_count):
                raise ValueError("certified selective risk coverage counts are invalid")
            expected_coverage = self.coverage_allowed_count / self.coverage_total_count
            if not math.isclose(
                float(self.coverage), expected_coverage,
                rel_tol=0.0, abs_tol=CERTIFICATE_NUMERIC_TOLERANCE,
            ):
                raise ValueError("certified selective risk coverage differs from its counters")
            expected_upper_bound = clopper_pearson_one_sided_upper_bound(
                self.harmful_event_count,
                self.certification_sample_count,
                self.confidence_level,
            )
            if not math.isclose(
                float(self.harm_upper_bound), expected_upper_bound,
                rel_tol=0.0, abs_tol=CERTIFICATE_NUMERIC_TOLERANCE,
            ):
                raise ValueError("certified selective risk upper bound differs from its event counts")
        elif (self.harm_upper_bound is not None
                or self.coverage is not None
                or self.coverage_allowed_count != 0
                or self.coverage_total_count != 0):
            raise ValueError("uncertified selective risk cannot claim certification metrics")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.hardware_action_authorized):
            raise ValueError("selective risk qualification cannot authorize selection, promotion, or hardware")
        if self.schema != "TrackBSelectiveRiskQualificationV1":
            raise ValueError("unsupported selective risk qualification schema")

    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class TrackBSelectiveRiskDecisionV1:
    """A fail-closed research decision; it never emits or authorizes an action."""

    allow: bool
    abstain: bool
    reason: str
    qualification_sha: str
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBSelectiveRiskDecisionV1"

    def validate(self) -> None:
        if type(self.allow) is not bool or type(self.abstain) is not bool or self.allow == self.abstain:
            raise ValueError("selective risk decision must be exactly allow or abstain")
        if type(self.reason) is not str or not self.reason or len(self.qualification_sha) != 64:
            raise ValueError("selective risk decision identity is invalid")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.hardware_action_authorized):
            raise ValueError("selective risk decision cannot authorize selection, promotion, or hardware")
        if self.schema != "TrackBSelectiveRiskDecisionV1":
            raise ValueError("unsupported selective risk decision schema")


def fail_closed_selective_risk_decision(
    qualification: TrackBSelectiveRiskQualificationV1,
    *,
    predicted_harm_probability: float,
    expected_system_type: str,
    expected_shared_encoder_sha: str,
    expected_critic_checkpoint_sha: str,
    expected_calibrator_sha: str,
    expected_frozen_policy_sha: str,
) -> TrackBSelectiveRiskDecisionV1:
    """Allow only certified confidence below a fixed research-only threshold."""
    qualification.validate()
    if (not math.isfinite(float(predicted_harm_probability))
            or not 0.0 <= float(predicted_harm_probability) <= 1.0
            or expected_system_type not in ("3qubit", "4qubit")
            or any(type(value) is not str or len(value) != 64 for value in (
                expected_shared_encoder_sha,
                expected_critic_checkpoint_sha,
                expected_calibrator_sha,
                expected_frozen_policy_sha,
            ))):
        raise ValueError("selective risk probabilities are invalid")
    if qualification.qualification_state != RISK_CALIBRATED_CERTIFIED:
        result = TrackBSelectiveRiskDecisionV1(
            allow=False,
            abstain=True,
            reason="UNQUALIFIED_CONFIDENCE",
            qualification_sha=qualification.sha256(),
        )
    elif (qualification.system_type != expected_system_type
            or qualification.shared_encoder_sha != expected_shared_encoder_sha
            or qualification.critic_checkpoint_sha != expected_critic_checkpoint_sha
            or qualification.calibrator_sha != expected_calibrator_sha
            or qualification.frozen_policy_sha != expected_frozen_policy_sha):
        result = TrackBSelectiveRiskDecisionV1(
            allow=False,
            abstain=True,
            reason="QUALIFICATION_IDENTITY_MISMATCH",
            qualification_sha=qualification.sha256(),
        )
    elif (qualification.harm_upper_bound is None
            or qualification.coverage is None
            or qualification.harm_upper_bound > qualification.maximum_harm_upper_bound
            or qualification.coverage < qualification.minimum_coverage):
        result = TrackBSelectiveRiskDecisionV1(
            allow=False,
            abstain=True,
            reason="CERTIFICATION_BOUND_OR_COVERAGE_INSUFFICIENT",
            qualification_sha=qualification.sha256(),
        )
    elif float(predicted_harm_probability) > qualification.policy_harm_probability_threshold:
        result = TrackBSelectiveRiskDecisionV1(
            allow=False,
            abstain=True,
            reason="PREDICTED_HARM_EXCEEDS_RESEARCH_THRESHOLD",
            qualification_sha=qualification.sha256(),
        )
    else:
        result = TrackBSelectiveRiskDecisionV1(
            allow=True,
            abstain=False,
            reason="CERTIFIED_RESEARCH_ONLY_CONFIDENCE_BELOW_THRESHOLD",
            qualification_sha=qualification.sha256(),
        )
    result.validate()
    return result


__all__ = [
    "RISK_NOT_TRAINED",
    "RISK_TRAINED_UNCALIBRATED",
    "RISK_CALIBRATED_CERTIFIED",
    "clopper_pearson_one_sided_upper_bound",
    "selective_risk_policy_sha",
    "TrackBSelectiveRiskQualificationV1",
    "TrackBSelectiveRiskDecisionV1",
    "fail_closed_selective_risk_decision",
]
