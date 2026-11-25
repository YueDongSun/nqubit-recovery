"""Registered-candidate Track B FD selection with fail-closed proposal gating."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from .contracts import canonical_sha256, require_sha256
from .candidate_library import (
    RegisteredActionCandidateLibraryV1,
    RegisteredActionCandidateV1,
    registered_candidate_score,
)
from .posterior import TrackBPosteriorV1
from .real_csm import RealCSMSupportV1
from .real_rsm import RealRSMSupportV1
from .rf_mapping_core import RegisteredRFMappingCoreV1


@dataclass(frozen=True)
class ProposalEligibilityGateV1:
    """Research-only hard gate before an actionful proposal can serialize."""

    max_harm_probability: float = 0.25
    max_stop_probability: float = 0.25
    max_abstain_probability: float = 0.25
    min_direction_stability: float = 0.50
    min_rf_confidence: float = 0.50
    max_support_leakage_norm: float = 1.0e-8
    max_requested_to_materialized_error: float = 1.0e-8
    min_preview_direction_cosine: float = 0.999999
    max_actual_loaded_discrepancy: float = 1.0e-3
    min_actionful_score_over_no_op: float = 0.0
    schema: str = field(default="TrackBProposalEligibilityGateV1", init=False)

    def __post_init__(self) -> None:
        values = (
            self.max_harm_probability, self.max_stop_probability,
            self.max_abstain_probability, self.min_direction_stability,
            self.min_rf_confidence,
        )
        if not all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("proposal eligibility gate values must lie in [0,1]")
        for value in (
            self.max_support_leakage_norm,
            self.max_requested_to_materialized_error,
            self.max_actual_loaded_discrepancy,
            self.min_actionful_score_over_no_op,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("proposal preview tolerances must be finite and nonnegative")
        if not math.isfinite(float(self.min_preview_direction_cosine)) \
                or not -1.0 <= float(self.min_preview_direction_cosine) <= 1.0:
            raise ValueError("proposal preview direction threshold must lie in [-1,1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "max_harm_probability": self.max_harm_probability,
            "max_stop_probability": self.max_stop_probability,
            "max_abstain_probability": self.max_abstain_probability,
            "min_direction_stability": self.min_direction_stability,
            "min_rf_confidence": self.min_rf_confidence,
            "max_support_leakage_norm": self.max_support_leakage_norm,
            "max_requested_to_materialized_error": self.max_requested_to_materialized_error,
            "min_preview_direction_cosine": self.min_preview_direction_cosine,
            "max_actual_loaded_discrepancy": self.max_actual_loaded_discrepancy,
            "min_actionful_score_over_no_op": self.min_actionful_score_over_no_op,
        }

    def sha256(self) -> str:
        from .contracts import canonical_sha256

        return canonical_sha256(self.to_dict())

    def evaluate(
        self,
        *,
        candidate: RegisteredActionCandidateV1,
        csm: RealCSMSupportV1,
        rsm: RealRSMSupportV1,
        harm_probability: float,
        stop_probability: float,
        abstain_probability: float,
        direction_stability: float,
        rf_confidence: float,
    ) -> tuple[bool, str]:
        """Return a typed result; no threshold miss is silently softened."""
        if candidate.source == "NO_OP":
            return False, "NO_OP_NO_ACTIONFUL_PROPOSAL"
        if not csm.input_identity_verified or not csm.nuisance_profiled:
            return False, "CSM_INPUT_IDENTITY_OR_NUISANCE_PROFILE_NOT_VERIFIED"
        if not rsm.metric_bound or not rsm.has_recoverable_support:
            return False, "RSM_METRIC_SUPPORT_NOT_ELIGIBLE"
        if not candidate.recoverable_action.projection_verified:
            return False, "RECOVERABLE_ACTION_PROJECTION_NOT_VERIFIED"
        try:
            candidate.recoverable_action.verify_projection(rsm=rsm, csm=csm)
        except (TypeError, ValueError):
            return False, "RECOVERABLE_ACTION_PROJECTION_PROOF_INVALID"
        preview = candidate.rf_preview
        if preview.support_leakage_norm > self.max_support_leakage_norm:
            return False, "RF_PREVIEW_SUPPORT_LEAKAGE_EXCEEDS_GATE"
        if preview.requested_to_materialized_logical_error > self.max_requested_to_materialized_error:
            return False, "RF_PREVIEW_LOGICAL_ERROR_EXCEEDS_GATE"
        if preview.direction_cosine < self.min_preview_direction_cosine:
            return False, "RF_PREVIEW_DIRECTION_COSINE_BELOW_GATE"
        if preview.actual_loaded_discrepancy is not None \
                and preview.actual_loaded_discrepancy > self.max_actual_loaded_discrepancy:
            return False, "RF_PREVIEW_ACTUAL_DISCREPANCY_EXCEEDS_GATE"
        if harm_probability > self.max_harm_probability:
            return False, "HARM_PROBABILITY_EXCEEDS_GATE"
        if stop_probability > self.max_stop_probability:
            return False, "STOP_PROBABILITY_EXCEEDS_GATE"
        if abstain_probability > self.max_abstain_probability:
            return False, "ABSTAIN_PROBABILITY_EXCEEDS_GATE"
        if direction_stability < self.min_direction_stability:
            return False, "DIRECTION_STABILITY_BELOW_GATE"
        if rf_confidence < self.min_rf_confidence:
            return False, "RF_CONFIDENCE_BELOW_GATE"
        return True, "ACTIONFUL_PROPOSAL_ELIGIBLE"


@dataclass(frozen=True)
class FDSelectionV1:
    """Static counterfactual selection, never a causal gain prediction."""

    selected_candidate_sha: str
    candidate_ranking: tuple[str, ...]
    global_gamma: float
    candidate_uncertainty: float
    harm_probability: float
    stop_probability: float
    abstain_probability: float
    direction_stability: float
    rf_confidence: float
    reason_code: str
    eligibility_policy_sha: str
    per_mode_gamma_enabled: bool = False
    causal_gain_available: bool = False
    hardware_action_authorized: bool = False
    registered_candidate_sha: str | None = None
    proposal_eligible: bool = False
    eligibility_reason_code: str = "PROPOSAL_NOT_EVALUATED"
    applied_candidate: RegisteredActionCandidateV1 | None = None
    schema: str = field(default="TrackBFDSelectionV1", init=False)

    def __post_init__(self) -> None:
        require_sha256(self.eligibility_policy_sha, "eligibility_policy_sha")
        values = (
            self.global_gamma, self.candidate_uncertainty, self.harm_probability,
            self.stop_probability, self.abstain_probability, self.direction_stability,
            self.rf_confidence,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("FD selection diagnostics must be finite")
        if not 0.0 <= self.global_gamma <= 1.0 or not all(0.0 <= float(value) <= 1.0 for value in (
            self.candidate_uncertainty, self.harm_probability, self.stop_probability,
            self.abstain_probability, self.direction_stability, self.rf_confidence,
        )):
            raise ValueError("FD gamma/probability diagnostics must lie in [0,1]")
        registered = self.registered_candidate_sha or self.selected_candidate_sha
        if not self.candidate_ranking or registered not in self.candidate_ranking:
            raise ValueError("registered FD candidate must be in the closed ranking")
        if type(self.proposal_eligible) is not bool:
            raise TypeError("proposal_eligible must be an exact boolean")
        if type(self.eligibility_reason_code) is not str or not self.eligibility_reason_code:
            raise ValueError("FD selection requires an eligibility reason code")
        if self.applied_candidate is not None:
            if self.applied_candidate.candidate_sha != self.selected_candidate_sha:
                raise ValueError("applied candidate SHA must be the selected proposal candidate SHA")
            if self.applied_candidate.source == "NO_OP":
                raise ValueError("NO_OP cannot be represented as an actionful applied candidate")
            if self.applied_candidate.base_candidate_sha != registered:
                raise ValueError("applied candidate must bind the selected registered candidate")
        if self.proposal_eligible and self.applied_candidate is None:
            raise ValueError("eligible FD selection requires a fresh applied candidate")
        if self.per_mode_gamma_enabled or self.causal_gain_available or self.hardware_action_authorized:
            raise ValueError("Track B FD is global-gamma, noncausal, and research-only")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": self.schema,
            "selected_candidate_sha": self.selected_candidate_sha,
            "candidate_ranking": list(self.candidate_ranking),
            "global_gamma": self.global_gamma,
            "candidate_uncertainty": self.candidate_uncertainty,
            "harm_probability": self.harm_probability,
            "stop_probability": self.stop_probability,
            "abstain_probability": self.abstain_probability,
            "direction_stability": self.direction_stability,
            "rf_confidence": self.rf_confidence,
            "reason_code": self.reason_code,
            "eligibility_policy_sha": self.eligibility_policy_sha,
            "per_mode_gamma_enabled": False,
            "causal_gain_available": False,
            "hardware_action_authorized": False,
            "registered_candidate_sha": self.registered_candidate_sha,
            "proposal_eligible": self.proposal_eligible,
            "eligibility_reason_code": self.eligibility_reason_code,
            "applied_candidate_sha": (
                None if self.applied_candidate is None else self.applied_candidate.candidate_sha
            ),
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


class TrackBFDModelV1:
    """Rank a closed action library, then re-damp and re-preview the selected action."""

    def __init__(
        self,
        system_type: str,
        *,
        eligibility_gate: ProposalEligibilityGateV1 | None = None,
    ) -> None:
        if system_type not in ("3qubit", "4qubit"):
            raise ValueError("Track B FD plate supports only 3qubit or 4qubit")
        self.system_type = system_type
        self.eligibility_gate = eligibility_gate or ProposalEligibilityGateV1()

    def select(
        self, *, posterior: TrackBPosteriorV1, csm: RealCSMSupportV1,
        rsm: RealRSMSupportV1, library: RegisteredActionCandidateLibraryV1,
        direction_conflict: bool = False,
        rf_mapping_core: RegisteredRFMappingCoreV1 | None = None,
    ) -> FDSelectionV1:
        if rf_mapping_core is not None and type(rf_mapping_core) is not RegisteredRFMappingCoreV1:
            raise TypeError(
                "Track B selection accepts only the exact registered RF mapping core"
            )
        if posterior.system_type != self.system_type or library.system_type != self.system_type:
            raise ValueError("FD inputs must use one system-specific plate")
        if posterior.forward_sha != csm.forward_sha:
            raise ValueError("FD selection requires one same-forward posterior and CSM identity")
        if (
            csm.forward_draw_set_sha is None
            or posterior.forward_draw_set_sha != csm.forward_draw_set_sha
            or posterior.forward_draw_ids_sha != csm.forward_draw_ids_sha
        ):
            raise ValueError(
                "FD selection requires one sealed same-forward posterior/CSM draw set"
            )
        if rsm.csm_sha != csm.sha256() or library.rsm_sha != rsm.sha256():
            raise ValueError("FD selection requires exact CSM/RSM/library identity chaining")
        ordered_candidates = library.ranked_candidates()
        ordered = tuple(candidate.candidate_sha for candidate in ordered_candidates)
        no_op = next(candidate for candidate in library.candidates if candidate.source == "NO_OP")
        selected = no_op
        registered_candidate = no_op
        applied_candidate: RegisteredActionCandidateV1 | None = None
        applied_global_gamma = 0.0
        forced_reason: str | None = None
        if direction_conflict:
            forced_reason = "TRACK_A_TRACK_B_DIRECTION_CONFLICT"
        elif not rsm.has_recoverable_support:
            forced_reason = "NO_RECOVERABLE_SUPPORT"
        if forced_reason is None and rf_mapping_core is None:
            forced_reason = "FRESH_RF_PREVIEW_REQUIRED_AFTER_GLOBAL_DAMPING"
        if forced_reason is None:
            rejection_reason = "NO_ACTIONFUL_CANDIDATE_PASSED_PREVIEW_GATE"
            no_op_score = registered_candidate_score(no_op)
            # The library is already ordered by the shared static score.  Find
            # the first candidate that passes the complete preliminary gate;
            # a risky top-ranked candidate must not suppress a safer registered
            # candidate below it.
            for candidate in ordered_candidates:
                if candidate.source == "NO_OP":
                    continue
                if registered_candidate_score(candidate) <= no_op_score + self.eligibility_gate.min_actionful_score_over_no_op:
                    rejection_reason = "ACTIONFUL_SCORE_NOT_ABOVE_NO_OP"
                    continue
                preliminary_harm = candidate.predicted_harm_probability
                preliminary_abstain = max(
                    candidate.abstain_risk,
                    1.0 - csm.confidence,
                    posterior.ood_probability,
                )
                preliminary_stop = max(preliminary_harm, preliminary_abstain)
                preliminary_direction = min(candidate.direction_stability, csm.direction_stability)
                preliminary_rf_confidence = candidate.apparatus_confidence
                prelim_ok, prelim_reason = self.eligibility_gate.evaluate(
                    candidate=candidate,
                    csm=csm,
                    rsm=rsm,
                    harm_probability=preliminary_harm,
                    stop_probability=preliminary_stop,
                    abstain_probability=preliminary_abstain,
                    direction_stability=preliminary_direction,
                    rf_confidence=preliminary_rf_confidence,
                )
                if prelim_ok:
                    uncertainty_scalar = max(float(posterior.uncertainty_scalar), 0.0)
                    global_gamma = min(
                        float(candidate.gamma), 1.0 / (1.0 + uncertainty_scalar),
                    )
                    if rf_mapping_core is None:
                        selected = candidate
                        break
                    try:
                        applied = candidate.re_damped(global_gamma, rf_mapping_core)
                    except (TypeError, ValueError):
                        rejection_reason = "POST_DAMPING_RF_PREVIEW_FAILED"
                        continue
                    final_harm = applied.predicted_harm_probability
                    final_abstain = max(
                        applied.abstain_risk,
                        1.0 - csm.confidence,
                        posterior.ood_probability,
                    )
                    final_ok, final_reason = self.eligibility_gate.evaluate(
                        candidate=applied,
                        csm=csm,
                        rsm=rsm,
                        harm_probability=final_harm,
                        stop_probability=max(final_harm, final_abstain),
                        abstain_probability=final_abstain,
                        direction_stability=min(
                            applied.direction_stability, csm.direction_stability,
                        ),
                        rf_confidence=applied.apparatus_confidence,
                    )
                    if final_ok:
                        selected = applied
                        registered_candidate = candidate
                        applied_candidate = applied
                        applied_global_gamma = global_gamma
                        break
                    rejection_reason = f"POST_DAMPING_GATE:{final_reason}"
                    continue
                rejection_reason = prelim_reason
            else:
                forced_reason = f"PROPOSAL_ELIGIBILITY_FORCED_NO_OP:{rejection_reason}"
        uncertainty_scalar = max(float(posterior.uncertainty_scalar), 0.0)
        if rf_mapping_core is not None and selected.source == "NO_OP" and forced_reason is None:
            forced_reason = f"PROPOSAL_ELIGIBILITY_FORCED_NO_OP:{rejection_reason}"
        global_gamma = (
            applied_global_gamma if applied_candidate is not None else 0.0
        )
        harm = selected.predicted_harm_probability
        abstain = max(selected.abstain_risk, 1.0 - csm.confidence, posterior.ood_probability)
        stop = max(harm, abstain)
        direction = min(selected.direction_stability, csm.direction_stability)
        rf_confidence = selected.apparatus_confidence
        if selected.source == "NO_OP":
            eligible = False
            eligibility_reason = forced_reason or "NO_OP_NO_ACTIONFUL_PROPOSAL"
        elif rf_mapping_core is None:
            eligible = False
            eligibility_reason = "FRESH_RF_PREVIEW_REQUIRED_AFTER_GLOBAL_DAMPING"
        else:
            # The candidate loop above has already performed the post-damping
            # pure preview and final gate.  Reusing that object ensures the
            # returned selection cannot diverge from the action that was checked.
            eligible = applied_candidate is not None
            eligibility_reason = (
                "ACTIONFUL_PROPOSAL_ELIGIBLE" if eligible
                else (forced_reason or "POST_DAMPING_CANDIDATE_NOT_ELIGIBLE")
            )
        return FDSelectionV1(
            selected_candidate_sha=(
                applied_candidate.candidate_sha if applied_candidate is not None else selected.candidate_sha
            ),
            candidate_ranking=ordered,
            global_gamma=float(np_clip(global_gamma)),
            candidate_uncertainty=selected.uncertainty,
            harm_probability=harm,
            stop_probability=stop,
            abstain_probability=abstain,
            direction_stability=direction,
            rf_confidence=rf_confidence,
            reason_code=forced_reason or "REGISTERED_STATIC_FD_CANDIDATE_SELECTED",
            eligibility_policy_sha=self.eligibility_gate.sha256(),
            registered_candidate_sha=registered_candidate.candidate_sha,
            proposal_eligible=eligible,
            eligibility_reason_code=eligibility_reason,
            applied_candidate=applied_candidate,
        )


def np_clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class F_3Q_HW_V10(TrackBFDModelV1):
    """Fixed 3Q research-only FD plate compatibility entry point."""

    def __init__(self) -> None:
        super().__init__("3qubit")


class F_4Q_HW_V10(TrackBFDModelV1):
    """Fixed 4Q research-only FD plate compatibility entry point."""

    def __init__(self) -> None:
        super().__init__("4qubit")


__all__ = [
    "FDSelectionV1", "F_3Q_HW_V10", "F_4Q_HW_V10", "ProposalEligibilityGateV1",
    "TrackBFDModelV1",
]
