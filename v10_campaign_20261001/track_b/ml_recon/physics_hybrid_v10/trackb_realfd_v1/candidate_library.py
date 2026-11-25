"""Registered-only Track B candidate actions and static FD ranking inputs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Sequence

from .coordinate_abi import RFMaterializationPreviewV1, RecoverableLogicalActionV1
from .contracts import canonical_sha256, require_sha256
from .real_rsm import RealRSMSupportV1
from .rf_mapping_core import RegisteredRFMappingCoreV1
from .rf_preview import preview_registered_rf_action


CANDIDATE_SOURCES = (
    "NO_OP", "TRACK_A_IDEAL", "V24_ANALYTIC", "TRACKB_LOCAL_ANALYTIC",
    "LOW_RANK_PARALLEL_RESIDUAL",
)
GAMMA_GRID = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)


def registered_candidate_score(candidate: "RegisteredActionCandidateV1") -> float:
    """One registered, noncausal score shared by FD and envelope selection.

    Candidate-specific uncertainty is intentionally part of the score. Global
    posterior uncertainty instead affects damping/eligibility, because a common
    term cannot rank two candidates differently.
    """
    return float(
        candidate.counterfactual_static_utility_lcb95
        - candidate.predicted_harm_probability
        - candidate.abstain_risk
        - candidate.uncertainty
    )


@dataclass(frozen=True)
class RegisteredActionCandidateV1:
    """A fully registered candidate; real-static FD has no free action head."""

    source: str
    gamma: float
    recoverable_action: RecoverableLogicalActionV1
    rsm_sha: str
    rf_preview: RFMaterializationPreviewV1
    uncertainty: float
    counterfactual_static_utility_mean: float
    counterfactual_static_utility_lcb95: float
    utility_quantiles: tuple[float, float, float]
    predicted_harm_probability: float
    direction_stability: float
    apparatus_confidence: float
    abstain_risk: float
    base_candidate_sha: str | None = None
    applied_global_gamma: float | None = None
    schema: str = field(default="RegisteredActionCandidateV1", init=False)

    def __post_init__(self) -> None:
        if self.source not in CANDIDATE_SOURCES:
            raise ValueError("candidate source is not registered")
        if self.source == "NO_OP":
            if float(self.gamma) != 0.0:
                raise ValueError("NO_OP candidate gamma must be exactly zero")
            if any(abs(value) > 1.0e-12 for value in self.recoverable_action.grouped_coordinates):
                raise ValueError("NO_OP candidate action must be exactly zero")
        elif self.gamma not in GAMMA_GRID:
            raise ValueError("actionful candidate gamma must be selected from the registered positive grid")
        if self.recoverable_action.ideal_action.analytic_action_application_count != 1:
            raise ValueError("candidate action has not bound exactly one ideal action application")
        if self.rsm_sha != self.recoverable_action.support_sha:
            raise ValueError("candidate recoverable action must bind the exact RSM/CSM support SHA")
        preview_delta = self.rf_preview.requested_rf_delta
        if (
            preview_delta.system_type != self.recoverable_action.ideal_action.system_type
            or preview_delta.source_logical_action_sha
            != self.recoverable_action.sha256()
        ):
            raise ValueError(
                "candidate RF preview must bind the exact recoverable logical action"
            )
        metrics = (
            self.uncertainty, self.counterfactual_static_utility_mean,
            self.counterfactual_static_utility_lcb95, *self.utility_quantiles,
            self.predicted_harm_probability, self.direction_stability,
            self.apparatus_confidence, self.abstain_risk,
        )
        if not all(math.isfinite(float(value)) for value in metrics):
            raise ValueError("candidate diagnostics must be finite")
        if tuple(sorted(self.utility_quantiles)) != self.utility_quantiles:
            raise ValueError("candidate utility quantiles must be ordered")
        if not all(0.0 <= float(value) <= 1.0 for value in (
            self.uncertainty, self.predicted_harm_probability, self.direction_stability,
            self.apparatus_confidence, self.abstain_risk,
        )):
            raise ValueError("candidate uncertainty/probabilities/stability must lie in [0,1]")
        if self.base_candidate_sha is not None:
            require_sha256(self.base_candidate_sha, "base_candidate_sha")
            if self.applied_global_gamma is None:
                raise ValueError("re-damped candidate requires an applied global gamma")
        if self.applied_global_gamma is not None:
            if self.base_candidate_sha is None:
                raise ValueError("applied global gamma requires a registered base candidate")
            if (
                not math.isfinite(float(self.applied_global_gamma))
                or not 0.0 <= float(self.applied_global_gamma) <= float(self.gamma)
            ):
                raise ValueError("applied global gamma must be finite in [0, registered gamma]")
        if (
            not self.rf_preview.preview_only
            or self.rf_preview.hardware_action_authorized
            or not self.rf_preview.accepted_for_external_review
            or not self.rf_preview.bounds_valid
            or not self.rf_preview.phase_wrap_valid
        ):
            raise ValueError(
                "candidate requires an accepted, bounds-valid, phase-valid "
                "external-review-only RF preview"
            )

    @property
    def candidate_sha(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "source": self.source,
            "gamma": self.gamma,
            "recoverable_action_sha": self.recoverable_action.sha256(),
            "rsm_sha": self.rsm_sha,
            "rf_preview_sha": self.rf_preview.sha256(),
            "uncertainty": self.uncertainty,
            "counterfactual_static_utility_mean": self.counterfactual_static_utility_mean,
            "counterfactual_static_utility_lcb95": self.counterfactual_static_utility_lcb95,
            "utility_quantiles": list(self.utility_quantiles),
            "predicted_harm_probability": self.predicted_harm_probability,
            "direction_stability": self.direction_stability,
            "apparatus_confidence": self.apparatus_confidence,
            "abstain_risk": self.abstain_risk,
            "base_candidate_sha": self.base_candidate_sha,
            "applied_global_gamma": self.applied_global_gamma,
        })

    def re_damped(
        self,
        global_gamma: float,
        rf_mapping_core: RegisteredRFMappingCoreV1,
    ) -> "RegisteredActionCandidateV1":
        """Apply global damping and require a fresh pure RF preview.

        The registered grid gamma remains the library identity. The derived
        candidate binds the post-damping action/preview and points back to that
        registered base, so stale preview bytes cannot reach serialization.
        """
        applied = float(global_gamma)
        if self.source == "NO_OP":
            if applied != 0.0:
                raise ValueError("NO_OP cannot receive a nonzero global gamma")
            return self
        if not math.isfinite(applied) or not 0.0 <= applied <= self.gamma:
            raise ValueError("global gamma must lie in [0, registered gamma]")
        if type(rf_mapping_core) is not RegisteredRFMappingCoreV1:
            raise TypeError("Track B re-damping requires the exact registered RF mapping core")
        action = self.recoverable_action.scaled(applied / self.gamma)
        preview = preview_registered_rf_action(rf_mapping_core, action)
        if not isinstance(preview, RFMaterializationPreviewV1):
            raise TypeError("registered RF mapping core must produce RFMaterializationPreviewV1")
        if preview.requested_rf_delta.source_logical_action_sha != action.sha256():
            raise ValueError("recomputed RF preview does not bind the re-damped action")
        return replace(
            self,
            recoverable_action=action,
            rf_preview=preview,
            base_candidate_sha=self.candidate_sha,
            applied_global_gamma=applied,
        )


@dataclass(frozen=True)
class RegisteredActionCandidateLibraryV1:
    """Closed candidate set; selected actions can only come from this library."""

    system_type: str
    candidates: tuple[RegisteredActionCandidateV1, ...]
    rsm_sha: str
    schema: str = field(default="RegisteredActionCandidateLibraryV1", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("candidate library system type is unsupported")
        if not self.candidates:
            raise ValueError("candidate library must contain the registered no-op candidate")
        if not any(candidate.source == "NO_OP" for candidate in self.candidates):
            raise ValueError("candidate library must always contain NO_OP")
        if len({candidate.candidate_sha for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate SHA identities must be unique")
        if any(candidate.recoverable_action.ideal_action.system_type != self.system_type for candidate in self.candidates):
            raise ValueError("candidate system type differs from library system type")
        if any(candidate.rsm_sha != self.rsm_sha for candidate in self.candidates):
            raise ValueError("all candidates must bind the same registered RSM support")
        if len({
            candidate.rf_preview.requested_rf_delta.rf_coordinate_layout_sha
            for candidate in self.candidates
        }) != 1:
            raise ValueError("candidate library cannot mix RF coordinate layouts")

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "system_type": self.system_type,
            "rsm_sha": self.rsm_sha,
            "candidate_shas": [candidate.candidate_sha for candidate in self.candidates],
        })

    def select_by_sha(self, candidate_sha: str) -> RegisteredActionCandidateV1:
        for candidate in self.candidates:
            if candidate.candidate_sha == candidate_sha:
                return candidate
        raise ValueError("candidate SHA is not registered in this library")

    def ranked_candidates(self) -> tuple[RegisteredActionCandidateV1, ...]:
        """Return the closed library under the shared registered score."""
        return tuple(sorted(
            self.candidates,
            key=lambda candidate: (-registered_candidate_score(candidate), candidate.candidate_sha),
        ))


def build_parallel_residual_candidate(
    *, source: str, gamma: float, base_action: RecoverableLogicalActionV1,
    rsm: RealRSMSupportV1, rf_preview: RFMaterializationPreviewV1,
    uncertainty: float, utility_mean: float, utility_lcb95: float, utility_quantiles: Sequence[float],
    harm_probability: float, direction_stability: float, apparatus_confidence: float, abstain_risk: float,
) -> RegisteredActionCandidateV1:
    """Build only a support-parallel candidate; callers cannot insert free directions."""
    effective_gamma = 0.0 if source == "NO_OP" else float(gamma)
    if source != "NO_OP" and effective_gamma not in GAMMA_GRID:
        raise ValueError("actionful candidate gamma must be selected from the registered positive grid")
    scaled = base_action.scaled(effective_gamma)
    if scaled.support_sha != rsm.sha256():
        raise ValueError("base action must bind the exact registered RSM support")
    if scaled.reason_code != rsm.reason_code:
        raise ValueError("base action reason must bind the exact registered RSM result")
    quantiles = tuple(float(value) for value in utility_quantiles)
    if len(quantiles) != 3:
        raise ValueError("candidate utility quantiles require p05/p50/p95")
    return RegisteredActionCandidateV1(
        source=source,
        gamma=effective_gamma,
        recoverable_action=scaled,
        rsm_sha=rsm.sha256(),
        rf_preview=rf_preview,
        uncertainty=float(uncertainty),
        counterfactual_static_utility_mean=float(utility_mean),
        counterfactual_static_utility_lcb95=float(utility_lcb95),
        utility_quantiles=(quantiles[0], quantiles[1], quantiles[2]),
        predicted_harm_probability=float(harm_probability),
        direction_stability=float(direction_stability),
        apparatus_confidence=float(apparatus_confidence),
        abstain_risk=float(abstain_risk),
    )


__all__ = [
    "CANDIDATE_SOURCES", "GAMMA_GRID", "RegisteredActionCandidateLibraryV1",
    "RegisteredActionCandidateV1", "build_parallel_residual_candidate", "registered_candidate_score",
]
