"""Registered tryable recovery envelope, explicitly not a physical maximum claim."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import inspect
import math

import numpy as np

from ..csm_contract import SingularSupportContract, WhiteningThresholdContract
from ..csm_covariance import support_whitening
from .candidate_library import (
    RegisteredActionCandidateLibraryV1,
    RegisteredActionCandidateV1,
    registered_candidate_score,
)
from .contracts import canonical_sha256, require_sha256
from .posterior import TrackBPosteriorV1
from .real_csm import RealCSMSupportV1, array_identity_sha256
from .real_rsm import RealRSMSupportV1
from .rf_mapping_core import RegisteredRFMappingCoreV1
from .rf_preview import preview_registered_rf_action
from .forward_draw_set import BoundTrackBForwardDrawSetV1


REGISTERED_TRYABLE_SCOPE = "REGISTERED_CANDIDATE_SET_AND_BOUND_RF_MAPPING_ONLY"
ENVELOPE_ESTIMATION_STATUSES = frozenset({
    "CONTRACT_SCAFFOLD_NOT_DATA_EVALUATED",
    "POSTERIOR_DRAW_COUNTERFACTUAL",
})


def _mean(values: tuple[float, ...]) -> float:
    return float(sum(values) / len(values))


def _quantiles(values: tuple[float, ...]) -> tuple[float, float, float]:
    ordered = tuple(sorted(values))
    def value(fraction: float) -> float:
        index = int(round((len(ordered) - 1) * fraction))
        return float(ordered[index])
    return value(0.05), value(0.50), value(0.95)


@dataclass(frozen=True)
class PosteriorCandidateEvaluationV1:
    """Counterfactual outcomes for one candidate over registered posterior draws.

    Values are supplied by a same-forward measurement/RF evaluator.  The class
    is deliberately evidence-shaped: it cannot state causal gain or a physical
    global maximum, but it makes the finite registered-candidate estimate
    reproducible and prevents hard-coded zero gap fields.
    """

    candidate_sha: str
    draw_ids: tuple[str, ...]
    utility_samples: tuple[float, ...]
    harm_samples: tuple[float, ...]
    recoverable_fraction_samples: tuple[float, ...]
    ideal_to_rf_realizability_samples: tuple[float, ...]
    observation_noise_gap_samples: tuple[float, ...]
    basis_gap_samples: tuple[float, ...]
    nuisance_gap_samples: tuple[float, ...]
    sector_ambiguity_gap_samples: tuple[float, ...]
    rf_reachability_gap_samples: tuple[float, ...]
    rf_bounds_gap_samples: tuple[float, ...]
    materialization_gap_samples: tuple[float, ...]
    uncertainty_gap_samples: tuple[float, ...]
    forward_sha: str
    forward_draw_set_sha: str
    posterior_sha: str
    grouped_bootstrap_sha: str
    uncertainty_samples: tuple[float, ...]
    abstain_samples: tuple[float, ...]
    direction_stability_samples: tuple[float, ...]
    rf_confidence_samples: tuple[float, ...]
    evaluation_recipe_sha: str
    observation_support_shas: tuple[str, ...] = ()
    schema: str = field(default="PosteriorCandidateEvaluationV1", init=False)

    def __post_init__(self) -> None:
        for name in (
            "candidate_sha", "forward_sha", "forward_draw_set_sha", "posterior_sha",
            "grouped_bootstrap_sha", "evaluation_recipe_sha",
        ):
            require_sha256(getattr(self, name), name)
        if not self.draw_ids or len(set(self.draw_ids)) != len(self.draw_ids) or any(
            type(value) is not str or not value for value in self.draw_ids
        ):
            raise ValueError("posterior candidate evaluation requires unique draw identities")
        arrays = (
            self.utility_samples, self.harm_samples, self.recoverable_fraction_samples,
            self.ideal_to_rf_realizability_samples, self.observation_noise_gap_samples,
            self.basis_gap_samples, self.nuisance_gap_samples,
            self.sector_ambiguity_gap_samples, self.rf_reachability_gap_samples,
            self.rf_bounds_gap_samples, self.materialization_gap_samples,
            self.uncertainty_gap_samples,
            self.uncertainty_samples, self.abstain_samples,
            self.direction_stability_samples, self.rf_confidence_samples,
        )
        if any(len(values) != len(self.draw_ids) for values in arrays):
            raise ValueError("every posterior candidate outcome must align to draw IDs")
        if any(not math.isfinite(float(value)) for values in arrays for value in values):
            raise ValueError("posterior candidate outcomes must be finite")
        bounded = arrays[1:]
        if any(not 0.0 <= float(value) <= 1.0 for values in bounded for value in values):
            raise ValueError("harm, fractions, and gap outcomes must lie in [0,1]")
        if self.observation_support_shas and (
            len(self.observation_support_shas) != len(self.draw_ids)
            or any(type(value) is not str for value in self.observation_support_shas)
        ):
            raise ValueError("observation support receipts must align to draw IDs")
        for value in self.observation_support_shas:
            require_sha256(value, "observation_support_sha")

    @property
    def draw_ids_sha(self) -> str:
        return canonical_sha256({"schema": "TrackBForwardDrawIDsV1", "draw_ids": list(self.draw_ids)})

    @property
    def uncertainty_mean(self) -> float:
        return _mean(self.uncertainty_samples)

    @property
    def abstain_mean(self) -> float:
        return _mean(self.abstain_samples)

    @property
    def direction_stability_mean(self) -> float:
        return _mean(self.direction_stability_samples)

    @property
    def rf_confidence_mean(self) -> float:
        return _mean(self.rf_confidence_samples)

    @property
    def utility_quantiles(self) -> tuple[float, float, float]:
        return _quantiles(self.utility_samples)

    @property
    def utility_mean(self) -> float:
        return _mean(self.utility_samples)

    @property
    def utility_lcb95(self) -> float:
        values = self.utility_samples
        if len(values) == 1:
            return float(values[0])
        mean = _mean(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return float(mean - 1.96 * math.sqrt(variance / len(values)))

    @property
    def harm_probability(self) -> float:
        return _mean(self.harm_samples)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "candidate_sha": self.candidate_sha,
            "draw_ids": list(self.draw_ids),
            "draw_ids_sha": self.draw_ids_sha,
            "utility_samples": list(self.utility_samples),
            "harm_samples": list(self.harm_samples),
            "recoverable_fraction_samples": list(self.recoverable_fraction_samples),
            "ideal_to_rf_realizability_samples": list(self.ideal_to_rf_realizability_samples),
            "observation_noise_gap_samples": list(self.observation_noise_gap_samples),
            "basis_gap_samples": list(self.basis_gap_samples),
            "nuisance_gap_samples": list(self.nuisance_gap_samples),
            "sector_ambiguity_gap_samples": list(self.sector_ambiguity_gap_samples),
            "rf_reachability_gap_samples": list(self.rf_reachability_gap_samples),
            "rf_bounds_gap_samples": list(self.rf_bounds_gap_samples),
            "materialization_gap_samples": list(self.materialization_gap_samples),
            "uncertainty_gap_samples": list(self.uncertainty_gap_samples),
            "forward_sha": self.forward_sha,
            "forward_draw_set_sha": self.forward_draw_set_sha,
            "posterior_sha": self.posterior_sha,
            "grouped_bootstrap_sha": self.grouped_bootstrap_sha,
            "uncertainty_samples": list(self.uncertainty_samples),
            "abstain_samples": list(self.abstain_samples),
            "direction_stability_samples": list(self.direction_stability_samples),
            "rf_confidence_samples": list(self.rf_confidence_samples),
            "evaluation_recipe_sha": self.evaluation_recipe_sha,
            "observation_support_shas": list(self.observation_support_shas),
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TrackBCandidateEvaluationRejectionV1:
    """Typed non-actionful outcome when an exact-core preview is rejected."""

    candidate_sha: str
    reason_code: str
    forward_draw_set_sha: str
    rf_mapping_core_sha: str
    schema: str = field(default="TrackBCandidateEvaluationRejectionV1", init=False)

    def __post_init__(self) -> None:
        for name in ("candidate_sha", "forward_draw_set_sha", "rf_mapping_core_sha"):
            require_sha256(getattr(self, name), name)
        if self.reason_code not in {"EXACT_CORE_PREVIEW_REJECTED"}:
            raise ValueError("candidate evaluation rejection reason is not registered")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "candidate_sha": self.candidate_sha,
            "reason_code": self.reason_code,
            "forward_draw_set_sha": self.forward_draw_set_sha,
            "rf_mapping_core_sha": self.rf_mapping_core_sha,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class BoundTrackBAppliedCandidateEvaluationV1:
    """Fresh draw-wise evaluation of the exact post-damping proposal action."""

    registered_candidate_sha: str
    applied_candidate_sha: str
    applied_global_gamma: float
    evaluation: PosteriorCandidateEvaluationV1
    schema: str = field(default="BoundTrackBAppliedCandidateEvaluationV1", init=False)

    def __post_init__(self) -> None:
        require_sha256(self.registered_candidate_sha, "registered_candidate_sha")
        require_sha256(self.applied_candidate_sha, "applied_candidate_sha")
        if self.evaluation.candidate_sha != self.applied_candidate_sha:
            raise ValueError("applied candidate evaluation must bind the exact applied candidate")
        if not math.isfinite(float(self.applied_global_gamma)) or not 0.0 < float(self.applied_global_gamma) <= 1.0:
            raise ValueError("applied candidate evaluation requires a positive bounded damping value")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registered_candidate_sha": self.registered_candidate_sha,
            "applied_candidate_sha": self.applied_candidate_sha,
            "applied_global_gamma": self.applied_global_gamma,
            "evaluation_sha": self.evaluation.sha256(),
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _lcb95(values: tuple[float, ...]) -> float:
    if len(values) == 1:
        return float(values[0])
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(mean - 1.96 * math.sqrt(variance / len(values)))


def _csm_supported_quadratic_form(
    *, csm: RealCSMSupportV1, covariance: np.ndarray,
    nuisance_jacobian: np.ndarray, nuisance_covariance: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Build the evaluator quadratic form from the sealed CSM contracts.

    This is deliberately not a ``pinv`` convenience path.  The same
    whitening and singular-support contracts that established CSM geometry
    define the observation support, then the registered nuisance subspace is
    projected from that support.  The returned receipt commits the effective
    covariance, whitener, nuisance projector and final quadratic operator.
    """
    whitening = csm.whitening_contract
    singular_support = csm.singular_support_contract
    if type(whitening) is not WhiteningThresholdContract \
            or type(singular_support) is not SingularSupportContract:
        raise ValueError("posterior evaluator requires sealed CSM support contracts")
    if whitening.sha256() != csm.whitening_contract_sha \
            or singular_support.sha256() != csm.singular_support_contract_sha:
        raise ValueError("posterior evaluator CSM support contract identities changed")
    effective_covariance = covariance + nuisance_jacobian @ nuisance_covariance @ nuisance_jacobian.T
    effective_covariance = 0.5 * (effective_covariance + effective_covariance.T)
    if not np.all(np.isfinite(effective_covariance)):
        raise ValueError("posterior evaluator effective covariance must be finite")
    whitener = support_whitening(effective_covariance, threshold_contract=whitening)
    whitened_nuisance = whitener.whiten_jacobian(nuisance_jacobian)
    if whitened_nuisance.shape[1]:
        left, singular_values, _right = np.linalg.svd(whitened_nuisance, full_matrices=False)
        threshold = singular_support.effective_threshold(float(singular_values.max(initial=0.0)))
        active = singular_values > threshold
        nuisance_basis = left[:, active]
    else:
        nuisance_basis = np.zeros((whitener.rank, 0), dtype=np.float64)
    projection = np.eye(whitener.rank, dtype=np.float64)
    if nuisance_basis.shape[1]:
        projection -= nuisance_basis @ nuisance_basis.T
    transform = projection @ whitener.matrix
    precision = transform.T @ transform
    support_receipt_sha = canonical_sha256({
        "schema": "TrackBCSMObservationSupportReceiptV1",
        "whitening_contract_sha": csm.whitening_contract_sha,
        "singular_support_contract_sha": csm.singular_support_contract_sha,
        "effective_covariance_sha": array_identity_sha256(effective_covariance),
        "whitener": whitener.to_dict(),
        "nuisance_projection_sha": array_identity_sha256(projection),
        "effective_support_rank": int(whitener.rank - nuisance_basis.shape[1]),
        "supported_precision_sha": array_identity_sha256(precision),
    })
    return transform, support_receipt_sha


def _evaluate_trackb_candidate_draw(
    candidate: RegisteredActionCandidateV1,
    posterior: TrackBPosteriorV1,
    jacobian: object,
    covariance: object,
    nuisance_jacobian: object,
    baseline_output: object,
    preview: object,
    csm: RealCSMSupportV1,
    rsm: RealRSMSupportV1,
) -> dict[str, float | str]:
    """Evaluate one logical candidate in its matching observation space.

    ``candidate`` and the RSM target both live in the logical action chart.
    They are compared only after the same draw Jacobian maps their difference
    into observation space.  The nuisance covariance is propagated through the
    draw's nuisance Jacobian and the resulting covariance support is used for
    every quadratic norm.
    """
    jacobian_array = np.asarray(jacobian, dtype=float)
    covariance_array = np.asarray(covariance, dtype=float)
    nuisance_array = np.asarray(nuisance_jacobian, dtype=float)
    baseline_array = np.asarray(baseline_output, dtype=float).reshape(-1)
    action = np.asarray(candidate.recoverable_action.grouped_coordinates, dtype=float)
    target = np.asarray(rsm.supported_target_direction, dtype=float)
    if jacobian_array.ndim != 2 or jacobian_array.shape[1] != action.size \
            or target.shape != action.shape or baseline_array.shape != (jacobian_array.shape[0],):
        raise ValueError("posterior evaluator logical/observation dimensions do not align")
    nuisance_covariance = np.asarray(posterior.nuisance_covariance, dtype=float)
    if nuisance_array.shape != (jacobian_array.shape[0], nuisance_covariance.shape[0]):
        raise ValueError("posterior evaluator nuisance Jacobian/covariance dimensions do not align")
    transform, support_receipt_sha = _csm_supported_quadratic_form(
        csm=csm,
        covariance=covariance_array,
        nuisance_jacobian=nuisance_array,
        nuisance_covariance=nuisance_covariance,
    )
    effective_covariance = covariance_array + nuisance_array @ nuisance_covariance @ nuisance_array.T
    effective_covariance = 0.5 * (effective_covariance + effective_covariance.T)

    def covariance_energy(vector: np.ndarray) -> float:
        return float(max(np.linalg.norm(transform @ vector) ** 2, 0.0))

    baseline_observation_delta = jacobian_array @ (-target)
    action_observation_delta = jacobian_array @ (action - target)
    baseline_energy = covariance_energy(baseline_observation_delta)
    action_energy = covariance_energy(action_observation_delta)
    information = float(max(np.linalg.norm(transform @ jacobian_array) ** 2, 0.0))
    uncertainty = float(np.clip(1.0 / (1.0 + math.sqrt(information)), 0.0, 1.0))
    harm = float(np.clip(
        preview.support_leakage_norm + preview.requested_to_materialized_logical_error
        + max(action_energy - baseline_energy, 0.0) / max(1.0 + baseline_energy, 1.0e-12)
        + 0.5 * posterior.ood_probability,
        0.0, 1.0,
    ))
    abstain = float(np.clip(max(
        posterior.ood_probability, uncertainty, 1.0 - posterior.apparatus_confidence,
    ), 0.0, 1.0))
    confidence = float(np.clip(
        posterior.apparatus_confidence * (1.0 - harm), 0.0, 1.0,
    ))
    direction = float(np.clip(preview.direction_cosine, -1.0, 1.0))
    logical_error = float(np.clip(
        np.linalg.norm(action - target) / max(1.0 + np.linalg.norm(target), 1.0e-12), 0.0, 1.0,
    ))
    error = float(np.clip(action_energy / max(1.0 + baseline_energy, 1.0e-12), 0.0, 1.0))
    utility = float(np.clip(
        (baseline_energy - action_energy) / max(1.0 + baseline_energy, 1.0e-12) * (1.0 - harm),
        -1.0, 1.0,
    ))
    return {
        "utility": utility,
        "harm": harm,
        "recoverable": float(np.clip(1.0 - error, 0.0, 1.0)),
        "realizability": float(np.clip(1.0 - preview.requested_to_materialized_logical_error, 0.0, 1.0)),
        "noise_gap": float(np.clip(1.0 / (1.0 + information), 0.0, 1.0)),
        "basis_gap": logical_error,
        "nuisance_gap": float(np.clip(
            np.trace(nuisance_array @ nuisance_covariance @ nuisance_array.T)
            / max(1.0 + np.trace(effective_covariance), 1.0e-12), 0.0, 1.0,
        )),
        "sector_gap": 0.0,
        "rf_reachability_gap": float(np.clip(1.0 - direction, 0.0, 1.0)),
        "rf_bounds_gap": 0.0 if preview.bounds_valid else 1.0,
        "materialization_gap": float(np.clip(preview.requested_to_materialized_logical_error, 0.0, 1.0)),
        "uncertainty_gap": uncertainty,
        "uncertainty": uncertainty,
        "abstain": abstain,
        "direction": float(np.clip(direction, 0.0, 1.0)),
        "confidence": confidence,
        "support_receipt_sha": support_receipt_sha,
    }


def evaluate_bound_trackb_candidate(
    *, candidate: RegisteredActionCandidateV1,
    forward_draw_set: BoundTrackBForwardDrawSetV1,
    posterior: TrackBPosteriorV1,
    csm: RealCSMSupportV1,
    rsm: RealRSMSupportV1,
    rf_mapping_core: RegisteredRFMappingCoreV1,
    grouped_bootstrap_sha: str,
    evaluation_recipe_sha: str,
) -> tuple[RegisteredActionCandidateV1, PosteriorCandidateEvaluationV1]:
    """Replay every draw for one candidate and derive its typed diagnostics."""
    candidate.recoverable_action.verify_projection(rsm=rsm, csm=csm)
    preview = preview_registered_rf_action(rf_mapping_core, candidate.recoverable_action)
    if not preview.accepted_for_external_review:
        raise ValueError("candidate exact-core RF preview is not accepted")
    metrics = tuple(
        _evaluate_trackb_candidate_draw(
            candidate, posterior, jacobian, covariance, nuisance_jacobian, baseline_output,
            preview, csm, rsm,
        )
        for jacobian, covariance, nuisance_jacobian, baseline_output in zip(
            forward_draw_set.jacobian_draws,
            forward_draw_set.covariance_draws,
            forward_draw_set.nuisance_jacobian_draws,
            forward_draw_set.baseline_outputs,
            strict=True,
        )
    )

    def values(name: str) -> tuple[float, ...]:
        return tuple(float(item[name]) for item in metrics)

    utility = values("utility")
    derived = replace(
        candidate,
        rf_preview=preview,
        uncertainty=_mean(values("uncertainty")),
        counterfactual_static_utility_mean=_mean(utility),
        counterfactual_static_utility_lcb95=_lcb95(utility),
        utility_quantiles=_quantiles(utility),
        predicted_harm_probability=_mean(values("harm")),
        direction_stability=_mean(values("direction")),
        apparatus_confidence=_mean(values("confidence")),
        abstain_risk=_mean(values("abstain")),
    )
    evaluation = PosteriorCandidateEvaluationV1(
        candidate_sha=derived.candidate_sha,
        draw_ids=forward_draw_set.draw_ids,
        utility_samples=values("utility"),
        harm_samples=values("harm"),
        recoverable_fraction_samples=values("recoverable"),
        ideal_to_rf_realizability_samples=values("realizability"),
        observation_noise_gap_samples=values("noise_gap"),
        basis_gap_samples=values("basis_gap"),
        nuisance_gap_samples=values("nuisance_gap"),
        sector_ambiguity_gap_samples=values("sector_gap"),
        rf_reachability_gap_samples=values("rf_reachability_gap"),
        rf_bounds_gap_samples=values("rf_bounds_gap"),
        materialization_gap_samples=values("materialization_gap"),
        uncertainty_gap_samples=values("uncertainty_gap"),
        forward_sha=forward_draw_set.forward_sha,
        forward_draw_set_sha=forward_draw_set.sha256(),
        posterior_sha=posterior.sha256(),
        grouped_bootstrap_sha=grouped_bootstrap_sha,
        uncertainty_samples=values("uncertainty"),
        abstain_samples=values("abstain"),
        direction_stability_samples=values("direction"),
        rf_confidence_samples=values("confidence"),
        evaluation_recipe_sha=evaluation_recipe_sha,
        observation_support_shas=tuple(str(item["support_receipt_sha"]) for item in metrics),
    )
    return derived, evaluation


@dataclass(frozen=True)
class BoundTrackBPosteriorEvaluationSetV1:
    """Authoritative evaluator output over one sealed draw set and candidate library."""

    forward_draw_set_sha: str
    posterior_sha: str
    csm_sha: str
    rsm_sha: str
    rf_mapping_core_sha: str
    source_candidate_library_sha: str
    candidate_library: RegisteredActionCandidateLibraryV1
    evaluations: tuple[PosteriorCandidateEvaluationV1, ...]
    evaluator_source_sha: str
    evaluator_config_sha: str
    forward_draw_ids_sha: str
    grouped_bootstrap_plan_sha: str
    rejected_candidates: tuple[TrackBCandidateEvaluationRejectionV1, ...] = ()
    producer_status: str = "UNVERIFIED_EVALUATION_SET"
    _replay_forward_draw_set: BoundTrackBForwardDrawSetV1 | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _replay_posterior: TrackBPosteriorV1 | None = field(default=None, init=False, repr=False, compare=False)
    _replay_csm: RealCSMSupportV1 | None = field(default=None, init=False, repr=False, compare=False)
    _replay_rsm: RealRSMSupportV1 | None = field(default=None, init=False, repr=False, compare=False)
    _replay_candidate_library: RegisteredActionCandidateLibraryV1 | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _replay_rf_mapping_core: RegisteredRFMappingCoreV1 | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    schema: str = field(default="BoundTrackBPosteriorEvaluationSetV1", init=False)

    def __post_init__(self) -> None:
        require_sha256(self.forward_draw_set_sha, "forward_draw_set_sha")
        require_sha256(self.posterior_sha, "posterior_sha")
        require_sha256(self.csm_sha, "csm_sha")
        require_sha256(self.rsm_sha, "rsm_sha")
        require_sha256(self.rf_mapping_core_sha, "rf_mapping_core_sha")
        require_sha256(self.source_candidate_library_sha, "source_candidate_library_sha")
        require_sha256(self.evaluator_source_sha, "evaluator_source_sha")
        require_sha256(self.evaluator_config_sha, "evaluator_config_sha")
        require_sha256(self.forward_draw_ids_sha, "forward_draw_ids_sha")
        require_sha256(self.grouped_bootstrap_plan_sha, "grouped_bootstrap_plan_sha")
        if self.producer_status not in {"UNVERIFIED_EVALUATION_SET", "REPLAYED_BOUND_FORWARD_DRAWS"}:
            raise ValueError("posterior evaluation producer status is not registered")
        if self.producer_status == "REPLAYED_BOUND_FORWARD_DRAWS" and any(value is None for value in (
            self._replay_forward_draw_set, self._replay_posterior, self._replay_csm,
            self._replay_rsm, self._replay_candidate_library, self._replay_rf_mapping_core,
        )):
            raise ValueError("replayed posterior evaluation set requires attached live inputs")
        if not self.evaluations:
            raise ValueError("authoritative posterior evaluation set cannot be empty")
        if len(self.evaluations) != len(self.candidate_library.candidates):
            raise ValueError("authoritative posterior evaluations must cover the candidate library")
        if {item.candidate_sha for item in self.evaluations} != {
            item.candidate_sha for item in self.candidate_library.candidates
        }:
            raise ValueError("authoritative posterior evaluation candidate identities do not match the library")
        if any(item.forward_draw_set_sha != self.forward_draw_set_sha for item in self.evaluations):
            raise ValueError("authoritative posterior evaluations disagree on forward draw-set identity")
        if any(item.draw_ids_sha != self.forward_draw_ids_sha for item in self.evaluations):
            raise ValueError("authoritative posterior evaluations disagree on draw-ID identity")
        if any(item.grouped_bootstrap_sha != self.grouped_bootstrap_plan_sha for item in self.evaluations):
            raise ValueError("authoritative posterior evaluations disagree on bootstrap plan identity")
        if self.producer_status == "REPLAYED_BOUND_FORWARD_DRAWS" and any(
            len(item.observation_support_shas) != len(item.draw_ids)
            for item in self.evaluations
        ):
            raise ValueError("replayed posterior evaluations require one CSM support receipt per draw")
        evaluated_shas = {item.candidate_sha for item in self.candidate_library.candidates}
        if any(item.candidate_sha in evaluated_shas for item in self.rejected_candidates):
            raise ValueError("candidate cannot be both evaluated and exact-core rejected")
        if any(
            item.forward_draw_set_sha != self.forward_draw_set_sha
            or item.rf_mapping_core_sha != self.rf_mapping_core_sha
            for item in self.rejected_candidates
        ):
            raise ValueError("candidate rejection does not bind the evaluator draw/core identities")

    @classmethod
    def from_bound_objects(
        cls,
        *,
        forward_draw_set: BoundTrackBForwardDrawSetV1,
        posterior: TrackBPosteriorV1,
        csm: RealCSMSupportV1,
        rsm: RealRSMSupportV1,
        candidate_library: RegisteredActionCandidateLibraryV1,
        rf_mapping_core: RegisteredRFMappingCoreV1,
    ) -> "BoundTrackBPosteriorEvaluationSetV1":
        if type(forward_draw_set) is not BoundTrackBForwardDrawSetV1:
            raise TypeError("posterior evaluator requires an authoritative forward draw set")
        forward_draw_set.assert_authoritative()
        if type(posterior) is not TrackBPosteriorV1 or type(csm) is not RealCSMSupportV1 \
                or type(rsm) is not RealRSMSupportV1 or type(candidate_library) is not RegisteredActionCandidateLibraryV1 \
                or type(rf_mapping_core) is not RegisteredRFMappingCoreV1:
            raise TypeError("posterior evaluator requires exact registered Track B objects")
        if csm.forward_draw_set_sha != forward_draw_set.sha256() \
                or posterior.forward_draw_set_sha != forward_draw_set.sha256():
            raise ValueError("posterior evaluator objects do not share the sealed forward draw set")
        if not csm.input_identity_verified:
            raise ValueError("posterior evaluator requires a verified CSM support identity")
        if rsm.csm_sha != csm.sha256() or candidate_library.rsm_sha != rsm.sha256():
            raise ValueError("posterior evaluator objects do not share CSM/RSM/library identity")
        draw_ids = forward_draw_set.draw_ids
        draw_ids_sha = canonical_sha256({
            "schema": "TrackBForwardDrawIDsV1", "draw_ids": list(draw_ids),
        })
        grouped_bootstrap_sha = canonical_sha256({
            "schema": "TrackBGroupedBootstrapPlanV1", "draw_ids": list(draw_ids),
            "grouping": "one posterior draw is one resampling group",
        })
        recipe_sha = canonical_sha256({
            "schema": "TrackBPosteriorCounterfactualEvaluatorRecipeV1",
            "source": inspect.getsource(_evaluate_trackb_candidate_draw),
            "formula_version": 2,
            "whitening_contract_sha": csm.whitening_contract_sha,
            "singular_support_contract_sha": csm.singular_support_contract_sha,
        })
        evaluator_source_sha = canonical_sha256(inspect.getsource(_evaluate_trackb_candidate_draw))
        evaluated_items = []
        rejected_candidates = []
        for candidate in candidate_library.candidates:
            try:
                evaluated_items.append(evaluate_bound_trackb_candidate(
                    candidate=candidate, forward_draw_set=forward_draw_set, posterior=posterior,
                    csm=csm, rsm=rsm, rf_mapping_core=rf_mapping_core,
                    grouped_bootstrap_sha=grouped_bootstrap_sha, evaluation_recipe_sha=recipe_sha,
                ))
            except ValueError as error:
                if candidate.source == "NO_OP" or "exact-core RF preview is not accepted" not in str(error):
                    raise
                rejected_candidates.append(TrackBCandidateEvaluationRejectionV1(
                    candidate_sha=candidate.candidate_sha,
                    reason_code="EXACT_CORE_PREVIEW_REJECTED",
                    forward_draw_set_sha=forward_draw_set.sha256(),
                    rf_mapping_core_sha=rf_mapping_core.sha256(),
                ))
        evaluated = tuple(evaluated_items)
        if not evaluated or not any(item[0].source == "NO_OP" for item in evaluated):
            raise ValueError("posterior evaluator requires an accepted exact-core NO_OP candidate")
        evaluated_library = RegisteredActionCandidateLibraryV1(
            candidate_library.system_type, tuple(item[0] for item in evaluated), rsm.sha256(),
        )
        evaluations = tuple(item[1] for item in evaluated)
        evaluation_set = cls(
            forward_draw_set_sha=forward_draw_set.sha256(),
            posterior_sha=posterior.sha256(),
            csm_sha=csm.sha256(),
            rsm_sha=rsm.sha256(),
            rf_mapping_core_sha=rf_mapping_core.sha256(),
            source_candidate_library_sha=candidate_library.sha256(),
            candidate_library=evaluated_library,
            evaluations=evaluations,
            evaluator_source_sha=evaluator_source_sha,
            evaluator_config_sha=recipe_sha,
            forward_draw_ids_sha=draw_ids_sha,
            grouped_bootstrap_plan_sha=grouped_bootstrap_sha,
            rejected_candidates=tuple(rejected_candidates),
            producer_status="UNVERIFIED_EVALUATION_SET",
        )
        object.__setattr__(evaluation_set, "_replay_forward_draw_set", forward_draw_set)
        object.__setattr__(evaluation_set, "_replay_posterior", posterior)
        object.__setattr__(evaluation_set, "_replay_csm", csm)
        object.__setattr__(evaluation_set, "_replay_rsm", rsm)
        object.__setattr__(evaluation_set, "_replay_candidate_library", candidate_library)
        object.__setattr__(evaluation_set, "_replay_rf_mapping_core", rf_mapping_core)
        object.__setattr__(evaluation_set, "producer_status", "REPLAYED_BOUND_FORWARD_DRAWS")
        return evaluation_set

    def assert_authoritative(self) -> None:
        """Recompute the evaluator result from live replay inputs before use."""
        if self.producer_status != "REPLAYED_BOUND_FORWARD_DRAWS":
            raise ValueError("posterior evaluation set is not replayed from a bound forward producer")
        forward_draw_set = self._replay_forward_draw_set
        posterior = self._replay_posterior
        csm = self._replay_csm
        rsm = self._replay_rsm
        candidate_library = self._replay_candidate_library
        rf_mapping_core = self._replay_rf_mapping_core
        if any(value is None for value in (
            forward_draw_set, posterior, csm, rsm, candidate_library, rf_mapping_core,
        )):
            raise ValueError("posterior evaluation set has no replay inputs")
        fresh = type(self).from_bound_objects(
            forward_draw_set=forward_draw_set,
            posterior=posterior,
            csm=csm,
            rsm=rsm,
            candidate_library=candidate_library,
            rf_mapping_core=rf_mapping_core,
        )
        if fresh.sha256() != self.sha256():
            raise ValueError("posterior evaluation set differs from its replayed bound inputs")

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "forward_draw_set_sha": self.forward_draw_set_sha,
            "posterior_sha": self.posterior_sha,
            "csm_sha": self.csm_sha,
            "rsm_sha": self.rsm_sha,
            "rf_mapping_core_sha": self.rf_mapping_core_sha,
            "source_candidate_library_sha": self.source_candidate_library_sha,
            "candidate_library_sha": self.candidate_library.sha256(),
            "evaluation_shas": [item.sha256() for item in self.evaluations],
            "evaluator_source_sha": self.evaluator_source_sha,
            "evaluator_config_sha": self.evaluator_config_sha,
            "forward_draw_ids_sha": self.forward_draw_ids_sha,
            "grouped_bootstrap_plan_sha": self.grouped_bootstrap_plan_sha,
            "rejected_candidate_shas": [item.sha256() for item in self.rejected_candidates],
            "producer_status": self.producer_status,
        })

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "forward_draw_set_sha": self.forward_draw_set_sha,
            "posterior_sha": self.posterior_sha,
            "csm_sha": self.csm_sha,
            "rsm_sha": self.rsm_sha,
            "rf_mapping_core_sha": self.rf_mapping_core_sha,
            "source_candidate_library_sha": self.source_candidate_library_sha,
            "candidate_library_sha": self.candidate_library.sha256(),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "evaluator_source_sha": self.evaluator_source_sha,
            "evaluator_config_sha": self.evaluator_config_sha,
            "forward_draw_ids_sha": self.forward_draw_ids_sha,
            "grouped_bootstrap_plan_sha": self.grouped_bootstrap_plan_sha,
            "rejected_candidates": [item.to_dict() for item in self.rejected_candidates],
            "producer_status": self.producer_status,
        }


@dataclass(frozen=True)
class TrackBRegisteredTryableRecoveryEnvelopeV1:
    """Best candidate inside a declared finite library and nominal RF preview bounds."""

    candidate_library_sha: str
    registered_tryable_best_candidate_sha: str
    registered_tryable_recovery_fraction: float
    unavoidable_model_floor: float
    ideal_to_rf_realizability_fraction: float
    counterfactual_static_utility_mean: float
    counterfactual_static_utility_lcb95: float
    counterfactual_static_utility_quantiles: tuple[float, float, float]
    predicted_harm_probability: float
    recoverable_support_fraction: float
    observation_noise_gap: float
    basis_gap: float
    nuisance_gap: float
    sector_ambiguity_gap: float
    rf_reachability_gap: float
    rf_bounds_gap: float
    materialization_gap: float
    uncertainty_gap: float
    selected_candidate_score: float | None = None
    estimation_status: str = "CONTRACT_SCAFFOLD_NOT_DATA_EVALUATED"
    evaluation_forward_sha: str | None = None
    evaluation_draw_set_sha: str | None = None
    evaluation_draw_ids_sha: str | None = None
    evaluation_recipe_sha: str | None = None
    evaluation_posterior_sha: str | None = None
    evaluation_grouped_bootstrap_sha: str | None = None
    scope: str = REGISTERED_TRYABLE_SCOPE
    causal_gain_available: bool = False
    physical_global_maximum_proven: bool = False
    real_transition_supervision: bool = False
    schema: str = field(default="TrackBRegisteredTryableRecoveryEnvelopeV1", init=False)

    def __post_init__(self) -> None:
        require_sha256(self.candidate_library_sha, "candidate_library_sha")
        require_sha256(
            self.registered_tryable_best_candidate_sha,
            "registered_tryable_best_candidate_sha",
        )
        values = (
            self.registered_tryable_recovery_fraction, self.unavoidable_model_floor,
            self.ideal_to_rf_realizability_fraction, self.counterfactual_static_utility_mean,
            self.counterfactual_static_utility_lcb95, *self.counterfactual_static_utility_quantiles,
            self.predicted_harm_probability, self.recoverable_support_fraction,
            self.observation_noise_gap, self.basis_gap, self.nuisance_gap, self.sector_ambiguity_gap,
            self.rf_reachability_gap, self.rf_bounds_gap, self.materialization_gap, self.uncertainty_gap,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("tryable envelope values must be finite")
        if not all(0.0 <= float(value) <= 1.0 for value in (
            self.registered_tryable_recovery_fraction, self.unavoidable_model_floor,
            self.ideal_to_rf_realizability_fraction, self.predicted_harm_probability,
            self.recoverable_support_fraction, self.observation_noise_gap, self.basis_gap,
            self.nuisance_gap, self.sector_ambiguity_gap, self.rf_reachability_gap,
            self.rf_bounds_gap, self.materialization_gap, self.uncertainty_gap,
        )):
            raise ValueError("tryable envelope fractions/probabilities must lie in [0,1]")
        if tuple(sorted(self.counterfactual_static_utility_quantiles)) != self.counterfactual_static_utility_quantiles:
            raise ValueError("tryable envelope utility quantiles must be ordered")
        if self.selected_candidate_score is not None and not math.isfinite(float(self.selected_candidate_score)):
            raise ValueError("tryable envelope selected candidate score must be finite when supplied")
        if self.estimation_status not in ENVELOPE_ESTIMATION_STATUSES:
            raise ValueError("tryable envelope estimation status is not registered")
        if self.estimation_status == "POSTERIOR_DRAW_COUNTERFACTUAL":
            for name in (
                "evaluation_forward_sha", "evaluation_draw_set_sha", "evaluation_draw_ids_sha",
                "evaluation_recipe_sha", "evaluation_posterior_sha",
                "evaluation_grouped_bootstrap_sha",
            ):
                require_sha256(getattr(self, name), name)
        elif any(value is not None for value in (
            self.evaluation_forward_sha, self.evaluation_posterior_sha,
            self.evaluation_draw_set_sha, self.evaluation_draw_ids_sha,
            self.evaluation_recipe_sha,
            self.evaluation_grouped_bootstrap_sha,
        )):
            raise ValueError("scaffold envelope cannot claim posterior-draw identities")
        if self.scope != REGISTERED_TRYABLE_SCOPE:
            raise ValueError("tryable envelope scope cannot be widened beyond registered candidates/RF bounds")
        if self.causal_gain_available or self.physical_global_maximum_proven or self.real_transition_supervision:
            raise ValueError("tryable envelope cannot claim causal gain, global maximum, or transition supervision")

    @classmethod
    def from_library(
        cls, library: RegisteredActionCandidateLibraryV1, *, recoverable_support_fraction: float,
        ideal_to_rf_realizability_fraction: float,
        selected_candidate_sha: str | None = None,
    ) -> "TrackBRegisteredTryableRecoveryEnvelopeV1":
        """Summarize the finite registered library under the FD score contract.

        ``selected_candidate_sha`` lets Fusion bind the same registered choice
        used by FD (including a forced no-op). Without it, the shared ranking is
        deterministic and uses the exact same score function.
        """
        selected = (
            library.select_by_sha(selected_candidate_sha)
            if selected_candidate_sha is not None else library.ranked_candidates()[0]
        )
        preview = selected.rf_preview
        realization = max(0.0, min(1.0, float(ideal_to_rf_realizability_fraction)))
        recovery = max(0.0, min(1.0, float(recoverable_support_fraction) * realization))
        return cls(
            candidate_library_sha=library.sha256(),
            registered_tryable_best_candidate_sha=selected.candidate_sha,
            registered_tryable_recovery_fraction=recovery,
            unavoidable_model_floor=1.0 - recovery,
            ideal_to_rf_realizability_fraction=realization,
            counterfactual_static_utility_mean=selected.counterfactual_static_utility_mean,
            counterfactual_static_utility_lcb95=selected.counterfactual_static_utility_lcb95,
            counterfactual_static_utility_quantiles=selected.utility_quantiles,
            predicted_harm_probability=selected.predicted_harm_probability,
            recoverable_support_fraction=float(recoverable_support_fraction),
            observation_noise_gap=0.0,
            basis_gap=0.0,
            nuisance_gap=0.0,
            sector_ambiguity_gap=0.0,
            rf_reachability_gap=max(0.0, 1.0 - abs(preview.direction_cosine)),
            rf_bounds_gap=max(0.0, 1.0 - preview.accepted_for_external_review),
            materialization_gap=min(1.0, max(0.0, preview.requested_to_materialized_logical_error)),
            uncertainty_gap=min(1.0, max(0.0, selected.uncertainty)),
            selected_candidate_score=registered_candidate_score(selected),
        )

    @classmethod
    def from_posterior_evaluations(
        cls,
        library: RegisteredActionCandidateLibraryV1,
        evaluations: tuple[PosteriorCandidateEvaluationV1, ...],
        *,
        selected_candidate_sha: str | None = None,
        tolerance: float = 1.0e-8,
        expected_forward_draw_set_sha: str | None = None,
        expected_draw_ids: tuple[str, ...] | None = None,
    ) -> "TrackBRegisteredTryableRecoveryEnvelopeV1":
        """Build a data-supported finite envelope from same-forward draw outcomes.

        The library metrics must be the summaries of the supplied posterior
        evaluator.  This makes FD and envelope selection share exactly the
        registered candidate score instead of silently using two rankers.
        """
        if not evaluations or len(evaluations) != len(library.candidates):
            raise ValueError("posterior evaluations must cover every registered candidate exactly once")
        by_sha = {item.candidate_sha: item for item in evaluations}
        if len(by_sha) != len(evaluations) or set(by_sha) != {
            candidate.candidate_sha for candidate in library.candidates
        }:
            raise ValueError("posterior evaluations do not match the registered candidate library")
        forward_shas = {item.forward_sha for item in evaluations}
        posterior_shas = {item.posterior_sha for item in evaluations}
        bootstrap_shas = {item.grouped_bootstrap_sha for item in evaluations}
        draw_set_shas = {item.forward_draw_set_sha for item in evaluations}
        draw_id_shas = {item.draw_ids_sha for item in evaluations}
        recipe_shas = {item.evaluation_recipe_sha for item in evaluations}
        if (
            len(forward_shas) != 1 or len(draw_set_shas) != 1
            or len(posterior_shas) != 1 or len(bootstrap_shas) != 1
            or len(draw_id_shas) != 1 or len(recipe_shas) != 1
        ):
            raise ValueError("candidate evaluations must use one same-forward/draw-set/posterior/bootstrap identity")
        if expected_forward_draw_set_sha is not None:
            require_sha256(expected_forward_draw_set_sha, "expected_forward_draw_set_sha")
            if next(iter(draw_set_shas)) != expected_forward_draw_set_sha:
                raise ValueError("candidate evaluations do not bind the expected forward draw set")
        if expected_draw_ids is not None:
            expected_draw_id_sha = canonical_sha256({
                "schema": "TrackBForwardDrawIDsV1",
                "draw_ids": list(expected_draw_ids),
            })
            if next(iter(draw_id_shas)) != expected_draw_id_sha:
                raise ValueError("candidate evaluation draw IDs do not match the sealed forward draw set")
        for candidate in library.candidates:
            evaluation = by_sha[candidate.candidate_sha]
            if (
                not math.isclose(candidate.counterfactual_static_utility_mean, evaluation.utility_mean, abs_tol=tolerance)
                or not math.isclose(candidate.counterfactual_static_utility_lcb95, evaluation.utility_lcb95, abs_tol=tolerance)
                or not math.isclose(candidate.predicted_harm_probability, evaluation.harm_probability, abs_tol=tolerance)
                or not math.isclose(candidate.uncertainty, evaluation.uncertainty_mean, abs_tol=tolerance)
                or not math.isclose(candidate.abstain_risk, evaluation.abstain_mean, abs_tol=tolerance)
                or not math.isclose(candidate.direction_stability, evaluation.direction_stability_mean, abs_tol=tolerance)
                or not math.isclose(candidate.apparatus_confidence, evaluation.rf_confidence_mean, abs_tol=tolerance)
                or any(
                    not math.isclose(left, right, abs_tol=tolerance)
                    for left, right in zip(candidate.utility_quantiles, evaluation.utility_quantiles, strict=True)
                )
            ):
                raise ValueError("candidate gate diagnostics are not bound to the supplied posterior evaluation")
        selected = library.select_by_sha(selected_candidate_sha) if selected_candidate_sha else library.ranked_candidates()[0]
        evaluation = by_sha[selected.candidate_sha]
        recovery = _mean(evaluation.recoverable_fraction_samples)
        realization = _mean(evaluation.ideal_to_rf_realizability_samples)
        return cls(
            candidate_library_sha=library.sha256(),
            registered_tryable_best_candidate_sha=selected.candidate_sha,
            registered_tryable_recovery_fraction=recovery,
            unavoidable_model_floor=1.0 - recovery,
            ideal_to_rf_realizability_fraction=realization,
            counterfactual_static_utility_mean=evaluation.utility_mean,
            counterfactual_static_utility_lcb95=evaluation.utility_lcb95,
            counterfactual_static_utility_quantiles=evaluation.utility_quantiles,
            predicted_harm_probability=evaluation.harm_probability,
            recoverable_support_fraction=recovery,
            observation_noise_gap=_mean(evaluation.observation_noise_gap_samples),
            basis_gap=_mean(evaluation.basis_gap_samples),
            nuisance_gap=_mean(evaluation.nuisance_gap_samples),
            sector_ambiguity_gap=_mean(evaluation.sector_ambiguity_gap_samples),
            rf_reachability_gap=_mean(evaluation.rf_reachability_gap_samples),
            rf_bounds_gap=_mean(evaluation.rf_bounds_gap_samples),
            materialization_gap=_mean(evaluation.materialization_gap_samples),
            uncertainty_gap=_mean(evaluation.uncertainty_gap_samples),
            selected_candidate_score=registered_candidate_score(selected),
            estimation_status="POSTERIOR_DRAW_COUNTERFACTUAL",
            evaluation_forward_sha=next(iter(forward_shas)),
            evaluation_draw_set_sha=next(iter(draw_set_shas)),
            evaluation_draw_ids_sha=next(iter(draw_id_shas)),
            evaluation_recipe_sha=next(iter(recipe_shas)),
            evaluation_posterior_sha=next(iter(posterior_shas)),
            evaluation_grouped_bootstrap_sha=next(iter(bootstrap_shas)),
        )

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "candidate_library_sha": self.candidate_library_sha,
            "registered_tryable_best_candidate_sha": self.registered_tryable_best_candidate_sha,
            "registered_tryable_recovery_fraction": self.registered_tryable_recovery_fraction,
            "unavoidable_model_floor": self.unavoidable_model_floor,
            "ideal_to_rf_realizability_fraction": self.ideal_to_rf_realizability_fraction,
            "counterfactual_static_utility_mean": self.counterfactual_static_utility_mean,
            "counterfactual_static_utility_lcb95": self.counterfactual_static_utility_lcb95,
            "counterfactual_static_utility_quantiles": list(self.counterfactual_static_utility_quantiles),
            "predicted_harm_probability": self.predicted_harm_probability,
            "recoverable_support_fraction": self.recoverable_support_fraction,
            "observation_noise_gap": self.observation_noise_gap,
            "basis_gap": self.basis_gap,
            "nuisance_gap": self.nuisance_gap,
            "sector_ambiguity_gap": self.sector_ambiguity_gap,
            "rf_reachability_gap": self.rf_reachability_gap,
            "rf_bounds_gap": self.rf_bounds_gap,
            "materialization_gap": self.materialization_gap,
            "uncertainty_gap": self.uncertainty_gap,
            "selected_candidate_score": self.selected_candidate_score,
            "estimation_status": self.estimation_status,
            "evaluation_forward_sha": self.evaluation_forward_sha,
            "evaluation_draw_set_sha": self.evaluation_draw_set_sha,
            "evaluation_draw_ids_sha": self.evaluation_draw_ids_sha,
            "evaluation_recipe_sha": self.evaluation_recipe_sha,
            "evaluation_posterior_sha": self.evaluation_posterior_sha,
            "evaluation_grouped_bootstrap_sha": self.evaluation_grouped_bootstrap_sha,
            "scope": self.scope,
            "causal_gain_available": False,
            "physical_global_maximum_proven": False,
            "real_transition_supervision": False,
        })


__all__ = [
    "BoundTrackBPosteriorEvaluationSetV1", "ENVELOPE_ESTIMATION_STATUSES",
    "PosteriorCandidateEvaluationV1", "TrackBCandidateEvaluationRejectionV1",
    "REGISTERED_TRYABLE_SCOPE", "TrackBRegisteredTryableRecoveryEnvelopeV1",
]
