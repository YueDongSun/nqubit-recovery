"""Pure Track B inference and the closed actionful-proposal pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .candidate_library import RegisteredActionCandidateLibraryV1, registered_candidate_score
from .contracts import (
    ResearchRFActionProposalV1,
    TrackBFDResearchConclusionV1,
    TrackBProposalEligibilityReceiptV1,
    canonical_sha256,
)
from .fd_model import FDSelectionV1, TrackBFDModelV1
from .forward_draw_set import BoundTrackBForwardDrawSetV1
from .posterior import TrackBPosteriorV1
from .posterior_receipt import BoundTrackBPosteriorReceiptV1
from .real_csm import RealCSMSupportV1
from .real_rsm import RealRSMSupportV1
from .recoverable_envelope import (
    BoundTrackBAppliedCandidateEvaluationV1,
    BoundTrackBPosteriorEvaluationSetV1,
    PosteriorCandidateEvaluationV1,
    TrackBRegisteredTryableRecoveryEnvelopeV1,
    evaluate_bound_trackb_candidate,
)
from .rf_mapping_core import RegisteredRFMappingCoreV1
from .rf_preview import preview_registered_rf_action


@dataclass(frozen=True)
class TrackBFusionResultV1:
    selection: FDSelectionV1
    envelope: TrackBRegisteredTryableRecoveryEnvelopeV1
    conclusion: TrackBFDResearchConclusionV1
    proposal: ResearchRFActionProposalV1 | None
    eligibility_receipt: TrackBProposalEligibilityReceiptV1 | None = None


def _scaffold_envelope(
    library: RegisteredActionCandidateLibraryV1,
    selection: FDSelectionV1,
    rsm: RealRSMSupportV1,
) -> TrackBRegisteredTryableRecoveryEnvelopeV1:
    return TrackBRegisteredTryableRecoveryEnvelopeV1.from_library(
        library,
        recoverable_support_fraction=rsm.recoverable_residual_fraction,
        ideal_to_rf_realizability_fraction=max(
            0.0, selection.rf_confidence * (1.0 - selection.abstain_probability),
        ),
        selected_candidate_sha=selection.registered_candidate_sha,
    )


def _proposal_context_fields(context: Mapping[str, object]) -> None:
    required_external = {
        "proposal_id", "parent_command_bundle_sha", "expires_at_utc",
    }
    optional_external = {"model_artifact_kind", "model_provider_type"}
    unknown = set(context) - required_external - optional_external
    if unknown:
        raise ValueError(f"proposal context contains non-external or unknown identities: {sorted(unknown)}")
    if required_external - set(context):
        raise ValueError("research proposal context is missing an external evidence identity")


def _final_gate_metrics(
    *, candidate, csm: RealCSMSupportV1, posterior: TrackBPosteriorV1,
) -> tuple[float, float, float, float, float]:
    harm = candidate.predicted_harm_probability
    abstain = max(candidate.abstain_risk, 1.0 - csm.confidence, posterior.ood_probability)
    return (
        harm,
        max(harm, abstain),
        abstain,
        min(candidate.direction_stability, csm.direction_stability),
        candidate.apparatus_confidence,
    )


class V10TrackBRealFDRFBridgeV1:
    """Compose diagnostics and a single closed proposal-producing pipeline.

    Raw posterior/CSM/RSM objects remain usable for a conclusion-only scaffold.
    An actionful proposal, however, can be produced only by this method's
    source-bound path; no public helper accepts a caller-built selection or
    envelope.
    """

    def __init__(self, system_type: str) -> None:
        self.fd_model = TrackBFDModelV1(system_type)
        self.system_type = system_type

    def infer(
        self, *, posterior: TrackBPosteriorV1, csm: RealCSMSupportV1, rsm: RealRSMSupportV1,
        library: RegisteredActionCandidateLibraryV1, source_commit: str, source_tree: str,
        data_manifest_sha: str, direction_conflict: bool = False,
        proposal_context: Mapping[str, object] | None = None,
        rf_mapping_core: RegisteredRFMappingCoreV1 | None = None,
        posterior_evaluations: tuple[PosteriorCandidateEvaluationV1, ...] | None = None,
        posterior_evaluation_set: BoundTrackBPosteriorEvaluationSetV1 | None = None,
        posterior_receipt: BoundTrackBPosteriorReceiptV1 | None = None,
        forward_draw_set: BoundTrackBForwardDrawSetV1 | None = None,
    ) -> TrackBFusionResultV1:
        if posterior_evaluations is not None or posterior_evaluation_set is not None:
            raise ValueError(
                "fusion no longer accepts caller-supplied posterior evaluations; "
                "the proposal path replays the bound producer itself"
            )
        if rf_mapping_core is not None and type(rf_mapping_core) is not RegisteredRFMappingCoreV1:
            raise TypeError("fusion requires the exact registered RF mapping core")

        active_library = library
        active_evaluation_set = None
        if forward_draw_set is not None and rf_mapping_core is not None:
            active_evaluation_set = BoundTrackBPosteriorEvaluationSetV1.from_bound_objects(
                forward_draw_set=forward_draw_set,
                posterior=posterior,
                csm=csm,
                rsm=rsm,
                candidate_library=library,
                rf_mapping_core=rf_mapping_core,
            )
            active_evaluation_set.assert_authoritative()
            active_library = active_evaluation_set.candidate_library

        selection = self.fd_model.select(
            posterior=posterior, csm=csm, rsm=rsm, library=active_library,
            direction_conflict=direction_conflict, rf_mapping_core=rf_mapping_core,
        )
        envelope = _scaffold_envelope(active_library, selection, rsm)
        proposal = None
        eligibility_receipt = None

        if proposal_context is not None and not direction_conflict \
                and posterior_receipt is not None and forward_draw_set is not None \
                and rf_mapping_core is not None:
            selection, envelope, proposal, eligibility_receipt = self._run_actionful_pipeline(
                posterior=posterior,
                posterior_receipt=posterior_receipt,
                forward_draw_set=forward_draw_set,
                csm=csm,
                rsm=rsm,
                source_library=library,
                rf_mapping_core=rf_mapping_core,
                source_commit=source_commit,
                source_tree=source_tree,
                data_manifest_sha=data_manifest_sha,
                context=proposal_context,
            )

        conclusion = TrackBFDResearchConclusionV1(
            source_commit=source_commit,
            source_tree=source_tree,
            data_manifest_sha=data_manifest_sha,
            selected_candidate_sha=selection.selected_candidate_sha,
            envelope_sha=envelope.sha256(),
            eligibility_policy_sha=selection.eligibility_policy_sha,
            proposal_sha=None if proposal is None else proposal.sha256(),
            reason_codes=(
                selection.reason_code,
                selection.eligibility_reason_code,
                envelope.estimation_status,
                "RESEARCH_ONLY_EXTERNAL_EXECUTION_REQUIRED",
            ),
        )
        return TrackBFusionResultV1(selection, envelope, conclusion, proposal, eligibility_receipt)

    def _run_actionful_pipeline(
        self, *, posterior: TrackBPosteriorV1, posterior_receipt: BoundTrackBPosteriorReceiptV1,
        forward_draw_set: BoundTrackBForwardDrawSetV1, csm: RealCSMSupportV1,
        rsm: RealRSMSupportV1, source_library: RegisteredActionCandidateLibraryV1,
        rf_mapping_core: RegisteredRFMappingCoreV1, source_commit: str, source_tree: str,
        data_manifest_sha: str, context: Mapping[str, object],
    ) -> tuple[
        FDSelectionV1,
        TrackBRegisteredTryableRecoveryEnvelopeV1,
        ResearchRFActionProposalV1 | None,
        TrackBProposalEligibilityReceiptV1 | None,
    ]:
        _proposal_context_fields(context)
        if csm.forward_draw_set_sha is None or csm.forward_draw_ids_sha is None:
            raise ValueError("actionful proposal requires a sealed CSM draw set")
        posterior_receipt.verify_for(
            posterior=posterior,
            forward_draw_set=forward_draw_set,
            forward_draw_set_sha=csm.forward_draw_set_sha,
            forward_draw_ids_sha=csm.forward_draw_ids_sha,
            source_commit=source_commit,
            source_tree=source_tree,
        )
        if posterior_receipt.data_manifest_sha != data_manifest_sha:
            raise ValueError("posterior receipt data manifest differs from proposal input")
        if rf_mapping_core.rf_to_awg_mapping is None or rf_mapping_core.rf_to_awg_recipe_sha is None:
            raise ValueError("actionful proposal requires a source-bound RF-to-AWG mapping")

        evaluation_set = BoundTrackBPosteriorEvaluationSetV1.from_bound_objects(
            forward_draw_set=forward_draw_set,
            posterior=posterior,
            csm=csm,
            rsm=rsm,
            candidate_library=source_library,
            rf_mapping_core=rf_mapping_core,
        )
        evaluation_set.assert_authoritative()
        evaluated_library = evaluation_set.candidate_library
        no_op = next(candidate for candidate in evaluated_library.candidates if candidate.source == "NO_OP")
        evaluation_by_sha = {item.candidate_sha: item for item in evaluation_set.evaluations}
        no_op_evaluation = evaluation_by_sha[no_op.candidate_sha]
        full_ranking = tuple(item.candidate_sha for item in evaluated_library.ranked_candidates())

        for registered_candidate in evaluated_library.ranked_candidates():
            if registered_candidate.source == "NO_OP":
                continue
            trial_library = RegisteredActionCandidateLibraryV1(
                evaluated_library.system_type, (no_op, registered_candidate), rsm.sha256(),
            )
            trial_selection = self.fd_model.select(
                posterior=posterior, csm=csm, rsm=rsm, library=trial_library,
                rf_mapping_core=rf_mapping_core,
            )
            if not trial_selection.proposal_eligible or trial_selection.applied_candidate is None:
                continue
            applied_candidate, applied_evaluation = evaluate_bound_trackb_candidate(
                candidate=trial_selection.applied_candidate,
                forward_draw_set=forward_draw_set,
                posterior=posterior,
                csm=csm,
                rsm=rsm,
                rf_mapping_core=rf_mapping_core,
                grouped_bootstrap_sha=evaluation_set.grouped_bootstrap_plan_sha,
                evaluation_recipe_sha=evaluation_set.evaluator_config_sha,
            )
            harm, stop, abstain, direction, confidence = _final_gate_metrics(
                candidate=applied_candidate, csm=csm, posterior=posterior,
            )
            gate_ok, gate_reason = self.fd_model.eligibility_gate.evaluate(
                candidate=applied_candidate,
                csm=csm,
                rsm=rsm,
                harm_probability=harm,
                stop_probability=stop,
                abstain_probability=abstain,
                direction_stability=direction,
                rf_confidence=confidence,
            )
            if not gate_ok or registered_candidate_score(applied_candidate) <= (
                registered_candidate_score(no_op)
                + self.fd_model.eligibility_gate.min_actionful_score_over_no_op
            ):
                continue
            applied_evaluation_receipt = BoundTrackBAppliedCandidateEvaluationV1(
                registered_candidate_sha=registered_candidate.candidate_sha,
                applied_candidate_sha=applied_candidate.candidate_sha,
                applied_global_gamma=float(trial_selection.global_gamma),
                evaluation=applied_evaluation,
            )
            final_library = RegisteredActionCandidateLibraryV1(
                evaluated_library.system_type, (no_op, applied_candidate), rsm.sha256(),
            )
            envelope = TrackBRegisteredTryableRecoveryEnvelopeV1.from_posterior_evaluations(
                final_library,
                (no_op_evaluation, applied_evaluation),
                selected_candidate_sha=applied_candidate.candidate_sha,
                expected_forward_draw_set_sha=csm.forward_draw_set_sha,
                expected_draw_ids=csm.forward_draw_ids,
            )
            selection = replace(
                trial_selection,
                selected_candidate_sha=applied_candidate.candidate_sha,
                candidate_ranking=full_ranking,
                candidate_uncertainty=applied_candidate.uncertainty,
                harm_probability=harm,
                stop_probability=stop,
                abstain_probability=abstain,
                direction_stability=direction,
                rf_confidence=confidence,
                eligibility_reason_code=gate_reason,
                applied_candidate=applied_candidate,
            )
            receipt = _build_pipeline_receipt(
                selection=selection,
                posterior_receipt=posterior_receipt,
                csm=csm,
                rsm=rsm,
                envelope=envelope,
                library=final_library,
                rf_mapping_core=rf_mapping_core,
                applied_evaluation=applied_evaluation_receipt,
                evaluation_set=evaluation_set,
            )
            proposal = _serialize_pipeline_proposal(
                selection=selection,
                posterior_receipt=posterior_receipt,
                rsm=rsm,
                rf_mapping_core=rf_mapping_core,
                receipt=receipt,
                context=context,
            )
            return selection, envelope, proposal, receipt

        no_op_library = RegisteredActionCandidateLibraryV1(
            evaluated_library.system_type, (no_op,), rsm.sha256(),
        )
        selection = self.fd_model.select(
            posterior=posterior, csm=csm, rsm=rsm, library=no_op_library,
            rf_mapping_core=rf_mapping_core,
        )
        envelope = TrackBRegisteredTryableRecoveryEnvelopeV1.from_posterior_evaluations(
            no_op_library,
            (no_op_evaluation,),
            selected_candidate_sha=no_op.candidate_sha,
            expected_forward_draw_set_sha=csm.forward_draw_set_sha,
            expected_draw_ids=csm.forward_draw_ids,
        )
        return selection, envelope, None, None


def _build_pipeline_receipt(
    *, selection: FDSelectionV1, posterior_receipt: BoundTrackBPosteriorReceiptV1,
    csm: RealCSMSupportV1, rsm: RealRSMSupportV1,
    envelope: TrackBRegisteredTryableRecoveryEnvelopeV1,
    library: RegisteredActionCandidateLibraryV1,
    rf_mapping_core: RegisteredRFMappingCoreV1,
    applied_evaluation: BoundTrackBAppliedCandidateEvaluationV1,
    evaluation_set: BoundTrackBPosteriorEvaluationSetV1,
) -> TrackBProposalEligibilityReceiptV1:
    if selection.applied_candidate is None or not selection.proposal_eligible:
        raise ValueError("pipeline receipt requires an eligible applied candidate")
    candidate = selection.applied_candidate
    candidate.recoverable_action.verify_projection(rsm=rsm, csm=csm)
    preview = preview_registered_rf_action(rf_mapping_core, candidate.recoverable_action)
    if preview.sha256() != candidate.rf_preview.sha256():
        raise ValueError("pipeline applied candidate preview does not replay the exact RF core")
    logical_action_sha = candidate.rf_preview.requested_rf_delta.logical_action_payload_sha
    if logical_action_sha is None or logical_action_sha != canonical_sha256({
        "schema": "TrackBLogicalActionPayloadV1",
        "system_type": candidate.recoverable_action.ideal_action.system_type,
        "amplitude_coordinates": list(candidate.recoverable_action.amplitude_coordinates),
        "phase_coordinates": list(candidate.recoverable_action.phase_coordinates),
        "coordinate_layout_sha": candidate.recoverable_action.ideal_action.coordinate_layout_sha,
    }):
        raise ValueError("pipeline RF delta does not bind the applied logical action payload")
    recipe_sha = rf_mapping_core.rf_to_awg_recipe_sha
    mapping = rf_mapping_core.rf_to_awg_mapping
    if recipe_sha is None or mapping is None:
        raise ValueError("pipeline receipt requires a source-bound RF-to-AWG recipe")
    return TrackBProposalEligibilityReceiptV1(
        selection_sha=selection.sha256(),
        candidate_library_sha=library.sha256(),
        registered_candidate_sha=selection.registered_candidate_sha or candidate.base_candidate_sha or candidate.candidate_sha,
        applied_candidate_sha=candidate.candidate_sha,
        posterior_sha=posterior_receipt.posterior_payload_sha,
        forward_draw_set_sha=csm.forward_draw_set_sha or "",
        csm_sha=csm.sha256(),
        rsm_sha=rsm.sha256(),
        envelope_sha=envelope.sha256(),
        eligibility_policy_sha=selection.eligibility_policy_sha,
        rf_mapping_core_sha=rf_mapping_core.sha256(),
        rf_preview_sha=candidate.rf_preview.sha256(),
        rf_command_delta_sha=candidate.rf_preview.requested_rf_delta.sha256(),
        evaluation_set_sha=evaluation_set.sha256(),
        posterior_receipt_sha=posterior_receipt.sha256(),
        applied_evaluation_sha=applied_evaluation.sha256(),
        logical_action_payload_sha=logical_action_sha,
        rf_to_awg_mapping_sha=mapping.sha256(),
        awg_recipe_sha=recipe_sha,
    )


def _serialize_pipeline_proposal(
    *, selection: FDSelectionV1, posterior_receipt: BoundTrackBPosteriorReceiptV1,
    rsm: RealRSMSupportV1, rf_mapping_core: RegisteredRFMappingCoreV1,
    receipt: TrackBProposalEligibilityReceiptV1, context: Mapping[str, object],
) -> ResearchRFActionProposalV1:
    _proposal_context_fields(context)
    candidate = selection.applied_candidate
    if candidate is None:
        raise ValueError("pipeline proposal requires an applied candidate")
    action = candidate.recoverable_action
    mapping = rf_mapping_core.rf_to_awg_mapping
    recipe_sha = rf_mapping_core.rf_to_awg_recipe_sha
    if mapping is None or recipe_sha is None:
        raise ValueError("pipeline proposal requires a source-bound RF-to-AWG recipe")
    if any(value is None for value in (
        rf_mapping_core.operation_registry_sha, rf_mapping_core.command_table_sha,
        rf_mapping_core.hardware_path_id, rf_mapping_core.operating_point_id,
    )):
        raise ValueError("pipeline RF core is missing operation/hardware source identities")
    proposal = ResearchRFActionProposalV1(
        proposal_id=context["proposal_id"],
        system_type=action.ideal_action.system_type,
        source_commit=posterior_receipt.source_commit,
        source_tree=posterior_receipt.source_tree,
        model_checkpoint_sha=posterior_receipt.model_checkpoint_sha,
        model_output_bundle_sha=posterior_receipt.model_output_bundle_sha,
        data_manifest_sha=posterior_receipt.data_manifest_sha,
        desired_target_sha=rsm.desired_target_sha,
        action_sha=receipt.logical_action_payload_sha,
        current_control_snapshot_sha=rf_mapping_core.current_control_snapshot_sha,
        parent_command_bundle_sha=context["parent_command_bundle_sha"],
        bounds_policy_sha=rf_mapping_core.bounds_policy_sha,
        rf_coordinate_layout_sha=rf_mapping_core.rf_coordinate_layout_sha,
        rf_command_basis_sha=rf_mapping_core.rf_command_basis_sha,
        logical_coordinate_layout_sha=action.ideal_action.coordinate_layout_sha,
        hardware_path_id=rf_mapping_core.hardware_path_id,
        operating_point_id=rf_mapping_core.operating_point_id,
        operation_registry_sha=rf_mapping_core.operation_registry_sha,
        command_table_sha=rf_mapping_core.command_table_sha,
        selected_candidate_sha=candidate.candidate_sha,
        eligibility_policy_sha=receipt.eligibility_policy_sha,
        eligibility_receipt_sha=receipt.sha256(),
        posterior_receipt_sha=posterior_receipt.sha256(),
        applied_evaluation_sha=receipt.applied_evaluation_sha,
        rf_to_awg_mapping_sha=mapping.sha256(),
        awg_recipe_sha=recipe_sha,
        expires_at_utc=context["expires_at_utc"],
        amplitude_coordinates=action.amplitude_coordinates,
        phase_coordinates=action.phase_coordinates,
        model_artifact_kind=context.get("model_artifact_kind", "FILE_BACKED_CHECKPOINT"),
        model_provider_type=context.get("model_provider_type", "TRACKB_REALFD_V1"),
    )
    proposal.validate()
    return proposal


__all__ = ["TrackBFusionResultV1", "V10TrackBRealFDRFBridgeV1"]
