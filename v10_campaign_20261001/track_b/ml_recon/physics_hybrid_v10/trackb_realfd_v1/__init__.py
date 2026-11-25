"""V10 Track B RealFD/RF research-only model package.

The package issues typed research conclusions and preview-only proposals.  It
does not import hardware execution code and cannot authorize RF/AWG actions.
"""

from .contracts import (
    ResearchRFActionProposalV1,
    TrackBFDResearchConclusionV1,
    TrackBProposalEligibilityReceiptV1,
    TrackBRealV5CompatibilityManifestV1,
    TrackBResearchAuthorityV1,
)
from .coordinate_abi import (
    IdealLogicalActionV1,
    ModeActionV1,
    RFCommandDeltaV1,
    RFMaterializationPreviewV1,
    RecoverableLogicalActionV1,
)
from .action_prior import (
    ActionPriorBundleV1,
    ActionPriorCompatibilityContextV1,
    ActionPriorFileBackedAdmissionV1,
    ActionPriorProviderV1,
)
from .actuator_transfer_v2 import ActuatorTransferContractV2, SingularSupportContractV1
from .candidate_library import RegisteredActionCandidateLibraryV1, RegisteredActionCandidateV1
from .dual_descriptor import DualOperationDescriptorV1
from .fd_model import F_3Q_HW_V10, F_4Q_HW_V10, ProposalEligibilityGateV1, TrackBFDModelV1
from .forward_draw_set import (
    BoundTrackBForwardDrawSetV1,
    build_bound_trackb_forward_draw_set,
    forward_callable_identity_sha,
)
from .measurement_rf_map import MeasurementRFMapV1, TrainableMeasurementRFMapV1
from .observation_plate import TrainableMaskedObservationPlateV1, TrainableMaskedObservationPlateConfigV1
from .datasets import (
    TrackBOuterFoldPlanV1,
    TrackBRealDataAdmissionManifestV1,
    TrackBRealFoldAdmissionV1,
    TrackBRealStaticDatasetV1,
    open_trackb_v24_real_static_fold,
    trackb_real_static_collate,
    trackb_real_static_evaluation_collate,
)
from .operation_metadata_sidecar import (
    TrackBV240OperationMetadataSidecarV1,
    VerifiedOperationMetadataV1,
    build_v240_operation_metadata_sidecar,
    load_verified_v240_operation_metadata_sidecar,
    load_verified_v240_operation_metadata_sidecar_bytes,
    write_v240_operation_metadata_sidecar,
)
from .historical_r6_read_only import (
    HISTORICAL_R6_SOURCE_ACCESS_PROFILE,
    TrackBV240HistoricalR6ReadOnlySidecarV2,
    build_historical_r6_read_only_sidecar,
    load_verified_historical_r6_read_only_sidecar,
    write_historical_r6_read_only_sidecar,
)
from .losses import TrackBRealStaticLossWeightsV1, trackb_masked_real_static_objective
from .evaluation import (
    TrackBFixedVisibleMarginalBaselineEvaluationV1,
    build_trackb_fixed_visible_marginal_prediction,
    evaluate_trackb_fixed_visible_marginal_baseline,
)
from .postpilot_diagnostics import (
    TrackBPostPilotDiagnosticRecordV1,
    evaluate_trackb_full_fold_postpilot_diagnostics,
)
from .read_only_pilot_audit import (
    TrackBReadOnlyPilotArtifactAuditV1,
    TrackBReadOnlyPilotCheckpointAuditV1,
    TrackBReadOnlyPilotReplayV1,
    audit_trackb_bounded_pilot_artifacts,
    load_trackb_read_only_pilot_checkpoint,
    load_trackb_read_only_pilot_checkpoint_bytes,
)
from .training import (
    TrackBLongTrainingAuthorizationV1,
    TrackBRealBatchPlanV1,
    seed_trackb_runtime,
    trackb_model_state_sha,
    TrackBRealRuntimeConfigV1,
    TrackBRealRuntimeIdentityV1,
    TrackBRealTrainingRuntimeV1,
    TrackBRuntimeEnvironmentV1,
    TrackBThreeSeedEnsemblePlanV1,
)
from .posterior import TrackBPosteriorV1
from .posterior_receipt import BoundTrackBPosteriorReceiptV1
from .native_posterior import (
    TrackBNativeEnsemblePosteriorV1,
    build_native_ensemble_posterior,
    validate_native_ensemble_posterior,
)
from .native_action_sensitivity import (
    TrackBNativeProjectiveSensitivityV1,
    build_bound_native_projective_sensitivity,
    build_native_projective_forward_instance,
    build_native_projective_sensitivity,
    native_projective_covariance,
    native_projective_forward,
    verify_native_projective_sensitivity_companions,
)
from .native_state_context import describe_native_state_context
from .native_state_ensemble import (
    Native_state_ensemble,
    build_native_state_ensemble,
    verify_native_state_companions,
)
from .native_state_sensitivity import (
    Native_state_sensitivity,
    build_native_state_sensitivity,
    verify_native_state_sensitivity_companions,
)
from .artifact_io import (
    TrackBArtifactPublicationV1,
    publish_new_bytes,
    publish_new_bytes_with_state,
)
from .source_artifacts import (
    TrackBActionfulSourceArtifactBundleV1,
    TrackBSourceArtifactFileV1,
)
from .projector_descriptor import IdealProjectorDescriptorV2
from .real_csm import CSM_GEOMETRY_STATUSES, RealCSMSupportV1
from .real_rsm import MetricProjectionContractV1, RSM_SUPPORT_STATUSES, RealRSMSupportV1
from .recoverable_envelope import (
    BoundTrackBAppliedCandidateEvaluationV1,
    BoundTrackBPosteriorEvaluationSetV1,
    PosteriorCandidateEvaluationV1,
    TrackBRegisteredTryableRecoveryEnvelopeV1,
)
from .rf_command_descriptor import RawRFCommandDescriptorV1
from .fusion import V10TrackBRealFDRFBridgeV1
from .rf_mapping_core import (
    RFCoordinateSpecV1,
    RegisteredRFMappingCoreV1,
    RegisteredSingularSupportV1,
    solve_registered_rf_mapping,
)
from .rf_to_awg_mapping import (
    RFToAWGCoordinateBindingV1,
    RegisteredRFToAWGMappingV1,
)
from .rf_preview import RegisteredRFPreviewProviderV1, preview_registered_rf_action
from .unequal_concentration_main_effects import (
    Unequal_concentration_config, unequal_concentration_four_cell_nll,
    unequal_concentration_zero_fallback, unequal_nested_four_cell_moments,
)
from .unequal_concentration_math_audit import (
    Unequal_concentration_math_audit_config,
    audit_unequal_concentration_math,
    independent_beta_binomial_log_pmf,
)
from .unequal_concentration_point import (
    Unequal_concentration_point_overlay,
    build_unequal_concentration_point_overlay,
    load_unequal_concentration_point_overlay_bytes,
    validate_unequal_concentration_point_artifact,
    verify_unequal_concentration_point_overlay,
)
from .unequal_concentration_pit import (
    build_unequal_concentration_pit_plan,
    execute_unequal_concentration_pit_plan,
    validate_unequal_concentration_pit_plan,
)
from .unequal_concentration_checkpoint import (
    Unequal_concentration_checkpoint_binding,
    build_unequal_concentration_checkpoint_binding,
    load_unequal_concentration_checkpoint_binding_bytes,
    validate_unequal_concentration_checkpoint_binding,
    verify_unequal_concentration_checkpoint_binding,
)
from .model_admission import (
    COMPARISON_METRICS, EFFECTIVE_FAMILY_CONTEXT_V3_VARIANT, LEGACY_V1_VARIANT,
    PROTOCOL_SCHEMA, REGISTERED_MODEL_VARIANTS, SUPPORTED_VARIANTS,
    TrackBModelComparisonProtocolV1, TrackBModelVariantSpecV1,
    build_recommended_trackb_model_protocol, load_trackb_model_protocol,
)
from .inverse_transformer import (
    BOUNDED_GAUGE_PROJECTION,
    RISK_NOT_TRAINED,
    TrackBInverseTransformerConfigV1,
    TrackBInverseTransformerDependencyV1,
    TrackBInverseTransformerV1,
    measure_inverse_transformer_dependency,
    project_bounded_gauge_action,
)
from .inverse_transformer_pilot import (
    TrackBV24TenRoundPilotConfigV1,
    build_trackb_ten_round_v24_plan,
    generate_trackb_ten_round_v24_dataset,
    run_trackb_v24_ten_round_pilot,
)
from .train_fit_diagnostic import (
    TRAIN_FIT_DIAGNOSTIC_SCHEMA,
    TrackBFrozenTrainFitDiagnosticConfigV1,
    run_trackb_frozen_train_fit_diagnostic,
)
from .recomputable_diagnostic import (
    RECOMPUTABLE_TRAIN_DIAGNOSTIC_SCHEMA,
    ROW_EQUAL_METRIC_AGGREGATION,
    TrackBRecomputableTrainDiagnosticConfigV1,
    run_trackb_recomputable_train_diagnostic,
)
from .gradient_diagnostic import (
    FROZEN_GRADIENT_DIAGNOSTIC_SCHEMA,
    TrackBFrozenGradientDiagnosticConfigV1,
    run_trackb_frozen_gradient_diagnostic,
)
from .recomputable_metrics import LEDGER_ORDER
from .ten_round_guidance import (
    TrackBGuidanceObservationV1,
    TrackBLogicalGuidanceV1,
    TrackBTenRoundGuidanceRecordV1,
    TrackBTenRoundGuidanceSessionV1,
    TrackBV24GuidanceSourceBindingV1,
    write_ten_round_guidance_session,
)
from .candidate_action_evaluator import (
    TrackBCandidateActionScoreV1,
    evaluate_trackb_model_candidate_actions,
    is_materially_negative_trackb_gain,
    score_trackb_candidate_action,
)
from .candidate_policy_evaluator import (
    FROZEN_VALIDATION_ROLE,
    TrackBActionScaleTailPolicyV1,
    TrackBActionScaleTailPolicySummaryV1,
    TrackBActionScaleTailPolicyEvaluationV1,
    TrackBPolicyEvaluationBindingV1,
    TrackBResearchOnlyPolicyChoiceV1,
    TRAIN_SCREEN_ROLE,
    choose_research_only_conservative_policy,
    evaluate_trackb_action_scale_tail_policies,
)
from .candidate_critic import (
    TrackBCandidateCriticBatchV1,
    TrackBCandidateCriticConfigV1,
    TrackBCandidateCriticOutputV1,
    TrackBCandidateCriticV1,
    candidate_critic_loss,
)
from .candidate_transition_contract import TrackBCandidateTransitionLabelV1
from .selective_risk_contract import (
    RISK_CALIBRATED_CERTIFIED,
    RISK_NOT_TRAINED as SELECTIVE_RISK_NOT_TRAINED,
    TrackBSelectiveRiskDecisionV1,
    TrackBSelectiveRiskQualificationV1,
    clopper_pearson_one_sided_upper_bound,
    fail_closed_selective_risk_decision,
    selective_risk_policy_sha,
)
from .shared_rf_architecture import (
    TrackBSharedRFEncoderConfigV1,
    TrackBSharedRFEncoderV1,
    TrackBSharedRFInputsV1,
    TrackBSystemTaskHeadConfigV1,
    TrackBSystemTaskHeadsV1,
)

__all__ = [
    "IdealLogicalActionV1", "ModeActionV1", "RecoverableLogicalActionV1",
    "RFCommandDeltaV1", "RFMaterializationPreviewV1", "ResearchRFActionProposalV1",
    "TrackBFDResearchConclusionV1", "TrackBProposalEligibilityReceiptV1", "TrackBRealV5CompatibilityManifestV1",
    "TrackBResearchAuthorityV1",
    "ActionPriorBundleV1", "ActionPriorCompatibilityContextV1",
    "ActionPriorFileBackedAdmissionV1", "ActionPriorProviderV1", "ActuatorTransferContractV2",
    "SingularSupportContractV1", "RegisteredActionCandidateLibraryV1",
    "RegisteredActionCandidateV1", "DualOperationDescriptorV1", "F_3Q_HW_V10",
    "F_4Q_HW_V10", "ProposalEligibilityGateV1", "TrackBFDModelV1", "MeasurementRFMapV1",
    "TrainableMeasurementRFMapV1", "TrainableMaskedObservationPlateV1",
    "TrainableMaskedObservationPlateConfigV1", "TrackBPosteriorV1",
    "BoundTrackBPosteriorReceiptV1",
    "TrackBNativeEnsemblePosteriorV1", "build_native_ensemble_posterior",
    "validate_native_ensemble_posterior",
    "TrackBNativeProjectiveSensitivityV1", "build_native_projective_sensitivity",
    "build_bound_native_projective_sensitivity", "build_native_projective_forward_instance",
    "native_projective_forward",
    "native_projective_covariance", "verify_native_projective_sensitivity_companions",
    "Native_state_ensemble", "build_native_state_ensemble", "verify_native_state_companions",
    "Native_state_sensitivity", "build_native_state_sensitivity",
    "verify_native_state_sensitivity_companions", "describe_native_state_context",
    "TrackBArtifactPublicationV1", "publish_new_bytes", "publish_new_bytes_with_state",
    "TrackBActionfulSourceArtifactBundleV1", "TrackBSourceArtifactFileV1",
    "IdealProjectorDescriptorV2", "RawRFCommandDescriptorV1", "RealCSMSupportV1",
    "MetricProjectionContractV1", "RealRSMSupportV1", "CSM_GEOMETRY_STATUSES", "RSM_SUPPORT_STATUSES",
    "BoundTrackBAppliedCandidateEvaluationV1", "BoundTrackBPosteriorEvaluationSetV1",
    "PosteriorCandidateEvaluationV1", "TrackBRegisteredTryableRecoveryEnvelopeV1",
    "V10TrackBRealFDRFBridgeV1",
    "BoundTrackBForwardDrawSetV1",
    "build_bound_trackb_forward_draw_set", "forward_callable_identity_sha",
    "TrackBOuterFoldPlanV1", "TrackBRealDataAdmissionManifestV1", "TrackBRealFoldAdmissionV1",
    "TrackBRealStaticDatasetV1", "open_trackb_v24_real_static_fold",
    "trackb_real_static_collate", "trackb_real_static_evaluation_collate",
    "TrackBRealStaticLossWeightsV1",
    "TrackBFixedVisibleMarginalBaselineEvaluationV1",
    "build_trackb_fixed_visible_marginal_prediction",
    "evaluate_trackb_fixed_visible_marginal_baseline",
    "TrackBPostPilotDiagnosticRecordV1",
    "evaluate_trackb_full_fold_postpilot_diagnostics",
    "TrackBV240OperationMetadataSidecarV1", "VerifiedOperationMetadataV1",
    "build_v240_operation_metadata_sidecar", "load_verified_v240_operation_metadata_sidecar",
    "load_verified_v240_operation_metadata_sidecar_bytes",
    "write_v240_operation_metadata_sidecar",
    "HISTORICAL_R6_SOURCE_ACCESS_PROFILE", "TrackBV240HistoricalR6ReadOnlySidecarV2",
    "build_historical_r6_read_only_sidecar", "load_verified_historical_r6_read_only_sidecar",
    "write_historical_r6_read_only_sidecar",
    "trackb_masked_real_static_objective", "TrackBRealRuntimeConfigV1",
    "TrackBRealRuntimeIdentityV1", "TrackBLongTrainingAuthorizationV1", "TrackBRealTrainingRuntimeV1",
    "TrackBRuntimeEnvironmentV1",
    "seed_trackb_runtime", "trackb_model_state_sha", "TrackBRealBatchPlanV1",
    "TrackBThreeSeedEnsemblePlanV1", "RFCoordinateSpecV1",
    "TrackBReadOnlyPilotArtifactAuditV1", "TrackBReadOnlyPilotCheckpointAuditV1",
    "TrackBReadOnlyPilotReplayV1", "audit_trackb_bounded_pilot_artifacts",
    "load_trackb_read_only_pilot_checkpoint", "load_trackb_read_only_pilot_checkpoint_bytes",
    "RegisteredRFMappingCoreV1", "RegisteredSingularSupportV1",
    "solve_registered_rf_mapping",
    "RFToAWGCoordinateBindingV1", "RegisteredRFToAWGMappingV1",
    "RegisteredRFPreviewProviderV1",
    "preview_registered_rf_action",
    "Unequal_concentration_config", "unequal_concentration_four_cell_nll",
    "unequal_concentration_zero_fallback", "unequal_nested_four_cell_moments",
    "Unequal_concentration_math_audit_config", "audit_unequal_concentration_math",
    "independent_beta_binomial_log_pmf",
    "Unequal_concentration_point_overlay", "build_unequal_concentration_point_overlay",
    "load_unequal_concentration_point_overlay_bytes",
    "validate_unequal_concentration_point_artifact",
    "verify_unequal_concentration_point_overlay",
    "build_unequal_concentration_pit_plan",
    "execute_unequal_concentration_pit_plan",
    "validate_unequal_concentration_pit_plan",
    "Unequal_concentration_checkpoint_binding",
    "build_unequal_concentration_checkpoint_binding",
    "load_unequal_concentration_checkpoint_binding_bytes",
    "validate_unequal_concentration_checkpoint_binding",
    "verify_unequal_concentration_checkpoint_binding",
    "COMPARISON_METRICS", "EFFECTIVE_FAMILY_CONTEXT_V3_VARIANT", "LEGACY_V1_VARIANT",
    "PROTOCOL_SCHEMA", "REGISTERED_MODEL_VARIANTS", "SUPPORTED_VARIANTS",
    "TrackBModelComparisonProtocolV1", "TrackBModelVariantSpecV1",
    "build_recommended_trackb_model_protocol", "load_trackb_model_protocol",
    "TrackBInverseTransformerConfigV1", "TrackBInverseTransformerDependencyV1",
    "TrackBInverseTransformerV1", "measure_inverse_transformer_dependency",
    "project_bounded_gauge_action", "BOUNDED_GAUGE_PROJECTION", "RISK_NOT_TRAINED",
    "TrackBV24TenRoundPilotConfigV1", "build_trackb_ten_round_v24_plan",
    "generate_trackb_ten_round_v24_dataset", "run_trackb_v24_ten_round_pilot",
    "TRAIN_FIT_DIAGNOSTIC_SCHEMA", "TrackBFrozenTrainFitDiagnosticConfigV1",
    "run_trackb_frozen_train_fit_diagnostic",
    "RECOMPUTABLE_TRAIN_DIAGNOSTIC_SCHEMA", "ROW_EQUAL_METRIC_AGGREGATION",
    "TrackBRecomputableTrainDiagnosticConfigV1", "run_trackb_recomputable_train_diagnostic",
    "FROZEN_GRADIENT_DIAGNOSTIC_SCHEMA", "TrackBFrozenGradientDiagnosticConfigV1",
    "run_trackb_frozen_gradient_diagnostic",
    "LEDGER_ORDER",
    "TrackBGuidanceObservationV1", "TrackBLogicalGuidanceV1",
    "TrackBTenRoundGuidanceRecordV1", "TrackBTenRoundGuidanceSessionV1",
    "TrackBV24GuidanceSourceBindingV1", "write_ten_round_guidance_session",
    "TrackBCandidateActionScoreV1", "evaluate_trackb_model_candidate_actions",
    "is_materially_negative_trackb_gain", "score_trackb_candidate_action",
    "TrackBActionScaleTailPolicyV1", "TrackBActionScaleTailPolicySummaryV1",
    "TrackBActionScaleTailPolicyEvaluationV1", "TrackBPolicyEvaluationBindingV1",
    "TrackBResearchOnlyPolicyChoiceV1", "TRAIN_SCREEN_ROLE", "FROZEN_VALIDATION_ROLE",
    "choose_research_only_conservative_policy", "evaluate_trackb_action_scale_tail_policies",
    "TrackBCandidateCriticBatchV1", "TrackBCandidateCriticConfigV1", "TrackBCandidateCriticOutputV1",
    "TrackBCandidateCriticV1", "candidate_critic_loss",
    "TrackBCandidateTransitionLabelV1",
    "RISK_CALIBRATED_CERTIFIED", "SELECTIVE_RISK_NOT_TRAINED",
    "TrackBSelectiveRiskDecisionV1", "TrackBSelectiveRiskQualificationV1",
    "clopper_pearson_one_sided_upper_bound", "fail_closed_selective_risk_decision",
    "selective_risk_policy_sha",
    "TrackBSharedRFEncoderConfigV1", "TrackBSharedRFEncoderV1", "TrackBSharedRFInputsV1",
    "TrackBSystemTaskHeadConfigV1", "TrackBSystemTaskHeadsV1",
]
