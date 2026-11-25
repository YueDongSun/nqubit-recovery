"""R16 physical Gate specifications; legacy specifications are not mutated."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from ml_recon.physics_hybrid_v10.contracts import sha256_of

from ..config import GateConfig
from .gate_specs import GATE_SPEC_TEMPLATES
from .physical_plan import AUTHORITY, PHYSICAL_PROGRAM_NAMESPACE


PHYSICAL_EVIDENCE_SCHEMA = "V10PhysicalGateEvidenceV2"
PHYSICAL_SPEC_SCHEMA = "V10PhysicalGateSpecV3"
PHYSICAL_IDENTITY_FIELDS = (
    "physical_budget_registry_sha",
    "operation_registry_sha",
    "semantic_abi_sha",
    "protocol_sha",
)
P0_SEMANTIC_BINDING_SCHEMA = "V10P0ActiveSemanticBindingsV5"
P0_CLASSIFIER_OBJECTIVE_BINDING = {
    "objective_supervision_schema": "V10PhysicalObjectiveSupervisionV1",
    "output_audit_schema": "V10ClassifierOutputSliceAuditV4",
    "sector_training_scope": "K1_VERSUS_REST_AGGREGATE",
    "k2_vs_ge3_trainable": False,
    "population_target_relation": "trace_normalized_target_R1_diagonal",
    "physical_calibration_evidence_schema": "V10HeldOutCalibrationEvidenceV6",
    "physical_calibration_report_schema": "V10HeldOutCalibrationReportV6",
    "physical_calibration_admission_schema": "V10CalibrationAdmissionReceiptV4",
    "physical_output_manifest_schema": "V10PhysicalPublishedOutputTrainingRegistryV3",
    "physical_k1_evidence_source": "TRACE_BOUND_AGGREGATE",
    "physical_confidence_supported": False,
}
P0_SEMANTIC_SOURCE_PATHS = (
    "ml_recon/physics_hybrid_v10/four_cell_layout.py",
    "ml_recon/physics_hybrid_v10/evidence_file_io.py",
    "ml_recon/physics_hybrid_v10/evidence_snapshot.py",
    "ml_recon/v10_recoverability_foundry/training/physical_gates.py",
    "ml_recon/configs/v10_track_a_physical_budget_v2.json",
    "ml_recon/physics_hybrid_v10/training/physical_program.py",
    "ml_recon/physics_hybrid_v10/training/physical_admission.py",
    "ml_recon/physics_hybrid_v10/training/review_source_manifest.py",
    "ml_recon/physics_hybrid_v10/training/physical_dataset.py",
    "ml_recon/v10_recoverability_foundry/training/physical_plan.py",
    "ml_recon/physics_hybrid_v10/physical_classifier.py",
    "ml_recon/physics_hybrid_v10/classifier_n_full.py",
    "ml_recon/physics_hybrid_v10/classifier_output_audit.py",
    "ml_recon/physics_hybrid_v10/published_output_registry.py",
    "ml_recon/physics_hybrid_v10/physical_objective_contract.py",
    "ml_recon/physics_hybrid_v10/physical_belief_contract.py",
    "ml_recon/physics_hybrid_v10/checkpoint_schema.py",
    "ml_recon/physics_hybrid_v10/training_runtime.py",
    "ml_recon/physics_hybrid_v10/train_classifier.py",
    "ml_recon/physics_hybrid_v10/control_coordinate_layout.py",
    "ml_recon/physics_hybrid_v10/belief_geometry.py",
    "ml_recon/physics_hybrid_v10/w_structure_head.py",
    "ml_recon/v10_recoverability_foundry/training/physical_supervision.py",
    "ml_recon/physics_hybrid_v10/csm_profile_contract.py",
    "ml_recon/physics_hybrid_v10/calibration_admission.py",
    "ml_recon/physics_hybrid_v10/calibration_update_evidence.py",
    "ml_recon/physics_hybrid_v10/training/physical_feedback.py",
    "ml_recon/v10_recoverability_foundry/physics/aod_feedback_forward.py",
    "ml_recon/v10_recoverability_foundry/contracts/memory_event_reducer.py",
    "ml_recon/v10_recoverability_foundry/contracts/memory_evidence.py",
    "ml_recon/v10_recoverability_foundry/runtime/memory_event_reducer.py",
    "ml_recon/v10_recoverability_foundry/contracts/memory_queue.py",
    "ml_recon/v10_recoverability_foundry/physics/memory_queue.py",
    "ml_recon/v10_recoverability_foundry/physics/memory_qualification_evaluator.py",
    "ml_recon/v10_recoverability_foundry/physics/memory_transition.py",
    "ml_recon/v10_recoverability_foundry/evaluation/memory_fusion.py",
)
ADDITIONAL_CHECKS = {
    "P0": (
        "physical_budget_conservation",
        "aod_w_state_contract",
        "xy_command_one_effective_phase",
        "operation_registry_binding",
        "success_multiplicity_not_fixed",
        "mu_succ_derived",
        "csm_target_independence",
        "final_action_semantics",
        "stream_only_ownership",
        "memory_identity_clock_and_consumption",
        "memory_compatibility_and_capacity",
        "memory_future_truth_excluded",
        "memory_finite_window_and_censoring",
        "memory_factors_and_parameter_sources",
        "memory_extension_preserves_core_abi",
        "r1_covariance_consumer_alignment",
        "active_classifier_nll_layout_identity",
        "csm_estimand_profile_admission",
        "memory_shared_reducer_replay_causality",
        "calibration_raw_evidence_and_scope",
        "pro_response_and_commit_boundary_binding",
    ),
    "E0": (
        "cpu_cuda_registered_parity",
        "stream_recipe_source_identity",
        "four_cell_conservation",
    ),
    "E1": (
        "dense_w_probability_parity",
        "actual_registry_parity",
        "projector_descriptor_parity",
        "noise_forward_parity",
        "csm_jacobian_gram_parity",
        "bounded_oracle_parity",
    ),
    "E2": (
        "fusion_accounting_oracle_ceiling",
        "final_bounded_action_semantics",
        "memory_parameter_mapping_accepted",
        "memory_exact_queue_reference_parity",
        "memory_finite_window_and_infinite_memory_ceilings",
    ),
    "E3": (
        "every_output_slice_supervised",
        "checkpoint_reload_exact",
        "negative_controls_operational",
    ),
    "S1": ("held_out_severity", "held_out_projector_family", "separate_system_reports"),
    "S2": (
        "legal_partition_coverage",
        "hard_negative_transitivity",
        "hyperedge_evidence",
    ),
    "S3": (
        "raw_profiled_posterior_separated",
        "subspace_principal_angle_coverage",
        "spectrum_calibration",
        "kappa_epsilon_coverage",
        "csm_confidence_calibration",
    ),
    "S4": (
        "bounded_action_semantics",
        "esm_csm_factorial_complete",
        "stop_abstain_calibration",
    ),
    "S5": (
        "fixed_quality_required_budget_curve",
        "measurement_dilution_difference_in_differences",
        "quality_preserving_budget_growth",
        "fusion_useful_rate_curve",
        "rate_quality_fusion_pareto",
        "thresholds_frozen_before_confirmation",
        "independent_threshold_registration",
        "model_sample_scaling_separate",
        "memory_m0_m1_m2_m3_matched_panels",
        "memory_initial_inventory_and_censoring",
        "memory_accepted_rate_grouped_uncertainty",
        "memory_parameter_and_port_mapping_bound",
    ),
}


def physical_gate_spec(gate_id: str, gate_config: GateConfig | None = None) -> dict:
    """Return a fresh, hash-bound specification in the physical namespace."""
    thresholds = gate_config or GateConfig()
    thresholds.validate()
    if gate_id not in GATE_SPEC_TEMPLATES:
        raise ValueError(f"unknown physical Gate {gate_id}")
    template = deepcopy(GATE_SPEC_TEMPLATES[gate_id])
    checks = tuple(template["required_checks"]) + ADDITIONAL_CHECKS[gate_id]
    rules = template["metric_rules"]
    if gate_id == "S5":
        # The non-zero effect is registered in an immutable S5 receipt rather
        # than silently falling back to a configuration default.
        rules = (
            ("model_value_over_baseline_lcb95", "strict_minimum", 0.0),
            ("confirmation_gain_lcb95", "strict_minimum", 0.0),
        )
    resolved_rules = []
    threshold_values = asdict(thresholds)
    for name, comparator, reference in rules:
        resolved_rules.append(
            {
                "metric": name,
                "comparator": comparator,
                "threshold": float(
                    threshold_values[reference]
                    if isinstance(reference, str)
                    else reference
                ),
            }
        )
    payload = {
        "schema": PHYSICAL_SPEC_SCHEMA,
        "namespace": PHYSICAL_PROGRAM_NAMESPACE,
        "gate_id": gate_id,
        "authorization_scope": f"track_a_physical_v2_{gate_id.lower()}_gate_closure",
        "evaluator_schema": f"V10Physical{gate_id}EvidenceV2",
        "required_checks": checks,
        "metric_rules": tuple(resolved_rules),
        "minimum_seed_count": template["minimum_seed_count"],
        "minimum_sample_rung": template["minimum_sample_rung"],
        "maximum_sample_rung": min(template["maximum_sample_rung"], 200000),
        "require_checkpoints": template["require_checkpoints"],
        "require_dataset_ranges": template["require_dataset_ranges"],
        "require_dataset_identity": template["require_dataset_identity"],
        "physical_identity_fields": PHYSICAL_IDENTITY_FIELDS,
        "threshold_registration_required": gate_id == "S5",
        "memory_contract_revision": "R18_SYNTHETIC_EVENT_QUEUE_AND_SEPARATE_EXTENSION",
        "p0_memory_scope": "CONTRACT_IMPLEMENTATION_ONLY_NOT_PARAMETER_ACCEPTANCE_OR_S5",
        "gate_thresholds": threshold_values,
        "authority": dict(AUTHORITY),
    }
    return {**payload, "gate_spec_sha": sha256_of(payload)}
