"""Named field-visibility views; training code may not assemble ad-hoc fields."""

from __future__ import annotations

from typing import Dict, FrozenSet

from .validators import validate_view_keys


CLASSIFIER_INPUT = frozenset({
    "counts", "basis_descriptors", "shots", "setting_uncertainty", "setting_mask",
    "global_outcome_histogram", "full_outcome_mask",
    "subset_parent_setting_id", "subset_qubits", "subset_order", "subset_axes",
    "subset_marginal_counts", "subset_marginal_mask", "subset_shots",
    "common_shot_group_id", "pair_coverage_mask", "hyperedge_coverage_mask",
    "pair_stat_qubits", "pair_stat_axes", "pair_stat_expectation",
    "pair_stat_variance", "pair_stat_shots", "pair_stat_coverage_mask",
    "mode_features", "mode_mask", "physical_adjacency", "correlation_adjacency",
    "incidence", "certificate_features", "certificate_mask", "global_multi",
    "global_multi_mask", "environment", "environment_mask",
})
GENERAL_N_CLASSIFIER_INPUT = CLASSIFIER_INPUT - frozenset({
    "global_outcome_histogram", "full_outcome_mask",
})
CLASSIFIER_TRUTH = frozenset({
    "target_p_clean", "target_p_noisy", "target_measurement_mask",
    "target_fine", "target_fine_mask", "target_fine_identifiable_mask",
    "target_coarse", "target_coarse_mask", "target_pair", "target_pair_mask", "target_partition",
    "target_partition_mask", "target_depth", "target_intactness",
    "target_stretchability", "target_structure_mask", "target_R1_re",
    "target_R1_im", "target_w_mask", "target_sector", "target_hyperedge",
    "target_hyperedge_mask",
    "target_operational_noise", "target_operational_noise_mask",
    "target_certificate_status", "target_certificate_mask", "target_ood",
})
FEEDBACK_CONTEXT = frozenset({
    "target_mode_tokens", "physical_mode_tokens", "gate_features", "mode_mask",
    "history_tokens", "history_mask",
})
FEEDBACK_BASE = FEEDBACK_CONTEXT | frozenset({
    "noisy_measurement_tokens", "measurement_covariance"})
FEEDBACK_TARGETS = frozenset({
    "target_action", "target_recoverability", "target_gain", "target_harm",
    "target_recoverable_class", "oracle_action_mask",
    "target_stop", "target_next_measurement", "target_noop_next_measurement",
    "target_noop_gain", "target_history_gain", "target_history_gain_mask",
    "recoverability_mask",
})
FEEDBACK_TRUTH = FEEDBACK_TARGETS | frozenset({
    "oracle_action", "analytic_baseline_gain",
    "analytic_baseline_next_measurement",
})
DERIVED_DENOISE = frozenset({"denoised_measurement_tokens", "covariance"})
DERIVED_CLASS = frozenset({"esm_mode_tokens"})
DERIVED_CONTROL = frozenset({
    "analytic_action", "analytic_action_raw", "analytic_action_denoised",
})
ORACLE_INPUT = frozenset({
    "clean_measurement_tokens", "true_mechanism", "true_severity",
    "true_esm_mode_tokens", "true_partition", "oracle_action",
    "current_coefficients_re", "current_coefficients_im",
})
TEACHER_INPUT = ORACLE_INPUT | FEEDBACK_BASE | frozenset({
    "true_next_measurement", "true_kappa", "epsilon_irr",
})
REAL_COUNT_INPUT = frozenset({
    "signal_counts", "idler_counts", "coin_counts", "read_count_scalar",
    "four_cell_counts", "operation_mask", "exposure_s",
    "exposure_known_mask", "global_multi", "global_multi_mask",
    "global_multi_scope", "target_amplitudes_raw", "target_phases_raw",
    "target_mask", "target_coefficients_re", "target_coefficients_im",
    "target_coefficients_mask", "target_sha", "environment",
    "environment_mask", "environment_missing_mask",
    "training_eligible_mask", "real_action_supervision_mask",
})
REAL_V4_COMPACT_CONTEXT = frozenset({
    "real_v4_sector_probabilities", "real_v4_sector_entropy",
    "real_v4_sector_confidence", "real_v4_sector_mask", "real_v4_chi_summary",
    "real_v4_gsi_proxy_provenance", "real_v4_selected_measurement_manifest_sha",
    "real_v4_w_posterior_compact", "real_v4_w_mask",
    "real_v4_esm_mode_summary", "real_v4_csm_node_summary",
    "real_v4_csm_global_summary", "real_v4_context_confidence",
    "real_v4_context_mask",
})
REAL_V4_FORBIDDEN_LONG_RUNG_FIELDS = frozenset({
    "real_v4_w_posterior_draws", "real_v4_dense_csm",
    "real_v4_dense_covariance", "real_v4_action_supervision",
})


def validate_real_v4_lowdisk_fields(fields: FrozenSet[str] | set[str]) -> None:
    leaked = REAL_V4_FORBIDDEN_LONG_RUNG_FIELDS & set(fields)
    if leaked:
        raise ValueError(f"dense or action V4 fields cannot persist in long-rung shards: {sorted(leaked)}")
    if set(fields) - REAL_V4_COMPACT_CONTEXT:
        raise ValueError("unregistered V4 compact field requested")


VIEW_FIELDS: Dict[str, FrozenSet[str]] = {
    "real_count_input": REAL_COUNT_INPUT,
    "real_v4_compact_context": REAL_V4_COMPACT_CONTEXT,
    "classifier_input": CLASSIFIER_INPUT,
    "classifier_truth": CLASSIFIER_TRUTH,
    "classifier_real_input": CLASSIFIER_INPUT,
    "target_conditioned_law_input": FEEDBACK_BASE,
    "feedback_student_base": FEEDBACK_BASE | frozenset({"analytic_action_raw"}),
    "feedback_teacher": FEEDBACK_BASE | FEEDBACK_TRUTH | DERIVED_DENOISE
                        | DERIVED_CLASS | DERIVED_CONTROL | TEACHER_INPUT,
    "feedback_hardware": FEEDBACK_BASE | DERIVED_DENOISE | DERIVED_CLASS | DERIVED_CONTROL,
    "ablation_A0": FEEDBACK_BASE | frozenset({"analytic_action_raw"}),
    "ablation_A1": FEEDBACK_CONTEXT | DERIVED_DENOISE
                   | frozenset({"analytic_action_denoised"}),
    "ablation_A2": FEEDBACK_BASE | DERIVED_CLASS
                   | frozenset({"analytic_action_raw"}),
    "ablation_A3": FEEDBACK_CONTEXT | DERIVED_DENOISE | DERIVED_CLASS
                   | frozenset({"analytic_action_denoised"}),
    "ablation_A4": FEEDBACK_CONTEXT | ORACLE_INPUT,
    "audit_full": CLASSIFIER_INPUT | CLASSIFIER_TRUTH | FEEDBACK_BASE | FEEDBACK_TRUTH
                  | DERIVED_DENOISE | DERIVED_CLASS | DERIVED_CONTROL | TEACHER_INPUT,
}


def fields_for_view(view_name: str) -> FrozenSet[str]:
    if view_name not in VIEW_FIELDS:
        raise ValueError(f"unknown V2.4 loader view: {view_name}")
    fields = VIEW_FIELDS[view_name]
    if view_name in ("classifier_input", "classifier_real_input", "feedback_student_base",
                     "feedback_hardware", "ablation_A0", "ablation_A1", "ablation_A2",
                     "ablation_A3"):
        validate_view_keys(view_name, fields)
    return fields


def classifier_input_for_family(dataset_family: str) -> FrozenSet[str]:
    """Bind classifier inputs to the family's registered measurement contract."""
    family = str(dataset_family)
    if "_negative_control_" in family:
        family = family.split("_negative_control_", 1)[0]
    elif family.endswith("_negative_control"):
        family = family[:-len("_negative_control")]
    if family.endswith("_locked_ood"):
        family = family[:-len("_locked_ood")]
    if family == "small_n_family":
        return CLASSIFIER_INPUT
    if family == "general_n_structure":
        return GENERAL_N_CLASSIFIER_INPUT
    raise ValueError(f"no classifier input contract for dataset family {family}")


def view_visibility_matrix() -> Dict[str, Dict[str, bool]]:
    universe = sorted(set().union(*VIEW_FIELDS.values()))
    return {view: {field: field in fields for field in universe}
            for view, fields in VIEW_FIELDS.items()}
