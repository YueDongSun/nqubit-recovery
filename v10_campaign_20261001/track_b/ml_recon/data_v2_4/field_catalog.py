"""Exhaustive catalog of fields persisted by V2.4 writers."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

from .enums import Authority, Identifiability, StorageTier, Visibility
from .schema_types import FieldSpec


def _field(store: str, name: str, dtype: str, shape: str, *,
           authority: Authority, visibility: Visibility,
           identifiability: Identifiability, unit: str = "dimensionless",
           required: bool = False, mask: str = "row_valid_mask",
           tier: StorageTier = StorageTier.CORE, static: bool = False,
           compression: str = "profiled", provenance: Tuple[str, ...] = ("synthetic",),
           authority_by_provenance: Dict[str, str] | None = None) -> FieldSpec:
    return FieldSpec(
        field_path=f"{store}/{name}", store=store, dtype=dtype, shape_rule=shape,
        unit=unit, required=required, mask_path=f"{store}/{mask}",
        authority=authority.value, visibility=visibility.value,
        identifiability=identifiability.value, storage_tier=tier.value,
        static_or_dynamic="static" if static else "dynamic",
        compression_policy=compression, provenance_applicability=provenance,
        authority_by_provenance=authority_by_provenance or {},
    )


def _catalog() -> Iterable[FieldSpec]:
    measured = Authority.MEASURED_OBSERVATION
    exact = Authority.EXACT_BY_CONSTRUCTION
    derived = Authority.DETERMINISTIC_DERIVED
    weak = Authority.WEAK_ESTIMATE
    deployable = Visibility.DEPLOYABLE_INPUT
    derived_input = Visibility.DERIVED_DEPLOYABLE_INPUT
    teacher = Visibility.PRIVILEGED_TEACHER_ONLY
    evaluation = Visibility.EVALUATION_ONLY
    provenance_only = Visibility.PROVENANCE_ONLY
    identifiable = Identifiability.IDENTIFIABLE_UNDER_THIS_VIEW
    equivalence = Identifiability.EQUIVALENCE_CLASS_ONLY
    hidden = Identifiability.NOT_IDENTIFIABLE
    both_authority = {
        "real": measured.value,
        "synthetic": exact.value,
    }

    for store, provenance in (("observation", ("both",)), ("truth", ("synthetic",)),
                              ("derived", ("both",)), ("cell_static", ("both",))):
        yield _field(store, "row_valid_mask", "bool", "[]", authority=exact,
                     visibility=provenance_only, identifiability=identifiable,
                     required=True, mask="row_valid_mask", compression="none",
                     provenance=provenance)
        yield _field(store, "sample_id", "bytes", "[]", authority=exact,
                     visibility=provenance_only, identifiability=identifiable,
                     required=store != "cell_static", compression="none",
                     provenance=provenance)

    observation_specs = {
        "counts": ("uint16|uint32|uint64|float32", "[settings,outcomes_or_cells]", "count"),
        "measurement_probability": ("float32", "[settings,outcomes_or_cells]", "probability"),
        "global_outcome_histogram": ("uint16|uint32|uint64", "[settings,2^N]", "count"),
        "full_outcome_mask": ("bool", "[settings,2^N]", "dimensionless"),
        "basis_descriptors": ("float32", "[settings,descriptor]", "dimensionless"),
        "shots": ("uint32|float32", "[settings]", "count"),
        "setting_uncertainty": ("float32", "[settings]", "probability"),
        "deterministic_expected_count_mask": ("bool", "[]", "dimensionless"),
        "setting_mask": ("bool", "[settings]", "dimensionless"),
        "mode_features": ("float32", "[N,features]", "dimensionless"),
        "mode_mask": ("bool", "[N]", "dimensionless"),
        "physical_adjacency": ("float32", "[N,N]", "dimensionless"),
        "correlation_adjacency": ("float32", "[N,N]", "correlation"),
        "incidence": ("float32", "[N,subsets]", "dimensionless"),
        "certificate_features": ("float32", "[subsets,features]", "dimensionless"),
        "certificate_mask": ("bool", "[subsets]", "dimensionless"),
        "global_multi": ("uint32|uint64", "[9]", "count"),
        "global_multi_mask": ("bool", "[]", "dimensionless"),
        "environment": ("float32", "[7]", "dimensionless"),
        "environment_mask": ("bool", "[7]", "dimensionless"),
        "environment_missing_mask": ("bool", "[7]", "dimensionless"),
        "noisy_measurement_tokens": ("float32", "[tokens,features]", "dimensionless"),
        "measurement_covariance": ("float32", "[tokens,features]", "dimensionless"),
        "csm_cartesian_stats": ("float32", "[statistics]", "dimensionless"),
        "csm_cartesian_covariance": ("float32", "[statistics,statistics]", "covariance"),
        "csm_statistic_valid_mask": ("bool", "[statistics]", "dimensionless"),
        "csm_detection_fraction": ("float32", "[settings]", "probability"),
        "csm_covariance_estimator_kind": ("bytes", "[]", "dimensionless"),
        "csm_dirichlet_alpha": ("float64", "[]", "dimensionless"),
        "csm_observation_contract_sha": ("bytes", "[]", "dimensionless"),
        "csm_observation_contract_kind": ("bytes", "[]", "dimensionless"),
        "csm_statistic_transform_sha": ("bytes", "[]", "dimensionless"),
        "csm_whitening_support_kind": ("bytes", "[]", "dimensionless"),
        "csm_covariance_instance_sha": ("bytes", "[]", "dimensionless"),
        "csm_geometry_source": ("bytes", "[]", "dimensionless"),
        "target_mode_tokens": ("float32", "[N,features]", "dimensionless"),
        "physical_mode_tokens": ("float32", "[N,features]", "dimensionless"),
        "gate_features": ("float32", "[features]", "dimensionless"),
        "history_tokens": ("float32", "[rounds,N,features]", "dimensionless"),
        "history_mask": ("bool", "[rounds]", "dimensionless"),
        "measurement_uniforms_sha": ("bytes", "[]", "dimensionless"),
        "next_measurement_uniforms_sha": ("bytes", "[]", "dimensionless"),
        "subset_parent_setting_id": ("int32", "[tokens]", "dimensionless"),
        "subset_qubits": ("int16", "[tokens,max_subset]", "dimensionless"),
        "subset_order": ("int8", "[tokens]", "dimensionless"),
        "subset_axes": ("int8", "[tokens,max_subset]", "dimensionless"),
        "subset_marginal_counts": ("uint16|uint32|uint64", "[tokens,max_outcomes]", "count"),
        "subset_marginal_mask": ("bool", "[tokens,max_outcomes]", "dimensionless"),
        "subset_shots": ("uint32", "[tokens]", "count"),
        "common_shot_group_id": ("int64", "[tokens]", "dimensionless"),
        "pair_coverage_mask": ("bool", "[N,N]", "dimensionless"),
        "hyperedge_coverage_mask": ("bool", "[subsets]", "dimensionless"),
        "pair_stat_qubits": ("int16", "[pair_axis_tokens,2]", "dimensionless"),
        "pair_stat_axes": ("int8", "[pair_axis_tokens,2]", "dimensionless"),
        "pair_stat_expectation": ("float32", "[pair_axis_tokens]", "correlation"),
        "pair_stat_variance": ("float32", "[pair_axis_tokens]", "variance"),
        "pair_stat_shots": ("uint32", "[pair_axis_tokens]", "count"),
        "pair_stat_coverage_mask": ("bool", "[pair_axis_tokens]", "dimensionless"),
    }
    for name, (dtype, shape, unit) in observation_specs.items():
        mask = name if name.endswith("_mask") else "row_valid_mask"
        visibility = deployable
        if name == "environment":
            mask = "environment_mask"
        elif name == "global_multi":
            mask = "global_multi_mask"
        elif name.startswith("subset_") and name not in ("subset_marginal_mask",):
            mask = "subset_marginal_mask"
        elif name.startswith("pair_stat_") and name != "pair_stat_coverage_mask":
            mask = "pair_stat_coverage_mask"
        elif name in ("measurement_uniforms_sha", "next_measurement_uniforms_sha"):
            visibility = provenance_only
        yield _field(
            "observation", name, dtype, shape, authority=measured,
            visibility=visibility, identifiability=identifiable, unit=unit,
            mask=mask, provenance=("both",), authority_by_provenance=both_authority)

    real_observation_specs = {
        "source_row_index": ("int64", "[]"),
        "signal_counts": ("uint32|uint64", "[operations]"),
        "idler_counts": ("uint32|uint64", "[operations]"),
        "coin_counts": ("uint32|uint64", "[operations]"),
        "read_count_scalar": ("uint32|uint64", "[]"),
        "four_cell_counts": ("uint32|uint64", "[operations,4]"),
        "operation_mask": ("bool", "[operations]"),
        "exposure_s": ("float64", "[]"),
        "exposure_known_mask": ("bool", "[]"),
        "global_multi_scope": ("bytes", "[]"),
        "target_amplitudes_raw": ("float32", "[4]"),
        "target_phases_raw": ("float32", "[4]"),
        "target_mask": ("bool", "[4]"),
        "target_coefficients_re": ("float32", "[4]"),
        "target_coefficients_im": ("float32", "[4]"),
        "target_coefficients_mask": ("bool", "[]"),
        "target_sha": ("bytes", "[]"),
        "training_eligible_mask": ("bool", "[]"),
        "real_action_supervision_mask": ("bool", "[]"),
    }
    for name, (dtype, shape) in real_observation_specs.items():
        mask = name if name.endswith("_mask") else "row_valid_mask"
        if name == "exposure_s":
            mask = "exposure_known_mask"
        elif name.startswith("target_coefficients") and not name.endswith("_mask"):
            mask = "target_coefficients_mask"
        yield _field("observation", name, dtype, shape, authority=measured,
                     visibility=deployable if name != "source_row_index" else provenance_only,
                     identifiability=identifiable, mask=mask, provenance=("real",),
                     unit="second" if name == "exposure_s" else
                          ("count" if "count" in name else "dimensionless"))

    truth_names = {
        "track_a_dlcz_provenance": ("bytes", "[]", "row_valid_mask"),
        "rho_clean_re": ("float64", "[2^N,2^N]", "rho_clean_mask"),
        "rho_clean_im": ("float64", "[2^N,2^N]", "rho_clean_mask"),
        "rho_clean_mask": ("bool", "[]", "rho_clean_mask"),
        "target_p_clean": ("float32", "[settings,outcomes]", "target_measurement_mask"),
        "target_p_noisy": ("float32", "[settings,outcomes]", "target_measurement_mask"),
        "target_measurement_mask": ("bool", "[settings]", "target_measurement_mask"),
        "target_fine": ("int64", "[]", "target_fine_mask"),
        "target_fine_mask": ("bool", "[]", "target_fine_mask"),
        "target_fine_identifiable_mask": ("bool", "[]", "target_fine_identifiable_mask"),
        "target_coarse": ("int64", "[]", "target_coarse_mask"),
        "target_coarse_mask": ("bool", "[]", "target_coarse_mask"),
        "target_pair": ("float32", "[N,N]", "target_pair_mask"),
        "target_pair_mask": ("bool", "[N,N]", "target_pair_mask"),
        "target_partition": ("float32", "[N,N]", "target_partition_mask"),
        "target_partition_mask": ("bool", "[N,N]", "target_partition_mask"),
        "partition_block_ids": ("int16", "[N]", "partition_exact_mask"),
        "partition_exact_mask": ("bool", "[]", "partition_exact_mask"),
        "target_depth": ("int64", "[]", "target_structure_mask"),
        "target_intactness": ("int64", "[]", "target_structure_mask"),
        "target_stretchability": ("int64", "[]", "target_structure_mask"),
        "target_structure_mask": ("bool", "[]", "target_structure_mask"),
        "target_hyperedge": ("float32", "[subsets]", "target_hyperedge_mask"),
        "target_hyperedge_mask": ("bool", "[subsets]", "target_hyperedge_mask"),
        "target_operational_noise": (
            "int64", "[]", "target_operational_noise_mask"),
        "target_operational_noise_mask": (
            "bool", "[]", "target_operational_noise_mask"),
        "target_R1_re": ("float32", "[N,N]", "target_w_mask"),
        "target_R1_im": ("float32", "[N,N]", "target_w_mask"),
        "target_w_mask": ("bool", "[]", "target_w_mask"),
        "target_sector": ("float32", "[3]", "target_w_mask"),
        "target_certificate_status": ("int64", "[]", "target_certificate_mask"),
        "target_certificate_mask": ("bool", "[]", "target_certificate_mask"),
        "target_ood": ("float32", "[]", "row_valid_mask"),
        "negative_control_mask": ("bool", "[]", "negative_control_mask"),
    }
    privileged_truth = {
        "correctable_mechanism", "irrecoverable_mechanism", "mechanism_severity",
        "epsilon_irr", "eta_target", "eta_achieved", "eta_matching_residual",
        "eta_matching_converged", "physical_noise_parameter", "state_infidelity",
        "oracle_action", "clean_coefficients_re", "clean_coefficients_im",
        "current_coefficients_re", "current_coefficients_im", "R1_re", "R1_im",
        "p0", "p1", "p2", "p3_plus", "pair_probability",
        "multi_excitation_backend_mask", "crn_consumed", "target_action",
        "target_recoverability", "target_gain", "target_harm", "target_stop",
        "target_recoverable_class", "target_state_fidelity_gain",
        "oracle_action_mask", "oracle_solver_success", "oracle_solver_status",
        "oracle_objective_before", "oracle_objective_after",
        "oracle_constraint_violation", "analytic_baseline_gain",
        "analytic_baseline_next_measurement",
        "target_next_measurement", "target_noop_next_measurement",
        "target_noop_gain", "history_round_count",
        "target_history_gain", "target_history_gain_mask",
        "true_esm_mode_tokens", "true_partition", "true_mechanism", "true_severity",
        "clean_measurement_tokens", "true_next_measurement", "true_kappa",
        "mechanism_name", "control_mechanism_name", "irrecoverable_mechanism_name",
        "eta_ctrl_target", "eta_irr_target", "eta_ctrl_achieved", "eta_irr_achieved",
        "kappa_inj", "kappa_achieved", "kappa_forward_achieved", "control_parameter", "irrecoverable_parameter",
        "kappa_target", "kappa_component_achieved", "kappa_component_mask",
        "kappa_operational", "kappa_operational_mask",
        "kappa_ctrl_geom", "kappa_ctrl_reg",
        "recoverable_class", "recoverability_mask", "solver_status",
        "d3_train_fit_sha", "d3_heldout_error", "intervention_forward_sha",
        "locked_ood_pair_id",
        "partition_truth_available", "certificate_complete",
        "epistemic_ambiguity", "distribution_ood",
    }
    for name, (dtype, shape, mask) in truth_names.items():
        visibility = evaluation if name.startswith("target_") or name.startswith("partition_") \
            or name == "negative_control_mask" \
            else teacher
        identifiability = identifiable if name in {
            "target_p_clean", "target_p_noisy", "target_fine", "target_coarse"} \
            else equivalence
        yield _field("truth", name, dtype, shape, authority=exact,
                     visibility=visibility, identifiability=identifiability, mask=mask,
                     tier=StorageTier.EXTENDED)
    for name in sorted(privileged_truth):
        dtype = "bytes" if name.endswith("_sha") or name in {
            "mechanism_name", "control_mechanism_name", "irrecoverable_mechanism_name",
            "solver_status", "oracle_solver_status", "locked_ood_pair_id"} else (
            "bool" if name.endswith("_mask") or name in {
            "crn_consumed", "eta_matching_converged", "recoverability_mask",
            "oracle_solver_success", "partition_truth_available",
            "certificate_complete", "epistemic_ambiguity", "distribution_ood"} else (
            "int64" if name in {"recoverable_class", "target_recoverable_class"}
            else ("int16" if name == "history_round_count" else "float32|float64"))
        )
        shape = "[]" if name.endswith("_mask") or name in {
            "epsilon_irr", "eta_target", "eta_achieved", "eta_matching_residual",
            "eta_matching_converged", "physical_noise_parameter", "state_infidelity",
            "p0", "p1", "p2", "p3_plus", "crn_consumed", "target_recoverability",
            "target_gain", "target_harm", "target_stop", "target_noop_gain",
            "target_history_gain",
            "history_round_count", "true_severity", "true_kappa",
            "target_recoverable_class", "target_state_fidelity_gain",
            "kappa_ctrl_geom", "kappa_ctrl_reg", "recoverable_class",
            "recoverability_mask", "oracle_action_mask", "oracle_solver_success",
            "oracle_objective_before", "oracle_objective_after",
            "oracle_constraint_violation", "analytic_baseline_gain",
            "partition_truth_available", "certificate_complete",
            "epistemic_ambiguity", "distribution_ood"} else "[variable]"
        if name in {
                "mechanism_name", "solver_status", "oracle_solver_status",
                "locked_ood_pair_id",
                "d3_train_fit_sha", "d3_heldout_error", "intervention_forward_sha"}:
            shape = "[]"
        yield _field("truth", name, dtype, shape, authority=exact,
                     visibility=teacher, identifiability=hidden,
                     tier=StorageTier.EXTENDED)

    derived_specs = {
        "analytic_action": ("float32", "[N,2]", "analytic_action_mask", derived),
        "analytic_action_raw": ("float32", "[N,2]", "analytic_action_raw_mask", derived),
        "analytic_action_denoised": ("float32", "[N,2]", "analytic_action_denoised_mask", derived),
        "analytic_action_mask": ("bool", "[variable]", "analytic_action_mask", derived),
        "analytic_action_raw_mask": ("bool", "[variable]", "analytic_action_raw_mask", derived),
        "analytic_action_denoised_mask": ("bool", "[variable]", "analytic_action_denoised_mask", derived),
        "denoised_measurement_tokens": ("float32", "[tokens,features]", "row_valid_mask", derived),
        "covariance": ("float32", "[tokens,features]", "row_valid_mask", derived),
        "esm_mode_tokens": ("float32", "[N,features]", "row_valid_mask", derived),
        "classifier_posterior": ("float32", "[features]", "classifier_mask", derived),
        "classifier_mask": ("bool", "[]", "classifier_mask", derived),
        "rho8": ("float32", "[8,2^N,2^N]", "rho8_mask", weak),
        "rho8_re": ("float32", "[8,2^N,2^N]", "rho8_mask", weak),
        "rho8_im": ("float32", "[8,2^N,2^N]", "rho8_mask", weak),
        "rho8_mask": ("bool", "[8]", "rho8_mask", weak),
        "rho8_cross_fit_nll": ("float64", "[8]", "rho8_cross_fit_mask", weak),
        "rho8_cross_fit_mask": (
            "bool", "[8]", "rho8_cross_fit_mask", weak),
        "count_mle": ("float32", "[2^N,2^N]", "count_mle_mask", weak),
        "count_mle_re": ("float32", "[N,N]", "count_mle_mask", weak),
        "count_mle_im": ("float32", "[N,N]", "count_mle_mask", weak),
        "count_mle_mask": ("bool", "[]", "count_mle_mask", weak),
        "count_mle_converged": (
            "bool", "[]", "count_mle_mask", weak),
        "count_mle_likelihood_gain": (
            "float64", "[]", "count_mle_mask", weak),
        "teacher_soft_action": (
            "float32", "[N,2]", "teacher_distillation_mask", derived),
        "teacher_soft_recoverability": (
            "float32", "[]", "teacher_distillation_mask", derived),
        "teacher_soft_gain_quantiles": (
            "float32", "[5]", "teacher_distillation_mask", derived),
        "teacher_soft_harm_probability": (
            "float32", "[]", "teacher_distillation_mask", derived),
        "teacher_soft_next_measurement": (
            "float32", "[N,2]", "teacher_distillation_mask", derived),
        "teacher_checkpoint_sha": ("bytes", "[]", "row_valid_mask", derived),
        "teacher_distillation_mask": (
            "bool", "[]", "teacher_distillation_mask", derived),
        "classifier_config_sha": ("bytes", "[]", "row_valid_mask", derived),
        "normalization_sha": ("bytes", "[]", "row_valid_mask", derived),
        "forward_sha": ("bytes", "[]", "row_valid_mask", derived),
        "data_build_sha": ("bytes", "[]", "row_valid_mask", derived),
    }
    for name, (dtype, shape, mask, authority) in derived_specs.items():
        yield _field("derived", name, dtype, shape, authority=authority,
                     visibility=derived_input, identifiability=equivalence, mask=mask,
                     tier=StorageTier.EXTENDED, provenance=("both",))

    static_specs = {
        "basis_descriptors": ("float32", "[settings,descriptor]"),
        "operation_ids": ("bytes", "[operations]"),
        "setting_mask": ("bool", "[settings]"),
        "shots": ("uint32|float32", "[settings]"),
        "physical_adjacency": ("float32", "[N,N]"),
        "incidence": ("float32", "[N,subsets]"),
        "mode_features": ("float32", "[N,features]"),
        "mode_mask": ("bool", "[N]"),
        "measurement_schedule_sha": ("bytes", "[]"),
        "operation_registry_sha": ("bytes", "[]"),
        "command_table_sha": ("bytes", "[]"),
    }
    for name, (dtype, shape) in static_specs.items():
        yield _field("cell_static", name, dtype, shape, authority=exact,
                     visibility=deployable, identifiability=identifiable,
                     static=True, compression="profiled", provenance=("both",))


def all_field_specs() -> Tuple[FieldSpec, ...]:
    fields = tuple(_catalog())
    paths = [field.field_path for field in fields]
    if len(paths) != len(set(paths)):
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        raise RuntimeError(f"duplicate V2.4 field specifications: {duplicates}")
    for field in fields:
        field.validate()
    return fields


def real_v4_compact_field_specs() -> Tuple[FieldSpec, ...]:
    """Versioned extension fields that do not mutate the sealed V2.4 schema SHA."""
    derived = Authority.DETERMINISTIC_DERIVED
    visibility = Visibility.DERIVED_DEPLOYABLE_INPUT
    identifiability = Identifiability.EQUIVALENCE_CLASS_ONLY
    specifications = {
        "real_v4_sector_probabilities": ("float32", "[4]", "real_v4_sector_mask"),
        "real_v4_sector_entropy": ("float32", "[]", "real_v4_sector_mask"),
        "real_v4_sector_confidence": ("float32", "[]", "real_v4_sector_mask"),
        "real_v4_sector_mask": ("bool", "[]", "real_v4_sector_mask"),
        "real_v4_chi_summary": ("float32", "[3]", "real_v4_sector_mask"),
        "real_v4_gsi_proxy_provenance": ("bytes", "[]", "real_v4_context_mask"),
        "real_v4_selected_measurement_manifest_sha": (
            "bytes", "[]", "real_v4_context_mask"),
        "real_v4_w_posterior_compact": ("float32", "[N,4]", "real_v4_w_mask"),
        "real_v4_w_mask": ("bool", "[]", "real_v4_w_mask"),
        "real_v4_esm_mode_summary": ("float32", "[N,4]", "real_v4_context_mask"),
        "real_v4_csm_node_summary": ("float32", "[N,4]", "real_v4_context_mask"),
        "real_v4_csm_global_summary": ("float32", "[8]", "real_v4_context_mask"),
        "real_v4_context_confidence": ("float32", "[]", "real_v4_context_mask"),
        "real_v4_context_mask": ("bool", "[]", "real_v4_context_mask"),
    }
    return tuple(_field(
        "derived", name, dtype, shape, authority=derived,
        visibility=visibility, identifiability=identifiability, mask=mask,
        tier=StorageTier.CORE, provenance=("both",))
        for name, (dtype, shape, mask) in specifications.items())
