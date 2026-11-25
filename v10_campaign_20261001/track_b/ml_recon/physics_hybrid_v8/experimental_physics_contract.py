"""Hash-bound V8 experimental-physics decisions recovered from operator ASK records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


EXPERIMENTAL_PHYSICS_CONTRACT_SCHEMA = (
    "physics_hybrid_v8_experimental_physics_contract_v3"
)
DEFAULT_EXPERIMENTAL_PHYSICS_CONTRACT_PATH = Path(
    "ml_recon/configs/physics_hybrid_v8_experimental_physics.json"
)
SUPPORTED_SYSTEMS = ("3qubit", "4qubit")
PRIMARY_REPORT_METRICS = (
    "calibrated_raw_core_fidelity",
    "gaussian_denoised_core_fidelity",
    "usable_entanglement_yield",
)
NEGATIVE_RESIDUAL_ABLATIONS = ("signed_gaussian", "censored_gaussian")


@dataclass(frozen=True)
class V8ExperimentalPhysicsContractV1:
    """Validated immutable payload and byte identity for training artifacts."""

    payload: Dict[str, Any]
    file_sha256: str
    semantic_sha256: str
    path: str

    def validate(self) -> None:
        validate_experimental_physics_contract(self.payload)
        _require_sha256(self.file_sha256, "file_sha256")
        _require_sha256(self.semantic_sha256, "semantic_sha256")


def load_experimental_physics_contract(
    path: str | Path = DEFAULT_EXPERIMENTAL_PHYSICS_CONTRACT_PATH,
) -> V8ExperimentalPhysicsContractV1:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V8 experimental physics contract must be a JSON object")
    validate_experimental_physics_contract(payload)
    repository_root = Path(__file__).resolve().parents[2]
    decision_source = dict(payload["decision_source"])
    for path_key, hash_key in (
        ("operator_ask_v2_amendment_path", "operator_ask_v2_amendment_sha256"),
        (
            "dlcz_double_excitation_contract_path",
            "dlcz_double_excitation_contract_sha256",
        ),
        (
            "aod_optical_detuning_contract_path",
            "aod_optical_detuning_contract_sha256",
        ),
    ):
        referenced_path = repository_root / str(decision_source[path_key])
        if not referenced_path.is_file():
            raise FileNotFoundError(
                f"referenced experimental-physics contract is missing: {referenced_path}"
            )
        if _sha256_file(referenced_path) != str(decision_source[hash_key]):
            raise ValueError(
                f"referenced experimental-physics contract hash changed: {path_key}"
            )
    contract = V8ExperimentalPhysicsContractV1(
        payload=payload,
        file_sha256=_sha256_file(source),
        semantic_sha256=_stable_json_sha256(payload),
        path=str(source),
    )
    contract.validate()
    return contract


def validate_experimental_physics_contract(contract: Mapping[str, Any]) -> None:
    value = dict(contract or {})
    if value.get("schema") != EXPERIMENTAL_PHYSICS_CONTRACT_SCHEMA:
        raise ValueError("unsupported V8 experimental physics contract schema")

    source = _mapping(value, "decision_source")
    _require_sha256(source.get("operator_ask_record_sha256"), "operator_ask_record_sha256")
    _require_sha256(source.get("original_review_prompt_sha256"), "original_review_prompt_sha256")
    _require_sha256(
        source.get("operator_ask_v2_amendment_sha256"),
        "operator_ask_v2_amendment_sha256",
    )
    _require_sha256(
        source.get("dlcz_double_excitation_contract_sha256"),
        "dlcz_double_excitation_contract_sha256",
    )
    _require_sha256(
        source.get("aod_optical_detuning_contract_sha256"),
        "aod_optical_detuning_contract_sha256",
    )
    _require_exact(
        source,
        "operator_ask_v2_amendment_path",
        "docs/ml/architecture/v8/V8_OPERATOR_ASK_PHYSICS_CONTRACT_V2_2026_08_11.md",
    )
    _require_exact(
        source,
        "dlcz_double_excitation_contract_path",
        "docs/ml/architecture/v8/V8_DLCZ_DOUBLE_EXCITATION_CONTRACT_2026_08_12.md",
    )
    _require_exact(
        source,
        "aod_optical_detuning_contract_path",
        "docs/ml/architecture/v8/V8_AOD_OPTICAL_DETUNING_CONTRACT_2026_08_14.md",
    )
    _require_exact(source, "consolidated_at_local_date", "2026-08-11")
    _require_exact(source, "latest_operator_amendment_local_date", "2026-08-14")
    _require_exact(source, "timezone", "Asia/Shanghai")
    _require_exact(source, "operator_authorized", True)
    _require_exact(source, "external_herald_binary_spad_amendment_authorized", True)

    sharing = _mapping(value, "system_sharing")
    if tuple(sharing.get("systems") or ()) != SUPPORTED_SYSTEMS:
        raise ValueError("experimental physics contract must cover exact 3Q/4Q systems")
    for name, expected in {
        "shared_dlcz_aod_physics_backbone": True,
        "system_specific_calibration_heads": True,
        "system_specific_uncertainty_and_selection_gates": True,
        "cross_system_real_conclusion_transfer_allowed": False,
    }.items():
        _require_exact(sharing, name, expected)

    state = _mapping(value, "state_contract")
    _require_exact(state, "feedback_state_space", "single_excitation_w_core_n_by_n")
    _require_exact(state, "latent_generator_space", "bosonic_fock_sectors_n0_n1_n2_n_ge_3")
    _require_exact(state, "n1_sector_dimension", "n")
    _require_exact(state, "n2_sector_dimension", "n_times_n_plus_1_div_2")
    for name, expected in {
        "hard_core_two_power_n_space_is_complete_for_double_occupancy": False,
        "sector_specific_measurement_expectation_required": True,
        "w_core_born_allocation_allowed_for_n2_or_tail": False,
        "hom_relative_phase_authoritatively_observable": False,
    }.items():
        _require_exact(state, name, expected)
    _require_exact(state, "full_two_power_n_density_claim_allowed", False)
    _require_exact(
        state,
        "rho_solver",
        "calibrated_povm_covariance_weighted_psd_trace_one_mle",
    )
    if tuple(state.get("primary_report_metrics") or ()) != PRIMARY_REPORT_METRICS:
        raise ValueError("V8 physical report metrics changed")
    _require_exact(
        state,
        "conditional_core_quality_and_unconditional_yield_must_be_separate",
        True,
    )
    for name, expected in {
        "measured_rho_runtime_input_allowed": False,
        "measured_rho_authoritative_supervision_allowed": False,
        "measured_rho_auxiliary_pseudo_target_ablation_allowed": True,
        "measured_rho_auxiliary_default_enabled": False,
        "synthetic_latent_rho_auxiliary_supervision_allowed": True,
        "rho_auxiliary_selection_authority_allowed": False,
        "rho_auxiliary_ablation_non_regression_required": True,
    }.items():
        _require_exact(state, name, expected)
    _require_exact(
        state,
        "rho_auxiliary_semantic",
        "output_only_physics_regularizer_with_uncertainty_not_runtime_input",
    )

    counts = _mapping(value, "count_and_denoising_contract")
    _require_exact(counts, "read_count_semantic", "shared_effective_trial_count_per_operation")
    read_rates = dict(counts.get("known_exposure_read_rate_hz_by_system") or {})
    if read_rates != {"3qubit": 419.6, "4qubit": 137.35}:
        raise ValueError("known-exposure read-rate references changed")
    _require_exact(counts, "known_exposure_forward_uses_per_row_read_count", True)
    _require_exact(counts, "legacy_reference_gate_rate_hz_5000_allowed", False)
    _require_exact(counts, "g_si_allowed_as_direct_rho_target", False)
    _require_exact(counts, "raw_processed_counts_retained_and_evaluated_in_parallel", True)
    _require_exact(counts, "low_count_fallback", "exact_four_outcome_multinomial_posterior")
    if tuple(counts.get("negative_residual_ablation") or ()) != NEGATIVE_RESIDUAL_ABLATIONS:
        raise ValueError("negative-residual ablation must remain signed versus censored Gaussian")
    for name, expected in {
        "hard_zero_clipping_allowed": False,
        "outcome_adaptive_same_window_operation_selection_allowed": False,
        "ordinary_measured_noise_must_remain_in_trusted_rows": True,
    }.items():
        _require_exact(counts, name, expected)

    detector = _mapping(value, "detector_and_multi_excitation_contract")
    if int(detector.get("detector_count", -1)) != 4:
        raise ValueError("V8 real systems require four detector outputs")
    efficiency = tuple(float(item) for item in detector.get("detector_efficiency_prior_interval") or ())
    if efficiency != (0.25, 0.40):
        raise ValueError("detector efficiency must remain the declared broad 25-40 percent prior")
    if float(detector.get("global_optical_path_efficiency_prior_center", -1.0)) != 0.40:
        raise ValueError("global optical-path efficiency prior changed")
    if tuple(
        float(item) for item in detector.get("combined_read_efficiency_prior_interval") or ()
    ) != (0.10, 0.16):
        raise ValueError("combined read efficiency prior changed")
    if float(
        detector.get("external_herald_singleton_association_window_s", -1.0)
    ) != 2.0e-7:
        raise ValueError("external-herald/singleton association gate must remain 200 ns")
    if float(
        detector.get("read_multi_detector_simultaneity_clock_s", -1.0)
    ) != 1.0e-8:
        raise ValueError("read multi-detector simultaneity must remain one 100 MHz clock")
    for name, expected in {
        "detector_efficiency_authoritatively_calibrated": False,
        "detector_and_optical_path_efficiency_must_be_separate": True,
        "signal_is_external_herald_photon_count": False,
        "signal_is_aggregate_external_herald_spad_click_count": True,
        "signal_is_eventwise_photon_number": False,
        "external_herald_equal_logical_acceptance_for_n1_n2_n_ge_3": True,
        "external_herald_default_equal_physical_click_likelihood_for_n_ge_1": True,
        "external_herald_efficiency_authoritatively_calibrated": False,
        "external_herald_may_directly_supervise_real_excitation_sector": False,
        "multi_click_detectors_are_all_read_idler_path": True,
        "arbitrary_qubit_mode_to_detector_routing_allowed": False,
        "multi2_is_direct_double_excitation_label": False,
        "sealed_corpus_has_off_window_background": False,
        "global_multi_click_may_be_expanded_to_operation_pseudo_labels": False,
        "estimated_true_multi_click_may_be_direct_real_label": False,
        "single_corrected_real_label_export_allowed": False,
        "dead_time_afterpulse_gate_crosstalk_real_supervision_available": False,
    }.items():
        _require_exact(detector, name, expected)
    if float(detector.get("external_herald_gate_window_s", -1.0)) != 2.0e-7:
        raise ValueError("external herald SPAD gate must remain 200 ns")
    _require_exact(
        detector,
        "external_herald_detector_model",
        "threshold_povm_one_minus_one_minus_dark_times_one_minus_eta_power_n",
    )
    _require_exact(
        detector,
        "external_herald_efficiency_sensitivity_role",
        "fixed_acquisition_bias_sensitivity_not_eventwise_number_resolution",
    )
    if tuple(float(value) for value in detector.get(
        "nominal_four_output_splitter_probability", ()
    )) != (0.25, 0.25, 0.25, 0.25):
        raise ValueError("fixed two-layer read splitter must remain nominally uniform")
    if tuple(float(value) for value in detector.get(
        "nominal_beamsplitter_layer_transmission", ()
    )) != (0.5, 0.5):
        raise ValueError("both read splitter layers must remain nominally 50:50")
    if float(detector.get(
        "relative_detector_response_variation_prior_fraction", -1.0
    )) != 0.10:
        raise ValueError("relative read-detector response variation prior changed")
    if tuple(detector.get("fpga_joint_four_cell_order") or ()) != (
        "R-S-I+C",
        "S-C",
        "I-C",
        "C",
    ):
        raise ValueError("FPGA four-cell observation order changed")
    if float(
        detector.get("valid_acquisition_total_multi_to_idler_empirical_upper_fraction", -1.0)
    ) != 0.20:
        raise ValueError("operator total-multi/idler apparatus cap changed")
    _require_exact(
        detector,
        "total_multi_ratio_numerator",
        "multi2_plus_multi3_plus_multi4_global_clock_counts",
    )
    _require_exact(
        detector,
        "total_multi_ratio_denominator",
        "idler_total_sticky_singleton_trial_count",
    )
    _require_exact(
        detector,
        "total_multi_ratio_role",
        "data_quality_hard_fault_only_not_likelihood_target",
    )
    _require_exact(
        detector,
        "substantial_real_multi2_dark_background_response_expected",
        True,
    )
    _require_exact(detector, "fpga_multi_origin_report_required", True)
    if tuple(detector.get("fpga_multi_origin_names") or ()) != (
        "no_herald_background_window",
        "herald_n1_window",
        "herald_n2_window",
        "herald_n_ge_3_window",
    ):
        raise ValueError("FPGA multi origin registry changed")
    _require_exact(
        detector,
        "background_zeroed_counterfactual_report_required",
        True,
    )
    _require_exact(
        detector,
        "fpga_multi_origin_eventwise_label_claim_allowed",
        False,
    )
    double_excitation = dict(detector.get("double_excitation_state_contract") or {})
    if tuple(double_excitation.get("component_names") or ()) != (
        "same_collective_mode_fock2",
        "target_plus_orthogonal_schmidt_mode",
        "occupation_dephased_fock2",
    ):
        raise ValueError("DLCZ double-excitation component registry changed")
    for name, expected in {
        "ideal_same_mode_one_body_reduction_equals_target": True,
        "double_excitation_implies_phase_violation": False,
        "eventwise_real_component_label_allowed": False,
    }.items():
        _require_exact(double_excitation, name, expected)
    _require_exact(
        double_excitation,
        "phase_violation_semantic",
        "one_minus_target_overlap_of_normalized_one_body_reduced_density",
    )

    time_space = _mapping(value, "temporal_and_spatial_contract")
    if (
        int(time_space.get("authoritative_window_s", -1)) != 1200
        or int(time_space.get("subwindow_count", -1)) != 8
        or int(time_space.get("subwindow_s", -1)) != 150
    ):
        raise ValueError("authoritative timing must remain 1200 s = 8 x 150 s")
    _require_exact(time_space, "operation_points_are_independent_samples", False)

    observable = _mapping(value, "observable_covariate_contract")
    for name, expected in {
        "actual_loaded_means_measured_rf_output": False,
        "independent_rf_power_phase_spectrum_measurement_available": False,
        "online_beam_position_or_intensity_telemetry_available": False,
        "beam_covariates_are_offline_calibration_or_latent_nuisance_only": True,
        "cell_lifetime_map_may_be_called_dephasing_rate_without_storage_sweep": False,
    }.items():
        _require_exact(observable, name, expected)
    if float(observable.get("nominal_storage_delay_us", -1.0)) != 55.0:
        raise ValueError("nominal storage delay must remain distinct and fixed at 55 us")
    _require_sha256(observable.get("temperature_source_sha256"), "temperature_source_sha256")
    for name, expected in {
        "environment_contract_schema": "physics_hybrid_v8_environment_contract_v2",
        "environment_sensor_serial": "HXA1379GS1A",
        "humidity_unit": "percent_rh",
        "environment_role": "weak_data_quality_analysis_only",
        "environment_may_change_hard_labels_or_likelihood_weight": False,
    }.items():
        _require_exact(observable, name, expected)
    if (
        float(observable.get("environment_sensor_distance_from_platform_m_approx", -1.0))
        != 2.0
        or float(observable.get("high_temperature_threshold_c", -1.0)) != 21.5
    ):
        raise ValueError("environment sensor geometry or temperature threshold changed")

    detuning = _mapping(value, "aod_optical_detuning_contract")
    if tuple(float(item) for item in detuning.get("operation_point_reference_mhz") or ()) != (
        103.0,
        103.0,
    ):
        raise ValueError("AOD optical reference operation point changed")
    if (
        float(detuning.get("write_reference_detuning_mhz", float("nan"))) != -10.0
        or float(detuning.get("read_reference_detuning_mhz", float("nan"))) != 0.0
        or float(detuning.get("write_diffraction_order_sign", float("nan"))) != 1.0
        or float(detuning.get("read_diffraction_order_sign", float("nan"))) != -1.0
    ):
        raise ValueError("AOD write/read detuning reference or sign changed")
    for name, expected in {
        "relative_shift_formula": "operation_x_plus_operation_y_minus_cell_x_minus_cell_y",
        "write_detuning_formula": "write_reference_plus_relative_shift",
        "read_detuning_formula": "read_reference_minus_relative_shift",
        "three_qubit_operation_order_authority": "awg_sequential_operation_index",
        "three_qubit_awg_operation_order": "basis_config_entries_11_special_then_64_information_row_major",
        "three_qubit_ui_64_plus_11_partition_has_physical_authority": False,
        "single_photon_demultiplexer_equalizes_frequency_before_interference": True,
        "atomic_write_read_detuning_is_cell_dependent_before_demultiplexing": True,
        "detuning_source_contract": "aod_frequency_point_difference_only",
        "positive_detuning_semantic": "blue_relative_to_atomic_transition",
        "negative_detuning_semantic": "red_relative_to_atomic_transition",
        "both_detuning_signs_supported": True,
        "detuning_cancellation_assumption_allowed": False,
        "operator_provided_aod_mapping_authoritative": True,
        "historical_per_row_operation_point_authoritative": False,
        "nominal_registry_feature_role": "static_synthetic_prior_and_sensitivity_not_historical_command",
        "legacy_retrieval_map_is_authoritative_historical_calibration": False,
        "real_point_identification_allowed": False,
        "real_product_or_effective_coupling_learning_allowed": True,
        "action_or_causal_evidence_created": False,
    }.items():
        _require_exact(detuning, name, expected)
    if (
        tuple(detuning.get("three_qubit_special_operation_range") or ())
        != (0, 11)
        or tuple(detuning.get("three_qubit_information_operation_range") or ())
        != (11, 75)
    ):
        raise ValueError("3Q AWG operation ranges changed")
    if (
        tuple(float(item) for item in detuning.get("operator_example_operation_point_mhz") or ())
        != (88.0, 110.0)
        or tuple(float(item) for item in detuning.get("operator_example_cell_frequency_mhz") or ())
        != (95.0, 88.0)
        or float(detuning.get("operator_example_relative_shift_mhz", float("nan"))) != 15.0
        or float(detuning.get("operator_example_write_detuning_mhz", float("nan"))) != 5.0
        or float(detuning.get("operator_example_read_detuning_mhz", float("nan"))) != -15.0
    ):
        raise ValueError("operator AOD detuning example changed")
    if tuple(detuning.get("selected_idler_coupling_factorization") or ()) != (
        "memory_retrieval_and_aod_detuning",
        "mode_overlap_before_splitter",
        "global_downstream_optical_path",
        "per_spad_efficiency_after_splitter",
    ):
        raise ValueError("selected-idler coupling factorization changed")

    action = _mapping(value, "action_and_runtime_contract")
    for name, expected in {
        "current_hardware_action_authorized": False,
        "w_ch3_amplitude_action_allowed": False,
        "production_neural_or_calibrator_parameter_updates_allowed": False,
        "posterior_and_history_state_updates_allowed": True,
        "episode_level_pre_authorization_required_before_automation": True,
    }.items():
        _require_exact(action, name, expected)
    if (
        int(action.get("episode_maximum_action_adjustments", -1)) != 10
        or int(action.get("episode_maximum_paired_transitions", -1)) != 9
        or action.get("episode_step_semantic")
        != "initial_state_plus_at_most_ten_sequential_bounded_adjustments"
    ):
        raise ValueError("episode action-step contract changed")
    if float(action.get("w_ch3_amplitude_v", -1.0)) != 1.0:
        raise ValueError("W_CH3 amplitude must remain fixed at the declared 1 V work point")
    if float(action.get("w_ch3_phase_candidate_maximum_step_rad", -1.0)) > 0.005:
        raise ValueError("W_CH3 phase step exceeds the operator-approved narrow neighborhood")
    if float(action.get("w_ch3_phase_candidate_maximum_cumulative_rad", -1.0)) > 0.01:
        raise ValueError("W_CH3 phase cumulative bound exceeds the operator-approved limit")

    supervision = _mapping(value, "real_and_synthetic_supervision_contract")
    for name, expected in {
        "all_trusted_real_rows_supervise_static_shared_representation": True,
        "state_pair_transport_is_real_action_evidence": False,
        "target_action_pairing_shuffle_negative_control_required": True,
        "target_context_shuffle_negative_control_required": True,
        "synthetic_latent_noise_truth_allowed": True,
        "synthetic_truth_may_be_exported_as_real_corrected_label": False,
        "synthetic_fock_leakage_classifier_pretraining_required": True,
        "synthetic_fock_encoder_may_initialize_static_feedback_representation": True,
        "synthetic_fock_encoder_transfer_creates_real_action_evidence": False,
    }.items():
        _require_exact(supervision, name, expected)
    _require_exact(
        supervision,
        "real_action_residual_jacobian_and_policy_require",
        "paired_executed_transition_or_hash_bound_signed_probe",
    )


def experimental_physics_contract_audit(
    contract: V8ExperimentalPhysicsContractV1,
) -> Dict[str, Any]:
    contract.validate()
    payload = contract.payload
    return {
        "schema": "physics_hybrid_v8_experimental_physics_contract_audit_v3",
        "contract_schema": payload["schema"],
        "contract_file_sha256": contract.file_sha256,
        "contract_semantic_sha256": contract.semantic_sha256,
        "operator_ask_record_sha256": payload["decision_source"][
            "operator_ask_record_sha256"
        ],
        "operator_ask_v2_amendment_sha256": payload["decision_source"][
            "operator_ask_v2_amendment_sha256"
        ],
        "dlcz_double_excitation_contract_sha256": payload["decision_source"][
            "dlcz_double_excitation_contract_sha256"
        ],
        "aod_optical_detuning_contract_sha256": payload["decision_source"][
            "aod_optical_detuning_contract_sha256"
        ],
        "systems": list(payload["system_sharing"]["systems"]),
        "hardware_action_authorized": False,
        "real_action_evidence_policy": payload[
            "real_and_synthetic_supervision_contract"
        ]["real_action_residual_jacobian_and_policy_require"],
        "real_corrected_label_export_allowed": False,
        "latent_generator_space": payload["state_contract"]["latent_generator_space"],
        "known_exposure_read_rate_hz_by_system": payload[
            "count_and_denoising_contract"
        ]["known_exposure_read_rate_hz_by_system"],
        "external_herald_singleton_association_window_s": payload[
            "detector_and_multi_excitation_contract"
        ]["external_herald_singleton_association_window_s"],
        "read_multi_detector_simultaneity_clock_s": payload[
            "detector_and_multi_excitation_contract"
        ]["read_multi_detector_simultaneity_clock_s"],
        "valid_acquisition_total_multi_to_idler_empirical_upper_fraction": payload[
            "detector_and_multi_excitation_contract"
        ]["valid_acquisition_total_multi_to_idler_empirical_upper_fraction"],
        "substantial_real_multi2_dark_background_response_expected": payload[
            "detector_and_multi_excitation_contract"
        ]["substantial_real_multi2_dark_background_response_expected"],
        "fpga_multi_origin_names": payload[
            "detector_and_multi_excitation_contract"
        ]["fpga_multi_origin_names"],
        "fpga_multi_origin_eventwise_label_claim_allowed": False,
        "environment_contract_schema": payload[
            "observable_covariate_contract"
        ]["environment_contract_schema"],
        "aod_optical_detuning_contract": dict(
            payload["aod_optical_detuning_contract"]
        ),
    }


def _mapping(value: Mapping[str, Any], name: str) -> Dict[str, Any]:
    result = dict(value.get(name) or {})
    if not result:
        raise ValueError(f"experimental physics contract section is missing: {name}")
    return result


def _require_exact(value: Mapping[str, Any], name: str, expected: Any) -> None:
    if value.get(name) != expected:
        raise ValueError(f"experimental physics contract changed {name}")


def _require_sha256(value: Any, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DEFAULT_EXPERIMENTAL_PHYSICS_CONTRACT_PATH",
    "EXPERIMENTAL_PHYSICS_CONTRACT_SCHEMA",
    "V8ExperimentalPhysicsContractV1",
    "experimental_physics_contract_audit",
    "load_experimental_physics_contract",
    "validate_experimental_physics_contract",
]
