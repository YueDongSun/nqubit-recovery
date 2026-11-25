"""Declare production-classifier supervision without inferring trained authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Mapping


@dataclass(frozen=True)
class Objective_binding:
    """One scalar objective's target and effective-support declaration."""

    objective_name: str
    authoritative_targets: tuple[str, ...]
    target_masks: tuple[str, ...]
    effective_mask_key: str
    coverage_unit: str
    relation: str | None = None
    scope: str | None = None


@dataclass(frozen=True)
class Published_output_training_spec:
    output_name: str
    authoritative_targets: tuple[str, ...] = ()
    target_masks: tuple[str, ...] = ()
    proper_loss: str | None = None
    promotion_required: bool = False
    downstream_consumers: tuple[str, ...] = ()
    status: str = "DIAGNOSTIC_UNTRAINED"
    objective_bindings: tuple[Objective_binding, ...] = ()


# The generic legacy classifier and physical wrapper have distinct feedback
# consumers.  Do not allow the generic eight-output contract to hide the
# physical aggregate-sector value consumed by the physical feedback adapter.
legacy_feedback_required_outputs = (
    "R1_re",
    "R1_im",
    "W_populations",
    "W_coherence_re",
    "W_coherence_im",
    "W_amplitude_variance",
    "W_phase_variance",
    "joint_control_covariance",
)
physical_feedback_required_outputs = (
    *legacy_feedback_required_outputs,
    "sector_aggregate_posterior",
)
# Preserve the established generic classifier API.  Physical consumers must
# request their larger contract explicitly so a physical receipt can never be
# mistaken for a legacy classifier receipt.
feedback_required_outputs = legacy_feedback_required_outputs


def feedback_consumer_required_outputs() -> tuple[str, ...]:
    """Return the established generic classifier feedback contract."""
    return feedback_required_outputs


physical_uncertainty_outputs = {
    "W_amplitude_variance": (
        ("target_W_amplitude_error",),
        ("target_physical_uncertainty_mask",),
        "synthetic_v2_gaussian_nll",
    ),
    "W_phase_variance": (
        ("target_W_phase_error",),
        ("target_physical_uncertainty_mask",),
        "synthetic_v2_circular_gaussian_nll",
    ),
    "joint_control_covariance": (
        ("target_W_amplitude_error", "target_W_phase_error"),
        ("target_physical_uncertainty_mask",),
        "synthetic_v2_helmert_joint_gaussian_nll",
    ),
}
objective_outputs = (
    (("p_clean",), ("target_p_clean",), (), "measurement_and_denoise"),
    (
        ("fine_family_logits", "fine_family_probs"),
        ("target_fine",),
        ("target_fine_mask", "target_fine_identifiable_mask"),
        "small_n_family_and_calibration",
    ),
    (
        ("pair_entanglement_probs", "pair_entanglement_logits"),
        ("target_pair",),
        ("target_pair_mask",),
        "pair_map",
    ),
    (
        ("hyperedge_entanglement_probs",),
        ("target_hyperedge",),
        ("target_hyperedge_mask",),
        "hyperedge",
    ),
    (
        ("partition_link_probs", "partition_link_logits", "partition_antichain_scores"),
        ("target_partition",),
        ("target_partition_mask",),
        "partition",
    ),
    (("depth_posterior",), ("target_depth",), ("target_structure_mask",), "depth"),
    (
        ("intactness_posterior",),
        ("target_intactness",),
        ("target_structure_mask",),
        "intactness",
    ),
    (
        ("stretchability_posterior",),
        ("target_stretchability",),
        ("target_structure_mask",),
        "stretchability",
    ),
    (
        ("coarse_template_posterior",),
        ("target_coarse",),
        ("target_coarse_mask",),
        "coarse",
    ),
    (
        ("R1_re", "R1_im", "W_populations", "W_coherence_re", "W_coherence_im"),
        ("target_R1_re", "target_R1_im"),
        ("target_w_mask",),
        "w_r1",
    ),
    (
        ("p0", "p1", "p_ge2", "leakage"),
        ("target_sector",),
        ("target_w_mask",),
        "w_sector",
    ),
    (
        ("operational_noise_logits", "operational_noise_posterior"),
        ("target_operational_noise",),
        ("target_operational_noise_mask",),
        "operational_noise",
    ),
    (("OOD_probability",), ("target_ood",), (), "ood"),
)
diagnostic_outputs = (
    "certificate_status_logits",
    "certificate_status_probs",
    "model_uncertainty",
    "W_amplitude_variance",
    "W_phase_variance",
    "amplitude_alr_covariance",
    "phase_helmert_covariance",
    "joint_control_covariance",
    "amplitude_error",
    "phase_error",
    "recoverability",
    "W_uncertainty",
    "entanglement_depth_score",
    "covariance_clean_diag",
    "nuisance_posterior_mean",
    "nuisance_posterior_variance",
    "abstention_probability",
)
metadata_outputs = (
    "stretchability_values",
    "partition_representation",
    "R1_factor_re",
    "R1_factor_im",
    "fine_family_primary_report_allowed",
    "semantic_latent",
    "sector_native_logits",
)
physical_objective_outputs = (
    (("amplitude_error",), ("target_amplitude_error",), ("target_mode_mask",),
     "physical_helmert_log_amplitude_mse"),
    (("phase_error",), ("target_phase_error",), ("target_mode_mask",),
     "physical_helmert_wrapped_phase_mse"),
    (("p_clean", "measurement_probability"), ("target_measurement_probability",),
     ("target_measurement_mask",), "physical_four_cell_cross_entropy"),
)


def _physical_binding(
    objective_name: str,
    targets: tuple[str, ...],
    masks: tuple[str, ...],
    effective_mask_key: str,
    coverage_unit: str,
    *,
    relation: str | None = None,
    scope: str | None = None,
) -> Objective_binding:
    return Objective_binding(
        objective_name, targets, masks, effective_mask_key, coverage_unit, relation, scope
    )


physical_feedback_objective_bindings = {
    "R1_re": (_physical_binding("R1_loss", ("target_R1_re", "target_R1_im"), ("target_mode_mask",), "state_rows", "ROWS", scope="FULL_COMPLEX_R1"),),
    "R1_im": (_physical_binding("R1_loss", ("target_R1_re", "target_R1_im"), ("target_mode_mask",), "state_rows", "ROWS", scope="FULL_COMPLEX_R1"),),
    "W_coherence_re": (_physical_binding("R1_loss", ("target_R1_re", "target_R1_im"), ("target_mode_mask",), "state_rows", "ROWS", scope="OFF_DIAGONAL_COHERENCE"),),
    "W_coherence_im": (_physical_binding("R1_loss", ("target_R1_re", "target_R1_im"), ("target_mode_mask",), "state_rows", "ROWS", scope="OFF_DIAGONAL_COHERENCE"),),
    "W_populations": (
        _physical_binding("W_population_loss", ("target_W_populations",), ("target_mode_mask",), "state_rows", "ROWS"),
        _physical_binding("R1_loss", ("target_R1_re", "target_R1_im"), ("target_mode_mask",), "state_rows", "ROWS", relation="target_W_populations_equals_diagonal_of_target_R1_normalized_by_positive_trace", scope="DIAGONAL_ONLY"),
    ),
    "W_amplitude_variance": (_physical_binding("joint_control_covariance_loss", ("target_R1_re", "target_R1_im"), ("target_mode_mask",), "covariance_rows", "ROWS", relation="point_belief_residual_detached_for_covariance_objective"),),
    "W_phase_variance": (_physical_binding("joint_control_covariance_loss", ("target_R1_re", "target_R1_im"), ("target_mode_mask",), "covariance_rows", "ROWS", relation="point_belief_residual_detached_for_covariance_objective"),),
    "joint_control_covariance": (_physical_binding("joint_control_covariance_loss", ("target_R1_re", "target_R1_im"), ("target_mode_mask",), "covariance_rows", "ROWS", relation="point_belief_residual_detached_for_covariance_objective"),),
}

physical_auxiliary_objective_bindings = {
    "p_clean": (_physical_binding("measurement_likelihood_loss", ("target_measurement_probability",), ("target_measurement_mask",), "measurement_operations", "OPERATIONS"),),
    "amplitude_error": (_physical_binding("amplitude_error_loss", ("target_amplitude_error",), ("target_mode_mask",), "state_rows", "ROWS"),),
    "phase_error": (_physical_binding("phase_error_loss", ("target_phase_error",), ("target_mode_mask",), "state_rows", "ROWS"),),
    "sector_aggregate_logits": (_physical_binding("sector_loss", ("target_sector",), ("target_sector_mask",), "sector_rows", "ROWS", scope="K1_VERSUS_REST_AGGREGATE"),),
    "sector_aggregate_posterior": (_physical_binding("sector_loss", ("target_sector",), ("target_sector_mask",), "sector_rows", "ROWS", scope="K1_VERSUS_REST_AGGREGATE"),),
    "measurement_probability": (_physical_binding("measurement_likelihood_loss", ("target_measurement_probability",), ("target_measurement_mask",), "measurement_operations", "OPERATIONS"),),
}


def published_output_registry() -> dict[str, Published_output_training_spec]:
    registry = {}
    for outputs, targets, masks, loss in objective_outputs:
        for name in outputs:
            registry[name] = Published_output_training_spec(
                name,
                targets,
                masks,
                loss,
                name in legacy_feedback_required_outputs,
                ("feedback_semantic_adapter", "mainline_bundle")
                if name in legacy_feedback_required_outputs
                else (),
                "OBJECTIVE_REGISTERED_NOT_VALIDATED",
            )
    for name in diagnostic_outputs:
        required = name in legacy_feedback_required_outputs or name.startswith(
            "certificate_status"
        )
        targets, masks, loss = physical_uncertainty_outputs.get(name, ((), (), None))
        registry[name] = Published_output_training_spec(
            name,
            targets,
            masks,
            loss,
            promotion_required=required,
            downstream_consumers=("feedback_semantic_adapter", "mainline_bundle")
            if name in legacy_feedback_required_outputs
            else (),
            status=(
                "OBJECTIVE_IMPLEMENTED_NOT_CALIBRATED"
                if loss
                else "DIAGNOSTIC_UNTRAINED"
            ),
        )
    for name in metadata_outputs:
        registry[name] = Published_output_training_spec(
            name, status="REPRESENTATION_OR_METADATA_ONLY"
        )
    return registry


def physical_output_training_registry(excitation_mode: str = "legacy") -> dict[str, Published_output_training_spec]:
    """Describe the physical C_N wrapper without changing the legacy registry.

    Physical W-state outputs are derived from the full R1 target.  The mode
    mask is the observed availability unit for every feedback-consumed output.
    """
    # Names are shared with the core, but its legacy targets and objectives are
    # not part of physical_classifier_step. In particular, target_sector is a
    # class label here, not the legacy soft target masked by target_w_mask.
    if excitation_mode not in ("legacy", "diagnostic", "coupled"):
        raise ValueError("unknown physical excitation mode")
    registry = {
        name: replace(
            spec,
            authoritative_targets=(),
            target_masks=(),
            proper_loss=None,
            status=("REPRESENTATION_OR_METADATA_ONLY" if name in metadata_outputs
                    else "DIAGNOSTIC_UNTRAINED"),
        )
        for name, spec in published_output_registry().items()
    }
    for outputs, targets, masks, loss in physical_objective_outputs:
        for name in outputs:
            registry[name] = Published_output_training_spec(
                name, targets, masks, loss, status="OBJECTIVE_REGISTERED_NOT_VALIDATED"
            )
    # Every feedback output declares its actual scalar objective.  In
    # particular, covariance support is belief-valid rows, not all available
    # mode elements.  This remains a training-contract declaration only.
    for name, bindings in physical_feedback_objective_bindings.items():
        primary = bindings[0]
        registry[name] = Published_output_training_spec(
            output_name=name,
            authoritative_targets=primary.authoritative_targets,
            target_masks=primary.target_masks,
            proper_loss=primary.objective_name,
            promotion_required=True,
            downstream_consumers=("feedback_semantic_adapter", "mainline_bundle"),
            status="OBJECTIVE_REGISTERED_NOT_VALIDATED",
            objective_bindings=bindings,
        )
    for name, bindings in physical_auxiliary_objective_bindings.items():
        primary = bindings[0]
        registry[name] = Published_output_training_spec(
            output_name=name,
            authoritative_targets=primary.authoritative_targets,
            target_masks=primary.target_masks,
            proper_loss=primary.objective_name,
            promotion_required=name in physical_feedback_required_outputs,
            downstream_consumers=("feedback_semantic_adapter", "mainline_bundle")
            if name in physical_feedback_required_outputs else (),
            status="OBJECTIVE_REGISTERED_NOT_VALIDATED",
            objective_bindings=bindings,
        )
    for name in ("sector_logits", "sector_posterior"):
        registry[name] = Published_output_training_spec(
            output_name=name,
            status="DIAGNOSTIC_UNTRAINED",
        )
    if excitation_mode != "legacy":
        for name, objective in {
            "operation_excitation_rate": "excitation_rate_oracle_loss",
            "operation_effective_mu": "intensity_oracle_loss",
            "operation_sector_probabilities": "operation_sector_loss",
        }.items():
            if name == "operation_excitation_rate":
                targets = ("target_excitation_rate_o",)
                masks = ("target_excitation_rate_o_mask",)
            elif name == "operation_effective_mu":
                targets = ("target_effective_mu_o",)
                masks = ("target_effective_mu_o_mask",)
            else:
                targets = ("target_sector_probabilities",)
                masks = ("target_sector_probability_mask",)
            registry[name] = Published_output_training_spec(
                output_name=name,
                authoritative_targets=targets,
                target_masks=masks,
                proper_loss=objective,
                status="OBJECTIVE_REGISTERED_NOT_VALIDATED",
            )
        query_bindings = (
            _physical_binding("measurement_likelihood_loss", ("target_four_cell_counts",), ("likelihood_target_mask",), "query_operations", "OPERATIONS"),
        )
        for name in ("p_clean", "measurement_probability"):
            registry[name] = replace(
                registry[name],
                authoritative_targets=("target_four_cell_counts",),
                target_masks=("likelihood_target_mask",),
                proper_loss="measurement_likelihood_loss",
                objective_bindings=query_bindings,
            )
        conditional_r1_targets = (
            "target_conditional_rho1_re",
            "target_conditional_rho1_im",
            "target_sector_probabilities",
        )
        conditional_r1_masks = (
            "target_mode_mask",
            "target_sector_probability_mask",
        )
        r1_binding = _physical_binding(
            "R1_loss",
            conditional_r1_targets,
            conditional_r1_masks,
            "state_rows",
            "ROWS",
            relation="target_R1_equals_opportunity_weighted_p1_times_conditional_rho1",
            scope="OPERATION_POISSON_SUBNORMALIZED_R1",
        )
        for name in ("R1_re", "R1_im", "W_coherence_re", "W_coherence_im"):
            registry[name] = replace(
                registry[name],
                authoritative_targets=conditional_r1_targets,
                target_masks=("target_mode_mask",),
                proper_loss="R1_loss",
                objective_bindings=(r1_binding,),
            )
        covariance_binding = _physical_binding(
            "joint_control_covariance_loss",
            conditional_r1_targets,
            conditional_r1_masks,
            "covariance_rows",
            "ROWS",
            relation="belief_residual_uses_opportunity_weighted_p1_times_conditional_rho1",
        )
        for name in (
            "W_amplitude_variance",
            "W_phase_variance",
            "joint_control_covariance",
        ):
            registry[name] = replace(
                registry[name],
                authoritative_targets=conditional_r1_targets,
                target_masks=("target_mode_mask",),
                proper_loss="joint_control_covariance_loss",
                objective_bindings=(covariance_binding,),
            )
        operation_sector_binding = _physical_binding(
            "sector_loss",
            ("target_sector_probabilities",),
            ("target_sector_probability_mask",),
            "sector_rows",
            "ROWS",
            relation="opportunity_weighted_operation_sector_mixture",
            scope="OPERATION_POISSON_K1_VERSUS_REST",
        )
        for name in ("sector_aggregate_logits", "sector_aggregate_posterior"):
            registry[name] = replace(
                registry[name],
                authoritative_targets=("target_sector_probabilities",),
                target_masks=("target_sector_probability_mask",),
                proper_loss="sector_loss",
                objective_bindings=(operation_sector_binding,),
            )
        operation_mu_bindings = (
            _physical_binding("intensity_oracle_loss", ("target_effective_mu_o",), ("target_effective_mu_o_mask",), "intensity_operations", "OPERATIONS"),
            _physical_binding("herald_binomial_loss", ("target_herald_counts",), ("likelihood_target_mask",), "query_operations", "OPERATIONS"),
            _physical_binding("operation_sector_loss", ("target_sector_probabilities",), ("target_sector_probability_mask",), "sector_operations", "OPERATIONS"),
        )
        if excitation_mode == "coupled":
            operation_mu_bindings += (
                _physical_binding("measurement_likelihood_loss", ("target_four_cell_counts",), ("likelihood_target_mask",), "query_operations", "OPERATIONS"),
                _physical_binding("sector_loss", ("target_sector_probabilities",), ("target_sector_probability_mask",), "sector_rows", "ROWS"),
                _physical_binding("R1_loss", conditional_r1_targets, conditional_r1_masks, "state_rows", "ROWS"),
                _physical_binding("joint_control_covariance_loss", conditional_r1_targets, conditional_r1_masks, "covariance_rows", "ROWS"),
            )
        registry["operation_effective_mu"] = replace(
            registry["operation_effective_mu"],
            objective_bindings=operation_mu_bindings,
        )
        registry["operation_excitation_rate"] = replace(
            registry["operation_excitation_rate"],
            objective_bindings=(
                _physical_binding("excitation_rate_oracle_loss", ("target_excitation_rate_o",), ("target_excitation_rate_o_mask",), "rate_operations", "OPERATIONS"),
                *operation_mu_bindings,
            ),
        )
        registry["operation_sector_probabilities"] = replace(registry["operation_sector_probabilities"], objective_bindings=(
            _physical_binding("operation_sector_loss", ("target_sector_probabilities",), ("target_sector_probability_mask",), "sector_operations", "OPERATIONS"),
        ))
    for name, spec in registry.items():
        if (
            spec.objective_bindings
            and spec.proper_loss != spec.objective_bindings[0].objective_name
        ):
            raise RuntimeError(
                f"physical output primary objective mismatch: {name}"
            )
    return registry


def output_training_registry_for_model(
    model: object,
) -> dict[str, Published_output_training_spec]:
    """Select the physical contract only for the physical AOD wrapper."""
    if type(model).__name__ == "V10_physical_aod_classifier" and hasattr(model, "core"):
        return physical_output_training_registry(getattr(model, "excitation_mode", "legacy"))
    return published_output_registry()


def output_training_metadata(outputs: Mapping) -> dict:
    registry = published_output_registry()
    unknown = set(outputs) - set(registry)
    if unknown:
        raise ValueError(
            f"published classifier outputs lack training contracts: {sorted(unknown)}"
        )
    return {
        "output_training_status": {name: registry[name].status for name in outputs},
        "output_target_coverage": {name: None for name in outputs},
        "uncertainty_source": "DIAGNOSTIC_UNTRAINED_NO_PRODUCTION_CALIBRATION",
    }


def output_training_manifest() -> dict:
    return {
        "schema": "V10PublishedOutputTrainingRegistryV1",
        "step_function": "classifier_registered_step",
        "outputs": {
            name: asdict(spec) for name, spec in published_output_registry().items()
        },
        "feedback_source_eligible": False,
        "held_out_calibration_verified": False,
        "coverage_status": "SEE_SOURCE_BOUND_OUTPUT_AUDIT_NOT_INFERRED_FROM_WEIGHTS",
    }


def physical_output_training_manifest(excitation_mode: str = "legacy") -> dict:
    """Return the exact physical classifier output-consumer contract."""
    from .physical_objective_contract import (
        OBJECTIVE_SUPERVISION_SCHEMA,
        OBJECTIVE_SUPERVISION_SCHEMA_V2,
    )

    if excitation_mode == "legacy":
        return {
        "schema": "V10PhysicalPublishedOutputTrainingRegistryV3",
        "step_function": "physical_classifier_step",
        "objective_supervision_schema": OBJECTIVE_SUPERVISION_SCHEMA,
        "sector_training_scope": "K1_VERSUS_REST_AGGREGATE",
        "feedback_required_outputs": list(physical_feedback_required_outputs),
        "outputs": {
            name: asdict(spec)
            for name, spec in physical_output_training_registry().items()
        },
        "feedback_source_eligible": False,
        "held_out_calibration_verified": False,
        "coverage_status": "SEE_COMMITTED_EFFECTIVE_OBJECTIVE_AUDIT",
        }
    if excitation_mode not in ("diagnostic", "coupled"):
        raise ValueError("unknown physical excitation mode")
    return {
        "schema": "V10PhysicalPublishedOutputTrainingRegistryV4",
        "excitation_mode": excitation_mode,
        "step_function": "physical_classifier_step",
        "objective_supervision_schema": OBJECTIVE_SUPERVISION_SCHEMA_V2,
        "sector_training_scope": "OPERATION_POISSON_K1_VERSUS_REST",
        "feedback_required_outputs": list(physical_feedback_required_outputs),
        "outputs": {name: asdict(spec) for name, spec in physical_output_training_registry(excitation_mode).items()},
        "feedback_source_eligible": False,
        "held_out_calibration_verified": False,
        "coverage_status": "SEE_COMMITTED_EFFECTIVE_OBJECTIVE_AUDIT",
    }


def _canonical_physical_output_manifest_bytes(manifest: object) -> bytes:
    if not isinstance(manifest, Mapping):
        raise ValueError("physical checkpoint output-training manifest mismatch")
    try:
        return json.dumps(
            dict(manifest),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("physical checkpoint output-training manifest mismatch") from error


def physical_output_training_manifest_sha(excitation_mode: str = "legacy") -> str:
    """Hash the exact physical output contract consumed by checkpoints/receipts."""
    return hashlib.sha256(
        _canonical_physical_output_manifest_bytes(
            physical_output_training_manifest(excitation_mode)
        )
    ).hexdigest()


def validate_physical_output_training_manifest(manifest: object) -> dict:
    """Require byte-equivalent semantic content for physical checkpoints."""
    if not isinstance(manifest, Mapping):
        raise ValueError("physical checkpoint output-training manifest mismatch")
    schema = manifest.get("schema")
    if schema == "V10PhysicalPublishedOutputTrainingRegistryV3":
        expected = physical_output_training_manifest()
    elif schema == "V10PhysicalPublishedOutputTrainingRegistryV4":
        expected = physical_output_training_manifest(manifest.get("excitation_mode"))
    else:
        raise ValueError("physical checkpoint output-training manifest mismatch")
    expected_bytes = _canonical_physical_output_manifest_bytes(expected)
    observed_bytes = _canonical_physical_output_manifest_bytes(manifest)
    if observed_bytes != expected_bytes:
        raise ValueError("physical checkpoint output-training manifest mismatch")
    return expected


def require_calibrated_feedback_outputs(
    admission_path: str | None = None,
    expected_bindings: Mapping[str, str] | None = None,
    required_outputs: tuple[str, ...] = feedback_required_outputs,
    excitation_mode: str = "legacy",
) -> dict:
    """Fail closed until a reviewed, source-bound calibration admission exists.

    No constructor flag, finite random variance, or self-declared checkpoint
    field can turn the current untrained covariance heads into a feedback ABI.
    A later proper-objective/calibration change must implement admission as a
    separate reviewed contract; the present pilot does not provide that proof.
    """
    if excitation_mode not in ("legacy", "diagnostic", "coupled"):
        raise ValueError("unknown physical excitation mode")
    if (
        tuple(required_outputs) == physical_feedback_required_outputs
        and excitation_mode != "legacy"
    ):
        raise ValueError(
            "operation-Poisson feedback calibration remains blocked pending a mode-specific protocol"
        )
    if tuple(required_outputs) == physical_feedback_required_outputs:
        registry = physical_output_training_registry()
    elif tuple(required_outputs) == legacy_feedback_required_outputs:
        registry = published_output_registry()
    else:
        raise ValueError("unknown feedback consumer output contract")
    missing = [
        name
        for name in required_outputs
        if registry.get(name) is None or registry[name].proper_loss is None
    ]
    if missing or admission_path is None or expected_bindings is None:
        raise ValueError(
            "untrained or uncalibrated classifier feedback outputs; "
            f"missing production objectives: {missing}; feedback freezing/cache blocked"
        )
    from .calibration_admission import validate_calibration_admission

    return validate_calibration_admission(
        admission_path, expected_bindings, tuple(required_outputs)
    )
