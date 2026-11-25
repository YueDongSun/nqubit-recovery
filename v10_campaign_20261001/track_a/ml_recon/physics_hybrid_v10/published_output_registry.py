"""Declare production-classifier supervision without inferring trained authority."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class Published_output_training_spec:
    output_name: str
    authoritative_targets: tuple[str, ...] = ()
    target_masks: tuple[str, ...] = ()
    proper_loss: str | None = None
    promotion_required: bool = False
    downstream_consumers: tuple[str, ...] = ()
    status: str = "DIAGNOSTIC_UNTRAINED"


feedback_required_outputs = (
    "W_populations", "W_coherence_re", "W_coherence_im",
    "W_amplitude_variance", "W_phase_variance",
)
objective_outputs = (
    (("p_clean",), ("target_p_clean",), (), "measurement_and_denoise"),
    (("fine_family_logits", "fine_family_probs"), ("target_fine",),
     ("target_fine_mask", "target_fine_identifiable_mask"), "small_n_family_and_calibration"),
    (("pair_entanglement_probs", "pair_entanglement_logits"), ("target_pair",),
     ("target_pair_mask",), "pair_map"),
    (("hyperedge_entanglement_probs",), ("target_hyperedge",),
     ("target_hyperedge_mask",), "hyperedge"),
    (("partition_link_probs", "partition_link_logits", "partition_antichain_scores"),
     ("target_partition",), ("target_partition_mask",), "partition"),
    (("depth_posterior",), ("target_depth",), ("target_structure_mask",), "depth"),
    (("intactness_posterior",), ("target_intactness",), ("target_structure_mask",), "intactness"),
    (("stretchability_posterior",), ("target_stretchability",),
     ("target_structure_mask",), "stretchability"),
    (("coarse_template_posterior",), ("target_coarse",), ("target_coarse_mask",), "coarse"),
    (("R1_re", "R1_im", "W_populations",
      "W_coherence_re", "W_coherence_im"), ("target_R1_re", "target_R1_im"),
     ("target_w_mask",), "w_r1"),
    (("p0", "p1", "p_ge2", "leakage"), ("target_sector",), ("target_w_mask",), "w_sector"),
    (("operational_noise_logits", "operational_noise_posterior"),
     ("target_operational_noise",), ("target_operational_noise_mask",), "operational_noise"),
    (("OOD_probability",), ("target_ood",), (), "ood"),
)
diagnostic_outputs = (
    "certificate_status_logits", "certificate_status_probs", "model_uncertainty",
    "W_amplitude_variance", "W_phase_variance", "amplitude_alr_covariance",
    "phase_helmert_covariance", "joint_control_covariance", "amplitude_error",
    "phase_error", "recoverability", "W_uncertainty", "entanglement_depth_score",
    "covariance_clean_diag", "nuisance_posterior_mean", "nuisance_posterior_variance",
    "abstention_probability",
)
metadata_outputs = ("stretchability_values", "partition_representation",
                    "R1_factor_re", "R1_factor_im",
                    "fine_family_primary_report_allowed", "semantic_latent")


def published_output_registry() -> dict[str, Published_output_training_spec]:
    registry = {}
    for outputs, targets, masks, loss in objective_outputs:
        for name in outputs:
            registry[name] = Published_output_training_spec(
                name, targets, masks, loss, name in feedback_required_outputs,
                ("feedback_semantic_adapter", "mainline_bundle") if name in feedback_required_outputs else (),
                "OBJECTIVE_REGISTERED_NOT_VALIDATED")
    for name in diagnostic_outputs:
        required = name in feedback_required_outputs or name.startswith("certificate_status")
        registry[name] = Published_output_training_spec(
            name, promotion_required=required,
            downstream_consumers=("feedback_semantic_adapter", "mainline_bundle")
            if name in feedback_required_outputs else ())
    for name in metadata_outputs:
        registry[name] = Published_output_training_spec(name, status="REPRESENTATION_OR_METADATA_ONLY")
    return registry


def output_training_metadata(outputs: Mapping) -> dict:
    registry = published_output_registry()
    unknown = set(outputs) - set(registry)
    if unknown:
        raise ValueError(f"published classifier outputs lack training contracts: {sorted(unknown)}")
    return {"output_training_status": {name: registry[name].status for name in outputs},
            "output_target_coverage": {name: None for name in outputs},
            "uncertainty_source": "DIAGNOSTIC_UNTRAINED_NO_PRODUCTION_CALIBRATION"}


def output_training_manifest() -> dict:
    return {"schema": "V10PublishedOutputTrainingRegistryV1",
            "step_function": "classifier_registered_step",
            "outputs": {name: asdict(spec) for name, spec in published_output_registry().items()},
            "feedback_source_eligible": False,
            "held_out_calibration_verified": False,
            "coverage_status": "SEE_SOURCE_BOUND_OUTPUT_AUDIT_NOT_INFERRED_FROM_WEIGHTS"}


def require_calibrated_feedback_outputs() -> None:
    """Fail closed until a reviewed, source-bound calibration admission exists.

    No constructor flag, finite random variance, or self-declared checkpoint
    field can turn the current untrained covariance heads into a feedback ABI.
    A later proper-objective/calibration change must implement admission as a
    separate reviewed contract; the present pilot does not provide that proof.
    """
    registry = published_output_registry()
    missing = [name for name in feedback_required_outputs if registry[name].proper_loss is None]
    raise ValueError("untrained or uncalibrated classifier feedback outputs; "
                     f"missing production objectives: {missing}; feedback freezing/cache blocked")
