"""Opt-in research training/evaluation glue for the joint command/action path.

This module intentionally does not replace the V4 trainer when all new config
flags are disabled.  It provides bounded unit/smoke-step primitives rather than
a large sweep or hardware workflow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .action_conditioned_model import (
    OperationComplexFeedbackModel,
    autograd_log_rate_jacobian,
    central_difference_log_rate_jacobian,
    fisher_from_integrable_count_map,
)
from .command_contract import PhysicsHybridCommandTableV1
from .command_features import command_tone_features
from .detector_click_model import FourDetectorClickModel, count_negative_log_likelihood
from .probe_system_identification import directional_supervision_loss
from .projective_action import ACTION_GEOMETRY_VERSION, action_dimensions
from .registry_graph import RegistryOperationGraph, graph_action_smoothness

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


RESEARCH_CURRICULUM_STAGES: Tuple[str, ...] = (
    "command_aod_forward_pretrain",
    "write_read_born_count_pretrain",
    "source_detector_nuisance_calibration",
    "joint_signed_probe_phase_adaptation",
    "constrained_inverse_teacher_generation",
    "controller_distillation",
    "real_count_adapter",
    "uq_conformal_calibration",
)


def _training(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(config.get("training") or config)


def validate_joint_research_config(config: Mapping[str, Any]) -> Dict[str, bool]:
    training = _training(config)
    required = (
        "command_conditioning", "detector_click_model", "joint_probe_fit",
        "joint_amp_phase_action", "autograd_count_jacobian",
        "registry_graph_regularization",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise ValueError(f"research config is missing registered keys: {missing}")
    flags = {name: bool(dict(training[name]).get("enabled", False)) for name in required}
    command_config = dict(training["command_conditioning"])
    policy_config = dict(command_config.get("policy_conditioning") or {})
    policy_required = bool(
        policy_config.get("required_for_joint_action", policy_config.get("required", True))
    )
    if (
        flags["command_conditioning"]
        and flags["joint_amp_phase_action"]
        and policy_required
        and not bool(policy_config.get("enabled", False))
    ):
        raise ValueError(
            "joint command-conditioned action requires policy_conditioning.enabled=true"
        )
    if bool(policy_config.get("enabled", False)) and not bool(
        policy_config.get("share_forward_command_encoder", True)
    ):
        raise ValueError("policy conditioning must share the forward command encoder")
    return flags


def build_research_feedback_model(
    config: Mapping[str, Any],
    command_table: PhysicsHybridCommandTableV1,
    *,
    hidden_dim: int = 64,
) -> OperationComplexFeedbackModel:
    if torch is None:
        raise RuntimeError("joint research model requires torch")
    flags = validate_joint_research_config(config)
    if not (flags["command_conditioning"] and flags["joint_amp_phase_action"]):
        raise ValueError("command_conditioning and joint_amp_phase_action must be explicitly enabled")
    features, _ = command_tone_features(command_table)
    training = _training(config)
    action = dict(training["joint_amp_phase_action"])
    command_config = dict(training["command_conditioning"])
    policy_config = dict(command_config.get("policy_conditioning") or {})
    detector_config = dict(training["detector_click_model"])
    detector_model = None
    if flags["detector_click_model"]:
        detector_model = FourDetectorClickModel(
            dark_prior=detector_config.get("dark_prior", (1e-5,) * 4),
            detector_efficiency=detector_config.get("detector_efficiency_prior", (1.0,) * 4),
            time_blocks=int(detector_config.get("time_blocks", 1)),
            time_block_residual_scale=float(detector_config.get("time_block_residual_std_logit", 0.05)),
            dark_prior_std_logit=float(detector_config.get("dark_prior_std_logit", 0.25)),
            efficiency_prior_std_logit=float(detector_config.get("efficiency_prior_std_logit", 0.25)),
        )
    return OperationComplexFeedbackModel(
        command_table.system_type,
        features.shape[-1],
        hidden_dim=hidden_dim,
        brightness_enabled=bool(action.get("brightness_enabled", False)),
        covariance_kind=str(action.get("action_covariance_kind", "full_lower_triangular")),
        detector_model=detector_model,
        policy_conditioning_enabled=bool(policy_config.get("enabled", False)),
        minimum_valid_tone_fraction=float(policy_config.get("minimum_valid_tone_fraction", 0.99)),
        missing_command_covariance_diagonal=float(
            policy_config.get("missing_command_covariance_diagonal", 1.0)
        ),
        gate_initial_bias=float(policy_config.get("gate_initial_bias", -2.0)),
        share_forward_command_encoder=bool(
            policy_config.get("share_forward_command_encoder", True)
        ),
        state_source=str(command_config.get("state_source", "write_command_forward")),
    )


def research_forward(
    model: OperationComplexFeedbackModel,
    observation: Mapping[str, Any],
    command: Any,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run the opt-in joint model and derive J/Fisher from its zero-action map."""
    if torch is None:
        raise RuntimeError("research_forward requires torch")
    flags = validate_joint_research_config(config)
    training = _training(config)
    action_config = dict(training["joint_amp_phase_action"])
    outputs = dict(model(
        observation, command,
        amplitude_limit=float(action_config.get("amplitude_limit", 0.03)),
        phase_limit_rad=float(action_config.get("phase_limit_rad", 0.20)),
        brightness_limit=float(action_config.get("brightness_limit", 0.0)),
    ))
    if not flags["autograd_count_jacobian"]:
        outputs["count_jacobian_source"] = "disabled_opt_in_false"
        return outputs

    zero = torch.zeros(
        (command.batch_size, model.dimensions["total"]),
        dtype=command.tone_features.dtype, device=command.tone_features.device,
    )
    forward_context = outputs["forward_context"]
    zero_output = model.count_forward_model.forward_map(
        observation, command, zero, forward_context=forward_context,
    )
    jacobian = autograd_log_rate_jacobian(
        model.count_forward_model, observation, command, action_point=zero,
        forward_context=forward_context,
        create_graph=bool(model.training),
    )
    jacobian_config = dict(training["autograd_count_jacobian"])
    detector_config = dict(training["detector_click_model"])
    fisher = fisher_from_integrable_count_map(
        zero_output["expected_counts"], jacobian,
        nb2_dispersion=float(detector_config.get("dispersion", 0.0)),
        detector_variance=observation.get("detector_variance"),
        ridge=float(jacobian_config.get("fisher_ridge", 1e-6)),
    )
    eigenvalues = torch.linalg.eigvalsh(fisher["fisher"])
    minimum_information = float(jacobian_config.get("minimum_information_eigenvalue", 1e-6))
    maximum_condition = float(jacobian_config.get("maximum_condition_number", 1e4))
    condition = eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(minimum_information)
    amp_eigen = torch.linalg.eigvalsh(fisher["fisher_aa"])
    phase_eigen = torch.linalg.eigvalsh(fisher["fisher_pp"])
    amp_min = amp_eigen[..., 0]
    phase_min = phase_eigen[..., 0]
    condition_amp = amp_eigen[..., -1] / amp_min.clamp_min(minimum_information)
    condition_phase = phase_eigen[..., -1] / phase_min.clamp_min(minimum_information)
    amplitude_dim = model.dimensions["amplitude"]
    phase_log_rate_jacobian = jacobian[..., amplitude_dim:]
    phase_rate_jacobian = zero_output["expected_rate"].unsqueeze(-1) * phase_log_rate_jacobian
    outputs.update(fisher)
    outputs.update({
        "count_jacobian": jacobian,
        "phase_log_rate_jacobian": phase_log_rate_jacobian,
        "phase_rate_jacobian": phase_rate_jacobian,
        "phase_jacobian_covariance": fisher["posterior_covariance"][..., amplitude_dim:, amplitude_dim:],
        "phase_jacobian_mask": torch.ones_like(phase_log_rate_jacobian, dtype=torch.bool),
        "amplitude_empirical_evidence_mask": torch.zeros_like(
            jacobian[..., :amplitude_dim], dtype=torch.bool,
        ),
        "zero_action_expected_counts": zero_output["expected_counts"],
        "count_jacobian_source": "autograd_same_forward_map",
        "fisher_condition": condition,
        "fisher_condition_amp": condition_amp,
        "fisher_condition_phase": condition_phase,
        "minimum_eigenvalue_amp": amp_min,
        "minimum_eigenvalue_phase": phase_min,
        "amplitude_actionable_mask": (amp_min >= minimum_information) & (condition_amp <= maximum_condition),
        "phase_actionable_mask": (phase_min >= minimum_information) & (condition_phase <= maximum_condition),
    })
    return outputs


@dataclass(frozen=True)
class JointResearchLosses:
    total: Any
    components: Mapping[str, Any]


def joint_research_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    graph: Optional[RegistryOperationGraph] = None,
) -> JointResearchLosses:
    """Mask-aware loss bundle; unavailable real causal labels contribute zero."""
    if torch is None:
        raise RuntimeError("joint_research_loss requires torch")
    training = _training(config)
    action_config = dict(training["joint_amp_phase_action"])
    probe_config = dict(training["joint_probe_fit"])
    jacobian_config = dict(training["autograd_count_jacobian"])
    graph_config = dict(training["registry_graph_regularization"])
    device = outputs["expected_counts"].device
    zero = outputs["expected_counts"].sum() * 0.0

    observed = torch.as_tensor(batch["point_counts"], dtype=outputs["expected_counts"].dtype, device=device)
    point_mask = torch.as_tensor(batch.get("point_count_mask", torch.ones_like(observed)), dtype=observed.dtype, device=device)
    mean = outputs["expected_counts"].clamp_min(1e-8)
    dispersion = float(dict(training.get("detector_click_model") or {}).get("dispersion", 0.0))
    if dispersion > 0.0:
        alpha = torch.as_tensor(dispersion, dtype=mean.dtype, device=device)
        r = 1.0 / alpha
        count_each = -(
            torch.lgamma(observed + r) - torch.lgamma(r) - torch.lgamma(observed + 1.0)
            + r * (torch.log(r) - torch.log(r + mean))
            + observed * (torch.log(mean) - torch.log(r + mean))
        )
    else:
        count_each = mean - observed * torch.log(mean)
    count_nll = (count_each * point_mask).sum() / point_mask.sum().clamp_min(1.0)

    amplitude_geometry = zero
    if batch.get("amplitude_target_coordinates") is not None:
        target = torch.as_tensor(batch["amplitude_target_coordinates"], dtype=mean.dtype, device=device)
        mask = torch.as_tensor(batch.get("amplitude_causal_supervision_mask", 0.0), dtype=mean.dtype, device=device)
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(-1)
        error = (outputs["amplitude_action_coordinates"] - target).square()
        amplitude_geometry = (error * mask).sum() / mask.expand_as(error).sum().clamp_min(1.0)

    phase_circular = zero
    if batch.get("phase_target_coordinates") is not None:
        target = torch.as_tensor(batch["phase_target_coordinates"], dtype=mean.dtype, device=device)
        mask = torch.as_tensor(batch.get("phase_causal_supervision_mask", 0.0), dtype=mean.dtype, device=device)
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(-1)
        delta = outputs["phase_action_coordinates"] - target
        error = 2.0 * (1.0 - torch.cos(delta))
        phase_circular = (error * mask).sum() / mask.expand_as(error).sum().clamp_min(1.0)

    directional = zero
    if all(batch.get(name) is not None for name in ("probe_directions", "directional_target", "directional_variance")):
        if outputs.get("count_jacobian") is None:
            raise ValueError("directional supervision requires count_jacobian from the forward map")
        directional_jacobian = (
            outputs["phase_rate_jacobian"]
            if str(batch.get("directional_target_kind", "log_rate")) == "rate"
            else outputs["phase_log_rate_jacobian"]
        )
        directional = directional_supervision_loss(
            directional_jacobian, batch["probe_directions"],
            batch["directional_target"], batch["directional_variance"],
            mask=batch.get("directional_mask"),
        )

    integrability = zero
    if outputs.get("jacobian_residual_head") is not None and outputs.get("count_jacobian") is not None:
        integrability = F.mse_loss(outputs["jacobian_residual_head"], outputs["count_jacobian"].detach())

    graph_loss = zero
    if bool(graph_config.get("enabled", False)):
        if graph is None:
            raise ValueError("registry graph regularization enabled without a snapshot-bound graph")
        local_action = outputs.get("local_hardware_action")
        if local_action is None:
            # Global coordinates are not repeated over operations.  Graph loss
            # is therefore exactly zero until an explicit local residual exists.
            graph_loss = zero
        else:
            graph_loss = graph_action_smoothness(local_action, graph)

    brightness = outputs.get("brightness_delta", zero)
    brightness_shrink = torch.mean(torch.square(brightness))
    cycle_shrink = (
        torch.square(outputs.get("aod_cycle_residual_norm", outputs.get("cycle_residual_norm", zero)))
        + torch.mean(torch.square(outputs.get("cycle_amplitude_log_gain", zero)))
        + torch.mean(torch.square(outputs.get("cycle_phase_rad", zero)))
    )
    amplitude_shrink = torch.mean(torch.square(outputs.get("local_amplitude_log_gain", outputs["amplitude_action_coordinates"])))

    global_detector = zero
    detector_config = dict(training["detector_click_model"])
    if bool(detector_config.get("enabled", False)):

        for name in ("multi2_pairs", "multi2", "multi3", "multi4"):
            target_name = f"global_{name}_count"
            mean_name = f"global_{name}_mean"
            if batch.get(target_name) is not None and outputs.get(mean_name) is not None:
                global_detector = global_detector + count_negative_log_likelihood(
                    batch[target_name], outputs[mean_name],
                    dispersion=detector_config.get("dispersion"),
                )
    detector_regularization = outputs.get("detector_regularization", zero)
    detector_dark_prior_deviation = outputs.get("detector_dark_prior_deviation", zero)
    detector_efficiency_prior_deviation = outputs.get("detector_efficiency_prior_deviation", zero)
    detector_time_block_residual_norm = outputs.get("detector_time_block_residual_norm", zero)

    feedback_weights = dict(config.get("losses", {}).get("feedback") or {})
    total = (
        float(feedback_weights.get("count_nll", 1.0)) * count_nll
        + float(action_config.get("amplitude_geometry_loss_weight", 0.0)) * amplitude_geometry
        + float(action_config.get("phase_circular_loss_weight", 0.0)) * phase_circular
        + float(probe_config.get("directional_loss_weight", 0.0)) * directional
        + float(jacobian_config.get("integrability_weight", 0.0)) * integrability
        + float(graph_config.get("loss_weight", 0.0)) * graph_loss
        + float(action_config.get("amplitude_shrink_weight", 0.0)) * amplitude_shrink
        + float(action_config.get("brightness_shrink_weight", 0.0)) * brightness_shrink
        + float(action_config.get("cycle_residual_weight", 0.0)) * cycle_shrink
        + float(detector_config.get("detector_prior_shrinkage_weight", 0.0))
        * detector_regularization
        + global_detector
    )
    return JointResearchLosses(total=total, components={
        "command_conditioned_count_nll": count_nll,
        "amplitude_geometry": amplitude_geometry,
        "phase_circular": phase_circular,
        "directional": directional,
        "integrability": integrability,
        "registry_graph": graph_loss,
        "amplitude_shrink": amplitude_shrink,
        "brightness_shrink": brightness_shrink,
        "cycle_residual_shrink": cycle_shrink,
        "global_detector_likelihood": global_detector,
        "detector_regularization": detector_regularization,
        "detector_dark_prior_deviation": detector_dark_prior_deviation,
        "detector_efficiency_prior_deviation": detector_efficiency_prior_deviation,
        "detector_time_block_residual_norm": detector_time_block_residual_norm,
    })


def curriculum_stage(stage_index: int) -> str:
    if not 0 <= int(stage_index) < len(RESEARCH_CURRICULUM_STAGES):
        raise IndexError(f"research curriculum stage must be in [0,{len(RESEARCH_CURRICULUM_STAGES)})")
    return RESEARCH_CURRICULUM_STAGES[int(stage_index)]


def research_checkpoint_metadata(
    config: Mapping[str, Any],
    command_table: Optional[PhysicsHybridCommandTableV1],
    *,
    registry_graph: Optional[RegistryOperationGraph] = None,
    feature_ordered_names: Sequence[str] = (),
) -> Dict[str, Any]:
    training = _training(config)
    config_hashes = {}
    for name in (
        "command_conditioning", "detector_click_model", "joint_probe_fit",
        "joint_amp_phase_action", "autograd_count_jacobian",
        "registry_graph_regularization", "research_curriculum",
    ):
        canonical = json.dumps(dict(training.get(name) or {}), sort_keys=True, separators=(",", ":")).encode("utf-8")
        config_hashes[name] = hashlib.sha256(canonical).hexdigest()
    return {
        "research_only": True,
        "command_schema": None if command_table is None else command_table.schema,
        "command_hash": None if command_table is None else command_table.table_hash,
        "action_geometry_version": ACTION_GEOMETRY_VERSION,
        "detector_model_version": 1,
        "registry_graph_hash": None if registry_graph is None else registry_graph.graph_hash,
        "count_jacobian_source": "autograd_same_forward_map",
        "feature_ordered_names": list(feature_ordered_names),
        "config_hashes": config_hashes,
    }


def evaluate_joint_research_outputs(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    autograd_jacobian: Optional[Any] = None,
    central_jacobian: Optional[Any] = None,
) -> Dict[str, Any]:
    """Emit the complete research evaluation surface with availability flags."""
    def value(name: str, default: float = float("nan")) -> float:
        item = outputs.get(name)
        if item is None:
            return default
        if torch is not None and isinstance(item, torch.Tensor):
            return float(item.detach().float().mean().cpu())
        return float(np.mean(np.asarray(item, dtype=float)))

    metrics = {
        "command_conditioned_heldout_count_nll": value("command_conditioned_heldout_count_nll"),
        "leave_frequency_group_out_nll": value("leave_frequency_group_out_nll"),
        "leave_row_out_nll": value("leave_row_out_nll"),
        "leave_cycle_out_nll": value("leave_cycle_out_nll"),
        "heldout_probe_direction_error": value("heldout_probe_direction_error"),
        "jacobian_cosine_error": value("jacobian_cosine_error"),
        "jacobian_mahalanobis_error": value("jacobian_mahalanobis_error"),
        "detector_dark_calibration_coverage": value("detector_dark_calibration_coverage"),
        "multi2_pair_likelihood": value("multi2_pair_likelihood"),
        "multi3_likelihood": value("multi3_likelihood"),
        "multi4_likelihood": value("multi4_likelihood"),
        "amplitude_coordinate_error": value("amplitude_coordinate_error"),
        "circular_phase_error": value("circular_phase_error"),
        "post_action_state_infidelity": value("post_action_state_infidelity"),
        "amplitude_action_saturation": value("amplitude_action_saturation"),
        "phase_action_saturation": value("phase_action_saturation"),
        "amplitude_skip_rate": value("amplitude_skip_rate"),
        "phase_skip_rate": value("phase_skip_rate"),
        "fisher_block_condition": value("fisher_block_condition"),
        "cycle_residual_norm": value("cycle_residual_norm"),
        "global_versus_local_residual_norm": value("global_versus_local_residual_norm"),
    }
    if autograd_jacobian is not None and central_jacobian is not None:
        if torch is not None and isinstance(autograd_jacobian, torch.Tensor):
            metrics["autograd_central_difference_max_error"] = float(
                torch.max(torch.abs(autograd_jacobian - central_jacobian)).detach().cpu()
            )
        else:
            metrics["autograd_central_difference_max_error"] = float(
                np.max(np.abs(np.asarray(autograd_jacobian) - np.asarray(central_jacobian)))
            )
    else:
        metrics["autograd_central_difference_max_error"] = float("nan")
    metrics["available_metrics"] = sorted(name for name, metric in metrics.items() if name != "available_metrics" and np.isfinite(metric))
    metrics["research_only"] = True
    return metrics


__all__ = [
    "build_research_feedback_model", "research_forward", "validate_joint_research_config",
    "joint_research_loss", "JointResearchLosses", "curriculum_stage",
    "research_checkpoint_metadata", "evaluate_joint_research_outputs",
    "RESEARCH_CURRICULUM_STAGES",
]
