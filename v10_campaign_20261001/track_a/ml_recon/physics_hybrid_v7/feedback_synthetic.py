"""Real-fitted synthetic transitions and trajectories for V7 feedback.

Static measured snapshots cannot identify an action response.  This generator
therefore uses measured train-group nuisance statistics for the observation
domain while keeping action effects explicitly synthetic.  It produces both a
one-step transition for Model A and up to nine causal history tokens for Model
B.  Common nuisance draws are shared across before/after rendering so the
action comparison does not acquire avoidable Monte-Carlo variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ml_recon.physics_hybrid_v5.full_synthetic_generator import sample_nb2
from ml_recon.physics_hybrid_v5.projective_action import (
    amplitude_to_coordinates,
    coordinates_to_amplitude,
    coordinates_to_phase,
    phase_to_coordinates,
)

from .feedback_data import (
    MAX_ACTION_DIM,
    FeedbackMeasuredArraysV1,
    FeedbackNuisanceProfileV1,
    build_operation_features,
    build_target_features,
)
from .geometry import build_operation_frequency_geometry


FEEDBACK_SYNTHETIC_SCHEMA = "v7_feedback_transition_batch_v1"
PHASE_ACTION_MIN_RAD = 0.01

ACTION_MODE_NAMES = (
    "no_op",
    "stop",
    "rollback",
    "phase_only",
    "amplitude_only",
    "combined",
)

MECHANISM_NAMES = (
    "aod_nonlinearity",
    "optical_overlap",
    "motion_dephasing",
    "collision_crosstalk",
    "detector_background",
    "unknown_residual",
)

HISTORY_FEATURE_NAMES = tuple(
    [f"actual_action_{index}" for index in range(MAX_ACTION_DIM)]
    + [
        "utility_delta",
        "fidelity_after",
        "retrieval_delta",
        "log_count_ratio",
        "multi_risk_delta",
        "accepted",
        "rollback",
        "uncertainty",
        "best_so_far",
        "model_hash_changed",
        "calibration_hash_changed",
        "command_hash_changed",
        "step_fraction",
    ]
)


@dataclass(frozen=True)
class FeedbackSyntheticConfig:
    maximum_history: int = 9
    maximum_steps: int = 10
    phase_step_max_rad: float = 0.10
    phase_step_min_rad: float = PHASE_ACTION_MIN_RAD
    phase_cumulative_hard_max_rad: float = 0.80
    amplitude_step_max: float = 0.005
    amplitude_cumulative_max: float = 0.03
    phase_initial_error_std_rad: float = 0.38
    # Keep the synthetic amplitude error inside the reachable cumulative trust
    # region (0.03).  The previous 0.10 prior placed most samples many episodes
    # outside the actuator contract, making amplitude-only control unlearnable.
    amplitude_initial_error_std: float = 0.02
    error_regime_probabilities: Tuple[float, float, float, float] = (
        0.15,
        0.25,
        0.25,
        0.35,
    )
    near_target_amplitude_scale: float = 0.02
    near_target_phase_scale: float = 0.03
    dominated_coordinate_residual_scale: float = 0.03
    phase_response_lower: float = 0.72
    phase_response_upper: float = 1.12
    amplitude_response_lower: float = 0.65
    amplitude_response_upper: float = 1.05
    phase_drift_std_rad: float = 0.025
    crosstalk_scale_max: float = 0.18
    dephasing_lower: float = 0.01
    dephasing_upper: float = 0.30
    multi_probability_lower: float = 0.01
    multi_probability_upper: float = 0.25
    serious_harm_threshold: float = 0.02
    operation_hide_fraction: float = 0.08
    amplitude_alignment_utility_weight: float = 0.012
    phase_alignment_utility_weight: float = 0.004
    policy_minimum_gain: float = 0.0005
    policy_stop_fidelity: float = 0.92
    policy_stop_gain: float = 0.002
    policy_subspace_utility_tolerance: float = 0.0005
    # Early local policies have O(1) normalized action error, so safety
    # supervision marginalizes across the full per-step trust region.  This is
    # epistemic action uncertainty, not additional physical actuator noise.
    policy_action_uncertainty_fraction: float = 1.0
    policy_action_uncertainty_samples: int = 6

    def validate(self) -> None:
        if self.maximum_history != 9 or self.maximum_steps != 10:
            raise ValueError("V7 feedback contract requires history 9 and maximum steps 10")
        if not 0.0 < self.phase_step_min_rad <= self.phase_step_max_rad <= 0.10:
            raise ValueError("phase trust-region bounds are invalid")
        if self.phase_cumulative_hard_max_rad > 0.80:
            raise ValueError("phase cumulative hard ceiling exceeds 0.8 rad")
        if self.amplitude_step_max > 0.005 or self.amplitude_cumulative_max > 0.03:
            raise ValueError("amplitude action exceeds the locked W_CH1 boundary")
        if not 0.0 <= self.policy_minimum_gain <= self.policy_stop_gain <= 0.02:
            raise ValueError("policy gain thresholds are invalid")
        if not 0.0 <= self.policy_subspace_utility_tolerance <= 0.005:
            raise ValueError("policy subspace utility tolerance is invalid")
        if (
            self.amplitude_alignment_utility_weight < 0.0
            or self.phase_alignment_utility_weight < 0.0
        ):
            raise ValueError("alignment utility weights must be non-negative")
        if not 0.0 <= self.policy_action_uncertainty_fraction <= 1.0:
            raise ValueError("policy action uncertainty fraction is invalid")
        if self.policy_action_uncertainty_samples < 1:
            raise ValueError("at least one policy uncertainty sample is required")
        if not 0.0 < self.policy_stop_fidelity <= 1.0:
            raise ValueError("policy stop fidelity is invalid")
        probabilities = np.asarray(self.error_regime_probabilities, dtype=np.float64)
        if probabilities.shape != (4,) or np.any(probabilities <= 0.0):
            raise ValueError("four positive error-regime probabilities are required")
        if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-8):
            raise ValueError("error-regime probabilities must sum to one")
        if not (
            0.0 < self.near_target_amplitude_scale <= 1.0
            and 0.0 < self.near_target_phase_scale <= 1.0
            and 0.0 < self.dominated_coordinate_residual_scale <= 1.0
        ):
            raise ValueError("error-regime scales must be in (0, 1]")


def canonical_action_mask(system_type: str) -> np.ndarray:
    if system_type == "3qubit":
        return np.asarray([1.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    if system_type == "4qubit":
        return np.ones(MAX_ACTION_DIM, dtype=np.float32)
    raise ValueError("system_type must be 3qubit or 4qubit")


def pack_action(
    amplitude_coordinates: np.ndarray,
    phase_coordinates: np.ndarray,
    system_type: str,
) -> np.ndarray:
    value = np.zeros(np.asarray(amplitude_coordinates).shape[:-1] + (MAX_ACTION_DIM,), dtype=np.float64)
    modes = 2 if system_type == "3qubit" else 3
    value[..., :modes] = amplitude_coordinates
    value[..., 3:3 + modes] = phase_coordinates
    return value


def enforce_phase_action_minimum(
    action: np.ndarray,
    minimum_rad: float = PHASE_ACTION_MIN_RAD,
) -> np.ndarray:
    """Map sub-resolution relative-phase proposals to a physical no-op."""

    value = np.asarray(action, dtype=np.float64).copy()
    if value.ndim != 2 or value.shape[1] != MAX_ACTION_DIM:
        raise ValueError("action must have shape [batch, 6]")
    phase_size = np.max(np.abs(value[:, 3:]), axis=1)
    below_resolution = (phase_size > 0.0) & (phase_size < float(minimum_rad))
    value[below_resolution, 3:] = 0.0
    return value


def unpack_action(action: np.ndarray, system_type: str) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(action, dtype=np.float64)
    modes = 2 if system_type == "3qubit" else 3
    return value[..., :modes], value[..., 3:3 + modes]


def generate_feedback_transition_batch(
    arrays: FeedbackMeasuredArraysV1,
    profile: FeedbackNuisanceProfileV1,
    *,
    batch_size: int,
    seed: int,
    config: FeedbackSyntheticConfig | None = None,
) -> Dict[str, np.ndarray]:
    """Generate a physically structured action transition and policy target."""

    config = config or FeedbackSyntheticConfig()
    config.validate()
    if profile.system_type != arrays.system_type or profile.manifest_hash != arrays.manifest_hash:
        raise ValueError("synthetic profile and measured arrays disagree")
    rng = np.random.default_rng(int(seed))
    train_indices = np.flatnonzero(arrays.split_mask("train"))
    selected = rng.choice(train_indices, size=int(batch_size), replace=True)
    system_type = arrays.system_type
    qubit_count = arrays.qubit_count
    modes = qubit_count - 1
    target_amplitude = arrays.target_amplitude[selected].astype(np.float64)
    target_phase = arrays.target_phase_rad[selected].astype(np.float64)
    target_mask = arrays.target_mask[selected].astype(np.float64)
    target_amplitude_active = target_amplitude[:, :qubit_count]
    target_phase_active = target_phase[:, :qubit_count]
    target_amp_coordinate = amplitude_to_coordinates(target_amplitude_active)
    target_phase_coordinate = phase_to_coordinates(target_phase_active)
    error_regime = rng.choice(
        4,
        size=batch_size,
        p=np.asarray(config.error_regime_probabilities, dtype=np.float64),
    )
    amplitude_scale = np.ones(batch_size, dtype=np.float64)
    phase_scale = np.ones(batch_size, dtype=np.float64)
    near_target = error_regime == 0
    amplitude_dominated = error_regime == 1
    phase_dominated = error_regime == 2
    amplitude_scale[near_target] = float(config.near_target_amplitude_scale)
    phase_scale[near_target] = float(config.near_target_phase_scale)
    phase_scale[amplitude_dominated] = float(config.dominated_coordinate_residual_scale)
    amplitude_scale[phase_dominated] = float(config.dominated_coordinate_residual_scale)
    amplitude_error = rng.normal(
        0.0,
        float(config.amplitude_initial_error_std),
        size=(batch_size, modes),
    ) * amplitude_scale[:, None]
    phase_error = rng.normal(
        0.0,
        float(config.phase_initial_error_std_rad),
        size=(batch_size, modes),
    ) * phase_scale[:, None]
    current_amp_coordinate = target_amp_coordinate + amplitude_error
    current_phase_coordinate = target_phase_coordinate + phase_error
    current_amplitude = coordinates_to_amplitude(current_amp_coordinate)
    current_phase = coordinates_to_phase(current_phase_coordinate)

    teacher_amp = np.clip(
        target_amp_coordinate - current_amp_coordinate,
        -float(config.amplitude_step_max),
        float(config.amplitude_step_max),
    )
    teacher_phase = np.clip(
        target_phase_coordinate - current_phase_coordinate,
        -float(config.phase_step_max_rad),
        float(config.phase_step_max_rad),
    )
    teacher_action = enforce_phase_action_minimum(
        pack_action(teacher_amp, teacher_phase, system_type),
        config.phase_step_min_rad,
    )
    action, sampled_action_kind = _sample_training_action(
        teacher_action,
        system_type,
        rng,
        config,
    )
    mechanism_weights = rng.dirichlet(
        np.asarray([1.8, 1.4, 1.2, 1.1, 1.5, 0.8], dtype=np.float64),
        size=batch_size,
    )
    effective_action, response_matrix = _physical_action_response(
        action,
        system_type,
        mechanism_weights,
        rng,
        config,
    )
    effective_amp, effective_phase = unpack_action(effective_action, system_type)
    next_amp_coordinate = current_amp_coordinate + effective_amp
    next_phase_coordinate = current_phase_coordinate + effective_phase
    next_amplitude = coordinates_to_amplitude(next_amp_coordinate)
    next_phase = coordinates_to_phase(next_phase_coordinate)

    dephasing = rng.uniform(config.dephasing_lower, config.dephasing_upper, size=batch_size)
    dephasing *= 0.55 + mechanism_weights[:, MECHANISM_NAMES.index("motion_dephasing")]
    dephasing = np.clip(dephasing, 0.0, 0.45)
    crosstalk_norm = np.linalg.norm(
        response_matrix - np.eye(MAX_ACTION_DIM)[None, :, :],
        axis=(1, 2),
    )
    source_brightness = rng.uniform(0.0, 1.0, size=batch_size)
    multi_probability = (
        config.multi_probability_lower
        + (config.multi_probability_upper - config.multi_probability_lower)
        * np.square(source_brightness)
    )
    multi_probability *= 0.75 + mechanism_weights[:, MECHANISM_NAMES.index("detector_background")]
    multi_probability = np.clip(
        multi_probability,
        config.multi_probability_lower,
        config.multi_probability_upper,
    )
    retrieval_scale = rng.uniform(0.82, 1.12, size=batch_size)
    current_rho = _core_density(current_amplitude, current_phase, dephasing)
    next_dephasing = np.clip(
        dephasing
        + 0.08 * np.linalg.norm(effective_phase, axis=1)
        + 0.04 * np.linalg.norm(effective_amp, axis=1),
        0.0,
        0.55,
    )
    next_rho = _core_density(next_amplitude, next_phase, next_dephasing)
    target_rho = _core_density(target_amplitude_active, target_phase_active, np.zeros(batch_size))
    current_fidelity = _density_overlap(current_rho, target_rho)
    next_fidelity = _density_overlap(next_rho, target_rho)

    shared_nuisance = _draw_count_nuisance(profile, batch_size, rng)
    current_counts = _render_counts(
        arrays,
        profile,
        current_amplitude,
        current_phase,
        retrieval_scale,
        multi_probability,
        mechanism_weights,
        shared_nuisance,
        rng,
    )
    next_counts = _render_counts(
        arrays,
        profile,
        next_amplitude,
        next_phase,
        retrieval_scale,
        multi_probability,
        mechanism_weights,
        shared_nuisance,
        rng,
    )
    visible = np.ones_like(current_counts["coin_counts"], dtype=np.float32)
    hidden_count = max(1, int(round(arrays.operation_count * config.operation_hide_fraction)))
    for row in range(batch_size):
        hidden = rng.choice(arrays.operation_count, size=hidden_count, replace=False)
        visible[row, hidden] = 0.0
    geometry = build_operation_frequency_geometry(system_type)
    operation_features = build_operation_features(
        coin_counts=current_counts["coin_counts"],
        signal_counts=current_counts["signal_counts"],
        idler_counts=current_counts["idler_counts"],
        read_count=current_counts["read_count"],
        target_amplitude=target_amplitude,
        target_phase_rad=target_phase,
        geometry=geometry,
        static_operation_features=arrays.static_operation_features,
        retrieval_relative_map=arrays.retrieval_relative_map,
        observed_coin_mask=visible,
    )
    amplitude_alignment_gain = np.clip(
        (
            np.square(current_amp_coordinate - target_amp_coordinate).sum(axis=1)
            - np.square(next_amp_coordinate - target_amp_coordinate).sum(axis=1)
        ) / max(float(config.amplitude_cumulative_max) ** 2, 1e-12),
        -1.0,
        1.0,
    )
    phase_alignment_gain = np.clip(
        (
            np.square(current_phase_coordinate - target_phase_coordinate).sum(axis=1)
            - np.square(next_phase_coordinate - target_phase_coordinate).sum(axis=1)
        ) / max(float(config.phase_cumulative_hard_max_rad) ** 2, 1e-12),
        -1.0,
        1.0,
    )
    utility = (
        next_fidelity - current_fidelity
        + float(config.amplitude_alignment_utility_weight) * amplitude_alignment_gain
        + float(config.phase_alignment_utility_weight) * phase_alignment_gain
        - 0.05 * np.square(action).sum(axis=1)
        - 0.10 * np.maximum(multi_probability - 0.10, 0.0)
    )
    action_induced_utility = utility + 0.10 * np.maximum(
        multi_probability - 0.10, 0.0
    )
    harmful = action_induced_utility < -float(config.serious_harm_threshold)
    log_gsi = _state_log_gsi(next_counts)
    nuisance_targets = np.stack((
        retrieval_scale,
        log_gsi,
        multi_probability,
        next_dephasing,
        crosstalk_norm,
    ), axis=1)
    history, history_mask = _generate_history(
        current_fidelity,
        multi_probability,
        teacher_action,
        rng,
        config,
    )
    action_mask = np.broadcast_to(canonical_action_mask(system_type), action.shape).copy()
    batch = {
        "schema": np.asarray(FEEDBACK_SYNTHETIC_SCHEMA),
        "system_type": np.asarray(system_type),
        "operation_features": operation_features.astype(np.float32),
        "target_features": build_target_features(
            target_amplitude,
            target_phase,
            target_mask,
        ).astype(np.float32),
        "action": action.astype(np.float32),
        "action_mask": action_mask.astype(np.float32),
        "sampled_action_kind": sampled_action_kind.astype(np.int64),
        "error_regime": error_regime.astype(np.int64),
        "current_coin_counts": current_counts["coin_counts"].astype(np.float32),
        "current_signal_counts": current_counts["signal_counts"].astype(np.float32),
        "current_idler_counts": current_counts["idler_counts"].astype(np.float32),
        "current_read_count": current_counts["read_count"].astype(np.float32),
        "next_coin_counts": next_counts["coin_counts"].astype(np.float32),
        "next_expected_counts": next_counts["expected_counts"].astype(np.float32),
        "next_count_loss_mask": np.ones_like(next_counts["coin_counts"], dtype=np.float32),
        "nb2_dispersion": next_counts["nb2_dispersion"].astype(np.float32),
        "current_rho": current_rho.astype(np.complex64),
        "next_rho": next_rho.astype(np.complex64),
        "rho_available_mask": np.ones(batch_size, dtype=np.float32),
        "utility": utility.astype(np.float32),
        "action_induced_utility": action_induced_utility.astype(np.float32),
        "amplitude_alignment_gain": amplitude_alignment_gain.astype(np.float32),
        "phase_alignment_gain": phase_alignment_gain.astype(np.float32),
        "harmful": harmful.astype(np.float32),
        "nuisance_targets": nuisance_targets.astype(np.float32),
        "mechanism_weights": mechanism_weights.astype(np.float32),
        "history": history.astype(np.float32),
        "history_mask": history_mask.astype(np.float32),
        "teacher_action": teacher_action.astype(np.float32),
        "teacher_mode": np.zeros(batch_size, dtype=np.int64),
        "teacher_gain": np.zeros(batch_size, dtype=np.float32),
        "teacher_harmful": np.zeros(batch_size, dtype=np.float32),
        "current_fidelity": current_fidelity.astype(np.float32),
        "next_fidelity": next_fidelity.astype(np.float32),
        "current_amplitude_coordinate": current_amp_coordinate.astype(np.float32),
        "current_phase_coordinate": current_phase_coordinate.astype(np.float32),
        "target_amplitude_coordinate": target_amp_coordinate.astype(np.float32),
        "target_phase_coordinate": target_phase_coordinate.astype(np.float32),
        "response_matrix": response_matrix.astype(np.float32),
        "current_dephasing": dephasing.astype(np.float32),
        "multi_probability": multi_probability.astype(np.float32),
        "retrieval_scale": retrieval_scale.astype(np.float32),
        "nuisance_signal_probability": np.asarray(
            shared_nuisance["signal_probability"], dtype=np.float32,
        ),
        "nuisance_idler_probability": np.asarray(
            shared_nuisance["idler_probability"], dtype=np.float32,
        ),
        "nuisance_coin_total": np.asarray(shared_nuisance["coin_total"], dtype=np.float32),
        "nuisance_signal_total": np.asarray(shared_nuisance["signal_total"], dtype=np.float32),
        "nuisance_idler_total": np.asarray(shared_nuisance["idler_total"], dtype=np.float32),
        "nuisance_read_count": np.asarray(shared_nuisance["read_count"], dtype=np.float32),
        "transition_supervision_mask": np.ones(batch_size, dtype=np.float32),
    }
    batch.update(_policy_teacher_targets(
        batch,
        ideal_teacher_action=teacher_action,
        history=history,
        history_mask=history_mask,
        config=config,
        rng=rng,
    ))
    return batch


def evaluate_counterfactual_action(
    batch: Mapping[str, np.ndarray],
    action: np.ndarray,
    *,
    config: FeedbackSyntheticConfig | None = None,
) -> Dict[str, np.ndarray]:
    """Evaluate an action against the batch's held-out synthetic mechanism.

    This uses the same response matrix, AOD saturation and dephasing law as the
    transition generator, but integrates out the zero-mean phase drift.  It is
    therefore suitable for policy screening without asking Model A to judge
    Model B's own action.
    """

    config = config or FeedbackSyntheticConfig()
    config.validate()
    system_type = str(np.asarray(batch["system_type"]).item())
    value = enforce_phase_action_minimum(action, config.phase_step_min_rad)
    response = np.asarray(batch["response_matrix"], dtype=np.float64)
    if value.ndim != 2 or response.shape != (value.shape[0], MAX_ACTION_DIM, MAX_ACTION_DIM):
        raise ValueError("counterfactual action and response shapes disagree")
    action_mask = canonical_action_mask(system_type)[None, :]
    value = value * action_mask
    raw_effective = np.einsum("bij,bj->bi", response, value)
    mechanism = np.asarray(batch["mechanism_weights"], dtype=np.float64)
    nonlinearity = mechanism[:, MECHANISM_NAMES.index("aod_nonlinearity")][:, None]
    limits = np.asarray(
        [config.amplitude_step_max] * 3 + [config.phase_step_max_rad] * 3,
        dtype=np.float64,
    )
    effective = limits * np.tanh(
        raw_effective / np.maximum(limits, 1e-8) * (1.0 - 0.30 * nonlinearity)
    )
    effective *= action_mask
    effective_amplitude, effective_phase = unpack_action(effective, system_type)
    current_amplitude_coordinate = np.asarray(
        batch["current_amplitude_coordinate"], dtype=np.float64,
    )
    current_phase_coordinate = np.asarray(
        batch["current_phase_coordinate"], dtype=np.float64,
    )
    next_amplitude_coordinate = current_amplitude_coordinate + effective_amplitude
    next_phase_coordinate = current_phase_coordinate + effective_phase
    next_amplitude = coordinates_to_amplitude(next_amplitude_coordinate)
    next_phase = coordinates_to_phase(next_phase_coordinate)
    current_dephasing = np.asarray(batch["current_dephasing"], dtype=np.float64)
    next_dephasing = np.clip(
        current_dephasing
        + 0.08 * np.linalg.norm(effective_phase, axis=1)
        + 0.04 * np.linalg.norm(effective_amplitude, axis=1),
        0.0,
        0.55,
    )
    next_rho = _core_density(next_amplitude, next_phase, next_dephasing)
    target_amplitude = coordinates_to_amplitude(
        np.asarray(batch["target_amplitude_coordinate"], dtype=np.float64),
    )
    target_phase = coordinates_to_phase(
        np.asarray(batch["target_phase_coordinate"], dtype=np.float64),
    )
    target_rho = _core_density(
        target_amplitude,
        target_phase,
        np.zeros(value.shape[0], dtype=np.float64),
    )
    next_fidelity = _density_overlap(next_rho, target_rho)
    current_fidelity = np.asarray(batch["current_fidelity"], dtype=np.float64)
    multi_probability = np.asarray(batch["multi_probability"], dtype=np.float64)
    target_amplitude_coordinate = np.asarray(
        batch["target_amplitude_coordinate"], dtype=np.float64,
    )
    target_phase_coordinate = np.asarray(
        batch["target_phase_coordinate"], dtype=np.float64,
    )
    amplitude_error_before = np.square(
        current_amplitude_coordinate - target_amplitude_coordinate
    ).sum(axis=1)
    amplitude_error_after = np.square(
        next_amplitude_coordinate - target_amplitude_coordinate
    ).sum(axis=1)
    phase_error_before = np.square(
        current_phase_coordinate - target_phase_coordinate
    ).sum(axis=1)
    phase_error_after = np.square(
        next_phase_coordinate - target_phase_coordinate
    ).sum(axis=1)
    amplitude_alignment_gain = np.clip(
        (amplitude_error_before - amplitude_error_after)
        / max(float(config.amplitude_cumulative_max) ** 2, 1e-12),
        -1.0,
        1.0,
    )
    phase_alignment_gain = np.clip(
        (phase_error_before - phase_error_after)
        / max(float(config.phase_cumulative_hard_max_rad) ** 2, 1e-12),
        -1.0,
        1.0,
    )
    utility = (
        next_fidelity
        - current_fidelity
        + float(config.amplitude_alignment_utility_weight) * amplitude_alignment_gain
        + float(config.phase_alignment_utility_weight) * phase_alignment_gain
        - 0.05 * np.square(value).sum(axis=1)
        - 0.10 * np.maximum(multi_probability - 0.10, 0.0)
    )
    action_induced_utility = utility + 0.10 * np.maximum(
        multi_probability - 0.10, 0.0
    )
    return {
        "effective_action": effective.astype(np.float32),
        "next_amplitude_coordinate": next_amplitude_coordinate.astype(np.float32),
        "next_phase_coordinate": next_phase_coordinate.astype(np.float32),
        "next_dephasing": next_dephasing.astype(np.float32),
        "next_rho": next_rho.astype(np.complex64),
        "next_fidelity": next_fidelity.astype(np.float32),
        "amplitude_alignment_gain": amplitude_alignment_gain.astype(np.float32),
        "phase_alignment_gain": phase_alignment_gain.astype(np.float32),
        "utility": utility.astype(np.float32),
        "action_induced_utility": action_induced_utility.astype(np.float32),
        "harmful": (
            action_induced_utility < -float(config.serious_harm_threshold)
        ).astype(np.float32),
    }


def render_feedback_latent_observation(
    arrays: FeedbackMeasuredArraysV1,
    profile: FeedbackNuisanceProfileV1,
    latent_state: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Render a fresh 20-minute-style count observation from latent state.

    The caller owns the causal state transition and rollback logic.  This
    function only maps the current Aitchison/Helmert state through the same
    measured-profile count model used for one-step synthetic supervision.
    """

    system_type = arrays.system_type
    amplitude_coordinate = np.asarray(
        latent_state["current_amplitude_coordinate"], dtype=np.float64,
    )
    phase_coordinate = np.asarray(
        latent_state["current_phase_coordinate"], dtype=np.float64,
    )
    amplitude = coordinates_to_amplitude(amplitude_coordinate)
    phase = coordinates_to_phase(phase_coordinate)
    batch_size = amplitude.shape[0]
    nuisance = {
        "signal_probability": np.asarray(
            latent_state["nuisance_signal_probability"], dtype=np.float64,
        ),
        "idler_probability": np.asarray(
            latent_state["nuisance_idler_probability"], dtype=np.float64,
        ),
        "coin_total": np.asarray(latent_state["nuisance_coin_total"], dtype=np.float64),
        "signal_total": np.asarray(latent_state["nuisance_signal_total"], dtype=np.float64),
        "idler_total": np.asarray(latent_state["nuisance_idler_total"], dtype=np.float64),
        "read_count": np.asarray(latent_state["nuisance_read_count"], dtype=np.float64),
    }
    counts = _render_counts(
        arrays,
        profile,
        amplitude,
        phase,
        np.asarray(latent_state["retrieval_scale"], dtype=np.float64),
        np.asarray(latent_state["multi_probability"], dtype=np.float64),
        np.asarray(latent_state["mechanism_weights"], dtype=np.float64),
        nuisance,
        np.random.default_rng(int(seed)),
    )
    qubit_count = arrays.qubit_count
    target_amplitude = coordinates_to_amplitude(
        np.asarray(latent_state["target_amplitude_coordinate"], dtype=np.float64),
    )
    target_phase = coordinates_to_phase(
        np.asarray(latent_state["target_phase_coordinate"], dtype=np.float64),
    )
    padded_amplitude = np.zeros((batch_size, 4), dtype=np.float32)
    padded_phase = np.zeros_like(padded_amplitude)
    padded_mask = np.zeros_like(padded_amplitude)
    padded_amplitude[:, :qubit_count] = target_amplitude.astype(np.float32)
    padded_phase[:, :qubit_count] = target_phase.astype(np.float32)
    padded_mask[:, :qubit_count] = 1.0
    operation_features = build_operation_features(
        coin_counts=counts["coin_counts"],
        signal_counts=counts["signal_counts"],
        idler_counts=counts["idler_counts"],
        read_count=counts["read_count"],
        target_amplitude=padded_amplitude,
        target_phase_rad=padded_phase,
        geometry=build_operation_frequency_geometry(system_type),
        static_operation_features=arrays.static_operation_features,
        retrieval_relative_map=arrays.retrieval_relative_map,
        observed_coin_mask=np.ones_like(counts["coin_counts"], dtype=np.float32),
    )
    current_rho = _core_density(
        amplitude,
        phase,
        np.asarray(latent_state["current_dephasing"], dtype=np.float64),
    )
    target_rho = _core_density(
        target_amplitude,
        target_phase,
        np.zeros(batch_size, dtype=np.float64),
    )
    return {
        "operation_features": operation_features.astype(np.float32),
        "target_features": build_target_features(
            padded_amplitude, padded_phase, padded_mask,
        ).astype(np.float32),
        "coin_counts": counts["coin_counts"].astype(np.float32),
        "signal_counts": counts["signal_counts"].astype(np.float32),
        "idler_counts": counts["idler_counts"].astype(np.float32),
        "read_count": counts["read_count"].astype(np.float32),
        "current_rho": current_rho.astype(np.complex64),
        "current_fidelity": _density_overlap(current_rho, target_rho).astype(np.float32),
        "log_gsi": _state_log_gsi(counts).astype(np.float32),
    }


def generate_feedback_policy_batch(
    arrays: FeedbackMeasuredArraysV1,
    profile: FeedbackNuisanceProfileV1,
    *,
    batch_size: int,
    seed: int,
    config: FeedbackSyntheticConfig | None = None,
) -> Dict[str, np.ndarray]:
    """Generate a policy batch with history causally bound to current state.

    Each history action is evaluated by the same response/crosstalk forward
    map used for the final teacher.  Accepted actions advance the latent state;
    rejected actions never become parents; serious harm and two consecutive
    no-improvement rounds stop the rollout.  This prevents Model B from
    learning against histories that are statistically plausible but physically
    unrelated to the state it must control.
    """

    config = config or FeedbackSyntheticConfig()
    config.validate()
    base = generate_feedback_transition_batch(
        arrays,
        profile,
        batch_size=batch_size,
        seed=int(seed),
        config=config,
    )
    rng = np.random.default_rng(int(seed) + 104729)
    state = {
        key: np.asarray(base[key]).copy()
        for key in (
            "current_amplitude_coordinate",
            "current_phase_coordinate",
            "target_amplitude_coordinate",
            "target_phase_coordinate",
            "response_matrix",
            "current_dephasing",
            "multi_probability",
            "retrieval_scale",
            "mechanism_weights",
            "nuisance_signal_probability",
            "nuisance_idler_probability",
            "nuisance_coin_total",
            "nuisance_signal_total",
            "nuisance_idler_total",
            "nuisance_read_count",
        )
    }
    state["system_type"] = np.asarray(arrays.system_type)
    history = np.zeros(
        (batch_size, config.maximum_history, len(HISTORY_FEATURE_NAMES)),
        dtype=np.float32,
    )
    history_mask = np.zeros(
        (batch_size, config.maximum_history), dtype=np.float32,
    )
    requested_lengths = rng.integers(
        0, config.maximum_history + 1, size=batch_size,
    )
    active = np.ones(batch_size, dtype=bool)
    stalled_rounds = np.zeros(batch_size, dtype=np.int64)
    cumulative_action = np.zeros((batch_size, MAX_ACTION_DIM), dtype=np.float32)
    previous_count_total: np.ndarray | None = None
    best_fidelity = np.asarray(base["current_fidelity"], dtype=np.float32).copy()
    cumulative_limits = np.asarray(
        [config.amplitude_cumulative_max] * 3
        + [config.phase_cumulative_hard_max_rad] * 3,
        dtype=np.float32,
    )
    for step in range(config.maximum_history):
        participating = active & (step < requested_lengths)
        if not np.any(participating):
            continue
        observation = render_feedback_latent_observation(
            arrays,
            profile,
            state,
            seed=int(seed + 100003 * (step + 1)),
        )
        current_fidelity = np.asarray(
            observation["current_fidelity"], dtype=np.float32,
        )
        teacher_amplitude = np.clip(
            np.asarray(state["target_amplitude_coordinate"], dtype=np.float64)
            - np.asarray(state["current_amplitude_coordinate"], dtype=np.float64),
            -config.amplitude_step_max,
            config.amplitude_step_max,
        )
        teacher_phase = np.clip(
            np.asarray(state["target_phase_coordinate"], dtype=np.float64)
            - np.asarray(state["current_phase_coordinate"], dtype=np.float64),
            -config.phase_step_max_rad,
            config.phase_step_max_rad,
        )
        ideal_action = enforce_phase_action_minimum(
            pack_action(teacher_amplitude, teacher_phase, arrays.system_type),
            config.phase_step_min_rad,
        )
        behavior_action = _sample_policy_rollout_action(
            ideal_action,
            arrays.system_type,
            rng,
            config,
        )
        behavior_action[~participating] = 0.0
        behavior_action = np.clip(
            behavior_action,
            -cumulative_limits[None, :] - cumulative_action,
            cumulative_limits[None, :] - cumulative_action,
        )
        behavior_action = enforce_phase_action_minimum(
            behavior_action,
            config.phase_step_min_rad,
        ).astype(np.float32)
        state["current_fidelity"] = current_fidelity
        outcome = evaluate_counterfactual_action(
            state,
            behavior_action,
            config=config,
        )
        utility_delta = np.asarray(
            outcome["action_induced_utility"], dtype=np.float32,
        )
        next_fidelity = np.asarray(outcome["next_fidelity"], dtype=np.float32)
        serious_harm = (
            np.asarray(outcome["harmful"], dtype=np.float32) > 0.5
        ) & participating
        accepted = (
            (utility_delta >= config.policy_minimum_gain)
            & participating
            & ~serious_harm
            & (np.linalg.norm(behavior_action, axis=1) > 1e-8)
        )
        for state_name, outcome_name in (
            ("current_amplitude_coordinate", "next_amplitude_coordinate"),
            ("current_phase_coordinate", "next_phase_coordinate"),
            ("current_dephasing", "next_dephasing"),
        ):
            state[state_name][accepted] = np.asarray(outcome[outcome_name])[accepted]
        cumulative_action[accepted] += behavior_action[accepted]
        fidelity_after = current_fidelity.copy()
        fidelity_after[accepted] = next_fidelity[accepted]
        best_fidelity = np.maximum(best_fidelity, fidelity_after)
        no_improvement = participating & ~accepted & ~serious_harm
        stalled_rounds[accepted] = 0
        stalled_rounds[no_improvement] += 1
        plateau_stop = no_improvement & (stalled_rounds >= 2)
        count_total = np.sum(observation["coin_counts"], axis=1)
        if previous_count_total is None:
            log_count_ratio = np.zeros(batch_size, dtype=np.float32)
        else:
            log_count_ratio = np.log(
                (count_total + 1.0) / (previous_count_total + 1.0)
            ).astype(np.float32)
        token = np.concatenate((
            behavior_action,
            np.stack((
                utility_delta,
                fidelity_after,
                np.zeros(batch_size, dtype=np.float32),
                log_count_ratio,
                np.zeros(batch_size, dtype=np.float32),
                accepted.astype(np.float32),
                serious_harm.astype(np.float32),
                np.full(batch_size, 0.05, dtype=np.float32),
                best_fidelity,
                np.zeros(batch_size, dtype=np.float32),
                np.zeros(batch_size, dtype=np.float32),
                np.ones(batch_size, dtype=np.float32),
                np.full(
                    batch_size,
                    float(step + 1) / float(config.maximum_steps),
                    dtype=np.float32,
                ),
            ), axis=1),
        ), axis=1)
        history[participating, step] = token[participating]
        history_mask[participating, step] = 1.0
        previous_count_total = count_total
        active[serious_harm | plateau_stop] = False

    final_observation = render_feedback_latent_observation(
        arrays,
        profile,
        state,
        seed=int(seed + 999983),
    )
    final_teacher_amplitude = np.clip(
        np.asarray(state["target_amplitude_coordinate"], dtype=np.float64)
        - np.asarray(state["current_amplitude_coordinate"], dtype=np.float64),
        -config.amplitude_step_max,
        config.amplitude_step_max,
    )
    final_teacher_phase = np.clip(
        np.asarray(state["target_phase_coordinate"], dtype=np.float64)
        - np.asarray(state["current_phase_coordinate"], dtype=np.float64),
        -config.phase_step_max_rad,
        config.phase_step_max_rad,
    )
    final_teacher_action = enforce_phase_action_minimum(
        pack_action(
            final_teacher_amplitude,
            final_teacher_phase,
            arrays.system_type,
        ),
        config.phase_step_min_rad,
    )
    base.update({
        "operation_features": final_observation["operation_features"],
        "target_features": final_observation["target_features"],
        "current_coin_counts": final_observation["coin_counts"],
        "current_signal_counts": final_observation["signal_counts"],
        "current_idler_counts": final_observation["idler_counts"],
        "current_read_count": final_observation["read_count"],
        "current_rho": final_observation["current_rho"],
        "rho_available_mask": np.ones(batch_size, dtype=np.float32),
        "current_fidelity": final_observation["current_fidelity"],
        "current_amplitude_coordinate": np.asarray(
            state["current_amplitude_coordinate"], dtype=np.float32,
        ),
        "current_phase_coordinate": np.asarray(
            state["current_phase_coordinate"], dtype=np.float32,
        ),
        "current_dephasing": np.asarray(
            state["current_dephasing"], dtype=np.float32,
        ),
        "history": history,
        "history_mask": history_mask,
        "teacher_action": final_teacher_action.astype(np.float32),
        "causal_policy_history": np.ones(batch_size, dtype=np.float32),
    })
    base.update(_policy_teacher_targets(
        base,
        ideal_teacher_action=final_teacher_action,
        history=history,
        history_mask=history_mask,
        config=config,
        rng=rng,
    ))
    return base


def _sample_policy_rollout_action(
    teacher_action: np.ndarray,
    system_type: str,
    rng: np.random.Generator,
    config: FeedbackSyntheticConfig,
) -> np.ndarray:
    """Sample mostly safe on-policy history actions plus sparse exploration."""

    batch_size = teacher_action.shape[0]
    kind = rng.choice(
        7,
        size=batch_size,
        p=[0.05, 0.38, 0.24, 0.12, 0.12, 0.06, 0.03],
    )
    action = teacher_action.copy()
    for row, selected in enumerate(kind):
        if selected == 0:
            action[row] = 0.0
        elif selected == 1:
            action[row] *= rng.uniform(0.75, 1.0)
        elif selected == 2:
            action[row] *= rng.uniform(0.45, 0.70)
        elif selected == 3:
            action[row, :3] = 0.0
        elif selected == 4:
            action[row, 3:] = 0.0
        elif selected == 5:
            action[row] += rng.normal(0.0, 0.12, size=MAX_ACTION_DIM) * np.asarray(
                [config.amplitude_step_max] * 3
                + [config.phase_step_max_rad] * 3,
            )
        else:
            action[row] = -rng.uniform(0.60, 1.0) * teacher_action[row]
    action[:, :3] = np.clip(
        action[:, :3], -config.amplitude_step_max, config.amplitude_step_max,
    )
    action[:, 3:] = np.clip(
        action[:, 3:], -config.phase_step_max_rad, config.phase_step_max_rad,
    )
    action *= canonical_action_mask(system_type)[None, :]
    return enforce_phase_action_minimum(action, config.phase_step_min_rad)


def _sample_training_action(
    teacher_action: np.ndarray,
    system_type: str,
    rng: np.random.Generator,
    config: FeedbackSyntheticConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    batch = teacher_action.shape[0]
    kind = rng.choice(7, size=batch, p=[0.10, 0.24, 0.14, 0.12, 0.16, 0.12, 0.12])
    action = teacher_action.copy()
    mask = canonical_action_mask(system_type)
    for row, value in enumerate(kind):
        if value == 0:
            action[row] = 0.0
        elif value == 1:
            action[row] *= 0.5
        elif value == 2:
            action[row, :3] = 0.0
        elif value == 3:
            action[row, 3:] = 0.0
        elif value == 4:
            action[row] = -teacher_action[row]
        elif value == 5:
            action[row] += rng.normal(0.0, 0.25, size=MAX_ACTION_DIM) * np.asarray(
                [config.amplitude_step_max] * 3 + [config.phase_step_max_rad] * 3
            )
        else:
            action[row] = rng.uniform(-1.0, 1.0, size=MAX_ACTION_DIM) * np.asarray(
                [config.amplitude_step_max] * 3 + [config.phase_step_max_rad] * 3
            )
    action[:, :3] = np.clip(action[:, :3], -config.amplitude_step_max, config.amplitude_step_max)
    action[:, 3:] = np.clip(action[:, 3:], -config.phase_step_max_rad, config.phase_step_max_rad)
    action *= mask[None, :]
    action = enforce_phase_action_minimum(action, config.phase_step_min_rad)
    return action, kind


def _physical_action_response(
    action: np.ndarray,
    system_type: str,
    mechanism_weights: np.ndarray,
    rng: np.random.Generator,
    config: FeedbackSyntheticConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    batch = action.shape[0]
    action_mask = canonical_action_mask(system_type)
    response = np.broadcast_to(np.eye(MAX_ACTION_DIM), (batch, MAX_ACTION_DIM, MAX_ACTION_DIM)).copy()
    amp_gain = rng.uniform(config.amplitude_response_lower, config.amplitude_response_upper, size=(batch, 3))
    phase_gain = rng.uniform(config.phase_response_lower, config.phase_response_upper, size=(batch, 3))
    diagonal = np.concatenate((amp_gain, phase_gain), axis=1)
    response *= diagonal[:, :, None]
    for row in range(batch):
        crosstalk = rng.normal(0.0, 1.0, size=(MAX_ACTION_DIM, MAX_ACTION_DIM))
        np.fill_diagonal(crosstalk, 0.0)
        scale = config.crosstalk_scale_max * mechanism_weights[row, MECHANISM_NAMES.index("collision_crosstalk")]
        response[row] += scale * crosstalk / max(float(np.linalg.norm(crosstalk, ord=2)), 1e-8)
    response *= action_mask[None, :, None] * action_mask[None, None, :]
    effective = np.einsum("bij,bj->bi", response, action)
    nonlinearity = mechanism_weights[:, MECHANISM_NAMES.index("aod_nonlinearity")][:, None]
    limits = np.asarray(
        [config.amplitude_step_max] * 3 + [config.phase_step_max_rad] * 3,
        dtype=np.float64,
    )
    effective = limits * np.tanh(effective / np.maximum(limits, 1e-8) * (1.0 - 0.30 * nonlinearity))
    drift = rng.normal(0.0, config.phase_drift_std_rad, size=(batch, 3))
    effective[:, 3:] += drift * action_mask[None, 3:]
    effective *= action_mask[None, :]
    return effective, response


def _core_density(
    amplitude: np.ndarray,
    phase: np.ndarray,
    dephasing: np.ndarray,
) -> np.ndarray:
    state = amplitude * np.exp(1j * phase)
    state /= np.maximum(np.linalg.norm(state, axis=1, keepdims=True), 1e-12)
    density = state[:, :, None] * state.conj()[:, None, :]
    diagonal = np.zeros_like(density)
    indices = np.arange(state.shape[1])
    diagonal[:, indices, indices] = np.real(density[:, indices, indices])
    weight = np.asarray(dephasing, dtype=np.float64)[:, None, None]
    return (1.0 - weight) * density + weight * diagonal


def _density_overlap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("bij,bji->b", first, second)).clip(0.0, 1.0)


def _draw_count_nuisance(
    profile: FeedbackNuisanceProfileV1,
    batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    signal_probability = _draw_probability_modes(
        profile.signal_probability_mean,
        profile.signal_log_share_modes,
        batch_size,
        rng,
    )
    idler_probability = _draw_probability_modes(
        profile.idler_probability_mean,
        profile.idler_log_share_modes,
        batch_size,
        rng,
    )
    return {
        "signal_probability": signal_probability,
        "idler_probability": idler_probability,
        "coin_total": np.maximum(
            1.0,
            np.expm1(rng.normal(profile.log_coin_total_mean, profile.log_coin_total_std, size=batch_size)),
        ),
        "signal_total": np.maximum(
            1.0,
            np.expm1(rng.normal(profile.log_signal_total_mean, profile.log_signal_total_std, size=batch_size)),
        ),
        "idler_total": np.maximum(
            1.0,
            np.expm1(rng.normal(profile.log_idler_total_mean, profile.log_idler_total_std, size=batch_size)),
        ),
        "read_count": np.maximum(
            1.0,
            np.expm1(rng.normal(profile.log_read_count_mean, profile.log_read_count_std, size=batch_size)),
        ),
    }


def _draw_probability_modes(
    mean_probability: np.ndarray,
    modes: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coefficient = rng.normal(0.0, 0.35, size=(batch_size, modes.shape[0]))
    log_probability = np.log(np.clip(mean_probability, 1e-12, None))[None, :] + coefficient @ modes
    log_probability -= log_probability.max(axis=1, keepdims=True)
    probability = np.exp(log_probability)
    return probability / probability.sum(axis=1, keepdims=True)


def _render_counts(
    arrays: FeedbackMeasuredArraysV1,
    profile: FeedbackNuisanceProfileV1,
    amplitude: np.ndarray,
    phase: np.ndarray,
    retrieval_scale: np.ndarray,
    multi_probability: np.ndarray,
    mechanism_weights: np.ndarray,
    nuisance: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    geometry = build_operation_frequency_geometry(arrays.system_type)
    batch = amplitude.shape[0]
    born = np.stack([
        geometry.target_measurement_overlap({
            "amplitudes": amplitude[index],
            "phases_rad": phase[index],
        })
        for index in range(batch)
    ])
    retrieval_vectors = np.asarray(profile.retrieval_rate_vectors, dtype=np.float64)
    population = np.square(np.abs(amplitude))
    effective_retrieval = np.einsum("bq,oq->bo", population, retrieval_vectors)
    born *= effective_retrieval
    born /= np.maximum(born.sum(axis=1, keepdims=True), 1e-12)
    signal_probability = np.asarray(nuisance["signal_probability"], dtype=np.float64)
    idler_probability = np.asarray(nuisance["idler_probability"], dtype=np.float64)
    accidental = signal_probability * idler_probability
    accidental /= np.maximum(accidental.sum(axis=1, keepdims=True), 1e-12)
    background_fraction = 0.03 + 0.16 * mechanism_weights[:, MECHANISM_NAMES.index("detector_background")]
    coherent_mid = profile.coherent_fraction_quantiles[1]
    coherent_fraction = np.clip(
        coherent_mid * retrieval_scale * (1.0 - 0.55 * multi_probability),
        0.10,
        0.90,
    )
    accidental_fraction = np.clip(1.0 - coherent_fraction - background_fraction, 0.02, 0.80)
    normalization = coherent_fraction + accidental_fraction + background_fraction
    coherent_fraction /= normalization
    accidental_fraction /= normalization
    background_fraction /= normalization
    probability = (
        coherent_fraction[:, None] * born
        + accidental_fraction[:, None] * accidental
        + background_fraction[:, None] / float(arrays.operation_count)
    )
    probability /= probability.sum(axis=1, keepdims=True)
    expected_counts = np.asarray(nuisance["coin_total"], dtype=np.float64)[:, None] * probability
    alpha = np.clip(
        profile.nb2_dispersion * (0.8 + 1.5 * multi_probability),
        1e-5,
        0.75,
    )
    coin_counts = np.stack([
        sample_nb2(expected_counts[index], float(alpha[index]), rng)
        for index in range(batch)
    ]).astype(np.float64)
    signal_counts = rng.poisson(
        np.asarray(nuisance["signal_total"], dtype=np.float64)[:, None] * signal_probability,
    ).astype(np.float64)
    idler_counts = rng.poisson(
        np.asarray(nuisance["idler_total"], dtype=np.float64)[:, None] * idler_probability,
    ).astype(np.float64)
    coin_counts = np.minimum(coin_counts, np.minimum(signal_counts, idler_counts))
    return {
        "coin_counts": coin_counts,
        "signal_counts": signal_counts,
        "idler_counts": idler_counts,
        "read_count": np.asarray(nuisance["read_count"], dtype=np.float64),
        "expected_counts": expected_counts,
        "nb2_dispersion": alpha,
    }


def _state_log_gsi(counts: Mapping[str, np.ndarray]) -> np.ndarray:
    coin = np.asarray(counts["coin_counts"], dtype=np.float64).sum(axis=1)
    signal = np.asarray(counts["signal_counts"], dtype=np.float64).sum(axis=1)
    idler = np.asarray(counts["idler_counts"], dtype=np.float64).sum(axis=1)
    read = np.asarray(counts["read_count"], dtype=np.float64)
    return np.log(
        (coin + 0.5) * np.maximum(read, 1.0)
        / ((signal + 0.5) * (idler + 0.5))
    )


def _generate_history(
    current_fidelity: np.ndarray,
    multi_probability: np.ndarray,
    teacher_action: np.ndarray,
    rng: np.random.Generator,
    config: FeedbackSyntheticConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    batch = current_fidelity.size
    history = np.zeros((batch, config.maximum_history, len(HISTORY_FEATURE_NAMES)), dtype=np.float64)
    mask = np.zeros((batch, config.maximum_history), dtype=np.float64)
    lengths = rng.integers(0, config.maximum_history + 1, size=batch)
    for row, length in enumerate(lengths):
        fidelity = max(0.0, current_fidelity[row] - rng.uniform(0.0, 0.20))
        best = fidelity
        for step in range(int(length)):
            scale = 0.5 + 0.5 * (step + 1) / max(length, 1)
            action = teacher_action[row] * scale + rng.normal(0.0, 0.08, MAX_ACTION_DIM) * np.asarray(
                [config.amplitude_step_max] * 3 + [config.phase_step_max_rad] * 3
            )
            action = enforce_phase_action_minimum(
                action[None, :], config.phase_step_min_rad,
            )[0]
            gain = rng.normal(0.025, 0.025) * (1.0 - fidelity)
            if rng.random() < 0.12:
                gain = -abs(rng.normal(0.025, 0.015))
            after = float(np.clip(fidelity + gain, 0.0, 1.0))
            accepted = gain >= 0.0
            rollback = gain < -config.serious_harm_threshold
            if accepted:
                fidelity = after
                best = max(best, after)
            token = np.concatenate((
                action,
                np.asarray([
                    gain,
                    after,
                    rng.normal(0.0, 0.01),
                    rng.normal(gain, 0.04),
                    rng.normal(0.0, 0.02) + 0.1 * multi_probability[row],
                    float(accepted),
                    float(rollback),
                    abs(rng.normal(0.04, 0.02)),
                    best,
                    0.0,
                    0.0,
                    float(step > 0),
                    float(step + 1) / float(config.maximum_steps),
                ]),
            ))
            history[row, step] = token
            mask[row, step] = 1.0
    return history, mask


def _policy_teacher_targets(
    batch: Mapping[str, np.ndarray],
    *,
    ideal_teacher_action: np.ndarray,
    history: np.ndarray,
    history_mask: np.ndarray,
    config: FeedbackSyntheticConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Choose a policy target by physical counterfactual utility.

    The old norm-threshold rule made phase-only, amplitude-only and no-op
    effectively unreachable because both sampled error coordinates were almost
    always nonzero.  Here all actionable subspaces are evaluated through the
    same nonlinear/crosstalk response used by the simulator.  Stop and rollback
    remain episode-level decisions and therefore always carry a zero action.
    """

    value = np.asarray(ideal_teacher_action, dtype=np.float64)
    batch_size = value.shape[0]
    no_op = np.zeros_like(value)
    phase_only = value.copy()
    phase_only[:, :3] = 0.0
    amplitude_only = value.copy()
    amplitude_only[:, 3:] = 0.0
    candidates = np.stack((
        no_op,
        0.25 * phase_only,
        0.50 * phase_only,
        phase_only,
        amplitude_only,
        value,
    ), axis=1)
    candidates = enforce_phase_action_minimum(
        candidates.reshape(-1, MAX_ACTION_DIM),
        config.phase_step_min_rad,
    ).reshape(batch_size, 6, MAX_ACTION_DIM)
    outcomes = [
        evaluate_counterfactual_action(batch, candidates[:, index], config=config)
        for index in range(candidates.shape[1])
    ]
    utility = np.stack(
        [np.asarray(outcome["utility"], dtype=np.float64) for outcome in outcomes],
        axis=1,
    )
    next_fidelity = np.stack(
        [np.asarray(outcome["next_fidelity"], dtype=np.float64) for outcome in outcomes],
        axis=1,
    )
    selected_index = np.argmax(utility, axis=1)
    # Prefer a lower-dimensional control subspace when it is effectively tied
    # with the best utility.  This is a trust-region prior, not a fabricated
    # label: all alternatives have already passed through the same physical
    # forward map, and the tolerance is smaller than the minimum actionable
    # gain.  The ordering keeps no-op safest, then lets phase-only and
    # amplitude-only compete on their actual utility before combined control.
    best_utility = np.max(utility, axis=1)
    tolerance = float(config.policy_subspace_utility_tolerance)
    for row in range(batch_size):
        eligible = np.flatnonzero(utility[row] >= best_utility[row] - tolerance)
        if 0 in eligible:
            selected_index[row] = 0
        else:
            phase_indices = np.asarray([1, 2, 3], dtype=np.int64)
            phase_index = int(phase_indices[np.argmax(utility[row, phase_indices])])
            phase_eligible = phase_index in eligible
            amplitude_eligible = 4 in eligible
            if phase_eligible and amplitude_eligible:
                selected_index[row] = (
                    phase_index if utility[row, phase_index] >= utility[row, 4] else 4
                )
            elif phase_eligible:
                selected_index[row] = phase_index
            elif amplitude_eligible:
                selected_index[row] = 4
    rows = np.arange(batch_size)
    selected_action = candidates[rows, selected_index].copy()
    current_fidelity = np.asarray(batch["current_fidelity"], dtype=np.float64)
    selected_gain = utility[rows, selected_index] - utility[:, 0]
    mode_lookup = np.asarray([
        ACTION_MODE_NAMES.index("no_op"),
        ACTION_MODE_NAMES.index("phase_only"),
        ACTION_MODE_NAMES.index("phase_only"),
        ACTION_MODE_NAMES.index("phase_only"),
        ACTION_MODE_NAMES.index("amplitude_only"),
        ACTION_MODE_NAMES.index("combined"),
    ])
    mode = mode_lookup[selected_index]
    no_improvement = selected_gain < float(config.policy_minimum_gain)
    stop = (
        (current_fidelity >= float(config.policy_stop_fidelity))
        & (selected_gain < float(config.policy_stop_gain))
    )
    accepted_index = HISTORY_FEATURE_NAMES.index("accepted")
    utility_index = HISTORY_FEATURE_NAMES.index("utility_delta")
    for row in range(history.shape[0]):
        length = int(np.sum(history_mask[row]))
        if length >= 2:
            tail = history[row, length - 2:length]
            if np.all(
                (tail[:, accepted_index] <= 0.5)
                | (tail[:, utility_index] < config.policy_minimum_gain)
            ):
                stop[row] = True
    rank_index = selected_index.copy()
    # Candidate zero is part of Model A's rank set.  If the best actionable
    # candidate does not clear the minimum physical utility, the rank target
    # itself must be no-op; otherwise training says "execute" while the policy
    # label says "no-op" for the same transition.
    rank_index[no_improvement] = 0
    mode[no_improvement] = ACTION_MODE_NAMES.index("no_op")
    mode[stop] = ACTION_MODE_NAMES.index("stop")
    for row in range(history.shape[0]):
        length = int(np.sum(history_mask[row]))
        if length > 0:
            rollback_index = HISTORY_FEATURE_NAMES.index("rollback")
            if history[row, length - 1, rollback_index] > 0.5:
                mode[row] = ACTION_MODE_NAMES.index("rollback")
    inactive = np.isin(mode, [
        ACTION_MODE_NAMES.index("no_op"),
        ACTION_MODE_NAMES.index("stop"),
        ACTION_MODE_NAMES.index("rollback"),
    ])
    selected_action[inactive] = 0.0
    selected_gain[inactive] = 0.0
    limits = np.asarray(
        [config.amplitude_step_max] * 3 + [config.phase_step_max_rad] * 3,
        dtype=np.float64,
    )
    harmful_probability = np.zeros(batch_size, dtype=np.float64)
    for _ in range(int(config.policy_action_uncertainty_samples)):
        perturbed = selected_action + rng.normal(size=selected_action.shape) * (
            float(config.policy_action_uncertainty_fraction) * limits[None, :]
        )
        perturbed = np.clip(perturbed, -limits[None, :], limits[None, :])
        perturbed[inactive] = 0.0
        harmful_probability += np.asarray(
            evaluate_counterfactual_action(batch, perturbed, config=config)["harmful"],
            dtype=np.float64,
        )
    harmful_probability /= float(config.policy_action_uncertainty_samples)
    return {
        "teacher_action": value.astype(np.float32),
        "teacher_selected_action": selected_action.astype(np.float32),
        "teacher_mode": mode.astype(np.int64),
        "teacher_gain": selected_gain.astype(np.float32),
        "teacher_harmful": harmful_probability.astype(np.float32),
        "candidate_actions": candidates.astype(np.float32),
        "candidate_utilities": utility.astype(np.float32),
        "candidate_fidelity_gain": (next_fidelity - current_fidelity[:, None]).astype(
            np.float32
        ),
        "candidate_harmful": np.stack(
            [np.asarray(outcome["harmful"], dtype=np.float32) for outcome in outcomes],
            axis=1,
        ),
        "teacher_candidate_index": rank_index.astype(np.int64),
    }


__all__ = [
    "ACTION_MODE_NAMES",
    "FEEDBACK_SYNTHETIC_SCHEMA",
    "FeedbackSyntheticConfig",
    "HISTORY_FEATURE_NAMES",
    "MECHANISM_NAMES",
    "PHASE_ACTION_MIN_RAD",
    "canonical_action_mask",
    "enforce_phase_action_minimum",
    "evaluate_counterfactual_action",
    "generate_feedback_policy_batch",
    "generate_feedback_transition_batch",
    "pack_action",
    "render_feedback_latent_observation",
    "unpack_action",
]
