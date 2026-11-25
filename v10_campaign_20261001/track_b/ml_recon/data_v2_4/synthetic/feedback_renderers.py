"""Phase-sensitive W feedback samples with estimated same-forward analytic control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import math
import numpy as np

from ml_recon.physics_hybrid_v10.control_tangent import (
    aitchison_amplitude_coordinates,
    apply_tangent_action,
    helmert_matrix,
    phase_coordinates,
)
from ml_recon.physics_hybrid_v10.measurement_model import born_probabilities
from ml_recon.physics_hybrid_v10.noise_interventions import (
    V24NoiseIntervention,
    registered_feedback_settings,
    w_density_matrix,
)
from ml_recon.physics_hybrid_v10.noise_matching import (
    evaluated_severity_matching_kind, hellinger_squared, legacy_severity_matching_kind,
    state_fidelity, validate_severity_matching_kind,
)
from ml_recon.physics_hybrid_v10.recoverability import (
    recoverability,
    same_forward_jacobian,
)
from ml_recon.physics_hybrid_v10.state_backends import (
    DenseDensityStateBackend,
    V10StateBackend,
)
from ml_recon.physics_hybrid_v10.w_renderer import nonlinear_corrective_oracle
from ml_recon.physics_hybrid_v10.csm_observation_contract import (
    CartesianFeedbackAccumulator,
    CSM_FEEDBACK_CARTESIAN_STATS_V1,
    make_csm_observation_contract,
)
from ml_recon.physics_hybrid_v10.forward_instance import forward_callable_sha

from ..hashing import sha256_array, sha256_json
from .noise_renderers import render_noise_truth
from .trackb_teacher_policy import (
    TRACKB_TEACHER_POLICY_V1,
    TRACKB_TEACHER_POLICY_V2,
    TRACKB_TEACHER_POLICY_V2_SHA256,
    TRACKB_TEACHER_POLICY_V2_CELL_IDS,
    validate_trackb_policy_namespace,
    validate_trackb_v2_sample_key,
)
from .trackb_phase_lift import trackb_receding_horizon_action_v2


CONTROL_AMPLITUDE_BOUND = 0.35
CONTROL_PHASE_BOUND = 0.70
CONTROL_RIDGE_WEIGHT = 1e-3
RECOVERABLE_CLASS_THRESHOLD = 0.50
LOW_RECOVERABILITY_STOP_THRESHOLD = 0.25
FEEDBACK_HISTORY_ROUNDS = 5
TRACKB_TEN_ROUND_HISTORY = 10
TRACKB_RECEDING_HORIZON_DAMPING = 0.35
TRACKB_RECEDING_HORIZON_POLICY = "receding_horizon_target_oracle_v1"
FIXED_FRACTION_HISTORY_POLICY = "fixed_fraction_teacher_v1"


def _sample_detected_setting(
    backend: V10StateBackend,
    axes: tuple[str, ...],
    uniforms: np.ndarray,
    intervention: V24NoiseIntervention,
    physical_parameter: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcomes, state_probability = born_probabilities(backend, axes)
    apparatus_probability = intervention.apply_apparatus_channel(
        state_probability, outcomes, physical_parameter)
    detector_law = intervention.apply_detector_channel(
        apparatus_probability, physical_parameter)
    indices = np.searchsorted(np.cumsum(detector_law), uniforms, side="right")
    detected = indices < len(outcomes)
    sampled = np.zeros((len(uniforms), backend.N), dtype=np.int8)
    sampled[detected] = outcomes[indices[detected]]
    return sampled, detected, outcomes


def phase_sensitive_w_measurement(coefficients: np.ndarray, shots: int | np.ndarray,
                                  uniforms: np.ndarray) -> Dict[str, np.ndarray]:
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    return phase_sensitive_density_measurement(
        w_density_matrix(coefficients), shots, uniforms,
        V24NoiseIntervention("identity", "none", "none", 0.0, "none"), 0.0)


def phase_sensitive_density_measurement(
    rho: np.ndarray,
    shots: int | np.ndarray,
    uniforms: np.ndarray,
    intervention: V24NoiseIntervention,
    physical_parameter: float,
    *,
    ideal_probability: bool = False,
) -> Dict[str, np.ndarray]:
    backend = DenseDensityStateBackend(np.asarray(rho, dtype=np.complex128))
    N = backend.N
    required_settings = 1 + 2 * (N - 1)
    shots_by_setting = np.broadcast_to(
        np.asarray(shots, dtype=np.int64), (required_settings,)).copy()
    if np.any(shots_by_setting <= 0):
        raise ValueError("feedback shots must be positive")
    maximum_shots = int(shots_by_setting.max())
    if uniforms.shape != (required_settings, maximum_shots):
        raise ValueError("feedback measurement uniforms have the wrong registered shape")
    cartesian_accumulator = CartesianFeedbackAccumulator(N)

    def setting_statistics(axes: tuple[str, ...], setting_index: int
                           ) -> tuple[np.ndarray, np.ndarray, float]:
        setting_shots = int(shots_by_setting[setting_index])
        if ideal_probability:
            outcomes, state_probability = born_probabilities(backend, axes)
            apparatus = intervention.apply_apparatus_channel(
                state_probability, outcomes, physical_parameter)
            law = intervention.apply_detector_channel(apparatus, physical_parameter)
            cartesian_accumulator.add_setting_counts(
                setting_index, law * setting_shots, setting_shots,
                sampling_enabled=False)
            return outcomes, law[:-1], float(law[:-1].sum())
        sampled, detected, outcomes = _sample_detected_setting(
            backend, axes, uniforms[setting_index, :setting_shots],
            intervention, physical_parameter)
        observed = np.zeros(len(outcomes) + 1, dtype=np.float64)
        if np.any(detected):
            outcome_indices = ((sampled[detected] > 0).astype(np.int64)
                               @ (2 ** np.arange(N - 1, -1, -1)))
            observed[:-1] = np.bincount(
                outcome_indices, minlength=len(outcomes)).astype(np.float64)
        observed[-1] = float(setting_shots - int(detected.sum()))
        observed /= setting_shots
        cartesian_accumulator.add_setting_counts(
            setting_index, observed * setting_shots, setting_shots)
        return (sampled, detected.astype(np.float64) / setting_shots,
                float(detected.mean()))

    z_outcomes, z_weight, z_detection = setting_statistics(
        tuple("Z" for _ in range(N)), 0)
    populations = np.sum((z_outcomes < 0) * z_weight[:, None], axis=0)
    detection_fractions = [z_detection]
    coherence = np.zeros(N, dtype=np.complex128)
    coherence[0] = max(populations[0], 1e-8)
    cursor = 1
    for mode in range(1, N):
        xx_axes = ["Z"] * N
        xy_axes = ["Z"] * N
        xx_axes[0] = xx_axes[mode] = "X"
        xy_axes[0], xy_axes[mode] = "X", "Y"
        xx, xx_weight, xx_detection = setting_statistics(tuple(xx_axes), cursor)
        xy, xy_weight, xy_detection = setting_statistics(tuple(xy_axes), cursor + 1)
        cursor += 2
        detection_fractions.extend((xx_detection, xy_detection))
        re = 0.5 * np.sum(xx[:, 0] * xx[:, mode] * xx_weight)
        im = -0.5 * np.sum(xy[:, 0] * xy[:, mode] * xy_weight)
        coherence[mode] = re + 1j * im
    reference = max(np.sqrt(max(populations[0], 1e-8)), 1e-4)
    estimated = np.sqrt(np.maximum(populations, 1e-8)).astype(np.complex128)
    estimated[0] = reference
    for mode in range(1, N):
        estimated[mode] *= np.exp(1j * np.angle(coherence[mode] + 1e-12))
    estimated /= np.linalg.norm(estimated)
    phase_proxy = np.angle(estimated * estimated[0].conj()) / np.pi
    tokens = np.stack((populations, phase_proxy), axis=-1).astype(np.float32)
    z_shots = int(shots_by_setting[0])
    variance = (np.zeros_like(populations) if ideal_probability else
                np.maximum(populations * (1.0 - populations) / z_shots, 1e-6))
    covariance = np.stack((variance, np.sqrt(variance)), axis=-1).astype(np.float32)
    detection_fraction = np.asarray(detection_fractions, dtype=np.float32)
    cartesian_result = cartesian_accumulator.finalize()
    cartesian_observation_contract = make_csm_observation_contract(
        CSM_FEEDBACK_CARTESIAN_STATS_V1)
    return {"tokens": tokens, "covariance": covariance,
            "estimated_coefficients": estimated,
            "detection_fraction": detection_fraction,
            "cartesian_stats": cartesian_result["stats"],
            "cartesian_covariance": cartesian_result["covariance"],
            "cartesian_statistic_valid_mask": cartesian_result["valid_mask"],
            "cartesian_detection_fraction": cartesian_result["detection_fraction"],
            "cartesian_covariance_estimator_kind": cartesian_result[
                "covariance_estimator_kind"],
            "cartesian_covariance_instance_sha": cartesian_result[
                "covariance_instance_sha"],
            "cartesian_observation_contract_sha": cartesian_observation_contract.sha256(),
            "cartesian_observation_contract_kind": cartesian_observation_contract.kind,
            "cartesian_statistic_transform_sha": cartesian_result["statistic_transform_sha"],
            "cartesian_dirichlet_alpha": cartesian_result["dirichlet_alpha"],
            "cartesian_whitening_support_kind": "THRESHOLDED_COVARIANCE_EIGEN_SUPPORT",
            "cartesian_geometry_source": "PLUGIN_MEASURED_STATE_IDENTITY_FORWARD"}


def _feedback_shots(setting_count: int, shot_mode: str,
                    shot_budget: int) -> np.ndarray:
    if shot_mode not in ("S_total", "S_setting", "S_ideal") or shot_budget <= 0:
        raise ValueError("invalid registered feedback shot contract")
    if shot_mode == "S_total":
        if shot_budget < setting_count:
            raise ValueError("S_total feedback budget must cover every setting")
        quotient, remainder = divmod(shot_budget, setting_count)
        return np.asarray([
            quotient + int(index < remainder) for index in range(setting_count)
        ], dtype=np.int64)
    return np.full(setting_count, shot_budget, dtype=np.int64)


def _deployable_analytic_action(
    estimated_coefficients: np.ndarray,
    target: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Solve an action using observed estimates only; no latent noise truth."""
    N = len(target)
    helmert = helmert_matrix(N)
    identity = V24NoiseIntervention("identity", "none", "none", 0.0, "none")
    exact_shots = np.ones(1 + 2 * (N - 1), dtype=np.int64)
    exact_panel = np.zeros((len(exact_shots), 1), dtype=np.float64)

    def forward(flat_action: np.ndarray) -> np.ndarray:
        action = np.asarray(flat_action, dtype=np.float64).reshape(N - 1, 2)
        state = apply_tangent_action(
            estimated_coefficients, action[:, 0], action[:, 1])
        measured = phase_sensitive_density_measurement(
            w_density_matrix(state), exact_shots, exact_panel, identity, 0.0,
            ideal_probability=True)
        return measured["tokens"].astype(np.float64).reshape(-1)

    zero = np.zeros(2 * (N - 1), dtype=np.float64)
    target_tokens = phase_sensitive_density_measurement(
        w_density_matrix(target), exact_shots, exact_panel, identity, 0.0,
        ideal_probability=True)["tokens"].astype(np.float64).reshape(-1)
    residual = target_tokens - forward(zero)
    jacobian = same_forward_jacobian(forward, zero, epsilon=1e-4)
    covariance_diagonal = np.maximum(
        np.asarray(covariance, dtype=np.float64).reshape(-1), 1e-5)
    lower = np.tile((-CONTROL_AMPLITUDE_BOUND, -CONTROL_PHASE_BOUND), N - 1)
    upper = np.tile((CONTROL_AMPLITUDE_BOUND, CONTROL_PHASE_BOUND), N - 1)
    solved = recoverability(
        residual, jacobian, covariance_diagonal, lambda_u=CONTROL_RIDGE_WEIGHT,
        bounds=(lower, upper), solver_backend="box_coordinate")
    coordinates = np.asarray(solved["u_star"], dtype=np.float64).reshape(N - 1, 2)
    solved = {
        **solved,
        "actual_lambda_u": CONTROL_RIDGE_WEIGHT,
        "action_coordinate_layout": "logical_interleaved_amp_phase",
        "solver_backend": "box_coordinate",
        "action_forward_sha": forward_callable_sha(forward),
        "projection_contract_sha": sha256_json({
            "kind": "helmert_mode_to_logical",
            "N": N,
        }),
    }
    return np.stack((helmert.T @ coordinates[:, 0],
                     helmert.T @ coordinates[:, 1]), axis=-1), solved


def _full_forward_oracle_action(current: np.ndarray, target: np.ndarray,
                                intervention,
                                settings: tuple[tuple[str, ...], ...],
                                physical_parameter: float
                                ) -> Dict[str, object]:
    """Solve the bounded action against the complete noisy measurement forward."""
    N = len(target)
    helmert = helmert_matrix(N)
    identity = V24NoiseIntervention("identity", "none", "none", 0.0, "none")
    target_law = identity.measurement_law(w_density_matrix(target), settings, 0.0)
    lower = np.tile((-CONTROL_AMPLITUDE_BOUND, -CONTROL_PHASE_BOUND), N - 1)
    upper = np.tile((CONTROL_AMPLITUDE_BOUND, CONTROL_PHASE_BOUND), N - 1)

    def forward(flat_action: np.ndarray) -> np.ndarray:
        action = np.asarray(flat_action, dtype=np.float64).reshape(N - 1, 2)
        state = apply_tangent_action(current, action[:, 0], action[:, 1])
        return intervention.measurement_law(
            w_density_matrix(state), settings, physical_parameter)

    def objective(flat_action: np.ndarray) -> float:
        residual = hellinger_squared(forward(flat_action), target_law)
        return float(residual + CONTROL_RIDGE_WEIGHT * np.dot(flat_action, flat_action))

    zero = np.zeros(2 * (N - 1), dtype=np.float64)
    solver_status = "not_run"
    solver_success = False
    try:
        from scipy.optimize import minimize
        result = minimize(objective, zero, method="L-BFGS-B", bounds=tuple(zip(lower, upper)),
                          options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
        solver_status = str(result.message)
        solver_success = bool(result.success and np.all(np.isfinite(result.x)))
        solved = np.asarray(result.x if solver_success else zero, dtype=np.float64)
    except ImportError:
        jacobian = same_forward_jacobian(forward, zero, epsilon=1e-4)
        residual = target_law - forward(zero)
        fallback = recoverability(
            residual, jacobian, np.maximum(target_law * 0.0 + 1e-5, 1e-5),
            lambda_u=CONTROL_RIDGE_WEIGHT, bounds=(lower, upper),
            solver_backend="box_coordinate")
        solved = np.asarray(fallback["u_star"], dtype=np.float64)
        solver_success = bool(fallback["solver_success"])
        solver_status = str(fallback["solver_status"])
    objective_before = objective(zero)
    objective_after = objective(solved)
    constraint_violation = float(max(
        np.maximum(lower - solved, 0.0).max(initial=0.0),
        np.maximum(solved - upper, 0.0).max(initial=0.0)))
    solver_success = bool(
        solver_success and constraint_violation <= 1e-9
        and objective_after <= objective_before + 1e-12)
    coordinates = solved.reshape(N - 1, 2)
    action = np.stack((helmert.T @ coordinates[:, 0],
                       helmert.T @ coordinates[:, 1]), axis=-1)
    return {
        "action": action,
        "solver_success": solver_success,
        "solver_status": solver_status,
        "objective_before": float(objective_before),
        "objective_after": float(objective_after),
        "constraint_violation": constraint_violation,
    }

def render_feedback_sample(
    key,
    *,
    shot_mode: str = "S_setting",
    shot_budget: int = 256,
    noise_family: str = "smoke_mixed",
    eta: float = 0.08,
    kappa: float = 0.5,
    shots: int | None = None,
    composite_noise: bool = False,
    severity_matching_kind: str = legacy_severity_matching_kind,
) -> Dict[str, Dict[str, np.ndarray]]:
    validate_severity_matching_kind(severity_matching_kind)
    corrected = severity_matching_kind == evaluated_severity_matching_kind
    validate_trackb_policy_namespace(
        key, history_policy=TRACKB_TEACHER_POLICY_V2 if corrected else TRACKB_TEACHER_POLICY_V1,
        history_policy_sha256=TRACKB_TEACHER_POLICY_V2_SHA256 if corrected else None)
    if shots is not None:
        shot_mode, shot_budget = "S_setting", int(shots)
    N = key.N
    target_rng = key.rng("target")
    action_rng = key.rng("action")
    measurement_rng = key.rng("measurement")
    target = target_rng.normal(size=N) + 1j * target_rng.normal(size=N)
    target /= np.linalg.norm(target)
    injected_amplitude = action_rng.normal(scale=0.14, size=N - 1)
    injected_phase = action_rng.normal(scale=0.30, size=N - 1)
    current = apply_tangent_action(target, injected_amplitude, injected_phase)
    current_rho = w_density_matrix(current)
    settings = registered_feedback_settings(N)
    shots_by_setting = _feedback_shots(len(settings), shot_mode, shot_budget)
    ideal_probability = shot_mode == "S_ideal"
    noise = render_noise_truth(
        key.rng("noise"), rho_clean=current_rho, settings=settings,
        noise_family=noise_family, eta_target=eta, correctable_fraction=kappa,
        composite=composite_noise, severity_matching_kind=severity_matching_kind)
    panel = measurement_rng.random((len(settings), int(shots_by_setting.max())))
    identity_intervention = V24NoiseIntervention(
        "identity", "none", "none", 0.0, "none")
    clean_measured = phase_sensitive_density_measurement(
        current_rho, shots_by_setting, panel, identity_intervention, 0.0,
        ideal_probability=ideal_probability)
    measured = phase_sensitive_density_measurement(
        noise.rho_noisy, shots_by_setting, panel,
        noise.intervention, noise.physical_parameter,
        ideal_probability=ideal_probability)
    oracle = nonlinear_corrective_oracle(current, target)
    helmert = helmert_matrix(N)
    oracle_action = np.stack((helmert.T @ oracle["delta_amplitude"],
                              helmert.T @ oracle["delta_phase"]), axis=-1)
    analytic_action_raw, _ = _deployable_analytic_action(
        measured["estimated_coefficients"], target, measured["covariance"])
    if np.allclose(analytic_action_raw, oracle_action, atol=1e-7):
        analytic_action_raw *= 0.95
    denoised_population = np.maximum(measured["tokens"][:, 0], 1e-8)
    denoised_population /= denoised_population.sum()
    denoised_coefficients = np.sqrt(denoised_population).astype(np.complex128)
    denoised_coefficients *= np.exp(1j * np.pi * measured["tokens"][:, 1])
    analytic_action_denoised, _ = _deployable_analytic_action(
        denoised_coefficients, target, measured["covariance"])
    analytic_baseline_state = apply_tangent_action(
        current, helmert @ analytic_action_denoised[:, 0],
        helmert @ analytic_action_denoised[:, 1])
    analytic_baseline_rho_clean = w_density_matrix(analytic_baseline_state)
    analytic_baseline_rho_noisy = noise.intervention.apply_state_channel(
        analytic_baseline_rho_clean, noise.physical_parameter)
    target_rho = w_density_matrix(target)
    noop_fidelity = state_fidelity(target_rho, noise.rho_noisy)
    identity_target_law = identity_intervention.measurement_law(
        target_rho, settings, 0.0)
    noop_law = noise.intervention.measurement_law(
        current_rho, settings, noise.physical_parameter)
    analytic_baseline_law = noise.intervention.measurement_law(
        analytic_baseline_rho_clean, settings, noise.physical_parameter)
    analytic_baseline_gain = (
        hellinger_squared(noop_law, identity_target_law)
        - hellinger_squared(analytic_baseline_law, identity_target_law))

    def operational_forward(flat_action: np.ndarray) -> np.ndarray:
        action = np.asarray(flat_action, dtype=np.float64).reshape(N - 1, 2)
        state = apply_tangent_action(
            current, action[:, 0], action[:, 1])
        return noise.intervention.measurement_law(
            w_density_matrix(state), settings, noise.physical_parameter)

    oracle_result = _full_forward_oracle_action(
        current, target, noise.intervention, settings,
        noise.physical_parameter)
    full_forward_oracle = np.asarray(oracle_result["action"], dtype=np.float64)
    oracle_state = apply_tangent_action(
        current, helmert @ full_forward_oracle[:, 0],
        helmert @ full_forward_oracle[:, 1])
    oracle_rho_clean = w_density_matrix(oracle_state)
    oracle_rho_noisy = noise.intervention.apply_state_channel(
        oracle_rho_clean, noise.physical_parameter)
    oracle_law = noise.intervention.measurement_law(
        oracle_rho_clean, settings, noise.physical_parameter)
    gain = (hellinger_squared(noop_law, identity_target_law)
            - hellinger_squared(oracle_law, identity_target_law))
    state_fidelity_gain = (
        state_fidelity(target_rho, oracle_rho_noisy) - noop_fidelity)
    operational_zero = np.zeros(2 * (N - 1), dtype=np.float64)
    operational_residual = identity_target_law - operational_forward(operational_zero)
    operational_jacobian = same_forward_jacobian(
        operational_forward, operational_zero, epsilon=1e-4)
    average_shots = max(float(shots_by_setting.mean()), 1.0)
    operational_covariance = np.maximum(
        noop_law * (1.0 - noop_law) / average_shots, 1e-7)
    lower = np.tile((-CONTROL_AMPLITUDE_BOUND, -CONTROL_PHASE_BOUND), N - 1)
    upper = np.tile((CONTROL_AMPLITUDE_BOUND, CONTROL_PHASE_BOUND), N - 1)
    operational = recoverability(
        operational_residual, operational_jacobian, operational_covariance,
        lambda_u=CONTROL_RIDGE_WEIGHT, bounds=(lower, upper),
        solver_backend="box_coordinate")
    kappa_ctrl_reg = float(operational["kappa_ctrl"])
    kappa_ctrl_geom = float(operational["geometry_kappa_ctrl"])
    epsilon_irr = float(operational["epsilon_irr"])
    recoverability_mask = bool(
        operational["solver_success"] and np.isfinite(kappa_ctrl_reg)
        and np.isfinite(epsilon_irr))
    recoverable_class = int(kappa_ctrl_reg >= RECOVERABLE_CLASS_THRESHOLD)
    next_panel = key.rng("next_measurement").random(
        (len(settings), int(shots_by_setting.max())))
    oracle_measured = phase_sensitive_density_measurement(
        oracle_rho_noisy, shots_by_setting, next_panel,
        noise.intervention, noise.physical_parameter,
        ideal_probability=ideal_probability)
    analytic_baseline_measured = phase_sensitive_density_measurement(
        analytic_baseline_rho_noisy, shots_by_setting, next_panel,
        noise.intervention, noise.physical_parameter,
        ideal_probability=ideal_probability)
    noop_measured = phase_sensitive_density_measurement(
        noise.rho_noisy, shots_by_setting, next_panel,
        noise.intervention, noise.physical_parameter,
        ideal_probability=ideal_probability)
    harm = float(gain < 0.0)
    stop = float(harm or kappa_ctrl_reg < LOW_RECOVERABILITY_STOP_THRESHOLD)
    target_phase = np.angle(target * target[0].conj()) / np.pi
    target_tokens = np.stack((np.abs(target), target_phase), axis=-1).astype(np.float32)
    physical_tokens = np.stack((np.arange(N) / max(N - 1, 1), np.ones(N)), axis=-1).astype(np.float32)
    observable_error = float(np.mean(np.abs(measured["tokens"][:, 0]
                                             - np.abs(target) ** 2)))
    measured_probability = np.clip(measured["tokens"][:, 0], 1e-8, 1.0)
    target_probability = np.clip(np.abs(target) ** 2, 1e-8, 1.0)
    gate_features = np.asarray((
        observable_error, float(measured["covariance"].mean()), N / 256.0,
        float(shots_by_setting.sum()) / 1024.0,
        float(-(measured_probability * np.log(measured_probability)).sum()),
        float(-(target_probability * np.log(target_probability)).sum()),
        float(measured_probability.max()), float(measured["detection_fraction"].mean()),
    ), dtype=np.float32)
    denoised = measured["tokens"].copy()
    denoised[:, 0] = denoised_population
    esm = np.stack((
        np.clip(2.0 * np.abs(measured["estimated_coefficients"])
                * np.abs(measured["estimated_coefficients"][0]), 0.0, 1.0),
        1.0 - measured["covariance"][:, 0],
    ), axis=-1).astype(np.float32)
    true_esm = np.stack((
        np.clip(2.0 * np.abs(current) * np.abs(current[0]), 0.0, 1.0),
        np.ones(N, dtype=np.float64),
    ), axis=-1).astype(np.float32)
    return {
        "observation": {
            "noisy_measurement_tokens": measured["tokens"],
            "measurement_covariance": measured["covariance"],
            "csm_cartesian_stats": np.asarray(
                measured["cartesian_stats"], dtype=np.float32),
            "csm_cartesian_covariance": np.asarray(
                measured["cartesian_covariance"], dtype=np.float32),
            "csm_statistic_valid_mask": measured["cartesian_statistic_valid_mask"],
            "csm_detection_fraction": measured["cartesian_detection_fraction"].astype(np.float32),
            "csm_covariance_estimator_kind": np.asarray(
                str(measured["cartesian_covariance_estimator_kind"]).encode("ascii"), dtype="S64"),
            "csm_dirichlet_alpha": np.asarray(
                measured["cartesian_dirichlet_alpha"], dtype=np.float64),
            "csm_observation_contract_sha": np.asarray(
                str(measured["cartesian_observation_contract_sha"]).encode("ascii"), dtype="S64"),
            "csm_observation_contract_kind": np.asarray(
                str(measured["cartesian_observation_contract_kind"]).encode("ascii"), dtype="S64"),
            "csm_statistic_transform_sha": np.asarray(
                str(measured["cartesian_statistic_transform_sha"]).encode("ascii"), dtype="S64"),
            "csm_whitening_support_kind": np.asarray(
                str(measured["cartesian_whitening_support_kind"]).encode("ascii"), dtype="S64"),
            "csm_covariance_instance_sha": np.asarray(
                str(measured["cartesian_covariance_instance_sha"]).encode("ascii"), dtype="S64"),
            "csm_geometry_source": np.asarray(
                str(measured["cartesian_geometry_source"]).encode("ascii"), dtype="S64"),
            "target_mode_tokens": target_tokens,
            "physical_mode_tokens": physical_tokens,
            "gate_features": gate_features,
            "mode_mask": np.ones(N, dtype=np.bool_),
            "history_tokens": np.zeros(
                (FEEDBACK_HISTORY_ROUNDS, N, 7), dtype=np.float32),
            "history_mask": np.zeros(FEEDBACK_HISTORY_ROUNDS, dtype=np.bool_),
            "measurement_uniforms_sha": np.asarray(sha256_array(panel).encode("ascii"), dtype="S64"),
            "next_measurement_uniforms_sha": np.asarray(
                sha256_array(next_panel).encode("ascii"), dtype="S64"),
        },
        "truth": {
            "oracle_action": full_forward_oracle.astype(np.float32),
            "target_action": full_forward_oracle.astype(np.float32),
            "oracle_action_mask": np.asarray(
                oracle_result["solver_success"], dtype=np.bool_),
            "oracle_solver_success": np.asarray(
                oracle_result["solver_success"], dtype=np.bool_),
            "oracle_solver_status": np.asarray(
                str(oracle_result["solver_status"]).encode("utf-8"), dtype="S128"),
            "oracle_objective_before": np.asarray(
                oracle_result["objective_before"], dtype=np.float64),
            "oracle_objective_after": np.asarray(
                oracle_result["objective_after"], dtype=np.float64),
            "oracle_constraint_violation": np.asarray(
                oracle_result["constraint_violation"], dtype=np.float64),
            "target_recoverability": np.asarray(kappa_ctrl_reg, dtype=np.float32),
            "target_recoverable_class": np.asarray(
                recoverable_class, dtype=np.int64),
            "kappa_ctrl_geom": np.asarray(kappa_ctrl_geom, dtype=np.float64),
            "kappa_ctrl_reg": np.asarray(kappa_ctrl_reg, dtype=np.float64),
            "recoverable_class": np.asarray(recoverable_class, dtype=np.int64),
            "recoverability_mask": np.asarray(recoverability_mask, dtype=np.bool_),
            "solver_status": np.asarray(
                str(operational["solver_status"]).encode("utf-8"), dtype="S128"),
            "target_gain": np.asarray(gain, dtype=np.float32),
            "target_state_fidelity_gain": np.asarray(
                state_fidelity_gain, dtype=np.float32),
            "target_harm": np.asarray(harm, dtype=np.float32),
            "target_stop": np.asarray(stop, dtype=np.float32),
            "target_next_measurement": oracle_measured["tokens"],
            "analytic_baseline_gain": np.asarray(
                analytic_baseline_gain, dtype=np.float32),
            "analytic_baseline_next_measurement": analytic_baseline_measured["tokens"],
            "target_noop_next_measurement": noop_measured["tokens"],
            "target_noop_gain": np.asarray(0.0, dtype=np.float32),
            "target_history_gain": np.asarray(0.0, dtype=np.float32),
            "target_history_gain_mask": np.asarray(False, dtype=np.bool_),
            "history_round_count": np.asarray(0, dtype=np.int16),
            "current_coefficients_re": current.real.astype(np.float64),
            "current_coefficients_im": current.imag.astype(np.float64),
            "correctable_mechanism": noise.correctable_multihot,
            "irrecoverable_mechanism": noise.irrecoverable_multihot,
            "mechanism_severity": noise.severity,
            "epsilon_irr": np.asarray(epsilon_irr, dtype=np.float64),
            "eta_target": np.asarray(noise.eta_target, dtype=np.float64),
            "eta_achieved": np.asarray(noise.eta_achieved, dtype=np.float64),
            "eta_matching_residual": np.asarray(
                noise.matching_residual, dtype=np.float64),
            "eta_matching_converged": np.asarray(
                noise.matching_converged, dtype=np.bool_),
            "physical_noise_parameter": np.asarray(
                noise.physical_parameter, dtype=np.float64),
            "state_infidelity": np.asarray(noise.state_infidelity, dtype=np.float64),
            "mechanism_name": np.asarray(
                noise.mechanism_name.encode("ascii"), dtype="S64"),
            "d3_train_fit_sha": np.asarray(
                noise.train_fit_sha.encode("ascii"), dtype="S64"),
            "d3_heldout_error": np.asarray(noise.heldout_error, dtype=np.float64),
            "intervention_forward_sha": np.asarray(
                noise.intervention_forward_sha.encode("ascii"), dtype="S64"),
            "clean_measurement_tokens": clean_measured["tokens"],
            "true_next_measurement": oracle_measured["tokens"],
            "true_esm_mode_tokens": true_esm,
            "true_partition": np.ones((N, N), dtype=np.float32),
            "eta_ctrl_target": np.asarray([noise.eta_ctrl_target], dtype=np.float64),
            "eta_irr_target": np.asarray([noise.eta_irr_target], dtype=np.float64),
            "eta_ctrl_achieved": np.asarray([noise.eta_ctrl_achieved], dtype=np.float64),
            "eta_irr_achieved": np.asarray([noise.eta_irr_achieved], dtype=np.float64),
            "kappa_inj": np.asarray([noise.kappa_inj], dtype=np.float64),
            "kappa_achieved": np.asarray([noise.kappa_achieved], dtype=np.float64),
            "kappa_forward_achieved": np.asarray([noise.kappa_forward_achieved], dtype=np.float64),
            "kappa_target": np.asarray([noise.kappa_target], dtype=np.float64),
            "kappa_component_achieved": np.asarray(
                [noise.kappa_component_achieved], dtype=np.float64),
            "kappa_component_mask": np.asarray(
                noise.kappa_component_mask, dtype=np.bool_),
            "kappa_operational": np.asarray(
                [kappa_ctrl_reg], dtype=np.float64),
            "kappa_operational_mask": np.asarray(
                recoverability_mask, dtype=np.bool_),
            "control_parameter": np.asarray([noise.control_parameter], dtype=np.float64),
            "irrecoverable_parameter": np.asarray([noise.irrecoverable_parameter], dtype=np.float64),
            "control_mechanism_name": np.asarray([noise.control_mechanism_name.encode("ascii")], dtype="S64"),
            "irrecoverable_mechanism_name": np.asarray([noise.irrecoverable_mechanism_name.encode("ascii")], dtype="S64"),
            "true_mechanism": np.concatenate((
                noise.correctable_multihot, noise.irrecoverable_multihot)).astype(np.float32),
            "true_severity": np.asarray(noise.physical_parameter, dtype=np.float64),
            "true_kappa": np.asarray(kappa_ctrl_reg, dtype=np.float64),
            "crn_consumed": np.asarray(True, dtype=np.bool_),
        },
        "derived_classifier": {
            "denoised_measurement_tokens": denoised,
            "covariance": measured["covariance"],
            "esm_mode_tokens": esm,
            "classifier_config_sha": np.asarray(sha256_json({"kind": "fixture_denoiser"}).encode(), dtype="S64"),
            "normalization_sha": np.asarray(sha256_json({"kind": "identity"}).encode(), dtype="S64"),
        },
        "derived_control": {
            "analytic_action": analytic_action_denoised.astype(np.float32),
            "analytic_action_raw": analytic_action_raw.astype(np.float32),
            "analytic_action_denoised": analytic_action_denoised.astype(np.float32),
            "analytic_action_mask": np.ones((N, 2), dtype=np.bool_),
            "analytic_action_raw_mask": np.ones((N, 2), dtype=np.bool_),
            "analytic_action_denoised_mask": np.ones((N, 2), dtype=np.bool_),
            "forward_sha": np.asarray(sha256_json({"forward": "global_pauli_w_v24"}).encode(), dtype="S64"),
        },
        "identities": {
            "latent_state_id": sha256_json({"target": target.real.tolist(), "target_im": target.imag.tolist()}),
            "transition_id": sha256_json({"sample": key.sample_id, "kind": "one_step"}),
            "crn_group_id": sha256_array(panel),
        },
        "internal": {
            "rho_noisy": noise.rho_noisy,
            "intervention": noise.intervention,
            "physical_parameter": noise.physical_parameter,
        },
    }


def _trackb_receding_horizon_action(
    before: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one bounded target-oracle action and its resulting clean state."""
    N = len(before)
    helmert = helmert_matrix(N)
    residual_amplitude = (
        aitchison_amplitude_coordinates(np.abs(target))
        - aitchison_amplitude_coordinates(np.abs(before))
    )
    residual_phase = np.angle(np.exp(1j * (
        phase_coordinates(np.angle(target))
        - phase_coordinates(np.angle(before))
    )))
    step_amplitude_coordinates = TRACKB_RECEDING_HORIZON_DAMPING * np.clip(
        residual_amplitude, -CONTROL_AMPLITUDE_BOUND, CONTROL_AMPLITUDE_BOUND)
    step_phase_coordinates = TRACKB_RECEDING_HORIZON_DAMPING * np.clip(
        residual_phase, -CONTROL_PHASE_BOUND, CONTROL_PHASE_BOUND)
    after = apply_tangent_action(
        before, step_amplitude_coordinates, step_phase_coordinates)
    action = np.stack((
        helmert.T @ step_amplitude_coordinates,
        helmert.T @ step_phase_coordinates,
    ), axis=-1)
    return action, after


@dataclass(frozen=True)
class TrackBRecedingHorizonForwardContextV1:
    """One synthetic pre-state used for independent logical-action scoring."""

    sample_id: str
    round_index: int
    before_coefficients: np.ndarray
    target_coefficients: np.ndarray
    intervention: Any
    physical_parameter: float
    settings: tuple[tuple[str, ...], ...]
    teacher_action: np.ndarray
    teacher_gain: float
    pre_state_sha: str

    def validate(self) -> None:
        N = int(np.asarray(self.before_coefficients).size)
        if type(self.sample_id) is not str or not self.sample_id or not 0 <= self.round_index < TRACKB_TEN_ROUND_HISTORY:
            raise ValueError("Track B candidate forward context identity is invalid")
        for value, name in ((self.before_coefficients, "before_coefficients"),
                            (self.target_coefficients, "target_coefficients")):
            array = np.asarray(value, dtype=np.complex128).reshape(-1)
            if array.size != N or not np.all(np.isfinite(array)) or np.linalg.norm(array) <= 0.0:
                raise ValueError(f"{name} is invalid")
        _validate_trackb_logical_action(self.teacher_action, N)
        if not math.isfinite(float(self.teacher_gain)) or len(self.pre_state_sha) != 64:
            raise ValueError("Track B candidate forward context gain or SHA is invalid")


def _validate_trackb_logical_action(action: np.ndarray, N: int) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (N, 2) or not np.all(np.isfinite(value)):
        raise ValueError("Track B candidate logical action must have finite [N,2] shape")
    if not np.allclose(value.sum(axis=0), 0.0, atol=1e-6, rtol=0.0):
        raise ValueError("Track B candidate logical action must remain global-gauge centered")
    return value


def evaluate_trackb_candidate_transition(
    context: TrackBRecedingHorizonForwardContextV1,
    candidate_action: np.ndarray,
) -> Dict[str, object]:
    """Score one candidate against the exact synthetic pre-state forward only."""
    context.validate()
    before = np.asarray(context.before_coefficients, dtype=np.complex128).reshape(-1)
    target = np.asarray(context.target_coefficients, dtype=np.complex128).reshape(-1)
    N = before.size
    action = _validate_trackb_logical_action(candidate_action, N)
    helmert = helmert_matrix(N)
    action_coordinates = np.stack((
        helmert @ action[:, 0],
        helmert @ action[:, 1],
    ), axis=-1)
    after = apply_tangent_action(before, action_coordinates[:, 0], action_coordinates[:, 1])
    identity = V24NoiseIntervention("identity", "none", "none", 0.0, "none")
    target_law = identity.measurement_law(w_density_matrix(target), context.settings, 0.0)
    before_law = context.intervention.measurement_law(
        w_density_matrix(before), context.settings, context.physical_parameter)
    after_law = context.intervention.measurement_law(
        w_density_matrix(after), context.settings, context.physical_parameter)
    before_distance = hellinger_squared(before_law, target_law)
    after_distance = hellinger_squared(after_law, target_law)
    return {
        "candidate_action": action,
        "candidate_action_sha": sha256_array(action),
        "after_coefficients": after,
        "after_state_sha": sha256_array(after),
        "before_distance": float(before_distance),
        "after_distance": float(after_distance),
        "gain": float(before_distance - after_distance),
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def replay_trackb_ten_round_forward_contexts(
    key,
    *,
    shot_mode: str,
    shot_budget: int,
    noise_family: str,
    eta: float,
    kappa: float,
    history_policy: str = TRACKB_TEACHER_POLICY_V1,
    history_policy_sha256: str | None = None,
    severity_matching_kind: str = legacy_severity_matching_kind,
) -> tuple[TrackBRecedingHorizonForwardContextV1, ...]:
    """Replay the registered ten-round source without exposing latent state to a model."""
    validate_severity_matching_kind(severity_matching_kind)
    if (history_policy == TRACKB_TEACHER_POLICY_V2) != (
            severity_matching_kind == evaluated_severity_matching_kind):
        raise ValueError("Track B replay policy and severity_matching_kind differ")
    validate_trackb_policy_namespace(
        key, history_policy=history_policy,
        history_policy_sha256=history_policy_sha256)
    if history_policy == TRACKB_TEACHER_POLICY_V2:
        if history_policy_sha256 != TRACKB_TEACHER_POLICY_V2_SHA256:
            raise ValueError("Track B V2 forward replay requires its registered policy SHA")
        N = int(getattr(key, "N", -1))
        expected_cell_id = TRACKB_TEACHER_POLICY_V2_CELL_IDS.get(N)
        if expected_cell_id is None:
            raise ValueError("Track B V2 forward replay supports only N=3 or N=4")
        validate_trackb_v2_sample_key(
            key, N=N, dataset_family="feedback_history", cell_id=expected_cell_id)
    elif history_policy != TRACKB_TEACHER_POLICY_V1 or history_policy_sha256 is not None:
        raise ValueError("Track B forward replay policy identity is invalid")
    rendered = render_feedback_sample(
        key,
        shot_mode=shot_mode,
        shot_budget=shot_budget,
        noise_family=noise_family,
        eta=eta,
        kappa=kappa,
        composite_noise=True,
        severity_matching_kind=severity_matching_kind,
    )
    truth = rendered["truth"]
    observation = rendered["observation"]
    internal = rendered["internal"]
    rolling_state = (
        np.asarray(truth["current_coefficients_re"], dtype=np.float64)
        + 1j * np.asarray(truth["current_coefficients_im"], dtype=np.float64)
    )
    target_tokens = np.asarray(observation["target_mode_tokens"], dtype=np.float64)
    target = target_tokens[:, 0] * np.exp(1j * np.pi * target_tokens[:, 1])
    target /= np.linalg.norm(target)
    settings = registered_feedback_settings(len(target))
    contexts = []
    for round_index in range(TRACKB_TEN_ROUND_HISTORY):
        if history_policy == TRACKB_TEACHER_POLICY_V1:
            teacher_action, after = _trackb_receding_horizon_action(rolling_state, target)
        else:
            teacher_action, after = trackb_receding_horizon_action_v2(rolling_state, target)
        provisional = TrackBRecedingHorizonForwardContextV1(
            sample_id=key.sample_id,
            round_index=round_index,
            before_coefficients=rolling_state,
            target_coefficients=target,
            intervention=internal["intervention"],
            physical_parameter=float(internal["physical_parameter"]),
            settings=settings,
            teacher_action=teacher_action,
            teacher_gain=0.0,
            pre_state_sha=sha256_array(rolling_state),
        )
        teacher_score = evaluate_trackb_candidate_transition(provisional, teacher_action)
        contexts.append(TrackBRecedingHorizonForwardContextV1(
            sample_id=key.sample_id,
            round_index=round_index,
            before_coefficients=rolling_state,
            target_coefficients=target,
            intervention=internal["intervention"],
            physical_parameter=float(internal["physical_parameter"]),
            settings=settings,
            teacher_action=teacher_action,
            teacher_gain=float(teacher_score["gain"]),
            pre_state_sha=sha256_array(rolling_state),
        ))
        rolling_state = after
    return tuple(contexts)


def _render_feedback_history(
    key,
    rendered: Dict[str, Dict[str, np.ndarray]],
    *,
    shot_mode: str,
    shot_budget: int,
    rounds: int,
    policy: str,
    policy_sha256: str | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Attach a physical `(y_t,u_t,y_{t+1},G_t)` rollout history."""
    validate_trackb_policy_namespace(
        key, history_policy=policy, history_policy_sha256=policy_sha256)
    if rounds not in (FEEDBACK_HISTORY_ROUNDS, TRACKB_TEN_ROUND_HISTORY):
        raise ValueError("V2.4 feedback history supports only registered five- or ten-round trajectories")
    if policy not in (FIXED_FRACTION_HISTORY_POLICY, TRACKB_RECEDING_HORIZON_POLICY,
                      TRACKB_TEACHER_POLICY_V2):
        raise ValueError("V2.4 feedback history policy is unregistered")
    if policy in (TRACKB_RECEDING_HORIZON_POLICY, TRACKB_TEACHER_POLICY_V2) and rounds != TRACKB_TEN_ROUND_HISTORY:
        raise ValueError("the Track B receding-horizon policy requires ten rounds")
    truth = rendered["truth"]
    observation = rendered["observation"]
    internal = rendered["internal"]
    current = (np.asarray(truth["current_coefficients_re"], dtype=np.float64)
               + 1j * np.asarray(truth["current_coefficients_im"], dtype=np.float64))
    target_tokens = np.asarray(observation["target_mode_tokens"], dtype=np.float64)
    target = target_tokens[:, 0] * np.exp(1j * np.pi * target_tokens[:, 1])
    target /= np.linalg.norm(target)
    N = len(current)
    settings = registered_feedback_settings(N)
    shots_by_setting = _feedback_shots(len(settings), shot_mode, shot_budget)
    ideal_probability = shot_mode == "S_ideal"
    intervention = internal["intervention"]
    physical_parameter = float(internal["physical_parameter"])
    helmert = helmert_matrix(N)
    action = np.asarray(truth["target_action"], dtype=np.float64)
    amplitude_coordinates = helmert @ action[:, 0]
    phase_action_coordinates = helmert @ action[:, 1]
    identity = V24NoiseIntervention("identity", "none", "none", 0.0, "none")
    target_law = identity.measurement_law(w_density_matrix(target), settings, 0.0)
    history = []
    gains = []
    rolling_state = current
    for round_index in range(rounds):
        if policy == FIXED_FRACTION_HISTORY_POLICY:
            before_fraction = round_index / rounds
            after_fraction = (round_index + 1) / rounds
            before = apply_tangent_action(
                current, before_fraction * amplitude_coordinates,
                before_fraction * phase_action_coordinates)
            after = apply_tangent_action(
                current, after_fraction * amplitude_coordinates,
                after_fraction * phase_action_coordinates)
            step_action = action / rounds
        else:
            before = rolling_state
            if policy == TRACKB_TEACHER_POLICY_V2:
                step_action, after = trackb_receding_horizon_action_v2(before, target)
            else:
                step_action, after = _trackb_receding_horizon_action(before, target)
        before_clean = w_density_matrix(before)
        after_clean = w_density_matrix(after)
        before_noisy = intervention.apply_state_channel(
            before_clean, physical_parameter)
        after_noisy = intervention.apply_state_channel(
            after_clean, physical_parameter)
        panel = key.rng("history_measurement", round_index).random(
            (len(settings), int(shots_by_setting.max())))
        before_measured = phase_sensitive_density_measurement(
            before_noisy, shots_by_setting, panel,
            intervention, physical_parameter,
            ideal_probability=ideal_probability)
        after_measured = phase_sensitive_density_measurement(
            after_noisy, shots_by_setting, panel,
            intervention, physical_parameter,
            ideal_probability=ideal_probability)
        before_law = intervention.measurement_law(
            before_clean, settings, physical_parameter)
        after_law = intervention.measurement_law(
            after_clean, settings, physical_parameter)
        gain = (hellinger_squared(before_law, target_law)
                - hellinger_squared(after_law, target_law))
        history.append(np.concatenate((
            before_measured["tokens"], step_action,
            after_measured["tokens"],
            np.full((N, 1), gain, dtype=np.float64),
        ), axis=-1))
        gains.append(gain)
        rolling_state = after
    observation["history_tokens"] = np.asarray(history, dtype=np.float32)
    observation["history_mask"] = np.ones(rounds, dtype=np.bool_)
    truth["history_round_count"] = np.asarray(rounds, dtype=np.int16)
    truth["target_history_gain"] = np.asarray(sum(gains), dtype=np.float32)
    truth["target_history_gain_mask"] = np.asarray(True, dtype=np.bool_)
    return rendered


def render_feedback_history(
    key,
    rendered: Dict[str, Dict[str, np.ndarray]],
    *,
    shot_mode: str,
    shot_budget: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Attach the canonical five-round V2.4 feedback-history training target."""
    return _render_feedback_history(
        key, rendered, shot_mode=shot_mode, shot_budget=shot_budget,
        rounds=FEEDBACK_HISTORY_ROUNDS,
        policy=FIXED_FRACTION_HISTORY_POLICY)


def render_trackb_ten_round_history(
    key,
    rendered: Dict[str, Dict[str, np.ndarray]],
    *,
    shot_mode: str,
    shot_budget: int,
    history_policy: str = TRACKB_TEACHER_POLICY_V1,
    history_policy_sha256: str | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Attach the Track B-specific ten-round V2.4 training trajectory.

    The layout remains the registered V2.4 history token `[rounds, N, 7]`;
    only the explicitly selected Track B data source uses ten rounds.
    """
    if history_policy == TRACKB_TEACHER_POLICY_V2:
        if history_policy_sha256 != TRACKB_TEACHER_POLICY_V2_SHA256:
            raise ValueError("Track B V2 renderer requires its registered policy SHA")
        N = int(getattr(key, "N", -1))
        expected_cell_id = TRACKB_TEACHER_POLICY_V2_CELL_IDS.get(N)
        if expected_cell_id is None:
            raise ValueError("Track B V2 renderer supports only N=3 or N=4")
        validate_trackb_v2_sample_key(
            key, N=N, dataset_family="feedback_history", cell_id=expected_cell_id)
    elif history_policy != TRACKB_TEACHER_POLICY_V1 or history_policy_sha256 is not None:
        raise ValueError("Track B renderer policy identity is invalid")
    return _render_feedback_history(
        key, rendered, shot_mode=shot_mode, shot_budget=shot_budget,
        rounds=TRACKB_TEN_ROUND_HISTORY, policy=history_policy,
        policy_sha256=history_policy_sha256)
