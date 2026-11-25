"""Causal, multi-family synthetic transitions for V8 feedback training.

The measured corpus identifies static count and density structure but contains
no authoritative action-response labels.  This module therefore keeps every
action label explicitly synthetic, binds history and the queried transition to
the same latent plant, and separates train, development, and locked-OOD plant
families.  It never converts synthetic responses into measured supervision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ml_recon.physics_hybrid_v5.projective_action import (
    coordinates_to_amplitude,
    coordinates_to_phase,
)
from ml_recon.physics_hybrid_v7.feedback_data import (
    OPERATION_FEATURE_NAMES,
    FeedbackMeasuredArraysV1,
    FeedbackNuisanceProfileV1,
    build_operation_features,
)
from ml_recon.physics_hybrid_v7.feedback_synthetic import (
    FeedbackSyntheticConfig,
    _render_counts,
    _sample_training_action,
    _state_log_gsi,
    canonical_action_mask,
    evaluate_counterfactual_action,
    generate_feedback_policy_batch,
)
from ml_recon.physics_hybrid_v7.geometry import build_operation_frequency_geometry


V8_SYNTHETIC_SCHEMA = "v8_feedback_transition_batch_v1"
REFERENCE_EXPOSURE_S = 1200.0
EXPOSURE_LEVELS_S = (150.0, 300.0, 600.0, 1200.0)
SYNTHETIC_FAMILY_ROLES = ("train", "development", "locked_ood")


@dataclass(frozen=True)
class V8SyntheticFamilyConfig:
    """Immutable definition of one falsifiable simulator family."""

    family_id: str
    role: str
    phase_response_lower: float
    phase_response_upper: float
    amplitude_response_lower: float
    amplitude_response_upper: float
    phase_drift_std_rad: float
    crosstalk_scale_max: float
    dephasing_lower: float
    dephasing_upper: float
    multi_probability_lower: float
    multi_probability_upper: float
    uniform_exploration_floor: float = 0.20
    locked: bool = False

    def validate(self) -> None:
        if self.role not in SYNTHETIC_FAMILY_ROLES:
            raise ValueError("synthetic family role is invalid")
        if self.role == "locked_ood" and not self.locked:
            raise ValueError("locked_ood family must be immutable")
        if self.role != "locked_ood" and self.locked:
            raise ValueError("only locked_ood families may be locked")
        if not 0.20 <= self.uniform_exploration_floor <= 1.0:
            raise ValueError("uniform exploration floor must be at least 20 percent")
        if not 0.0 <= self.dephasing_lower < self.dephasing_upper < 1.0:
            raise ValueError("dephasing range is invalid")
        if not 0.0 <= self.multi_probability_lower < self.multi_probability_upper < 1.0:
            raise ValueError("multi-excitation range is invalid")
        if min(
            self.phase_response_lower,
            self.amplitude_response_lower,
            self.phase_drift_std_rad,
            self.crosstalk_scale_max,
        ) < 0.0:
            raise ValueError("response and drift parameters must be non-negative")

    @property
    def family_hash(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def simulator_config(self) -> FeedbackSyntheticConfig:
        self.validate()
        return replace(
            FeedbackSyntheticConfig(),
            phase_response_lower=float(self.phase_response_lower),
            phase_response_upper=float(self.phase_response_upper),
            amplitude_response_lower=float(self.amplitude_response_lower),
            amplitude_response_upper=float(self.amplitude_response_upper),
            phase_drift_std_rad=float(self.phase_drift_std_rad),
            crosstalk_scale_max=float(self.crosstalk_scale_max),
            dephasing_lower=float(self.dephasing_lower),
            dephasing_upper=float(self.dephasing_upper),
            multi_probability_lower=float(self.multi_probability_lower),
            multi_probability_upper=float(self.multi_probability_upper),
        )


V8_SYNTHETIC_FAMILIES: Mapping[str, V8SyntheticFamilyConfig] = {
    "measured_correlated_train": V8SyntheticFamilyConfig(
        family_id="measured_correlated_train",
        role="train",
        phase_response_lower=0.72,
        phase_response_upper=1.12,
        amplitude_response_lower=0.65,
        amplitude_response_upper=1.05,
        phase_drift_std_rad=0.025,
        crosstalk_scale_max=0.18,
        dephasing_lower=0.01,
        dephasing_upper=0.30,
        multi_probability_lower=0.01,
        multi_probability_upper=0.25,
    ),
    "mechanism_shift_development": V8SyntheticFamilyConfig(
        family_id="mechanism_shift_development",
        role="development",
        phase_response_lower=0.62,
        phase_response_upper=1.18,
        amplitude_response_lower=0.55,
        amplitude_response_upper=1.10,
        phase_drift_std_rad=0.045,
        crosstalk_scale_max=0.24,
        dephasing_lower=0.04,
        dephasing_upper=0.38,
        multi_probability_lower=0.025,
        multi_probability_upper=0.30,
    ),
    "detector_drift_locked_ood": V8SyntheticFamilyConfig(
        family_id="detector_drift_locked_ood",
        role="locked_ood",
        phase_response_lower=0.50,
        phase_response_upper=1.25,
        amplitude_response_lower=0.45,
        amplitude_response_upper=1.15,
        phase_drift_std_rad=0.070,
        crosstalk_scale_max=0.30,
        dephasing_lower=0.08,
        dephasing_upper=0.48,
        multi_probability_lower=0.05,
        multi_probability_upper=0.36,
        locked=True,
    ),
}


def resolve_synthetic_family(
    family_id: str,
    *,
    expected_role: Optional[str] = None,
    allow_locked_evaluation: bool = False,
) -> V8SyntheticFamilyConfig:
    """Resolve a family while preventing locked-OOD training leakage."""

    try:
        family = V8_SYNTHETIC_FAMILIES[str(family_id)]
    except KeyError as exc:
        raise ValueError(f"unknown V8 synthetic family: {family_id}") from exc
    family.validate()
    if expected_role is not None and family.role != str(expected_role):
        raise ValueError(
            f"synthetic family {family.family_id} has role {family.role}, "
            f"expected {expected_role}"
        )
    if family.locked and not allow_locked_evaluation:
        raise PermissionError(
            "locked simulator family is evaluation-only and requires an explicit unlock"
        )
    return family


def generate_v8_transition_batch(
    arrays: FeedbackMeasuredArraysV1,
    profile: FeedbackNuisanceProfileV1,
    *,
    batch_size: int,
    seed: int,
    family_id: str = "measured_correlated_train",
    exposure_s: Optional[float] = None,
    allow_locked_evaluation: bool = False,
) -> Dict[str, Any]:
    """Generate a causal same-plant transition with explicit exposure.

    The history is first rolled forward with accepted/rejected/rollback
    semantics.  The final state from that exact plant becomes the current
    state for the queried action.  This fixes the V7 one-step generator's
    statistically plausible but plant-unbound Model-A history.
    """

    family = resolve_synthetic_family(
        family_id,
        allow_locked_evaluation=allow_locked_evaluation,
    )
    simulator_config = family.simulator_config()
    base = generate_feedback_policy_batch(
        arrays,
        profile,
        batch_size=int(batch_size),
        seed=int(seed),
        config=simulator_config,
    )
    exposure_rng = np.random.default_rng(int(seed) + 7919)
    current_count_rng = np.random.default_rng(int(seed) + 104729)
    action_rng = np.random.default_rng(int(seed) + 130363)
    next_count_rng = np.random.default_rng(int(seed) + 169087)
    if exposure_s is None:
        exposure = exposure_rng.choice(
            np.asarray(EXPOSURE_LEVELS_S, dtype=np.float64),
            size=int(batch_size),
            replace=True,
        )
    else:
        if not np.isfinite(float(exposure_s)) or float(exposure_s) <= 0.0:
            raise ValueError("synthetic exposure_s must be finite and positive")
        exposure = np.full(int(batch_size), float(exposure_s), dtype=np.float64)
    exposure_scale = exposure / REFERENCE_EXPOSURE_S

    nuisance = {
        "signal_probability": np.asarray(
            base["nuisance_signal_probability"], dtype=np.float64
        ),
        "idler_probability": np.asarray(
            base["nuisance_idler_probability"], dtype=np.float64
        ),
        "coin_total": np.asarray(base["nuisance_coin_total"], dtype=np.float64)
        * exposure_scale,
        "signal_total": np.asarray(base["nuisance_signal_total"], dtype=np.float64)
        * exposure_scale,
        "idler_total": np.asarray(base["nuisance_idler_total"], dtype=np.float64)
        * exposure_scale,
        "read_count": np.asarray(base["nuisance_read_count"], dtype=np.float64)
        * exposure_scale,
    }
    current_amplitude = coordinates_to_amplitude(
        np.asarray(base["current_amplitude_coordinate"], dtype=np.float64)
    )
    current_phase = coordinates_to_phase(
        np.asarray(base["current_phase_coordinate"], dtype=np.float64)
    )
    current_counts = _render_counts(
        arrays,
        profile,
        current_amplitude,
        current_phase,
        np.asarray(base["retrieval_scale"], dtype=np.float64),
        np.asarray(base["multi_probability"], dtype=np.float64),
        np.asarray(base["mechanism_weights"], dtype=np.float64),
        nuisance,
        current_count_rng,
    )

    teacher_action = np.asarray(base["teacher_action"], dtype=np.float64)
    action, sampled_action_kind = _sample_training_action(
        teacher_action,
        arrays.system_type,
        action_rng,
        simulator_config,
    )
    outcome = evaluate_counterfactual_action(
        base,
        action,
        config=simulator_config,
    )
    next_amplitude = coordinates_to_amplitude(
        np.asarray(outcome["next_amplitude_coordinate"], dtype=np.float64)
    )
    next_phase = coordinates_to_phase(
        np.asarray(outcome["next_phase_coordinate"], dtype=np.float64)
    )
    next_counts = _render_counts(
        arrays,
        profile,
        next_amplitude,
        next_phase,
        np.asarray(base["retrieval_scale"], dtype=np.float64),
        np.asarray(base["multi_probability"], dtype=np.float64),
        np.asarray(base["mechanism_weights"], dtype=np.float64),
        nuisance,
        next_count_rng,
    )

    qubit_count = arrays.qubit_count
    target_amplitude = coordinates_to_amplitude(
        np.asarray(base["target_amplitude_coordinate"], dtype=np.float64)
    )
    target_phase = coordinates_to_phase(
        np.asarray(base["target_phase_coordinate"], dtype=np.float64)
    )
    padded_amplitude = np.zeros((int(batch_size), 4), dtype=np.float32)
    padded_phase = np.zeros_like(padded_amplitude)
    padded_amplitude[:, :qubit_count] = target_amplitude.astype(np.float32)
    padded_phase[:, :qubit_count] = target_phase.astype(np.float32)
    operation_features = build_operation_features(
        coin_counts=current_counts["coin_counts"],
        signal_counts=current_counts["signal_counts"],
        idler_counts=current_counts["idler_counts"],
        read_count=current_counts["read_count"],
        target_amplitude=padded_amplitude,
        target_phase_rad=padded_phase,
        geometry=build_operation_frequency_geometry(arrays.system_type),
        static_operation_features=arrays.static_operation_features,
        retrieval_relative_map=arrays.retrieval_relative_map,
        observed_coin_mask=np.ones_like(current_counts["coin_counts"], dtype=np.float32),
    ).astype(np.float32)
    command_features = build_synthetic_command_features(
        operation_features,
        current_amplitude,
        current_phase,
    )
    nuisance_targets = np.stack(
        (
            np.asarray(base["retrieval_scale"], dtype=np.float64),
            _state_log_gsi(next_counts),
            np.asarray(base["multi_probability"], dtype=np.float64),
            np.asarray(outcome["next_dephasing"], dtype=np.float64),
            np.linalg.norm(
                np.asarray(base["response_matrix"], dtype=np.float64)
                - np.eye(6, dtype=np.float64)[None, :, :],
                axis=(1, 2),
            ),
        ),
        axis=1,
    )
    action_mask = np.broadcast_to(
        canonical_action_mask(arrays.system_type),
        (int(batch_size), 6),
    ).copy()
    base.update(
        {
            "schema": np.asarray(V8_SYNTHETIC_SCHEMA),
            "synthetic_family_id": np.asarray(family.family_id),
            "synthetic_family_role": np.asarray(family.role),
            "synthetic_family_hash": np.asarray(family.family_hash),
            "causal_history_bound": np.ones(int(batch_size), dtype=np.float32),
            "real_action_supervision_mask": np.zeros(int(batch_size), dtype=np.float32),
            "synthetic_action_supervision_mask": np.ones(int(batch_size), dtype=np.float32),
            "operation_features": operation_features,
            "command_features": command_features,
            "command_mask": np.ones(
                (int(batch_size), arrays.operation_count), dtype=np.float32
            ),
            "command_supervision_provenance": np.ones(
                int(batch_size), dtype=np.float32
            ),
            "action": action.astype(np.float32),
            "action_mask": action_mask.astype(np.float32),
            "sampled_action_kind": sampled_action_kind.astype(np.int64),
            "current_coin_counts": current_counts["coin_counts"].astype(np.float32),
            "current_signal_counts": current_counts["signal_counts"].astype(np.float32),
            "current_idler_counts": current_counts["idler_counts"].astype(np.float32),
            "current_read_count": current_counts["read_count"].astype(np.float32),
            "next_coin_counts": next_counts["coin_counts"].astype(np.float32),
            "next_expected_counts": next_counts["expected_counts"].astype(np.float32),
            "next_count_loss_mask": np.ones_like(
                next_counts["coin_counts"], dtype=np.float32
            ),
            "nb2_dispersion": next_counts["nb2_dispersion"].astype(np.float32),
            "next_rho": np.asarray(outcome["next_rho"], dtype=np.complex64),
            "next_fidelity": np.asarray(outcome["next_fidelity"], dtype=np.float32),
            "utility": np.asarray(outcome["utility"], dtype=np.float32),
            "action_induced_utility": np.asarray(
                outcome["action_induced_utility"], dtype=np.float32
            ),
            "amplitude_alignment_gain": np.asarray(
                outcome["amplitude_alignment_gain"], dtype=np.float32
            ),
            "phase_alignment_gain": np.asarray(
                outcome["phase_alignment_gain"], dtype=np.float32
            ),
            "harmful": np.asarray(outcome["harmful"], dtype=np.float32),
            "nuisance_targets": nuisance_targets.astype(np.float32),
            "exposure_s": exposure.astype(np.float32),
            "exposure_known_mask": np.ones(int(batch_size), dtype=np.float32),
            "reference_exposure_s": np.full(
                int(batch_size), REFERENCE_EXPOSURE_S, dtype=np.float32
            ),
            "transition_supervision_mask": np.ones(
                int(batch_size), dtype=np.float32
            ),
            "nuisance_coin_total": nuisance["coin_total"].astype(np.float32),
            "nuisance_signal_total": nuisance["signal_total"].astype(np.float32),
            "nuisance_idler_total": nuisance["idler_total"].astype(np.float32),
            "nuisance_read_count": nuisance["read_count"].astype(np.float32),
        }
    )
    return base


def build_synthetic_command_features(
    operation_features: np.ndarray,
    current_amplitude: np.ndarray,
    current_phase: np.ndarray,
) -> np.ndarray:
    """Create a bounded synthetic-only proxy for the 18 command features.

    The function is intentionally named and tagged as synthetic.  It exists to
    pretrain the command-conditioning branch without claiming that historical
    target-only measured rows contain command supervision.
    """

    operation = np.asarray(operation_features, dtype=np.float64)
    if operation.ndim != 3:
        raise ValueError("operation_features must be [batch, operation, feature]")
    batch, operation_count, _ = operation.shape
    amplitude = np.asarray(current_amplitude, dtype=np.float64)
    phase = np.asarray(current_phase, dtype=np.float64)
    if amplitude.shape[0] != batch or phase.shape != amplitude.shape:
        raise ValueError("current amplitude/phase batch shapes disagree")
    basis_amplitude = operation[
        :, :, OPERATION_FEATURE_NAMES.index("basis_amplitude_mean")
    ]
    basis_power = operation[
        :, :, OPERATION_FEATURE_NAMES.index("basis_amplitude_power")
    ]
    basis_phase_sin = operation[
        :, :, OPERATION_FEATURE_NAMES.index("basis_phase_sin_mean")
    ]
    basis_phase_cos = operation[
        :, :, OPERATION_FEATURE_NAMES.index("basis_phase_cos_mean")
    ]
    retrieval = operation[
        :, :, OPERATION_FEATURE_NAMES.index("retrieval_relative_map")
    ]
    x_coordinate = operation[
        :, :, OPERATION_FEATURE_NAMES.index("x_frequency")
    ]
    y_coordinate = operation[
        :, :, OPERATION_FEATURE_NAMES.index("y_frequency")
    ]
    amplitude_scale = np.mean(np.abs(amplitude), axis=1, keepdims=True)
    phase_sin = np.mean(np.sin(phase), axis=1, keepdims=True)
    phase_cos = np.mean(np.cos(phase), axis=1, keepdims=True)
    w_x_log_amplitude = np.log(
        np.maximum(np.abs(basis_amplitude) * amplitude_scale, 1e-6)
    )
    r_x_log_amplitude = w_x_log_amplitude + np.log(
        np.maximum(retrieval, 1e-6)
    )
    w_y_amplitude = np.clip(amplitude_scale * (1.0 + 0.10 * y_coordinate), 0.0, 2.0)
    r_y_amplitude = np.clip(w_y_amplitude * retrieval, 0.0, 2.0)
    total_rf_power = np.maximum(basis_power, 0.0)
    w_rf_power = total_rf_power * np.square(amplitude_scale)
    r_rf_power = w_rf_power * np.square(np.maximum(retrieval, 0.0))
    zeros = np.zeros((batch, operation_count), dtype=np.float64)
    values = np.stack(
        (
            w_x_log_amplitude,
            np.tanh(basis_phase_sin + phase_sin),
            np.tanh(basis_phase_cos + phase_cos),
            r_x_log_amplitude,
            np.tanh(basis_phase_sin - phase_sin),
            np.tanh(basis_phase_cos - phase_cos),
            w_y_amplitude,
            np.broadcast_to(phase_sin, (batch, operation_count)),
            np.broadcast_to(phase_cos, (batch, operation_count)),
            r_y_amplitude,
            np.tanh(np.broadcast_to(phase_sin, (batch, operation_count)) + x_coordinate),
            np.tanh(np.broadcast_to(phase_cos, (batch, operation_count)) + y_coordinate),
            total_rf_power,
            w_rf_power,
            r_rf_power,
            zeros,
            zeros,
            zeros,
        ),
        axis=-1,
    )
    if values.shape != (batch, operation_count, 18):
        raise AssertionError("synthetic command feature contract changed")
    return np.nan_to_num(values, nan=0.0, posinf=20.0, neginf=-20.0).astype(
        np.float32
    )


__all__ = [
    "EXPOSURE_LEVELS_S",
    "REFERENCE_EXPOSURE_S",
    "SYNTHETIC_FAMILY_ROLES",
    "V8_SYNTHETIC_FAMILIES",
    "V8_SYNTHETIC_SCHEMA",
    "V8SyntheticFamilyConfig",
    "build_synthetic_command_features",
    "generate_v8_transition_batch",
    "resolve_synthetic_family",
]
