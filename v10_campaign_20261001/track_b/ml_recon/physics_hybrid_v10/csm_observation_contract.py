"""Explicit observation and action-source contracts for V10 CSM experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .csm_contract import OBSERVATION_CONTRACT_KINDS
from .contracts import sha256_of
from .csm_covariance import (
    block_diagonal_covariance, multinomial_setting_covariance,
    sufficient_stat_covariance,
)
from .noise_interventions import registered_feedback_settings
from .state_backends import all_pauli_outcomes


CSM_ORACLE_CATEGORICAL_V1 = "CSM_ORACLE_CATEGORICAL_V1"
CSM_FEEDBACK_CARTESIAN_STATS_V1 = "CSM_FEEDBACK_CARTESIAN_STATS_V1"
CSM_FEEDBACK_PHASE_PROXY_TOKENS_V1 = "CSM_FEEDBACK_PHASE_PROXY_TOKENS_V1"
CSM_ACTION_ORACLE_SOURCE_BOUND_LOCAL_LINEAR = (
    "ORACLE_SOURCE_BOUND_LOCAL_LINEAR")
CSM_ACTION_DEPLOYABLE_ANALYTIC = "DEPLOYABLE_ANALYTIC"
CSM_ACTION_FROZEN_STUDENT = "FROZEN_STUDENT"
OBSERVATION_KINDS = set(OBSERVATION_CONTRACT_KINDS)
ACTION_SOURCES = {
    CSM_ACTION_ORACLE_SOURCE_BOUND_LOCAL_LINEAR,
    CSM_ACTION_DEPLOYABLE_ANALYTIC,
    CSM_ACTION_FROZEN_STUDENT,
}


@dataclass(frozen=True)
class CSMObservationContract:
    """Bind the information level used to construct a CSM geometry."""

    kind: str
    feature_names: Tuple[str, ...]
    feature_layout: str
    state_information_level: str
    measurement_semantics: str
    research_only: bool = True
    hardware_action_authorized: bool = False

    def validate(self) -> None:
        if self.kind not in OBSERVATION_KINDS:
            raise ValueError("unknown CSM observation contract kind")
        if not self.feature_names:
            raise ValueError("CSM observation contract needs feature names")
        if not self.feature_layout or not self.state_information_level:
            raise ValueError("CSM observation contract metadata is incomplete")
        if not self.measurement_semantics:
            raise ValueError("CSM observation semantics are required")
        if not self.research_only or self.hardware_action_authorized:
            raise ValueError("CSM observations are research-only and hardware-free")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {"schema": "V10CSMObservationContractV1", **asdict(self)}

    def sha256(self) -> str:
        return sha256_of(self.to_dict())


@dataclass(frozen=True)
class CSMActionSourceContract:
    """Bind the source and information level of an action used in a rollout."""

    action_source: str
    lambda_u: float
    covariance_contract_sha: str
    observation_contract_kind: str
    state_information_level: str
    direction_authorized: bool = True
    research_only: bool = True
    hardware_action_authorized: bool = False
    actual_lambda_u: float | None = None
    action_coordinate_layout: str = "logical_interleaved_amp_phase"
    control_manifold_sha: str = ""
    bounds_sha: str = ""
    solver_backend: str = ""
    solver_status: str = ""
    solver_success: bool | None = None
    action_forward_sha: str = ""
    estimator_sha: str = ""
    checkpoint_sha: str = ""
    covariance_semantics_kind: str = ""
    covariance_floor: float | None = None
    phase_uncertainty_semantics: str = ""
    projection_contract_sha: str = ""
    covariance_semantics_sha: str = ""
    covariance_instance_sha: str = ""

    def validate(self) -> None:
        if self.action_source not in ACTION_SOURCES:
            raise ValueError("unknown CSM action source")
        if not np.isfinite(self.lambda_u) or self.lambda_u < 0.0:
            raise ValueError("action-source lambda_u must be finite and non-negative")
        actual_lambda = self.lambda_u if self.actual_lambda_u is None else self.actual_lambda_u
        if not np.isfinite(actual_lambda) or actual_lambda < 0.0:
            raise ValueError("actual action-source lambda_u must be finite and non-negative")
        if not self.action_coordinate_layout:
            raise ValueError("action coordinate layout is required")
        for name in (
            "control_manifold_sha", "bounds_sha", "action_forward_sha",
            "estimator_sha", "checkpoint_sha", "projection_contract_sha",
            "covariance_semantics_sha", "covariance_instance_sha",
        ):
            value = getattr(self, name)
            if value and len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest when supplied")
        if len(self.covariance_contract_sha) != 64:
            raise ValueError("action-source covariance SHA must be bound")
        if self.observation_contract_kind not in OBSERVATION_KINDS:
            raise ValueError("action-source observation kind is unknown")
        if not self.state_information_level:
            raise ValueError("action-source state information level is required")
        if self.action_source == CSM_ACTION_ORACLE_SOURCE_BOUND_LOCAL_LINEAR:
            if "truth" not in self.state_information_level:
                raise ValueError("oracle action source must bind truth information")
        elif self.action_source == CSM_ACTION_DEPLOYABLE_ANALYTIC:
            if "truth" in self.state_information_level:
                raise ValueError("deployable action source cannot bind truth information")
            if self.observation_contract_kind != CSM_FEEDBACK_PHASE_PROXY_TOKENS_V1:
                raise ValueError(
                    "deployable analytic action must bind phase-proxy tokens")
            if self.covariance_semantics_kind != "FEEDBACK_PHASE_PROXY_DIAGONAL_APPROX_V1":
                raise ValueError("deployable action covariance semantics are not registered")
            if self.covariance_floor is None or not np.isfinite(self.covariance_floor) \
                    or self.covariance_floor < 0.0:
                raise ValueError("deployable action covariance floor must be non-negative")
            if self.phase_uncertainty_semantics != "RENDERER_SQRT_VARIANCE_PROXY_V1":
                raise ValueError("deployable phase uncertainty semantics are not registered")
        elif self.action_source == CSM_ACTION_FROZEN_STUDENT:
            if len(self.checkpoint_sha) != 64:
                raise ValueError("frozen student action needs checkpoint SHA")
        if not self.research_only or self.hardware_action_authorized:
            raise ValueError("CSM actions are research-only and hardware-free")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {"schema": "V10CSMActionSourceContractV1", **asdict(self)}

    def sha256(self) -> str:
        return sha256_of(self.to_dict())


def make_csm_observation_contract(kind: str) -> CSMObservationContract:
    if kind == CSM_ORACLE_CATEGORICAL_V1:
        return CSMObservationContract(
            kind=kind,
            feature_names=("setting_outcome_probability",),
            feature_layout="registered_categorical_per_setting",
            state_information_level="truth_state_truth_intervention_truth_severity",
            measurement_semantics="full_registered_intervention_measurement_law",
        )
    if kind == CSM_FEEDBACK_PHASE_PROXY_TOKENS_V1:
        return CSMObservationContract(
            kind=kind,
            feature_names=("population", "phase_proxy"),
            feature_layout="per_mode_phase_proxy_tokens",
            state_information_level="measured_estimated_coefficients",
            measurement_semantics="deployable_feedback_phase_proxy_tokens",
        )
    if kind == CSM_FEEDBACK_CARTESIAN_STATS_V1:
        return CSMObservationContract(
            kind=kind,
            feature_names=("population", "coherence_re", "coherence_im"),
            feature_layout="per_mode_cartesian_w_stats",
            state_information_level="measured_estimated_coefficients",
            measurement_semantics="deployable_feedback_sufficient_statistics",
        )
    raise ValueError("unknown CSM observation contract kind")


def w_cartesian_feedback_stats(coefficients: np.ndarray) -> np.ndarray:
    """Return stable population/coherence statistics for deployable feedback."""
    values = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    if values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("W coefficients must contain at least three finite modes")
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("W coefficients must have positive norm")
    values = values / norm
    coherence = np.conj(values[0]) * values[1:]
    return np.concatenate((
        np.abs(values) ** 2,
        coherence.real,
        coherence.imag,
    )).astype(np.float64)


CARTESIAN_STATS_LAYOUT = "grouped_population_coherence_re_im_v1"


def _feedback_settings(
    N: int, settings: Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, ...], ...]:
    selected = (registered_feedback_settings(N) if settings is None else
                tuple(tuple(str(axis) for axis in value) for value in settings))
    expected = registered_feedback_settings(N)
    if selected != expected:
        raise ValueError("Cartesian statistics require the registered feedback settings")
    return selected



class CartesianFeedbackAccumulator:
    """Streaming Cartesian sufficient statistics from registered setting counts."""

    def __init__(
        self,
        N: int,
        settings: Sequence[Sequence[str]] | None = None,
        *,
        dirichlet_alpha: float = 0.0,
    ) -> None:
        self.N = int(N)
        self.settings = _feedback_settings(self.N, settings)
        self.dirichlet_alpha = float(dirichlet_alpha)
        if not np.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha < 0.0:
            raise ValueError("Dirichlet covariance smoothing alpha must be non-negative")
        self.statistic_dimension = 3 * self.N - 2
        self.block_size = 2 ** self.N + 1
        self._outcomes = all_pauli_outcomes(self.N)
        self._stats = np.zeros(self.statistic_dimension, dtype=np.float64)
        self._covariance = np.zeros(
            (self.statistic_dimension, self.statistic_dimension), dtype=np.float64)
        self._valid_mask = np.zeros(self.statistic_dimension, dtype=np.bool_)
        self._seen = np.zeros(len(self.settings), dtype=np.bool_)
        self._detection_fraction = np.zeros(len(self.settings), dtype=np.float64)

    @property
    def covariance_estimator_kind(self) -> str:
        return (
            "EMPIRICAL_MULTINOMIAL_MLE_V1"
            if self.dirichlet_alpha == 0.0
            else "DIRICHLET_MULTINOMIAL_COVARIANCE_V1"
        )

    def _feature_values(self, setting: tuple[str, ...]) -> np.ndarray:
        values = np.zeros(
            (self.block_size, self.statistic_dimension), dtype=np.float64)
        outcomes = self._outcomes
        if setting == tuple("Z" for _ in range(self.N)):
            values[:-1, :self.N] = (outcomes < 0).astype(np.float64)
            return values
        for mode in range(1, self.N):
            xx = ["Z"] * self.N
            xy = ["Z"] * self.N
            xx[0] = xx[mode] = "X"
            xy[0], xy[mode] = "X", "Y"
            product = outcomes[:, 0] * outcomes[:, mode]
            if setting == tuple(xx):
                values[:-1, self.N + mode - 1] = 0.5 * product
                return values
            if setting == tuple(xy):
                values[:-1, 2 * self.N - 2 + mode] = 0.5 * product
                return values
        raise ValueError("unregistered Cartesian feedback setting")

    def add_setting_counts(
        self,
        setting_index: int,
        counts: np.ndarray,
        shots: float,
        *,
        sampling_enabled: bool = True,
    ) -> None:
        index = int(setting_index)
        if index < 0 or index >= len(self.settings):
            raise ValueError("Cartesian setting index is out of range")
        values = np.asarray(counts, dtype=np.float64).reshape(-1)
        if values.size != self.block_size or not np.all(np.isfinite(values)):
            raise ValueError("Cartesian setting counts have the wrong dimension")
        if np.any(values < -1e-12):
            raise ValueError("Cartesian setting counts cannot be negative")
        shot_value = float(shots)
        total = float(values.sum())
        if not np.isfinite(shot_value) or shot_value <= 0.0 or total <= 0.0:
            raise ValueError("Cartesian setting shots must be positive")
        if not np.isclose(total, shot_value, rtol=1e-6, atol=1e-8):
            raise ValueError("Cartesian setting counts do not match shot contract")
        probabilities = np.maximum(values, 0.0) / total
        covariance_probabilities = probabilities
        if self.dirichlet_alpha > 0.0:
            covariance_probabilities = (
                values + self.dirichlet_alpha) / (
                    total + self.dirichlet_alpha * self.block_size)
        features = self._feature_values(self.settings[index])
        active = np.flatnonzero(np.any(np.abs(features) > 0.0, axis=0))
        mean = covariance_probabilities @ features
        if sampling_enabled:
            second = (features.T * covariance_probabilities) @ features
            covariance = (second - np.outer(mean, mean)) / shot_value
        else:
            covariance = np.zeros_like(self._covariance)
        self._stats[active] += (probabilities @ features)[active]
        self._covariance[np.ix_(active, active)] += covariance[np.ix_(active, active)]
        self._valid_mask[active] = True
        self._seen[index] = True
        self._detection_fraction[index] = float(1.0 - probabilities[-1])

    def finalize(self) -> dict[str, np.ndarray | str]:
        if not np.all(self._seen):
            missing = np.flatnonzero(~self._seen).tolist()
            raise ValueError(f"Cartesian accumulator missing settings: {missing}")
        covariance = 0.5 * (self._covariance + self._covariance.T)
        return {
            "stats": self._stats.copy(),
            "covariance": covariance,
            "valid_mask": self._valid_mask.copy(),
            "detection_fraction": self._detection_fraction.copy(),
            "covariance_estimator_kind": self.covariance_estimator_kind,
            "dirichlet_alpha": float(self.dirichlet_alpha),
            "statistic_transform_sha": cartesian_feedback_transform_sha(
                self.N, self.settings),
            "covariance_instance_sha": sha256_of({
                "kind": self.covariance_estimator_kind,
                "dirichlet_alpha": self.dirichlet_alpha,
                "stats": self._stats.tolist(),
                "covariance": covariance.tolist(),
            }),
        }





def cartesian_feedback_statistic_transform(
    N: int, settings: Sequence[Sequence[str]] | None = None,
) -> np.ndarray:
    """Map registered categorical setting laws to Cartesian W statistics.

    The output order is ``[population[N], coherence_re[N-1],
    coherence_im[N-1]]``.  Each input setting is a categorical block with
    ``2**N`` Pauli outcomes followed by the detector no-click outcome.  The
    no-click column is zero because these are raw linear sufficient statistics;
    detector exposure therefore remains visible in the statistic covariance.
    """
    selected = _feedback_settings(N, settings)
    outcomes = all_pauli_outcomes(N)
    block_size = int(outcomes.shape[0] + 1)
    transform = np.zeros((3 * N - 2, block_size * len(selected)), dtype=np.float64)
    offsets = {value: index * block_size for index, value in enumerate(selected)}
    z_setting = tuple("Z" for _ in range(N))
    z_offset = offsets[z_setting]
    for outcome_index, outcome in enumerate(outcomes):
        for mode in range(N):
            transform[mode, z_offset + outcome_index] = float(outcome[mode] < 0)
    for mode in range(1, N):
        xx_axes = ["Z"] * N
        xy_axes = ["Z"] * N
        xx_axes[0] = xx_axes[mode] = "X"
        xy_axes[0], xy_axes[mode] = "X", "Y"
        xx_offset = offsets[tuple(xx_axes)]
        xy_offset = offsets[tuple(xy_axes)]
        for outcome_index, outcome in enumerate(outcomes):
            product_xx = float(outcome[0] * outcome[mode])
            product_xy = float(outcome[0] * outcome[mode])
            transform[N + mode - 1, xx_offset + outcome_index] = 0.5 * product_xx
            transform[2 * N - 2 + mode, xy_offset + outcome_index] = 0.5 * product_xy
    return transform


def _normalized_feedback_law_blocks(
    law: np.ndarray, N: int, settings: Sequence[Sequence[str]] | None,
) -> tuple[tuple[tuple[str, ...], ...], np.ndarray]:
    selected = _feedback_settings(N, settings)
    values = np.asarray(law, dtype=np.float64).reshape(-1)
    block_size = 2 ** N + 1
    expected_size = len(selected) * block_size
    if values.size != expected_size or not np.all(np.isfinite(values)):
        raise ValueError("feedback law has the wrong Cartesian statistic dimension")
    blocks = values.reshape(len(selected), block_size)
    totals = blocks.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("feedback law contains a non-positive setting block")
    normalized = blocks / totals
    if np.any(normalized < -1e-12):
        raise ValueError("feedback law contains negative probabilities")
    return selected, normalized


def cartesian_feedback_stats_from_law(
    law: np.ndarray, N: int, settings: Sequence[Sequence[str]] | None = None,
) -> np.ndarray:
    """Return measured Cartesian statistics from per-setting categorical laws."""
    selected, blocks = _normalized_feedback_law_blocks(law, N, settings)
    return cartesian_feedback_statistic_transform(N, selected) @ blocks.reshape(-1)


def cartesian_feedback_covariance_from_law(
    law: np.ndarray, shots: Sequence[float] | np.ndarray, N: int,
    settings: Sequence[Sequence[str]] | None = None,
) -> np.ndarray:
    """Propagate exact per-setting multinomial covariance to Cartesian stats."""
    selected, blocks = _normalized_feedback_law_blocks(law, N, settings)
    shot_values = np.broadcast_to(
        np.asarray(shots, dtype=np.float64).reshape(-1), (len(selected),)).copy()
    if np.any(~np.isfinite(shot_values)) or np.any(shot_values <= 0.0):
        raise ValueError("Cartesian statistic shots must be finite and positive")
    raw_covariance = block_diagonal_covariance(tuple(
        multinomial_setting_covariance(block, shot_values[index])
        for index, block in enumerate(blocks)))
    return sufficient_stat_covariance(
        raw_covariance, cartesian_feedback_statistic_transform(N, selected))


def cartesian_feedback_transform_sha(
    N: int, settings: Sequence[Sequence[str]] | None = None,
) -> str:
    return sha256_of(cartesian_feedback_statistic_transform(N, settings).tolist())


def coefficients_from_cartesian_feedback_stats(
    stats: np.ndarray, N: int,
) -> np.ndarray:
    """Recover a W coefficient gauge from observed Cartesian sufficient stats."""
    values = np.asarray(stats, dtype=np.float64).reshape(-1)
    if values.size != 3 * N - 2 or not np.all(np.isfinite(values)):
        raise ValueError("Cartesian W statistics have the wrong dimension")
    populations = np.maximum(values[:N], 0.0)
    total = float(populations.sum())
    if total <= 0.0:
        raise ValueError("Cartesian W statistics have no positive population")
    populations /= total
    coherence = values[N:N + N - 1] + 1j * values[2 * N - 1:]
    coefficients = np.sqrt(populations).astype(np.complex128)
    coefficients[1:] *= np.exp(1j * np.angle(coherence))
    coefficients /= np.linalg.norm(coefficients)
    return coefficients


__all__ = [
    "ACTION_SOURCES",
    "CartesianFeedbackAccumulator",
    "CARTESIAN_STATS_LAYOUT",
    "CSM_ACTION_DEPLOYABLE_ANALYTIC",
    "CSM_ACTION_FROZEN_STUDENT",
    "CSM_ACTION_ORACLE_SOURCE_BOUND_LOCAL_LINEAR",
    "CSMActionSourceContract",
    "CSM_FEEDBACK_CARTESIAN_STATS_V1",
    "CSM_FEEDBACK_PHASE_PROXY_TOKENS_V1",
    "CSM_ORACLE_CATEGORICAL_V1",
    "CSMObservationContract",
    "make_csm_observation_contract",
    "cartesian_feedback_covariance_from_law",
    "cartesian_feedback_statistic_transform",
    "cartesian_feedback_stats_from_law",
    "cartesian_feedback_transform_sha",
    "coefficients_from_cartesian_feedback_stats",
    "w_cartesian_feedback_stats",
]
