"""Paired V6 feedback utility, acceptance, and local 6x6 adaptation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_ACCEPTANCE_PROBABILITY = 0.90
DEFAULT_MAXIMUM_RETRIEVAL_DROP = 0.02
DEFAULT_ONE_SIDED_Z = 1.645
DEFAULT_SERIOUS_HARMFUL_DELTA = -0.05


@dataclass(frozen=True)
class UtilityWeightsV1:
    fidelity: float = 1.0
    count: float = 0.05
    cycle: float = 0.10
    retrieval: float = 1.0
    multi: float = 0.10
    action: float = 0.01


@dataclass(frozen=True)
class UtilityObservationV1:
    fidelity_lcb: float
    target_count_nll: float
    cycle_inconsistency: float
    retrieval_drop_penalty: float
    multi_click_anomaly: float
    action_energy: float

    def utility(self, weights: UtilityWeightsV1) -> float:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("utility observation must contain finite values")
        return float(
            weights.fidelity * self.fidelity_lcb
            - weights.count * self.target_count_nll
            - weights.cycle * self.cycle_inconsistency
            - weights.retrieval * self.retrieval_drop_penalty
            - weights.multi * self.multi_click_anomaly
            - weights.action * self.action_energy
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "UtilityObservationV1":
        return cls(**{
            name: float(payload[name])
            for name in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class UtilityDeltaPosteriorV1:
    mean: float
    std: float
    lower_confidence_bound: float
    probability_improvement: float
    paired_subwindow_count: int

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionDecisionV1:
    decision: str
    before_utility: float
    after_utility: float
    utility_delta_posterior: UtilityDeltaPosteriorV1
    hard_guards_passed: bool
    retrieval_drop: float
    cycle_consistency_passed: bool
    serious_harmful: bool
    reasons: Tuple[str, ...]

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["utility_delta_posterior"] = self.utility_delta_posterior.to_json_dict()
        payload["reasons"] = list(self.reasons)
        return payload


class FeedbackTransitionEvaluator:
    """Evaluate one complete before/after pair without unpaired shortcuts."""

    def __init__(
        self,
        weights: UtilityWeightsV1 | None = None,
        *,
        acceptance_probability: float = DEFAULT_ACCEPTANCE_PROBABILITY,
        maximum_retrieval_drop: float = DEFAULT_MAXIMUM_RETRIEVAL_DROP,
        serious_harmful_delta: float = DEFAULT_SERIOUS_HARMFUL_DELTA,
    ) -> None:
        self.weights = weights or UtilityWeightsV1()
        self.acceptance_probability = float(acceptance_probability)
        self.maximum_retrieval_drop = float(maximum_retrieval_drop)
        self.serious_harmful_delta = float(serious_harmful_delta)
        if not 0.5 < self.acceptance_probability < 1.0:
            raise ValueError("acceptance_probability must be between 0.5 and 1")
        if self.maximum_retrieval_drop < 0.0:
            raise ValueError("maximum_retrieval_drop must be non-negative")

    def evaluate(
        self,
        before_subwindows: Sequence[UtilityObservationV1 | Mapping[str, Any]],
        after_subwindows: Sequence[UtilityObservationV1 | Mapping[str, Any]],
        *,
        hard_guards: Mapping[str, bool],
        retrieval_drop: float,
        cycle_consistency_passed: bool,
        serious_safety_failure: bool = False,
    ) -> TransitionDecisionV1:
        if not before_subwindows or len(before_subwindows) != len(after_subwindows):
            raise ValueError("before and after require equal non-empty paired subwindows")
        before = [self._observation(value) for value in before_subwindows]
        after = [self._observation(value) for value in after_subwindows]
        before_utility = np.asarray([item.utility(self.weights) for item in before], dtype=float)
        after_utility = np.asarray([item.utility(self.weights) for item in after], dtype=float)
        paired_delta = after_utility - before_utility
        mean = float(np.mean(paired_delta))
        if paired_delta.size > 1:
            std = float(np.std(paired_delta, ddof=1) / math.sqrt(paired_delta.size))
        else:
            std = 0.0
        lower = mean - DEFAULT_ONE_SIDED_Z * std
        if std <= np.finfo(float).eps:
            probability = 1.0 if mean > 0.0 else (0.5 if mean == 0.0 else 0.0)
        else:
            probability = float(NormalDist().cdf(mean / std))
        posterior = UtilityDeltaPosteriorV1(
            mean=mean,
            std=std,
            lower_confidence_bound=lower,
            probability_improvement=probability,
            paired_subwindow_count=int(paired_delta.size),
        )
        failed_guards = sorted(str(name) for name, passed in hard_guards.items() if not bool(passed))
        guards_passed = not failed_guards
        reasons = [f"hard_guard:{name}" for name in failed_guards]
        retrieval_drop_value = float(retrieval_drop)
        if retrieval_drop_value > self.maximum_retrieval_drop:
            reasons.append("retrieval_drop")
        if not cycle_consistency_passed:
            reasons.append("cycle_consistency")
        if probability < self.acceptance_probability:
            reasons.append("probability_improvement")
        if lower <= 0.0:
            reasons.append("utility_delta_lcb")
        serious_harmful = bool(
            serious_safety_failure or lower <= self.serious_harmful_delta
        )
        if serious_safety_failure:
            reasons.append("serious_safety_failure")
        accepted = (
            guards_passed
            and probability >= self.acceptance_probability
            and lower > 0.0
            and retrieval_drop_value <= self.maximum_retrieval_drop
            and bool(cycle_consistency_passed)
            and not serious_safety_failure
        )
        return TransitionDecisionV1(
            decision="accepted" if accepted else "rollback",
            before_utility=float(np.mean(before_utility)),
            after_utility=float(np.mean(after_utility)),
            utility_delta_posterior=posterior,
            hard_guards_passed=guards_passed,
            retrieval_drop=retrieval_drop_value,
            cycle_consistency_passed=bool(cycle_consistency_passed),
            serious_harmful=serious_harmful,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _observation(
        value: UtilityObservationV1 | Mapping[str, Any],
    ) -> UtilityObservationV1:
        if isinstance(value, UtilityObservationV1):
            return value
        return UtilityObservationV1.from_mapping(value)


class OnlineLocalResponseAdapter:
    """Episode-local Broyden update restricted to one 6x6 response matrix."""

    kind = "broyden_6x6"
    reset_each_state = True
    dimension = 6

    def __init__(
        self,
        initial_response: Iterable[Iterable[float]],
        *,
        regularization: float = 1e-8,
        trust_radius: float = 1.0,
        bias: Iterable[float] | None = None,
    ) -> None:
        response = np.asarray(initial_response, dtype=float)
        if response.shape != (self.dimension, self.dimension):
            raise ValueError("initial_response must be 6x6")
        if not np.all(np.isfinite(response)):
            raise ValueError("initial_response must be finite")
        self._initial_response = response.copy()
        self.response = response.copy()
        self.regularization = float(regularization)
        self.trust_radius = float(trust_radius)
        self.bias = np.zeros(self.dimension, dtype=float) if bias is None else np.asarray(bias, dtype=float)
        if self.bias.shape != (self.dimension,):
            raise ValueError("bias must contain six values")
        if self.regularization <= 0.0 or self.trust_radius <= 0.0:
            raise ValueError("regularization and trust_radius must be positive")

    def update(
        self,
        action: Iterable[float],
        error_before: Iterable[float],
        error_after: Iterable[float],
    ) -> np.ndarray:
        action_vector = self._vector(action, "action")
        error_delta = self._vector(error_after, "error_after") - self._vector(
            error_before, "error_before"
        )
        denominator = float(action_vector @ action_vector + self.regularization)
        residual = error_delta - self.response @ action_vector
        self.response = self.response + np.outer(residual, action_vector) / denominator
        return self.response.copy()

    def project_action(self, error: Iterable[float]) -> np.ndarray:
        error_vector = self._vector(error, "error") - self.bias
        action = -np.linalg.pinv(self.response) @ error_vector
        norm = float(np.linalg.norm(action))
        if norm > self.trust_radius:
            action *= self.trust_radius / norm
        return action

    def reset_for_state(self, initial_response: Iterable[Iterable[float]] | None = None) -> None:
        if initial_response is not None:
            response = np.asarray(initial_response, dtype=float)
            if response.shape != (self.dimension, self.dimension):
                raise ValueError("initial_response must be 6x6")
            self._initial_response = response.copy()
        self.response = self._initial_response.copy()
        self.bias = np.zeros(self.dimension, dtype=float)

    @classmethod
    def allowed_online_parameters(cls) -> Tuple[str, ...]:
        return ("local_response_6x6", "low_dimensional_bias", "trust_radius")

    @staticmethod
    def forbidden_online_parameters() -> Tuple[str, ...]:
        return ("neural_model", "detector_dark", "source_multiplicity")

    @classmethod
    def _vector(cls, value: Iterable[float], name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (cls.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain six finite values")
        return vector


__all__ = [
    "FeedbackTransitionEvaluator",
    "OnlineLocalResponseAdapter",
    "TransitionDecisionV1",
    "UtilityDeltaPosteriorV1",
    "UtilityObservationV1",
    "UtilityWeightsV1",
]
