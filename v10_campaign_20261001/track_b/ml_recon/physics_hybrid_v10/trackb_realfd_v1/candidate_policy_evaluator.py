"""Train-only conservative policy selection for Track B model-only actions.

The evaluator scores fixed action-scale and active-round policies against the
same V2.4 synthetic pre-states used by the candidate-action evaluator.  It is
strictly research-only: the result is neither model-selection evidence nor an
authorization to emit RF, AWG, or hardware commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import math

import numpy as np
import torch

from ml_recon.data_v2_4.hashing import sha256_array

from .candidate_action_evaluator import (
    _iter_trackb_model_action_contexts,
    is_materially_negative_trackb_gain,
    score_trackb_candidate_action,
)
from .contracts import canonical_sha256
from .inverse_transformer import TrackBInverseTransformerV1
from .training import trackb_model_state_sha


TEN_ROUND_COUNT = 10
TRAIN_SCREEN_ROLE = "TRAIN_SCREEN"
FROZEN_VALIDATION_ROLE = "FROZEN_VALIDATION"
POLICY_EVALUATION_ROLES = (TRAIN_SCREEN_ROLE, FROZEN_VALIDATION_ROLE)


@dataclass(frozen=True)
class TrackBPolicyEvaluationBindingV1:
    """Immutable model/data/split identity for one policy-screen result."""

    v24_build_id: str
    v24_ready_sha: str
    v24_contract_sha: str
    v24_schema_sha: str
    v24_generator_sha: str
    v24_build_manifest_sha: str
    v24_index_sha: str
    v24_split_sha: str
    parent_split_sample_set_sha: str
    evaluation_sample_set_sha: str
    evaluation_sample_count: int
    cell_id: str
    split: str
    system_type: str
    mode_count: int
    model_checkpoint_sha: str
    model_config_sha: str
    model_state_sha: str
    model_source_commit: str
    model_source_tree: str
    evaluation_input_sha: str
    global_seed: int
    noise_family: str
    eta: float
    kappa: float
    replay_shot_mode: str
    replay_shot_budget: int
    evaluation_role: str
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBPolicyEvaluationBindingV1"

    def validate(self) -> None:
        if (type(self.v24_build_id) is not str or not self.v24_build_id
                or self.system_type not in ("3qubit", "4qubit")
                or self.mode_count != int(self.system_type[0])
                or self.cell_id != f"trackb_feedback_history_n{self.mode_count}"):
            raise ValueError("Track B policy binding system identity is invalid")
        if self.split not in ("train", "validation") or self.evaluation_role not in POLICY_EVALUATION_ROLES:
            raise ValueError("Track B policy binding split role is invalid")
        if ((self.evaluation_role == TRAIN_SCREEN_ROLE and self.split != "train")
                or (self.evaluation_role == FROZEN_VALIDATION_ROLE and self.split != "validation")):
            raise ValueError("Track B policy binding role does not match its V2.4 split")
        if type(self.evaluation_sample_count) is not int or self.evaluation_sample_count <= 0:
            raise ValueError("Track B policy binding sample count is invalid")
        for name in (
            "v24_ready_sha",
            "v24_contract_sha",
            "v24_schema_sha",
            "v24_generator_sha",
            "v24_build_manifest_sha",
            "v24_index_sha",
            "v24_split_sha",
            "parent_split_sample_set_sha",
            "evaluation_sample_set_sha",
            "model_checkpoint_sha",
            "model_config_sha",
            "model_state_sha",
            "evaluation_input_sha",
        ):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ValueError(f"Track B policy binding {name} is invalid")
        for name in ("model_source_commit", "model_source_tree"):
            value = getattr(self, name)
            if type(value) is not str or len(value) not in (40, 64):
                raise ValueError(f"Track B policy binding {name} is invalid")
        if (type(self.global_seed) is not int
                or type(self.replay_shot_budget) is not int
                or self.replay_shot_budget <= 0
                or type(self.noise_family) is not str
                or not self.noise_family
                or self.replay_shot_mode not in ("S_total", "S_setting", "S_ideal")
                or not math.isfinite(float(self.eta))
                or not math.isfinite(float(self.kappa))
                or not 0.0 <= float(self.eta) <= 1.0
                or not 0.0 <= float(self.kappa) <= 1.0):
            raise ValueError("Track B policy replay binding is invalid")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.hardware_action_authorized):
            raise ValueError("policy binding cannot authorize selection, promotion, or hardware")
        if self.schema != "TrackBPolicyEvaluationBindingV1":
            raise ValueError("unsupported Track B policy binding schema")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TrackBActionScaleTailPolicyV1:
    """A fixed logical-action scale followed by a no-op tail."""

    action_scale: float
    active_round_count: int
    schema: str = "TrackBActionScaleTailPolicyV1"

    def validate(self) -> None:
        if (type(self.action_scale) not in (int, float)
                or not math.isfinite(float(self.action_scale))
                or not 0.0 <= float(self.action_scale) <= 1.0):
            raise ValueError("Track B action scale must be a finite value in [0, 1]")
        if type(self.active_round_count) is not int or not 0 <= self.active_round_count <= TEN_ROUND_COUNT:
            raise ValueError("Track B active round count is invalid")
        if (float(self.action_scale) == 0.0) != (self.active_round_count == 0):
            raise ValueError("the no-action policy must be represented canonically")
        if self.schema != "TrackBActionScaleTailPolicyV1":
            raise ValueError("unsupported Track B action-scale tail policy schema")

    @property
    def policy_id(self) -> str:
        self.validate()
        scale = f"{float(self.action_scale):.4f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"model_scale_{scale}_active_rounds_{self.active_round_count}"

    def action_for_round(self, model_action: np.ndarray, *, round_index: int) -> np.ndarray:
        """Apply the policy without changing the logical action gauge."""
        self.validate()
        if type(round_index) is not int or not 0 <= round_index < TEN_ROUND_COUNT:
            raise ValueError("Track B policy round index is invalid")
        action = np.asarray(model_action, dtype=np.float64)
        if round_index >= self.active_round_count:
            return np.zeros_like(action)
        return action * float(self.action_scale)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TrackBActionScaleTailPolicySummaryV1:
    """Aggregate forward result for one immutable action-scale tail policy."""

    policy: TrackBActionScaleTailPolicyV1
    sample_count: int
    active_step_count: int
    mean_gain_by_round: tuple[float, ...]
    mean_active_step_gain: float
    negative_active_step_gain_rate: float
    mean_cumulative_gain: float
    median_cumulative_gain: float
    minimum_cumulative_gain: float
    negative_trajectory_gain_rate: float
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBActionScaleTailPolicySummaryV1"

    def validate(self) -> None:
        self.policy.validate()
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("Track B policy summary requires at least one sample")
        if type(self.active_step_count) is not int or self.active_step_count < 0:
            raise ValueError("Track B policy active-step count is invalid")
        if self.active_step_count != self.sample_count * self.policy.active_round_count:
            raise ValueError("Track B policy active-step count disagrees with its fixed tail")
        if len(self.mean_gain_by_round) != TEN_ROUND_COUNT:
            raise ValueError("Track B policy summary must report all ten rounds")
        for value in (
            *self.mean_gain_by_round,
            self.mean_active_step_gain,
            self.mean_cumulative_gain,
            self.median_cumulative_gain,
            self.minimum_cumulative_gain,
            self.negative_active_step_gain_rate,
            self.negative_trajectory_gain_rate,
        ):
            if not math.isfinite(float(value)):
                raise ValueError("Track B policy summary contains a non-finite metric")
        for value in (self.negative_active_step_gain_rate, self.negative_trajectory_gain_rate):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("Track B policy summary risk rate is outside [0, 1]")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.hardware_action_authorized):
            raise ValueError("policy scoring cannot authorize selection, promotion, or hardware")
        if self.schema != "TrackBActionScaleTailPolicySummaryV1":
            raise ValueError("unsupported Track B policy summary schema")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TrackBActionScaleTailPolicyEvaluationV1:
    """Source-bound teacher-forced policy evaluation for one V2.4 split."""

    binding: TrackBPolicyEvaluationBindingV1
    sample_ids_sha: str
    sample_count: int
    policy_summaries: tuple[TrackBActionScaleTailPolicySummaryV1, ...]
    evaluation_kind: str = "SYNTHETIC_TEACHER_FORCED_ACTION_SCALE_TAIL_ENGINEERING_ONLY"
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBActionScaleTailPolicyEvaluationV1"

    def validate(self) -> None:
        self.binding.validate()
        if len(self.sample_ids_sha) != 64 or type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("Track B policy evaluation identity is invalid")
        if (self.sample_ids_sha != self.binding.evaluation_sample_set_sha
                or self.sample_count != self.binding.evaluation_sample_count):
            raise ValueError("Track B policy evaluation differs from its immutable sample binding")
        if not self.policy_summaries:
            raise ValueError("Track B policy evaluation requires at least one policy")
        identifiers = []
        for summary in self.policy_summaries:
            summary.validate()
            if summary.sample_count != self.sample_count:
                raise ValueError("Track B policy evaluation mixes sample counts")
            identifiers.append(summary.policy.policy_id)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Track B policy evaluation repeats a policy identity")
        if self.evaluation_kind != "SYNTHETIC_TEACHER_FORCED_ACTION_SCALE_TAIL_ENGINEERING_ONLY":
            raise ValueError("unsupported Track B policy evaluation kind")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.hardware_action_authorized):
            raise ValueError("policy evaluation cannot authorize selection, promotion, or hardware")
        if self.schema != "TrackBActionScaleTailPolicyEvaluationV1":
            raise ValueError("unsupported Track B policy evaluation schema")

    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TrackBResearchOnlyPolicyChoiceV1:
    """A train-only conservative choice that never grants external action authority."""

    policy: TrackBActionScaleTailPolicyV1
    training_evaluation_sha: str
    training_binding: TrackBPolicyEvaluationBindingV1
    safe_nonzero_policy_count: int
    max_negative_active_step_gain_rate: float
    minimum_mean_cumulative_gain: float
    decision_reason: str
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBResearchOnlyPolicyChoiceV1"

    def validate(self) -> None:
        self.policy.validate()
        self.training_binding.validate()
        if self.training_binding.evaluation_role != TRAIN_SCREEN_ROLE:
            raise ValueError("Track B policy choice requires a registered TRAIN screen")
        if len(self.training_evaluation_sha) != 64:
            raise ValueError("Track B research-only policy choice requires a training evaluation SHA")
        if type(self.safe_nonzero_policy_count) is not int or self.safe_nonzero_policy_count < 0:
            raise ValueError("Track B safe policy count is invalid")
        if (not math.isfinite(float(self.max_negative_active_step_gain_rate))
                or not 0.0 <= float(self.max_negative_active_step_gain_rate) <= 1.0
                or not math.isfinite(float(self.minimum_mean_cumulative_gain))
                or not self.decision_reason):
            raise ValueError("Track B research-only policy choice thresholds are invalid")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.hardware_action_authorized):
            raise ValueError("policy choice cannot authorize selection, promotion, or hardware")
        if self.schema != "TrackBResearchOnlyPolicyChoiceV1":
            raise ValueError("unsupported Track B research-only policy choice schema")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def evaluate_trackb_action_scale_tail_policies(
    model: TrackBInverseTransformerV1,
    rows: Mapping[int, Mapping[str, Any]],
    *,
    policies: Sequence[TrackBActionScaleTailPolicyV1],
    binding: TrackBPolicyEvaluationBindingV1,
    global_seed: int,
    generator_sha: str,
    cell_id: str,
    split: str,
    noise_family: str,
    eta: float,
    kappa: float,
    shot_mode: str,
    shot_budget: int,
    device: str = "cpu",
) -> TrackBActionScaleTailPolicyEvaluationV1:
    """Evaluate fixed scales and no-op tails on exactly one bound V2.4 split."""
    binding.validate()
    if (binding.system_type != model.config.system_type
            or binding.mode_count != model.config.mode_count
            or binding.cell_id != cell_id
            or binding.split != split
            or binding.v24_generator_sha != generator_sha
            or binding.global_seed != global_seed
            or binding.noise_family != noise_family
            or binding.eta != float(eta)
            or binding.kappa != float(kappa)
            or binding.replay_shot_mode != shot_mode
            or binding.replay_shot_budget != shot_budget):
        raise ValueError("Track B policy evaluator arguments differ from the immutable binding")
    if (binding.model_config_sha != canonical_sha256(model.config.to_dict())
            or binding.model_state_sha != trackb_model_state_sha(model)):
        raise ValueError("Track B policy evaluator model differs from the immutable binding")
    if binding.evaluation_input_sha != _policy_rows_input_sha(rows):
        raise ValueError("Track B policy evaluator rows differ from the immutable binding")
    candidates = tuple(policies)
    if not candidates:
        raise ValueError("Track B policy evaluator requires at least one policy")
    identifiers = []
    for policy in candidates:
        policy.validate()
        identifiers.append(policy.policy_id)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Track B policy evaluator received duplicate policies")
    values = {
        policy.policy_id: {
            "per_round": [[] for _ in range(TEN_ROUND_COUNT)],
            "cumulative": {},
            "negative_active_steps": 0,
        }
        for policy in candidates
    }
    sample_ids: list[str] = []
    for context, model_action in _iter_trackb_model_action_contexts(
        model,
        rows,
        global_seed=global_seed,
        generator_sha=generator_sha,
        cell_id=cell_id,
        split=split,
        noise_family=noise_family,
        eta=eta,
        kappa=kappa,
        shot_mode=shot_mode,
        shot_budget=shot_budget,
        device=device,
    ):
        if context.round_index == 0:
            sample_ids.append(context.sample_id)
        for policy in candidates:
            action = policy.action_for_round(model_action, round_index=context.round_index)
            score = score_trackb_candidate_action(context, action, candidate_kind="model")
            current = values[policy.policy_id]
            current["per_round"][context.round_index].append(score.gain)
            current["cumulative"][context.sample_id] = (
                float(current["cumulative"].get(context.sample_id, 0.0)) + score.gain)
            if context.round_index < policy.active_round_count:
                current["negative_active_steps"] += int(is_materially_negative_trackb_gain(score.gain))
    summaries = []
    for policy in candidates:
        current = values[policy.policy_id]
        cumulative = np.asarray(list(current["cumulative"].values()), dtype=np.float64)
        if cumulative.size != len(sample_ids):
            raise RuntimeError("Track B policy evaluator lost a source-bound sample")
        active_rounds = current["per_round"][:policy.active_round_count]
        active_gains = np.asarray(
            [gain for round_gains in active_rounds for gain in round_gains], dtype=np.float64)
        if policy.active_round_count and active_gains.size != len(sample_ids) * policy.active_round_count:
            raise RuntimeError("Track B policy evaluator lost an active candidate action")
        summary = TrackBActionScaleTailPolicySummaryV1(
            policy=policy,
            sample_count=len(sample_ids),
            active_step_count=int(active_gains.size),
            mean_gain_by_round=tuple(
                float(np.asarray(round_gains, dtype=np.float64).mean())
                for round_gains in current["per_round"]
            ),
            mean_active_step_gain=float(active_gains.mean()) if active_gains.size else 0.0,
            negative_active_step_gain_rate=(
                float(current["negative_active_steps"] / active_gains.size)
                if active_gains.size else 0.0
            ),
            mean_cumulative_gain=float(cumulative.mean()),
            median_cumulative_gain=float(np.median(cumulative)),
            minimum_cumulative_gain=float(cumulative.min()),
            negative_trajectory_gain_rate=float(
                np.asarray([is_materially_negative_trackb_gain(value) for value in cumulative]).mean()),
        )
        summary.validate()
        summaries.append(summary)
    result = TrackBActionScaleTailPolicyEvaluationV1(
        binding=binding,
        sample_ids_sha=canonical_sha256(sample_ids),
        sample_count=len(sample_ids),
        policy_summaries=tuple(summaries),
    )
    result.validate()
    return result


def _policy_rows_input_sha(rows: Mapping[int, Mapping[str, Any]]) -> str:
    """Hash exactly the model-visible inputs before a policy evaluator forward."""
    required = (
        "history_tokens",
        "history_mask",
        "noisy_measurement_tokens",
        "target_mode_tokens",
        "mode_mask",
    )
    records = []
    for source_index in sorted(rows):
        row = rows[source_index]
        if int(str(row["sample_id"]).rsplit(":", 1)[-1]) != int(source_index):
            raise ValueError("Track B policy input row identity is invalid")
        field_hashes = {}
        for field in required:
            value = row.get(field)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Track B policy input lacks tensor {field}")
            field_hashes[field] = sha256_array(value.detach().cpu().numpy())
        records.append({"sample_id": str(row["sample_id"]), **field_hashes})
    return canonical_sha256(records)


def choose_research_only_conservative_policy(
    training_evaluation: TrackBActionScaleTailPolicyEvaluationV1,
    *,
    max_negative_active_step_gain_rate: float = 0.25,
    minimum_mean_cumulative_gain: float = 0.0,
) -> TrackBResearchOnlyPolicyChoiceV1:
    """Choose on the train split only; fall back to an explicit no-action policy."""
    training_evaluation.validate()
    if training_evaluation.binding.evaluation_role != TRAIN_SCREEN_ROLE:
        raise ValueError("Track B conservative-policy selection accepts only a TRAIN screen")
    if (not math.isfinite(float(max_negative_active_step_gain_rate))
            or not 0.0 <= float(max_negative_active_step_gain_rate) <= 1.0
            or not math.isfinite(float(minimum_mean_cumulative_gain))):
        raise ValueError("Track B conservative-policy thresholds are invalid")
    no_action = next(
        (summary for summary in training_evaluation.policy_summaries
         if summary.policy.active_round_count == 0),
        None,
    )
    if no_action is None:
        raise ValueError("Track B conservative-policy selection requires an explicit no-action baseline")
    safe = [
        summary for summary in training_evaluation.policy_summaries
        if summary.policy.active_round_count > 0
        and summary.negative_active_step_gain_rate <= max_negative_active_step_gain_rate
        and summary.negative_trajectory_gain_rate <= max_negative_active_step_gain_rate
        and summary.mean_cumulative_gain > minimum_mean_cumulative_gain
    ]
    if safe:
        selected = max(
            safe,
            key=lambda summary: (
                summary.mean_cumulative_gain,
                -summary.negative_active_step_gain_rate,
                -summary.negative_trajectory_gain_rate,
                summary.minimum_cumulative_gain,
                summary.policy.active_round_count,
                summary.policy.action_scale,
            ),
        )
        reason = "TRAIN_ONLY_NONZERO_POLICY_MET_CONSERVATIVE_RISK_AND_GAIN_LIMITS"
    else:
        selected = no_action
        reason = "NO_NONZERO_POLICY_MET_TRAIN_ONLY_CONSERVATIVE_RISK_AND_GAIN_LIMITS"
    result = TrackBResearchOnlyPolicyChoiceV1(
        policy=selected.policy,
        training_evaluation_sha=training_evaluation.sha256(),
        training_binding=training_evaluation.binding,
        safe_nonzero_policy_count=len(safe),
        max_negative_active_step_gain_rate=float(max_negative_active_step_gain_rate),
        minimum_mean_cumulative_gain=float(minimum_mean_cumulative_gain),
        decision_reason=reason,
    )
    result.validate()
    return result


__all__ = [
    "TEN_ROUND_COUNT",
    "TRAIN_SCREEN_ROLE",
    "FROZEN_VALIDATION_ROLE",
    "TrackBPolicyEvaluationBindingV1",
    "TrackBActionScaleTailPolicyV1",
    "TrackBActionScaleTailPolicySummaryV1",
    "TrackBActionScaleTailPolicyEvaluationV1",
    "TrackBResearchOnlyPolicyChoiceV1",
    "evaluate_trackb_action_scale_tail_policies",
    "choose_research_only_conservative_policy",
]
