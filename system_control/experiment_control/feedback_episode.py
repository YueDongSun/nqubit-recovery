"""Guarded 900 second V6 feedback episode contracts and state machine.

The episode controller coordinates immutable command bundles through injected
services.  It has no legacy delta path and cannot promote model checkpoints.
Hardware access is possible only through the command bundle store's injected
uploader transaction.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .data_protocol import SnapshotManifestV3, SnapshotManifestV4
from .feedback_transition_evaluator import FeedbackTransitionEvaluator, TransitionDecisionV1
from .subwindow_accumulator import SubwindowAccumulator


DEFAULT_MEASUREMENT_WINDOW_S = 900.0
DEFAULT_SUBWINDOW_S = 150.0
DEFAULT_MAXIMUM_ACTION_ROUNDS = 5
MAXIMUM_SUPPORTED_ROUNDS = 10
DEFAULT_MAXIMUM_ACCEPTED_ACTIONS = 5

EPISODE_STATUSES = (
    "SESSION_CREATED",
    "BASELINE_PREPARED",
    "BASELINE_UPLOADED",
    "BASELINE_MEASURING",
    "BASELINE_MEASURED",
    "PROPOSAL_GENERATED",
    "CANDIDATE_MATERIALIZED",
    "CANDIDATE_GUARDED",
    "WAITING_OPERATOR_APPROVAL",
    "CANDIDATE_ACTIVATED",
    "CANDIDATE_MEASURING",
    "OUTCOME_EVALUATED",
    "ITERATION_ACCEPTED",
    "ITERATION_ROLLED_BACK",
    "STOPPED_KEEP_BEST",
    "ABORTED_RESTORED_BASELINE",
    "COMPLETED",
)
TERMINAL_EPISODE_STATUSES = {
    "STOPPED_KEEP_BEST", "ABORTED_RESTORED_BASELINE", "COMPLETED",
}
_EPISODE_TRANSITIONS = {
    "SESSION_CREATED": {"BASELINE_PREPARED"},
    "BASELINE_PREPARED": {"BASELINE_UPLOADED"},
    "BASELINE_UPLOADED": {"BASELINE_MEASURING"},
    "BASELINE_MEASURING": {"BASELINE_MEASURED"},
    "BASELINE_MEASURED": {"PROPOSAL_GENERATED", "COMPLETED"},
    "PROPOSAL_GENERATED": {"CANDIDATE_MATERIALIZED"},
    "CANDIDATE_MATERIALIZED": {"CANDIDATE_GUARDED"},
    "CANDIDATE_GUARDED": {"WAITING_OPERATOR_APPROVAL", "ITERATION_ROLLED_BACK"},
    "WAITING_OPERATOR_APPROVAL": {"CANDIDATE_ACTIVATED", "ITERATION_ROLLED_BACK"},
    "CANDIDATE_ACTIVATED": {"CANDIDATE_MEASURING", "ITERATION_ROLLED_BACK"},
    "CANDIDATE_MEASURING": {"OUTCOME_EVALUATED", "ITERATION_ROLLED_BACK"},
    "OUTCOME_EVALUATED": {"ITERATION_ACCEPTED", "ITERATION_ROLLED_BACK"},
    "ITERATION_ACCEPTED": {"PROPOSAL_GENERATED", "COMPLETED"},
    "ITERATION_ROLLED_BACK": {"PROPOSAL_GENERATED", "STOPPED_KEEP_BEST", "COMPLETED"},
    "STOPPED_KEEP_BEST": set(),
    "ABORTED_RESTORED_BASELINE": set(),
    "COMPLETED": set(),
}


def required_action_candidates(full_action: Sequence[float]) -> Tuple[Dict[str, Any], ...]:
    """Return the six mandatory V6 episode candidates for a 6D full action."""

    values = tuple(float(value) for value in full_action)
    if len(values) != 6:
        raise ValueError("V6 episode full action must contain six values")
    amplitude = values[:3]
    phase = values[3:]
    candidates = (
        ("zero", (0.0,) * 6, 0.0),
        ("phase_only", (0.0,) * 3 + phase, 1.0),
        ("amplitude_only", amplitude + (0.0,) * 3, 1.0),
        ("full_0_25", tuple(0.25 * value for value in values), 0.25),
        ("full_0_50", tuple(0.50 * value for value in values), 0.50),
        ("full_1_00", values, 1.0),
    )
    return tuple({
        "action_kind": kind,
        "action": list(action),
        "trust_radius": scale,
        "brightness_delta": 0.0,
        "read_action": 0.0,
        "local_residual": 0.0,
        "cycle_residual": 0.0,
    } for kind, action, scale in candidates)


@dataclass(frozen=True)
class FeedbackEpisodeV1:
    episode_id: str
    system_type: str
    target_state: Mapping[str, Any]
    target_hash: str
    baseline_bundle_id: Optional[str]
    baseline_bundle_hash: Optional[str]
    model_checkpoint_hash: str
    detector_calibration_hash: str
    measurement_window_s: float = DEFAULT_MEASUREMENT_WINDOW_S
    subwindow_s: float = DEFAULT_SUBWINDOW_S
    maximum_action_rounds: int = DEFAULT_MAXIMUM_ACTION_ROUNDS
    maximum_accepted_actions: int = DEFAULT_MAXIMUM_ACCEPTED_ACTIONS
    active_iteration: int = 0
    accepted_iteration: int = 0
    best_iteration: int = 0
    active_bundle_id: Optional[str] = None
    active_bundle_hash: Optional[str] = None
    best_bundle_id: Optional[str] = None
    best_bundle_hash: Optional[str] = None
    accepted_snapshot_id: Optional[str] = None
    accepted_snapshot_hash: Optional[str] = None
    cumulative_action: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    accepted_action_count: int = 0
    consecutive_no_improvement: int = 0
    status: str = "SESSION_CREATED"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id is required")
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("system_type must be 3qubit or 4qubit")
        if not self.target_hash or not self.model_checkpoint_hash or not self.detector_calibration_hash:
            raise ValueError("target, checkpoint, and detector calibration hashes are required")
        if self.status not in EPISODE_STATUSES:
            raise ValueError(f"unsupported episode status: {self.status}")
        if self.maximum_action_rounds < 1 or self.maximum_action_rounds > MAXIMUM_SUPPORTED_ROUNDS:
            raise ValueError("maximum_action_rounds must be in 1..10")
        if self.maximum_accepted_actions < 1 or self.maximum_accepted_actions > self.maximum_action_rounds:
            raise ValueError("maximum_accepted_actions must be in 1..maximum_action_rounds")
        if self.active_iteration < 0 or self.active_iteration > MAXIMUM_SUPPORTED_ROUNDS:
            raise ValueError("active_iteration must be in 0..10")
        if len(self.cumulative_action) != 6:
            raise ValueError("cumulative_action must contain six values")
        ratio = self.measurement_window_s / self.subwindow_s
        if self.measurement_window_s <= 0.0 or self.subwindow_s <= 0.0 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError("measurement_window_s must be a positive multiple of subwindow_s")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["target_state"] = dict(self.target_state)
        payload["cumulative_action"] = list(self.cumulative_action)
        payload["schema"] = "FeedbackEpisodeV1"
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "FeedbackEpisodeV1":
        values = dict(payload)
        if values.pop("schema", "FeedbackEpisodeV1") != "FeedbackEpisodeV1":
            raise ValueError("unsupported feedback episode schema")
        values["cumulative_action"] = tuple(float(value) for value in values["cumulative_action"])
        return cls(**values)


@dataclass(frozen=True)
class FeedbackIterationV2:
    iteration_index: int
    parent_accepted_iteration: int
    before_snapshot_id: str
    before_snapshot_hash: str
    before_bundle_id: str
    before_bundle_hash: str
    proposal: Mapping[str, Any]
    requested_action: Mapping[str, Any]
    projected_action: Mapping[str, Any]
    actual_loaded_action: Mapping[str, Any] = field(default_factory=dict)
    candidate_bundle_id: Optional[str] = None
    candidate_bundle_hash: Optional[str] = None
    awg_upload_receipt: Optional[Mapping[str, Any]] = None
    awg_upload_receipt_hash: Optional[str] = None
    after_snapshot_id: Optional[str] = None
    after_snapshot_hash: Optional[str] = None
    predicted_utility: Optional[float] = None
    observed_utility: Optional[float] = None
    utility_delta_posterior: Mapping[str, Any] = field(default_factory=dict)
    probability_of_improvement: Optional[float] = None
    decision: Optional[str] = None
    rollback_receipt: Optional[Mapping[str, Any]] = None
    operator_approval: Mapping[str, Any] = field(default_factory=dict)
    timestamps: Mapping[str, float] = field(default_factory=dict)
    hard_guard_result: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.iteration_index < 1 or self.iteration_index > MAXIMUM_SUPPORTED_ROUNDS:
            raise ValueError("iteration_index must be in 1..10")
        if self.parent_accepted_iteration < 0 or self.parent_accepted_iteration >= self.iteration_index:
            raise ValueError("parent_accepted_iteration must identify an earlier accepted state")
        for value_name in (
            "before_snapshot_id", "before_snapshot_hash", "before_bundle_id", "before_bundle_hash",
        ):
            if not str(getattr(self, value_name)):
                raise ValueError(f"{value_name} is required")
        if (self.candidate_bundle_id is None) != (self.candidate_bundle_hash is None):
            raise ValueError("candidate bundle ID/hash must be recorded together")
        if (self.after_snapshot_id is None) != (self.after_snapshot_hash is None):
            raise ValueError("after snapshot ID/hash must be recorded together")
        if (self.awg_upload_receipt is None) != (self.awg_upload_receipt_hash is None):
            raise ValueError("AWG upload receipt/hash must be recorded together")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = "FeedbackIterationV2"
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "FeedbackIterationV2":
        values = dict(payload)
        if values.pop("schema", "FeedbackIterationV2") != "FeedbackIterationV2":
            raise ValueError("unsupported feedback iteration schema")
        return cls(**values)


class FeedbackEpisodeController:
    """Persist every V6 episode transition before proceeding to the next side effect."""

    def __init__(
        self,
        episode_store: Any,
        bundle_store: Any,
        uploader: Any,
        *,
        evaluator: FeedbackTransitionEvaluator | None = None,
        stop_after_serious_harmful: bool = True,
        snapshot_validator: Any | None = None,
    ) -> None:
        self.episode_store = episode_store
        self.bundle_store = bundle_store
        self.uploader = uploader
        self.evaluator = evaluator or FeedbackTransitionEvaluator()
        self.stop_after_serious_harmful = bool(stop_after_serious_harmful)
        self.snapshot_validator = snapshot_validator

    def create_episode(
        self,
        *,
        system_type: str,
        target_state: Mapping[str, Any],
        target_hash: str,
        model_checkpoint_hash: str,
        detector_calibration_hash: str,
        measurement_window_s: float = DEFAULT_MEASUREMENT_WINDOW_S,
        subwindow_s: float = DEFAULT_SUBWINDOW_S,
        maximum_action_rounds: int = DEFAULT_MAXIMUM_ACTION_ROUNDS,
        maximum_accepted_actions: int = DEFAULT_MAXIMUM_ACCEPTED_ACTIONS,
        episode_id: str | None = None,
    ) -> FeedbackEpisodeV1:
        episode = FeedbackEpisodeV1(
            episode_id=episode_id or f"episode_{uuid.uuid4().hex[:20]}",
            system_type=system_type,
            target_state=dict(target_state),
            target_hash=str(target_hash),
            baseline_bundle_id=None,
            baseline_bundle_hash=None,
            model_checkpoint_hash=str(model_checkpoint_hash),
            detector_calibration_hash=str(detector_calibration_hash),
            measurement_window_s=float(measurement_window_s),
            subwindow_s=float(subwindow_s),
            maximum_action_rounds=int(maximum_action_rounds),
            maximum_accepted_actions=int(maximum_accepted_actions),
        )
        self.episode_store.create(episode)
        return episode

    def register_baseline_prepared(self, episode_id: str, bundle: Any) -> FeedbackEpisodeV1:
        episode = self._require_status(episode_id, "SESSION_CREATED")
        if bundle.parent_bundle_id is not None or bundle.target_hash != episode.target_hash:
            raise ValueError("baseline bundle must be a target-matched root bundle")
        episode = replace(
            episode,
            baseline_bundle_id=bundle.bundle_id,
            baseline_bundle_hash=bundle.bundle_hash,
            active_bundle_id=bundle.bundle_id,
            active_bundle_hash=bundle.bundle_hash,
            best_bundle_id=bundle.bundle_id,
            best_bundle_hash=bundle.bundle_hash,
        )
        return self._transition(episode, "BASELINE_PREPARED")

    def upload_baseline(
        self, episode_id: str, *, operator: str, approval_note: str = "",
    ) -> FeedbackEpisodeV1:
        episode = self._require_status(episode_id, "BASELINE_PREPARED")
        approved = self.bundle_store.approve(
            episode.baseline_bundle_id, operator=operator, approval_note=approval_note,
        )
        active = self.bundle_store.activate_for_trial(approved.bundle_id, self.uploader)
        if not active.upload_receipt_hash:
            raise RuntimeError("baseline upload did not produce a receipt hash")
        return self._transition(episode, "BASELINE_UPLOADED")

    def start_baseline_measurement(self, episode_id: str) -> FeedbackEpisodeV1:
        return self._transition(
            self._require_status(episode_id, "BASELINE_UPLOADED"),
            "BASELINE_MEASURING",
        )

    def complete_baseline_measurement(
        self,
        episode_id: str,
        *,
        snapshot: SnapshotManifestV3 | SnapshotManifestV4,
        snapshot_hash: str,
        accumulator: SubwindowAccumulator,
        prepared_artifact: Any,
    ) -> FeedbackEpisodeV1:
        episode = self._require_status(episode_id, "BASELINE_MEASURING")
        accumulator.require_authoritative_window()
        bundle = self.bundle_store.verify_bundle(episode.baseline_bundle_id)
        self._validate_snapshot(
            snapshot,
            prepared_artifact=prepared_artifact,
            bundle=bundle,
            episode_id=episode.episode_id,
            iteration_index=0,
            model_checkpoint_hash=episode.model_checkpoint_hash,
        )
        measured = self.bundle_store.mark_measurement_completed(
            bundle.bundle_id, snapshot_hash=str(snapshot_hash),
        )
        accepted = self.bundle_store.accept(measured.bundle_id)
        episode = replace(
            episode,
            active_bundle_id=accepted.bundle_id,
            active_bundle_hash=accepted.bundle_hash,
            best_bundle_id=accepted.bundle_id,
            best_bundle_hash=accepted.bundle_hash,
            accepted_snapshot_id=snapshot.sample_id,
            accepted_snapshot_hash=str(snapshot_hash),
        )
        return self._transition(episode, "BASELINE_MEASURED")

    def generate_proposal(
        self,
        episode_id: str,
        *,
        proposal: Mapping[str, Any],
        requested_action: Mapping[str, Any],
        projected_action: Mapping[str, Any],
        predicted_utility: float | None = None,
    ) -> FeedbackIterationV2:
        episode = self.episode_store.load(episode_id)
        if episode.status not in {"BASELINE_MEASURED", "ITERATION_ACCEPTED", "ITERATION_ROLLED_BACK"}:
            raise ValueError(f"cannot generate proposal from {episode.status}")
        if episode.consecutive_no_improvement >= 2:
            self._transition(episode, "STOPPED_KEEP_BEST")
            raise RuntimeError("episode stopped after two iterations without improvement")
        next_index = episode.active_iteration + 1
        if next_index > episode.maximum_action_rounds:
            self._transition(episode, "COMPLETED")
            raise RuntimeError("episode reached maximum_action_rounds")
        if not episode.accepted_snapshot_id or not episode.accepted_snapshot_hash:
            raise RuntimeError("accepted parent snapshot is missing")
        iteration = FeedbackIterationV2(
            iteration_index=next_index,
            parent_accepted_iteration=episode.accepted_iteration,
            before_snapshot_id=episode.accepted_snapshot_id,
            before_snapshot_hash=episode.accepted_snapshot_hash,
            before_bundle_id=str(episode.active_bundle_id),
            before_bundle_hash=str(episode.active_bundle_hash),
            proposal=dict(proposal),
            requested_action=dict(requested_action),
            projected_action=dict(projected_action),
            predicted_utility=None if predicted_utility is None else float(predicted_utility),
            timestamps={"proposal_generated_at": time.time()},
        )
        self.episode_store.save_iteration(episode_id, iteration)
        self._transition(replace(episode, active_iteration=next_index), "PROPOSAL_GENERATED")
        return iteration

    def register_candidate_materialized(
        self,
        episode_id: str,
        *,
        bundle: Any,
        prepared_artifact: Any,
    ) -> FeedbackIterationV2:
        episode = self._require_status(episode_id, "PROPOSAL_GENERATED")
        iteration = self.episode_store.load_iteration(episode_id, episode.active_iteration)
        if bundle.parent_bundle_id != iteration.before_bundle_id:
            raise ValueError("candidate parent must be the most recent accepted bundle")
        if bundle.target_hash != episode.target_hash or prepared_artifact.target_hash != episode.target_hash:
            raise ValueError("candidate changed the immutable target")
        if prepared_artifact.parent_control_hash != self.bundle_store.load_manifest(
            iteration.before_bundle_id
        ).control_hash:
            raise ValueError("prepared artifact parent control hash mismatch")
        iteration = replace(
            iteration,
            actual_loaded_action=dict(
                prepared_artifact.materialization.get("actual_loaded_action") or {}
            ),
            candidate_bundle_id=bundle.bundle_id,
            candidate_bundle_hash=bundle.bundle_hash,
            timestamps={**dict(iteration.timestamps), "candidate_materialized_at": time.time()},
        )
        self.episode_store.save_iteration(episode_id, iteration)
        self._transition(episode, "CANDIDATE_MATERIALIZED")
        return iteration

    def guard_candidate(
        self, episode_id: str, guard_result: Mapping[str, Any],
    ) -> FeedbackIterationV2:
        episode = self._require_status(episode_id, "CANDIDATE_MATERIALIZED")
        iteration = self.episode_store.load_iteration(episode_id, episode.active_iteration)
        passed = bool(guard_result.get("passed", False))
        iteration = replace(
            iteration,
            hard_guard_result=dict(guard_result),
            timestamps={**dict(iteration.timestamps), "candidate_guarded_at": time.time()},
        )
        self.episode_store.save_iteration(episode_id, iteration)
        guarded = self._transition(episode, "CANDIDATE_GUARDED")
        if not passed:
            self.bundle_store.reject(iteration.candidate_bundle_id, reason="hard guard failed")
            iteration = replace(iteration, decision="rejected_pre_activation")
            self.episode_store.save_iteration(episode_id, iteration)
            self._transition(
                replace(guarded, consecutive_no_improvement=guarded.consecutive_no_improvement + 1),
                "ITERATION_ROLLED_BACK",
            )
            return iteration
        self._transition(guarded, "WAITING_OPERATOR_APPROVAL")
        return iteration

    def approve_and_activate_candidate(
        self,
        episode_id: str,
        *,
        operator: str,
        approval_note: str = "",
        fault_injector: Any | None = None,
    ) -> FeedbackIterationV2:
        episode = self._require_status(episode_id, "WAITING_OPERATOR_APPROVAL")
        iteration = self.episode_store.load_iteration(episode_id, episode.active_iteration)
        if not str(operator).strip():
            raise ValueError("operator identity is required")
        candidate = self.bundle_store.load_manifest(iteration.candidate_bundle_id)
        if candidate.status == "staged":
            approved = self.bundle_store.approve(
                iteration.candidate_bundle_id,
                operator=operator,
                approval_note=approval_note,
            )
        elif candidate.status == "operator_approved":
            approved = candidate
        else:
            raise ValueError(f"candidate cannot be activated from {candidate.status}")
        activated = self.bundle_store.activate_for_trial(
            approved.bundle_id,
            self.uploader,
            fault_injector=fault_injector,
        )
        if not activated.upload_receipt or not activated.upload_receipt_hash:
            raise RuntimeError("candidate activation is missing upload receipt/hash")
        approval = {
            "operator": str(operator),
            "approval_note": str(approval_note),
            "approved_at": time.time(),
        }
        iteration = replace(
            iteration,
            awg_upload_receipt=dict(activated.upload_receipt),
            awg_upload_receipt_hash=activated.upload_receipt_hash,
            operator_approval=approval,
            timestamps={**dict(iteration.timestamps), "candidate_activated_at": time.time()},
        )
        self.episode_store.save_iteration(episode_id, iteration)
        self._transition(
            replace(
                episode,
                active_bundle_id=activated.bundle_id,
                active_bundle_hash=activated.bundle_hash,
            ),
            "CANDIDATE_ACTIVATED",
        )
        return iteration

    def start_candidate_measurement(self, episode_id: str) -> FeedbackEpisodeV1:
        return self._transition(
            self._require_status(episode_id, "CANDIDATE_ACTIVATED"),
            "CANDIDATE_MEASURING",
        )

    def complete_candidate_measurement(
        self,
        episode_id: str,
        *,
        snapshot: SnapshotManifestV3 | SnapshotManifestV4,
        snapshot_hash: str,
        accumulator: SubwindowAccumulator,
        prepared_artifact: Any,
        before_subwindows: Sequence[Mapping[str, Any]],
        after_subwindows: Sequence[Mapping[str, Any]],
        hard_guards: Mapping[str, bool],
        retrieval_drop: float,
        cycle_consistency_passed: bool,
    ) -> TransitionDecisionV1:
        episode = self._require_status(episode_id, "CANDIDATE_MEASURING")
        accumulator.require_authoritative_window()
        iteration = self.episode_store.load_iteration(episode_id, episode.active_iteration)
        bundle = self.bundle_store.verify_bundle(iteration.candidate_bundle_id)
        self._validate_snapshot(
            snapshot,
            prepared_artifact=prepared_artifact,
            bundle=bundle,
            episode_id=episode.episode_id,
            iteration_index=iteration.iteration_index,
            model_checkpoint_hash=episode.model_checkpoint_hash,
        )
        self.bundle_store.mark_measurement_completed(
            bundle.bundle_id, snapshot_hash=str(snapshot_hash),
        )
        decision = self.evaluator.evaluate(
            before_subwindows,
            after_subwindows,
            hard_guards=hard_guards,
            retrieval_drop=retrieval_drop,
            cycle_consistency_passed=cycle_consistency_passed,
        )
        iteration = replace(
            iteration,
            after_snapshot_id=snapshot.sample_id,
            after_snapshot_hash=str(snapshot_hash),
            observed_utility=decision.after_utility,
            utility_delta_posterior=decision.utility_delta_posterior.to_json_dict(),
            probability_of_improvement=decision.utility_delta_posterior.probability_improvement,
            decision=decision.decision,
            timestamps={**dict(iteration.timestamps), "outcome_evaluated_at": time.time()},
        )
        self.episode_store.save_iteration(episode_id, iteration)
        evaluated = self._transition(episode, "OUTCOME_EVALUATED")
        if decision.decision == "accepted":
            accepted = self.bundle_store.accept(bundle.bundle_id)
            cumulative = self._six_values(
                list(prepared_artifact.materialization.get("cumulative_delta_log_amp") or ())
                + list(prepared_artifact.materialization.get("cumulative_delta_phase") or ())
            )
            episode = replace(
                evaluated,
                accepted_iteration=iteration.iteration_index,
                best_iteration=iteration.iteration_index,
                active_bundle_id=accepted.bundle_id,
                active_bundle_hash=accepted.bundle_hash,
                best_bundle_id=accepted.bundle_id,
                best_bundle_hash=accepted.bundle_hash,
                accepted_snapshot_id=snapshot.sample_id,
                accepted_snapshot_hash=str(snapshot_hash),
                cumulative_action=cumulative,
                accepted_action_count=evaluated.accepted_action_count + 1,
                consecutive_no_improvement=0,
            )
            episode = self._transition(episode, "ITERATION_ACCEPTED")
            if (
                episode.active_iteration >= episode.maximum_action_rounds
                or episode.accepted_action_count >= episode.maximum_accepted_actions
            ):
                self._transition(episode, "COMPLETED")
        else:
            rolled_back = self.bundle_store.rollback(bundle.bundle_id, self.uploader)
            status_record = self.episode_store.bundle_status_record(
                self.bundle_store.bundle_path(bundle.bundle_id)
            )
            rollback_event = next(
                event for event in reversed(status_record["history"])
                if event.get("status") == "rolled_back"
            )
            iteration = replace(iteration, rollback_receipt=rollback_event.get("rollback_receipt"))
            self.episode_store.save_iteration(episode_id, iteration)
            parent = self.bundle_store.load_manifest(iteration.before_bundle_id)
            episode = replace(
                evaluated,
                active_bundle_id=parent.bundle_id,
                active_bundle_hash=parent.bundle_hash,
                consecutive_no_improvement=evaluated.consecutive_no_improvement + 1,
            )
            episode = self._transition(episode, "ITERATION_ROLLED_BACK")
            if decision.serious_harmful and self.stop_after_serious_harmful:
                self._transition(episode, "STOPPED_KEEP_BEST")
            elif episode.consecutive_no_improvement >= 2:
                self._transition(episode, "STOPPED_KEEP_BEST")
            elif episode.active_iteration >= episode.maximum_action_rounds:
                self._transition(episode, "COMPLETED")
        return decision

    def abort_and_restore_baseline(self, episode_id: str, *, reason: str) -> FeedbackEpisodeV1:
        episode = self.episode_store.load(episode_id)
        if episode.status in TERMINAL_EPISODE_STATUSES:
            return episode
        pointer = self.bundle_store.active_pointer()
        if pointer and pointer.get("bundle_id") != episode.baseline_bundle_id:
            active = self.bundle_store.load_manifest(str(pointer["bundle_id"]))
            if active.status in {"activated_for_trial", "measurement_completed"}:
                self.bundle_store.rollback(active.bundle_id, self.uploader)
        return self._transition(episode, "ABORTED_RESTORED_BASELINE", details={"reason": str(reason)})

    def recover_episode(self, episode_id: str) -> FeedbackEpisodeV1:
        """Reconcile an interrupted bundle transaction without inventing a decision."""

        episode = self.episode_store.load(episode_id)
        self.bundle_store.recover_active_bundle(self.uploader)
        pointer = self.bundle_store.active_pointer()
        if episode.status == "BASELINE_PREPARED":
            baseline = self.bundle_store.load_manifest(episode.baseline_bundle_id)
            if baseline.status == "activated_for_trial" and pointer and pointer.get("bundle_id") == baseline.bundle_id:
                episode = self._transition(
                    replace(
                        episode,
                        active_bundle_id=baseline.bundle_id,
                        active_bundle_hash=baseline.bundle_hash,
                    ),
                    "BASELINE_UPLOADED",
                    details={"recovered": True},
                )
        elif episode.status == "WAITING_OPERATOR_APPROVAL":
            iteration = self.episode_store.load_iteration(episode_id, episode.active_iteration)
            candidate = self.bundle_store.load_manifest(iteration.candidate_bundle_id)
            if (
                candidate.status == "activated_for_trial"
                and pointer
                and pointer.get("bundle_id") == candidate.bundle_id
                and candidate.upload_receipt
                and candidate.upload_receipt_hash
            ):
                iteration = replace(
                    iteration,
                    awg_upload_receipt=dict(candidate.upload_receipt),
                    awg_upload_receipt_hash=candidate.upload_receipt_hash,
                    timestamps={**dict(iteration.timestamps), "activation_recovered_at": time.time()},
                )
                self.episode_store.save_iteration(episode_id, iteration)
                episode = self._transition(
                    replace(
                        episode,
                        active_bundle_id=candidate.bundle_id,
                        active_bundle_hash=candidate.bundle_hash,
                    ),
                    "CANDIDATE_ACTIVATED",
                    details={"recovered": True},
                )
        return self.episode_store.resume(
            episode_id,
            target_hash=episode.target_hash,
            model_checkpoint_hash=episode.model_checkpoint_hash,
            detector_calibration_hash=episode.detector_calibration_hash,
            bundle_store=self.bundle_store,
        )

    def _require_status(self, episode_id: str, expected: str) -> FeedbackEpisodeV1:
        episode = self.episode_store.load(episode_id)
        if episode.status != expected:
            raise ValueError(f"expected episode status {expected}, got {episode.status}")
        return episode

    def _validate_snapshot(self, manifest: Any, **kwargs: Any) -> None:
        if self.snapshot_validator is not None:
            self.snapshot_validator(manifest, **kwargs)
            return
        try:
            from .data_protocol import validate_v6_episode_snapshot
        except ImportError:  # Phase 7 compatibility before strict Phase 8 provenance lands.
            if abs(float(manifest.duration_s) - DEFAULT_MEASUREMENT_WINDOW_S) > 1e-6:
                raise ValueError("episode snapshot must cover the 900 second authority window")
            return
        validate_v6_episode_snapshot(manifest, **kwargs)

    def _transition(
        self,
        episode: FeedbackEpisodeV1,
        next_status: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> FeedbackEpisodeV1:
        if next_status not in EPISODE_STATUSES:
            raise ValueError(f"unsupported episode status: {next_status}")
        allowed = _EPISODE_TRANSITIONS.get(episode.status, set())
        if next_status not in allowed and next_status != "ABORTED_RESTORED_BASELINE":
            raise ValueError(f"invalid episode transition {episode.status} -> {next_status}")
        updated = replace(episode, status=next_status, updated_at=time.time())
        self.episode_store.save(updated)
        self.episode_store.record_transition(
            updated.episode_id,
            previous_status=episode.status,
            next_status=next_status,
            details={} if details is None else dict(details),
        )
        return updated

    @staticmethod
    def _six_values(value: Any) -> Tuple[float, ...]:
        if isinstance(value, Mapping):
            vector = value.get("vector") or value.get("action") or [0.0] * 6
        else:
            vector = value
        values = tuple(float(item) for item in vector)
        if len(values) > 6:
            raise ValueError("cumulative action cannot exceed six values")
        return values + (0.0,) * (6 - len(values))


__all__ = [
    "EPISODE_STATUSES",
    "FeedbackEpisodeController",
    "FeedbackEpisodeV1",
    "FeedbackIterationV2",
    "MAXIMUM_SUPPORTED_ROUNDS",
    "required_action_candidates",
]
