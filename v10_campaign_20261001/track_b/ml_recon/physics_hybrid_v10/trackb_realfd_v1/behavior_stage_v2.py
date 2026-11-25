"""Serial V2 behavior engineering stage with one owned output lifecycle.

External exact-source review and run-specific preregistration are workflow
prerequisites, not claims minted by this module. The entry has no CLI or generic
training bypass. Failure preserves partial evidence and never resumes a root.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Callable

from ml_recon.data_v2_4.admission_snapshot import Canonical_json_snapshot
from ml_recon.data_v2_4.hashing import canonical_json_bytes

from .artifact_directory_lease import Directory_lease
from .behavior_artifact_owner import Behavior_artifact_owner
from .behavior_dataset_v2 import publish_behavior_dataset
from .behavior_self_rollout_v2 import evaluate_behavior_model_v2, summarize_behavior_measurements
from .behavior_training_v2 import train_behavior_model_v2
from .behavior_v2_contract import Behavior_stage_plan_v2
from .contracts import require_sha256
from .teacher_diagnostic_admission import inspect_teacher_diagnostic, require_unchanged_teacher_evidence
from .teacher_diagnostic_preflight import require_unchanged_teacher_diagnostic_preflight


stage_ready_filename = "behavior_stage_ready.json"


@dataclass(frozen=True)
class Behavior_stage_receipt_v2:
    ready: Canonical_json_snapshot
    ready_file_identity: tuple[int, str]


def run_behavior_stage_v2(
    plan: Behavior_stage_plan_v2, *, expected_source: tuple[str, str], expected_plan_sha256: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> Behavior_stage_receipt_v2:
    """Run the fixed data -> 191-update behavior -> validation200 self-rollout.

    Caller-retained source/plan identities are independent of the candidate
    plan. This entry enforces engineering contracts, not external review,
    preregistration, scientific Gate, model selection or hardware authority.
    """
    if os.name != "nt" or type(plan) is not Behavior_stage_plan_v2:
        raise ValueError("behavior stage requires an explicit plan and Windows native leases")
    plan.validate()
    require_sha256(expected_plan_sha256, "behavior_stage_plan_sha")
    if expected_source != plan.expected_source or expected_plan_sha256 != plan.sha256():
        raise ValueError("behavior stage differs from independently pinned source or plan")
    if on_progress is not None and not callable(on_progress):
        raise ValueError("behavior progress callback must be callable")
    pin = plan.teacher_evidence_pin
    teacher_admission = inspect_teacher_diagnostic(
        pin, expected_source=expected_source, expected_plan_sha256=pin.expected_plan_sha256)

    def check_inputs() -> None:
        # All per-key teacher evidence was inspected at phase admission. The
        # same physical files/source are rechecked at every publish boundary;
        # complete durable teacher evidence is re-inspected between phases.
        require_unchanged_teacher_diagnostic_preflight(
            pin.plan.preflight_bindings[0], sample_key=pin.plan.sample_keys[0],
            config=pin.plan.config, expected_source=expected_source)

    def check_teacher() -> None:
        require_unchanged_teacher_evidence(
            teacher_admission, pin, expected_source=expected_source,
            expected_plan_sha256=pin.expected_plan_sha256,
            borrowed_parent=parent if parent is not None and parent.path == pin.plan.trusted_output_parent else None)

    check_inputs()  # No mkdir, rendering, or optimizer activity before this point.
    parent = None
    owner = None
    body_error = None
    try:
        parent = Directory_lease.open(plan.trusted_output_parent)
        owner = Behavior_artifact_owner.create_child(parent, plan.run_id, check_inputs=check_inputs)
        owner.publish("behavior_stage_plan.json", canonical_json_bytes({
            "plan": plan.portable_payload(), "plan_sha256": expected_plan_sha256,
            "root_identity": {"volume_serial_number": owner.root.volume_serial_number,
                              "file_id": owner.root.file_id}}))
        dataset = publish_behavior_dataset(
            plan, owner, teacher_admission, expected_source=expected_source,
            expected_plan_sha256=expected_plan_sha256, on_progress=on_progress)
        check_teacher()
        trained = train_behavior_model_v2(
            plan, owner, dataset, expected_source=expected_source,
            expected_plan_sha256=expected_plan_sha256, on_progress=on_progress)
        check_teacher()
        if owner.read("pilot/behavior_pilot_receipt.json") != trained.receipt.payload:
            raise ValueError("behavior model receipt changed before measurement")
        pilot_receipt = trained.receipt.decode()
        if (pilot_receipt["completed_optimizer_updates"] != plan.recipe.optimizer_steps
                or pilot_receipt["model"]["checkpoint"]["sha256"] != trained.checkpoint_sha256
                or pilot_receipt["model"]["config"] != trained.model.config.to_dict()):
            raise ValueError("behavior measurement does not bind the exact trained checkpoint/config")
        owner.create_directory("measurement")
        measurements = []
        for index, key in enumerate(plan.validation_keys):
            owner.verify()
            result = evaluate_behavior_model_v2(
                trained.model, key, recipe=plan.recipe, replicate_index=plan.evaluation_replicate_index)
            if result.sample_key != key or result.recipe_sha256 != plan.recipe.sha256():
                raise ValueError("behavior measurement unit differs from the planned key/recipe")
            owner.publish(f"measurement/unit_{index:06d}.json", canonical_json_bytes(asdict(result)))
            measurements.append(result)
            if on_progress is not None:
                on_progress({"phase": "MODEL_SELF_ROLLOUT_COMPLETED", "completed": index + 1,
                             "total": len(plan.validation_keys)})
        if len(measurements) != plan.recipe.validation_samples:
            raise ValueError("behavior measurement did not complete every registered validation key")
        summary = summarize_behavior_measurements(tuple(measurements))
        measurement_payload = canonical_json_bytes({
            "schema": "TrackBBehaviorMeasurementReceiptV2", "stage_plan_sha256": expected_plan_sha256,
            "source_commit": expected_source[0], "source_tree": expected_source[1],
            "dataset_receipt_sha256": dataset.receipt.sha256, "pilot_receipt_sha256": trained.receipt.sha256,
            "checkpoint_sha256": trained.checkpoint_sha256, "recipe_sha256": plan.recipe.sha256(),
            "evaluation_replicate_index": plan.evaluation_replicate_index,
            "summary": summary, "unit_inventory": [binding.to_dict() for binding in owner.inventory()
                                                       if binding.relative_path.startswith("measurement/unit_")],
            "research_only": True, "model_self_rollout_completed": True,
            "evaluation_scope": "EXPOSED_DEVELOPMENT_PANEL_NOT_FRESH_CONFIRMATORY_HOLDOUT",
            "scientific_gate": "NOT_PERFORMED", "selection_evidence": False,
            "promotion_ready": False, "hardware_action_authorized": False})
        measurement = owner.publish("measurement/behavior_measurement_receipt.json", measurement_payload)
        check_teacher()
        owner.verify()
        ready_payload = canonical_json_bytes({
            "schema": "TrackBBehaviorStageReadyV2", "stage_plan_sha256": expected_plan_sha256,
            "plan": plan.portable_payload(), "teacher_evidence": teacher_admission.identity_summary(),
            "source_commit": expected_source[0], "source_tree": expected_source[1],
            "root_identity": {"volume_serial_number": owner.root.volume_serial_number, "file_id": owner.root.file_id},
            "dataset_receipt_sha256": dataset.receipt.sha256, "pilot_receipt_sha256": trained.receipt.sha256,
            "checkpoint_sha256": trained.checkpoint_sha256, "measurement_receipt_sha256": measurement.sha256,
            "inventory": [binding.to_dict() for binding in owner.inventory()],
            "completed_optimizer_updates": plan.recipe.optimizer_steps,
            "completed_validation_episodes": len(measurements), "model_self_rollout_completed": True,
            "external_source_review": "NOT_CERTIFIED_BY_RUNNER", "registered_study_approval": "EXTERNAL_WORKFLOW_REQUIRED",
            "completion_boundary": "REQUIRES_SUCCESSFUL_RUN_RETURN_OR_INDEPENDENT_COMPLETION_INSPECTION",
            "scientific_gate": "NOT_PERFORMED", "learner_minus_behavior_gate": "NOT_PERFORMED_NO_LEARNER_TRAINED",
            "research_only": True, "selection_evidence": False,
            "promotion_ready": False, "hardware_action_authorized": False})
        ready_binding = owner.publish(stage_ready_filename, ready_payload)
        check_inputs()
        receipt = Behavior_stage_receipt_v2(
            Canonical_json_snapshot(ready_payload, ready_binding.sha256),
            (ready_binding.volume_serial_number, ready_binding.file_id))
    except BaseException as error:
        body_error = error
        if hasattr(error, "add_note"):
            error.add_note("RECOVERY_REQUIRED: preserve the partial behavior stage; no resume, deletion or completion claim")
        raise
    finally:
        failures = []
        for resource in (owner, parent):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    failures.append(error)
        if failures:
            if body_error is not None and hasattr(body_error, "add_note"):
                body_error.add_note(f"RECOVERY_REQUIRED: behavior lease cleanup failed: {failures!r}")
            elif body_error is None:
                raise RuntimeError("RECOVERY_REQUIRED: behavior stage leases did not close") from failures[0]
    return receipt


__all__ = ["Behavior_stage_receipt_v2", "run_behavior_stage_v2"]
