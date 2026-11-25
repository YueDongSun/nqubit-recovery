"""Fixed V2 behavior training after native publication and strict data admission.

This phase uses the unchanged oracle-history numerical core. It never invokes
or weakens the generic V2 training hold. Its returned checkpoint was reloaded
from the exact publisher-owned bytes, not an unverified pathname.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
from typing import Any, Callable

import torch

from ml_recon.data_v2_4.admission_snapshot import Canonical_json_snapshot
from ml_recon.data_v2_4.hashing import canonical_json_bytes, sha256_json

from . import inverse_transformer_pilot as pilot
from .behavior_artifact_owner import Behavior_artifact_owner
from .behavior_dataset_v2 import Behavior_dataset_result
from .behavior_v2_contract import Behavior_stage_plan_v2
from .contracts import require_sha256
from .inverse_transformer import TrackBInverseTransformerV1
from .pilot_dataset_admission import admit_pilot_dataset
from .pilot_history_dataset import Pilot_history_dataset
from .pilot_runtime_checks import require_unchanged_pilot_source
from .pilot_training_core import Pilot_training_dependencies, initialize_pilot_model, train_pilot_core
from .teacher_diagnostic_preflight import require_unchanged_teacher_diagnostic_preflight


@dataclass(frozen=True)
class Behavior_training_result:
    model: TrackBInverseTransformerV1
    receipt: Canonical_json_snapshot
    checkpoint_sha256: str


def _training_dependencies() -> Pilot_training_dependencies:
    return Pilot_training_dependencies(
        pilot._model_config_for_target_normalization, pilot.TrackBInverseTransformerV1,
        pilot._round_inputs, pilot._logical_action_loss_components, pilot._assign_training_gradients,
        pilot.checked_optimizer_step, pilot._evaluate_teacher_forced, pilot._round_coverage_batch_schedule,
        pilot.measure_inverse_transformer_dependency, pilot.RANDOM_TRAINING_SAMPLING,
        pilot.ROUND_COVERAGE_TRAINING_SAMPLING)


def train_behavior_model_v2(
    plan: Behavior_stage_plan_v2, owner: Behavior_artifact_owner, dataset: Behavior_dataset_result, *,
    expected_source: tuple[str, str], expected_plan_sha256: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> Behavior_training_result:
    """Train exactly 191 checked updates, then verify and publish new bytes."""
    if (type(plan) is not Behavior_stage_plan_v2 or type(owner) is not Behavior_artifact_owner
            or type(dataset) is not Behavior_dataset_result):
        raise ValueError("behavior training requires exact stage/owner/dataset types")
    plan.validate_identity()
    require_sha256(expected_plan_sha256, "behavior_stage_plan_sha")
    if (plan.expected_source != expected_source or plan.sha256() != expected_plan_sha256
            or owner.root.path != plan.output_root):
        raise ValueError("behavior training differs from independently pinned source/plan/owner")
    require_unchanged_pilot_source(expected_source)
    owner.verify()
    if "pilot" in owner.directories or plan.pilot_root.exists():
        raise ValueError("behavior pilot output must be fresh; no resume or reuse")
    recipe = plan.recipe
    pin = plan.teacher_evidence_pin

    def check_source() -> None:
        require_unchanged_teacher_diagnostic_preflight(
            pin.plan.preflight_bindings[0], sample_key=pin.plan.sample_keys[0],
            config=pin.plan.config, expected_source=expected_source)

    if owner.read("dataset/behavior_dataset_receipt.json") != dataset.receipt.payload:
        raise ValueError("behavior dataset receipt differs from the owned publication")
    source_receipt = dataset.receipt.decode()
    if (source_receipt["stage_plan_sha256"] != expected_plan_sha256
            or source_receipt["dataset_recipe_sha256"] != sha256_json(recipe.dataset_recipe())):
        raise ValueError("behavior dataset receipt belongs to a different stage or recipe")
    admission = admit_pilot_dataset(
        plan.dataset_root, cell_id=recipe.cell_id, history_policy=recipe.history_policy,
        history_policy_sha256=recipe.history_policy_sha256, strict_shard_audit=True)
    identity = admission.identity_dict()
    if (admission != dataset.admission or identity != source_receipt["dataset_identity"]
            or (identity["source_commit"], identity["source_tree"]) != expected_source):
        raise ValueError("behavior training dataset changed since its strict publication admission")
    config = pilot.TrackBV24TenRoundPilotConfigV1(
        output_root=str(plan.pilot_root), existing_v24_dataset_root=str(plan.dataset_root),
        **recipe.pilot_kwargs())
    config.validate()
    train = Pilot_history_dataset._from_admission(admission, split="train")
    validation = None
    body_error = None
    try:
        validation = Pilot_history_dataset._from_admission(admission, split="validation")
        if len(train) != recipe.train_samples or len(validation) != recipe.validation_samples:
            raise ValueError("behavior training row counts differ from the registered recipe")
        train_source = pilot._training_source_identity(config, train, identity)
        validation_source = pilot._training_source_identity(config, validation, identity)
        normalization = pilot._training_target_normalization(
            train, batch_size=config.batch_size,
            action_loss_normalization_kind=config.action_loss_normalization_kind,
            require_full_active_modes=True, source_view_kind=config.source_view_kind)
        check_source()
        owner.verify()
        dependencies = _training_dependencies()
        initialized = initialize_pilot_model(config, normalization, dependencies)
        check_source()
        owner.verify()

        def progress(record: dict[str, object]) -> None:
            check_source()
            owner.verify()
            if on_progress is not None:
                on_progress({"phase": "OPTIMIZER_UPDATE_COMPLETED", **record})

        trained = train_pilot_core(
            initialized, config, normalization, train, validation, dependencies, on_progress=progress)
        if trained.completed_updates != recipe.optimizer_steps or len(trained.round_schedule) != recipe.optimizer_steps:
            raise ValueError("behavior training did not complete exactly the fixed 191 updates")
        owner.verify()
        checkpoint_metadata = {
            "schema": "TrackBBehaviorCheckpointV2", "stage_plan_sha256": expected_plan_sha256,
            "source_commit": expected_source[0], "source_tree": expected_source[1],
            "history_policy": recipe.history_policy, "history_policy_sha256": recipe.history_policy_sha256,
            "source_view_kind": recipe.source_view_kind, "model_config": trained.model_config.to_dict(),
            "optimizer_steps": trained.completed_updates, "target_normalization": normalization.to_dict(),
            "normalization_source": "FRESH_CURRENT_DATASET_TRAIN_ONLY",
            "round_schedule_sha256": sha256_json(trained.round_schedule),
            "batch_schedule_sha256": sha256_json(trained.batch_schedule),
            "train_source": train_source, "validation_source": validation_source,
            "dataset_receipt_sha256": dataset.receipt.sha256, "recipe_sha256": recipe.sha256(),
            "research_only": True, "selection_evidence": False,
            "promotion_ready": False, "hardware_action_authorized": False}
        with io.BytesIO() as buffer:
            torch.save({**checkpoint_metadata, "model_state_dict": trained.model.state_dict()}, buffer)
            checkpoint_bytes = buffer.getvalue()
        owner.create_directory("pilot/model", before_create=check_source)
        checkpoint_binding = owner.publish("pilot/model/checkpoint.pt", checkpoint_bytes, before_publish=check_source)
        # weights_only avoids arbitrary pickle execution; the bytes remain under
        # the live native lease and are independently hash-checked by read().
        with io.BytesIO(owner.read("pilot/model/checkpoint.pt")) as buffer:
            checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
        if ({key: value for key, value in checkpoint.items() if key != "model_state_dict"} != checkpoint_metadata
                or set(checkpoint) != set(checkpoint_metadata) | {"model_state_dict"}):
            raise ValueError("published behavior checkpoint metadata failed exact roundtrip")
        reloaded = TrackBInverseTransformerV1(trained.model_config)
        reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
        expected_state = trained.model.state_dict()
        for name, value in reloaded.state_dict().items():
            if (value.dtype != expected_state[name].dtype or not torch.equal(value, expected_state[name])
                    or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
                raise ValueError("published behavior checkpoint tensors differ from the trained model")
        reloaded.eval()
        receipt_payload = canonical_json_bytes({
            "schema": "TrackBBehaviorPilotReceiptV2", "stage_plan_sha256": expected_plan_sha256,
            "source_commit": expected_source[0], "source_tree": expected_source[1],
            "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256(),
            "dataset_receipt_sha256": dataset.receipt.sha256, "dataset_identity": identity,
            "train_source": train_source, "validation_source": validation_source,
            "model": {"config": trained.model_config.to_dict(), "target_normalization": normalization.to_dict(),
                      "checkpoint": checkpoint_binding.to_dict(), "checkpoint_roundtrip_verified": True},
            "completed_optimizer_updates": trained.completed_updates, "training_progress": trained.progress,
            "round_schedule_sha256": sha256_json(trained.round_schedule),
            "batch_schedule_sha256": sha256_json(trained.batch_schedule),
            "teacher_forced_evaluation": trained.evaluation, "ten_round_guidance": trained.guidance,
            "inverse_transformer_dependency": {**asdict(trained.dependency),
                                               "dependency_sufficient": trained.dependency.dependency_sufficient},
            "model_self_rollout": "NOT_PERFORMED_IN_TRAINING_PHASE", "scientific_gate": "NOT_PERFORMED",
            "research_only": True, "selection_evidence": False,
            "promotion_ready": False, "hardware_action_authorized": False})
        receipt_binding = owner.publish("pilot/behavior_pilot_receipt.json", receipt_payload, before_publish=check_source)
        result = Behavior_training_result(
            reloaded, Canonical_json_snapshot(receipt_payload, receipt_binding.sha256), checkpoint_binding.sha256)
    except BaseException as error:
        body_error = error
        raise
    finally:
        failures = []
        for view in (validation, train):
            if view is not None:
                try:
                    view.close()
                except BaseException as error:
                    failures.append(error)
        if failures:
            if body_error is not None and hasattr(body_error, "add_note"):
                body_error.add_note(f"RECOVERY_REQUIRED: behavior data-view cleanup failed: {failures!r}")
            elif body_error is None:
                raise RuntimeError("RECOVERY_REQUIRED: behavior data views did not close") from failures[0]
    return result


__all__ = ["Behavior_training_result", "train_behavior_model_v2"]
