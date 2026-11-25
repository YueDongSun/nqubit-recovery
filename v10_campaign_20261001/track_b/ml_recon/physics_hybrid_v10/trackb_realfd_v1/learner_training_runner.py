"""Research-only CLI wrapper around the existing ten-round Track B pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ml_recon.data_v2_4.hashing import sha256_file

from .artifact_io import (
    PUBLICATION_COMMITTED,
    TrackBArtifactPublicationV1,
    publish_new_bytes_with_state,
)
from .artifact_directory_lease import (
    Directory_lease, close_directory_leases, open_directory_ancestry,
    verify_directory_leases,
)
from .artifact_file_lease import File_lease, close_file_leases
from .contracts import canonical_json_bytes, canonical_sha256, require_sha256
from .inverse_transformer_pilot import (
    TrackBV24TenRoundPilotConfigV1,
    run_trackb_v24_ten_round_pilot,
)
from .learner_runner_config import (
    LEARNER_RUNNER_V2_SCHEMA,
    Learner_training_run_config_v1,
    load_learner_training_config,
)
from .learner_recipe_contract import (
    MATCHED_BEHAVIOR_RECIPE,
    MATCHED_BEHAVIOR_RECIPE_KIND,
    resolve_matched_behavior_recipe,
    verify_matched_behavior_receipt_bytes,
)
from .learner_pilot_artifact_writer import Learner_pilot_artifact_writer
from .pilot_runtime_checks import pilot_source_identity


RUNNER_RECEIPT_SCHEMA = "TrackBLearnerTrainingRunReceiptV1"
LEARNER_PLAN_V2_SCHEMA = "TrackBLearnerTrainingPlanV2"
COMPLETION_EVIDENCE_SCHEMA = "TrackBLearnerCompletionEvidenceV1"
COMPLETION_EVIDENCE_NAME = "run_completion_evidence.json"
_max_json_receipt_bytes = 64 * 1024 * 1024


class Learner_receipt_publication_error(RuntimeError):
    """A receipt publication did not return clean committed evidence."""

    def __init__(self, status: str, path: Path, publication: Mapping[str, object]) -> None:
        self.status = status
        self.publication = dict(publication)
        super().__init__(f"{status}: receipt publication at {path}: {self.publication}")


def _bound_source(config: Learner_training_run_config_v1) -> tuple[str, str]:
    source = pilot_source_identity()
    if source != (config.expected_source_commit, config.expected_source_tree):
        raise ValueError("current clean source commit/tree differs from learner plan")
    return source


def build_learner_plan(config: Learner_training_run_config_v1) -> dict[str, object]:
    """Inspect pinned inputs and source without creating any output."""
    config.validate()
    _bound_source(config)
    plan = {
        "schema": "TrackBLearnerTrainingPlanV1",
        "run_id": config.run_id,
        "run_root": config.run_root,
        "expected_source_commit": config.expected_source_commit,
        "expected_source_tree": config.expected_source_tree,
        "source_view_kind": config.source_view_kind,
        "history_gain_input_kind": config.history_gain_input_kind,
        "train_samples": config.train_samples,
        "validation_samples": config.validation_samples,
        "history_rounds": config.history_rounds,
        "optimizer_steps": config.optimizer_steps,
        "device": config.device,
        "data_admission": "NOT_RUN",
        "training": "NOT_RUN",
        "gate_decision": "NOT_PERFORMED",
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    if config.schema == LEARNER_RUNNER_V2_SCHEMA:
        recipe = _learner_recipe(config)
        plan["schema"] = LEARNER_PLAN_V2_SCHEMA
        plan["recipe_identity"] = {
            "kind": recipe["identity"],
            "behavior_receipt_sha": config.behavior_pilot_receipt_sha,
        }
        plan["recipe_values"] = recipe["values"]
    return plan


def _learner_recipe(config: Learner_training_run_config_v1) -> dict[str, object]:
    if config.schema != LEARNER_RUNNER_V2_SCHEMA:
        return {"identity": "pilot_defaults_v1", "values": {}}
    values = resolve_matched_behavior_recipe(
        config.behavior_pilot_receipt_path, config.behavior_pilot_receipt_sha,
    )
    return {"identity": MATCHED_BEHAVIOR_RECIPE_KIND, "values": values}


def _publish_receipt(
    root: Path, name: str, payload: Mapping[str, object],
    root_lease: Directory_lease,
) -> TrackBArtifactPublicationV1:
    root_lease.verify()
    body = canonical_json_bytes(dict(payload))
    path = root / name
    try:
        publication = publish_new_bytes_with_state(
            path, body, expected_sha=hashlib.sha256(body).hexdigest(),
            parent_lease=root_lease,
        )
    except Exception as error:
        raise Learner_receipt_publication_error(
            "UNKNOWN_AFTER_EXCEPTION", path,
            {"target_path": str(path), "target_present_after_exception":
             path.exists() if _lease_is_current(root_lease) else "UNKNOWN",
             "cause_type": type(error).__name__, "cause": str(error)},
        ) from error
    if publication.status != PUBLICATION_COMMITTED or not publication.staging_cleanup_complete:
        raise Learner_receipt_publication_error(
            "RECOVERY_REQUIRED", path, publication.to_dict(),
        )
    if publication.file_identity is None:
        raise Learner_receipt_publication_error(
            "RECOVERY_REQUIRED", path,
            {**publication.to_dict(), "reason": "publisher omitted staged file identity"},
        )
    return publication


def _lease_is_current(lease: Directory_lease) -> bool:
    try:
        lease.verify()
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _directory_binding(anchor_lease: Directory_lease,
                       root_lease: Directory_lease,
                       pilot_lease: Directory_lease | None,
                       model_lease: Directory_lease | None) -> dict[str, object]:
    return {"schema": "TrackBLearnerDirectoryBindingV1",
            "output_parent": anchor_lease.binding(),
            "run_root": root_lease.binding(),
            "pilot": pilot_lease.binding() if pilot_lease is not None else None,
            "model": model_lease.binding() if model_lease is not None else None}


def _pilot_config(
    config: Learner_training_run_config_v1, *,
    recipe_values: Mapping[str, object] | None = None,
) -> TrackBV24TenRoundPilotConfigV1:
    if config.schema == LEARNER_RUNNER_V2_SCHEMA:
        if recipe_values is not None and type(recipe_values) is not dict:
            raise ValueError("learner pilot recipe values must be an exact object")
        recipe = (_learner_recipe(config)["values"]
                  if recipe_values is None else dict(recipe_values))
        if (set(recipe) != set(MATCHED_BEHAVIOR_RECIPE)
                or any(type(recipe[name]) is not type(expected)
                       or recipe[name] != expected
                       for name, expected in MATCHED_BEHAVIOR_RECIPE.items())):
            raise ValueError("learner pilot recipe differs from the admitted plan")
    else:
        if recipe_values is not None:
            raise ValueError("V1 learner pilot cannot accept recipe overrides")
        recipe = {}
    pilot = TrackBV24TenRoundPilotConfigV1(
        output_root=str(Path(config.run_root) / "pilot"),
        system_type=config.system_type,
        train_samples=config.train_samples,
        validation_samples=config.validation_samples,
        optimizer_steps=config.optimizer_steps,
        device=config.device,
        existing_v24_dataset_root=config.existing_v24_dataset_root,
        source_view_kind=config.source_view_kind,
        history_gain_input_kind=config.history_gain_input_kind,
        behavior_pilot_receipt_path=config.behavior_pilot_receipt_path,
        behavior_pilot_receipt_sha=config.behavior_pilot_receipt_sha,
        learner_visited_train_report_path=config.learner_visited_train_report_path,
        learner_visited_train_report_sha=config.learner_visited_train_report_sha,
        learner_visited_validation_report_path=config.learner_visited_validation_report_path,
        learner_visited_validation_report_sha=config.learner_visited_validation_report_sha,
        **recipe,
    )
    pilot.validate()
    return pilot


def _validate_pilot_result(
    config: Learner_training_run_config_v1, result: Mapping[str, object],
    artifact_writer: Learner_pilot_artifact_writer,
) -> dict[str, object]:
    receipt_path = Path(str(result.get("receipt_path", "")))
    if (receipt_path != Path(config.run_root) / "pilot" / "pilot_receipt.json"
            or artifact_writer.receipt_lease is None
            or artifact_writer.receipt_lease.path != receipt_path
            or artifact_writer.receipt_lease.expected_sha != result.get("receipt_sha")):
        raise ValueError("pilot result receipt is missing or differs")
    receipt_bytes = artifact_writer.receipt_lease.read_bytes(
        max_bytes=_max_json_receipt_bytes)
    payload = json.loads(receipt_bytes.decode("utf-8"))
    declared = payload.get("config")
    model = payload.get("model")
    progress = payload.get("training_progress")
    pinned_fields = (
        "existing_v24_dataset_root", "behavior_pilot_receipt_path",
        "behavior_pilot_receipt_sha", "learner_visited_train_report_path",
        "learner_visited_train_report_sha", "learner_visited_validation_report_path",
        "learner_visited_validation_report_sha", "train_samples",
        "validation_samples", "system_type", "device",
    )
    if (payload.get("source_commit") != config.expected_source_commit
            or payload.get("source_tree") != config.expected_source_tree
            or payload.get("source_view_kind") != config.source_view_kind
            or payload.get("research_only") is not True
            or payload.get("selection_evidence") is not False
            or payload.get("promotion_ready") is not False
            or payload.get("hardware_action_authorized") is not False
            or payload.get("external_waveform_write") != "NOT_IMPLEMENTED"
            or not isinstance(declared, dict)
            or declared.get("optimizer_steps") != config.optimizer_steps
            or declared.get("source_view_kind") != config.source_view_kind
            or declared.get("history_gain_input_kind") != config.history_gain_input_kind
            or any(declared.get(name) != getattr(config, name) for name in pinned_fields)
            or not isinstance(progress, list) or not progress
            or not isinstance(progress[-1], dict)
            or progress[-1].get("optimizer_step") != config.optimizer_steps
            or not isinstance(model, dict)):
        raise ValueError("pilot result is not the bound research-only 191-update learner")
    checkpoint_path = Path(str(model.get("checkpoint_path", "")))
    if (checkpoint_path != Path(config.run_root) / "pilot" / "model" / "checkpoint.pt"
            or artifact_writer.checkpoint_lease is None
            or artifact_writer.checkpoint_lease.path != checkpoint_path
            or artifact_writer.checkpoint_lease.expected_sha != model.get("checkpoint_sha")):
        raise ValueError("pilot model checkpoint is missing or differs")
    artifact_writer.checkpoint_lease.verify_binding()
    return {"pilot_receipt_path": str(receipt_path),
            "pilot_receipt_sha": str(result["receipt_sha"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha": str(model["checkpoint_sha"])}


def _input_identity(config: Learner_training_run_config_v1) -> dict[str, object]:
    return {
        "existing_v24_dataset_root": config.existing_v24_dataset_root,
        "behavior_pilot_receipt_path": config.behavior_pilot_receipt_path,
        "behavior_pilot_receipt_sha": config.behavior_pilot_receipt_sha,
        "learner_visited_train_report_path": config.learner_visited_train_report_path,
        "learner_visited_train_report_sha": config.learner_visited_train_report_sha,
        "learner_visited_validation_report_path": config.learner_visited_validation_report_path,
        "learner_visited_validation_report_sha": config.learner_visited_validation_report_sha,
    }


def _require_current_input_source_identity(config: Learner_training_run_config_v1) -> None:
    _bound_source(config)
    for stem in (
        "behavior_pilot_receipt", "learner_visited_train_report",
        "learner_visited_validation_report",
    ):
        path = Path(getattr(config, f"{stem}_path"))
        expected_sha = getattr(config, f"{stem}_sha")
        if stem == "behavior_pilot_receipt" and config.schema == LEARNER_RUNNER_V2_SCHEMA:
            verify_matched_behavior_receipt_bytes(str(path), expected_sha)
        elif sha256_file(path) != expected_sha:
            raise ValueError(f"{stem} changed during learner training")


def _completion_evidence(
    config: Learner_training_run_config_v1,
    publication: TrackBArtifactPublicationV1,
    directory_binding: Mapping[str, object],
    artifact_bindings: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": COMPLETION_EVIDENCE_SCHEMA,
        "run_id": config.run_id,
        "run_root": config.run_root,
        "config_sha": canonical_sha256(config.to_dict()),
        "expected_source_commit": config.expected_source_commit,
        "expected_source_tree": config.expected_source_tree,
        "inputs": _input_identity(config),
        "terminal_path": str(Path(config.run_root) / "run_terminal.json"),
        "terminal_sha": publication.artifact_sha256,
        "terminal_publication": publication.to_dict(),
        "directory_binding": dict(directory_binding),
        "artifact_bindings": dict(artifact_bindings),
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "gate_decision": "NOT_PERFORMED",
    }


def _read_canonical_json_object(lease: File_lease) -> dict[str, object]:
    body = lease.read_bytes(max_bytes=_max_json_receipt_bytes)
    payload = json.loads(body.decode("utf-8"))
    if type(payload) is not dict or canonical_json_bytes(payload) != body:
        raise ValueError(f"{lease.path.name} is not a canonical JSON object")
    return payload


def _require_inspected_success_terminal(
    config: Learner_training_run_config_v1,
    terminal: Mapping[str, object],
    directory_binding: Mapping[str, object],
    artifact_leases: Mapping[str, File_lease],
) -> None:
    base = {
        "schema": RUNNER_RECEIPT_SCHEMA, "run_id": config.run_id,
        "expected_source_commit": config.expected_source_commit,
        "expected_source_tree": config.expected_source_tree,
        "research_only": True, "selection_evidence": False,
        "promotion_ready": False, "hardware_action_authorized": False,
        "gate_decision": "NOT_PERFORMED",
        "execution_outcome": "TRAINING_SUCCEEDED",
        "data_admission": "VERIFIED", "pilot_invoked": True,
        "optimizer_steps": config.optimizer_steps,
        "directory_binding": dict(directory_binding),
    }
    artifact_names = ("pilot_receipt", "checkpoint")
    if (set(terminal) != set(base) | {f"{name}_{field}" for name in artifact_names
                                    for field in ("path", "sha")}
            or any(terminal.get(name) != expected for name, expected in base.items())):
        raise ValueError("terminal is not the exact research-only successful learner receipt")
    expected_paths = {
        "pilot_receipt": Path(config.run_root) / "pilot" / "pilot_receipt.json",
        "checkpoint": Path(config.run_root) / "pilot" / "model" / "checkpoint.pt",
    }
    for name, path in expected_paths.items():
        file_lease = artifact_leases[name]
        if (terminal.get(f"{name}_path") != str(path)
                or file_lease.path != path
                or file_lease.expected_sha != require_sha256(
                    terminal.get(f"{name}_sha"), f"{name}_sha")):
            raise ValueError(f"terminal {name} artifact identity differs")
        file_lease.verify_binding()


def inspect_learner_completion(config: Learner_training_run_config_v1) -> dict[str, object]:
    """Read only: verify the durable witness of a clean terminal publication."""
    config.validate(inspection=True)
    root = Path(config.run_root)
    marker_path = root / COMPLETION_EVIDENCE_NAME
    base = {
        "schema": "TrackBLearnerCompletionInspectionV1",
        "run_id": config.run_id,
        "training_completion": "COMPLETION_UNPROVEN",
        "terminal_publication": "UNPROVEN",
        "marker_publication": "UNKNOWN_AFTER_RESTART",
        "directory_cleanup": "NOT_PROVEN",
        "process_exit": "NOT_PROVEN",
        "gate_decision": "NOT_PERFORMED",
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    if not marker_path.exists() and not marker_path.is_symlink():
        return base
    leases: list[Directory_lease] = []
    file_leases: list[File_lease] = []
    try:
        leases = open_directory_ancestry(root / "pilot" / "model", anchor=root.parent)
        directory_binding = _directory_binding(leases[-4], leases[-3], leases[-2], leases[-1])
        marker_lease = File_lease.open_verified(
            marker_path, sha256_file(marker_path), leases[-3])
        file_leases.append(marker_lease)
        marker = _read_canonical_json_object(marker_lease)
        terminal_path = root / "run_terminal.json"
        terminal_lease = File_lease.open_verified(
            terminal_path, require_sha256(marker.get("terminal_sha"), "terminal_sha"),
            leases[-3])
        file_leases.append(terminal_lease)
        terminal = _read_canonical_json_object(terminal_lease)
        publication_payload = marker.get("terminal_publication")
        if type(publication_payload) is not dict or set(publication_payload) != {
            "schema", "target_path", "artifact_sha256", "status", "target_published",
            "staging_cleanup_complete", "staging_path", "cleanup_error",
        }:
            raise ValueError("marker lacks strict terminal publication metadata")
        publication = TrackBArtifactPublicationV1(**{
            name: value for name, value in publication_payload.items() if name != "schema"
        })
        if (publication_payload["schema"] != publication.schema
                or publication.status != PUBLICATION_COMMITTED
                or publication.target_path != str(terminal_path)
                or publication.target_published is not True
                or publication.staging_cleanup_complete is not True
                or publication.cleanup_error is not None
                or type(publication.staging_path) is not str
                or Path(publication.staging_path).parent != root
                or not Path(publication.staging_path).name.startswith(
                    ".run_terminal.json.staging-"
                )):
            raise ValueError("marker does not record clean terminal publication")
        artifact_leases = {"terminal": terminal_lease}
        for name, parent_lease, path in (
            ("pilot_receipt", leases[-2], root / "pilot" / "pilot_receipt.json"),
            ("checkpoint", leases[-1], root / "pilot" / "model" / "checkpoint.pt"),
        ):
            artifact_lease = File_lease.open_verified(
                path, require_sha256(terminal.get(f"{name}_sha"), f"{name}_sha"),
                parent_lease)
            file_leases.append(artifact_lease)
            artifact_leases[name] = artifact_lease
        artifact_bindings = {name: artifact_leases[name].binding()
                             for name in ("pilot_receipt", "checkpoint", "terminal")}
        expected = _completion_evidence(config, publication, directory_binding,
                                        artifact_bindings)
        if set(marker) != set(expected) or marker != expected:
            raise ValueError("marker schema or run/config/source/input binding differs")
        if terminal_lease.expected_sha != marker["terminal_sha"]:
            raise ValueError("terminal bytes differ from the completion evidence")
        _require_inspected_success_terminal(config, terminal, directory_binding,
                                            artifact_leases)
        _bound_source(config)
        if not Path(config.existing_v24_dataset_root).is_dir():
            raise ValueError("V2.4 source root is missing")
        for stem in (
            "behavior_pilot_receipt", "learner_visited_train_report",
            "learner_visited_validation_report",
        ):
            path = Path(getattr(config, f"{stem}_path"))
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{stem} input differs from its pinned identity")
            expected_sha = getattr(config, f"{stem}_sha")
            if stem == "behavior_pilot_receipt" and config.schema == LEARNER_RUNNER_V2_SCHEMA:
                verify_matched_behavior_receipt_bytes(str(path), expected_sha)
            elif sha256_file(path) != expected_sha:
                raise ValueError(f"{stem} input differs from its pinned identity")
        verify_directory_leases(leases)
        for lease in file_leases:
            lease.verify_binding()
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError,
            RuntimeError, KeyError, IndexError) as error:
        inspection = {**base, "training_completion": "INVALID_COMPLETION_EVIDENCE",
                      "reason": str(error)}
    else:
        inspection = {**base, "training_completion": "VERIFIED",
                      "terminal_publication": "CLEAN_CONFIRMED",
                      "terminal_sha": str(marker["terminal_sha"])}
    finally:
        if file_leases:
            try:
                close_file_leases(file_leases)
            except RuntimeError as error:
                inspection = {**base, "training_completion": "INVALID_COMPLETION_EVIDENCE",
                              "reason": str(error)}
        if leases:
            try:
                close_directory_leases(leases)
            except RuntimeError as error:
                inspection = {**base, "training_completion": "INVALID_COMPLETION_EVIDENCE",
                              "reason": str(error)}
    return inspection


def _run_learner_training_leased(
    config: Learner_training_run_config_v1, plan: Mapping[str, object],
    leases: list[Directory_lease],
    file_leases: list[File_lease],
) -> dict[str, object]:
    root = Path(config.run_root)
    root_lease = leases[-1]
    pilot_lease: Directory_lease | None = None
    artifact_writer: Learner_pilot_artifact_writer | None = None
    base = {
        "schema": RUNNER_RECEIPT_SCHEMA, "run_id": config.run_id,
        "expected_source_commit": config.expected_source_commit,
        "expected_source_tree": config.expected_source_tree,
        "research_only": True, "selection_evidence": False,
        "promotion_ready": False, "hardware_action_authorized": False,
        "gate_decision": "NOT_PERFORMED",
    }
    admission_verified = False
    pilot_invoked = False
    execution_outcome = "FAILED"
    execution_error: BaseException | None = None
    execution_traceback = None
    accepted: dict[str, object] = {}

    def on_data_admitted(
        dataset_result: Mapping[str, object],
        train_source: Mapping[str, object],
        validation_source: Mapping[str, object],
    ) -> None:
        nonlocal admission_verified
        if admission_verified:
            raise RuntimeError("data admission callback repeated")
        # The callback runs only after both dataset views and source bindings
        # passed.  Record that fact before publishing, whose post-link state
        # may itself be uncertain and must stop the optimizer.
        admission_verified = True
        verify_directory_leases(leases)
        _publish_receipt(root, "data_admitted.json", {
            **base, "status": "INCOMPLETE", "data_admission": "ADMITTED",
            "training": "NOT_STARTED", "dataset": dict(dataset_result),
            "train_source": dict(train_source),
            "validation_source": dict(validation_source),
        }, root_lease)
    try:
        verify_directory_leases(leases)
        _publish_receipt(root, "run_started.json", {
            **base, "status": "INCOMPLETE", "data_admission": "NOT_RUN",
            "training": "NOT_RUN", "plan": plan,
        }, root_lease)
        pilot_lease = Directory_lease.create_child(root_lease, "pilot")
        leases.append(pilot_lease)
        artifact_writer = Learner_pilot_artifact_writer(root_lease, pilot_lease,
                                                        leases, file_leases)
        pilot_config = _pilot_config(config, recipe_values=plan.get("recipe_values"))
        pilot_invoked = True
        result = run_trackb_v24_ten_round_pilot(
            pilot_config, on_data_admitted=on_data_admitted,
            artifact_writer=artifact_writer,
        )
        if not admission_verified:
            raise RuntimeError("pilot returned without verified data admission")
        artifact_writer.verify()
        accepted = _validate_pilot_result(config, result, artifact_writer)
        artifact_writer.admit_model_directory()
        verify_directory_leases(leases)
        # Repeat source and input bindings after all pilot/model checks, just
        # before the sole terminal publication decision.
        _require_current_input_source_identity(config)
        execution_outcome = "TRAINING_SUCCEEDED"
    except KeyboardInterrupt as error:
        execution_outcome = "CANCELLED"
        execution_error = error
        execution_traceback = error.__traceback__
    except Exception as error:
        execution_error = error
        execution_traceback = error.__traceback__

    terminal = {
        **base,
        "directory_binding": _directory_binding(
            leases[0], root_lease, pilot_lease,
            artifact_writer.model_lease if artifact_writer is not None else None),
        "execution_outcome": execution_outcome,
        "data_admission": "VERIFIED" if admission_verified else "NOT_VERIFIED",
        "pilot_invoked": pilot_invoked,
        **({"optimizer_steps": config.optimizer_steps, **accepted}
           if execution_outcome == "TRAINING_SUCCEEDED" else {}),
        **({"error_type": type(execution_error).__name__, "error": str(execution_error)}
           if execution_error is not None else {}),
    }
    try:
        verify_directory_leases(leases)
        publication = _publish_receipt(root, "run_terminal.json", terminal, root_lease)
    except Exception as publication_error:
        if execution_error is not None:
            if hasattr(execution_error, "add_note"):
                execution_error.add_note(f"terminal receipt publication: {publication_error}")
            raise execution_error.with_traceback(execution_traceback) from publication_error
        raise
    try:
        terminal_lease = File_lease.open_verified(
            root / "run_terminal.json", publication.artifact_sha256, root_lease,
            expected_identity=publication.file_identity)
    except Exception as lease_error:
        if execution_error is not None:
            if hasattr(execution_error, "add_note"):
                execution_error.add_note(f"terminal file lease: {lease_error}")
            raise execution_error.with_traceback(execution_traceback) from lease_error
        raise
    file_leases.append(terminal_lease)
    if execution_error is not None:
        raise execution_error.with_traceback(execution_traceback)
    verify_directory_leases(leases)
    if artifact_writer is None or artifact_writer.receipt_lease is None \
            or artifact_writer.checkpoint_lease is None:
        raise ValueError("successful pilot artifacts lack retained file leases")
    artifact_writer.verify()
    terminal_lease.verify_binding()
    _require_current_input_source_identity(config)
    artifact_bindings = {
        "pilot_receipt": artifact_writer.receipt_lease.binding(),
        "checkpoint": artifact_writer.checkpoint_lease.binding(),
        "terminal": terminal_lease.binding(),
    }
    marker_publication = _publish_receipt(
        root, COMPLETION_EVIDENCE_NAME,
        _completion_evidence(config, publication, terminal["directory_binding"],
                             artifact_bindings), root_lease,
    )
    marker_lease = File_lease.open_verified(root / COMPLETION_EVIDENCE_NAME,
                                            marker_publication.artifact_sha256, root_lease,
                                            expected_identity=marker_publication.file_identity)
    file_leases.append(marker_lease)
    for lease in file_leases:
        lease.verify_binding()
    return {**terminal, "receipt_publication": publication.to_dict(),
            "completion_evidence_publication": marker_publication.to_dict()}


def run_learner_training(config: Learner_training_run_config_v1) -> dict[str, object]:
    """Claim a fresh run and hold its directory ancestry until final publication."""
    plan = build_learner_plan(config)
    root = Path(config.run_root)
    leases = open_directory_ancestry(root.parent, anchor=root.parent)
    file_leases: list[File_lease] = []
    try:
        leases.append(Directory_lease.create_child(leases[-1], root.name))
        verify_directory_leases(leases)
        result = _run_learner_training_leased(config, plan, leases, file_leases)
    except BaseException as error:
        try:
            close_file_leases(file_leases)
        except RuntimeError as close_error:
            if hasattr(error, "add_note"):
                error.add_note(str(close_error))
        try:
            close_directory_leases(leases)
        except RuntimeError as close_error:
            if hasattr(error, "add_note"):
                error.add_note(str(close_error))
        raise
    close_errors: list[RuntimeError] = []
    for close_operation in (lambda: close_file_leases(file_leases),
                            lambda: close_directory_leases(leases)):
        try:
            close_operation()
        except RuntimeError as close_error:
            close_errors.append(close_error)
    if close_errors:
        raise RuntimeError(f"RECOVERY_REQUIRED: artifact lease close failed: {close_errors}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "inspect"))
    parser.add_argument("--config", required=True, help="absolute path to strict versioned JSON")
    arguments = parser.parse_args(argv)
    config = load_learner_training_config(
        arguments.config, inspection=arguments.command == "inspect",
    )
    if arguments.command == "plan":
        result = build_learner_plan(config)
    elif arguments.command == "run":
        result = run_learner_training(config)
    else:
        result = inspect_learner_completion(config)
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    if arguments.command == "inspect" and result["training_completion"] != "VERIFIED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
