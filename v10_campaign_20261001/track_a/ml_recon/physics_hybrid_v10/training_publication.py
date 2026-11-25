"""Producer-owned checkpoint validation and committed completion publication.

Only freshly serialized, producer-owned staging files are deserialized here.
Observers consume the resulting hashes/attestation and never load remote pickle.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .checkpoint_schema import validate_checkpoint_envelope
from .checkpoint_storage import (Checkpoint_writer_lease, cleanup_checkpoint_commit_artifacts, copy_rooted_artifact,
                                  json_identity, publish_checkpoint_commit)
from .training_telemetry import AUTHORITY, append_training_record, atomic_training_json
from .training_telemetry import validate_committed_metrics


PUBLICATION_FILES = ("checkpoint_latest.pt", "checkpoint_audit.pt", "config_snapshot.json", "checkpoint_validation.json",
                     "training_summary.json", "training_completion.json")
REQUIRED_RUNTIME_STATES = ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                           "rng_state", "trainer_state", "model_build_spec")
EXECUTION_EVIDENCE_SCHEMA = "V10TrainingExecutionEvidenceV1"
EXECUTION_EVIDENCE_KEYS = frozenset({
    "schema", "run_id", "execution_id", "producer_pid", "telemetry_started_at",
    "initial_global_step", "final_global_step", "optimizer_step_delta",
    "optimizer_steps_this_execution", "finite_gradient_audit_passed",
    "gradient_audit_summary", "gradient_audit_summary_sha256",
    "output_audit_summary", "output_audit_summary_sha256", "source_identity",
    "registered_cells", "authority",
})


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_training_execution_evidence(*, run_id: str, execution_id: str,
                                      producer_pid: int, telemetry_started_at: str,
                                      initial_global_step: int, final_global_step: int,
                                      optimizer_steps_this_execution: int,
                                      finite_gradient_audit_passed: bool,
                                      gradient_audit_summary: Mapping[str, Any],
                                      output_audit_summary: Mapping[str, Any],
                                      source_identity: Mapping[str, Any],
                                      registered_cells: list[str]) -> dict[str, Any]:
    """Build and validate immutable producer evidence for a completed stream pilot."""
    evidence = {
        "schema": EXECUTION_EVIDENCE_SCHEMA,
        "run_id": run_id,
        "execution_id": execution_id,
        "producer_pid": producer_pid,
        "telemetry_started_at": telemetry_started_at,
        "initial_global_step": initial_global_step,
        "final_global_step": final_global_step,
        "optimizer_step_delta": final_global_step - initial_global_step,
        "optimizer_steps_this_execution": optimizer_steps_this_execution,
        "finite_gradient_audit_passed": finite_gradient_audit_passed,
        "gradient_audit_summary": dict(gradient_audit_summary),
        "gradient_audit_summary_sha256": _canonical_digest(gradient_audit_summary),
        "output_audit_summary": dict(output_audit_summary),
        "output_audit_summary_sha256": _canonical_digest(output_audit_summary),
        "source_identity": dict(source_identity),
        "registered_cells": sorted(registered_cells),
        "authority": dict(AUTHORITY),
    }
    validate_training_execution_evidence(evidence)
    return evidence


def validate_training_execution_evidence(evidence: Mapping[str, Any], *,
                                        expected_run_id: str | None = None,
                                        expected_execution_id: str | None = None,
                                        expected_producer_pid: int | None = None,
                                        expected_source_identity: Mapping[str, Any] | None = None) -> None:
    """Fail closed on incomplete or internally inconsistent producer evidence."""
    if evidence.get("schema") != EXECUTION_EVIDENCE_SCHEMA:
        raise ValueError("execution evidence schema mismatch")
    if set(evidence) != EXECUTION_EVIDENCE_KEYS:
        raise ValueError("execution evidence top-level schema is not exact")
    if (not isinstance(evidence.get("run_id"), str) or not evidence["run_id"]
            or not isinstance(evidence.get("execution_id"), str) or not evidence["execution_id"]
            or type(evidence.get("producer_pid")) is not int or evidence["producer_pid"] <= 0
            or not isinstance(evidence.get("telemetry_started_at"), str)
            or not evidence["telemetry_started_at"]):
        raise ValueError("execution evidence producer identity is invalid")
    try:
        started_at = datetime.fromisoformat(evidence["telemetry_started_at"])
    except ValueError as exc:
        raise ValueError("execution evidence telemetry start time is invalid") from exc
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("execution evidence telemetry start time lacks timezone")
    if (expected_run_id is not None and evidence["run_id"] != expected_run_id
            or expected_execution_id is not None and evidence["execution_id"] != expected_execution_id
            or expected_producer_pid is not None and evidence["producer_pid"] != expected_producer_pid):
        raise ValueError("execution evidence producer identity mismatch")
    if (expected_source_identity is not None
            and evidence["source_identity"] != dict(expected_source_identity)):
        raise ValueError("execution evidence source identity mismatch")
    initial = evidence.get("initial_global_step")
    final = evidence.get("final_global_step")
    delta = evidence.get("optimizer_step_delta")
    steps = evidence.get("optimizer_steps_this_execution")
    if any(type(value) is not int or value < 0 for value in (initial, final, delta)):
        raise ValueError("execution evidence optimizer steps are invalid")
    if (final <= initial or delta != final - initial or delta <= 0
            or type(steps) is not int or steps != delta):
        raise ValueError("execution evidence optimizer step delta is inconsistent")
    if evidence.get("finite_gradient_audit_passed") is not True:
        raise ValueError("execution evidence gradient audit is not finite")
    gradient = evidence.get("gradient_audit_summary")
    output = evidence.get("output_audit_summary")
    if not isinstance(gradient, dict) or not isinstance(output, dict):
        raise ValueError("execution evidence audit summaries are missing")
    if evidence.get("gradient_audit_summary_sha256") != _canonical_digest(gradient):
        raise ValueError("execution evidence gradient audit identity mismatch")
    if evidence.get("output_audit_summary_sha256") != _canonical_digest(output):
        raise ValueError("execution evidence output audit identity mismatch")
    if (type(gradient.get("audited_steps")) is not int
            or gradient["audited_steps"] < 1
            or gradient.get("audited_step_semantics") != "PRE_BACKWARD_ATTEMPT"
            or gradient.get("summary_scope") != "CURRENT_OBSERVER_INSTANCE"
            or gradient.get("run_id") != evidence["run_id"]
            or gradient.get("execution_id") != evidence["execution_id"]
            or gradient.get("source_identity") != evidence["source_identity"]):
        raise ValueError("execution evidence requires a gradient audit step")
    cells = evidence.get("registered_cells")
    if (output.get("schema") != "V10ClassifierOutputSliceAuditV1"
            or output.get("run_id") != evidence["run_id"]
            or output.get("execution_id") != evidence["execution_id"]
            or output.get("delta_reference") != "CURRENT_EXECUTION_INITIAL_STATE"
            or type(output.get("audited_steps")) is not int
            or output["audited_steps"] < 1
            or output.get("audit_complete") is not True
            or output.get("final_committed_step_observed") is not True
            or type(output.get("final_committed_step")) is not int
            or output.get("final_committed_step") != final
            or not isinstance(cells, list) or not cells
            or any(type(cell) is not str or not cell for cell in cells)
            or len(cells) != len(set(cells))
            or not isinstance(output.get("registered_cells"), list)
            or not isinstance(output.get("audited_cells"), list)
            or any(type(cell) is not str or not cell for cell in output["registered_cells"])
            or any(type(cell) is not str or not cell for cell in output["audited_cells"])
            or len(output["registered_cells"]) != len(set(output["registered_cells"]))
            or len(output["audited_cells"]) != len(set(output["audited_cells"]))
            or sorted(output["registered_cells"]) != sorted(cells)
            or sorted(output["audited_cells"]) != sorted(cells)
            or output.get("source_identity") != evidence["source_identity"]):
        raise ValueError("execution evidence registered-cell output audit is incomplete")
    authority = evidence.get("authority")
    if not isinstance(authority, dict) or authority != dict(AUTHORITY):
        raise ValueError("execution evidence authority flags must remain false")


def training_execution_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    """Return the canonical identity of the validated evidence payload."""
    validate_training_execution_evidence(evidence)
    return _canonical_digest(evidence)


def validate_execution_evidence_summary_binding(evidence: Mapping[str, Any],
                                                summary: Mapping[str, Any]) -> None:
    """Require the immutable audit payload to equal the producer summary facts."""
    if (evidence.get("gradient_audit_summary")
            != summary.get("observer_V10GradientAuditObserver")
            or evidence.get("output_audit_summary")
            != summary.get("observer_Classifier_output_audit")):
        raise ValueError("execution evidence audit summary diverges from training summary")


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def validate_owned_snapshot(path: Path, expected: Mapping[str, Any],
                            compute_state_sha: Callable[[Mapping[str, torch.Tensor]], str]) -> dict[str, object]:
    """Validate this producer's new staging output, not an arbitrary input file."""
    if not zipfile.is_zipfile(path):
        raise ValueError("saved checkpoint is not the required PyTorch ZIP snapshot")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict) or any(key not in loaded for key in REQUIRED_RUNTIME_STATES):
        raise ValueError("saved checkpoint lacks required runtime states")
    validate_checkpoint_envelope(loaded.get("metadata", {}))
    if loaded["metadata"] != expected["metadata"]:
        raise ValueError("saved checkpoint metadata changed during serialization")
    state = loaded.get("model_state_dict")
    expected_state = expected["model_state_dict"]
    if not isinstance(state, dict) or not state or set(state) != set(expected_state):
        raise ValueError("saved checkpoint model state keys differ")
    for name, tensor in state.items():
        reference = expected_state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != reference.shape
                or tensor.dtype != reference.dtype or not bool(torch.isfinite(tensor).all())):
            raise ValueError("saved checkpoint model tensor shape, dtype or finiteness differs")
    model_sha = compute_state_sha(state)
    if model_sha != expected["metadata"]["model_state_sha"]:
        raise ValueError("saved checkpoint model-state SHA mismatch")
    if loaded["model_build_spec"] != expected["model_build_spec"]:
        raise ValueError("saved checkpoint model construction specification changed")
    specification = json.dumps(loaded["model_build_spec"], sort_keys=True, ensure_ascii=True,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"checkpoint_format": "PYTORCH_ZIP", "model_state_sha": model_sha,
            "state_dict_tensor_count": len(state), "model_build_spec_sha256": hashlib.sha256(specification).hexdigest(),
            "checks": {"metadata_identity": True, "state_keys_shapes_dtypes": True,
                       "model_state_finite": True, "model_state_content_sha": True,
                       "runtime_state_fields_present": True, "model_build_spec": True}}


def publish_training_completion(*, output_dir: Path, checkpoint: dict[str, Any],
                                config: dict[str, Any], source_identity: dict[str, Any],
                                summary: dict[str, Any], execution_id: str,
                                compute_state_sha: Callable[[Mapping[str, torch.Tensor]], str],
                                checkpoint_commit: dict[str, Any] | None = None,
                                writer_lease: Checkpoint_writer_lease | None = None) -> dict[str, Any]:
    """Stage all outputs and publish a hash-bound completion receipt last."""
    if checkpoint["metadata"]["model_state_sha"] != summary["checkpoint_model_state_sha"]:
        raise RuntimeError("observer finalization changed the verified model state")
    if checkpoint_commit is None:
        raise RuntimeError("completion requires an explicit staged checkpoint descriptor")
    commit = checkpoint_commit
    if commit.get("execution_id") != execution_id:
        raise RuntimeError("completion requires the committed checkpoint descriptor")
    if writer_lease is None:
        raise RuntimeError("completion publication requires the active writer lease")
    writer_lease.assert_active()
    writer_lease.assert_rooted_path("training_completion.json", allow_missing=True)
    stream_publication = source_identity.get("input_delivery") == "stream_only"
    execution_evidence = summary.get("execution_evidence") if stream_publication else None
    if stream_publication and config.get("run_mode") == "pilot" and execution_evidence is None:
        raise RuntimeError("completed stream pilot requires immutable execution evidence")
    if execution_evidence is not None:
        validate_training_execution_evidence(
            execution_evidence, expected_run_id=config.get("run_id") or "",
            expected_execution_id=execution_id, expected_producer_pid=os.getpid(),
            expected_source_identity=source_identity)
        if (execution_evidence.get("run_id") != (config.get("run_id") or "")
                or execution_evidence.get("execution_id") != execution_id
                or execution_evidence.get("producer_pid") != os.getpid()
                or execution_evidence.get("final_global_step") != summary.get("global_step")
                or execution_evidence.get("source_identity") != source_identity):
            raise ValueError("execution evidence completion identity mismatch")
        validate_execution_evidence_summary_binding(execution_evidence, summary)
        execution_evidence_sha = training_execution_evidence_sha256(execution_evidence)
    else:
        execution_evidence_sha = None
    stream_recipe_path = output_dir / "stream_recipe.json"
    stream_recipe_identity = None
    if stream_publication:
        writer_lease.assert_rooted_path("stream_recipe.json")
        expected_recipe_sha = source_identity.get("stream_recipe_file_sha")
        if (not stream_recipe_path.is_file() or stream_recipe_path.is_symlink()
                or not isinstance(expected_recipe_sha, str)):
            raise ValueError("stream completion requires producer-owned stream_recipe.json")
        stream_recipe_identity = file_identity(stream_recipe_path)
        if stream_recipe_identity["sha256"] != expected_recipe_sha:
            raise ValueError("stream recipe file SHA mismatch before publication")
    metadata = checkpoint.get("metadata", {})
    expected_pairs = {
        "execution_id": execution_id,
        "run_id": config.get("run_id") or None,
        "model_id": config.get("model_id"),
        "global_step": summary.get("global_step"),
        "training_complete": True,
    }
    if any(metadata.get(key) != value for key, value in expected_pairs.items()):
        raise RuntimeError("completion checkpoint metadata identity mismatch")
    if metadata.get("source_identity") != source_identity:
        raise RuntimeError("completion checkpoint source identity mismatch")
    if metadata.get("config_sha") != source_identity.get("config_sha"):
        raise RuntimeError("completion checkpoint config identity mismatch")
    if summary.get("checkpoint_commit") != commit:
        raise RuntimeError("completion summary commit identity mismatch")
    if commit.get("source_identity") != source_identity:
        raise RuntimeError("completion descriptor source identity mismatch")
    if (commit.get("global_step") != summary.get("global_step")
            or commit.get("model_state_sha") != checkpoint["metadata"]["model_state_sha"]
            or commit.get("training_complete") is not True):
        raise RuntimeError("completion checkpoint commit identity mismatch")
    metrics_path = output_dir / commit["metrics_journal"]["path"]
    if not metrics_path.is_file() or summary.get("metrics_journal") != commit.get("metrics_journal"):
        raise RuntimeError("completion requires the immutable generation metrics journal")
    try:
        validate_committed_metrics(output_dir, execution_id=execution_id, final_commit=commit)
    except (OSError, ValueError) as exc:
        raise RuntimeError("completion metrics journal is not fully bound to committed generations") from exc
    with tempfile.TemporaryDirectory(prefix="completion_stage_", dir=output_dir,
                                     ignore_cleanup_errors=True) as temporary:
        staging = Path(temporary).resolve()
        if staging.parent != output_dir.resolve():
            raise ValueError("completion staging escaped the producer output root")
        staged_checkpoint = staging / "checkpoint_latest.pt"
        with staged_checkpoint.open("wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        validation = validate_owned_snapshot(staged_checkpoint, checkpoint, compute_state_sha)
        audit_projection = {
            "schema": "V10CheckpointAuditProjectionV1",
            "metadata": checkpoint["metadata"],
            "model_build_spec": checkpoint["model_build_spec"],
            "model_state_dict": checkpoint["model_state_dict"],
        }
        staged_audit_projection = staging / "checkpoint_audit.pt"
        with staged_audit_projection.open("wb") as handle:
            torch.save(audit_projection, handle)
            handle.flush()
            os.fsync(handle.fileno())
        loaded_projection = torch.load(staged_audit_projection, map_location="cpu", weights_only=True)
        if (not isinstance(loaded_projection, dict)
                or loaded_projection.get("schema") != "V10CheckpointAuditProjectionV1"
                or loaded_projection.get("metadata") != checkpoint["metadata"]
                or compute_state_sha(loaded_projection.get("model_state_dict", {}))
                != checkpoint["metadata"]["model_state_sha"]):
            raise ValueError("audit checkpoint projection validation failed")
        if stream_publication:
            writer_lease.assert_rooted_path("stream_recipe.json")
            staged_recipe = staging / "stream_recipe.json"
            shutil.copyfile(stream_recipe_path, staged_recipe)
            if file_identity(staged_recipe) != stream_recipe_identity:
                raise RuntimeError("stream recipe changed during completion staging")
        checkpoint_identity = file_identity(staged_checkpoint)
        validation.update(schema="V10CheckpointValidationV1", validated=True,
                          observed_at=datetime.now(timezone.utc).isoformat(),
                          run_id=config["run_id"] or None, execution_id=execution_id,
                          model_id=config["model_id"], global_seed=config["global_seed"],
                          global_step=summary["global_step"], source_identity=source_identity,
                          checkpoint_sha256=checkpoint_identity["sha256"],
                          checkpoint_bytes=checkpoint_identity["bytes"], authority=dict(AUTHORITY))
        if execution_evidence is not None:
            validation["execution_evidence"] = execution_evidence
            validation["execution_evidence_sha256"] = execution_evidence_sha
            validation["execution_evidence_validated"] = True
        validation["audit_projection"] = file_identity(staged_audit_projection)
        config_snapshot = dict(config)
        config_snapshot["config_sha"] = source_identity["config_sha"]
        atomic_training_json(staging / "config_snapshot.json", config_snapshot)
        atomic_training_json(staging / "checkpoint_validation.json", validation)
        publication_summary = dict(summary)
        if execution_evidence is not None:
            publication_summary["execution_evidence"] = execution_evidence
            publication_summary["execution_evidence_sha256"] = execution_evidence_sha
            publication_summary["execution_evidence_validated"] = True
        if checkpoint_commit is not None:
            commit = dict(commit)
            commit["completion_receipt"] = {
                "path": commit["manifest"]["path"].rsplit("/", 1)[0] + "/training_completion.json"
            }
            publication_summary["checkpoint_commit"] = commit
            summary["checkpoint_commit"] = commit
        atomic_training_json(staging / "training_summary.json", publication_summary)
        artifacts = {name: file_identity(staging / name) for name in PUBLICATION_FILES[:-1]}
        if stream_publication:
            artifacts["stream_recipe.json"] = {
                **file_identity(staging / "stream_recipe.json"), "path": "stream_recipe.json"}
        artifacts["checkpoint_commit.json"] = {
            **(json_identity(commit) if checkpoint_commit is not None
               else file_identity(output_dir / "checkpoint_commit.json")),
            "path": "checkpoint_commit.json"}
        generation_manifest = output_dir / commit["manifest"]["path"]
        generation_artifact_names = (
            "checkpoint_audit.pt", "checkpoint_validation.json", "config_snapshot.json", "training_summary.json")
        for name in generation_artifact_names:
            relative_target = f".checkpoint_generations/{commit['generation_id']}/{name}"
            generation_identity = copy_rooted_artifact(staging / name, relative_target, writer_lease)
            if generation_identity != {key: value for key, value in artifacts[name].items()
                                       if key in {"sha256", "bytes"}}:
                raise RuntimeError(f"generation artifact identity changed: {name}")
            artifacts[name]["path"] = relative_target
        artifacts["checkpoint_latest.pt"]["path"] = commit["checkpoint"]["path"]
        artifacts["checkpoint_generation_manifest.json"] = {
            **file_identity(generation_manifest), "path": commit["manifest"]["path"]}
        artifacts["metrics_journal.jsonl"] = {
            **file_identity(metrics_path), "path": commit["metrics_journal"]["path"]}
        metric_record_path = output_dir / commit["metric_record"]["path"]
        artifacts["metric_record.json"] = {
            **file_identity(metric_record_path), "path": commit["metric_record"]["path"]}
        completion = {"schema": "V10TrainingCompletionV1", "status": "COMPLETED",
                      "run_id": config["run_id"] or None, "execution_id": execution_id,
                      "model_id": config["model_id"], "global_seed": config["global_seed"],
                      "global_step": summary["global_step"], "source_identity": source_identity,
                      "model_state_sha": validation["model_state_sha"],
                      "checkpoint_commit": commit, "metrics_journal": artifacts["metrics_journal.jsonl"],
                      "artifacts": artifacts,
                      "authority": dict(AUTHORITY), "scientific_gate_evaluated": False}
        if execution_evidence is not None:
            completion["execution_evidence"] = execution_evidence
            completion["execution_evidence_sha256"] = execution_evidence_sha
            completion["execution_evidence_validated"] = True
        atomic_training_json(staging / "training_completion.json", completion)
        if checkpoint_commit is not None:
            generation_receipt = output_dir / commit["completion_receipt"]["path"]
            atomic_training_json(generation_receipt, completion)
            if file_identity(generation_receipt) != file_identity(staging / "training_completion.json"):
                raise RuntimeError("generation completion receipt staging mismatch")
            staged_descriptor = output_dir / ("checkpoint_commit_" + commit["generation_id"] + ".staged.json")
            atomic_training_json(staged_descriptor, commit)
            publish_checkpoint_commit(output_dir, commit, writer_lease=writer_lease)
            writer_lease.assert_active()
            cleanup_checkpoint_commit_artifacts(output_dir, commit["generation_id"])
            # Canonical files are projections only after the descriptor transition.
            publication_files = list(PUBLICATION_FILES)
            if stream_publication:
                publication_files.append("stream_recipe.json")
            for name in publication_files:
                try:
                    if writer_lease is not None:
                        writer_lease.assert_active()
                        writer_lease.assert_rooted_path(name, allow_missing=True)
                    os.replace(staging / name, output_dir / name)
                except OSError:
                    continue
            try:
                metric_projection = output_dir / commit["metric_record"]["path"] if isinstance(commit.get("metric_record"), dict) else None
                if metric_projection is not None and metric_projection.is_file():
                    append_training_record(output_dir / "metrics.jsonl",
                                           json.loads(metric_projection.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            return completion
        previous = {}
        publication_files = list(PUBLICATION_FILES)
        if stream_publication:
            publication_files.append("stream_recipe.json")
        for name in publication_files:
            destination = output_dir / name
            if destination.exists():
                if destination.is_symlink() or destination.stat().st_nlink != 1:
                    raise ValueError("unsafe canonical completion artifact")
                previous[name] = staging / ("previous_" + name)
                shutil.copyfile(destination, previous[name])
        published = []
        try:
            if writer_lease is not None:
                writer_lease.assert_active()
            for name in publication_files:
                writer_lease.assert_rooted_path(name, allow_missing=True)
                os.replace(staging / name, output_dir / name)
                published.append(name)
        except OSError:
            # Preserve the prior committed/incomplete state if publication is
            # interrupted. No new COMPLETED receipt exists before the final rename.
            for name in reversed(published):
                destination = output_dir / name
                if name in previous:
                    os.replace(previous[name], destination)
                elif destination.exists():
                    failed = output_dir / ("failed_publication_" + execution_id + "_" + name)
                    os.replace(destination, failed)
            raise
        return completion
