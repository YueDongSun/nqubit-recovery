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
from .training_telemetry import AUTHORITY, atomic_training_json


PUBLICATION_FILES = ("checkpoint_latest.pt", "config_snapshot.json", "checkpoint_validation.json",
                     "training_summary.json", "training_completion.json")
REQUIRED_RUNTIME_STATES = ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                           "rng_state", "trainer_state", "model_build_spec")


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
                                compute_state_sha: Callable[[Mapping[str, torch.Tensor]], str]) -> dict[str, Any]:
    """Stage all outputs and publish a hash-bound completion receipt last."""
    if checkpoint["metadata"]["model_state_sha"] != summary["checkpoint_model_state_sha"]:
        raise RuntimeError("observer finalization changed the verified model state")
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
        checkpoint_identity = file_identity(staged_checkpoint)
        validation.update(schema="V10CheckpointValidationV1", validated=True,
                          observed_at=datetime.now(timezone.utc).isoformat(),
                          run_id=config["run_id"] or None, execution_id=execution_id,
                          model_id=config["model_id"], global_seed=config["global_seed"],
                          global_step=summary["global_step"], source_identity=source_identity,
                          checkpoint_sha256=checkpoint_identity["sha256"],
                          checkpoint_bytes=checkpoint_identity["bytes"], authority=dict(AUTHORITY))
        atomic_training_json(staging / "config_snapshot.json", config)
        atomic_training_json(staging / "checkpoint_validation.json", validation)
        atomic_training_json(staging / "training_summary.json", summary)
        artifacts = {name: file_identity(staging / name) for name in PUBLICATION_FILES[:-1]}
        completion = {"schema": "V10TrainingCompletionV1", "status": "COMPLETED",
                      "run_id": config["run_id"] or None, "execution_id": execution_id,
                      "model_id": config["model_id"], "global_seed": config["global_seed"],
                      "global_step": summary["global_step"], "source_identity": source_identity,
                      "model_state_sha": validation["model_state_sha"], "artifacts": artifacts,
                      "authority": dict(AUTHORITY), "scientific_gate_evaluated": False}
        atomic_training_json(staging / "training_completion.json", completion)
        previous = {}
        for name in PUBLICATION_FILES:
            destination = output_dir / name
            if destination.exists():
                if destination.is_symlink() or destination.stat().st_nlink != 1:
                    raise ValueError("unsafe canonical completion artifact")
                previous[name] = staging / ("previous_" + name)
                shutil.copyfile(destination, previous[name])
        published = []
        try:
            for name in PUBLICATION_FILES:
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
