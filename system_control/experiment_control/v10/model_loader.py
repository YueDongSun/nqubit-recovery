"""Fail-closed loader for hash-bound V10 research checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import torch

from ml_recon.physics_hybrid_v10.checkpoint_schema import validate_checkpoint_envelope

from .schemas import V10ModelPackage


def load_v10_model_package(
    checkpoint_path: str | Path,
    *,
    expected_model_id: str,
    operation_registry_hash: str,
    measurement_contract_sha: str,
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    raw = path.read_bytes()
    checkpoint_sha = hashlib.sha256(raw).hexdigest()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    validate_checkpoint_envelope(metadata)
    if metadata["model_id"] != expected_model_id:
        raise ValueError("checkpoint model_id does not match requested hardware cell")
    package = V10ModelPackage(
        model_id=expected_model_id,
        checkpoint_sha=checkpoint_sha,
        source_commit=str(metadata["source_commit"]),
        operation_registry_hash=operation_registry_hash,
        measurement_contract_sha=measurement_contract_sha,
        selection_evidence=False,
        promotion_ready=False,
        hardware_action_authorized=False,
        research_only=True,
    )
    package.validate()
    return {"package": package, "checkpoint": checkpoint,
            "shadow_inference_only": True}
