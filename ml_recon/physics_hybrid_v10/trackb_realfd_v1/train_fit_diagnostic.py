"""Read-only frozen-endpoint TRAIN diagnostics for Track B ten-round pilots.

This module binds an existing pilot receipt, checkpoint, and V2.4 TRAIN split
before evaluating them with no optimizer, gradient, checkpoint, or receipt
mutation.  Its output is diagnostic-only and cannot establish validation,
selection, promotion, waveform, or hardware authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import json
import os
import subprocess
import tempfile

import torch

from ml_recon.data_v2_4.hashing import sha256_bytes, sha256_file
from ml_recon.data_v2_4.loaders import V24TrainingDataset

from .contracts import canonical_json_bytes, canonical_sha256
from .inverse_transformer_pilot import _evaluate_teacher_forced
from .policy_ablation import (
    _bound_v24_source,
    _dataset_source_identity,
    _load_bound_model,
    _matching_trackb_cell,
    _require_matching_source_identity,
    _require_oracle_history_reader,
    _require_research_only_flags,
)


TRAIN_FIT_DIAGNOSTIC_SCHEMA = "TrackBFrozenTrainFitDiagnosticV1"
EQUAL_BATCH_METRIC_AGGREGATION = "EQUAL_BATCH"


@dataclass(frozen=True)
class TrackBFrozenTrainFitDiagnosticConfigV1:
    """Immutable inputs and output for one no-gradient TRAIN split diagnostic."""

    pilot_receipt_path: str
    expected_pilot_receipt_sha: str
    output_path: str
    batch_size: int = 8
    device: str = "cpu"
    schema: str = TRAIN_FIT_DIAGNOSTIC_SCHEMA

    def validate(self) -> None:
        for name in ("pilot_receipt_path", "output_path"):
            if not Path(str(getattr(self, name))).is_absolute():
                raise ValueError(f"{name} must be absolute")
        if type(self.expected_pilot_receipt_sha) is not str or len(self.expected_pilot_receipt_sha) != 64:
            raise ValueError("expected_pilot_receipt_sha must be a SHA-256 hex value")
        try:
            int(self.expected_pilot_receipt_sha, 16)
        except ValueError as error:
            raise ValueError("expected_pilot_receipt_sha must be a SHA-256 hex value") from error
        if type(self.batch_size) is not int or self.batch_size <= 0 or not self.device:
            raise ValueError("train-fit diagnostic batch_size or device is invalid")
        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError("train-fit diagnostic device is invalid") from error
        if device.type != "cpu":
            raise ValueError("train-fit diagnostic supports CPU only")
        if self.schema != TRAIN_FIT_DIAGNOSTIC_SCHEMA:
            raise ValueError("unsupported Track B train-fit diagnostic schema")


def _diagnostic_repository_identity() -> tuple[str, str]:
    repository_root = Path(__file__).resolve().parents[3]
    status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=repository_root,
        text=True,
    ).strip()
    if status:
        raise ValueError("Track B train-fit diagnostic requires committed source")
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        text=True,
    ).strip()
    tree = subprocess.check_output(
        ("git", "rev-parse", "HEAD^{tree}"),
        cwd=repository_root,
        text=True,
    ).strip()
    return commit, tree


def _read_verified_pilot_receipt(
    path: Path,
    expected_sha: str,
) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("Track B pilot receipt is unavailable") from error
    if sha256_bytes(payload) != expected_sha:
        raise ValueError("Track B pilot receipt SHA differs from the diagnostic request")
    try:
        receipt = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Track B pilot receipt is invalid JSON") from error
    if type(receipt) is not dict:
        raise ValueError("Track B pilot receipt must be a JSON object")
    _require_research_only_flags(receipt, label="Track B pilot receipt")
    if receipt.get("external_waveform_write") != "NOT_IMPLEMENTED":
        raise ValueError("Track B train-fit diagnostic refuses waveform-capable receipts")
    return receipt


def _prepare_new_output_path(path: Path) -> Path:
    """Resolve a new report target without following an existing output alias."""
    if os.path.lexists(path):
        raise ValueError("Track B train-fit diagnostic output path already exists")
    parent = path.parent.resolve()
    if not parent.is_dir():
        raise ValueError("Track B train-fit diagnostic output parent is unavailable")
    result = parent / path.name
    if os.path.lexists(result):
        raise ValueError("Track B train-fit diagnostic output path already exists")
    return result


def _is_within_path(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_isolated_report_output_path(
    output_path: Path,
    *,
    dataset_root: Path,
    receipt_path: Path,
    checkpoint_path: Path,
) -> None:
    """Keep new diagnostic reports out of immutable inputs and pilot artifacts."""
    protected_roots = {
        "V2.4 dataset root": dataset_root.resolve(),
        "pilot receipt directory": receipt_path.parent.resolve(),
        "checkpoint directory": checkpoint_path.parent.resolve(),
    }
    for label, protected_root in protected_roots.items():
        if _is_within_path(output_path, protected_root):
            raise ValueError(
                f"Track B train-fit diagnostic output must be isolated from {label}",
            )


def _bound_batch_plan(
    receipt: Mapping[str, Any],
    *,
    requested_batch_size: int,
) -> dict[str, object]:
    """Bind diagnostic aggregation to the original pilot's equal-batch plan."""
    config_record = receipt.get("config")
    if not isinstance(config_record, Mapping):
        raise ValueError("Track B pilot receipt lacks a bound batch plan")
    pilot_batch_size = config_record.get("batch_size")
    if type(pilot_batch_size) is not int or pilot_batch_size <= 0:
        raise ValueError("Track B pilot receipt batch plan is invalid")
    if requested_batch_size != pilot_batch_size:
        raise ValueError("Track B train-fit diagnostic batch_size differs from the pilot receipt")
    return {
        "batch_size": pilot_batch_size,
        "metric_aggregation": EQUAL_BATCH_METRIC_AGGREGATION,
    }


def _checkpoint_path_from_receipt(receipt: Mapping[str, Any]) -> Path:
    model_record = receipt.get("model")
    if not isinstance(model_record, Mapping):
        raise ValueError("Track B pilot receipt lacks a model record")
    checkpoint_value = model_record.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("Track B pilot receipt lacks a checkpoint path")
    return Path(checkpoint_value).resolve()


def _atomic_publish_new_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes once without reusing a temporary name or replacing a target."""
    if os.path.lexists(path):
        raise ValueError("Track B train-fit diagnostic output path already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError("Track B train-fit diagnostic output path already exists") from error
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def run_trackb_frozen_train_fit_diagnostic(
    config: TrackBFrozenTrainFitDiagnosticConfigV1,
) -> dict[str, object]:
    """Evaluate the frozen 191-step checkpoint on TRAIN without mutation."""
    config.validate()
    output_path = _prepare_new_output_path(Path(config.output_path))
    diagnostic_source_commit, diagnostic_source_tree = _diagnostic_repository_identity()
    receipt_path = Path(config.pilot_receipt_path).resolve()
    receipt = _read_verified_pilot_receipt(
        receipt_path,
        config.expected_pilot_receipt_sha,
    )
    _require_oracle_history_reader(receipt, label="Track B frozen TRAIN fit diagnostic")
    batch_plan = _bound_batch_plan(receipt, requested_batch_size=config.batch_size)
    root, build_id, _identity, manifest = _bound_v24_source(receipt)
    checkpoint_path = _checkpoint_path_from_receipt(receipt)
    _require_isolated_report_output_path(
        output_path,
        dataset_root=root,
        receipt_path=receipt_path,
        checkpoint_path=checkpoint_path,
    )
    model, model_identity, checkpoint = _load_bound_model(receipt)
    cell = _matching_trackb_cell(manifest, system_type=model.config.system_type)
    train_dataset = V24TrainingDataset(
        root,
        build_id=build_id,
        cell_id=str(cell["cell_id"]),
        split="train",
        task="feedback",
        arm="A0_RAW",
    )
    validation_dataset = V24TrainingDataset(
        root,
        build_id=build_id,
        cell_id=str(cell["cell_id"]),
        split="validation",
        task="feedback",
        arm="A0_RAW",
    )
    try:
        receipt_dataset = receipt.get("dataset")
        if not isinstance(receipt_dataset, Mapping):
            raise ValueError("Track B pilot receipt lacks dataset identities")
        checkpoint_train_source = checkpoint.get("train_source")
        checkpoint_validation_source = checkpoint.get("validation_source")
        receipt_train_source = receipt_dataset.get("train")
        receipt_validation_source = receipt_dataset.get("validation")
        if not all(isinstance(value, Mapping) for value in (
            checkpoint_train_source,
            checkpoint_validation_source,
            receipt_train_source,
            receipt_validation_source,
        )):
            raise ValueError("Track B pilot receipt lacks bound split identities")
        train_source = _dataset_source_identity(train_dataset)
        validation_source = _dataset_source_identity(validation_dataset)
        _require_matching_source_identity(
            checkpoint_train_source,
            receipt_train_source,
            label="checkpoint/receipt train",
        )
        _require_matching_source_identity(
            checkpoint_train_source,
            train_source,
            label="checkpoint/opened train",
        )
        _require_matching_source_identity(
            checkpoint_validation_source,
            receipt_validation_source,
            label="checkpoint/receipt validation",
        )
        _require_matching_source_identity(
            checkpoint_validation_source,
            validation_source,
            label="checkpoint/opened validation",
        )
        evaluation, _guidance = _evaluate_teacher_forced(
            model,
            train_dataset,
            batch_size=config.batch_size,
            device=torch.device(config.device),
        )
        evaluation = {
            **evaluation,
            "diagnostic_kind": "TRAIN_FIT_DIAGNOSTIC",
            "split": "train",
            "diagnostic_only": True,
            "sample_count": len(train_dataset),
            "direct_supervision_coverage": "NOT_RECORDED_BY_BASELINE_PILOT",
            "metric_aggregation": batch_plan["metric_aggregation"],
            "selection_evidence": False,
            "promotion_ready": False,
            "hardware_action_authorized": False,
        }
        output = {
            "schema": TRAIN_FIT_DIAGNOSTIC_SCHEMA,
            "config": asdict(config),
            "diagnostic_source_commit": diagnostic_source_commit,
            "diagnostic_source_tree": diagnostic_source_tree,
            "pilot": {
                "receipt_path": str(receipt_path),
                "receipt_file_sha": config.expected_pilot_receipt_sha,
                "pilot_source_commit": receipt.get("source_commit"),
                "pilot_source_tree": receipt.get("source_tree"),
                "checkpoint_sha": model_identity["checkpoint_sha"],
                "checkpoint_model_state_sha": model_identity["model_state_sha"],
                "original_validation_evaluation_sha": canonical_sha256(receipt["evaluation"]),
            },
            "model": model_identity,
            "batch_plan": batch_plan,
            "train_source": train_source,
            "validation_source": validation_source,
            "evaluation": evaluation,
            "research_only": True,
            "selection_evidence": False,
            "promotion_ready": False,
            "hardware_action_authorized": False,
            "external_waveform_write": "NOT_IMPLEMENTED",
        }
        _atomic_publish_new_bytes(output_path, canonical_json_bytes(output))
        output["output_path"] = str(output_path)
        output["output_sha"] = sha256_file(output_path)
        return output
    finally:
        train_dataset.close()
        validation_dataset.close()


__all__ = [
    "EQUAL_BATCH_METRIC_AGGREGATION",
    "TRAIN_FIT_DIAGNOSTIC_SCHEMA",
    "TrackBFrozenTrainFitDiagnosticConfigV1",
    "run_trackb_frozen_train_fit_diagnostic",
]
