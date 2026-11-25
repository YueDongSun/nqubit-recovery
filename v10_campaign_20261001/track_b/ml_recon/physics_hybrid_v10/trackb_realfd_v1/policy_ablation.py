"""Source-bound runner for Track B conservative logical-action policy studies.

This runner is intentionally limited to synthetic V2.4 teacher-forced
candidate pre-states.  It selects a fixed action-scale/no-op-tail policy from
the training split, freezes it, and reports the held-out validation result.
It never emits an RF, AWG, or hardware action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import argparse
import json
import math
import os
import subprocess

import torch

from ml_recon.data_v2_4.hashing import sha256_bytes, sha256_file
from ml_recon.data_v2_4.loaders import V24TrainingDataset, validate_dataset_root

from .candidate_policy_evaluator import (
    FROZEN_VALIDATION_ROLE,
    TrackBActionScaleTailPolicyEvaluationV1,
    TrackBActionScaleTailPolicyV1,
    TrackBPolicyEvaluationBindingV1,
    TrackBResearchOnlyPolicyChoiceV1,
    TRAIN_SCREEN_ROLE,
    _policy_rows_input_sha,
    choose_research_only_conservative_policy,
    evaluate_trackb_action_scale_tail_policies,
)
from .contracts import (
    LEARNER_VISITED_HISTORY_VIEW,
    V24_ORACLE_HISTORY_VIEW,
    canonical_json_bytes,
    canonical_sha256,
    require_git_oid,
)
from .inverse_transformer import (
    CENTERED_LOGIT_BOUNDED_GAUGE_PROJECTION,
    TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION,
    UNOBSERVED_ZERO_HISTORY_GAIN_INPUT,
    GAIN_PARAMETERIZATION_KINDS,
    RAW_GAIN_PARAMETERIZATION_V1,
    TrackBInverseTransformerConfigV1,
    TrackBInverseTransformerV1,
    require_gain_parameterization_binding,
    require_action_decode_scale_binding,
)
from .inverse_transformer_pilot import (
    LEARNER_VIEW_CONFIG_FIELDS,
    PER_ROUND_ACTION_LOSS_NORMALIZATION,
    TrackBV24TenRoundPilotConfigV1,
    _open_pilot_dataset,
    _training_source_identity,
    _training_target_normalization,
)
from .pilot_history_dataset import Pilot_history_dataset
from .training import trackb_model_state_sha


POLICY_ABLATION_SCHEMA = "TrackBActionScaleTailPolicyAblationV1"
DEFAULT_SAMPLE_COUNT = 32
MAX_LEARNER_BEHAVIOR_CHAIN_DEPTH = 16
REPLAY_SHOT_MODE_BY_MANIFEST_SHOT_MODE = {
    "fixed_per_setting": "S_setting",
}


@dataclass
class _Model_read_context_v1:
    """Bound recursive receipt verification to this one model-load call."""

    active_receipt_shas: set[str] = field(default_factory=set)
    validated_models: dict[str, tuple[TrackBInverseTransformerV1, dict[str, Any], dict[str, Any]]] = field(
        default_factory=dict)


@dataclass(frozen=True)
class TrackBPolicyAblationConfigV1:
    """Reproducible input/output binding for a conservative-policy study."""

    pilot_receipt_path: str
    output_path: str
    train_sample_count: int = DEFAULT_SAMPLE_COUNT
    validation_sample_count: int = DEFAULT_SAMPLE_COUNT
    max_negative_active_step_gain_rate: float = 0.25
    minimum_mean_cumulative_gain: float = 0.0
    device: str = "cpu"
    schema: str = "TrackBPolicyAblationConfigV1"

    def validate(self) -> None:
        for name in ("pilot_receipt_path", "output_path"):
            value = Path(str(getattr(self, name)))
            if not value.is_absolute():
                raise ValueError(f"{name} must be absolute")
        for name in ("train_sample_count", "validation_sample_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not math.isfinite(float(self.max_negative_active_step_gain_rate))
                or not 0.0 <= float(self.max_negative_active_step_gain_rate) <= 1.0
                or not math.isfinite(float(self.minimum_mean_cumulative_gain))
                or not self.device):
            raise ValueError("Track B policy ablation thresholds or device are invalid")
        if self.schema != "TrackBPolicyAblationConfigV1":
            raise ValueError("unsupported Track B policy ablation config schema")


def default_trackb_action_scale_tail_policies() -> tuple[TrackBActionScaleTailPolicyV1, ...]:
    """Return the compact, explicit policy grid used for train-only screening."""
    policies = [TrackBActionScaleTailPolicyV1(action_scale=0.0, active_round_count=0)]
    for scale in (0.25, 0.5, 1.0):
        for active_round_count in (1, 4, 5, 6, 10):
            policies.append(TrackBActionScaleTailPolicyV1(
                action_scale=scale,
                active_round_count=active_round_count,
            ))
    return tuple(policies)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required Track B artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Track B artifact must contain a JSON object")
    return value


def _require_research_only_flags(value: Mapping[str, Any], *, label: str) -> None:
    if (value.get("research_only") is not True
            or value.get("selection_evidence") is not False
            or value.get("promotion_ready") is not False
            or value.get("hardware_action_authorized") is not False):
        raise ValueError(f"{label} violates the Track B research-only authority boundary")


def _require_oracle_history_reader(receipt: Mapping[str, Any], *, label: str) -> None:
    """Fail before a legacy diagnostic consumes learner-state supervision."""
    pilot_config = receipt.get("config")
    if not isinstance(pilot_config, Mapping):
        raise ValueError(f"{label} lacks a pilot source-view declaration")
    if pilot_config.get("source_view_kind", V24_ORACLE_HISTORY_VIEW) != V24_ORACLE_HISTORY_VIEW:
        raise ValueError(f"{label} does not support learner-visited source views")


def _declared_gain_parameterization(
    payload: Mapping[str, Any] | None,
    *,
    label: str,
) -> str | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} configuration is invalid")
    if "gain_parameterization" not in payload:
        return None
    value = payload["gain_parameterization"]
    if type(value) is not str or value not in GAIN_PARAMETERIZATION_KINDS:
        raise ValueError(f"{label} gain parameterization is invalid")
    return value


def _require_gain_parameterization_consistency(
    receipt: Mapping[str, Any],
    model_record: Mapping[str, Any],
    checkpoint_config_payload: Mapping[str, Any],
    model_config: TrackBInverseTransformerConfigV1,
    checkpoint_normalization: Mapping[str, Any],
) -> None:
    """Bind checkpoint, receipt-model, and pilot declarations before model creation."""
    pilot_config_payload = receipt.get("config")
    declarations = [
        _declared_gain_parameterization(checkpoint_config_payload, label="Track B checkpoint"),
        _declared_gain_parameterization(model_record.get("config"), label="Track B pilot receipt model"),
        _declared_gain_parameterization(pilot_config_payload, label="Track B pilot receipt"),
    ]
    if any(value is not None and value != model_config.gain_parameterization for value in declarations):
        raise ValueError("Track B gain parameterization declarations disagree")
    receipt_config_payload = model_record.get("config")
    receipt_normalization = model_record.get("target_normalization")
    if model_config.gain_parameterization == RAW_GAIN_PARAMETERIZATION_V1:
        if isinstance(receipt_config_payload, Mapping):
            receipt_config = TrackBInverseTransformerConfigV1(**receipt_config_payload)
            receipt_config.validate()
            if receipt_config.gain_parameterization != RAW_GAIN_PARAMETERIZATION_V1:
                raise ValueError("Track B pilot receipt gain parameterization differs from checkpoint")
        return
    if declarations[2] is None:
        raise ValueError("Track B pilot receipt lacks gain parameterization declaration")
    if not isinstance(receipt_config_payload, Mapping):
        raise ValueError("Track B pilot receipt lacks gain model configuration")
    if not isinstance(receipt_normalization, Mapping):
        raise ValueError("Track B pilot receipt lacks gain target normalization")
    receipt_config = TrackBInverseTransformerConfigV1(**receipt_config_payload)
    receipt_config.validate()
    require_gain_parameterization_binding(receipt_config, receipt_normalization)
    if (
        receipt_config.gain_parameterization != model_config.gain_parameterization
        or receipt_config.gain_center != model_config.gain_center
        or receipt_config.gain_scale != model_config.gain_scale
    ):
        raise ValueError("Track B pilot receipt gain parameterization differs from checkpoint")
    require_gain_parameterization_binding(model_config, receipt_normalization)
    require_gain_parameterization_binding(model_config, checkpoint_normalization)


def _require_model_configuration_consistency(
    receipt: Mapping[str, Any],
    model_record: Mapping[str, Any],
    model_config: TrackBInverseTransformerConfigV1,
) -> None:
    """Bind declared model semantics without changing legacy omission defaults."""
    declared_model = model_record.get("config")
    declared_pilot = receipt.get("config")
    if declared_model is not None:
        if not isinstance(declared_model, Mapping):
            raise ValueError("Track B receipt model configuration is invalid")
        parsed = TrackBInverseTransformerConfigV1(**declared_model)
        if parsed.to_dict() != model_config.to_dict():
            raise ValueError("Track B receipt model configuration differs from checkpoint")
    if declared_pilot is not None:
        if not isinstance(declared_pilot, Mapping):
            raise ValueError("Track B receipt pilot configuration is invalid")
        for name in ("system_type", "width", "heads", "blocks", "action_projection_kind",
                     "action_value_feature_kind", "gain_parameterization", "history_gain_input_kind"):
            if name in declared_pilot and declared_pilot[name] != getattr(model_config, name):
                raise ValueError(f"Track B receipt pilot configuration {name} differs from checkpoint")
    if model_config.action_projection_kind in (
        CENTERED_LOGIT_BOUNDED_GAUGE_PROJECTION,
        TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION,
    ):
        if (not isinstance(declared_model, Mapping) or "action_projection_kind" not in declared_model
                or not isinstance(declared_pilot, Mapping) or "action_projection_kind" not in declared_pilot):
            raise ValueError("Track B centered decoder requires explicit model and pilot projection declarations")
    if model_config.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION:
        if declared_pilot.get("action_loss_normalization_kind") != "per_round_v1":
            raise ValueError("Track B TRAIN-RMS pilot requires explicit per-round TRAIN normalization")
        receipt_normalization = model_record.get("target_normalization")
        if not isinstance(receipt_normalization, Mapping):
            raise ValueError("Track B TRAIN-RMS receipt lacks TRAIN normalization")
        require_action_decode_scale_binding(model_config, receipt_normalization)
    if model_config.history_gain_input_kind == UNOBSERVED_ZERO_HISTORY_GAIN_INPUT:
        if (not isinstance(declared_model, Mapping) or "history_gain_input_kind" not in declared_model
                or not isinstance(declared_pilot, Mapping) or "history_gain_input_kind" not in declared_pilot):
            raise ValueError("Track B unobserved history gain requires explicit model and pilot declarations")


def _require_train_rms_source_binding(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    model_config: TrackBInverseTransformerConfigV1,
) -> None:
    """Recompute opt-in decoder scales from the exact admitted TRAIN split."""
    if model_config.action_projection_kind != TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION:
        return
    pilot_config = receipt.get("config")
    batch_size = pilot_config.get("batch_size") if isinstance(pilot_config, Mapping) else None
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Track B TRAIN-RMS receipt lacks the original batch size")
    dataset_record = receipt.get("dataset")
    checkpoint_train_source = checkpoint.get("train_source")
    receipt_train_source = dataset_record.get("train") if isinstance(dataset_record, Mapping) else None
    if not isinstance(checkpoint_train_source, Mapping) or not isinstance(receipt_train_source, Mapping):
        raise ValueError("Track B TRAIN-RMS receipt lacks exact TRAIN source records")
    root, build_id, _identity, manifest = _bound_v24_source(receipt)
    cell = _matching_trackb_cell(manifest, system_type=model_config.system_type)
    dataset = Pilot_history_dataset(
        root, build_id=build_id, cell_id=str(cell["cell_id"]),
        split="train", task="feedback", arm="A0_RAW",
    )
    try:
        opened_source = _dataset_source_identity(dataset)
        _require_matching_source_identity(
            checkpoint_train_source, receipt_train_source, label="checkpoint/receipt TRAIN-RMS train")
        _require_matching_source_identity(
            checkpoint_train_source, opened_source, label="checkpoint/opened TRAIN-RMS train")
        normalization = _training_target_normalization(
            dataset,
            batch_size=batch_size,
            action_loss_normalization_kind=PER_ROUND_ACTION_LOSS_NORMALIZATION,
            require_full_active_modes=True,
        )
    finally:
        dataset.close()
    observed_normalization = canonical_json_bytes(normalization.to_dict())
    model_record = receipt.get("model")
    receipt_normalization = model_record.get("target_normalization") if isinstance(model_record, Mapping) else None
    for label, declared in (
        ("checkpoint", checkpoint.get("target_normalization")),
        ("receipt", receipt_normalization),
    ):
        if not isinstance(declared, Mapping) or canonical_json_bytes(declared) != observed_normalization:
            raise ValueError(f"Track B TRAIN-RMS {label} normalization differs from bound TRAIN source")
    require_action_decode_scale_binding(model_config, normalization.to_dict())


def _require_learner_view_source_binding(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    model_config: TrackBInverseTransformerConfigV1,
    verification_context: _Model_read_context_v1,
) -> bool:
    """Replay both admitted views and TRAIN labels for an opt-in pilot."""
    config_payload = receipt.get("config")
    if config_payload is None:
        if receipt.get("source_view_kind") is not None or checkpoint.get("source_view_kind") is not None:
            raise ValueError("Track B pilot lacks an explicit learner source-view configuration")
        config_payload = {}
    if not isinstance(config_payload, Mapping):
        raise ValueError("Track B pilot source-view configuration is invalid")
    source_view_kind = config_payload.get("source_view_kind", V24_ORACLE_HISTORY_VIEW)
    if source_view_kind == V24_ORACLE_HISTORY_VIEW:
        if (receipt.get("source_view_kind") not in (None, V24_ORACLE_HISTORY_VIEW)
                or checkpoint.get("source_view_kind") not in (None, V24_ORACLE_HISTORY_VIEW)
                or any(config_payload.get(name) for name in LEARNER_VIEW_CONFIG_FIELDS)):
            raise ValueError("Track B oracle pilot source-view declarations disagree")
        dataset_record = receipt.get("dataset")
        for split in ("train", "validation"):
            records = (
                checkpoint.get(f"{split}_source"),
                dataset_record.get(split) if isinstance(dataset_record, Mapping) else None,
            )
            if any(isinstance(record, Mapping)
                   and ("source_view_kind" in record or "learner_visited" in record)
                   for record in records):
                raise ValueError("Track B oracle pilot contains learner-visited source markers")
        return False
    if (source_view_kind != LEARNER_VISITED_HISTORY_VIEW
            or receipt.get("source_view_kind") != source_view_kind
            or checkpoint.get("source_view_kind") != source_view_kind
            or model_config.history_gain_input_kind != UNOBSERVED_ZERO_HISTORY_GAIN_INPUT):
        raise ValueError("Track B learner-visited source-view declarations disagree")
    try:
        config = TrackBV24TenRoundPilotConfigV1(**dict(config_payload))
    except TypeError as error:
        raise ValueError("Track B learner-visited pilot configuration fields differ") from error
    config.validate()
    root, build_id, _identity, manifest = _bound_v24_source(receipt)
    if Path(config.existing_v24_dataset_root).resolve() != root:
        raise ValueError("Track B learner-visited V2.4 root differs from receipt")
    cell = _matching_trackb_cell(manifest, system_type=model_config.system_type)
    dataset_record = receipt.get("dataset")
    if not isinstance(dataset_record, Mapping) or not isinstance(dataset_record.get("result"), Mapping):
        raise ValueError("Track B learner-visited pilot lacks its V2.4 result")
    result = dataset_record["result"]
    train_dataset = _open_pilot_dataset(
        config, root=root, build_id=build_id, cell_id=str(cell["cell_id"]), split="train",
        verification_context=verification_context)
    validation_dataset = None
    try:
        validation_dataset = _open_pilot_dataset(
            config, root=root, build_id=build_id, cell_id=str(cell["cell_id"]), split="validation",
            verification_context=verification_context)
        for split, dataset in (("train", train_dataset), ("validation", validation_dataset)):
            expected = _training_source_identity(config, dataset, result)
            expected_bytes = canonical_json_bytes(expected)
            for label, declared in (
                ("checkpoint", checkpoint.get(f"{split}_source")),
                ("receipt", dataset_record.get(split)),
            ):
                if not isinstance(declared, Mapping) or canonical_json_bytes(declared) != expected_bytes:
                    raise ValueError(f"Track B learner-visited {label} {split} source identity differs")
        normalization = _training_target_normalization(
            train_dataset,
            batch_size=config.batch_size,
            action_loss_normalization_kind=config.action_loss_normalization_kind,
            require_full_active_modes=(
                model_config.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION
            ),
            source_view_kind=LEARNER_VISITED_HISTORY_VIEW,
        )
    finally:
        train_dataset.close()
        if validation_dataset is not None:
            validation_dataset.close()
    expected_normalization = canonical_json_bytes(normalization.to_dict())
    model_record = receipt.get("model")
    receipt_normalization = model_record.get("target_normalization") if isinstance(model_record, Mapping) else None
    for label, declared in (
        ("checkpoint", checkpoint.get("target_normalization")),
        ("receipt", receipt_normalization),
    ):
        if not isinstance(declared, Mapping) or canonical_json_bytes(declared) != expected_normalization:
            raise ValueError(f"Track B learner-visited {label} TRAIN normalization differs")
    require_action_decode_scale_binding(model_config, normalization.to_dict())
    return True


def _load_bound_model(
    receipt: Mapping[str, Any],
    *,
    _verification_context: _Model_read_context_v1 | None = None,
) -> tuple[TrackBInverseTransformerV1, dict[str, Any], dict[str, Any]]:
    """Load once per exact receipt within a bounded provenance chain."""
    context = _verification_context if _verification_context is not None else _Model_read_context_v1()
    receipt_sha = canonical_sha256(receipt)
    if receipt_sha in context.active_receipt_shas:
        raise ValueError("Track B behavior receipt chain contains a cycle")
    cached = context.validated_models.get(receipt_sha)
    if cached is not None:
        checkpoint_path = Path(str(cached[1]["checkpoint_path"]))
        try:
            checkpoint_bytes = checkpoint_path.read_bytes()
        except OSError as error:
            raise ValueError("cached Track B behavior checkpoint is unavailable") from error
        if sha256_bytes(checkpoint_bytes) != cached[1]["checkpoint_sha"]:
            raise ValueError("cached Track B behavior checkpoint bytes have changed")
        return cached
    if len(context.active_receipt_shas) >= MAX_LEARNER_BEHAVIOR_CHAIN_DEPTH:
        raise ValueError("Track B behavior receipt chain exceeds its verified depth")
    context.active_receipt_shas.add(receipt_sha)
    try:
        result = _load_bound_model_uncached(receipt, verification_context=context)
        context.validated_models[receipt_sha] = result
        return result
    finally:
        context.active_receipt_shas.remove(receipt_sha)


def _load_bound_model_uncached(
    receipt: Mapping[str, Any],
    *,
    verification_context: _Model_read_context_v1,
) -> tuple[TrackBInverseTransformerV1, dict[str, Any], dict[str, Any]]:
    model_record = receipt.get("model")
    if not isinstance(model_record, dict):
        raise ValueError("Track B pilot receipt lacks a model record")
    checkpoint_path = Path(str(model_record.get("checkpoint_path", ""))).resolve()
    expected_sha = str(model_record.get("checkpoint_sha", ""))
    try:
        checkpoint_bytes = checkpoint_path.read_bytes()
    except OSError as error:
        raise ValueError("Track B checkpoint is unavailable") from error
    if len(expected_sha) != 64 or sha256_bytes(checkpoint_bytes) != expected_sha:
        raise ValueError("Track B checkpoint identity differs from its pilot receipt")
    try:
        checkpoint = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("Track B checkpoint loading requires weights-only PyTorch support") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != "TrackBInverseTransformerCheckpointV1":
        raise ValueError("Track B checkpoint schema is unsupported")
    _require_research_only_flags(checkpoint, label="Track B checkpoint")
    _require_checkpoint_producer_binding(receipt, checkpoint)
    config_payload = checkpoint.get("model_config")
    if not isinstance(config_payload, dict):
        raise ValueError("Track B checkpoint lacks model configuration")
    model_config = TrackBInverseTransformerConfigV1(**config_payload)
    model_config.validate()
    checkpoint_normalization = checkpoint.get("target_normalization")
    if not isinstance(checkpoint_normalization, Mapping):
        checkpoint_normalization = {}
    require_gain_parameterization_binding(model_config, checkpoint_normalization)
    require_action_decode_scale_binding(model_config, checkpoint_normalization)
    _require_gain_parameterization_consistency(
        receipt,
        model_record,
        config_payload,
        model_config,
        checkpoint_normalization,
    )
    _require_model_configuration_consistency(receipt, model_record, model_config)
    if not _require_learner_view_source_binding(
        receipt, checkpoint, model_config, verification_context):
        _require_train_rms_source_binding(receipt, checkpoint, model_config)
    model = TrackBInverseTransformerV1(model_config)
    model.load_state_dict(checkpoint.get("model_state_dict"), strict=True)
    return model, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha": expected_sha,
        "model_config": model_config.to_dict(),
        "model_config_sha": canonical_sha256(model_config.to_dict()),
        "model_state_sha": trackb_model_state_sha(model),
        "source_commit": checkpoint.get("source_commit"),
        "source_tree": checkpoint.get("source_tree"),
    }, checkpoint


def _require_checkpoint_producer_binding(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    """Reconcile an explicitly sealed receipt producer with its checkpoint.

    Older pilot receipts did not carry the top-level producer identity, so
    omission remains compatible.  Once either receipt field is present,
    however, both fields must be present and must match the exact producer
    fields embedded in the hashed checkpoint.  This check belongs before any
    source-qualified learner admission: resealing a receipt must not be able
    to relabel an otherwise unchanged checkpoint.
    """
    config_payload = receipt.get("config")
    source_view_kind = (
        config_payload.get("source_view_kind", V24_ORACLE_HISTORY_VIEW)
        if isinstance(config_payload, Mapping)
        else V24_ORACLE_HISTORY_VIEW
    )
    receipt_values = tuple(receipt.get(name) for name in ("source_commit", "source_tree"))
    checkpoint_values = tuple(checkpoint.get(name) for name in ("source_commit", "source_tree"))
    if all(value is None for value in receipt_values):
        if source_view_kind == LEARNER_VISITED_HISTORY_VIEW:
            raise ValueError("Track B learner-visited receipt producer identity is required")
        if any(value is None for value in checkpoint_values):
            raise ValueError("Track B checkpoint producer identity is incomplete")
        require_git_oid(checkpoint_values[0], "Track B checkpoint source_commit")
        require_git_oid(checkpoint_values[1], "Track B checkpoint source_tree")
        return
    if any(value is None for value in receipt_values):
        raise ValueError("Track B receipt producer identity is incomplete")
    if any(value is None for value in checkpoint_values):
        raise ValueError("Track B checkpoint producer identity is incomplete")
    require_git_oid(receipt_values[0], "Track B receipt source_commit")
    require_git_oid(receipt_values[1], "Track B receipt source_tree")
    require_git_oid(checkpoint_values[0], "Track B checkpoint source_commit")
    require_git_oid(checkpoint_values[1], "Track B checkpoint source_tree")
    if receipt_values != checkpoint_values:
        raise ValueError("Track B receipt producer identity differs from checkpoint")


def _bound_v24_source(receipt: Mapping[str, Any]) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    dataset_record = receipt.get("dataset")
    if not isinstance(dataset_record, dict) or not isinstance(dataset_record.get("result"), dict):
        raise ValueError("Track B pilot receipt lacks a V2.4 dataset record")
    result = dataset_record["result"]
    root = Path(str(result.get("dataset_root", ""))).resolve()
    identity = validate_dataset_root(root, build_id=str(result.get("build_id", "")))
    expected_generator_sha = _receipt_generator_sha(dataset_record, result)
    for name, expected in (
        ("contract_sha", result.get("contract_sha")),
        ("schema_sha", result.get("schema_sha")),
        ("generator_sha", expected_generator_sha),
    ):
        observed = str(identity.get(name, ""))
        if type(expected) is not str or not expected or expected != observed:
            raise ValueError(f"Track B V2.4 {name} differs from the pilot receipt")
    build_id = str(identity["build_id"])
    manifest = _load_json(root / "manifests" / f"{build_id}.json")
    _require_research_only_flags(manifest, label="Track B V2.4 manifest")
    return root, build_id, identity, manifest


def _receipt_generator_sha(
    dataset_record: Mapping[str, Any],
    result: Mapping[str, Any],
) -> str:
    """Resolve the one generator identity from result or both split identities."""
    candidates = []
    result_value = result.get("generator_sha")
    if type(result_value) is str and result_value:
        candidates.append(result_value)
    for split_name in ("train", "validation"):
        split_record = dataset_record.get(split_name)
        if not isinstance(split_record, Mapping):
            raise ValueError("Track B pilot receipt lacks bound split generator identity")
        split_value = split_record.get("generator_sha")
        if type(split_value) is not str or not split_value:
            raise ValueError("Track B pilot receipt has invalid split generator identity")
        candidates.append(split_value)
    if len(set(candidates)) != 1:
        raise ValueError("Track B pilot receipt generator identities disagree")
    return candidates[0]


def _matching_trackb_cell(manifest: Mapping[str, Any], *, system_type: str) -> dict[str, Any]:
    expected = f"trackb_feedback_history_n{int(system_type[0])}"
    cells = [cell for cell in manifest.get("cells", ()) if isinstance(cell, dict) and cell.get("cell_id") == expected]
    if len(cells) != 1:
        raise ValueError("Track B V2.4 manifest lacks exactly one bound feedback-history cell")
    cell = cells[0]
    if (cell.get("dataset_family") != "feedback_history"
            or cell.get("history_rounds") != 10
            or cell.get("history_policy") != "receding_horizon_target_oracle_v1"):
        raise ValueError("Track B V2.4 cell does not meet the ten-round receding-horizon contract")
    return cell


def _load_split_rows(dataset: V24TrainingDataset, *, sample_count: int) -> dict[int, Mapping[str, Any]]:
    if sample_count > len(dataset):
        raise ValueError("Track B policy ablation requested more rows than the bound V2.4 split contains")
    rows: dict[int, Mapping[str, Any]] = {}
    for position in range(sample_count):
        row = dataset[position]
        source_index = int(str(row["sample_id"]).rsplit(":", 1)[-1])
        if source_index in rows:
            raise ValueError("Track B V2.4 source split repeats a sample identity")
        rows[source_index] = row
    return rows


_SOURCE_BINDING_FIELDS = (
    "build_id",
    "ready_sha",
    "contract_sha",
    "schema_sha",
    "generator_sha",
    "build_manifest_sha",
    "index_sha",
    "split_sha",
    "sample_set_sha",
    "cell_id",
    "split",
    "source_commit",
    "source_tree",
)


def _dataset_source_identity(dataset: V24TrainingDataset) -> dict[str, object]:
    """Bind the entire opened V2.4 split, not only a selected row subset."""
    identity = dataset.identity
    return {
        "build_id": identity["build_id"],
        "ready_sha": identity["ready_sha"],
        "contract_sha": identity["contract_sha"],
        "schema_sha": identity["schema_sha"],
        "generator_sha": identity["generator_sha"],
        "build_manifest_sha": identity["build_manifest_sha"],
        "index_sha": identity["index_sha"],
        "split_sha": identity["split_sha"],
        "sample_set_sha": canonical_sha256(list(dataset.sample_ids)),
        "cell_id": dataset.cell_id,
        "split": dataset.split,
        "source_commit": identity["source_commit"],
        "source_tree": identity["source_tree"],
    }


def _require_matching_source_identity(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Reject same-schema roots that are not the exact checkpoint split source."""
    for field in _SOURCE_BINDING_FIELDS:
        expected_value = expected.get(field)
        observed_value = observed.get(field)
        if expected_value is None or observed_value is None or expected_value != observed_value:
            raise ValueError(f"Track B {label} source identity differs at {field}")


def _policy_binding(
    *,
    source: Mapping[str, Any],
    rows: Mapping[int, Mapping[str, Any]],
    model_identity: Mapping[str, Any],
    system_type: str,
    evaluation_role: str,
    evaluation_kwargs: Mapping[str, Any],
) -> TrackBPolicyEvaluationBindingV1:
    sample_ids = [str(rows[index]["sample_id"]) for index in sorted(rows)]
    binding = TrackBPolicyEvaluationBindingV1(
        v24_build_id=str(source["build_id"]),
        v24_ready_sha=str(source["ready_sha"]),
        v24_contract_sha=str(source["contract_sha"]),
        v24_schema_sha=str(source["schema_sha"]),
        v24_generator_sha=str(source["generator_sha"]),
        v24_build_manifest_sha=str(source["build_manifest_sha"]),
        v24_index_sha=str(source["index_sha"]),
        v24_split_sha=str(source["split_sha"]),
        parent_split_sample_set_sha=str(source["sample_set_sha"]),
        evaluation_sample_set_sha=canonical_sha256(sample_ids),
        evaluation_sample_count=len(sample_ids),
        cell_id=str(source["cell_id"]),
        split=str(source["split"]),
        system_type=system_type,
        mode_count=int(system_type[0]),
        model_checkpoint_sha=str(model_identity["checkpoint_sha"]),
        model_config_sha=str(model_identity["model_config_sha"]),
        model_state_sha=str(model_identity["model_state_sha"]),
        model_source_commit=str(model_identity["source_commit"]),
        model_source_tree=str(model_identity["source_tree"]),
        evaluation_input_sha=_policy_rows_input_sha(rows),
        global_seed=int(evaluation_kwargs["global_seed"]),
        noise_family=str(evaluation_kwargs["noise_family"]),
        eta=float(evaluation_kwargs["eta"]),
        kappa=float(evaluation_kwargs["kappa"]),
        replay_shot_mode=str(evaluation_kwargs["shot_mode"]),
        replay_shot_budget=int(evaluation_kwargs["shot_budget"]),
        evaluation_role=evaluation_role,
    )
    binding.validate()
    return binding


def _evaluation_kwargs(
    *,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    split: str,
    device: str,
) -> dict[str, Any]:
    manifest_shot_mode = str(cell["shot_mode"])
    replay_shot_mode = REPLAY_SHOT_MODE_BY_MANIFEST_SHOT_MODE.get(manifest_shot_mode)
    if replay_shot_mode is None:
        raise ValueError(
            "Track B policy ablation has no explicit renderer replay mapping for "
            f"manifest shot mode {manifest_shot_mode!r}")
    return {
        "global_seed": int(manifest["global_seed"]),
        "generator_sha": str(manifest["generator_sha"]),
        "cell_id": str(cell["cell_id"]),
        "split": split,
        "noise_family": str(cell["noise_family"]),
        "eta": float(cell["eta"]),
        "kappa": float(cell["kappa"]),
        "shot_mode": replay_shot_mode,
        "shot_budget": int(cell["shot_budget"]),
        "device": device,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _module_repository_root() -> Path:
    """Locate the checkout that owns this loaded evaluator module."""
    module_path = Path(__file__).resolve()
    for candidate in (module_path.parent, *module_path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Track B evaluator module is not inside a Git checkout")


def _committed_evaluator_identity() -> tuple[str, str]:
    repository_root = _module_repository_root()
    status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository_root,
        text=True,
    ).strip()
    if status:
        raise ValueError("Track B policy ablation requires committed evaluator source")
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=repository_root, text=True).strip()
    tree = subprocess.check_output(
        ("git", "rev-parse", "HEAD^{tree}"), cwd=repository_root, text=True).strip()
    return commit, tree


def run_trackb_action_scale_tail_policy_ablation(
    config: TrackBPolicyAblationConfigV1,
) -> dict[str, object]:
    """Select on train, freeze, and evaluate a conservative policy on validation."""
    config.validate()
    output_path = Path(config.output_path).resolve()
    if output_path.exists():
        raise ValueError("Track B policy ablation output path already exists")
    evaluator_source_commit, evaluator_source_tree = _committed_evaluator_identity()
    receipt = _load_json(Path(config.pilot_receipt_path).resolve())
    _require_research_only_flags(receipt, label="Track B pilot receipt")
    _require_oracle_history_reader(receipt, label="Track B policy ablation")
    if receipt.get("external_waveform_write") != "NOT_IMPLEMENTED":
        raise ValueError("Track B policy ablation refuses a receipt with waveform-write authority")
    model, model_identity, checkpoint = _load_bound_model(receipt)
    root, build_id, identity, manifest = _bound_v24_source(receipt)
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
        policies = default_trackb_action_scale_tail_policies()
        receipt_dataset = receipt.get("dataset")
        if not isinstance(receipt_dataset, Mapping):
            raise ValueError("Track B pilot receipt lacks split source identities")
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
            raise ValueError("Track B pilot lineage lacks exact train/validation source records")
        train_source = _dataset_source_identity(train_dataset)
        validation_source = _dataset_source_identity(validation_dataset)
        _require_matching_source_identity(
            checkpoint_train_source, receipt_train_source, label="checkpoint/receipt train")
        _require_matching_source_identity(
            checkpoint_train_source, train_source, label="checkpoint/opened train")
        _require_matching_source_identity(
            checkpoint_validation_source, receipt_validation_source, label="checkpoint/receipt validation")
        _require_matching_source_identity(
            checkpoint_validation_source, validation_source, label="checkpoint/opened validation")
        train_rows = _load_split_rows(train_dataset, sample_count=config.train_sample_count)
        validation_rows = _load_split_rows(validation_dataset, sample_count=config.validation_sample_count)
        train_evaluation_kwargs = _evaluation_kwargs(
            manifest=manifest,
            cell=cell,
            split="train",
            device=config.device,
        )
        validation_evaluation_kwargs = _evaluation_kwargs(
            manifest=manifest,
            cell=cell,
            split="validation",
            device=config.device,
        )
        train_evaluation = evaluate_trackb_action_scale_tail_policies(
            model,
            train_rows,
            policies=policies,
            binding=_policy_binding(
                source=train_source,
                rows=train_rows,
                model_identity=model_identity,
                system_type=model.config.system_type,
                evaluation_role=TRAIN_SCREEN_ROLE,
                evaluation_kwargs=train_evaluation_kwargs,
            ),
            **train_evaluation_kwargs,
        )
        choice = choose_research_only_conservative_policy(
            train_evaluation,
            max_negative_active_step_gain_rate=config.max_negative_active_step_gain_rate,
            minimum_mean_cumulative_gain=config.minimum_mean_cumulative_gain,
        )
        validation_policies = (policies[0], choice.policy)
        if choice.policy == policies[0]:
            validation_policies = (policies[0],)
        validation_evaluation = evaluate_trackb_action_scale_tail_policies(
            model,
            validation_rows,
            policies=validation_policies,
            binding=_policy_binding(
                source=validation_source,
                rows=validation_rows,
                model_identity=model_identity,
                system_type=model.config.system_type,
                evaluation_role=FROZEN_VALIDATION_ROLE,
                evaluation_kwargs=validation_evaluation_kwargs,
            ),
            **validation_evaluation_kwargs,
        )
    finally:
        train_dataset.close()
        validation_dataset.close()
    output = {
        "schema": POLICY_ABLATION_SCHEMA,
        "config": asdict(config),
        "model": model_identity,
        "evaluator": {
            "source_commit": evaluator_source_commit,
            "source_tree": evaluator_source_tree,
        },
        "v24_source": {
            "dataset_root": str(root),
            "build_id": build_id,
            "ready_sha": identity["ready_sha"],
            "contract_sha": identity["contract_sha"],
            "schema_sha": identity["schema_sha"],
            "generator_sha": identity["generator_sha"],
            "cell_id": cell["cell_id"],
            "history_rounds": cell["history_rounds"],
            "history_policy": cell["history_policy"],
            "manifest_shot_mode": cell["shot_mode"],
            "replay_shot_mode": REPLAY_SHOT_MODE_BY_MANIFEST_SHOT_MODE[str(cell["shot_mode"])],
            "replay_shot_mode_mapping": "MANIFEST_FIXED_PER_SETTING_TO_RENDERER_S_SETTING",
        },
        "train_only_screen": {
            "evaluation": train_evaluation.to_dict(),
            "evaluation_sha": train_evaluation.sha256(),
            "choice": choice.to_dict(),
        },
        "frozen_validation": {
            "evaluation": validation_evaluation.to_dict(),
            "evaluation_sha": validation_evaluation.sha256(),
            "selection_on_validation": "PROHIBITED",
        },
        "evaluation_semantics": (
            "SYNTHETIC_TEACHER_FORCED_CANDIDATE_ACTION_FORWARD_ENGINEERING_ONLY;"
            "NOT_A_SELF_ROLLOUT_OR_HARDWARE_EXECUTION"
        ),
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "external_waveform_write": "NOT_IMPLEMENTED",
    }
    _atomic_write(output_path, canonical_json_bytes(output))
    output["output_path"] = str(output_path)
    output["output_sha"] = sha256_file(output_path)
    return output


def _parse_args() -> TrackBPolicyAblationConfigV1:
    parser = argparse.ArgumentParser(description="Research-only Track B conservative policy ablation")
    parser.add_argument("--pilot-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--validation-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--max-negative-active-step-gain-rate", type=float, default=0.25)
    parser.add_argument("--minimum-mean-cumulative-gain", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    return TrackBPolicyAblationConfigV1(
        pilot_receipt_path=str(Path(args.pilot_receipt).resolve()),
        output_path=str(Path(args.output).resolve()),
        train_sample_count=args.train_samples,
        validation_sample_count=args.validation_samples,
        max_negative_active_step_gain_rate=args.max_negative_active_step_gain_rate,
        minimum_mean_cumulative_gain=args.minimum_mean_cumulative_gain,
        device=args.device,
    )


def main() -> None:
    result = run_trackb_action_scale_tail_policy_ablation(_parse_args())
    print(json.dumps({
        "output_path": result["output_path"],
        "output_sha": result["output_sha"],
        "selected_policy": result["train_only_screen"]["choice"]["policy"],
        "decision_reason": result["train_only_screen"]["choice"]["decision_reason"],
        "hardware_action_authorized": result["hardware_action_authorized"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_SAMPLE_COUNT",
    "POLICY_ABLATION_SCHEMA",
    "TrackBPolicyAblationConfigV1",
    "default_trackb_action_scale_tail_policies",
    "run_trackb_action_scale_tail_policy_ablation",
]
