"""Admit a frozen Track B rollout as a distinct learner-visited data view.

This reader never changes the immutable V2.4 root or the frozen checkpoint.
It validates complete rollout and source lineage before exposing separate
executed-history inputs and learner-state oracle supervision to training.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ml_recon.data_v2_4.hashing import sha256_bytes

from .contracts import (
    LEARNER_VISITED_HISTORY_VIEW,
    LEARNER_VISITED_ROW_SCHEMA,
    canonical_json_bytes,
    canonical_sha256,
    require_git_oid,
    require_sha256,
)
from .inverse_transformer import UNOBSERVED_ZERO_HISTORY_GAIN_INPUT
from .learner_visited_contract import ROW_TENSOR_NAMES, validate_learner_visited_tensors
from .learner_visited_history import derive_learner_visited_row
from .pilot_history_dataset import Pilot_history_dataset
from .pilot_runtime_checks import pilot_source_identity
from .policy_ablation import (
    _bound_v24_source,
    _evaluation_kwargs,
    _load_bound_model,
    _matching_trackb_cell,
    _require_matching_source_identity,
)
from .self_rollout_diagnostic import (
    TrackBSelfRolloutDiagnosticConfigV1,
    _bound_development_rows,
    _read_verified_pilot_receipt,
    _validate_self_rollout_result,
    _verified_replay_key,
)
from .self_rollout_evaluator import (
    Self_rollout_config_v1,
    evaluate_trackb_model_self_rollout,
)
from .self_rollout_shift_contract import (
    LEGACY_ORACLE_GAIN_SHIFT, history_gain_distribution_shift_kind,
)


LEARNER_VISITED_REPORT_VIEW_SCHEMA = "TrackBLearnerVisitedReportViewV2"
REGISTERED_SPLIT_COUNTS = {"train": 1_000, "validation": 200}
LEGACY_V2_COMPATIBILITY_MODE = "LEGACY_V2_NORMALIZED_FOR_V3_REPLAY"
NATIVE_V3_COMPATIBILITY_MODE = "NATIVE_V3_REPLAY"
ROW_METADATA_NAMES = (
    "schema", "row_sha", "row_identity_json", "sample_id", "split",
    "research_only", "selection_evidence", "promotion_ready",
    "hardware_action_authorized", "deployment_validation", "held_out",
)


@dataclass(frozen=True)
class Learner_visited_report_view_config_v1:
    """Exact paths and identities for one development-only derived split."""

    pilot_receipt_path: str
    expected_pilot_receipt_sha: str
    rollout_report_path: str
    expected_rollout_report_sha: str
    split: str
    sample_count: int
    allow_partial_development: bool = False
    schema: str = "LearnerVisitedReportViewConfigV1"

    def validate(self) -> None:
        for name in ("pilot_receipt_path", "rollout_report_path"):
            value = getattr(self, name)
            if type(value) is not str or not Path(value).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        require_sha256(self.expected_pilot_receipt_sha, "expected_pilot_receipt_sha")
        require_sha256(self.expected_rollout_report_sha, "expected_rollout_report_sha")
        if self.split not in REGISTERED_SPLIT_COUNTS:
            raise ValueError("learner-visited report view accepts TRAIN or validation only")
        if type(self.sample_count) is not int or not 0 < self.sample_count <= REGISTERED_SPLIT_COUNTS[self.split]:
            raise ValueError("learner-visited report sample count is invalid")
        if type(self.allow_partial_development) is not bool:
            raise TypeError("allow_partial_development must be a strict boolean")
        if not self.allow_partial_development and self.sample_count != REGISTERED_SPLIT_COUNTS[self.split]:
            raise ValueError("learner-visited training view requires the full registered split")
        if self.schema != "LearnerVisitedReportViewConfigV1":
            raise ValueError("learner-visited report view config schema is unsupported")


def _read_verified_report(config: Learner_visited_report_view_config_v1) -> dict[str, Any]:
    path = Path(config.rollout_report_path).resolve()
    payload = path.read_bytes()
    if sha256_bytes(payload) != config.expected_rollout_report_sha:
        raise ValueError("learner-visited rollout report differs from its expected SHA")
    try:
        report = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("learner-visited rollout report is not UTF-8 JSON") from error
    if type(report) is not dict or canonical_json_bytes(report) != payload:
        raise ValueError("learner-visited rollout report is not canonical JSON")
    return report


def _require_report_binding(
    report: Mapping[str, Any],
    config: Learner_visited_report_view_config_v1,
    model_identity: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> None:
    declared = report.get("config")
    accounting = report.get("episode_accounting")
    sample_ids = report.get("sample_ids")
    if not isinstance(declared, Mapping):
        raise ValueError("learner-visited rollout report lacks a configuration")
    try:
        declared_config = TrackBSelfRolloutDiagnosticConfigV1(**declared)
    except TypeError as error:
        raise ValueError("learner-visited rollout configuration fields differ") from error
    declared_config.validate()
    require_git_oid(report.get("evaluator_source_commit"), "rollout_evaluator_source_commit")
    require_git_oid(report.get("evaluator_source_tree"), "rollout_evaluator_source_tree")
    if (not isinstance(accounting, Mapping)
            or not isinstance(sample_ids, list)
            or report.get("schema") != "TrackBSelfRolloutDiagnosticV1"
            or report.get("status") != "COMPLETE"
            or report.get("evaluation_role") != "DEVELOPMENT_DIAGNOSTIC"
            or report.get("gate_decision") != "NOT_PERFORMED"
            or report.get("research_only") is not True
            or report.get("diagnostic_only") is not True
            or report.get("selection_evidence") is not False
            or report.get("promotion_ready") is not False
            or report.get("hardware_action_authorized") is not False
            or report.get("external_waveform_write") != "NOT_IMPLEMENTED"
            or report.get("history_gain_input_kind") != UNOBSERVED_ZERO_HISTORY_GAIN_INPUT
            or report.get("history_distribution_shift")
            != "UNOBSERVED_ZERO_GAIN_INPUT_MATCHED_LEARNER_STATE_SHIFT_REMAINS"
            or report.get("evaluation_recipe")
            != "COMPLETE_ACQUISITION_SELF_ROLLOUT_WITH_MATCHED_TEACHER_CONTROL_V2"
            or report.get("pilot_receipt_sha") != config.expected_pilot_receipt_sha
            or declared.get("expected_pilot_receipt_sha") != config.expected_pilot_receipt_sha
            or Path(str(declared.get("pilot_receipt_path", ""))).resolve()
            != Path(config.pilot_receipt_path).resolve()
            or Path(str(declared.get("output_path", ""))).resolve()
            != Path(config.rollout_report_path).resolve()
            or declared.get("split") != config.split
            or declared.get("sample_count") != config.sample_count
            or report.get("sample_count") != config.sample_count
            or len(sample_ids) != config.sample_count
            or len(set(sample_ids)) != config.sample_count
            or report.get("sample_set_sha") != canonical_sha256(sample_ids)
            or any(accounting.get(name) != expected for name, expected in (
                ("requested_count", config.sample_count),
                ("attempted_count", config.sample_count),
                ("completed_count", config.sample_count),
                ("failed_count", 0), ("not_attempted_count", 0),
            ))
            or canonical_sha256(report.get("model")) != canonical_sha256(model_identity)
            or canonical_sha256(report.get("replay")) != canonical_sha256(replay)):
        raise ValueError("learner-visited rollout report binding or authority differs")


class Learner_visited_report_dataset:
    """Validated in-memory rows derived from one immutable rollout report."""

    def __init__(
        self, config: Learner_visited_report_view_config_v1,
        *, verification_context: Any = None,
    ) -> None:
        config.validate()
        producer_commit, producer_tree = pilot_source_identity()
        report = _read_verified_report(config)
        receipt = _read_verified_pilot_receipt(
            Path(config.pilot_receipt_path).resolve(), config.expected_pilot_receipt_sha)
        model, model_identity, checkpoint = _load_bound_model(
            receipt, _verification_context=verification_context)
        if model.config.history_gain_input_kind != UNOBSERVED_ZERO_HISTORY_GAIN_INPUT:
            raise ValueError("learner-visited source requires an observable-gain behavior checkpoint")
        root, build_id, _identity, manifest = _bound_v24_source(receipt)
        cell = _matching_trackb_cell(manifest, system_type=model.config.system_type)
        replay = _evaluation_kwargs(manifest=manifest, cell=cell, split=config.split, device="cpu")
        _require_report_binding(report, config, model_identity, replay)
        base_dataset = Pilot_history_dataset(
            root, build_id=build_id, cell_id=str(cell["cell_id"]),
            split=config.split, task="feedback", arm="A0_RAW",
        )
        try:
            base_source, base_rows = _bound_development_rows(
                receipt, checkpoint, base_dataset, sample_count=config.sample_count)
            report_source = report.get("source")
            if not isinstance(report_source, Mapping):
                raise ValueError("learner-visited rollout report lacks split source identity")
            _require_matching_source_identity(report_source, base_source, label="rollout/opened learner source")
            source_indices = sorted(base_rows)
            expected_ids = [str(base_rows[index]["sample_id"]) for index in source_indices]
            if report["sample_ids"] != expected_ids:
                raise ValueError("learner-visited rollout sample order differs from V2.4 source")
            samples = report.get("samples")
            if not isinstance(samples, list) or len(samples) != config.sample_count:
                raise ValueError("learner-visited rollout report lacks complete samples")
            sample_schemas: list[str] = []
            for sample in samples:
                if not isinstance(sample, Mapping):
                    raise ValueError("learner-visited rollout report contains an unavailable sample")
                sample_schema = sample.get("schema")
                if sample_schema not in (
                    "TrackBSelfRolloutEvaluationV2", "TrackBSelfRolloutEvaluationV3",
                ):
                    raise ValueError("learner-visited sample schema is unsupported")
                sample_schemas.append(str(sample_schema))
            if len(set(sample_schemas)) != 1:
                raise ValueError("learner-visited rollout report mixes sample schemas")
            original_sample_schema = sample_schemas[0]
            compatibility_mode = (
                LEGACY_V2_COMPATIBILITY_MODE
                if original_sample_schema == "TrackBSelfRolloutEvaluationV2"
                else NATIVE_V3_COMPATIBILITY_MODE
            )
            derived_rows: list[dict[str, Any]] = []
            for source_index, sample in zip(source_indices, samples):
                key = _verified_replay_key(
                    source_index, base_rows[source_index],
                    mode_count=model.config.mode_count, replay=replay)
                if not isinstance(sample, Mapping):
                    raise ValueError("learner-visited rollout sample is unavailable")
                if sample.get("schema") != original_sample_schema:
                    raise ValueError("learner-visited rollout sample schema differs from report mode")
                expected_shift = history_gain_distribution_shift_kind(
                    model.config.history_gain_input_kind)
                allow_legacy_v2 = compatibility_mode == LEGACY_V2_COMPATIBILITY_MODE
                if compatibility_mode == LEGACY_V2_COMPATIBILITY_MODE:
                    if sample.get("input_distribution_shift") != LEGACY_ORACLE_GAIN_SHIFT:
                        raise ValueError("legacy V2 learner-visited sample has an unknown distribution shift")
                    comparison_sample = deepcopy(sample)
                    comparison_sample["schema"] = "TrackBSelfRolloutEvaluationV3"
                    comparison_sample["input_distribution_shift"] = expected_shift
                elif compatibility_mode == NATIVE_V3_COMPATIBILITY_MODE:
                    if sample.get("input_distribution_shift") != expected_shift:
                        raise ValueError("V3 learner-visited sample distribution shift differs from bound model")
                    comparison_sample = sample
                else:
                    raise ValueError("learner-visited sample schema is unsupported")
                _validate_self_rollout_result(
                    sample, key=key, model=model, replay=replay,
                    replicate_index=int(report["config"]["replicate_index"]),
                    pairing_kind=str(report["config"]["pairing_kind"]),
                    allow_legacy_v2=allow_legacy_v2,
                )
                if not isinstance(sample, Mapping) or not isinstance(sample.get("config"), Mapping):
                    raise ValueError("learner-visited rollout sample config is unavailable")
                sample_config = Self_rollout_config_v1(**sample["config"])
                reproduced = evaluate_trackb_model_self_rollout(
                    model, key,
                    replicate_index=int(report["config"]["replicate_index"]),
                    pairing_kind=str(report["config"]["pairing_kind"]),
                    include_teacher_state_control=sample_config.include_teacher_state_control,
                    **{name: replay[name] for name in (
                        "shot_mode", "shot_budget", "noise_family", "eta", "kappa",
                    )},
                )
                if canonical_json_bytes(reproduced) != canonical_json_bytes(comparison_sample):
                    raise ValueError("learner-visited report differs from frozen behavior rollout regeneration")
                row = derive_learner_visited_row(
                    sample, key, config=sample_config)
                source_target = base_rows[source_index]["target_mode_tokens"].detach().cpu().numpy()
                if not np.allclose(row["target_mode_tokens"], source_target, atol=1e-6, rtol=0):
                    raise ValueError("learner-visited target tokens differ from bound V2.4 source")
                validate_learner_visited_tensors(
                    row["row_identity_json"], row["row_sha"],
                    {name: row[name] for name in ROW_TENSOR_NAMES})
                derived_rows.append(row)
            sample_ids = tuple(expected_ids)
            row_shas = tuple(row["row_sha"] for row in derived_rows)
            self.source_identity = {
                "schema": LEARNER_VISITED_REPORT_VIEW_SCHEMA,
                "source_view_kind": LEARNER_VISITED_HISTORY_VIEW,
                "row_schema": LEARNER_VISITED_ROW_SCHEMA,
                "split": config.split,
                "sample_count": config.sample_count,
                "registered_budget_complete": config.sample_count == REGISTERED_SPLIT_COUNTS[config.split],
                "original_sample_schema": original_sample_schema,
                "compatibility_mode": compatibility_mode,
                "pilot_receipt_sha": config.expected_pilot_receipt_sha,
                "rollout_report_sha": config.expected_rollout_report_sha,
                "behavior_checkpoint_sha": model_identity["checkpoint_sha"],
                "behavior_model_config_sha": model_identity["model_config_sha"],
                "behavior_model_state_sha": model_identity["model_state_sha"],
                "behavior_source_commit": model_identity["source_commit"],
                "behavior_source_tree": model_identity["source_tree"],
                "rollout_evaluator_commit": report["evaluator_source_commit"],
                "rollout_evaluator_tree": report["evaluator_source_tree"],
                "view_producer_commit": producer_commit,
                "view_producer_tree": producer_tree,
                "base_v24_source": base_source,
                "replay": replay,
                "sample_set_sha": canonical_sha256(sample_ids),
                "row_set_sha": canonical_sha256(row_shas),
                "research_only": True,
                "selection_evidence": False,
                "promotion_ready": False,
                "hardware_action_authorized": False,
                "held_out": False,
            }
            self.sample_ids = sample_ids
            self.split = config.split
            self.cell_id = str(cell["cell_id"])
            self.base_root = root
            self.base_identity = dict(base_dataset.identity)
            self._rows = tuple(derived_rows)
        finally:
            base_dataset.close()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self._rows):
            raise IndexError("learner-visited row index is out of bounds")
        row = self._rows[index]
        return {
            **{name: row[name].copy() for name in ROW_TENSOR_NAMES},
            **{name: row[name] for name in ROW_METADATA_NAMES},
        }

    def close(self) -> None:
        self._rows = ()


__all__ = [
    "LEARNER_VISITED_REPORT_VIEW_SCHEMA",
    "Learner_visited_report_view_config_v1",
    "Learner_visited_report_dataset",
]
