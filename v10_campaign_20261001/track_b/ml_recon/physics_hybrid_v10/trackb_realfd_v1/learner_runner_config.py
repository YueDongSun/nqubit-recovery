"""Strict, source-bound configuration for the research-only Track B learner run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import os
import re

from ml_recon.data_v2_4.hashing import sha256_file

from .contracts import LEARNER_VISITED_HISTORY_VIEW, require_git_oid, require_sha256
from .inverse_transformer import UNOBSERVED_ZERO_HISTORY_GAIN_INPUT
from .pilot_runtime_checks import PILOT_REPOSITORY_ROOT


LEARNER_RUNNER_SCHEMA = "TrackBLearnerTrainingRunConfigV1"
LEARNER_RUNNER_V2_SCHEMA = "TrackBLearnerTrainingRunConfigV2"
RUN_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _absolute_canonical_path(value: object, name: str) -> Path:
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    path = Path(value)
    if path.resolve() != path or os.path.normcase(str(path)) != os.path.normcase(os.path.normpath(value)):
        raise ValueError(f"{name} must not use a path alias")
    return path


@dataclass(frozen=True)
class Learner_training_run_config_v1:
    """One explicit, immutable input set and one exclusively claimed output."""

    schema: str
    run_id: str
    expected_source_commit: str
    expected_source_tree: str
    run_root: str
    existing_v24_dataset_root: str
    behavior_pilot_receipt_path: str
    behavior_pilot_receipt_sha: str
    learner_visited_train_report_path: str
    learner_visited_train_report_sha: str
    learner_visited_validation_report_path: str
    learner_visited_validation_report_sha: str
    system_type: str
    device: str
    source_view_kind: str
    history_gain_input_kind: str
    train_samples: int
    validation_samples: int
    history_rounds: int
    optimizer_steps: int

    def validate(self, *, inspection: bool = False) -> None:
        if type(inspection) is not bool:
            raise TypeError("inspection must be a strict boolean")
        supported_schemas = {LEARNER_RUNNER_SCHEMA, LEARNER_RUNNER_V2_SCHEMA}
        if type(self.schema) is not str or self.schema not in supported_schemas:
            raise ValueError("unsupported learner runner schema")
        if type(self) is Learner_training_run_config_v1 and self.schema != LEARNER_RUNNER_SCHEMA:
            raise ValueError("V1 learner config cannot declare another schema")
        if type(self.run_id) is not str or RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a lowercase snake_case identifier")
        require_git_oid(self.expected_source_commit, "expected_source_commit")
        require_git_oid(self.expected_source_tree, "expected_source_tree")
        for name in (
            "behavior_pilot_receipt_sha", "learner_visited_train_report_sha",
            "learner_visited_validation_report_sha",
        ):
            require_sha256(getattr(self, name), name)
        fixed = {
            "system_type": "4qubit", "device": "cpu",
            "source_view_kind": LEARNER_VISITED_HISTORY_VIEW,
            "history_gain_input_kind": UNOBSERVED_ZERO_HISTORY_GAIN_INPUT,
            "train_samples": 1_000, "validation_samples": 200,
            "history_rounds": 10, "optimizer_steps": 191,
        }
        for name, expected in fixed.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"{name} must equal the registered learner budget/value {expected!r}")
        paths = {
            name: _absolute_canonical_path(getattr(self, name), name)
            for name in (
                "run_root", "existing_v24_dataset_root", "behavior_pilot_receipt_path",
                "learner_visited_train_report_path", "learner_visited_validation_report_path",
            )
        }
        output = paths["run_root"]
        if output.name != self.run_id:
            raise ValueError("run_root basename must equal run_id")
        if output == PILOT_REPOSITORY_ROOT or PILOT_REPOSITORY_ROOT in output.parents:
            raise ValueError("run_root must be outside the pilot source repository")
        if inspection:
            if output.exists() and not output.is_dir():
                raise ValueError("inspected run_root must be a directory when present")
        elif output.exists() or not output.parent.is_dir() or output.parent.resolve() != output.parent:
            raise ValueError("run_root must be new beneath an existing non-aliased parent")
        source_root = paths["existing_v24_dataset_root"]
        if not inspection and not source_root.is_dir():
            raise ValueError("existing_v24_dataset_root must exist")
        inputs = [source_root]
        for name, path in paths.items():
            if name in ("run_root", "existing_v24_dataset_root"):
                continue
            if not inspection:
                if not path.is_file():
                    raise ValueError(f"{name} must be an existing file")
                sha_name = name.removesuffix("_path") + "_sha"
                expected_sha = getattr(self, sha_name)
                if name == "behavior_pilot_receipt_path" and self.schema == LEARNER_RUNNER_V2_SCHEMA:
                    from .learner_recipe_contract import resolve_matched_behavior_recipe

                    resolve_matched_behavior_recipe(str(path), expected_sha)
                elif sha256_file(path) != expected_sha:
                    raise ValueError(f"{name} differs from its expected SHA")
            inputs.append(path)
        if len(set(inputs)) != len(inputs):
            raise ValueError("learner inputs must be distinct")
        if any(output == item or output in item.parents or item in output.parents for item in inputs):
            raise ValueError("run_root and input paths must not overlap")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Learner_training_run_config_v2(Learner_training_run_config_v1):
    """V2 opts into the SHA-bound matched behavior recipe."""

    recipe_kind: str

    def validate(self, *, inspection: bool = False) -> None:
        if self.schema != LEARNER_RUNNER_V2_SCHEMA:
            raise ValueError("V2 learner config has an unsupported schema")
        if self.recipe_kind != "matched_behavior_v1":
            raise ValueError("V2 learner config requires matched_behavior_v1")
        super().validate(inspection=inspection)


def load_learner_training_config(
    path_value: str, *, inspection: bool = False,
) -> Learner_training_run_config_v1:
    """Read an exact JSON object; no optional authority or bypass fields exist."""
    path = _absolute_canonical_path(path_value, "config")
    if not path.is_file():
        raise ValueError("config must be an existing file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("config must be UTF-8 JSON") from error
    if type(payload) is not dict:
        raise ValueError("config must be an exact JSON object")
    schema = payload.get("schema")
    if type(schema) is not str:
        raise ValueError("unsupported learner runner schema")
    config_type = {
        LEARNER_RUNNER_SCHEMA: Learner_training_run_config_v1,
        LEARNER_RUNNER_V2_SCHEMA: Learner_training_run_config_v2,
    }.get(schema)
    if config_type is None:
        raise ValueError("unsupported learner runner schema")
    expected = {field.name for field in fields(config_type)}
    if set(payload) != expected:
        raise ValueError(f"config fields differ; missing={sorted(expected - set(payload))}, unknown={sorted(set(payload) - expected)}")
    config = config_type(**payload)
    config.validate(inspection=inspection)
    if path == Path(config.run_root) or Path(config.run_root) in path.parents:
        raise ValueError("config cannot be inside run_root")
    return config


__all__ = ["LEARNER_RUNNER_SCHEMA", "LEARNER_RUNNER_V2_SCHEMA",
           "Learner_training_run_config_v1", "Learner_training_run_config_v2",
           "load_learner_training_config"]
