"""Fixed Stage 3 learner recipe and source-bound view plan contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import re
from typing import Any

from ml_recon.data_v2_4.hashing import sha256_json
from ml_recon.data_v2_4.sample_keys import V24SampleKey, make_sample_key

from .behavior_v2_contract import Behavior_recipe_v2
from .contracts import require_git_oid, require_sha256
from .pilot_runtime_checks import PILOT_REPOSITORY_ROOT


LEARNER_STAGE_RECIPE_SCHEMA_V2 = "TrackBLearnerStageRecipeV2"
LEARNER_STAGE_RECIPE_KIND_V2 = "matched_behavior_v2"
LEARNER_VIEW_PLAN_SCHEMA_V2 = "TrackBLearnerViewPlanV2"
LEARNER_VIEW_KIND_V2 = "learner_visited_history_v2"
LEARNER_VISITED_ROW_SCHEMA_V2 = "TrackBLearnerVisitedHistoryRowV2"
LEARNER_HISTORY_KIND_V2 = "MODEL_BRANCH_EXECUTED_ACTIONS_UNOBSERVED_ZERO_GAIN_V2"
LEARNER_LABEL_KIND_V2 = "TEACHER_V2_ACTION_AT_LEARNER_PRE_STATE_V2"
LEARNER_TEACHER_GAIN_KIND_V2 = "TEACHER_V2_TRANSITION_GAIN_NOT_MODEL_ACTION_GAIN"
LEARNER_REPLICATE_INDEX_V2 = 0
LEARNER_PAIRING_KIND_V2 = "common_random_variates_v1"
_run_id_pattern = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_behavior_recipe_sha256_v2 = Behavior_recipe_v2().sha256()


@dataclass(frozen=True)
class Learner_stage_recipe_v2:
    """One fixed learner-state imitation recipe; Stage 2 remains unchanged."""

    schema: str = LEARNER_STAGE_RECIPE_SCHEMA_V2
    recipe_kind: str = LEARNER_STAGE_RECIPE_KIND_V2
    behavior_recipe_sha256: str = _behavior_recipe_sha256_v2
    source_view_kind: str = LEARNER_VIEW_KIND_V2
    history_gain_input_kind: str = "unobserved_zero_v1"
    history_kind: str = LEARNER_HISTORY_KIND_V2
    label_kind: str = LEARNER_LABEL_KIND_V2
    teacher_gain_kind: str = LEARNER_TEACHER_GAIN_KIND_V2
    train_samples: int = 1_000
    validation_samples: int = 200
    optimizer_steps: int = 191
    history_rounds: int = 10
    batch_size: int = 64
    evaluation_replicate_index: int = LEARNER_REPLICATE_INDEX_V2
    pairing_kind: str = LEARNER_PAIRING_KIND_V2
    normalization_source: str = "FRESH_CURRENT_LEARNER_TRAIN_VIEWS_ONLY"
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def validate(self) -> None:
        expected = Learner_stage_recipe_v2.__dataclass_fields__
        for name, field in expected.items():
            value = getattr(self, name)
            default = field.default
            if type(value) is not type(default) or value != default:
                raise ValueError(f"fixed Stage 3 learner recipe differs at {name}")
        behavior_recipe = Behavior_recipe_v2()
        behavior_recipe.validate()
        if (self.behavior_recipe_sha256 != behavior_recipe.sha256()
                or self.train_samples != behavior_recipe.train_samples
                or self.validation_samples != behavior_recipe.validation_samples
                or self.optimizer_steps != behavior_recipe.optimizer_steps
                or self.batch_size != behavior_recipe.batch_size
                or self.history_rounds != behavior_recipe.history_rounds
                or self.evaluation_replicate_index != LEARNER_REPLICATE_INDEX_V2):
            raise ValueError("Stage 3 learner recipe differs from the registered Stage 2 numerical budget")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "behavior_recipe": Behavior_recipe_v2().to_dict(),
        }

    def sha256(self) -> str:
        return sha256_json(self.to_dict())


def _canonical_local_path(value: object, name: str) -> Path:
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    path = Path(value)
    normalized = os.path.normcase(os.path.normpath(value))
    if (path.resolve() != path or os.path.normcase(str(path)) != normalized
            or len(path.drive) != 2 or path.drive[1] != ":"
            or str(path).startswith(("\\\\", "//")) or ".." in path.parts):
        raise ValueError(f"{name} must be a canonical local Windows path")
    return path


def _registered_sample_keys(split: str) -> tuple[V24SampleKey, ...]:
    recipe = Behavior_recipe_v2()
    recipe.validate()
    if split == "train":
        count = recipe.train_samples
    elif split == "validation":
        count = recipe.validation_samples
    else:
        raise ValueError("learner view plan only supports TRAIN and validation splits")
    return tuple(
        make_sample_key(
            global_seed=recipe.dataset_seed,
            split=split,
            dataset_family=recipe.dataset_family,
            N=recipe.mode_count,
            index=index,
            generator_sha=recipe.generator_sha,
            cell_id=recipe.cell_id,
        )
        for index in range(count)
    )


def registered_sample_set_sha256(split: str) -> str:
    return sha256_json([key.sample_id for key in _registered_sample_keys(split)])


@dataclass(frozen=True)
class Learner_view_plan_v2:
    """Complete ordered split identities for new Stage 3 learner views."""

    run_id: str
    expected_consumer_source: tuple[str, str]
    behavior_source: tuple[str, str]
    behavior_stage_root: str
    trusted_output_parent: str
    output_root: str
    behavior_stage_plan_sha256: str
    behavior_ready_sha256: str
    behavior_pilot_receipt_sha256: str
    behavior_checkpoint_sha256: str
    behavior_dataset_receipt_sha256: str
    behavior_measurement_receipt_sha256: str
    stage2_completion_observation_sha256: str
    train_sample_set_sha256: str
    validation_sample_set_sha256: str
    recipe: Learner_stage_recipe_v2
    schema: str = LEARNER_VIEW_PLAN_SCHEMA_V2

    def sample_key_for(self, split: str, index: int) -> V24SampleKey:
        recipe = Behavior_recipe_v2()
        recipe.validate()
        if split == "train":
            count = self.recipe.train_samples
        elif split == "validation":
            count = self.recipe.validation_samples
        else:
            raise ValueError("learner view plan only supports TRAIN and validation splits")
        if type(index) is not int or not 0 <= index < count:
            raise ValueError("learner view sample index is outside the registered split")
        return make_sample_key(
            global_seed=recipe.dataset_seed,
            split=split,
            dataset_family=recipe.dataset_family,
            N=recipe.mode_count,
            index=index,
            generator_sha=recipe.generator_sha,
            cell_id=recipe.cell_id,
        )

    def registered_sample_set_sha256(self, split: str) -> str:
        if split not in ("train", "validation"):
            raise ValueError("learner view plan only supports TRAIN and validation splits")
        return registered_sample_set_sha256(split)

    def require_registered_sample_key(self, sample_key: V24SampleKey) -> tuple[str, int]:
        if type(sample_key) is not V24SampleKey:
            raise ValueError("learner view producer requires the registered V2.4 sample-key type")
        for split in ("train", "validation"):
            recipe = Behavior_recipe_v2()
            prefix = (
                f"v2_4:{recipe.dataset_family}:{recipe.cell_id}:{split}:N{recipe.mode_count}:"
            )
            if not sample_key.sample_id.startswith(prefix):
                continue
            suffix = sample_key.sample_id[len(prefix):]
            if len(suffix) != 12 or not suffix.isdigit():
                break
            index = int(suffix)
            if index < (self.recipe.train_samples if split == "train" else self.recipe.validation_samples):
                expected_key = self.sample_key_for(split, index)
                if sample_key == expected_key:
                    return split, index
            break
        raise ValueError("learner view sample key is not in the complete registered split plan")

    def validate_identity(self) -> None:
        if self.schema != LEARNER_VIEW_PLAN_SCHEMA_V2:
            raise ValueError("learner view plan schema is unsupported")
        if type(self.recipe) is not Learner_stage_recipe_v2:
            raise ValueError("learner view plan requires the fixed matched_behavior_v2 recipe")
        self.recipe.validate()
        if type(self.run_id) is not str or _run_id_pattern.fullmatch(self.run_id) is None:
            raise ValueError("learner view run_id must be a bounded snake_case identifier")
        for name, source in (
            ("expected_consumer_source", self.expected_consumer_source),
            ("behavior_source", self.behavior_source),
        ):
            if type(source) is not tuple or len(source) != 2:
                raise ValueError(f"{name} must bind one exact commit/tree pair")
            for value in source:
                require_git_oid(value, name)
        for name in (
            "behavior_stage_plan_sha256", "behavior_ready_sha256",
            "behavior_pilot_receipt_sha256", "behavior_checkpoint_sha256",
            "behavior_dataset_receipt_sha256", "behavior_measurement_receipt_sha256",
            "stage2_completion_observation_sha256",
            "train_sample_set_sha256", "validation_sample_set_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (self.train_sample_set_sha256 != self.registered_sample_set_sha256("train")
                or self.validation_sample_set_sha256 != self.registered_sample_set_sha256("validation")):
            raise ValueError("learner view sample sets differ from the registered ordered TRAIN/validation keys")
        stage_root = _canonical_local_path(self.behavior_stage_root, "behavior_stage_root")
        output_parent = _canonical_local_path(self.trusted_output_parent, "trusted_output_parent")
        output_root = _canonical_local_path(self.output_root, "output_root")
        if (stage_root == Path(stage_root.anchor)
                or stage_root == Path.home()
                or output_parent == Path(output_parent.anchor)
                or output_parent == Path.home()
                or not output_parent.is_dir()
                or not stage_root.is_dir()):
            raise ValueError("learner view inputs and output parent must be narrow existing directories")
        if output_root != output_parent / self.run_id:
            raise ValueError("learner view output must be the exact run_id child of its trusted parent")
        for protected_root in (stage_root, PILOT_REPOSITORY_ROOT):
            if (output_root == protected_root or output_root in protected_root.parents
                    or protected_root in output_root.parents):
                raise ValueError("learner view output must be isolated from Stage 2 inputs and source")

    def validate(self) -> None:
        self.validate_identity()
        if os.path.lexists(self.output_root):
            raise ValueError("learner view output must be a fresh root; reuse/resume is not admitted")

    def portable_payload(self) -> dict[str, Any]:
        self.validate_identity()
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "consumer_source": {"commit": self.expected_consumer_source[0], "tree": self.expected_consumer_source[1]},
            "behavior_source": {"commit": self.behavior_source[0], "tree": self.behavior_source[1]},
            "behavior_stage_plan_sha256": self.behavior_stage_plan_sha256,
            "behavior_ready_sha256": self.behavior_ready_sha256,
            "behavior_pilot_receipt_sha256": self.behavior_pilot_receipt_sha256,
            "behavior_checkpoint_sha256": self.behavior_checkpoint_sha256,
            "behavior_dataset_receipt_sha256": self.behavior_dataset_receipt_sha256,
            "behavior_measurement_receipt_sha256": self.behavior_measurement_receipt_sha256,
            "stage2_completion_observation_sha256": self.stage2_completion_observation_sha256,
            "recipe": self.recipe.to_dict(),
            "recipe_sha256": self.recipe.sha256(),
            "train_source": {
                "split": "train", "sample_count": self.recipe.train_samples,
                "sample_set_sha256": self.train_sample_set_sha256,
            },
            "validation_source": {
                "split": "validation", "sample_count": self.recipe.validation_samples,
                "sample_set_sha256": self.validation_sample_set_sha256,
            },
            "evaluation_replicate_index": self.recipe.evaluation_replicate_index,
            "pairing_kind": LEARNER_PAIRING_KIND_V2,
            "location_binding_sha256": sha256_json({
                "behavior_stage_root": self.behavior_stage_root,
                "trusted_output_parent": self.trusted_output_parent,
                "output_root": self.output_root,
            }),
            "research_only": True,
            "learner_views_admitted": False,
            "training_admitted": False,
            "scientific_gate": "NOT_PERFORMED",
            "selection_evidence": False,
            "promotion_ready": False,
            "hardware_action_authorized": False,
        }

    def sha256(self) -> str:
        return sha256_json(self.portable_payload())


__all__ = [
    "LEARNER_STAGE_RECIPE_SCHEMA_V2",
    "LEARNER_STAGE_RECIPE_KIND_V2",
    "LEARNER_VIEW_PLAN_SCHEMA_V2",
    "LEARNER_VIEW_KIND_V2",
    "LEARNER_VISITED_ROW_SCHEMA_V2",
    "LEARNER_HISTORY_KIND_V2",
    "LEARNER_LABEL_KIND_V2",
    "LEARNER_TEACHER_GAIN_KIND_V2",
    "registered_sample_set_sha256",
    "Learner_stage_recipe_v2",
    "Learner_view_plan_v2",
]
