"""Complete fixed Stage 2 recipe and plans, separate from execution authority.

The recipe is traced to exact historical receipt e713e863..., whose training
and model configurations explicitly use unobserved_zero_v1. The original
oracle-history data view is retained. That receipt is a recipe reference only:
it is not V2 evidence, authorization, or a source of reusable normalization.
No historical matched_behavior_v1 constant or generic training hold is changed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
from typing import Any

from ml_recon.data_v2_4.hashing import sha256_json
from ml_recon.data_v2_4.sample_keys import V24SampleKey, make_sample_key
from ml_recon.data_v2_4.synthetic.trackb_teacher_policy import (
    TRACKB_TEACHER_POLICY_V2, TRACKB_TEACHER_POLICY_V2_CELL_IDS, TRACKB_TEACHER_POLICY_V2_SHA256,
)
from ml_recon.data_v2_4.version import GENERATOR_SHA
from ml_recon.physics_hybrid_v10.noise_matching import evaluated_severity_matching_kind

from . import pilot_runtime_checks
from .contracts import require_git_oid
from .teacher_diagnostic_admission import Teacher_evidence_pin
from .teacher_diagnostic_run_contract import _canonical_directory, reserved_run_ids, run_id_pattern
from .teacher_diagnostic_v2 import Teacher_diagnostic_config_v2


@dataclass(frozen=True)
class Behavior_recipe_v2:
    """The one fixed 4Q phase-policy successor; no editable training axes."""

    schema: str = "TrackBBehaviorRecipeV2"
    system_type: str = "4qubit"
    mode_count: int = 4
    train_samples: int = 1000
    validation_samples: int = 200
    optimizer_steps: int = 191
    dataset_seed: int = 20261001
    model_seed: int = 20261001
    batch_size: int = 64
    width: int = 64
    heads: int = 4
    blocks: int = 2
    learning_rate: float = 0.0005
    gain_loss_weight: float = 0.5
    device: str = "cpu"
    action_loss_normalization_kind: str = "per_round_v1"
    action_projection_kind: str = "train_rms_centered_bounded_gauge_v3"
    action_value_feature_kind: str = "legacy_linear_mean_v0"
    gain_parameterization: str = "standardized_latent_gain_v1"
    gradient_composition_kind: str = "shared_balanced_pcgrad_v1"
    training_sampling_kind: str = "round_coverage_training_sampling_v1"
    source_view_kind: str = "v24_oracle_history_v1"
    history_gain_input_kind: str = "unobserved_zero_v1"
    allow_noncanonical_test_fixture: bool = False
    real_static_source_root: str = ""
    pilot_config_schema: str = "TrackBV24TenRoundPilotConfigV1"
    history_rounds: int = 10
    history_policy: str = TRACKB_TEACHER_POLICY_V2
    history_policy_sha256: str = TRACKB_TEACHER_POLICY_V2_SHA256
    generator_sha: str = GENERATOR_SHA
    generator_profile_id: str = "LEGACY_V240_GLOBAL_PAULI"
    cell_id: str = TRACKB_TEACHER_POLICY_V2_CELL_IDS[4]
    dataset_family: str = "feedback_history"
    measurement_domain: str = "global_pauli_v24"
    measurement_schedule: str = "phase_sensitive_w_v1"
    noise_family: str = "feedback_mixed"
    eta: float = 0.08
    kappa: float = 0.15
    severity_matching_kind: str = evaluated_severity_matching_kind
    shot_mode: str = "S_setting"
    shot_budget: int = 256
    evaluation_history_gain_input_kind: str = "unobserved_zero_v1"
    evaluation_action_precision: str = "float32_common_model_zero_teacher_v1"
    history_distribution_shift: str = "UNOBSERVED_ZERO_GAIN_INPUT_MATCHED_LEARNER_STATE_SHIFT_REMAINS"
    reference_behavior_receipt_sha256: str = "e713e8638c5e8b0b0b864eed75c9fe53e086d6a5ddb2b7794cafb2ce549c7834"
    reference_behavior_source_commit: str = "765aec8f31257e55a247d908d10cc57583014bef"
    reference_behavior_source_tree: str = "deecfcac9eda3bc60fc680d02ac9723602a1ba16"
    reference_dataset_recipe_sha256: str = "c84e798fd99da81084977d09cb1c6ef64291d5a850c5859cf9e4ea9d3aa1905b"
    reference_dataset_source_commit: str = "d6c4f0c8ac80dcf482deca8422d81d0bd64604e3"
    reference_dataset_source_tree: str = "03f52b4f4487ed80ce6c0bdce02d26d3b4786278"

    def validate(self) -> None:
        for field in fields(self):
            actual = getattr(self, field.name)
            if type(actual) is not type(field.default) or actual != field.default:
                raise ValueError(f"fixed Stage 2 recipe differs at {field.name}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def teacher_config(self) -> Teacher_diagnostic_config_v2:
        self.validate()
        return Teacher_diagnostic_config_v2(
            system_type=self.system_type, shot_mode=self.shot_mode, shot_budget=self.shot_budget,
            noise_family=self.noise_family, eta=self.eta, kappa=self.kappa,
            history_policy=self.history_policy, history_policy_sha256=self.history_policy_sha256,
            severity_matching_kind=self.severity_matching_kind)

    def dataset_cell(self) -> dict[str, Any]:
        self.validate()
        return {"cell_id": self.cell_id, "dataset_family": self.dataset_family, "N": self.mode_count,
                "system_type": self.system_type, "measurement_domain": self.measurement_domain,
                "train": self.train_samples, "validation": self.validation_samples,
                "noise_family": self.noise_family, "eta": self.eta, "kappa": self.kappa,
                "shot_mode": self.shot_mode, "shot_budget": self.shot_budget,
                "measurement_schedule": self.measurement_schedule, "history_rounds": self.history_rounds,
                "history_policy": self.history_policy, "history_policy_sha256": self.history_policy_sha256,
                "severity_matching_kind": self.severity_matching_kind}

    def dataset_recipe(self) -> dict[str, Any]:
        return {"cell": {**self.dataset_cell(), "shot_mode": "fixed_per_setting",
                         "schedule_id": "full_pauli", "storage_tier": "CORE"},
                "dataset_seed": self.dataset_seed, "generator_profile_id": self.generator_profile_id}

    def pilot_kwargs(self) -> dict[str, Any]:
        """Return the full non-output pilot recipe; this does not open its V2 hold."""
        self.validate()
        names = ("system_type", "train_samples", "validation_samples", "optimizer_steps", "batch_size",
                 "learning_rate", "device", "width", "heads", "blocks", "real_static_source_root",
                 "gain_loss_weight", "gradient_composition_kind", "action_value_feature_kind",
                 "action_projection_kind", "action_loss_normalization_kind", "gain_parameterization",
                 "training_sampling_kind", "allow_noncanonical_test_fixture", "history_gain_input_kind",
                 "source_view_kind", "history_policy", "history_policy_sha256")
        return {**{name: getattr(self, name) for name in names}, "seed": self.model_seed,
                "schema": self.pilot_config_schema}


@dataclass(frozen=True)
class Behavior_stage_plan_v2:
    expected_source: tuple[str, str]
    run_id: str
    trusted_output_parent: Path
    output_root: Path
    protected_input_roots: tuple[Path, ...]
    teacher_evidence_pin: Teacher_evidence_pin
    recipe: Behavior_recipe_v2
    train_keys: tuple[V24SampleKey, ...]
    validation_keys: tuple[V24SampleKey, ...]
    evaluation_replicate_index: int

    @property
    def dataset_root(self) -> Path:
        return self.output_root / "dataset"

    @property
    def pilot_root(self) -> Path:
        return self.output_root / "pilot"

    @property
    def measurement_root(self) -> Path:
        return self.output_root / "measurement"

    def validate_identity(self) -> None:
        """Validate immutable choices, including every TRAIN and validation key."""
        if type(self.recipe) is not Behavior_recipe_v2 or type(self.teacher_evidence_pin) is not Teacher_evidence_pin:
            raise ValueError("Stage 2 requires a fixed recipe and independently pinned teacher evidence")
        self.recipe.validate()
        self.teacher_evidence_pin.validate()
        if type(self.expected_source) is not tuple or len(self.expected_source) != 2:
            raise ValueError("Stage 2 requires one exact final source commit/tree")
        for value in self.expected_source:
            require_git_oid(value, "behavior_final_source")
        if self.teacher_evidence_pin.expected_source != self.expected_source:
            raise ValueError("teacher, data and behavior must use the same exact final source")
        expected_teacher_config = self.recipe.teacher_config()
        actual_teacher_config = self.teacher_evidence_pin.plan.config
        # Observation replicate is independently declared by the teacher plan.
        if any(getattr(actual_teacher_config, name) != getattr(expected_teacher_config, name)
               for name in ("system_type", "shot_mode", "shot_budget", "noise_family", "eta", "kappa",
                            "history_policy", "history_policy_sha256", "severity_matching_kind")):
            raise ValueError("teacher diagnostic recipe differs from the fixed Stage 2 physical recipe")
        for split, keys, count in (("train", self.train_keys, self.recipe.train_samples),
                                    ("validation", self.validation_keys, self.recipe.validation_samples)):
            if type(keys) is not tuple or len(keys) != count:
                raise ValueError("Stage 2 requires explicit complete TRAIN1000 and validation200 key tuples")
            for index, key in enumerate(keys):
                if type(key) is not V24SampleKey:
                    raise ValueError("Stage 2 key must be the exact canonical sample-key type")
                expected_teacher_config.validate(key)
                expected_key = make_sample_key(
                    global_seed=self.recipe.dataset_seed, split=split, dataset_family=self.recipe.dataset_family,
                    N=self.recipe.mode_count, index=index, generator_sha=self.recipe.generator_sha, cell_id=self.recipe.cell_id)
                if key != expected_key:
                    raise ValueError("Stage 2 keys must exactly cover the registered ordered split and dataset seed")
        if type(self.evaluation_replicate_index) is not int or self.evaluation_replicate_index < 0:
            raise ValueError("evaluation replicate must be an explicit nonnegative integer")
        if type(self.run_id) is not str or not run_id_pattern.fullmatch(self.run_id) or self.run_id in reserved_run_ids:
            raise ValueError("Stage 2 run_id must be a bounded snake_case basename")
        if type(self.protected_input_roots) is not tuple:
            raise ValueError("protected input roots must be an explicit tuple")
        for path in (self.trusted_output_parent, self.output_root, *self.protected_input_roots):
            if (not isinstance(path, Path) or not path.is_absolute() or len(path.drive) != 2
                    or path.drive[1] != ":" or str(path).startswith(("\\\\", "//")) or ".." in path.parts):
                raise ValueError("Stage 2 paths must be canonical local Windows paths")
        if (self.trusted_output_parent.parent == self.trusted_output_parent or self.trusted_output_parent == Path.home()
                or self.output_root != self.trusted_output_parent / self.run_id):
            raise ValueError("Stage 2 output must be the exact fresh child of its task-owned parent")
        for root in (pilot_runtime_checks.PILOT_REPOSITORY_ROOT, self.teacher_evidence_pin.root, *self.protected_input_roots):
            if self.output_root == root or root in self.output_root.parents or self.output_root in root.parents:
                raise ValueError("Stage 2 output overlaps source or immutable inputs")

    def validate(self) -> None:
        self.validate_identity()
        for path in (self.trusted_output_parent, pilot_runtime_checks.PILOT_REPOSITORY_ROOT,
                     self.teacher_evidence_pin.root, *self.protected_input_roots):
            _canonical_directory(path)
        if os.path.lexists(self.output_root):
            raise ValueError("Stage 2 output already exists; no reuse or resume is admitted")

    def portable_payload(self) -> dict[str, Any]:
        self.validate_identity()
        return {"schema": "TrackBBehaviorStagePlanV2", "run_id": self.run_id,
                "source_commit": self.expected_source[0], "source_tree": self.expected_source[1],
                "recipe": self.recipe.to_dict(), "recipe_sha256": self.recipe.sha256(),
                "teacher_evidence": self.teacher_evidence_pin.identity_summary(),
                "train_keys": [asdict(key) for key in self.train_keys],
                "validation_keys": [asdict(key) for key in self.validation_keys],
                "evaluation_replicate_index": self.evaluation_replicate_index,
                "evaluation_pairing_kind": "common_random_variates_v1",
                "dataset_recipe_sha256": sha256_json(self.recipe.dataset_recipe()),
                "location_binding_sha256": sha256_json({"output_root": str(self.output_root),
                    "trusted_output_parent": str(self.trusted_output_parent),
                    "protected_input_roots": [str(path) for path in self.protected_input_roots]}),
                "research_only": True, "execution_admission": "NOT_GRANTED_BY_PLAN",
                "external_source_review": "NOT_CERTIFIED_BY_PLAN", "model_measurement": "NOT_PERFORMED",
                "gate_decision": "NOT_PERFORMED", "selection_evidence": False,
                "promotion_ready": False, "hardware_action_authorized": False}

    def sha256(self) -> str:
        return sha256_json(self.portable_payload())


__all__ = ["Behavior_recipe_v2", "Behavior_stage_plan_v2"]
