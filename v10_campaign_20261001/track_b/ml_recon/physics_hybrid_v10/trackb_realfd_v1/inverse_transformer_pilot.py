"""Source-bound V2.4 ten-round inverse-Transformer engineering pilot.

The pilot trains a logical Track B controller only from explicit V2.4
``feedback_history`` records.  It uses teacher-forced V2.4 histories for the
ten-round validation readout; it never materializes RF/AWG data or asserts a
real feedback transition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import hashlib
import io
import json
import math
import os
import random

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from ml_recon.data_v2_4.hashing import sha256_file
from ml_recon.data_v2_4.loaders import V24TrainingDataset, validate_dataset_root
from ml_recon.data_v2_4.schema_registry import schema_sha256
from ml_recon.data_v2_4.synthetic.generator import (
    SMOKE_PROFILE_LEGACY_V240_GLOBAL_PAULI,
    generate_smoke_build,
)
from ml_recon.data_v2_4.version import GENERATOR_SHA
from ml_recon.physics_hybrid_v10.noise_matching import evaluated_severity_matching_kind
from ml_recon.data_v2_4.synthetic.trackb_teacher_policy import (
    TRACKB_TEACHER_POLICY_V1,
    TRACKB_TEACHER_POLICY_V2,
    TRACKB_TEACHER_POLICY_V2_CELL_IDS,
    TRACKB_TEACHER_POLICY_V2_SHA256,
    validate_trackb_v2_cell,
)

from .balanced_gradients import compose_balanced_shared_gradients
from .contracts import (
    LEARNER_VISITED_HISTORY_VIEW,
    LEARNER_VISITED_ROW_SCHEMA,
    V24_ORACLE_HISTORY_VIEW,
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
)
from .learner_visited_contract import ROW_TENSOR_NAMES, validate_learner_visited_tensors
from .inverse_transformer import (
    BOUNDED_GAUGE_PROJECTION,
    CENTERED_LOGIT_BOUNDED_GAUGE_PROJECTION,
    TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION,
    LEGACY_LINEAR_MEAN_ACTION_VALUE,
    HISTORY_GAIN_INPUT_KINDS,
    LEGACY_ORACLE_HISTORY_GAIN_INPUT,
    UNOBSERVED_ZERO_HISTORY_GAIN_INPUT,
    V24_HISTORY_GAIN_FEATURE_INDEX,
    RAW_GAIN_PARAMETERIZATION_V1,
    STANDARDIZED_LATENT_GAIN_PARAMETERIZATION_V1,
    TrackBInverseTransformerConfigV1,
    TrackBInverseTransformerV1,
    measure_inverse_transformer_dependency,
    require_gain_parameterization_binding,
    require_action_decode_scale_binding,
)
from .ten_round_guidance import TEN_ROUND_COUNT
from .pilot_history_dataset import Pilot_history_dataset
from .pilot_dataset_admission import (
    MAX_PILOT_MANIFEST_BYTES,
    Pilot_dataset_admission,
    admit_pilot_dataset,
)
from ml_recon.data_v2_4.admission_snapshot import read_canonical_snapshot
from .pilot_runtime_checks import (
    checked_optimizer_step,
    pilot_source_identity,
    require_unchanged_pilot_source,
)
from .pilot_training_core import (
    Pilot_training_dependencies,
    initialize_pilot_model,
    train_pilot_core,
)
from .learner_pilot_artifact_writer import Learner_pilot_artifact_writer


PILOT_SCHEMA = "TrackBV24TenRoundInverseTransformerPilotV1"
DEFAULT_OPTIMIZER_STEPS = 191
DEFAULT_GAIN_LOSS_WEIGHT = 0.5
STANDARDIZED_HUBER_DELTA = 1.0
WEIGHTED_SUM_GRADIENT_COMPOSITION = "weighted_sum_v1"
SHARED_PCGD_GRADIENT_COMPOSITION = "shared_pcgrad_v1"
SHARED_BALANCED_PCGD_GRADIENT_COMPOSITION = "shared_balanced_pcgrad_v1"
GRADIENT_COMPOSITION_KINDS = (
    WEIGHTED_SUM_GRADIENT_COMPOSITION,
    SHARED_PCGD_GRADIENT_COMPOSITION,
    SHARED_BALANCED_PCGD_GRADIENT_COMPOSITION,
)
GLOBAL_ACTION_LOSS_NORMALIZATION = "global_v1"
PER_ROUND_ACTION_LOSS_NORMALIZATION = "per_round_v1"
ACTION_LOSS_NORMALIZATION_KINDS = (
    GLOBAL_ACTION_LOSS_NORMALIZATION,
    PER_ROUND_ACTION_LOSS_NORMALIZATION,
)
RANDOM_TRAINING_SAMPLING = "random_training_sampling_v1"
ROUND_COVERAGE_TRAINING_SAMPLING = "round_coverage_training_sampling_v1"
TRAINING_SAMPLING_KINDS = (
    RANDOM_TRAINING_SAMPLING,
    ROUND_COVERAGE_TRAINING_SAMPLING,
)
LEARNER_VIEW_CONFIG_FIELDS = (
    "behavior_pilot_receipt_path", "behavior_pilot_receipt_sha",
    "learner_visited_train_report_path", "learner_visited_train_report_sha",
    "learner_visited_validation_report_path", "learner_visited_validation_report_sha",
)


@dataclass(frozen=True)
class TrackBV24TenRoundPilotConfigV1:
    """Bounded current-source data/model pilot configuration."""

    output_root: str
    system_type: str = "4qubit"
    train_samples: int = 1_000
    validation_samples: int = 200
    optimizer_steps: int = DEFAULT_OPTIMIZER_STEPS
    batch_size: int = 8
    learning_rate: float = 5e-4
    seed: int = 20261001
    device: str = "cpu"
    width: int = 64
    heads: int = 4
    blocks: int = 2
    real_static_source_root: str = ""
    existing_v24_dataset_root: str = ""
    gain_loss_weight: float = DEFAULT_GAIN_LOSS_WEIGHT
    gradient_composition_kind: str = WEIGHTED_SUM_GRADIENT_COMPOSITION
    action_value_feature_kind: str = LEGACY_LINEAR_MEAN_ACTION_VALUE
    action_projection_kind: str = BOUNDED_GAUGE_PROJECTION
    action_loss_normalization_kind: str = PER_ROUND_ACTION_LOSS_NORMALIZATION
    gain_parameterization: str = RAW_GAIN_PARAMETERIZATION_V1
    training_sampling_kind: str = RANDOM_TRAINING_SAMPLING
    allow_noncanonical_test_fixture: bool = False
    schema: str = "TrackBV24TenRoundPilotConfigV1"
    history_gain_input_kind: str = LEGACY_ORACLE_HISTORY_GAIN_INPUT
    source_view_kind: str = V24_ORACLE_HISTORY_VIEW
    behavior_pilot_receipt_path: str = ""
    behavior_pilot_receipt_sha: str = ""
    learner_visited_train_report_path: str = ""
    learner_visited_train_report_sha: str = ""
    learner_visited_validation_report_path: str = ""
    learner_visited_validation_report_sha: str = ""
    history_policy: str = TRACKB_TEACHER_POLICY_V1
    history_policy_sha256: str = ""

    def validate(self) -> None:
        if not Path(self.output_root).is_absolute():
            raise ValueError("ten-round pilot output_root must be absolute")
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("ten-round pilot supports only 3qubit or 4qubit")
        if self.train_samples <= 0 or self.validation_samples <= 0:
            raise ValueError("ten-round pilot requires nonempty train and validation samples")
        if (not self.allow_noncanonical_test_fixture
                and (self.train_samples != 1_000 or self.validation_samples != 200)):
            raise ValueError("the Track B ten-round pilot requires the registered 1,000/200 data budget")
        if self.optimizer_steps != DEFAULT_OPTIMIZER_STEPS:
            raise ValueError("the bounded Track B pilot must use exactly 191 optimizer steps")
        if self.batch_size <= 0 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("ten-round pilot batch size and learning rate are invalid")
        if not math.isfinite(self.gain_loss_weight) or self.gain_loss_weight <= 0.0:
            raise ValueError("ten-round pilot gain loss weight must be positive")
        if self.gradient_composition_kind not in GRADIENT_COMPOSITION_KINDS:
            raise ValueError("ten-round pilot gradient composition kind is unsupported")
        if self.action_loss_normalization_kind not in ACTION_LOSS_NORMALIZATION_KINDS:
            raise ValueError("ten-round pilot action-loss normalization kind is unsupported")
        if self.training_sampling_kind not in TRAINING_SAMPLING_KINDS:
            raise ValueError("ten-round pilot training sampling kind is unsupported")
        if self.history_gain_input_kind not in HISTORY_GAIN_INPUT_KINDS:
            raise ValueError("ten-round pilot history gain input kind is unsupported")
        if self.source_view_kind == V24_ORACLE_HISTORY_VIEW:
            if any(getattr(self, name) for name in LEARNER_VIEW_CONFIG_FIELDS):
                raise ValueError("V2.4 oracle pilot cannot declare learner-visited artifacts")
        elif self.source_view_kind == LEARNER_VISITED_HISTORY_VIEW:
            if (self.history_gain_input_kind != UNOBSERVED_ZERO_HISTORY_GAIN_INPUT
                    or not self.existing_v24_dataset_root):
                raise ValueError("learner-visited pilot requires zero-gain inputs and a bound V2.4 root")
            if self.train_samples != 1_000 or self.validation_samples != 200:
                raise ValueError("learner-visited training requires the full registered 1,000/200 budget")
            for name in LEARNER_VIEW_CONFIG_FIELDS:
                value = getattr(self, name)
                if type(value) is not str or not value:
                    raise ValueError(f"learner-visited pilot requires {name}")
                if name.endswith("_path"):
                    if not Path(value).is_absolute():
                        raise ValueError(f"{name} must be absolute")
                else:
                    require_sha256(value, name)
        else:
            raise ValueError("ten-round pilot source view is unsupported")
        if self.history_policy == TRACKB_TEACHER_POLICY_V1:
            if type(self.history_policy_sha256) is not str or self.history_policy_sha256:
                raise ValueError("historical V1 pilot cannot declare a policy SHA")
        elif self.history_policy == TRACKB_TEACHER_POLICY_V2:
            if self.history_policy_sha256 != TRACKB_TEACHER_POLICY_V2_SHA256:
                raise ValueError("Track B V2 pilot requires the registered history policy SHA")
            if self.source_view_kind != V24_ORACLE_HISTORY_VIEW:
                raise ValueError("Track B V2 pilot training is not ready for learner-visited labels")
        else:
            raise ValueError("ten-round pilot history policy is unsupported")
        if (self.training_sampling_kind == ROUND_COVERAGE_TRAINING_SAMPLING
                and self.optimizer_steps < math.ceil(self.train_samples / self.batch_size) * TEN_ROUND_COUNT):
            raise ValueError("round coverage training sampling exceeds the authorized optimizer budget")
        if self.action_value_feature_kind != LEGACY_LINEAR_MEAN_ACTION_VALUE:
            raise ValueError("candidate-conditioned values require the separate Track B critic contract")
        if self.action_projection_kind not in (
            BOUNDED_GAUGE_PROJECTION, CENTERED_LOGIT_BOUNDED_GAUGE_PROJECTION,
            TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION,
        ):
            raise ValueError("new Track B pilots require bounded gauge projection")
        if (self.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION
                and self.action_loss_normalization_kind != PER_ROUND_ACTION_LOSS_NORMALIZATION):
            raise ValueError("TRAIN-RMS decoder requires per-round action normalization")
        if type(self.allow_noncanonical_test_fixture) is not bool:
            raise TypeError("allow_noncanonical_test_fixture must be a strict boolean")
        if self.real_static_source_root and not Path(self.real_static_source_root).is_absolute():
            raise ValueError("real_static_source_root must be absolute when supplied")
        if self.existing_v24_dataset_root and not Path(self.existing_v24_dataset_root).is_absolute():
            raise ValueError("existing_v24_dataset_root must be absolute when supplied")
        if self.existing_v24_dataset_root:
            source_root = Path(self.existing_v24_dataset_root).resolve()
            output_root = Path(self.output_root).resolve()
            if output_root == source_root or source_root in output_root.parents:
                raise ValueError("ten-round pilot output cannot be inside its immutable V2.4 source")
        # Validate decoder shape here; the actual scales bind after TRAIN statistics are read.
        TrackBInverseTransformerConfigV1(
            self.system_type,
            int(self.system_type[0]),
            self.width,
            self.heads,
            self.blocks,
            action_value_feature_kind=self.action_value_feature_kind,
            action_projection_kind=self.action_projection_kind,
            gain_parameterization=self.gain_parameterization,
            history_gain_input_kind=self.history_gain_input_kind,
            amplitude_decode_scale_by_round=(
                (1.0,) * TEN_ROUND_COUNT
                if self.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION else ()
            ),
            phase_decode_scale_by_round=(
                (1.0,) * TEN_ROUND_COUNT
                if self.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION else ()
            ),
        ).validate()
        if self.schema != "TrackBV24TenRoundPilotConfigV1":
            raise ValueError("unsupported ten-round pilot configuration schema")


@dataclass(frozen=True)
class TrackBTargetNormalizationV1:
    """Train-split-only target statistics for balanced pilot optimization."""

    amplitude_scale: float
    phase_scale: float
    gain_center: float
    gain_scale: float
    action_loss_normalization_kind: str = GLOBAL_ACTION_LOSS_NORMALIZATION
    amplitude_scale_by_round: tuple[float, ...] = ()
    phase_scale_by_round: tuple[float, ...] = ()
    schema: str = "TrackBTargetNormalizationV1"

    def validate(self) -> None:
        for name in ("amplitude_scale", "phase_scale", "gain_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.gain_center)):
            raise ValueError("gain_center must be finite")
        if self.action_loss_normalization_kind not in ACTION_LOSS_NORMALIZATION_KINDS:
            raise ValueError("unsupported Track B action-loss normalization kind")
        if self.action_loss_normalization_kind == PER_ROUND_ACTION_LOSS_NORMALIZATION:
            if (len(self.amplitude_scale_by_round) != TEN_ROUND_COUNT
                    or len(self.phase_scale_by_round) != TEN_ROUND_COUNT):
                raise ValueError("per-round Track B action normalization requires all ten scales")
            for value in (*self.amplitude_scale_by_round, *self.phase_scale_by_round):
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError("per-round Track B action scale must be finite and positive")
        elif self.amplitude_scale_by_round or self.phase_scale_by_round:
            raise ValueError("global Track B action normalization cannot carry per-round scales")
        if self.schema != "TrackBTargetNormalizationV1":
            raise ValueError("unsupported target normalization schema")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    def action_scales_for_round(self, round_index: int) -> tuple[float, float]:
        """Return train-split-only action scales for one registered round."""
        self.validate()
        if not 0 <= round_index < TEN_ROUND_COUNT:
            raise ValueError("Track B action normalization round index is invalid")
        if self.action_loss_normalization_kind == PER_ROUND_ACTION_LOSS_NORMALIZATION:
            return self.amplitude_scale_by_round[round_index], self.phase_scale_by_round[round_index]
        return self.amplitude_scale, self.phase_scale


def build_trackb_ten_round_v24_plan(
    config: TrackBV24TenRoundPilotConfigV1,
) -> tuple[dict[str, object], ...]:
    """Build the one-cell V2.4 source explicitly used by this bounded pilot."""
    config.validate()
    mode_count = int(config.system_type[0])
    if config.history_policy == TRACKB_TEACHER_POLICY_V2:
        cell_id = TRACKB_TEACHER_POLICY_V2_CELL_IDS[mode_count]
    else:
        cell_id = f"trackb_feedback_history_n{mode_count}"
    cell = {
        "cell_id": cell_id,
        "dataset_family": "feedback_history",
        "N": mode_count,
        "system_type": config.system_type,
        "measurement_domain": "global_pauli_v24",
        "train": config.train_samples,
        "validation": config.validation_samples,
        "noise_family": "feedback_mixed",
        "eta": 0.08,
        "kappa": 0.15,
        "shot_mode": "S_setting",
        "shot_budget": 256,
        "measurement_schedule": "phase_sensitive_w_v1",
        "history_rounds": TEN_ROUND_COUNT,
        "history_policy": config.history_policy,
    }
    if config.history_policy == TRACKB_TEACHER_POLICY_V2:
        cell["history_policy_sha256"] = config.history_policy_sha256
        cell["severity_matching_kind"] = evaluated_severity_matching_kind
        validate_trackb_v2_cell(cell)
    return (cell,)


def generate_trackb_ten_round_v24_dataset(
    config: TrackBV24TenRoundPilotConfigV1,
) -> dict[str, object]:
    """Generate a fresh current-source V2.4 engineering dataset under output_root."""
    config.validate()
    if config.history_policy == TRACKB_TEACHER_POLICY_V2:
        raise RuntimeError(
            "Track B V2 generation is blocked until generic generation performs clean-source admission before writes"
        )
    root = Path(config.output_root).resolve() / "v24_ten_round_data"
    if root.exists() and any(root.iterdir()):
        raise ValueError("ten-round V2.4 dataset root must be empty")
    root.mkdir(parents=True, exist_ok=True)
    result = generate_smoke_build(
        root,
        global_seed=config.seed,
        plan=build_trackb_ten_round_v24_plan(config),
    )
    if not (root / "READY.json").is_file() or result.get("dataset_root") != str(root):
        raise RuntimeError("current V2.4 ten-round dataset generation did not produce a READY root")
    manifest_path = root / "manifests" / f"{result['build_id']}.json"
    manifest = read_canonical_snapshot(
        manifest_path, label="Track B generated build manifest",
        expected_sha=str(result["build_manifest_sha"]),
        max_bytes=MAX_PILOT_MANIFEST_BYTES, reject_reparse=True,
    ).decode()
    return {
        **dict(result), **_bound_dataset_recipe(config, manifest),
        "generator_sha": manifest["generator_sha"],
        "source_commit": manifest["source_commit"],
        "source_tree": manifest["source_tree_sha"],
    }


def _bound_dataset_recipe(
    config: TrackBV24TenRoundPilotConfigV1,
    manifest: Mapping[str, Any],
) -> dict[str, object]:
    """Bind the full physical recipe, keeping data and optimizer seeds distinct."""
    expected = {
        **build_trackb_ten_round_v24_plan(config)[0],
        "shot_mode": "fixed_per_setting", "schedule_id": "full_pauli", "storage_tier": "CORE",
    }
    cells = manifest.get("cells")
    if (type(cells) is not list or len(cells) != 1
            or canonical_json_bytes(cells[0]) != canonical_json_bytes(expected)
            or manifest.get("generator_profile_id") != SMOKE_PROFILE_LEGACY_V240_GLOBAL_PAULI
            or type(manifest.get("global_seed")) is not int):
        raise ValueError("V2.4 dataset recipe differs from the registered Track B pilot")
    if config.history_policy == TRACKB_TEACHER_POLICY_V2:
        validate_trackb_v2_cell(cells[0])
    recipe = {
        "cell": expected,
        "dataset_seed": manifest["global_seed"],
        "generator_profile_id": manifest["generator_profile_id"],
    }
    return {
        **recipe,
        "dataset_recipe_sha": canonical_sha256(recipe),
        **({"history_policy": TRACKB_TEACHER_POLICY_V2,
            "history_policy_sha256": TRACKB_TEACHER_POLICY_V2_SHA256}
           if config.history_policy == TRACKB_TEACHER_POLICY_V2 else {}),
    }


def _reuse_compatible_trackb_ten_round_v24_dataset(
    config: TrackBV24TenRoundPilotConfigV1,
) -> dict[str, object]:
    """Return the compatible public identity from one bounded admission."""
    result, _ = _admit_compatible_trackb_ten_round_v24_dataset(config)
    return result


def _admit_compatible_trackb_ten_round_v24_dataset(
    config: TrackBV24TenRoundPilotConfigV1,
) -> tuple[dict[str, object], Pilot_dataset_admission]:
    """Bind recipe, identity, and split rows to one immutable root admission."""
    root = Path(config.existing_v24_dataset_root).resolve()
    expected_cell = f"trackb_feedback_history_n{config.system_type[0]}"
    if config.history_policy == TRACKB_TEACHER_POLICY_V2:
        expected_cell = TRACKB_TEACHER_POLICY_V2_CELL_IDS[int(config.system_type[0])]
    admission = admit_pilot_dataset(
        root, cell_id=expected_cell, history_policy=config.history_policy,
        history_policy_sha256=config.history_policy_sha256,
        strict_shard_audit=True,
    )
    identity = admission.identity_dict()
    build_id = admission.build_id
    manifest = admission.manifest_dict()
    matching = [cell for cell in manifest.get("cells", []) if cell.get("cell_id") == expected_cell]
    if len(matching) != 1:
        raise ValueError("existing V2.4 root lacks the expected Track B feedback-history cell")
    cell = matching[0]
    recipe = _bound_dataset_recipe(config, manifest)
    expected_policy = config.history_policy
    policy_ok = cell.get("history_policy", TRACKB_TEACHER_POLICY_V1) == expected_policy
    if expected_policy == TRACKB_TEACHER_POLICY_V2:
        try:
            validate_trackb_v2_cell(cell)
            policy_ok = cell.get("history_policy_sha256") == config.history_policy_sha256
        except ValueError:
            policy_ok = False
    if (cell.get("history_rounds") != TEN_ROUND_COUNT
            or not policy_ok
            or cell.get("train") != config.train_samples
            or cell.get("validation") != config.validation_samples
            or manifest.get("generator_sha") != GENERATOR_SHA
            or identity.get("schema_sha") != schema_sha256()):
        raise ValueError("existing V2.4 root is incompatible with the current Track B ten-round pilot")
    if expected_policy == TRACKB_TEACHER_POLICY_V2:
        source_commit, source_tree = _git_identity()
        if (manifest.get("source_commit"), manifest.get("source_tree_sha")) != (
                source_commit, source_tree):
            raise ValueError("existing Track B V2 root source identity differs from the clean pilot source")
    result = {
        "dataset_root": str(root),
        "build_id": build_id,
        "ready_sha": identity["ready_sha"],
        "build_manifest_sha": identity["build_manifest_sha"],
        "contract_sha": identity["contract_sha"],
        "schema_sha": identity["schema_sha"],
        "index_sha": identity["index_sha"],
        "generator_sha": manifest["generator_sha"],
        "source_commit": manifest.get("source_commit"),
        "source_tree": manifest.get("source_tree_sha"),
        "data_to_model_compatibility": "SAME_V24_CONTRACT_SCHEMA_AND_GENERATOR",
        "reused_existing_v24_root": True,
        **recipe,
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        **({"history_policy": TRACKB_TEACHER_POLICY_V2,
            "history_policy_sha256": TRACKB_TEACHER_POLICY_V2_SHA256}
           if expected_policy == TRACKB_TEACHER_POLICY_V2 else {}),
    }
    return result, admission


def _supervision_targets(
    batch: Mapping[str, Any], history: torch.Tensor,
    *, source_view_kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep executed learner history separate from teacher-state labels."""
    has_actions = "teacher_action_tokens" in batch
    has_gains = "teacher_transition_gain" in batch
    if source_view_kind == V24_ORACLE_HISTORY_VIEW:
        if has_actions or has_gains:
            raise ValueError("V2.4 oracle history cannot accept learner-visited target arrays")
        return history[..., 2:4], history[:, :, 0, V24_HISTORY_GAIN_FEATURE_INDEX]
    if source_view_kind != LEARNER_VISITED_HISTORY_VIEW:
        raise ValueError("Track B supervision source view is unsupported")
    if not has_actions or not has_gains:
        raise ValueError("learner-visited supervision requires both teacher target arrays")
    identity_jsons = batch.get("row_identity_json")
    row_shas = batch.get("row_sha")
    schemas = batch.get("schema")
    splits = batch.get("split")
    sample_ids = batch.get("sample_id")
    if any(not isinstance(value, (tuple, list)) or len(value) != history.shape[0]
           for value in (identity_jsons, row_shas, schemas, splits, sample_ids)):
        raise ValueError("learner-visited source row declarations are incomplete")
    for name in ROW_TENSOR_NAMES:
        value = batch.get(name)
        if not isinstance(value, torch.Tensor) or value.shape[0] != history.shape[0]:
            raise ValueError(f"learner-visited {name} tensor is unavailable")
    for row_index in range(history.shape[0]):
        tensors = {
            name: batch[name][row_index].detach().cpu().numpy()
            for name in ROW_TENSOR_NAMES
        }
        identity = validate_learner_visited_tensors(
            identity_jsons[row_index], row_shas[row_index], tensors)
        if (schemas[row_index] != LEARNER_VISITED_ROW_SCHEMA
                or splits[row_index] != identity["split"]
                or sample_ids[row_index] != identity["sample_id"]):
            raise ValueError("learner-visited batch declaration differs from row identity")
        for name in ("research_only", "selection_evidence", "promotion_ready",
                     "hardware_action_authorized", "deployment_validation", "held_out"):
            declared = batch.get(name)
            if (not isinstance(declared, torch.Tensor) or declared.dtype != torch.bool
                    or declared.shape != history.shape[:1]
                    or bool(declared[row_index]) != identity[name]):
                raise ValueError(f"learner-visited {name} authority declaration differs")
    actions = batch["teacher_action_tokens"]
    gains = batch["teacher_transition_gain"]
    if not isinstance(actions, torch.Tensor) or not isinstance(gains, torch.Tensor):
        raise TypeError("learner-visited teacher targets must be tensors")
    actions = actions.to(device=history.device, dtype=history.dtype)
    gains = gains.to(device=history.device, dtype=history.dtype)
    if (actions.shape != history.shape[:3] + (2,)
            or gains.shape != history.shape[:2]):
        raise ValueError("learner-visited teacher target or executed history contract is invalid")
    return actions, gains


def _round_inputs(batch: Mapping[str, Any], *, round_index: int,
                  device: torch.device, source_view_kind: str = V24_ORACLE_HISTORY_VIEW,
                  ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if not 0 <= round_index < TEN_ROUND_COUNT:
        raise ValueError("V2.4 ten-round index is invalid")
    if (not isinstance(batch["history_mask"], torch.Tensor)
            or batch["history_mask"].dtype != torch.bool
            or not isinstance(batch["mode_mask"], torch.Tensor)
            or batch["mode_mask"].dtype != torch.bool):
        raise ValueError("V2.4 feedback history masks must be boolean tensors")
    history = batch["history_tokens"].to(device=device, dtype=torch.float32)
    history_mask = batch["history_mask"].to(device=device, dtype=torch.bool)
    noisy = batch["noisy_measurement_tokens"].to(device=device, dtype=torch.float32)
    target = batch["target_mode_tokens"].to(device=device, dtype=torch.float32)
    mode_mask = batch["mode_mask"].to(device=device, dtype=torch.bool)
    if (history.ndim != 4 or history.shape[1] != TEN_ROUND_COUNT or history.shape[-1] != 7
            or history_mask.shape != history.shape[:2] or noisy.shape != history.shape[0:1] + history.shape[2:3] + (2,)
            or target.shape != noisy.shape or mode_mask.shape != noisy.shape[:2]):
        raise ValueError("V2.4 feedback history batch does not meet the ten-round source contract")
    prefix = history[:, :round_index]
    prefix_mask = history_mask[:, :round_index]
    if round_index and not bool(prefix_mask.all()):
        raise ValueError("V2.4 training row lost a required preceding history record")
    action_targets, gain_targets = _supervision_targets(
        batch, history, source_view_kind=source_view_kind)
    target_action = action_targets[:, round_index]
    target_gain = gain_targets[:, round_index]
    return {
        "history_tokens": prefix,
        "history_mask": prefix_mask,
        "noisy_measurement_tokens": history[:, round_index, :, :2],
        "target_mode_tokens": target,
        "mode_mask": mode_mask,
    }, target_action, target_gain


def _training_target_normalization(
    dataset: V24TrainingDataset,
    *,
    batch_size: int,
    action_loss_normalization_kind: str,
    require_full_active_modes: bool = False,
    source_view_kind: str = V24_ORACLE_HISTORY_VIEW,
) -> TrackBTargetNormalizationV1:
    """Calculate normalization only from the bound V2.4 training split."""
    if action_loss_normalization_kind not in ACTION_LOSS_NORMALIZATION_KINDS:
        raise ValueError("Track B action-loss normalization kind is unsupported")
    if getattr(dataset, "split", "train") != "train":
        raise ValueError("Track B target normalization requires the TRAIN split")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    amplitude_square_sum = 0.0
    phase_square_sum = 0.0
    action_count = 0
    gain_sum = 0.0
    gain_square_sum = 0.0
    gain_count = 0
    amplitude_square_sum_by_round = [0.0] * TEN_ROUND_COUNT
    phase_square_sum_by_round = [0.0] * TEN_ROUND_COUNT
    action_count_by_round = [0] * TEN_ROUND_COUNT
    for batch in loader:
        if (source_view_kind == LEARNER_VISITED_HISTORY_VIEW
                and any(split != "train" for split in batch.get("split", ()))):
            raise ValueError("learner-visited target normalization requires TRAIN source rows")
        if require_full_active_modes and not bool(batch["mode_mask"].all()):
            raise ValueError("TRAIN-RMS decoder requires full active-mode TRAIN rows")
        history = batch["history_tokens"].to(dtype=torch.float64)
        action, gains = _supervision_targets(batch, history, source_view_kind=source_view_kind)
        amplitude_square_sum += float(action[..., 0].square().sum())
        phase_square_sum += float(action[..., 1].square().sum())
        action_count += int(action[..., 0].numel())
        for round_index in range(TEN_ROUND_COUNT):
            round_action = action[:, round_index]
            amplitude_square_sum_by_round[round_index] += float(round_action[..., 0].square().sum())
            phase_square_sum_by_round[round_index] += float(round_action[..., 1].square().sum())
            action_count_by_round[round_index] += int(round_action[..., 0].numel())
        gain_sum += float(gains.sum())
        gain_square_sum += float(gains.square().sum())
        gain_count += int(gains.numel())
    if not action_count or not gain_count:
        raise ValueError("V2.4 training source has no ten-round targets")
    gain_center = gain_sum / gain_count
    gain_variance = max(gain_square_sum / gain_count - gain_center**2, 0.0)
    result = TrackBTargetNormalizationV1(
        amplitude_scale=max(math.sqrt(amplitude_square_sum / action_count), 1e-6),
        phase_scale=max(math.sqrt(phase_square_sum / action_count), 1e-6),
        gain_center=gain_center,
        gain_scale=max(math.sqrt(gain_variance), 1e-6),
        action_loss_normalization_kind=action_loss_normalization_kind,
        amplitude_scale_by_round=(
            tuple(max(math.sqrt(total / count), 1e-6)
                  for total, count in zip(amplitude_square_sum_by_round, action_count_by_round))
            if action_loss_normalization_kind == PER_ROUND_ACTION_LOSS_NORMALIZATION else ()
        ),
        phase_scale_by_round=(
            tuple(max(math.sqrt(total / count), 1e-6)
                  for total, count in zip(phase_square_sum_by_round, action_count_by_round))
            if action_loss_normalization_kind == PER_ROUND_ACTION_LOSS_NORMALIZATION else ()
        ),
    )
    result.validate()
    return result


def _model_config_for_target_normalization(
    config: TrackBV24TenRoundPilotConfigV1,
    normalization: TrackBTargetNormalizationV1,
) -> TrackBInverseTransformerConfigV1:
    """Bind the model's gain decoding semantics to TRAIN-only normalization."""
    model_config = TrackBInverseTransformerConfigV1(
        config.system_type,
        int(config.system_type[0]),
        config.width,
        config.heads,
        config.blocks,
        action_value_feature_kind=config.action_value_feature_kind,
        action_projection_kind=config.action_projection_kind,
        gain_parameterization=config.gain_parameterization,
        gain_center=(
            normalization.gain_center
            if config.gain_parameterization == STANDARDIZED_LATENT_GAIN_PARAMETERIZATION_V1
            else 0.0
        ),
        gain_scale=(
            normalization.gain_scale
            if config.gain_parameterization == STANDARDIZED_LATENT_GAIN_PARAMETERIZATION_V1
            else 1.0
        ),
        amplitude_decode_scale_by_round=(
            normalization.amplitude_scale_by_round
            if config.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION
            else ()
        ),
        phase_decode_scale_by_round=(
            normalization.phase_scale_by_round
            if config.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION
            else ()
        ),
        history_gain_input_kind=config.history_gain_input_kind,
    )
    require_gain_parameterization_binding(model_config, normalization.to_dict())
    require_action_decode_scale_binding(model_config, normalization.to_dict())
    return model_config


def _logical_action_loss_components(
    output: Mapping[str, torch.Tensor],
    target_action: torch.Tensor,
    target_gain: torch.Tensor,
    normalization: TrackBTargetNormalizationV1,
    *,
    round_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return differentiable task losses without changing their semantics."""
    prediction = torch.stack((output["amplitude_delta"], output["phase_delta"]), dim=-1)
    amplitude_scale, phase_scale = normalization.action_scales_for_round(round_index)
    amplitude_loss = functional.mse_loss(
        prediction[..., 0] / amplitude_scale,
        target_action[..., 0] / amplitude_scale,
    )
    phase_loss = functional.mse_loss(
        prediction[..., 1] / phase_scale,
        target_action[..., 1] / phase_scale,
    )
    gain_loss = functional.smooth_l1_loss(
        (output["predicted_teacher_transition_gain"] - normalization.gain_center) / normalization.gain_scale,
        (target_gain - normalization.gain_center) / normalization.gain_scale,
        beta=STANDARDIZED_HUBER_DELTA,
    )
    action_loss = 0.5 * (amplitude_loss + phase_loss)
    return action_loss, gain_loss, {
        "action_normalized_mse": float(action_loss.detach().cpu()),
        "amplitude_normalized_mse": float(amplitude_loss.detach().cpu()),
        "phase_normalized_mse": float(phase_loss.detach().cpu()),
        "gain_standardized_huber": float(gain_loss.detach().cpu()),
        "action_mse": float(functional.mse_loss(prediction, target_action).detach().cpu()),
        "gain_mse": float(functional.mse_loss(
            output["predicted_teacher_transition_gain"], target_gain).detach().cpu()),
    }


def _logical_action_loss(
    output: Mapping[str, torch.Tensor],
    target_action: torch.Tensor,
    target_gain: torch.Tensor,
    normalization: TrackBTargetNormalizationV1,
    *,
    round_index: int,
    gain_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    action_loss, gain_loss, metrics = _logical_action_loss_components(
        output,
        target_action,
        target_gain,
        normalization,
        round_index=round_index,
    )
    loss = action_loss + gain_loss_weight * gain_loss
    return loss, {
        **metrics,
        "loss": float(loss.detach().cpu()),
    }


def _compose_shared_pcgrad(
    parameter_names: tuple[str, ...],
    parameters: tuple[torch.nn.Parameter, ...],
    action_gradients: tuple[torch.Tensor | None, ...],
    gain_gradients: tuple[torch.Tensor | None, ...],
    *,
    gain_loss_weight: float,
    balance_gain_gradients: bool = False,
) -> tuple[tuple[torch.Tensor | None, ...], bool]:
    """Project only conflicting shared gradients before the optimizer step."""
    shared_indices = tuple(
        index for index, name in enumerate(parameter_names)
        if name.startswith((
            "history_encoder.",
            "last_history_encoder.",
            "prefix_length_encoder.",
            "mode_encoder.",
            "inverse_transformer.",
        ))
    )
    if balance_gain_gradients:
        return compose_balanced_shared_gradients(
            parameters, action_gradients, gain_gradients,
            shared_indices=shared_indices, gain_loss_weight=gain_loss_weight,
        )
    action_gradients_for_math = action_gradients
    weighted_gain_gradients = tuple(
        None if gradient is None else gain_loss_weight * gradient
        for gradient in gain_gradients
    )
    action_norm_squared = math.fsum(
        float(
            action_gradients[index]
            .detach()
            .to(dtype=torch.float64)
            .square()
            .sum()
        )
        for index in shared_indices
        if action_gradients[index] is not None
    )
    shared_inner_product = math.fsum(
        float(
            (
                action_gradients_for_math[index].detach().to(dtype=torch.float64)
                * weighted_gain_gradients[index].detach().to(dtype=torch.float64)
            ).sum()
        )
        for index in shared_indices
        if action_gradients_for_math[index] is not None
        and weighted_gain_gradients[index] is not None
    )
    projection_applied = shared_inner_product < 0.0 and action_norm_squared > 0.0
    if projection_applied:
        projection_coefficient = shared_inner_product / action_norm_squared
        projected_gain_gradients = list(weighted_gain_gradients)
        for index in shared_indices:
            parameter = parameters[index]
            action_gradient = (
                torch.zeros_like(parameter)
                if action_gradients_for_math[index] is None
                else action_gradients_for_math[index]
            )
            gain_gradient = (
                torch.zeros_like(parameter)
                if weighted_gain_gradients[index] is None
                else weighted_gain_gradients[index]
            )
            projected = gain_gradient - projection_coefficient * action_gradient
            if action_gradients_for_math[index] is None and weighted_gain_gradients[index] is None:
                projected_gain_gradients[index] = None
            else:
                projected_gain_gradients[index] = projected
        weighted_gain_gradients = tuple(projected_gain_gradients)
    combined_gradients = []
    for parameter, action_gradient, gain_gradient in zip(
        parameters,
        action_gradients_for_math,
        weighted_gain_gradients,
        strict=True,
    ):
        if action_gradient is None and gain_gradient is None:
            combined_gradients.append(None)
            continue
        combined = torch.zeros_like(parameter)
        if action_gradient is not None:
            combined = combined + action_gradient
        if gain_gradient is not None:
            combined = combined + gain_gradient
        combined_gradients.append(combined)
    return tuple(combined_gradients), projection_applied


def _assign_training_gradients(
    model: TrackBInverseTransformerV1,
    action_loss: torch.Tensor,
    gain_loss: torch.Tensor,
    *,
    gain_loss_weight: float,
    gradient_composition_kind: str,
) -> bool:
    """Assign optimizer gradients while keeping the default weighted sum unchanged."""
    if gradient_composition_kind == WEIGHTED_SUM_GRADIENT_COMPOSITION:
        (action_loss + gain_loss_weight * gain_loss).backward()
        return False
    if gradient_composition_kind not in (
        SHARED_PCGD_GRADIENT_COMPOSITION,
        SHARED_BALANCED_PCGD_GRADIENT_COMPOSITION,
    ):
        raise ValueError("unsupported Track B gradient composition kind")
    named_parameters = tuple(model.named_parameters())
    parameter_names = tuple(name for name, _parameter in named_parameters)
    parameters = tuple(parameter for _name, parameter in named_parameters)
    action_gradients = torch.autograd.grad(
        action_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    gain_gradients = torch.autograd.grad(
        gain_loss,
        parameters,
        allow_unused=True,
    )
    combined_gradients, projection_applied = _compose_shared_pcgrad(
        parameter_names,
        parameters,
        action_gradients,
        gain_gradients,
        gain_loss_weight=gain_loss_weight,
        balance_gain_gradients=(
            gradient_composition_kind == SHARED_BALANCED_PCGD_GRADIENT_COMPOSITION
        ),
    )
    for parameter, gradient in zip(parameters, combined_gradients, strict=True):
        parameter.grad = gradient
    return projection_applied


def _source_identity(dataset: V24TrainingDataset) -> dict[str, object]:
    identity = dict(dataset.identity)
    result = {
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
    policy = getattr(dataset, "history_policy", TRACKB_TEACHER_POLICY_V1)
    if policy == TRACKB_TEACHER_POLICY_V2:
        result["history_policy"] = policy
        result["history_policy_sha256"] = dataset.history_policy_sha256
    return result


def _open_pilot_dataset(
    config: TrackBV24TenRoundPilotConfigV1,
    *, root: Path, build_id: str, cell_id: str, split: str,
    verification_context: Any = None,
    root_admission: Pilot_dataset_admission | None = None,
) -> Any:
    """Open the explicit source view without changing the V2.4 base root."""
    if config.source_view_kind == V24_ORACLE_HISTORY_VIEW:
        if root_admission is not None:
            if (root_admission.root != root.resolve()
                    or root_admission.build_id != build_id
                    or root_admission.cell_id != cell_id
                    or root_admission.history_policy != config.history_policy
                    or root_admission.history_policy_sha256 != config.history_policy_sha256):
                raise ValueError("Track B shared root admission differs from the requested view")
            return Pilot_history_dataset._from_admission(root_admission, split=split)
        return Pilot_history_dataset(
            root, build_id=build_id, cell_id=cell_id,
            split=split, task="feedback", arm="A0_RAW",
            history_policy=config.history_policy,
            history_policy_sha256=config.history_policy_sha256,
        )
    from .learner_visited_dataset import (
        Learner_visited_report_dataset,
        Learner_visited_report_view_config_v1,
    )

    report_path = getattr(config, f"learner_visited_{split}_report_path")
    report_sha = getattr(config, f"learner_visited_{split}_report_sha")
    return Learner_visited_report_dataset(Learner_visited_report_view_config_v1(
        pilot_receipt_path=config.behavior_pilot_receipt_path,
        expected_pilot_receipt_sha=config.behavior_pilot_receipt_sha,
        rollout_report_path=report_path,
        expected_rollout_report_sha=report_sha,
        split=split,
        sample_count=config.train_samples if split == "train" else config.validation_samples,
        allow_partial_development=False,
    ), verification_context=verification_context)


def _training_source_identity(
    config: TrackBV24TenRoundPilotConfigV1,
    dataset: Any,
    dataset_result: Mapping[str, Any],
) -> dict[str, object]:
    """Bind both the V2.4 split and the distinct learner-visited row view."""
    if config.source_view_kind == V24_ORACLE_HISTORY_VIEW:
        source = _source_identity(dataset)
        for name in (
            "build_id", "ready_sha", "build_manifest_sha", "contract_sha",
            "schema_sha", "index_sha", "generator_sha", "source_commit", "source_tree",
        ):
            if source.get(name) != dataset_result.get(name):
                raise ValueError(f"oracle-history V2.4 base differs at {name}")
        return source
    view = getattr(dataset, "source_identity", None)
    base = view.get("base_v24_source") if isinstance(view, Mapping) else None
    if (not isinstance(base, Mapping)
            or view.get("source_view_kind") != LEARNER_VISITED_HISTORY_VIEW
            or view.get("split") != dataset.split
            or view.get("sample_count") != len(dataset)
            or view.get("registered_budget_complete") is not True
            or Path(dataset.base_root).resolve() != Path(config.existing_v24_dataset_root).resolve()):
        raise ValueError("learner-visited pilot source view is not bound to its V2.4 split")
    for name in (
        "build_id", "ready_sha", "build_manifest_sha", "contract_sha",
        "schema_sha", "index_sha", "generator_sha", "source_commit", "source_tree",
    ):
        if base.get(name) != dataset_result.get(name):
            raise ValueError(f"learner-visited V2.4 base differs at {name}")
    return {
        **dict(base),
        "source_view_kind": LEARNER_VISITED_HISTORY_VIEW,
        "learner_visited": dict(view),
    }


def _real_static_source_identity(root_value: str) -> dict[str, object] | None:
    if not root_value:
        return None
    root = Path(root_value).resolve()
    ready_path = root / "READY.json"
    if not root.is_dir() or not ready_path.is_file():
        raise ValueError("real_static_source_root must contain READY.json")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("research_only") is not True or ready.get("selection_evidence") is not False \
            or ready.get("promotion_ready") is not False or ready.get("hardware_action_authorized") is not False:
        raise ValueError("real-static source flags violate the research-only Track B boundary")
    return {
        "root": str(root),
        "ready_sha": sha256_file(ready_path),
        "build_id": ready.get("build_id"),
        "contract_sha": ready.get("contract_sha"),
        "schema_sha": ready.get("schema_sha"),
        "used_for_action_training": False,
        "reason": "REAL_STATIC_HAS_NO_ACTION_TRANSITIONS",
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def _git_identity() -> tuple[str, str]:
    return pilot_source_identity()


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    before_replace: Callable[[], None] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_checkpoint_artifact(
    checkpoint_path: Path,
    payload: Mapping[str, object],
    *,
    expected_source_identity: tuple[str, str],
    artifact_writer: Learner_pilot_artifact_writer | None,
) -> str:
    """Publish a checkpoint only while the original clean source is unchanged."""
    if artifact_writer is not None:
        prepared = io.BytesIO()
        torch.save(dict(payload), prepared)
        checkpoint_bytes = prepared.getvalue()
        if not checkpoint_bytes:
            raise RuntimeError("serialized pilot checkpoint is empty")
        return artifact_writer.write_checkpoint_bytes(
            checkpoint_bytes,
            before_publish=lambda: require_unchanged_pilot_source(expected_source_identity),
        )

    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    try:
        torch.save(dict(payload), temporary_checkpoint)
        require_unchanged_pilot_source(expected_source_identity)
        os.replace(temporary_checkpoint, checkpoint_path)
    finally:
        if temporary_checkpoint.exists():
            temporary_checkpoint.unlink()
    return sha256_file(checkpoint_path)


def _publish_receipt_artifact(
    receipt_path: Path,
    payload: Mapping[str, object],
    *,
    expected_source_identity: tuple[str, str],
    artifact_writer: Learner_pilot_artifact_writer | None,
) -> str:
    """Publish a receipt only while the original clean source is unchanged."""
    receipt_bytes = canonical_json_bytes(payload)
    if artifact_writer is not None:
        return artifact_writer.write_receipt_bytes(
            receipt_bytes,
            before_publish=lambda: require_unchanged_pilot_source(expected_source_identity),
        )
    _atomic_write_bytes(
        receipt_path,
        receipt_bytes,
        before_replace=lambda: require_unchanged_pilot_source(expected_source_identity),
    )
    return sha256_file(receipt_path)


def _evaluate_teacher_forced(
    model: TrackBInverseTransformerV1,
    dataset: V24TrainingDataset,
    *,
    batch_size: int,
    device: torch.device,
    source_view_kind: str = V24_ORACLE_HISTORY_VIEW,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate registered oracle histories or separate learner-state labels."""
    if source_view_kind not in (V24_ORACLE_HISTORY_VIEW, LEARNER_VISITED_HISTORY_VIEW):
        raise ValueError("offline Track B evaluation source view is unsupported")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    action_error_by_round: list[list[float]] = [[] for _ in range(TEN_ROUND_COUNT)]
    gain_error_by_round: list[list[float]] = [[] for _ in range(TEN_ROUND_COUNT)]
    zero_action_error_by_round: list[list[float]] = [[] for _ in range(TEN_ROUND_COUNT)]
    previous_action_error_by_round: list[list[float]] = [[] for _ in range(TEN_ROUND_COUNT)]
    zero_gain_error_by_round: list[list[float]] = [[] for _ in range(TEN_ROUND_COUNT)]
    previous_gain_error_by_round: list[list[float]] = [[] for _ in range(TEN_ROUND_COUNT)]
    selected_guidance: list[dict[str, object]] = []
    learner_visited = source_view_kind == LEARNER_VISITED_HISTORY_VIEW
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            for round_index in range(TEN_ROUND_COUNT):
                inputs, target_action, target_gain = _round_inputs(
                    batch, round_index=round_index, device=device,
                    source_view_kind=source_view_kind)
                output = model(inputs)
                predicted = torch.stack((output["amplitude_delta"], output["phase_delta"]), dim=-1)
                action_error_by_round[round_index].append(float(
                    functional.mse_loss(predicted, target_action).detach().cpu()))
                gain_error_by_round[round_index].append(float(
                    functional.l1_loss(
                        output["predicted_teacher_transition_gain"], target_gain).detach().cpu()))
                zero_action_error_by_round[round_index].append(float(
                    functional.mse_loss(torch.zeros_like(target_action), target_action).detach().cpu()))
                zero_gain_error_by_round[round_index].append(float(
                    functional.l1_loss(torch.zeros_like(target_gain), target_gain).detach().cpu()))
                history = batch["history_tokens"].to(device=device, dtype=torch.float32)
                if round_index:
                    previous_action = history[:, round_index - 1, :, 2:4]
                    previous_gain = history[:, round_index - 1, 0, 6]
                else:
                    previous_action = torch.zeros_like(target_action)
                    previous_gain = torch.zeros_like(target_gain)
                previous_action_error_by_round[round_index].append(float(
                    functional.mse_loss(previous_action, target_action).detach().cpu()))
                previous_gain_error_by_round[round_index].append(float(
                    functional.l1_loss(previous_gain, target_gain).detach().cpu()))
                if batch_index == 0:
                    guidance = {
                        "round_index": round_index + 1,
                        "history_records_consumed": round_index,
                        "logical_amplitude_delta": predicted[0, :, 0].detach().cpu().tolist(),
                        "logical_phase_delta": predicted[0, :, 1].detach().cpu().tolist(),
                        "target_amplitude_delta": target_action[0, :, 0].detach().cpu().tolist(),
                        "target_phase_delta": target_action[0, :, 1].detach().cpu().tolist(),
                        "predicted_teacher_transition_gain": float(
                            output["predicted_teacher_transition_gain"][0].detach().cpu()),
                        "hardware_action_authorized": False,
                    }
                    gain_label = (
                        "teacher_transition_gain_at_learner_pre_state"
                        if learner_visited else "observed_v24_gain"
                    )
                    guidance[gain_label] = float(target_gain[0].detach().cpu())
                    selected_guidance.append(guidance)
    summary = {
        "evaluation_kind": "V24_TEACHER_FORCED_TEN_ROUND_ENGINEERING",
        "action_mse_by_round": [float(sum(values) / len(values)) for values in action_error_by_round],
        "gain_mae_by_round": [float(sum(values) / len(values)) for values in gain_error_by_round],
        "zero_action_mse_by_round": [float(sum(values) / len(values)) for values in zero_action_error_by_round],
        "previous_action_mse_by_round": [float(sum(values) / len(values)) for values in previous_action_error_by_round],
        "zero_gain_mae_by_round": [float(sum(values) / len(values)) for values in zero_gain_error_by_round],
        "previous_gain_mae_by_round": [float(sum(values) / len(values)) for values in previous_gain_error_by_round],
        "gain_target_semantics": "TEACHER_ACTION_TRANSITION_GAIN_NOT_MODEL_ACTION_GAIN",
        "hardware_action_authorized": False,
    }
    if learner_visited:
        summary.update({
            "evaluation_kind": "LEARNER_VISITED_OFFLINE_TEN_ROUND_ENGINEERING",
            "gain_target_semantics": "TEACHER_ACTION_TRANSITION_GAIN_AT_LEARNER_PRE_STATE_NOT_MODEL_ACTION_GAIN",
            "previous_action_baseline_kind": "PREVIOUS_LEARNER_EXECUTED_ACTION",
            "previous_gain_baseline_kind": "UNOBSERVED_ZERO_HISTORY_GAIN",
            "development_only": True,
            "selection_evidence": False,
            "promotion_ready": False,
            "hardware_action_authorized": False,
        })
    return summary, selected_guidance


def _round_coverage_batch_schedule(
    *,
    dataset_size: int,
    batch_size: int,
    optimizer_steps: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Cover every TRAIN row for every round before deterministically repeating batches."""
    if dataset_size <= 0 or batch_size <= 0 or optimizer_steps <= 0:
        raise ValueError("Track B round coverage schedule requires positive dimensions")
    batches_per_round = math.ceil(dataset_size / batch_size)
    per_round_batches: list[tuple[tuple[int, ...], ...]] = []
    for round_index in range(TEN_ROUND_COUNT):
        indices = list(range(dataset_size))
        random.Random(seed + round_index).shuffle(indices)
        required = batches_per_round * batch_size
        padding = required - dataset_size
        if padding:
            indices.extend((indices * math.ceil(padding / dataset_size))[:padding])
        per_round_batches.append(tuple(
            tuple(indices[offset:offset + batch_size])
            for offset in range(0, required, batch_size)
        ))
    batch_schedule: list[tuple[int, ...]] = []
    round_schedule: list[int] = []
    for batch_index in range(batches_per_round):
        for round_index in range(TEN_ROUND_COUNT):
            batch_schedule.append(per_round_batches[round_index][batch_index])
            round_schedule.append(round_index)
            if len(batch_schedule) == optimizer_steps:
                return tuple(batch_schedule), tuple(round_schedule)
    repeat_index = 0
    while len(batch_schedule) < optimizer_steps:
        round_index = repeat_index % TEN_ROUND_COUNT
        batch_index = (repeat_index // TEN_ROUND_COUNT) % batches_per_round
        batch_schedule.append(per_round_batches[round_index][batch_index])
        round_schedule.append(round_index)
        repeat_index += 1
    return tuple(batch_schedule), tuple(round_schedule)


def _pilot_config_payload(config: TrackBV24TenRoundPilotConfigV1) -> dict[str, object]:
    """Preserve the legacy random-pilot payload while declaring opt-in sampling."""
    payload = asdict(config)
    if config.training_sampling_kind == RANDOM_TRAINING_SAMPLING:
        del payload["training_sampling_kind"]
    if config.gradient_composition_kind == WEIGHTED_SUM_GRADIENT_COMPOSITION:
        del payload["gradient_composition_kind"]
    if config.history_gain_input_kind == LEGACY_ORACLE_HISTORY_GAIN_INPUT:
        del payload["history_gain_input_kind"]
    if config.source_view_kind == V24_ORACLE_HISTORY_VIEW:
        del payload["source_view_kind"]
        for name in LEARNER_VIEW_CONFIG_FIELDS:
            del payload[name]
    if config.history_policy == TRACKB_TEACHER_POLICY_V1:
        del payload["history_policy"]
        del payload["history_policy_sha256"]
    return payload


def _batch_schedule_metadata(
    batch_schedule: tuple[tuple[int, ...], ...] | None,
) -> dict[str, object]:
    """Record a batch schedule only when an opt-in coverage schedule exists."""
    if batch_schedule is None:
        return {}
    return {"batch_schedule_sha": canonical_sha256(batch_schedule)}


def run_trackb_v24_ten_round_pilot(
    config: TrackBV24TenRoundPilotConfigV1,
    *,
    emit_progress: bool = False,
    on_data_admitted: Callable[[Mapping[str, object], Mapping[str, object], Mapping[str, object]], None] | None = None,
    artifact_writer: Learner_pilot_artifact_writer | None = None,
) -> dict[str, object]:
    """Generate, train, and evaluate the bounded current-source V2.4 pilot."""
    config.validate()
    if type(emit_progress) is not bool:
        raise TypeError("emit_progress must be a strict boolean")
    if on_data_admitted is not None and not callable(on_data_admitted):
        raise TypeError("on_data_admitted must be callable")
    if artifact_writer is not None:
        if not isinstance(artifact_writer, Learner_pilot_artifact_writer):
            raise TypeError("artifact_writer must be a learner pilot artifact writer")
        artifact_writer.verify()
        if artifact_writer.pilot_lease.path != Path(config.output_root):
            raise ValueError("artifact writer does not bind the configured pilot root")
    source_commit, source_tree = _git_identity()
    if config.history_policy == TRACKB_TEACHER_POLICY_V2:
        raise RuntimeError(
            "Track B V2 pilot training is not ready until learner-visited and evaluator contracts bind the policy"
        )
    output_root = Path(config.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("ten-round pilot output root must be empty")
    if config.existing_v24_dataset_root:
        dataset_result, root_admission = _admit_compatible_trackb_ten_round_v24_dataset(config)
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        dataset_result = generate_trackb_ten_round_v24_dataset(config)
        root_admission = None
    dataset_root = Path(str(dataset_result["dataset_root"]))
    build_id = str(dataset_result["build_id"])
    cell_id = (
        TRACKB_TEACHER_POLICY_V2_CELL_IDS[int(config.system_type[0])]
        if config.history_policy == TRACKB_TEACHER_POLICY_V2
        else f"trackb_feedback_history_n{config.system_type[0]}"
    )
    if root_admission is None:
        root_admission = admit_pilot_dataset(
            dataset_root, build_id=build_id, cell_id=cell_id,
            history_policy=config.history_policy,
            history_policy_sha256=config.history_policy_sha256,
            strict_shard_audit=True,
        )
        root_identity = root_admission.identity_dict()
        for name in (
            "build_id", "ready_sha", "build_manifest_sha", "contract_sha",
            "schema_sha", "index_sha", "generator_sha", "source_commit", "source_tree",
        ):
            if root_identity.get(name) != dataset_result.get(name):
                raise ValueError(f"generated Track B root differs from its receipt at {name}")
        recipe = _bound_dataset_recipe(config, root_admission.manifest_dict())
        if recipe["dataset_recipe_sha"] != dataset_result.get("dataset_recipe_sha"):
            raise ValueError("generated Track B root recipe differs from its receipt")
    verification_context = None
    if config.source_view_kind == LEARNER_VISITED_HISTORY_VIEW:
        from .policy_ablation import _Model_read_context_v1

        verification_context = _Model_read_context_v1()
    train_dataset = _open_pilot_dataset(
        config, root=dataset_root, build_id=build_id, cell_id=cell_id, split="train",
        verification_context=verification_context, root_admission=root_admission)
    validation_dataset = None
    try:
        validation_dataset = _open_pilot_dataset(
            config, root=dataset_root, build_id=build_id, cell_id=cell_id, split="validation",
            verification_context=verification_context, root_admission=root_admission)
        if len(train_dataset) != config.train_samples or len(validation_dataset) != config.validation_samples:
            raise RuntimeError("generated V2.4 ten-round row counts differ from the bound pilot plan")
        train_source = _training_source_identity(config, train_dataset, dataset_result)
        validation_source = _training_source_identity(config, validation_dataset, dataset_result)
        if on_data_admitted is not None:
            on_data_admitted(dataset_result, train_source, validation_source)
        output_root.mkdir(parents=True, exist_ok=True)
        normalization = _training_target_normalization(
            train_dataset,
            batch_size=config.batch_size,
            action_loss_normalization_kind=config.action_loss_normalization_kind,
            require_full_active_modes=(
                config.action_projection_kind == TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION
            ),
            source_view_kind=config.source_view_kind,
        )
        if artifact_writer is not None:
            artifact_writer.verify()
        training_dependencies = Pilot_training_dependencies(
            model_config_for_target_normalization=_model_config_for_target_normalization,
            model_factory=TrackBInverseTransformerV1,
            round_inputs=_round_inputs,
            logical_action_loss_components=_logical_action_loss_components,
            assign_training_gradients=_assign_training_gradients,
            checked_optimizer_step=checked_optimizer_step,
            evaluate_teacher_forced=_evaluate_teacher_forced,
            round_coverage_batch_schedule=_round_coverage_batch_schedule,
            measure_dependency=measure_inverse_transformer_dependency,
            random_sampling_kind=RANDOM_TRAINING_SAMPLING,
            coverage_sampling_kind=ROUND_COVERAGE_TRAINING_SAMPLING,
        )
        initialized = initialize_pilot_model(config, normalization, training_dependencies)
        if artifact_writer is not None:
            artifact_writer.verify()
        def emit_training_progress(record: dict[str, object]) -> None:
            print(canonical_json_bytes({"event": "optimizer_step_completed", **record}).decode("utf-8"),
                  flush=True)

        trained = train_pilot_core(
            initialized, config, normalization, train_dataset, validation_dataset,
            training_dependencies, on_progress=emit_training_progress if emit_progress else None,
        )
        if trained.completed_updates != config.optimizer_steps:
            raise RuntimeError("ten-round pilot did not complete the registered optimizer budget")
        model = trained.model
        model_config = trained.model_config
        round_schedule = trained.round_schedule
        batch_schedule = trained.batch_schedule
        progress = [dict(record) for record in trained.progress]
        evaluation = trained.evaluation
        guidance = [dict(record) for record in trained.guidance]
        dependency = trained.dependency
        require_unchanged_pilot_source((source_commit, source_tree))
        real_static_source = _real_static_source_identity(config.real_static_source_root)
        model_dir = output_root / "model"
        if artifact_writer is None:
            model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / "checkpoint.pt"
        checkpoint_payload = {
            "schema": "TrackBInverseTransformerCheckpointV1",
            "model_config": model_config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_steps": config.optimizer_steps,
            "target_normalization": normalization.to_dict(),
            "round_schedule_sha": canonical_sha256(round_schedule),
            **_batch_schedule_metadata(batch_schedule),
            "source_commit": source_commit,
            "source_tree": source_tree,
            "train_source": train_source,
            "validation_source": validation_source,
            **({"source_view_kind": config.source_view_kind}
               if config.source_view_kind == LEARNER_VISITED_HISTORY_VIEW else {}),
            "research_only": True,
            "selection_evidence": False,
            "promotion_ready": False,
            "hardware_action_authorized": False,
        }
        checkpoint_sha = _publish_checkpoint_artifact(
            checkpoint_path,
            checkpoint_payload,
            expected_source_identity=(source_commit, source_tree),
            artifact_writer=artifact_writer,
        )
        receipt = {
            "schema": PILOT_SCHEMA,
            "config": _pilot_config_payload(config),
            "source_commit": source_commit,
            "source_tree": source_tree,
            "dataset": {
                "result": dataset_result,
                "train": train_source,
                "validation": validation_source,
                "real_static_observation_source": real_static_source,
                "data_to_model_compatibility": dataset_result.get(
                    "data_to_model_compatibility", "SAME_SOURCE_COMMIT_AND_TREE"),
            },
            "model": {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha": checkpoint_sha,
                "config": model_config.to_dict(),
                "target_normalization": normalization.to_dict(),
            },
            "training_progress": progress,
            "round_schedule_sha": canonical_sha256(round_schedule),
            **_batch_schedule_metadata(batch_schedule),
            "evaluation": evaluation,
            "inverse_transformer_dependency": {
                **asdict(dependency),
                "dependency_sufficient": dependency.dependency_sufficient,
            },
            "ten_round_guidance": guidance,
            **({"source_view_kind": config.source_view_kind}
               if config.source_view_kind == LEARNER_VISITED_HISTORY_VIEW else {}),
            "research_only": True,
            "selection_evidence": False,
            "promotion_ready": False,
            "hardware_action_authorized": False,
            "external_waveform_write": "NOT_IMPLEMENTED",
        }
        receipt_path = output_root / "pilot_receipt.json"
        receipt_sha = _publish_receipt_artifact(
            receipt_path,
            receipt,
            expected_source_identity=(source_commit, source_tree),
            artifact_writer=artifact_writer,
        )
        receipt["receipt_path"] = str(receipt_path)
        receipt["receipt_sha"] = receipt_sha
        return receipt
    finally:
        train_dataset.close()
        if validation_dataset is not None:
            validation_dataset.close()


__all__ = [
    "GRADIENT_COMPOSITION_KINDS",
    "SHARED_BALANCED_PCGD_GRADIENT_COMPOSITION",
    "SHARED_PCGD_GRADIENT_COMPOSITION",
    "WEIGHTED_SUM_GRADIENT_COMPOSITION",
    "DEFAULT_OPTIMIZER_STEPS",
    "PILOT_SCHEMA",
    "TrackBV24TenRoundPilotConfigV1",
    "build_trackb_ten_round_v24_plan",
    "generate_trackb_ten_round_v24_dataset",
    "run_trackb_v24_ten_round_pilot",
]
