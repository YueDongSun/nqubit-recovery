"""Train V8 feedback candidates through a strict, auditable V7 reuse seam.

The implementation intentionally reuses the battle-tested V7 optimization
loops while replacing the parts whose contracts changed in V8:

* V8 model construction and independent artifact schemas;
* repeat-aware measured auxiliary losses with an immutable unit NB2 weight;
* development-only model selection (the final test split stays sealed);
* explicit exposure masks, provenance hashes and synthetic-family partitions;
* fail-closed uncertainty state until real paired episodes can calibrate it.

This module is research/training code.  It has no experiment-runner, AWG,
FPGA, matrix-delta, promotion, or hardware-action authority.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import subprocess
import types
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
except ImportError:  # pragma: no cover - repository supports NumPy-only installs
    torch = None
    functional = None

from ml_recon.physics_hybrid_v7 import train_feedback_models as v7_training
from ml_recon.physics_hybrid_v7.feedback_metrics import (
    torch_nb2_log_probability,
)


DEFAULT_CONFIG_PATH = Path("ml_recon/configs/physics_hybrid_v8_feedback.json")
CONFIG_SCHEMA = "physics_hybrid_v8_feedback_training_v1"
TRAINING_REPORT_SCHEMA = "v8_feedback_training_report_v1"
CHECKPOINT_SCHEMA = "v8_feedback_inference_checkpoint_v1"
SOURCE_AUDIT_SCHEMA = "v8_source_tree_audit_v1"
UNCERTAINTY_SCHEMA = "v8_real_episode_uncertainty_contract_v1"
LOCKED_FINAL_STATE_SCHEMA = "v8_locked_final_test_state_v1"
WEAK_PAIRED_COMPARISON_CONTRACT_SCHEMA = (
    "v8_weak_supervision_paired_comparison_contract_v1"
)
WEAK_PAIRED_COMPARISON_SCHEMA = "v8_weak_supervision_paired_comparison_v1"
WEAK_PAIRED_COMPARISON_GATE_SCHEMA = (
    "v8_weak_supervision_paired_comparison_gate_v1"
)
REQUIRED_ABLATIONS = ("A0", "A1", "A2", "A3", "A4")
HASH_CHUNK_BYTES = 1024 * 1024
RHO_SOFT_TEACHER_MAXIMUM_FRACTION = 0.05


@dataclass(frozen=True)
class V8FeedbackTrainingProfile:
    """V7-loop-compatible profile carrying V8 model and ablation settings."""

    name: str
    execution_tier: str
    width: int
    operation_layers: int
    history_layers: int
    attention_heads: int
    feedforward_multiplier: float
    dropout: float
    mechanism_top_k: int
    classifier_latent_dim: int
    history_identification_enabled: bool
    real_batch_size: int
    synthetic_batch_size: int
    measured_pretrain_epochs: int
    synthetic_model_a_steps: int
    synthetic_model_b_steps: int
    measured_adapter_epochs: int
    maximum_real_batches_per_system: int
    evaluation_max_samples_per_split: int
    policy_backbone_gradient_scale: float
    classifier_prior_source: str
    loss_weight_overrides: Mapping[str, float]
    weak_state_pair_supervision_enabled: bool
    weak_supervision_pair_id: str
    weak_supervision_control_for: str
    state_pair_budget_phase: str
    real_training_temperature_max_c: Optional[float]
    real_training_temperature_filter_systems: Tuple[str, ...]
    ablation_id: str
    v8_model_overrides: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        name: str,
        value: Mapping[str, Any],
        *,
        ablation_id: str,
        v8_model_overrides: Mapping[str, Any],
    ) -> "V8FeedbackTrainingProfile":
        base = v7_training.FeedbackTrainingProfile.from_mapping(name, value)
        return cls(
            **asdict(base),
            weak_state_pair_supervision_enabled=bool(
                value.get("weak_state_pair_supervision_enabled", False)
            ),
            weak_supervision_pair_id=str(value.get("weak_supervision_pair_id", "")),
            weak_supervision_control_for=str(
                value.get("weak_supervision_control_for", "")
            ),
            state_pair_budget_phase=str(value.get("state_pair_budget_phase", "screen")),
            real_training_temperature_max_c=(
                None
                if value.get("real_training_temperature_max_c") is None
                else float(value["real_training_temperature_max_c"])
            ),
            real_training_temperature_filter_systems=tuple(
                str(item)
                for item in value.get(
                    "real_training_temperature_filter_systems", ()
                )
            ),
            ablation_id=str(ablation_id),
            v8_model_overrides=copy.deepcopy(dict(v8_model_overrides)),
        )

    def model_config(self) -> Any:
        from .feedback_models import V8FeedbackModelConfig

        # Build the common shape payload directly.  Calling the V7 profile's
        # ``model_config`` here is unsafe while the reuse seam has temporarily
        # replaced V7FeedbackModelConfig with its V8 counterpart: the V7
        # method would pass the legacy history keyword into the V8 dataclass.
        payload = {
            "width": int(self.width),
            "operation_layers": int(self.operation_layers),
            "history_layers": int(self.history_layers),
            "attention_heads": int(self.attention_heads),
            "feedforward_multiplier": float(self.feedforward_multiplier),
            "dropout": float(self.dropout),
            "mechanism_top_k": int(self.mechanism_top_k),
            "classifier_latent_dim": int(self.classifier_latent_dim),
            "analytic_history_identification_enabled": bool(
                self.history_identification_enabled
            ),
        }
        payload.update(copy.deepcopy(dict(self.v8_model_overrides)))
        return V8FeedbackModelConfig.from_mapping(payload)


@dataclass(frozen=True)
class V8FeedbackSystemContext:
    """V7 context plus exposure and paired-transition evidence metadata."""

    arrays: Any
    profile: Any
    geometry: Any
    exposure_s: np.ndarray
    exposure_known_mask: np.ndarray
    declared_calibration_hashes: Tuple[Optional[str], ...]
    calibration_known_mask: np.ndarray
    calibration_identity_match_mask: np.ndarray
    paired_episode_group_keys: Tuple[str, ...]
    reference_exposure_s: float
    measured_bundle_audit: Mapping[str, Any]
    locked_final_declaration: Mapping[str, Any]
    state_pair_transport_set: Any
    state_pair_validation_set: Any


class _FilteredTrainingArraysView:
    """Delegate immutable measured arrays while narrowing only train rows."""

    def __init__(self, arrays: Any, allowed_training_mask: np.ndarray) -> None:
        self._arrays = arrays
        allowed = np.asarray(allowed_training_mask, dtype=bool)
        if allowed.shape != (arrays.sample_count,):
            raise ValueError("training-view mask does not match measured rows")
        self._allowed_training_mask = allowed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._arrays, name)

    def split_mask(self, purpose: str) -> np.ndarray:
        base = np.asarray(self._arrays.split_mask(purpose), dtype=bool)
        if purpose == "train":
            return base & self._allowed_training_mask
        return base


@dataclass(frozen=True)
class V8RealActionSupervisionBatch:
    """Strict batch seam for future verified transitions or signed probes.

    Ordinary target-only snapshots never enter this structure.  Numeric action,
    Jacobian and rank labels must be supplied by a separate hash-bound loader;
    the current legacy corpus therefore produces the validated empty batch.
    """

    transition_ids: Tuple[str, ...]
    episode_ids: Tuple[str, ...]
    actual_action: np.ndarray
    action_mask: np.ndarray
    count_jacobian: np.ndarray
    jacobian_mask: np.ndarray
    rank_label: np.ndarray
    rank_mask: np.ndarray
    real_paired_or_signed_probe_mask: np.ndarray

    def validate(self) -> None:
        row_count = len(self.transition_ids)
        if len(self.episode_ids) != row_count:
            raise ValueError("real action transition and episode IDs differ")
        expected_shapes = {
            "actual_action": (row_count, 6),
            "action_mask": (row_count, 6),
            "rank_label": (row_count,),
            "rank_mask": (row_count,),
            "real_paired_or_signed_probe_mask": (row_count,),
        }
        for name, expected in expected_shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        jacobian = np.asarray(self.count_jacobian)
        jacobian_mask = np.asarray(self.jacobian_mask)
        if (
            jacobian.ndim != 3
            or jacobian.shape[0] != row_count
            or jacobian.shape[-1] != 6
            or jacobian_mask.shape != jacobian.shape
        ):
            raise ValueError("real count Jacobian and mask shapes changed")
        if row_count and not np.all(
            np.asarray(self.real_paired_or_signed_probe_mask) == 1.0
        ):
            raise ValueError("verified real action evidence mask must equal one")


def empty_real_action_supervision() -> V8RealActionSupervisionBatch:
    batch = V8RealActionSupervisionBatch(
        transition_ids=(),
        episode_ids=(),
        actual_action=np.empty((0, 6), dtype=np.float32),
        action_mask=np.empty((0, 6), dtype=np.float32),
        count_jacobian=np.empty((0, 0, 6), dtype=np.float32),
        jacobian_mask=np.empty((0, 0, 6), dtype=np.float32),
        rank_label=np.empty((0,), dtype=np.int64),
        rank_mask=np.empty((0,), dtype=np.float32),
        real_paired_or_signed_probe_mask=np.empty((0,), dtype=np.float32),
    )
    batch.validate()
    return batch


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("V8 feedback training requires torch")


def _stable_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_path(path: str | Path, *, relative_to: Optional[Path] = None) -> Path:
    selected = Path(path)
    if not selected.is_absolute() and relative_to is not None:
        selected = relative_to / selected
    return selected.resolve()


def _state_pair_builder_config(value: Mapping[str, Any]) -> Any:
    """Parse an explicit, fully hash-bound state-pair builder declaration."""

    from .state_pair_transport import StatePairTransportConfig

    defaults = asdict(StatePairTransportConfig())
    payload = dict(value)
    if set(payload) != set(defaults):
        missing = sorted(set(defaults) - set(payload))
        unknown = sorted(set(payload) - set(defaults))
        raise ValueError(
            "state-pair pair_builder must explicitly declare every field; "
            f"missing={missing}, unknown={unknown}"
        )
    parsed: Dict[str, Any] = {}
    for name, default in defaults.items():
        raw = payload[name]
        if isinstance(default, bool):
            if not isinstance(raw, bool):
                raise ValueError(f"state-pair pair_builder.{name} must be boolean")
            parsed[name] = raw
        elif isinstance(default, int):
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(f"state-pair pair_builder.{name} must be an integer")
            parsed[name] = int(raw)
        else:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"state-pair pair_builder.{name} must be numeric")
            parsed[name] = float(raw)
    builder = StatePairTransportConfig(**parsed)
    builder.validate()
    return builder


def validate_v8_config(config: Mapping[str, Any]) -> None:
    """Fail closed on every training, split, confidence and safety invariant."""

    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("V8 feedback config schema changed")
    from .feedback_data import validate_operator_curated_corpus_contract

    validate_operator_curated_corpus_contract(
        dict(config.get("operator_curated_corpus") or {})
    )
    environment = dict(
        dict(config.get("operator_curated_corpus") or {}).get(
            "environment_quality_contract"
        ) or {}
    )
    if (
        environment.get("temperature_training_view_candidate_allowed") is not True
        or environment.get("temperature_training_view_changes_hard_labels") is not False
        or environment.get("primary_validation_temperature_filter_allowed") is not False
        or environment.get("adoption_requires_multi_seed_full_validation_improvement")
        is not True
        or float(environment.get("candidate_temperature_max_c", math.nan)) != 22.0
        or tuple(environment.get("candidate_systems") or ()) != ("4qubit",)
    ):
        raise ValueError("temperature training-view candidate contract is invalid")
    environment_sha256 = str(environment.get("source_sha256") or "")
    if len(environment_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in environment_sha256
    ):
        raise ValueError("temperature training-view source SHA-256 is invalid")
    temperature_gate = dict(environment.get("candidate_adoption_gate") or {})
    if (
        int(temperature_gate.get("minimum_screen_seed_count", 0)) < 3
        or int(temperature_gate.get("minimum_final_seed_count", 0)) < 5
        or temperature_gate.get(
            "same_full_validation_sample_ids_and_hidden_masks_required"
        ) is not True
        or float(
            temperature_gate.get("mean_nb2_improvement_lcb95_exclusive", math.nan)
        ) != 0.0
        or temperature_gate.get("conditional_operation_nll_non_regression_required")
        is not True
        or temperature_gate.get("filtered_validation_allowed") is not False
        or float(
            temperature_gate.get(
                "operation_mapping_shuffle_degradation_lcb95_exclusive",
                math.nan,
            )
        ) != 0.0
    ):
        raise ValueError("temperature training-view adoption gate is invalid")
    real_loss = dict(config.get("real_loss_contract") or {})
    if float(real_loss.get("real_count_nb2_weight", -1.0)) != 1.0:
        raise ValueError("real NB2 weight is immutable and must equal 1.0")
    if real_loss.get("dynamic_real_count_downweighting_allowed") is not False:
        raise ValueError("dynamic real-count down-weighting must be disabled")
    if real_loss.get("all_real_auxiliary_losses_use_repeat_weight") is not True:
        raise ValueError("all measured auxiliary losses must use repeat weights")
    if (
        real_loss.get(
            "registered_operation_mapping_claim_requires_shuffle_degradation"
        ) is not True
        or real_loss.get("single_seed_operation_shuffle_has_selection_authority")
        is not False
    ):
        raise ValueError("measured operation-mapping negative control is invalid")
    weak_transport = dict(config.get("state_pair_transport_weak_contract") or {})
    if weak_transport.get("counts_as_real_causality_evidence") is not False:
        raise ValueError("weak state-pair transport cannot count as real causality")
    if int(weak_transport.get("real_action_supervision_mask", -1)) != 0:
        raise ValueError("weak state pairs cannot receive real action labels")
    if float(weak_transport.get("loss_weight", -1.0)) < 0.0:
        raise ValueError("weak state-pair transport weight must be non-negative")
    if weak_transport.get("budget_scope") != "per_system":
        raise ValueError("weak state-pair budgets must be declared per system")
    if min(
        int(weak_transport.get("screen_maximum_batches_per_system", 0)),
        int(weak_transport.get("full_maximum_batches_per_system", 0)),
    ) <= 0:
        raise ValueError("weak screen/full per-system budgets must be positive")
    if int(weak_transport.get("screen_validation_maximum_pair_rows_per_system", 0)) <= 0:
        raise ValueError("weak screen validation budget must be positive")
    if int(weak_transport.get("full_validation_maximum_pair_rows_per_system", -1)) < 0:
        raise ValueError("weak full validation budget must be zero (all) or positive")
    if not tuple(weak_transport.get("validation_split_names") or ()):
        raise ValueError("weak validation split names must be explicit")
    if weak_transport.get("prioritize_every_unique_anchor_before_repeating") is not True:
        raise ValueError("weak sampler must prioritize unique anchors before repeats")
    if float(weak_transport.get("endpoint_hidden_fraction", -1.0)) != 0.0:
        raise ValueError("weak endpoint transport must use fully observed operation rows")
    if weak_transport.get("weak_classifier_prior_allowed") is not False:
        raise ValueError("weak endpoint transport cannot ingest rho classifier priors")
    backbone_scale = float(weak_transport.get("weak_backbone_gradient_scale", -1.0))
    if not 0.0 <= backbone_scale <= 0.1:
        raise ValueError("weak backbone gradient scale must stay in [0, 0.1]")
    from .feedback_models import (
        MEASUREMENT_TRANSPORT_CROSS_CALIBRATION_FEATURES,
        MEASUREMENT_TRANSPORT_EXCLUDED_SHORTCUT_FEATURES,
        MEASUREMENT_TRANSPORT_FEATURE_NAMES,
        V8FeedbackModelConfig,
    )

    semantic_contracts = (
        (
            "measurement_transport_feature_names",
            MEASUREMENT_TRANSPORT_FEATURE_NAMES,
        ),
        (
            "cross_calibration_feature_names",
            MEASUREMENT_TRANSPORT_CROSS_CALIBRATION_FEATURES,
        ),
        (
            "measurement_transport_excluded_shortcuts",
            MEASUREMENT_TRANSPORT_EXCLUDED_SHORTCUT_FEATURES,
        ),
    )
    for key, expected in semantic_contracts:
        actual = tuple(str(value) for value in weak_transport.get(key, ()))
        if actual != tuple(expected):
            raise ValueError(f"weak transport semantic contract disagrees for {key}")
    expected_proxy_scales = tuple(
        float(value) for value in V8FeedbackModelConfig().target_transport_proxy_scales
    )
    actual_proxy_scales = tuple(
        float(value) for value in weak_transport.get(
            "target_transport_proxy_scales", ()
        )
    )
    if actual_proxy_scales != expected_proxy_scales:
        raise ValueError("weak target-transport proxy scales disagree with the model")
    _state_pair_builder_config(dict(weak_transport.get("pair_builder") or {}))
    if weak_transport.get(
        "strict_effective_tier_requires_known_equal_calibration_hashes"
    ) is not True:
        raise ValueError("strict weak transport requires known equal calibration hashes")
    if weak_transport.get("strict_effective_tier_requires_known_exposure") is not True:
        raise ValueError("strict weak transport requires known exposure")
    if weak_transport.get("unknown_strict_pair_fallback") != (
        "cross_calibration_scale_free_encoder"
    ):
        raise ValueError("unknown weak pair fallback must use scale-free features")
    for weight_name in (
        "endpoint_static_term_weight",
        "relational_transport_term_weight",
    ):
        if float(weak_transport.get(weight_name, -1.0)) <= 0.0:
            raise ValueError(f"{weight_name} must be positive")
    paired_comparison = dict(weak_transport.get("paired_comparison") or {})
    if paired_comparison.get("schema") != WEAK_PAIRED_COMPARISON_GATE_SCHEMA:
        raise ValueError("weak paired-comparison gate schema changed")
    for required_true in (
        "allowed_for_model_selection",
        "require_same_seed",
        "require_checkpoint_hash_difference",
        "require_development_validation_negative_control",
    ):
        if paired_comparison.get(required_true) is not True:
            raise ValueError(f"weak paired-comparison {required_true} must be true")
    expected_static_components = {
        "negative_log_real_nb2",
        "negative_real_ece",
        "negative_coverage_error",
        "real_rho_fidelity",
    }
    static_component_weights = dict(
        paired_comparison.get("static_component_weights") or {}
    )
    if set(static_component_weights) != expected_static_components or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in static_component_weights.values()
    ):
        raise ValueError(
            "weak paired-comparison static component weights must be explicit and positive"
        )
    finite_thresholds = (
        "minimum_static_composite_gain_exclusive",
        "minimum_joint_selection_score_gain",
        "minimum_absolute_static_component_delta",
    )
    if any(
        not math.isfinite(float(paired_comparison.get(name, float("nan"))))
        for name in finite_thresholds
    ):
        raise ValueError("weak paired-comparison thresholds must be finite")
    if float(paired_comparison["minimum_absolute_static_component_delta"]) <= 0.0:
        raise ValueError("weak paired-comparison non-degeneracy threshold must be positive")
    improved_count = int(
        paired_comparison.get("minimum_improved_static_component_count", 0)
    )
    if not 1 <= improved_count <= len(expected_static_components):
        raise ValueError("weak paired-comparison improved component count is invalid")

    ablations = dict(config.get("ablations") or {})
    if tuple(sorted(ablations)) != REQUIRED_ABLATIONS:
        raise ValueError("V8 config must declare exactly A0 through A4")
    for ablation_id, value in ablations.items():
        if str(value.get("id")) != ablation_id:
            raise ValueError(f"ablation identity mismatch for {ablation_id}")
        overrides = dict(value.get("loss_weight_overrides") or {})
        if "real_count_nb2" in overrides and float(overrides["real_count_nb2"]) != 1.0:
            raise ValueError("an ablation may not down-weight real NB2")

    profiles = dict(config.get("training_profiles") or {})
    if not profiles:
        raise ValueError("at least one V8 training profile is required")
    for profile_name, value in profiles.items():
        ablation_id = str(value.get("ablation_id", ""))
        if ablation_id not in ablations:
            raise ValueError(f"profile {profile_name} has an unknown ablation")
        overrides = dict(value.get("loss_weight_overrides") or {})
        if "real_count_nb2" in overrides and float(overrides["real_count_nb2"]) != 1.0:
            raise ValueError(f"profile {profile_name} may not down-weight real NB2")
        if str(value.get("state_pair_budget_phase", "screen")) not in {
            "screen", "full"
        }:
            raise ValueError(
                f"profile {profile_name} has an invalid state-pair budget phase"
            )
    paired_profiles: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = {}
    for profile_name, value in profiles.items():
        pair_id = str(value.get("weak_supervision_pair_id", ""))
        if pair_id:
            paired_profiles.setdefault(pair_id, []).append((profile_name, value))
    for pair_id, pair_rows in paired_profiles.items():
        if len(pair_rows) != 2:
            raise ValueError(
                f"weak-supervision comparison {pair_id} must contain exactly two profiles"
            )
        enabled_rows = [
            (name, value)
            for name, value in pair_rows
            if bool(value.get("weak_state_pair_supervision_enabled", False))
        ]
        disabled_rows = [
            (name, value)
            for name, value in pair_rows
            if not bool(value.get("weak_state_pair_supervision_enabled", False))
        ]
        if len(enabled_rows) != 1 or len(disabled_rows) != 1:
            raise ValueError(
                f"weak-supervision comparison {pair_id} requires one enabled and one disabled profile"
            )
        enabled_name, enabled_value = enabled_rows[0]
        disabled_name, disabled_value = disabled_rows[0]
        if str(disabled_value.get("weak_supervision_control_for", "")) != enabled_name:
            raise ValueError(
                f"weak-supervision control {disabled_name} must name {enabled_name} as its counterpart"
            )
        comparison_keys = {
            "weak_state_pair_supervision_enabled",
            "weak_supervision_pair_id",
            "weak_supervision_control_for",
        }
        normalized_enabled = {
            key: copy.deepcopy(item)
            for key, item in enabled_value.items()
            if key not in comparison_keys
        }
        normalized_disabled = {
            key: copy.deepcopy(item)
            for key, item in disabled_value.items()
            if key not in comparison_keys
        }
        if normalized_enabled != normalized_disabled:
            raise ValueError(
                f"weak-supervision comparison {pair_id} is not capacity and budget matched"
            )

    family_contract = dict(config.get("synthetic_plant_families") or {})
    partitions = {
        key: tuple(str(item) for item in family_contract.get(key, ()))
        for key in ("train", "development", "locked_ood")
    }
    if any(not value for value in partitions.values()):
        raise ValueError("train, development and locked OOD plants must all be declared")
    if len(set().union(*[set(value) for value in partitions.values()])) < 3:
        raise ValueError("V8 requires at least three synthetic plant families")
    for left_name, left in partitions.items():
        for right_name, right in partitions.items():
            if left_name < right_name and set(left).intersection(right):
                raise ValueError("synthetic plant-family partitions must be disjoint")

    evaluation = dict(config.get("evaluation_contract") or {})
    development = dict(evaluation.get("development") or {})
    locked = dict(evaluation.get("locked_final_test") or {})
    if development.get("allowed_for_model_selection") is not True:
        raise ValueError("development evaluation must be the only selection scope")
    if locked.get("allowed_for_model_selection") is not False:
        raise ValueError("locked final-test metrics must never select a model")
    if locked.get("single_use") is not True or int(locked.get("maximum_access_count", 0)) != 1:
        raise ValueError("locked final test must be declared one-time access")
    if locked.get("declaration_required") is not True:
        raise ValueError("locked final test requires a hash-bound declaration")

    uncertainty = dict(config.get("uncertainty_contract") or {})
    if uncertainty.get("calibration_source") != "real_paired_transition_episode_only":
        raise ValueError("uncertainty calibration may use real paired episodes only")
    if uncertainty.get("group_key") != "episode_id":
        raise ValueError("bootstrap and conformal calibration must group by episode")
    if uncertainty.get("synthetic_calibration_allowed") is not False:
        raise ValueError("synthetic rows may not calibrate reported confidence")
    if uncertainty.get("fail_closed_without_real_pairs") is not True:
        raise ValueError("missing real paired transitions must force abstention")

    exposure = dict(config.get("real_exposure_contract") or {})
    if exposure.get("known_mask_required") is not True:
        raise ValueError("rate-exposure training requires exposure_known_mask")
    if exposure.get("unknown_exposure_enters_equivariance_gate") is not False:
        raise ValueError("unknown exposure may not enter exposure-equivariance gates")
    if exposure.get("legacy_3qubit_zero_means_unknown") is not True:
        raise ValueError("legacy 3Q zero exposure must be interpreted as unknown")

    rho_estimator = dict(config.get("rho_estimator_contract") or {})
    if (
        rho_estimator.get("schema")
        != "physics_hybrid_v8_rho_estimator_contract_v2"
        or rho_estimator.get("source_semantic")
        != "calibrated_reconstruction_estimate_from_the_same_count_snapshot"
        or rho_estimator.get("independent_experimental_measurement") is not False
        or rho_estimator.get("soft_teacher_and_physical_consistency_training_allowed")
        is not True
        or rho_estimator.get("independent_evidence_double_counting_allowed") is not False
        or rho_estimator.get("missing_rho_requires_zero_mask") is not True
        or rho_estimator.get("full_density_ground_truth_claim_allowed") is not False
        or rho_estimator.get(
            "derived_rho_classifier_latent_allowed_in_real_count_path"
        ) is not False
    ):
        raise ValueError("rho estimator provenance/soft-teacher contract is invalid")
    rho_contribution_cap = float(
        rho_estimator.get(
            "maximum_weighted_contribution_fraction_of_real_count_nb2", 0.0
        )
    )
    if not 0.0 < rho_contribution_cap <= RHO_SOFT_TEACHER_MAXIMUM_FRACTION:
        raise ValueError("rho soft-teacher contribution cap must lie in (0, 0.05]")
    rho_backbone_gradient_scale = float(
        rho_estimator.get("shared_backbone_gradient_scale_on_real_rows", -1.0)
    )
    if not 0.0 <= rho_backbone_gradient_scale <= 1.0:
        raise ValueError("real rho shared-backbone gradient scale must lie in [0, 1]")

    safety = dict(config.get("safety") or {})
    if safety.get("promotion_ready") is not False:
        raise ValueError("V8 training config cannot grant promotion readiness")
    if safety.get("hardware_action_authorized") is not False:
        raise ValueError("V8 training config cannot grant hardware authority")
    if safety.get("live_gates_required") is not True:
        raise ValueError("live gates must remain required")

    execution = dict(config.get("execution_plans") or {})
    required_plans = {
        "contract_smoke",
        "vram_probe",
        "local_3seed",
        "confirmation_5seed",
        "server_4x4090",
        "server_8x4090",
    }
    if not required_plans.issubset(execution):
        raise ValueError("V8 execution plan is incomplete")


def load_v8_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    config = _read_json(config_path)
    validate_v8_config(config)
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = _file_sha256(config_path)
    return config


def _load_v7_base_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    config_path = Path(str(config["_config_path"]))
    declared_path = Path(str(config["v7_base_config_path"]))
    base_path = (
        declared_path.resolve()
        if declared_path.is_absolute()
        else (Path.cwd() / declared_path).resolve()
    )
    base = _read_json(base_path)
    if base.get("schema") != "physics_hybrid_v7_feedback_training_v1":
        raise ValueError("V8 requires the declared V7 feedback base config")
    base["loss_weights"]["real_count_nb2"] = 1.0
    base["loss_weights"]["real_state_pair_transport_weak"] = float(
        config["state_pair_transport_weak_contract"]["loss_weight"]
    )
    for key in ("data", "optimization", "policy", "selection"):
        base[key].update(copy.deepcopy(dict(config.get(key) or {})))
    return base


def _resolve_profile_mapping(
    profile_name: str,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    *,
    smoke: bool,
) -> V8FeedbackTrainingProfile:
    profile_decl = dict(config["training_profiles"][profile_name])
    base_name = str(profile_decl.pop("v7_base_profile"))
    base_value = copy.deepcopy(dict(base_config["profiles"][base_name]))
    ablation_id = str(profile_decl.pop("ablation_id"))
    ablation = dict(config["ablations"][ablation_id])
    profile_model_overrides = dict(profile_decl.pop("model_overrides", {}))
    base_value.update(profile_decl)
    loss_overrides = dict(base_value.get("loss_weight_overrides", {}))
    loss_overrides.update(dict(ablation.get("loss_weight_overrides", {})))
    if "real_count_nb2" in loss_overrides and float(loss_overrides["real_count_nb2"]) != 1.0:
        raise ValueError("real NB2 may not be changed by a resolved profile")
    loss_overrides["real_count_nb2"] = 1.0
    base_value["loss_weight_overrides"] = loss_overrides
    if smoke:
        smoke_plan = dict(config["execution_plans"]["contract_smoke"])
        base_value.update({
            "real_batch_size": int(smoke_plan["real_batch_size"]),
            "synthetic_batch_size": int(smoke_plan["synthetic_batch_size"]),
            "measured_pretrain_epochs": int(smoke_plan["measured_pretrain_epochs"]),
            "synthetic_model_a_steps": int(smoke_plan["synthetic_model_a_steps"]),
            "synthetic_model_b_steps": int(smoke_plan["synthetic_model_b_steps"]),
            "measured_adapter_epochs": int(smoke_plan["measured_adapter_epochs"]),
            "maximum_real_batches_per_system": int(
                smoke_plan["maximum_real_batches_per_system"]
            ),
            "evaluation_max_samples_per_split": int(
                smoke_plan["evaluation_max_samples_per_split"]
            ),
        })
    model_overrides = dict(config.get("model") or {})
    model_overrides.update(dict(ablation.get("model_overrides") or {}))
    model_overrides.update(profile_model_overrides)
    return V8FeedbackTrainingProfile.from_mapping(
        profile_name,
        base_value,
        ablation_id=ablation_id,
        v8_model_overrides=model_overrides,
    )


def resolve_v8_profiles(
    config: Mapping[str, Any],
    profile_names: Optional[Sequence[str]] = None,
    *,
    ablation_id: Optional[str] = None,
    smoke: bool = False,
) -> Tuple[Dict[str, Any], List[V8FeedbackTrainingProfile]]:
    base_config = _load_v7_base_config(config)
    selected_names = list(profile_names or config["default_profiles"])
    unknown = sorted(set(selected_names).difference(config["training_profiles"]))
    if unknown:
        raise ValueError(f"unknown V8 training profiles: {unknown}")
    profiles = [
        _resolve_profile_mapping(name, config, base_config, smoke=smoke)
        for name in selected_names
    ]
    if ablation_id is not None:
        if ablation_id not in REQUIRED_ABLATIONS:
            raise ValueError(f"unknown ablation {ablation_id}")
        remapped: List[V8FeedbackTrainingProfile] = []
        for profile in profiles:
            profile_decl = copy.deepcopy(dict(config["training_profiles"][profile.name]))
            profile_decl["ablation_id"] = str(ablation_id)
            temporary = copy.deepcopy(dict(config))
            temporary["training_profiles"] = copy.deepcopy(
                dict(config["training_profiles"])
            )
            temporary["training_profiles"][profile.name] = profile_decl
            remapped.append(
                _resolve_profile_mapping(profile.name, temporary, base_config, smoke=smoke)
            )
        profiles = remapped
    return base_config, profiles


def _optional_v8_bundle_metadata(
    config: Mapping[str, Any],
) -> Tuple[
    Dict[str, Dict[str, Tuple[float, float, str, Optional[str], float]]],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    """Load V8 contract rows when available; absence remains fail-closed."""

    try:
        from .contracts import MeasuredAuditWaiverV1
        from .feedback_data import load_v8_measured_data_bundle
    except (ImportError, AttributeError):
        return (
            {},
            {
                "audit_profile_passed": False,
                "waiver_applied": False,
                "failed_checks": ["v8_measured_bundle_loader_unavailable"],
            },
            {},
            {},
            {},
        )
    from .feedback_data import declare_locked_final_test
    waiver_payload = dict(config.get("measured_audit_waiver") or {})
    waiver = (
        MeasuredAuditWaiverV1.from_json_dict(waiver_payload)
        if waiver_payload
        else None
    )
    bundle = load_v8_measured_data_bundle(
        config["data"]["manifest_dir"],
        audit_waiver=waiver,
        curated_corpus_contract=config["operator_curated_corpus"],
    )
    result: Dict[
        str,
        Dict[str, Tuple[float, float, str, Optional[str], float]],
    ] = {}
    for row in bundle.rows:
        record = row.record
        system_type = str(record.system_type)
        episode_key = str(getattr(row, "paired_episode_group_key", "") or "")
        result.setdefault(system_type, {})[str(record.sample_id)] = (
            float(row.exposure_s) if row.exposure_s is not None else math.nan,
            float(row.exposure_known_mask),
            episode_key,
            row.calibration_hash,
            float(row.calibration_known_mask),
        )
    locked_config = config["evaluation_contract"]["locked_final_test"]
    locked_declaration = declare_locked_final_test(
        bundle,
        final_test_split_names=locked_config["split_names"],
        locked_at=str(locked_config["locked_at"]),
    )
    from .state_pair_transport import build_state_pair_transport

    train_split_names = tuple(
        str(item) for item in config["state_pair_transport_weak_contract"]["train_split_names"]
    )
    validation_split_names = tuple(
        str(item)
        for item in config["state_pair_transport_weak_contract"][
            "validation_split_names"
        ]
    )
    pair_builder_config = _state_pair_builder_config(
        config["state_pair_transport_weak_contract"]["pair_builder"]
    )
    state_pair_sets: Dict[str, Any] = {}
    state_pair_validation_sets: Dict[str, Any] = {}
    for system_type in config["data"]["systems"]:
        system_rows = [
            row for row in bundle.rows if row.system_type == system_type
        ]
        available_splits = {row.split_name for row in system_rows}
        selected_splits = tuple(
            name for name in train_split_names if name in available_splits
        )
        state_pair_sets[str(system_type)] = build_state_pair_transport(
            system_rows,
            split_names=selected_splits,
            purpose="train_gradient",
            config=pair_builder_config,
            locked_split_names=locked_config["split_names"],
        )
        selected_validation_splits = tuple(
            name for name in validation_split_names if name in available_splits
        )
        state_pair_validation_sets[str(system_type)] = build_state_pair_transport(
            system_rows,
            split_names=selected_validation_splits,
            purpose="validation_evaluation",
            config=pair_builder_config,
            locked_split_names=locked_config["split_names"],
        )
    measured_audit = bundle.audit.to_json_dict()
    measured_audit["operator_curated_corpus"] = copy.deepcopy(
        bundle.curated_corpus_audit
    )
    return (
        result,
        measured_audit,
        locked_declaration.to_json_dict(),
        state_pair_sets,
        state_pair_validation_sets,
    )


def load_v8_contexts(
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> Dict[str, V8FeedbackSystemContext]:
    del runtime_config
    from .dlcz_noise_data import (
        build_v8_a0_development_contexts,
        load_v8_dlcz_development_data,
    )

    waiver_payload = dict(config.get("measured_audit_waiver") or {})
    development_data = load_v8_dlcz_development_data(
        config["data"]["manifest_dir"],
        data_root=config["data"].get("data_root", "data"),
        audit_waiver=waiver_payload or None,
        curated_corpus_contract=config["operator_curated_corpus"],
        include_raw_registers=False,
    )
    base_contexts, development_audit = build_v8_a0_development_contexts(
        development_data,
        manifest_dir=config["data"]["manifest_dir"],
        audit_waiver=waiver_payload or None,
        curated_corpus_contract=config["operator_curated_corpus"],
        reference_exposure_s=float(
            config["real_exposure_contract"]["reference_exposure_s"]
        ),
    )
    (
        metadata,
        measured_bundle_audit,
        locked_final_declaration,
        state_pair_sets,
        state_pair_validation_sets,
    ) = _optional_v8_bundle_metadata(config)
    result: Dict[str, V8FeedbackSystemContext] = {}
    for system_type, context in base_contexts.items():
        if system_type not in metadata:
            raise ValueError(f"V8 metadata has no {system_type} rows")
        expected_sample_ids = set(metadata[system_type])
        if not set(context.arrays.sample_ids).issubset(expected_sample_ids):
            raise ValueError("development context sample IDs are absent from V8 metadata")
        context_audit = copy.deepcopy(measured_bundle_audit)
        context_audit["development_payload_audit"] = copy.deepcopy(
            development_audit
        )
        result[system_type] = replace(
            context,
            measured_bundle_audit=context_audit,
            locked_final_declaration=copy.deepcopy(locked_final_declaration),
            state_pair_transport_set=state_pair_sets.get(system_type),
            state_pair_validation_set=state_pair_validation_sets.get(system_type),
        )
    return result


def weighted_real_auxiliary_mean(
    value: "torch.Tensor",
    repeat_weight: "torch.Tensor",
) -> "torch.Tensor":
    """Return a row-wise repeat-aware mean used by every measured auxiliary."""

    selected = value
    while selected.ndim > 1:
        selected = selected.mean(dim=-1)
    weight = repeat_weight.to(dtype=selected.dtype).reshape(-1)
    if selected.shape[0] != weight.shape[0]:
        raise ValueError("real auxiliary and repeat weight batch sizes differ")
    return (selected * weight).sum() / weight.sum().clamp_min(1e-8)


def bounded_auxiliary_contribution(
    requested: "torch.Tensor",
    primary: "torch.Tensor",
    maximum_fraction: float,
) -> "torch.Tensor":
    """Bound an auxiliary value while preserving its optimization gradient.

    ``torch.minimum(requested, detached_cap)`` makes the auxiliary gradient
    exactly zero whenever the cap is active.  The detached multiplicative
    scale below enforces the same forward bound without disabling the teacher.
    """

    fraction = float(maximum_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("auxiliary maximum_fraction must lie in (0, 1]")
    if requested.ndim != 0 or primary.ndim != 0:
        raise ValueError("bounded auxiliary contribution requires scalar losses")
    cap = fraction * primary.detach().clamp_min(0.0)
    scale = torch.minimum(
        torch.ones_like(requested),
        cap / requested.detach().clamp_min(1e-12),
    )
    return requested * scale


def shared_gamma_total_nb2_conditional_multinomial_nll(
    observed: "torch.Tensor",
    mean: "torch.Tensor",
    dispersion: "torch.Tensor",
    hidden_mask: "torch.Tensor",
    repeat_weight: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Exact row likelihood for shared brightness and operation allocation.

    A Gamma-distributed row brightness gives an NB2 hidden total.  Conditional
    on that total, registered hidden operations follow a multinomial with
    probabilities proportional to the predicted operation means.  Repeat
    weights are applied once per row, never to the integer event counts.
    """

    if observed.shape != mean.shape or hidden_mask.shape != observed.shape:
        raise ValueError("shared-brightness tensors must share [B, O] shape")
    if repeat_weight.shape != (observed.shape[0],):
        raise ValueError("shared-brightness repeat weights must have shape [B]")
    mask = (hidden_mask > 0).to(dtype=mean.dtype)
    valid = mask.sum(dim=1) > 0
    if not bool(torch.any(valid)):
        raise ValueError("shared-brightness likelihood requires hidden operations")
    counts = observed.to(dtype=mean.dtype) * mask
    means = mean.clamp_min(1e-8) * mask
    observed_total = counts.sum(dim=1)
    expected_total = means.sum(dim=1).clamp_min(1e-8)
    total_log_probability = torch_nb2_log_probability(
        observed_total,
        expected_total,
        dispersion,
    )
    probability = means / expected_total[:, None]
    conditional_log_probability = (
        torch.lgamma(observed_total + 1.0)
        - torch.lgamma(counts + 1.0).sum(dim=1)
        + (counts * torch.log(probability.clamp_min(1e-8))).sum(dim=1)
    )
    weight = repeat_weight.to(dtype=mean.dtype) * valid.to(dtype=mean.dtype)
    denominator = weight.sum().clamp_min(1e-8)
    total_nll = -(total_log_probability * weight).sum() / denominator
    conditional_nll = -(conditional_log_probability * weight).sum() / denominator
    return total_nll + conditional_nll, total_nll, conditional_nll


def _v8_measured_batch(
    context: V8FeedbackSystemContext,
    indices: Sequence[int],
    **kwargs: Any,
) -> Dict[str, "torch.Tensor"]:
    selected = np.asarray(indices, dtype=np.int64)
    device = kwargs["device"]
    hidden_fraction = float(kwargs["hidden_fraction"])
    original_kwargs = dict(kwargs)
    # ``rho_full`` is reconstructed from this same measurement snapshot.  A
    # frozen classifier latent derived from it can therefore carry information
    # about coin cells that the deterministic hidden mask is supposed to hide.
    # Real count fitting and validation must remain count/target/registry only;
    # synthetic classifier-prior experiments are kept on their separate path.
    original_kwargs["classifier_prior"] = None
    if hidden_fraction == 0.0:
        # V7's deterministic mask utility intentionally accepts only (0, 1).
        # Build its non-mask fields through the existing path, then replace the
        # operation view with an exact fully observed endpoint representation.
        original_kwargs["hidden_fraction"] = 0.5
    batch = _ORIGINAL_V7_FUNCTIONS["_measured_batch"](
        context, indices, **original_kwargs
    )
    if hidden_fraction == 0.0:
        arrays = context.arrays
        visible = np.ones(
            (len(selected), arrays.operation_count), dtype=np.float32
        )
        operation_features = v7_training.build_operation_features(
            coin_counts=arrays.coin_counts[selected],
            signal_counts=arrays.signal_counts[selected],
            idler_counts=arrays.idler_counts[selected],
            read_count=arrays.read_count[selected],
            target_amplitude=arrays.target_amplitude[selected],
            target_phase_rad=arrays.target_phase_rad[selected],
            geometry=context.geometry,
            static_operation_features=arrays.static_operation_features,
            retrieval_relative_map=arrays.retrieval_relative_map,
            observed_coin_mask=visible,
        )
        batch["operation_features"] = torch.from_numpy(operation_features).to(device)
        batch["hidden_mask"] = torch.zeros(
            len(selected), arrays.operation_count, dtype=torch.float32, device=device
        )
    elif not 0.0 < hidden_fraction < 1.0:
        raise ValueError("measured hidden_fraction must be 0 or lie in (0, 1)")
    batch["exposure_s"] = torch.from_numpy(context.exposure_s[selected]).to(device)
    batch["exposure_known_mask"] = torch.from_numpy(
        context.exposure_known_mask[selected]
    ).to(device)
    batch["calibration_known_mask"] = torch.from_numpy(
        context.calibration_known_mask[selected]
    ).to(device)
    batch["calibration_identity_match_mask"] = torch.from_numpy(
        context.calibration_identity_match_mask[selected]
    ).to(device)
    batch["real_paired_or_signed_probe_mask"] = torch.zeros(
        len(selected), dtype=torch.float32, device=device
    )
    batch["command_features"] = torch.zeros(
        len(selected),
        context.arrays.operation_count,
        18,
        dtype=torch.float32,
        device=device,
    )
    batch["command_mask"] = torch.zeros_like(batch["command_features"])
    batch["derived_rho_classifier_latent_allowed_mask"] = torch.zeros(
        len(selected), dtype=torch.float32, device=device
    )
    return batch


def _v8_measured_loss(
    model: Any,
    batch: Mapping[str, "torch.Tensor"],
    context: V8FeedbackSystemContext,
    weights: Mapping[str, float],
) -> Tuple["torch.Tensor", Dict[str, float]]:
    """Unit-weight real NB2 plus repeat-weighted measured auxiliaries."""

    if float(weights.get("real_count_nb2", -1.0)) != 1.0:
        raise ValueError("real NB2 weight is immutable and must equal 1.0")
    output = model.model_a(
        batch["operation_features"],
        batch["target_features"],
        batch["graph_kernel"],
        batch["action"],
        batch["action_mask"],
        system_type=context.arrays.system_type,
        classifier_latent=batch.get("classifier_latent"),
        classifier_uncertainty=batch.get("classifier_uncertainty"),
        exposure_s=batch.get("exposure_s"),
        exposure_known_mask=batch.get("exposure_known_mask"),
        command_features=batch.get("command_features"),
        command_mask=batch.get("command_mask"),
        calibration_known_mask=batch.get("calibration_known_mask"),
        calibration_identity_match_mask=batch.get(
            "calibration_identity_match_mask"
        ),
        rho_shared_backbone_gradient_scale=float(
            weights.get("real_rho_shared_backbone_gradient_scale", 0.0)
        ),
    )
    repeat_weight = batch["sample_weight"]
    count_mask = batch["hidden_mask"] * repeat_weight[:, None]
    likelihood_kind = str(model.config.real_count_likelihood_kind)
    if likelihood_kind == "operation_independent_nb2":
        primary = v7_training.torch_nb2_nll(
            batch["coin_counts"],
            output["expected_counts"],
            output["nb2_dispersion"],
            count_mask,
        )
        shared_total_nll = primary.detach() * 0.0
        conditional_multinomial_nll = primary.detach() * 0.0
    elif likelihood_kind == "shared_gamma_total_nb2_plus_conditional_multinomial":
        primary, shared_total_nll, conditional_multinomial_nll = (
            shared_gamma_total_nb2_conditional_multinomial_nll(
                batch["coin_counts"],
                output["expected_counts"],
                output["nb2_dispersion"],
                batch["hidden_mask"],
                repeat_weight,
            )
        )
    else:
        raise ValueError(f"unsupported real count likelihood: {likelihood_kind}")
    total_per_row = functional.smooth_l1_loss(
        torch.log1p(output["expected_total"]),
        torch.log1p(batch["coin_counts"].sum(dim=1)),
        reduction="none",
    )
    total_loss = weighted_real_auxiliary_mean(total_per_row, repeat_weight)
    rho_loss = v7_training._weighted_density_infidelity(
        output["rho_current"],
        batch["rho"],
        batch["rho_mask"],
        repeat_weight,
    )
    target_dispersion = torch.full_like(
        output["nb2_dispersion"],
        float(context.profile.nb2_dispersion),
    )
    dispersion_per_row = functional.smooth_l1_loss(
        torch.log(output["nb2_dispersion"].clamp_min(1e-6)),
        torch.log(target_dispersion.clamp_min(1e-6)),
        reduction="none",
    )
    dispersion_loss = weighted_real_auxiliary_mean(
        dispersion_per_row,
        repeat_weight,
    )
    multi_per_row = torch.square(functional.relu(
        output["multi_interval_prediction"] - batch["multi_upper_rates"]
    )).mean(dim=1)
    multi_weight = repeat_weight * batch["multi_interval_mask"]
    multi_raw = weighted_real_auxiliary_mean(multi_per_row, multi_weight)
    requested_multi = float(weights["real_multi_interval"]) * multi_raw
    multi_contribution = bounded_auxiliary_contribution(
        requested_multi,
        primary,
        0.01,
    )
    transport_weight = float(weights.get("real_state_pair_transport_weak", 0.0))
    transport_loss = output["static_transport_embedding"].sum() * 0.0
    transport_pair_count = 0
    if (
        transport_weight > 0.0
        and "state_pair_positive_index" in batch
        and "state_pair_deranged_index" in batch
    ):
        from .feedback_models import static_transport_compatibility

        positive_index = batch["state_pair_positive_index"].long()
        negative_index = batch["state_pair_deranged_index"].long()
        embedding = output["static_transport_embedding"]
        row_index = torch.arange(embedding.shape[0], device=embedding.device)
        if bool(torch.any(negative_index == row_index)):
            raise ValueError("state endpoint negative control contains a self-pair")
        positive = static_transport_compatibility(
            embedding,
            embedding[positive_index],
            temperature=float(model.config.transport_temperature),
        )
        negative = static_transport_compatibility(
            embedding,
            embedding[negative_index],
            temperature=float(model.config.transport_temperature),
        )
        transport_per_row = functional.softplus(negative - positive)
        transport_loss = weighted_real_auxiliary_mean(
            transport_per_row,
            repeat_weight,
        )
        transport_pair_count = int(embedding.shape[0])
    requested_rho = float(weights["real_rho"]) * rho_loss
    rho_cap_fraction = float(
        weights.get(
            "real_rho_soft_teacher_maximum_fraction",
            RHO_SOFT_TEACHER_MAXIMUM_FRACTION,
        )
    )
    rho_contribution = bounded_auxiliary_contribution(
        requested_rho,
        primary,
        rho_cap_fraction,
    )
    loss = (
        primary
        + float(weights["real_total"]) * total_loss
        + rho_contribution
        + float(weights["real_dispersion"]) * dispersion_loss
        + multi_contribution
        + transport_weight * transport_loss
    )
    return loss, {
        "loss": float(loss.detach()),
        "count_nb2": float(primary.detach()),
        "real_count_likelihood_kind_shared_gamma": float(
            likelihood_kind
            == "shared_gamma_total_nb2_plus_conditional_multinomial"
        ),
        "shared_brightness_total_nb2": float(shared_total_nll.detach()),
        "conditional_operation_multinomial": float(
            conditional_multinomial_nll.detach()
        ),
        "real_count_nb2_effective_weight": 1.0,
        "total": float(total_loss.detach()),
        "rho": float(rho_loss.detach()),
        "rho_estimator_is_independent_measurement": 0.0,
        "rho_soft_teacher_contribution": float(rho_contribution.detach()),
        "rho_soft_teacher_fraction_of_primary": float(
            rho_contribution.detach() / primary.detach().clamp_min(1e-8)
        ),
        "rho_soft_teacher_maximum_fraction_of_primary": rho_cap_fraction,
        "dispersion": float(dispersion_loss.detach()),
        "multi_raw": float(multi_raw.detach()),
        "multi_contribution": float(multi_contribution.detach()),
        "multi_fraction_of_primary": float(
            multi_contribution.detach() / primary.detach().clamp_min(1e-8)
        ),
        "all_real_auxiliary_losses_repeat_weighted": 1.0,
        "state_pair_transport_weak": float(transport_loss.detach()),
        "state_pair_transport_weak_pair_count": float(transport_pair_count),
        "state_pair_transport_is_real_causality_evidence": 0.0,
        "real_action_response_loss_contribution": 0.0,
        "real_action_supervision_mask_sum": float(
            batch["real_paired_or_signed_probe_mask"].sum().detach()
        ),
        "historical_command_authoritative_mask_sum": float(
            batch["command_mask"].sum().detach()
        ),
    }


def _initialize_v8_physical_head_biases(
    model: Any,
    contexts: Mapping[str, V8FeedbackSystemContext],
) -> None:
    """Initialize log-rate, not log-count, while preserving other V7 priors."""

    with torch.no_grad():
        for system_type, context in contexts.items():
            reference_exposure = max(float(context.reference_exposure_s), 1e-8)
            model.model_a.total_heads[system_type].bias.fill_(
                float(context.profile.log_coin_total_mean) - math.log(reference_exposure)
            )
            model.model_a.uncalibrated_log_total_heads[system_type].bias.fill_(
                float(context.profile.log_coin_total_mean)
            )
            model.model_a.dispersion_heads[system_type].bias.fill_(
                v7_training._inverse_softplus(
                    min(context.profile.nb2_dispersion, 0.70)
                )
            )
            upper = np.maximum(
                np.asarray(
                    context.profile.multi_upper_rate_quantiles[1],
                    dtype=np.float64,
                ),
                1e-6,
            )
            model.model_a.multi_interval_heads[system_type].bias.copy_(
                torch.as_tensor(
                    [v7_training._inverse_softplus(float(item)) for item in upper],
                    dtype=model.model_a.multi_interval_heads[system_type].bias.dtype,
                )
            )


def _evaluate_development_measured_data(
    model: Any,
    contexts: Mapping[str, V8FeedbackSystemContext],
    profile: V8FeedbackTrainingProfile,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: "torch.device",
    classifier_prior: Any = None,
) -> Dict[str, Any]:
    """Evaluate validation rows only; no locked/holdout row is touched."""

    hidden_fraction = float(config["data"]["hidden_operation_fraction"])
    grid_size = int(config["selection"]["temperature_grid_size"])
    result: Dict[str, Any] = {}
    for system_offset, (system_type, context) in enumerate(contexts.items()):
        validation_indices = v7_training._select_evaluation_indices(
            np.flatnonzero(context.arrays.split_mask("validation")),
            maximum_samples=profile.evaluation_max_samples_per_split,
            seed=int(seed + system_offset),
        )
        if validation_indices.size == 0:
            raise ValueError(f"{system_type} has no development validation rows")
        prediction = _collect_v8_measured_predictions(
            model,
            context,
            validation_indices,
            batch_size=profile.real_batch_size,
            hidden_fraction=hidden_fraction,
            seed=int(seed + 701),
            device=device,
            classifier_prior=classifier_prior,
        )
        probability = v7_training._masked_probability(
            prediction["probability"], prediction["mask"]
        )
        counts = prediction["observed"] * prediction["mask"]
        temperature, validation_nll = v7_training.fit_categorical_temperature(
            probability,
            counts,
            grid_size=grid_size,
        )
        result[system_type] = {
            "validation_temperature": float(temperature),
            "validation_temperature_fit_nll": float(validation_nll),
            "splits": {
                "validation_development": v7_training._measured_prediction_metrics(
                    prediction,
                    temperature=float(temperature),
                    seed=int(seed + system_offset),
                )
            },
            "selection_scope": "development_only",
            "locked_final_rows_accessed": 0,
            "operation_mapping_negative_control": prediction[
                "operation_mapping_negative_control"
            ],
        }
    return result


def _collect_v8_measured_predictions(
    model: Any,
    context: V8FeedbackSystemContext,
    indices: np.ndarray,
    *,
    batch_size: int,
    hidden_fraction: float,
    seed: int,
    device: "torch.device",
    classifier_prior: Any = None,
) -> Dict[str, Any]:
    """Collect V8 validation outputs and an operation-pairing control."""

    del classifier_prior  # same-snapshot rho latent is forbidden on real rows.
    observed_rows: List[np.ndarray] = []
    mean_rows: List[np.ndarray] = []
    shuffled_mean_rows: List[np.ndarray] = []
    probability_rows: List[np.ndarray] = []
    dispersion_rows: List[np.ndarray] = []
    total_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []
    weight_rows: List[np.ndarray] = []
    rho_fidelity_rows: List[np.ndarray] = []
    classifier_gate_rows: List[np.ndarray] = []

    def forward(
        batch: Mapping[str, "torch.Tensor"],
        operation_features: "torch.Tensor",
    ) -> Any:
        return model.model_a(
            operation_features,
            batch["target_features"],
            batch["graph_kernel"],
            batch["action"],
            batch["action_mask"],
            system_type=context.arrays.system_type,
            classifier_latent=None,
            classifier_uncertainty=None,
            exposure_s=batch.get("exposure_s"),
            exposure_known_mask=batch.get("exposure_known_mask"),
            command_features=batch.get("command_features"),
            command_mask=batch.get("command_mask"),
            calibration_known_mask=batch.get("calibration_known_mask"),
            calibration_identity_match_mask=batch.get(
                "calibration_identity_match_mask"
            ),
            rho_shared_backbone_gradient_scale=0.0,
        )

    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            selected = indices[start:start + int(batch_size)]
            batch = _v8_measured_batch(
                context,
                selected,
                hidden_fraction=hidden_fraction,
                mask_seed=int(seed),
                device=device,
                classifier_prior=None,
            )
            output = forward(batch, batch["operation_features"])
            shuffled_features = batch["operation_features"].clone()
            operation_count = int(shuffled_features.shape[1])
            for local_index, array_index in enumerate(selected):
                sample_id = context.arrays.sample_ids[int(array_index)]
                digest = hashlib.sha256(
                    f"{int(seed)}:operation_mapping:{sample_id}".encode("utf-8")
                ).digest()
                shift = 1 + int.from_bytes(digest[:8], "big") % (operation_count - 1)
                shuffled_features[local_index, :, :8] = torch.roll(
                    shuffled_features[local_index, :, :8],
                    shifts=shift,
                    dims=0,
                )
            shuffled_output = forward(batch, shuffled_features)
            observed_rows.append(batch["coin_counts"].cpu().numpy())
            mean_rows.append(output["expected_counts"].float().cpu().numpy())
            shuffled_mean_rows.append(
                shuffled_output["expected_counts"].float().cpu().numpy()
            )
            probability_rows.append(output["count_probability"].float().cpu().numpy())
            dispersion_rows.append(output["nb2_dispersion"].float().cpu().numpy())
            total_rows.append(output["expected_total"].float().cpu().numpy())
            mask_rows.append(batch["hidden_mask"].cpu().numpy())
            weight_rows.append(batch["sample_weight"].cpu().numpy())
            classifier_gate_rows.append(
                output["classifier_gate"].float().abs().mean(dim=1).cpu().numpy()
            )
            fidelity = v7_training.torch_density_fidelity(
                output["rho_current"], batch["rho"]
            )
            rho_mask = batch["rho_mask"] > 0.5
            if bool(torch.any(rho_mask)):
                rho_fidelity_rows.append(fidelity[rho_mask].float().cpu().numpy())

    observed = np.concatenate(observed_rows, axis=0)
    mean = np.concatenate(mean_rows, axis=0)
    shuffled_mean = np.concatenate(shuffled_mean_rows, axis=0)
    dispersion = np.concatenate(dispersion_rows, axis=0)
    mask = np.concatenate(mask_rows, axis=0)
    repeat_weight = np.concatenate(weight_rows, axis=0)
    observed_tensor = torch.from_numpy(observed)
    mean_tensor = torch.from_numpy(mean)
    shuffled_mean_tensor = torch.from_numpy(shuffled_mean)
    dispersion_tensor = torch.from_numpy(dispersion)
    mask_tensor = torch.from_numpy(mask)
    repeat_tensor = torch.from_numpy(repeat_weight)
    independent_mask = mask_tensor * repeat_tensor[:, None]
    original_independent = v7_training.torch_nb2_nll(
        observed_tensor, mean_tensor, dispersion_tensor, independent_mask
    )
    shuffled_independent = v7_training.torch_nb2_nll(
        observed_tensor, shuffled_mean_tensor, dispersion_tensor, independent_mask
    )
    likelihood_kind = str(model.config.real_count_likelihood_kind)
    if likelihood_kind == "shared_gamma_total_nb2_plus_conditional_multinomial":
        original_matched = shared_gamma_total_nb2_conditional_multinomial_nll(
            observed_tensor,
            mean_tensor,
            dispersion_tensor,
            mask_tensor,
            repeat_tensor,
        )[0]
        shuffled_matched = shared_gamma_total_nb2_conditional_multinomial_nll(
            observed_tensor,
            shuffled_mean_tensor,
            dispersion_tensor,
            mask_tensor,
            repeat_tensor,
        )[0]
    else:
        original_matched = original_independent
        shuffled_matched = shuffled_independent
    degradation = shuffled_matched - original_matched
    control = {
        "schema": "v8_measured_operation_mapping_negative_control_v1",
        "shuffle_kind": "within_row_dynamic_measurement_block_derangement",
        "dynamic_feature_indices": list(range(8)),
        "registered_projector_and_target_features_kept_fixed": True,
        "same_hidden_mask": True,
        "same_snapshot_rho_classifier_latent_used": False,
        "row_count": int(len(indices)),
        "sample_ids_sha256": _stable_json_hash(
            [context.arrays.sample_ids[int(index)] for index in indices]
        ),
        "hidden_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
        ).hexdigest(),
        "seed": int(seed),
        "likelihood_kind": likelihood_kind,
        "original_matched_nll": float(original_matched),
        "shuffled_matched_nll": float(shuffled_matched),
        "absolute_degradation": float(degradation),
        "relative_degradation": float(
            degradation / original_matched.abs().clamp_min(1e-8)
        ),
        "original_independent_nb2_nll": float(original_independent),
        "shuffled_independent_nb2_nll": float(shuffled_independent),
        "selection_authority": False,
        "interpretation": (
            "positive_degradation_is_required_before_claiming_registered_"
            "operation_mapping_learning"
        ),
    }
    control["control_sha256"] = _stable_json_hash(control)
    return {
        "observed": observed,
        "mean": mean,
        "probability": np.concatenate(probability_rows, axis=0),
        "dispersion": dispersion,
        "expected_total": np.concatenate(total_rows, axis=0),
        "mask": mask,
        "rho_fidelity": (
            np.concatenate(rho_fidelity_rows, axis=0)
            if rho_fidelity_rows else np.empty((0,))
        ),
        "classifier_gate": (
            np.concatenate(classifier_gate_rows, axis=0)
            if classifier_gate_rows else np.empty((0,))
        ),
        "operation_mapping_negative_control": control,
    }


def _expanded_condition(
    value: "torch.Tensor",
    *,
    requested_batch: int,
    base_batch: int,
) -> "torch.Tensor":
    if requested_batch == base_batch:
        return value
    if requested_batch % base_batch != 0:
        raise ValueError("conditioned Model-A batch is not an integer expansion")
    repeats = requested_batch // base_batch
    return value[:, None].expand(
        -1, repeats, *([-1] * (value.ndim - 1))
    ).reshape(requested_batch, *value.shape[1:])


@contextlib.contextmanager
def _condition_v8_model_a(
    model_a: Any,
    tensor: Mapping[str, "torch.Tensor"],
) -> Iterator[None]:
    """Inject causal history, exposure and command context into reused V7 loss calls."""

    original_forward = model_a.forward
    base_batch = int(tensor["action"].shape[0])
    condition_names = (
        "history",
        "history_mask",
        "context_target_features",
        "context_mask",
        "exposure_s",
        "exposure_known_mask",
        "command_features",
        "command_mask",
        "calibration_known_mask",
        "calibration_identity_match_mask",
    )

    def conditioned_forward(
        self: Any,
        operation_features: "torch.Tensor",
        target_features: "torch.Tensor",
        graph_kernel: "torch.Tensor",
        action: "torch.Tensor",
        action_mask: "torch.Tensor",
        **kwargs: Any,
    ) -> Mapping[str, "torch.Tensor"]:
        requested_batch = int(operation_features.shape[0])
        for name in condition_names:
            if name in tensor and name not in kwargs:
                kwargs[name] = _expanded_condition(
                    tensor[name],
                    requested_batch=requested_batch,
                    base_batch=base_batch,
                )
        return original_forward(
            operation_features,
            target_features,
            graph_kernel,
            action,
            action_mask,
            **kwargs,
        )

    model_a.forward = types.MethodType(conditioned_forward, model_a)
    try:
        yield
    finally:
        model_a.forward = original_forward


def _v8_synthetic_tensors(
    batch: Mapping[str, Any],
    *,
    device: "torch.device",
) -> Dict[str, "torch.Tensor"]:
    result = v7_training._synthetic_tensors(batch, device=device)
    command_features = result.get("command_features")
    if command_features is not None:
        if command_features.ndim != 3 or command_features.shape[-1] != 18:
            raise ValueError("V8 synthetic command features must have shape [B,O,18]")
        feature_support = torch.ones(
            18, dtype=command_features.dtype, device=device
        )
        # The present simulator varies RF/phase/power proxies only.  Its three
        # timing slots are neutral placeholders, not authoritative observations.
        feature_support[15:] = 0.0
        result["command_mask"] = torch.ones_like(command_features) * feature_support
        result["synthetic_command_feature_support_mask"] = feature_support
    batch_size = int(result["operation_features"].shape[0])
    result["calibration_known_mask"] = torch.ones(
        batch_size,
        dtype=result["operation_features"].dtype,
        device=device,
    )
    result["calibration_identity_match_mask"] = torch.ones_like(
        result["calibration_known_mask"]
    )
    return result


def _train_v8_synthetic_model_a_stage(
    model: Any,
    contexts: Mapping[str, V8FeedbackSystemContext],
    profile: V8FeedbackTrainingProfile,
    config: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    seed: int,
    device: "torch.device",
    classifier_prior: Any,
) -> Dict[str, Any]:
    """Train Model A on the declared causal V8 training plant family."""

    from .feedback_synthetic import generate_v8_transition_batch

    steps = int(profile.synthetic_model_a_steps)
    if steps <= 0:
        return {"steps": 0, "mean": {}, "by_system": {}, "family_role": "train"}
    family_ids = tuple(str(item) for item in config["_v8_train_families"])
    optimization = config["optimization"]
    optimizer = v7_training._make_optimizer(
        model,
        learning_rate=float(optimization["synthetic_learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    amp_dtype = str(optimization.get("amp_dtype", "bfloat16"))
    scaler = v7_training._make_scaler(
        device, bool(optimization["use_amp"]), amp_dtype
    )
    rank_temperature = float(
        dict(config.get("policy") or {}).get("candidate_rank_temperature", 0.02)
    )
    system_names = tuple(contexts)
    all_logs: List[Dict[str, float]] = []
    system_logs: Dict[str, List[Dict[str, float]]] = {key: [] for key in contexts}
    family_steps: Dict[str, int] = {key: 0 for key in family_ids}
    model.train()
    for step in range(steps):
        system_type = system_names[step % len(system_names)]
        family_id = family_ids[step % len(family_ids)]
        context = contexts[system_type]
        synthetic = generate_v8_transition_batch(
            context.arrays,
            context.profile,
            batch_size=profile.synthetic_batch_size,
            seed=int(seed + step * 104729),
            family_id=family_id,
        )
        tensor = _v8_synthetic_tensors(synthetic, device=device)
        tensor["context_target_features"] = tensor["target_features"][:, None, :].expand(
            -1, tensor["history"].shape[1], -1
        )
        tensor["context_mask"] = tensor["history_mask"]
        tensor.update(v7_training._classifier_prior_features(
            classifier_prior,
            system_type=system_type,
            target_features=tensor["target_features"],
            rho_available_mask=tensor["rho_available_mask"],
            core_rho=tensor["current_rho"],
        ))
        with v7_training._autocast(
            device, bool(optimization["use_amp"]), amp_dtype
        ):
            with _condition_v8_model_a(model.model_a, tensor):
                loss, log, output = v7_training._synthetic_model_a_loss(
                    model,
                    tensor,
                    context,
                    weights,
                    candidate_rank_temperature=rank_temperature,
                )
                physics_weight = float(weights.get("synthetic_jacobian", 0.0))
                if physics_weight > 0.0:
                    zero_output = model.model_a(
                        tensor["operation_features"],
                        tensor["target_features"],
                        torch.from_numpy(context.arrays.graph_kernel).to(device),
                        torch.zeros_like(tensor["action"]),
                        tensor["action_mask"],
                        system_type=system_type,
                        classifier_latent=tensor.get("classifier_latent"),
                        classifier_uncertainty=tensor.get("classifier_uncertainty"),
                    )
                    negative_output = model.model_a(
                        tensor["operation_features"],
                        tensor["target_features"],
                        torch.from_numpy(context.arrays.graph_kernel).to(device),
                        -tensor["action"],
                        tensor["action_mask"],
                        system_type=system_type,
                        classifier_latent=tensor.get("classifier_latent"),
                        classifier_uncertainty=tensor.get("classifier_uncertainty"),
                    )
                    zero_residual = torch.square(
                        zero_output["action_effect"]
                    ).mean()
                    odd_residual = torch.square(
                        output["action_effect"]
                        + negative_output["action_effect"]
                        - 2.0 * zero_output["action_effect"]
                    ).mean()
                    physics_loss = zero_residual + odd_residual
                    loss = loss + physics_weight * physics_loss
                    log["physics_zero_action_residual"] = float(
                        zero_residual.detach()
                    )
                    log["physics_action_odd_residual"] = float(
                        odd_residual.detach()
                    )
                    log["loss"] = float(loss.detach())
        v7_training._optimization_step(
            loss=loss,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=float(optimization["gradient_clip_norm"]),
        )
        family_steps[family_id] += 1
        all_logs.append(log)
        system_logs[system_type].append(log)
    return {
        **v7_training._stage_log_summary(all_logs),
        "by_system": {
            system_type: v7_training._average_logs(rows)
            for system_type, rows in system_logs.items()
        },
        "family_role": "train",
        "family_steps": family_steps,
        "causal_history_bound": True,
        "exposure_mask_explicit": True,
        "command_mask_explicit": True,
        "synthetic_command_feature_support_mask": [1] * 15 + [0] * 3,
        "synthetic_timing_features_authoritative": False,
    }


def _balanced_state_pair_schedule(
    pairs: Sequence[Any],
    *,
    seed: int,
    maximum_pair_rows: int,
) -> List[Any]:
    """Seeded tier/anchor round-robin with every anchor visited before reuse."""

    rng = np.random.default_rng(int(seed))
    queues: Dict[str, Dict[str, List[Any]]] = {}
    for pair in pairs:
        queues.setdefault(pair.source_sample_id, {}).setdefault(
            pair.calibration_transport_tier, []
        ).append(pair)
    anchor_ids = sorted(queues)
    rng.shuffle(anchor_ids)
    for tier_queues in queues.values():
        for values in tier_queues.values():
            rng.shuffle(values)
    tier_names = ("strict_calibration", "cross_calibration")
    result: List[Any] = []
    tier_counts = {name: 0 for name in tier_names}
    for anchor_id in anchor_ids:
        available = [name for name in tier_names if queues[anchor_id].get(name)]
        if not available or len(result) >= maximum_pair_rows:
            continue
        selected_tier = min(available, key=lambda name: (tier_counts[name], name))
        result.append(queues[anchor_id][selected_tier].pop())
        tier_counts[selected_tier] += 1
    round_index = 0
    while len(result) < maximum_pair_rows:
        progress = False
        tier_order = tier_names[round_index % len(tier_names):] + tier_names[:round_index % len(tier_names)]
        for tier_name in tier_order:
            for anchor_id in anchor_ids:
                values = queues[anchor_id].get(tier_name, [])
                if not values:
                    continue
                result.append(values.pop())
                tier_counts[tier_name] += 1
                progress = True
                if len(result) >= maximum_pair_rows:
                    break
            if len(result) >= maximum_pair_rows:
                break
        if not progress:
            break
        round_index += 1
    return result


def _exact_state_pair_derangement(
    pairs: Sequence[Any],
    *,
    shuffle_kind: str,
    seed: int,
    maximum_attempts: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Find a deterministic perfect donor matching with strict source exclusions."""

    if not pairs:
        return {}, {"status": "empty_stratum", "attempts": 0}
    stratum_id = str(pairs[0].comparability_stratum_id)
    if any(pair.comparability_stratum_id != stratum_id for pair in pairs):
        raise ValueError("exact derangement input crossed comparability strata")
    if shuffle_kind not in {"state_endpoint", "target_action"}:
        raise ValueError(f"unsupported exact state-pair shuffle: {shuffle_kind}")
    if int(maximum_attempts) <= 0:
        raise ValueError("maximum_attempts must be positive")

    pair_count = len(pairs)
    rng = np.random.default_rng(int(seed))
    recipient_order = list(range(pair_count))
    rng.shuffle(recipient_order)
    donor_tie_break = rng.permutation(pair_count).tolist()
    donor_rank = {
        donor_index: rank for rank, donor_index in enumerate(donor_tie_break)
    }
    adjacency: List[List[int]] = []
    for recipient in pairs:
        candidates = []
        for donor_index, donor in enumerate(pairs):
            if donor.target_sample_id in {
                recipient.source_sample_id,
                recipient.target_sample_id,
            }:
                continue
            if donor.target_measurement_group_id in {
                recipient.source_measurement_group_id,
                recipient.target_measurement_group_id,
            }:
                continue
            candidates.append(donor_index)
        candidates.sort(key=lambda value: (donor_rank[value], value))
        adjacency.append(candidates)

    donor_to_recipient = [-1] * pair_count

    def augment(recipient_index: int, visited_donors: set[int]) -> bool:
        for donor_index in adjacency[recipient_index]:
            if donor_index in visited_donors:
                continue
            visited_donors.add(donor_index)
            previous_recipient = donor_to_recipient[donor_index]
            if previous_recipient < 0 or augment(
                previous_recipient, visited_donors
            ):
                donor_to_recipient[donor_index] = recipient_index
                return True
        return False

    matched_recipient_count = 0
    for recipient_index in recipient_order:
        if augment(recipient_index, set()):
            matched_recipient_count += 1
    if matched_recipient_count == pair_count:
        recipient_to_donor = [-1] * pair_count
        for donor_index, recipient_index in enumerate(donor_to_recipient):
            recipient_to_donor[recipient_index] = donor_index
        donors = {
            pairs[recipient_index].pair_id: pairs[donor_index]
            for recipient_index, donor_index in enumerate(recipient_to_donor)
        }
        plan_payload = {
            "algorithm": "seeded_deterministic_bipartite_augmenting_path_v1",
            "seed": int(seed),
            "shuffle_kind": shuffle_kind,
            "stratum_id": stratum_id,
            "pair_ids": [pair.pair_id for pair in pairs],
            "recipient_to_donor": recipient_to_donor,
        }
        return donors, {
            "status": "accepted_exact_whole_stratum",
            "attempts": 1,
            "shuffle_kind": shuffle_kind,
            "algorithm": plan_payload["algorithm"],
            "plan_hash": _stable_json_hash(plan_payload),
            "item_count": pair_count,
            "matched_recipient_count": matched_recipient_count,
            "minimum_candidate_donor_count": min(map(len, adjacency)),
            "maximum_candidate_donor_count": max(map(len, adjacency)),
            "source_endpoint_exclusion_verified": True,
            "source_group_exclusion_verified": True,
            "recipient_target_exclusion_verified": True,
            "exact_item_marginal_preserved": True,
            "each_donor_used_exactly_once": True,
        }
    return {}, {
        "status": "fail_closed_no_exact_source_excluding_perfect_matching",
        "attempts": 1,
        "shuffle_kind": shuffle_kind,
        "algorithm": "seeded_deterministic_bipartite_augmenting_path_v1",
        "item_count": pair_count,
        "matched_recipient_count": matched_recipient_count,
        "unmatched_recipient_count": pair_count - matched_recipient_count,
        "minimum_candidate_donor_count": min(map(len, adjacency)),
        "maximum_candidate_donor_count": max(map(len, adjacency)),
        "reason": "bipartite_graph_has_no_perfect_source_excluding_matching",
        "no_rows_individually_filtered": True,
    }


def _weak_transport_optimizer(
    model: Any,
    contract: Mapping[str, Any],
    optimization: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any], List["torch.nn.Parameter"], List["torch.nn.Parameter"]]:
    parameter_groups = model.weak_transport_parameter_groups()
    learning_rate = float(optimization["real_learning_rate"])
    scale_by_group = {
        "shared_measurement_static_encoder": float(
            contract["weak_backbone_gradient_scale"]
        ),
        "independent_weak_response_operator": float(
            contract["weak_operator_learning_rate_scale"]
        ),
        "shared_coordinate_geometry_basis": 1.0,
    }
    optimizer_groups: List[Dict[str, Any]] = []
    optimized: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    audit: Dict[str, Any] = {}
    for name in (
        "shared_measurement_static_encoder",
        "independent_weak_response_operator",
        "shared_coordinate_geometry_basis",
    ):
        parameters = [
            parameter
            for parameter in parameter_groups[name]
            if parameter.requires_grad and id(parameter) not in seen
        ]
        for parameter in parameters:
            seen.add(id(parameter))
        scale = scale_by_group[name]
        audit[name] = {
            "parameter_tensor_count": len(parameters),
            "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "learning_rate_scale": scale,
            "learning_rate": learning_rate * scale,
        }
        if parameters and scale > 0.0:
            optimizer_groups.append({"params": parameters, "lr": learning_rate * scale})
            optimized.extend(parameters)
    if not optimizer_groups:
        raise ValueError("weak transport optimizer has no authorized parameters")
    excluded = [parameter for parameter in model.parameters() if id(parameter) not in seen]
    audit["excluded_real_count_rho_action_jacobian_policy"] = {
        "parameter_tensor_count": len(excluded),
        "parameter_count": int(sum(parameter.numel() for parameter in excluded)),
        "optimizer_membership": False,
    }
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(optimization["weight_decay"]),
        betas=(0.9, 0.98),
    )
    return optimizer, audit, optimized, excluded


def _weak_transport_optimization_step(
    *,
    loss: "torch.Tensor",
    model: Any,
    optimizer: Any,
    scaler: Any,
    optimized_parameters: Sequence["torch.nn.Parameter"],
    excluded_parameters: Sequence["torch.nn.Parameter"],
    parameter_audit: Mapping[str, Any],
    gradient_clip_norm: float,
) -> Dict[str, float]:
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("weak transport loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    group_norms: Dict[str, float] = {}
    for group_name, group in zip(
        (
            "shared_measurement_static_encoder",
            "independent_weak_response_operator",
        ),
        optimizer.param_groups,
    ):
        squared = torch.zeros((), dtype=torch.float64, device=loss.device)
        for parameter in group["params"]:
            if parameter.grad is not None:
                squared = squared + parameter.grad.detach().double().square().sum()
        group_norms[f"{group_name}_gradient_norm"] = float(torch.sqrt(squared).cpu())
    nonzero_excluded = 0
    for parameter in excluded_parameters:
        if parameter.grad is not None and bool(torch.any(parameter.grad != 0.0)):
            nonzero_excluded += 1
    if nonzero_excluded:
        raise RuntimeError("weak loss reached excluded real-action/count/policy parameters")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        list(optimized_parameters),
        float(gradient_clip_norm),
        error_if_nonfinite=True,
    )
    scaler.step(optimizer)
    scaler.update()
    return {
        **group_norms,
        "optimized_gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "excluded_parameter_nonzero_gradient_count": float(nonzero_excluded),
        "authorized_parameter_count": float(sum(
            int(value.get("parameter_count", 0))
            for key, value in parameter_audit.items()
            if key != "excluded_real_count_rho_action_jacobian_policy"
        )),
    }


def _train_state_pair_transport_stage(
    model: Any,
    contexts: Mapping[str, V8FeedbackSystemContext],
    config: Mapping[str, Any],
    *,
    seed: int,
    device: "torch.device",
    classifier_prior: Any,
) -> Dict[str, Any]:
    """Train/evaluate the isolated measurement-only weak proxy operator."""

    from .causal_negative_control import (
        CausalNegativeControlConfig,
        evaluate_paired_vs_shuffled_loss,
    )

    del classifier_prior  # rho/classifier priors are forbidden in this channel.
    contract = dict(config["_v8_state_pair_contract"])
    weight = float(contract["loss_weight"])
    supervision_mode = str(
        contract.get("supervision_mode", "relational_state_pair_transport")
    )
    if supervision_mode not in {
        "relational_state_pair_transport",
        "endpoint_identity_compute_matched_control",
    }:
        raise ValueError(f"unknown weak supervision mode: {supervision_mode}")
    relational_supervision_enabled = (
        supervision_mode == "relational_state_pair_transport"
    )
    endpoint_static_term_weight = float(contract["endpoint_static_term_weight"])
    relational_transport_term_weight = float(
        contract["relational_transport_term_weight"]
    )
    pair_builder_contract = dict(contract["pair_builder"])
    maximum_batches_per_system = int(
        contract["active_maximum_batches_per_system"]
    )
    batch_size = int(contract["batch_size"])
    if weight <= 0.0 or maximum_batches_per_system <= 0:
        return {
            "status": "disabled",
            "steps": 0,
            "evidence_scope": "weak",
            "real_action_supervision_mask_sum": 0,
        }
    optimization = dict(config["optimization"])
    optimizer, parameter_audit, optimized_parameters, excluded_parameters = (
        _weak_transport_optimizer(model, contract, optimization)
    )
    scaler = v7_training._make_scaler(
        device,
        bool(optimization["use_amp"]),
        str(optimization.get("amp_dtype", "bfloat16")),
    )
    model.zero_grad(set_to_none=True)
    negative_config = CausalNegativeControlConfig(
        bootstrap_replicates=int(
            config["_v8_causal_negative_control_contract"].get(
                "bootstrap_replicates", 2000
            )
        ),
        confidence_level=float(
            config["_v8_causal_negative_control_contract"].get(
                "confidence_level", 0.95
            )
        ),
        minimum_groups=int(
            config["_v8_causal_negative_control_contract"].get("minimum_groups", 8)
        ),
        minimum_relative_degradation=float(
            config["_v8_causal_negative_control_contract"].get(
                "minimum_relative_degradation", 0.05
            )
        ),
        seed=int(config["_v8_causal_negative_control_contract"].get("seed", seed)),
    )
    maximum_attempts = int(contract["derangement_maximum_seed_attempts_per_stratum"])
    by_system: Dict[str, Dict[str, Any]] = {}
    final_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    gradient_logs: List[Dict[str, float]] = []
    total_steps = 0

    def effective_transport_tier(pair: Any) -> Tuple[str, Optional[str]]:
        if pair.calibration_transport_tier != "strict_calibration":
            return str(pair.calibration_transport_tier), None
        calibration_known = bool(
            pair.source_calibration_hash
            and pair.target_calibration_hash
            and pair.source_calibration_hash == pair.target_calibration_hash
        )
        exposure_known = pair.exposure_ratio is not None
        if not calibration_known or not exposure_known:
            reasons = []
            if not calibration_known:
                reasons.append("unknown_or_non_authoritative_calibration")
            if not exposure_known:
                reasons.append("unknown_exposure")
            return "cross_calibration", "+".join(reasons)
        return "strict_calibration", None

    def effective_pair_weight(pair: Any) -> float:
        effective_tier, reason = effective_transport_tier(pair)
        pair_weight = float(pair.pair_weight)
        if effective_tier != "cross_calibration" or reason is None:
            return pair_weight
        calibration_known = bool(
            pair.source_calibration_hash and pair.target_calibration_hash
        )
        original_calibration_weight = (
            1.0
            if calibration_known
            else float(pair_builder_contract["unknown_calibration_weight"])
        )
        effective_calibration_weight = float(
            pair_builder_contract[
                "cross_calibration_known_weight"
                if calibration_known
                else "cross_calibration_unknown_weight"
            ]
        )
        return max(
            float(pair_builder_contract["minimum_pair_weight"]),
            pair_weight
            * effective_calibration_weight
            / max(original_calibration_weight, 1e-12),
        )

    def pair_batch_outputs(
        selected_pairs: Sequence[Any],
        *,
        context: V8FeedbackSystemContext,
        endpoint_donors: Mapping[str, Any],
        proxy_donors: Mapping[str, Any],
        batch_seed: int,
        include_shuffle_controls: bool,
    ) -> Tuple["torch.Tensor", List[Dict[str, Any]]]:
        index_by_sample = {
            sample_id: index
            for index, sample_id in enumerate(context.arrays.sample_ids)
        }
        row_losses: List[torch.Tensor] = []
        row_weights: List[torch.Tensor] = []
        row_payloads: List[Dict[str, Any]] = []
        for tier in ("strict_calibration", "cross_calibration"):
            tier_pairs = [
                pair for pair in selected_pairs
                if effective_transport_tier(pair)[0] == tier
            ]
            if not tier_pairs:
                continue
            source_indices = [index_by_sample[pair.source_sample_id] for pair in tier_pairs]
            target_indices = [index_by_sample[pair.target_sample_id] for pair in tier_pairs]
            batch_kwargs = {
                "hidden_fraction": 0.0,
                "mask_seed": int(batch_seed),
                "device": device,
                "classifier_prior": None,
            }
            source_batch = _v8_measured_batch(context, source_indices, **batch_kwargs)
            target_batch = _v8_measured_batch(context, target_indices, **batch_kwargs)
            proxy = torch.as_tensor(
                [pair.target_transport_proxy for pair in tier_pairs],
                dtype=source_batch["operation_features"].dtype,
                device=device,
            )
            proxy_mask = torch.as_tensor(
                [pair.target_transport_proxy_mask for pair in tier_pairs],
                dtype=source_batch["operation_features"].dtype,
                device=device,
            )
            zero_mask = torch.zeros_like(proxy_mask)
            zero_proxy = torch.zeros_like(proxy)
            real_zero = torch.zeros(len(tier_pairs), dtype=proxy.dtype, device=device)
            calibration_source_known = torch.as_tensor(
                [pair.source_calibration_hash is not None for pair in tier_pairs],
                dtype=proxy.dtype,
                device=device,
            )
            calibration_target_known = torch.as_tensor(
                [pair.target_calibration_hash is not None for pair in tier_pairs],
                dtype=proxy.dtype,
                device=device,
            )
            calibration_identity_match = torch.as_tensor(
                [
                    pair.source_calibration_hash is not None
                    and pair.target_calibration_hash is not None
                    and pair.source_calibration_hash == pair.target_calibration_hash
                    for pair in tier_pairs
                ],
                dtype=proxy.dtype,
                device=device,
            )
            exposure_known = torch.as_tensor(
                [pair.exposure_ratio is not None for pair in tier_pairs],
                dtype=proxy.dtype,
                device=device,
            )
            weak_provenance_masks = {
                "source_calibration_known_mask": calibration_source_known,
                "target_calibration_known_mask": calibration_target_known,
                "calibration_identity_match_mask": calibration_identity_match,
                "source_exposure_known_mask": exposure_known,
                "target_exposure_known_mask": exposure_known,
                "exposure_compatible_mask": exposure_known,
            }
            correct_output = model.predict_weak_target_transport(
                source_batch["operation_features"],
                proxy,
                proxy_mask,
                system_type=context.arrays.system_type,
                calibration_transport_tier=tier,
                **weak_provenance_masks,
                real_action_supervision_mask=real_zero,
                real_count_jacobian_supervision_mask=real_zero,
            )
            target_output = model.predict_weak_target_transport(
                target_batch["operation_features"],
                zero_proxy,
                zero_mask,
                system_type=context.arrays.system_type,
                calibration_transport_tier=tier,
                **weak_provenance_masks,
            )
            target_source_embedding = target_output["source_transport_embedding"]
            target_embedding = target_source_embedding.detach()
            correct_per_row = 1.0 - functional.cosine_similarity(
                correct_output["predicted_target_transport_embedding"],
                target_embedding,
                dim=-1,
            )
            proxy_per_row: Optional[torch.Tensor] = None
            endpoint_per_row: Optional[torch.Tensor] = None
            if include_shuffle_controls:
                endpoint_indices = [
                    index_by_sample[
                        endpoint_donors[pair.pair_id].target_sample_id
                    ]
                    for pair in tier_pairs
                ]
                endpoint_batch = _v8_measured_batch(
                    context, endpoint_indices, **batch_kwargs
                )
                shuffled_proxy = torch.as_tensor(
                    [
                        proxy_donors[pair.pair_id].target_transport_proxy
                        for pair in tier_pairs
                    ],
                    dtype=proxy.dtype,
                    device=device,
                )
                shuffled_proxy_mask = torch.as_tensor(
                    [
                        proxy_donors[pair.pair_id].target_transport_proxy_mask
                        for pair in tier_pairs
                    ],
                    dtype=proxy.dtype,
                    device=device,
                )
                proxy_output = model.predict_weak_target_transport(
                    source_batch["operation_features"],
                    shuffled_proxy,
                    shuffled_proxy_mask,
                    system_type=context.arrays.system_type,
                    calibration_transport_tier=tier,
                    **weak_provenance_masks,
                    real_action_supervision_mask=real_zero,
                    real_count_jacobian_supervision_mask=real_zero,
                )
                endpoint_output = model.predict_weak_target_transport(
                    endpoint_batch["operation_features"],
                    zero_proxy,
                    zero_mask,
                    system_type=context.arrays.system_type,
                    calibration_transport_tier=tier,
                    **weak_provenance_masks,
                )
                endpoint_embedding = endpoint_output[
                    "source_transport_embedding"
                ].detach()
                proxy_per_row = 1.0 - functional.cosine_similarity(
                    proxy_output["predicted_target_transport_embedding"],
                    target_embedding,
                    dim=-1,
                )
                endpoint_per_row = 1.0 - functional.cosine_similarity(
                    correct_output["predicted_target_transport_embedding"],
                    endpoint_embedding,
                    dim=-1,
                )
            supervision_mask = (
                correct_output["weak_transport_supervision_mask"]
                if relational_supervision_enabled
                else torch.ones_like(correct_output["weak_transport_supervision_mask"])
            )
            pair_weight = torch.as_tensor(
                [effective_pair_weight(pair) for pair in tier_pairs],
                dtype=correct_per_row.dtype,
                device=device,
            ) * supervision_mask
            if not torch.all(
                correct_output["real_action_supervision_mask_passthrough"] == 0.0
            ) or not torch.all(
                correct_output["real_count_jacobian_supervision_mask_passthrough"] == 0.0
            ):
                raise RuntimeError("weak transport received real action/Jacobian supervision")
            # Both paired arms receive the identical, permutation-invariant
            # endpoint anti-collapse objective.  The registered weak-on arm
            # differs only by the ordered proxy-to-target relational term.
            endpoint_embeddings = torch.cat(
                [
                    correct_output["source_transport_embedding"],
                    target_source_embedding,
                ],
                dim=0,
            )
            endpoint_center = endpoint_embeddings.mean(dim=0)
            endpoint_variance = endpoint_embeddings.var(dim=0, unbiased=False)
            endpoint_static_loss = (
                endpoint_center.square().mean()
                + torch.relu(1.0 - torch.sqrt(endpoint_variance + 1e-4))
                .square()
                .mean()
            )
            training_per_row = (
                endpoint_static_term_weight * endpoint_static_loss
            ).expand_as(correct_per_row)
            if relational_supervision_enabled:
                training_per_row = training_per_row + (
                    relational_transport_term_weight * correct_per_row
                )
            row_losses.append(training_per_row)
            row_weights.append(pair_weight)
            support = correct_output[
                "weak_transport_measurement_feature_support_mask"
            ][0, 0].detach().float().cpu().numpy().astype(int).tolist()
            if include_shuffle_controls:
                assert proxy_per_row is not None and endpoint_per_row is not None
                for row_index, pair in enumerate(tier_pairs):
                    effective_tier, reclassification_reason = effective_transport_tier(
                        pair
                    )
                    row_payloads.append({
                        "pair": pair,
                        "system_type": context.arrays.system_type,
                        "correct_loss": correct_per_row[row_index],
                        "proxy_shuffled_loss": proxy_per_row[row_index],
                        "endpoint_shuffled_loss": endpoint_per_row[row_index],
                        "pair_weight": pair_weight[row_index],
                        "feature_support_mask": support,
                        "effective_calibration_transport_tier": effective_tier,
                        "tier_reclassification_reason": reclassification_reason,
                    })
        combined_loss = weighted_real_auxiliary_mean(
            torch.cat(row_losses), torch.cat(row_weights)
        )
        return combined_loss, row_payloads

    model.train()
    for system_offset, (system_type, context) in enumerate(contexts.items()):
        pair_set = context.state_pair_transport_set
        if pair_set is None:
            by_system[system_type] = {"pair_count": 0, "steps": 0}
            continue
        pair_set.validate()
        by_stratum: Dict[str, List[Any]] = {}
        for pair in pair_set.pairs:
            by_stratum.setdefault(pair.comparability_stratum_id, []).append(pair)
        endpoint_donors: Dict[str, Any] = {}
        proxy_donors: Dict[str, Any] = {}
        derangement_audit: Dict[str, Any] = {}
        for stratum_offset, (stratum_id, stratum_pairs) in enumerate(
            sorted(by_stratum.items())
        ):
            endpoint_map, endpoint_audit = _exact_state_pair_derangement(
                stratum_pairs,
                shuffle_kind="state_endpoint",
                seed=int(seed + system_offset * 100003 + stratum_offset * 2003),
                maximum_attempts=maximum_attempts,
            )
            proxy_map, proxy_audit = _exact_state_pair_derangement(
                stratum_pairs,
                shuffle_kind="target_action",
                seed=int(seed + system_offset * 100003 + stratum_offset * 4001),
                maximum_attempts=maximum_attempts,
            )
            accepted = bool(endpoint_map and proxy_map)
            derangement_audit[stratum_id] = {
                "status": "accepted" if accepted else "fail_closed",
                "endpoint": endpoint_audit,
                "proxy": proxy_audit,
                "whole_stratum_row_count": len(stratum_pairs),
                "no_individual_row_filtering": True,
            }
            if accepted:
                endpoint_donors.update(endpoint_map)
                proxy_donors.update(proxy_map)
        derangement_eligible_pool = [
            pair for pair in pair_set.pairs
            if pair.pair_id in endpoint_donors and pair.pair_id in proxy_donors
        ]
        scheduled_pairs = _balanced_state_pair_schedule(
            pair_set.pairs,
            seed=int(seed + system_offset * 100003),
            maximum_pair_rows=maximum_batches_per_system * batch_size,
        )
        diagnostic_pairs = _balanced_state_pair_schedule(
            derangement_eligible_pool,
            seed=int(seed + 9000011 + system_offset * 100003),
            maximum_pair_rows=maximum_batches_per_system * batch_size,
        )
        system_steps = 0
        for start in range(0, len(scheduled_pairs), batch_size):
            selected_pairs = scheduled_pairs[start:start + batch_size]
            transport_loss, _ = pair_batch_outputs(
                selected_pairs,
                context=context,
                endpoint_donors=endpoint_donors,
                proxy_donors=proxy_donors,
                batch_seed=int(seed + system_offset * 100003 + start),
                include_shuffle_controls=False,
            )
            gradient_logs.append(_weak_transport_optimization_step(
                loss=weight * transport_loss,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                optimized_parameters=optimized_parameters,
                excluded_parameters=excluded_parameters,
                parameter_audit=parameter_audit,
                gradient_clip_norm=float(optimization["gradient_clip_norm"]),
            ))
            total_steps += 1
            system_steps += 1

        model.eval()
        with torch.no_grad():
            for start in range(0, len(diagnostic_pairs), batch_size):
                _, payloads = pair_batch_outputs(
                    diagnostic_pairs[start:start + batch_size],
                    context=context,
                    endpoint_donors=endpoint_donors,
                    proxy_donors=proxy_donors,
                    batch_seed=int(seed + system_offset * 100003 + start),
                    include_shuffle_controls=True,
                )
                final_rows.extend(payloads)
        validation_pair_set = context.state_pair_validation_set
        validation_derangement_audit: Dict[str, Any] = {}
        validation_eligible_pairs: List[Any] = []
        if validation_pair_set is not None:
            validation_pair_set.validate()
            validation_by_stratum: Dict[str, List[Any]] = {}
            for pair in validation_pair_set.pairs:
                validation_by_stratum.setdefault(
                    pair.comparability_stratum_id, []
                ).append(pair)
            validation_stratum_ids = sorted(validation_by_stratum)
            validation_rng = np.random.default_rng(
                int(seed + 50000017 + system_offset * 100003)
            )
            validation_rng.shuffle(validation_stratum_ids)
            validation_maximum_rows = int(
                contract["active_validation_maximum_pair_rows_per_system"]
            )
            selected_validation_strata: List[str] = []
            selected_validation_row_count = 0
            for stratum_id in validation_stratum_ids:
                if (
                    validation_maximum_rows > 0
                    and selected_validation_row_count >= validation_maximum_rows
                ):
                    break
                selected_validation_strata.append(stratum_id)
                selected_validation_row_count += len(
                    validation_by_stratum[stratum_id]
                )
            validation_endpoint_donors: Dict[str, Any] = {}
            validation_proxy_donors: Dict[str, Any] = {}
            for stratum_offset, stratum_id in enumerate(selected_validation_strata):
                stratum_pairs = validation_by_stratum[stratum_id]
                endpoint_map, endpoint_audit = _exact_state_pair_derangement(
                    stratum_pairs,
                    shuffle_kind="state_endpoint",
                    seed=int(
                        seed + 51000019 + system_offset * 100003
                        + stratum_offset * 2003
                    ),
                    maximum_attempts=maximum_attempts,
                )
                proxy_map, proxy_audit = _exact_state_pair_derangement(
                    stratum_pairs,
                    shuffle_kind="target_action",
                    seed=int(
                        seed + 52000031 + system_offset * 100003
                        + stratum_offset * 4001
                    ),
                    maximum_attempts=maximum_attempts,
                )
                accepted = bool(endpoint_map and proxy_map)
                validation_derangement_audit[stratum_id] = {
                    "status": "accepted" if accepted else "fail_closed",
                    "endpoint": endpoint_audit,
                    "proxy": proxy_audit,
                    "whole_stratum_row_count": len(stratum_pairs),
                    "exact_validation_marginal": accepted,
                    "no_individual_row_filtering": True,
                }
                if accepted:
                    validation_endpoint_donors.update(endpoint_map)
                    validation_proxy_donors.update(proxy_map)
                    validation_eligible_pairs.extend(stratum_pairs)
            with torch.no_grad():
                for start in range(0, len(validation_eligible_pairs), batch_size):
                    _, payloads = pair_batch_outputs(
                        validation_eligible_pairs[start:start + batch_size],
                        context=context,
                        endpoint_donors=validation_endpoint_donors,
                        proxy_donors=validation_proxy_donors,
                        batch_seed=int(
                            seed + 53000033 + system_offset * 100003 + start
                        ),
                        include_shuffle_controls=True,
                    )
                    validation_rows.extend(payloads)
        model.train()
        trained_sources = {pair.source_sample_id for pair in scheduled_pairs}
        trained_targets = {pair.target_sample_id for pair in scheduled_pairs}
        declared_tier_counts = {
            tier: sum(
                pair.calibration_transport_tier == tier for pair in scheduled_pairs
            )
            for tier in ("strict_calibration", "cross_calibration")
        }
        effective_tier_counts = {
            tier: sum(
                effective_transport_tier(pair)[0] == tier
                for pair in scheduled_pairs
            )
            for tier in ("strict_calibration", "cross_calibration")
        }
        tier_reclassification_counts: Dict[str, int] = {}
        for pair in scheduled_pairs:
            _, reason = effective_transport_tier(pair)
            if reason is not None:
                tier_reclassification_counts[reason] = (
                    tier_reclassification_counts.get(reason, 0) + 1
                )
        by_system[system_type] = {
            "source_sample_count": int(pair_set.source_sample_count),
            "pair_count": len(pair_set.pairs),
            "derangement_eligible_pool_pair_count": len(derangement_eligible_pool),
            "scheduled_pair_count": len(scheduled_pairs),
            "derangement_eligible_pair_count": len(derangement_eligible_pool),
            "diagnostic_scheduled_pair_count": len(diagnostic_pairs),
            "unique_source_endpoint_count": len(trained_sources),
            "unique_target_endpoint_count": len(trained_targets),
            "unique_source_coverage_fraction": (
                len(trained_sources) / max(1, int(pair_set.source_sample_count))
            ),
            "declared_tier_pair_count": declared_tier_counts,
            "effective_tier_pair_count": effective_tier_counts,
            "strict_to_scale_free_reclassification_count": int(sum(
                tier_reclassification_counts.values()
            )),
            "tier_reclassification_reason_count": tier_reclassification_counts,
            "steps": system_steps,
            "supervision_mode": supervision_mode,
            "compute_matched": bool(contract.get("compute_matched", False)),
            "scheduled_endpoint_ids_sha256": _stable_json_hash([
                [pair.source_sample_id, pair.target_sample_id]
                for pair in scheduled_pairs
            ]),
            "scheduled_pair_ids_sha256": _stable_json_hash([
                pair.pair_id for pair in scheduled_pairs
            ]),
            "maximum_batches_per_system": maximum_batches_per_system,
            "pair_set_sha256": _stable_json_hash(asdict(pair_set)),
            "pair_set_audit": asdict(pair_set.audit),
            "derangement_strata": derangement_audit,
            "validation": {
                "pair_set_sha256": (
                    _stable_json_hash(asdict(validation_pair_set))
                    if validation_pair_set is not None else None
                ),
                "pair_count": (
                    len(validation_pair_set.pairs)
                    if validation_pair_set is not None else 0
                ),
                "evaluated_exact_pair_count": len(validation_eligible_pairs),
                "unique_source_count": len({
                    pair.source_sample_id for pair in validation_eligible_pairs
                }),
                "unique_target_count": len({
                    pair.target_sample_id for pair in validation_eligible_pairs
                }),
                "maximum_pair_rows_budget": int(
                    contract["active_validation_maximum_pair_rows_per_system"]
                ),
                "derangement_strata": validation_derangement_audit,
                "locked_test_accessed": False,
                "gradient_used": False,
            },
        }

    def build_weak_controls(
        rows: Sequence[Mapping[str, Any]],
        *,
        channel: str,
        allowed_for_model_selection: bool,
        exact_evaluation_marginal: bool,
    ) -> Dict[str, Any]:
        if not rows:
            return {
                "status": "fail_closed_no_exact_derangement_rows",
                "channel": channel,
                "allowed_for_model_selection": allowed_for_model_selection,
                "passed": False,
            }
        paired_losses = [float(row["correct_loss"].cpu()) for row in rows]
        weights = [float(row["pair_weight"].cpu()) for row in rows]
        groups = [row["pair"].transport_group_id for row in rows]
        controls: Dict[str, Any] = {}
        for name, loss_key, shuffle_kind in (
            ("proxy_target_action", "proxy_shuffled_loss", "target_action"),
            ("state_endpoint", "endpoint_shuffled_loss", "state_endpoint"),
        ):
            result = evaluate_paired_vs_shuffled_loss(
                paired_losses,
                [float(row[loss_key].cpu()) for row in rows],
                groups,
                evidence_source="state_pair_transport_weak",
                shuffle_kind=shuffle_kind,
                sample_weights=weights,
                config=negative_config,
            )
            payload = asdict(result)
            payload["result_hash"] = result.result_hash
            payload["same_frozen_checkpoint_evaluation"] = True
            payload["gradient_used"] = False
            payload["channel"] = channel
            payload["allowed_for_model_selection"] = allowed_for_model_selection
            payload["exact_evaluation_marginal"] = exact_evaluation_marginal
            controls[name] = payload
        return {
            "status": "evaluated",
            "channel": channel,
            "allowed_for_model_selection": allowed_for_model_selection,
            "row_count": len(rows),
            "passed": bool(all(
                control["passed_within_evidence_scope"]
                for control in controls.values()
            )),
            "counts_as_real_causality_evidence": False,
            "controls": controls,
        }

    train_fit_diagnostics = build_weak_controls(
        final_rows,
        channel="train_fit_diagnostic_only",
        allowed_for_model_selection=False,
        exact_evaluation_marginal=False,
    )
    validation_controls_by_system = {
        system_type: build_weak_controls(
            [row for row in validation_rows if row["system_type"] == system_type],
            channel="development_validation_state_pair_weak",
            allowed_for_model_selection=True,
            exact_evaluation_marginal=True,
        )
        for system_type in contexts
    }
    weak_validation_gate_passed = bool(
        relational_supervision_enabled
        and total_steps > 0
        and
        validation_controls_by_system
        and all(
            value.get("passed", False)
            for value in validation_controls_by_system.values()
        )
    )
    tier_metrics: Dict[str, Any] = {}
    for tier in ("strict_calibration", "cross_calibration"):
        selected_rows = [
            row for row in final_rows
            if row["effective_calibration_transport_tier"] == tier
        ]
        tier_metrics[tier] = {
            "pair_count": len(selected_rows),
            "unique_source_count": len({
                row["pair"].source_sample_id for row in selected_rows
            }),
            "unique_target_count": len({
                row["pair"].target_sample_id for row in selected_rows
            }),
            "mean_correct_loss": float(np.mean([
                float(row["correct_loss"].cpu()) for row in selected_rows
            ])) if selected_rows else None,
            "feature_support_mask": (
                selected_rows[0]["feature_support_mask"] if selected_rows else None
            ),
            "declared_strict_reclassified_count": sum(
                row["pair"].calibration_transport_tier == "strict_calibration"
                and row["effective_calibration_transport_tier"]
                == "cross_calibration"
                for row in selected_rows
            ),
        }
    return {
        "status": "trained" if total_steps else "no_exact_derangement_available",
        "steps": total_steps,
        "loss_weight": weight,
        "supervision_mode": supervision_mode,
        "relational_state_pair_supervision_enabled": relational_supervision_enabled,
        "endpoint_static_term_weight": endpoint_static_term_weight,
        "relational_transport_term_weight": relational_transport_term_weight,
        "endpoint_static_term_identical_in_paired_arms": True,
        "only_relational_term_toggled_in_registered_pair": True,
        "effective_cross_calibration_weight_recomputed": True,
        "compute_matched": bool(contract.get("compute_matched", False)),
        "same_schedule_seed_contract": int(seed),
        "only_relational_transport_supervision_differs_within_registered_pair": bool(
            contract.get("compute_matched", False)
        ),
        "budget_profile": contract["active_budget_profile"],
        "budget_scope": "per_system",
        "endpoint_hidden_fraction": 0.0,
        "evidence_scope": "weak",
        "operator": "measurement_only_endpoint_plus_masked_proxy",
        "strict_effective_tier_requires_known_equal_calibration_hashes": True,
        "strict_effective_tier_requires_known_exposure": True,
        "unknown_strict_pairs_use_scale_free_cross_calibration_encoder": True,
        "target_transport_proxy_order": [
            "relative_amplitude_0",
            "relative_amplitude_1",
            "relative_amplitude_2",
            "relative_phase_0",
            "relative_phase_1",
            "relative_phase_2",
        ],
        "target_transport_proxy_scales": list(contract["target_transport_proxy_scales"]),
        "real_action_supervision_mask_sum": 0,
        "jacobian_supervision_mask_sum": 0,
        "counts_as_real_causality_evidence": False,
        "locked_test_accessed": False,
        "all_real_pair_auxiliaries_repeat_weighted": True,
        "parameter_groups": parameter_audit,
        "gradient_norms": v7_training._average_logs(gradient_logs),
        "by_system": by_system,
        "by_calibration_tier": tier_metrics,
        "train_fit_diagnostics": train_fit_diagnostics,
        "development_validation_negative_controls": (
            validation_controls_by_system
        ),
        "weak_validation_gate_passed": weak_validation_gate_passed,
        "weak_branch_eligible_for_selection": weak_validation_gate_passed,
    }


def _temperature_filtered_training_contexts(
    contexts: Mapping[str, V8FeedbackSystemContext],
    config: Mapping[str, Any],
    profile: V8FeedbackTrainingProfile,
) -> Tuple[Mapping[str, V8FeedbackSystemContext], Dict[str, Any]]:
    """Build an auditable 4Q-only <=22 C training view.

    This never mutates trusted/hard-fault labels and never filters validation.
    Rows without a causally aligned sensor reading are retained.
    """

    threshold = profile.real_training_temperature_max_c
    if threshold is None:
        raise ValueError("temperature training view requires an explicit threshold")
    systems = tuple(profile.real_training_temperature_filter_systems)
    operator_contract = dict(
        config.get("_v8_operator_curated_corpus")
        or config["operator_curated_corpus"]
    )
    environment = dict(operator_contract["environment_quality_contract"])
    if float(threshold) != float(environment["candidate_temperature_max_c"]):
        raise ValueError("profile temperature threshold disagrees with the contract")
    if systems != tuple(environment["candidate_systems"]):
        raise ValueError("profile temperature systems disagree with the contract")

    from .environment_context import align_environment_context, load_environment_log
    from .feedback_data import load_v8_measured_data_bundle

    waiver = dict(config.get("measured_audit_waiver") or {}) or None
    bundle = load_v8_measured_data_bundle(
        dict(config.get("_v8_data_config") or config["data"])["manifest_dir"],
        audit_waiver=waiver,
        curated_corpus_contract=operator_contract,
    )
    log = load_environment_log(
        environment["source_path_external"],
        expected_sha256=environment["source_sha256"],
        expected_sensor_serial=environment["sensor_serial"],
    )
    features = align_environment_context(bundle.records, log)
    if tuple(row.sample_id for row in bundle.rows) != tuple(
        record.sample_id for record in bundle.records
    ):
        raise ValueError("measured rows and environment records are misaligned")
    temperature_by_id = {
        record.sample_id: (
            float(features.causal_temperature_c[index]),
            bool(features.causal_context_mask[index]),
        )
        for index, record in enumerate(bundle.records)
    }

    filtered_contexts: Dict[str, V8FeedbackSystemContext] = {}
    by_system: Dict[str, Any] = {}
    for system_type, context in contexts.items():
        base_train = np.asarray(context.arrays.split_mask("train"), dtype=bool)
        allowed = np.ones(context.arrays.sample_count, dtype=bool)
        known_temperature = np.zeros(context.arrays.sample_count, dtype=bool)
        temperature = np.full(context.arrays.sample_count, np.nan, dtype=np.float64)
        for index, sample_id in enumerate(context.arrays.sample_ids):
            if sample_id not in temperature_by_id:
                raise ValueError("temperature training view is missing a sample ID")
            temperature[index], known_temperature[index] = temperature_by_id[sample_id]
        allowed = _temperature_training_allowed_mask(
            temperature,
            known_temperature,
            threshold_c=float(threshold),
            enabled=system_type in systems,
        )
        removed = base_train & ~allowed
        kept = base_train & allowed
        filtered_contexts[system_type] = replace(
            context,
            arrays=_FilteredTrainingArraysView(context.arrays, allowed),
        )
        by_system[system_type] = {
            "original_training_row_count": int(np.sum(base_train)),
            "retained_training_row_count": int(np.sum(kept)),
            "removed_training_row_count": int(np.sum(removed)),
            "unknown_temperature_training_row_count": int(
                np.sum(base_train & ~known_temperature)
            ),
            "removed_sample_ids_sha256": _stable_json_hash(
                [
                    context.arrays.sample_ids[index]
                    for index in np.flatnonzero(removed)
                ]
            ),
        }
    audit = {
        "schema": "v8_real_temperature_training_view_v1",
        "enabled": True,
        "threshold_c": float(threshold),
        "systems": list(systems),
        "source_sha256": log.contract.source_sha256,
        "environment_log_stable_hash": log.stable_hash,
        "environment_feature_stable_hash": features.stable_hash,
        "unknown_temperature_rows_retained": True,
        "primary_validation_filtered": False,
        "hard_labels_changed": False,
        "by_system": by_system,
    }
    audit["audit_sha256"] = _stable_json_hash(audit)
    return filtered_contexts, audit


def _temperature_training_allowed_mask(
    temperature_c: np.ndarray,
    known_mask: np.ndarray,
    *,
    threshold_c: float,
    enabled: bool,
) -> np.ndarray:
    temperature = np.asarray(temperature_c, dtype=np.float64)
    known = np.asarray(known_mask, dtype=bool)
    if temperature.shape != known.shape or temperature.ndim != 1:
        raise ValueError("temperature training arrays must share one-dimensional shape")
    if not math.isfinite(float(threshold_c)):
        raise ValueError("temperature training threshold must be finite")
    if not enabled:
        return np.ones(temperature.shape, dtype=bool)
    if np.any(known & ~np.isfinite(temperature)):
        raise ValueError("known temperature rows must be finite")
    return ~(known & (temperature > float(threshold_c)))


def _train_measured_stage_with_transport(
    model: Any,
    contexts: Mapping[str, V8FeedbackSystemContext],
    profile: V8FeedbackTrainingProfile,
    config: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: "torch.device",
    classifier_prior: Any,
) -> Dict[str, Any]:
    measured_contexts = contexts
    training_view_audit: Dict[str, Any] = {
        "schema": "v8_real_temperature_training_view_v1",
        "enabled": False,
        "primary_validation_filtered": False,
        "hard_labels_changed": False,
    }
    if profile.real_training_temperature_max_c is not None:
        measured_contexts, training_view_audit = _temperature_filtered_training_contexts(
            contexts,
            config,
            profile,
        )
    measured = _ORIGINAL_V7_FUNCTIONS["_train_measured_stage"](
        model,
        measured_contexts,
        profile,
        config,
        weights,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        classifier_prior=classifier_prior,
    )
    state_pair_optimizer_required = bool(
        profile.weak_state_pair_supervision_enabled
        or profile.weak_supervision_pair_id
    )
    transport = (
        _train_state_pair_transport_stage(
            model,
            contexts,
            config,
            seed=int(seed + 7000003),
            device=device,
            classifier_prior=classifier_prior,
        )
        if int(epochs) > 0 and state_pair_optimizer_required
        else {
            "status": "stage_skipped",
            "steps": 0,
            "evidence_scope": "weak",
            "real_action_supervision_mask_sum": 0,
            "skip_reason": (
                "zero_measured_epochs"
                if int(epochs) <= 0
                else "profile_not_registered_for_weak_supervision"
            ),
            "relational_state_pair_supervision_enabled": False,
            "compute_matched": False,
            "weak_branch_eligible_for_selection": False,
        }
    )
    return {
        **measured,
        "real_temperature_training_view": training_view_audit,
        "state_pair_transport_weak": transport,
    }


def _train_v8_synthetic_model_b_stage(
    model: Any,
    contexts: Mapping[str, V8FeedbackSystemContext],
    profile: V8FeedbackTrainingProfile,
    config: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    seed: int,
    device: "torch.device",
    classifier_prior: Any,
) -> Dict[str, Any]:
    """Train the policy on causal histories from the same declared train plant."""

    from .feedback_synthetic import generate_v8_transition_batch

    steps = int(profile.synthetic_model_b_steps)
    if steps <= 0:
        return {"steps": 0, "mean": {}, "by_system": {}, "family_role": "train"}
    family_ids = tuple(str(item) for item in config["_v8_train_families"])
    optimization = config["optimization"]
    policy_config = dict(config.get("policy") or {})
    optimizer = v7_training._make_optimizer(
        model,
        learning_rate=float(optimization["synthetic_learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    amp_dtype = str(optimization.get("amp_dtype", "bfloat16"))
    scaler = v7_training._make_scaler(
        device, bool(optimization["use_amp"]), amp_dtype
    )
    system_names = tuple(contexts)
    all_logs: List[Dict[str, float]] = []
    system_logs: Dict[str, List[Dict[str, float]]] = {key: [] for key in contexts}
    family_steps: Dict[str, int] = {key: 0 for key in family_ids}
    model.train()
    for step in range(steps):
        system_type = system_names[step % len(system_names)]
        family_id = family_ids[step % len(family_ids)]
        context = contexts[system_type]
        synthetic = generate_v8_transition_batch(
            context.arrays,
            context.profile,
            batch_size=profile.synthetic_batch_size,
            seed=int(seed + step * 130363),
            family_id=family_id,
        )
        tensor = _v8_synthetic_tensors(synthetic, device=device)
        tensor.update(v7_training._classifier_prior_features(
            classifier_prior,
            system_type=system_type,
            target_features=tensor["target_features"],
            rho_available_mask=tensor["rho_available_mask"],
            core_rho=tensor["current_rho"],
        ))
        with v7_training._autocast(
            device, bool(optimization["use_amp"]), amp_dtype
        ):
            loss, log = v7_training._synthetic_model_b_loss(
                model,
                tensor,
                context,
                weights,
                backbone_gradient_scale=profile.policy_backbone_gradient_scale,
                mode_class_weights=policy_config.get(
                    "mode_class_weights",
                    [1.0] * len(v7_training.ACTION_MODE_NAMES),
                ),
                mode_label_smoothing=float(
                    policy_config.get("mode_label_smoothing", 0.0)
                ),
            )
        v7_training._optimization_step(
            loss=loss,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=float(optimization["gradient_clip_norm"]),
        )
        family_steps[family_id] += 1
        all_logs.append(log)
        system_logs[system_type].append(log)
    return {
        **v7_training._stage_log_summary(all_logs),
        "by_system": {
            system_type: v7_training._average_logs(rows)
            for system_type, rows in system_logs.items()
        },
        "family_role": "train",
        "family_steps": family_steps,
        "causal_history_bound": True,
    }


def _v8_development_transition_generator(
    arrays: Any,
    profile: Any,
    *,
    batch_size: int,
    seed: int,
    config: Any = None,
) -> Dict[str, Any]:
    from .feedback_synthetic import generate_v8_transition_batch

    family_id = str(_ACTIVE_DEVELOPMENT_FAMILIES[0])
    return generate_v8_transition_batch(
        arrays,
        profile,
        batch_size=int(batch_size),
        seed=int(seed),
        family_id=family_id,
    )


def _v8_development_policy_generator(
    arrays: Any,
    profile: Any,
    *,
    batch_size: int,
    seed: int,
    config: Any = None,
) -> Dict[str, Any]:
    return _v8_development_transition_generator(
        arrays,
        profile,
        batch_size=batch_size,
        seed=seed,
        config=config,
    )


def _evaluate_development_synthetic_data_with_controls(
    model: Any,
    contexts: Mapping[str, V8FeedbackSystemContext],
    profile: V8FeedbackTrainingProfile,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: "torch.device",
    classifier_prior: Any = None,
    stop_probability_thresholds: Optional[Mapping[str, Any]] = None,
    classifier_rho_available: Optional[bool] = None,
) -> Dict[str, Any]:
    result = _ORIGINAL_V7_FUNCTIONS["evaluate_synthetic_data"](
        model,
        contexts,
        profile,
        config,
        seed=seed,
        device=device,
        classifier_prior=classifier_prior,
        stop_probability_thresholds=stop_probability_thresholds,
        classifier_rho_available=classifier_rho_available,
    )
    if classifier_rho_available is False:
        return result
    from .evaluate_feedback_models import (
        evaluate_synthetic_causal_negative_controls,
    )

    configured_samples = int(config["selection"]["synthetic_evaluation_samples"])
    sample_count = (
        min(configured_samples, max(16, profile.evaluation_max_samples_per_split))
        if profile.evaluation_max_samples_per_split > 0
        else configured_samples
    )
    result["v8_causal_negative_controls"] = (
        evaluate_synthetic_causal_negative_controls(
            model,
            contexts,
            config,
            device=device,
            classifier_prior=classifier_prior,
            family_ids=config["_v8_development_families"],
            sample_count=sample_count,
            base_seed=int(seed + 71000041),
            allow_locked_evaluation=False,
            channel="synthetic_development_selection_control",
            allowed_for_model_selection=True,
            state_endpoint_contract_reference=_stable_json_hash(
                config["_v8_state_pair_contract"]
            ),
        )
    )
    result["v8_causal_negative_controls_count_as_real_causality"] = False
    return result


_ORIGINAL_V7_FUNCTIONS: Dict[str, Any] = {}
_ACTIVE_DEVELOPMENT_FAMILIES: Tuple[str, ...] = (
    "mechanism_shift_development",
)


def _v8_selection_score(
    measured: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    closed_loop: Mapping[str, Any],
) -> Dict[str, float]:
    """Keep V8 evaluation metadata out of the inherited V7 system reducer."""

    system_keys = tuple(
        key for key, value in measured.items()
        if isinstance(value, Mapping) and "splits" in value
    )
    if not system_keys:
        raise ValueError("V8 selection received no measured system results")
    missing_synthetic = [key for key in system_keys if key not in synthetic]
    missing_closed_loop = [key for key in system_keys if key not in closed_loop]
    if missing_synthetic or missing_closed_loop:
        raise KeyError(
            "V8 selection is missing system results: "
            f"synthetic={missing_synthetic}, closed_loop={missing_closed_loop}"
        )
    return _ORIGINAL_V7_FUNCTIONS["_selection_score"](
        measured,
        {key: synthetic[key] for key in system_keys},
        {key: closed_loop[key] for key in system_keys},
    )


@contextlib.contextmanager
def v8_v7_reuse_seam(
    *,
    development_families: Optional[Sequence[str]] = None,
) -> Iterator[None]:
    """Temporarily route generic V7 loops through strict V8 components."""

    from .feedback_models import (
        V8FeedbackModelBundle,
        V8FeedbackModelConfig,
        parameter_count,
    )
    from .batched_stop_calibration import (
        calibrate_policy_stop_thresholds as calibrate_v8_policy_stop_thresholds,
    )

    global _ACTIVE_DEVELOPMENT_FAMILIES
    previous_development_families = _ACTIVE_DEVELOPMENT_FAMILIES
    if development_families is not None:
        _ACTIVE_DEVELOPMENT_FAMILIES = tuple(
            str(item) for item in development_families
        )
    replacements = {
        "V7FeedbackModelBundle": V8FeedbackModelBundle,
        "V7FeedbackModelConfig": V8FeedbackModelConfig,
        "parameter_count": parameter_count,
        "_measured_batch": _v8_measured_batch,
        "_measured_loss": _v8_measured_loss,
        "_initialize_physical_head_biases": _initialize_v8_physical_head_biases,
        "_train_measured_stage": _train_measured_stage_with_transport,
        "_train_synthetic_model_a_stage": _train_v8_synthetic_model_a_stage,
        "_train_synthetic_model_b_stage": _train_v8_synthetic_model_b_stage,
        "evaluate_measured_data": _evaluate_development_measured_data,
        "evaluate_synthetic_data": _evaluate_development_synthetic_data_with_controls,
        "calibrate_policy_stop_thresholds": calibrate_v8_policy_stop_thresholds,
        "_selection_score": _v8_selection_score,
        "generate_feedback_transition_batch": _v8_development_transition_generator,
        "generate_feedback_policy_batch": _v8_development_policy_generator,
    }
    previous: Dict[str, Any] = {}
    for name, replacement in replacements.items():
        previous[name] = getattr(v7_training, name)
        _ORIGINAL_V7_FUNCTIONS[name] = previous[name]
        setattr(v7_training, name, replacement)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(v7_training, name, value)
        _ORIGINAL_V7_FUNCTIONS.clear()
        _ACTIVE_DEVELOPMENT_FAMILIES = previous_development_families


def _git_bytes(arguments: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
        )
        return bytes(result.stdout)
    except (OSError, subprocess.CalledProcessError):
        return b"unavailable"


def collect_source_tree_audit(config_path: str | Path) -> Dict[str, Any]:
    """Hash HEAD, dirty patch, untracked V8 sources and dependency sources."""

    root_text = _git_bytes(("rev-parse", "--show-toplevel")).decode(
        "utf-8", errors="replace"
    ).strip()
    repository_root = Path(root_text).resolve() if root_text != "unavailable" else Path.cwd().resolve()
    source_paths = [
        Path("ml_recon/physics_hybrid_v8"),
        Path("ml_recon/physics_hybrid_v7/train_feedback_models.py"),
        Path("ml_recon/physics_hybrid_v7/feedback_data.py"),
        Path("ml_recon/physics_hybrid_v7/feedback_synthetic.py"),
        Path("ml_recon/physics_hybrid_v7/feedback_models.py"),
        Path("ml_recon/physics_hybrid_v7/geometry.py"),
        Path("ml_recon/physics_hybrid_v7/feedback_classifier_prior.py"),
        Path("ml_recon/physics_hybrid_v7/feedback_metrics.py"),
        Path("ml_recon/physics_hybrid_v7/physical_ceiling.py"),
        Path("ml_recon/physics_hybrid_v5/command_contract.py"),
        Path("ml_recon/physics_hybrid_v5/projective_action.py"),
        Path("ml_recon/physics_hybrid_v5/full_synthetic_generator.py"),
        Path("ml_recon/physics_hybrid_v4/phase_geometry.py"),
        Path("ml_recon/measured_data/contracts.py"),
        Path("ml_recon/measured_data/datasets.py"),
        Path("ml_recon/physics_identification/operation_point_calibration.py"),
        Path(
            "system_control/qubit_recon_files/matrix/"
            "retrieval_efficiency_3qubit.txt"
        ),
        Path(
            "system_control/qubit_recon_files/matrix/"
            "retrieval_efficiency_4qubit.txt"
        ),
        Path(config_path),
    ]
    files: List[Path] = []
    for path in source_paths:
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob("*.py") if item.is_file()))
        elif path.is_file():
            files.append(path)
    source_hashes: Dict[str, str] = {}
    for path in sorted(set(files), key=lambda item: str(item)):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(f"source audit path escapes repository root: {resolved}") from exc
        source_hashes[relative.as_posix()] = _file_sha256(resolved)
    status = _git_bytes(("status", "--porcelain=v1", "--untracked-files=all"))
    patch = _git_bytes(("diff", "--binary", "HEAD", "--"))
    untracked_v8 = b"".join(
        path.read_bytes()
        for path in sorted(set(files), key=lambda item: str(item))
        if _git_bytes(("ls-files", "--error-unmatch", "--", str(path))) == b"unavailable"
    )
    patch_digest = hashlib.sha256(patch + b"\0" + untracked_v8).hexdigest()
    head = _git_bytes(("rev-parse", "HEAD")).decode("utf-8", errors="replace").strip()
    audit = {
        "schema": SOURCE_AUDIT_SCHEMA,
        "git_head": head,
        "dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
        "dirty_patch_sha256": patch_digest,
        "source_file_sha256": source_hashes,
        "source_tree_sha256": _stable_json_hash(source_hashes),
    }
    audit["audit_sha256"] = _stable_json_hash(audit)
    return audit


def split_hash_by_system(
    contexts: Mapping[str, V8FeedbackSystemContext],
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for system_type, context in contexts.items():
        digest = hashlib.sha256()
        arrays = context.arrays
        for index, sample_id in enumerate(arrays.sample_ids):
            row = (
                str(sample_id),
                str(arrays.split_names[index]),
                str(arrays.repeat_group_ids[index]),
                str(arrays.time_block_ids[index]),
                format(float(arrays.sample_weight[index]), ".9g"),
                str(int(arrays.rho_available_mask[index] > 0.5)),
                str(int(context.exposure_known_mask[index] > 0.5)),
            )
            digest.update(("|".join(row) + "\n").encode("utf-8"))
        result[system_type] = digest.hexdigest()
    return result


def build_uncertainty_state(
    config: Mapping[str, Any],
    *,
    paired_episode_group_count: int,
    calibration_artifact: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return calibrated state or an explicit fail-closed abstention state."""

    from .feedback_confidence import (
        REAL_PAIRED_EVIDENCE_SOURCE,
        V8ConfidenceConfig,
        empty_real_conformal_gate,
    )

    contract = dict(config["uncertainty_contract"])
    minimum = int(contract["minimum_real_paired_episode_groups"])
    confidence_config = V8ConfidenceConfig(
        conformal_alpha=float(contract["conformal_alpha"]),
        minimum_calibration_groups=minimum,
        require_real_paired_episode=True,
    )
    base = {
        "schema": UNCERTAINTY_SCHEMA,
        "group_key": "episode_id",
        "paired_episode_group_count": int(paired_episode_group_count),
        "minimum_real_paired_episode_groups": minimum,
        "synthetic_calibration_used": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    if paired_episode_group_count < minimum or calibration_artifact is None:
        empty_gate = empty_real_conformal_gate(config=confidence_config)
        return {
            **base,
            "status": "abstain_insufficient_real_paired_episode_calibration",
            "grouped_bootstrap_fitted": False,
            "conformal_fitted": False,
            "policy_execution_allowed": False,
            "abstain": True,
            "conformal_gate": empty_gate.to_json_dict(),
        }
    artifact = dict(calibration_artifact)
    if artifact.get("source") not in {
        "real_paired_transition_episode_only",
        REAL_PAIRED_EVIDENCE_SOURCE,
    }:
        raise ValueError("uncertainty artifact is not real-episode calibrated")
    if artifact.get("group_key") != "episode_id":
        raise ValueError("uncertainty artifact is not grouped by episode")
    if artifact.get("synthetic_rows_used") not in (0, False):
        raise ValueError("synthetic rows cannot calibrate reported uncertainty")
    if artifact.get("grouped_bootstrap_fitted") is not True:
        raise ValueError("grouped bootstrap ensemble is not fitted")
    if artifact.get("conformal_fitted") is not True:
        raise ValueError("episode-group conformal calibration is not fitted")
    return {
        **base,
        "status": "real_episode_calibrated_shadow_only",
        "grouped_bootstrap_fitted": True,
        "conformal_fitted": True,
        "policy_execution_allowed": False,
        "abstain": False,
        "calibration_artifact_sha256": _stable_json_hash(artifact),
    }


def build_causal_context_gate(
    config: Mapping[str, Any],
    *,
    real_paired_or_signed_probe_count: int,
    negative_control_metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Gate causal context on pre-registered real-evidence shuffle controls."""

    contract = dict(config["causal_negative_control_contract"])
    base = {
        "schema": "v8_causal_context_negative_control_gate_v1",
        "real_paired_or_signed_probe_count": int(
            real_paired_or_signed_probe_count
        ),
        "target_only_rows_count_as_causal_evidence": False,
        "state_pair_transport_weak_counts_as_causal_evidence": False,
        "synthetic_negative_controls_reported_separately": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    if real_paired_or_signed_probe_count <= 0 or negative_control_metrics is None:
        return {
            **base,
            "causal_context_gate": False,
            "branch_eligible": False,
            "status": "fail_closed_no_real_causal_negative_control_evidence",
            "required_controls": [
                "target_context_group_derangement",
                "target_action_group_derangement",
                "state_endpoint_group_derangement",
            ],
        }
    metrics = dict(negative_control_metrics)
    checks = {
        "action_prediction_degradation": float(
            metrics["shuffled_action_normalized_mae"]
            - metrics["correct_action_normalized_mae"]
        ) >= float(contract["minimum_action_mae_degradation"]),
        "rank_accuracy_drop": float(
            metrics["correct_rank_top1_accuracy"]
            - metrics["shuffled_rank_top1_accuracy"]
        ) >= float(contract["minimum_rank_accuracy_drop"]),
        "jacobian_alignment_drop": float(
            metrics["correct_jacobian_alignment"]
            - metrics["shuffled_jacobian_alignment"]
        ) >= float(contract["minimum_jacobian_alignment_drop"]),
        "grouped_significance": float(metrics["grouped_bootstrap_p_value"])
        <= float(contract["maximum_grouped_bootstrap_p_value"]),
    }
    passed = bool(all(checks.values()))
    return {
        **base,
        "causal_context_gate": passed,
        "branch_eligible": passed,
        "status": "passed" if passed else "failed_preregistered_shuffle_control",
        "checks": checks,
        "metrics": metrics,
        "metrics_sha256": _stable_json_hash(metrics),
    }


def _measured_file_audit(config: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_dir = Path(config["data"]["manifest_dir"])
    names = (
        "measured_data_summary.json",
        "measured_splits_v1.json",
        "split_audit.json",
    )
    hashes = {
        name: _file_sha256(manifest_dir / name)
        for name in names
        if (manifest_dir / name).is_file()
    }
    waiver = dict(config.get("measured_audit_waiver") or {})
    expected_summary = str(waiver.get("summary_sha256", "")).lower()
    actual_summary = hashes.get("measured_data_summary.json", "").lower()
    if expected_summary and actual_summary != expected_summary:
        raise ValueError("measured audit waiver is not bound to the current summary hash")
    return {
        "file_sha256": hashes,
        "waiver": waiver,
        "audit_sha256": _stable_json_hash({"file_sha256": hashes, "waiver": waiver}),
    }


def _weak_supervision_comparison_contract(
    config: Mapping[str, Any],
    profile: V8FeedbackTrainingProfile,
) -> Dict[str, Any]:
    """Bind one arm to the external comparison produced after both runs exist."""

    pair_id = str(profile.weak_supervision_pair_id)
    if not pair_id:
        value = {
            "schema": WEAK_PAIRED_COMPARISON_CONTRACT_SCHEMA,
            "status": "not_registered_for_this_profile",
            "profile": profile.name,
            "weak_state_pair_supervision_enabled": bool(
                profile.weak_state_pair_supervision_enabled
            ),
            "eligible_for_paired_comparison": False,
        }
        value["contract_sha256"] = _stable_json_hash(value)
        return value

    pair_members = sorted(
        name
        for name, declaration in config["training_profiles"].items()
        if str(declaration.get("weak_supervision_pair_id", "")) == pair_id
    )
    if len(pair_members) != 2 or profile.name not in pair_members:
        raise ValueError("weak-supervision pair registration changed after validation")
    counterpart = next(name for name in pair_members if name != profile.name)
    normalized_declarations: Dict[str, Any] = {}
    ignored_keys = {
        "weak_state_pair_supervision_enabled",
        "weak_supervision_pair_id",
        "weak_supervision_control_for",
    }
    for name in pair_members:
        normalized_declarations[name] = {
            key: copy.deepcopy(item)
            for key, item in config["training_profiles"][name].items()
            if key not in ignored_keys
        }
    comparison_basis = {
        "pair_id": pair_id,
        "member_profiles": pair_members,
        "normalized_profile_declarations": normalized_declarations,
        "ablation_id": profile.ablation_id,
        "model_config_sha256": _stable_json_hash(
            profile.model_config().to_json_dict()
        ),
        "shared_seed_plans": {
            key: copy.deepcopy(config["execution_plans"][key])
            for key in ("local_3seed", "confirmation_5seed")
        },
        "same_checkpoint_selection_rule_required": True,
        "common_random_numbers_required": True,
        "paired_comparison_gate": copy.deepcopy(
            config["state_pair_transport_weak_contract"]["paired_comparison"]
        ),
    }
    value = {
        "schema": WEAK_PAIRED_COMPARISON_CONTRACT_SCHEMA,
        "status": "registered_not_yet_compared",
        "pair_id": pair_id,
        "profile": profile.name,
        "counterpart_profile": counterpart,
        "role": (
            "weak_on_candidate"
            if profile.weak_state_pair_supervision_enabled
            else "weak_off_capacity_matched_control"
        ),
        "weak_state_pair_supervision_enabled": bool(
            profile.weak_state_pair_supervision_enabled
        ),
        "capacity_and_budget_matched": True,
        "eligible_for_paired_comparison": True,
        "comparison_basis": comparison_basis,
        "comparison_basis_sha256": _stable_json_hash(comparison_basis),
        "comparison_result_status": (
            "pending_external_hash_bound_artifact_after_both_reports_exist"
        ),
        "comparison_result_schema": WEAK_PAIRED_COMPARISON_SCHEMA,
    }
    value["contract_sha256"] = _stable_json_hash(value)
    return value


def _load_weak_comparison_training_report(
    path_value: str | Path,
) -> Tuple[Path, Dict[str, Any]]:
    """Load one development report and verify its checkpoint binding."""

    path = Path(path_value).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != TRAINING_REPORT_SCHEMA:
        raise ValueError(f"unsupported weak-comparison training report: {path}")
    expected_report_hash = _stable_json_hash({
        key: value for key, value in report.items() if key != "report_hash"
    })
    if str(report.get("report_hash", "")) != expected_report_hash:
        raise ValueError(f"weak-comparison training report hash mismatch: {path}")
    if (
        report.get("selection_scope") != "development_only"
        or report.get("locked_final_metrics_present") is not False
        or dict(report.get("locked_final_test") or {}).get("status")
        != "sealed_not_accessed"
    ):
        raise ValueError("weak comparison may use development evidence only")
    checkpoint = dict(report.get("checkpoint") or {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if _file_sha256(checkpoint_path) != str(checkpoint.get("sha256", "")):
        raise ValueError("weak-comparison checkpoint hash differs from its report")
    eligibility = dict(report.get("development_selection_eligibility") or {})
    expected_eligibility_hash = _stable_json_hash({
        key: value for key, value in eligibility.items()
        if key != "eligibility_sha256"
    })
    if str(eligibility.get("eligibility_sha256", "")) != expected_eligibility_hash:
        raise ValueError("weak-comparison development eligibility hash mismatch")
    contract = dict(report.get("weak_supervision_paired_comparison") or {})
    if contract.get("schema") != WEAK_PAIRED_COMPARISON_CONTRACT_SCHEMA:
        raise ValueError("weak-comparison report lacks its registered pair contract")
    expected_contract_hash = _stable_json_hash({
        key: value for key, value in contract.items() if key != "contract_sha256"
    })
    if str(contract.get("contract_sha256", "")) != expected_contract_hash:
        raise ValueError("weak-comparison pair contract hash mismatch")
    return path, report


def _latest_weak_stage(
    report: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    available: List[Tuple[str, Dict[str, Any]]] = []
    training = dict(report.get("training") or {})
    for stage_name in ("measured_pretrain", "measured_adapter"):
        weak = dict(
            dict(training.get(stage_name) or {}).get(
                "state_pair_transport_weak"
            ) or {}
        )
        if weak and weak.get("status") != "stage_skipped":
            available.append((stage_name, weak))
    if not available:
        raise ValueError("weak-comparison report has no executed state-pair stage")
    return available[-1]


def build_weak_supervision_paired_comparison(
    weak_on_report_path: str | Path,
    weak_off_control_report_path: str | Path,
    *,
    config: Mapping[str, Any],
    output_path: str | Path,
) -> Dict[str, Any]:
    """Create one seed-matched, hash-bound weak-on versus weak-off artifact."""

    weak_on_path, weak_on = _load_weak_comparison_training_report(
        weak_on_report_path
    )
    control_path, control = _load_weak_comparison_training_report(
        weak_off_control_report_path
    )
    weak_on_profile = dict(weak_on.get("profile") or {})
    control_profile = dict(control.get("profile") or {})
    weak_on_contract = dict(
        weak_on.get("weak_supervision_paired_comparison") or {}
    )
    control_contract = dict(
        control.get("weak_supervision_paired_comparison") or {}
    )
    pair_id = str(weak_on_contract.get("pair_id", ""))
    structural_checks = {
        "same_seed": int(weak_on.get("seed", -1)) == int(control.get("seed", -2)),
        "opposite_registered_roles": (
            weak_on_profile.get("weak_state_pair_supervision_enabled") is True
            and control_profile.get("weak_state_pair_supervision_enabled") is False
            and weak_on_contract.get("role") == "weak_on_candidate"
            and control_contract.get("role")
            == "weak_off_capacity_matched_control"
        ),
        "same_pair_id": bool(pair_id)
        and pair_id == str(control_contract.get("pair_id", "")),
        "mutual_counterpart_profiles": (
            str(weak_on_contract.get("counterpart_profile", ""))
            == str(control_profile.get("name", ""))
            and str(control_contract.get("counterpart_profile", ""))
            == str(weak_on_profile.get("name", ""))
        ),
        "same_comparison_basis": (
            str(weak_on_contract.get("comparison_basis_sha256", ""))
            == str(control_contract.get("comparison_basis_sha256", ""))
            and bool(weak_on_contract.get("comparison_basis_sha256"))
        ),
        "capacity_and_budget_matched": (
            weak_on_contract.get("capacity_and_budget_matched") is True
            and control_contract.get("capacity_and_budget_matched") is True
        ),
        "same_config": (
            str(weak_on.get("config_sha256", ""))
            == str(control.get("config_sha256", ""))
            == str(config.get("_config_sha256", ""))
        ),
        "same_split": (
            weak_on.get("split_hash_by_system")
            == control.get("split_hash_by_system")
        ),
        "same_measured_and_model_audits": all(
            weak_on.get(field_name) == control.get(field_name)
            for field_name in (
                "measured_audit_sha256",
                "model_audit_sha256",
                "source_tree_audit",
            )
        ),
    }
    if not all(structural_checks.values()):
        failed = [name for name, passed in structural_checks.items() if not passed]
        raise ValueError(f"weak paired-comparison structural mismatch: {failed}")

    weak_on_stage_name, weak_on_stage = _latest_weak_stage(weak_on)
    control_stage_name, control_stage = _latest_weak_stage(control)
    schedule_fields = (
        "scheduled_endpoint_ids_sha256",
        "scheduled_pair_ids_sha256",
        "pair_set_sha256",
        "maximum_batches_per_system",
        "steps",
    )
    weak_on_by_system = dict(weak_on_stage.get("by_system") or {})
    control_by_system = dict(control_stage.get("by_system") or {})
    schedule_match = bool(
        weak_on_stage_name == control_stage_name
        and weak_on_stage.get("compute_matched") is True
        and control_stage.get("compute_matched") is True
        and weak_on_stage.get("same_schedule_seed_contract")
        == control_stage.get("same_schedule_seed_contract")
        and set(weak_on_by_system) == set(control_by_system)
        and all(
            all(
                dict(weak_on_by_system[system_type]).get(field_name)
                == dict(control_by_system[system_type]).get(field_name)
                for field_name in schedule_fields
            )
            for system_type in weak_on_by_system
        )
    )
    if not schedule_match:
        raise ValueError("weak paired-comparison endpoint schedule or compute budget differs")

    gate_contract = copy.deepcopy(
        dict(config["state_pair_transport_weak_contract"]["paired_comparison"])
    )
    registered_gate = dict(
        dict(weak_on_contract.get("comparison_basis") or {}).get(
            "paired_comparison_gate"
        ) or {}
    )
    if registered_gate != gate_contract:
        raise ValueError("weak paired-comparison gate differs from the trained contract")
    weights = {
        str(name): float(value)
        for name, value in dict(gate_contract["static_component_weights"]).items()
    }
    weak_on_selection = dict(weak_on.get("selection") or {})
    control_selection = dict(control.get("selection") or {})
    component_values: Dict[str, Dict[str, float]] = {}
    for name, weight in sorted(weights.items()):
        weak_on_value = float(weak_on_selection[name])
        control_value = float(control_selection[name])
        if not all(math.isfinite(value) for value in (weak_on_value, control_value)):
            raise ValueError(f"weak paired-comparison component is non-finite: {name}")
        component_values[name] = {
            "weak_on": weak_on_value,
            "weak_off_control": control_value,
            "paired_gain": weak_on_value - control_value,
            "weight": weight,
        }
    static_composite_gain = float(sum(
        value["weight"] * value["paired_gain"]
        for value in component_values.values()
    ))
    joint_selection_gain = float(
        weak_on_selection["joint_selection_score"]
        - control_selection["joint_selection_score"]
    )
    absolute_component_delta = max(
        abs(value["paired_gain"]) for value in component_values.values()
    )
    improved_component_count = sum(
        value["paired_gain"] > 0.0 for value in component_values.values()
    )
    weak_on_eligibility = dict(
        weak_on.get("development_selection_eligibility") or {}
    )
    control_eligibility = dict(
        control.get("development_selection_eligibility") or {}
    )
    result_checks = {
        "candidate_development_negative_control_passed": bool(
            weak_on_stage.get("relational_state_pair_supervision_enabled") is True
            and weak_on_stage.get("weak_validation_gate_passed") is True
            and weak_on_stage.get("weak_branch_eligible_for_selection") is True
            and weak_on_eligibility.get("selection_eligible") is True
        ),
        "control_is_diagnostic_only": bool(
            control_eligibility.get("selection_eligible") is False
            and control_eligibility.get("reason")
            == "compute_matched_non_relational_control_is_diagnostic_only"
        ),
        "checkpoint_response_is_non_degenerate": bool(
            str(weak_on["checkpoint"]["sha256"])
            != str(control["checkpoint"]["sha256"])
            if gate_contract["require_checkpoint_hash_difference"]
            else True
        ),
        "static_component_response_is_non_degenerate": bool(
            absolute_component_delta
            >= float(gate_contract["minimum_absolute_static_component_delta"])
        ),
        "minimum_improved_static_component_count_passed": bool(
            improved_component_count
            >= int(gate_contract["minimum_improved_static_component_count"])
        ),
        "static_composite_gain_passed": bool(
            static_composite_gain
            > float(gate_contract["minimum_static_composite_gain_exclusive"])
        ),
        "joint_selection_non_regression_passed": bool(
            joint_selection_gain
            >= float(gate_contract["minimum_joint_selection_score_gain"])
        ),
    }
    passed = bool(all(result_checks.values()))
    artifact = {
        "schema": WEAK_PAIRED_COMPARISON_SCHEMA,
        "status": "passed" if passed else "failed_preregistered_gate",
        "comparison_result_present": True,
        "selection_scope": "development_only",
        "locked_final_metrics_present": False,
        "allowed_for_model_selection": True,
        "pair_id": pair_id,
        "seed": int(weak_on["seed"]),
        "weak_on": {
            "profile": str(weak_on_profile["name"]),
            "training_report_path": str(weak_on_path),
            "training_report_file_sha256": _file_sha256(weak_on_path),
            "training_report_hash": str(weak_on["report_hash"]),
            "checkpoint_sha256": str(weak_on["checkpoint"]["sha256"]),
            "selection_eligibility_sha256": str(
                weak_on_eligibility["eligibility_sha256"]
            ),
            "weak_stage_sha256": _stable_json_hash(weak_on_stage),
        },
        "weak_off_control": {
            "profile": str(control_profile["name"]),
            "training_report_path": str(control_path),
            "training_report_file_sha256": _file_sha256(control_path),
            "training_report_hash": str(control["report_hash"]),
            "checkpoint_sha256": str(control["checkpoint"]["sha256"]),
            "selection_eligibility_sha256": str(
                control_eligibility["eligibility_sha256"]
            ),
            "weak_stage_sha256": _stable_json_hash(control_stage),
        },
        "config_sha256": str(config["_config_sha256"]),
        "comparison_basis_sha256": str(
            weak_on_contract["comparison_basis_sha256"]
        ),
        "gate_contract": gate_contract,
        "gate_contract_sha256": _stable_json_hash(gate_contract),
        "compute_match": {
            "same_stage": weak_on_stage_name,
            "same_schedule_seed": int(
                weak_on_stage["same_schedule_seed_contract"]
            ),
            "same_endpoint_and_pair_schedule": True,
            "same_optimizer_step_count": True,
            "same_endpoint_static_objective": True,
            "only_relational_transport_supervision_differs": True,
        },
        "structural_checks": structural_checks,
        "result_checks": result_checks,
        "metrics": {
            "static_components": component_values,
            "static_composite_gain": static_composite_gain,
            "joint_selection_score_gain": joint_selection_gain,
            "maximum_absolute_static_component_delta": absolute_component_delta,
            "improved_static_component_count": improved_component_count,
        },
        "negative_control_evidence": {
            "candidate_weak_validation_gate_passed": bool(
                weak_on_stage.get("weak_validation_gate_passed") is True
            ),
            "candidate_weak_stage_sha256": _stable_json_hash(weak_on_stage),
            "development_validation_negative_controls_sha256": _stable_json_hash(
                weak_on_stage.get("development_validation_negative_controls")
                or {}
            ),
        },
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    artifact["artifact_sha256"] = _stable_json_hash(artifact)
    selected_output = Path(output_path).resolve()
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected_output.with_name(f"{selected_output.name}.tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, selected_output)
    return artifact


def development_selection_eligibility(
    report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fail closed before any weak relational branch enters a leaderboard."""

    profile = dict(report.get("profile") or {})
    if (
        int(profile.get("synthetic_model_a_steps", -1)) == 0
        and int(profile.get("synthetic_model_b_steps", -1)) == 0
    ):
        value = {
            "schema": "v8_development_selection_eligibility_v1",
            "selection_eligible": False,
            "reason": (
                "static_real_only_candidate_requires_dedicated_multi_seed_"
                "count_and_negative_control_gate"
            ),
            "relational_weak_branch": False,
            "weak_validation_gate_required": False,
            "weak_supervision_pair_id": str(
                profile.get("weak_supervision_pair_id", "")
            ),
            "temperature_training_view_enabled": (
                profile.get("real_training_temperature_max_c") is not None
            ),
        }
        value["eligibility_sha256"] = _stable_json_hash(value)
        return value
    relational_enabled = bool(
        profile.get("weak_state_pair_supervision_enabled", False)
    )
    pair_id = str(profile.get("weak_supervision_pair_id", ""))
    if not relational_enabled:
        if pair_id:
            value = {
                "schema": "v8_development_selection_eligibility_v1",
                "selection_eligible": False,
                "reason": "compute_matched_non_relational_control_is_diagnostic_only",
                "relational_weak_branch": False,
                "weak_validation_gate_required": False,
                "weak_supervision_pair_id": pair_id,
            }
        else:
            value = {
                "schema": "v8_development_selection_eligibility_v1",
                "selection_eligible": True,
                "reason": "profile_has_no_relational_weak_branch",
                "relational_weak_branch": False,
                "weak_validation_gate_required": False,
                "weak_supervision_pair_id": "",
            }
        value["eligibility_sha256"] = _stable_json_hash(value)
        return value

    training = dict(report.get("training") or {})
    available: List[Tuple[str, Mapping[str, Any]]] = []
    for stage_name in ("measured_pretrain", "measured_adapter"):
        weak = dict(
            dict(training.get(stage_name) or {}).get(
                "state_pair_transport_weak"
            ) or {}
        )
        if weak and weak.get("status") != "stage_skipped":
            available.append((stage_name, weak))
    if not available:
        value = {
            "schema": "v8_development_selection_eligibility_v1",
            "selection_eligible": False,
            "reason": "missing_development_validation_weak_negative_control",
            "relational_weak_branch": True,
            "weak_validation_gate_required": True,
            "weak_supervision_pair_id": pair_id,
        }
    else:
        stage_name, weak = available[-1]
        passed = bool(
            weak.get("relational_state_pair_supervision_enabled") is True
            and weak.get("weak_validation_gate_passed") is True
            and weak.get("weak_branch_eligible_for_selection") is True
        )
        value = {
            "schema": "v8_development_selection_eligibility_v1",
            "selection_eligible": passed,
            "reason": (
                "development_validation_weak_negative_control_passed"
                if passed
                else "development_validation_weak_negative_control_failed_or_insufficient"
            ),
            "relational_weak_branch": True,
            "weak_validation_gate_required": True,
            "weak_supervision_pair_id": pair_id,
            "selected_stage": stage_name,
            "weak_stage_sha256": _stable_json_hash(weak),
        }
    value["eligibility_sha256"] = _stable_json_hash(value)
    return value


def _upgrade_v7_artifacts_to_v8(
    report: Dict[str, Any],
    *,
    profile: V8FeedbackTrainingProfile,
    config: Mapping[str, Any],
    contexts: Mapping[str, V8FeedbackSystemContext],
    source_audit: Mapping[str, Any],
    state_pair_budget_selection: Mapping[str, Any],
) -> Dict[str, Any]:
    checkpoint_path = Path(report["checkpoint"]["path"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != v7_training.CHECKPOINT_SCHEMA:
        raise ValueError("V7 reuse seam emitted an unexpected checkpoint schema")
    split_hashes = split_hash_by_system(contexts)
    state_pair_by_system: Dict[str, Any] = {}
    for system_type, context in contexts.items():
        pair_set = context.state_pair_transport_set
        if pair_set is None:
            raise ValueError(f"{system_type} state-pair transport set is missing")
        pair_set.validate()
        validation_pair_set = context.state_pair_validation_set
        if validation_pair_set is None:
            raise ValueError(f"{system_type} validation state-pair set is missing")
        validation_pair_set.validate()
        state_pair_by_system[system_type] = {
            "pair_set_sha256": _stable_json_hash(asdict(pair_set)),
            "pair_count": len(pair_set.pairs),
            "source_sample_count": int(pair_set.source_sample_count),
            "included_split_names": list(pair_set.included_split_names),
            "locked_test_accessed": bool(pair_set.locked_test_accessed),
            "locked_test_metric_inheritance_allowed": bool(
                pair_set.locked_test_metric_inheritance_allowed
            ),
            "audit": asdict(pair_set.audit),
            "validation_pair_set_sha256": _stable_json_hash(
                asdict(validation_pair_set)
            ),
            "validation_pair_count": len(validation_pair_set.pairs),
            "validation_source_sample_count": int(
                validation_pair_set.source_sample_count
            ),
            "validation_included_split_names": list(
                validation_pair_set.included_split_names
            ),
            "validation_locked_test_accessed": bool(
                validation_pair_set.locked_test_accessed
            ),
            "validation_audit": asdict(validation_pair_set.audit),
        }
    state_pair_transport_contract = {
        "schema": "v8_state_pair_transport_training_contract_v1",
        "evidence_scope": "weak",
        "pair_builder": copy.deepcopy(
            dict(config["state_pair_transport_weak_contract"]["pair_builder"])
        ),
        "pair_builder_sha256": _stable_json_hash(
            config["state_pair_transport_weak_contract"]["pair_builder"]
        ),
        "by_system": state_pair_by_system,
        "locked_test_accessed": False,
        "real_action_supervision_mask": 0,
        "real_jacobian_supervision_mask": 0,
        "counts_as_real_causality_evidence": False,
    }
    state_pair_transport_contract["contract_sha256"] = _stable_json_hash(
        state_pair_transport_contract
    )
    measured_audit = _measured_file_audit(config)
    weak_supervision_comparison = _weak_supervision_comparison_contract(
        config, profile
    )
    selection_eligibility = development_selection_eligibility(report)
    paired_groups = set().union(*(
        set(context.paired_episode_group_keys) for context in contexts.values()
    ))
    uncertainty = build_uncertainty_state(
        config,
        paired_episode_group_count=len(paired_groups),
    )
    real_action_evidence_count = int(sum(
        int(context.measured_bundle_audit.get(
            "real_action_supervision_count_by_system", {}
        ).get(system_type, 0))
        for system_type, context in contexts.items()
    ))
    causal_context_gate = build_causal_context_gate(
        config,
        real_paired_or_signed_probe_count=real_action_evidence_count,
    )
    locked_state = {
        "schema": LOCKED_FINAL_STATE_SCHEMA,
        "status": "sealed_not_accessed",
        "access_count": 0,
        "maximum_access_count": 1,
        "declaration_required": True,
        "allowed_for_model_selection": False,
        "declaration": copy.deepcopy(
            next(iter(contexts.values())).locked_final_declaration
        ),
        "declaration_sha256": _stable_json_hash(
            next(iter(contexts.values())).locked_final_declaration
        ),
    }
    contract = {
        "config_schema": CONFIG_SCHEMA,
        "config_sha256": str(config["_config_sha256"]),
        "ablation_id": profile.ablation_id,
        "synthetic_plant_families": copy.deepcopy(
            dict(config["synthetic_plant_families"])
        ),
        "real_loss_contract": copy.deepcopy(dict(config["real_loss_contract"])),
        "real_exposure_contract": copy.deepcopy(
            dict(config["real_exposure_contract"])
        ),
        "split_hash_by_system": split_hashes,
        "measured_file_audit": measured_audit,
        "source_tree_audit": copy.deepcopy(dict(source_audit)),
        "model_audit_sha256": _stable_json_hash(checkpoint.get("audit") or {}),
        "measured_bundle_audit": copy.deepcopy(
            next(iter(contexts.values())).measured_bundle_audit
        ),
        "state_pair_transport_weak": state_pair_transport_contract,
        "state_pair_budget_selection": copy.deepcopy(
            dict(state_pair_budget_selection)
        ),
        "weak_supervision_paired_comparison": weak_supervision_comparison,
        "development_selection_eligibility": selection_eligibility,
        "synthetic_command_feature_support": {
            "schema": "v8_synthetic_command_feature_support_v1",
            "feature_count": 18,
            "feature_support_mask": [1] * 15 + [0] * 3,
            "unsupported_feature_names": [
                "digital_delay",
                "digital_width",
                "analog_delay",
            ],
            "unsupported_nonzero_timing_is_ood": True,
            "cannot_increase_action_confidence": True,
        },
        "causal_context_gate": causal_context_gate,
        "development_selection_only": True,
        "locked_final_test": locked_state,
    }
    contract["contract_audit_sha256"] = _stable_json_hash(contract)
    checkpoint.update({
        "schema": CHECKPOINT_SCHEMA,
        "training_profile": asdict(profile),
        "v8_training_contract": contract,
        "source_tree_audit": copy.deepcopy(dict(source_audit)),
        "config_sha256": str(config["_config_sha256"]),
        "split_hash_by_system": split_hashes,
        "measured_audit_sha256": measured_audit["audit_sha256"],
        "model_audit_sha256": contract["model_audit_sha256"],
        "contract_audit_sha256": contract["contract_audit_sha256"],
        "uncertainty_calibration": uncertainty,
        "causal_context_gate": causal_context_gate,
        "state_pair_transport_weak": state_pair_transport_contract,
        "state_pair_budget_selection": copy.deepcopy(
            dict(state_pair_budget_selection)
        ),
        "weak_supervision_paired_comparison": weak_supervision_comparison,
        "development_selection_eligibility": selection_eligibility,
        "locked_final_test": locked_state,
        "selection_scope": "development_only",
        "frozen_inference": True,
        "online_training_allowed": False,
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    })
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint_path)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    report.update({
        "schema": TRAINING_REPORT_SCHEMA,
        "profile": asdict(profile),
        "v8_training_contract": contract,
        "source_tree_audit": copy.deepcopy(dict(source_audit)),
        "config_sha256": str(config["_config_sha256"]),
        "split_hash_by_system": split_hashes,
        "measured_audit_sha256": measured_audit["audit_sha256"],
        "model_audit_sha256": contract["model_audit_sha256"],
        "contract_audit_sha256": contract["contract_audit_sha256"],
        "uncertainty_calibration": uncertainty,
        "causal_context_gate": causal_context_gate,
        "state_pair_transport_weak": state_pair_transport_contract,
        "state_pair_budget_selection": copy.deepcopy(
            dict(state_pair_budget_selection)
        ),
        "weak_supervision_paired_comparison": weak_supervision_comparison,
        "development_selection_eligibility": selection_eligibility,
        "locked_final_test": locked_state,
        "selection_scope": "development_only",
        "locked_final_metrics_present": False,
        "checkpoint": {
            **dict(report["checkpoint"]),
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "schema": CHECKPOINT_SCHEMA,
            "frozen_inference": True,
        },
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "evidence_limit": (
            "Development selection uses measured validation plus declared synthetic "
            "development plants only. Locked final rows remain sealed; action "
            "confidence abstains until grouped real paired episodes calibrate it."
        ),
    })
    report.pop("report_hash", None)
    report["report_hash"] = _stable_json_hash(report)
    report_path = checkpoint_path.parent / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def train_v8_profile(
    *,
    profile: V8FeedbackTrainingProfile,
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    contexts: Mapping[str, V8FeedbackSystemContext],
    output_dir: str | Path,
    seed: int,
    device: "torch.device",
) -> Dict[str, Any]:
    source_audit = collect_source_tree_audit(config["_config_path"])
    runtime = copy.deepcopy(dict(runtime_config))
    runtime["loss_weights"]["real_count_nb2"] = 1.0
    rho_estimator_contract = dict(config["rho_estimator_contract"])
    runtime["loss_weights"]["real_rho_soft_teacher_maximum_fraction"] = float(
        rho_estimator_contract[
            "maximum_weighted_contribution_fraction_of_real_count_nb2"
        ]
    )
    runtime["loss_weights"]["real_rho_shared_backbone_gradient_scale"] = float(
        rho_estimator_contract["shared_backbone_gradient_scale_on_real_rows"]
    )
    runtime["_v8_train_families"] = list(
        config["synthetic_plant_families"]["train"]
    )
    runtime["_v8_development_families"] = list(
        config["synthetic_plant_families"]["development"]
    )
    runtime["_v8_operator_curated_corpus"] = copy.deepcopy(
        dict(config["operator_curated_corpus"])
    )
    runtime["_v8_data_config"] = copy.deepcopy(dict(config["data"]))
    runtime["measured_audit_waiver"] = copy.deepcopy(
        dict(config.get("measured_audit_waiver") or {})
    )
    state_pair_contract = copy.deepcopy(
        dict(config["state_pair_transport_weak_contract"])
    )
    state_pair_contract["supervision_mode"] = (
        "relational_state_pair_transport"
        if profile.weak_state_pair_supervision_enabled
        else "endpoint_identity_compute_matched_control"
    )
    state_pair_contract["compute_matched"] = bool(
        profile.weak_supervision_pair_id
    )
    budget_phase = str(
        config.get("_state_pair_budget_phase_override")
        or profile.state_pair_budget_phase
    )
    if budget_phase not in {"screen", "full"}:
        raise ValueError("state-pair budget phase must be screen or full")
    budget_key = f"{budget_phase}_maximum_batches_per_system"
    state_pair_contract["active_maximum_batches_per_system"] = int(
        state_pair_contract[budget_key]
    )
    validation_budget_key = (
        f"{budget_phase}_validation_maximum_pair_rows_per_system"
    )
    state_pair_contract["active_validation_maximum_pair_rows_per_system"] = int(
        state_pair_contract[validation_budget_key]
    )
    state_pair_contract["active_budget_profile"] = budget_phase
    if bool(config.get("_smoke_mode", False)):
        smoke_plan = dict(config["execution_plans"]["contract_smoke"])
        state_pair_contract["active_maximum_batches_per_system"] = min(
            int(state_pair_contract["active_maximum_batches_per_system"]),
            int(smoke_plan.get("state_pair_maximum_batches_per_system", 1)),
        )
        state_pair_contract["active_budget_profile"] = "contract_smoke"
        state_pair_contract["active_validation_maximum_pair_rows_per_system"] = int(
            smoke_plan.get("state_pair_validation_maximum_pair_rows_per_system", 32)
        )
    runtime["_v8_state_pair_contract"] = state_pair_contract
    state_pair_budget_selection = {
        "schema": "v8_state_pair_budget_selection_v1",
        "requested_phase": budget_phase,
        "active_budget_profile": state_pair_contract["active_budget_profile"],
        "maximum_batches_per_system": int(
            state_pair_contract["active_maximum_batches_per_system"]
        ),
        "validation_maximum_pair_rows_per_system": int(
            state_pair_contract["active_validation_maximum_pair_rows_per_system"]
        ),
        "budget_scope": "per_system",
        "smoke_override_applied": bool(config.get("_smoke_mode", False)),
        "supervision_mode": state_pair_contract["supervision_mode"],
        "compute_matched": bool(state_pair_contract["compute_matched"]),
    }
    state_pair_budget_selection["selection_sha256"] = _stable_json_hash(
        state_pair_budget_selection
    )
    runtime["_v8_causal_negative_control_contract"] = copy.deepcopy(
        dict(config["causal_negative_control_contract"])
    )
    with v8_v7_reuse_seam(
        development_families=config["synthetic_plant_families"]["development"]
    ):
        report = v7_training.train_feedback_profile(
            profile=profile,
            config=runtime,
            contexts=contexts,
            output_dir=output_dir,
            seed=int(seed),
            device=device,
        )
    return _upgrade_v7_artifacts_to_v8(
        report,
        profile=profile,
        config=config,
        contexts=contexts,
        source_audit=source_audit,
        state_pair_budget_selection=state_pair_budget_selection,
    )


def contract_smoke(config: Mapping[str, Any]) -> Dict[str, Any]:
    base_config, profiles = resolve_v8_profiles(config, smoke=True)
    result = {
        "schema": "v8_feedback_contract_smoke_v1",
        "config_sha256": str(config["_config_sha256"]),
        "profile_count": len(profiles),
        "profiles": [profile.name for profile in profiles],
        "ablations": list(REQUIRED_ABLATIONS),
        "synthetic_plant_families": copy.deepcopy(
            dict(config["synthetic_plant_families"])
        ),
        "real_count_nb2_weight": float(base_config["loss_weights"]["real_count_nb2"]),
        "development_selection_only": True,
        "locked_final_test_sealed": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    result["smoke_hash"] = _stable_json_hash(result)
    return result


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seed_csv(value: Optional[str], defaults: Sequence[int]) -> List[int]:
    if value is None:
        return [int(item) for item in defaults]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--profiles", help="Comma-separated V8 profile names")
    parser.add_argument("--ablation", choices=REQUIRED_ABLATIONS)
    parser.add_argument("--seeds", help="Comma-separated integer seeds")
    parser.add_argument(
        "--state-pair-budget-phase",
        choices=("screen", "full"),
        help="Explicit weak state-pair screen/full budget phase",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="ml_recon/reports/physics_hybrid_v8/feedback_training",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--contract-smoke", action="store_true")
    parser.add_argument("--allow-server-profile", action="store_true")
    parser.add_argument("--reevaluate-checkpoint", action="append")
    parser.add_argument(
        "--evaluation-scope",
        choices=("development", "locked_final"),
        default="development",
    )
    parser.add_argument("--locked-declaration")
    parser.add_argument(
        "--weak-on-report",
        help="Weak relational training_report.json for a paired comparison",
    )
    parser.add_argument(
        "--weak-off-control-report",
        help="Same-seed compute-matched weak-off training_report.json",
    )
    parser.add_argument(
        "--weak-comparison-output",
        help="Output path for the hash-bound paired comparison artifact",
    )
    args = parser.parse_args(argv)

    config = load_v8_config(args.config)
    config["_smoke_mode"] = bool(args.smoke)
    config["_state_pair_budget_phase_override"] = args.state_pair_budget_phase
    weak_comparison_arguments = (
        args.weak_on_report,
        args.weak_off_control_report,
        args.weak_comparison_output,
    )
    if any(value is not None for value in weak_comparison_arguments):
        if not all(value is not None for value in weak_comparison_arguments):
            raise ValueError(
                "weak comparison requires --weak-on-report, "
                "--weak-off-control-report and --weak-comparison-output"
            )
        artifact = build_weak_supervision_paired_comparison(
            args.weak_on_report,
            args.weak_off_control_report,
            config=config,
            output_path=args.weak_comparison_output,
        )
        print(json.dumps({
            "event": "v8_weak_supervision_paired_comparison_complete",
            "seed": artifact["seed"],
            "status": artifact["status"],
            "artifact_sha256": artifact["artifact_sha256"],
            "promotion_ready": False,
            "hardware_action_authorized": False,
        }, sort_keys=True), flush=True)
        return 0
    if args.contract_smoke:
        print(json.dumps(contract_smoke(config), sort_keys=True), flush=True)
        return 0
    if args.reevaluate_checkpoint:
        from .evaluate_feedback_models import evaluate_v8_checkpoint

        for checkpoint_path in args.reevaluate_checkpoint:
            evaluate_v8_checkpoint(
                checkpoint_path,
                config=config,
                output_dir=args.output_dir,
                device_name=args.device,
                scope=args.evaluation_scope,
                locked_declaration_path=args.locked_declaration,
            )
        return 0

    runtime_config, profiles = resolve_v8_profiles(
        config,
        _parse_csv(args.profiles),
        ablation_id=args.ablation,
        smoke=bool(args.smoke),
    )
    blocked = [
        profile.name
        for profile in profiles
        if not profile.execution_tier.startswith("local")
    ]
    if blocked and not args.allow_server_profile:
        raise ValueError(f"server profiles require --allow-server-profile: {blocked}")
    device = v7_training._resolve_device(args.device)
    contexts = load_v8_contexts(config, runtime_config)
    seeds = _parse_seed_csv(args.seeds, config["execution_plans"]["local_3seed"]["seeds"])
    reports: List[Dict[str, Any]] = []
    for profile in profiles:
        for seed in seeds:
            reports.append(train_v8_profile(
                profile=profile,
                config=config,
                runtime_config=runtime_config,
                contexts=contexts,
                output_dir=args.output_dir,
                seed=int(seed),
                device=device,
            ))
    leaderboard = {
        "schema": "v8_feedback_development_leaderboard_v1",
        "selection_scope": "development_only",
        "locked_final_metrics_present": False,
        "rows": [{
            "profile": str(report["profile"]["name"]),
            "ablation_id": str(report["profile"]["ablation_id"]),
            "seed": int(report["seed"]),
            "selection_eligible": bool(
                report["development_selection_eligibility"]["selection_eligible"]
            ),
            "selection_ineligibility_reason": (
                None
                if report["development_selection_eligibility"]["selection_eligible"]
                else report["development_selection_eligibility"]["reason"]
            ),
            "selection_eligibility_sha256": report[
                "development_selection_eligibility"
            ]["eligibility_sha256"],
            "joint_selection_score": (
                float(report["selection"]["joint_selection_score"])
                if report["development_selection_eligibility"]["selection_eligible"]
                else None
            ),
            "diagnostic_joint_selection_score": float(
                report["selection"]["joint_selection_score"]
            ),
            "checkpoint": str(report["checkpoint"]["path"]),
        } for report in reports],
        "winner_selection_rule": (
            "rank only selection_eligible rows; relational weak branches require "
            "development validation shuffle pass and non-relational controls are diagnostic"
        ),
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    leaderboard["leaderboard_hash"] = _stable_json_hash(leaderboard)
    (output_path / "development_leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "v8_feedback_development_screen_complete",
        "report_count": len(reports),
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG_PATH",
    "LOCKED_FINAL_STATE_SCHEMA",
    "SOURCE_AUDIT_SCHEMA",
    "TRAINING_REPORT_SCHEMA",
    "UNCERTAINTY_SCHEMA",
    "V8FeedbackSystemContext",
    "V8FeedbackTrainingProfile",
    "V8RealActionSupervisionBatch",
    "WEAK_PAIRED_COMPARISON_CONTRACT_SCHEMA",
    "WEAK_PAIRED_COMPARISON_GATE_SCHEMA",
    "WEAK_PAIRED_COMPARISON_SCHEMA",
    "build_weak_supervision_paired_comparison",
    "build_uncertainty_state",
    "collect_source_tree_audit",
    "contract_smoke",
    "empty_real_action_supervision",
    "load_v8_config",
    "load_v8_contexts",
    "main",
    "resolve_v8_profiles",
    "split_hash_by_system",
    "train_v8_profile",
    "validate_v8_config",
    "v8_v7_reuse_seam",
    "weighted_real_auxiliary_mean",
]
