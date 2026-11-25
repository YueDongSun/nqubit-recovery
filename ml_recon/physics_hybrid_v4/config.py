"""Validated configuration and capacity planning for Physics Hybrid V4."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .architecture import (
    AUTOMATIC_CAPACITY_MAX_PARAMETERS,
    AUTOMATIC_CAPACITY_MIN_PARAMETERS,
    validate_architecture_config,
)


MODEL_IDS = (
    "classifier_3q",
    "feedback_3q",
    "classifier_4q",
    "feedback_4q",
    "classifier_n",
    "feedback_n",
)
MODEL_ID_SET = frozenset(MODEL_IDS)
CAPACITY_MIN_PARAMETERS = AUTOMATIC_CAPACITY_MIN_PARAMETERS
CAPACITY_MAX_PARAMETERS = AUTOMATIC_CAPACITY_MAX_PARAMETERS
SOURCE_NAMES = ("clean_ideal", "synthetic_noisy", "real")


@dataclass(frozen=True)
class PhysicsHybridV4Config:
    path: Path
    payload: Dict[str, Any]
    sha256: str
    smoke: bool = False

    @property
    def output_root(self) -> Path:
        return Path(str(self.payload["output_root"]))

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self.payload["models"].keys())

    def model(self, model_id: str) -> Dict[str, Any]:
        if model_id not in MODEL_ID_SET:
            raise ValueError(f"unknown Physics Hybrid V4 model: {model_id}")
        return copy.deepcopy(dict(self.payload["models"][model_id]))

    def to_json_dict(self) -> Dict[str, Any]:
        result = copy.deepcopy(self.payload)
        result["resolved_config_sha256"] = self.sha256
        result["smoke"] = bool(self.smoke)
        return result


def load_v4_config(
    path: str | Path,
    *,
    smoke: bool = False,
    overrides: Sequence[str] = (),
) -> PhysicsHybridV4Config:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if overrides:
        payload = _apply_dot_overrides(payload, overrides)
    _validate_config(payload)
    if smoke:
        payload = _apply_smoke_overrides(payload)
    hash_payload = _payload_for_hash(payload)
    canonical = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return PhysicsHybridV4Config(config_path, payload, digest, smoke=smoke)


def parse_model_selection(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None or value == "all":
        return MODEL_IDS
    if isinstance(value, str):
        requested = [item.strip() for item in value.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in value if str(item).strip()]
    unknown = sorted(set(requested) - MODEL_ID_SET)
    if unknown:
        raise ValueError(f"unknown models: {unknown}; expected any of {list(MODEL_IDS)}")
    if not requested:
        raise ValueError("at least one model must be selected")
    return tuple(dict.fromkeys(requested))


def capacity_target(config: PhysicsHybridV4Config, model_id: str, profile: str | None = None) -> int:
    model_config = config.model(model_id)
    profile_name = str(profile or model_config.get("capacity_profile") or "medium_20m")
    profiles = config.payload["capacity_profiles"]
    if profile_name not in profiles:
        raise ValueError(f"capacity profile {profile_name!r} is not defined")
    target = int(profiles[profile_name]["target_parameters"])
    if not CAPACITY_MIN_PARAMETERS <= target <= CAPACITY_MAX_PARAMETERS:
        raise ValueError(
            f"target_parameters must be in [{CAPACITY_MIN_PARAMETERS}, {CAPACITY_MAX_PARAMETERS}], got {target}"
        )
    return target


def source_probabilities(config: PhysicsHybridV4Config, *, real_available: bool) -> Dict[str, float]:
    ratios = config.payload["data"]["source_identity_ratio"]
    weights = {name: float(ratios[name]) for name in SOURCE_NAMES}
    if not real_available:
        weights["real"] = 0.0
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("source identity ratio has no positive entries")
    return {name: value / total for name, value in weights.items()}


def stage_order(requested: str) -> tuple[str, ...]:
    valid = ("generate", "train", "evaluate", "report")
    if requested == "all":
        return valid
    if requested not in valid:
        raise ValueError(f"stage must be one of {valid + ('all',)}")
    return (requested,)


def _validate_config(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "PhysicsHybridV4Config":
        raise ValueError("config schema must be PhysicsHybridV4Config")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("PhysicsHybridV4Config schema_version must be 1")
    if not payload.get("output_root"):
        raise ValueError("output_root is required")
    models = payload.get("models")
    if not isinstance(models, Mapping) or set(models) != MODEL_ID_SET:
        raise ValueError(f"models must define exactly {list(MODEL_IDS)}")
    profiles = payload.get("capacity_profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("capacity_profiles must be a non-empty mapping")
    targets = []
    for name, record in profiles.items():
        target = int(record.get("target_parameters", 0))
        if not CAPACITY_MIN_PARAMETERS <= target <= CAPACITY_MAX_PARAMETERS:
            raise ValueError(f"capacity profile {name} is outside the 1M-1.2B safety interval")
        targets.append(target)
    if min(targets) > CAPACITY_MIN_PARAMETERS or max(targets) < 200_000_000:
        raise ValueError("capacity profiles must expose both the 1M and 200M boundaries")
    validate_architecture_config(payload, MODEL_IDS)
    data = payload.get("data", {})
    ratio = data.get("source_identity_ratio", {})
    # The default corpus mix is clean_ideal:synthetic_noisy:real = 1000:10000:1.
    # A 2026-07-24 three-type data spec produces separate standard (clean-only)
    # and synthetic-noise (noisy-only) datasets by overriding the ratio, so the
    # mix is no longer required to be the single fixed 1000:10000:1 value. Only
    # enforce that all three source names are present as non-negative integers
    # and that at least one source is non-zero (otherwise nothing is generated).
    ratio_values = tuple(int(ratio.get(name, -1)) for name in SOURCE_NAMES)
    if any(value < 0 for value in ratio_values):
        raise ValueError("source_identity_ratio must have non-negative integer values")
    if sum(ratio_values) == 0:
        raise ValueError("source_identity_ratio must select at least one source")
    training = payload.get("training", {})
    sampling_strategy = str(training.get("sampling_strategy", "corpus_ratio"))
    if sampling_strategy not in {
        "corpus_ratio", "domain_compensated", "v5_streaming_stratified"
    }:
        raise ValueError(
            "training.sampling_strategy is not supported"
        )
    if sampling_strategy == "domain_compensated":
        domain_batch = dict(training.get("domain_batch") or {})
        clean_fraction = float(domain_batch.get("clean_fraction_of_synthetic", -1.0))
        if not 0.0 <= clean_fraction <= 1.0:
            raise ValueError(
                "training.domain_batch.clean_fraction_of_synthetic must be in [0, 1]"
            )
        if int(domain_batch.get("real_rows_per_batch", -1)) < 0:
            raise ValueError(
                "training.domain_batch.real_rows_per_batch must be non-negative"
            )
    domain_objective = dict(training.get("domain_objective") or {})
    if bool(domain_objective.get("enabled", False)):
        weights = dict(domain_objective.get("weights") or {})
        if set(weights) != {"clean", "noisy", "real"}:
            raise ValueError(
                "training.domain_objective.weights must define clean, noisy, and real"
            )
        values = [float(weights[name]) for name in ("clean", "noisy", "real")]
        if any(value < 0.0 for value in values) or not math.isclose(
            sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "training.domain_objective.weights must be non-negative and sum to 1"
            )
    phase = payload.get("phase", {})
    if float(phase.get("action_limit_rad", 0.0)) != 0.20:
        raise ValueError("phase.action_limit_rad must remain 0.20 for the first V4 hardware stage")
    probe_values = tuple(float(value) for value in phase.get("probe_amplitudes_rad", ()))
    if probe_values != (0.02, 0.05, 0.08):
        raise ValueError("phase.probe_amplitudes_rad must be [0.02, 0.05, 0.08]")
    if float(phase.get("minimum_fisher_information", 0.0)) <= 0.0:
        raise ValueError("phase.minimum_fisher_information must be positive")
    if float(phase.get("maximum_posterior_std_rad", 0.0)) <= 0.0:
        raise ValueError("phase.maximum_posterior_std_rad must be positive")
    crosstalk = payload.get("crosstalk", {})
    maximum = float(crosstalk.get("maximum_transfer_probability", 0.0))
    if maximum <= 0.0 or maximum > 0.15:
        raise ValueError("crosstalk.maximum_transfer_probability must be in (0, 0.15]")
    channel_weights = tuple(float(value) for value in payload.get("noise", {}).get("count_channel_weights", ()))
    if len(channel_weights) != 6 or any(value <= 0.0 for value in channel_weights):
        raise ValueError("noise.count_channel_weights must contain six positive values")
    evaluation = payload.get("evaluation", {})
    if tuple(int(value) for value in evaluation.get("feedback_horizons", ())) != (1, 2, 3, 5):
        raise ValueError("evaluation.feedback_horizons must be [1, 2, 3, 5]")
    for key in (
        "classifier_minimum_metric",
        "feedback_minimum_success",
        "maximum_ece",
        "feedback_minimum_localization",
        "feedback_minimum_sign_accuracy",
        "feedback_maximum_off_target_energy",
        "minimum_jacobian_cosine",
    ):
        value = float(evaluation.get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"evaluation.{key} must be in [0, 1]")
    logit_temperature = evaluation.get("family_logit_temperature", {})
    if not 0.0 < float(logit_temperature.get("minimum", 0.0)) <= float(
        logit_temperature.get("maximum", 0.0)
    ):
        raise ValueError("evaluation.family_logit_temperature range is invalid")
    if int(logit_temperature.get("steps", 0)) < 2:
        raise ValueError("evaluation.family_logit_temperature.steps must be at least two")


def _apply_smoke_overrides(payload: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(dict(payload))
    smoke = dict(resolved.get("smoke_overrides") or {})
    resolved["training"].update(smoke.get("training") or {})
    resolved["data"].update(smoke.get("data") or {})
    resolved["evaluation"].update(smoke.get("evaluation") or {})
    smoke_profile = str(smoke.get("capacity_profile", "smoke_sub_million"))
    if smoke_profile not in resolved["capacity_profiles"]:
        resolved["capacity_profiles"][smoke_profile] = {
            "target_parameters": CAPACITY_MIN_PARAMETERS,
            "smoke_width_override": 64,
            "smoke_layers_override": 1,
        }
    for model in resolved["models"].values():
        model["capacity_profile"] = smoke_profile
        if "epochs" in resolved["training"]:
            model["epochs"] = int(resolved["training"]["epochs"])
        if "batch_size" in resolved["training"]:
            model["batch_size"] = int(resolved["training"]["batch_size"])
        if smoke.get("data", {}).get("sample_counts"):
            model["sample_counts"] = copy.deepcopy(smoke["data"]["sample_counts"])
    return resolved


def _apply_dot_overrides(payload: Mapping[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    resolved = copy.deepcopy(dict(payload))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must use dotted.path=JSON_VALUE syntax: {item!r}")
        dotted_path, raw_value = item.split("=", 1)
        keys = [key for key in dotted_path.split(".") if key]
        if not keys:
            raise ValueError("override path must not be empty")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        cursor: Dict[str, Any] = resolved
        for key in keys[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                raise ValueError(f"override path {dotted_path!r} does not resolve through mapping key {key!r}")
            cursor = child
        if keys[-1] not in cursor:
            raise ValueError(f"override path {dotted_path!r} does not exist in the base config")
        cursor[keys[-1]] = value
    return resolved


def _payload_for_hash(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep the default-auto profile compatible with pre-profile V4 artifacts.

    Architecture profiles are inert declarations while ``auto`` is selected.
    Excluding that declaration block reproduces the hash of the original V4
    config, so existing auto-capacity datasets and checkpoints remain resumable.
    Any selected non-auto global or per-model profile is hashed in full.
    """

    resolved = copy.deepcopy(dict(payload))
    training = resolved.get("training")
    if (
        isinstance(training, dict)
        and int(training.get("intra_epoch_checkpoint_interval_updates", 0)) == 0
    ):
        training.pop("intra_epoch_checkpoint_interval_updates", None)
    architecture = resolved.get("architecture", {})
    default_profile = str(architecture.get("default_profile", "auto"))
    per_model_profiles = {
        str(record.get("architecture_profile", "auto"))
        for record in resolved.get("models", {}).values()
        if isinstance(record, Mapping)
    }
    if default_profile == "auto" and per_model_profiles <= {"auto"}:
        resolved.pop("architecture", None)
    return resolved
