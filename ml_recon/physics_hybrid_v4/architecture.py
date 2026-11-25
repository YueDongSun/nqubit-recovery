"""Architecture topology, capacity planning, and hardware profile resolution.

This module is intentionally free of PyTorch imports.  It is the canonical
location for width/depth/backbone decisions; ``models.py`` only implements the
resolved neural modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence


MODEL_INPUT_DEFAULTS = {
    "operation_feature_dim": 32,
    "cell_feature_dim": 12,
    "global_feature_dim": 24,
    "num_families": 6,
    "num_family_groups": 3,
    "num_mechanisms": 11,
}
COUNT_CHANNEL_COUNT = 6
AUTOMATIC_CAPACITY_MIN_PARAMETERS = 1_000_000
AUTOMATIC_CAPACITY_MAX_PARAMETERS = 1_200_000_000
EXPLICIT_CAPACITY_MAX_PARAMETERS = 1_200_000_000
VALID_BACKBONES = frozenset({"auto", "transformer", "linear_set"})
VALID_DECODERS = frozenset({"dct", "grid_spectral"})

MODEL_TOPOLOGY: Dict[str, Dict[str, Any]] = {
    "classifier_3q": {
        "family": "classifier",
        "operation_count": 75,
        "cell_count": 3,
        "default_backbone": "transformer",
    },
    "feedback_3q": {
        "family": "feedback",
        "operation_count": 75,
        "cell_count": 3,
        "default_backbone": "transformer",
    },
    "classifier_4q": {
        "family": "classifier",
        "operation_count": 256,
        "cell_count": 4,
        "default_backbone": "transformer",
    },
    "feedback_4q": {
        "family": "feedback",
        "operation_count": 256,
        "cell_count": 4,
        "default_backbone": "transformer",
    },
    "classifier_n": {
        "family": "classifier",
        "operation_count": "maximum_n",
        "cell_count": 2,
        "default_backbone": "linear_set",
    },
    "feedback_n": {
        "family": "feedback",
        "operation_count": "maximum_n",
        "cell_count": 2,
        "default_backbone": "linear_set",
    },
}

COMMON_CAPACITY_TARGETS = (
    1_000_000,
    5_000_000,
    20_000_000,
    50_000_000,
    100_000_000,
    200_000_000,
)
MICROBATCH_LADDER = (32, 16, 8, 4, 2, 1)

# Explicit shapes are the reproducible search space; measured VRAM, not these
# estimates, decides the release-stable and engineering ceilings.
MODEL_ARCHITECTURE_LADDERS: Dict[str, tuple[tuple[int, int, int, str], ...]] = {
    "classifier_3q": (
        (1024, 20, 261_000_000, "transformer"),
        (1280, 20, 408_000_000, "transformer"),
        (1280, 24, 486_000_000, "transformer"),
        (1536, 20, 587_000_000, "transformer"),
        (1536, 24, 700_000_000, "transformer"),
        (1792, 20, 799_000_000, "transformer"),
        (1792, 24, 953_000_000, "transformer"),
        (2048, 20, 1_045_000_000, "transformer"),
    ),
    "feedback_3q": (
        (1024, 20, 262_000_000, "transformer"),
        (1280, 20, 408_000_000, "transformer"),
        (1280, 24, 487_000_000, "transformer"),
        (1536, 20, 588_000_000, "transformer"),
        (1536, 24, 701_000_000, "transformer"),
        (1792, 20, 800_000_000, "transformer"),
        (1792, 24, 955_000_000, "transformer"),
        (2048, 20, 1_045_000_000, "transformer"),
    ),
    "classifier_4q": (
        (768, 16, 119_000_000, "transformer"),
        (1024, 16, 211_000_000, "transformer"),
        (1024, 20, 261_000_000, "transformer"),
        (1280, 20, 408_000_000, "transformer"),
        (1280, 24, 486_000_000, "transformer"),
        (1536, 20, 587_000_000, "transformer"),
        (1536, 24, 700_000_000, "transformer"),
        (1792, 20, 799_000_000, "transformer"),
        (1792, 24, 953_000_000, "transformer"),
        (2048, 20, 1_045_000_000, "transformer"),
    ),
    "feedback_4q": (
        (1024, 20, 219_000_000, "linear_set"),
        (1280, 20, 343_000_000, "linear_set"),
        (1280, 24, 408_000_000, "linear_set"),
        (1536, 20, 494_000_000, "linear_set"),
        (1536, 24, 588_000_000, "linear_set"),
        (1792, 20, 672_000_000, "linear_set"),
        (1792, 24, 800_000_000, "linear_set"),
        (2048, 20, 877_000_000, "linear_set"),
        (2048, 24, 1_045_000_000, "linear_set"),
    ),
    "classifier_n": (
        (1024, 20, 219_000_000, "linear_set"),
        (1280, 20, 342_000_000, "linear_set"),
        (1280, 24, 408_000_000, "linear_set"),
        (1536, 20, 492_000_000, "linear_set"),
        (1536, 24, 587_000_000, "linear_set"),
        (1792, 20, 670_000_000, "linear_set"),
        (1792, 24, 799_000_000, "linear_set"),
        (2048, 20, 875_000_000, "linear_set"),
        (2048, 24, 1_043_000_000, "linear_set"),
    ),
    "feedback_n": (
        (1024, 20, 212_000_000, "linear_set"),
        (1280, 20, 331_000_000, "linear_set"),
        (1280, 24, 397_000_000, "linear_set"),
        (1536, 20, 477_000_000, "linear_set"),
        (1536, 24, 571_000_000, "linear_set"),
        (1792, 20, 649_000_000, "linear_set"),
        (1792, 24, 778_000_000, "linear_set"),
        (2048, 20, 848_000_000, "linear_set"),
        (2048, 24, 1_016_000_000, "linear_set"),
    ),
}


def model_architecture_ladder(model_id: str) -> list[Dict[str, Any]]:
    if model_id not in MODEL_ARCHITECTURE_LADDERS:
        raise ValueError(f"unknown model_id {model_id!r}")
    effective_batch = 128 if model_id.startswith("classifier") else 64
    return [
        {
            "width": width,
            "layers": layers,
            "target_parameters": target,
            "backbone": backbone,
            "maximum_n": 256 if model_id.endswith("_n") else None,
            "microbatch_attempts": list(MICROBATCH_LADDER),
            "effective_batch": effective_batch,
            "activation_checkpointing": True,
        }
        for width, layers, target, backbone in MODEL_ARCHITECTURE_LADDERS[model_id]
    ]


@dataclass(frozen=True)
class CapacityPlan:
    target_parameters: int
    hidden_dim: int
    transformer_layers: int
    attention_heads: int
    feedforward_multiplier: int
    estimated_parameters: int
    relative_error: float
    backbone: str

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelBuildSpec:
    model_id: str
    target_parameters: int
    operation_feature_dim: int = MODEL_INPUT_DEFAULTS["operation_feature_dim"]
    cell_feature_dim: int = MODEL_INPUT_DEFAULTS["cell_feature_dim"]
    global_feature_dim: int = MODEL_INPUT_DEFAULTS["global_feature_dim"]
    num_families: int = MODEL_INPUT_DEFAULTS["num_families"]
    num_family_groups: int = MODEL_INPUT_DEFAULTS["num_family_groups"]
    num_mechanisms: int = MODEL_INPUT_DEFAULTS["num_mechanisms"]
    dropout: float = 0.10
    maximum_n: int = 64
    width_override: Optional[int] = None
    layers_override: Optional[int] = None
    backbone: str = "auto"
    architecture_profile: str = "auto"
    activation_checkpointing: bool = False
    minimum_fisher_information: float = 0.001
    maximum_posterior_std_rad: float = 0.05
    maximum_fisher_condition: float = 10_000.0
    decoder_kind: str = "dct"
    density_head_enabled: bool = True
    fisher_gate_enabled: bool = True
    # V5 physics-math upgrade: opt-in inductive biases (neutral defaults keep
    # the historical six-model shapes byte-identical when disabled).
    coherent_pooling: bool = False
    witness_head_enabled: bool = False
    v5_nuisance_head_enabled: bool = False
    v5_nuisance_dim: int = 4
    v5_temporal_dynamics_enabled: bool = False


@dataclass(frozen=True)
class ArchitectureSelection:
    """One model's resolved entry from a named hardware architecture profile."""

    profile_name: str
    target_parameters: Optional[int] = None
    width: Optional[int] = None
    layers: Optional[int] = None
    backbone: str = "auto"
    activation_checkpointing: Optional[bool] = None
    micro_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    evidence: str = "automatic_capacity_planner"
    release_eligible: bool = False
    notes: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def model_topology(model_id: str, *, maximum_n: int = 64) -> Dict[str, Any]:
    if model_id not in MODEL_TOPOLOGY:
        raise ValueError(f"unknown model_id {model_id!r}")
    result = dict(MODEL_TOPOLOGY[model_id])
    if result["operation_count"] == "maximum_n":
        result["operation_count"] = int(maximum_n)
    return result


def resolved_backbone(spec: ModelBuildSpec) -> str:
    requested = str(spec.backbone or "auto")
    if requested not in VALID_BACKBONES:
        raise ValueError(f"unsupported backbone {requested!r}; expected one of {sorted(VALID_BACKBONES)}")
    if requested == "auto":
        return str(model_topology(spec.model_id, maximum_n=spec.maximum_n)["default_backbone"])
    return requested


def plan_capacity(spec: ModelBuildSpec) -> CapacityPlan:
    """Choose a width/depth pair or validate an explicit hardware profile.

    Automatic and explicit search share the 1M-1.2B safety boundary.  Release
    eligibility is decided by measured VRAM stability, not a 200M proxy cap.
    """

    target = int(spec.target_parameters)
    if spec.decoder_kind not in VALID_DECODERS:
        raise ValueError(
            f"unsupported decoder_kind {spec.decoder_kind!r}; expected one of {sorted(VALID_DECODERS)}"
        )
    explicit = spec.width_override is not None or spec.layers_override is not None
    if explicit:
        if not AUTOMATIC_CAPACITY_MIN_PARAMETERS <= target <= EXPLICIT_CAPACITY_MAX_PARAMETERS:
            raise ValueError(
                "explicit target_parameters must be between 1M and 1.2B"
            )
        candidate_widths = (
            (int(spec.width_override),)
            if spec.width_override is not None
            else (
                96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768,
                896, 1024, 1280, 1536, 1792, 2048,
            )
        )
        candidate_layers = (
            (int(spec.layers_override),)
            if spec.layers_override is not None
            else (2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32)
        )
        _, layers, width = min(
            (
                abs(estimate_parameter_count(spec, width, layers) - target),
                layers,
                width,
            )
            for width in candidate_widths
            for layers in candidate_layers
        )
        if width < 16 or layers < 1:
            raise ValueError("explicit architecture width must be >=16 and layers must be >=1")
        heads = _attention_heads(width)
        estimate = estimate_parameter_count(spec, width, layers)
        return CapacityPlan(
            target_parameters=target,
            hidden_dim=width,
            transformer_layers=layers,
            attention_heads=heads,
            feedforward_multiplier=4,
            estimated_parameters=estimate,
            relative_error=abs(estimate - target) / target,
            backbone=resolved_backbone(spec),
        )
    if not AUTOMATIC_CAPACITY_MIN_PARAMETERS <= target <= AUTOMATIC_CAPACITY_MAX_PARAMETERS:
        raise ValueError("automatic target_parameters must be between 1M and 1.2B")
    candidates = []
    for width in (
        96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896,
        1024, 1280, 1536, 1792, 2048,
    ):
        for layers in (2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32):
            estimate = estimate_parameter_count(spec, width, layers)
            candidates.append((abs(estimate - target), layers, width, estimate))
    _, layers, width, estimate = min(candidates)
    return CapacityPlan(
        target_parameters=target,
        hidden_dim=width,
        transformer_layers=layers,
        attention_heads=_attention_heads(width),
        feedforward_multiplier=4,
        estimated_parameters=estimate,
        relative_error=abs(estimate - target) / target,
        backbone=resolved_backbone(spec),
    )


def estimate_parameter_count(spec: ModelBuildSpec, hidden_dim: int, layers: int) -> int:
    hidden = int(hidden_dim)
    backbone_name = resolved_backbone(spec)
    if spec.model_id == "feedback_n":
        input_projection = (spec.operation_feature_dim + 3) * hidden
        linear_set_layers = int(layers) * (10 * hidden * hidden + 9 * hidden)
        pool_gate = hidden + 1
        global_projection = spec.global_feature_dim * hidden + hidden
        output_heads = 4 * (hidden + 1) + (2 * hidden + 2)
        utility = 2 * hidden * hidden + 3 * hidden + 2
        return int(input_projection + linear_set_layers + pool_gate + global_projection + output_heads + utility)
    per_layer = 10 * hidden * hidden + 9 * hidden
    if backbone_name == "transformer":
        per_layer = 12 * hidden * hidden + 13 * hidden
    backbone = int(layers) * per_layer
    common = (
        spec.operation_feature_dim * hidden
        + spec.global_feature_dim * hidden
        + 6 * hidden * hidden
        + 20 * hidden
    )
    if spec.model_id.startswith("classifier"):
        heads = hidden * (spec.num_families + spec.num_family_groups + spec.num_mechanisms + 9)
        experts = 5 * (hidden * hidden // 2 + 2 * hidden)
        return int(backbone + common + heads + experts)
    if spec.model_id.startswith("feedback"):
        cell = spec.cell_feature_dim * hidden + 3 * hidden * hidden
        phase_modes = 3 if spec.model_id == "feedback_4q" else 2
        heads = hidden * (phase_modes * 3 + 6)
        return int(backbone + common + cell + heads)
    raise ValueError(f"unknown model_id {spec.model_id!r}")


def resolve_architecture_selection(
    payload: Mapping[str, Any],
    model_id: str,
    *,
    profile_name: Optional[str] = None,
) -> ArchitectureSelection:
    architecture = payload.get("architecture", {})
    selected_name = str(
        profile_name
        or payload.get("models", {}).get(model_id, {}).get("architecture_profile")
        or architecture.get("default_profile")
        or "auto"
    )
    profiles = architecture.get("profiles", {})
    if selected_name == "auto" and selected_name not in profiles:
        return ArchitectureSelection(profile_name="auto")
    if selected_name not in profiles:
        raise ValueError(f"architecture profile {selected_name!r} is not defined")
    profile = profiles[selected_name]
    record = dict(profile.get("models", {}).get(model_id, {}))
    if not record:
        return ArchitectureSelection(
            profile_name=selected_name,
            evidence=str(profile.get("evidence", "automatic_capacity_planner")),
            release_eligible=bool(profile.get("release_eligible", False)),
            notes=str(profile.get("notes", "")),
        )
    return ArchitectureSelection(
        profile_name=selected_name,
        target_parameters=_optional_int(record.get("target_parameters")),
        width=_optional_int(record.get("width")),
        layers=_optional_int(record.get("layers")),
        backbone=str(record.get("backbone", "auto")),
        activation_checkpointing=_optional_bool(record.get("activation_checkpointing")),
        micro_batch_size=_optional_int(record.get("micro_batch_size")),
        gradient_accumulation_steps=_optional_int(record.get("gradient_accumulation_steps")),
        evidence=str(record.get("evidence", profile.get("evidence", "unspecified"))),
        release_eligible=bool(record.get("release_eligible", profile.get("release_eligible", False))),
        notes=str(record.get("notes", profile.get("notes", ""))),
    )


def validate_architecture_config(payload: Mapping[str, Any], model_ids: Sequence[str]) -> None:
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("architecture must be a mapping")
    profiles = architecture.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("architecture.profiles must be a non-empty mapping")
    default_profile = str(architecture.get("default_profile", ""))
    if default_profile not in profiles:
        raise ValueError("architecture.default_profile must name a defined profile")
    known_models = set(model_ids)
    for model_id in model_ids:
        requested = payload.get("models", {}).get(model_id, {}).get("architecture_profile")
        if requested is not None and str(requested) not in profiles:
            raise ValueError(
                f"models.{model_id}.architecture_profile names undefined profile {requested!r}"
            )
    for profile_name, raw_profile in profiles.items():
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"architecture profile {profile_name!r} must be a mapping")
        records = raw_profile.get("models", {})
        if not isinstance(records, Mapping):
            raise ValueError(f"architecture profile {profile_name!r}.models must be a mapping")
        unknown = set(records) - known_models
        if unknown:
            raise ValueError(f"architecture profile {profile_name!r} has unknown models: {sorted(unknown)}")
        for model_id, raw_record in records.items():
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"architecture profile {profile_name!r}.{model_id} must be a mapping")
            width = _optional_int(raw_record.get("width"))
            layers = _optional_int(raw_record.get("layers"))
            if (width is None) != (layers is None):
                raise ValueError(
                    f"architecture profile {profile_name!r}.{model_id} must define width and layers together"
                )
            if width is not None and (width < 16 or layers is None or layers < 1):
                raise ValueError(f"architecture profile {profile_name!r}.{model_id} has invalid width/layers")
            target = _optional_int(raw_record.get("target_parameters"))
            if width is not None and target is None:
                raise ValueError(
                    f"architecture profile {profile_name!r}.{model_id} must define target_parameters"
                )
            if target is not None and not (
                AUTOMATIC_CAPACITY_MIN_PARAMETERS <= target <= EXPLICIT_CAPACITY_MAX_PARAMETERS
            ):
                raise ValueError(
                    f"architecture profile {profile_name!r}.{model_id} target must be in [1M, 1.2B]"
                )
            if (
                target is not None
                and target > 200_000_000
                and width is None
            ):
                raise ValueError(
                    f"architecture profile {profile_name!r}.{model_id} targets above 200M require width/layers"
                )
            backbone = str(raw_record.get("backbone", "auto"))
            if backbone not in VALID_BACKBONES:
                raise ValueError(
                    f"architecture profile {profile_name!r}.{model_id} has unsupported backbone {backbone!r}"
                )
            for key in ("micro_batch_size", "gradient_accumulation_steps"):
                value = _optional_int(raw_record.get(key))
                if value is not None and value < 1:
                    raise ValueError(f"architecture profile {profile_name!r}.{model_id}.{key} must be positive")


def _attention_heads(hidden_dim: int) -> int:
    for heads in (16, 12, 8, 6, 4, 3, 2, 1):
        if hidden_dim % heads == 0 and hidden_dim // heads >= 16:
            return heads
    return 1


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> Optional[bool]:
    return None if value is None else bool(value)
