"""Contracts for physical coefficient evidence and model roles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PhysicalCoefficientPosteriorV1:
    coefficient_name: str
    scope: str
    value: float
    std: float
    unit: str
    bootstrap_inclusion_probability: float
    sign_stability: float
    heldout_delta_nll: float
    intervention_score: float
    probe_score: float
    identifiable: bool
    confounded_with: Sequence[str]
    provenance: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.coefficient_name or not self.unit or not self.provenance:
            raise ValueError("coefficient name, unit, and provenance are required")
        for name in ("bootstrap_inclusion_probability", "sign_stability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.provenance == "classifier_logit":
            raise ValueError("classifier logits are not physical coefficient evidence")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


MODEL_ROLE_SPECS: Dict[str, Dict[str, Any]] = {
    "feedback_3q": {
        "role_kind": "joint_calibrator",
        "training_sources": ("real", "synthetic"),
        "real_only": False,
    },
    "feedback_4q": {
        "role_kind": "joint_calibrator",
        "training_sources": ("real", "synthetic"),
        "real_only": False,
    },
    "classifier_3q": {
        "role_kind": "real_noise_system_identification",
        "training_sources": ("real",),
        "real_only": True,
    },
    "classifier_4q": {
        "role_kind": "real_noise_system_identification",
        "training_sources": ("real",),
        "real_only": True,
    },
    "classifier_n": {
        "role_kind": "shared_aod_law_discovery",
        "training_sources": ("real_3q", "real_4q"),
        "real_only": True,
        "real_n_generalization_claim_allowed": False,
    },
    "feedback_n": {
        "role_kind": "shared_noise_dynamics_hyperprior",
        "training_sources": ("real_3q", "real_4q"),
        "real_only": True,
        "hardware_action_enabled": False,
        "real_n_generalization_claim_allowed": False,
    },
}


def validate_training_sources(model_id: str, source_ids: Iterable[str]) -> Dict[str, Any]:
    if model_id not in MODEL_ROLE_SPECS:
        raise ValueError(f"unknown model ID: {model_id}")
    spec = MODEL_ROLE_SPECS[model_id]
    source_ids = tuple(str(value) for value in source_ids)
    if spec["real_only"] and any(value == "synthetic" for value in source_ids):
        raise ValueError(f"{model_id} is real-only and rejects synthetic rows")
    allowed = set(spec["training_sources"])
    for source_id in source_ids:
        normalized = source_id if source_id != "real" or "real" in allowed else "real_3q"
        if normalized not in allowed and source_id != "real":
            raise ValueError(f"{model_id} rejects training source {source_id!r}")
    return {
        **spec,
        "model_id": model_id,
        "real_only": bool(spec["real_only"]),
        "synthetic_rows": sum(value == "synthetic" for value in source_ids),
    }


__all__ = [
    "MODEL_ROLE_SPECS",
    "PhysicalCoefficientPosteriorV1",
    "validate_training_sources",
]
