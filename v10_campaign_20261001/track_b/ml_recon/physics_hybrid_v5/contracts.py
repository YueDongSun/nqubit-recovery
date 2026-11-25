"""Machine-readable contracts for the Physics Hybrid V5 research track."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


DATA_CONTRACT_SCHEMA = "PhysicsHybridV5DataContract"
EXPERIMENT_CONTRACT_SCHEMA = "PhysicsHybridV5ExperimentContract"
DATA_CONTRACT_VERSION = 1
EXPERIMENT_CONTRACT_VERSION = 1
VALID_SOURCES = frozenset({"synthetic", "measured", "derived"})
VALID_HARDWARE_MODES = frozenset({"none", "read_only", "shadow", "write"})
VALID_LABEL_STATES = frozenset({"valid", "missing", "masked", "derived"})


@dataclass(frozen=True)
class PhysicsHybridV5DataContract:
    dataset_id: str
    system_type: str
    model_id: str
    source: str
    unique_latent_count: int
    noise_redraw_count: int
    optimizer_exposure_count: int
    simulator_hash: str
    parameter_prior_hash: str
    basis_registry_hash: str
    operation_registry_hash: str
    calibration_hash: str
    campaign_id: str | None
    time_block_id: str | None
    probe_pair_id: str | None
    parent_dataset_ids: tuple[str, ...] = ()
    label_provenance: Dict[str, str] = field(default_factory=dict)
    label_validity: Dict[str, str] = field(default_factory=dict)
    exposure_s_summary: Dict[str, float] = field(default_factory=dict)
    count_strata: Dict[str, int] = field(default_factory=dict)
    missingness_summary: Dict[str, float] = field(default_factory=dict)
    split_group_fields: tuple[str, ...] = (
        "campaign_id",
        "calibration_hash",
        "time_block_id",
        "probe_pair_id",
    )
    synthetic_parent_relation: str | None = None
    schema: str = DATA_CONTRACT_SCHEMA
    schema_version: int = DATA_CONTRACT_VERSION

    def validate(self) -> None:
        validate_data_contract(asdict(self))

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["parent_dataset_ids"] = list(self.parent_dataset_ids)
        payload["split_group_fields"] = list(self.split_group_fields)
        return payload

    def sha256(self) -> str:
        return stable_json_hash(self.to_json_dict())


@dataclass(frozen=True)
class PhysicsHybridV5ExperimentContract:
    experiment_id: str
    status: str
    hypothesis: str
    model_ids: tuple[str, ...]
    evidence_scope: str
    data_contract_hashes: tuple[str, ...]
    split_policy: str
    comparison_axis: str
    baselines: tuple[str, ...]
    ablations: tuple[str, ...]
    metrics: tuple[str, ...]
    seeds: tuple[int, ...]
    optimizer_update_budget: int
    wall_time_budget_h: float
    hardware_mode: str = "none"
    hardware_approval_required: bool = True
    acceptance_rules: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = (
        "non_finite_loss",
        "contract_violation",
        "accelerator_oom",
    )
    schema: str = EXPERIMENT_CONTRACT_SCHEMA
    schema_version: int = EXPERIMENT_CONTRACT_VERSION

    def validate(self) -> None:
        validate_experiment_contract(asdict(self))

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in (
            "model_ids",
            "data_contract_hashes",
            "baselines",
            "ablations",
            "metrics",
            "seeds",
            "acceptance_rules",
            "stop_conditions",
        ):
            payload[name] = list(payload[name])
        return payload

    def sha256(self) -> str:
        return stable_json_hash(self.to_json_dict())


def validate_data_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != DATA_CONTRACT_SCHEMA:
        raise ValueError(f"data contract schema must be {DATA_CONTRACT_SCHEMA}")
    if int(payload.get("schema_version", 0)) != DATA_CONTRACT_VERSION:
        raise ValueError(f"data contract schema_version must be {DATA_CONTRACT_VERSION}")
    for field_name in ("dataset_id", "system_type", "model_id"):
        if not str(payload.get(field_name, "")).strip():
            raise ValueError(f"{field_name} is required")
    source = str(payload.get("source", ""))
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}")
    unique = int(payload.get("unique_latent_count", -1))
    redraws = int(payload.get("noise_redraw_count", -1))
    exposures = int(payload.get("optimizer_exposure_count", -1))
    if min(unique, redraws, exposures) < 0:
        raise ValueError("data-volume counters must be non-negative")
    if redraws < unique and source != "measured":
        raise ValueError("synthetic/derived noise_redraw_count cannot be below unique_latent_count")
    if exposures < redraws:
        raise ValueError("optimizer_exposure_count cannot be below noise_redraw_count")
    for field_name in (
        "simulator_hash",
        "parameter_prior_hash",
        "basis_registry_hash",
        "operation_registry_hash",
        "calibration_hash",
    ):
        _require_hash_or_explicit_missing(payload.get(field_name), field_name)
    label_validity = dict(payload.get("label_validity") or {})
    label_provenance = dict(payload.get("label_provenance") or {})
    if not label_validity:
        raise ValueError("label_validity must declare every supervised target")
    for label, state in label_validity.items():
        if state not in VALID_LABEL_STATES:
            raise ValueError(f"label_validity[{label!r}] has unknown state {state!r}")
        if state in {"valid", "derived"} and not str(label_provenance.get(label, "")).strip():
            raise ValueError(f"valid label {label!r} requires provenance")
    parents = tuple(payload.get("parent_dataset_ids") or ())
    if source == "derived" and not parents:
        raise ValueError("derived data must identify at least one parent dataset")
    if source == "measured" and payload.get("synthetic_parent_relation"):
        raise ValueError("measured data cannot declare a synthetic parent relation")
    exposure = dict(payload.get("exposure_s_summary") or {})
    if exposure:
        minimum = float(exposure.get("min", -1.0))
        maximum = float(exposure.get("max", -1.0))
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError("exposure_s_summary must have 0 < min <= max")
    strata = dict(payload.get("count_strata") or {})
    expected_strata = {"lt10", "10_99", "100_999", "ge1000"}
    if strata and set(strata) != expected_strata:
        raise ValueError(f"count_strata must define exactly {sorted(expected_strata)}")
    if any(int(value) < 0 for value in strata.values()):
        raise ValueError("count_strata values must be non-negative")


def validate_experiment_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != EXPERIMENT_CONTRACT_SCHEMA:
        raise ValueError(f"experiment contract schema must be {EXPERIMENT_CONTRACT_SCHEMA}")
    if int(payload.get("schema_version", 0)) != EXPERIMENT_CONTRACT_VERSION:
        raise ValueError(
            f"experiment contract schema_version must be {EXPERIMENT_CONTRACT_VERSION}"
        )
    if not str(payload.get("experiment_id", "")).strip():
        raise ValueError("experiment_id is required")
    if not str(payload.get("hypothesis", "")).strip():
        raise ValueError("hypothesis is required")
    if not tuple(payload.get("model_ids") or ()):
        raise ValueError("model_ids must not be empty")
    if not tuple(payload.get("data_contract_hashes") or ()):
        raise ValueError("data_contract_hashes must not be empty")
    if not tuple(payload.get("baselines") or ()):
        raise ValueError("at least one baseline is required")
    if not tuple(payload.get("ablations") or ()):
        raise ValueError("at least one ablation is required")
    if not tuple(payload.get("metrics") or ()):
        raise ValueError("metrics must not be empty")
    seeds = tuple(int(seed) for seed in payload.get("seeds") or ())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if int(payload.get("optimizer_update_budget", 0)) <= 0:
        raise ValueError("optimizer_update_budget must be positive")
    if float(payload.get("wall_time_budget_h", 0.0)) <= 0.0:
        raise ValueError("wall_time_budget_h must be positive")
    hardware_mode = str(payload.get("hardware_mode", "none"))
    if hardware_mode not in VALID_HARDWARE_MODES:
        raise ValueError(f"hardware_mode must be one of {sorted(VALID_HARDWARE_MODES)}")
    if hardware_mode == "write" and not bool(payload.get("hardware_approval_required", False)):
        raise ValueError("hardware write mode must require explicit approval")


def load_contract(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") == DATA_CONTRACT_SCHEMA:
        validate_data_contract(payload)
    elif payload.get("schema") == EXPERIMENT_CONTRACT_SCHEMA:
        validate_experiment_contract(payload)
    else:
        raise ValueError("unknown V5 contract schema")
    return payload


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash_or_explicit_missing(value: Any, field_name: str) -> None:
    text = str(value or "")
    if text == "missing":
        return
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 or 'missing'")
