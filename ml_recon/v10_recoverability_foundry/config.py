"""Validated external configuration for the Track A foundry."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Mapping

from ml_recon.physics_hybrid_v10.contracts import sha256_of


DENSE_N_HARD_LIMIT = 16


@dataclass(frozen=True)
class PhysicsConfig:
    n_values: tuple[int, ...] = (3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256)
    k_values: tuple[int, ...] = (1, 2, 3, 4, 5)
    max_sparse_terms: int = 4096
    dense_n_limit: int = 16
    collision_arms: tuple[str, ...] = (
        "coherent_exact",
        "sector_dephased",
        "classical_mixture",
    )

    def validate(self) -> None:
        if (not self.n_values
                or any(type(value) is not int for value in self.n_values)
                or tuple(sorted(set(self.n_values))) != self.n_values
                or min(self.n_values) < 3
                or max(self.n_values) > 256):
            raise ValueError("Track A requires 3 <= N <= 256")
        if (not self.k_values
                or any(type(value) is not int for value in self.k_values)
                or tuple(sorted(set(self.k_values))) != self.k_values
                or min(self.k_values) < 1
                or max(self.k_values) > 5):
            raise ValueError("Track A supports registered fixed-K values 1..5")
        if (type(self.max_sparse_terms) is not int
                or self.max_sparse_terms <= 0
                or type(self.dense_n_limit) is not int
                or not 1 <= self.dense_n_limit <= DENSE_N_HARD_LIMIT):
            raise ValueError("invalid sparse or dense state boundary")
        expected = {"coherent_exact", "sector_dephased", "classical_mixture"}
        if (len(self.collision_arms) != len(expected)
                or any(not isinstance(value, str) for value in self.collision_arms)
                or set(self.collision_arms) != expected):
            raise ValueError("all three registered collision arms are required")


@dataclass(frozen=True)
class DataConfig:
    global_seed: int = 20261001
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    smoke_shots: int = 64
    token_budget_per_batch: int = 4096
    rank_disjoint_keys: bool = True

    def validate(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction)
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(float(value)) for value in fractions):
            raise ValueError("data fractions must be finite real values")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie in (0,1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0,1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must leave a test split")
        if (type(self.global_seed) is not int or self.global_seed < 0
                or type(self.smoke_shots) is not int or self.smoke_shots <= 0
                or type(self.token_budget_per_batch) is not int
                or self.token_budget_per_batch <= 0):
            raise ValueError("shot and token budgets must be positive")
        if type(self.rank_disjoint_keys) is not bool or not self.rank_disjoint_keys:
            raise ValueError("Track A requires rank-disjoint counter keys")


@dataclass(frozen=True)
class ModelConfig:
    classifier_profile: str = "C-TEST"
    feedback_width: int = 32
    feedback_heads: int = 4
    feedback_blocks: int = 1
    partition_candidate_count: int = 8
    posterior_coverage_targets: tuple[float, ...] = (0.5, 0.9, 0.95)

    def validate(self) -> None:
        if (not isinstance(self.classifier_profile, str)
                or self.classifier_profile not in {"C-TEST", "C-S", "C-M", "C-L", "C-XL"}):
            raise ValueError("unknown V10 classifier profile")
        dimensions = (
            self.feedback_width,
            self.feedback_heads,
            self.feedback_blocks,
            self.partition_candidate_count,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("feedback dimensions must be positive")
        if self.feedback_width % self.feedback_heads:
            raise ValueError("feedback width must be divisible by the head count")
        if not 1 <= self.partition_candidate_count <= 1024:
            raise ValueError("partition candidate count must be in [1,1024]")
        if (not self.posterior_coverage_targets
                or any(isinstance(value, bool) or not isinstance(value, Real)
                       or not math.isfinite(float(value))
                       or not 0.0 < value < 1.0
                       for value in self.posterior_coverage_targets)
                or tuple(sorted(set(self.posterior_coverage_targets)))
                != self.posterior_coverage_targets):
            raise ValueError("coverage targets must lie in (0,1)")


@dataclass(frozen=True)
class TrainingConfig:
    run_mode: str = "smoke"
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 3e-4
    amp_enabled: bool = False
    sample_rungs: tuple[int, ...] = (1_000, 10_000, 50_000, 200_000, 2_000_000)
    training_authorized: bool = False
    ten_k_training_authorized: bool = False

    def validate(self) -> None:
        if not isinstance(self.run_mode, str) or self.run_mode not in {"smoke", "pilot", "full"}:
            raise ValueError("run_mode must be smoke, pilot, or full")
        if (type(self.batch_size) is not int or self.batch_size <= 0
                or type(self.epochs) is not int or self.epochs <= 0
                or isinstance(self.learning_rate, bool)
                or not isinstance(self.learning_rate, Real)
                or not math.isfinite(float(self.learning_rate))
                or self.learning_rate <= 0.0):
            raise ValueError("training values must be positive")
        if (not self.sample_rungs
                or any(type(value) is not int or value <= 0 for value in self.sample_rungs)
                or tuple(sorted(set(self.sample_rungs))) != self.sample_rungs):
            raise ValueError("sample rungs must be unique and increasing")
        if (type(self.amp_enabled) is not bool
                or type(self.training_authorized) is not bool
                or type(self.ten_k_training_authorized) is not bool):
            raise ValueError("training flags must be strict booleans")
        if self.training_authorized or self.ten_k_training_authorized:
            raise ValueError("Track A config cannot self-authorize training")
        if self.run_mode != "smoke":
            raise ValueError(
                "non-smoke execution requires an external gate artifact and runtime entry point"
            )


@dataclass(frozen=True)
class GateConfig:
    fisher_rank_fraction_minimum: float = 0.25
    control_auroc_minimum: float = 0.65
    action_cosine_minimum: float = 0.8
    serious_harm_u95_maximum: float = 0.01
    posterior_coverage_error_maximum: float = 0.05

    def validate(self) -> None:
        values = (
            self.fisher_rank_fraction_minimum,
            self.control_auroc_minimum,
            self.action_cosine_minimum,
            self.serious_harm_u95_maximum,
            self.posterior_coverage_error_maximum,
        )
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(float(value))
               or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("gate probabilities and fractions must lie in [0,1]")


@dataclass(frozen=True)
class TrackAConfig:
    schema: str = "V10TrackAConfigV1"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gates: GateConfig = field(default_factory=GateConfig)

    def validate(self) -> None:
        if self.schema != "V10TrackAConfigV1":
            raise ValueError("unknown Track A configuration schema")
        self.physics.validate()
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.gates.validate()

    def sha256(self) -> str:
        self.validate()
        return sha256_of(asdict(self))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrackAConfig":
        physics_value = dict(value.get("physics", {}))
        for key in ("n_values", "k_values", "collision_arms"):
            if key in physics_value:
                physics_value[key] = tuple(physics_value[key])
        model_value = dict(value.get("model", {}))
        if "posterior_coverage_targets" in model_value:
            model_value["posterior_coverage_targets"] = tuple(
                model_value["posterior_coverage_targets"])
        training_value = dict(value.get("training", {}))
        if "sample_rungs" in training_value:
            training_value["sample_rungs"] = tuple(training_value["sample_rungs"])
        config = cls(
            schema=str(value.get("schema", "")),
            physics=PhysicsConfig(**physics_value),
            data=DataConfig(**dict(value.get("data", {}))),
            model=ModelConfig(**model_value),
            training=TrainingConfig(**training_value),
            gates=GateConfig(**dict(value.get("gates", {}))),
        )
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_track_a_config(path: str | Path | None = None) -> TrackAConfig:
    if path is None:
        config = TrackAConfig()
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            config = TrackAConfig.from_mapping(json.load(handle))
    config.validate()
    return config
