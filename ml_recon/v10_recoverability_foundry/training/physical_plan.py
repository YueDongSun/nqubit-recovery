"""Validated physical and asynchronous-memory plan; never run authorization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from ml_recon.physics_hybrid_v10.contracts import sha256_of


PHYSICAL_PROGRAM_NAMESPACE = "TRACK_A_PHYSICAL_V2"
PHYSICAL_PLAN_SCHEMA = "V10PhysicalTrainingPlanV4"
PHYSICAL_BRANCH = "codex/v10_tracka_physical_budget_v2"
PRIMARY_N_VALUES = (3, 4, 6, 8, 12, 18, 24, 36)
STRESS_N_VALUES = (4, 6, 8, 12, 18)
BUDGET_CAPS = (256, 512, 1024, 2048, 4096)
STAGES = (
    "P0_CONTRACTS",
    "E0_QUALIFICATION",
    "E1_EXACT_TEACHER",
    "E2_ORACLE_SCREEN",
    "E3_ANTI_COLLAPSE",
    "S1_GENERALIZATION",
    "S2_STRUCTURE",
    "S3_POSTERIOR",
    "S4_FACTORIAL",
    "S5_CONFIRMATION",
)
AUTHORITY = {
    "research_only": True,
    "selection_evidence": False,
    "promotion_ready": False,
    "hardware_action_authorized": False,
    "ten_k_training_authorized": False,
    "real_feedback_authorized": False,
}

MEMORY_ARMS = (
    "M0_NO_MEMORY",
    "M1_IDEAL_MEMORY",
    "M2_PRL_ANCHORED_MEMORY",
    "M3_MEMORY_SENSITIVITY",
)


@dataclass(frozen=True)
class Memory_training_plan:
    schema: str = "V10MemoryTrainingPlanV1"
    comparison_arms: tuple[str, ...] = MEMORY_ARMS
    primary_estimand: str = (
        "finite_horizon_expected_accepted_useful_count_per_elapsed_second"
    )
    clock: str = "registered_integer_ticks_with_explicit_seconds_per_tick"
    common_random_numbers: str = (
        "matched_exogenous_trial_opportunities_not_copied_admitted_successes"
    )
    e0_memory_targets: str = "NOT_AVAILABLE"
    e0_memory_loss_enabled: bool = False
    memory_policy_training_enabled: bool = False
    m2_parameter_mapping_status: str = "BLOCKED_PARAMETER_UNBOUND"
    memory_s5_status: str = "NOT_RUN"
    published_primitive_scope: str = "W3_PLUS_W3_TO_W4_ONLY"
    modeled_gain_is_real_gain: bool = False

    def validate(self) -> None:
        expected = asdict(Memory_training_plan())
        if asdict(self) != expected or any(
            type(getattr(self, name)) is not bool
            for name in (
                "e0_memory_loss_enabled",
                "memory_policy_training_enabled",
                "modeled_gain_is_real_gain",
            )
        ):
            raise ValueError(
                "memory research requires separate parameter, queue and stage evidence; E0 cannot enable it"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Memory_training_plan:
        if not isinstance(value, Mapping) or set(value) != {
            item.name for item in fields(cls)
        }:
            raise ValueError("memory plan fields mismatch")
        payload = dict(value)
        if not isinstance(payload["comparison_arms"], (list, tuple)):
            raise ValueError("memory comparison arms must be an array")
        payload["comparison_arms"] = tuple(payload["comparison_arms"])
        result = cls(**payload)
        result.validate()
        return result


@dataclass(frozen=True)
class Physical_training_plan:
    schema: str = PHYSICAL_PLAN_SCHEMA
    namespace: str = PHYSICAL_PROGRAM_NAMESPACE
    primary_n_values: tuple[int, ...] = PRIMARY_N_VALUES
    primary_k: int = 1
    stress_n_values: tuple[int, ...] = STRESS_N_VALUES
    stress_k: int = 2
    mode_shot_caps: tuple[int, ...] = BUDGET_CAPS
    budget_protocols: tuple[str, ...] = (
        "FIXED_RESOURCE",
        "FIXED_TIME",
        "FIXED_QUALITY",
        "SENSITIVITY",
    )
    resource_interpretation: str = (
        "fixed_mode_opportunity_cap_with_reported_integer_remainder"
    )
    sample_rungs: tuple[int, ...] = (1000, 2000, 10000, 50000, 200000)
    development_seeds: tuple[int, ...] = (20261001, 20261002, 20261003)
    data_seed: int = 20261001
    confirmation_seeds: tuple[int, ...] = (
        20261101,
        20261102,
        20261103,
        20261104,
        20261105,
    )
    exact_teacher_rows: int = 1000
    oracle_screen_rows: int = 1000
    anti_collapse_rows: int = 2000
    qualification_rows: int = 8
    qualification_optimizer_steps: int = 2
    qualification_validation_rows: int = 4
    qualification_batch_size: int = 4
    qualification_epochs: int = 1
    qualification_profile: str = "C-TEST"
    qualification_n_values: tuple[int, ...] = (3, 4)
    qualification_requested_mode_shots: int = 1024
    qualification_samples_per_n: int = 256
    qualification_noise_per_cell_success_probability: float = 0.005
    qualification_split_sizes: tuple[int, int, int] = (8, 4, 4)
    validation_rows: int = 200
    batch_size: int = 16
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    amp_enabled: bool = False
    amp_initial_scale: float = 256.0
    telemetry_interval_seconds: float = 1200.0
    quality_threshold: float | None = None
    minimum_meaningful_effect: float | None = None
    threshold_registration_sha: str | None = None
    required_review_mode: str = "pro"
    required_review_surface: str = "chatgpt_web_chat"
    train_after_complete_code_review_only: bool = True
    automatic_scale_up: bool = False
    allow_two_million_rung: bool = False
    memory_plan: Memory_training_plan = field(default_factory=Memory_training_plan)
    authority: Mapping[str, bool] = field(default_factory=lambda: dict(AUTHORITY))

    def validate(self) -> None:
        if type(self.memory_plan) is not Memory_training_plan:
            raise ValueError("physical plan requires the typed memory subplan")
        self.memory_plan.validate()
        if type(self.data_seed) is not int or self.data_seed < 0:
            raise ValueError(
                "data_seed must be fixed independently of model initialization seeds"
            )
        if (
            self.schema != PHYSICAL_PLAN_SCHEMA
            or self.namespace != PHYSICAL_PROGRAM_NAMESPACE
        ):
            raise ValueError("unknown physical training plan schema or namespace")
        if (
            self.primary_n_values != PRIMARY_N_VALUES
            or self.stress_n_values != STRESS_N_VALUES
        ):
            raise ValueError(
                "primary and stress physical N panels must match the registered plan"
            )
        if (
            type(self.primary_k) is not int
            or type(self.stress_k) is not int
            or (self.primary_k, self.stress_k) != (1, 2)
        ):
            raise ValueError(
                "K=1 is primary and K=2 is a separately gated stress panel"
            )
        if self.mode_shot_caps != BUDGET_CAPS:
            raise ValueError(
                "physical mode-shot caps must retain the registered 1024-based grid"
            )
        if self.budget_protocols != (
            "FIXED_RESOURCE",
            "FIXED_TIME",
            "FIXED_QUALITY",
            "SENSITIVITY",
        ):
            raise ValueError(
                "fixed time, fixed mode resource, quality search and sensitivity are separate protocols"
            )
        if (
            self.resource_interpretation
            != "fixed_mode_opportunity_cap_with_reported_integer_remainder"
        ):
            raise ValueError(
                "integer budget allocation must report actual expenditure and remainder"
            )
        for name in (
            "primary_n_values",
            "stress_n_values",
            "mode_shot_caps",
            "sample_rungs",
            "development_seeds",
            "confirmation_seeds",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or not value
                or any(type(item) is not int or item < 0 for item in value)
                or tuple(sorted(set(value))) != value
            ):
                raise ValueError(
                    f"{name} must be an immutable sorted unique nonnegative integer panel"
                )
        if self.sample_rungs != (1000, 2000, 10000, 50000, 200000):
            raise ValueError(
                "unregistered model sample rung; two million is not a default target"
            )
        if len(self.development_seeds) < 3 or len(self.confirmation_seeds) < 5:
            raise ValueError(
                "development requires three seeds; confirmation requires five"
            )
        if set(self.development_seeds) & set(self.confirmation_seeds):
            raise ValueError("development and confirmation seeds must be disjoint")
        for name in (
            "exact_teacher_rows",
            "oracle_screen_rows",
            "anti_collapse_rows",
            "qualification_rows",
            "qualification_optimizer_steps",
            "qualification_validation_rows",
            "qualification_batch_size",
            "qualification_epochs",
            "qualification_requested_mode_shots",
            "qualification_samples_per_n",
            "validation_rows",
            "batch_size",
            "epochs",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.exact_teacher_rows,
            self.oracle_screen_rows,
            self.anti_collapse_rows,
        ) != (1000, 1000, 2000):
            raise ValueError(
                "teacher/oracle/anti-collapse rungs are frozen at 1k/1k/2k"
            )
        if self.qualification_rows != 8 or self.qualification_optimizer_steps != 2:
            raise ValueError("qualification is a bounded engineering run, not a pilot")
        if (
            self.qualification_validation_rows,
            self.qualification_batch_size,
            self.qualification_epochs,
            self.qualification_profile,
        ) != (4, 4, 1, "C-TEST"):
            raise ValueError(
                "E0 requires four validation rows, batch four, one epoch and C-TEST"
            )
        if (
            self.qualification_n_values != (3, 4)
            or self.qualification_requested_mode_shots != 1024
            or self.qualification_samples_per_n != 256
            or self.qualification_split_sizes != (
                self.qualification_rows,
                self.qualification_validation_rows,
                4,
            )
            or isinstance(
                self.qualification_noise_per_cell_success_probability, bool
            )
            or not isinstance(
                self.qualification_noise_per_cell_success_probability, Real
            )
            or not math.isfinite(
                self.qualification_noise_per_cell_success_probability
            )
            or self.qualification_noise_per_cell_success_probability != 0.005
        ):
            raise ValueError(
                "E0 recipe must remain the exact N=3/4 C-TEST budget and split contract"
            )
        for name in (
            "learning_rate",
            "amp_initial_scale",
            "telemetry_interval_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, Real)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and nonnegative")
        for name in (
            "amp_enabled",
            "train_after_complete_code_review_only",
            "automatic_scale_up",
            "allow_two_million_rung",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a strict boolean")
        if (
            self.required_review_mode != "pro"
            or self.required_review_surface != "chatgpt_web_chat"
            or not self.train_after_complete_code_review_only
            or self.automatic_scale_up
            or self.allow_two_million_rung
        ):
            raise ValueError(
                "the user requires completed code, a new Chat Pro review and explicit stage admission"
            )
        if dict(self.authority) != AUTHORITY or any(
            type(value) is not bool for value in self.authority.values()
        ):
            raise ValueError(
                "a training plan cannot grant training scale, promotion or hardware authority"
            )
        registration = (
            self.quality_threshold,
            self.minimum_meaningful_effect,
            self.threshold_registration_sha,
        )
        if any(value is not None for value in registration):
            if any(value is None for value in registration):
                raise ValueError(
                    "quality/effect thresholds require one complete development registration"
                )
            for name in ("quality_threshold", "minimum_meaningful_effect"):
                value = getattr(self, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                    or not 0 < value < 1
                ):
                    raise ValueError(f"{name} must be finite and strictly within (0,1)")
            if (
                not isinstance(self.threshold_registration_sha, str)
                or len(self.threshold_registration_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.threshold_registration_sha
                )
            ):
                raise ValueError(
                    "threshold registration requires a canonical source-bound SHA"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return sha256_of(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Physical_training_plan:
        if not isinstance(value, Mapping):
            raise ValueError("physical plan must be a JSON object")
        allowed = {item.name for item in fields(cls)}
        if set(value) != allowed:
            raise ValueError(
                f"physical plan fields mismatch: missing={sorted(allowed - set(value))}, extra={sorted(set(value) - allowed)}"
            )
        payload = dict(value)
        payload["memory_plan"] = Memory_training_plan.from_mapping(
            payload["memory_plan"]
        )
        for name in (
            "primary_n_values",
            "stress_n_values",
            "mode_shot_caps",
            "budget_protocols",
            "sample_rungs",
            "development_seeds",
            "confirmation_seeds",
            "qualification_n_values",
            "qualification_split_sizes",
        ):
            if not isinstance(payload[name], (list, tuple)):
                raise ValueError(f"{name} must be an array")
            payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def load_physical_plan(path: str | Path) -> Physical_training_plan:
    return Physical_training_plan.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def stage_requirements(stage: str) -> tuple[str, ...]:
    if stage not in STAGES:
        raise ValueError("unknown physical program stage")
    index = STAGES.index(stage)
    order = ("P0", "E0", "E1", "E2", "E3", "S1", "S2", "S3", "S4", "S5")
    return order[:index]
