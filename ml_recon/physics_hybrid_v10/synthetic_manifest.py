"""Order-independent, exactly replayable V10 synthetic manifest."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, Tuple

import numpy as np

from .contracts import sha256_of

V10_GENERATOR_SHA = hashlib.sha256(b"V10SyntheticGeneratorV1").hexdigest()


def _seed(sample_id: str, stream: str, generator_sha: str = V10_GENERATOR_SHA,
          global_seed: int = 0, split: str = "train", substream: str = "0") -> int:
    digest = hashlib.sha256(
        f"{generator_sha}|{global_seed}|{split}|{sample_id}|{stream}|{substream}".encode(
            "utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


@dataclass(frozen=True)
class V10SyntheticSampleKey:
    sample_id: str
    N: int
    dataset_family: str
    state_seed: int
    partition_seed: int
    measurement_seed: int
    noise_seed: int
    shot_seed: int
    action_seed: int
    generator_sha: str
    measurement_contract_sha: str
    shot_budget_mode: str
    global_seed: int = 0
    split: str = "train"

    def sha256(self) -> str:
        return sha256_of(asdict(self))


class V10CounterBasedRng:
    """Independent named RNG streams; enumeration order never affects a sample."""

    def __init__(self, sample_id: str, generator_sha: str = V10_GENERATOR_SHA,
                 global_seed: int = 0, split: str = "train"):
        self.sample_id = sample_id
        self.generator_sha = generator_sha
        self.global_seed = int(global_seed)
        self.split = str(split)

    def stream(self, name: str, substream: str | int = "0") -> np.random.Generator:
        return np.random.Generator(np.random.Philox(_seed(
            self.sample_id, name, self.generator_sha, self.global_seed,
            self.split, str(substream))))


@dataclass(frozen=True)
class V10SyntheticManifest:
    pool_sizes: Dict[str, int]
    N_values: Tuple[int, ...]
    measurement_contract_sha: str
    shot_budget_modes: Tuple[str, ...] = ("S-total", "S-setting", "S-ideal")
    generator_sha: str = V10_GENERATOR_SHA
    global_seed: int = 20261001
    split_thresholds: Tuple[int, int] = (8000, 9000)

    def validate(self) -> None:
        if min(self.N_values) < 3 or max(self.N_values) > 256:
            raise ValueError("manifest N outside V10 range")
        if any(v < 0 for v in self.pool_sizes.values()):
            raise ValueError("pool sizes must be non-negative")
        if not (0 < self.split_thresholds[0] < self.split_thresholds[1] < 10_000):
            raise ValueError("split thresholds must define train/validation/test buckets")

    def sha256(self) -> str:
        self.validate()
        return sha256_of(asdict(self))

    def split_for_sample(self, sample_id: str) -> str:
        bucket = _seed(sample_id, "split", self.generator_sha,
                       self.global_seed, "split_registry", "0") % 10_000
        if bucket < self.split_thresholds[0]:
            return "train"
        if bucket < self.split_thresholds[1]:
            return "validation"
        return "test"

    def sample_key(self, dataset_family: str, index: int, N: int,
                   shot_budget_mode: str, split: str | None = None) -> V10SyntheticSampleKey:
        if dataset_family not in self.pool_sizes or not 0 <= index < self.pool_sizes[dataset_family]:
            raise IndexError("sample outside declared pool")
        if N not in self.N_values or shot_budget_mode not in self.shot_budget_modes:
            raise ValueError("sample key outside manifest strata")
        sample_id = f"v10:{dataset_family}:N{N}:{shot_budget_mode}:{index:012d}"
        assigned_split = self.split_for_sample(sample_id) if split is None else str(split)
        if assigned_split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        return V10SyntheticSampleKey(
            sample_id, N, dataset_family,
            _seed(sample_id, "state", self.generator_sha, self.global_seed, assigned_split),
            _seed(sample_id, "partition", self.generator_sha, self.global_seed, assigned_split),
            _seed(sample_id, "measurement", self.generator_sha, self.global_seed, assigned_split),
            _seed(sample_id, "noise", self.generator_sha, self.global_seed, assigned_split),
            _seed(sample_id, "shot", self.generator_sha, self.global_seed, assigned_split),
            _seed(sample_id, "action", self.generator_sha, self.global_seed, assigned_split),
            self.generator_sha, self.measurement_contract_sha, shot_budget_mode,
            self.global_seed, assigned_split)


def default_manifest(measurement_contract_sha: str) -> V10SyntheticManifest:
    return V10SyntheticManifest(
        pool_sizes={
            "small_n_clean": 1_000_000,
            "small_n_noisy": 1_000_000,
            "general_n_structure": 2_000_000,
            "w_scaling": 2_000_000,
            "feedback_one_step": 2_000_000,
            "feedback_history": 500_000,
            "feedback_noop": 250_000,
            "mechanism_shift": 250_000,
            "locked_ood": 250_000,
        },
        N_values=(3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256),
        measurement_contract_sha=measurement_contract_sha,
    )
