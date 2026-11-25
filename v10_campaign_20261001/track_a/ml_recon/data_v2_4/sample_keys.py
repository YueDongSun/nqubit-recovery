"""Prefix-stable SHA-256/Philox sample identities and RNG streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .hashing import sha256_json


@dataclass(frozen=True)
class V24SampleKey:
    global_seed: int
    split: str
    sample_id: str
    dataset_family: str
    N: int
    generator_sha: str
    cell_id: str

    def validate(self) -> None:
        allowed_splits = {
            "train", "validation", "test", "locked_ood", "train_target_family",
            "validation_target_family", "probe_train", "probe_validation",
            "negative_control", "audit",
        }
        if self.split not in allowed_splits:
            raise ValueError("invalid V2.4 split")
        if self.N < 3 or self.N > 256 or len(self.generator_sha) != 64:
            raise ValueError("invalid V2.4 sample key")

    def sha256(self) -> str:
        self.validate()
        return sha256_json(asdict(self))

    def rng(self, stream: str, substream: str | int = "0") -> np.random.Generator:
        digest = sha256_json({
            "global_seed": self.global_seed,
            "split": self.split,
            "sample_id": self.sample_id,
            "stream": str(stream),
            "substream": str(substream),
            "generator_sha": self.generator_sha,
        })
        return np.random.Generator(np.random.Philox(int(digest[:16], 16)))

    def _seed(self, stream: str) -> int:
        return int(sha256_json({"sample_key_sha": self.sha256(), "stream": stream})[:16], 16)

    @property
    def state_seed(self) -> int:
        return self._seed("state")

    @property
    def partition_seed(self) -> int:
        return self._seed("partition")

    @property
    def measurement_seed(self) -> int:
        return self._seed("measurement")

    @property
    def noise_seed(self) -> int:
        return self._seed("noise")

    @property
    def shot_seed(self) -> int:
        return self._seed("shot")

    @property
    def action_seed(self) -> int:
        return self._seed("action")


def make_sample_key(*, global_seed: int, split: str, dataset_family: str,
                    N: int, index: int, generator_sha: str, cell_id: str) -> V24SampleKey:
    sample_id = f"v2_4:{dataset_family}:{cell_id}:{split}:N{N}:{int(index):012d}"
    return V24SampleKey(global_seed, split, sample_id, dataset_family,
                        int(N), generator_sha, cell_id)


def deterministic_value_sha(value: Any) -> str:
    if isinstance(value, np.ndarray):
        from .hashing import sha256_array
        return sha256_array(value)
    return sha256_json(value)
