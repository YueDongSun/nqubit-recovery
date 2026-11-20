"""Deterministic, lazy synthetic banks for the sustained corrected-law study."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import numpy as np
import torch

from .effective_projector_observation import EFFECTIVE_LAW
from .training_ideas_data import make_projector_training_batch
from .training_ideas_experiment import _take

SIZES = (3, 4, 8)
CHUNK = 256
BATCH = 32
LARGE = 32768
SMALL = 1024
REUSE = 16
SPLIT = {"train": 971330, "validation": 974107, "calibration": 977711,
         "confirmation": 979153, "fixture": 970211}


def train_bank_namespace(identity):
    """Resolve the TRAIN-only override, including archived runs without named splits."""
    return identity.get("train_namespace", identity.get("split_namespace"))


def bank_seed(n, role, chunk, split_namespace=None):
    """Return the exact integer generator seed for a registered state-bank chunk."""
    entropy = [SPLIT[role], n, chunk]
    if split_namespace is not None:
        # Domain-separate the new study while preserving the historical draws.
        entropy.extend(sha256(split_namespace.encode("utf-8")).digest())
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def build_bank(n, size, role, chunk, teachers, *, split_namespace=None, fixed_calibration=None):
    seed = bank_seed(n, role, chunk, split_namespace)
    return make_projector_training_batch(
        n, size, 4096, seed, observation_law=EFFECTIVE_LAW, include_teachers=teachers, device="cpu",
        fixed_calibration=fixed_calibration,
        report_fixed_geometry=role in {"fixture", "confirmation"},
    )


class TrainingBanks:
    def __init__(self, cache_root, rows, teachers, *, split_namespace=None):
        if rows < CHUNK or rows % CHUNK:
            raise ValueError("train states per N must be a positive multiple of the chunk size")
        self.split_namespace = split_namespace
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows = rows
        self.teachers = teachers
        self.loaded = {}
        self.seen = {n: torch.zeros(rows, dtype=torch.bool) for n in SIZES}

    def take(self, step, device):
        n = SIZES[step % len(SIZES)]
        visit = step // len(SIZES)
        chunks = self.rows // CHUNK
        chunk = (visit // REUSE) % chunks
        if n not in self.loaded or self.loaded[n][0] != chunk:
            path = self.root / f"n{n}_chunk{chunk:03d}.pt"
            if path.exists():
                bank = torch.load(path, map_location="cpu", weights_only=True)
            else:
                bank = build_bank(n, CHUNK, "train", chunk, self.teachers, split_namespace=self.split_namespace)
                temp = path.with_suffix(".tmp")
                torch.save(bank, temp)
                temp.replace(path)
            self.loaded[n] = (chunk, bank)
        gen = torch.Generator().manual_seed(829201 + step)
        ix = torch.randperm(CHUNK, generator=gen)[:BATCH]
        self.seen[n][chunk * CHUNK + ix] = True
        batch = _take(self.loaded[n][1], ix)
        return n, {key: val.to(device) for key, val in batch.items()}
