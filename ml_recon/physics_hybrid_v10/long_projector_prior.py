"""Command shrinkage and pair-frequency reference from unique, actually visited TRAIN rows."""

from functools import lru_cache

import numpy as np
import torch

from .long_projector_data import BATCH, CHUNK, REUSE, SIZES, bank_seed
from .training_ideas_data import _states


@lru_cache(maxsize=512)
def _prior_terms(n, chunk, split_namespace):
    rng = np.random.Generator(
        np.random.Philox(bank_seed(n, "train", chunk, split_namespace))
    )
    command, rho, classes, partition, _, _ = _states(n, CHUNK, rng)
    off = ~np.eye(n, dtype=bool)
    prior = (command[:, :, None] * command[:, None, :].conj())[:, off]
    return (
        (prior.conj() * rho[:, off]).real.sum(-1),
        (abs(prior) ** 2).sum(-1),
        partition[:, off].sum(-1),
        classes,
    )


def prior_for_seen(n, seen, split_namespace=None):
    mask = np.asarray(seen, dtype=bool)
    if mask.ndim != 1 or mask.size % CHUNK:
        raise ValueError("TRAIN visit mask must contain complete chunks")
    num = den = pair_sum = 0.0
    count = int(mask.sum())
    for chunk, valid in enumerate(mask.reshape(-1, CHUNK)):
        if valid.any():
            numerators, denominators, pairs, _ = _prior_terms(n, chunk, split_namespace)
            num += numerators[valid].sum()
            den += denominators[valid].sum()
            pair_sum += pairs[valid].sum()
    if not count or den <= 0:
        raise ValueError("candidate has no observed TRAIN prior")
    # Constrained least squares on the convex command-dephasing family.
    return (
        float(np.clip(num / den, 0, 1)),
        float(pair_sum / (count * n * (n - 1))),
        count,
    )


def training_seen_mask(n, rows, step):
    """Reconstruct the unique TRAIN identities visited through this optimizer update."""
    if rows < CHUNK or rows % CHUNK or step < 0:
        raise ValueError("invalid TRAIN bank size or optimizer step")
    seen = torch.zeros(rows, dtype=torch.bool)
    offset = SIZES.index(n)
    for i in range(offset, step, len(SIZES)):
        chunk = ((i // len(SIZES)) // REUSE) % (rows // CHUNK)
        sub = seen[chunk * CHUNK : (chunk + 1) * CHUNK]
        if not sub.all():
            gen = torch.Generator().manual_seed(829201 + i)
            sub[torch.randperm(CHUNK, generator=gen)[:BATCH]] = True
        if i % 256 == offset and seen.all():
            break
    return seen


def actual_training_prior(n, rows, step, split_namespace=None):
    """Fit the prior from this candidate's actual registered sample visits."""
    return prior_for_seen(n, training_seen_mask(n, rows, step), split_namespace)


@lru_cache(maxsize=256)
def visited_noise_class_counts(n, rows, step, split_namespace):
    """Count noise-labeled unique TRAIN states actually used at this checkpoint."""
    from .training_ideas import NOISE_CLASS_NAMES

    seen = training_seen_mask(n, rows, step).numpy().reshape(-1, CHUNK)
    counts = np.zeros(len(NOISE_CLASS_NAMES), dtype=np.int64)
    for chunk, active in enumerate(seen):
        if active.any():
            classes = _prior_terms(n, chunk, split_namespace)[3]
            counts += np.bincount(classes[active & (classes >= 0)], minlength=len(counts))
    return tuple(int(value) for value in counts)
