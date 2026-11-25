"""Measurement helpers that preserve global outcomes and derive marginals from them."""

from __future__ import annotations

import numpy as np


def outcome_histogram(outcomes: np.ndarray, N: int) -> np.ndarray:
    sampled = np.asarray(outcomes, dtype=np.int8)
    if sampled.ndim != 2 or sampled.shape[1] != N:
        raise ValueError("global outcomes must have shape [shots,N]")
    bits = (sampled > 0).astype(np.int64)
    powers = 1 << np.arange(N - 1, -1, -1)
    indices = bits @ powers
    return np.bincount(indices, minlength=2 ** N)


def pair_marginal_from_global(outcomes: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    return subset_marginal_from_global(outcomes, pair)


def subset_marginal_from_global(outcomes: np.ndarray,
                                subset: tuple[int, ...]) -> np.ndarray:
    sampled = np.asarray(outcomes, dtype=np.int8)
    if sampled.ndim != 2 or not 1 <= len(subset) <= 4 \
            or len(set(subset)) != len(subset) \
            or min(subset) < 0 or max(subset) >= sampled.shape[1]:
        raise ValueError("registered subset must contain 1-4 unique in-range qubits")
    bits = (sampled[:, subset] > 0).astype(np.int64)
    powers = 1 << np.arange(len(subset) - 1, -1, -1)
    indices = bits @ powers
    return np.bincount(indices, minlength=2 ** len(subset))


def observed_correlation_adjacency(outcome_panels: list[np.ndarray], N: int) -> np.ndarray:
    """Build a deployable correlation graph from noisy sampled outcomes only."""
    if not outcome_panels:
        raise ValueError("correlation graph requires sampled outcomes")
    accumulator = np.zeros((N, N), dtype=np.float64)
    observed_panel_count = 0
    for panel in outcome_panels:
        sampled = np.asarray(panel, dtype=np.float64)
        if sampled.ndim != 2 or sampled.shape[1] != N:
            raise ValueError("outcome panel shape disagrees with N")
        if sampled.shape[0] == 0:
            continue
        mean = sampled.mean(axis=0)
        connected = sampled.T @ sampled / sampled.shape[0] - np.outer(mean, mean)
        accumulator += np.abs(connected)
        observed_panel_count += 1
    if observed_panel_count == 0:
        return np.zeros((N, N), dtype=np.float32)
    correlation = accumulator / observed_panel_count
    np.fill_diagonal(correlation, 0.0)
    maximum = float(correlation.max(initial=0.0))
    if maximum > 0.0:
        correlation /= maximum
    return correlation.astype(np.float32)
