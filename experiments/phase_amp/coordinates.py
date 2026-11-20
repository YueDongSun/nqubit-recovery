"""Centered log-amplitude and wrapped relative-phase coordinates for synthetic W preparation."""

import numpy as np


def centered(x):
    return x - x.mean(axis=-2, keepdims=True)


def error_sizes(coords):
    n = coords.shape[-2]
    i, j = np.triu_indices(n, 1)
    amplitude = coords[..., 0] - coords[..., 0].mean(axis=-1, keepdims=True)
    phi = coords[..., 1]
    difference = phi[..., i] - phi[..., j]
    wrapped = np.arctan2(np.sin(difference), np.cos(difference))
    a = np.sqrt(np.mean(amplitude**2, axis=-1))
    p = np.sqrt(np.mean(wrapped**2, axis=-1) * (n - 1) / (2 * n))
    return a, p


def fidelity(coords):
    v = np.exp(coords[..., 0] + 1j * coords[..., 1])
    return np.abs(v.sum(-1)) ** 2 / (coords.shape[-2] * (np.abs(v) ** 2).sum(-1))


def primary_initial(episode_ids, n, spread, drift):
    bias = []
    velocity = []
    for start in range(0, len(episode_ids), 64):
        group = [str(x).split("_") for x in episode_ids[start : start + 64]]
        seed = int(group[0][1])
        size = len(group)
        if any(
            tag != f"n{n}" or int(s) != seed or int(k) != i
            for i, (tag, s, k) in enumerate(group)
        ):
            raise ValueError("trajectory IDs do not bind the generator batch")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 17]))
        initial = (
            centered(rng.normal(size=(size, n, 2)) * np.array([0.30, 0.40])) * spread
        )
        v = (
            centered(rng.normal(size=(size, n, 2)) * np.array([0.05, 0.08]))
            if drift
            else np.zeros_like(initial)
        )
        bias.append(initial)
        velocity.append(v)
    return np.concatenate(bias), np.concatenate(velocity)


def grid_coords(n, rows, seed, amp, phase):
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 739]))
    directions = centered(rng.normal(size=(rows, n, 2)))
    norm = np.sqrt(np.mean(directions**2, axis=1, keepdims=True))
    if np.any(norm == 0):
        raise ValueError("degenerate random direction")
    result = directions / norm * np.array([amp, phase])[None, None, :]
    a, p = error_sizes(result)
    if not np.allclose(a, amp, rtol=1e-11, atol=1e-12) or not np.allclose(
        p, phase, rtol=1e-11, atol=1e-12
    ):
        raise ValueError(
            "realized gauge-invariant RMS differs from registered grid point"
        )
    return result
