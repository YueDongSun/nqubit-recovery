"""Helpers for the 3-qubit W_CH1 15x5 global amplitude matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_W_CH1_GLOBAL_AMP_SHAPE = (15, 5)


def load_w_ch1_global_amp_matrix(
    path: str | Path,
    shape: tuple[int, int] = DEFAULT_W_CH1_GLOBAL_AMP_SHAPE,
    default: float = 1.0,
) -> np.ndarray:
    """Load a W_CH1 global amplitude matrix, falling back to a default matrix."""

    fallback = np.full(shape, float(default), dtype=float)
    try:
        matrix = np.loadtxt(Path(path), dtype=float)
    except (OSError, ValueError):
        return fallback

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        return fallback
    return matrix


def save_w_ch1_global_amp_matrix(path: str | Path, matrix: np.ndarray) -> None:
    """Persist a W_CH1 global amplitude matrix using the project text format."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != DEFAULT_W_CH1_GLOBAL_AMP_SHAPE:
        raise ValueError(
            f"W_CH1 global amplitude matrix must have shape "
            f"{DEFAULT_W_CH1_GLOBAL_AMP_SHAPE}, got {matrix.shape}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.6f")
