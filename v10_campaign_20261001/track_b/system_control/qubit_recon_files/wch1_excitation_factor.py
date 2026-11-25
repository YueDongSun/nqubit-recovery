"""W_CH1 point-wise excitation-rate compensation factors."""

from __future__ import annotations

import numpy as np


def build_wch1_excitation_factor_matrix(freq_steps: int) -> np.ndarray:
    """Return the W_CH1 excitation compensation factor matrix.

    The 16x16 source model mirrors ``system_control/awg256seq.py``.  The
    3-qubit sequence uses the matching 15x15 subset by dropping the final row
    and column so operation-point ordering stays row/column aligned.
    """

    freq_steps = int(freq_steps)
    if freq_steps not in (15, 16):
        raise ValueError(f"freq_steps must be 15 or 16, got {freq_steps}")

    n = 16
    factors = np.zeros((n, n), dtype=float)
    for row in range(n):
        for col in range(n):
            i = row + 1
            j = col + 1
            factors[row, col] = (80 - 2.5 * i - 2.5 * j) ** 1.18 / 65
            di = (i - 12.0) / 2.5
            dj = (j - 13.0) / 1.8
            hot = np.exp(-(di * di + dj * dj))
            factors[row, col] *= 1.0 / (1.0 + 80.0 * hot)

    max_val = float(np.max(factors)) if factors.size else 0.0
    if max_val > 0.0:
        factors = factors / max_val
    factors = np.maximum(factors, 0.1)
    if freq_steps == 15:
        return factors[:15, :15].copy()
    return factors
