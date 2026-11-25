"""Numerically stable candidate response forms for AOD power compression."""

from __future__ import annotations

from typing import Any

import numpy as np


EPSILON = 1e-12


def rational_compression(power: Any, scale: float = 1.0) -> np.ndarray:
    value = np.maximum(np.asarray(power, dtype=float), 0.0)
    scale = max(float(scale), EPSILON)
    return value / (1.0 + value / scale)


def exponential_compression(power: Any, scale: float = 1.0) -> np.ndarray:
    value = np.maximum(np.asarray(power, dtype=float), 0.0)
    scale = max(float(scale), EPSILON)
    return scale * (1.0 - np.exp(-value / scale))


def hill_compression(
    power: Any,
    half_saturation: float = 1.0,
    exponent: float = 2.0,
) -> np.ndarray:
    value = np.maximum(np.asarray(power, dtype=float), 0.0)
    half = max(float(half_saturation), EPSILON)
    exponent = max(float(exponent), EPSILON)
    numerator = np.power(value, exponent)
    return numerator / (np.power(half, exponent) + numerator + EPSILON)


def cubic_spline_hinges(value: Any, knots: Any) -> np.ndarray:
    x = np.asarray(value, dtype=float).reshape(-1)
    knot_values = np.asarray(knots, dtype=float).reshape(-1)
    if knot_values.size == 0:
        return np.zeros((x.size, 0), dtype=float)
    return np.stack([np.maximum(x - knot, 0.0) ** 3 for knot in knot_values], axis=-1)


__all__ = [
    "cubic_spline_hinges",
    "exponential_compression",
    "hill_compression",
    "rational_compression",
]
