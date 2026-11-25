"""Helmert/Aitchison coordinates for the 2N-2 W control tangent."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def helmert_matrix(N: int) -> np.ndarray:
    if N < 2:
        raise ValueError("Helmert coordinates require N >= 2")
    h = np.zeros((N - 1, N), dtype=np.float64)
    for i in range(1, N):
        h[i - 1, :i] = 1.0 / np.sqrt(i * (i + 1.0))
        h[i - 1, i] = -i / np.sqrt(i * (i + 1.0))
    return h


def aitchison_amplitude_coordinates(amplitudes: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if np.any(a < 0.0) or np.linalg.norm(a) <= eps:
        raise ValueError("amplitudes must be non-negative and non-zero")
    return helmert_matrix(len(a)) @ np.log(a + eps)


def recover_amplitudes(coordinates: np.ndarray) -> np.ndarray:
    xi = np.asarray(coordinates, dtype=np.float64).reshape(-1)
    raw = np.exp(helmert_matrix(len(xi) + 1).T @ xi)
    return raw / np.linalg.norm(raw)


def phase_coordinates(phases: np.ndarray) -> np.ndarray:
    phi = np.asarray(phases, dtype=np.float64).reshape(-1)
    return helmert_matrix(len(phi)) @ phi


def recover_phases(coordinates: np.ndarray) -> np.ndarray:
    xi = np.asarray(coordinates, dtype=np.float64).reshape(-1)
    phases = helmert_matrix(len(xi) + 1).T @ xi
    phases = np.angle(np.exp(1j * phases))
    return np.angle(np.exp(1j * (phases - phases[0])))


def circular_phase_loss(prediction: np.ndarray, truth: np.ndarray) -> float:
    delta = np.asarray(prediction) - np.asarray(truth)
    return float(np.mean(1.0 - np.cos(delta)))


def tangent_dimension(N: int) -> int:
    return 2 * N - 2


def apply_tangent_action(coefficients: np.ndarray, delta_amplitude: np.ndarray,
                         delta_phase: np.ndarray) -> np.ndarray:
    c = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    amp = np.abs(c)
    phase = np.angle(c)
    new_amp = recover_amplitudes(aitchison_amplitude_coordinates(amp) + delta_amplitude)
    new_phase = recover_phases(phase_coordinates(phase) + delta_phase)
    return new_amp * np.exp(1j * new_phase)
