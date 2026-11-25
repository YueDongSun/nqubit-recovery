import os
from pathlib import Path

import numpy as np


def _strict_loading_enabled():
    return os.environ.get("NQUBIT_STRICT_MATRIX_LOADING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_text_matrix(path, shape, fill_value=0.0):
    """Load a text matrix, creating a default one if missing or invalid."""
    _ensure_parent(path)
    expected_shape = tuple(shape)

    if os.path.exists(path):
        try:
            matrix = np.loadtxt(path)
            if matrix.shape == expected_shape:
                return matrix
        except Exception:
            pass

    if _strict_loading_enabled():
        raise FileNotFoundError(
            f"strict matrix loading rejected missing or malformed matrix: {path}; expected shape {expected_shape}"
        )
    matrix = np.full(expected_shape, fill_value, dtype=float)
    np.savetxt(path, matrix, fmt="%.6f")
    return matrix


def ensure_channel_matrices(base_dir, suffix="", freq_steps=15):
    """Ensure all 8 AWG channel amp/phase matrices exist and return their paths."""
    matrix_dir = Path(base_dir) / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    channels = ["W_CH1", "W_CH2", "W_CH3", "W_CH4", "R_CH1", "R_CH2", "R_CH3", "R_CH4"]
    created = {}
    for channel in channels:
        amp_path = matrix_dir / f"{channel}_amp{suffix}.txt"
        phase_path = matrix_dir / f"{channel}_phase{suffix}.txt"
        ensure_text_matrix(str(amp_path), (freq_steps, freq_steps), 1.0)
        ensure_text_matrix(str(phase_path), (freq_steps, freq_steps), 0.0)
        created[channel] = {"amp": str(amp_path), "phase": str(phase_path)}
    return created


def ensure_phase_calibration(path, shape):
    return ensure_text_matrix(path, shape, 0.0)


def ensure_w_ch1_global_amp_matrix(path, rows, cols):
    return ensure_text_matrix(path, (rows, cols), 1.0)


def ensure_text_matrix_rectangular(path, rows, cols, fill_value=0.0):
    """Load a rectangular matrix (rows × cols), creating default if missing."""
    _ensure_parent(path)
    expected_shape = (rows, cols)

    if os.path.exists(path):
        try:
            matrix = np.loadtxt(path)
            if matrix.shape == expected_shape:
                return matrix
        except Exception:
            pass

    if _strict_loading_enabled():
        raise FileNotFoundError(
            f"strict matrix loading rejected missing or malformed matrix: {path}; expected shape {expected_shape}"
        )
    matrix = np.full(expected_shape, fill_value, dtype=float)
    np.savetxt(path, matrix, fmt="%.6f")
    return matrix


def apply_additive_delta(matrix, delta_path, limit=None):
    """Return matrix plus optional ML delta without modifying the source file."""
    matrix = np.asarray(matrix, dtype=float)
    if not os.path.exists(delta_path):
        return matrix

    try:
        delta = np.loadtxt(delta_path)
    except Exception as exc:
        if _strict_loading_enabled():
            raise ValueError(f"strict matrix loading rejected malformed delta {delta_path}: {exc}") from exc
        return matrix

    if delta.shape != matrix.shape:
        if _strict_loading_enabled():
            raise ValueError(
                f"strict matrix loading rejected delta {delta_path} shape {delta.shape}; expected {matrix.shape}"
            )
        return matrix
    if limit is not None:
        delta = np.clip(delta, -limit, limit)
    return matrix + delta


def ensure_y_channel_matrices(
    base_dir,
    suffix="",
    x_freq_steps=15,
    y_freq_steps=15,
    ops_per_row=None,
    matrix_dir=None,
):
    """Ensure matrices for X (square) and Y (rectangular) channels separately.

    Args:
        base_dir: Base directory for matrix files
        suffix: File name suffix (e.g., "_4qubit")
        x_freq_steps: Frequency steps for X channels (square matrix dimension)
        y_freq_steps: Frequency steps for Y channels (rows)
        ops_per_row: Number of operation points per row for Y channels (columns)
    """
    matrix_root = Path(matrix_dir) if matrix_dir is not None else Path(base_dir) / "matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)

    x_channels = ["W_CH1", "W_CH3", "R_CH1", "R_CH3"]
    y_channels = ["W_CH2", "W_CH4", "R_CH2", "R_CH4"]

    created = {}

    for channel in x_channels:
        amp_path = matrix_root / f"{channel}_amp{suffix}.txt"
        phase_path = matrix_root / f"{channel}_phase{suffix}.txt"
        ensure_text_matrix_rectangular(str(amp_path), x_freq_steps, x_freq_steps, 1.0)
        ensure_text_matrix_rectangular(str(phase_path), x_freq_steps, x_freq_steps, 0.0)
        created[channel] = {"amp": str(amp_path), "phase": str(phase_path)}

    if ops_per_row is None:
        ops_per_row = x_freq_steps

    for channel in y_channels:
        amp_path = matrix_root / f"{channel}_amp{suffix}.txt"
        phase_path = matrix_root / f"{channel}_phase{suffix}.txt"
        ensure_text_matrix_rectangular(str(amp_path), y_freq_steps, ops_per_row, 1.0)
        ensure_text_matrix_rectangular(str(phase_path), y_freq_steps, ops_per_row, 0.0)
        created[channel] = {"amp": str(amp_path), "phase": str(phase_path)}

    return created


def ensure_y_channel_cycle_matrices(
    channel,
    base_dir,
    suffix,
    cycle_count,
    y_freq_steps,
    ops_per_row,
    matrix_dir=None,
):
    """Ensure cycle-specific rectangular matrices for Y channels (R_CH2 for 4-qubit)."""
    matrix_root = Path(matrix_dir) if matrix_dir is not None else Path(base_dir) / "matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)

    matrices = []
    for cycle in range(1, cycle_count + 1):
        amp_path = matrix_root / f"{channel}_amp_{cycle}{suffix}.txt"
        phase_path = matrix_root / f"{channel}_phase_{cycle}{suffix}.txt"
        matrices.append({
            "amp": ensure_text_matrix_rectangular(str(amp_path), y_freq_steps, ops_per_row, 1.0),
            "phase": ensure_text_matrix_rectangular(str(phase_path), y_freq_steps, ops_per_row, 0.0)
        })
    return matrices
