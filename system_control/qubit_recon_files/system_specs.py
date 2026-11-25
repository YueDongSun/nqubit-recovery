"""Centralized system specifications for 3-qubit and 4-qubit W-state experiments.

All physical and sampling constants that were previously scattered across
entangle_sequence.py, entangle_sequence_4qubit.py, fourier_sequence_guard.py,
phase_cal.py, phase_cal_4qubit.py, calibrated_density_reconstruction.py,
delta_manager.py, density_reconstruction.py, and runner.py are unified here.

Usage:
    from system_specs import SYSTEM_SPECS, get_spec

    spec = get_spec("3qubit")
    n_steps = spec["freq_steps"]          # 15
    row_dur = spec["op_duration_ns"]      # 3000
"""

from typing import Dict, Any

SYSTEM_SPECS: Dict[str, Dict[str, Any]] = {
    "3qubit": {
        "freq_steps": 15,
        "groups_per_row": 5,
        "group_size": 3,
        "total_operations": 75,
        "op_duration_ns": 3000,
        "sampling_rate": 1e9,
        "freq_step_mhz": 2.5,
        "freq_start_mhz": 85.0,
        "freq_end_mhz": 120.0,
        "core_dim": 3,
        "full_dim": 8,
        "full_indices": [4, 2, 1],
        "core_indices": [1, 2, 4],
        "entries_per_cycle": 75,
        "efficiency_shape": (15, 15),
        "num_cells": 75,
        "x_channels": {"W_CH1", "W_CH3", "R_CH1", "R_CH3"},
        "y_channels": {"W_CH2", "W_CH4", "R_CH2", "R_CH4"},
    },
    "4qubit": {
        "freq_steps": 16,
        "groups_per_row": 4,
        "group_size": 4,
        "total_operations": 256,
        "ops_per_cycle": 64,
        "total_cycles": 4,
        "op_duration_ns": 4000,
        "cycle_active_ns": 64 * 4000,
        "cycle_gap_ns": 0,
        "sampling_rate": 1e9,
        "freq_step_mhz": 2.5,
        "freq_start_mhz": 85.0,
        "freq_end_mhz": 122.5,
        "core_dim": 4,
        "full_dim": 16,
        "full_indices": [8, 4, 2, 1],
        "core_indices": [1, 2, 4, 8],
        "entries_per_cycle": 64,
        "efficiency_shape": (16, 16),
        "num_cells": 256,
        "x_channels": {"W_CH1", "W_CH3", "R_CH1", "R_CH3"},
        "y_channels": {"W_CH2", "W_CH4", "R_CH2", "R_CH4"},
    },
}

_VALID_TYPES = set(SYSTEM_SPECS.keys())


def get_spec(system_type: str) -> Dict[str, Any]:
    if system_type not in _VALID_TYPES:
        raise ValueError(f"system_type must be one of {_VALID_TYPES}, got {system_type!r}")
    return SYSTEM_SPECS[system_type]


def freq_steps(system_type: str) -> int:
    return get_spec(system_type)["freq_steps"]


def groups_per_row(system_type: str) -> int:
    return get_spec(system_type)["groups_per_row"]


def group_size(system_type: str) -> int:
    return get_spec(system_type)["group_size"]