#!/usr/bin/env python3
"""
Phase calculation program for the 3-qubit entanglement workflow.
"""

import argparse
import hashlib
import json
import math
import os

import numpy as np

from basis_configs import ensure_basis_config
from matrix_utils import ensure_phase_calibration, ensure_text_matrix, ensure_w_ch1_global_amp_matrix
from system_specs import SYSTEM_SPECS as _SPECS

_SPEC_3Q = _SPECS["3qubit"]

FREQ_STEPS_3Q = _SPEC_3Q["freq_steps"]
GROUP_SIZE_3Q = _SPEC_3Q["group_size"]
GROUPS_PER_ROW_3Q = _SPEC_3Q["groups_per_row"]
CORE_INDICES_3Q = _SPEC_3Q["core_indices"]
X_CHANNELS_3Q = _SPEC_3Q["x_channels"]
PHASE_GENERATION_MANIFEST_3Q = "phase_cal_manifest.json"


def save_matrix(matrix, file_path):
    np.savetxt(file_path, matrix, fmt="%.6f")
    print(f"Matrix saved to {file_path}")


def _state_hash(schema, system_type, amplitudes, phases_rad, excitation_mode):
    payload = {
        "schema": str(schema),
        "system_type": str(system_type),
        "amplitudes": [float(value) for value in amplitudes],
        "phases_rad": [float(value) for value in phases_rad],
        "excitation_mode": int(excitation_mode),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_target_control(args, cell_count):
    legacy_amplitudes = [getattr(args, f"user_amp{index}") for index in range(1, cell_count + 1)]
    legacy_phases = [getattr(args, f"user_phase{index}") for index in range(1, cell_count + 1)]
    target_amplitudes = [
        getattr(args, f"target_amp{index}")
        if getattr(args, f"target_amp{index}") is not None else legacy_amplitudes[index - 1]
        for index in range(1, cell_count + 1)
    ]
    target_phases = [
        getattr(args, f"target_phase{index}")
        if getattr(args, f"target_phase{index}") is not None else legacy_phases[index - 1]
        for index in range(1, cell_count + 1)
    ]
    command_amplitudes = [
        getattr(args, f"command_amp{index}")
        if getattr(args, f"command_amp{index}") is not None else legacy_amplitudes[index - 1]
        for index in range(1, cell_count + 1)
    ]
    command_phases = [
        getattr(args, f"command_phase{index}")
        if getattr(args, f"command_phase{index}") is not None else legacy_phases[index - 1]
        for index in range(1, cell_count + 1)
    ]
    if any(value is None for value in target_amplitudes + target_phases + command_amplitudes + command_phases):
        raise ValueError("provide complete legacy user_* values or complete target_*/command_* values")
    return (
        [float(value) for value in target_amplitudes],
        [float(value) for value in target_phases],
        [float(value) for value in command_amplitudes],
        [float(value) for value in command_phases],
    )


def write_phase_generation_manifest(
    output_dir,
    *,
    target_amplitudes,
    target_phases_rad,
    commanded_amplitudes,
    commanded_phases_rad,
    excitation_mode,
    target_hash="",
    control_hash="",
):
    directory = os.path.abspath(output_dir)
    files = {}
    for name in ("W_CH1_amp.txt", "W_CH1_phase.txt", "R_CH1_amp.txt", "R_CH1_phase.txt"):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing generated matrix: {path}")
        with open(path, "rb") as handle:
            files[name] = hashlib.sha256(handle.read()).hexdigest()
    resolved_target_hash = target_hash or _state_hash(
        "TargetStateSpecV1", "3qubit", target_amplitudes, target_phases_rad, excitation_mode,
    )
    resolved_control_hash = control_hash or _state_hash(
        "ControlStateSpecV1", "3qubit", commanded_amplitudes, commanded_phases_rad, excitation_mode,
    )
    payload = {
        "schema": "PhaseCalculation3QubitManifestV1",
        "schema_version": 1,
        "system_type": "3qubit",
        "target_amplitudes": [float(value) for value in target_amplitudes],
        "target_phases_rad": [float(value) for value in target_phases_rad],
        "commanded_amplitudes": [float(value) for value in commanded_amplitudes],
        "commanded_phases_rad": [float(value) for value in commanded_phases_rad],
        "target_hash": str(resolved_target_hash),
        "control_hash": str(resolved_control_hash),
        "matrix_files": files,
        "matrix_bundle_hash": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "excitation_mode": int(excitation_mode),
    }
    destination = os.path.join(directory, PHASE_GENERATION_MANIFEST_3Q)
    temporary = destination + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, destination)
    return payload


def calculate_rho(amplitudes, phases, mode=1):
    rho = np.zeros((8, 8), dtype=complex)
    state_indices = [4, 2, 1]
    alpha = amplitudes * np.exp(1j * phases)

    if mode == 1:
        psi = np.zeros(8, dtype=complex)
        norm = np.linalg.norm(alpha)
        if norm > 0:
            for idx, basis_index in enumerate(state_indices):
                psi[basis_index] = alpha[idx] / norm
        rho = np.outer(psi, psi.conj())
    elif mode == 2:
        psi = np.zeros(8, dtype=complex)
        alpha_pair = alpha[:2]
        norm = np.linalg.norm(alpha_pair)
        if norm > 0:
            psi[4] = alpha[0] / norm
            psi[2] = alpha[1] / norm
        rho = np.outer(psi, psi.conj())
    elif mode == 3:
        norm_sq = float(np.sum(np.abs(alpha) ** 2))
        if norm_sq > 0:
            for idx, basis_index in enumerate(state_indices):
                rho[basis_index, basis_index] = (np.abs(alpha[idx]) ** 2) / norm_sq

    return np.real(rho), np.imag(rho)


def calculate_rho_3x3_core(amplitudes, phases, mode=1):
    re_rho, im_rho = calculate_rho(amplitudes, phases, mode)
    return re_rho[np.ix_(CORE_INDICES_3Q, CORE_INDICES_3Q)], im_rho[np.ix_(CORE_INDICES_3Q, CORE_INDICES_3Q)]


def build_r_ch1_matrices_from_config(config):
    amp_matrix = np.zeros((FREQ_STEPS_3Q, FREQ_STEPS_3Q))
    phase_matrix = np.zeros((FREQ_STEPS_3Q, FREQ_STEPS_3Q))

    entries = config["entries"]
    for op_index, entry in enumerate(entries[:75]):
        base_col = (op_index * GROUP_SIZE_3Q) % FREQ_STEPS_3Q
        base_row = (op_index * GROUP_SIZE_3Q) // FREQ_STEPS_3Q
        amp_values = np.asarray(entry["amp"], dtype=float)
        phase_values = np.asarray(entry["phase"], dtype=float)

        if base_col <= FREQ_STEPS_3Q - GROUP_SIZE_3Q:
            for offset in range(GROUP_SIZE_3Q):
                amp_matrix[base_row, base_col + offset] = amp_values[offset]
                phase_matrix[base_row, base_col + offset] = phase_values[offset]
        else:
            first_chunk = FREQ_STEPS_3Q - base_col
            for offset in range(first_chunk):
                amp_matrix[base_row, base_col + offset] = amp_values[offset]
                phase_matrix[base_row, base_col + offset] = phase_values[offset]
            for offset in range(first_chunk, GROUP_SIZE_3Q):
                amp_matrix[base_row + 1, offset - first_chunk] = amp_values[offset]
                phase_matrix[base_row + 1, offset - first_chunk] = phase_values[offset]

    return amp_matrix, phase_matrix


def generate_w_ch1_matrices(user_amp_values, user_phase_values, phase_cali_matrix, w_ch3_phase_matrix, r_ch3_phase_matrix, global_amp_matrix):
    amp_matrix = np.zeros((FREQ_STEPS_3Q, FREQ_STEPS_3Q))
    phase_matrix = np.zeros((FREQ_STEPS_3Q, FREQ_STEPS_3Q))

    for row in range(FREQ_STEPS_3Q):
        for group in range(GROUPS_PER_ROW_3Q):
            group_gain = global_amp_matrix[row, group]
            col_start = group * GROUP_SIZE_3Q
            for offset in range(GROUP_SIZE_3Q):
                col = col_start + offset
                amp_matrix[row, col] = user_amp_values[offset] * group_gain
                phase_matrix[row, col] = (
                    phase_cali_matrix[row, col]
                    #+ user_amp_values[offset]
                    + user_phase_values[offset]
                    - (w_ch3_phase_matrix[row, col] - r_ch3_phase_matrix[row, col])
                )

    return amp_matrix, phase_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate phase and amplitude matrices for the 3-qubit workflow")
    parser.add_argument("--w_ch3_amp", required=True)
    parser.add_argument("--w_ch3_phase", required=True)
    parser.add_argument("--w_ch4_amp", required=True)
    parser.add_argument("--w_ch4_phase", required=True)
    parser.add_argument("--r_ch3_amp", required=True)
    parser.add_argument("--r_ch3_phase", required=True)
    parser.add_argument("--r_ch4_amp", required=True)
    parser.add_argument("--r_ch4_phase", required=True)
    parser.add_argument("--phase_cali", required=True)
    parser.add_argument("--basis_config", default=None, help="Path to the 3-qubit basis JSON config")
    for prefix in ("user_amp", "user_phase", "target_amp", "target_phase", "command_amp", "command_phase"):
        for index in range(1, 4):
            parser.add_argument(f"--{prefix}{index}", type=float, default=None)
    parser.add_argument("--target_hash", default="")
    parser.add_argument("--control_hash", default="")
    parser.add_argument("--excitation_mode", type=int, default=1)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--w_ch1_global_amp", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    basis_config_path = args.basis_config or os.path.join(args.output_dir, "basis_config_3qubit.json")
    basis_config = ensure_basis_config(basis_config_path, "3qubit")

    fixed_inputs = {
        "W_CH3_amp": ensure_text_matrix(args.w_ch3_amp, (FREQ_STEPS_3Q, FREQ_STEPS_3Q), 1.0),
        "W_CH3_phase": ensure_text_matrix(args.w_ch3_phase, (FREQ_STEPS_3Q, FREQ_STEPS_3Q), 0.0),
        "W_CH4_amp": ensure_text_matrix(args.w_ch4_amp, (FREQ_STEPS_3Q, FREQ_STEPS_3Q), 1.0),
        "W_CH4_phase": ensure_text_matrix(args.w_ch4_phase, (FREQ_STEPS_3Q, FREQ_STEPS_3Q), 0.0),
        "R_CH3_amp": ensure_text_matrix(args.r_ch3_amp, (FREQ_STEPS_3Q, FREQ_STEPS_3Q), 1.0),
        "R_CH3_phase": ensure_text_matrix(args.r_ch3_phase, (FREQ_STEPS_3Q, FREQ_STEPS_3Q), 0.0),
        "R_CH4_amp": ensure_text_matrix(args.r_ch4_amp, (FREQ_STEPS_3Q, FREQ_STEPS_3Q), 1.0),
        "R_CH4_phase": ensure_text_matrix(args.r_ch4_phase, (FREQ_STEPS_3Q, FREQ_STEPS_3Q), 0.0),
    }
    phase_cali = ensure_phase_calibration(args.phase_cali, (FREQ_STEPS_3Q, FREQ_STEPS_3Q))
    global_amp_path = args.w_ch1_global_amp or os.path.join(args.output_dir, "W_CH1_global_amp.txt")
    global_amp_matrix = ensure_w_ch1_global_amp_matrix(global_amp_path, FREQ_STEPS_3Q, GROUPS_PER_ROW_3Q)

    target_amp_values, target_phase_values, command_amp_values, command_phase_values = _resolve_target_control(args, 3)

    r_ch1_amp, r_ch1_phase = build_r_ch1_matrices_from_config(basis_config)
    save_matrix(r_ch1_amp, os.path.join(args.output_dir, "R_CH1_amp.txt"))
    save_matrix(r_ch1_phase, os.path.join(args.output_dir, "R_CH1_phase.txt"))

    w_ch1_amp, w_ch1_phase = generate_w_ch1_matrices(
        command_amp_values,
        command_phase_values,
        phase_cali,
        fixed_inputs["W_CH3_phase"],
        fixed_inputs["R_CH3_phase"],
        global_amp_matrix,
    )
    save_matrix(w_ch1_amp, os.path.join(args.output_dir, "W_CH1_amp.txt"))
    save_matrix(w_ch1_phase, os.path.join(args.output_dir, "W_CH1_phase.txt"))

    re_rho, im_rho = calculate_rho(np.asarray(target_amp_values), np.asarray(target_phase_values), args.excitation_mode)
    re_rho_3x3, im_rho_3x3 = calculate_rho_3x3_core(
        np.asarray(target_amp_values),
        np.asarray(target_phase_values),
        args.excitation_mode,
    )

    save_matrix(re_rho, os.path.join(args.output_dir, "re_rho.txt"))
    save_matrix(im_rho, os.path.join(args.output_dir, "im_rho.txt"))
    save_matrix(re_rho_3x3, os.path.join(args.output_dir, "re_rho_3x3.txt"))
    save_matrix(im_rho_3x3, os.path.join(args.output_dir, "im_rho_3x3.txt"))

    phase_generation_manifest = write_phase_generation_manifest(
        args.output_dir,
        target_amplitudes=target_amp_values,
        target_phases_rad=target_phase_values,
        commanded_amplitudes=command_amp_values,
        commanded_phases_rad=command_phase_values,
        excitation_mode=args.excitation_mode,
        target_hash=args.target_hash,
        control_hash=args.control_hash,
    )

    density_data = {
        "system_type": "3qubit",
        "basis_config_path": basis_config_path,
        "x_phase_channels": sorted(X_CHANNELS_3Q),
        "re_rho": re_rho.tolist(),
        "im_rho": im_rho.tolist(),
        "re_rho_3x3": re_rho_3x3.tolist(),
        "im_rho_3x3": im_rho_3x3.tolist(),
        "excitation_mode": args.excitation_mode,
        "user_amplitudes": command_amp_values,
        "user_phases_degrees": [math.degrees(value) for value in command_phase_values],
        "target_amplitudes": target_amp_values,
        "target_phases_rad": target_phase_values,
        "commanded_amplitudes": command_amp_values,
        "commanded_phases_rad": command_phase_values,
        "target_hash": phase_generation_manifest["target_hash"],
        "control_hash": phase_generation_manifest["control_hash"],
        "phase_generation_manifest": PHASE_GENERATION_MANIFEST_3Q,
    }
    with open(os.path.join(args.output_dir, "density_matrix.json"), "w", encoding="utf-8") as handle:
        json.dump(density_data, handle, indent=2)

    print("Phase calculation completed successfully!")
    print(f"Generated matrices saved to {args.output_dir}")


if __name__ == "__main__":
    main()
