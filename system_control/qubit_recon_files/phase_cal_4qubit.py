#!/usr/bin/env python3
"""
Phase calculation program for the 4-qubit entanglement workflow.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from basis_configs import ensure_basis_config
from matrix_utils import ensure_phase_calibration, ensure_text_matrix, ensure_w_ch1_global_amp_matrix
from system_specs import SYSTEM_SPECS as _SPECS

_SPEC_4Q = _SPECS["4qubit"]

FREQ_STEPS_4Q = _SPEC_4Q["freq_steps"]
GROUP_SIZE_4Q = _SPEC_4Q["group_size"]
GROUPS_PER_ROW_4Q = _SPEC_4Q["groups_per_row"]
CORE_INDICES_4Q = _SPEC_4Q["core_indices"]
X_CHANNELS_4Q = _SPEC_4Q["x_channels"]
X_MATRIX_SHAPE_4Q = (FREQ_STEPS_4Q, FREQ_STEPS_4Q)
Y_MATRIX_SHAPE_4Q = (FREQ_STEPS_4Q, GROUPS_PER_ROW_4Q)
PHASE_GENERATION_MANIFEST_4Q = "phase_cal_4qubit_manifest.json"


def save_matrix(matrix, file_path):
    np.savetxt(file_path, matrix, fmt="%.6f")
    print(f"Matrix saved to {file_path}")


def write_phase_generation_manifest(
    output_dir,
    *,
    user_amplitudes,
    user_phases_rad,
    excitation_mode,
    target_amplitudes=None,
    target_phases_rad=None,
    commanded_amplitudes=None,
    commanded_phases_rad=None,
    target_hash="",
    control_hash="",
):
    """Atomically attest the final unprobed W_CH1 matrices for this target.

    The automation layer uses this identity *after* target-state phase
    calculation.  This prevents an intentional coverage-to-anchor rewrite from
    being confused with an external mutation while an ephemeral probe is being
    rendered.
    """

    directory = Path(output_dir)
    phase_files = {}
    for cycle_index in range(1, 5):
        path = directory / f"W_CH1_phase_{cycle_index}_4qubit.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing generated W_CH1 phase matrix: {path}")
        phase_files[path.name] = _sha256_file(path)
    encoded = json.dumps(
        phase_files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    base_hash = hashlib.sha256(encoded).hexdigest()
    target_amplitudes = list(user_amplitudes if target_amplitudes is None else target_amplitudes)
    target_phases_rad = list(user_phases_rad if target_phases_rad is None else target_phases_rad)
    commanded_amplitudes = list(user_amplitudes if commanded_amplitudes is None else commanded_amplitudes)
    commanded_phases_rad = list(user_phases_rad if commanded_phases_rad is None else commanded_phases_rad)
    matrix_files = dict(phase_files)
    for cycle_index in range(1, 5):
        path = directory / f"W_CH1_amp_{cycle_index}_4qubit.txt"
        if path.is_file():
            matrix_files[path.name] = _sha256_file(path)
    target_hash = target_hash or _state_hash(
        "TargetStateSpecV1", "4qubit", target_amplitudes, target_phases_rad, excitation_mode,
    )
    control_hash = control_hash or _state_hash(
        "ControlStateSpecV1", "4qubit", commanded_amplitudes, commanded_phases_rad, excitation_mode,
    )
    payload = {
        "schema": "PhaseCalculation4QubitManifestV1",
        "schema_version": 1,
        "system_type": "4qubit",
        "w_ch1_phase_files": phase_files,
        "w_ch1_base_matrix_hash": base_hash,
        "user_amplitudes": [float(value) for value in user_amplitudes],
        "user_phases_rad": [float(value) for value in user_phases_rad],
        "user_phases_degrees": [math.degrees(float(value)) for value in user_phases_rad],
        "target_amplitudes": [float(value) for value in target_amplitudes],
        "target_phases_rad": [float(value) for value in target_phases_rad],
        "commanded_amplitudes": [float(value) for value in commanded_amplitudes],
        "commanded_phases_rad": [float(value) for value in commanded_phases_rad],
        "target_hash": str(target_hash),
        "control_hash": str(control_hash),
        "matrix_files": matrix_files,
        "matrix_bundle_hash": hashlib.sha256(
            json.dumps(matrix_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "excitation_mode": int(excitation_mode),
    }
    destination = directory / PHASE_GENERATION_MANIFEST_4Q
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(f"PHASE_CAL_4Q_BASE_HASH={base_hash}")
    return payload


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    groups = []
    for prefix, legacy in (
        ("target_amp", legacy_amplitudes),
        ("target_phase", legacy_phases),
        ("command_amp", legacy_amplitudes),
        ("command_phase", legacy_phases),
    ):
        groups.append([
            getattr(args, f"{prefix}{index}")
            if getattr(args, f"{prefix}{index}") is not None else legacy[index - 1]
            for index in range(1, cell_count + 1)
        ])
    if any(value is None for values in groups for value in values):
        raise ValueError("provide complete legacy user_* values or complete target_*/command_* values")
    return tuple([float(value) for value in values] for values in groups)


def ensure_fixed_channel_matrices(args):
    """Load fixed 4Q AOD matrices using the generator's X/Y shape contract."""
    fixed_inputs = {
        "W_CH3_phase": ensure_text_matrix(args.w_ch3_phase, X_MATRIX_SHAPE_4Q, 0.0),
        "R_CH3_phase": ensure_text_matrix(args.r_ch3_phase, X_MATRIX_SHAPE_4Q, 0.0),
    }
    ensure_text_matrix(args.w_ch3_amp, X_MATRIX_SHAPE_4Q, 1.0)
    ensure_text_matrix(args.w_ch4_amp, Y_MATRIX_SHAPE_4Q, 1.0)
    ensure_text_matrix(args.w_ch4_phase, Y_MATRIX_SHAPE_4Q, 0.0)
    ensure_text_matrix(args.r_ch3_amp, X_MATRIX_SHAPE_4Q, 1.0)
    ensure_text_matrix(args.r_ch4_amp, Y_MATRIX_SHAPE_4Q, 1.0)
    ensure_text_matrix(args.r_ch4_phase, Y_MATRIX_SHAPE_4Q, 0.0)
    return fixed_inputs


def calculate_rho_4qubit(amplitudes, phases, mode=1):
    rho = np.zeros((16, 16), dtype=complex)
    state_indices = [8, 4, 2, 1]
    alpha = amplitudes * np.exp(1j * phases)

    if mode == 1:
        psi = np.zeros(16, dtype=complex)
        norm = np.linalg.norm(alpha)
        if norm > 0:
            for idx, basis_index in enumerate(state_indices):
                psi[basis_index] = alpha[idx] / norm
        rho = np.outer(psi, psi.conj())
    elif mode == 2:
        psi = np.zeros(16, dtype=complex)
        alpha_triple = alpha[:3]
        norm = np.linalg.norm(alpha_triple)
        if norm > 0:
            psi[8] = alpha[0] / norm
            psi[4] = alpha[1] / norm
            psi[2] = alpha[2] / norm
        rho = np.outer(psi, psi.conj())
    elif mode == 3:
        psi = np.zeros(16, dtype=complex)
        alpha_pair = alpha[:2]
        norm = np.linalg.norm(alpha_pair)
        if norm > 0:
            psi[8] = alpha[0] / norm
            psi[4] = alpha[1] / norm
        rho = np.outer(psi, psi.conj())
    elif mode == 4:
        norm_sq = float(np.sum(np.abs(alpha) ** 2))
        if norm_sq > 0:
            for idx, basis_index in enumerate(state_indices):
                rho[basis_index, basis_index] = (np.abs(alpha[idx]) ** 2) / norm_sq

    return np.real(rho), np.imag(rho)


def calculate_rho_4x4_core(amplitudes, phases, mode=1):
    re_rho, im_rho = calculate_rho_4qubit(amplitudes, phases, mode)
    return re_rho[np.ix_(CORE_INDICES_4Q, CORE_INDICES_4Q)], im_rho[np.ix_(CORE_INDICES_4Q, CORE_INDICES_4Q)]


def build_cycle_matrices(cycle_entries):
    amp_matrix = np.zeros((FREQ_STEPS_4Q, FREQ_STEPS_4Q))
    phase_matrix = np.zeros((FREQ_STEPS_4Q, FREQ_STEPS_4Q))

    for op_index, entry in enumerate(cycle_entries[:64]):
        base_col = (op_index * GROUP_SIZE_4Q) % FREQ_STEPS_4Q
        base_row = (op_index * GROUP_SIZE_4Q) // FREQ_STEPS_4Q
        amp_values = np.asarray(entry["amp"], dtype=float)
        phase_values = np.asarray(entry["phase"], dtype=float)

        if base_col <= FREQ_STEPS_4Q - GROUP_SIZE_4Q:
            for offset in range(GROUP_SIZE_4Q):
                amp_matrix[base_row, base_col + offset] = amp_values[offset]
                phase_matrix[base_row, base_col + offset] = phase_values[offset]
        else:
            first_chunk = FREQ_STEPS_4Q - base_col
            for offset in range(first_chunk):
                amp_matrix[base_row, base_col + offset] = amp_values[offset]
                phase_matrix[base_row, base_col + offset] = phase_values[offset]
            for offset in range(first_chunk, GROUP_SIZE_4Q):
                amp_matrix[base_row + 1, offset - first_chunk] = amp_values[offset]
                phase_matrix[base_row + 1, offset - first_chunk] = phase_values[offset]

    return amp_matrix, phase_matrix


def generate_w_ch1_matrices_4qubit(user_amp_values, user_phase_values, phase_cali_matrix, w_ch3_phase_matrix, r_ch3_phase_matrix, global_amp_matrix):
    amp_matrix = np.zeros((FREQ_STEPS_4Q, FREQ_STEPS_4Q))
    phase_matrix = np.zeros((FREQ_STEPS_4Q, FREQ_STEPS_4Q))

    for row in range(FREQ_STEPS_4Q):
        for group in range(GROUPS_PER_ROW_4Q):
            group_gain = global_amp_matrix[row, group]
            col_start = group * GROUP_SIZE_4Q
            for offset in range(GROUP_SIZE_4Q):
                col = col_start + offset
                amp_matrix[row, col] = user_amp_values[offset] * group_gain
                phase_matrix[row, col] = (
                    phase_cali_matrix[row, col]
                    # + user_amp_values[offset]
                    + user_phase_values[offset]
                    - (w_ch3_phase_matrix[row, col] - r_ch3_phase_matrix[row, col])
                )

    return amp_matrix, phase_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate phase and amplitude matrices for the 4-qubit workflow")
    parser.add_argument("--w_ch3_amp", required=True)
    parser.add_argument("--w_ch3_phase", required=True)
    parser.add_argument("--w_ch4_amp", required=True)
    parser.add_argument("--w_ch4_phase", required=True)
    parser.add_argument("--r_ch3_amp", required=True)
    parser.add_argument("--r_ch3_phase", required=True)
    parser.add_argument("--r_ch4_amp", required=True)
    parser.add_argument("--r_ch4_phase", required=True)
    parser.add_argument("--phase_cali", required=True)
    parser.add_argument("--basis_config", default=None, help="Path to the 4-qubit basis JSON config")
    for prefix in ("user_amp", "user_phase", "target_amp", "target_phase", "command_amp", "command_phase"):
        for index in range(1, 5):
            parser.add_argument(f"--{prefix}{index}", type=float, default=None)
    parser.add_argument("--target_hash", default="")
    parser.add_argument("--control_hash", default="")
    parser.add_argument("--excitation_mode", type=int, default=1)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--w_ch1_global_amp_1", default=None)
    parser.add_argument("--w_ch1_global_amp_2", default=None)
    parser.add_argument("--w_ch1_global_amp_3", default=None)
    parser.add_argument("--w_ch1_global_amp_4", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    basis_config_path = args.basis_config or os.path.join(args.output_dir, "basis_config_4qubit.json")
    basis_config = ensure_basis_config(basis_config_path, "4qubit")

    fixed_inputs = ensure_fixed_channel_matrices(args)

    phase_cali = ensure_phase_calibration(args.phase_cali, X_MATRIX_SHAPE_4Q)
    target_amp_values, target_phase_values, command_amp_values, command_phase_values = _resolve_target_control(args, 4)

    global_amp_paths = [
        args.w_ch1_global_amp_1 or os.path.join(args.output_dir, "W_CH1_global_amp_1_4qubit.txt"),
        args.w_ch1_global_amp_2 or os.path.join(args.output_dir, "W_CH1_global_amp_2_4qubit.txt"),
        args.w_ch1_global_amp_3 or os.path.join(args.output_dir, "W_CH1_global_amp_3_4qubit.txt"),
        args.w_ch1_global_amp_4 or os.path.join(args.output_dir, "W_CH1_global_amp_4_4qubit.txt"),
    ]

    for cycle_index, cycle_info in enumerate(basis_config["cycles"], start=1):
        r_ch1_amp, r_ch1_phase = build_cycle_matrices(cycle_info["entries"])
        save_matrix(r_ch1_amp, os.path.join(args.output_dir, f"R_CH1_amp_{cycle_index}_4qubit.txt"))
        save_matrix(r_ch1_phase, os.path.join(args.output_dir, f"R_CH1_phase_{cycle_index}_4qubit.txt"))

        global_amp_matrix = ensure_w_ch1_global_amp_matrix(global_amp_paths[cycle_index - 1], FREQ_STEPS_4Q, GROUPS_PER_ROW_4Q)
        w_ch1_amp, w_ch1_phase = generate_w_ch1_matrices_4qubit(
            command_amp_values,
            command_phase_values,
            phase_cali,
            fixed_inputs["W_CH3_phase"],
            fixed_inputs["R_CH3_phase"],
            global_amp_matrix,
        )
        save_matrix(w_ch1_amp, os.path.join(args.output_dir, f"W_CH1_amp_{cycle_index}_4qubit.txt"))
        save_matrix(w_ch1_phase, os.path.join(args.output_dir, f"W_CH1_phase_{cycle_index}_4qubit.txt"))

        if cycle_index == 1:
            save_matrix(r_ch1_amp, os.path.join(args.output_dir, "R_CH1_amp_4qubit.txt"))
            save_matrix(r_ch1_phase, os.path.join(args.output_dir, "R_CH1_phase_4qubit.txt"))
            save_matrix(w_ch1_amp, os.path.join(args.output_dir, "W_CH1_amp_4qubit.txt"))
            save_matrix(w_ch1_phase, os.path.join(args.output_dir, "W_CH1_phase_4qubit.txt"))

    re_rho, im_rho = calculate_rho_4qubit(np.asarray(target_amp_values), np.asarray(target_phase_values), args.excitation_mode)
    re_rho_4x4, im_rho_4x4 = calculate_rho_4x4_core(
        np.asarray(target_amp_values),
        np.asarray(target_phase_values),
        args.excitation_mode,
    )

    save_matrix(re_rho, os.path.join(args.output_dir, "re_rho_4qubit.txt"))
    save_matrix(im_rho, os.path.join(args.output_dir, "im_rho_4qubit.txt"))
    save_matrix(re_rho_4x4, os.path.join(args.output_dir, "re_rho_4x4.txt"))
    save_matrix(im_rho_4x4, os.path.join(args.output_dir, "im_rho_4x4.txt"))

    phase_generation_manifest = write_phase_generation_manifest(
        args.output_dir,
        user_amplitudes=command_amp_values,
        user_phases_rad=command_phase_values,
        target_amplitudes=target_amp_values,
        target_phases_rad=target_phase_values,
        commanded_amplitudes=command_amp_values,
        commanded_phases_rad=command_phase_values,
        excitation_mode=args.excitation_mode,
        target_hash=args.target_hash,
        control_hash=args.control_hash,
    )

    density_data = {
        "system_type": "4qubit",
        "basis_config_path": basis_config_path,
        "x_phase_channels": sorted(X_CHANNELS_4Q),
        "re_rho": re_rho.tolist(),
        "im_rho": im_rho.tolist(),
        "re_rho_4x4": re_rho_4x4.tolist(),
        "im_rho_4x4": im_rho_4x4.tolist(),
        "excitation_mode": args.excitation_mode,
        "user_amplitudes": command_amp_values,
        "user_phases_degrees": [math.degrees(value) for value in command_phase_values],
        "target_amplitudes": target_amp_values,
        "target_phases_rad": target_phase_values,
        "commanded_amplitudes": command_amp_values,
        "commanded_phases_rad": command_phase_values,
        "target_hash": phase_generation_manifest["target_hash"],
        "control_hash": phase_generation_manifest["control_hash"],
        "w_ch1_base_matrix_hash": phase_generation_manifest["w_ch1_base_matrix_hash"],
        "phase_generation_manifest": PHASE_GENERATION_MANIFEST_4Q,
    }
    with open(os.path.join(args.output_dir, "density_matrix_4qubit.json"), "w", encoding="utf-8") as handle:
        json.dump(density_data, handle, indent=2)

    print("Phase calculation completed successfully for 4-qubit system!")
    print(f"Generated matrices saved to {args.output_dir}")


if __name__ == "__main__":
    main()
