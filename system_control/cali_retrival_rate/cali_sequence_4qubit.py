import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
QUBIT_RECON_DIR = REPO_ROOT / "system_control" / "qubit_recon_files"
if str(QUBIT_RECON_DIR) not in sys.path:
    sys.path.insert(0, str(QUBIT_RECON_DIR))

from matrix_utils import (
    apply_additive_delta,
    ensure_text_matrix,
    ensure_text_matrix_rectangular,
    ensure_y_channel_cycle_matrices,
    ensure_y_channel_matrices,
)


FREQ_START = 85
FREQ_END = 122.5
FREQ_STEPS = 16
GROUPS_PER_ROW = 4
GROUP_SIZE = 4
OPS_PER_CYCLE = 64
OP_DURATION_NS = 4000
CYCLE_ACTIVE_NS = OPS_PER_CYCLE * OP_DURATION_NS
CYCLE_GAP_NS = 0
TOTAL_CYCLES = 4
SAMPLING_RATE = 1e9
OUTPUT_DIR = REPO_ROOT / "cali_rate"

CHANNELS = ["W_CH1", "W_CH2", "W_CH3", "W_CH4", "R_CH1", "R_CH2", "R_CH3", "R_CH4"]
Y_CHANNELS = {"W_CH2", "W_CH4", "R_CH2", "R_CH4"}
R_CH1_CELL_COUNT = 4

CHANNEL_DIGITAL_DELAYS = {
    "W_CH1": 0,
    "W_CH2": 9690,
    "W_CH3": 9740,
    "W_CH4": 9790,
    "R_CH1": 0,
    "R_CH2": 9980,
    "R_CH3": 10030,
    "R_CH4": 10080,
}

CHANNEL_ANALOG_DELAYS = {
    "W_CH1": 4300,
    "W_CH2": 4950,
    "W_CH3": 1400,
    "W_CH4": 0,
    "R_CH1": 1800,
    "R_CH2": 0,
    "R_CH3": 700,
    "R_CH4": 4900,
}

CHANNEL_DIGITAL_WIDTHS = {
    "W_CH1": 120,
    "W_CH2": 120,
    "W_CH3": 120,
    "W_CH4": 120,
    "R_CH1": 400,
    "R_CH2": 400,
    "R_CH3": 400,
    "R_CH4": 400,
}


class CaliRetrievalConfig4Qubit:
    def __init__(self):
        self.freq_start = FREQ_START
        self.freq_end = FREQ_END
        self.freq_steps = FREQ_STEPS
        self.sampling_rate = SAMPLING_RATE
        self.output_dir = str(OUTPUT_DIR)
        self.total_duration: Optional[int] = None
        self.digital_delay = 0
        self.digital_delay2 = 0
        self.channel_digital_delays: Dict[str, int] = dict(CHANNEL_DIGITAL_DELAYS)
        self.channel_digital2_delays: Dict[str, int] = dict(CHANNEL_DIGITAL_DELAYS)
        self.channel_analog_delays: Dict[str, int] = dict(CHANNEL_ANALOG_DELAYS)
        self.channel_digital_widths: Dict[str, int] = dict(CHANNEL_DIGITAL_WIDTHS)
        self.analog_active_until: Optional[int] = None
        self.amp_matrices = {}
        self.phase_matrices = {}
        self.w_ch1_cycle_amp_matrices = []
        self.w_ch1_cycle_phase_matrices = []
        self.cycle_amp_matrices = {"R_CH1": [], "R_CH2": []}
        self.cycle_phase_matrices = {"R_CH1": [], "R_CH2": []}

        matrix_paths = ensure_y_channel_matrices(
            str(QUBIT_RECON_DIR),
            suffix="_4qubit",
            x_freq_steps=FREQ_STEPS,
            y_freq_steps=FREQ_STEPS,
            ops_per_row=GROUPS_PER_ROW,
        )

        for channel in ["W_CH3", "R_CH3"]:
            self.amp_matrices[channel] = np.loadtxt(matrix_paths[channel]["amp"])
            self.phase_matrices[channel] = np.loadtxt(matrix_paths[channel]["phase"])

        for channel in Y_CHANNELS:
            self.amp_matrices[channel] = ensure_text_matrix_rectangular(
                matrix_paths[channel]["amp"],
                FREQ_STEPS,
                GROUPS_PER_ROW,
                1.0,
            )
            self.phase_matrices[channel] = ensure_text_matrix_rectangular(
                matrix_paths[channel]["phase"],
                FREQ_STEPS,
                GROUPS_PER_ROW,
                0.0,
            )

        for cycle_index in range(1, TOTAL_CYCLES + 1):
            self.w_ch1_cycle_amp_matrices.append(
                ensure_text_matrix(
                    str(QUBIT_RECON_DIR / "matrix" / f"W_CH1_amp_{cycle_index}_4qubit.txt"),
                    (FREQ_STEPS, FREQ_STEPS),
                    1.0,
                )
            )
            self.w_ch1_cycle_phase_matrices.append(
                ensure_text_matrix(
                    str(QUBIT_RECON_DIR / "matrix" / f"W_CH1_phase_{cycle_index}_4qubit.txt"),
                    (FREQ_STEPS, FREQ_STEPS),
                    0.0,
                )
            )
            self.cycle_amp_matrices["R_CH1"].append(
                ensure_text_matrix(
                    str(QUBIT_RECON_DIR / "matrix" / f"R_CH1_amp_{cycle_index}_4qubit.txt"),
                    (FREQ_STEPS, FREQ_STEPS),
                    1.0,
                )
            )
            self.cycle_phase_matrices["R_CH1"].append(
                ensure_text_matrix(
                    str(QUBIT_RECON_DIR / "matrix" / f"R_CH1_phase_{cycle_index}_4qubit.txt"),
                    (FREQ_STEPS, FREQ_STEPS),
                    0.0,
                )
            )

        r_ch2_matrices = ensure_y_channel_cycle_matrices(
            "R_CH2",
            str(QUBIT_RECON_DIR),
            "_4qubit",
            TOTAL_CYCLES,
            FREQ_STEPS,
            GROUPS_PER_ROW,
        )
        for cycle_index in range(TOTAL_CYCLES):
            self.cycle_amp_matrices["R_CH2"].append(r_ch2_matrices[cycle_index]["amp"])
            self.cycle_phase_matrices["R_CH2"].append(r_ch2_matrices[cycle_index]["phase"])

        self._apply_ml_deltas()

    def _apply_ml_deltas(self):
        delta_dir = QUBIT_RECON_DIR / "matrix" / "ml_delta"

        for channel in ["W_CH3", "R_CH3"]:
            self.phase_matrices[channel] = apply_additive_delta(
                self.phase_matrices[channel],
                str(delta_dir / f"{channel}_phase_delta_4qubit.txt"),
                0.20,
            )
            self.amp_matrices[channel] = apply_additive_delta(
                self.amp_matrices[channel],
                str(delta_dir / f"{channel}_amp_delta_4qubit.txt"),
                0.03,
            )

        for cycle_index in range(TOTAL_CYCLES):
            cycle = cycle_index + 1
            self.w_ch1_cycle_phase_matrices[cycle_index] = apply_additive_delta(
                self.w_ch1_cycle_phase_matrices[cycle_index],
                str(delta_dir / f"W_CH1_phase_delta_{cycle}_4qubit.txt"),
                0.20,
            )
            self.cycle_phase_matrices["R_CH1"][cycle_index] = apply_additive_delta(
                self.cycle_phase_matrices["R_CH1"][cycle_index],
                str(delta_dir / f"R_CH1_phase_delta_{cycle}_4qubit.txt"),
                0.20,
            )


def _operation_indices(local_t_ns):
    op_index = min(local_t_ns // OP_DURATION_NS, OPS_PER_CYCLE - 1)
    row = op_index // GROUPS_PER_ROW
    group = op_index % GROUPS_PER_ROW
    col_start = group * GROUP_SIZE
    cols = [col_start + offset for offset in range(GROUP_SIZE)]
    return op_index, row, cols


def _base_channel_value(config, channel_name, row, cols, frequencies, t_sec, cycle_index, op_index, selected_cell_index=None):
    if channel_name == "W_CH1":
        value = 0.0
        amp_matrix = config.w_ch1_cycle_amp_matrices[cycle_index]
        phase_matrix = config.w_ch1_cycle_phase_matrices[cycle_index]
        for col in cols:
            amp = amp_matrix[row, col]
            phase = phase_matrix[row, col]
            value += amp * np.sin(2 * np.pi * frequencies[col] * 1e6 * t_sec + phase)
        return value

    if channel_name == "R_CH2":
        amp_matrix = config.cycle_amp_matrices["R_CH2"][cycle_index]
        phase_matrix = config.cycle_phase_matrices["R_CH2"][cycle_index]
        col = op_index % GROUPS_PER_ROW
        amp = amp_matrix[row, col]
        phase = phase_matrix[row, col]
        return amp * np.sin(2 * np.pi * frequencies[row] * 1e6 * t_sec + phase)

    if channel_name in {"W_CH3", "R_CH3"}:
        value = 0.0
        active_cols = cols
        if channel_name == "R_CH3" and selected_cell_index is not None:
            active_cols = [cols[selected_cell_index]]
        for col in active_cols:
            amp = np.sqrt(config.amp_matrices[channel_name][row, col])
            phase = config.phase_matrices[channel_name][row, col]
            value += amp * np.sin(2 * np.pi * frequencies[col] * 1e6 * t_sec + phase)
        return value

    if channel_name in Y_CHANNELS:
        col = op_index % GROUPS_PER_ROW
        amp = config.amp_matrices[channel_name][row, col]
        phase = config.phase_matrices[channel_name][row, col]
        return amp * np.sin(2 * np.pi * frequencies[row] * 1e6 * t_sec + phase)

    return 0.0


def _r_ch1_cell_value(config, row, cols, cell_index, frequencies, t_sec, cycle_index):
    col = cols[cell_index]
    amp = config.cycle_amp_matrices["R_CH1"][cycle_index][row, col]
    phase = config.cycle_phase_matrices["R_CH1"][cycle_index][row, col]
    return amp * np.sin(2 * np.pi * frequencies[col] * 1e6 * t_sec + phase)


def _validate_read_cell_index(read_cell_index):
    if not isinstance(read_cell_index, int) or isinstance(read_cell_index, bool):
        raise ValueError("read_cell_index must be an integer")
    if not 1 <= read_cell_index <= R_CH1_CELL_COUNT:
        raise ValueError(f"read_cell_index must be between 1 and {R_CH1_CELL_COUNT}")
    return read_cell_index - 1


def _schedule_digital1(config, data, channel_name, selected_cell_index=None):
    data[:, 1] = 0
    data[:, 2] = 0
    width = config.channel_digital_widths[channel_name]
    delay = config.digital_delay + config.channel_digital_delays[channel_name]

    for cycle_index in range(TOTAL_CYCLES):
        cycle_start_ns = cycle_index * (CYCLE_ACTIVE_NS + CYCLE_GAP_NS)
        for op_index in range(OPS_PER_CYCLE):
            op_start_ns = cycle_start_ns + op_index * OP_DURATION_NS
            if channel_name.startswith("R_") and selected_cell_index is not None:
                positions = [selected_cell_index]
            elif channel_name == "W_CH1":
                positions = [0]
            else:
                positions = range(GROUP_SIZE)

            for pos in positions:
                start = op_start_ns + pos * 1000 + delay
                end = start + width
                s = int(max(0, start))
                e = int(min(len(data), end))
                if s < e:
                    data[s:e, 1] = 1

        gap_start = cycle_start_ns + CYCLE_ACTIVE_NS + config.digital_delay2
        dig2_start = gap_start + config.channel_digital2_delays[channel_name]
        dig2_end = dig2_start + 5000
        s2 = int(max(0, dig2_start))
        e2 = int(min(len(data), dig2_end))
        if s2 < e2:
            data[s2:e2, 2] = 1


def _total_duration(config):
    if config.total_duration is not None:
        return config.total_duration
    max_delay = max(
        max(config.digital_delay + value for value in config.channel_digital_delays.values()),
        max(config.digital_delay2 + value for value in config.channel_digital2_delays.values()),
        max(config.channel_analog_delays.values()),
    )
    return max_delay + TOTAL_CYCLES * (CYCLE_ACTIVE_NS + CYCLE_GAP_NS)


def _save_channel(output_dir, filename, data):
    filepath = os.path.join(output_dir, filename)
    np.savetxt(
        filepath,
        data,
        fmt=["%.7f", "%d", "%d"],
        delimiter=",",
        header="analog,digital1,digital2",
        comments="",
    )


def _generate_cell_outputs_4qubit(config, read_cell_index):
    selected_cell_index = _validate_read_cell_index(read_cell_index)
    total_duration = _total_duration(config)
    buffer_size = total_duration + 10000

    frequencies = np.linspace(config.freq_start, config.freq_end, config.freq_steps)
    outputs = {}

    for channel_name in CHANNELS:
        data = np.zeros((buffer_size, 3))
        for cycle_index in range(TOTAL_CYCLES):
            cycle_start_ns = cycle_index * (CYCLE_ACTIVE_NS + CYCLE_GAP_NS)
            for local_t_ns in range(CYCLE_ACTIVE_NS):
                absolute_t_ns = cycle_start_ns + local_t_ns
                t_sec = absolute_t_ns / config.sampling_rate
                op_index, row, cols = _operation_indices(local_t_ns)
                analog_time_idx = absolute_t_ns + config.channel_analog_delays[channel_name]
                if not (0 <= analog_time_idx < buffer_size):
                    continue

                if channel_name == "R_CH1":
                    analog_value = _r_ch1_cell_value(
                        config,
                        row,
                        cols,
                        selected_cell_index,
                        frequencies,
                        t_sec,
                        cycle_index,
                    )
                else:
                    analog_value = _base_channel_value(
                        config,
                        channel_name,
                        row,
                        cols,
                        frequencies,
                        t_sec,
                        cycle_index,
                        op_index,
                        selected_cell_index,
                    )
                if config.analog_active_until is not None and analog_time_idx >= config.analog_active_until:
                    analog_value = 0.0
                data[int(analog_time_idx), 0] = analog_value

        digital_cell_index = selected_cell_index if channel_name.startswith("R_") else None
        _schedule_digital1(config, data, channel_name, selected_cell_index=digital_cell_index)
        outputs[channel_name] = data[:total_duration]

    return outputs


def _save_cell_outputs(config, read_cell_index, outputs):
    output_dir = os.path.join(config.output_dir, f"4qubit_cell{read_cell_index}")
    os.makedirs(output_dir, exist_ok=True)
    for channel_name, data in outputs.items():
        _save_channel(output_dir, f"{channel_name}_4qubit.txt", data)


def generate_cali_retrieval_sequence_4qubit(config=None, read_cell_index=None):
    if config is None:
        config = CaliRetrievalConfig4Qubit()

    os.makedirs(config.output_dir, exist_ok=True)
    if read_cell_index is None:
        read_cell_indices = range(1, R_CH1_CELL_COUNT + 1)
    else:
        _validate_read_cell_index(read_cell_index)
        read_cell_indices = [read_cell_index]

    grouped_outputs = {}
    for cell_index in read_cell_indices:
        cell_outputs = _generate_cell_outputs_4qubit(config, cell_index)
        _save_cell_outputs(config, cell_index, cell_outputs)
        grouped_outputs[f"cell{cell_index}"] = cell_outputs

    return grouped_outputs, config


if __name__ == "__main__":
    generated_outputs, generated_config = generate_cali_retrieval_sequence_4qubit()
    total_files = sum(len(outputs) for outputs in generated_outputs.values())
    print(
        f"Generated {len(generated_outputs)} calibration cell groups "
        f"({total_files} files) in {generated_config.output_dir}"
    )
