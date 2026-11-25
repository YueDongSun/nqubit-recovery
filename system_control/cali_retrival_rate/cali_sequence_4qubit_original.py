import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if __package__:
    from . import cali_sequence_4qubit as base
else:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import cali_sequence_4qubit as base


OUTPUT_DIR = base.OUTPUT_DIR / "original_phase0_4qubit"


class CaliRetrievalOriginalConfig4Qubit(base.CaliRetrievalConfig4Qubit):
    def __init__(self):
        super().__init__()
        self.output_dir = str(OUTPUT_DIR)


def _operation_indices(local_t_ns):
    op_index = min(local_t_ns // base.OP_DURATION_NS, base.OPS_PER_CYCLE - 1)
    row = op_index // base.GROUPS_PER_ROW
    group = op_index % base.GROUPS_PER_ROW
    col_start = group * base.GROUP_SIZE
    cols = [col_start + offset for offset in range(base.GROUP_SIZE)]
    return op_index, row, cols


def _base_channel_value(config, channel_name, row, cols, frequencies, t_sec, cycle_index, op_index):
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
        col = op_index % base.GROUPS_PER_ROW
        amp = amp_matrix[row, col]
        return amp * np.sin(2 * np.pi * frequencies[row] * 1e6 * t_sec)

    if channel_name == "W_CH3":
        value = 0.0
        for col in cols:
            amp = np.sqrt(config.amp_matrices[channel_name][row, col])
            phase = config.phase_matrices[channel_name][row, col]
            value += amp * np.sin(2 * np.pi * frequencies[col] * 1e6 * t_sec + phase)
        return value

    if channel_name == "R_CH3":
        value = 0.0
        for col in cols:
            amp = np.sqrt(config.amp_matrices[channel_name][row, col])
            value += amp * np.sin(2 * np.pi * frequencies[col] * 1e6 * t_sec)
        return value

    if channel_name in base.Y_CHANNELS:
        col = op_index % base.GROUPS_PER_ROW
        amp = config.amp_matrices[channel_name][row, col]
        phase = 0.0 if channel_name.startswith("R_") else config.phase_matrices[channel_name][row, col]
        return amp * np.sin(2 * np.pi * frequencies[row] * 1e6 * t_sec + phase)

    return 0.0


def _r_ch1_all_cells_value_phase0(config, row, cols, frequencies, t_sec, cycle_index):
    value = 0.0
    amp_matrix = config.cycle_amp_matrices["R_CH1"][cycle_index]
    for col in cols:
        amp = amp_matrix[row, col]
        value += amp * np.sin(2 * np.pi * frequencies[col] * 1e6 * t_sec)
    return value


def _schedule_digital1(config, data, channel_name, selected_cell_index=None):
    data[:, 1] = 0
    data[:, 2] = 0
    width = config.channel_digital_widths[channel_name]
    delay = config.digital_delay + config.channel_digital_delays[channel_name]

    for cycle_index in range(base.TOTAL_CYCLES):
        cycle_start_ns = cycle_index * (base.CYCLE_ACTIVE_NS + base.CYCLE_GAP_NS)
        for op_index in range(base.OPS_PER_CYCLE):
            op_start_ns = cycle_start_ns + op_index * base.OP_DURATION_NS
            positions = [selected_cell_index] if selected_cell_index is not None else range(base.GROUP_SIZE)

            if channel_name == "W_CH1":
                positions = [0]

            for pos in positions:
                start = op_start_ns + pos * 1000 + delay
                end = start + width
                s = int(max(0, start))
                e = int(min(len(data), end))
                if s < e:
                    data[s:e, 1] = 1

        gap_start = cycle_start_ns + base.CYCLE_ACTIVE_NS + config.digital_delay2
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
    return max_delay + base.TOTAL_CYCLES * (base.CYCLE_ACTIVE_NS + base.CYCLE_GAP_NS)


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


def generate_cali_retrieval_sequence_4qubit_original(config=None):
    if config is None:
        config = CaliRetrievalOriginalConfig4Qubit()

    total_duration = _total_duration(config)
    buffer_size = total_duration + 10000
    os.makedirs(config.output_dir, exist_ok=True)

    frequencies = np.linspace(config.freq_start, config.freq_end, config.freq_steps)
    outputs = {}

    for channel_name in base.CHANNELS:
        data = np.zeros((buffer_size, 3))
        for cycle_index in range(base.TOTAL_CYCLES):
            cycle_start_ns = cycle_index * (base.CYCLE_ACTIVE_NS + base.CYCLE_GAP_NS)
            for local_t_ns in range(base.CYCLE_ACTIVE_NS):
                absolute_t_ns = cycle_start_ns + local_t_ns
                t_sec = absolute_t_ns / config.sampling_rate
                op_index, row, cols = _operation_indices(local_t_ns)
                analog_time_idx = absolute_t_ns + config.channel_analog_delays[channel_name]
                if not (0 <= analog_time_idx < buffer_size):
                    continue

                if channel_name == "R_CH1":
                    analog_value = _r_ch1_all_cells_value_phase0(
                        config,
                        row,
                        cols,
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
                    )
                if config.analog_active_until is not None and analog_time_idx >= config.analog_active_until:
                    analog_value = 0.0
                data[int(analog_time_idx), 0] = analog_value

        _schedule_digital1(config, data, channel_name)
        outputs[channel_name] = data[:total_duration]

    for channel_name, data in outputs.items():
        _save_channel(config.output_dir, f"{channel_name}_4qubit.txt", data)

    return outputs, config


generate_cali_retrieval_sequence_4qubit = generate_cali_retrieval_sequence_4qubit_original


if __name__ == "__main__":
    generated_outputs, generated_config = generate_cali_retrieval_sequence_4qubit_original()
    print(f"Generated {len(generated_outputs)} original calibration sequence files in {generated_config.output_dir}")
