#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Dict, Optional

# Force the non-GUI Agg backend BEFORE importing pyplot.
# This script is called from a QThread worker in the experiment UI;
# matplotlib's Qt/Tk backends require the main thread and would warn
# (or crash) if initialized here. Agg only writes PNGs via savefig()
# and makes plt.show() a no-op.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from matrix_utils import apply_additive_delta, ensure_text_matrix, ensure_text_matrix_rectangular, ensure_y_channel_matrices, ensure_y_channel_cycle_matrices
from fourier_sequence_guard import run_and_attach_fourier_guard
from system_specs import SYSTEM_SPECS
from digital_delay_defaults import DEFAULT_DIGITAL_DELAY_OFFSET_NS, digital_delay_values
from wch1_excitation_factor import build_wch1_excitation_factor_matrix


def _maybe_show_plot():
    """Pop the current figure only when explicitly opted in.

    The main experiment UI drives this script on a worker thread; the
    default behaviour is to save PNG only. Set the environment variable
    AWG_SHOW_PLOTS=1 (true/yes/on also accepted) to pop interactive
    windows when running standalone on the main thread.
    """
    flag = os.environ.get("AWG_SHOW_PLOTS", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        plt.show()

_SPEC_4Q = SYSTEM_SPECS["4qubit"]

FREQ_STEPS_4Q = _SPEC_4Q["freq_steps"]
GROUPS_PER_ROW_4Q = _SPEC_4Q["groups_per_row"]
GROUP_SIZE_4Q = _SPEC_4Q["group_size"]
OPS_PER_CYCLE_4Q = _SPEC_4Q["ops_per_cycle"]
OP_DURATION_NS_4Q = _SPEC_4Q["op_duration_ns"]
CYCLE_ACTIVE_NS_4Q = _SPEC_4Q["cycle_active_ns"]
CYCLE_GAP_NS_4Q = _SPEC_4Q["cycle_gap_ns"]
TOTAL_CYCLES_4Q = _SPEC_4Q["total_cycles"]
Y_CHANNELS_4Q = _SPEC_4Q["y_channels"]


class EntangleConfig4Qubit:
    sampling_rate: float
    total_duration: Optional[int]
    channel_digital_delays: Dict[str, int]
    channel_digital2_delays: Dict[str, int]
    channel_analog_delays: Dict[str, int]
    channel_physical_delays: Optional[Dict[str, int]]
    output_dir: str
    analog_active_until: Optional[int]

    def __init__(self, matrix_dir=None, include_ml_delta=True):
        self.freq_start = _SPEC_4Q["freq_start_mhz"]
        self.freq_end = _SPEC_4Q["freq_end_mhz"]
        self.freq_step = _SPEC_4Q["freq_step_mhz"]
        self.freq_steps = FREQ_STEPS_4Q
        self.digital_delay = 0
        self.digital_delay2 = 0
        self.digital_width_write = 120
        self.digital_width_read = 400
        self.digital_width2_write = 0
        self.digital_width2_read = 0
        self.digital_delay_offset_ns = DEFAULT_DIGITAL_DELAY_OFFSET_NS
        default_digital_delays = digital_delay_values(self.digital_delay_offset_ns)
        self.w_ch1_digital_width = 120
        self.w_ch1_digital_delay = default_digital_delays["W_CH1"]
        self.r_ch1_digital_width = 400
        self.r_ch1_digital_delay = default_digital_delays["R_CH1"]
        self.w_ch1_excitation_mode = 1
        self.w_ch1_digital1_positions = [True, False, False, False]
        self.user_amp_values = [1.0, 1.0, 1.0, 1.0]
        self.user_phase_values = [0.0, 0.0, 0.0, 0.0]

        self.channel_digital_widths = {
            "W_CH1": self.w_ch1_digital_width,
            "W_CH2": self.digital_width_write,
            "W_CH3": self.digital_width_write,
            "W_CH4": self.digital_width_write,
            "R_CH1": self.r_ch1_digital_width,
            "R_CH2": self.digital_width_read,
            "R_CH3": self.digital_width_read,
            "R_CH4": self.digital_width_read,
        }
        self.channel_digital2_widths = {
            "W_CH1": self.digital_width2_write,
            "W_CH2": self.digital_width2_write,
            "W_CH3": self.digital_width2_write,
            "W_CH4": self.digital_width2_write,
            "R_CH1": self.digital_width2_read,
            "R_CH2": self.digital_width2_read,
            "R_CH3": self.digital_width2_read,
            "R_CH4": self.digital_width2_read,
        }
        self.channel_digital_delays = {
            channel: int(delay)
            for channel, delay in default_digital_delays.items()
        }
        self.channel_digital2_delays = dict(self.channel_digital_delays)
        self.channel_analog_delays = {
            "W_CH1": 4300,
            "W_CH2": 4950,
            "W_CH3": 1400,
            "W_CH4": 0,
            "R_CH1": 1800,
            "R_CH2": 0,
            "R_CH3": 700,
            "R_CH4": 4900,
        }
        self.sampling_rate = _SPEC_4Q["sampling_rate"]
        self.total_duration = None
        self.channel_physical_delays = None
        self.output_dir = os.path.join("AWG_Sequence", "entangle_sequences_4qubit")
        self.analog_active_until = None

        self.amp_matrices = {}
        self.phase_matrices = {}
        self.w_ch1_cycle_amp_matrices = []
        self.w_ch1_cycle_phase_matrices = []
        self.cycle_amp_matrices = {"R_CH1": [], "R_CH2": []}
        self.cycle_phase_matrices = {"R_CH1": [], "R_CH2": []}
        self.w_ch1_excitation_factor = build_wch1_excitation_factor_matrix(self.freq_steps)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        matrix_root = Path(matrix_dir) if matrix_dir is not None else Path(script_dir) / "matrix"
        matrix_paths = ensure_y_channel_matrices(
            script_dir,
            suffix="_4qubit",
            x_freq_steps=FREQ_STEPS_4Q,
            y_freq_steps=FREQ_STEPS_4Q,
            ops_per_row=GROUPS_PER_ROW_4Q,
            matrix_dir=str(matrix_root),
        )

        for channel in ["W_CH1", "W_CH3", "R_CH3"]:
            self.amp_matrices[channel] = np.loadtxt(matrix_paths[channel]["amp"])
            self.phase_matrices[channel] = np.loadtxt(matrix_paths[channel]["phase"])

        for channel in Y_CHANNELS_4Q:
            self.amp_matrices[channel] = ensure_text_matrix_rectangular(
                matrix_paths[channel]["amp"],
                FREQ_STEPS_4Q,
                GROUPS_PER_ROW_4Q,
                1.0
            )
            self.phase_matrices[channel] = ensure_text_matrix_rectangular(
                matrix_paths[channel]["phase"],
                FREQ_STEPS_4Q,
                GROUPS_PER_ROW_4Q,
                0.0
            )

        for cycle_index in range(1, TOTAL_CYCLES_4Q + 1):
            self.w_ch1_cycle_amp_matrices.append(
                ensure_text_matrix(str(matrix_root / f"W_CH1_amp_{cycle_index}_4qubit.txt"), (self.freq_steps, self.freq_steps), 1.0)
            )
            self.w_ch1_cycle_phase_matrices.append(
                ensure_text_matrix(str(matrix_root / f"W_CH1_phase_{cycle_index}_4qubit.txt"), (self.freq_steps, self.freq_steps), 0.0)
            )
            self.cycle_amp_matrices["R_CH1"].append(
                ensure_text_matrix(str(matrix_root / f"R_CH1_amp_{cycle_index}_4qubit.txt"), (self.freq_steps, self.freq_steps), 1.0)
            )
            self.cycle_phase_matrices["R_CH1"].append(
                ensure_text_matrix(str(matrix_root / f"R_CH1_phase_{cycle_index}_4qubit.txt"), (self.freq_steps, self.freq_steps), 0.0)
            )

        r_ch2_matrices = ensure_y_channel_cycle_matrices(
            "R_CH2",
            script_dir,
            "_4qubit",
            TOTAL_CYCLES_4Q,
            FREQ_STEPS_4Q,
            GROUPS_PER_ROW_4Q,
            matrix_dir=str(matrix_root),
        )
        for cycle_index in range(TOTAL_CYCLES_4Q):
            self.cycle_amp_matrices["R_CH2"].append(r_ch2_matrices[cycle_index]["amp"])
            self.cycle_phase_matrices["R_CH2"].append(r_ch2_matrices[cycle_index]["phase"])

        if include_ml_delta:
            self._apply_ml_deltas(matrix_root)

    def _apply_ml_deltas(self, matrix_root):
        delta_dir = os.path.join(str(matrix_root), "ml_delta")
        phase_limit = 0.20
        amp_limit = 0.03

        for channel in ["W_CH3", "R_CH3"]:
            self.phase_matrices[channel] = apply_additive_delta(
                self.phase_matrices[channel],
                os.path.join(delta_dir, f"{channel}_phase_delta_4qubit.txt"),
                phase_limit,
            )
            self.amp_matrices[channel] = apply_additive_delta(
                self.amp_matrices[channel],
                os.path.join(delta_dir, f"{channel}_amp_delta_4qubit.txt"),
                amp_limit,
            )

        for cycle_index in range(TOTAL_CYCLES_4Q):
            cycle = cycle_index + 1
            self.w_ch1_cycle_phase_matrices[cycle_index] = apply_additive_delta(
                self.w_ch1_cycle_phase_matrices[cycle_index],
                os.path.join(delta_dir, f"W_CH1_phase_delta_{cycle}_4qubit.txt"),
                phase_limit,
            )
            self.cycle_phase_matrices["R_CH1"][cycle_index] = apply_additive_delta(
                self.cycle_phase_matrices["R_CH1"][cycle_index],
                os.path.join(delta_dir, f"R_CH1_phase_delta_{cycle}_4qubit.txt"),
                phase_limit,
            )

    def apply_phase_probe_overlay(self, phase_probe_delta_matrices):
        """Apply an ephemeral operation-level probe to in-memory W_CH1 phases.

        This method deliberately performs no file I/O.  The base matrices and
        persistent ``matrix/ml_delta`` namespace remain unchanged.
        """

        if phase_probe_delta_matrices is None:
            return
        if len(phase_probe_delta_matrices) != TOTAL_CYCLES_4Q:
            raise ValueError("4Q phase probe overlay must contain four cycle matrices")
        overlaid = []
        for cycle_index, (base, delta) in enumerate(
            zip(self.w_ch1_cycle_phase_matrices, phase_probe_delta_matrices)
        ):
            delta_array = np.asarray(delta, dtype=float)
            if delta_array.shape != (FREQ_STEPS_4Q, FREQ_STEPS_4Q):
                raise ValueError(
                    f"phase probe cycle {cycle_index + 1} must have shape "
                    f"{(FREQ_STEPS_4Q, FREQ_STEPS_4Q)}"
                )
            if not np.all(np.isfinite(delta_array)):
                raise ValueError("phase probe overlay must contain only finite values")
            if float(np.max(np.abs(delta_array), initial=0.0)) > 0.08000001:
                raise ValueError("phase probe overlay exceeds the 0.08 rad campaign limit")
            overlaid.append(np.asarray(base, dtype=float).copy() + delta_array)
        self.w_ch1_cycle_phase_matrices = overlaid


def _operation_indices_4q(local_t_ns):
    op_index = min(local_t_ns // OP_DURATION_NS_4Q, OPS_PER_CYCLE_4Q - 1)
    row = op_index // GROUPS_PER_ROW_4Q
    group = op_index % GROUPS_PER_ROW_4Q
    col_start = group * GROUP_SIZE_4Q
    cols = [col_start + offset for offset in range(GROUP_SIZE_4Q)]
    return op_index, row, group, cols


def _wch1_selected_columns(mode, within_op_ns, cols):
    if mode == 1:
        return cols
    if mode == 2:
        return cols[:3] if within_op_ns < 3000 else [cols[3]]
    if mode == 3:
        if within_op_ns < 2000:
            return cols[:2]
        return [cols[2 + min((within_op_ns - 2000) // 1000, 1)]]
    if mode == 4:
        return [cols[min(within_op_ns // 1000, 3)]]
    return cols


def _channel_value(config, channel_name, row, cols, within_op_ns, frequencies, t_sec, cycle_index, op_index):
    analog_value = 0.0

    if channel_name == "W_CH1":
        amp_matrix = config.w_ch1_cycle_amp_matrices[cycle_index]
        phase_matrix = config.w_ch1_cycle_phase_matrices[cycle_index]
        factor_matrix = getattr(config, "w_ch1_excitation_factor", None)
        for col in _wch1_selected_columns(config.w_ch1_excitation_mode, within_op_ns, cols):
            amp = amp_matrix[row, col]
            if factor_matrix is not None:
                amp *= factor_matrix[row, col]
            phase = phase_matrix[row, col]
            analog_value += amp * np.sin(2 * np.pi * frequencies[col] * 1e6 * t_sec + phase)
    elif channel_name == "R_CH1":
        amp_matrix = config.cycle_amp_matrices["R_CH1"][cycle_index]
        phase_matrix = config.cycle_phase_matrices["R_CH1"][cycle_index]
        for col in cols:
            amp = amp_matrix[row, col]
            phase = phase_matrix[row, col]
            analog_value += amp * np.sin(2 * np.pi * frequencies[col] * 1e6 * t_sec + phase)
    elif channel_name == "R_CH2":
        amp_matrix = config.cycle_amp_matrices["R_CH2"][cycle_index]
        phase_matrix = config.cycle_phase_matrices["R_CH2"][cycle_index]
        col = op_index % GROUPS_PER_ROW_4Q
        amp = amp_matrix[row, col]
        phase = phase_matrix[row, col]
        analog_value = amp * np.sin(2 * np.pi * frequencies[row] * 1e6 * t_sec + phase)
    elif channel_name in {"W_CH3", "R_CH3"}:
        for col in cols:
            amp = config.amp_matrices[channel_name][row, col]
            if channel_name in {"W_CH3", "R_CH3"}:
                amp = np.sqrt(amp)
            phase = config.phase_matrices[channel_name][row, col]
            analog_value += amp * np.sin(2 * np.pi * frequencies[col] * 1e6 * t_sec + phase)
    elif channel_name in Y_CHANNELS_4Q:
        col = op_index % GROUPS_PER_ROW_4Q
        amp = config.amp_matrices[channel_name][row, col]
        phase = config.phase_matrices[channel_name][row, col]
        analog_value = amp * np.sin(2 * np.pi * frequencies[row] * 1e6 * t_sec + phase)

    return analog_value


def _schedule_digital_pulses(config, channels, buffer_size):
    for channel_name in channels:
        channels[channel_name][:, 1] = 0
        channels[channel_name][:, 2] = 0

    for cycle_index in range(TOTAL_CYCLES_4Q):
        cycle_start_us = cycle_index * ((CYCLE_ACTIVE_NS_4Q + CYCLE_GAP_NS_4Q) // 1000)

        for op_index in range(OPS_PER_CYCLE_4Q):
            op_start_us = cycle_start_us + op_index * 4

            for channel_name in channels:
                if channel_name == "W_CH1":
                    if config.w_ch1_excitation_mode == 1:
                        enabled_positions = [0] if config.w_ch1_digital1_positions[0] else []
                    elif config.w_ch1_excitation_mode == 2:
                        enabled_positions = [pos for pos in [0, 3] if config.w_ch1_digital1_positions[pos]]
                    elif config.w_ch1_excitation_mode == 3:
                        enabled_positions = [pos for pos in [0, 2, 3] if config.w_ch1_digital1_positions[pos]]
                    else:
                        enabled_positions = [pos for pos in range(4) if config.w_ch1_digital1_positions[pos]]
                elif channel_name.startswith("R_"):
                    # Read marker1 follows the operation cadence: one pulse per 4 us.
                    enabled_positions = [0]
                else:
                    enabled_positions = range(4)

                for pos in enabled_positions:
                    start = (op_start_us + pos) * 1000 + config.digital_delay + config.channel_digital_delays[channel_name]
                    end = start + config.channel_digital_widths.get(
                        channel_name,
                        config.digital_width_write if channel_name.startswith("W_") else config.digital_width_read,
                    )
                    s = int(max(0, start))
                    e = int(min(buffer_size, end))
                    if s < e:
                        channels[channel_name][s:e, 1] = 1

        gap_start = cycle_start_us * 1000 + CYCLE_ACTIVE_NS_4Q + config.digital_delay2
        for channel_name in channels:
            dig2_start = gap_start + config.channel_digital2_delays[channel_name]
            dig2_end = dig2_start + 5000
            s2 = int(max(0, dig2_start))
            e2 = int(min(buffer_size, dig2_end))
            if s2 < e2:
                channels[channel_name][s2:e2, 2] = 1


def generate_entangle_sequence_4qubit(config=None, return_channels=True, make_preview=True):
    if config is None:
        config = EntangleConfig4Qubit()

    if config.total_duration is not None:
        total_duration = config.total_duration
    else:
        max_delay = max(
            max(config.digital_delay + value for value in config.channel_digital_delays.values()),
            max(config.digital_delay2 + value for value in config.channel_digital2_delays.values()),
            max(config.channel_analog_delays.values()),
        )
        total_duration = max_delay + TOTAL_CYCLES_4Q * (CYCLE_ACTIVE_NS_4Q + CYCLE_GAP_NS_4Q)

    os.makedirs(config.output_dir, exist_ok=True)
    frequencies = np.linspace(config.freq_start, config.freq_end, config.freq_steps)
    buffer_size = total_duration + 10000
    channels = {
        name: np.zeros((buffer_size, 3), dtype=np.float32)
        for name in ["W_CH1", "W_CH2", "W_CH3", "W_CH4", "R_CH1", "R_CH2", "R_CH3", "R_CH4"]
    }

    for cycle_index in range(TOTAL_CYCLES_4Q):
        cycle_start_ns = cycle_index * (CYCLE_ACTIVE_NS_4Q + CYCLE_GAP_NS_4Q)
        for local_t_ns in range(CYCLE_ACTIVE_NS_4Q):
            absolute_t_ns = cycle_start_ns + local_t_ns
            t_sec = absolute_t_ns / config.sampling_rate
            op_index, row, _, cols = _operation_indices_4q(local_t_ns)
            within_op_ns = local_t_ns % OP_DURATION_NS_4Q

            for channel_name in channels:
                analog_time_idx = absolute_t_ns + config.channel_analog_delays[channel_name]
                if not (0 <= analog_time_idx < buffer_size):
                    continue

                analog_value = _channel_value(config, channel_name, row, cols, within_op_ns, frequencies, t_sec, cycle_index, op_index)
                if config.analog_active_until is not None and analog_time_idx >= config.analog_active_until:
                    analog_value = 0.0
                channels[channel_name][int(analog_time_idx), 0] = analog_value

    _schedule_digital_pulses(config, channels, buffer_size)

    for channel_name in channels:
        channels[channel_name] = channels[channel_name][:total_duration]

    run_and_attach_fourier_guard("4qubit", channels, config)

    common_prefix = f"T{total_duration / 1000:.0f}us"
    for channel_name, data in channels.items():
        filepath = os.path.join(config.output_dir, f"{channel_name}_{common_prefix}.txt")
        np.savetxt(filepath, data, fmt=["%.7f", "%d", "%d"], delimiter=",", header="analog,digital1,digital2", comments="")

    if make_preview:
        plot_signals_4qubit(channels, config, common_prefix)
    return (channels if return_channels else None), config


def plot_signals_4qubit(channels, config, common_prefix=""):
    display_points = min(50000, len(channels["W_CH1"]))
    plt.figure(figsize=(20, 16))

    for idx, (channel_name, data) in enumerate(channels.items()):
        plt.subplot(4, 2, idx + 1)
        display_data = data[:display_points]
        time_ns = np.arange(len(display_data))
        analog_signal = display_data[:, 0]
        digital1_signal = display_data[:, 1]
        digital2_signal = display_data[:, 2]
        scale = max(np.max(np.abs(analog_signal)), 1.0) if len(analog_signal) else 1.0

        plt.plot(time_ns, analog_signal, "b-", label="Analog", linewidth=0.5, alpha=0.7)
        plt.plot(time_ns, digital1_signal * 1.2 * scale, "r-", label="Digital1", linewidth=1)
        plt.plot(time_ns, digital2_signal * 1.4 * scale, "g-", label="Digital2", linewidth=1)
        plt.title(
            f"{channel_name}\nDig1: {config.digital_delay + config.channel_digital_delays[channel_name]}ns, "
            f"Dig2: {config.digital_delay2 + config.channel_digital2_delays[channel_name]}ns, "
            f"Ana: {config.channel_analog_delays[channel_name]}ns"
        )
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plot_filename = f"signals_{common_prefix}.png" if common_prefix else "signals_4qubit.png"
    plt.savefig(os.path.join(config.output_dir, plot_filename), dpi=300, bbox_inches="tight")
    _maybe_show_plot()
    plt.close()


if __name__ == "__main__":
    generated_channels, generated_config = generate_entangle_sequence_4qubit(EntangleConfig4Qubit())
    print(f"Generated {len(generated_channels)} channels with {len(generated_channels['W_CH1'])} points each")
