"""
3-Qubit Data Receiver - Serial Communication with Basic UI
Date: 2025-10-24

Features:
- Read 1044-byte frames from COM port
- Decode byte-aligned 75-cell protocol (modulo=300)
- Display parsed numbers in UI
- Signal/Idler sum displayed (no plots)
"""

import sys, time, threading
import numpy as np
import serial
import os
import importlib.util
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QGroupBox, QGridLayout,
    QTableWidget, QTableWidgetItem, QScrollArea, QProgressBar,
    QAbstractItemView, QAbstractScrollArea, QFrame, QSizePolicy, QHeaderView
)
import serial.tools.list_ports

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtGui import QColor

# Local import for the baseline-subtracting counter (with carry).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_counter import BaselineCounter

try:
    from .snapshot_export import save_numpy_snapshot
except Exception:
    from snapshot_export import save_numpy_snapshot

try:
    from .frame_protocol import (
        LAYOUT_3Q,
        decode_3qubit_counter_words,
        decode_multi2_pair_counts,
        decode_multi_click_counts,
    )
except Exception:
    from frame_protocol import (
        LAYOUT_3Q,
        decode_3qubit_counter_words,
        decode_multi2_pair_counts,
        decode_multi_click_counts,
    )

try:
    from .feedback_dataset_actions import (
        save_feedback_sample_from_window,
    )
except Exception:
    from feedback_dataset_actions import (
        save_feedback_sample_from_window,
    )

try:
    from .rho_fidelity_summary import (
        ESTIMATOR_METRICS,
        VARIANT_LABELS,
        format_model_audit,
        format_variant_metric,
        primary_variant,
        summarize_rho_variants,
        top_fidelity_fits,
    )
except Exception:
    from rho_fidelity_summary import (
        ESTIMATOR_METRICS,
        VARIANT_LABELS,
        format_model_audit,
        format_variant_metric,
        primary_variant,
        summarize_rho_variants,
        top_fidelity_fits,
    )

try:
    from .rho_reconstruction_variants import build_diagnostic_rho_variants
except Exception:
    from rho_reconstruction_variants import build_diagnostic_rho_variants

try:
    from .visual_diagnostics import draw_state_family_chart, draw_w_subspace_summary
except Exception:
    from visual_diagnostics import draw_state_family_chart, draw_w_subspace_summary

try:
    from .physics_dashboard import (
        InferenceSummaryTable,
        PhysicsMetricGrid,
        compute_physics_dashboard_metrics,
        draw_physics_dashboard,
    )
except Exception:
    from physics_dashboard import (
        InferenceSummaryTable,
        PhysicsMetricGrid,
        compute_physics_dashboard_metrics,
        draw_physics_dashboard,
    )

try:
    from .pointwise_diagnostics import draw_operation_point_diagnostics, pointwise_values_from_shadow
except Exception:
    from pointwise_diagnostics import draw_operation_point_diagnostics, pointwise_values_from_shadow

try:
    from system_control.experiment_control.feedback_evidence import summarize_wch1_feedback_evidence
except Exception:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from system_control.experiment_control.feedback_evidence import summarize_wch1_feedback_evidence

QUBIT_RECON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qubit_recon_files')
if QUBIT_RECON_DIR not in sys.path:
    sys.path.insert(0, QUBIT_RECON_DIR)

# Import density matrix reconstruction
DENSITY_RECON_AVAILABLE = False
density_reconstruction = None
try:
    spec = importlib.util.spec_from_file_location(
        "density_reconstruction",
        os.path.join(QUBIT_RECON_DIR, 'density_reconstruction.py')
    )
    if spec and spec.loader:
        density_reconstruction = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(density_reconstruction)
        DENSITY_RECON_AVAILABLE = True
except Exception as e:
    print(f"Could not import density_reconstruction: {e}")
    DENSITY_RECON_AVAILABLE = False

CALIBRATED_DENSITY_RECON_AVAILABLE = False
calibrated_density_reconstruction = None
try:
    spec = importlib.util.spec_from_file_location(
        "calibrated_density_reconstruction",
        os.path.join(QUBIT_RECON_DIR, 'calibrated_density_reconstruction.py')
    )
    if spec and spec.loader:
        calibrated_density_reconstruction = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(calibrated_density_reconstruction)
        CALIBRATED_DENSITY_RECON_AVAILABLE = True
except Exception as e:
    print(f"Could not import calibrated_density_reconstruction: {e}")
    CALIBRATED_DENSITY_RECON_AVAILABLE = False

# Import ML correction interface
ML_CORRECTION_AVAILABLE = False
ml_correction_interface = None
try:
    spec = importlib.util.spec_from_file_location(
        "ml_correction_interface",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_correction_interface.py')
    )
    if spec and spec.loader:
        ml_correction_interface = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ml_correction_interface)
        ML_CORRECTION_AVAILABLE = True
except Exception as e:
    print(f"Could not import ML correction interface: {e}")
    ML_CORRECTION_AVAILABLE = False


# ============================================================
# Simple Heatmap Table Widget
# ============================================================

class HeatmapTable(QTableWidget):
    def __init__(self, rows, cols, title, color_scheme="red"):
        """Simple heatmap table with linear (row-major) ordering."""
        super().__init__(rows, cols)
        self.title = title
        self.color_scheme = color_scheme  # "red", "green", "blue", "gray", "purple", "orange"
        self._rows = rows
        self._cols = cols

        cell_size = 20 if max(rows, cols) >= 16 else 24

        # Keep the heatmap as a tight grid, without the default table chrome
        # adding a small blank gutter around the data cells.
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setMinimumSectionSize(16)
        self.verticalHeader().setMinimumSectionSize(16)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.setFrameShape(QFrame.NoFrame)
        self.setLineWidth(0)
        self.setMidLineWidth(0)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Set equal column widths
        for i in range(cols):
            self.setColumnWidth(i, cell_size)

        # Set equal row heights
        for i in range(rows):
            self.setRowHeight(i, cell_size)

        # Set larger font for table cells
        font = self.font()
        font.setPointSize(7 if cell_size <= 20 else 8)
        self.setFont(font)
        self._fit_to_cells()

    def _fit_to_cells(self):
        width = sum(self.columnWidth(i) for i in range(self._cols))
        height = sum(self.rowHeight(i) for i in range(self._rows))
        self.setFixedSize(width + self.frameWidth() * 2, height + self.frameWidth() * 2)

    def update_data(self, data):
        """Update table with data and apply heatmap coloring."""
        total = self._rows * self._cols
        if len(data) < total:
            return

        # Normalize data for coloring
        try:
            arr = np.asarray(data[:total], dtype=float)
            max_val = float(arr.max()) if arr.size > 0 and arr.max() > 0 else 1.0
        except Exception:
            max_val = max(data[:total]) if max(data[:total]) > 0 else 1

        # Fill table row by row (linear ordering)
        for r in range(self._rows):
            for c in range(self._cols):
                idx = r * self._cols + c
                try:
                    value_f = float(data[idx])
                except Exception:
                    value_f = 0.0

                # Display value as text
                if abs(value_f - round(value_f)) < 1e-9:
                    text = str(int(round(value_f)))
                else:
                    text = f"{value_f:.1f}"

                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)

                # Compute color intensity. Guard against overflow if value_f
                # is astronomical (e.g. segment_value exploded because
                # signal_bins==0 upstream); clamp value_f first.
                if not (-1e6 <= value_f <= 1e6):
                    value_f = 0.0
                intensity = 0
                if max_val > 0:
                    intensity = min(255, int(255.0 * (value_f / max_val)))

                def _byte(v: int) -> int:
                    """Clamp channel value to unsigned byte range for QColor."""
                    return max(0, min(255, int(v)))

                # Apply color scheme
                if self.color_scheme == "red":
                    item.setBackground(QColor(255, _byte(255 - intensity), _byte(255 - intensity)))
                elif self.color_scheme == "green":
                    item.setBackground(QColor(_byte(255 - intensity), 255, _byte(255 - intensity)))
                elif self.color_scheme == "blue":
                    item.setBackground(QColor(_byte(255 - intensity), _byte(255 - intensity), 255))
                elif self.color_scheme == "gray":
                    # Gray scale: 0 = black, 255 = white
                    gray = _byte(255 * (value_f / max_val)) if max_val > 0 else 0
                    item.setBackground(QColor(gray, gray, gray))
                elif self.color_scheme == "purple":
                    # Purple scale: from white to purple (RGB: 128,0,128)
                    red = _byte(255 - intensity * 127 / 255)
                    green = _byte(255 - intensity * 255 / 255)
                    blue = _byte(255 - intensity * 127 / 255)
                    item.setBackground(QColor(red, green, blue))
                elif self.color_scheme == "orange":
                    # Orange scale: from white to orange (RGB: 255,165,0)
                    red = 255
                    green = _byte(255 - intensity * 90 / 255)
                    blue = _byte(255 - intensity * 255 / 255)
                    item.setBackground(QColor(red, green, blue))

                self.setItem(r, c, item)


class ScalarSummaryTable(QTableWidget):
    """Compact table for scalar counter/status values."""

    def __init__(self, labels):
        super().__init__(len(labels), 2)
        self._items = {}
        self.verticalHeader().setVisible(False)
        self.setHorizontalHeaderLabels(["Metric", "Value"])
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(True)
        self.setWordWrap(True)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        row_height = 18
        self.horizontalHeader().setFixedHeight(22)
        font = self.font()
        font.setPointSize(9)
        self.setFont(font)
        for row, label in enumerate(labels):
            self.setRowHeight(row, row_height)
            name_item = QTableWidgetItem(label)
            name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value_item = QTableWidgetItem("0")
            value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, value_item)
            self._items[label] = value_item
        self.setFixedHeight(26 + len(labels) * row_height)

    def set_value(self, label, value):
        item = self._items.get(label)
        if item is not None:
            item.setText(str(value))


class Driver(QObject):
    data_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.alive = False
        self.lock = threading.Lock()
        self.read_thread = None

        # Counter configuration: 75 cells, modulo=300
        self.num_cells = 75

        self._last_read_times = None
        self._last_write_seq = None
        self.read_overflows = 0
        self.write_overflows = 0
        self.overflow_sensitivity = 10
        self.overflow_sensitivity_signal = 10
        self.overflow_sensitivity_idler = 10
        self.overflow_sensitivity_coin = 10

        self.last_signal_bins = None
        self.last_idler_bins = None
        self.last_coin_bins = None
        self.signal_overflows = None
        self.idler_overflows = None
        self.coin_overflows = None

        # --- Baseline-subtracting counters (UI-friendly, overflow-safe) ---
        # Start at 0 on Start; survive wraps. 3Q protocol uses 4-byte (32-bit)
        # bins for signal/idler/coin per cell, plus 40-bit read/write.
        self.baseline_counters = {
            'read_times': BaselineCounter(modulus=1 << 40, sensitivity=10, name='read_times'),
            'write_seq':  BaselineCounter(modulus=1 << 40, sensitivity=10, name='write_seq'),
            # 3Q bins are 32-bit (4-byte reads), but per-protocol modulo=300
            # for histogram-like stats; sensitivity=1 still catches wraps.
            'signal_bins': [BaselineCounter(modulus=1 << 32, sensitivity=1, name=f'signal[{i}]')
                            for i in range(self.num_cells)],
            'idler_bins':  [BaselineCounter(modulus=1 << 32, sensitivity=1, name=f'idler[{i}]')
                            for i in range(self.num_cells)],
            'coin_bins':   [BaselineCounter(modulus=1 << 32, sensitivity=1, name=f'coin[{i}]')
                            for i in range(self.num_cells)],
            'signal_hist': [BaselineCounter(modulus=1 << 32, sensitivity=1, name=f'signal_hist[{i}]')
                            for i in range(10)],
            'idler_hist':  [BaselineCounter(modulus=1 << 32, sensitivity=1, name=f'idler_hist[{i}]')
                            for i in range(10)],
        }
        self._baseline_captured = False

    def open(self, port_name):
        try:
            self.serial_port = serial.Serial(port=port_name, baudrate=500000, timeout=0.1)
            self.serial_port.reset_input_buffer()
            self.alive = True
            self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
            self.read_thread.start()
            print(f"Opened {port_name}")
        except Exception as e:
            print(f"Failed to open {port_name}: {e}")

    def close(self):
        self.alive = False
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                print("Serial port closed")
        except Exception as e:
            print("Error closing serial:", e)

    def set_idle(self):
        with self.lock:
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b"n")
                    self.serial_port.reset_input_buffer()
                except Exception as e:
                    print("set_idle error:", e)

    def set_output(self):
        # Capture current counter values as UI baseline so the display starts
        # at 0 at Start and only grows.
        self._capture_baselines()
        with self.lock:
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b"y")
                    self.serial_port.reset_input_buffer()
                except Exception as e:
                    print("set_output error:", e)

        if not self.alive:
            self.alive = True
            self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
            self.read_thread.start()

    def _read_aligned_frame(self, n, timeout_s=2.0, idle_gap_s=0.2):
        if not (self.serial_port and self.serial_port.is_open):
            return b''
        end = time.time() + timeout_s
        buf = bytearray()
        last_byte_time = None
        while len(buf) < n and time.time() < end and self.alive:
            waiting = self.serial_port.in_waiting
            if waiting <= 0:
                time.sleep(0.002)
                continue

            now = time.time()
            if last_byte_time is not None and (now - last_byte_time) >= idle_gap_s and buf:
                buf.clear()

            chunk = self.serial_port.read(min(waiting, n - len(buf)))
            if chunk:
                buf.extend(chunk)
                last_byte_time = time.time()
        return bytes(buf) if len(buf) == n else b''

    def _capture_baselines(self):
        """Snapshot current corrected counter values as UI baselines (Start)."""
        for key, c in self.baseline_counters.items():
            if isinstance(c, list):
                for sub in c:
                    try:
                        sub.set_baseline_from_current()
                    except RuntimeError:
                        pass
            else:
                try:
                    c.set_baseline_from_current()
                except RuntimeError:
                    pass
        self._baseline_captured = True

    def _reset_baseline_counters(self):
        """Forget baseline + carry state. Called from clear_data()."""
        for key, c in self.baseline_counters.items():
            if isinstance(c, list):
                for sub in c:
                    sub.reset()
            else:
                c.reset()
        self._baseline_captured = False

    def read_loop(self):
        layout = LAYOUT_3Q
        FRAME_BYTES = layout.frame_bytes
        print("[INFO] Read loop started - byte-aligned 75-cell protocol (modulo=300)")

        while self.alive:
            try:
                if not (self.serial_port and self.serial_port.is_open):
                    time.sleep(0.1)
                    continue

                raw = self._read_aligned_frame(FRAME_BYTES, timeout_s=1.5)

                if len(raw) < FRAME_BYTES:
                    continue

                # Parse according to 75-cell protocol:
                # [0~11] Multi-click (12bits)
                # [12~21] Signal interval (10bits)
                # [22~1520] Signal hist (75 × stride=20, 19bits/cell)
                # [1521~1530] Idler interval (10bits)
                # [1531~3030] Idler hist (75 × stride=20, 19bits/cell)
                # [3031~3040] Coin interval (10bits)
                # [3041~3866] Coin hist (75 × stride=11, 9bits/cell)
                # [3867~3906] Read times (40bits)
                # [3907~3946] Write times (40bits)
                # [3947~4036] Signal cellhist (10 × stride=9, 8bits/bin)
                # [4037~4126] Idler cellhist (10 × stride=9, 8bits/bin)

                # Signal histogram (75 cells, 19 bits each)
                # Idler histogram (75 cells, 19 bits each)
                # Coin histogram (75 cells, 9 bits each)
                # Signal cellhist (10 bins, 8 bits each)
                # Idler cellhist (10 bins, 8 bits each)
                # Current byte offsets are defined in frame_protocol.LAYOUT_3Q:
                # 0: multi2, 4: multi3, 8: multi4,
                # 12: six multi2-pair counters, 36: read, 44: write,
                # 52/56/60: signal/idler/coin totals, 64: histogram payload.
                def read_uint(offset, size):
                    return int.from_bytes(raw[offset:offset + size], byteorder='big', signed=False)

                multi_counts = decode_multi_click_counts(raw, layout)
                multi2 = multi_counts['multi2_count']
                multi3 = multi_counts['multi3_count']
                multi4 = multi_counts['multi4_count']
                multi2_pairs = decode_multi2_pair_counts(raw, offset=layout.pair_offset)
                multi_click = multi_counts['multi_click']
                raw_read_times = read_uint(layout.read_offset, layout.read_size)
                raw_write_seq = read_uint(layout.write_offset, layout.write_size)
                counter_decode = decode_3qubit_counter_words(raw_read_times, raw_write_seq)
                read_times = counter_decode.read_count
                write_seq = counter_decode.write_count
                signal_interval = read_uint(layout.signal_total_offset, layout.scalar_size)
                idler_interval = read_uint(layout.idler_total_offset, layout.scalar_size)
                coin_interval = read_uint(layout.coin_total_offset, layout.scalar_size)

                signal_bins_raw = [
                    read_uint(layout.signal_base + i * layout.signal_stride, layout.signal_stride)
                    for i in range(self.num_cells)
                ]
                idler_bins_raw = [
                    read_uint(layout.idler_base + i * layout.idler_stride, layout.idler_stride)
                    for i in range(self.num_cells)
                ]
                coin_bins_raw = [
                    read_uint(layout.coin_base + i * layout.coin_stride, layout.coin_stride)
                    for i in range(self.num_cells)
                ]
                signal_hist_raw = [
                    read_uint(layout.signal_cell_base + i * layout.signal_cell_stride, layout.signal_cell_stride)
                    for i in range(layout.cell_hist_bins)
                ]
                idler_hist_raw = [
                    read_uint(layout.idler_cell_base + i * layout.idler_cell_stride, layout.idler_cell_stride)
                    for i in range(layout.cell_hist_bins)
                ]

                # Overflow handling for signal_hist (32-bit)
                if not hasattr(self, 'last_signal_hist_raw') or self.last_signal_hist_raw is None:
                    self.last_signal_hist_raw = [int(x) for x in signal_hist_raw]
                    self.signal_hist_overflows = [0] * 10
                    signal_hist = [int(x) for x in signal_hist_raw]
                else:
                    if getattr(self, 'signal_hist_overflows', None) is None:
                        self.signal_hist_overflows = [0] * 10
                    signal_hist = []
                    for i in range(10):
                        raw = int(signal_hist_raw[i])
                        last_raw = int(self.last_signal_hist_raw[i])
                        if raw < last_raw:
                            if (last_raw - raw) <= self.overflow_sensitivity_signal:
                                raw = last_raw
                            else:
                                self.signal_hist_overflows[i] += 1
                        corrected = raw + (self.signal_hist_overflows[i] << 32)
                        signal_hist.append(int(corrected))
                        self.last_signal_hist_raw[i] = raw

                # Overflow handling for idler_hist (32-bit)
                if not hasattr(self, 'last_idler_hist_raw') or self.last_idler_hist_raw is None:
                    self.last_idler_hist_raw = [int(x) for x in idler_hist_raw]
                    self.idler_hist_overflows = [0] * 10
                    idler_hist = [int(x) for x in idler_hist_raw]
                else:
                    if getattr(self, 'idler_hist_overflows', None) is None:
                        self.idler_hist_overflows = [0] * 10
                    idler_hist = []
                    for i in range(10):
                        raw = int(idler_hist_raw[i])
                        last_raw = int(self.last_idler_hist_raw[i])
                        if raw < last_raw:
                            if (last_raw - raw) <= self.overflow_sensitivity_idler:
                                raw = last_raw
                            else:
                                self.idler_hist_overflows[i] += 1
                        corrected = raw + (self.idler_hist_overflows[i] << 32)
                        idler_hist.append(int(corrected))
                        self.last_idler_hist_raw[i] = raw

                # Overflow handling for read_times (64-bit)
                if self._last_read_times is None:
                    self._last_read_times = int(read_times)
                    self.read_overflows = 0
                    real_read_times = int(read_times)
                else:
                    cur_rt = int(read_times)
                    last_rt = int(self._last_read_times)
                    if cur_rt < last_rt:
                        if (last_rt - cur_rt) <= self.overflow_sensitivity:
                            cur_rt = last_rt
                        else:
                            self.read_overflows += 1
                    self._last_read_times = cur_rt
                    real_read_times = cur_rt + (self.read_overflows << 64)

                # Overflow handling for write_seq (64-bit)
                if self._last_write_seq is None:
                    self._last_write_seq = int(write_seq)
                    self.write_overflows = 0
                    real_write_seq = int(write_seq)
                else:
                    cur_ws = int(write_seq)
                    last_ws = int(self._last_write_seq)
                    if cur_ws < last_ws:
                        if (last_ws - cur_ws) <= self.overflow_sensitivity:
                            cur_ws = last_ws
                        else:
                            self.write_overflows += 1
                    self._last_write_seq = cur_ws
                    real_write_seq = cur_ws + (self.write_overflows << 64)

                # Overflow handling for signal_bins (32-bit)
                if self.last_signal_bins is None:
                    self.last_signal_bins = [int(x) for x in signal_bins_raw]
                    self.signal_overflows = [0] * self.num_cells
                    signal_bins = [int(x) for x in signal_bins_raw]
                else:
                    if self.signal_overflows is None:
                        self.signal_overflows = [0] * self.num_cells
                    signal_bins = []
                    for i in range(self.num_cells):
                        raw = int(signal_bins_raw[i])
                        last_raw = int(self.last_signal_bins[i])
                        if raw < last_raw:
                            if (last_raw - raw) <= self.overflow_sensitivity_signal:
                                raw = last_raw
                            else:
                                self.signal_overflows[i] += 1
                        corrected = raw + (self.signal_overflows[i] << 32)
                        signal_bins.append(int(corrected))
                        self.last_signal_bins[i] = raw

                # Overflow handling for idler_bins (32-bit)
                if self.last_idler_bins is None:
                    self.last_idler_bins = [int(x) for x in idler_bins_raw]
                    self.idler_overflows = [0] * self.num_cells
                    idler_bins = [int(x) for x in idler_bins_raw]
                else:
                    if self.idler_overflows is None:
                        self.idler_overflows = [0] * self.num_cells
                    idler_bins = []
                    for i in range(self.num_cells):
                        raw = int(idler_bins_raw[i])
                        last_raw = int(self.last_idler_bins[i])
                        if raw < last_raw:
                            if (last_raw - raw) <= self.overflow_sensitivity_idler:
                                raw = last_raw
                            else:
                                self.idler_overflows[i] += 1
                        corrected = raw + (self.idler_overflows[i] << 32)
                        idler_bins.append(int(corrected))
                        self.last_idler_bins[i] = raw

                # Overflow handling for coin_bins (32-bit)
                if self.last_coin_bins is None:
                    self.last_coin_bins = [int(x) for x in coin_bins_raw]
                    self.coin_overflows = [0] * self.num_cells
                    coin_bins = [int(x) for x in coin_bins_raw]
                else:
                    if self.coin_overflows is None:
                        self.coin_overflows = [0] * self.num_cells
                    coin_bins = []
                    for i in range(self.num_cells):
                        raw = int(coin_bins_raw[i])
                        last_raw = int(self.last_coin_bins[i])
                        if raw < last_raw:
                            if (last_raw - raw) <= self.overflow_sensitivity_coin:
                                raw = last_raw
                            else:
                                self.coin_overflows[i] += 1
                        corrected = raw + (self.coin_overflows[i] << 32)
                        coin_bins.append(int(corrected))
                        self.last_coin_bins[i] = raw

                # Calculate per-segment values: coin/signal - idler/read_count
                segment_values = []
                coin_idler_values = []
                for i in range(self.num_cells):
                    sig = float(signal_bins[i])
                    idl = float(idler_bins[i])
                    coi = float(coin_bins[i])
                    read = float(real_read_times) if real_read_times > 0 else 1.0
                    if sig <= 0:
                        # No signal -> no contribution to (coin/signal - idler/read).
                        value = 0.0
                    else:
                        try:
                            value = (coi / sig) - (idl / read)
                        except Exception:
                            value = 0.0
                    # Clamp to a safe range so the downstream QColor mapping
                    # never sees a value that would overflow int32.
                    if not (-1e6 <= value <= 1e6):
                        value = 0.0
                    segment_values.append(value)
                    coin_idler_values.append((coi / idl) if idl > 0.0 else 0.0)

                result = {
                    'signal_bins': np.array(signal_bins, dtype=np.int64),
                    'idler_bins': np.array(idler_bins, dtype=np.int64),
                    'coin_bins': np.array(coin_bins, dtype=np.int64),
                    'signal_bins_raw': np.array(signal_bins_raw, dtype=np.int64),
                    'idler_bins_raw': np.array(idler_bins_raw, dtype=np.int64),
                    'coin_bins_raw': np.array(coin_bins_raw, dtype=np.int64),
                    'write_count': int(real_write_seq),
                    'read_count': int(real_read_times),
                    'write_count_raw': int(raw_write_seq),
                    'read_count_raw': int(raw_read_times),
                    'counter_decode_layout': counter_decode.layout,
                    'counter_decode_warning': counter_decode.warning,
                    'multi2_count': int(multi2),
                    'multi3_count': int(multi3),
                    'multi4_count': int(multi4),
                    'multi2_pairs': {key: int(value) for key, value in multi2_pairs.items()},
                    'signal_hist': np.array(signal_hist, dtype=np.int64),
                    'idler_hist': np.array(idler_hist, dtype=np.int64),
                    'signal_hist_raw': np.array(signal_hist_raw, dtype=np.int64),
                    'idler_hist_raw': np.array(idler_hist_raw, dtype=np.int64),
                    'segment_values': np.array(segment_values, dtype=np.float64),
                    'coin_idler_values': np.array(coin_idler_values, dtype=np.float64),
                    # Preserve the signed physical estimator.  The clipped
                    # legacy copy is display/backward-compatibility only.
                    'signed_segment_values': np.array(segment_values, dtype=np.float64),
                    'legacy_clipped_segment_values': np.maximum(np.array(segment_values, dtype=np.float64), 0.0),
                    'multi_click': int(multi_click),
                    'signal_interval': int(signal_interval),
                    'idler_interval': int(idler_interval),
                    'coin_interval': int(coin_interval),
                    'fpga_protocol_version': 'legacy_counter_frame_v1',
                    'counter_semantics': 'cumulative_since_start',
                    'timestamp': time.time()
                }

                # Baseline-subtracted display values (UI only). They are kept
                # in the frame so snapshots can distinguish absolute FPGA
                # counters from Start-window counters.
                bc = self.baseline_counters
                if not self._baseline_captured:
                    try:
                        self._capture_baselines()
                    except Exception:
                        pass
                try:
                    result['display_signal_bins'] = np.array(
                        [bc['signal_bins'][i].update(int(signal_bins_raw[i]))
                         for i in range(self.num_cells)], dtype=np.int64)
                    result['display_idler_bins'] = np.array(
                        [bc['idler_bins'][i].update(int(idler_bins_raw[i]))
                         for i in range(self.num_cells)], dtype=np.int64)
                    result['display_coin_bins'] = np.array(
                        [bc['coin_bins'][i].update(int(coin_bins_raw[i]))
                         for i in range(self.num_cells)], dtype=np.int64)
                    result['display_read_count'] = int(bc['read_times'].update(int(read_times)))
                    result['display_write_count'] = int(bc['write_seq'].update(int(write_seq)))
                    result['display_signal_hist'] = np.array(
                        [bc['signal_hist'][i].update(int(signal_hist_raw[i]))
                         for i in range(10)], dtype=np.int64)
                    result['display_idler_hist'] = np.array(
                        [bc['idler_hist'][i].update(int(idler_hist_raw[i]))
                         for i in range(10)], dtype=np.int64)
                except Exception as exc:
                    print(f"[baseline_counter] update error: {exc!r}")

                self.data_ready.emit(result)

            except Exception as e:
                print(f"Error reading serial: {e}")
                self.alive = False
                break


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3-Qubit Data Receiver (75 cells)")
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(
            min(2400, max(640, available.width() - 48)),
            min(1320, max(600, available.height() - 64)),
        )

        self.driver = Driver()
        self.driver.data_ready.connect(self.on_data_ready)

        self.current_data = None
        self.frame_count = 0
        self.last_frame_timestamp = None
        self.latest_rho = None
        self.latest_raw_rho = None
        self.latest_calibrated_rho = None
        self.latest_idler_calibrated_rho = None
        self.latest_retrieval_calibrated_rho = None
        self.latest_rho_variants = {}
        self.latest_analysis_rho_variants = {}
        self.latest_rho_fidelity_summary = {}
        self.feedback_context = {}
        self.rate_history = []
        self._last_rate_sample = None
        self.max_rate_history_points = 300

        # Initialize ML correction interface
        self.ml_interface = None
        if ML_CORRECTION_AVAILABLE and ml_correction_interface is not None:
            try:
                self.ml_interface = ml_correction_interface.MLCorrectionInterface(num_qubits=3)
                # Try to load trained model
                self.ml_interface.load_model()
                print("ML Correction Interface initialized for 3-qubit system")
            except Exception as e:
                print(f"Warning: Failed to initialize ML interface: {e}")

        self.setup_ui()
        self.realtime_diag_timer = QTimer(self)
        self.realtime_diag_timer.timeout.connect(self.update_realtime_diagnosis)
        self.realtime_diag_timer.start(30000)

    def setup_ui(self):
        # Create a scroll area for the main content
        scroll_area = QScrollArea()
        self.main_scroll_area = scroll_area
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create a widget to hold all the content
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(3)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(6)
        self.com_combo = QComboBox()
        self.refresh_ports()

        self.btn_open = QPushButton("Open")
        self.btn_close = QPushButton("Close")
        self.btn_start = QPushButton("Start (y)")
        self.btn_stop = QPushButton("Stop (n)")
        self.btn_clear = QPushButton("Clear")
        self.btn_ml_correction = QPushButton("ML Correction")
        self.btn_save_feedback_sample = QPushButton("Save Feedback Sample")

        # Make buttons larger for better visibility
        for btn in [
            self.btn_open,
            self.btn_close,
            self.btn_start,
            self.btn_stop,
            self.btn_clear,
            self.btn_ml_correction,
            self.btn_save_feedback_sample,
        ]:
            btn.setMinimumSize(92, 32)
            btn.setStyleSheet("font-size: 11px;")

        # Style ML button differently
        self.btn_ml_correction.setStyleSheet("font-size: 12px; background-color: #4CAF50; color: white;")
        self.btn_save_feedback_sample.setStyleSheet("font-size: 12px; background-color: #1976D2; color: white;")

        ctrl_layout.addWidget(QLabel("Port:"))
        ctrl_layout.addWidget(self.com_combo)
        ctrl_layout.addWidget(self.btn_open)
        ctrl_layout.addWidget(self.btn_close)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.btn_ml_correction)
        ctrl_layout.addWidget(self.btn_save_feedback_sample)

        main_layout.addLayout(ctrl_layout)

        self.text_edit = QTextEdit()
        self.text_edit.setFixedHeight(32)
        main_layout.addWidget(self.text_edit)

        self.lbl_feedback_context = QLabel("Feedback context: not set")
        self.lbl_feedback_context.setWordWrap(True)
        self.lbl_feedback_context.setMaximumHeight(32)
        main_layout.addWidget(self.lbl_feedback_context)

        # Display area for parsed numbers
        display_group = QGroupBox("Parsed Data (75 cells, modulo=300)")
        display_layout = QGridLayout()
        display_layout.setContentsMargins(6, 6, 6, 6)
        display_layout.setHorizontalSpacing(8)

        # Labels for sums and intervals
        self.lbl_multi_click = QLabel("Clicks: 2=0 3=0 4=0 total=0")
        self.lbl_multi2_pairs = QLabel("2-click pairs: 12=0 13=0 14=0 23=0 24=0 34=0")
        self.lbl_signal_interval = QLabel("Signal Interval: 0")
        self.lbl_idler_interval = QLabel("Idler Interval: 0")
        self.lbl_coin_interval = QLabel("Coin Interval: 0")
        self.lbl_write_count = QLabel("Write Count: 0")
        self.lbl_read_count = QLabel("Read Count: 0")
        self.lbl_signal_sum = QLabel("Signal Sum: 0")
        self.lbl_idler_sum = QLabel("Idler Sum: 0")
        self.lbl_coin_sum = QLabel("Coin Sum: 0")

        # Increase font size for labels
        font = self.lbl_multi_click.font()
        font.setPointSize(9)
        for lbl in [self.lbl_multi_click, self.lbl_multi2_pairs, self.lbl_signal_interval, self.lbl_idler_interval,
                   self.lbl_coin_interval, self.lbl_write_count, self.lbl_read_count,
                   self.lbl_signal_sum, self.lbl_idler_sum, self.lbl_coin_sum]:
            lbl.setFont(font)

        self.summary_table = ScalarSummaryTable([
            "2-click",
            "3-click",
            "4-click",
            "Multi-click Total",
            "2-click pairs",
            "Signal Interval",
            "Idler Interval",
            "Coin Interval",
            "Write Count",
            "Read Count",
            "Signal Sum",
            "Idler Sum",
            "Coin Sum",
        ])
        display_layout.addWidget(self.summary_table, 0, 0, Qt.AlignTop)

        self.rate_fig = Figure(figsize=(3.2, 1.8))
        self.rate_canvas = FigureCanvas(self.rate_fig)
        self.rate_canvas.setFixedHeight(190)
        display_layout.addWidget(self.rate_canvas, 0, 1, Qt.AlignTop)
        display_layout.setColumnStretch(0, 5)
        display_layout.setColumnStretch(1, 4)
        self._draw_rate_plot()

        display_group.setLayout(display_layout)

        realtime_group = QGroupBox("Real-time Entanglement Diagnosis (30s)")
        realtime_layout = QGridLayout()
        realtime_layout.setContentsMargins(6, 6, 6, 6)
        realtime_layout.setHorizontalSpacing(8)
        realtime_layout.setVerticalSpacing(4)
        self.lbl_diag_model = QLabel("Model: not loaded")
        self.lbl_diag_modes = QLabel("Excitation modes: waiting for rho")
        self.lbl_diag_families = QLabel("State families: --")
        self.lbl_diag_errors = QLabel("Error sources: waiting for rho")
        self.lbl_diag_metrics = QLabel("Fidelity/Purity/OOD: --")
        self.lbl_diag_metrics.setWordWrap(True)
        self.diagnosis_table = ScalarSummaryTable([
            "Model",
            "Excitation Modes",
            "State Families",
            "Error Sources",
            "Fidelity",
            "Top F Fits",
            "Purity",
            "OOD",
            "Ideal F",
            "Model Audit",
        ])
        realtime_layout.addWidget(self.diagnosis_table, 0, 0)
        self._update_diagnosis_table(
            model="not loaded",
            excitation_modes="waiting for rho",
            state_families="--",
            error_sources="waiting for rho",
            fidelity="--",
            top_fits="--",
            purity="--",
            ood="--",
            ideal_f="--",
            model_audit="--",
        )
        self.estimator_table = QTableWidget(len(ESTIMATOR_METRICS), len(VARIANT_LABELS) + 1)
        self.estimator_table.setHorizontalHeaderLabels(
            ["Metric"] + [label for _key, label in VARIANT_LABELS]
        )
        self.estimator_table.verticalHeader().setVisible(False)
        self.estimator_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.estimator_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.estimator_table.setAlternatingRowColors(True)
        for row in range(self.estimator_table.rowCount()):
            self.estimator_table.setRowHeight(row, 17)
        self.estimator_table.horizontalHeader().setFixedHeight(22)
        self.estimator_table.setFixedHeight(26 + self.estimator_table.rowCount() * 18)
        estimator_header = self.estimator_table.horizontalHeader()
        estimator_header.setStretchLastSection(False)
        estimator_header.setMinimumSectionSize(72)
        estimator_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for column in range(1, self.estimator_table.columnCount()):
            estimator_header.setSectionResizeMode(column, QHeaderView.Stretch)
        realtime_layout.addWidget(self.estimator_table, 0, 1, 2, 1)
        self._update_estimator_table({})
        self.mode_probability_panel = QWidget()
        mode_probability_layout = QGridLayout(self.mode_probability_panel)
        mode_probability_layout.setContentsMargins(0, 0, 0, 0)
        mode_probability_layout.setHorizontalSpacing(5)
        mode_probability_layout.setVerticalSpacing(2)
        self.mode_probability_labels = {}
        self.mode_probability_bars = {}
        for mode_index in range(1, 4):
            key = f"mode_{mode_index}"
            label = QLabel(f"Mode {mode_index}")
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setFormat("0.0%")
            bar.setTextVisible(True)
            bar.setFixedHeight(16)
            self.mode_probability_labels[key] = label
            self.mode_probability_bars[key] = bar
            mode_probability_layout.addWidget(label, mode_index - 1, 0)
            mode_probability_layout.addWidget(bar, mode_index - 1, 1)
        realtime_layout.addWidget(self.mode_probability_panel, 1, 0, Qt.AlignTop)
        self.w_subspace_fig = Figure(figsize=(4.4, 1.6))
        self.w_subspace_canvas = FigureCanvas(self.w_subspace_fig)
        self.w_subspace_canvas.setFixedHeight(185)
        realtime_layout.addWidget(self.w_subspace_canvas, 0, 2)
        draw_w_subspace_summary(self.w_subspace_fig, None, 3, message="waiting for rho")
        self.w_subspace_canvas.draw()

        self.state_family_fig = Figure(figsize=(4.4, 1.6))
        self.state_family_canvas = FigureCanvas(self.state_family_fig)
        self.state_family_canvas.setFixedHeight(185)
        realtime_layout.addWidget(self.state_family_canvas, 1, 2)
        self._clear_state_family_table("waiting for rho")
        realtime_layout.setColumnStretch(0, 4)
        realtime_layout.setColumnStretch(1, 6)
        realtime_layout.setColumnStretch(2, 5)
        realtime_group.setLayout(realtime_layout)
        self.status_widget = QWidget()
        status_layout = QHBoxLayout(self.status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)
        status_layout.addWidget(display_group, 3)
        status_layout.addWidget(realtime_group, 7)
        self.status_widget.setMaximumHeight(410)
        self.status_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        main_layout.addWidget(self.status_widget)

        self.physics_dashboard_group = QGroupBox("Best-F Physics & Inference Dashboard")
        physics_dashboard_layout = QHBoxLayout(self.physics_dashboard_group)
        physics_dashboard_layout.setContentsMargins(7, 7, 7, 7)
        physics_dashboard_layout.setSpacing(8)
        self.physics_metric_grid = PhysicsMetricGrid()
        self.inference_summary_table = InferenceSummaryTable()
        self.physics_dashboard_fig = Figure(figsize=(9.0, 2.0))
        self.physics_dashboard_canvas = FigureCanvas(self.physics_dashboard_fig)
        self.physics_dashboard_canvas.setFixedHeight(180)
        physics_dashboard_layout.addWidget(self.physics_metric_grid, 5)
        physics_dashboard_layout.addWidget(self.inference_summary_table, 3)
        physics_dashboard_layout.addWidget(self.physics_dashboard_canvas, 6)
        self.physics_dashboard_group.setFixedHeight(205)
        main_layout.addWidget(self.physics_dashboard_group)
        self._update_physics_dashboard({}, {})

        # Combined display area with two rows
        combined_group = QGroupBox("Data Analysis")
        combined_group.setMaximumHeight(700)
        combined_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        combined_layout = QVBoxLayout()
        combined_layout.setContentsMargins(8, 8, 8, 8)
        combined_layout.setSpacing(6)

        # First row: All four matrices (Signal, Idler, Coin, Calculated)
        row1_group = QGroupBox("Matrices (8x8 + 1x11)")
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(5, 5, 5, 5)
        row1_layout.setSpacing(5)

        # Signal matrix
        signal_matrix_layout = QVBoxLayout()
        signal_matrix_layout.setSpacing(3)
        signal_label = QLabel("Signal")
        signal_label.setAlignment(Qt.AlignCenter)
        signal_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        signal_matrix_layout.addWidget(signal_label)
        signal_8x8_layout = QHBoxLayout()
        signal_8x8_layout.setContentsMargins(0, 0, 0, 0)
        self.signal_8x8_table = HeatmapTable(8, 8, "Signal 8x8", color_scheme="red")
        signal_8x8_layout.addWidget(self.signal_8x8_table, 0, Qt.AlignHCenter)
        signal_matrix_layout.addLayout(signal_8x8_layout)
        self.signal_1x11_table = HeatmapTable(1, 11, "Signal 1x11", color_scheme="red")
        signal_matrix_layout.addWidget(self.signal_1x11_table, 0, Qt.AlignHCenter)

        # Idler matrix
        idler_matrix_layout = QVBoxLayout()
        idler_matrix_layout.setSpacing(3)
        idler_label = QLabel("Idler")
        idler_label.setAlignment(Qt.AlignCenter)
        idler_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        idler_matrix_layout.addWidget(idler_label)
        idler_8x8_layout = QHBoxLayout()
        idler_8x8_layout.setContentsMargins(0, 0, 0, 0)
        self.idler_8x8_table = HeatmapTable(8, 8, "Idler 8x8", color_scheme="green")
        idler_8x8_layout.addWidget(self.idler_8x8_table, 0, Qt.AlignHCenter)
        idler_matrix_layout.addLayout(idler_8x8_layout)
        self.idler_1x11_table = HeatmapTable(1, 11, "Idler 1x11", color_scheme="green")
        idler_matrix_layout.addWidget(self.idler_1x11_table, 0, Qt.AlignHCenter)

        # Coin matrix
        coin_matrix_layout = QVBoxLayout()
        coin_matrix_layout.setSpacing(3)
        coin_label = QLabel("Coin")
        coin_label.setAlignment(Qt.AlignCenter)
        coin_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        coin_matrix_layout.addWidget(coin_label)
        coin_8x8_layout = QHBoxLayout()
        coin_8x8_layout.setContentsMargins(0, 0, 0, 0)
        self.coin_8x8_table = HeatmapTable(8, 8, "Coin 8x8", color_scheme="blue")
        coin_8x8_layout.addWidget(self.coin_8x8_table, 0, Qt.AlignHCenter)
        coin_matrix_layout.addLayout(coin_8x8_layout)
        self.coin_1x11_table = HeatmapTable(1, 11, "Coin 1x11", color_scheme="blue")
        coin_matrix_layout.addWidget(self.coin_1x11_table, 0, Qt.AlignHCenter)

        # Calculated values matrix display (8x8 + 1x11)
        calc_matrix_layout = QVBoxLayout()
        calc_matrix_layout.setSpacing(3)
        calc_label = QLabel("Calculated x1000 (coin/signal - idler/read)")
        calc_label.setAlignment(Qt.AlignCenter)
        calc_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        calc_matrix_layout.addWidget(calc_label)

        calc_8x8_layout = QHBoxLayout()
        calc_8x8_layout.setContentsMargins(0, 0, 0, 0)
        self.calc_8x8_table = HeatmapTable(8, 8, "Calculated 8x8", color_scheme="purple")
        calc_8x8_layout.addWidget(self.calc_8x8_table, 0, Qt.AlignHCenter)
        calc_matrix_layout.addLayout(calc_8x8_layout)

        self.calc_1x11_table = HeatmapTable(1, 11, "Calculated 1x11", color_scheme="purple")
        calc_matrix_layout.addWidget(self.calc_1x11_table, 0, Qt.AlignHCenter)

        row1_layout.addLayout(signal_matrix_layout)
        row1_layout.addLayout(idler_matrix_layout)
        row1_layout.addLayout(coin_matrix_layout)
        row1_layout.addLayout(calc_matrix_layout)
        row1_group.setLayout(row1_layout)
        combined_layout.addWidget(row1_group)

        # Second row: Histograms + raw/calibrated density matrix plots
        row2_group = QGroupBox("Histograms & Density Matrix ρ")
        row2_group.setTitle("Cell Timing Histograms & Density Matrix")
        row2_group.setMaximumHeight(185)
        row2_layout = QHBoxLayout()
        row2_layout.setContentsMargins(5, 5, 5, 5)
        row2_layout.setSpacing(5)

        # Signal histogram
        signal_hist_layout = QVBoxLayout()
        signal_hist_label = QLabel("Signal Cell Histogram (20 ns bins)")
        signal_hist_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        signal_hist_layout.addWidget(signal_hist_label)
        self.signal_hist_canvas = FigureCanvas(Figure(figsize=(2.8, 1.6)))
        self.signal_hist_ax = self.signal_hist_canvas.figure.add_subplot(111)
        signal_hist_layout.addWidget(self.signal_hist_canvas)

        # Idler histogram
        idler_hist_layout = QVBoxLayout()
        idler_hist_label = QLabel("Idler Cell Histogram (20 ns bins)")
        idler_hist_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        idler_hist_layout.addWidget(idler_hist_label)
        self.idler_hist_canvas = FigureCanvas(Figure(figsize=(2.8, 1.6)))
        self.idler_hist_ax = self.idler_hist_canvas.figure.add_subplot(111)
        idler_hist_layout.addWidget(self.idler_hist_canvas)

        # Re(ρ) 3D histogram
        re_rho_layout = QVBoxLayout()
        re_rho_label = QLabel("Re(ρ)")
        re_rho_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        re_rho_label.setText("Cal Re(rho)")
        re_rho_layout.addWidget(re_rho_label)
        self.re_rho_canvas = FigureCanvas(Figure(figsize=(2.6, 1.6)))
        self.re_rho_ax = self.re_rho_canvas.figure.add_subplot(111, projection='3d')
        re_rho_layout.addWidget(self.re_rho_canvas)

        # Im(ρ) 3D histogram
        im_rho_layout = QVBoxLayout()
        im_rho_label = QLabel("Im(ρ)")
        im_rho_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        im_rho_label.setText("Cal Im(rho)")
        im_rho_layout.addWidget(im_rho_label)
        self.im_rho_canvas = FigureCanvas(Figure(figsize=(2.6, 1.6)))
        self.im_rho_ax = self.im_rho_canvas.figure.add_subplot(111, projection='3d')
        im_rho_layout.addWidget(self.im_rho_canvas)

        raw_re_rho_layout = QVBoxLayout()
        raw_re_rho_label = QLabel("Raw Re(rho)")
        raw_re_rho_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        raw_re_rho_layout.addWidget(raw_re_rho_label)
        self.raw_re_rho_canvas = FigureCanvas(Figure(figsize=(2.6, 1.6)))
        self.raw_re_rho_ax = self.raw_re_rho_canvas.figure.add_subplot(111, projection='3d')
        raw_re_rho_layout.addWidget(self.raw_re_rho_canvas)

        raw_im_rho_layout = QVBoxLayout()
        raw_im_rho_label = QLabel("Raw Im(rho)")
        raw_im_rho_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        raw_im_rho_layout.addWidget(raw_im_rho_label)
        self.raw_im_rho_canvas = FigureCanvas(Figure(figsize=(2.6, 1.6)))
        self.raw_im_rho_ax = self.raw_im_rho_canvas.figure.add_subplot(111, projection='3d')
        raw_im_rho_layout.addWidget(self.raw_im_rho_canvas)

        row2_layout.addLayout(signal_hist_layout)
        row2_layout.addLayout(idler_hist_layout)
        row2_layout.addLayout(re_rho_layout)
        row2_layout.addLayout(im_rho_layout)
        row2_layout.addLayout(raw_re_rho_layout)
        row2_layout.addLayout(raw_im_rho_layout)
        row2_group.setLayout(row2_layout)
        combined_layout.addWidget(row2_group)

        combined_group.setLayout(combined_layout)
        main_layout.addWidget(combined_group)

        pointwise_group = QGroupBox("Operation-point digital-twin diagnostics (shadow only)")
        pointwise_group.setCheckable(True)
        pointwise_group.setChecked(False)
        pointwise_group_layout = QVBoxLayout(pointwise_group)
        self.pointwise_content = QWidget()
        pointwise_layout = QVBoxLayout(self.pointwise_content)
        pointwise_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_pointwise_status = QLabel("Shadow diagnostics: waiting for an operation-point snapshot")
        self.lbl_pointwise_status.setWordWrap(True)
        pointwise_layout.addWidget(self.lbl_pointwise_status)
        self.pointwise_fig = Figure(figsize=(10.5, 5.1))
        self.pointwise_canvas = FigureCanvas(self.pointwise_fig)
        self.pointwise_canvas.setFixedHeight(390)
        pointwise_layout.addWidget(self.pointwise_canvas)
        draw_operation_point_diagnostics(self.pointwise_fig, "3qubit", {})
        self.pointwise_canvas.draw()
        pointwise_group_layout.addWidget(self.pointwise_content)
        self.pointwise_content.setVisible(False)
        pointwise_group.toggled.connect(self.pointwise_content.setVisible)
        main_layout.addWidget(pointwise_group)
        main_layout.addStretch(1)

        # Set the content widget to the scroll area
        scroll_area.setWidget(content_widget)

        # Create the main layout for the window and add the scroll area
        window_layout = QVBoxLayout(self)
        window_layout.addWidget(scroll_area)

        self.btn_open.clicked.connect(self.open_serial)
        self.btn_close.clicked.connect(self.close_serial)
        self.btn_start.clicked.connect(self.start_output)
        self.btn_stop.clicked.connect(self.stop_output)
        self.btn_clear.clicked.connect(self.clear_data)
        self.btn_ml_correction.clicked.connect(self.request_ml_correction)
        self.btn_save_feedback_sample.clicked.connect(self.save_feedback_sample)

    def set_feedback_context(self, context):
        self.feedback_context = dict(context or {})
        state = self.feedback_context.get("target_state_params", {})
        name = state.get("name", "ui_state")
        mode = self.feedback_context.get("excitation_mode", "--")
        model_path = self.feedback_context.get("model_path", "ml_recon/models/feedback_model_3qubit.json")
        evidence = summarize_wch1_feedback_evidence(self.feedback_context, "3qubit")
        if hasattr(self, "lbl_feedback_context"):
            self.lbl_feedback_context.setText(
                f"Feedback context: {name}, excitation={mode}, model={model_path}; "
                f"W_CH1 feedback evidence: {evidence.detail}"
            )
        if hasattr(self, "text_edit"):
            self.text_edit.append(f"Feedback context updated: {self.feedback_context}")

    def refresh_ports(self):
        self.com_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        items = ports[:] if ports else []
        if 'COM6' not in items:
            items.insert(0, 'COM6')
        self.com_combo.addItems(items if items else ["No ports found"])
        try:
            self.com_combo.setCurrentText('COM6')
        except Exception:
            pass

    def open_serial(self):
        port = self.com_combo.currentText()
        if "No ports" in port:
            self.text_edit.append("No valid COM port found.")
            return
        self.driver.open(port)
        self.text_edit.append(f"Opened {port}")

    def close_serial(self):
        self.driver.close()
        self.text_edit.append("Closed serial port.")

    def start_output(self):
        self._reset_rate_history()
        self.driver.set_output()
        self.text_edit.append("Sent 'y' to start output.")

    def stop_output(self):
        self.driver.set_idle()
        self.text_edit.append("Sent 'n' to stop output.")

    def clear_data(self):
        self.text_edit.append("Data cleared - counters reset")
        self.driver._last_read_times = None
        self.driver._last_write_seq = None
        self.driver.read_overflows = 0
        self.driver.write_overflows = 0
        self.driver.last_signal_bins = None
        self.driver.last_idler_bins = None
        self.driver.last_coin_bins = None
        self.driver.signal_overflows = None
        self.driver.idler_overflows = None
        self.driver.coin_overflows = None
        if hasattr(self.driver, 'last_signal_hist_raw'):
            self.driver.last_signal_hist_raw = None
        if hasattr(self.driver, 'last_idler_hist_raw'):
            self.driver.last_idler_hist_raw = None
        # Reset baseline-subtracting UI counters too, so the next Start begins
        # from a fresh 0 instead of carrying over wraps + baseline.
        try:
            self.driver._reset_baseline_counters()
        except Exception:
            pass
        self.current_data = None
        self.frame_count = 0
        self.last_frame_timestamp = None
        self.latest_rho = None
        self.latest_raw_rho = None
        self.latest_calibrated_rho = None
        self.latest_idler_calibrated_rho = None
        self.latest_retrieval_calibrated_rho = None
        self.latest_rho_variants = {}
        self.latest_analysis_rho_variants = {}
        self.latest_rho_fidelity_summary = {}
        self.update_display({})
        self.clear_plots()
        self.update_realtime_diagnosis()
        self.text_edit.append("Counter reset complete")

    def save_current_numpy_snapshot(self, output_root=None):
        snapshot_data = None
        if self.current_data is not None:
            snapshot_data = dict(self.current_data)
            snapshot_data["feedback_context"] = dict(getattr(self, "feedback_context", {}) or {})
            if self.latest_raw_rho is not None:
                snapshot_data["rho_raw"] = np.asarray(self.latest_raw_rho)
            if self.latest_calibrated_rho is not None:
                snapshot_data["rho_calibrated"] = np.asarray(self.latest_calibrated_rho)
            if self.latest_rho is not None:
                snapshot_data["rho_posterior_mean"] = np.asarray(self.latest_rho)
            snapshot_data["posterior_uncertainty"] = dict(
                getattr(self, "latest_rho_variant_diagnostics", {}) or {}
            )
        path = save_numpy_snapshot(snapshot_data, "3qubit", output_root=output_root)
        if path is None:
            self.text_edit.append("No current data to save before clear.")
        else:
            self.text_edit.append(f"Saved current data snapshot: {path}")
        return path

    def save_feedback_sample(self):
        result = save_feedback_sample_from_window(self, "3qubit")
        self.text_edit.append(result.message)
        return result.path if result.success else None

    def on_data_ready(self, data):
        self.current_data = data
        self._update_pointwise_diagnostics(data)
        self.frame_count += 1
        self.last_frame_timestamp = time.time()
        self.update_display(data)

    def update_display(self, data):
        if not data:
            self.lbl_multi_click.setText("Clicks: 2=0 3=0 4=0 total=0")
            self.lbl_multi2_pairs.setText("2-click pairs: 12=0 13=0 14=0 23=0 24=0 34=0")
            self.lbl_signal_interval.setText("Signal Interval: 0")
            self.lbl_idler_interval.setText("Idler Interval: 0")
            self.lbl_coin_interval.setText("Coin Interval: 0")
            self.lbl_write_count.setText("Write Count: 0")
            self.lbl_read_count.setText("Read Count: 0")
            self.lbl_signal_sum.setText("Signal Sum: 0")
            self.lbl_idler_sum.setText("Idler Sum: 0")
            self.lbl_coin_sum.setText("Coin Sum: 0")
            self._update_summary_table(
                multi_click=0,
                multi2_count=0,
                multi3_count=0,
                multi4_count=0,
                pair_text="12=0 13=0 14=0 23=0 24=0 34=0",
                signal_interval=0,
                idler_interval=0,
                coin_interval=0,
                write_count=0,
                read_count=0,
                signal_sum=0,
                idler_sum=0,
                coin_sum=0,
            )
            self._reset_rate_history()
            return

        signal_sum = int(np.sum(data.get('signal_bins', np.zeros(self.driver.num_cells))))
        idler_sum = int(np.sum(data.get('idler_bins', np.zeros(self.driver.num_cells))))
        coin_sum = int(np.sum(data.get('coin_bins', np.zeros(self.driver.num_cells))))

        pairs = data.get('multi2_pairs', {})
        pair_text = " ".join(f"{name}={int(pairs.get(name, 0))}" for name in ("12", "13", "14", "23", "24", "34"))
        multi2_count = int(data.get('multi2_count', 0) or 0)
        multi3_count = int(data.get('multi3_count', 0) or 0)
        multi4_count = int(data.get('multi4_count', 0) or 0)
        multi_click = int(data.get('multi_click', multi2_count + multi3_count + multi4_count) or 0)
        self.lbl_multi_click.setText(
            f"Clicks: 2={multi2_count} 3={multi3_count} 4={multi4_count} total={multi_click}"
        )
        self.lbl_multi2_pairs.setText(f"2-click pairs: {pair_text}")
        self.lbl_signal_interval.setText(f"Signal Interval: {data.get('signal_interval', 0)}")
        self.lbl_idler_interval.setText(f"Idler Interval: {data.get('idler_interval', 0)}")
        self.lbl_coin_interval.setText(f"Coin Interval: {data.get('coin_interval', 0)}")
        self.lbl_write_count.setText(f"Write Count: {data.get('write_count', 0)}")
        self.lbl_read_count.setText(f"Read Count: {data.get('read_count', 0)}")
        self.lbl_signal_sum.setText(f"Signal Sum: {signal_sum}")
        self.lbl_idler_sum.setText(f"Idler Sum: {idler_sum}")
        self.lbl_coin_sum.setText(f"Coin Sum: {coin_sum}")
        self._update_summary_table(
            multi_click=multi_click,
            multi2_count=multi2_count,
            multi3_count=multi3_count,
            multi4_count=multi4_count,
            pair_text=pair_text,
            signal_interval=data.get('signal_interval', 0),
            idler_interval=data.get('idler_interval', 0),
            coin_interval=data.get('coin_interval', 0),
            write_count=data.get('write_count', 0),
            read_count=data.get('read_count', 0),
            signal_sum=signal_sum,
            idler_sum=idler_sum,
            coin_sum=coin_sum,
        )
        self._update_rate_history(data, signal_sum, idler_sum, coin_sum)

        # Update histogram plot
        self.update_histogram(data)

        # Update matrix displays
        self.update_matrices(data)

        # Update density matrix reconstruction
        self.update_density_matrix(data)

    def _update_summary_table(
        self,
        *,
        multi_click,
        multi2_count,
        multi3_count,
        multi4_count,
        pair_text,
        signal_interval,
        idler_interval,
        coin_interval,
        write_count,
        read_count,
        signal_sum,
        idler_sum,
        coin_sum,
    ):
        self.summary_table.set_value("2-click", multi2_count)
        self.summary_table.set_value("3-click", multi3_count)
        self.summary_table.set_value("4-click", multi4_count)
        self.summary_table.set_value("Multi-click Total", multi_click)
        self.summary_table.set_value("2-click pairs", pair_text)
        self.summary_table.set_value("Signal Interval", signal_interval)
        self.summary_table.set_value("Idler Interval", idler_interval)
        self.summary_table.set_value("Coin Interval", coin_interval)
        self.summary_table.set_value("Write Count", write_count)
        self.summary_table.set_value("Read Count", read_count)
        self.summary_table.set_value("Signal Sum", signal_sum)
        self.summary_table.set_value("Idler Sum", idler_sum)
        self.summary_table.set_value("Coin Sum", coin_sum)

    def _reset_rate_history(self):
        self.rate_history = []
        self._last_rate_sample = None
        self._draw_rate_plot()

    def _update_rate_history(self, data, signal_sum, idler_sum, coin_sum):
        try:
            timestamp = float(data.get('timestamp', time.time()) or time.time())
        except Exception:
            timestamp = time.time()
        sample = {
            "timestamp": timestamp,
            "elapsed": 0.0,
            "signal": float(signal_sum),
            "idler": float(idler_sum),
            "coin": float(coin_sum),
        }
        previous = self._last_rate_sample
        if previous is None:
            self._last_rate_sample = sample
            self._draw_rate_plot()
            return

        dt = timestamp - float(previous["timestamp"])
        if dt <= 0.0:
            self._last_rate_sample = sample
            return

        d_signal = float(signal_sum) - float(previous["signal"])
        d_idler = float(idler_sum) - float(previous["idler"])
        d_coin = float(coin_sum) - float(previous["coin"])
        if d_signal < 0.0 or d_idler < 0.0 or d_coin < 0.0:
            self.rate_history = []
            self._last_rate_sample = sample
            self._draw_rate_plot()
            return

        signal_rate = d_signal / dt
        idler_rate = d_idler / dt
        coin_rate = d_coin / dt
        ratio = coin_rate / signal_rate if signal_rate > 0.0 else 0.0
        elapsed = float(previous.get("elapsed", 0.0)) + dt
        sample["elapsed"] = elapsed
        self.rate_history.append({
            "elapsed": elapsed,
            "signal_rate": signal_rate,
            "idler_rate": idler_rate,
            "coin_rate": coin_rate,
            "coin_signal_ratio": ratio,
            "coin_idler_ratio": coin_rate / idler_rate if idler_rate > 0.0 else 0.0,
        })
        if len(self.rate_history) > self.max_rate_history_points:
            self.rate_history = self.rate_history[-self.max_rate_history_points:]
        self._last_rate_sample = sample
        self._draw_rate_plot()

    def _draw_rate_plot(self):
        if not hasattr(self, 'rate_canvas'):
            return
        self.rate_fig.clear()
        ax = self.rate_fig.add_subplot(111)
        if not self.rate_history:
            ax.text(0.5, 0.5, "waiting for count-rate samples", ha="center", va="center")
            ax.set_axis_off()
            self.rate_canvas.draw()
            return

        x = np.asarray([point["elapsed"] for point in self.rate_history], dtype=float)
        signal_rate = np.asarray([point["signal_rate"] for point in self.rate_history], dtype=float)
        idler_rate = np.asarray([point["idler_rate"] for point in self.rate_history], dtype=float)
        coin_rate = np.asarray([point["coin_rate"] for point in self.rate_history], dtype=float)
        coin_signal_ratio = np.asarray([point["coin_signal_ratio"] for point in self.rate_history], dtype=float)
        coin_idler_ratio = np.asarray([point["coin_idler_ratio"] for point in self.rate_history], dtype=float)

        ax.plot(x, signal_rate, color="#c62828", linewidth=1.2, label="Signal/s")
        ax.plot(x, idler_rate, color="#1565c0", linewidth=1.2, label="Idler/s")
        ax.plot(x, coin_rate, color="#6a1b9a", linewidth=1.2, label="Coin/s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Counts/s")
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(x, coin_signal_ratio, color="#2e7d32", linewidth=1.2, label="Coin/Signal")
        ax2.plot(x, coin_idler_ratio, color="#ef6c00", linewidth=1.2, label="Coin/Idler")
        ax2.set_ylabel("Per-second ratios")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=7)
        self.rate_fig.tight_layout(pad=0.6)
        self.rate_canvas.draw()

    def update_histogram(self, data):
        if not data:
            self.signal_hist_ax.clear()
            self.idler_hist_ax.clear()
            self.signal_hist_canvas.draw()
            self.idler_hist_canvas.draw()
            return

        signal_hist = data.get('display_signal_hist', data.get('signal_hist', np.zeros(10)))
        idler_hist = data.get('display_idler_hist', data.get('idler_hist', np.zeros(10)))
        signal_hist = np.asarray(signal_hist, dtype=np.int64)
        idler_hist = np.asarray(idler_hist, dtype=np.int64)
        x_ticks = np.arange(10)
        tick_labels = [f"{i * 20}-{(i + 1) * 20}" for i in range(10)]

        self.signal_hist_ax.clear()
        self.signal_hist_ax.bar(x_ticks, signal_hist, color='red', alpha=0.7)
        self.signal_hist_ax.set_xlabel('Time in selected cell (ns)', fontsize=8)
        self.signal_hist_ax.set_ylabel('Count', fontsize=8)
        self.signal_hist_ax.set_xticks(x_ticks)
        self.signal_hist_ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=7)
        self.signal_hist_ax.tick_params(labelsize=7)
        self.signal_hist_canvas.figure.tight_layout(pad=0.3)
        self.signal_hist_canvas.draw()

        self.idler_hist_ax.clear()
        self.idler_hist_ax.bar(x_ticks, idler_hist, color='blue', alpha=0.7)
        self.idler_hist_ax.set_xlabel('Time in selected cell (ns)', fontsize=8)
        self.idler_hist_ax.set_ylabel('Count', fontsize=8)
        self.idler_hist_ax.set_xticks(x_ticks)
        self.idler_hist_ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=7)
        self.idler_hist_ax.tick_params(labelsize=7)
        self.idler_hist_canvas.figure.tight_layout(pad=0.3)
        self.idler_hist_canvas.draw()

    def _update_pointwise_diagnostics(self, data):
        if not hasattr(self, 'pointwise_canvas'):
            return
        shadow = (data or {}).get('noise_aware_reconstruction') or (data or {}).get('shadow_diagnostics')
        if not shadow:
            self.lbl_pointwise_status.setText("Shadow diagnostics: no operation-point reconstruction attached to this frame")
            return
        values = pointwise_values_from_shadow(shadow, (data or {}).get('calibration_numeric'))
        draw_operation_point_diagnostics(self.pointwise_fig, "3qubit", values)
        self.pointwise_canvas.draw_idle()
        self.lbl_pointwise_status.setText(
            "Shadow diagnostics — conditional F={:.3f}; leakage-adjusted F={:.3f}; delta application disabled".format(
                float(shadow.get('conditional_fidelity_median', 0.0)),
                float(shadow.get('leakage_adjusted_fidelity_median', 0.0)),
            )
        )

    def update_matrices(self, data):
        if not data:
            self.signal_8x8_table.update_data(np.zeros(64))
            self.signal_1x11_table.update_data(np.zeros(11))
            self.idler_8x8_table.update_data(np.zeros(64))
            self.idler_1x11_table.update_data(np.zeros(11))
            self.coin_8x8_table.update_data(np.zeros(64))
            self.coin_1x11_table.update_data(np.zeros(11))
            self.calc_8x8_table.update_data(np.zeros(64))
            self.calc_1x11_table.update_data(np.zeros(11))
            return

        signal_bins = data.get('signal_bins', np.zeros(self.driver.num_cells))
        idler_bins = data.get('idler_bins', np.zeros(self.driver.num_cells))
        coin_bins = data.get('coin_bins', np.zeros(self.driver.num_cells))
        segment_values = data.get('segment_values', np.zeros(self.driver.num_cells))

        # Update signal matrices (8x8 + 1x11)
        self.signal_8x8_table.update_data(signal_bins[:64])
        self.signal_1x11_table.update_data(signal_bins[64:75])

        # Update idler matrices (8x8 + 1x11)
        self.idler_8x8_table.update_data(idler_bins[:64])
        self.idler_1x11_table.update_data(idler_bins[64:75])

        # Update coin matrices (8x8 + 1x11)
        self.coin_8x8_table.update_data(coin_bins[:64])
        self.coin_1x11_table.update_data(coin_bins[64:75])

        # Display calculated values in milli-units for readability only.
        calc_display = np.asarray(segment_values, dtype=float) * 1000.0
        self.calc_8x8_table.update_data(calc_display[:64])
        self.calc_1x11_table.update_data(calc_display[64:75])

    def update_density_matrix(self, data):
        """Reconstruct and display density matrix from segment values."""
        recon_available = (
            CALIBRATED_DENSITY_RECON_AVAILABLE and calibrated_density_reconstruction is not None
        ) or (DENSITY_RECON_AVAILABLE and density_reconstruction is not None)
        if not data or not recon_available:
            self.re_rho_ax.clear()
            self.im_rho_ax.clear()
            if hasattr(self, 'raw_re_rho_ax'):
                self.raw_re_rho_ax.clear()
                self.raw_im_rho_ax.clear()
            self.re_rho_canvas.draw()
            self.im_rho_canvas.draw()
            if hasattr(self, 'raw_re_rho_canvas'):
                self.raw_re_rho_canvas.draw()
                self.raw_im_rho_canvas.draw()
            self.latest_rho = None
            self.latest_rho_variants = {}
            self.latest_analysis_rho_variants = {}
            self.latest_rho_fidelity_summary = {}
            self._update_w_subspace_panel({})
            self._clear_state_family_table("waiting for rho")
            self._update_physics_dashboard({}, {"message": "waiting for rho"})
            return

        try:
            # Get segment values and set negative values to 0
            segment_values = data.get('segment_values', np.zeros(self.driver.num_cells))
            counts = np.maximum(segment_values, 0)  # Clip negative values to 0

            # Get path to basis config file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            basis_config_path = os.path.join(os.path.dirname(script_dir), 'qubit_recon_files', 'matrix', 'basis_config_3qubit.json')

            if not os.path.exists(basis_config_path):
                print(f"Basis config not found: {basis_config_path}")
                return

            # Reconstruct density matrix.  The calibrated module returns both
            # raw and calibrated states; the raw module is kept as a fallback.
            if CALIBRATED_DENSITY_RECON_AVAILABLE and calibrated_density_reconstruction is not None:
                result = calibrated_density_reconstruction.reconstruct_density_matrix(
                    '3qubit',
                    counts,
                    basis_config_path,
                    efficiency_map_path=calibrated_density_reconstruction.default_efficiency_map_path('3qubit'),
                )
                raw_result = result.get('raw', {})
                calibrated_result = result.get('calibrated', result)
                retrieval_result = None
                retrieval_map_path = calibrated_density_reconstruction.legacy_efficiency_map_path('3qubit')
                if os.path.exists(retrieval_map_path):
                    retrieval_wrapper = calibrated_density_reconstruction.reconstruct_density_matrix(
                        '3qubit',
                        counts,
                        basis_config_path,
                        efficiency_map_path=retrieval_map_path,
                    )
                    retrieval_result = retrieval_wrapper.get('calibrated', retrieval_wrapper)
            elif density_reconstruction is not None:
                result = density_reconstruction.reconstruct_density_matrix('3qubit', counts, basis_config_path)
                raw_result = result
                calibrated_result = result
                retrieval_result = None
            else:
                self.text_edit.append("Error: Density reconstruction module not loaded")
                if 'self.re_rho_ax' in locals():
                    self.re_rho_ax.clear()
                    self.im_rho_ax.clear()
                    if hasattr(self, 'raw_re_rho_ax'):
                        self.raw_re_rho_ax.clear()
                        self.raw_im_rho_ax.clear()
                    self.re_rho_canvas.draw()
                    self.im_rho_canvas.draw()
                    if hasattr(self, 'raw_re_rho_canvas'):
                        self.raw_re_rho_canvas.draw()
                        self.raw_im_rho_canvas.draw()
                return

            # Extract real and imaginary parts
            re_rho = calibrated_result['re_rho']
            im_rho = calibrated_result['im_rho']
            raw_re_rho = raw_result.get('re_rho', re_rho)
            raw_im_rho = raw_result.get('im_rho', im_rho)
            self.latest_idler_calibrated_rho = calibrated_result.get('rho_full', re_rho + 1j * im_rho)
            self.latest_calibrated_rho = self.latest_idler_calibrated_rho
            self.latest_raw_rho = raw_result.get('rho_full', raw_re_rho + 1j * raw_im_rho)
            if retrieval_result is not None:
                retrieval_re_rho = retrieval_result.get('re_rho')
                retrieval_im_rho = retrieval_result.get('im_rho')
                if retrieval_re_rho is not None and retrieval_im_rho is not None:
                    self.latest_retrieval_calibrated_rho = retrieval_result.get(
                        'rho_full',
                        retrieval_re_rho + 1j * retrieval_im_rho,
                    )
                else:
                    self.latest_retrieval_calibrated_rho = retrieval_result.get('rho_full')
            else:
                self.latest_retrieval_calibrated_rho = None
            self.latest_rho_variants = {}
            if self.latest_raw_rho is not None:
                self.latest_rho_variants['raw'] = self.latest_raw_rho
            if self.latest_idler_calibrated_rho is not None:
                self.latest_rho_variants['idler'] = self.latest_idler_calibrated_rho
            if self.latest_retrieval_calibrated_rho is not None:
                self.latest_rho_variants['retrieval'] = self.latest_retrieval_calibrated_rho
            self.latest_rho = self.latest_idler_calibrated_rho
            if self.latest_rho is None:
                self.latest_rho = self.latest_retrieval_calibrated_rho
            if self.latest_rho is None:
                self.latest_rho = self.latest_raw_rho

            # Update 3D plots
            self.plot_3d_density_matrix(self.re_rho_ax, re_rho, "Idler Cal Re(rho)")
            self.plot_3d_density_matrix(self.im_rho_ax, im_rho, "Idler Cal Im(rho)")
            if hasattr(self, 'raw_re_rho_ax'):
                self.plot_3d_density_matrix(self.raw_re_rho_ax, raw_re_rho, "Raw Re(rho)")
                self.plot_3d_density_matrix(self.raw_im_rho_ax, raw_im_rho, "Raw Im(rho)")

            self.re_rho_canvas.draw()
            self.im_rho_canvas.draw()
            if hasattr(self, 'raw_re_rho_canvas'):
                self.raw_re_rho_canvas.draw()
                self.raw_im_rho_canvas.draw()

        except Exception as e:
            print(f"Error in density matrix reconstruction: {e}")

    def _build_rho_fidelity_summary(self):
        variants = dict(getattr(self, 'latest_rho_variants', {}) or {})
        if not variants and self.latest_rho is not None:
            variants['idler'] = self.latest_rho
        diagnostic_variants = build_diagnostic_rho_variants(
            '3qubit',
            self.current_data,
            calibrated_module=calibrated_density_reconstruction
            if CALIBRATED_DENSITY_RECON_AVAILABLE
            else None,
        )
        variants.update(diagnostic_variants.get('rho_variants', {}))
        self.latest_analysis_rho_variants = dict(variants)
        self.latest_rho_variant_diagnostics = diagnostic_variants.get('diagnostics', {})
        summary = summarize_rho_variants(
            variants,
            num_qubits=3,
            data=self.current_data,
            ml_interface=self.ml_interface,
            target_state_params=(getattr(self, 'feedback_context', {}) or {}).get('target_state_params'),
        )
        self.latest_rho_fidelity_summary = summary
        return summary

    def update_realtime_diagnosis(self):
        """Refresh lightweight excitation/error diagnosis from the latest rho."""
        if not hasattr(self, 'lbl_diag_model'):
            return
        if self.latest_rho is None:
            backend = (
                getattr(self.ml_interface, 'diagnosis_model_backend', None)
                if self.ml_interface is not None
                else None
            ) or "not loaded"
            self.lbl_diag_model.setText(f"Model: {backend}")
            self.lbl_diag_modes.setText("Excitation modes: waiting for rho")
            self.lbl_diag_families.setText("State families: waiting for rho")
            self.lbl_diag_errors.setText("Error sources: waiting for rho")
            self.lbl_diag_metrics.setText("Fidelity/Purity/OOD: --")
            self._update_diagnosis_table(
                model=backend,
                excitation_modes="waiting for rho",
                state_families="waiting for rho",
                error_sources="waiting for rho",
                fidelity="--",
                top_fits="--",
                purity="--",
                ood="--",
                ideal_f="--",
                model_audit="--",
            )
            self._update_estimator_table({})
            self._set_mode_probability_bars({})
            self._update_w_subspace_panel({})
            self._clear_state_family_table("waiting for rho")
            self._update_physics_dashboard({}, {"backend": backend, "message": "waiting for rho"})
            return

        summary = self._build_rho_fidelity_summary()
        _primary_key, primary_metrics = primary_variant(summary)
        diagnosis = (primary_metrics or {}).get('diagnosis', {}) or {}
        diagnosis_available = bool(diagnosis.get('available'))
        if self.ml_interface is None:
            backend = "ML interface unavailable"
        elif diagnosis_available:
            backend = diagnosis.get('backend', 'unknown')
        else:
            backend = diagnosis.get('message', 'direct rho metrics')

        mode_probabilities = diagnosis.get('mode_probabilities', {}) if diagnosis_available else {}
        mode_text = self._format_probabilities(mode_probabilities) if diagnosis_available else "--"
        family_text = (
            self._format_probabilities(diagnosis.get('state_family_group_probabilities', {}))
            if diagnosis_available
            else "--"
        )
        error_text = (
            self._format_probabilities(diagnosis.get('error_source_probabilities', {}))
            if diagnosis_available
            else "--"
        )
        fidelity_text = format_variant_metric(summary, 'fidelity')
        top_fit_text = self._format_top_fits(summary)
        purity_text = format_variant_metric(summary, 'purity')
        ood_text = format_variant_metric(summary, 'ood', percent=True)
        ideal_text = format_variant_metric(summary, 'ideal_fidelity_ceiling')
        audit_text = format_model_audit(summary)
        model_text = self._format_model_paths(backend, summary.get('model_audit', {}))
        multi_leakage = float(summary.get('multi_click_leakage', 0.0) or 0.0)
        ideal_row_text = f"{ideal_text} (ideal W 1.000, multi {multi_leakage * 100.0:.2f}%)"

        self.lbl_diag_model.setText(f"Model: {model_text}")
        self.lbl_diag_modes.setText("Excitation modes: " + mode_text)
        self.lbl_diag_families.setText("State families: groups " + family_text)
        self.lbl_diag_errors.setText("Error sources: " + error_text)
        target_desc = summary.get('target_description', 'target: --')
        self.lbl_diag_metrics.setText(
            f"Target F: {fidelity_text}; top fits: {top_fit_text}; {target_desc}; "
            f"IdealF ceiling: {ideal_text}; OOD: {ood_text}; audit: {audit_text}"
        )
        self._update_diagnosis_table(
            model=model_text,
            excitation_modes=mode_text,
            state_families=family_text,
            error_sources=error_text,
            fidelity=f"Target {fidelity_text}",
            top_fits=top_fit_text,
            purity=purity_text,
            ood=ood_text,
            ideal_f=ideal_row_text,
            model_audit=audit_text,
        )
        self._update_estimator_table(summary)
        self._set_mode_probability_bars(mode_probabilities)
        self._update_w_subspace_panel(summary)
        if diagnosis_available:
            self._update_state_family_table(diagnosis)
        else:
            self._clear_state_family_table("state-family model unavailable")
        self._update_physics_dashboard(summary, diagnosis)

    def _update_diagnosis_table(
        self,
        *,
        model,
        excitation_modes,
        state_families,
        error_sources,
        fidelity,
        top_fits,
        purity,
        ood,
        ideal_f,
        model_audit,
    ):
        if not hasattr(self, 'diagnosis_table'):
            return
        self.diagnosis_table.set_value("Model", model)
        self.diagnosis_table.set_value("Excitation Modes", excitation_modes)
        self.diagnosis_table.set_value("State Families", state_families)
        self.diagnosis_table.set_value("Error Sources", error_sources)
        self.diagnosis_table.set_value("Fidelity", fidelity)
        self.diagnosis_table.set_value("Top F Fits", top_fits)
        self.diagnosis_table.set_value("Purity", purity)
        self.diagnosis_table.set_value("OOD", ood)
        self.diagnosis_table.set_value("Ideal F", ideal_f)
        self.diagnosis_table.set_value("Model Audit", model_audit)

    def _format_top_fits(self, summary):
        fits = top_fidelity_fits(summary, limit=4)
        if not fits:
            return "--"
        return " | ".join(
            f"{item['variant_label']} {item['metric_label']} {item['fidelity']:.3f}"
            for item in fits
        )

    def _update_estimator_table(self, summary):
        if not hasattr(self, 'estimator_table'):
            return
        variants = (summary or {}).get('variants', {}) or {}
        for row, (metric_key, metric_label) in enumerate(ESTIMATOR_METRICS):
            label_item = QTableWidgetItem(metric_label)
            label_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.estimator_table.setItem(row, 0, label_item)
            for column, (variant_key, _variant_label) in enumerate(VARIANT_LABELS, start=1):
                values = variants.get(variant_key, {}) or {}
                value = values.get(metric_key)
                if metric_key.lower().startswith('ood'):
                    text = "--" if value is None else f"{float(value) * 100.0:.1f}%"
                elif value is None:
                    text = "--"
                else:
                    text = f"{float(value):.3f}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.estimator_table.setItem(row, column, item)

    def _format_model_paths(self, backend, audit):
        audit = audit or {}
        diag_backend = audit.get('diagnosis_backend') or backend or "--"
        diag_path = audit.get('diagnosis_model_path') or "--"
        family_backend = audit.get('state_family_backend') or "state_family"
        family_path = audit.get('state_family_model_path') or "--"
        return f"{diag_backend}: {diag_path}; {family_backend}: {family_path}"

    def _set_mode_probability_bars(self, values):
        if not hasattr(self, 'mode_probability_bars'):
            return
        values = values or {}
        for label, bar in self.mode_probability_bars.items():
            probability = float(values.get(label, 0.0) or 0.0)
            probability = max(0.0, min(1.0, probability))
            percent = probability * 100.0
            bar.setValue(int(round(percent * 10.0)))
            bar.setFormat(f"{percent:.1f}%")
            text_label = self.mode_probability_labels.get(label)
            if text_label is not None:
                text_label.setText(label.replace("_", " ").title())

    def _update_w_subspace_panel(self, summary):
        if not hasattr(self, 'w_subspace_canvas'):
            return
        draw_w_subspace_summary(
            self.w_subspace_fig,
            getattr(self, 'latest_rho', None),
            3,
            summary or {},
            message="waiting for rho",
        )
        self.w_subspace_canvas.draw()

    def _update_physics_dashboard(self, summary, diagnosis):
        if not hasattr(self, 'physics_dashboard_canvas'):
            return
        metrics = compute_physics_dashboard_metrics(
            getattr(self, 'latest_rho', None),
            3,
            summary or {},
            diagnosis or {},
            rho_variants=(
                getattr(self, 'latest_analysis_rho_variants', None)
                or getattr(self, 'latest_rho_variants', None)
                or {}
            ),
        )
        self.latest_physics_dashboard_metrics = metrics
        self.physics_metric_grid.update_metrics(metrics)
        self.inference_summary_table.update_metrics(metrics)
        draw_physics_dashboard(self.physics_dashboard_fig, metrics, 3)
        self.physics_dashboard_canvas.draw()

    def _clear_state_family_table(self, message):
        if not hasattr(self, 'state_family_canvas'):
            return
        draw_state_family_chart(self.state_family_fig, {}, {}, message=str(message))
        self.state_family_canvas.draw()

    def _update_state_family_table(self, diagnosis):
        if not hasattr(self, 'state_family_canvas'):
            return
        probabilities = diagnosis.get('state_family_probabilities', {}) or {}
        groups = diagnosis.get('state_family_groups', {}) or {}
        if not probabilities:
            self._clear_state_family_table("no fine-state probabilities")
            return
        draw_state_family_chart(self.state_family_fig, probabilities, groups, top_n=10)
        self.state_family_canvas.draw()

    def _format_probabilities(self, values):
        if not values:
            return "--"
        top = sorted(values.items(), key=lambda item: item[1], reverse=True)[:4]
        return ", ".join(f"{label}: {prob * 100:.1f}%" for label, prob in top)

    def plot_3d_density_matrix(self, ax, data, title):
        """Plot 3D bar chart for density matrix."""
        ax.clear()

        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot 3D bar chart
        dx = dy = 0.8
        dz = data.flatten()
        xpos = X.flatten()
        ypos = Y.flatten()
        zpos = np.zeros_like(dz)

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

        # Add labels for 3-qubit basis states: |q2 q1 q0⟩
        subspace_labels = [
            '|000⟩', '|001⟩', '|010⟩', '|011⟩',
            '|100⟩', '|101⟩', '|110⟩', '|111⟩'
        ]

        ax.set_title(title)
        ax.set_xlabel("Bra ⟨i|")
        ax.set_ylabel("Ket |j⟩")
        ax.set_zlabel("Value")

        if data.shape[0] == 8 and data.shape[1] == 8:
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_xticklabels(subspace_labels, rotation=45, ha='right')
            ax.set_yticklabels(subspace_labels, rotation=-45, ha='right')

    def request_ml_correction(self):
        """Request ML-based correction suggestions for current density matrix"""
        self.text_edit.append("\n--- ML Correction Request ---")

        if self.ml_interface is None:
            self.text_edit.append("Error: ML Correction Interface not available")
            self.text_edit.append("Please ensure ml_recon directory exists and dependencies are installed")
            return

        if self.current_data is None:
            self.text_edit.append("Error: No density matrix data available")
            self.text_edit.append("Please collect data first")
            return

        try:
            # Reconstruct density matrix from current data
            segment_values = self.current_data.get('segment_values', np.zeros(self.driver.num_cells))
            counts = np.maximum(segment_values, 0)

            # Get basis config path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            basis_config_path = os.path.join(os.path.dirname(script_dir), 'qubit_recon_files', 'matrix', 'basis_config_3qubit.json')

            if not os.path.exists(basis_config_path):
                self.text_edit.append(f"Error: Basis config not found: {basis_config_path}")
                return

            # Reconstruct density matrix
            if CALIBRATED_DENSITY_RECON_AVAILABLE and calibrated_density_reconstruction is not None:
                result = calibrated_density_reconstruction.reconstruct_density_matrix(
                    '3qubit',
                    counts,
                    basis_config_path,
                    efficiency_map_path=calibrated_density_reconstruction.default_efficiency_map_path('3qubit'),
                )
            elif density_reconstruction is not None:
                result = density_reconstruction.reconstruct_density_matrix('3qubit', counts, basis_config_path)
            else:
                self.text_edit.append("Error: Density reconstruction module not loaded")
                if 'self.re_rho_ax' in locals():
                    self.re_rho_ax.clear()
                    self.im_rho_ax.clear()
                    if hasattr(self, 'raw_re_rho_ax'):
                        self.raw_re_rho_ax.clear()
                        self.raw_im_rho_ax.clear()
                    self.re_rho_canvas.draw()
                    self.im_rho_canvas.draw()
                    if hasattr(self, 'raw_re_rho_canvas'):
                        self.raw_re_rho_canvas.draw()
                        self.raw_im_rho_canvas.draw()
                return
            rho_current = result['rho_full']

            # Analyze with ML interface
            analysis = self.ml_interface.analyze_density_matrix(rho_current)

            if not getattr(self, 'latest_rho_variants', None):
                self.latest_rho_variants = {'idler': rho_current}
            summary = self._build_rho_fidelity_summary()
            self.text_edit.append(f"Target: {summary.get('target_description', '--')}")
            self.text_edit.append(f"Target F amp variants: {format_variant_metric(summary, 'F_target_amp')}")
            self.text_edit.append(f"Target F sqrt-amp variants: {format_variant_metric(summary, 'F_target_sqrt_amp')}")
            self.text_edit.append(f"Target F amp^2 variants: {format_variant_metric(summary, 'F_target_amp_squared')}")
            self.text_edit.append(f"Target F sin(amp)/amp variants: {format_variant_metric(summary, 'F_target_sinc_amp')}")
            self.text_edit.append(f"Depol-corrected F variants: {format_variant_metric(summary, 'F_depolarized_corrected')}")
            self.text_edit.append(f"Phase-opt target-amp witness: {format_variant_metric(summary, 'F_phase_opt_target_amp')}")
            self.text_edit.append(f"Phase-opt balanced witness: {format_variant_metric(summary, 'F_phase_opt_balanced')}")
            self.text_edit.append(f"Best W-subspace variants: {format_variant_metric(summary, 'F_best_W_subspace')}")
            self.text_edit.append(f"Purity variants: {format_variant_metric(summary, 'purity')}")
            self.text_edit.append(
                "Ideal F ceiling variants: "
                f"{format_variant_metric(summary, 'ideal_fidelity_ceiling')}"
            )
            self.text_edit.append(f"Model audit: {format_model_audit(summary)}")
            audit = summary.get('model_audit', {}) or {}
            self.text_edit.append(
                f"Diagnosis model: {audit.get('diagnosis_backend', '--')} "
                f"{audit.get('diagnosis_model_path', '--')}"
            )
            self.text_edit.append(
                f"State-family model: {audit.get('state_family_backend', '--')} "
                f"{audit.get('state_family_model_path', '--')}"
            )
            self.text_edit.append(f"Trace (active rho): {analysis['trace']:.4f}")

            self.text_edit.append("\nTop eigenstates:")
            for idx, val in analysis['top_eigenstates']:
                binary = format(idx, f'0{self.ml_interface.num_qubits}b')
                self.text_edit.append(f"  |{binary}>: {val:.4f}")

            self.text_edit.append("\nCorrection suggestions:")
            for suggestion in analysis['suggestions']:
                self.text_edit.append(f"  - {suggestion}")

            diagnosis = self.ml_interface.diagnose_density_matrix(
                rho_current,
                summary.get('rho_target'),
            )
            if diagnosis.get('available'):
                self.text_edit.append("\nQuantum state diagnosis:")
                self.text_edit.append("  Excitation mode probabilities:")
                for label, prob in diagnosis.get('mode_probabilities', {}).items():
                    self.text_edit.append(f"    {label}: {prob * 100:.1f}%")
                self.text_edit.append("  Error source probabilities:")
                for label, prob in diagnosis.get('error_source_probabilities', {}).items():
                    self.text_edit.append(f"    {label}: {prob * 100:.1f}%")
                self.text_edit.append(f"  Unknown/OOD probability: {diagnosis.get('unknown_probability', 0.0) * 100:.1f}%")
            else:
                self.text_edit.append(f"\nQuantum state diagnosis: {diagnosis.get('message', 'not available')}")

            # Get ML-based parameter suggestions if model is loaded
            if self.ml_interface.model is not None:
                self.text_edit.append("\nML-based parameter suggestions:")
                # Simulate current parameters (would need to get from actual system)
                current_params = {
                    'freq': 90.0,  # Example value
                    'phase': 0.0,
                    'amp': 1.0
                }
                ml_suggestions = self.ml_interface.get_correction_suggestions(current_params)

                if 'error' not in ml_suggestions:
                    self.text_edit.append(f"  Phase correction: {ml_suggestions['phase_correction']:.3f}")
                    self.text_edit.append(f"  Amplitude correction: {ml_suggestions['amp_correction']:.3f}")

                    self.text_edit.append("\nDetailed suggestions:")
                    for suggestion in ml_suggestions['suggestions']:
                        self.text_edit.append(f"  - {suggestion}")
                else:
                    self.text_edit.append(f"  {ml_suggestions['error']}")
            else:
                self.text_edit.append("\nNote: ML model not loaded. Only basic analysis available.")
                self.text_edit.append("Run training in ml_recon directory to enable ML suggestions.")

        except Exception as e:
            self.text_edit.append(f"Error in ML correction: {e}")
            import traceback
            self.text_edit.append(traceback.format_exc())

    def clear_plots(self):
        # Clear histogram plots
        self.signal_hist_ax.clear()
        self.idler_hist_ax.clear()
        self.signal_hist_canvas.draw()
        self.idler_hist_canvas.draw()

        # Clear density matrix plots
        self.re_rho_ax.clear()
        self.im_rho_ax.clear()
        if hasattr(self, 'raw_re_rho_ax'):
            self.raw_re_rho_ax.clear()
            self.raw_im_rho_ax.clear()
        self.re_rho_canvas.draw()
        self.im_rho_canvas.draw()
        if hasattr(self, 'raw_re_rho_canvas'):
            self.raw_re_rho_canvas.draw()
            self.raw_im_rho_canvas.draw()

        # Clear matrix tables
        self.signal_8x8_table.update_data(np.zeros(64))
        self.signal_1x11_table.update_data(np.zeros(11))
        self.idler_8x8_table.update_data(np.zeros(64))
        self.idler_1x11_table.update_data(np.zeros(11))
        self.coin_8x8_table.update_data(np.zeros(64))
        self.coin_1x11_table.update_data(np.zeros(11))
        self.calc_8x8_table.update_data(np.zeros(64))
        self.calc_1x11_table.update_data(np.zeros(11))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
