"""
Histogram Acquisition GUI with Serial Communication 
Date: 2025-10-24

Features:
- Read 1159-byte frames from COM port
- Decode histogram using 225-bin byte-aligned protocol
- Display heatmaps and tables with overflow protection
- Save data to timestamped CSV
- Virtual test mode for GUI debugging
- Clear function to reset baseline
- Additional plots for calculated values
- Time-series plots for monitoring values over time
- Real-time increment calculation between updates
"""

import sys, time, threading, datetime, csv
import numpy as np
import serial
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QGroupBox, QTableWidget, 
    QTableWidgetItem, QSplitter, QTabWidget, QSizePolicy, QSpinBox,
    QLineEdit
)
from PyQt5.QtGui import QColor
import serial.tools.list_ports

GRID_SIZE = 15
CELL_COUNT = GRID_SIZE * GRID_SIZE
SERIAL_FRAME_BYTES = 1159
CELL_COUNTER_BITS = 11
COIN_COUNTER_BITS = 7
HIST_COUNTER_BITS = 8
HIST_BINS = 10
READ_COUNT_BYTE = 0
WRITE_COUNT_BYTE = READ_COUNT_BYTE + 3
IDLER_ALL_BYTE = WRITE_COUNT_BYTE + 3
COIN_ALL_BYTE = IDLER_ALL_BYTE + 3
SIGNAL_ALL_BYTE = COIN_ALL_BYTE + 2
SIGNAL_CELL_BYTE = SIGNAL_ALL_BYTE + 3
IDLER_CELL_BYTE = SIGNAL_CELL_BYTE + CELL_COUNT * 2
COIN_CELL_BYTE = IDLER_CELL_BYTE + CELL_COUNT * 2
SIGNAL_HIST_BYTE = COIN_CELL_BYTE + CELL_COUNT
IDLER_HIST_BYTE = SIGNAL_HIST_BYTE + HIST_BINS

# For plotting
import matplotlib.pyplot as plt
import importlib
# FigureCanvas moved in newer matplotlib releases (3.8+). Import dynamically
# to avoid static import-time errors on systems with different backends.
class _FigureCanvasBase(QWidget):
    def __init__(self, fig=None, *args, **kwargs):
        super().__init__()
        self.fig = fig
    def setParent(self, parent):
        super().setParent(parent)
    def draw(self):
        pass
    def updateGeometry(self):
        super().updateGeometry()
    def clear(self):
        pass
    def setSizePolicy(self, *args, **kwargs):
        try:
            super().setSizePolicy(*args, **kwargs)
        except Exception:
            pass

# Default to base stub; may be overridden by backend-specific canvas below.
FigureCanvas = _FigureCanvasBase
try:
    m = importlib.import_module('matplotlib.backends.backend_qtagg')
    FigureCanvas = getattr(m, 'FigureCanvasQTAgg')
except Exception:
    try:
        m = importlib.import_module('matplotlib.backends.backend_qt5agg')
        FigureCanvas = getattr(m, 'FigureCanvasQTAgg')
    except Exception:
        # Fallback stub for environments without Qt backends (allows import).
        # keep default _FigureCanvasBase
        pass
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# ============================================================
# Serial Driver (1159-byte 225 protocol with overflow protection)
# ============================================================

class Driver(QObject):
    data_ready = pyqtSignal(dict)  # Emit full decoded data dict

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.alive = False
        self.lock = threading.Lock()
        
        # Overflow protection for counters
        self._last_read_times = None
        self._last_write_seq = None
        self.read_overflows = 0
        self.write_overflows = 0
        # tolerance for treating small decreases as noise instead of wrap
        # can be overridden by UI before start
        self.overflow_sensitivity = 10
        # additional diff threshold: compare scalar total vs sum(bins)
        # if the difference is below this threshold, suppress treating
        # small decreases as wrap (acts as an extra protection).
        # Default to same initial value as overflow_sensitivity; UI will
        # set this to the same value on Start so they remain consistent.
        self.overflow_diff_threshold = 10
        # Per-type thresholds (allow independent control for signal/idler/coin)
        self.overflow_sensitivity_signal = 10
        self.overflow_sensitivity_idler = 10
        self.overflow_sensitivity_coin = 10
        self.overflow_diff_threshold_signal = 10
        self.overflow_diff_threshold_idler = 10
        self.overflow_diff_threshold_coin = 10
        # Overflow handling for scalar totals (signal_all, idler_all, coin_all)
        self._last_signal_all = None
        self._last_idler_all = None
        self._last_coin_all = None
        self.signal_all_overflows = 0
        self.idler_all_overflows = 0
        self.coin_all_overflows = 0
        
        # Overflow protection for bins (to handle counter wrap-around)
        # Use None to indicate "not initialized" so the first received frame
        # will establish the baseline and avoid false overflow detection
        self.last_signal_bins = None  # will be set to list of raw values on first frame
        self.last_idler_bins = None
        self.last_coin_bins = None
        # Overflow counters are also initialized on first frame
        self.signal_overflows = None
        self.idler_overflows = None
        self.coin_overflows = None

    def open(self, port_name):
        try:
            self.serial_port = serial.Serial(port=port_name, baudrate=500000, timeout=0.1)
            self.alive = True
            threading.Thread(target=self.read_loop, daemon=True).start()
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
        """Send 'n' to stop output"""
        with self.lock:
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b"n")
                    self.serial_port.reset_input_buffer()
                except Exception as e:
                    print("set_idle error:", e)

    def set_output(self):
        """Send 'y' to start output"""
        with self.lock:
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b"y")
                    self.serial_port.reset_input_buffer()
                except Exception as e:
                    print("set_output error:", e)

        if not self.alive:
            self.alive = True
            threading.Thread(target=self.read_loop, daemon=True).start()

    def _read_exact(self, n, timeout_s=1.0):
        """Read exactly n bytes with timeout"""
        if not (self.serial_port and self.serial_port.is_open):
            return b''
        end = time.time() + timeout_s
        buf = bytearray()
        while len(buf) < n and time.time() < end:
            chunk = self.serial_port.read(n - len(buf))
            if chunk:
                buf.extend(chunk)
            else:
                time.sleep(0.001)
        return bytes(buf)

    def read_loop(self):
        """Continuously wait for frames and decode them using 1159-byte 225 protocol"""
        frame_bytes = SERIAL_FRAME_BYTES
        print("[INFO] Read loop started - 1159-byte 225 protocol")

        while self.alive:
            try:
                if not (self.serial_port and self.serial_port.is_open):
                    time.sleep(0.1)
                    continue

                # Wait for enough data
                if self.serial_port.in_waiting >= frame_bytes:
                    raw = self._read_exact(frame_bytes, timeout_s=0.5)
                else:
                    time.sleep(0.01)
                    continue

                if len(raw) < frame_bytes:
                    continue

                def get_u24(offset):
                    return int.from_bytes(raw[offset:offset + 3], byteorder='big', signed=False) & ((1 << 19) - 1)

                def get_u16(offset):
                    return int.from_bytes(raw[offset:offset + 2], byteorder='big', signed=False)

                # Extract scalar counters according to byte-aligned Verilog layout.
                read_times = get_u24(READ_COUNT_BYTE)
                write_seq  = get_u24(WRITE_COUNT_BYTE)
                idler_all  = get_u24(IDLER_ALL_BYTE)
                coin_all   = get_u16(COIN_ALL_BYTE) & ((1 << 15) - 1)
                signal_all = get_u24(SIGNAL_ALL_BYTE)

                # --- Overflow handling for scalar totals ---
                # Defer robust handling for `signal_all` until after per-bin
                # corrections are computed so we can compare against the
                # sum(signal_bins). Store raw value now and finalize later.
                raw_signal_all = int(signal_all)

                # Defer robust handling for idler_all and coin_all similar to signal_all.
                raw_idler_all = int(idler_all)
                raw_coin_all = int(coin_all)

                # Extract arrays (raw values)
                signal_bins_raw = []
                for i in range(CELL_COUNT):
                    signal_bins_raw.append(get_u16(SIGNAL_CELL_BYTE + i * 2) & ((1 << CELL_COUNTER_BITS) - 1))

                idler_bins_raw = []
                for i in range(CELL_COUNT):
                    idler_bins_raw.append(get_u16(IDLER_CELL_BYTE + i * 2) & ((1 << CELL_COUNTER_BITS) - 1))

                coin_bins_raw = []
                for i in range(CELL_COUNT):
                    coin_bins_raw.append(raw[COIN_CELL_BYTE + i] & ((1 << COIN_COUNTER_BITS) - 1))

                signal_hist_raw = [int(x) for x in raw[SIGNAL_HIST_BYTE:SIGNAL_HIST_BYTE + HIST_BINS]]
                idler_hist_raw = [int(x) for x in raw[IDLER_HIST_BYTE:IDLER_HIST_BYTE + HIST_BINS]]

                # --- per-hist overflow handling (8-bit counters) ---
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
                            # If small decrease, treat as noise and keep previous value
                            if (last_raw - raw) <= self.overflow_sensitivity_signal:
                                raw = last_raw
                            else:
                                self.signal_hist_overflows[i] += 1
                        corrected = raw + (self.signal_hist_overflows[i] << 8)
                        signal_hist.append(int(corrected))
                        self.last_signal_hist_raw[i] = raw

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
                        corrected = raw + (self.idler_hist_overflows[i] << 8)
                        idler_hist.append(int(corrected))
                        self.last_idler_hist_raw[i] = raw

                # --- Robust handling for scalar counters (initialize on first frame) ---
                if self._last_read_times is None:
                    # First frame: set baseline and don't count overflow
                    self._last_read_times = int(read_times)
                    self.read_overflows = 0
                    real_read_times = int(read_times)
                else:
                    cur_rt = int(read_times)
                    last_rt = int(self._last_read_times)
                    if cur_rt < last_rt:
                        # small drops treated as noise (<= overflow_sensitivity): keep previous value
                        if (last_rt - cur_rt) <= self.overflow_sensitivity:
                            cur_rt = last_rt
                        else:
                            self.read_overflows += 1
                    self._last_read_times = cur_rt
                    real_read_times = cur_rt + (self.read_overflows << 19)

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
                    real_write_seq = cur_ws + (self.write_overflows << 19)

                # --- Per-bin overflow handling with first-frame initialization ---
                # Use single-wrap increment by default; initialize per-bin arrays on first frame
                if self.last_signal_bins is None:
                    # set baseline raw arrays and zero overflow counters
                    self.last_signal_bins = [int(x) for x in signal_bins_raw]
                    self.signal_overflows = [0] * CELL_COUNT
                    signal_bins = [int(x) for x in signal_bins_raw]
                else:
                    # ensure overflow list exists (defensive check for analyzer)
                    if self.signal_overflows is None:
                        self.signal_overflows = [0] * CELL_COUNT
                    signal_bins = []
                    for i in range(CELL_COUNT):
                        raw = int(signal_bins_raw[i])
                        last_raw = int(self.last_signal_bins[i])
                        if raw < last_raw:
                            # If decrease is small, treat as jitter and keep previous
                            # value instead of counting as overflow.
                            if (last_raw - raw) <= self.overflow_sensitivity_signal:
                                raw = last_raw
                            else:
                                # assume single wrap-around (device counter overflowed)
                                self.signal_overflows[i] += 1
                        corrected = raw + (self.signal_overflows[i] << 11)
                        signal_bins.append(int(corrected))
                        self.last_signal_bins[i] = raw

                if self.last_idler_bins is None:
                    self.last_idler_bins = [int(x) for x in idler_bins_raw]
                    self.idler_overflows = [0] * CELL_COUNT
                    idler_bins = [int(x) for x in idler_bins_raw]
                else:
                    if self.idler_overflows is None:
                        self.idler_overflows = [0] * CELL_COUNT
                    idler_bins = []
                    for i in range(CELL_COUNT):
                        raw = int(idler_bins_raw[i])
                        last_raw = int(self.last_idler_bins[i])
                        if raw < last_raw:
                            if (last_raw - raw) <= self.overflow_sensitivity_idler:
                                raw = last_raw
                            else:
                                self.idler_overflows[i] += 1
                        corrected = raw + (self.idler_overflows[i] << 11)
                        idler_bins.append(int(corrected))
                        self.last_idler_bins[i] = raw

                if self.last_coin_bins is None:
                    self.last_coin_bins = [int(x) for x in coin_bins_raw]
                    self.coin_overflows = [0] * CELL_COUNT
                    coin_bins = [int(x) for x in coin_bins_raw]
                else:
                    if self.coin_overflows is None:
                        self.coin_overflows = [0] * CELL_COUNT
                    coin_bins = []
                    for i in range(CELL_COUNT):
                        raw = int(coin_bins_raw[i])
                        last_raw = int(self.last_coin_bins[i])
                        if raw < last_raw:
                            if (last_raw - raw) <= self.overflow_sensitivity_coin:
                                raw = last_raw
                            else:
                                self.coin_overflows[i] += 1
                        corrected = raw + (self.coin_overflows[i] << 7)
                        coin_bins.append(int(corrected))
                        self.last_coin_bins[i] = raw

                # --- Robust handling for signal_all using bins sum protection ---
                # Ensure overflow counters exist
                if not hasattr(self, 'signal_all_overflows') or self.signal_all_overflows is None:
                    self.signal_all_overflows = 0
                if self._last_signal_all is None:
                    # first frame: set baseline
                    self._last_signal_all = int(raw_signal_all)
                    real_signal_all = int(raw_signal_all)
                else:
                    cur_sig = int(raw_signal_all)
                    last_sig = int(self._last_signal_all)
                    if cur_sig < last_sig:
                        # small drops treated as noise
                        if (last_sig - cur_sig) <= self.overflow_sensitivity_signal:
                            cur_sig = last_sig
                        else:
                            # candidate overflow: compare estimated scalar (with
                            # current overflow count) to sum of per-bin totals.
                            # If they are still close (within overflow_diff_threshold_signal)
                            # then this drop is likely not a real wrap — keep
                            # previous value instead of incrementing the overflow.
                            current_est = cur_sig + (self.signal_all_overflows << 19)
                            try:
                                sum_sig_bins = int(sum(signal_bins))
                            except Exception:
                                sum_sig_bins = 0
                            if abs(current_est - sum_sig_bins) <= getattr(self, 'overflow_diff_threshold_signal', self.overflow_sensitivity_signal):
                                # treat as noise — keep previous raw
                                cur_sig = last_sig
                            else:
                                # confirmed wrap
                                self.signal_all_overflows += 1
                    self._last_signal_all = cur_sig
                    real_signal_all = cur_sig + (self.signal_all_overflows << 19)
                # --- Robust handling for idler_all using bins sum protection ---
                if not hasattr(self, 'idler_all_overflows') or self.idler_all_overflows is None:
                    self.idler_all_overflows = 0
                if self._last_idler_all is None:
                    self._last_idler_all = int(raw_idler_all)
                    real_idler_all = int(raw_idler_all)
                else:
                    cur_idl = int(raw_idler_all)
                    last_idl = int(self._last_idler_all)
                    if cur_idl < last_idl:
                        if (last_idl - cur_idl) <= self.overflow_sensitivity_idler:
                            cur_idl = last_idl
                        else:
                            current_est_idl = cur_idl + (self.idler_all_overflows << 19)
                            try:
                                sum_idl_bins = int(sum(idler_bins))
                            except Exception:
                                sum_idl_bins = 0
                            if abs(current_est_idl - sum_idl_bins) <= getattr(self, 'overflow_diff_threshold_idler', self.overflow_sensitivity_idler):
                                cur_idl = last_idl
                            else:
                                self.idler_all_overflows += 1
                    self._last_idler_all = cur_idl
                    real_idler_all = cur_idl + (self.idler_all_overflows << 19)

                # --- Robust handling for coin_all using bins sum protection ---
                if not hasattr(self, 'coin_all_overflows') or self.coin_all_overflows is None:
                    self.coin_all_overflows = 0
                if self._last_coin_all is None:
                    self._last_coin_all = int(raw_coin_all)
                    real_coin_all = int(raw_coin_all)
                else:
                    cur_coin = int(raw_coin_all)
                    last_coin = int(self._last_coin_all)
                    if cur_coin < last_coin:
                        if (last_coin - cur_coin) <= self.overflow_sensitivity_coin:
                            cur_coin = last_coin
                        else:
                            current_est_coin = cur_coin + (self.coin_all_overflows << 15)
                            try:
                                sum_coin_bins = int(sum(coin_bins))
                            except Exception:
                                sum_coin_bins = 0
                            if abs(current_est_coin - sum_coin_bins) <= getattr(self, 'overflow_diff_threshold_coin', self.overflow_sensitivity_coin):
                                cur_coin = last_coin
                            else:
                                self.coin_all_overflows += 1
                    self._last_coin_all = cur_coin
                    real_coin_all = cur_coin + (self.coin_all_overflows << 15)

                # Build complete data dictionary (use int64 arrays to avoid dtype truncation)
                result = {
                    'signal_bins': np.array(signal_bins, dtype=np.int64),
                    'idler_bins': np.array(idler_bins, dtype=np.int64),
                    'coin_bins': np.array(coin_bins, dtype=np.int64),
                    'signal_bins_raw': np.array(signal_bins_raw, dtype=np.int64),
                    'idler_bins_raw': np.array(idler_bins_raw, dtype=np.int64),
                    'coin_bins_raw': np.array(coin_bins_raw, dtype=np.int64),
                    'write_count': int(real_write_seq),
                    'read_count': int(real_read_times),
                    'signal_all': int(real_signal_all),
                    'idler_all': int(real_idler_all),
                    'coin_all': int(real_coin_all),
                    # also include raw (un-corrected) scalar values for robust delta calculation
                    'signal_all_raw': int(signal_all),
                    'idler_all_raw': int(idler_all),
                    'coin_all_raw': int(coin_all),
                    'write_count_raw': int(write_seq),
                    'read_count_raw': int(read_times),
                    'timestamp': time.time()
                }

                # attach extra hist arrays
                result['signal_hist'] = np.array(signal_hist, dtype=np.int64)
                result['idler_hist'] = np.array(idler_hist, dtype=np.int64)
                result['signal_hist_raw'] = np.array(signal_hist_raw, dtype=np.int64)
                result['idler_hist_raw'] = np.array(idler_hist_raw, dtype=np.int64)

                self.data_ready.emit(result)

            except Exception as e:
                print(f"Error reading serial: {e}")
                self.alive = False
                break


# ============================================================
# Heatmap Table Widget with Different Colors
# ============================================================

class HeatmapTable(QTableWidget):
    def __init__(self, rows, cols, title, color_scheme="red", interleaved=None):
        """Heatmap table.

        interleaved: if None, use global HEATMAP_INTERLEAVED; if 1 use the
        original odd/even checkerboard mapping, if 0 use simple row-major
        ordering.
        """
        super().__init__(rows, cols)
        self.title = title
        self.color_scheme = color_scheme  # "red", "green", or "blue"
        # Decide interleaving mode: prefer explicit constructor arg; if None
        # default to True to preserve original behavior (checkerboard).
        if interleaved is None:
            interleaved = 1
        # Normalize to boolean
        self.interleaved = bool(int(interleaved))
        
        # Make it larger and set fixed size
        self.setFixedSize(500, 500)
        
        # Hide headers
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        
        # Set equal column widths
        for i in range(cols):
            self.setColumnWidth(i, 500 // cols)
        
        # Set equal row heights
        for i in range(rows):
            self.setRowHeight(i, 500 // rows)
        # store dims
        self._rows = rows
        self._cols = cols

        # mapping mode: 'cross' (checkerboard), 'linear' (row-major), or 'spiral' (center-out)
        if self.interleaved:
            self.mapping_mode = 'cross'
        else:
            self.mapping_mode = 'linear'

        # build initial coords map
        self._build_coords_map()

    def set_mapping(self, mode: str):
        """Set mapping mode: 'cross', 'linear', or 'spiral' and rebuild coords map."""
        mode = str(mode).lower()
        if mode not in ('cross', 'linear', 'spiral'):
            return
        self.mapping_mode = mode
        # keep interleaved flag in sync for backward compatibility
        self.interleaved = True if mode == 'cross' else False
        self._build_coords_map()

    def _build_coords_map(self):
        """(Re)build the internal coords mapping based on current
        interleaved setting and stored rows/cols."""
        coords = []
        mode = getattr(self, 'mapping_mode', 'cross' if self.interleaved else 'linear')
        if mode == 'cross':
            for r in range(self._rows):
                for c in range(self._cols):
                    if (r + c) % 2 == 0:
                        coords.append((r, c))
            for r in range(self._rows):
                for c in range(self._cols):
                    if (r + c) % 2 == 1:
                        coords.append((r, c))
        elif mode == 'linear':
            for r in range(self._rows):
                for c in range(self._cols):
                    coords.append((r, c))
        elif mode == 'spiral':
            # center-out spiral mapping: first index -> center cell
            nrows = self._rows
            ncols = self._cols
            # assume square grid; fall back to row-major for non-square
            if nrows != ncols:
                for r in range(nrows):
                    for c in range(ncols):
                        coords.append((r, c))
            else:
                n = nrows
                cx = (n - 1) // 2
                cy = (n - 1) // 2
                coords = [(cx, cy)]
                step = 1
                dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                x, y = cx, cy
                d = 0
                while len(coords) < n * n:
                    for _ in range(2):
                        dx, dy = dirs[d % 4]
                        for _ in range(step):
                            x += dx
                            y += dy
                            if 0 <= x < n and 0 <= y < n:
                                coords.append((x, y))
                                if len(coords) >= n * n:
                                    break
                        d += 1
                        if len(coords) >= n * n:
                            break
                    step += 1

        # safety: ensure coords length equals cell count
        total = self._rows * self._cols
        if len(coords) < total:
            # fill remaining with row-major
            exist = set(coords)
            for r in range(self._rows):
                for c in range(self._cols):
                    if (r, c) not in exist:
                        coords.append((r, c))

        self._coords_map = coords

    def set_interleaved(self, on: bool):
        """Set interleaved mapping on (True) or off (False) and rebuild map.

        After calling this, caller should re-render/update the table (e.g.
        call update_data with the current data) to reflect the new mapping.
        """
        self.interleaved = bool(on)
        self._build_coords_map()
    
    def update_data(self, data):
        """Update table with data and apply heatmap coloring with different color schemes"""
        total = self._rows * self._cols
        if len(data) != total:
            return

        # Normalize using NumPy for safety
        try:
            arr = np.asarray(data, dtype=float)
            max_val = float(arr.max()) if arr.size > 0 and arr.max() > 0 else 1.0
        except Exception:
            # fallback
            max_val = max(data) if max(data) > 0 else 1

        # Fill table according to the precomputed coords map
        for idx, (r, c) in enumerate(self._coords_map):
            # Support float values: format text for display but keep
            # numeric value for intensity calculation.
            try:
                value_f = float(data[idx])
            except Exception:
                value_f = 0.0

            # Pretty-print: show integer-like numbers without decimals,
            # otherwise show 3 decimal places.
            if abs(value_f - round(value_f)) < 1e-9:
                text = str(int(round(value_f)))
            else:
                text = f"{value_f:.3f}"

            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)

            # Compute intensity based on float value and max_val
            try:
                intensity = 0
                if max_val > 0:
                    intensity = min(255, int(255.0 * (value_f / max_val)))
            except Exception:
                intensity = 0

            if self.color_scheme == "red":
                item.setBackground(QColor(255, 255 - intensity, 255 - intensity))
            elif self.color_scheme == "green":
                item.setBackground(QColor(255 - intensity, 255, 255 - intensity))
            elif self.color_scheme == "blue":
                item.setBackground(QColor(255 - intensity, 255 - intensity, 255))

            self.setItem(r, c, item)


# ============================================================
# Matplotlib Figure for Time Series Plots
# ============================================================

class TimeSeriesPlot(QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # create a backend-specific canvas instance
        try:
            self.canvas = FigureCanvas(self.fig)
            self.canvas.setParent(self)
        except Exception:
            # fallback: no-op canvas
            self.canvas = None

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.grid(True, alpha=0.3)

        # Initialize data storage
        self.timestamps = []
        self.data_series = {}

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
    
    def add_series(self, name, color, label):
        """Add a new data series to the plot"""
        self.data_series[name] = {
            'values': [],
            'color': color,
            'label': label
        }
    
    def update_plot(self, timestamp, new_data):
        """Update the plot with new data point"""
        # Add new timestamp
        self.timestamps.append(timestamp)
        
        # Add new values for each registered series. If a series is missing in
        # new_data, append np.nan so lengths remain equal and matplotlib can
        # handle gaps.
        for name, series in self.data_series.items():
            if name in new_data:
                series['values'].append(new_data[name])
            else:
                # keep shape consistent; use NaN for missing samples
                series['values'].append(np.nan)
        
        # Keep only last 100 points to prevent memory issues
        max_points = 100
        if len(self.timestamps) > max_points:
            self.timestamps = self.timestamps[-max_points:]
            for series in self.data_series.values():
                series['values'] = series['values'][-max_points:]
        
        # Clear and redraw
        self.ax.clear()
        
        # Plot each series
        for name, series in self.data_series.items():
            if len(series['values']) > 0:
                # Calculate relative time (seconds from first point)
                if len(self.timestamps) == len(series['values']):
                    relative_times = [t - self.timestamps[0] for t in self.timestamps]
                    self.ax.plot(relative_times, series['values'], 
                                color=series['color'], label=series['label'], linewidth=2)
        
        self.ax.set_xlabel('Time (s)')
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
        
        # Format y-axis to use integer ticks
        self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        if self.canvas is not None:
            try:
                self.canvas.draw()
            except Exception:
                pass

    def clear_data(self):
        """Clear stored timestamps and series values so the plot restarts from empty."""
        self.timestamps = []
        for series in self.data_series.values():
            series['values'] = []
        # Redraw empty plot
        self.ax.clear()
        self.ax.set_xlabel('Time (s)')
        self.ax.grid(True, alpha=0.3)
        if self.canvas is not None:
            try:
                self.canvas.draw()
            except Exception:
                pass


# ============================================================
# Simple Bar (Histogram) Plot for 1D arrays
# ============================================================
class BarPlot(QWidget):
    def __init__(self, parent=None, width=5, height=2, dpi=100, xlabel='Index'):
        super().__init__(parent)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        try:
            self.canvas = FigureCanvas(self.fig)
            self.canvas.setParent(self)
        except Exception:
            self.canvas = None
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(xlabel)
        self.ax.grid(True, alpha=0.3)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def update_hist(self, timestamp, values, labels=None, color='tab:blue'):
        # values: 1D iterable of raw counts. Display raw numbers (no normalization).
        vals = np.asarray(values, dtype=float)

        # Clear and draw raw counts as bars
        self.ax.clear()
        x = np.arange(1, len(vals) + 1)
        self.ax.bar(x, vals, color=color, alpha=0.8)
        if labels is not None:
            try:
                # use provided labels as x tick labels if length matches
                if len(labels) == len(vals):
                    self.ax.set_xticks(x)
                    self.ax.set_xticklabels(labels)
            except Exception:
                pass

        self.ax.set_xlabel('Index')
        self.ax.set_ylabel('Count')
        # autoscale y-axis with small headroom; ensure at least 1 to avoid zero-range
        try:
            ymax = float(vals.max()) if vals.size > 0 else 1.0
            ymax = max(ymax, 1.0) * 1.1
            self.ax.set_ylim(0, ymax)
        except Exception:
            self.ax.set_ylim(0, 1.0)

        # Prefer integer ticks for counts
        try:
            self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except Exception:
            pass

        self.ax.grid(True, alpha=0.3)
        if self.canvas is not None:
            try:
                self.canvas.draw()
            except Exception:
                pass

    def clear_data(self):
        self.ax.clear()
        self.ax.set_xlabel('Index')
        self.ax.set_ylabel('Count')
        if self.canvas is not None:
            try:
                self.canvas.draw()
            except Exception:
                pass


# ============================================================
# Main GUI
# ============================================================

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram Data Viewer (1159-byte protocol, 15x15/225)")
        self.resize(2400, 1350)

        self.driver = Driver()
        self.driver.data_ready.connect(self.update_from_serial)

        self.virtual_mode = False
        
        # Baseline storage for clear functionality
        self.baseline = None
        self.current_data = None
        
        # Data storage
        self.signal_bins = np.zeros(CELL_COUNT, dtype=int)
        self.idler_bins = np.zeros(CELL_COUNT, dtype=int)
        self.coin_bins = np.zeros(CELL_COUNT, dtype=int)
        self.signal_increment = np.zeros(CELL_COUNT, dtype=int)
        self.idler_increment = np.zeros(CELL_COUNT, dtype=int)
        self.coin_increment = np.zeros(CELL_COUNT, dtype=int)
        self.write_count = 0
        self.read_count = 0
        self.signal_all = 0
        self.idler_all = 0
        self.coin_all = 0
        # Average excitation value (signal_all / (write_count * CELL_COUNT))
        self.avg_excitation = 0.0
        
        
        self.prev_signal_bins = np.zeros(CELL_COUNT, dtype=int)
        self.prev_idler_bins = np.zeros(CELL_COUNT, dtype=int)
        self.prev_coin_bins = np.zeros(CELL_COUNT, dtype=int)
        self.prev_signal_all = 0
        self.prev_idler_all = 0
        self.prev_coin_all = 0
        # raw scalar previous values for modular-delta calculation
        self.prev_signal_all_raw = None
        self.prev_idler_all_raw = None
        self.prev_coin_all_raw = None
        self.prev_write_count_raw = None
        self.prev_read_count_raw = None
        self.prev_write_count = 0
        self.prev_read_count = 0
    
        self.signal_increment_upd = np.zeros(CELL_COUNT, dtype=int)
        self.idler_increment_upd = np.zeros(CELL_COUNT, dtype=int)
        self.coin_increment_upd = np.zeros(CELL_COUNT, dtype=int)
        self.signal_all_upd = 0
        self.idler_all_upd = 0
        self.coin_all_upd = 0
        self.write_count_upd = 0
        self.read_count_upd = 0
        
        # For time series tracking
        self.start_time = time.time()
        self.data_history = []
        # Track previous update time to compute per-second rates
        self.prev_update_time = None

        # New histogram arrays (10 bins each parsed from extra bits)
        self.signal_hist = np.zeros(10, dtype=int)
        self.idler_hist = np.zeros(10, dtype=int)
        self.prev_signal_hist = np.zeros(10, dtype=int)
        self.prev_idler_hist = np.zeros(10, dtype=int)
        self.signal_hist_upd = np.zeros(10, dtype=int)
        self.idler_hist_upd = np.zeros(10, dtype=int)
        # Excitation formula: configured in code. Default uses total signal
        # counts divided by (write_count * CELL_COUNT) to match the requested default.
        # Small epsilon avoids division-by-zero.
        self.exc_formula_text = "signal_all/(write_count*CELL_COUNT)"

        self.setup_ui()

    # --------------------- UI Layout ---------------------

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Top info row: counts, calculations, and a small log
        top_info_layout = QHBoxLayout()
        top_info_layout.setContentsMargins(5, 5, 5, 5)

        # Sum values display
        sums_group = QGroupBox("Total Counts")
        sums_layout = QGridLayout()
        self.lbl_signal_all = QLabel("Signal All: 0")
        self.lbl_idler_all = QLabel("Idler All: 0")
        self.lbl_coin_all = QLabel("Coin All: 0")
        self.lbl_signal_sum = QLabel("Signal Sum: 0")
        self.lbl_idler_sum = QLabel("Idler Sum: 0")
        self.lbl_coin_sum = QLabel("Coin Sum: 0")
        self.lbl_signal_all_upd = QLabel("Signal All Δ: 0")
        self.lbl_idler_all_upd = QLabel("Idler All Δ: 0")
        self.lbl_coin_all_upd = QLabel("Coin All Δ: 0")
        self.lbl_write_count = QLabel("Write Count: 0")
        self.lbl_read_count = QLabel("Read Count: 0")
        self.lbl_write_count_upd = QLabel("Write Count Δ: 0")
        self.lbl_read_count_upd = QLabel("Read Count Δ: 0")

        sums_layout.addWidget(self.lbl_signal_all, 0, 0)
        sums_layout.addWidget(self.lbl_idler_all, 0, 1)
        sums_layout.addWidget(self.lbl_coin_all, 0, 2)
        sums_layout.addWidget(self.lbl_signal_sum, 1, 0)
        sums_layout.addWidget(self.lbl_idler_sum, 1, 1)
        sums_layout.addWidget(self.lbl_coin_sum, 1, 2)
        sums_layout.addWidget(self.lbl_signal_all_upd, 2, 0)
        sums_layout.addWidget(self.lbl_idler_all_upd, 2, 1)
        sums_layout.addWidget(self.lbl_coin_all_upd, 2, 2)
        sums_layout.addWidget(self.lbl_write_count, 3, 0)
        sums_layout.addWidget(self.lbl_read_count, 3, 1)
        sums_layout.addWidget(self.lbl_write_count_upd, 4, 0)
        sums_layout.addWidget(self.lbl_read_count_upd, 4, 1)
        sums_group.setLayout(sums_layout)

        # Calculation results group
        calc_group = QGroupBox("Calculation Results")
        calc_layout = QVBoxLayout()
        # keep initial labels consistent with update_ui naming (g2 / retrival)
        self.lbl_calc1 = QLabel("g2: 1.0000")
        self.lbl_calc2 = QLabel("Retrival Rate: 0.000000")
        self.lbl_calc3 = QLabel("g2_per_second: 1.0000")
        self.lbl_calc4 = QLabel("Retriva Rate_per_second: 0.000000")
        self.lbl_avg_excitation = QLabel("Avg Excitation: 0.000000")
        # additional realtime calculation displays
        self.lbl_real_retrivel = QLabel("Real Retrivel: 0.0000")
        self.lbl_factor_g2 = QLabel("Factor g2: 1.0000")
        calc_layout.addWidget(self.lbl_calc1)
        calc_layout.addWidget(self.lbl_calc2)
        calc_layout.addWidget(self.lbl_calc3)
        calc_layout.addWidget(self.lbl_calc4)
        calc_layout.addWidget(self.lbl_avg_excitation)
        calc_layout.addWidget(self.lbl_real_retrivel)
        calc_layout.addWidget(self.lbl_factor_g2)
        calc_group.setLayout(calc_layout)

        # Free-form user notes box (editable) placed next to Calculation Results
        self.user_notes = QTextEdit()
        self.user_notes.setFixedHeight(100)
        self.user_notes.setPlaceholderText('Enter notes here...')
        # Excitation formula input (single-line) and apply button
        self.edit_exc_formula = QLineEdit()
        self.edit_exc_formula.setPlaceholderText("Enter python expression, use variables: np, CELL_COUNT, signal_increment, idler_increment, coin_increment, write_count, write_count_upd, signal_all_upd, idler_all_upd, coin_all_upd, signal_all")
        try:
            # prefill textbox with current default formula
            self.edit_exc_formula.setText(str(self.exc_formula_text))
        except Exception:
            pass
        self.btn_apply_exc = QPushButton('Apply Exc Formula')
        self.btn_apply_exc.setToolTip('Compile and apply the excitation formula (safe eval with numpy as np)')
        self.btn_apply_exc.clicked.connect(self.apply_exc_formula)

        # Text log (reduced height to make room for extra hist plots)
        self.text_edit = QTextEdit()
        self.text_edit.setFixedHeight(60)
        self.text_edit.setMaximumHeight(100)

        top_info_layout.addWidget(sums_group, 1)
        top_info_layout.addWidget(calc_group, 1)
        # left: user notes + excitation formula controls
        notes_layout = QVBoxLayout()
        notes_layout.addWidget(self.user_notes)
        # small horizontal for formula + apply
        frm_row = QHBoxLayout()
        frm_row.addWidget(self.edit_exc_formula)
        frm_row.addWidget(self.btn_apply_exc)
        notes_layout.addLayout(frm_row)
        top_info_layout.addLayout(notes_layout, 1)
        top_info_layout.addWidget(self.text_edit, 1)
        main_layout.addLayout(top_info_layout)

        # Main content: left (2x2 plots) and right (heatmaps + controls)
        content_layout = QHBoxLayout()

        # Left: 2x2 plots grid
        plots_grid = QGridLayout()
        self.plot_sums = TimeSeriesPlot(self, width=4, height=2)
        self.plot_sums.add_series('signal_rate', 'red', 'Signal rate (counts/s)')
        self.plot_sums.add_series('idler_rate', 'green', 'Idler rate (counts/s)')
        self.plot_sums.add_series('coin_sum_delta', 'blue', 'Coin Sum Δ')

        self.plot_cal1_cal3 = TimeSeriesPlot(self, width=4, height=2)
        self.plot_cal1_cal3.add_series('cal1', 'red', 'g2')
        self.plot_cal1_cal3.add_series('cal3', 'blue', 'g2_per_second')

        self.plot_cal2_cal4 = TimeSeriesPlot(self, width=4, height=2)
        self.plot_cal2_cal4.add_series('cal2', 'green', 'Retrieval Rate')
        self.plot_cal2_cal4.add_series('cal4', 'orange', 'Retrieval Rate_per_second')

        self.plot_avg_excitation = TimeSeriesPlot(self, width=4, height=2)
        self.plot_avg_excitation.add_series('avg_excitation', 'purple', 'Avg excitation (signal_all / (writecount*CELL_COUNT))')
        # also plot raw signal_all as a second line
        self.plot_avg_excitation.add_series('signal_all', 'black', 'Signal All/(write_count*CELL_COUNT)')

        # --- New extra plots (placeholders for user formulas) ---
        self.plot_extra1 = TimeSeriesPlot(self, width=4, height=2)
        self.plot_extra1.add_series('real_retrivel', 'magenta', 'Real_retrivel Rate')

        self.plot_extra2 = TimeSeriesPlot(self, width=4, height=2)
        self.plot_extra2.add_series('factor_g2', 'cyan', 'Factor_g2')

        # New 10-bin histogram bar plots (raw counts)
        self.hist_plot_signal = BarPlot(self, width=4, height=2, xlabel='Bin')
        self.hist_plot_idler = BarPlot(self, width=4, height=2, xlabel='Bin')

        plots_grid.addWidget(self.plot_sums, 0, 0)
        plots_grid.addWidget(self.plot_cal1_cal3, 0, 1)
        plots_grid.addWidget(self.plot_cal2_cal4, 1, 0)
        plots_grid.addWidget(self.plot_avg_excitation, 1, 1)
        # place the two new time-series plots on a new row
        plots_grid.addWidget(self.plot_extra1, 2, 0)
        plots_grid.addWidget(self.plot_extra2, 2, 1)
        # place the two new histogram plots on a new row (row 3)
        plots_grid.addWidget(self.hist_plot_signal, 3, 0)
        plots_grid.addWidget(self.hist_plot_idler, 3, 1)

        content_layout.addLayout(plots_grid, 1)

        # Right: heatmaps (2x3) and controls under them
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(5, 5, 5, 5)

        heatmap_layout = QGridLayout()
        # Create heatmap tables with different colors
        self.heatmap_signal = HeatmapTable(GRID_SIZE, GRID_SIZE, "Signal Heatmap", "red")
        self.heatmap_idler = HeatmapTable(GRID_SIZE, GRID_SIZE, "Idler Heatmap", "green")
        self.heatmap_coin = HeatmapTable(GRID_SIZE, GRID_SIZE, "Coin Heatmap", "blue")
        self.heatmap_calc1 = HeatmapTable(GRID_SIZE, GRID_SIZE, "Calc Heatmap 1", "red")
        self.heatmap_calc2 = HeatmapTable(GRID_SIZE, GRID_SIZE, "Calc Heatmap 2", "green")
        self.heatmap_calc3 = HeatmapTable(GRID_SIZE, GRID_SIZE, "Calc Heatmap 3", "blue")

        heatmap_layout.addWidget(QLabel("Signal Increment"), 0, 0)
        heatmap_layout.addWidget(QLabel("Idler Increment"), 0, 1)
        heatmap_layout.addWidget(QLabel("Coin Increment"), 0, 2)
        heatmap_layout.addWidget(self.heatmap_signal, 1, 0)
        heatmap_layout.addWidget(self.heatmap_idler, 1, 1)
        heatmap_layout.addWidget(self.heatmap_coin, 1, 2)

        heatmap_layout.addWidget(QLabel("Calc Matrix 1"), 2, 0)
        heatmap_layout.addWidget(QLabel("Calc Matrix 2"), 2, 1)
        heatmap_layout.addWidget(QLabel("Calc Matrix 3"), 2, 2)
        heatmap_layout.addWidget(self.heatmap_calc1, 3, 0)
        heatmap_layout.addWidget(self.heatmap_calc2, 3, 1)
        heatmap_layout.addWidget(self.heatmap_calc3, 3, 2)

        right_panel.addLayout(heatmap_layout)

        # Controls (placed under heatmaps)
        ctrl_layout = QHBoxLayout()
        self.com_combo = QComboBox()
        self.refresh_ports()
        # Mapping selection: linear / cross / spiral
        self.mapping_combo = QComboBox()
        self.mapping_combo.addItems(['linear', 'cross', 'spiral'])
        # default to cross to match original checkerboard behaviour
        try:
            self.mapping_combo.setCurrentText('cross')
        except Exception:
            pass
        # Overflow sensitivity controls (per-type) and diff thresholds
        # Signal
        self.spin_noise_signal = QSpinBox()
        self.spin_noise_signal.setRange(0, 1000000)
        self.spin_noise_signal.setValue(100)
        self.spin_diff_signal = QSpinBox()
        self.spin_diff_signal.setRange(0, 10000)
        self.spin_diff_signal.setValue(3500)
        self.spin_noise_signal.setToolTip('Signal: ignore decreases <= this as noise')
        self.spin_diff_signal.setToolTip('Signal: diff threshold vs sum(bins)')
        # Idler
        self.spin_noise_idler = QSpinBox()
        self.spin_noise_idler.setRange(0, 1000000)
        self.spin_noise_idler.setValue(100)
        self.spin_diff_idler = QSpinBox()
        self.spin_diff_idler.setRange(0, 10000)
        self.spin_diff_idler.setValue(3500)
        self.spin_noise_idler.setToolTip('Idler: ignore decreases <= this as noise')
        self.spin_diff_idler.setToolTip('Idler: diff threshold vs sum(bins)')
        # Coin
        self.spin_noise_coin = QSpinBox()
        self.spin_noise_coin.setRange(0, 1000000)
        self.spin_noise_coin.setValue(100)
        self.spin_diff_coin = QSpinBox()
        self.spin_diff_coin.setRange(0, 10000)
        self.spin_diff_coin.setValue(1000)
        self.spin_noise_coin.setToolTip('Coin: ignore decreases <= this as noise')
        self.spin_diff_coin.setToolTip('Coin: diff threshold vs sum(bins)')
        # Factor K control: checkbox + factor value (double)
        from PyQt5.QtWidgets import QCheckBox, QDoubleSpinBox
        self.chk_use_k = QCheckBox('Use K*g2')
        self.spin_factor_k = QDoubleSpinBox()
        self.spin_factor_k.setDecimals(6)
        self.spin_factor_k.setRange(0.0, 100.0)
        # default K = 2/CELL_COUNT
        self.spin_factor_k.setValue(2.0)
        self.spin_factor_k.setToolTip('When enabled, set factor_g2 = K * g2')
        self.btn_open = QPushButton("Open")
        self.btn_close = QPushButton("Close")
        self.btn_start = QPushButton("Start (y)")
        self.btn_stop = QPushButton("Stop (n)")
        self.btn_virtual = QPushButton("Virtual Test")
        self.btn_clear = QPushButton("Clear Baseline")
        self.btn_save = QPushButton("Save CSV")
        # cross toggle removed — feature completed

        ctrl_layout.addWidget(QLabel("Port:"))
        ctrl_layout.addWidget(self.com_combo)
        ctrl_layout.addWidget(QLabel("Mapping:"))
        ctrl_layout.addWidget(self.mapping_combo)
        # factor K widgets
        ctrl_layout.addWidget(self.chk_use_k)
        ctrl_layout.addWidget(QLabel('K:'))
        ctrl_layout.addWidget(self.spin_factor_k)
        # Add per-type noise/diff controls (compact labels)
        ctrl_layout.addWidget(QLabel("Sig N:"))
        ctrl_layout.addWidget(self.spin_noise_signal)
        ctrl_layout.addWidget(QLabel("Sig D:"))
        ctrl_layout.addWidget(self.spin_diff_signal)
        ctrl_layout.addWidget(QLabel("Idl N:"))
        ctrl_layout.addWidget(self.spin_noise_idler)
        ctrl_layout.addWidget(QLabel("Idl D:"))
        ctrl_layout.addWidget(self.spin_diff_idler)
        ctrl_layout.addWidget(QLabel("Coin N:"))
        ctrl_layout.addWidget(self.spin_noise_coin)
        ctrl_layout.addWidget(QLabel("Coin D:"))
        ctrl_layout.addWidget(self.spin_diff_coin)
        ctrl_layout.addWidget(self.btn_open)
        ctrl_layout.addWidget(self.btn_close)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_virtual)
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.btn_save)

        right_panel.addLayout(ctrl_layout)

        content_layout.addLayout(right_panel, 2)
        main_layout.addLayout(content_layout)

        # Connections
        self.btn_open.clicked.connect(self.open_serial)
        self.btn_close.clicked.connect(self.close_serial)
        self.btn_start.clicked.connect(self.start_output)
        self.btn_stop.clicked.connect(self.stop_output)
        self.btn_virtual.clicked.connect(self.toggle_virtual)
        self.btn_clear.clicked.connect(self.clear_baseline)
        self.btn_save.clicked.connect(self.save_csv)

        # Timer for virtual test
        self.timer = QTimer()
        self.timer.timeout.connect(self.generate_virtual_data)

        # cross toggle removed
        # connect mapping combo
        try:
            self.mapping_combo.currentTextChanged.connect(self.change_mapping)
        except Exception:
            pass

    def apply_exc_formula(self):
        """Read the formula from the textbox, test-evaluate with safe locals,
        and store it in `self.exc_formula_text`. Empty string clears custom formula.
        """
        try:
            text = str(self.edit_exc_formula.text()).strip()
        except Exception:
            text = ''

        if not text:
            self.exc_formula_text = ''
            try:
                self.text_edit.append('Cleared excitation formula; using default avg_excitation')
            except Exception:
                pass
            return

        # quick sanity test: try eval with minimal locals to catch syntax/runtime errors
        test_locals = {
            'np': np,
            'CELL_COUNT': CELL_COUNT,
            'signal_increment': np.zeros(1),
            'idler_increment': np.zeros(1),
            'coin_increment': np.zeros(1),
            'write_count': 1,
            'write_count_upd': 1,
            'signal_all_upd': 1,
            'idler_all_upd': 1,
            'coin_all_upd': 1,
            'signal_rate': 1.0,
            'idler_rate': 1.0,
            'coin_delta': 1.0,
            'avg_excitation': 0.0,
            'signal_all': 1
        }
        try:
            _ = eval(text, {'np': np, 'CELL_COUNT': CELL_COUNT}, test_locals)
        except Exception as e:
            try:
                self.text_edit.append(f'Invalid excitation formula: {e}')
            except Exception:
                pass
            return

        # passed sanity test; apply
        self.exc_formula_text = text
        try:
            self.text_edit.append(f'Applied excitation formula: {text}')
        except Exception:
            pass

    def toggle_cross(self):
        """(removed)"""

    def change_mapping(self, text: str):
        """Handler for mapping combo changes. Applies mapping to all heatmaps and refreshes UI."""
        mode = str(text).lower()
        try:
            for heat in (self.heatmap_signal, self.heatmap_idler, self.heatmap_coin,
                         self.heatmap_calc1, self.heatmap_calc2, self.heatmap_calc3):
                try:
                    heat.set_mapping(mode)
                except Exception:
                    # if heat doesn't support set_mapping, try set_interleaved()
                    if mode == 'cross':
                        try:
                            heat.set_interleaved(True)
                        except Exception:
                            pass
                    else:
                        try:
                            heat.set_interleaved(False)
                        except Exception:
                            pass

            # mapping change applied (legacy cross button removed)

            # refresh displayed data according to new mapping
            try:
                self.heatmap_signal.update_data(self.signal_increment)
                self.heatmap_idler.update_data(self.idler_increment)
                self.heatmap_coin.update_data(self.coin_increment)
                # also refresh calc heatmaps
                try:
                    self.heatmap_calc1.update_data(self.calc_heat1)
                    self.heatmap_calc2.update_data(self.calc_heat2)
                    self.heatmap_calc3.update_data(self.calc_heat3)
                except Exception:
                    pass
            except Exception as e:
                self.text_edit.append(f"mapping apply error: {e}")
        except Exception as e:
            self.text_edit.append(f"change_mapping error: {e}")

    def refresh_ports(self):
        self.com_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        items = ports[:] if ports else []
        # Ensure COM6 is available as the default selection (insert if missing)
        if 'COM6' not in items:
            items.insert(0, 'COM6')
        self.com_combo.addItems(items if items else ["No ports found"])
        try:
            self.com_combo.setCurrentText('COM6')
        except Exception:
            pass

    # --------------------- Serial Control ---------------------

    def open_serial(self):
        port = self.com_combo.currentText()
        if "No ports" in port:
            self.text_edit.append("⚠️ No valid COM port found.")
            return
        self.driver.open(port)
        self.text_edit.append(f"Opened {port}")

    def close_serial(self):
        self.driver.close()
        self.text_edit.append("Closed serial port.")

    def start_output(self):
        # Lock in overflow sensitivity and disable editing after start
        try:
            # Signal
            vs = int(self.spin_noise_signal.value())
            vd = int(self.spin_diff_signal.value())
            self.driver.overflow_sensitivity_signal = vs
            self.driver.overflow_diff_threshold_signal = vd
            self.spin_noise_signal.setEnabled(False)
            self.spin_diff_signal.setEnabled(False)
            # Idler
            vi = int(self.spin_noise_idler.value())
            vdi = int(self.spin_diff_idler.value())
            self.driver.overflow_sensitivity_idler = vi
            self.driver.overflow_diff_threshold_idler = vdi
            self.spin_noise_idler.setEnabled(False)
            self.spin_diff_idler.setEnabled(False)
            # Coin
            vc = int(self.spin_noise_coin.value())
            vdc = int(self.spin_diff_coin.value())
            self.driver.overflow_sensitivity_coin = vc
            self.driver.overflow_diff_threshold_coin = vdc
            self.spin_noise_coin.setEnabled(False)
            self.spin_diff_coin.setEnabled(False)
        except Exception:
            pass
        self.driver.set_output()
        self.text_edit.append("Sent 'y' to start output.")
        # disable formula editing while running
        try:
            self.edit_exc_formula.setEnabled(False)
            self.btn_apply_exc.setEnabled(False)
        except Exception:
            pass

    def stop_output(self):
        self.driver.set_idle()
        try:
            # re-enable editing so user can change threshold between runs
            self.spin_noise_signal.setEnabled(True)
            self.spin_diff_signal.setEnabled(True)
            self.spin_noise_idler.setEnabled(True)
            self.spin_diff_idler.setEnabled(True)
            self.spin_noise_coin.setEnabled(True)
            self.spin_diff_coin.setEnabled(True)
        except Exception:
            pass
        # excitation formula is code-configured; no UI widgets to re-enable
        try:
            self.edit_exc_formula.setEnabled(True)
            self.btn_apply_exc.setEnabled(True)
        except Exception:
            pass
        self.text_edit.append("Sent 'n' to stop output.")

    def toggle_virtual(self):
        self.virtual_mode = not self.virtual_mode
        if self.virtual_mode:
            self.timer.start(500)
            self.text_edit.append(" Virtual test mode ON")
        else:
            self.timer.stop()
            self.text_edit.append(" Virtual test mode OFF")

    def clear_baseline(self):
        """Set current data as baseline for incremental display and reset all counters"""
        if self.current_data is not None:
            self.baseline = {
                'signal_bins': self.current_data['signal_bins'].copy(),
                'idler_bins': self.current_data['idler_bins'].copy(), 
                'coin_bins': self.current_data['coin_bins'].copy(),
                'signal_hist': self.current_data.get('signal_hist', np.zeros(10, dtype=int)).copy(),
                'idler_hist': self.current_data.get('idler_hist', np.zeros(10, dtype=int)).copy(),
                'signal_all': self.current_data['signal_all'],
                'idler_all': self.current_data['idler_all'],
                'coin_all': self.current_data['coin_all'],
                'write_count': self.current_data['write_count'],
                'read_count': self.current_data['read_count'],
                'timestamp': self.current_data['timestamp']
            }
            
            # Reset all previous values to current values so increments start from zero
            self.prev_signal_bins = self.current_data['signal_bins'].copy()
            self.prev_idler_bins = self.current_data['idler_bins'].copy()
            self.prev_coin_bins = self.current_data['coin_bins'].copy()
            self.prev_signal_all = self.current_data['signal_all']
            self.prev_idler_all = self.current_data['idler_all']
            self.prev_coin_all = self.current_data['coin_all']
            self.prev_write_count = self.current_data['write_count']
            self.prev_read_count = self.current_data['read_count']
            # hist prev
            try:
                self.prev_signal_hist = self.current_data.get('signal_hist', np.zeros(10, dtype=int)).copy()
                self.prev_idler_hist = self.current_data.get('idler_hist', np.zeros(10, dtype=int)).copy()
            except Exception:
                self.prev_signal_hist = np.zeros(10, dtype=int)
                self.prev_idler_hist = np.zeros(10, dtype=int)
            
            # Reset increment values to zero
            self.signal_increment_upd = np.zeros(CELL_COUNT, dtype=int)
            self.idler_increment_upd = np.zeros(CELL_COUNT, dtype=int)
            self.coin_increment_upd = np.zeros(CELL_COUNT, dtype=int)
            self.signal_all_upd = 0
            self.idler_all_upd = 0
            self.coin_all_upd = 0
            self.write_count_upd = 0
            self.read_count_upd = 0
            
            # Also reset the absolute values that are displayed in UI to zero
            self.signal_all = 0
            self.idler_all = 0
            self.coin_all = 0
            self.write_count = 0
            self.read_count = 0
            # Reset avg excitation display/state
            self.avg_excitation = 0.0
            try:
                if hasattr(self, 'lbl_avg_excitation'):
                    self.lbl_avg_excitation.setText("Avg Excitation: 0.000000")
            except Exception:
                pass
            self.signal_bins = np.zeros(CELL_COUNT, dtype=int)
            self.idler_bins = np.zeros(CELL_COUNT, dtype=int)
            self.coin_bins = np.zeros(CELL_COUNT, dtype=int)
            # reset hist arrays
            self.signal_hist = np.zeros(10, dtype=int)
            self.idler_hist = np.zeros(10, dtype=int)
            self.prev_signal_hist = np.zeros(10, dtype=int)
            self.prev_idler_hist = np.zeros(10, dtype=int)
            
            self.text_edit.append(" Baseline cleared - all counters reset to zero")
        else:
            self.text_edit.append(" No data available to set as baseline")
        
        # Also clear time-series plots so they restart drawing from an empty state
        try:
            for plot in (self.plot_sums, self.plot_cal1_cal3, self.plot_cal2_cal4, self.plot_avg_excitation):
                if hasattr(plot, 'clear_data'):
                    plot.clear_data()
            # clear the newly added plots as well
            try:
                if hasattr(self, 'plot_extra1'):
                    self.plot_extra1.clear_data()
                if hasattr(self, 'plot_extra2'):
                    self.plot_extra2.clear_data()
                # clear new histogram plots
                try:
                    if hasattr(self, 'hist_plot_signal'):
                        self.hist_plot_signal.clear_data()
                    if hasattr(self, 'hist_plot_idler'):
                        self.hist_plot_idler.clear_data()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        # Reset timing for rate calculations and history
        self.prev_update_time = None
        self.start_time = time.time()
        self.data_history = []
            
    
    # --------------------- Data Update ---------------------

    def update_from_serial(self, data):
        # Store current data
        self.current_data = data
        
        # Calculate increments since last update for per-bin arrays (already handled by driver overflows)
        self.signal_increment_upd = data['signal_bins'] - self.prev_signal_bins
        self.idler_increment_upd = data['idler_bins'] - self.prev_idler_bins
        self.coin_increment_upd = data['coin_bins'] - self.prev_coin_bins

        # Use modular-delta on raw scalar fields to avoid negative deltas caused by wrap
        def modular_delta(cur, prev, bits):
            try:
                if prev is None:
                    return 0
                modulus = 1 << bits
                d = (int(cur) - int(prev)) % modulus
                return int(d)
            except Exception:
                try:
                    return int(cur) - int(prev)
                except Exception:
                    return 0

        # compute scalar updates from raw fields if present, otherwise fall back
        if 'signal_all_raw' in data:
            self.signal_all_upd = modular_delta(data['signal_all_raw'], self.prev_signal_all_raw, 19)
        else:
            self.signal_all_upd = int(data.get('signal_all', 0)) - int(self.prev_signal_all)

        if 'idler_all_raw' in data:
            self.idler_all_upd = modular_delta(data['idler_all_raw'], self.prev_idler_all_raw, 19)
        else:
            self.idler_all_upd = int(data.get('idler_all', 0)) - int(self.prev_idler_all)

        if 'coin_all_raw' in data:
            self.coin_all_upd = modular_delta(data['coin_all_raw'], self.prev_coin_all_raw, 15)
        else:
            self.coin_all_upd = int(data.get('coin_all', 0)) - int(self.prev_coin_all)

        if 'write_count_raw' in data:
            self.write_count_upd = modular_delta(data['write_count_raw'], self.prev_write_count_raw, 19)
        else:
            self.write_count_upd = int(data.get('write_count', 0)) - int(self.prev_write_count)

        if 'read_count_raw' in data:
            self.read_count_upd = modular_delta(data['read_count_raw'], self.prev_read_count_raw, 19)
        else:
            self.read_count_upd = int(data.get('read_count', 0)) - int(self.prev_read_count)
        
        # Update previous values
        self.prev_signal_bins = data['signal_bins'].copy()
        self.prev_idler_bins = data['idler_bins'].copy()
        self.prev_coin_bins = data['coin_bins'].copy()
        # update previous absolute (corrected) scalars
        self.prev_signal_all = data.get('signal_all', self.prev_signal_all)
        self.prev_idler_all = data.get('idler_all', self.prev_idler_all)
        self.prev_coin_all = data.get('coin_all', self.prev_coin_all)
        self.prev_write_count = data.get('write_count', self.prev_write_count)
        self.prev_read_count = data.get('read_count', self.prev_read_count)
        # update previous raw scalars for modular delta
        if 'signal_all_raw' in data:
            self.prev_signal_all_raw = int(data.get('signal_all_raw'))
        if 'idler_all_raw' in data:
            self.prev_idler_all_raw = int(data.get('idler_all_raw'))
        if 'coin_all_raw' in data:
            self.prev_coin_all_raw = int(data.get('coin_all_raw'))
        if 'write_count_raw' in data:
            self.prev_write_count_raw = int(data.get('write_count_raw'))
        if 'read_count_raw' in data:
            self.prev_read_count_raw = int(data.get('read_count_raw'))
        
        # Update absolute values
        self.signal_bins = data['signal_bins']
        self.idler_bins = data['idler_bins']
        self.coin_bins = data['coin_bins']
        self.write_count = data['write_count']
        self.read_count = data['read_count']
        self.signal_all = data['signal_all']
        self.idler_all = data['idler_all']
        self.coin_all = data['coin_all']
        
        # Calculate increments if baseline exists
        if self.baseline is not None:
            self.signal_increment = data['signal_bins'] - self.baseline['signal_bins']
            self.idler_increment = data['idler_bins'] - self.baseline['idler_bins']
            self.coin_increment = data['coin_bins'] - self.baseline['coin_bins']
            # histogram baseline-corrected
            try:
                self.signal_hist = data.get('signal_hist', np.zeros(10, dtype=int)) - self.baseline.get('signal_hist', np.zeros(10, dtype=int))
                self.idler_hist = data.get('idler_hist', np.zeros(10, dtype=int)) - self.baseline.get('idler_hist', np.zeros(10, dtype=int))
            except Exception:
                self.signal_hist = data.get('signal_hist', np.zeros(10, dtype=int))
                self.idler_hist = data.get('idler_hist', np.zeros(10, dtype=int))
        else:
            # If no baseline, increments are just current values
            self.signal_increment = data['signal_bins']
            self.idler_increment = data['idler_bins']
            self.coin_increment = data['coin_bins']
            # histogram raw values
            self.signal_hist = data.get('signal_hist', np.zeros(10, dtype=int))
            self.idler_hist = data.get('idler_hist', np.zeros(10, dtype=int))
        
        # Update time series plots
        current_time = time.time()
        totals_data = {
            'signal_all': data['signal_all'],
            'idler_all': data['idler_all'],
            'coin_all': data['coin_all']
        }
        
        sums_data = {
            'signal_sum': np.sum(data['signal_bins']),
            'idler_sum': np.sum(data['idler_bins']),
            'coin_sum': np.sum(data['coin_bins'])
        }
        
        write_read_data = {
            'write_count': data['write_count'],
            'read_count': data['read_count']
        }
        sums_delta_data = {
            'signal_sum_delta': np.sum(self.signal_increment_upd),
            'idler_sum_delta': np.sum(self.idler_increment_upd), 
            'coin_sum_delta': np.sum(self.coin_increment_upd)
        }

        calc1 = self.coin_all*self.write_count*CELL_COUNT/(self.signal_all*self.idler_all+0.00001)
        calc2 = self.coin_all/ (self.signal_all+0.00001)
        calc3 = (self.coin_all_upd*self.write_count_upd *CELL_COUNT)/ (self.signal_all_upd* self.idler_all_upd+0.00001)
        calc4 = self.coin_all_upd/(self.signal_all_upd+0.00001)

        cal1_cal3_data = {
            'cal1': calc1,
            'cal3': calc3
        }

        cal2_cal4_data = {
            'cal2': calc2, 
            'cal4': calc4
        }

        # Compute per-second rates for signal_all and idler_all.
        # Skip plotting on the very first update after start/clear (user requested)
        if self.prev_update_time is None:
            # initialize prev_update_time but do not draw plots for the first second
            self.prev_update_time = current_time
            should_draw = False
            dt = 1.0
        else:
            dt = max(1e-6, current_time - self.prev_update_time)
            should_draw = True

        signal_rate = float(self.signal_all_upd) / dt
        idler_rate = float(self.idler_all_upd) / dt
        coin_delta = np.sum(self.coin_increment_upd)

        # Compute average excitation per write per bin (safe divide)
        # Use write_count_upd (the number of new writes) to avoid dividing
        # by zero when write_count is nonzero but no new writes occurred.
        if (self.write_count_upd * CELL_COUNT) > 0:
            avg_excitation = signal_rate / (self.write_count_upd * float(CELL_COUNT))
        else:
            avg_excitation = 0.0

        # --- Custom excitation calculation (user-defined) ---
        # default to existing avg_excitation unless user provided a formula
        custom_exc = float(avg_excitation)
        if hasattr(self, 'exc_formula_text') and self.exc_formula_text:
            # prepare locals available to user formula
            local_vars = {
                'np': np,
                'signal_increment': self.signal_increment,
                'idler_increment': self.idler_increment,
                'coin_increment': self.coin_increment,
                'signal_all_upd': self.signal_all_upd,
                'idler_all_upd': self.idler_all_upd,
                'coin_all_upd': self.coin_all_upd,
                'write_count': self.write_count,
                'write_count_upd': self.write_count_upd,
                'signal_rate': signal_rate,
                'idler_rate': idler_rate,
                'coin_delta': coin_delta,
                'avg_excitation': avg_excitation,
                'CELL_COUNT': CELL_COUNT
            }
            try:
                # evaluate under restricted globals (only numpy)
                custom_val = eval(self.exc_formula_text, {'np': np, 'CELL_COUNT': CELL_COUNT}, local_vars)
                custom_exc = float(custom_val)
            except Exception as e:
                try:
                    self.text_edit.append(f"Exc formula error: {e}")
                except Exception:
                    print("Exc formula error:", e)
                custom_exc = float(avg_excitation)

        # store latest computed custom excitation for UI and plotting
        self.custom_excitation = float(custom_exc)

        # Update first plot with rates (counts per second), coin delta and avg_excitation
        if should_draw:
            self.plot_sums.update_plot(current_time, {
                'signal_rate': signal_rate,
                'idler_rate': idler_rate,
                'coin_sum_delta': coin_delta
            })
            # update separate avg_excitation plot
            try:
                # plot the user-defined/custom excitation value in the avg_excitation plot
                # plot normalized signal_all per bin: signal_all / (write_count * CELL_COUNT)
                try:
                    signal_all_norm = float(self.signal_all) / (float(self.write_count) * float(CELL_COUNT) + 1e-9)
                except Exception:
                    signal_all_norm = 0.0
                self.plot_avg_excitation.update_plot(current_time, {'avg_excitation': self.custom_excitation, 'signal_all': signal_all_norm})
            except Exception:
                # defensive: ignore plotting errors to avoid breaking read loop
                pass

        # --- Update histogram increments and bar plots ---
        # compute per-update hist increments (since prev hist)
        try:
            self.signal_hist_upd = self.signal_hist - self.prev_signal_hist
            self.idler_hist_upd = self.idler_hist - self.prev_idler_hist
        except Exception:
            self.signal_hist_upd = self.signal_hist.copy()
            self.idler_hist_upd = self.idler_hist.copy()

        # update prev hist for next iteration
        try:
            self.prev_signal_hist = np.asarray(self.signal_hist).copy()
            self.prev_idler_hist = np.asarray(self.idler_hist).copy()
        except Exception:
            pass

        # Draw histograms every second (i.e., when should_draw True)
        if should_draw:
            # displayed arrays are already baseline-corrected in self.signal_hist/self.idler_hist
            labels = [str(i) for i in range(1, 11)]
            try:
                try:
                    self.hist_plot_signal.update_hist(current_time, np.asarray(self.signal_hist, dtype=float), labels=labels, color='tab:purple')
                except Exception as e:
                    # log exception for debugging
                    try:
                        self.text_edit.append(f"hist_plot_signal error: {e}")
                    except Exception:
                        print("hist_plot_signal error:", e)
                try:
                    self.hist_plot_idler.update_hist(current_time, np.asarray(self.idler_hist, dtype=float), labels=labels, color='tab:olive')
                except Exception as e:
                    try:
                        self.text_edit.append(f"hist_plot_idler error: {e}")
                    except Exception:
                        print("hist_plot_idler error:", e)
            except Exception as e:
                try:
                    self.text_edit.append(f"hist plotting outer error: {e}")
                except Exception:
                    print("hist plotting outer error:", e)

        # ------------------ EXTRA FORMULAS REGION START ------------------
        # Compute two extra metrics for the new time-series plots.
        # Use the same available variables as before: self.signal_increment,
        # self.idler_increment, self.coin_increment, self.signal_all_upd,
        # self.idler_all_upd, self.coin_all_upd, self.write_count_upd,
        # self.write_count, etc.
        # Replace the example formulas below with your desired expressions.
        try:
            # Example placeholder formulas (safe divides). Change as needed.
            real_retrivel = 100*((float(np.sum(self.coin_increment))/ float(np.sum(self.signal_increment)+1e-12)) -(float(np.sum(self.idler_increment))/(float(self.write_count)*CELL_COUNT+1e-6)))
            # Compute factor_g2 either from a user-provided multiplier K * g2
            # or fall back to the original formula. Prefer K*g2 when enabled.
            if getattr(self, 'chk_use_k', None) is not None and self.chk_use_k.isChecked():
                # use local calc1 (g2) computed earlier in update_from_serial
                try:
                    factor_g2 = float(self.spin_factor_k.value()) * float(calc1)
                except Exception:
                    factor_g2 = 0.0
            else:
                factor_g2 = CELL_COUNT*2* (float(np.sum(self.coin_increment)) * float(self.write_count)) /(float(np.sum(self.idler_increment))*float(np.sum(self.signal_increment))+1e-6)
            # store for UI display
            self.real_retrivel = float(real_retrivel)
            self.factor_g2 = float(factor_g2)
        except Exception:
            real_retrivel = 0.0
            factor_g2 = 0.0
            self.real_retrivel = 0.0
            self.factor_g2 = 0.0
        # ------------------- EXTRA FORMULAS REGION END -------------------

        # Update extra plots (only draw after first interval to have sensible dt)
        if should_draw:
            try:
                self.plot_extra1.update_plot(current_time, {'real_retrivel': float(real_retrivel)})
            except Exception:
                pass
            try:
                self.plot_extra2.update_plot(current_time, {'factor_g2': float(factor_g2)})
            except Exception:
                pass

        # store latest avg value for UI text
        self.avg_excitation = avg_excitation

        # Remember current time for next rate calculation
        self.prev_update_time = current_time
        # Skip drawing the other per-second plots on the first update after start/clear
        if should_draw:
            self.plot_cal1_cal3.update_plot(current_time, cal1_cal3_data)
            self.plot_cal2_cal4.update_plot(current_time, cal2_cal4_data)
                
        
        #self.plot_totals.update_plot(current_time, totals_data)
        
        #self.plot_write_read.update_plot(current_time, write_read_data)
        
        # Compute and store calc heatmaps (placeholders) from current increments
        try:
            self.calc_heat1, self.calc_heat2, self.calc_heat3 = self.compute_calc_heats()
        except Exception:
            # fallback to zero arrays on error
            self.calc_heat1 = np.zeros(CELL_COUNT, dtype=int)
            self.calc_heat2 = np.zeros(CELL_COUNT, dtype=int)
            self.calc_heat3 = np.zeros(CELL_COUNT, dtype=int)

        self.update_ui()

    def generate_virtual_data(self):
        # Generate virtual data with overflow simulation
        signal_bins = np.random.randint(0, 2048, CELL_COUNT)
        idler_bins = np.random.randint(0, 2048, CELL_COUNT)
        coin_bins = np.random.randint(0, 128, CELL_COUNT)
        
        # Simulate occasional overflow
        if np.random.random() < 0.1:
            signal_bins = np.random.randint(0, 500, CELL_COUNT)
            idler_bins = np.random.randint(0, 500, CELL_COUNT)
            coin_bins = np.random.randint(0, 50, CELL_COUNT)
        
        # build scalar values and simulated raw (masked by device bit-width)
        sig_all = int(np.random.randint(0, 500000))
        idl_all = int(np.random.randint(0, 500000))
        coin_all = int(np.random.randint(0, 50000))
        write_cnt = int(time.time() * 100) % 10000
        read_cnt = (int(time.time() * 100) % 10000) + 100

        # mask to device bit widths to simulate raw counters
        sig_raw = sig_all & ((1 << 19) - 1)
        idl_raw = idl_all & ((1 << 19) - 1)
        coin_raw = coin_all & ((1 << 15) - 1)
        write_raw = write_cnt & ((1 << 19) - 1)
        read_raw = read_cnt & ((1 << 19) - 1)

        virtual_data = {
            'signal_bins': signal_bins,
            'idler_bins': idler_bins,
            'coin_bins': coin_bins,
            'signal_bins_raw': signal_bins,
            'idler_bins_raw': idler_bins, 
            'coin_bins_raw': coin_bins,
            'signal_hist': np.random.randint(0, 100, 10),
            'idler_hist': np.random.randint(0, 100, 10),
            'write_count': write_cnt,
            'read_count': read_cnt,
            'signal_all': sig_all,
            'idler_all': idl_all,
            'coin_all': coin_all,
            'signal_all_raw': sig_raw,
            'idler_all_raw': idl_raw,
            'coin_all_raw': coin_raw,
            'write_count_raw': write_raw,
            'read_count_raw': read_raw,
            'timestamp': time.time()
        }
        
        self.update_from_serial(virtual_data)

    def compute_calc_heats(self):
        """Compute three CELL_COUNT-bin arrays (calc_heat1/2/3) from current increment
        data. This is a placeholder where you should implement the actual
        per-segment formulas using the following available arrays/values:

        - self.coin_increment  (np.array length CELL_COUNT)
        - self.signal_increment (np.array length CELL_COUNT)
        - self.idler_increment  (np.array length CELL_COUNT)
        - self.write_count_upd  (int)

        Please put your formulas between the markers below (FORMULA_REGION_START/END).
        After implementing, return three numpy arrays with dtype=int (length CELL_COUNT).
        """
        # default zero arrays (use float so per-segment formulas can return floats)
        calc1 = np.zeros(CELL_COUNT, dtype=float)
        calc2 = np.zeros(CELL_COUNT, dtype=float)
        calc3 = np.zeros(CELL_COUNT, dtype=float)

        # ----------------------- FORMULA_REGION_START -----------------------
        # Insert your per-segment calculation here. Changed to use the
        # UI-displayed increments (self.signal_increment / self.idler_increment /
        # self.coin_increment) and the current write count (self.write_count).
        # Also ensure results are numeric (floats) so heatmap displays correctly.
        for i in range(CELL_COUNT):
            # Use UI-displayed increments as floats
            sig = float(self.signal_increment[i])
            idl = float(self.idler_increment[i])
            con = float(self.coin_increment[i])

            # use the displayed write_count (not the per-update delta)
            wc = float(self.write_count)

            # calc1: show the per-segment displayed signal count (float)
            calc1[i] = sig/(wc+0.00001)*1000
            # calc2: show the per-segment displayed coin count (float)
            # changed multiplier from 100 to 1000 as requested
            calc2[i] = (con/(sig+0.0001) -idl/(wc+0.00001)) *1000

            # calc3: scaled metric con * write_count / (sig + idl) with safe divide
            calc3[i] = (con * wc) /(sig*idl+ 0.0001)
        # ------------------------ FORMULA_REGION_END ------------------------

        # Return numeric arrays (floats allowed); HeatmapTable now handles floats
        return calc1, calc2, calc3

    def update_ui(self):
        # Update heatmaps with increment data
        self.heatmap_signal.update_data(self.signal_increment)
        self.heatmap_idler.update_data(self.idler_increment)
        self.heatmap_coin.update_data(self.coin_increment)
        # Update calc heatmaps (computed from increments)
        try:
            self.heatmap_calc1.update_data(self.calc_heat1)
            self.heatmap_calc2.update_data(self.calc_heat2)
            self.heatmap_calc3.update_data(self.calc_heat3)
        except Exception:
            # if calc arrays not present yet, skip
            pass

                
        # Update sum displays - show increments relative to baseline if baseline exists
        if self.baseline is not None:
            # Show increments from baseline
            signal_all_display = self.signal_all - self.baseline['signal_all']
            idler_all_display = self.idler_all - self.baseline['idler_all']
            coin_all_display = self.coin_all - self.baseline['coin_all']
            signal_sum_display = np.sum(self.signal_bins) - np.sum(self.baseline['signal_bins'])
            idler_sum_display = np.sum(self.idler_bins) - np.sum(self.baseline['idler_bins'])
            coin_sum_display = np.sum(self.coin_bins) - np.sum(self.baseline['coin_bins'])
            write_count_display = self.write_count - self.baseline['write_count']
            read_count_display = self.read_count - self.baseline['read_count']
        else:
            # Show absolute values
            signal_all_display = self.signal_all
            idler_all_display = self.idler_all
            coin_all_display = self.coin_all
            signal_sum_display = np.sum(self.signal_bins)
            idler_sum_display = np.sum(self.idler_bins)
            coin_sum_display = np.sum(self.coin_bins)
            write_count_display = self.write_count
            read_count_display = self.read_count

        self.lbl_signal_all.setText(f"Signal All: {signal_all_display}")
        self.lbl_idler_all.setText(f"Idler All: {idler_all_display}")
        self.lbl_coin_all.setText(f"Coin All: {coin_all_display}")
        self.lbl_signal_sum.setText(f"Signal Sum: {signal_sum_display}")
        self.lbl_idler_sum.setText(f"Idler Sum: {idler_sum_display}")
        self.lbl_coin_sum.setText(f"Coin Sum: {coin_sum_display}")

        self.lbl_signal_all_upd.setText(f"Signal All Δ: {self.signal_all_upd}")
        self.lbl_idler_all_upd.setText(f"Idler All Δ: {self.idler_all_upd}")
        self.lbl_coin_all_upd.setText(f"Coin All Δ: {self.coin_all_upd}")

        # Update write/read count displays
        self.lbl_write_count.setText(f"Write Count: {write_count_display}")
        self.lbl_read_count.setText(f"Read Count: {read_count_display}")
        self.lbl_write_count_upd.setText(f"Write Count Δ: {self.write_count_upd}")
        self.lbl_read_count_upd.setText(f"Read Count Δ: {self.read_count_upd}")
                
        # Update calculation results 
        calc1 = (float(np.sum(self.coin_increment))*float(self.write_count)*CELL_COUNT )/ (float(np.sum(self.signal_increment))* float(np.sum(self.idler_increment))+0.0001)
        calc2 = float(np.sum(self.coin_increment)) / (float(np.sum(self.signal_increment)) +0.0001)
        
        calc3 = (float(np.sum(self.coin_increment_upd))*float(self.write_count_upd)*CELL_COUNT )/ (np.sum(self.signal_increment_upd)* np.sum(self.idler_increment_upd)+0.0001)
        calc4 = np.sum(self.coin_increment_upd) / (np.sum(self.signal_increment_upd) +0.0001)
        
        self.calc1 = calc1
        self.calc2 = calc2
        self.calc3 = calc3  
        self.calc4 = calc4
        
        self.lbl_calc1.setText(f"g2: {calc1:.4f}")
        self.lbl_calc2.setText(f"Retrival Rate: {calc2:.6f}")
        self.lbl_calc3.setText(f"g2_per_second: {calc3:.4f}")
        self.lbl_calc4.setText(f"Retriva Rate_per_second: {calc4:.6f}")
        # Display numeric avg_excitation alongside other calculation labels
        try:
            # show the custom excitation value if available
            val = getattr(self, 'custom_excitation', self.avg_excitation)
            self.lbl_avg_excitation.setText(f"Avg Excitation: {val:.6f}")
        except Exception:
            # if label not present, silently ignore
            pass
        # Display extra calculated values in realtime (if available)
        try:
            self.lbl_real_retrivel.setText(f"Real Retrivel: {getattr(self, 'real_retrivel', 0.0):.4f}")
        except Exception:
            pass
        try:
            self.lbl_factor_g2.setText(f"Factor g2: {getattr(self, 'factor_g2', 0.0):.4f}")
        except Exception:
            pass

    # --------------------- Data Access Methods ---------------------
    
    def get_signal_data(self):
        """Get signal bins data (absolute values)"""
        return self.signal_bins
    
    def get_idler_data(self):
        """Get idler bins data (absolute values)""" 
        return self.idler_bins
    
    def get_coin_data(self):
        """Get coin bins data (absolute values)"""
        return self.coin_bins
    
    def get_signal_increment(self):
        """Get signal bins increment data (relative to baseline)"""
        return self.signal_increment
    
    def get_idler_increment(self):
        """Get idler bins increment data (relative to baseline)"""
        return self.idler_increment
    
    def get_coin_increment(self):
        """Get coin bins increment data (relative to baseline)"""
        return self.coin_increment
    
    def get_signal_increment_upd(self):
        """Get signal bins increment since last update"""
        return self.signal_increment_upd
    
    def get_idler_increment_upd(self):
        """Get idler bins increment since last update"""
        return self.idler_increment_upd
    
    def get_coin_increment_upd(self):
        """Get coin bins increment since last update"""
        return self.coin_increment_upd
    
    def get_signal_all_upd(self):
        """Get signal_all increment since last update"""
        return self.signal_all_upd
    
    def get_idler_all_upd(self):
        """Get idler_all increment since last update"""
        return self.idler_all_upd
    
    def get_coin_all_upd(self):
        """Get coin_all increment since last update"""
        return self.coin_all_upd
    
    def get_write_count_upd(self):
        """Get write_count increment since last update"""
        return self.write_count_upd
    
    def get_read_count_upd(self):
        """Get read_count increment since last update"""
        return self.read_count_upd
    
    def get_total_counts(self):
        """Get all total count values"""
        return {
            'signal_all': self.signal_all,
            'idler_all': self.idler_all, 
            'coin_all': self.coin_all,
            'signal_sum': np.sum(self.signal_bins),
            'idler_sum': np.sum(self.idler_bins),
            'coin_sum': np.sum(self.coin_bins),
            'write_count': self.write_count,
            'read_count': self.read_count
        }
    
    def get_time_series_data(self):
        """Get the complete time series data for external use"""
        return self.data_history

    # --------------------- Save CSV ---------------------

    def save_csv(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"histogram_{timestamp}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Signal Absolute"] + list(self.signal_bins))
            writer.writerow(["Idler Absolute"] + list(self.idler_bins))
            writer.writerow(["Coin Absolute"] + list(self.coin_bins))
            writer.writerow(["Signal Increment"] + list(self.signal_increment))
            writer.writerow(["Idler Increment"] + list(self.idler_increment))
            writer.writerow(["Coin Increment"] + list(self.coin_increment))
            writer.writerow(["Signal Increment Update"] + list(self.signal_increment_upd))
            writer.writerow(["Idler Increment Update"] + list(self.idler_increment_upd))
            writer.writerow(["Coin Increment Update"] + list(self.coin_increment_upd))
            writer.writerow(["Signal All", self.signal_all])
            writer.writerow(["Idler All", self.idler_all])
            writer.writerow(["Coin All", self.coin_all])
            writer.writerow(["Signal All Update", self.signal_all_upd])
            writer.writerow(["Idler All Update", self.idler_all_upd])
            writer.writerow(["Coin All Update", self.coin_all_upd])
            writer.writerow(["Write Count", self.write_count])
            writer.writerow(["Read Count", self.read_count])
            writer.writerow(["Write Count Update", self.write_count_upd])
            writer.writerow(["Read Count Update", self.read_count_upd])
        self.text_edit.append(f" Data saved: {filename}")


# ============================================================
# Main Entry
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
