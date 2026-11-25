#!/usr/bin/env python3
"""
UI for 4-Qubit Entanglement Sequence Generation
Date: 2025-10-24 (Updated for 4-Qubit System)

Features:
- Generate button to trigger sequence generation
- Waveform preview display
- Two 3D histograms (16x16) for Re(ρ) and Im(ρ)
- Support for 4-Qubit system with 38 frequencies and 4 excitation modes
"""

import json
import os
import sys
from pathlib import Path
import numpy as np

try:
    from .digital_delay_defaults import DEFAULT_DIGITAL_DELAY_OFFSET_NS, channel_digital_delay_values, digital_delay_values
except ImportError:
    from digital_delay_defaults import DEFAULT_DIGITAL_DELAY_OFFSET_NS, channel_digital_delay_values, digital_delay_values

try:
    from .system_specs import SYSTEM_SPECS
except ImportError:
    from system_specs import SYSTEM_SPECS

_SPEC_4Q = SYSTEM_SPECS["4qubit"]


def load_generated_preview_channels(output_dir, max_rows=10000):
    """Load the newest coherent W_CH1..4 waveform set for a lightweight UI preview."""
    output_path = Path(output_dir)
    candidates = sorted(
        output_path.glob("W_CH1_*.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for w_ch1_path in candidates:
        suffix = w_ch1_path.name[len("W_CH1_"):]
        paths = {name: output_path / f"{name}_{suffix}" for name in ("W_CH1", "W_CH2", "W_CH3", "W_CH4")}
        if not all(path.exists() for path in paths.values()):
            continue
        return {
            name: np.loadtxt(path, delimiter=",", skiprows=1, max_rows=int(max_rows))
            for name, path in paths.items()
        }
    raise FileNotFoundError(f"No complete W_CH1..4 waveform set found in {output_path}")

# Try to import GUI libraries with fallback
try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QComboBox, QTextEdit, QGroupBox,
        QTableWidget, QTableWidgetItem, QSplitter, QTabWidget,
        QSizePolicy, QSpinBox, QLineEdit, QDoubleSpinBox, QCheckBox,
        QFileDialog, QMessageBox, QScrollArea
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 not available, GUI will not be functional")

# Try to import matplotlib with fallback
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    # Use the 3D toolkit (there is no mplot4d). Axes3D enables 3D bar plotting via bar3d.
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available, plotting will be disabled")

# Try to import our 4-qubit entanglement sequence generator
try:
    from .entangle_sequence_4qubit import EntangleConfig4Qubit, generate_entangle_sequence_4qubit
    ENTANGLE_AVAILABLE = True
except ImportError as e:
    try:
        sequence_dir = os.path.dirname(os.path.abspath(__file__))
        if sequence_dir not in sys.path:
            sys.path.insert(0, sequence_dir)
        from entangle_sequence_4qubit import EntangleConfig4Qubit, generate_entangle_sequence_4qubit
        ENTANGLE_AVAILABLE = True
    except ImportError as fallback_exc:
        ENTANGLE_AVAILABLE = False
        print(f"Could not import entangle_sequence_4qubit: {fallback_exc or e}")

class EntangleUI(QWidget):
    def __init__(self):
        super().__init__()
        if not PYQT_AVAILABLE:
            print("Cannot initialize UI without PyQt5")
            return

        self.setWindowTitle("4-Qubit Entangle Sequence Generator")
        self.resize(1280, 900)
        self.setMinimumSize(960, 700)

        # Data storage
        self.channels_data = None
        self.config = None

        # Re(ρ) and Im(ρ) matrices (16x16) for 4-qubit system
        self.re_rho = np.ones((16, 16))
        self.im_rho = np.ones((16, 16))

        # Re(ρ) and Im(ρ) core matrices (4x4)
        self.re_rho_4x4 = np.ones((4, 4))
        self.im_rho_4x4 = np.ones((4, 4))

        # Excitation mode for W_CH1 (now 4 modes)
        self.w_ch1_excitation_mode = 1  # Default to mode 1

        # Digital1 pulse positions for W_CH1 (microseconds 1, 2, 3, 4)
        self.w_ch1_digital1_positions = [True, False, False, False]  # Default: pulse at 1st microsecond only
        self.digital_delay_offset_ns = DEFAULT_DIGITAL_DELAY_OFFSET_NS

        # User amplitude and phase values for W_CH1 (4 values for 4-qubit system)
        self.user_amp_values = [1.0, 1.0, 1.0, 1.0]  # Default amplitude values
        self.user_phase_values = [0.0, 0.0, 0.0, 0.0]  # Default phase values (radians)
        self.w_ch1_global_amp_matrices = [np.ones((16, 4)) for _ in range(4)]

        self.setup_ui()

    def setup_ui(self):
        if not PYQT_AVAILABLE:
            return

        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        content_layout = QVBoxLayout(content_widget)

        # Top control panel
        control_layout = QGridLayout()
        control_layout.setColumnStretch(4, 1)

        # Generate button
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(self.generate_sequence)

        # Frequency settings
        freq_group = QGroupBox("Frequency Settings (4-Qubit)")
        freq_layout = QGridLayout()

        freq_layout.addWidget(QLabel("Start (MHz):"), 0, 0)
        self.spin_freq_start = QDoubleSpinBox()
        self.spin_freq_start.setRange(50, 200)
        self.spin_freq_start.setSingleStep(0.5)
        self.spin_freq_start.setValue(_SPEC_4Q["freq_start_mhz"])
        freq_layout.addWidget(self.spin_freq_start, 0, 1)

        freq_layout.addWidget(QLabel("End (MHz):"), 1, 0)
        self.spin_freq_end = QDoubleSpinBox()
        self.spin_freq_end.setRange(50, 200)
        self.spin_freq_end.setSingleStep(0.5)
        self.spin_freq_end.setValue(_SPEC_4Q["freq_end_mhz"])
        freq_layout.addWidget(self.spin_freq_end, 1, 1)

        freq_layout.addWidget(QLabel("Step (MHz):"), 2, 0)
        self.spin_freq_step = QDoubleSpinBox()
        self.spin_freq_step.setRange(0.1, 10)
        self.spin_freq_step.setSingleStep(2.5)
        self.spin_freq_step.setValue(2.5)
        freq_layout.addWidget(self.spin_freq_step, 2, 1)
        freq_layout.addWidget(QLabel("(Fixed at 2.5 MHz for 4-qubit)"), 3, 1, 1, 4)

        freq_group.setLayout(freq_layout)

        # W_CH1 Excitation Mode settings (4 modes)
        mode_group = QGroupBox("W_CH1 Excitation Mode")
        mode_layout = QVBoxLayout()

        mode_layout.addWidget(QLabel("Mode Selection:"))
        self.combo_excitation_mode = QComboBox()
        self.combo_excitation_mode.addItem("Mode 1: Four simultaneous (4 µs)")
        self.combo_excitation_mode.addItem("Mode 2: Three simultaneous + one solo")
        self.combo_excitation_mode.addItem("Mode 3: Two simultaneous + two sequential")
        self.combo_excitation_mode.addItem("Mode 4: Purely sequential")
        self.combo_excitation_mode.setCurrentIndex(0)  # Default to mode 1
        self.combo_excitation_mode.currentIndexChanged.connect(self.on_excitation_mode_changed)
        mode_layout.addWidget(self.combo_excitation_mode)

        # Mode description
        self.lbl_mode_description = QLabel()
        self.lbl_mode_description.setWordWrap(True)
        self.lbl_mode_description.setText("")
        self.lbl_mode_description.setStyleSheet("color: #666; font-size: 10px;")
        mode_layout.addWidget(self.lbl_mode_description)

        mode_group.setLayout(mode_layout)

        # W_CH1 Amplitude and Phase settings
        amp_phase_group = QGroupBox("W_CH1 Amplitude & Phase")
        amp_phase_layout = QGridLayout()

        # Amplitude inputs (4 values for 4-qubit system)
        amp_phase_layout.addWidget(QLabel("Amp:"), 0, 0)
        self.spin_amp1 = QDoubleSpinBox()
        self.spin_amp1.setRange(0.0, 5.0)
        self.spin_amp1.setSingleStep(0.1)
        self.spin_amp1.setValue(1.0)
        self.spin_amp1.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(self.spin_amp1, 0, 1)

        self.spin_amp2 = QDoubleSpinBox()
        self.spin_amp2.setRange(0.0, 5.0)
        self.spin_amp2.setSingleStep(0.1)
        self.spin_amp2.setValue(1.0)
        self.spin_amp2.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(self.spin_amp2, 0, 2)

        self.spin_amp3 = QDoubleSpinBox()
        self.spin_amp3.setRange(0.0, 5.0)
        self.spin_amp3.setSingleStep(0.1)
        self.spin_amp3.setValue(1.0)
        self.spin_amp3.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(self.spin_amp3, 0, 3)

        self.spin_amp4 = QDoubleSpinBox()
        self.spin_amp4.setRange(0.0, 5.0)
        self.spin_amp4.setSingleStep(0.1)
        self.spin_amp4.setValue(1.0)
        self.spin_amp4.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(self.spin_amp4, 0, 4)

        # Phase inputs (in degrees for user convenience, converted to radians internally)
        amp_phase_layout.addWidget(QLabel("Phase (°):"), 1, 0)
        self.spin_phase1 = QDoubleSpinBox()
        self.spin_phase1.setRange(-360.0, 360.0)
        self.spin_phase1.setSingleStep(1.0)
        self.spin_phase1.setValue(0.0)
        self.spin_phase1.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(self.spin_phase1, 1, 1)

        self.spin_phase2 = QDoubleSpinBox()
        self.spin_phase2.setRange(-360.0, 360.0)
        self.spin_phase2.setSingleStep(1.0)
        self.spin_phase2.setValue(0.0)
        self.spin_phase2.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(self.spin_phase2, 1, 2)

        self.spin_phase3 = QDoubleSpinBox()
        self.spin_phase3.setRange(-360.0, 360.0)
        self.spin_phase3.setSingleStep(1.0)
        self.spin_phase3.setValue(0.0)
        self.spin_phase3.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(self.spin_phase3, 1, 3)

        self.spin_phase4 = QDoubleSpinBox()
        self.spin_phase4.setRange(-360.0, 360.0)
        self.spin_phase4.setSingleStep(1.0)
        self.spin_phase4.setValue(0.0)
        self.spin_phase4.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(self.spin_phase4, 1, 4)

        amp_phase_group.setLayout(amp_phase_layout)

        matrix_group = QGroupBox("W_CH1 Global Amplitude Matrices")
        matrix_layout = QGridLayout()
        self.table_w_ch1_global_amp = []
        for idx in range(4):
            label_row = (idx // 2) * 2
            table_row = label_row + 1
            table_col = idx % 2
            matrix_layout.addWidget(QLabel(f"Cycle {idx + 1}"), label_row, table_col)
            table = QTableWidget(16, 4)
            table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            table.setMaximumHeight(240)
            for row in range(16):
                for col in range(4):
                    table.setItem(row, col, QTableWidgetItem("1.0"))
            self.table_w_ch1_global_amp.append(table)
            matrix_layout.addWidget(table, table_row, table_col)
        self.btn_load_signal_calibration = QPushButton("Load Signal Calibration JSON...")
        self.btn_load_signal_calibration.clicked.connect(self.browse_signal_calibration_json)
        matrix_layout.addWidget(self.btn_load_signal_calibration, 4, 0, 1, 2)
        matrix_group.setLayout(matrix_layout)
        self.load_persisted_w_ch1_global_amp_matrices()

        # Digital1 pulse position checkboxes (like 3-qubit version)
        mode_layout.addWidget(QLabel("Digital1 Pulses:"))
        self.chk_digital1_pos1 = QCheckBox("1st µs")
        self.chk_digital1_pos1.setChecked(True)
        self.chk_digital1_pos1.stateChanged.connect(self.on_digital1_position_changed)
        mode_layout.addWidget(self.chk_digital1_pos1)

        self.chk_digital1_pos2 = QCheckBox("2nd µs")
        self.chk_digital1_pos2.setChecked(False)
        self.chk_digital1_pos2.stateChanged.connect(self.on_digital1_position_changed)
        mode_layout.addWidget(self.chk_digital1_pos2)

        self.chk_digital1_pos3 = QCheckBox("3rd µs")
        self.chk_digital1_pos3.setChecked(False)
        self.chk_digital1_pos3.stateChanged.connect(self.on_digital1_position_changed)
        mode_layout.addWidget(self.chk_digital1_pos3)

        self.chk_digital1_pos4 = QCheckBox("4th µs")
        self.chk_digital1_pos4.setChecked(False)
        self.chk_digital1_pos4.stateChanged.connect(self.on_digital1_position_changed)
        mode_layout.addWidget(self.chk_digital1_pos4)

        # Initialize checkbox states based on default mode
        self.update_digital1_checkboxes()

        # W_CH1 and R_CH1 Digital1 Settings
        digital_settings_group = QGroupBox("Digital1 Settings (ns)")
        digital_settings_layout = QHBoxLayout()
        default_digital_delays = digital_delay_values(self.digital_delay_offset_ns)

        # W_CH1 Digital1 Width
        digital_settings_layout.addWidget(QLabel("W_CH1 Width:"))
        self.spin_w_ch1_digital_width = QSpinBox()
        self.spin_w_ch1_digital_width.setRange(0, 10000)
        self.spin_w_ch1_digital_width.setSingleStep(10)
        self.spin_w_ch1_digital_width.setValue(120)  # Default from config
        digital_settings_layout.addWidget(self.spin_w_ch1_digital_width)

        # W_CH1 Digital1 Delay
        digital_settings_layout.addWidget(QLabel("W_CH1 Delay:"))
        self.spin_w_ch1_digital_delay = QSpinBox()
        self.spin_w_ch1_digital_delay.setRange(0, 100000)
        self.spin_w_ch1_digital_delay.setSingleStep(100)
        self.spin_w_ch1_digital_delay.setValue(default_digital_delays["W_CH1"])
        digital_settings_layout.addWidget(self.spin_w_ch1_digital_delay)

        # R_CH1 Digital1 Width
        digital_settings_layout.addWidget(QLabel("R_CH1 Width:"))
        self.spin_r_ch1_digital_width = QSpinBox()
        self.spin_r_ch1_digital_width.setRange(0, 10000)
        self.spin_r_ch1_digital_width.setSingleStep(10)
        self.spin_r_ch1_digital_width.setValue(400)  # Default from config
        digital_settings_layout.addWidget(self.spin_r_ch1_digital_width)

        # R_CH1 Digital1 Delay
        digital_settings_layout.addWidget(QLabel("R_CH1 Delay:"))
        self.spin_r_ch1_digital_delay = QSpinBox()
        self.spin_r_ch1_digital_delay.setRange(0, 100000)
        self.spin_r_ch1_digital_delay.setSingleStep(100)
        self.spin_r_ch1_digital_delay.setValue(default_digital_delays["R_CH1"])
        digital_settings_layout.addWidget(self.spin_r_ch1_digital_delay)

        digital_settings_group.setLayout(digital_settings_layout)

        # Output directory
        output_layout = QHBoxLayout()
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_directory)
        self.txt_output_dir = QLineEdit(os.path.join("AWG_Sequence", "entangle_sequences_4qubit"))

        output_layout.addWidget(QLabel("Output Dir:"))
        output_layout.addWidget(self.txt_output_dir)
        output_layout.addWidget(self.btn_browse)

        control_layout.addWidget(self.btn_generate, 0, 0)
        control_layout.addWidget(freq_group, 0, 1)
        control_layout.addWidget(mode_group, 0, 2)
        control_layout.addWidget(amp_phase_group, 0, 3, 1, 2)
        control_layout.addWidget(digital_settings_group, 1, 0, 1, 3)
        control_layout.addWidget(QLabel("Output Dir:"), 1, 3)
        control_layout.addWidget(self.txt_output_dir, 1, 4)
        control_layout.addWidget(self.btn_browse, 1, 5)

        # Status area
        self.txt_status = QTextEdit()
        self.txt_status.setMaximumHeight(100)
        self.txt_status.setReadOnly(True)

        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Waveform preview
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        waveform_group = QGroupBox("Waveform Preview")
        waveform_layout = QVBoxLayout()

        # Create matplotlib figure for waveform preview (if available)
        if MATPLOTLIB_AVAILABLE:
            self.waveform_fig = Figure(figsize=(6, 3.8))
            self.waveform_canvas = FigureCanvas(self.waveform_fig)
            self.waveform_canvas.setMinimumSize(360, 240)
            waveform_layout.addWidget(self.waveform_canvas)
        else:
            waveform_layout.addWidget(QLabel("Matplotlib not available for plotting"))

        waveform_group.setLayout(waveform_layout)
        left_layout.addWidget(waveform_group)

        # Right panel - 3D histograms
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Full 16x16 3D histogram group
        hist_group = QGroupBox("3D Histograms (16x16)")
        hist_layout = QHBoxLayout()

        # Re(ρ) 3D histogram
        re_rho_group = QGroupBox("Re(ρ)")
        re_rho_layout = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.re_rho_fig = Figure(figsize=(3.4, 2.6))
            self.re_rho_canvas = FigureCanvas(self.re_rho_fig)
            self.re_rho_canvas.setMinimumSize(280, 220)
            re_rho_layout.addWidget(self.re_rho_canvas)
        else:
            re_rho_layout.addWidget(QLabel("Matplotlib not available"))
        re_rho_group.setLayout(re_rho_layout)

        # Im(ρ) 3D histogram
        im_rho_group = QGroupBox("Im(ρ)")
        im_rho_layout = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.im_rho_fig = Figure(figsize=(3.4, 2.6))
            self.im_rho_canvas = FigureCanvas(self.im_rho_fig)
            self.im_rho_canvas.setMinimumSize(280, 220)
            im_rho_layout.addWidget(self.im_rho_canvas)
        else:
            im_rho_layout.addWidget(QLabel("Matplotlib not available"))
        im_rho_group.setLayout(im_rho_layout)

        hist_layout.addWidget(re_rho_group)
        hist_layout.addWidget(im_rho_group)
        hist_group.setLayout(hist_layout)

        right_layout.addWidget(hist_group)

        # 4x4 core 3D histogram group
        core_hist_group = QGroupBox("3D Histograms (4x4 Core)")
        core_hist_layout = QHBoxLayout()

        # Re(ρ) 4x4 core 3D histogram
        re_rho_4x4_group = QGroupBox("Re(ρ) Core")
        re_rho_4x4_layout = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.re_rho_4x4_fig = Figure(figsize=(3.4, 2.6))
            self.re_rho_4x4_canvas = FigureCanvas(self.re_rho_4x4_fig)
            self.re_rho_4x4_canvas.setMinimumSize(280, 220)
            re_rho_4x4_layout.addWidget(self.re_rho_4x4_canvas)
        else:
            re_rho_4x4_layout.addWidget(QLabel("Matplotlib not available"))
        re_rho_4x4_group.setLayout(re_rho_4x4_layout)

        # Im(ρ) 4x4 core 3D histogram
        im_rho_4x4_group = QGroupBox("Im(ρ) Core")
        im_rho_4x4_layout = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.im_rho_4x4_fig = Figure(figsize=(3.4, 2.6))
            self.im_rho_4x4_canvas = FigureCanvas(self.im_rho_4x4_fig)
            self.im_rho_4x4_canvas.setMinimumSize(280, 220)
            im_rho_4x4_layout.addWidget(self.im_rho_4x4_canvas)
        else:
            im_rho_4x4_layout.addWidget(QLabel("Matplotlib not available"))
        im_rho_4x4_group.setLayout(im_rho_4x4_layout)

        core_hist_layout.addWidget(re_rho_4x4_group)
        core_hist_layout.addWidget(im_rho_4x4_group)
        core_hist_group.setLayout(core_hist_layout)

        right_layout.addWidget(core_hist_group)

        # Right panel layout
        right_panel.setLayout(right_layout)

        # Add panels to splitter
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([560, 640])

        # Add all to main layout
        content_layout.addLayout(control_layout)
        content_layout.addWidget(self.txt_status)
        content_layout.addWidget(matrix_group)
        content_layout.addWidget(content_splitter)

        # Initial plot update
        if MATPLOTLIB_AVAILABLE:
            self.update_plots()

    def browse_directory(self):
        if not PYQT_AVAILABLE:
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.txt_output_dir.setText(directory)

    def set_digital_delay_offset(self, offset_ns):
        self.digital_delay_offset_ns = int(offset_ns)
        delays = digital_delay_values(self.digital_delay_offset_ns)
        if hasattr(self, "spin_w_ch1_digital_delay"):
            self.spin_w_ch1_digital_delay.setValue(delays["W_CH1"])
        if hasattr(self, "spin_r_ch1_digital_delay"):
            self.spin_r_ch1_digital_delay.setValue(delays["R_CH1"])

    def on_excitation_mode_changed(self, index):
        """Handle excitation mode selection change"""
        self.w_ch1_excitation_mode = index + 1
        self.update_mode_description()
        self.update_digital1_checkboxes()

    def on_amp_phase_changed(self):
        """Handle amplitude and phase value changes"""
        import math
        self.user_amp_values = [
            self.spin_amp1.value(),
            self.spin_amp2.value(),
            self.spin_amp3.value(),
            self.spin_amp4.value()
        ]
        self.user_phase_values = [
            math.radians(self.spin_phase1.value()),
            math.radians(self.spin_phase2.value()),
            math.radians(self.spin_phase3.value()),
            math.radians(self.spin_phase4.value())
        ]

    def set_target_state(self, amplitudes, phases_deg, excitation_mode):
        """Atomically synchronize an automation target with the visible 4Q UI."""
        if len(amplitudes) != 4 or len(phases_deg) != 4:
            raise ValueError("4Q target state requires four amplitudes and phases")
        for idx, value in enumerate(amplitudes, start=1):
            getattr(self, f"spin_amp{idx}").setValue(float(value))
        for idx, value in enumerate(phases_deg, start=1):
            getattr(self, f"spin_phase{idx}").setValue(float(value))
        self.combo_excitation_mode.setCurrentIndex(int(excitation_mode) - 1)
        self.w_ch1_excitation_mode = int(excitation_mode)
        self.on_amp_phase_changed()

    def refresh_from_generated_output(self, output_dir, density_json_path=None):
        """Refresh preview and density histograms from the sequence actually generated in back end."""
        script_dir = Path(__file__).resolve().parent
        density_path = Path(density_json_path) if density_json_path else script_dir / "matrix" / "density_matrix_4qubit.json"
        payload = json.loads(density_path.read_text(encoding="utf-8"))
        self.re_rho = np.asarray(payload["re_rho"], dtype=float)
        self.im_rho = np.asarray(payload["im_rho"], dtype=float)
        if "re_rho_4x4" in payload and "im_rho_4x4" in payload:
            self.re_rho_4x4 = np.asarray(payload["re_rho_4x4"], dtype=float)
            self.im_rho_4x4 = np.asarray(payload["im_rho_4x4"], dtype=float)
        else:
            core_indices = _SPEC_4Q["core_indices"]
            self.re_rho_4x4 = self.re_rho[np.ix_(core_indices, core_indices)]
            self.im_rho_4x4 = self.im_rho[np.ix_(core_indices, core_indices)]
        self.channels_data = load_generated_preview_channels(output_dir)
        self.txt_output_dir.setText(str(output_dir))
        if MATPLOTLIB_AVAILABLE:
            self.update_plots()
        self.txt_status.append(f"Refreshed preview/histograms from generated output: {output_dir}")

    def set_w_ch1_global_amp_matrices(self, matrices):
        values = np.asarray(matrices, dtype=float)
        if values.shape != (4, 16, 4) or not np.all(np.isfinite(values)):
            raise ValueError("4Q signal calibration must have shape [4,16,4] with finite values")
        for cycle, table in enumerate(self.table_w_ch1_global_amp):
            for row in range(16):
                for col in range(4):
                    table.setItem(row, col, QTableWidgetItem(f"{values[cycle, row, col]:.6f}"))
        self.w_ch1_global_amp_matrices = [matrix.copy() for matrix in values]

    def load_persisted_w_ch1_global_amp_matrices(self):
        matrix_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrix")
        matrices = []
        for cycle in range(1, 5):
            path = os.path.join(matrix_dir, f"W_CH1_global_amp_{cycle}_4qubit.txt")
            if os.path.exists(path):
                matrix = np.loadtxt(path, dtype=float)
                if matrix.shape != (16, 4):
                    raise ValueError(f"{path} must have shape 16x4, got {matrix.shape}")
            else:
                matrix = np.ones((16, 4), dtype=float)
            matrices.append(matrix)
        self.set_w_ch1_global_amp_matrices(matrices)

    def load_signal_calibration_json(self, path):
        payload = json.loads(open(path, "r", encoding="utf-8").read())
        matrices = payload.get("result", {}).get("updated_global_amp")
        if matrices is None:
            matrices = payload.get("updated_global_amp")
        self.set_w_ch1_global_amp_matrices(matrices)
        self.txt_status.append(f"Loaded signal calibration: {path}")

    def browse_signal_calibration_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load 4Q Signal Calibration",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrix", "calibration_reports"),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            self.load_signal_calibration_json(path)
        except Exception as exc:
            QMessageBox.critical(self, "Signal calibration", str(exc))

    def on_digital1_position_changed(self, state):
        """Handle digital1 position checkbox changes"""
        # Update W_CH1 digital1 positions based on checkbox states
        self.w_ch1_digital1_positions = [
            self.chk_digital1_pos1.isChecked(),
            self.chk_digital1_pos2.isChecked(),
            self.chk_digital1_pos3.isChecked(),
            self.chk_digital1_pos4.isChecked()
        ]

    def read_w_ch1_global_amp_matrices(self):
        matrices = []
        for table in self.table_w_ch1_global_amp:
            matrix = np.ones((16, 4))
            for row in range(16):
                for col in range(4):
                    item = table.item(row, col)
                    text = item.text().strip() if item and item.text() else "1.0"
                    try:
                        matrix[row, col] = float(text)
                    except ValueError:
                        matrix[row, col] = 1.0
            matrices.append(matrix)
        self.w_ch1_global_amp_matrices = matrices
        return matrices

    def update_mode_description(self):
        """Update mode description based on selected mode"""
        mode = self.w_ch1_excitation_mode
        descriptions = {
            1: "Mode 1: Four simultaneous frequencies (f1,f2,f3,f4) for 4 µs, digital1 only at 1st µs",
            2: "Mode 2: Three simultaneous for 3 µs (f1,f2,f3), then f4 alone for 1 µs, digital1 at 1st and 4th µs",
            3: "Mode 3: Two simultaneous for 2 µs (f1,f2), then f3,f4 sequentially for 2 µs, digital1 at 1st, 3rd, and 4th µs",
            4: "Mode 4: Purely sequential (one per µs), digital1 at every µs"
        }
        self.lbl_mode_description.setText(descriptions.get(mode, ""))

    def update_digital1_checkboxes(self):
        """Update digital1 checkbox states based on excitation mode"""
        mode = self.w_ch1_excitation_mode

        if mode == 1:
            # Mode 1: Only first position can be toggled
            self.chk_digital1_pos1.setEnabled(True)
            self.chk_digital1_pos2.setEnabled(False)
            self.chk_digital1_pos3.setEnabled(False)
            self.chk_digital1_pos4.setEnabled(False)
            self.chk_digital1_pos2.setChecked(False)
            self.chk_digital1_pos3.setChecked(False)
            self.chk_digital1_pos4.setChecked(False)
        elif mode == 2:
            # Mode 2: First and fourth positions can be toggled
            self.chk_digital1_pos1.setEnabled(True)
            self.chk_digital1_pos2.setEnabled(False)
            self.chk_digital1_pos3.setEnabled(False)
            self.chk_digital1_pos4.setEnabled(True)
            self.chk_digital1_pos2.setChecked(False)
            self.chk_digital1_pos3.setChecked(False)
        elif mode == 3:
            # Mode 3: First, third, and fourth positions can be toggled
            self.chk_digital1_pos1.setEnabled(True)
            self.chk_digital1_pos2.setEnabled(False)
            self.chk_digital1_pos3.setEnabled(True)
            self.chk_digital1_pos4.setEnabled(True)
            self.chk_digital1_pos2.setChecked(False)
        elif mode == 4:
            # Mode 4: All positions can be toggled
            self.chk_digital1_pos1.setEnabled(True)
            self.chk_digital1_pos2.setEnabled(True)
            self.chk_digital1_pos3.setEnabled(True)
            self.chk_digital1_pos4.setEnabled(True)

    def generate_sequence(self, retain_channels=True, make_preview=True):
        if not PYQT_AVAILABLE or not ENTANGLE_AVAILABLE:
            return

        try:
            self.txt_status.append("Calculating phase matrices...")
            QApplication.processEvents()

            # Run phase calculation to generate W_CH1 and R_CH1 matrices
            import subprocess
            import sys
            import os

            # Check if phase_cal_4qubit.py exists using absolute path
            script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
            phase_cal_path = os.path.join(script_dir, "phase_cal_4qubit.py")

            if not os.path.exists(phase_cal_path):
                self.txt_status.append(f"Error: phase_cal_4qubit.py not found at {phase_cal_path}")
                return

            # Prepare command arguments with absolute paths
            w_ch1_global_amp_paths = []
            basis_config_path = os.path.join(script_dir, "matrix/basis_config_4qubit.json")
            for idx, matrix in enumerate(self.read_w_ch1_global_amp_matrices(), start=1):
                matrix_path = os.path.join(script_dir, f"matrix/W_CH1_global_amp_{idx}_4qubit.txt")
                np.savetxt(matrix_path, matrix, fmt='%.6f')
                w_ch1_global_amp_paths.append(matrix_path)
            cmd = [
                sys.executable, phase_cal_path,
                "--w_ch3_amp", os.path.join(script_dir, "matrix/W_CH3_amp_4qubit.txt"),
                "--w_ch3_phase", os.path.join(script_dir, "matrix/W_CH3_phase_4qubit.txt"),
                "--w_ch4_amp", os.path.join(script_dir, "matrix/W_CH4_amp_4qubit.txt"),
                "--w_ch4_phase", os.path.join(script_dir, "matrix/W_CH4_phase_4qubit.txt"),
                "--r_ch3_amp", os.path.join(script_dir, "matrix/R_CH3_amp_4qubit.txt"),
                "--r_ch3_phase", os.path.join(script_dir, "matrix/R_CH3_phase_4qubit.txt"),
                "--r_ch4_amp", os.path.join(script_dir, "matrix/R_CH4_amp_4qubit.txt"),
                "--r_ch4_phase", os.path.join(script_dir, "matrix/R_CH4_phase_4qubit.txt"),
                "--phase_cali", os.path.join(script_dir, "phase_Cali_4qubit.txt"),
                "--basis_config", basis_config_path,
                "--user_amp1", str(self.user_amp_values[0]),
                "--user_amp2", str(self.user_amp_values[1]),
                "--user_amp3", str(self.user_amp_values[2]),
                "--user_amp4", str(self.user_amp_values[3]),
                "--user_phase1", str(self.user_phase_values[0]),
                "--user_phase2", str(self.user_phase_values[1]),
                "--user_phase3", str(self.user_phase_values[2]),
                "--user_phase4", str(self.user_phase_values[3]),
                "--excitation_mode", str(self.w_ch1_excitation_mode),
                "--w_ch1_global_amp_1", w_ch1_global_amp_paths[0],
                "--w_ch1_global_amp_2", w_ch1_global_amp_paths[1],
                "--w_ch1_global_amp_3", w_ch1_global_amp_paths[2],
                "--w_ch1_global_amp_4", w_ch1_global_amp_paths[3],
                "--output_dir", os.path.join(script_dir, "matrix")
            ]

            # Run phase calculation
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.txt_status.append(f"Phase calculation failed: {result.stderr}")
                return

            self.txt_status.append("Phase calculation completed!")

            # Load the calculated density matrix
            try:
                import json
                import os
                density_file = os.path.join(script_dir, "matrix/density_matrix_4qubit.json")
                if os.path.exists(density_file):
                    with open(density_file, 'r') as f:
                        density_data = json.load(f)
                        self.re_rho = np.array(density_data['re_rho'])
                        self.im_rho = np.array(density_data['im_rho'])
                        # Load 4x4 core matrices if available
                        if 're_rho_4x4' in density_data and 'im_rho_4x4' in density_data:
                            self.re_rho_4x4 = np.array(density_data['re_rho_4x4'])
                            self.im_rho_4x4 = np.array(density_data['im_rho_4x4'])
                        else:
                            # Fallback to extracting from full matrix
                            core_indices = [1, 2, 4, 8]
                            self.re_rho_4x4 = self.re_rho[np.ix_(core_indices, core_indices)]
                            self.im_rho_4x4 = self.im_rho[np.ix_(core_indices, core_indices)]
                    self.txt_status.append("Density matrix loaded successfully!")
                else:
                    self.txt_status.append("Warning: Density matrix file not found")
            except Exception as e:
                self.txt_status.append(f"Warning: Failed to load density matrix: {e}")

            self.txt_status.append("Generating 4-qubit entanglement sequence...")
            QApplication.processEvents()

            # Create configuration
            config = EntangleConfig4Qubit()
            config.freq_start = self.spin_freq_start.value()
            config.freq_end = self.spin_freq_end.value()
            # Fixed step at 2.5 MHz for 4-qubit system
            config.output_dir = self.txt_output_dir.text()

            # Set W_CH1 excitation mode and digital control
            config.w_ch1_excitation_mode = self.w_ch1_excitation_mode
            config.w_ch1_digital1_positions = self.w_ch1_digital1_positions
            config.digital_delay_offset_ns = self.digital_delay_offset_ns

            # Set W_CH1 amplitude and phase values
            config.user_amp_values = self.user_amp_values
            config.user_phase_values = self.user_phase_values

            # Set W_CH1 and R_CH1 digital settings
            config.w_ch1_digital_width = self.spin_w_ch1_digital_width.value()
            config.w_ch1_digital_delay = self.spin_w_ch1_digital_delay.value()
            config.r_ch1_digital_width = self.spin_r_ch1_digital_width.value()
            config.r_ch1_digital_delay = self.spin_r_ch1_digital_delay.value()

            # Update channel digital widths and delays
            config.channel_digital_widths['W_CH1'] = config.w_ch1_digital_width
            config.channel_digital_widths['R_CH1'] = config.r_ch1_digital_width
            channel_delays = channel_digital_delay_values(
                config.w_ch1_digital_delay,
                config.r_ch1_digital_delay,
            )
            config.channel_digital_delays.update(channel_delays)
            config.channel_digital2_delays.update(channel_delays)

            # Generate sequence
            channels_data, self.config = generate_entangle_sequence_4qubit(
                config,
                return_channels=retain_channels,
                make_preview=make_preview,
            )
            self.channels_data = channels_data if retain_channels else None

            self.txt_status.append("Sequence generation completed!")
            fourier_warnings = getattr(self.config, "fourier_warnings", [])
            if fourier_warnings:
                self.txt_status.append("Fourier guard warnings:")
                for warning in fourier_warnings:
                    self.txt_status.append(f"  {warning}")
                if getattr(self.config, "fourier_has_frequency_issues", False):
                    QMessageBox.warning(
                        self,
                        "Fourier guard warning",
                        "\n".join(fourier_warnings[:8]),
                    )
            else:
                self.txt_status.append("Fourier guard OK: W_CH1/R_CH1 peaks match intended frequencies.")

            # Update plots
            if MATPLOTLIB_AVAILABLE and self.channels_data is not None:
                self.update_plots()

        except Exception as e:
            self.txt_status.append(f"Error: {str(e)}")
            if PYQT_AVAILABLE:
                QMessageBox.critical(self, "Error", f"Failed to generate sequence:\n{str(e)}")

    def update_plots(self):
        if not MATPLOTLIB_AVAILABLE:
            return

        # Update waveform preview
        self.update_waveform_plot()

        # Update 3D histograms
        self.update_4d_histograms()

    def update_waveform_plot(self):
        if not MATPLOTLIB_AVAILABLE:
            return

        self.waveform_fig.clear()

        if self.channels_data is not None:
            # Plot first few channels as example
            axes = self.waveform_fig.subplots(2, 2)
            channels_to_plot = ['W_CH1', 'W_CH2', 'W_CH3', 'W_CH4']
            for i, ch_name in enumerate(channels_to_plot):
                if ch_name in self.channels_data:
                    row, col = i // 2, i % 2
                    ax = axes[row, col]

                    data = self.channels_data[ch_name]
                    # Robustly extract analog waveform samples from possible data shapes
                    try:
                        if isinstance(data, dict):
                            # Common keys used by sequence generators
                            if 'analog' in data:
                                analog = np.asarray(data['analog'])
                            elif 'wave' in data:
                                analog = np.asarray(data['wave'])
                            elif 'samples' in data:
                                analog = np.asarray(data['samples'])
                            else:
                                # Fallback: try first value
                                analog = np.asarray(list(data.values())[0])
                        else:
                            analog = np.asarray(data)

                        # If 2D (N x M), assume first column is the analog channel
                        if analog.ndim == 2:
                            if analog.shape[1] >= 1:
                                analog = analog[:, 0]
                            else:
                                analog = analog.ravel()

                        display_points = min(10000, analog.shape[0])
                        time_points = np.arange(display_points)
                        ax.plot(time_points, analog[:display_points], lw=0.8)
                    except Exception:
                        # If anything goes wrong, show a placeholder line
                        ax.plot([0, 1], [0, 0], linestyle='--', color='#aaa')

                    ax.set_title(ch_name)
                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True)
        else:
            # Show placeholder
            ax = self.waveform_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Click 'Generate' to preview waveforms",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        self.waveform_fig.tight_layout()
        self.waveform_canvas.draw()

    def update_4d_histograms(self):
        if not MATPLOTLIB_AVAILABLE:
            return
        # Update Re(ρ) histogram (3D surface/bar plot)
        self.re_rho_fig.clear()
        ax1 = self.re_rho_fig.add_subplot(111, projection='3d')
        self.plot_4d_histogram(ax1, self.re_rho, "Re(ρ)")

        self.re_rho_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        self.re_rho_canvas.draw()

        # Update Im(ρ) histogram (3D)
        self.im_rho_fig.clear()
        ax2 = self.im_rho_fig.add_subplot(111, projection='3d')
        self.plot_4d_histogram(ax2, self.im_rho, "Im(ρ)")

        self.im_rho_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        self.im_rho_canvas.draw()

        # Update Re(ρ) 4x4 core histogram (3D)
        self.re_rho_4x4_fig.clear()
        ax3 = self.re_rho_4x4_fig.add_subplot(111, projection='3d')
        self.plot_4d_histogram_4x4(ax3, self.re_rho_4x4, "Re(ρ) Core")

        self.re_rho_4x4_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        self.re_rho_4x4_canvas.draw()

        # Update Im(ρ) 4x4 core histogram (3D)
        self.im_rho_4x4_fig.clear()
        ax4 = self.im_rho_4x4_fig.add_subplot(111, projection='3d')
        self.plot_4d_histogram_4x4(ax4, self.im_rho_4x4, "Im(ρ) Core")

        self.im_rho_4x4_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        self.im_rho_4x4_canvas.draw()

    def plot_4d_histogram(self, ax, data, title):
        if not MATPLOTLIB_AVAILABLE:
            return

        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot 3D bar chart using Axes3D.bar3d
        dx = dy = 0.8
        dz = data.flatten()
        xpos = X.flatten()
        ypos = Y.flatten()
        zpos = np.zeros_like(dz)

        # bar3d takes x, y, z, dx, dy, dz
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

        # Add meaningful labels for 16x16 density matrix
        # Define subspace labels for 4-qubit system: |q3 q2 q1 q0⟩ where q3 is most significant
        subspace_labels = [
            '|0000⟩', '|0001⟩', '|0010⟩', '|0011⟩',
            '|0100⟩', '|0101⟩', '|0110⟩', '|0111⟩',
            '|1000⟩', '|1001⟩', '|1010⟩', '|1011⟩',
            '|1100⟩', '|1101⟩', '|1110⟩', '|1111⟩'
        ]

        # Set axis labels and title
        ax.set_title(title)
        ax.set_xlabel("Bra ⟨i|")
        ax.set_ylabel("Ket |j⟩")
        ax.set_zlabel("Value")

        # Set tick labels for better visualization
        if data.shape[0] == 16 and data.shape[1] == 16:
            ax.set_xticks(range(16))
            ax.set_yticks(range(16))
            ax.set_xticklabels(subspace_labels, rotation=45, ha='right')
            ax.set_yticklabels(subspace_labels, rotation=-45, ha='right')


    def plot_4d_histogram_4x4(self, ax, data, title):
        if not MATPLOTLIB_AVAILABLE:
            return

        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot 3D bar chart using Axes3D.bar3d
        dx = dy = 0.8
        dz = data.flatten()
        xpos = X.flatten()
        ypos = Y.flatten()
        zpos = np.zeros_like(dz)

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

        # Add meaningful labels for the 4x4 core density matrix
        # Define subspace labels: |0001⟩, |0010⟩, |0100⟩, |1000⟩
        core_subspace_labels = ['|0001⟩', '|0010⟩', '|0100⟩', '|1000⟩']

        # Set axis labels and title
        ax.set_title(title)
        ax.set_xlabel("Bra ⟨i|")
        ax.set_ylabel("Ket |j⟩")
        ax.set_zlabel("Value")

        # Set tick labels for better visualization
        if data.shape[0] == 4 and data.shape[1] == 4:
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(core_subspace_labels, rotation=45, ha='right')
            ax.set_yticklabels(core_subspace_labels, rotation=-45, ha='right')


def main():
    if not PYQT_AVAILABLE:
        print("PyQt5 is required to run the GUI. Please install it with: pip install PyQt5")
        return

    app = QApplication(sys.argv)
    window = EntangleUI()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
