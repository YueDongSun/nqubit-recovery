"""
UI for Entangle Sequence Generation
Date: 2025-10-24

Features:
- Generate button to trigger sequence generation
- Waveform preview display
- Two 3D histograms (8x8) for Re(ρ) and Im(ρ)
"""

import os
import sys
import numpy as np

try:
    from .digital_delay_defaults import DEFAULT_DIGITAL_DELAY_OFFSET_NS, channel_digital_delay_values, digital_delay_values
except ImportError:
    from digital_delay_defaults import DEFAULT_DIGITAL_DELAY_OFFSET_NS, channel_digital_delay_values, digital_delay_values

try:
    from .wch1_global_amp import load_w_ch1_global_amp_matrix
except ImportError:
    from wch1_global_amp import load_w_ch1_global_amp_matrix


def default_w_ch1_global_amp_path():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    return os.path.join(script_dir, "matrix", "W_CH1_global_amp.txt")

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
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available, plotting will be disabled")

# Try to import our entangle sequence generator
try:
    from .entangle_sequence import EntangleConfig, generate_entangle_sequence
    ENTANGLE_AVAILABLE = True
except ImportError as e:
    try:
        sequence_dir = os.path.dirname(os.path.abspath(__file__))
        if sequence_dir not in sys.path:
            sys.path.insert(0, sequence_dir)
        from entangle_sequence import EntangleConfig, generate_entangle_sequence
        ENTANGLE_AVAILABLE = True
    except ImportError as fallback_exc:
        ENTANGLE_AVAILABLE = False
        print(f"Could not import entangle_sequence: {fallback_exc or e}")

class EntangleUI(QWidget):
    def __init__(self):
        super().__init__()
        if not PYQT_AVAILABLE:
            print("Cannot initialize UI without PyQt5")
            return

        self.setWindowTitle("Entangle Sequence Generator")
        self.resize(1280, 900)
        self.setMinimumSize(960, 700)

        # Data storage
        self.channels_data = None
        self.config = None

        # Re(ρ) and Im(ρ) matrices (8x8)
        self.re_rho = np.ones((8, 8))
        self.im_rho = np.ones((8, 8))

        # Re(ρ) and Im(ρ) core matrices (3x3)
        self.re_rho_3x3 = np.ones((3, 3))
        self.im_rho_3x3 = np.ones((3, 3))

        # Excitation mode for W_CH1
        self.w_ch1_excitation_mode = 1  # Default to mode 1
        self.w_ch1_digital1_positions = [True, False, False]  # Default positions
        self.digital_delay_offset_ns = DEFAULT_DIGITAL_DELAY_OFFSET_NS

        # User amplitude and phase values for W_CH1
        self.user_amp_values = [1.0, 1.0, 1.0]  # Default amplitude values
        self.user_phase_values = [0.0, 0.0, 0.0]  # Default phase values (radians)
        self.w_ch1_global_amp_matrix = load_w_ch1_global_amp_matrix(default_w_ch1_global_amp_path())

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
        control_layout.setColumnStretch(6, 1)

        # Generate button
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(self.generate_sequence)

        # Frequency settings
        freq_group = QGroupBox("Frequency Settings")
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Start (MHz):"))
        self.spin_freq_start = QDoubleSpinBox()
        self.spin_freq_start.setRange(50, 200)
        self.spin_freq_start.setSingleStep(0.5)
        self.spin_freq_start.setValue(85)
        freq_layout.addWidget(self.spin_freq_start)

        freq_layout.addWidget(QLabel("End (MHz):"))
        self.spin_freq_end = QDoubleSpinBox()
        self.spin_freq_end.setRange(50, 200)
        self.spin_freq_end.setSingleStep(0.5)
        self.spin_freq_end.setValue(120)
        freq_layout.addWidget(self.spin_freq_end)

        freq_layout.addWidget(QLabel("Step (MHz):"))
        self.spin_freq_step = QDoubleSpinBox()
        self.spin_freq_step.setRange(0.1, 10)
        self.spin_freq_step.setSingleStep(0.1)
        self.spin_freq_step.setValue(2.5)
        self.spin_freq_step.setEnabled(False)  # Fixed at 2.5 MHz
        freq_layout.addWidget(self.spin_freq_step)

        freq_group.setLayout(freq_layout)

        # W_CH1 Excitation Mode settings
        mode_group = QGroupBox("W_CH1 Excitation Mode")
        mode_layout = QHBoxLayout()

        mode_layout.addWidget(QLabel("Mode:"))
        self.combo_excitation_mode = QComboBox()
        self.combo_excitation_mode.addItem("Mode 1: Simultaneous frequencies (1 pulse)")
        self.combo_excitation_mode.addItem("Mode 2: (1,2)/3 pattern (2 pulses)")
        self.combo_excitation_mode.addItem("Mode 3: Sequential frequencies (3 pulses)")
        self.combo_excitation_mode.setCurrentIndex(0)  # Default to mode 1
        self.combo_excitation_mode.currentIndexChanged.connect(self.on_excitation_mode_changed)
        mode_layout.addWidget(self.combo_excitation_mode)

        # Digital1 pulse position checkboxes
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

        # Initialize checkbox states based on default mode
        self.update_digital1_checkboxes()

        mode_group.setLayout(mode_layout)

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

        # W_CH1 Amplitude and Phase settings
        amp_phase_group = QGroupBox("W_CH1 Amplitude & Phase")
        amp_phase_layout = QHBoxLayout()

        # Amplitude inputs
        amp_phase_layout.addWidget(QLabel("Amp:"))
        self.spin_amp1 = QDoubleSpinBox()
        self.spin_amp1.setRange(0.0, 5.0)
        self.spin_amp1.setSingleStep(0.1)
        self.spin_amp1.setValue(1.0)
        self.spin_amp1.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(QLabel("1:"))
        amp_phase_layout.addWidget(self.spin_amp1)

        self.spin_amp2 = QDoubleSpinBox()
        self.spin_amp2.setRange(0.0, 5.0)
        self.spin_amp2.setSingleStep(0.1)
        self.spin_amp2.setValue(1.0)
        self.spin_amp2.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(QLabel("2:"))
        amp_phase_layout.addWidget(self.spin_amp2)

        self.spin_amp3 = QDoubleSpinBox()
        self.spin_amp3.setRange(0.0, 5.0)
        self.spin_amp3.setSingleStep(0.1)
        self.spin_amp3.setValue(1.0)
        self.spin_amp3.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(QLabel("3:"))
        amp_phase_layout.addWidget(self.spin_amp3)

        # Phase inputs (in degrees for user convenience, converted to radians internally)
        amp_phase_layout.addWidget(QLabel("Phase (°):"))
        self.spin_phase1 = QDoubleSpinBox()
        self.spin_phase1.setRange(-360.0, 360.0)
        self.spin_phase1.setSingleStep(1.0)
        self.spin_phase1.setValue(0.0)
        self.spin_phase1.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(QLabel("1:"))
        amp_phase_layout.addWidget(self.spin_phase1)

        self.spin_phase2 = QDoubleSpinBox()
        self.spin_phase2.setRange(-360.0, 360.0)
        self.spin_phase2.setSingleStep(1.0)
        self.spin_phase2.setValue(0.0)
        self.spin_phase2.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(QLabel("2:"))
        amp_phase_layout.addWidget(self.spin_phase2)

        self.spin_phase3 = QDoubleSpinBox()
        self.spin_phase3.setRange(-360.0, 360.0)
        self.spin_phase3.setSingleStep(1.0)
        self.spin_phase3.setValue(0.0)
        self.spin_phase3.valueChanged.connect(self.on_amp_phase_changed)
        amp_phase_layout.addWidget(QLabel("3:"))
        amp_phase_layout.addWidget(self.spin_phase3)

        amp_phase_group.setLayout(amp_phase_layout)

        matrix_group = QGroupBox("W_CH1 Global Amplitude Matrix (15x5)")
        matrix_layout = QVBoxLayout()
        self.table_w_ch1_global_amp = QTableWidget(15, 5)
        self.table_w_ch1_global_amp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_w_ch1_global_amp.setMaximumHeight(260)
        for row in range(15):
            for col in range(5):
                value = self.w_ch1_global_amp_matrix[row, col]
                self.table_w_ch1_global_amp.setItem(row, col, QTableWidgetItem(f"{value:.6g}"))
        matrix_layout.addWidget(self.table_w_ch1_global_amp)
        matrix_group.setLayout(matrix_layout)

        # Output directory
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_directory)
        self.txt_output_dir = QLineEdit(os.path.join("AWG_Sequence", "entangle_sequences"))

        control_layout.addWidget(self.btn_generate, 0, 0)
        control_layout.addWidget(freq_group, 0, 1, 1, 2)
        control_layout.addWidget(mode_group, 0, 3, 1, 4)
        control_layout.addWidget(digital_settings_group, 1, 0, 1, 3)
        control_layout.addWidget(amp_phase_group, 1, 3, 1, 4)
        control_layout.addWidget(QLabel("Output Dir:"), 2, 0)
        control_layout.addWidget(self.txt_output_dir, 2, 1, 1, 5)
        control_layout.addWidget(self.btn_browse, 2, 6)

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

        # Full 8x8 3D histogram group
        hist_group = QGroupBox("3D Histograms (8x8)")
        hist_layout = QHBoxLayout()

        # Re(ρ) 3D histogram
        re_rho_group = QGroupBox("Re(ρ)")
        re_rho_layout = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.re_rho_fig = Figure(figsize=(3.2, 2.4))
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
            self.im_rho_fig = Figure(figsize=(3.2, 2.4))
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

        # 3x3 core 3D histogram group
        core_hist_group = QGroupBox("3D Histograms (3x3 Core)")
        core_hist_layout = QHBoxLayout()

        # Re(ρ) 3x3 core 3D histogram
        re_rho_3x3_group = QGroupBox("Re(ρ) Core")
        re_rho_3x3_layout = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.re_rho_3x3_fig = Figure(figsize=(3.2, 2.4))
            self.re_rho_3x3_canvas = FigureCanvas(self.re_rho_3x3_fig)
            self.re_rho_3x3_canvas.setMinimumSize(280, 220)
            re_rho_3x3_layout.addWidget(self.re_rho_3x3_canvas)
        else:
            re_rho_3x3_layout.addWidget(QLabel("Matplotlib not available"))
        re_rho_3x3_group.setLayout(re_rho_3x3_layout)

        # Im(ρ) 3x3 core 3D histogram
        im_rho_3x3_group = QGroupBox("Im(ρ) Core")
        im_rho_3x3_layout = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.im_rho_3x3_fig = Figure(figsize=(3.2, 2.4))
            self.im_rho_3x3_canvas = FigureCanvas(self.im_rho_3x3_fig)
            self.im_rho_3x3_canvas.setMinimumSize(280, 220)
            im_rho_3x3_layout.addWidget(self.im_rho_3x3_canvas)
        else:
            im_rho_3x3_layout.addWidget(QLabel("Matplotlib not available"))
        im_rho_3x3_group.setLayout(im_rho_3x3_layout)

        core_hist_layout.addWidget(re_rho_3x3_group)
        core_hist_layout.addWidget(im_rho_3x3_group)
        core_hist_group.setLayout(core_hist_layout)

        right_layout.addWidget(core_hist_group)

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
        self.update_digital1_checkboxes()

    def on_digital1_position_changed(self, state):
        """Handle digital1 position checkbox changes"""
        self.w_ch1_digital1_positions[0] = self.chk_digital1_pos1.isChecked()
        self.w_ch1_digital1_positions[1] = self.chk_digital1_pos2.isChecked()
        self.w_ch1_digital1_positions[2] = self.chk_digital1_pos3.isChecked()

    def update_digital1_checkboxes(self):
        """Update digital1 checkbox states based on excitation mode"""
        mode = self.w_ch1_excitation_mode

        if mode == 1:
            # Mode 1: Only first position can be toggled
            self.chk_digital1_pos1.setEnabled(True)
            self.chk_digital1_pos2.setEnabled(False)
            self.chk_digital1_pos3.setEnabled(False)
            self.chk_digital1_pos2.setChecked(False)
            self.chk_digital1_pos3.setChecked(False)
        elif mode == 2:
            # Mode 2: First and third positions can be toggled
            self.chk_digital1_pos1.setEnabled(True)
            self.chk_digital1_pos2.setEnabled(False)
            self.chk_digital1_pos3.setEnabled(True)
            self.chk_digital1_pos2.setChecked(False)
        elif mode == 3:
            # Mode 3: All positions can be toggled
            self.chk_digital1_pos1.setEnabled(True)
            self.chk_digital1_pos2.setEnabled(True)
            self.chk_digital1_pos3.setEnabled(True)

    def on_amp_phase_changed(self):
        """Handle amplitude and phase value changes"""
        # Convert degrees to radians for internal storage
        import math
        self.user_amp_values = [
            self.spin_amp1.value(),
            self.spin_amp2.value(),
            self.spin_amp3.value()
        ]
        self.user_phase_values = [
            math.radians(self.spin_phase1.value()),
            math.radians(self.spin_phase2.value()),
            math.radians(self.spin_phase3.value())
        ]

    def set_target_state(self, amplitudes, phases_deg, excitation_mode):
        """Atomically synchronize automation state into widgets and generator state."""
        if len(amplitudes) != 3 or len(phases_deg) != 3:
            raise ValueError("3Q target state requires three amplitudes and phases")
        for idx, value in enumerate(amplitudes, start=1):
            getattr(self, f"spin_amp{idx}").setValue(float(value))
        for idx, value in enumerate(phases_deg, start=1):
            getattr(self, f"spin_phase{idx}").setValue(float(value))
        self.combo_excitation_mode.setCurrentIndex(int(excitation_mode) - 1)
        self.w_ch1_excitation_mode = int(excitation_mode)
        self.on_amp_phase_changed()

    def read_w_ch1_global_amp_matrix(self):
        matrix = np.ones((15, 5))
        for row in range(15):
            for col in range(5):
                item = self.table_w_ch1_global_amp.item(row, col)
                text = item.text().strip() if item and item.text() else "1.0"
                try:
                    matrix[row, col] = float(text)
                except ValueError:
                    matrix[row, col] = 1.0
        self.w_ch1_global_amp_matrix = matrix
        return matrix

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

            # Check if phase_cal.py exists using absolute path
            script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
            phase_cal_path = os.path.join(script_dir, "phase_cal.py")

            if not os.path.exists(phase_cal_path):
                self.txt_status.append(f"Error: phase_cal.py not found at {phase_cal_path}")
                return

            # Prepare command arguments with absolute paths
            w_ch1_global_amp_path = default_w_ch1_global_amp_path()
            basis_config_path = os.path.join(script_dir, "matrix/basis_config_3qubit.json")
            np.savetxt(w_ch1_global_amp_path, self.read_w_ch1_global_amp_matrix(), fmt='%.6f')
            cmd = [
                sys.executable, phase_cal_path,
                "--w_ch3_amp", os.path.join(script_dir, "matrix/W_CH3_amp.txt"),
                "--w_ch3_phase", os.path.join(script_dir, "matrix/W_CH3_phase.txt"),
                "--w_ch4_amp", os.path.join(script_dir, "matrix/W_CH4_amp.txt"),
                "--w_ch4_phase", os.path.join(script_dir, "matrix/W_CH4_phase.txt"),
                "--r_ch3_amp", os.path.join(script_dir, "matrix/R_CH3_amp.txt"),
                "--r_ch3_phase", os.path.join(script_dir, "matrix/R_CH3_phase.txt"),
                "--r_ch4_amp", os.path.join(script_dir, "matrix/R_CH4_amp.txt"),
                "--r_ch4_phase", os.path.join(script_dir, "matrix/R_CH4_phase.txt"),
                "--phase_cali", os.path.join(script_dir, "phase_Cali.txt"),
                "--basis_config", basis_config_path,
                "--user_amp1", str(self.user_amp_values[0]),
                "--user_amp2", str(self.user_amp_values[1]),
                "--user_amp3", str(self.user_amp_values[2]),
                "--user_phase1", str(self.user_phase_values[0]),
                "--user_phase2", str(self.user_phase_values[1]),
                "--user_phase3", str(self.user_phase_values[2]),
                "--excitation_mode", str(self.w_ch1_excitation_mode),
                "--w_ch1_global_amp", w_ch1_global_amp_path,
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
                density_file = os.path.join(script_dir, "matrix/density_matrix.json")
                if os.path.exists(density_file):
                    with open(density_file, 'r') as f:
                        density_data = json.load(f)
                        self.re_rho = np.array(density_data['re_rho'])
                        self.im_rho = np.array(density_data['im_rho'])
                        # Load 3x3 core matrices if available
                        if 're_rho_3x3' in density_data and 'im_rho_3x3' in density_data:
                            self.re_rho_3x3 = np.array(density_data['re_rho_3x3'])
                            self.im_rho_3x3 = np.array(density_data['im_rho_3x3'])
                        else:
                            # Fallback to extracting from full matrix
                            core_indices = [1, 2, 4]
                            self.re_rho_3x3 = self.re_rho[np.ix_(core_indices, core_indices)]
                            self.im_rho_3x3 = self.im_rho[np.ix_(core_indices, core_indices)]
                    self.txt_status.append("Density matrix loaded successfully!")
                else:
                    self.txt_status.append("Warning: Density matrix file not found")
            except Exception as e:
                self.txt_status.append(f"Warning: Failed to load density matrix: {e}")

            self.txt_status.append("Generating sequence...")
            QApplication.processEvents()

            # Create configuration
            config = EntangleConfig()
            config.freq_start = self.spin_freq_start.value()
            config.freq_end = self.spin_freq_end.value()
            config.output_dir = self.txt_output_dir.text()

            # Set W_CH1 excitation mode and digital1 positions
            config.w_ch1_excitation_mode = self.w_ch1_excitation_mode
            config.w_ch1_digital1_positions = self.w_ch1_digital1_positions
            config.digital_delay_offset_ns = self.digital_delay_offset_ns

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
            channels_data, self.config = generate_entangle_sequence(
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
        self.update_3d_histograms()

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
                    # Display first 10000 points for performance
                    display_points = min(10000, len(data))
                    time_points = np.arange(display_points)
                    analog_data = data[:display_points, 0]

                    ax.plot(time_points, analog_data)
                    ax.set_title(ch_name)
                    ax.set_xlabel("Time (ns)")
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

    def update_3d_histograms(self):
        if not MATPLOTLIB_AVAILABLE:
            return

        # Update Re(ρ) histogram
        self.re_rho_fig.clear()
        ax1 = self.re_rho_fig.add_subplot(111, projection='3d')
        self.plot_3d_histogram(ax1, self.re_rho, "Re(ρ)")

        self.re_rho_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        self.re_rho_canvas.draw()

        # Update Im(ρ) histogram
        self.im_rho_fig.clear()
        ax2 = self.im_rho_fig.add_subplot(111, projection='3d')
        self.plot_3d_histogram(ax2, self.im_rho, "Im(ρ)")

        self.im_rho_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        self.im_rho_canvas.draw()

        # Update Re(ρ) 3x3 core histogram
        self.re_rho_3x3_fig.clear()
        ax3 = self.re_rho_3x3_fig.add_subplot(111, projection='3d')
        self.plot_3d_histogram_3x3(ax3, self.re_rho_3x3, "Re(ρ) Core")

        self.re_rho_3x3_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        self.re_rho_3x3_canvas.draw()

        # Update Im(ρ) 3x3 core histogram
        self.im_rho_3x3_fig.clear()
        ax4 = self.im_rho_3x3_fig.add_subplot(111, projection='3d')
        self.plot_3d_histogram_3x3(ax4, self.im_rho_3x3, "Im(ρ) Core")

        self.im_rho_3x3_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        self.im_rho_3x3_canvas.draw()

    def plot_3d_histogram(self, ax, data, title):
        if not MATPLOTLIB_AVAILABLE:
            return

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

        # Add meaningful labels for the 8x8 density matrix
        # Define subspace labels: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩
        subspace_labels = [
            '|000⟩', '|001⟩', '|010⟩', '|011⟩',
            '|100⟩', '|101⟩', '|110⟩', '|111⟩'
        ]

        # Set axis labels and title
        ax.set_title(title)
        ax.set_xlabel("Bra ⟨i|")
        ax.set_ylabel("Ket |j⟩")
        ax.set_zlabel("Value")

        # Set tick labels for better visualization
        if data.shape[0] == 8 and data.shape[1] == 8:
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_xticklabels(subspace_labels, rotation=45, ha='right')
            ax.set_yticklabels(subspace_labels, rotation=-45, ha='right')
        else:
            # Fallback to numeric labels
            ax.set_xlabel("X Index")
            ax.set_ylabel("Y Index")


    def plot_3d_histogram_3x3(self, ax, data, title):
        if not MATPLOTLIB_AVAILABLE:
            return

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

        # Add meaningful labels for the 3x3 core density matrix
        # Define subspace labels: |001⟩, |010⟩, |100⟩
        core_subspace_labels = ['|001⟩', '|010⟩', '|100⟩']

        # Set axis labels and title
        ax.set_title(title)
        ax.set_xlabel("Bra ⟨i|")
        ax.set_ylabel("Ket |j⟩")
        ax.set_zlabel("Value")

        # Set tick labels for better visualization
        if data.shape[0] == 3 and data.shape[1] == 3:
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(core_subspace_labels, rotation=45, ha='right')
            ax.set_yticklabels(core_subspace_labels, rotation=-45, ha='right')
        else:
            # Fallback to numeric labels
            ax.set_xlabel("X Index")
            ax.set_ylabel("Y Index")


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
