#!/usr/bin/env python3
"""Unified experiment entry UI for normal and feedback workflows."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QTextEdit,
    QDoubleSpinBox,
    QFormLayout,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .automation_schedule import (
    MAX_AUTO_STATES,
    V4_4Q_COVERAGE_STATES,
    V4_4Q_SIGNED_PROBE_ANCHORS,
    V4_4Q_ZERO_PROBE_ANCHORS,
    generate_coverage_schedule,
    generate_drift_aware_coverage_schedule,
    load_state_schedule_file,
    parse_state_schedule,
    schedule_to_json,
    validate_v4_4qubit_campaign_schedule,
)
from .awg_uploader import FrontPanelAwgUploader
from .data_source import MeasurementAccumulator
from .dependency_report import collect_dependency_items, format_dependency_report, has_blocking_missing
from .fpga_programmer import ENV_CABLE, ENV_PROGRAMMER, find_quartus_pgm, program_fpga
from .model_paths import LEGACY_FEEDBACK_MODEL_PATH, display_feedback_model_path
from .data_protocol import (
    CampaignManifestV1,
    atomic_write_json,
    load_snapshot_manifest,
    stable_json_hash,
)
from .dataset_store import matrix_hash_snapshot
from .models import (
    DEFAULT_MEASUREMENT_DURATION_S,
    V4_4Q_CAMPAIGN_DURATION_S,
    ExperimentRunConfig,
)
from .runner import ExperimentRunner
from .ui_controllers import DataUiController, SequenceUiController
from .ui_readiness import ExperimentUiReadiness, build_experiment_readiness
from .v6_episode_ui import V6EpisodePanel, V6EpisodeUiStateV1
from system_control.qubit_recon_files.digital_delay_defaults import (
    DEFAULT_DIGITAL_DELAY_OFFSET_NS,
    MAX_DIGITAL_DELAY_OFFSET_NS,
    MIN_DIGITAL_DELAY_OFFSET_NS,
    digital_delay_values,
)


class TerminalLoggingTextEdit(QTextEdit):
    """QTextEdit that mirrors every append() call to stderr for terminal visibility.

    Keeps a single source of truth (the UI log widget) while also writing to
    the terminal that launched the UI, so headless-style observation of the
    auto cycle is possible without watching the GUI. Set the env var
    ``EXPERIMENT_UI_TERMINAL_LOG=0`` to silence the terminal mirror.

    Thread-safe: append() is invoked on the Qt main thread via queued signal
    connections from runner / AWG threads, and ``print(file=...)`` is itself
    thread-safe under the GIL.
    """

    _ENABLED = os.environ.get("EXPERIMENT_UI_TERMINAL_LOG", "1").lower() not in ("0", "false", "no", "off")

    def append(self, text: str) -> None:  # type: ignore[override]
        super().append(text)
        if TerminalLoggingTextEdit._ENABLED:
            try:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] {text}", file=sys.stderr, flush=True)
            except Exception:
                # Never let logging failure break the UI flow.
                pass


class RunnerThread(QThread):
    log = pyqtSignal(str)
    finished_snapshot = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config: ExperimentRunConfig):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.log.emit(f"Running mock {self.config.workflow_mode} window")
            runner = ExperimentRunner(self.config)
            snapshot = runner.run_dry_cycle()
            self.log.emit(f"Finished mock dry cycle: state={runner.state.value}")
            payload = snapshot.to_json_dict()
            if runner.last_sample_path is not None:
                payload["sample_path"] = str(runner.last_sample_path)
            self.finished_snapshot.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))


class AwgUploadThread(QThread):
    log = pyqtSignal(str)
    uploaded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, uploader: FrontPanelAwgUploader, sequence_dir: str, metadata: Dict):
        super().__init__()
        self.uploader = uploader
        self.sequence_dir = sequence_dir
        self.metadata = metadata

    def run(self):
        try:
            self.log.emit(f"Uploading sequence to AWG: {self.sequence_dir}")
            result = self.uploader.upload(self.sequence_dir, self.metadata, log=self.log.emit)
            if not isinstance(result, dict):
                result = {"uploaded": bool(result)}
            self.uploaded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class FpgaProgramThread(QThread):
    log = pyqtSignal(str)
    programmed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    DEFAULT_MAX_ATTEMPTS = 10
    DEFAULT_RETRY_DELAY_S = 5.0

    def __init__(
        self,
        system_type: str,
        programmer_path: str = "",
        cable: str = "",
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
    ):
        super().__init__()
        self.system_type = system_type
        self.programmer_path = programmer_path
        self.cable = cable
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_s = max(0.0, float(retry_delay_s))

    def run(self):
        failures: List[str] = []
        for attempt in range(1, self.max_attempts + 1):
            if self.isInterruptionRequested():
                return
            self.log.emit(
                f"Programming FPGA for {self.system_type}: "
                f"attempt {attempt}/{self.max_attempts}..."
            )
            try:
                result = program_fpga(
                    self.system_type,
                    programmer_path=self.programmer_path or None,
                    cable=self.cable or None,
                    log=self.log.emit,
                )
            except Exception as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt >= self.max_attempts:
                    self.failed.emit(
                        f"FPGA programming failed after {self.max_attempts} attempts. "
                        f"Last error: {exc}"
                    )
                    return
                self.log.emit(
                    f"FPGA programming attempt {attempt}/{self.max_attempts} failed: {exc}. "
                    f"Retrying in {self.retry_delay_s:g}s."
                )
                if not self._wait_for_retry():
                    return
                continue

            payload = result.to_json_dict()
            payload.update(
                {
                    "attempt": attempt,
                    "max_attempts": self.max_attempts,
                    "prior_failures": failures,
                }
            )
            self.programmed.emit(payload)
            return

    def _wait_for_retry(self) -> bool:
        """Wait in the worker thread while remaining responsive to Stop."""

        remaining_ms = int(round(self.retry_delay_s * 1000.0))
        while remaining_ms > 0:
            if self.isInterruptionRequested():
                return False
            step_ms = min(100, remaining_ms)
            self.msleep(step_ms)
            remaining_ms -= step_ms
        return not self.isInterruptionRequested()


def _bridge_256_sequences_to_standard_dir(
    source_dir: Path,
    standard_dir: Path,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Path]:
    """Bridge ``source_dir`` linear-sweep files into the standard 4Q naming.

    ``awg256seq`` emits files like ``AWG_W_CH1_T266us_D0ns_W170ns_linear_*.txt``;
    ``AWGcontrol_port/set_front_4qubit.py`` expects plain
    ``{W,R}_CH{n}_T{xxx}us.txt``. This helper symlinks (or copies on Windows
    without developer mode) each source file into ``standard_dir`` under its
    canonical name, clearing any prior bridge artifacts first so the upload
    sees a clean directory.

    Returns the bridge map ``{canonical_path: source_path}``. Raises
    ``FileNotFoundError`` if the source contains fewer than 8 linear files
    (one per channel, 4 W + 4 R).
    """
    import re as _re
    import shutil as _shutil

    def _emit(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    source_dir = Path(source_dir)
    standard_dir = Path(standard_dir)
    standard_dir.mkdir(parents=True, exist_ok=True)

    # Wipe stale bridge artifacts from previous runs.
    bridge_pat = _re.compile(r'^(W_CH\d|R_CH\d)_T(\d+)us\.txt$', _re.IGNORECASE)
    stale = [p for p in standard_dir.glob('*.txt') if bridge_pat.match(p.name)]
    for s in stale:
        s.unlink()
    if stale:
        _emit(f"Cleared {len(stale)} stale bridge file(s) from {standard_dir}.")

    linear_files = sorted(source_dir.glob('*_linear_*'))
    if len(linear_files) < 8:
        raise FileNotFoundError(
            f"Bridge needs 8 linear files (4 W + 4 R); found {len(linear_files)} in {source_dir}."
        )

    duration_re = _re.compile(r'_T(\d+)us_', _re.IGNORECASE)
    ch_re = _re.compile(r'(W_CH\d|R_CH\d)')
    bridge_map: Dict[str, Path] = {}
    for f in linear_files:
        m = duration_re.search(f.stem)
        if not m:
            _emit(f"  [bridge warn] no duration in filename: {f.name}")
            continue
        cm = ch_re.search(f.stem)
        if not cm:
            _emit(f"  [bridge warn] no channel in filename: {f.name}")
            continue
        duration = m.group(1)
        ch_name = cm.group(1)
        link = standard_dir / f'{ch_name}_T{duration}us.txt'
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(f.resolve())
        except (OSError, NotImplementedError):
            _shutil.copy2(f, link)
        bridge_map[str(link)] = f
        _emit(f"  [bridge] {link.name} → {f.name}")

    if len(bridge_map) != 8:
        raise FileNotFoundError(
            f"Bridge produced {len(bridge_map)}/8 files from {source_dir}; aborting."
        )
    _emit(f"Bridged {len(bridge_map)}/8 linear sequence files into {standard_dir}.")
    return bridge_map


class RetrievalCalibrationUploadThread(QThread):
    """Generate 256 AWG sequences and upload linear channels to AWGs."""

    log = pyqtSignal(str)
    uploaded = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, awg1_ip: str = '192.168.2.122::4011',
                 awg2_ip: str = '192.168.2.110::3992',
                 marker_high: float = 2.5,
                 marker_low: float = 0.0):
        super().__init__()
        self.awg1_ip = awg1_ip
        self.awg2_ip = awg2_ip
        self.marker_high = marker_high
        self.marker_low = marker_low

    def run(self):
        try:
            # Step 1: Generate sequences
            self.log.emit("Generating 256 AWG sequences...")
            # Import here to avoid circular import at module level
            repo_root = Path(__file__).resolve().parents[2]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from system_control.awg256seq import generate_awg_sequence
            channels, config = generate_awg_sequence()
            output_dir = Path(config.output_dir)
            self.log.emit(f"Sequences generated in {output_dir}")

            # Step 2: Find linear sequence files
            linear_files = sorted(output_dir.glob('*_linear_*'))
            if not linear_files:
                self.failed.emit(f"No linear sequence files found in {output_dir}")
                return

            # Group by channel: W_CH* -> AWG1, R_CH* -> AWG2
            w_files = {}
            r_files = {}
            for f in linear_files:
                name = f.stem  # e.g. AWG_W_CH1_..._linear_...
                if '_W_CH' in name:
                    # Extract channel number
                    for i in range(1, 5):
                        if f'_W_CH{i}_' in name or f'_W_CH{i}.' in name:
                            w_files[i] = f
                            break
                elif '_R_CH' in name:
                    for i in range(1, 5):
                        if f'_R_CH{i}_' in name or f'_R_CH{i}.' in name:
                            r_files[i] = f
                            break

            self.log.emit(f"Found {len(w_files)} W channels, {len(r_files)} R channels")

            # Sanity check: must have all 8 channels for the upload to succeed.
            expected = {('W', i) for i in range(1, 5)} | {('R', i) for i in range(1, 5)}
            found = {('W', i) for i in w_files} | {('R', i) for i in r_files}
            missing = expected - found
            if missing:
                self.failed.emit(
                    f"Missing linear sequence files for {len(missing)} channel(s): "
                    + ", ".join(f"{p}_CH{i}" for p, i in sorted(missing))
                    + f". Looked in {output_dir}. Cannot upload to AWG."
                )
                return

            # Step 3: Bridge file naming so AWGcontrol_port/set_front_4qubit.py
            # can pick up the 256 generator's output.
            standard_dir = repo_root / 'AWG_Sequence' / 'entangle_sequences_4qubit'
            try:
                _bridge_256_sequences_to_standard_dir(
                    output_dir, standard_dir, log=self.log.emit
                )
            except FileNotFoundError as exc:
                self.failed.emit(str(exc))
                return

            # Step 4: Upload via the canonical AWGcontrol_port path.
            # FrontPanelAwgUploader dispatches to AWGcontrol_port/set_front_4qubit.py
            # which already implements reset / sample-rate / run-mode / trigger /
            # waveform-write / channel-assign / output-on / start for both AWGs.
            self.log.emit(
                f"Uploading via AWGcontrol_port/set_front_4qubit.py "
                f"(AWG1={self.awg1_ip}, AWG2={self.awg2_ip}) ..."
            )
            uploader = FrontPanelAwgUploader(
                awg1_ip=self.awg1_ip,
                awg2_ip=self.awg2_ip,
                sample_rate=getattr(config, 'sampling_rate', 1e9) or 1e9,
            )
            result = uploader.upload(
                str(standard_dir),
                {
                    "system_type": "4qubit",
                    "mode": "retrieval_calibration_256",
                    "source": "awg256seq",
                    "duration_ns": int(getattr(config, 'total_duration', 0) or 0),
                },
                log=self.log.emit,  # pipe per-file AWG progress into the UI log
                marker_high=self.marker_high,
                marker_low=self.marker_low,
            )
            self.log.emit(
                f"AWG upload complete (backend={result.get('backend')}, "
                f"system_type={result.get('system_type')}, "
                f"duration={result.get('time_duration')})."
            )
            self.uploaded.emit()

        except Exception as exc:
            self.failed.emit(str(exc))


class ExperimentEntryUI(QWidget):
    """Top-level control surface for 3Q/4Q normal and feedback runs."""

    def __init__(self, auto_open_child_uis_on_startup: bool = True):
        super().__init__()
        self.setWindowTitle("Unified W-State Experiment Control")
        self.resize(1100, 900)
        self.auto_open_child_uis_on_startup = auto_open_child_uis_on_startup
        self.runner_thread = None
        self.fpga_thread = None
        self.awg_thread = None
        self.awg_uploader = FrontPanelAwgUploader()
        self.last_sequence_dir = None
        self.sequence_controller = SequenceUiController()
        self.data_source = DataUiController(self._on_data_frame)
        self.real_runner = None
        self.accumulator = None
        self.auto_running = False
        self.auto_state_index = 0
        self.auto_schedule = []
        self.auto_run_dir: Optional[Path] = None
        self.auto_resume_index = 0
        self.auto_schedule_source_path: Optional[Path] = None
        self.auto_schedule_source_description = ""
        self._state_syncing = False
        self._model_path_user_edited = False
        self._retrieval_upload_thread = None
        self._retrieval_data_ui_process = None
        self._latest_readiness = None
        self._continue_auto_after_fpga = False
        self._last_fpga_program_success = False
        self._auto_after_fpga_action = ""
        self._v6_feedback_backend = "legacy"
        self._v6_episode_action_handler: Optional[Callable[[str, Dict], Optional[Dict]]] = None
        self._setup_ui()
        if self.auto_open_child_uis_on_startup:
            QTimer.singleShot(0, self.startup_open_child_uis)

    def _setup_ui(self):
        main = QVBoxLayout(self)

        run_group = QGroupBox("Run")
        run_layout = QGridLayout(run_group)

        self.system_combo = QComboBox()
        self.system_combo.addItems(["3qubit", "4qubit"])
        self.system_combo.currentTextChanged.connect(self._on_system_changed)

        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems(["collect_dataset", "feedback_optimize"])

        self.excitation_spin = QSpinBox()
        self.excitation_spin.setRange(1, 4)
        self.excitation_spin.setValue(1)

        self.digital_delay_offset_spin = QSpinBox()
        self.digital_delay_offset_spin.setRange(MIN_DIGITAL_DELAY_OFFSET_NS, MAX_DIGITAL_DELAY_OFFSET_NS)
        self.digital_delay_offset_spin.setSingleStep(100)
        self.digital_delay_offset_spin.setValue(DEFAULT_DIGITAL_DELAY_OFFSET_NS)
        self.digital_delay_offset_spin.setSuffix(" ns")
        self.digital_delay_offset_spin.valueChanged.connect(self._on_digital_delay_offset_changed)
        self.digital_delay_actual_label = QLabel()

        self.feedback_interval_spin = QDoubleSpinBox()
        self.feedback_interval_spin.setRange(1.0, 86400.0)
        self.feedback_interval_spin.setValue(1800.0)
        self.feedback_interval_spin.setSuffix(" s")

        self.normal_duration_spin = QDoubleSpinBox()
        self.normal_duration_spin.setRange(1.0, 86400.0)
        self.normal_duration_spin.setValue(DEFAULT_MEASUREMENT_DURATION_S)
        self.normal_duration_spin.setSuffix(" s")

        # Settle delay between AWG upload and the data UI Start. Default 30s,
        # long enough for the FPGA histogramming counters to come up cleanly
        # after the AWG trigger is armed, short enough not to feel sluggish.
        self.auto_pre_start_settle_spin = QDoubleSpinBox()
        self.auto_pre_start_settle_spin.setRange(0.0, 600.0)
        self.auto_pre_start_settle_spin.setValue(30.0)
        self.auto_pre_start_settle_spin.setSingleStep(1.0)
        self.auto_pre_start_settle_spin.setSuffix(" s")

        self.target_fidelity_spin = QDoubleSpinBox()
        self.target_fidelity_spin.setRange(0.0, 1.0)
        self.target_fidelity_spin.setSingleStep(0.01)
        self.target_fidelity_spin.setValue(0.95)

        self.efficiency_guard_spin = QDoubleSpinBox()
        self.efficiency_guard_spin.setRange(0.0, 0.5)
        self.efficiency_guard_spin.setSingleStep(0.005)
        self.efficiency_guard_spin.setValue(0.02)

        self.phase_strategy_combo = QComboBox()
        self.phase_strategy_combo.addItems([
            "w_subspace_phase",
            "geometric_phase_experimental",
            "classifier_enhanced",
        ])
        self.strict_hardware_check = QCheckBox("Strict hardware mode")
        self.strict_hardware_check.setChecked(True)
        self.v6_controls_check = QCheckBox("Show V6 feedback controls (disabled by default)")
        self.v6_controls_check.setChecked(False)
        self.v6_controls_check.toggled.connect(self._on_v6_controls_toggled)

        self.model_path_edit = QLineEdit(display_feedback_model_path(self.system_combo.currentText()))
        self.model_path_edit.textEdited.connect(lambda _text: setattr(self, "_model_path_user_edited", True))
        self.awg1_endpoint_edit = QLineEdit("192.168.2.122::4011")
        self.awg2_endpoint_edit = QLineEdit("192.168.2.110::3992")
        default_programmer = os.environ.get(ENV_PROGRAMMER) or str(find_quartus_pgm() or "quartus_pgm")
        self.fpga_programmer_edit = QLineEdit(default_programmer)
        self.fpga_cable_edit = QLineEdit(os.environ.get(ENV_CABLE, ""))

        fields = [
            ("System", self.system_combo),
            ("Workflow", self.workflow_combo),
            ("Excitation", self.excitation_spin),
            ("Digital delay offset", self.digital_delay_offset_spin),
            ("Digital delays", self.digital_delay_actual_label),
            ("Feedback interval", self.feedback_interval_spin),
            ("Collection duration", self.normal_duration_spin),
            ("Target fidelity", self.target_fidelity_spin),
            ("Efficiency guard", self.efficiency_guard_spin),
            ("Phase strategy", self.phase_strategy_combo),
            ("Hardware safety", self.strict_hardware_check),
            ("V6 controls", self.v6_controls_check),
            ("Pre-start settle", self.auto_pre_start_settle_spin),
            ("Model path", self.model_path_edit),
            ("AWG1 endpoint", self.awg1_endpoint_edit),
            ("AWG2 endpoint", self.awg2_endpoint_edit),
            ("Quartus programmer", self.fpga_programmer_edit),
            ("FPGA cable", self.fpga_cable_edit),
        ]
        for row, (label, widget) in enumerate(fields):
            run_layout.addWidget(QLabel(label), row, 0)
            run_layout.addWidget(widget, row, 1)

        main.addWidget(run_group)
        self._update_digital_delay_actual_label()

        state_group = QGroupBox("State")
        state_layout = QGridLayout(state_group)
        self.amp_spins: List[QDoubleSpinBox] = []
        self.phase_spins: List[QDoubleSpinBox] = []
        for i in range(4):
            amp = QDoubleSpinBox()
            amp.setRange(0.0, 5.0)
            amp.setSingleStep(0.05)
            amp.setValue(1.0)
            phase = QDoubleSpinBox()
            phase.setRange(-360.0, 360.0)
            phase.setSingleStep(1.0)
            phase.setValue(0.0)
            amp.valueChanged.connect(self._on_main_state_changed)
            phase.valueChanged.connect(self._on_main_state_changed)
            self.amp_spins.append(amp)
            self.phase_spins.append(phase)
            state_layout.addWidget(QLabel(f"Amp {i + 1}"), 0, i)
            state_layout.addWidget(amp, 1, i)
            state_layout.addWidget(QLabel(f"Phase {i + 1} deg"), 2, i)
            state_layout.addWidget(phase, 3, i)
        main.addWidget(state_group)

        self.v6_episode_panel = V6EpisodePanel(self)
        self.v6_episode_panel.action_requested.connect(self._on_v6_episode_action)
        main.addWidget(self.v6_episode_panel)

        sequence_group = QGroupBox("Sequence")
        sequence_layout = QGridLayout(sequence_group)
        self.sequence_output_edit = QLineEdit()
        self.sequence_output_edit.setReadOnly(True)
        self.sequence_status_label = QLabel("Sequence UI: not open")
        self.btn_open_sequence = QPushButton("Open Sequence UI")
        self.btn_open_sequence.clicked.connect(self.open_sequence_ui)
        self.btn_generate_sequence = QPushButton("Generate / Upload")
        self.btn_generate_sequence.clicked.connect(self.generate_sequence_in_ui)
        sequence_layout.addWidget(QLabel("Last output"), 0, 0)
        sequence_layout.addWidget(self.sequence_output_edit, 0, 1, 1, 3)
        sequence_layout.addWidget(self.btn_open_sequence, 1, 0)
        sequence_layout.addWidget(self.btn_generate_sequence, 1, 1)
        sequence_layout.addWidget(self.sequence_status_label, 1, 2, 1, 2)
        self.auto_start_after_upload_check = QCheckBox("Auto-start manual window after AWG upload")
        self.auto_start_after_upload_check.setChecked(False)
        sequence_layout.addWidget(self.auto_start_after_upload_check, 2, 0, 1, 4)
        main.addWidget(sequence_group)

        data_group = QGroupBox("Data")
        data_layout = QGridLayout(data_group)
        self.data_port_combo = QComboBox()
        self.data_port_combo.currentTextChanged.connect(self._on_main_com_changed)
        self.btn_refresh_ports = QPushButton("Refresh Ports")
        self.btn_refresh_ports.clicked.connect(self.refresh_data_ports)
        self.btn_open_data = QPushButton("Open Data UI")
        self.btn_open_data.clicked.connect(self.open_data_ui)
        self.btn_data_open_port = QPushButton("Open Selected COM")
        self.btn_data_open_port.clicked.connect(self.open_data_port)
        self.btn_data_close_port = QPushButton("Close COM")
        self.btn_data_close_port.clicked.connect(self.close_data_port)
        self.btn_data_start = QPushButton("Send y / Start FPGA Output")
        self.btn_data_start.clicked.connect(self.start_data_output)
        self.btn_data_stop = QPushButton("Send n / Stop FPGA Output")
        self.btn_data_stop.clicked.connect(self.stop_data_output)
        self.btn_data_save = QPushButton("Save Snapshot")
        self.btn_data_save.clicked.connect(self.save_data_snapshot)
        self.btn_data_clear = QPushButton("Clear UI Counters")
        self.btn_data_clear.clicked.connect(self.clear_data_window)
        self.btn_data_prepare = QPushButton("Save + Clear + Send y")
        self.btn_data_prepare.clicked.connect(self.prepare_data_window_from_ui)
        self.data_status_label = QLabel("Data UI: not open")
        data_layout.addWidget(QLabel("COM"), 0, 0)
        data_layout.addWidget(self.data_port_combo, 0, 1)
        data_layout.addWidget(self.btn_refresh_ports, 0, 2)
        data_layout.addWidget(self.btn_open_data, 0, 3)
        data_layout.addWidget(self.btn_data_open_port, 1, 0)
        data_layout.addWidget(self.btn_data_close_port, 1, 1)
        data_layout.addWidget(self.btn_data_start, 1, 2)
        data_layout.addWidget(self.btn_data_stop, 1, 3)
        data_layout.addWidget(self.btn_data_save, 2, 0)
        data_layout.addWidget(self.btn_data_clear, 2, 1)
        data_layout.addWidget(self.btn_data_prepare, 2, 2)
        data_layout.addWidget(self.data_status_label, 2, 3)
        main.addWidget(data_group)

        auto_group = QGroupBox("Automation")
        auto_layout = QGridLayout(auto_group)
        self.auto_cycle_check = QCheckBox("Auto cycle states after each clean measurement window")
        self.auto_repeat_check = QCheckBox("Loop schedule")
        self.auto_repeat_check.setChecked(False)
        self.auto_state_count_spin = QSpinBox()
        self.auto_state_count_spin.setRange(1, MAX_AUTO_STATES)
        self.auto_state_count_spin.setValue(MAX_AUTO_STATES)
        self.schedule_file_edit = QLineEdit()
        self.schedule_file_edit.setPlaceholderText("Automation scheme JSON file")
        self.btn_browse_schedule_file = QPushButton("Browse JSON")
        self.btn_browse_schedule_file.clicked.connect(self.browse_auto_schedule_file)
        self.btn_latest_schedule_file = QPushButton("Use Latest 1000")
        self.btn_latest_schedule_file.clicked.connect(self.use_latest_auto_schedule_file)
        self.btn_generate_1000_schedule = QPushButton("Generate JSON")
        self.btn_generate_1000_schedule.clicked.connect(self.generate_auto_schedule_file)
        self.btn_preview_schedule = QPushButton("Preview Schedule")
        self.btn_preview_schedule.clicked.connect(self.preview_auto_schedule)
        self.btn_start_auto = QPushButton("Start Auto Cycle")
        self.btn_start_auto.clicked.connect(self.start_auto_cycle)
        self.btn_resume_auto = QPushButton("Resume...")
        self.btn_resume_auto.clicked.connect(self.show_auto_resume_dialog)
        self.fpga_auto_program_check = QCheckBox("Program FPGA after each AWG upload")
        self.fpga_auto_program_check.setChecked(True)
        self.fpga_auto_program_check.stateChanged.connect(lambda _state: self.refresh_dependency_report())
        self.btn_program_fpga = QPushButton("Program FPGA Now")
        self.btn_program_fpga.clicked.connect(self.program_fpga_now)
        self.fpga_status_label = QLabel("FPGA: idle")
        self.auto_progress_label = QLabel("Resume: no active automation plan")
        self.schedule_text = QTextEdit()
        self.schedule_text.setMaximumHeight(110)
        self.schedule_text.setPlaceholderText(
            "Read-only preview / legacy inline JSON. Automation start prefers the scheme JSON file above."
        )
        auto_layout.addWidget(self.auto_cycle_check, 0, 0, 1, 2)
        auto_layout.addWidget(self.auto_repeat_check, 0, 2)
        auto_layout.addWidget(QLabel("Scheme JSON"), 1, 0)
        auto_layout.addWidget(self.schedule_file_edit, 1, 1, 1, 2)
        auto_layout.addWidget(self.btn_browse_schedule_file, 1, 3)
        auto_layout.addWidget(QLabel("Generate states"), 2, 0)
        auto_layout.addWidget(self.auto_state_count_spin, 2, 1)
        auto_layout.addWidget(self.btn_generate_1000_schedule, 2, 2)
        auto_layout.addWidget(self.btn_latest_schedule_file, 2, 3)
        auto_layout.addWidget(self.btn_preview_schedule, 3, 0)
        auto_layout.addWidget(self.btn_start_auto, 3, 1)
        auto_layout.addWidget(self.btn_resume_auto, 3, 2)
        auto_layout.addWidget(self.btn_program_fpga, 3, 3)
        auto_layout.addWidget(self.fpga_auto_program_check, 4, 0, 1, 2)
        auto_layout.addWidget(self.fpga_status_label, 4, 2, 1, 2)
        auto_layout.addWidget(self.auto_progress_label, 5, 0, 1, 4)
        auto_layout.addWidget(self.schedule_text, 6, 0, 1, 4)
        main.addWidget(auto_group)

        retrieval_group = QGroupBox("Retrieval Rate Calibration")
        retrieval_layout = QGridLayout(retrieval_group)
        self.btn_retrieval_upload = QPushButton("Sequence Upload")
        self.btn_retrieval_upload.clicked.connect(self._on_retrieval_calibration_upload)
        self.btn_retrieval_data_ui = QPushButton("Data UI: Off")
        self.btn_retrieval_data_ui.clicked.connect(self._on_retrieval_data_ui_toggle)
        self.retrieval_status_label = QLabel("Status: idle")
        retrieval_layout.addWidget(self.btn_retrieval_upload, 0, 0)
        retrieval_layout.addWidget(self.btn_retrieval_data_ui, 0, 1)
        retrieval_layout.addWidget(self.retrieval_status_label, 0, 2)
        main.addWidget(retrieval_group)

        dependency_group = QGroupBox("Dependency Report")
        dependency_layout = QVBoxLayout(dependency_group)
        dependency_buttons = QHBoxLayout()
        self.btn_refresh_dependencies = QPushButton("Refresh Dependencies")
        self.btn_refresh_dependencies.clicked.connect(self.refresh_dependency_report)
        dependency_buttons.addWidget(self.btn_refresh_dependencies)
        dependency_buttons.addStretch()
        self.dependency_text = QTextEdit()
        self.dependency_text.setReadOnly(True)
        self.dependency_text.setMaximumHeight(150)
        dependency_layout.addLayout(dependency_buttons)
        dependency_layout.addWidget(self.dependency_text)
        main.addWidget(dependency_group)

        buttons = QHBoxLayout()
        self.btn_awg_ready = QPushButton("Start Experiment Window")
        self.btn_awg_ready.clicked.connect(self.start_real_collection_window)
        self.btn_dry_run = QPushButton("Mock Dry Run")
        self.btn_dry_run.clicked.connect(self.start_dry_run)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_runner)
        self.btn_stop.setEnabled(False)
        buttons.addWidget(self.btn_awg_ready)
        buttons.addWidget(self.btn_dry_run)
        buttons.addWidget(self.btn_stop)
        main.addLayout(buttons)

        self.log_text = TerminalLoggingTextEdit()
        self.log_text.setReadOnly(True)
        main.addWidget(self.log_text)
        self.excitation_spin.valueChanged.connect(self._on_main_state_changed)
        self._sync_mode_count()
        self._set_default_auto_schedule_file(log=False)
        self.refresh_dependency_report()

    def startup_open_child_uis(self) -> None:
        config = self._build_config()
        try:
            self._open_sequence_ui_without_generation(config)
            self.log_text.append(f"Startup opened {config.system_type} sequence UI without generation.")
        except Exception as exc:
            self.log_text.append(f"Startup sequence UI open failed: {exc}")
        try:
            self._open_data_ui_without_com(config)
            self._sync_data_ports_from_child()
            self.log_text.append(f"Startup opened {config.system_type} data UI. COM remains closed.")
        except Exception as exc:
            self.log_text.append(f"Startup data UI open failed: {exc}")
        self.refresh_dependency_report()

    def open_sequence_ui(self):
        config = self._build_config()
        self._open_sequence_ui_without_generation(config)
        self.log_text.append(f"Opened {config.system_type} sequence UI with current target state.")
        self.refresh_dependency_report()

    def _open_sequence_ui_without_generation(self, config: ExperimentRunConfig):
        window = self.sequence_controller.open(config.system_type, config.state_schedule[0], config.excitation_mode)
        self._wire_sequence_state_sync(window, config.system_type)
        self._sync_sequence_from_main()
        self._update_sequence_status()
        return window

    def generate_sequence_in_ui(self):
        config = self._build_config()
        if self._is_v4_4q_campaign():
            config.campaign_run_dir = str(self._ensure_auto_run_dir())
            config.campaign_id = self.auto_run_dir.name
            self.real_runner = ExperimentRunner(config)
            output_dir = self.real_runner.prepare_sequence()
            realized_state = self.real_runner.config.state_schedule[0]
            if self.auto_running and 0 <= self.auto_state_index < len(self.auto_schedule):
                provenance_keys = (
                    "requested_phases_deg", "actual_loaded_phases_deg", "actual_loaded_phases_rad",
                    "fourier_guard_phase_adjustment_deg", "fourier_guard_attempt_count",
                    "fourier_guard_passed", "fourier_guard_report",
                )
                self.auto_schedule[self.auto_state_index].update({
                    key: realized_state[key] for key in provenance_keys if key in realized_state
                })
            self.sequence_controller.apply_state(realized_state, config.excitation_mode)
            if hasattr(self.sequence_controller, "refresh_generated_output"):
                self.sequence_controller.refresh_generated_output(output_dir)
            self._record_campaign_active_probe()
            self.last_sequence_dir = output_dir
            self.sequence_output_edit.setText(output_dir or "")
            self.log_text.append(
                f"V4 campaign sequence generated through SequenceService. Output: {output_dir}"
            )
            self._start_awg_upload(self.real_runner.config, output_dir)
            self._update_sequence_status()
            self.refresh_dependency_report()
            return
        if self.sequence_controller.window is None or self.sequence_controller.system_type != config.system_type:
            self._open_sequence_ui_without_generation(config)
            self.log_text.append(f"Opened {config.system_type} sequence UI with current target state.")
        else:
            self._sync_sequence_from_main()
        output_dir = self.sequence_controller.generate_current()
        self.last_sequence_dir = output_dir
        self.sequence_output_edit.setText(output_dir or "")
        self.log_text.append(f"Sequence UI generation requested. Output: {output_dir or 'unknown'}")
        if not output_dir:
            self.log_text.append("Sequence output directory was not reported; AWG upload was not started.")
            if self.auto_running:
                self._stop_auto_cycle("Auto cycle stopped because sequence generation did not report an output directory.")
            self._update_sequence_status()
            self.refresh_dependency_report()
            return
        # Output dir is valid: actually start the AWG upload (was previously dead code).
        self._start_awg_upload(config, output_dir)
        self._update_sequence_status()
        self.refresh_dependency_report()

    def program_fpga_now(self):
        config = self._build_config()
        self._start_fpga_programming(config.system_type, continue_auto_after=False)

    def _start_fpga_programming(
        self,
        system_type: str,
        *,
        continue_auto_after: bool,
        auto_after_fpga_action: str = "generate_sequence",
    ) -> bool:
        if self.fpga_thread is not None and self.fpga_thread.isRunning():
            self.log_text.append("FPGA programming is already running.")
            return False
        self._continue_auto_after_fpga = bool(continue_auto_after)
        self._auto_after_fpga_action = auto_after_fpga_action if continue_auto_after else ""
        self._last_fpga_program_success = False
        self.fpga_status_label.setText(f"FPGA: programming {system_type}...")
        self.btn_program_fpga.setEnabled(False)
        self.btn_generate_sequence.setEnabled(False)
        self.btn_start_auto.setEnabled(False)
        self.fpga_thread = FpgaProgramThread(
            system_type,
            self.fpga_programmer_edit.text().strip(),
            self.fpga_cable_edit.text().strip(),
        )
        self.fpga_thread.log.connect(self.log_text.append)
        self.fpga_thread.programmed.connect(self._on_fpga_programmed)
        self.fpga_thread.failed.connect(self._on_fpga_failed)
        self.fpga_thread.finished.connect(self._on_fpga_thread_finished)
        self.fpga_thread.start()
        self.refresh_dependency_report()
        return True

    def _on_fpga_programmed(self, result: Dict):
        self._last_fpga_program_success = True
        self.fpga_status_label.setText(f"FPGA: programmed {result.get('system_type', self.system_combo.currentText())}")
        attempt = int(result.get("attempt", 1) or 1)
        max_attempts = int(result.get("max_attempts", 1) or 1)
        self.log_text.append(
            f"FPGA programming complete for {result.get('system_type')}: "
            f"{result.get('bitstream_path')} (attempt {attempt}/{max_attempts})"
        )

    def _on_fpga_failed(self, message: str):
        self._last_fpga_program_success = False
        self._continue_auto_after_fpga = False
        self._auto_after_fpga_action = ""
        self.fpga_status_label.setText("FPGA: programming failed")
        self.log_text.append(f"FPGA programming failed: {message}")
        if self.auto_running:
            self._stop_auto_cycle("Auto cycle stopped after FPGA programming failure.")

    def _on_fpga_thread_finished(self):
        self.fpga_thread = None
        should_continue = self._continue_auto_after_fpga and self._last_fpga_program_success
        action = self._auto_after_fpga_action
        self._continue_auto_after_fpga = False
        self._auto_after_fpga_action = ""
        self.refresh_dependency_report()
        if should_continue and self.auto_running:
            if action == "start_data":
                self._schedule_auto_data_start()
            else:
                self.generate_sequence_in_ui()

    def _start_awg_upload(self, config: ExperimentRunConfig, output_dir: str):
        if self.awg_thread is not None and self.awg_thread.isRunning():
            self.log_text.append("AWG upload is already running.")
            return
        self.awg_uploader.awg1_ip = self.awg1_endpoint_edit.text().strip()
        self.awg_uploader.awg2_ip = self.awg2_endpoint_edit.text().strip()
        digital_delays = digital_delay_values(config.digital_delay_offset_ns)
        metadata = {
            "system_type": config.system_type,
            "mode": config.mode,
            "workflow_mode": config.workflow_mode,
            "state": config.state_schedule[0],
            "excitation_mode": config.excitation_mode,
            "strict_hardware_mode": config.strict_hardware_mode,
            "digital_delay_offset_ns": config.digital_delay_offset_ns,
            "w_ch1_digital_delay_ns": digital_delays["W_CH1"],
            "r_ch1_digital_delay_ns": digital_delays["R_CH1"],
            "channel_digital_delay_ns": dict(digital_delays),
            "phase_probe": (
                dict(self.real_runner.last_phase_probe)
                if self.real_runner is not None and self.real_runner.last_phase_probe
                else None
            ),
        }
        self.btn_generate_sequence.setEnabled(False)
        self.btn_awg_ready.setEnabled(False)
        self.btn_start_auto.setEnabled(False)
        self.awg_thread = AwgUploadThread(self.awg_uploader, output_dir, metadata)
        self.awg_thread.log.connect(self.log_text.append)
        self.awg_thread.uploaded.connect(self._on_awg_uploaded)
        self.awg_thread.failed.connect(self._on_awg_failed)
        self.awg_thread.finished.connect(self._on_awg_thread_finished)
        self.awg_thread.start()
        self.refresh_dependency_report()

    def _on_awg_uploaded(self, result: Dict):
        detail = f", T={result.get('time_duration')}" if result.get('time_duration') else ""
        self.log_text.append(f"AWG upload complete via {result.get('backend', 'unknown')}{detail}.")
        if (
            self.auto_running
            and self.real_runner is not None
            and getattr(self.real_runner, "probe_recovery_reset_pending", False)
        ):
            self.real_runner.probe_recovery_reset_pending = False
            self.log_text.append(
                "Recovered an unfinished probe: unprobed reset sequence uploaded; "
                "regenerating the scheduled state now."
            )
            self.generate_sequence_in_ui()
            return
        if self.auto_running and self._is_v4_4q_campaign() and self.real_runner is not None:
            try:
                self.real_runner.mark_active_probe_uploaded()
            except Exception as exc:
                self._stop_auto_cycle(
                    f"Auto cycle stopped because phase-probe upload evidence failed: {exc}"
                )
                return
        if self.auto_running:
            if self.fpga_auto_program_check.isChecked():
                config = self._build_config()
                state_name = config.state_schedule[0].get("name", "state")
                self.log_text.append(f"Auto: programming FPGA for {state_name} after AWG upload...")
                started = self._start_fpga_programming(
                    config.system_type,
                    continue_auto_after=True,
                    auto_after_fpga_action="start_data",
                )
                if not started:
                    self._stop_auto_cycle("Auto cycle stopped because FPGA programming could not start.")
            else:
                self._schedule_auto_data_start()
        elif self.auto_start_after_upload_check.isChecked():
            if not self.start_real_collection_window():
                self.log_text.append("Data UI or COM port was not ready for collection window.")
        else:
            self.log_text.append(
                "AWG upload complete. COM must be open; click Start Experiment Window to begin data collection."
            )
        self._update_sequence_status()
        self.refresh_dependency_report()

    def _schedule_auto_data_start(self) -> None:
        if not self.auto_running:
            return
        settle = self._auto_pre_start_settle_s()
        self.log_text.append(
            f"Auto: pre-start settle for {settle:.0f}s before triggering data Start..."
        )
        QTimer.singleShot(int(settle * 1000), self._auto_start_data_window)

    def _auto_pre_start_settle_s(self) -> float:
        """Settle time between AWG upload and data Start. Default 30s."""
        try:
            return float(self.auto_pre_start_settle_spin.value())
        except Exception:
            return 30.0

    def _auto_start_data_window(self) -> None:
        """Triggered by QTimer after the settle delay. Starts the data window."""
        if not self.auto_running:
            return  # user pressed Stop during the settle wait
        if not self.start_real_collection_window():
            self._stop_auto_cycle("Auto cycle stopped because the data UI or COM port was not ready.")

    def _on_awg_failed(self, message: str):
        self.log_text.append(f"AWG upload failed: {message}")
        self.log_text.append("Data UI was not cleared and counting was not restarted.")
        if self.auto_running:
            self._stop_auto_cycle("Auto cycle stopped after AWG upload failure.")
        self.refresh_dependency_report()

    def _on_awg_thread_finished(self):
        self.awg_thread = None
        self.refresh_dependency_report()

    def open_data_ui(self):
        config = self._build_config()
        self._open_data_ui_without_com(config)
        self._sync_data_ports_from_child()
        self.log_text.append(f"Opened {config.system_type} data UI. COM and acquisition controls are available here.")
        self.refresh_dependency_report()

    def _open_data_ui_without_com(self, config: ExperimentRunConfig):
        window = self.data_source.open(config.system_type, self._feedback_context(config))
        self._update_data_status_label()
        return window

    def refresh_data_ports(self):
        if self._ensure_data_ui_open():
            self._sync_data_ports_from_child()
            self._update_data_status_label()
            self.log_text.append("COM port list refreshed.")
            self.refresh_dependency_report()

    def open_data_port(self):
        if not self._ensure_data_ui_open():
            return
        self._on_main_com_changed(self.data_port_combo.currentText())
        if not self._compute_experiment_readiness().can_open_com:
            self.log_text.append("Cannot open COM: data UI is not open, no valid COM is selected, or COM is already open.")
            self.refresh_dependency_report()
            return
        result = self.data_source.open_port(self.data_port_combo.currentText())
        self.log_text.append(result.message)
        self._update_data_status_label()
        self.refresh_dependency_report()

    def close_data_port(self):
        if not self._ensure_data_ui_open():
            return
        result = self.data_source.close_port()
        self.log_text.append(result.message)
        self._update_data_status_label()
        self.refresh_dependency_report()

    def start_data_output(self):
        if not self._ensure_data_ui_open():
            return
        if not self._compute_experiment_readiness().com_open:
            self.log_text.append("Cannot send y/start: COM port is not open.")
            self.refresh_dependency_report()
            return
        result = self.data_source.start_output()
        self.log_text.append(result.message)
        self._update_data_status_label()
        self.refresh_dependency_report()

    def stop_data_output(self):
        if not self._ensure_data_ui_open():
            return
        if not self._compute_experiment_readiness().com_open:
            self.log_text.append("Cannot send n/stop: COM port is not open.")
            self.refresh_dependency_report()
            return
        result = self.data_source.stop_output()
        self.log_text.append(result.message)
        self._update_data_status_label()
        self.refresh_dependency_report()

    def save_data_snapshot(self):
        if not self._ensure_data_ui_open():
            return
        result = self.data_source.save_current()
        self.log_text.append(result.message)
        self._update_data_status_label()

    def clear_data_window(self):
        if not self._ensure_data_ui_open():
            return
        result = self.data_source.clear_data()
        self.log_text.append(result.message)
        self._update_data_status_label()

    def prepare_data_window_from_ui(self):
        if not self._ensure_data_ui_open():
            return
        if not self._compute_experiment_readiness().com_open:
            self.log_text.append("Cannot prepare data window: COM port is not open.")
            self.refresh_dependency_report()
            return
        if self.auto_running and hasattr(self.data_source, "clear_start"):
            result = self.data_source.clear_start()
        else:
            result = self.data_source.save_clear_start()
        self.log_text.append(result.message)
        self._update_data_status_label()
        self.refresh_dependency_report()

    def start_real_collection_window(self):
        config = self._build_config()
        context_config = (
            self.real_runner.config
            if self._is_v4_4q_campaign()
            and self.real_runner is not None
            and self.real_runner.config.system_type == config.system_type
            else config
        )
        self.data_source.set_feedback_context(self._feedback_context(context_config))
        readiness = self._compute_experiment_readiness()
        if not readiness.data_ui_open:
            self.log_text.append("Cannot start experiment window: Data UI is not open.")
            self.refresh_dependency_report()
            return False
        if not readiness.com_open:
            self.log_text.append("Cannot start experiment window: COM port is not open.")
            self.refresh_dependency_report()
            return False
        if not readiness.awg_upload_idle:
            self.log_text.append("Cannot start experiment window: AWG upload is still running.")
            self.refresh_dependency_report()
            return False
        if not readiness.measurement_idle:
            self.log_text.append("Cannot start experiment window: measurement window is already running.")
            self.refresh_dependency_report()
            return False
        result = self.data_source.save_clear_start()
        self.log_text.append(result.message)
        self._update_data_status_label()
        if not result.success:
            self.refresh_dependency_report()
            return False
        duration = config.collection_duration_s if config.workflow_mode == "collect_dataset" else config.feedback_interval_s
        if not (
            self.auto_running
            and self._is_v4_4q_campaign()
            and self.real_runner is not None
        ):
            self.real_runner = ExperimentRunner(config)
        else:
            self.real_runner.config.collection_duration_s = duration
        if self.auto_running and self._is_v4_4q_campaign():
            try:
                self.real_runner.mark_active_probe_collecting()
            except Exception as exc:
                try:
                    self.data_source.stop_output()
                except Exception:
                    pass
                self.log_text.append(
                    f"Cannot start authoritative collection: phase-probe lifecycle failed: {exc}"
                )
                self.refresh_dependency_report()
                return False
        self.accumulator = MeasurementAccumulator(config.system_type, duration)
        self.log_text.append(
            f"Collecting real data window for {duration:.0f}s in {config.workflow_mode}. "
            "Frames are taken from the opened data UI."
        )
        self.refresh_dependency_report()
        return True

    def _prepare_data_ui_for_next_window(self, system_type: str) -> bool:
        return self.start_real_collection_window()

    def start_dry_run(self):
        config = self._build_config()
        config.use_mock_data = True
        config.strict_hardware_mode = False
        self.btn_dry_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_text.append(
            f"Starting MOCK {config.workflow_mode} dry cycle for {config.system_type}; "
            "no data UI, serial port, or AWG hardware is used."
        )
        self.runner_thread = RunnerThread(config)
        self.runner_thread.log.connect(self.log_text.append)
        self.runner_thread.failed.connect(self._on_failed)
        self.runner_thread.finished_snapshot.connect(self._on_finished)
        self.runner_thread.start()

    def stop_runner(self):
        if self.auto_running:
            self._stop_auto_cycle("Auto cycle stopped by user.")
        self._continue_auto_after_fpga = False
        self._auto_after_fpga_action = ""
        if self.runner_thread and self.runner_thread.isRunning():
            self.runner_thread.requestInterruption()
            self.runner_thread.quit()
            self.runner_thread.wait(1000)
        if self.fpga_thread and self.fpga_thread.isRunning():
            self.fpga_thread.requestInterruption()
            self.log_text.append("FPGA programming is already in progress; the Quartus command will finish in the background.")
        self.accumulator = None
        self.btn_dry_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.refresh_dependency_report()

    def _on_failed(self, message: str):
        self.log_text.append(f"Error: {message}")
        self.btn_dry_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.auto_running:
            self._stop_auto_cycle("Auto cycle stopped after runner error.")

    def _on_finished(self, snapshot: Dict):
        self.log_text.append(
            "Mock snapshot saved: "
            f"fidelity={snapshot.get('fidelity')} "
            f"retrieval_rate={snapshot.get('retrieval_rate')} "
            f"deltas={len(snapshot.get('deltas', []))}"
        )
        if snapshot.get("sample_path"):
            self.log_text.append(f"Feedback sample saved: {snapshot['sample_path']}")
        self.btn_dry_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_data_frame(self, frame: Dict):
        if self.accumulator is None:
            self._update_data_status_label()
            self._update_experiment_readiness()
            return
        self.accumulator.add_frame(frame)
        if self.accumulator.frames == 1 or self.accumulator.frames % 20 == 0:
            self.log_text.append(
                f"Collected frames={self.accumulator.frames}, "
                f"read={self.accumulator.read_count}, write={self.accumulator.write_count}"
            )
        if self.accumulator.ready():
            snapshot = self.accumulator.snapshot()
            self.accumulator = None
            self._process_real_snapshot(snapshot)
            self.refresh_dependency_report()
            return
        self._update_data_status_label()
        self._update_experiment_readiness()

    def _process_real_snapshot(self, snapshot):
        if self.real_runner is None:
            self.real_runner = ExperimentRunner(self._build_config())
        if self._is_v4_4q_campaign():
            snapshot.duration_s = V4_4Q_CAMPAIGN_DURATION_S
        try:
            if self.real_runner.config.workflow_mode == "collect_dataset":
                result = self.real_runner.process_dataset_snapshot(snapshot)
                self.log_text.append(
                    f"Dataset sample complete: fidelity={result.fidelity}, "
                    f"retrieval_rate={result.retrieval_rate}, path={self.real_runner.last_sample_path}"
                )
            else:
                result = self.real_runner.process_feedback_snapshot(snapshot)
                guard = self.real_runner.last_prediction.guard_result if self.real_runner.last_prediction else {}
                self.log_text.append(
                    f"Feedback window complete: fidelity={result.fidelity}, "
                    f"retrieval_rate={result.retrieval_rate}, deltas={len(result.deltas)}, guard={guard}"
                )
            if self.real_runner.config.legacy_feedback_sample_export:
                self._save_training_sample(result)
            # Auto cycle: stop the data UI's output so the next state can
            # take a clean window, then advance.
            if self.auto_running:
                try:
                    stop_result = self.data_source.stop_output()
                    self.log_text.append(stop_result.message)
                except Exception as exc:
                    self._stop_auto_cycle(f"Auto cycle stopped after data stop error: {exc}")
                    return
                try:
                    if self._is_v4_4q_campaign():
                        self._enrich_authoritative_window(result)
                    save_result = self.data_source.save_current()
                    self.log_text.append(save_result.message)
                except Exception as exc:
                    self._stop_auto_cycle(f"Auto cycle stopped after data save error: {exc}")
                    return
                if not save_result.success:
                    self._stop_auto_cycle("Auto cycle stopped because current data could not be saved.")
                    return
                if self._is_v4_4q_campaign():
                    snapshot_path = (getattr(save_result, "payload", None) or {}).get("path")
                    if not snapshot_path:
                        snapshot_path = getattr(save_result, "saved_path", None)
                    if not snapshot_path:
                        self._stop_auto_cycle(
                            "Auto cycle stopped because the authoritative snapshot path is missing."
                        )
                        return
                    try:
                        manifest = load_snapshot_manifest(snapshot_path)
                        self.real_runner.finalize_authoritative_snapshot(
                            snapshot_path,
                            sample_id=manifest.sample_id,
                        )
                        self._record_campaign_sample(manifest, Path(snapshot_path))
                        self.real_runner.clear_finalized_probe()
                        self._clear_campaign_active_probe_after_commit(manifest.sample_id)
                    except Exception as exc:
                        self._stop_auto_cycle(
                            f"Auto cycle stopped during authoritative snapshot commit: {exc}"
                        )
                        return
                try:
                    clear_result = self.data_source.clear_data()
                    self.log_text.append(clear_result.message)
                except Exception as exc:
                    self._stop_auto_cycle(f"Auto cycle stopped after data clear error: {exc}")
                    return
                if not clear_result.success:
                    self._stop_auto_cycle("Auto cycle stopped because data UI counters could not be cleared.")
                    return
                self._advance_auto_cycle_after_success()
            else:
                self.log_text.append(
                    "State saved. Modify the target state, regenerate sequence, then start the next window."
                )
        except Exception as exc:
            self.log_text.append(f"Real snapshot processing error: {exc}")
            if self.auto_running:
                self._stop_auto_cycle("Auto cycle stopped after snapshot processing error.")

    def _enrich_authoritative_window(self, result) -> None:
        window = self.data_source.window if self.data_source is not None else None
        current = getattr(window, "current_data", None) if window is not None else None
        if not isinstance(current, dict):
            raise RuntimeError("data UI has no current_data for the authoritative V4 snapshot")
        if float(result.duration_s) != V4_4Q_CAMPAIGN_DURATION_S:
            raise RuntimeError("V4 campaign result must have exactly 1200 s exposure")
        if not result.operation_point_registry_hash or not result.calibration_snapshot_hash:
            raise RuntimeError("V4 strict collection requires registry and calibration hashes")
        if not result.fpga_protocol_version:
            raise RuntimeError("V4 strict collection requires an explicit FPGA protocol version")
        waveform_dir = Path(self.real_runner.last_sequence_dir or "")
        if not waveform_dir.exists() or not any(path.is_file() for path in waveform_dir.rglob("*")):
            raise RuntimeError("V4 strict collection requires generated AWG waveform files")
        target = self.real_runner.current_state_params()
        is_anchor = target.get("state_role") == "anchor"
        if is_anchor != bool(self.real_runner.last_phase_probe):
            raise RuntimeError("anchor/probe lifecycle does not match the scheduled state role")
        current["duration_s"] = float(result.duration_s)
        current["started_at"] = float(result.timestamp) - float(result.duration_s)
        current["signal_bins"] = np.asarray(result.signal_bins)
        current["idler_bins"] = np.asarray(result.idler_bins)
        current["coin_bins"] = np.asarray(result.coin_bins)
        current["signal_bins_raw"] = np.asarray(
            result.signal_bins_raw if result.signal_bins_raw is not None else result.signal_bins
        )
        current["idler_bins_raw"] = np.asarray(
            result.idler_bins_raw if result.idler_bins_raw is not None else result.idler_bins
        )
        current["coin_bins_raw"] = np.asarray(
            result.coin_bins_raw if result.coin_bins_raw is not None else result.coin_bins
        )
        if result.density_matrix is not None:
            current["rho_calibrated"] = np.asarray(result.density_matrix, dtype=complex)
        noise_aware = dict(result.diagnostics.get("noise_aware_reconstruction") or {})
        posterior = noise_aware.get("posterior_mean_rho_full")
        if posterior is not None:
            if isinstance(posterior, dict) and {"real", "imag"} <= set(posterior):
                posterior = np.asarray(posterior["real"]) + 1j * np.asarray(posterior["imag"])
            current["rho_posterior_mean"] = np.asarray(posterior, dtype=complex)
        elif result.density_matrix is not None:
            current["rho_posterior_mean"] = np.asarray(result.density_matrix, dtype=complex)
        latest_raw = getattr(window, "latest_raw_rho", None)
        if latest_raw is not None:
            current["rho_raw"] = np.asarray(latest_raw, dtype=complex)

        context = self._feedback_context(self.real_runner.config)
        matrix_hash = stable_json_hash(
            matrix_hash_snapshot(self.real_runner.sequence_service.matrix_dir)
        )
        context.update(
            {
                "operation_registry_hash": result.operation_point_registry_hash,
                "calibration_snapshot_hash": result.calibration_snapshot_hash,
                "matrix_snapshot_hash": matrix_hash,
                "phase_probe": dict(self.real_runner.last_phase_probe or {}),
                "awg_state": {
                    "waveform_dir": self.real_runner.last_sequence_dir,
                    "waveform_hash": self.real_runner._waveform_directory_hash(
                        self.real_runner.last_sequence_dir or ""
                    ),
                    "probe_applied": bool(self.real_runner.last_phase_probe),
                },
            }
        )
        current["feedback_context"] = context
        current["operation_registry_hash"] = result.operation_point_registry_hash
        current["calibration_snapshot_hash"] = result.calibration_snapshot_hash
        current["matrix_snapshot_hash"] = matrix_hash
        current["model_schema_hash"] = stable_json_hash(
            {
                "feature_schema_version": result.feature_schema_version,
                "system_type": result.system_type,
            }
        )
        current["phase_probe"] = dict(self.real_runner.last_phase_probe or {})
        probe_store = self.real_runner.phase_probe_store
        if (
            self.real_runner.last_phase_probe
            and probe_store is not None
            and probe_store.active_record() is not None
        ):
            pointer = probe_store.active_record() or {}
            if pointer.get("lifecycle_status") != "collecting":
                raise RuntimeError("active phase probe is not in the collecting lifecycle state")
            contract = probe_store.load_active_contract()
            expected_probe = dict(self.real_runner.last_phase_probe)
            for field_name in (
                "plan_id",
                "anchor_id",
                "repeat_index",
                "code_pair_id",
                "pair_sign",
                "phase_tensor_sha256",
                "operation_registry_hash",
            ):
                if expected_probe.get(field_name) != getattr(contract, field_name):
                    raise RuntimeError(f"active phase probe does not match generated state: {field_name}")
            current["phase_probe_tensor"] = probe_store.load_active_tensor()
        current["awg_state"] = dict(context["awg_state"])
        masks = dict(result.missing_value_masks or {})
        masks["temperature_c"] = current.get("temperature_c") is None
        current["missing_value_masks"] = masks
        if current.get("temperature_c") is None:
            current.pop("temperature_c", None)

    def _record_campaign_sample(self, snapshot_manifest, snapshot_path: Path) -> None:
        if not self._is_v4_4q_campaign():
            return
        path = self._ensure_auto_run_dir() / "campaign_manifest.json"
        campaign = CampaignManifestV1.load(path)
        ledger = campaign.states[self.auto_state_index]
        if ledger.get("status") == "completed":
            raise RuntimeError(
                f"campaign state {self.auto_state_index + 1} is already completed; duplicate blocked"
            )
        if float(snapshot_manifest.duration_s) != V4_4Q_CAMPAIGN_DURATION_S:
            raise RuntimeError("authoritative snapshot does not contain the fixed 1200 s exposure")
        expected_state = self.auto_schedule[self.auto_state_index]
        if snapshot_manifest.campaign_id != campaign.campaign_id:
            raise RuntimeError("snapshot campaign_id does not match the resume ledger")
        if snapshot_manifest.plan_id != campaign.plan_id:
            raise RuntimeError("snapshot plan_id does not match the resume ledger")
        if int(snapshot_manifest.state_index or -1) != self.auto_state_index + 1:
            raise RuntimeError("snapshot state_index does not match the active resume position")
        if snapshot_manifest.state_role != expected_state.get("state_role"):
            raise RuntimeError("snapshot state_role does not match the automation plan")
        expected_anchor = expected_state.get("anchor_id")
        if (snapshot_manifest.anchor_id or None) != (
            None if expected_anchor is None else str(expected_anchor)
        ):
            raise RuntimeError("snapshot anchor_id does not match the automation plan")
        if not snapshot_path.is_dir() or not (snapshot_path / "manifest.json").is_file():
            raise RuntimeError("campaign snapshot path is not a completed manifest directory")
        existing_ids = {
            str(item.get("sample_id"))
            for item in campaign.states
            if item.get("sample_id")
        }
        if snapshot_manifest.sample_id in existing_ids:
            raise RuntimeError(f"duplicate sample_id blocked: {snapshot_manifest.sample_id}")
        ledger.update(
            {
                "status": "completed",
                "completed_at": time.time(),
                "sample_id": snapshot_manifest.sample_id,
                "snapshot_path": str(snapshot_path),
                "duration_s": float(snapshot_manifest.duration_s),
                "hashes": {
                    "schema": getattr(snapshot_manifest, "schema_hash", None),
                    "registry": snapshot_manifest.operation_registry_hash,
                    "calibration": snapshot_manifest.calibration_snapshot_hash,
                    "matrix": snapshot_manifest.matrix_snapshot_hash,
                    "model_schema": snapshot_manifest.model_schema_hash,
                    "awg": (snapshot_manifest.awg_state or {}).get("waveform_hash"),
                },
            }
        )
        store = self.real_runner.phase_probe_store if self.real_runner is not None else None
        campaign.active_probe = store.active_record() if store is not None else None
        campaign.next_resume_state_index = min(
            self.auto_state_index + 2,
            len(campaign.states),
        )
        campaign.save(path)

    def _clear_campaign_active_probe_after_commit(self, sample_id: str) -> None:
        """Finish the campaign half of the two-file probe commit transaction."""

        if not self._is_v4_4q_campaign():
            return
        path = self._ensure_auto_run_dir() / "campaign_manifest.json"
        campaign = CampaignManifestV1.load(path)
        ledger = campaign.states[self.auto_state_index]
        if ledger.get("status") != "completed" or ledger.get("sample_id") != sample_id:
            raise RuntimeError("campaign ledger was not committed before phase-probe clear")
        store = self.real_runner.phase_probe_store if self.real_runner is not None else None
        if store is not None and store.active_record() is not None:
            raise RuntimeError("phase-probe pointer still exists after finalized clear")
        campaign.active_probe = None
        campaign.save(path)

    def _record_campaign_active_probe(self) -> None:
        if not self._is_v4_4q_campaign():
            return
        path = self._ensure_auto_run_dir() / "campaign_manifest.json"
        campaign = CampaignManifestV1.load(path)
        ledger = campaign.states[self.auto_state_index]
        if ledger.get("status") == "completed":
            raise RuntimeError("completed campaign state cannot be regenerated")
        ledger["status"] = "running"
        ledger["started_at"] = time.time()
        store = self.real_runner.phase_probe_store if self.real_runner is not None else None
        campaign.active_probe = store.active_record() if store is not None else None
        campaign.next_resume_state_index = self.auto_state_index + 1
        campaign.save(path)

    def _save_training_sample(self, result) -> None:
        """Persist raw CSV + ideal state + entanglement classification.

        Combines the just-completed measurement window (still on the data UI
        as ``current_data``) with the ExperimentRunner's reconstructed density
        matrix and ideal target state, then writes a FeedbackSample via
        ``save_feedback_sample_from_window``. Result is also appended to
        ``ml_recon/data/feedback_samples/{3qubit,4qubit}/`` for offline
        feedback-model training.
        """
        window = self.data_source.window if self.data_source is not None else None
        if window is None:
            self.log_text.append("Training sample skipped: data UI window is gone.")
            return
        try:
            from system_control.data_ui.feedback_dataset_actions import (
                save_feedback_sample_from_window,
            )
        except Exception as exc:
            self.log_text.append(f"Training sample skipped: import error {exc!r}")
            return
        try:
            # Inject rho_measured into the window so the helper can pick it up.
            if getattr(result, "density_matrix", None) is not None:
                window.latest_rho = np.asarray(result.density_matrix, dtype=complex)
                window.latest_raw_rho = window.latest_rho
            save_result = save_feedback_sample_from_window(
                window, self.system_combo.currentText()
            )
            self.log_text.append(save_result.message)
            if not save_result.success:
                self.log_text.append(
                    "Training sample save failed; auto cycle will continue."
                )
        except Exception as exc:
            self.log_text.append(f"Training sample save error (ignored): {exc!r}")

    def browse_auto_schedule_file(self):
        start_dir = self._automation_plan_root()
        start_dir.mkdir(parents=True, exist_ok=True)
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select Automation Scheme JSON",
            str(start_dir),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        self.schedule_file_edit.setText(path)
        self.log_text.append(f"Automation scheme JSON selected: {path}")

    def use_latest_auto_schedule_file(self):
        path = self._latest_1000_schedule_file(self.system_combo.currentText())
        if path is None:
            self.log_text.append("No saved 1000-state automation plan found; generating one now.")
            path = self._write_generated_auto_schedule_file(MAX_AUTO_STATES)
        self.schedule_file_edit.setText(str(path))
        self.log_text.append(f"Using latest 1000-state automation scheme: {path}")

    def generate_auto_schedule_file(self):
        count = int(self.auto_state_count_spin.value())
        path = self._write_generated_auto_schedule_file(count)
        self.schedule_file_edit.setText(str(path))
        self.log_text.append(f"Generated automation scheme JSON: {path}")
        self.preview_auto_schedule()

    def preview_auto_schedule(self):
        try:
            schedule = self._auto_schedule_from_ui()
        except Exception as exc:
            self.log_text.append(f"Auto schedule error: {exc}")
            return
        preview = schedule if len(schedule) <= 40 else schedule[:40]
        preview_text = schedule_to_json(preview)
        if len(preview) < len(schedule):
            preview_text += f"\n\n... previewing first {len(preview)} of {len(schedule)} states ..."
        self.schedule_text.setPlainText(preview_text)
        source = self.auto_schedule_source_description or "generated schedule"
        self.log_text.append(f"Auto schedule ready: {len(schedule)} states from {source}.")

    def show_auto_resume_dialog(self):
        try:
            schedule = self._auto_schedule_from_ui()
            self.auto_schedule = schedule
            self._restore_latest_compatible_auto_progress()
        except Exception as exc:
            self.log_text.append(f"Resume unavailable: {exc}")
            return
        if not schedule:
            self.log_text.append("Resume unavailable: automation schedule is empty.")
            return

        total = len(schedule)
        recommended = min(max(self.auto_resume_index + 1, 1), total)
        dialog = QDialog(self)
        dialog.setWindowTitle("Resume Automation")
        layout = QFormLayout(dialog)
        status = QLabel(
            f"Plan has {total} states. Resume will start from state {recommended}; "
            "completed earlier states are skipped."
        )
        status.setWordWrap(True)
        resume_combo = QComboBox()
        for index, state in enumerate(schedule, start=1):
            name = state.get("name", f"state {index}")
            resume_combo.addItem(f"{index:04d} - {name}", index)
        resume_combo.setCurrentIndex(recommended - 1)
        layout.addRow(status)
        layout.addRow("Resume from JSON state", resume_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec_() == QDialog.Accepted:
            resume_from = resume_combo.currentData() or (resume_combo.currentIndex() + 1)
            self.start_auto_cycle(resume_from=int(resume_from))

    def _restore_latest_compatible_auto_progress(self) -> Optional[Dict]:
        """Reconnect Resume to the newest interrupted run for the selected source JSON."""
        source_path = self.auto_schedule_source_path
        if source_path is None or not self.auto_schedule:
            return None
        source_key = os.path.normcase(str(source_path.resolve()))
        run_root = self._repo_root() / "data" / "automation_runs"
        if not run_root.exists():
            return None
        candidates = sorted(
            run_root.glob("*/automation_progress.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for progress_path in candidates:
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if progress.get("status") not in {"interrupted", "running", "completed"}:
                    continue
                if str(progress.get("system_type")) != self.system_combo.currentText():
                    continue
                if int(progress.get("state_count", 0)) != len(self.auto_schedule):
                    continue
                recorded_source = progress.get("source_plan_path")
                if not recorded_source:
                    continue
                if os.path.normcase(str(Path(recorded_source).resolve())) != source_key:
                    continue
                run_dir = Path(progress.get("run_dir") or progress_path.parent).resolve()
                if run_dir != progress_path.parent.resolve():
                    continue
                plan_path = run_dir / "automation_plan.json"
                if not plan_path.exists() or not self._is_schedule_file_for(
                    plan_path,
                    self.system_combo.currentText(),
                    len(self.auto_schedule),
                ):
                    continue
                resume_number = int(progress.get("next_resume_state_index", 1))
                if self._is_v4_4q_campaign():
                    campaign_path = run_dir / "campaign_manifest.json"
                    if not campaign_path.exists():
                        continue
                    campaign = CampaignManifestV1.load(campaign_path)
                    if campaign.state_count != len(self.auto_schedule):
                        continue
                    resume_number = int(campaign.next_resume_state_index or resume_number)
                    if not 1 <= resume_number <= len(campaign.states):
                        continue
                    if campaign.states[resume_number - 1].get("status") == "completed":
                        continue
                if not 1 <= resume_number <= len(self.auto_schedule):
                    continue
                self.auto_run_dir = run_dir
                self.auto_resume_index = resume_number - 1
                self._update_auto_progress_label(progress)
                self.log_text.append(
                    f"Restored interrupted automation run {run_dir.name}; "
                    f"resume from state {resume_number}."
                )
                return progress
            except Exception:
                continue
        return None

    def start_auto_cycle(self, resume_from: Optional[int] = None):
        self.auto_cycle_check.setChecked(True)
        items = self.refresh_dependency_report()
        if has_blocking_missing(items):
            self.log_text.append("Auto cycle not started: required dependencies are missing.")
            return
        readiness = self._compute_experiment_readiness()
        if not readiness.can_start_auto_cycle:
            reasons = " ".join(readiness.blocking_reasons) or "experiment is not ready."
            self.log_text.append(f"Auto cycle not started: {reasons}")
            return
        status = self.data_source.status() if self.data_source is not None else {}
        try:
            self.auto_schedule = self._auto_schedule_from_ui()
        except Exception as exc:
            self.log_text.append(f"Auto cycle not started: {exc}")
            return
        if (
            self.system_combo.currentText() == "4qubit"
            and len(self.auto_schedule) == MAX_AUTO_STATES
        ):
            try:
                self._validate_v4_4q_campaign_schedule()
            except Exception as exc:
                self.log_text.append(f"Auto cycle not started: {exc}")
                return
            self.normal_duration_spin.setValue(V4_4Q_CAMPAIGN_DURATION_S)
            self.strict_hardware_check.setChecked(True)
            self.auto_repeat_check.setChecked(False)
        if resume_from is None:
            start_index = 0
            self.auto_run_dir = None
        else:
            start_index = int(resume_from) - 1
            if start_index < 0 or start_index >= len(self.auto_schedule):
                self.log_text.append(
                    f"Auto cycle not started: resume state must be 1..{len(self.auto_schedule)}."
                )
                return
        self.auto_running = True
        self.auto_state_index = start_index
        self.auto_resume_index = start_index
        self.btn_stop.setEnabled(True)
        self._ensure_auto_run_dir()
        self._apply_state_to_spins(self.auto_schedule[self.auto_state_index])
        state_name = self.auto_schedule[self.auto_state_index].get("name", "state")
        window_s = self.normal_duration_spin.value()
        resume_detail = "" if resume_from is None else f" Resuming from state {resume_from}."
        self.log_text.append(
            f"Auto cycle started with {len(self.auto_schedule)} states. "
            f"Current state: {state_name}. Window={window_s:.0f}s, "
            f"pre-start settle={self._auto_pre_start_settle_s():.0f}s, "
            f"FPGA per-state={'on' if self.fpga_auto_program_check.isChecked() else 'off'}, "
            f"COM={status.get('port') or '?'} open. "
            f"Scheme={self.auto_schedule_source_description or 'unknown'}.{resume_detail}"
        )
        self.refresh_dependency_report()
        # Per-state auto pipeline:
        # generate sequence -> AWG upload -> optional FPGA program -> data start.
        self._start_auto_state()

    def _advance_auto_cycle_after_success(self):
        if not self.auto_running:
            return
        self._record_auto_progress("completed")
        next_index = self.auto_state_index + 1
        if next_index >= len(self.auto_schedule):
            if not self.auto_repeat_check.isChecked():
                self._stop_auto_cycle("Auto cycle completed all scheduled states.")
                return
            next_index = 0
        self.auto_state_index = next_index
        self.auto_resume_index = next_index
        state = self.auto_schedule[self.auto_state_index]
        self._apply_state_to_spins(state)
        self.log_text.append(
            f"Auto cycle advancing to {state.get('name', f'state {self.auto_state_index + 1}')} "
            f"({self.auto_state_index + 1}/{len(self.auto_schedule)})."
        )
        self._start_auto_state()

    def _stop_auto_cycle(self, reason: str):
        completed_plan = "completed all scheduled states" in reason.lower()
        if self.auto_schedule:
            if completed_plan:
                self.auto_resume_index = 0
                self._record_auto_progress("completed_plan")
            else:
                self.auto_resume_index = min(self.auto_state_index, len(self.auto_schedule) - 1)
                self._record_auto_progress("interrupted")
        self.auto_running = False
        self.btn_stop.setEnabled(False)
        self.log_text.append(reason)
        self.refresh_dependency_report()

    def _start_auto_state(self) -> None:
        if not self.auto_running:
            return
        state = self.auto_schedule[self.auto_state_index]
        self._apply_state_to_spins(state)
        self._record_auto_progress("running")
        self.generate_sequence_in_ui()

    def _ensure_auto_run_dir(self) -> Path:
        if self.auto_run_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            root = repo_root / "data" / "automation_runs"
            root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            system_type = self.system_combo.currentText()
            self.auto_run_dir = root / f"{stamp}_{system_type}_{len(self.auto_schedule)}states"
            self.auto_run_dir.mkdir(parents=True, exist_ok=False)
            self._write_auto_plan()
        return self.auto_run_dir

    def _write_auto_plan(self) -> None:
        if self.auto_run_dir is None:
            return
        plan = {
            "system_type": self.system_combo.currentText(),
            "state_count": len(self.auto_schedule),
            "algorithm": (
                "deterministic_low_discrepancy_coverage_v2"
                if any(state.get("state_role") == "anchor" for state in self.auto_schedule)
                else "deterministic_low_discrepancy_coverage_v1"
            ),
            "created_at": time.time(),
            "source_plan_path": str(self.auto_schedule_source_path) if self.auto_schedule_source_path else None,
            "source_plan_description": self.auto_schedule_source_description,
            "states": self.auto_schedule,
        }
        plan_path = self.auto_run_dir / "automation_plan.json"
        atomic_write_json(plan_path, plan)
        if self._is_v4_4q_campaign():
            states = []
            for index, state in enumerate(self.auto_schedule, start=1):
                probe = dict(state.get("phase_probe") or {})
                states.append(
                    {
                        "state_index": index,
                        "name": state.get("name"),
                        "state_role": state.get("state_role"),
                        "anchor_id": state.get("anchor_id"),
                        "phase_probe": probe,
                        "status": "pending",
                        "sample_id": None,
                        "snapshot_path": None,
                        "hashes": {},
                    }
                )
            manifest = CampaignManifestV1(
                campaign_id=self.auto_run_dir.name,
                system_type="4qubit",
                plan_id=str(
                    next(
                        (
                            state.get("plan_id")
                            for state in self.auto_schedule
                            if state.get("plan_id")
                        ),
                        "deterministic_low_discrepancy_coverage_v2",
                    )
                ),
                measurement_duration_s=V4_4Q_CAMPAIGN_DURATION_S,
                state_count=len(states),
                coverage_count=sum(item["state_role"] == "coverage" for item in states),
                zero_anchor_count=sum(
                    item["state_role"] == "anchor"
                    and int(item["phase_probe"].get("pair_sign", 0)) == 0
                    for item in states
                ),
                paired_probe_anchor_count=sum(
                    item["state_role"] == "anchor"
                    and int(item["phase_probe"].get("pair_sign", 0)) != 0
                    for item in states
                ),
                states=states,
            )
            manifest.save(self.auto_run_dir / "campaign_manifest.json")

    def _record_auto_progress(self, status: str) -> None:
        if not self.auto_schedule:
            return
        try:
            run_dir = self._ensure_auto_run_dir()
            current_number = self.auto_state_index + 1
            if status == "completed_plan":
                next_resume_number = 1
            elif status == "completed":
                next_resume_number = min(current_number + 1, len(self.auto_schedule))
            else:
                next_resume_number = min(max(self.auto_resume_index + 1, 1), len(self.auto_schedule))
            state = dict(self.auto_schedule[self.auto_state_index])
            progress = {
                "status": status,
                "updated_at": time.time(),
                "system_type": self.system_combo.currentText(),
                "state_count": len(self.auto_schedule),
                "current_state_index": current_number,
                "current_state_name": state.get("name"),
                "next_resume_state_index": next_resume_number,
                "run_dir": str(run_dir),
                "plan_path": str(run_dir / "automation_plan.json"),
                "source_plan_path": str(self.auto_schedule_source_path) if self.auto_schedule_source_path else None,
            }
            atomic_write_json(run_dir / "automation_progress.json", progress)
            self._update_auto_progress_label(progress)
        except Exception as exc:
            self.log_text.append(f"Auto progress save warning: {exc}")

    def _is_v4_4q_campaign(self) -> bool:
        return bool(
            self.system_combo.currentText() == "4qubit"
            and len(self.auto_schedule) == MAX_AUTO_STATES
            and any(state.get("state_role") == "anchor" for state in self.auto_schedule)
        )

    def _validate_v4_4q_campaign_schedule(self) -> None:
        if self.system_combo.currentText() != "4qubit":
            raise ValueError("the V4 1000-state hardware campaign is 4qubit-only")
        if self.workflow_combo.currentText() != "collect_dataset":
            raise ValueError("the V4 1000-state campaign must use collect_dataset workflow")
        validate_v4_4qubit_campaign_schedule(self.auto_schedule)

    def _update_auto_progress_label(self, progress: Dict) -> None:
        status = progress.get("status", "unknown")
        current = progress.get("current_state_index", "?")
        total = progress.get("state_count", "?")
        resume = progress.get("next_resume_state_index", "?")
        run_dir = progress.get("run_dir", "")
        if status == "completed_plan":
            text = f"Resume: plan complete ({total}/{total}). Run: {run_dir}"
        elif status == "interrupted":
            text = f"Resume: interrupted at state {current}/{total}; resume from {resume}. Run: {run_dir}"
        elif status == "completed":
            text = f"Resume: state {current}/{total} saved; next resume point {resume}. Run: {run_dir}"
        else:
            text = f"Resume: running state {current}/{total}; resume from {resume} if interrupted. Run: {run_dir}"
        self.auto_progress_label.setText(text)

    # ---- Retrieval Rate Calibration handlers ----

    def _on_retrieval_calibration_upload(self):
        if self._retrieval_upload_thread is not None and self._retrieval_upload_thread.isRunning():
            self.log_text.append("Retrieval calibration upload is already running.")
            return
        self.btn_retrieval_upload.setEnabled(False)
        self.retrieval_status_label.setText("Status: uploading...")
        awg1_ip = self.awg1_endpoint_edit.text().strip()
        awg2_ip = self.awg2_endpoint_edit.text().strip()
        self._retrieval_upload_thread = RetrievalCalibrationUploadThread(awg1_ip, awg2_ip)
        self._retrieval_upload_thread.log.connect(self.log_text.append)
        self._retrieval_upload_thread.uploaded.connect(self._on_retrieval_upload_done)
        self._retrieval_upload_thread.failed.connect(self._on_retrieval_upload_failed)
        self._retrieval_upload_thread.start()

    def _on_retrieval_upload_done(self):
        self.btn_retrieval_upload.setEnabled(True)
        self.retrieval_status_label.setText("Status: upload complete")

    def _on_retrieval_upload_failed(self, message: str):
        self.btn_retrieval_upload.setEnabled(True)
        self.retrieval_status_label.setText("Status: upload failed")
        self.log_text.append(f"Retrieval calibration upload failed: {message}")

    def _on_retrieval_data_ui_toggle(self):
        if self._retrieval_data_ui_process is not None and self._retrieval_data_ui_process.poll() is None:
            # Process is running — terminate it
            self._retrieval_data_ui_process.terminate()
            try:
                self._retrieval_data_ui_process.wait(timeout=3)
            except Exception:
                self._retrieval_data_ui_process.kill()
            self._retrieval_data_ui_process = None
            self.btn_retrieval_data_ui.setText("Data UI: Off")
            self.retrieval_status_label.setText("Status: data UI closed")
            self.log_text.append("Retrieval rate calibration data UI closed.")
        else:
            # Launch the data UI
            repo_root = Path(__file__).resolve().parents[2]
            script = repo_root / 'system_control' / 'ui_heat_all_256.py'
            if not script.exists():
                self.log_text.append(f"Data UI script not found: {script}")
                return
            self._retrieval_data_ui_process = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(repo_root),
            )
            self.btn_retrieval_data_ui.setText("Data UI: On")
            self.retrieval_status_label.setText("Status: data UI running")
            self.log_text.append(f"Launched retrieval rate calibration data UI (PID={self._retrieval_data_ui_process.pid}).")

    def closeEvent(self, event):
        """Clean up subprocesses on window close."""
        if self._retrieval_data_ui_process is not None and self._retrieval_data_ui_process.poll() is None:
            self._retrieval_data_ui_process.terminate()
            try:
                self._retrieval_data_ui_process.wait(timeout=3)
            except Exception:
                self._retrieval_data_ui_process.kill()
        super().closeEvent(event)

    def _auto_schedule_from_ui(self):
        system_type = self.system_combo.currentText()
        file_text = self.schedule_file_edit.text().strip() if hasattr(self, "schedule_file_edit") else ""
        if file_text:
            path = Path(file_text)
            schedule = load_state_schedule_file(path, system_type)
            self.auto_schedule_source_path = path
            self.auto_schedule_source_description = str(path)
            return self._normalize_auto_schedule_provenance(schedule, system_type)

        latest_path = self._latest_1000_schedule_file(system_type)
        if latest_path is not None:
            self.schedule_file_edit.setText(str(latest_path))
            schedule = load_state_schedule_file(latest_path, system_type)
            self.auto_schedule_source_path = latest_path
            self.auto_schedule_source_description = f"latest 1000-state plan {latest_path}"
            return self._normalize_auto_schedule_provenance(schedule, system_type)

        text = self.schedule_text.toPlainText().strip()
        if text and "... previewing first" not in text:
            schedule = parse_state_schedule(text, system_type)
            path = self._write_auto_schedule_payload(
                schedule,
                algorithm="legacy_inline_schedule",
                suffix="inline",
            )
            self.schedule_file_edit.setText(str(path))
            schedule = load_state_schedule_file(path, system_type)
            self.auto_schedule_source_path = path
            self.auto_schedule_source_description = f"converted inline schedule {path}"
            return self._normalize_auto_schedule_provenance(schedule, system_type)

        path = self._write_generated_auto_schedule_file(MAX_AUTO_STATES)
        self.schedule_file_edit.setText(str(path))
        schedule = load_state_schedule_file(path, system_type)
        self.auto_schedule_source_path = path
        self.auto_schedule_source_description = f"generated 1000-state plan {path}"
        return self._normalize_auto_schedule_provenance(schedule, system_type)

    @staticmethod
    def _normalize_auto_schedule_provenance(
        schedule: List[Dict], system_type: str
    ) -> List[Dict]:
        """Bind coverage and anchor states to one in-memory campaign plan ID."""

        normalized = [dict(state) for state in schedule]
        if system_type != "4qubit" or len(normalized) != MAX_AUTO_STATES:
            return normalized
        canonical_plan_id = next(
            (
                str(state["plan_id"])
                for state in normalized
                if state.get("plan_id") not in (None, "")
            ),
            "deterministic_low_discrepancy_coverage_v2",
        )
        for state in normalized:
            state.setdefault("plan_id", canonical_plan_id)
        return normalized

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _automation_plan_root(self) -> Path:
        return self._repo_root() / "data" / "automation_plans"

    def _write_generated_auto_schedule_file(self, count: int) -> Path:
        system_type = self.system_combo.currentText()
        count = max(1, min(MAX_AUTO_STATES, int(count)))
        if system_type == "4qubit" and count == MAX_AUTO_STATES:
            schedule = generate_drift_aware_coverage_schedule(
                system_type,
                self._current_ui_state(),
                count,
                anchor_count=60,
            )
            algorithm = "deterministic_low_discrepancy_coverage_v2"
            suffix = "940coverage_60anchors_v4"
        else:
            schedule = generate_coverage_schedule(
                system_type,
                self._current_ui_state(),
                count,
            )
            algorithm = "deterministic_low_discrepancy_coverage_v1"
            suffix = ""
        return self._write_auto_schedule_payload(
            schedule,
            algorithm=algorithm,
            suffix=suffix,
        )

    def _write_auto_schedule_payload(
        self,
        schedule: List[Dict],
        *,
        algorithm: str,
        suffix: str = "",
    ) -> Path:
        system_type = self.system_combo.currentText()
        plan_root = self._automation_plan_root()
        plan_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        extra = f"_{suffix}" if suffix else ""
        path = plan_root / f"{stamp}_{system_type}_{len(schedule)}states{extra}.json"
        payload = {
            "system_type": system_type,
            "state_count": len(schedule),
            "algorithm": algorithm,
            "created_at": time.time(),
            "states": schedule,
        }
        if system_type == "4qubit" and len(schedule) == MAX_AUTO_STATES:
            validate_v4_4qubit_campaign_schedule(schedule)
            payload.update(
                {
                    "plan_id": next(
                        (
                            state.get("plan_id")
                            for state in schedule
                            if state.get("plan_id")
                        ),
                        "deterministic_low_discrepancy_coverage_v2",
                    ),
                    "campaign_composition": {
                        "coverage_states": V4_4Q_COVERAGE_STATES,
                        "zero_probe_anchors": V4_4Q_ZERO_PROBE_ANCHORS,
                        "signed_probe_anchors": V4_4Q_SIGNED_PROBE_ANCHORS,
                        "signed_probe_pairs": V4_4Q_SIGNED_PROBE_ANCHORS // 2,
                    },
                }
            )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.auto_schedule_source_path = path
        self.auto_schedule_source_description = str(path)
        return path

    def _latest_1000_schedule_file(self, system_type: str) -> Optional[Path]:
        candidates: List[Path] = []
        plan_root = self._automation_plan_root()
        if plan_root.exists():
            candidates.extend(plan_root.glob("*.json"))
        run_root = self._repo_root() / "data" / "automation_runs"
        if run_root.exists():
            candidates.extend(run_root.glob("*/automation_plan.json"))
        candidates = sorted(
            (path for path in candidates if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            if self._is_schedule_file_for(path, system_type, MAX_AUTO_STATES):
                return path
        return None

    def _is_schedule_file_for(self, path: Path, system_type: str, state_count: int) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload_system = payload.get("system_type")
                if payload_system and str(payload_system) != system_type:
                    return False
                states = payload.get("states")
            else:
                states = payload
            if not isinstance(states, list) or len(states) != int(state_count):
                return False
            schedule = load_state_schedule_file(path, system_type)
            if system_type == "4qubit" and int(state_count) == MAX_AUTO_STATES:
                validate_v4_4qubit_campaign_schedule(schedule)
            return True
        except Exception:
            return False

    def _set_default_auto_schedule_file(self, *, log: bool = True) -> None:
        if not hasattr(self, "schedule_file_edit"):
            return
        if self.schedule_file_edit.text().strip():
            return
        path = self._latest_1000_schedule_file(self.system_combo.currentText())
        if path is None:
            return
        self.schedule_file_edit.setText(str(path))
        if log and hasattr(self, "log_text"):
            self.log_text.append(f"Default automation scheme JSON: {path}")

    def _apply_auto_schedule_for_selected_system(self) -> None:
        """Select the authoritative campaign JSON when the measurement system changes."""

        if not hasattr(self, "schedule_file_edit"):
            return
        system_type = self.system_combo.currentText()
        current_text = self.schedule_file_edit.text().strip()

        if system_type == "4qubit":
            if current_text and self._is_schedule_file_for(
                Path(current_text), system_type, MAX_AUTO_STATES
            ):
                return
            path = self._latest_1000_schedule_file(system_type)
            generated = path is None
            if path is None:
                path = self._write_generated_auto_schedule_file(MAX_AUTO_STATES)
            self.schedule_file_edit.setText(str(path))
            self.schedule_text.clear()
            if hasattr(self, "log_text"):
                action = "Generated and auto-applied" if generated else "Auto-applied"
                self.log_text.append(
                    f"{action} 4qubit 940+60 campaign JSON: {path}"
                )
            return

        if current_text:
            try:
                payload = json.loads(Path(current_text).read_text(encoding="utf-8"))
                states = payload.get("states") if isinstance(payload, dict) else payload
                if isinstance(states, list) and self._is_schedule_file_for(
                    Path(current_text), system_type, len(states)
                ):
                    return
            except Exception:
                pass
            self.schedule_file_edit.clear()
        self._set_default_auto_schedule_file(log=hasattr(self, "log_text"))

    def _current_ui_state(self) -> Dict:
        system_type = self.system_combo.currentText()
        n = 3 if system_type == "3qubit" else 4
        return {
            "name": "ui_state",
            "amplitudes": [spin.value() for spin in self.amp_spins[:n]],
            "phases_deg": [spin.value() for spin in self.phase_spins[:n]],
        }

    def _apply_state_to_spins(self, state: Dict):
        self._state_syncing = True
        try:
            self._apply_state_to_spins_unchecked(state)
        finally:
            self._state_syncing = False
        self._sync_sequence_from_main()

    def _apply_state_to_spins_unchecked(self, state: Dict):
        for idx, value in enumerate(state.get("amplitudes", [])):
            if idx < len(self.amp_spins):
                self.amp_spins[idx].setValue(float(value))
        for idx, value in enumerate(state.get("phases_deg", [])):
            if idx < len(self.phase_spins):
                self.phase_spins[idx].setValue(float(value))

    def _on_main_state_changed(self, *args):
        if self._state_syncing:
            return
        self._sync_sequence_from_main()

    def _sync_sequence_from_main(self):
        if self._state_syncing:
            return
        if self.sequence_controller.window is None:
            self.sequence_controller.apply_digital_delay_offset(self._digital_delay_offset_ns())
            return
        if self.sequence_controller.system_type != self.system_combo.currentText():
            self.sequence_controller.apply_digital_delay_offset(self._digital_delay_offset_ns())
            return
        self._state_syncing = True
        try:
            self.sequence_controller.apply_state(self._current_ui_state(), self.excitation_spin.value())
            self.sequence_controller.apply_digital_delay_offset(self._digital_delay_offset_ns())
        finally:
            self._state_syncing = False

    def _digital_delay_offset_ns(self) -> int:
        return int(self.digital_delay_offset_spin.value())

    def _digital_delay_values(self) -> Dict[str, int]:
        return digital_delay_values(self._digital_delay_offset_ns())

    def _update_digital_delay_actual_label(self) -> None:
        delays = self._digital_delay_values()
        w_values = "/".join(str(delays[f"W_CH{index}"]) for index in range(1, 5))
        r_values = "/".join(str(delays[f"R_CH{index}"]) for index in range(1, 5))
        self.digital_delay_actual_label.setText(
            f"W {w_values} ns / R {r_values} ns"
        )

    def _on_digital_delay_offset_changed(self, *_args) -> None:
        self._update_digital_delay_actual_label()
        self._sync_sequence_from_main()

    def _wire_sequence_state_sync(self, window, system_type: str):
        if window is None:
            return
        n = 3 if system_type == "3qubit" else 4
        for idx in range(1, n + 1):
            for name in (f"spin_amp{idx}", f"spin_phase{idx}"):
                widget = getattr(window, name, None)
                if widget is not None and not getattr(widget, "_experiment_sync_connected", False):
                    widget.valueChanged.connect(self._on_sequence_state_changed)
                    widget._experiment_sync_connected = True
        combo = getattr(window, "combo_excitation_mode", None)
        if combo is not None and not getattr(combo, "_experiment_sync_connected", False):
            combo.currentIndexChanged.connect(self._on_sequence_state_changed)
            combo._experiment_sync_connected = True

    def _on_sequence_state_changed(self, *args):
        if self._state_syncing:
            return
        window = self.sequence_controller.window
        if window is None:
            return
        system_type = self.sequence_controller.system_type or self.system_combo.currentText()
        n = 3 if system_type == "3qubit" else 4
        state = {"name": "ui_state", "amplitudes": [], "phases_deg": []}
        for idx in range(1, n + 1):
            amp = getattr(window, f"spin_amp{idx}", None)
            phase = getattr(window, f"spin_phase{idx}", None)
            state["amplitudes"].append(float(amp.value()) if amp is not None else 1.0)
            state["phases_deg"].append(float(phase.value()) if phase is not None else 0.0)
        combo = getattr(window, "combo_excitation_mode", None)
        if combo is not None:
            excitation_mode = int(combo.currentIndex()) + 1
        else:
            excitation_mode = int(getattr(window, "w_ch1_excitation_mode", self.excitation_spin.value()))
        self._state_syncing = True
        try:
            self._apply_state_to_spins_unchecked(state)
            self.excitation_spin.setValue(excitation_mode)
        finally:
            self._state_syncing = False

    def _compute_experiment_readiness(self) -> ExperimentUiReadiness:
        return build_experiment_readiness(
            self.sequence_controller.status(),
            self.data_source.status(),
            selected_com_port=self.data_port_combo.currentText(),
            fpga_programming_running=self.fpga_thread is not None and self.fpga_thread.isRunning(),
            awg_upload_running=self.awg_thread is not None and self.awg_thread.isRunning(),
            measurement_running=self.accumulator is not None,
            auto_running=self.auto_running,
            last_sequence_dir=self.last_sequence_dir,
        )

    def _update_experiment_readiness(self) -> ExperimentUiReadiness:
        readiness = self._compute_experiment_readiness()
        self._latest_readiness = readiness
        self.btn_generate_sequence.setEnabled(readiness.can_generate)
        self.btn_data_open_port.setEnabled(readiness.can_open_com)
        self.btn_data_close_port.setEnabled(readiness.com_open)
        self.btn_data_start.setEnabled(readiness.com_open)
        self.btn_data_stop.setEnabled(readiness.com_open)
        self.btn_data_prepare.setEnabled(readiness.com_open)
        self.btn_data_save.setEnabled(readiness.data_ui_open)
        self.btn_data_clear.setEnabled(readiness.data_ui_open)
        self.btn_awg_ready.setEnabled(readiness.can_start_manual_window)
        self.btn_start_auto.setEnabled(readiness.can_start_auto_cycle)
        self.btn_program_fpga.setEnabled(readiness.fpga_programming_idle and not self.auto_running)
        return readiness

    def refresh_dependency_report(self):
        config = self._build_config()
        feedback_context = self._feedback_context(config)
        data_status = self.data_source.status()
        awg_upload_running = self.awg_thread is not None and self.awg_thread.isRunning()
        fpga_programming_running = self.fpga_thread is not None and self.fpga_thread.isRunning()
        self._update_sequence_status()
        self._update_data_status_label()
        items = collect_dependency_items(
            model_path=self.model_path_edit.text().strip(),
            awg1_endpoint=self.awg1_endpoint_edit.text().strip(),
            awg2_endpoint=self.awg2_endpoint_edit.text().strip(),
            fpga_programming_enabled=self.fpga_auto_program_check.isChecked(),
            fpga_programming_running=fpga_programming_running,
            fpga_programmer_path=self.fpga_programmer_edit.text().strip(),
            sequence_ui_open=self.sequence_controller.window is not None,
            data_ui_open=bool(data_status["open"]),
            com_open=bool(data_status["com_open"]),
            com_port=data_status.get("port"),
            data_frame_count=data_status.get("frame_count"),
            driver_alive=data_status.get("driver_alive"),
            read_thread_alive=data_status.get("read_thread_alive"),
            last_frame_age_s=data_status.get("last_frame_age_s"),
            awg_upload_running=awg_upload_running,
            accumulator_running=self.accumulator is not None,
            last_sequence_dir=self.last_sequence_dir,
            system_type=config.system_type,
            feedback_context=feedback_context,
        )
        self.dependency_text.setPlainText(format_dependency_report(items))
        self._update_experiment_readiness()
        return items

    def _build_config(self) -> ExperimentRunConfig:
        system_type = self.system_combo.currentText()
        state = self._active_target_state()
        collection_duration = (
            V4_4Q_CAMPAIGN_DURATION_S
            if self._is_v4_4q_campaign()
            else self.normal_duration_spin.value()
        )
        return ExperimentRunConfig(
            system_type=system_type,
            workflow_mode=self.workflow_combo.currentText(),
            state_schedule=[state],
            excitation_mode=self.excitation_spin.value(),
            feedback_interval_s=self.feedback_interval_spin.value(),
            normal_state_duration_s=collection_duration,
            collection_duration_s=collection_duration,
            target_fidelity=self.target_fidelity_spin.value(),
            efficiency_drop_threshold=self.efficiency_guard_spin.value(),
            model_path=self.model_path_edit.text().strip() or display_feedback_model_path(system_type),
            feedback_backend=self._v6_feedback_backend,
            v6_checkpoint_path=self.v6_episode_panel.settings()["checkpoint_path"] or None,
            v6_normalization_path=self.v6_episode_panel.settings()["normalization_path"] or None,
            v6_config_path=self.v6_episode_panel.settings()["config_path"] or None,
            v6_shadow_output_dir=self.v6_episode_panel.settings()["output_dir"] or None,
            v6_episode_output_dir=self.v6_episode_panel.settings()["output_dir"] or None,
            v6_episode_enabled=self._v6_feedback_backend == "v6_episode",
            retrieval_map_path=None,
            phase_feedback_strategy=self.phase_strategy_combo.currentText(),
            strict_hardware_mode=self.strict_hardware_check.isChecked(),
            use_mock_data=False,
            digital_delay_offset_ns=self._digital_delay_offset_ns(),
            campaign_run_dir=str(self.auto_run_dir) if self.auto_run_dir is not None else None,
            campaign_id=self.auto_run_dir.name if self.auto_run_dir is not None else None,
            legacy_feedback_sample_export=not self._is_v4_4q_campaign(),
        )

    def _active_target_state(self) -> Dict:
        if self.auto_running and self.auto_schedule:
            if 0 <= self.auto_state_index < len(self.auto_schedule):
                return dict(self.auto_schedule[self.auto_state_index])
        return self._current_ui_state()

    def _feedback_context(self, config: ExperimentRunConfig) -> Dict:
        canonical_plan_id = config.state_schedule[0].get("plan_id")
        if canonical_plan_id in (None, "") and self.auto_schedule:
            canonical_plan_id = next(
                (
                    state.get("plan_id")
                    for state in self.auto_schedule
                    if state.get("plan_id") not in (None, "")
                ),
                None,
            )
        context = {
            "system_type": config.system_type,
            "target_state_params": config.state_schedule[0],
            "excitation_mode": config.excitation_mode,
            "workflow_mode": config.workflow_mode,
            "model_path": config.model_path,
            "feedback_backend": config.feedback_backend,
            "phase_feedback_strategy": config.phase_feedback_strategy,
            "strict_hardware_mode": config.strict_hardware_mode,
            "collection_duration_s": config.collection_duration_s,
            "window_policy": "fixed_duration",
            "campaign_id": config.campaign_id,
            "phase_probe": dict(config.state_schedule[0].get("phase_probe") or {}),
        }
        if self.auto_schedule:
            context["automation"] = {
                "state_index": self.auto_state_index + 1,
                "state_count": len(self.auto_schedule),
                "state_name": config.state_schedule[0].get("name"),
                "run_dir": str(self.auto_run_dir) if self.auto_run_dir is not None else None,
                "resume_state_index": self.auto_resume_index + 1,
                "source_plan_path": str(self.auto_schedule_source_path) if self.auto_schedule_source_path else None,
                "source_plan_description": self.auto_schedule_source_description,
                "state_role": config.state_schedule[0].get("state_role", "coverage"),
                "anchor_id": config.state_schedule[0].get("anchor_id"),
                "campaign_id": config.campaign_id,
                # Coverage states intentionally omit the repeated plan_id in
                # the 940+60 JSON.  Bind every snapshot to the one canonical
                # plan identifier carried by its anchor states/campaign.
                "plan_id": canonical_plan_id,
            }
        return context

    def set_v6_episode_action_handler(
        self,
        handler: Optional[Callable[[str, Dict], Optional[Dict]]],
    ) -> None:
        """Attach the explicit engineering-side episode coordinator."""

        self._v6_episode_action_handler = handler

    def update_v6_episode_panel(
        self, state: V6EpisodeUiStateV1 | Dict,
    ) -> None:
        self.v6_episode_panel.update_state(state)

    def _on_v6_controls_toggled(self, checked: bool) -> None:
        self.v6_episode_panel.setVisible(bool(checked))
        if not checked:
            self._v6_feedback_backend = "legacy"
            self.v6_episode_panel.update_state(V6EpisodeUiStateV1())
        self.log_text.append(
            "V6 feedback controls shown; write actions remain explicit and guarded."
            if checked
            else "V6 feedback controls hidden; backend restored to legacy."
        )

    def _on_v6_episode_action(self, action: str) -> None:
        settings = self.v6_episode_panel.settings()
        state = self.v6_episode_panel.state
        request = {
            "action": action,
            "settings": settings,
            "status": state.status,
            "current_bundle_id": state.current_bundle_id,
            "current_bundle_hash": state.current_bundle_hash,
            "best_bundle_id": state.best_bundle_id,
            "best_bundle_hash": state.best_bundle_hash,
            "desired_target": dict(state.desired_target),
            "current_applied_control": dict(state.current_applied_control),
        }
        if action in {"start_shadow", "start_gated_episode"}:
            missing = [
                name for name in ("checkpoint_path", "normalization_path", "config_path", "output_dir")
                if not settings.get(name)
            ]
            if missing:
                self.log_text.append(
                    "V6 start blocked; missing settings: " + ", ".join(missing)
                )
                return
        if action == "start_shadow" and self._v6_episode_action_handler is None:
            self._v6_feedback_backend = "v6_shadow"
            self.v6_episode_panel.update_state({
                **asdict(state),
                "status": "IDLE",
            })
            self.log_text.append(
                "V6 shadow backend armed. It will save diagnostics only; no materialization, "
                "staging, upload, or active-command change is permitted."
            )
            return
        if self._v6_episode_action_handler is None:
            self.log_text.append(
                f"V6 action '{action}' blocked: no guarded episode coordinator is attached. "
                f"Bundle summary: {state.bundle_summary()}"
            )
            return
        try:
            result = self._v6_episode_action_handler(action, request)
            if action == "start_shadow":
                self._v6_feedback_backend = "v6_shadow"
            elif action == "start_gated_episode":
                self._v6_feedback_backend = "v6_episode"
            if isinstance(result, dict):
                self.v6_episode_panel.update_state(result)
            self.log_text.append(
                f"V6 action '{action}' completed. Bundle summary: "
                f"{self.v6_episode_panel.state.bundle_summary()}"
            )
        except Exception as exc:
            self.log_text.append(
                f"V6 action '{action}' failed closed: {exc}. Bundle summary: "
                f"{state.bundle_summary()}"
            )

    def _ensure_data_ui_open(self) -> bool:
        config = self._build_config()
        if self.data_source.window is None or self.data_source.system_type != config.system_type:
            self._open_data_ui_without_com(config)
            self._sync_data_ports_from_child()
            self.refresh_dependency_report()
        else:
            self.data_source.set_feedback_context(self._feedback_context(config))
        return self.data_source.window is not None

    def _sync_data_ports_from_child(self):
        ports = self.data_source.available_ports()
        current = self.data_port_combo.currentText()
        self.data_port_combo.blockSignals(True)
        try:
            self.data_port_combo.clear()
            self.data_port_combo.addItems(ports if ports else ["COM6"])
            if current:
                self.data_port_combo.setCurrentText(current)
        finally:
            self.data_port_combo.blockSignals(False)
        self._on_main_com_changed(self.data_port_combo.currentText())

    def _on_main_com_changed(self, port: str) -> None:
        if self.data_source.window is not None:
            self.data_source.select_port(port)
        self._update_experiment_readiness()

    def _update_sequence_status(self):
        status = self.sequence_controller.status()
        if status.get("output_dir"):
            self.sequence_output_edit.setText(status["output_dir"])
        state = "open" if status["open"] else "not open"
        system = status.get("system_type") or self.system_combo.currentText()
        self.sequence_status_label.setText(f"Sequence UI: {state} ({system})")

    def _update_data_status_label(self):
        status = self.data_source.status()
        if not status["open"]:
            self.data_status_label.setText("Data UI: not open")
            return
        com = "open" if status["com_open"] else "closed"
        if status.get("frame_count") is not None:
            frame = f"frames={status['frame_count']}"
            if status.get("last_frame_age_s") is not None:
                frame += f", last={status['last_frame_age_s']:.1f}s"
        else:
            frame = "frame" if status["has_current_data"] else "no frame"
        self.data_status_label.setText(f"Data UI: {status.get('port') or '--'} {com}, {frame}")

    def _on_system_changed(self):
        self._sync_mode_count()
        self._apply_auto_schedule_for_selected_system()
        current = self.model_path_edit.text().strip()
        known_defaults = {
            LEGACY_FEEDBACK_MODEL_PATH,
            display_feedback_model_path("3qubit"),
            display_feedback_model_path("4qubit"),
        }
        if not self._model_path_user_edited or current in known_defaults:
            self.model_path_edit.setText(display_feedback_model_path(self.system_combo.currentText()))
        config = self._build_config()
        try:
            self._open_sequence_ui_without_generation(config)
        except Exception as exc:
            self.log_text.append(f"Sequence UI switch failed: {exc}")
        try:
            self._open_data_ui_without_com(config)
            self._sync_data_ports_from_child()
        except Exception as exc:
            self.log_text.append(f"Data UI switch failed: {exc}")
        self.refresh_dependency_report()

    def _sync_mode_count(self):
        n = 3 if self.system_combo.currentText() == "3qubit" else 4
        self.excitation_spin.setMaximum(n)
        for i, (amp, phase) in enumerate(zip(self.amp_spins, self.phase_spins)):
            amp.setEnabled(i < n)
            phase.setEnabled(i < n)


def main():
    app = QApplication(sys.argv)
    window = ExperimentEntryUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
