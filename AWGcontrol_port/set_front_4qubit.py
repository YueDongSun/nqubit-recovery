"""
Set Front 4 Qubit Sequences to AWGs

Uploads entangle_sequences_4qubit (W and R states) to two AWG5014C instruments:
- AWG1: 192.168.2.122::4011 (W state - CH1-4)
- AWG2: 192.168.2.110::3990 (R state - CH1-4)

Configuration:
- Sample rate: 1 GHz
- Mode: Triggered
- Trigger input: External, 1.4 V threshold, 1 kOhm impedance
- Run mode: Enabled

Available time durations:
    T266us, T1034us, T1054us

Files processed from AWG_Sequence/entangle_sequences_4qubit/:
    W_CH1_T*.txt, W_CH2_T*.txt, W_CH3_T*.txt, W_CH4_T*.txt
    R_CH1_T*.txt, R_CH2_T*.txt, R_CH3_T*.txt, R_CH4_T*.txt
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable

from awg5014c_transport import open_instrument_resource


CH1_ANALOG_AMPLITUDE_VPP = 0.4


class AWG5014CController:
    """Controller for Tektronix AWG5014C arbitrary waveform generator."""

    def __init__(self, visa_address: str, buffer_size: int = 2048):
        self.visa_address = visa_address
        self.buffer_size = buffer_size
        self.rm = None
        self.awg = None

    def connect(self):
        """Establish connection to the AWG5014C."""
        self.awg = open_instrument_resource(self.visa_address)
        self.awg.read_termination = '\n'
        print(f"[{self.visa_address}] Connected to AWG5014C")
        return self

    def disconnect(self):
        """Close the connection to the AWG."""
        if self.awg:
            self.awg.close()
        self.awg = None
        self.rm = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def query_idn(self) -> str:
        """Query instrument identification."""
        return self.awg.query('*IDN?')

    def reset(self):
        """Reset the instrument."""
        self.awg.write('*RST')
        self.awg.write('*CLS')

    def set_sample_rate(self, rate_hz: float):
        """Set the sample rate in Hz."""
        self.awg.write(f'SOURce1:FREQuency {rate_hz}')
        print(f"[{self.visa_address}] Sample rate set to {rate_hz/1e9} GHz")

    def set_run_mode(self, mode: str = 'TRIG'):
        """Set the run mode (CONT, TRIG, GAT)."""
        self.awg.write(f'AWGControl:RMODe {mode}')
        print(f"[{self.visa_address}] Run mode set to {mode}")

    def configure_trigger_input(self, source: str = 'EXT', level: float = 1.4, impedance: float = 1000.0):
        """Configure the sequence trigger input.

        AWG5014C supports the trigger sequence command family for the front
        trigger input. Valid external impedances are 50 ohm and 1 kOhm.
        """
        normalized_source = str(source).strip().upper()
        if normalized_source.startswith('EXT'):
            source_command = 'EXTernal'
        elif normalized_source.startswith('INT'):
            source_command = 'INTernal'
        else:
            raise ValueError(f"trigger source must be INT or EXT (got {source!r})")
        if source_command == 'EXTernal' and float(impedance) not in (50.0, 1000.0):
            raise ValueError(
                f"external trigger impedance must be 50 or 1000 ohm (got {impedance})"
            )

        self.awg.write(f'TRIGger:SEQuence:SOURce {source_command}')
        if source_command == 'EXTernal':
            self.awg.write(f'TRIGger:SEQuence:LEVel {float(level):g}')
            self.awg.write(f'TRIGger:SEQuence:IMPedance {float(impedance):g}')
            print(
                f"[{self.visa_address}] Trigger input set: source={source_command}, "
                f"level={float(level):g} V, impedance={float(impedance):g} ohm"
            )
        else:
            print(f"[{self.visa_address}] Trigger input set: source={source_command}")

    def set_trigger_source(self, source: str = 'INT'):
        """Backward-compatible source-only trigger helper."""
        self.configure_trigger_input(source=source)

    def set_marker_voltage(self, high: float = 2.5, low: float = 0.0):
        """Set marker output HIGH/LOW voltage level on every source & marker.

        Marker voltage is global per AWG (per source, not per channel), and each
        source has two markers (M1, M2). Valid range is [-2.7, +2.7] V for both
        levels with HIGH > LOW and (HIGH - LOW) <= 5.4 V. AWG5014C defaults are
        HIGH=1.0 V / LOW=0.0 V; we override to 2.5 V / 0.0 V for the W-state
        experiment so digital pulses drive downstream hardware at full scale.

        Args:
            high: Marker HIGH level in volts (default 2.5).
            low:  Marker LOW level in volts (default 0.0).
        """
        if high <= low:
            raise ValueError(
                f"marker HIGH ({high} V) must be > LOW ({low} V)"
            )
        if abs(high) > 2.7 or abs(low) > 2.7:
            raise ValueError(
                f"marker voltages must be in [-2.7, +2.7] V (got HIGH={high}, LOW={low})"
            )
        if (high - low) > 5.4:
            raise ValueError(
                f"marker swing (HIGH-LOW={high - low} V) exceeds 5.4 V maximum"
            )
        for src in range(1, 5):
            for mk in (1, 2):
                self.awg.write(
                    f"SOURce{src}:MARKer{mk}:VOLTage:LEVel:IMMediate:HIGH {high}"
                )
                self.awg.write(
                    f"SOURce{src}:MARKer{mk}:VOLTage:LEVel:IMMediate:LOW {low}"
                )
        print(
            f"[{self.visa_address}] Marker voltage set: HIGH={high} V, LOW={low} V "
            f"(4 sources x 2 markers = 8 outputs)"
        )

    def set_analog_amplitude(self, channel: int, amplitude_vpp: float):
        """Set analog output amplitude for one AWG channel."""

        channel = int(channel)
        amplitude_vpp = float(amplitude_vpp)
        if channel not in (1, 2, 3, 4):
            raise ValueError(f"channel must be 1..4 (got {channel})")
        if not 0.02 <= amplitude_vpp <= 4.5:
            raise ValueError(f"analog amplitude must be 0.02..4.5 Vpp (got {amplitude_vpp})")
        self.awg.write(f'SOURce{channel}:VOLTage:LEVel:IMMediate:AMPLitude {amplitude_vpp:g}')
        print(f"[{self.visa_address}] CH{channel} analog amplitude set to {amplitude_vpp:g} Vpp")

    def delete_all_waveforms(self):
        """Delete all waveforms."""
        try:
            self.awg.write('WLIST:WAVEFORM:DELETE ALL')
        except:
            pass

    def delete_waveform(self, name: str):
        """Delete a specific waveform."""
        try:
            self.awg.write(f'WLIST:WAVEFORM:DELETE "{name}"')
        except:
            pass

    def create_waveform(self, name: str, samples: int):
        """Create a new waveform."""
        self.awg.write(f':WLIST:WAVEFORM:NEW "{name}", {samples}, REAL')

    def send_waveform_data(self, name: str, analog_data: np.ndarray, marker_bits: np.ndarray):
        """Send waveform data to AWG."""
        samples = len(analog_data)
        analog_data = np.asarray(analog_data, dtype=np.float32)
        analog_data = np.clip(analog_data, -1.0, 1.0).astype(np.float32, copy=False)

        # Prepare marker data
        marker_data = np.zeros(samples, dtype=np.uint8)
        if marker_bits is not None and len(marker_bits) == samples:
            marker_data = np.asarray(marker_bits, dtype=np.uint8)

        # Build binary block (5 bytes per sample: 4 for float32 + 1 for marker)
        binblock = np.empty(samples * 5, dtype=np.uint8)
        rows = binblock.reshape(samples, 5)
        rows[:, :4] = analog_data.astype("<f4", copy=False).view(np.uint8).reshape(samples, 4)
        rows[:, 4] = marker_data

        # Build binary block header
        num_bytes = len(binblock)
        bytes_str = str(num_bytes)
        header = f'#{len(bytes_str)}{bytes_str}'

        # Send waveform data
        cmd = f':WLIST:WAVEFORM:DATA "{name}",0,{samples},'
        cmd_bytes = cmd.encode('ascii')
        header_bytes = header.encode('ascii')

        old_termination = self.awg.write_termination
        self.awg.write_termination = ''
        try:
            self.awg.write_raw(cmd_bytes + header_bytes + binblock.tobytes())
            self.awg.write_raw((old_termination or '\n').encode('ascii'))
        finally:
            self.awg.write_termination = old_termination

    def assign_waveform_to_channel(self, channel: int, waveform_name: str):
        """Assign a waveform to a channel."""
        self.awg.write(f'SOURCE{channel}:WAVEFORM "{waveform_name}"')
        print(f"[{self.visa_address}] Waveform '{waveform_name}' assigned to CH{channel}")

    def set_channel_output(self, channel: int, enabled: bool):
        """Turn channel output on or off."""
        state = 'ON' if enabled else 'OFF'
        self.awg.write(f'OUTPUT{channel}:STATE {state}')

    def run(self):
        """Start waveform generation."""
        self.awg.write('AWGControl:RUN:IMMediate')
        print(f"[{self.visa_address}] AWG started")

    def stop(self):
        """Stop waveform generation."""
        self.awg.write('AWGControl:STOP:IMMediate')
        print(f"[{self.visa_address}] AWG stopped")


def load_sequence_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a sequence file containing analog and digital data.

    Args:
        filepath: Path to the CSV file

    Returns:
        Tuple of (analog_data, marker_bits)
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)
    data = np.atleast_2d(data)
    analog_data = data[:, 0].astype(np.float32, copy=False)
    digital1 = data[:, 1].astype(np.uint8, copy=False)
    digital2 = data[:, 2].astype(np.uint8, copy=False)
    marker_bits = ((digital1 & 1) << 6) | ((digital2 & 1) << 7)
    return analog_data, marker_bits.astype(np.uint8, copy=False)


def get_available_time_durations(sequence_dir: Path) -> List[str]:
    """Get list of available time durations from file names."""
    durations = set()
    for f in sequence_dir.glob('*.txt'):
        # Extract time duration from filename like "W_CH1_T266us.txt"
        parts = f.stem.split('_T')
        if len(parts) == 2:
            duration = parts[1].replace('us', '') + 'us'
            durations.add(duration)
    return sorted(list(durations))


def set_front_4qubit(time_duration: str = '266us',
                     awg1_ip: str = '192.168.2.122::4011',
                     awg2_ip: str = '192.168.2.110::3992',
                     sequence_dir: str = os.path.join('AWG_Sequence', 'entangle_sequences_4qubit'),
                     sample_rate: float = 1e9,
                     marker_high: float = 2.5,
                     marker_low: float = 0.0,
                     trigger_source: str = 'EXT',
                     trigger_level: float = 1.4,
                     trigger_impedance: float = 1000.0,
                     log: Optional[Callable[[str], None]] = None):
    """
    Upload 4 qubit sequences to two AWGs.

    Args:
        time_duration: Time duration to use (e.g., '266us', '1034us', '1054us')
        awg1_ip: IP address of first AWG (W state)
        awg2_ip: IP address of second AWG (R state)
        sequence_dir: Directory containing sequence files
        sample_rate: Sample rate in Hz (default: 1 GHz)
        trigger_source: Sequence trigger source (default EXTernal).
        trigger_level: External trigger threshold in volts (default 1.4 V).
        trigger_impedance: External trigger input impedance in ohms (default 1 kOhm).
    """
    sequence_path = Path(sequence_dir)
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

    # Normalize time_duration format
    if not time_duration.endswith('us'):
        time_duration = time_duration + 'us'

    # Get available durations and validate
    available = get_available_time_durations(sequence_path)
    if time_duration not in available:
        raise ValueError(f"Invalid time duration '{time_duration}'. Available: {available}")

    # Extract time number for waveform naming
    time_num = time_duration.replace('us', '')

    def _emit(msg: str) -> None:
        """Forward progress to the optional UI callback, otherwise print."""
        if log is not None:
            try:
                log(msg)
            except Exception:  # never let UI plumbing break the AWG job
                pass
        else:
            print(msg)

    _emit("=" * 60)
    _emit(f"Setting Front 4 Qubit Sequences (T{time_num}us)")
    _emit("=" * 60)

    # Define file mappings — 4 channels per AWG (W state on AWG1, R state on AWG2).
    w_files = {
        1: sequence_path / f'W_CH1_T{time_num}us.txt',
        2: sequence_path / f'W_CH2_T{time_num}us.txt',
        3: sequence_path / f'W_CH3_T{time_num}us.txt',
        4: sequence_path / f'W_CH4_T{time_num}us.txt',
    }

    r_files = {
        1: sequence_path / f'R_CH1_T{time_num}us.txt',
        2: sequence_path / f'R_CH2_T{time_num}us.txt',
        3: sequence_path / f'R_CH3_T{time_num}us.txt',
        4: sequence_path / f'R_CH4_T{time_num}us.txt',
    }

    # Check all 8 files exist before opening any instrument session.
    missing = [str(f) for f in {**w_files, **r_files}.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} sequence file(s) for T{time_num}us:\n  - "
            + "\n  - ".join(missing)
        )
    _emit(f"Verified all 8 sequence files present (4 W + 4 R, T{time_num}us).")

    # Connect to AWGs
    _emit(f"\nConnecting to AWG1 ({awg1_ip}) for W state (CH1-4)...")
    with AWG5014CController(awg1_ip) as awg1:
        _emit(f"AWG1 ID: {awg1.query_idn().strip()}")

        _emit(f"\nConnecting to AWG2 ({awg2_ip}) for R state (CH1-4)...")
        awg2 = AWG5014CController(awg2_ip)
        awg2.connect()
        _emit(f"AWG2 ID: {awg2.query_idn().strip()}")

        try:
            # Configure both AWGs
            _emit("\n" + "-" * 60)
            _emit("Configuring AWGs (reset / 1 GHz / TRIG / EXT / clear waveforms)...")
            _emit("-" * 60)

            for awg in [awg1, awg2]:
                awg.reset()
                awg.set_sample_rate(sample_rate)
                awg.set_run_mode('TRIG')
                awg.configure_trigger_input(
                    source=trigger_source,
                    level=trigger_level,
                    impedance=trigger_impedance,
                )
                awg.set_marker_voltage(high=marker_high, low=marker_low)
                awg.delete_all_waveforms()
            # Only W_CH1 (AWG1 CH1) and R_CH1 (AWG2 CH1) use reduced analog amplitude.
            awg1.set_analog_amplitude(1, CH1_ANALOG_AMPLITUDE_VPP)
            awg2.set_analog_amplitude(1, CH1_ANALOG_AMPLITUDE_VPP)

            # Upload W state sequences to AWG1 — 4 files, 4 channels.
            _emit("\n" + "-" * 60)
            _emit(f"Uploading W state sequences (T{time_num}us) → AWG1 ...")
            _emit("-" * 60)
            for ch, filepath in w_files.items():
                _emit(f"  [AWG1] {filepath.name} → CH{ch} ({filepath.stat().st_size} bytes)")
                analog_data, marker_bits = load_sequence_file(filepath)
                waveform_name = f'W_CH{ch}_T{time_num}us'
                awg1.create_waveform(waveform_name, len(analog_data))
                awg1.send_waveform_data(waveform_name, analog_data, marker_bits)
                awg1.assign_waveform_to_channel(ch, waveform_name)
                _emit(f"           waveform '{waveform_name}' ({len(analog_data)} samples) written & assigned")
            for ch in w_files:
                awg1.set_channel_output(ch, True)
            _emit(f"AWG1: outputs CH1-4 ENABLED.")

            # Upload R state sequences to AWG2 — 4 files, 4 channels.
            _emit("\n" + "-" * 60)
            _emit(f"Uploading R state sequences (T{time_num}us) → AWG2 ...")
            _emit("-" * 60)
            for ch, filepath in r_files.items():
                _emit(f"  [AWG2] {filepath.name} → CH{ch} ({filepath.stat().st_size} bytes)")
                analog_data, marker_bits = load_sequence_file(filepath)
                waveform_name = f'R_CH{ch}_T{time_num}us'
                awg2.create_waveform(waveform_name, len(analog_data))
                awg2.send_waveform_data(waveform_name, analog_data, marker_bits)
                awg2.assign_waveform_to_channel(ch, waveform_name)
                _emit(f"           waveform '{waveform_name}' ({len(analog_data)} samples) written & assigned")
            for ch in r_files:
                awg2.set_channel_output(ch, True)
            _emit(f"AWG2: outputs CH1-4 ENABLED.")

            # Start both AWGs
            _emit("\n" + "-" * 60)
            _emit("Starting AWGs (AWGControl:RUN:IMMediate)...")
            _emit("-" * 60)
            awg1.run()
            awg2.run()

            _emit("\n" + "=" * 60)
            _emit(f"4 Qubit Sequences Successfully Loaded! (T{time_num}us, 8 files / 4+4 channels)")
            _emit("=" * 60)
            _emit(f"AWG1 ({awg1_ip}): W state - CH1-4")
            _emit(f"AWG2 ({awg2_ip}): R state - CH1-4")
            _emit(f"Time Duration: {time_duration}")
            _emit(f"Sample Rate: {sample_rate/1e9} GHz")
            _emit(
                f"Trigger: {trigger_source.upper()}, "
                f"level={trigger_level} V, impedance={trigger_impedance} ohm"
            )
            _emit("Run Mode: TRIGGERED")
            _emit("=" * 60)

        finally:
            awg2.disconnect()


def list_available_durations(sequence_dir: str = os.path.join('AWG_Sequence', 'entangle_sequences_4qubit')):
    """List all available time durations."""
    sequence_path = Path(sequence_dir)
    if not sequence_path.exists():
        print(f"Sequence directory not found: {sequence_path}")
        return

    durations = get_available_time_durations(sequence_path)
    print("Available time durations:")
    for d in durations:
        print(f"  - {d}")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['--list', '-l', 'list']:
        # List available durations
        seq_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('AWG_Sequence', 'entangle_sequences_4qubit')
        list_available_durations(seq_dir)
        sys.exit(0)

    time_duration = sys.argv[1] if len(sys.argv) > 1 else '266us'
    awg1_ip = sys.argv[2] if len(sys.argv) > 2 else '192.168.2.122::4011'
    awg2_ip = sys.argv[3] if len(sys.argv) > 3 else '192.168.2.110::3992'
    sequence_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join('AWG_Sequence', 'entangle_sequences_4qubit')

    print(f"Arguments:")
    print(f"  Time duration: {time_duration}")
    print(f"  AWG1 (W state): {awg1_ip}")
    print(f"  AWG2 (R state): {awg2_ip}")
    print(f"  Sequence dir: {sequence_dir}")
    print()

    set_front_4qubit(time_duration=time_duration,
                     awg1_ip=awg1_ip,
                     awg2_ip=awg2_ip,
                     sequence_dir=sequence_dir,
                     sample_rate=1e9)
