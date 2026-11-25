"""
Set Front 3 Qubit Sequences to AWGs

Uploads entangle_sequences (W and R states) to two AWG5014C instruments:
- AWG1: 192.168.2.122::4011 (W state - CH1-4)
- AWG2: 192.168.2.110::3990 (R state - CH1-4)

Configuration:
- Sample rate: 1 GHz
- Mode: Triggered
- Trigger input: External, 1.4 V threshold, 1 kOhm impedance
- Run mode: Enabled

Files processed from AWG_Sequence/entangle_sequences/:
    W_CH1_T*.txt, W_CH2_T*.txt, W_CH3_T*.txt, W_CH4_T*.txt
    R_CH1_T*.txt, R_CH2_T*.txt, R_CH3_T*.txt, R_CH4_T*.txt
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable

from awg5014c_transport import open_instrument_resource


CH1_ANALOG_AMPLITUDE_VPP = 0.4
REQUIRED_SEQUENCE_CHANNELS = (
    "W_CH1",
    "W_CH2",
    "W_CH3",
    "W_CH4",
    "R_CH1",
    "R_CH2",
    "R_CH3",
    "R_CH4",
)
_SEQUENCE_FILE_RE = re.compile(r"^(?P<channel>[WR]_CH[1-4])_T(?P<duration>\d+)us\.txt$", re.IGNORECASE)


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


def _sequence_duration_groups(sequence_dir: Path) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = {}
    for path in sequence_dir.glob("*.txt"):
        match = _SEQUENCE_FILE_RE.match(path.name)
        if not match:
            continue
        channel = match.group("channel").upper()
        duration = f"{int(match.group('duration'))}us"
        groups.setdefault(duration, {})[channel] = path
    return groups


def get_available_time_durations(sequence_dir: Path) -> List[str]:
    """Get complete 8-channel sequence durations from file names."""
    groups = _sequence_duration_groups(sequence_dir)
    durations = [
        duration
        for duration, files in groups.items()
        if all(channel in files for channel in REQUIRED_SEQUENCE_CHANNELS)
    ]
    return sorted(durations, key=lambda item: int(item[:-2]))


def latest_complete_time_duration(sequence_dir: Path) -> str:
    """Return the newest duration that has all W/R channel files present."""
    groups = _sequence_duration_groups(sequence_dir)
    complete = []
    for duration, files in groups.items():
        if all(channel in files for channel in REQUIRED_SEQUENCE_CHANNELS):
            latest_mtime = max(files[channel].stat().st_mtime for channel in REQUIRED_SEQUENCE_CHANNELS)
            complete.append((latest_mtime, int(duration[:-2]), duration))
    if complete:
        complete.sort(reverse=True)
        return complete[0][2]

    if not groups:
        raise FileNotFoundError(f"No generated 3Q sequence files found in {sequence_dir}")

    details = []
    for duration, files in sorted(groups.items(), key=lambda item: int(item[0][:-2])):
        missing = [channel for channel in REQUIRED_SEQUENCE_CHANNELS if channel not in files]
        details.append(f"T{duration} missing {', '.join(missing)}")
    raise FileNotFoundError(
        "No complete 8-channel 3Q sequence duration found in "
        f"{sequence_dir}. Partial groups: {'; '.join(details)}"
    )


def set_front_3qubit(awg1_ip: str = '192.168.2.122::4011',
                     awg2_ip: str = '192.168.2.110::3992',
                     sequence_dir: str = os.path.join('AWG_Sequence', 'entangle_sequences'),
                     sample_rate: float = 1e9,
                     marker_high: float = 2.5,
                     marker_low: float = 0.0,
                     trigger_source: str = 'EXT',
                     trigger_level: float = 1.4,
                     trigger_impedance: float = 1000.0,
                     log: Optional[Callable[[str], None]] = None,
                     time_duration: Optional[str] = None):
    """
    Upload 3 qubit sequences to two AWGs.

    Args:
        awg1_ip: IP address of first AWG (W state)
        awg2_ip: IP address of second AWG (R state)
        sequence_dir: Directory containing sequence files
        sample_rate: Sample rate in Hz (default: 1 GHz)
        marker_high: Marker HIGH voltage in volts (default 2.5 V).
        marker_low:  Marker LOW voltage in volts (default 0.0 V).
        trigger_source: Sequence trigger source (default EXTernal).
        trigger_level: External trigger threshold in volts (default 1.4 V).
        trigger_impedance: External trigger input impedance in ohms (default 1 kOhm).
        log: Optional UI log callback; falls back to print() when None.
        time_duration: Optional duration to load, e.g. '237us'. When omitted,
            the newest complete 8-channel duration in sequence_dir is used.
    """
    sequence_path = Path(sequence_dir)
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_path}")

    if time_duration is None:
        time_duration = latest_complete_time_duration(sequence_path)
    time_duration = str(time_duration).strip().lower()
    if not time_duration.endswith('us'):
        time_duration = time_duration + 'us'
    time_num = time_duration.replace('us', '')
    if time_num.startswith('t'):
        time_num = time_num[1:]
    time_duration = f"{time_num}us"

    print("=" * 60)
    print(f"Setting Front 3 Qubit Sequences (T{time_num}us)")
    print("=" * 60)

    # Define file mappings
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

    # Check all files exist
    for ch, f in {**w_files, **r_files}.items():
        if not f.exists():
            raise FileNotFoundError(f"File not found: {f}")

    # Connect to AWGs
    print(f"\nConnecting to AWG1 ({awg1_ip}) for W state...")
    with AWG5014CController(awg1_ip) as awg1:
        print(f"AWG1 ID: {awg1.query_idn()}")

        print(f"\nConnecting to AWG2 ({awg2_ip}) for R state...")
        awg2 = AWG5014CController(awg2_ip)
        awg2.connect()
        print(f"AWG2 ID: {awg2.query_idn()}")

        try:
            # Configure both AWGs
            print("\n" + "-" * 60)
            print("Configuring AWGs...")
            print("-" * 60)

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

            # Upload W state sequences to AWG1
            print("\n" + "-" * 60)
            print("Uploading W state sequences to AWG1...")
            print("-" * 60)

            for ch, filepath in w_files.items():
                print(f"\nLoading {filepath.name}...")
                analog_data, marker_bits = load_sequence_file(filepath)
                waveform_name = f'W_CH{ch}_T{time_num}us'
                awg1.create_waveform(waveform_name, len(analog_data))
                awg1.send_waveform_data(waveform_name, analog_data, marker_bits)
                awg1.assign_waveform_to_channel(ch, waveform_name)

            for ch in w_files:
                awg1.set_channel_output(ch, True)

            # Upload R state sequences to AWG2
            print("\n" + "-" * 60)
            print("Uploading R state sequences to AWG2...")
            print("-" * 60)

            for ch, filepath in r_files.items():
                print(f"\nLoading {filepath.name}...")
                analog_data, marker_bits = load_sequence_file(filepath)
                waveform_name = f'R_CH{ch}_T{time_num}us'
                awg2.create_waveform(waveform_name, len(analog_data))
                awg2.send_waveform_data(waveform_name, analog_data, marker_bits)
                awg2.assign_waveform_to_channel(ch, waveform_name)

            for ch in r_files:
                awg2.set_channel_output(ch, True)

            # Start both AWGs
            print("\n" + "-" * 60)
            print("Starting AWGs...")
            print("-" * 60)
            awg1.run()
            awg2.run()

            print("\n" + "=" * 60)
            print("3 Qubit Sequences Successfully Loaded!")
            print("=" * 60)
            print(f"AWG1 ({awg1_ip}): W state - CH1-4")
            print(f"AWG2 ({awg2_ip}): R state - CH1-4")
            print(f"Time Duration: {time_duration}")
            print(f"Sample Rate: {sample_rate/1e9} GHz")
            print(
                f"Trigger: {trigger_source.upper()}, "
                f"level={trigger_level} V, impedance={trigger_impedance} ohm"
            )
            print("Run Mode: TRIGGERED")
            print("=" * 60)

        finally:
            awg2.disconnect()


def list_available_durations(sequence_dir: str = os.path.join('AWG_Sequence', 'entangle_sequences')):
    """List all complete 3Q time durations."""
    sequence_path = Path(sequence_dir)
    if not sequence_path.exists():
        print(f"Sequence directory not found: {sequence_path}")
        return

    durations = get_available_time_durations(sequence_path)
    print("Available complete time durations:")
    for duration in durations:
        print(f"  - {duration}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ['--list', '-l', 'list']:
        seq_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('AWG_Sequence', 'entangle_sequences')
        list_available_durations(seq_dir)
        sys.exit(0)

    # Parse command line arguments
    awg1_ip = sys.argv[1] if len(sys.argv) > 1 else '192.168.2.122::4011'
    awg2_ip = sys.argv[2] if len(sys.argv) > 2 else '192.168.2.110::3992'
    sequence_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join('AWG_Sequence', 'entangle_sequences')
    time_duration = sys.argv[4] if len(sys.argv) > 4 else None

    print(f"Arguments:")
    print(f"  AWG1 (W state): {awg1_ip}")
    print(f"  AWG2 (R state): {awg2_ip}")
    print(f"  Sequence dir: {sequence_dir}")
    print(f"  Time duration: {time_duration or 'latest complete'}")
    print()

    set_front_3qubit(awg1_ip=awg1_ip,
                     awg2_ip=awg2_ip,
                     sequence_dir=sequence_dir,
                     sample_rate=1e9,
                     time_duration=time_duration)
