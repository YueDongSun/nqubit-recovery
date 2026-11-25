"""AWG upload abstraction and front-panel pyvisa uploader."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from .hardware_safety import hardware_validation_summary, validate_sequence_dir_for_hardware


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


class AwgUploader(Protocol):
    def upload(self, sequence_dir: str, metadata: Dict[str, Any]) -> Any:
        """Upload generated sequences to the AWG."""


class NoOpAwgUploader:
    """Dry-run uploader used until websocket AWG control is implemented."""

    def upload(self, sequence_dir: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[AWG] no-op upload: {sequence_dir} metadata={metadata}")
        return {"uploaded": False, "backend": "noop", "sequence_dir": sequence_dir}


class FrontPanelAwgUploader:
    """Call local AWG front-panel scripts after sequence generation.

    Endpoints may be plain IP/VISA resources or raw SCPI websocket URLs
    beginning with ws:// or wss://.
    """

    def __init__(
        self,
        awg1_ip: str = "192.168.2.122::4011",
        awg2_ip: str = "192.168.2.110::3992",
        sample_rate: float = 1e9,
        trigger_source: str = "EXT",
        trigger_level: float = 1.4,
        trigger_impedance: float = 1000.0,
        upload_3q: Optional[Callable[..., Any]] = None,
        upload_4q: Optional[Callable[..., Any]] = None,
    ):
        self.awg1_ip = awg1_ip
        self.awg2_ip = awg2_ip
        self.sample_rate = sample_rate
        self.trigger_source = trigger_source
        self.trigger_level = trigger_level
        self.trigger_impedance = trigger_impedance
        self._upload_3q = upload_3q
        self._upload_4q = upload_4q

    def upload(self, sequence_dir: str, metadata: Dict[str, Any], log: Optional[Callable[[str], None]] = None, marker_high: float = 2.5, marker_low: float = 0.0) -> Dict[str, Any]:
        system_type = metadata.get("system_type")
        sequence_path = _resolve_sequence_dir(sequence_dir)
        if not sequence_path.exists():
            raise FileNotFoundError(f"Sequence directory not found: {sequence_path}")
        strict_hardware_mode = bool(metadata.get("strict_hardware_mode", False))
        guard_report = validate_sequence_dir_for_hardware(
            sequence_path,
            str(system_type),
            excitation_mode=int(metadata.get("excitation_mode", 1) or 1),
            strict=strict_hardware_mode,
        )
        if guard_report is not None and log is not None:
            log(hardware_validation_summary(guard_report))
            for note in guard_report.note_lines():
                log(note)
        validation_payload = {
            "strict_hardware_mode": strict_hardware_mode,
            "fourier_guard_checked_segments": (
                None if guard_report is None else int(guard_report.checked_segments)
            ),
            "fourier_guard_skipped_segments": (
                None if guard_report is None else int(guard_report.skipped_segments)
            ),
        }

        if system_type == "3qubit":
            time_duration = _latest_complete_time_duration(sequence_path, "3Q")
            func = self._upload_3q or _load_set_front_3qubit()
            func(
                awg1_ip=self.awg1_ip,
                awg2_ip=self.awg2_ip,
                sequence_dir=str(sequence_path),
                sample_rate=self.sample_rate,
                marker_high=marker_high,
                marker_low=marker_low,
                trigger_source=self.trigger_source,
                trigger_level=self.trigger_level,
                trigger_impedance=self.trigger_impedance,
                log=log,
                time_duration=time_duration,
            )
            return {
                "uploaded": True,
                "backend": self._backend_name(),
                "system_type": system_type,
                "sequence_dir": str(sequence_path),
                "time_duration": time_duration,
                **validation_payload,
            }

        if system_type == "4qubit":
            time_duration = _latest_complete_time_duration(sequence_path, "4Q")
            func = self._upload_4q or _load_set_front_4qubit()
            func(
                time_duration=time_duration,
                awg1_ip=self.awg1_ip,
                awg2_ip=self.awg2_ip,
                sequence_dir=str(sequence_path),
                sample_rate=self.sample_rate,
                marker_high=marker_high,
                marker_low=marker_low,
                trigger_source=self.trigger_source,
                trigger_level=self.trigger_level,
                trigger_impedance=self.trigger_impedance,
                log=log,
            )
            return {
                "uploaded": True,
                "backend": self._backend_name(),
                "system_type": system_type,
                "sequence_dir": str(sequence_path),
                "time_duration": time_duration,
                **validation_payload,
            }

        raise ValueError("metadata['system_type'] must be '3qubit' or '4qubit'")

    def _backend_name(self) -> str:
        endpoints = [self.awg1_ip.strip().lower(), self.awg2_ip.strip().lower()]
        if any(endpoint.startswith(("ws://", "wss://")) for endpoint in endpoints):
            if all(endpoint.startswith(("ws://", "wss://")) for endpoint in endpoints):
                return "front_panel_websocket"
            return "front_panel_mixed_pyvisa_websocket"
        return "front_panel_pyvisa"


def _resolve_sequence_dir(sequence_dir: str) -> Path:
    path = Path(sequence_dir)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _load_set_front_3qubit():
    port_dir = Path(__file__).resolve().parents[2] / "AWGcontrol_port"
    if str(port_dir) not in sys.path:
        sys.path.insert(0, str(port_dir))
    from set_front_3qubit import set_front_3qubit
    return set_front_3qubit


def _load_set_front_4qubit():
    port_dir = Path(__file__).resolve().parents[2] / "AWGcontrol_port"
    if str(port_dir) not in sys.path:
        sys.path.insert(0, str(port_dir))
    from set_front_4qubit import set_front_4qubit
    return set_front_4qubit


def _latest_complete_time_duration(sequence_dir: Path, label: str) -> str:
    groups: Dict[str, Dict[str, Path]] = {}
    for path in sequence_dir.glob("*.txt"):
        match = _SEQUENCE_FILE_RE.match(path.name)
        if not match:
            continue
        channel = match.group("channel").upper()
        duration = f"{int(match.group('duration'))}us"
        groups.setdefault(duration, {})[channel] = path

    complete = []
    for duration, files in groups.items():
        if all(channel in files for channel in REQUIRED_SEQUENCE_CHANNELS):
            latest_mtime = max(files[channel].stat().st_mtime for channel in REQUIRED_SEQUENCE_CHANNELS)
            complete.append((latest_mtime, int(duration[:-2]), duration))
    if complete:
        complete.sort(reverse=True)
        return complete[0][2]

    if not groups:
        raise FileNotFoundError(f"No {label} sequence files with Txxxus duration found in {sequence_dir}")

    details = []
    for duration, files in sorted(groups.items(), key=lambda item: int(item[0][:-2])):
        missing = [channel for channel in REQUIRED_SEQUENCE_CHANNELS if channel not in files]
        details.append(f"T{duration} missing {', '.join(missing)}")
    raise FileNotFoundError(
        f"No complete 8-channel {label} sequence duration found in {sequence_dir}. "
        f"Partial groups: {'; '.join(details)}"
    )
