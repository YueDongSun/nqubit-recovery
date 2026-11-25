"""Readiness gates for hardware-backed experiment UI actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class ExperimentUiReadiness:
    sequence_ui_open: bool
    data_ui_open: bool
    com_selected: bool
    com_open: bool
    fpga_programming_idle: bool
    awg_upload_idle: bool
    measurement_idle: bool
    has_sequence_dir: bool
    can_generate: bool
    can_upload: bool
    can_open_com: bool
    can_start_manual_window: bool
    can_start_auto_cycle: bool
    blocking_reasons: list[str]


def build_experiment_readiness(
    sequence_status: Mapping[str, Any],
    data_status: Mapping[str, Any],
    *,
    selected_com_port: Optional[str] = None,
    fpga_programming_running: bool = False,
    awg_upload_running: bool = False,
    measurement_running: bool = False,
    auto_running: bool = False,
    last_sequence_dir: Optional[str] = None,
) -> ExperimentUiReadiness:
    """Return UI action gates from current child UI and hardware state."""

    sequence_ui_open = bool(sequence_status.get("open"))
    data_ui_open = bool(data_status.get("open"))
    port = selected_com_port or data_status.get("port") or ""
    com_selected = _valid_com_port(port)
    com_open = bool(data_status.get("com_open"))
    fpga_programming_idle = not fpga_programming_running
    awg_upload_idle = not awg_upload_running
    hardware_idle = fpga_programming_idle and awg_upload_idle
    measurement_idle = not measurement_running
    has_sequence_dir = bool(last_sequence_dir or sequence_status.get("output_dir"))

    can_generate = sequence_ui_open and hardware_idle
    can_upload = can_generate and has_sequence_dir
    can_open_com = data_ui_open and com_selected and not com_open
    can_start_manual_window = data_ui_open and com_open and hardware_idle and measurement_idle
    can_start_auto_cycle = (
        sequence_ui_open
        and data_ui_open
        and com_open
        and hardware_idle
        and measurement_idle
        and not auto_running
    )

    reasons: list[str] = []
    if not sequence_ui_open:
        reasons.append("Sequence UI is not open.")
    if not data_ui_open:
        reasons.append("Data UI is not open.")
    if data_ui_open and not com_selected:
        reasons.append("No valid COM port is selected.")
    if data_ui_open and not com_open:
        reasons.append("COM port is not open.")
    if not fpga_programming_idle:
        reasons.append("FPGA programming is running.")
    if not awg_upload_idle:
        reasons.append("AWG upload is running.")
    if not measurement_idle:
        reasons.append("Measurement window is already running.")
    if auto_running:
        reasons.append("Auto cycle is already running.")

    return ExperimentUiReadiness(
        sequence_ui_open=sequence_ui_open,
        data_ui_open=data_ui_open,
        com_selected=com_selected,
        com_open=com_open,
        fpga_programming_idle=fpga_programming_idle,
        awg_upload_idle=awg_upload_idle,
        measurement_idle=measurement_idle,
        has_sequence_dir=has_sequence_dir,
        can_generate=can_generate,
        can_upload=can_upload,
        can_open_com=can_open_com,
        can_start_manual_window=can_start_manual_window,
        can_start_auto_cycle=can_start_auto_cycle,
        blocking_reasons=reasons,
    )


def _valid_com_port(port: object) -> bool:
    text = str(port or "").strip()
    return bool(text and "no ports" not in text.lower())
