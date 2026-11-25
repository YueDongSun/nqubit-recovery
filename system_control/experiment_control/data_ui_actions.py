"""Safe actions for transitioning data UI to the next measurement window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataUiPrepareResult:
    success: bool
    message: str
    saved_path: Optional[Path] = None


def save_clear_start_data_ui(window) -> DataUiPrepareResult:
    """Save current data before clear, then clear counters and start output."""

    if window is None:
        return DataUiPrepareResult(False, "Data UI is not open.")

    driver = getattr(window, "driver", None)
    serial_port = getattr(driver, "serial_port", None)
    if not (serial_port is not None and getattr(serial_port, "is_open", False)):
        return DataUiPrepareResult(False, "COM port is not open. Data was not cleared.")

    saved_path = None
    try:
        if getattr(window, "current_data", None) is not None:
            saved_path = window.save_current_numpy_snapshot()
        window.clear_data()
        window.start_output()
    except Exception as exc:
        return DataUiPrepareResult(False, f"Failed before starting next window: {exc}", saved_path)

    if saved_path is None:
        return DataUiPrepareResult(True, "No current data to save; counters cleared and output started.")
    return DataUiPrepareResult(True, f"Saved data to {saved_path}; counters cleared and output started.", saved_path)


def clear_start_data_ui(window) -> DataUiPrepareResult:
    """Clear counters and start output without saving partial current data."""

    if window is None:
        return DataUiPrepareResult(False, "Data UI is not open.")

    driver = getattr(window, "driver", None)
    serial_port = getattr(driver, "serial_port", None)
    if not (serial_port is not None and getattr(serial_port, "is_open", False)):
        return DataUiPrepareResult(False, "COM port is not open. Data was not cleared.")

    try:
        window.clear_data()
        window.start_output()
    except Exception as exc:
        return DataUiPrepareResult(False, f"Failed before starting next window: {exc}")

    return DataUiPrepareResult(True, "Partial current data discarded; counters cleared and output started.")
