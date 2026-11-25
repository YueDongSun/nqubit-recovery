"""Adapters for opening the existing sequence and data UIs from the main UI."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from system_control.qubit_recon_files.digital_delay_defaults import (
    DEFAULT_DIGITAL_DELAY_OFFSET_NS,
    digital_delay_values,
)

from .data_ui_actions import DataUiPrepareResult, clear_start_data_ui, save_clear_start_data_ui


@dataclass
class UiActionResult:
    success: bool
    message: str
    payload: Optional[Dict[str, Any]] = None


class SequenceUiController:
    """Open the existing entangle UI and optionally trigger generation."""

    def __init__(self):
        self.window = None
        self.system_type = None
        self.digital_delay_offset_ns = DEFAULT_DIGITAL_DELAY_OFFSET_NS

    def open(self, system_type: str, state: Dict[str, Any], excitation_mode: int):
        if self.window is not None and self.system_type == system_type:
            self.apply_state(state, excitation_mode)
            self.apply_digital_delay_offset(self.digital_delay_offset_ns)
            self.window.show()
            return self.window

        if self.window is not None and self.system_type != system_type:
            try:
                self.window.close()
            except Exception:
                pass
            self.window = None

        repo_root = Path(__file__).resolve().parents[2]
        recon_dir = repo_root / "system_control" / "qubit_recon_files"
        if str(recon_dir) not in sys.path:
            sys.path.insert(0, str(recon_dir))

        if system_type == "3qubit":
            from ui_entangle import EntangleUI
        elif system_type == "4qubit":
            from ui_entangle_4qubit import EntangleUI
        else:
            raise ValueError("system_type must be '3qubit' or '4qubit'")

        self.window = EntangleUI()
        self.system_type = system_type
        self.apply_state(state, excitation_mode)
        self.apply_digital_delay_offset(self.digital_delay_offset_ns)
        self.window.show()
        return self.window

    def generate_current(self, retain_channels: bool = True, make_preview: bool = True) -> Optional[str]:
        if self.window is None:
            return None
        try:
            self.window.generate_sequence(
                retain_channels=retain_channels,
                make_preview=make_preview,
            )
        except TypeError:
            self.window.generate_sequence()
        if hasattr(self.window, "txt_output_dir"):
            return self.window.txt_output_dir.text()
        return None

    def status(self) -> Dict[str, Any]:
        output_dir = None
        if self.window is not None and hasattr(self.window, "txt_output_dir"):
            output_dir = self.window.txt_output_dir.text()
        return {
            "open": self.window is not None,
            "system_type": self.system_type,
            "output_dir": output_dir,
            "digital_delay_offset_ns": self.digital_delay_offset_ns,
        }

    def apply_state(self, state: Dict[str, Any], excitation_mode: int) -> None:
        if self.window is None:
            return
        self._apply_state(self.window, state, excitation_mode)

    def refresh_generated_output(self, output_dir: str) -> None:
        if self.window is None:
            return
        if hasattr(self.window, "refresh_from_generated_output"):
            self.window.refresh_from_generated_output(output_dir)

    def apply_digital_delay_offset(self, offset_ns: int) -> None:
        self.digital_delay_offset_ns = int(offset_ns)
        if self.window is None:
            return
        if hasattr(self.window, "set_digital_delay_offset"):
            self.window.set_digital_delay_offset(self.digital_delay_offset_ns)
            return
        delays = digital_delay_values(self.digital_delay_offset_ns)
        w_spin = getattr(self.window, "spin_w_ch1_digital_delay", None)
        if w_spin is not None:
            w_spin.setValue(delays["W_CH1"])
        r_spin = getattr(self.window, "spin_r_ch1_digital_delay", None)
        if r_spin is not None:
            r_spin.setValue(delays["R_CH1"])

    def _apply_state(self, window, state: Dict[str, Any], excitation_mode: int) -> None:
        amps = state.get("amplitudes", [])
        phases = state.get("actual_loaded_phases_deg", state.get("phases_deg", []))
        if hasattr(window, "set_target_state"):
            window.set_target_state(amps, phases, excitation_mode)
            return
        for idx, value in enumerate(amps, start=1):
            spin = getattr(window, f"spin_amp{idx}", None)
            if spin is not None:
                spin.setValue(float(value))
        for idx, value in enumerate(phases, start=1):
            spin = getattr(window, f"spin_phase{idx}", None)
            if spin is not None:
                spin.setValue(float(value))
        if hasattr(window, "w_ch1_excitation_mode"):
            window.w_ch1_excitation_mode = int(excitation_mode)
        if hasattr(window, "combo_excitation_mode"):
            try:
                window.combo_excitation_mode.setCurrentIndex(int(excitation_mode) - 1)
            except Exception:
                pass


class DataUiController:
    """Open the existing data UI and forward parsed serial frames."""

    def __init__(self, on_frame: Optional[Callable[[Dict], None]] = None):
        self.window = None
        self.on_frame = on_frame
        self.system_type = None

    def open(self, system_type: str, feedback_context: Optional[Dict[str, Any]] = None):
        if self.window is not None and self.system_type == system_type:
            if feedback_context is not None and hasattr(self.window, "set_feedback_context"):
                self.window.set_feedback_context(feedback_context)
            self.window.show()
            return self.window
        if self.window is not None and self.system_type != system_type:
            try:
                if hasattr(self.window, "close_serial"):
                    self.window.close_serial()
            except Exception:
                pass
            try:
                self.window.close()
            except Exception:
                pass
            self.window = None

        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "system_control" / "data_ui"
        if str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))

        if system_type == "3qubit":
            from ui_3qubit import MainWindow
        elif system_type == "4qubit":
            from ui_4qubit import MainWindow
        else:
            raise ValueError("system_type must be '3qubit' or '4qubit'")

        self.window = MainWindow()
        self.system_type = system_type
        if self.on_frame is not None and hasattr(self.window, "driver"):
            self.window.driver.data_ready.connect(self.on_frame)
        if feedback_context is not None and hasattr(self.window, "set_feedback_context"):
            self.window.set_feedback_context(feedback_context)
        self.window.show()
        return self.window

    def set_feedback_context(self, context: Dict[str, Any]) -> None:
        if self.window is not None and hasattr(self.window, "set_feedback_context"):
            self.window.set_feedback_context(context)

    def latest_frame(self) -> Optional[Dict]:
        if self.window is None:
            return None
        return getattr(self.window, "current_data", None)

    def available_ports(self) -> List[str]:
        window = self.window
        if window is None:
            return []
        if hasattr(window, "refresh_ports"):
            window.refresh_ports()
        combo = getattr(window, "com_combo", None)
        if combo is None:
            return []
        return [combo.itemText(index) for index in range(combo.count())]

    def select_port(self, port: str) -> None:
        window = self.window
        combo = getattr(window, "com_combo", None) if window is not None else None
        if combo is not None and port:
            if combo.findText(port) < 0:
                combo.addItem(port)
            combo.setCurrentText(port)

    def open_port(self, port: Optional[str] = None) -> UiActionResult:
        window = self.window
        if window is None:
            return UiActionResult(False, "Data UI is not open.")
        if port:
            self.select_port(port)
        if not hasattr(window, "open_serial"):
            return UiActionResult(False, "Data UI has no open_serial action.")
        try:
            window.open_serial()
        except Exception as exc:
            return UiActionResult(False, f"Failed to open COM port: {exc}")
        status = self.status()
        if not status["com_open"]:
            return UiActionResult(False, f"COM port did not open: {status.get('port')}")
        return UiActionResult(True, f"Opened {status.get('port')}.", status)

    def close_port(self) -> UiActionResult:
        window = self.window
        if window is None:
            return UiActionResult(False, "Data UI is not open.")
        try:
            window.close_serial()
        except Exception as exc:
            return UiActionResult(False, f"Failed to close COM port: {exc}")
        return UiActionResult(True, "Closed COM port.", self.status())

    def start_output(self) -> UiActionResult:
        return self._call_window_action("start_output", "Started data output.")

    def stop_output(self) -> UiActionResult:
        return self._call_window_action("stop_output", "Stopped data output.")

    def clear_data(self) -> UiActionResult:
        return self._call_window_action("clear_data", "Cleared data counters.")

    def save_current(self) -> UiActionResult:
        window = self.window
        if window is None:
            return UiActionResult(False, "Data UI is not open.")
        if not hasattr(window, "save_current_numpy_snapshot"):
            return UiActionResult(False, "Data UI has no snapshot save action.")
        try:
            path = window.save_current_numpy_snapshot()
        except Exception as exc:
            return UiActionResult(False, f"Failed to save current data: {exc}")
        return UiActionResult(True, "Saved current data." if path else "No current data to save.", {"path": path})

    def save_clear_start(self) -> DataUiPrepareResult:
        return save_clear_start_data_ui(self.window)

    def clear_start(self) -> DataUiPrepareResult:
        return clear_start_data_ui(self.window)

    def status(self) -> Dict[str, Any]:
        window = self.window
        driver = getattr(window, "driver", None) if window is not None else None
        serial_port = getattr(driver, "serial_port", None) if driver is not None else None
        read_thread = getattr(driver, "read_thread", None) if driver is not None else None
        combo = getattr(window, "com_combo", None) if window is not None else None
        port = combo.currentText() if combo is not None else getattr(serial_port, "port", None)
        current_data = getattr(window, "current_data", None) if window is not None else None
        last_frame_timestamp = getattr(window, "last_frame_timestamp", None) if window is not None else None
        last_frame_age_s = None
        if last_frame_timestamp is not None:
            last_frame_age_s = max(0.0, time.time() - float(last_frame_timestamp))
        return {
            "open": window is not None,
            "system_type": self.system_type,
            "port": port,
            "com_open": bool(serial_port is not None and getattr(serial_port, "is_open", False)),
            "has_current_data": current_data is not None,
            "frame_keys": sorted(current_data.keys()) if isinstance(current_data, dict) else [],
            "driver_alive": bool(getattr(driver, "alive", False)) if driver is not None else False,
            "read_thread_alive": read_thread.is_alive() if read_thread is not None else None,
            "last_frame_age_s": last_frame_age_s,
            "frame_count": getattr(window, "frame_count", None) if window is not None else None,
        }

    def _call_window_action(self, action: str, success_message: str) -> UiActionResult:
        window = self.window
        if window is None:
            return UiActionResult(False, "Data UI is not open.")
        func = getattr(window, action, None)
        if func is None:
            return UiActionResult(False, f"Data UI has no {action} action.")
        try:
            func()
        except Exception as exc:
            return UiActionResult(False, f"{action} failed: {exc}")
        return UiActionResult(True, success_message, self.status())


RealDataUiSource = DataUiController
